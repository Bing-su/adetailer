from __future__ import annotations

import os
import platform
import re
import sys
import traceback
from contextlib import contextmanager
from copy import copy, deepcopy
from functools import partial
from pathlib import Path
from textwrap import dedent
from typing import Any

import gradio as gr
import torch
from rich import print

import modules
from adetailer import (
    AFTER_DETAILER,
    __version__,
    get_models,
    mediapipe_predict,
    ultralytics_predict,
)
from adetailer.args import ALL_ARGS, BBOX_SORTBY, ADetailerArgs, EnableChecker
from adetailer.common import PredictOutput
from adetailer.mask import (
    filter_by_ratio,
    filter_k_largest,
    mask_preprocess,
    sort_bboxes,
)
from adetailer.traceback import rich_traceback
from adetailer.ui import WebuiInfo, adui, ordinal, suffix
from controlnet_ext import ControlNetExt, controlnet_exists, get_cn_models
from controlnet_ext.restore import (
    CNHijackRestore,
    cn_allow_script_control,
)
from modules import images, safe, script_callbacks, scripts, shared
from modules.devices import NansException
from modules.paths import data_path, models_path
from modules.processing import (
    Processed,
    StableDiffusionProcessingImg2Img,
    create_infotext,
    process_images,
)
from modules.sd_samplers import all_samplers
from modules.shared import cmd_opts, opts, state

no_huggingface = getattr(cmd_opts, "ad_no_huggingface", False)
adetailer_dir = Path(models_path, "adetailer")
model_mapping = get_models(adetailer_dir, huggingface=not no_huggingface)
txt2img_submit_button = img2img_submit_button = None
SCRIPT_DEFAULT = "dynamic_prompting,dynamic_thresholding,wildcard_recursive,wildcards,lora_block_weight"

if (
    not adetailer_dir.exists()
    and adetailer_dir.parent.exists()
    and os.access(adetailer_dir.parent, os.W_OK)
):
    adetailer_dir.mkdir()

print(
    f"[-] ADetailer initialized. version: {__version__}, num models: {len(model_mapping)}"
)


@contextmanager
def change_torch_load():
    orig = torch.load
    try:
        torch.load = safe.unsafe_torch_load
        yield
    finally:
        torch.load = orig


@contextmanager
def pause_total_tqdm():
    orig = opts.data.get("multiple_tqdm", True)
    try:
        opts.data["multiple_tqdm"] = False
        yield
    finally:
        opts.data["multiple_tqdm"] = orig


@contextmanager
def preseve_prompts(p):
    all_pt = copy(p.all_prompts)
    all_ng = copy(p.all_negative_prompts)
    try:
        yield
    finally:
        p.all_prompts = all_pt
        p.all_negative_prompts = all_ng


class AfterDetailerScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.ultralytics_device = self.get_ultralytics_device()

        self.controlnet_ext = None

    def __repr__(self):
        return f"{self.__class__.__name__}(version={__version__})"

    def title(self):
        return AFTER_DETAILER

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        num_models = opts.data.get("ad_max_models", 2)
        ad_model_list = list(model_mapping.keys())
        sampler_names = [sampler.name for sampler in all_samplers]

        try:
            checkpoint_list = modules.sd_models.checkpoint_tiles(use_shorts=True)
        except TypeError:
            checkpoint_list = modules.sd_models.checkpoint_tiles()
        vae_list = modules.shared_items.sd_vae_items()

        webui_info = WebuiInfo(
            ad_model_list=ad_model_list,
            sampler_names=sampler_names,
            t2i_button=txt2img_submit_button,
            i2i_button=img2img_submit_button,
            checkpoints_list=checkpoint_list,
            vae_list=vae_list,
        )

        components, infotext_fields = adui(num_models, is_img2img, webui_info)

        self.infotext_fields = infotext_fields
        return components

    def init_controlnet_ext(self) -> None:
        if self.controlnet_ext is not None:
            return
        self.controlnet_ext = ControlNetExt()

        if controlnet_exists:
            try:
                self.controlnet_ext.init_controlnet()
            except ImportError:
                error = traceback.format_exc()
                print(
                    f"[-] ADetailer: ControlNetExt init failed:\n{error}",
                    file=sys.stderr,
                )

    def update_controlnet_args(self, p, args: ADetailerArgs) -> None:
        if self.controlnet_ext is None:
            self.init_controlnet_ext()

        if (
            self.controlnet_ext is not None
            and self.controlnet_ext.cn_available
            and args.ad_controlnet_model != "None"
        ):
            self.controlnet_ext.update_scripts_args(
                p,
                model=args.ad_controlnet_model,
                module=args.ad_controlnet_module,
                weight=args.ad_controlnet_weight,
                guidance_start=args.ad_controlnet_guidance_start,
                guidance_end=args.ad_controlnet_guidance_end,
            )

    def is_ad_enabled(self, *args_) -> bool:
        arg_list = [arg for arg in args_ if isinstance(arg, dict)]
        if not args_ or not arg_list or not isinstance(args_[0], (bool, dict)):
            message = f"""
                       [-] ADetailer: Invalid arguments passed to ADetailer.
                           input: {args_!r}
                           ADetailer disabled.
                       """
            print(dedent(message), file=sys.stderr)
            return False
        enable = args_[0] if isinstance(args_[0], bool) else True
        checker = EnableChecker(enable=enable, arg_list=arg_list)
        return checker.is_enabled()

    def get_args(self, p, *args_) -> list[ADetailerArgs]:
        """
        `args_` is at least 1 in length by `is_ad_enabled` immediately above
        """
        args = [arg for arg in args_ if isinstance(arg, dict)]

        if not args:
            message = f"[-] ADetailer: Invalid arguments passed to ADetailer: {args_!r}"
            raise ValueError(message)

        if hasattr(p, "adetailer_xyz"):
            args[0] = {**args[0], **p.adetailer_xyz}

        all_inputs = []

        for n, arg_dict in enumerate(args, 1):
            try:
                inp = ADetailerArgs(**arg_dict)
            except ValueError as e:
                msgs = [
                    f"[-] ADetailer: ValidationError when validating {ordinal(n)} arguments: {e}\n"
                ]
                for attr in ALL_ARGS.attrs:
                    arg = arg_dict.get(attr)
                    dtype = type(arg)
                    arg = "DEFAULT" if arg is None else repr(arg)
                    msgs.append(f"    {attr}: {arg} ({dtype})")
                raise ValueError("\n".join(msgs)) from e

            all_inputs.append(inp)

        return all_inputs

    def extra_params(self, arg_list: list[ADetailerArgs]) -> dict:
        params = {}
        for n, args in enumerate(arg_list):
            params.update(args.extra_params(suffix=suffix(n)))
        params["ADetailer version"] = __version__
        return params

    @staticmethod
    def get_ultralytics_device() -> str:
        if "adetailer" in shared.cmd_opts.use_cpu:
            return "cpu"

        if platform.system() == "Darwin":
            return ""

        vram_args = ["lowvram", "medvram", "medvram_sdxl"]
        if any(getattr(cmd_opts, vram, False) for vram in vram_args):
            return "cpu"

        return ""

    def prompt_blank_replacement(
        self, all_prompts: list[str], i: int, default: str
    ) -> str:
        if not all_prompts:
            return default
        if i < len(all_prompts):
            return all_prompts[i]
        j = i % len(all_prompts)
        return all_prompts[j]

    def _get_prompt(
        self, ad_prompt: str, all_prompts: list[str], i: int, default: str
    ) -> list[str]:
        prompts = re.split(r"\s*\[SEP\]\s*", ad_prompt)
        blank_replacement = self.prompt_blank_replacement(all_prompts, i, default)
        for n in range(len(prompts)):
            if not prompts[n]:
                prompts[n] = blank_replacement
            elif "[PROMPT]" in prompts[n]:
                prompts[n] = prompts[n].replace("[PROMPT]", f" {blank_replacement} ")
        return prompts

    def get_prompt(self, p, args: ADetailerArgs) -> tuple[list[str], list[str]]:
        i = p._ad_idx

        prompt = self._get_prompt(args.ad_prompt, p.all_prompts, i, p.prompt)
        negative_prompt = self._get_prompt(
            args.ad_negative_prompt, p.all_negative_prompts, i, p.negative_prompt
        )

        return prompt, negative_prompt

    def get_seed(self, p) -> tuple[int, int]:
        i = p._ad_idx

        if not p.all_seeds:
            seed = p.seed
        elif i < len(p.all_seeds):
            seed = p.all_seeds[i]
        else:
            j = i % len(p.all_seeds)
            seed = p.all_seeds[j]

        if not p.all_subseeds:
            subseed = p.subseed
        elif i < len(p.all_subseeds):
            subseed = p.all_subseeds[i]
        else:
            j = i % len(p.all_subseeds)
            subseed = p.all_subseeds[j]

        return seed, subseed

    def get_width_height(self, p, args: ADetailerArgs) -> tuple[int, int]:
        if args.ad_use_inpaint_width_height:
            width = args.ad_inpaint_width
            height = args.ad_inpaint_height
        else:
            width = p.width
            height = p.height

        return width, height

    def get_steps(self, p, args: ADetailerArgs) -> int:
        return args.ad_steps if args.ad_use_steps else p.steps

    def get_cfg_scale(self, p, args: ADetailerArgs) -> float:
        return args.ad_cfg_scale if args.ad_use_cfg_scale else p.cfg_scale

    def get_sampler(self, p, args: ADetailerArgs) -> str:
        return args.ad_sampler if args.ad_use_sampler else p.sampler_name

    def get_override_settings(self, p, args: ADetailerArgs) -> dict[str, Any]:
        d = {}

        if args.ad_use_clip_skip:
            d["CLIP_stop_at_last_layers"] = args.ad_clip_skip

        if (
            args.ad_use_checkpoint
            and args.ad_checkpoint
            and args.ad_checkpoint not in ("None", "Use same checkpoint")
        ):
            d["sd_model_checkpoint"] = args.ad_checkpoint

        if (
            args.ad_use_vae
            and args.ad_vae
            and args.ad_vae not in ("None", "Use same VAE")
        ):
            d["sd_vae"] = args.ad_vae
        return d

    def get_initial_noise_multiplier(self, p, args: ADetailerArgs) -> float | None:
        return args.ad_noise_multiplier if args.ad_use_noise_multiplier else None

    @staticmethod
    def infotext(p) -> str:
        return create_infotext(
            p, p.all_prompts, p.all_seeds, p.all_subseeds, None, 0, 0
        )

    def write_params_txt(self, p) -> None:
        infotext = self.infotext(p)
        params_txt = Path(data_path, "params.txt")
        params_txt.write_text(infotext, encoding="utf-8")

    def script_filter(self, p, args: ADetailerArgs):
        script_runner = copy(p.scripts)
        script_args = deepcopy(p.script_args)
        self.disable_controlnet_units(script_args)

        ad_only_seleted_scripts = opts.data.get("ad_only_seleted_scripts", True)
        if not ad_only_seleted_scripts:
            return script_runner, script_args

        ad_script_names = opts.data.get("ad_script_names", SCRIPT_DEFAULT)
        script_names_set = {
            name
            for script_name in ad_script_names.split(",")
            for name in (script_name, script_name.strip())
        }

        if args.ad_controlnet_model != "None":
            script_names_set.add("controlnet")

        filtered_alwayson = []
        for script_object in script_runner.alwayson_scripts:
            filepath = script_object.filename
            filename = Path(filepath).stem
            if filename in script_names_set:
                filtered_alwayson.append(script_object)

        script_runner.alwayson_scripts = filtered_alwayson
        return script_runner, script_args

    def disable_controlnet_units(self, script_args: list[Any]) -> None:
        for obj in script_args:
            if "controlnet" in obj.__class__.__name__.lower():
                if hasattr(obj, "enabled"):
                    obj.enabled = False
                if hasattr(obj, "input_mode"):
                    obj.input_mode = getattr(obj.input_mode, "SIMPLE", "simple")

            elif isinstance(obj, dict) and "module" in obj:
                obj["enabled"] = False

    def get_i2i_p(self, p, args: ADetailerArgs, image):
        seed, subseed = self.get_seed(p)
        width, height = self.get_width_height(p, args)
        steps = self.get_steps(p, args)
        cfg_scale = self.get_cfg_scale(p, args)
        initial_noise_multiplier = self.get_initial_noise_multiplier(p, args)
        sampler_name = self.get_sampler(p, args)
        override_settings = self.get_override_settings(p, args)

        i2i = StableDiffusionProcessingImg2Img(
            init_images=[image],
            resize_mode=0,
            denoising_strength=args.ad_denoising_strength,
            mask=None,
            mask_blur=args.ad_mask_blur,
            inpainting_fill=1,
            inpaint_full_res=args.ad_inpaint_only_masked,
            inpaint_full_res_padding=args.ad_inpaint_only_masked_padding,
            inpainting_mask_invert=0,
            initial_noise_multiplier=initial_noise_multiplier,
            sd_model=p.sd_model,
            outpath_samples=p.outpath_samples,
            outpath_grids=p.outpath_grids,
            prompt="",  # replace later
            negative_prompt="",
            styles=p.styles,
            seed=seed,
            subseed=subseed,
            subseed_strength=p.subseed_strength,
            seed_resize_from_h=p.seed_resize_from_h,
            seed_resize_from_w=p.seed_resize_from_w,
            sampler_name=sampler_name,
            batch_size=1,
            n_iter=1,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            restore_faces=args.ad_restore_face,
            tiling=p.tiling,
            extra_generation_params=p.extra_generation_params,
            do_not_save_samples=True,
            do_not_save_grid=True,
            override_settings=override_settings,
        )

        i2i.cached_c = [None, None]
        i2i.cached_uc = [None, None]
        i2i.scripts, i2i.script_args = self.script_filter(p, args)
        i2i._ad_disabled = True

        if args.ad_controlnet_model != "None":
            self.update_controlnet_args(i2i, args)
        else:
            i2i.control_net_enabled = False

        return i2i

    def save_image(self, p, image, *, condition: str, suffix: str) -> None:
        i = p._ad_idx
        if p.all_prompts:
            i %= len(p.all_prompts)
            save_prompt = p.all_prompts[i]
        else:
            save_prompt = p.prompt
        seed, _ = self.get_seed(p)

        if opts.data.get(condition, False):
            images.save_image(
                image=image,
                path=p.outpath_samples,
                basename="",
                seed=seed,
                prompt=save_prompt,
                extension=opts.samples_format,
                info=self.infotext(p),
                p=p,
                suffix=suffix,
            )

    def get_ad_model(self, name: str):
        if name not in model_mapping:
            msg = f"[-] ADetailer: Model {name!r} not found. Available models: {list(model_mapping.keys())}"
            raise ValueError(msg)
        return model_mapping[name]

    def sort_bboxes(self, pred: PredictOutput) -> PredictOutput:
        sortby = opts.data.get("ad_bbox_sortby", BBOX_SORTBY[0])
        sortby_idx = BBOX_SORTBY.index(sortby)
        return sort_bboxes(pred, sortby_idx)

    def pred_preprocessing(self, pred: PredictOutput, args: ADetailerArgs):
        pred = filter_by_ratio(
            pred, low=args.ad_mask_min_ratio, high=args.ad_mask_max_ratio
        )
        pred = filter_k_largest(pred, k=args.ad_mask_k_largest)
        pred = self.sort_bboxes(pred)
        return mask_preprocess(
            pred.masks,
            kernel=args.ad_dilate_erode,
            x_offset=args.ad_x_offset,
            y_offset=args.ad_y_offset,
            merge_invert=args.ad_mask_merge_invert,
        )

    @staticmethod
    def ensure_rgb_image(image: Any):
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")
        return image

    @staticmethod
    def i2i_prompts_replace(
        i2i, prompts: list[str], negative_prompts: list[str], j: int
    ) -> None:
        i1 = min(j, len(prompts) - 1)
        i2 = min(j, len(negative_prompts) - 1)
        prompt = prompts[i1]
        negative_prompt = negative_prompts[i2]
        i2i.prompt = prompt
        i2i.negative_prompt = negative_prompt

    @staticmethod
    def compare_prompt(p, processed, n: int = 0):
        if p.prompt != processed.all_prompts[0]:
            print(
                f"[-] ADetailer: applied {ordinal(n + 1)} ad_prompt: {processed.all_prompts[0]!r}"
            )

        if p.negative_prompt != processed.all_negative_prompts[0]:
            print(
                f"[-] ADetailer: applied {ordinal(n + 1)} ad_negative_prompt: {processed.all_negative_prompts[0]!r}"
            )

    @staticmethod
    def need_call_process(p) -> bool:
        i = p._ad_idx
        bs = p.batch_size
        return i % bs == bs - 1

    @staticmethod
    def need_call_postprocess(p) -> bool:
        i = p._ad_idx
        bs = p.batch_size
        return i % bs == 0

    @rich_traceback
    def process(self, p, *args_):
        if getattr(p, "_ad_disabled", False):
            return

        if self.is_ad_enabled(*args_):
            arg_list = self.get_args(p, *args_)
            extra_params = self.extra_params(arg_list)
            p.extra_generation_params.update(extra_params)

    def _postprocess_image_inner(
        self, p, pp, args: ADetailerArgs, *, n: int = 0
    ) -> bool:
        """
        Returns
        -------
            bool

            `True` if image was processed, `False` otherwise.
        """
        if state.interrupted or state.skipped:
            return False

        i = p._ad_idx

        i2i = self.get_i2i_p(p, args, pp.image)
        seed, subseed = self.get_seed(p)
        ad_prompts, ad_negatives = self.get_prompt(p, args)

        is_mediapipe = args.ad_model.lower().startswith("mediapipe")

        kwargs = {}
        if is_mediapipe:
            predictor = mediapipe_predict
            ad_model = args.ad_model
        else:
            predictor = ultralytics_predict
            ad_model = self.get_ad_model(args.ad_model)
            kwargs["device"] = self.ultralytics_device

        with change_torch_load():
            pred = predictor(ad_model, pp.image, args.ad_confidence, **kwargs)

        masks = self.pred_preprocessing(pred, args)

        if not masks:
            print(
                f"[-] ADetailer: nothing detected on image {i + 1} with {ordinal(n + 1)} settings."
            )
            return False

        self.save_image(
            p,
            pred.preview,
            condition="ad_save_previews",
            suffix="-ad-preview" + suffix(n, "-"),
        )

        steps = len(masks)
        processed = None
        state.job_count += steps

        if is_mediapipe:
            print(f"mediapipe: {steps} detected.")

        p2 = copy(i2i)
        for j in range(steps):
            p2.image_mask = masks[j]
            p2.init_images[0] = self.ensure_rgb_image(p2.init_images[0])
            self.i2i_prompts_replace(p2, ad_prompts, ad_negatives, j)

            if re.match(r"^\s*\[SKIP\]\s*$", p2.prompt):
                continue

            p2.seed = seed + j
            p2.subseed = subseed + j

            try:
                processed = process_images(p2)
            except NansException as e:
                msg = f"[-] ADetailer: 'NansException' occurred with {ordinal(n + 1)} settings.\n{e}"
                print(msg, file=sys.stderr)
                continue
            finally:
                p2.close()

            self.compare_prompt(p2, processed, n=n)
            p2 = copy(i2i)
            p2.init_images = [processed.images[0]]

        if processed is not None:
            pp.image = processed.images[0]
            return True

        return False

    @rich_traceback
    def postprocess_image(self, p, pp, *args_):
        if getattr(p, "_ad_disabled", False):
            return

        if not self.is_ad_enabled(*args_):
            return

        p._ad_idx = getattr(p, "_ad_idx", -1) + 1
        init_image = copy(pp.image)
        arg_list = self.get_args(p, *args_)

        if p.scripts is not None and self.need_call_postprocess(p):
            dummy = Processed(p, [], p.seed, "")
            with preseve_prompts(p):
                p.scripts.postprocess(copy(p), dummy)

        is_processed = False
        with CNHijackRestore(), pause_total_tqdm(), cn_allow_script_control():
            for n, args in enumerate(arg_list):
                if args.ad_model == "None":
                    continue
                is_processed |= self._postprocess_image_inner(p, pp, args, n=n)

        if is_processed:
            self.save_image(
                p, init_image, condition="ad_save_images_before", suffix="-ad-before"
            )

        if p.scripts is not None and self.need_call_process(p):
            with preseve_prompts(p):
                p.scripts.process(copy(p))

        try:
            ia = p._ad_idx
            lenp = len(p.all_prompts)
            if ia % lenp == lenp - 1:
                self.write_params_txt(p)
        except Exception:
            pass


def on_after_component(component, **_kwargs):
    global txt2img_submit_button, img2img_submit_button
    if getattr(component, "elem_id", None) == "txt2img_generate":
        txt2img_submit_button = component
        return

    if getattr(component, "elem_id", None) == "img2img_generate":
        img2img_submit_button = component


def on_ui_settings():
    section = ("ADetailer", AFTER_DETAILER)
    shared.opts.add_option(
        "ad_max_models",
        shared.OptionInfo(
            default=2,
            label="Max models",
            component=gr.Slider,
            component_args={"minimum": 1, "maximum": 10, "step": 1},
            section=section,
        ),
    )

    shared.opts.add_option(
        "ad_save_previews",
        shared.OptionInfo(False, "Save mask previews", section=section),
    )

    shared.opts.add_option(
        "ad_save_images_before",
        shared.OptionInfo(False, "Save images before ADetailer", section=section),
    )

    shared.opts.add_option(
        "ad_only_seleted_scripts",
        shared.OptionInfo(
            True, "Apply only selected scripts to ADetailer", section=section
        ),
    )

    textbox_args = {
        "placeholder": "comma-separated list of script names",
        "interactive": True,
    }

    shared.opts.add_option(
        "ad_script_names",
        shared.OptionInfo(
            default=SCRIPT_DEFAULT,
            label="Script names to apply to ADetailer (separated by comma)",
            component=gr.Textbox,
            component_args=textbox_args,
            section=section,
        ),
    )

    shared.opts.add_option(
        "ad_bbox_sortby",
        shared.OptionInfo(
            default="None",
            label="Sort bounding boxes by",
            component=gr.Radio,
            component_args={"choices": BBOX_SORTBY},
            section=section,
        ),
    )


# xyz_grid


def make_axis_on_xyz_grid():
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break

    if xyz_grid is None:
        return

    model_list = ["None", *model_mapping.keys()]
    samplers = [sampler.name for sampler in all_samplers]

    def set_value(p, x, xs, *, field: str):
        if not hasattr(p, "adetailer_xyz"):
            p.adetailer_xyz = {}
        p.adetailer_xyz[field] = x

    axis = [
        xyz_grid.AxisOption(
            "[ADetailer] ADetailer model 1st",
            str,
            partial(set_value, field="ad_model"),
            choices=lambda: model_list,
        ),
        xyz_grid.AxisOption(
            "[ADetailer] ADetailer prompt 1st",
            str,
            partial(set_value, field="ad_prompt"),
        ),
        xyz_grid.AxisOption(
            "[ADetailer] ADetailer negative prompt 1st",
            str,
            partial(set_value, field="ad_negative_prompt"),
        ),
        xyz_grid.AxisOption(
            "[ADetailer] Mask erosion / dilation 1st",
            int,
            partial(set_value, field="ad_dilate_erode"),
        ),
        xyz_grid.AxisOption(
            "[ADetailer] Inpaint denoising strength 1st",
            float,
            partial(set_value, field="ad_denoising_strength"),
        ),
        xyz_grid.AxisOption(
            "[ADetailer] Inpaint only masked 1st",
            str,
            partial(set_value, field="ad_inpaint_only_masked"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "[ADetailer] Inpaint only masked padding 1st",
            int,
            partial(set_value, field="ad_inpaint_only_masked_padding"),
        ),
        xyz_grid.AxisOption(
            "[ADetailer] ADetailer sampler 1st",
            str,
            partial(set_value, field="ad_sampler"),
            choices=lambda: samplers,
        ),
        xyz_grid.AxisOption(
            "[ADetailer] ControlNet model 1st",
            str,
            partial(set_value, field="ad_controlnet_model"),
            choices=lambda: ["None", *get_cn_models()],
        ),
    ]

    if not any(x.label.startswith("[ADetailer]") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)


def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(
            f"[-] ADetailer: xyz_grid error:\n{error}",
            file=sys.stderr,
        )


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(on_after_component)
script_callbacks.on_before_ui(on_before_ui)
