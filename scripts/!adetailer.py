from __future__ import annotations

import platform
import re
import sys
import traceback
from copy import copy
from functools import partial
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import gradio as gr
from PIL import Image, ImageChops
from rich import print

import modules
from aaaaaa.conditional import create_binary_mask, schedulers
from aaaaaa.helper import (
    change_torch_load,
    copy_extra_params,
    pause_total_tqdm,
    preseve_prompts,
)
from aaaaaa.p_method import (
    get_i,
    is_img2img_inpaint,
    is_inpaint_only_masked,
    need_call_postprocess,
    need_call_process,
)
from adetailer import (
    AFTER_DETAILER,
    __version__,
    get_models,
    mediapipe_predict,
    ultralytics_predict,
)
from adetailer.args import BBOX_SORTBY, SCRIPT_DEFAULT, ADetailerArgs, SkipImg2ImgOrig
from adetailer.common import PredictOutput, ensure_pil_image, safe_mkdir
from adetailer.mask import (
    filter_by_ratio,
    filter_k_largest,
    has_intersection,
    is_all_black,
    mask_preprocess,
    sort_bboxes,
)
from adetailer.traceback import rich_traceback
from adetailer.ui import WebuiInfo, adui, ordinal, suffix
from controlnet_ext import (
    CNHijackRestore,
    ControlNetExt,
    cn_allow_script_control,
    controlnet_exists,
    controlnet_type,
    get_cn_models,
)
from modules import images, paths, script_callbacks, scripts, shared
from modules.devices import NansException
from modules.processing import (
    Processed,
    StableDiffusionProcessingImg2Img,
    create_infotext,
    process_images,
)
from modules.sd_samplers import all_samplers
from modules.shared import cmd_opts, opts, state

if TYPE_CHECKING:
    from fastapi import FastAPI

PARAMS_TXT = "params.txt"

no_huggingface = getattr(cmd_opts, "ad_no_huggingface", False)
adetailer_dir = Path(paths.models_path, "adetailer")
safe_mkdir(adetailer_dir)

extra_models_dir = shared.opts.data.get("ad_extra_models_dir", "")
model_mapping = get_models(
    adetailer_dir,
    extra_models_dir,
    huggingface=not no_huggingface,
)

txt2img_submit_button = img2img_submit_button = None
txt2img_submit_button = cast(gr.Button, txt2img_submit_button)
img2img_submit_button = cast(gr.Button, img2img_submit_button)

print(
    f"[-] ADetailer initialized. version: {__version__}, num models: {len(model_mapping)}"
)


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
        scheduler_names = [x.label for x in schedulers]

        try:
            checkpoint_list = modules.sd_models.checkpoint_tiles(use_shorts=True)
        except TypeError:
            checkpoint_list = modules.sd_models.checkpoint_tiles()
        vae_list = modules.shared_items.sd_vae_items()

        webui_info = WebuiInfo(
            ad_model_list=ad_model_list,
            sampler_names=sampler_names,
            scheduler_names=scheduler_names,
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
        if not args_ or not arg_list:
            message = f"""
                       [-] ADetailer: Invalid arguments passed to ADetailer.
                           input: {args_!r}
                           ADetailer disabled.
                       """
            print(dedent(message), file=sys.stderr)
            return False

        ad_enabled = args_[0] if isinstance(args_[0], bool) else True
        not_none = any(arg.get("ad_model", "None") != "None" for arg in arg_list)
        return ad_enabled and not_none

    def set_skip_img2img(self, p, *args_) -> None:
        if (
            hasattr(p, "_ad_skip_img2img")
            or not hasattr(p, "init_images")
            or not p.init_images
        ):
            return

        if len(args_) >= 2 and isinstance(args_[1], bool):
            p._ad_skip_img2img = args_[1]
        else:
            p._ad_skip_img2img = False

        if not p._ad_skip_img2img:
            return

        if is_img2img_inpaint(p):
            p._ad_disabled = True
            msg = "[-] ADetailer: img2img inpainting with skip img2img is not supported. (because it's buggy)"
            print(msg)
            return

        p._ad_orig = SkipImg2ImgOrig(
            steps=p.steps,
            sampler_name=p.sampler_name,
            width=p.width,
            height=p.height,
        )
        p.steps = 1
        p.sampler_name = "Euler"
        p.width = 128
        p.height = 128

    def get_args(self, p, *args_) -> list[ADetailerArgs]:
        """
        `args_` is at least 1 in length by `is_ad_enabled` immediately above
        """
        args = [arg for arg in args_ if isinstance(arg, dict)]

        if not args:
            message = f"[-] ADetailer: Invalid arguments passed to ADetailer: {args_!r}"
            raise ValueError(message)

        if hasattr(p, "_ad_xyz"):
            args[0] = {**args[0], **p._ad_xyz}

        all_inputs = []

        for n, arg_dict in enumerate(args, 1):
            try:
                inp = ADetailerArgs(**arg_dict)
            except ValueError as e:
                msg = f"[-] ADetailer: ValidationError when validating {ordinal(n)} arguments"
                if hasattr(e, "add_note"):
                    e.add_note(msg)
                else:
                    print(msg, file=sys.stderr)
                raise

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
        self,
        ad_prompt: str,
        all_prompts: list[str],
        i: int,
        default: str,
        replacements: list[PromptSR],
    ) -> list[str]:
        prompts = re.split(r"\s*\[SEP\]\s*", ad_prompt)
        blank_replacement = self.prompt_blank_replacement(all_prompts, i, default)
        for n in range(len(prompts)):
            if not prompts[n]:
                prompts[n] = blank_replacement
            elif "[PROMPT]" in prompts[n]:
                prompts[n] = prompts[n].replace("[PROMPT]", blank_replacement)

            for pair in replacements:
                prompts[n] = prompts[n].replace(pair.s, pair.r)
        return prompts

    def get_prompt(self, p, args: ADetailerArgs) -> tuple[list[str], list[str]]:
        i = get_i(p)
        prompt_sr = p._ad_xyz_prompt_sr if hasattr(p, "_ad_xyz_prompt_sr") else []

        prompt = self._get_prompt(
            ad_prompt=args.ad_prompt,
            all_prompts=p.all_prompts,
            i=i,
            default=p.prompt,
            replacements=prompt_sr,
        )
        negative_prompt = self._get_prompt(
            ad_prompt=args.ad_negative_prompt,
            all_prompts=p.all_negative_prompts,
            i=i,
            default=p.negative_prompt,
            replacements=prompt_sr,
        )

        return prompt, negative_prompt

    def get_seed(self, p) -> tuple[int, int]:
        i = get_i(p)

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
        elif hasattr(p, "_ad_orig"):
            width = p._ad_orig.width
            height = p._ad_orig.height
        else:
            width = p.width
            height = p.height

        return width, height

    def get_steps(self, p, args: ADetailerArgs) -> int:
        if args.ad_use_steps:
            return args.ad_steps
        if hasattr(p, "_ad_orig"):
            return p._ad_orig.steps
        return p.steps

    def get_cfg_scale(self, p, args: ADetailerArgs) -> float:
        return args.ad_cfg_scale if args.ad_use_cfg_scale else p.cfg_scale

    def get_sampler(self, p, args: ADetailerArgs) -> str:
        if args.ad_use_sampler:
            return args.ad_sampler
        if hasattr(p, "_ad_orig"):
            return p._ad_orig.sampler_name
        return p.sampler_name

    def get_scheduler(self, p, args: ADetailerArgs) -> dict[str, str]:
        "webui >= 1.9.0"
        if not args.ad_use_sampler:
            return {}

        if args.ad_scheduler == "Use same scheduler":
            value = getattr(p, "scheduler", "Automatic")
        else:
            value = args.ad_scheduler
        return {"scheduler": value}

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

    def read_params_txt(self) -> str:
        params_txt = Path(paths.data_path, PARAMS_TXT)
        if params_txt.exists():
            return params_txt.read_text(encoding="utf-8")
        return ""

    def write_params_txt(self, content: str) -> None:
        params_txt = Path(paths.data_path, PARAMS_TXT)
        if params_txt.exists() and content:
            params_txt.write_text(content, encoding="utf-8")

    @staticmethod
    def script_args_copy(script_args):
        type_: type[list] | type[tuple] = type(script_args)
        result = []
        for arg in script_args:
            try:
                a = copy(arg)
            except TypeError:
                a = arg
            result.append(a)
        return type_(result)

    def script_filter(self, p, args: ADetailerArgs):
        script_runner = copy(p.scripts)
        script_args = self.script_args_copy(p.script_args)

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

    def disable_controlnet_units(
        self, script_args: list[Any] | tuple[Any, ...]
    ) -> None:
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

        version_args = {}
        if schedulers:
            version_args.update(self.get_scheduler(p, args))

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
            extra_generation_params=copy_extra_params(p.extra_generation_params),
            do_not_save_samples=True,
            do_not_save_grid=True,
            override_settings=override_settings,
            **version_args,
        )

        i2i.cached_c = [None, None]
        i2i.cached_uc = [None, None]
        i2i.scripts, i2i.script_args = self.script_filter(p, args)
        i2i._ad_disabled = True
        i2i._ad_inner = True

        if args.ad_controlnet_model != "Passthrough" and controlnet_type != "forge":
            self.disable_controlnet_units(i2i.script_args)

        if args.ad_controlnet_model not in ["None", "Passthrough"]:
            self.update_controlnet_args(i2i, args)
        elif args.ad_controlnet_model == "None":
            i2i.control_net_enabled = False

        return i2i

    def save_image(self, p, image, *, condition: str, suffix: str) -> None:
        i = get_i(p)
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

    def pred_preprocessing(self, p, pred: PredictOutput, args: ADetailerArgs):
        pred = filter_by_ratio(
            pred, low=args.ad_mask_min_ratio, high=args.ad_mask_max_ratio
        )
        pred = filter_k_largest(pred, k=args.ad_mask_k_largest)
        pred = self.sort_bboxes(pred)
        masks = mask_preprocess(
            pred.masks,
            kernel=args.ad_dilate_erode,
            x_offset=args.ad_x_offset,
            y_offset=args.ad_y_offset,
            merge_invert=args.ad_mask_merge_invert,
        )

        if is_img2img_inpaint(p) and not is_inpaint_only_masked(p):
            image_mask = self.get_image_mask(p)
            masks = self.inpaint_mask_filter(image_mask, masks)
        return masks

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
    def compare_prompt(p, extra_params: dict[str, Any], processed, n: int = 0):
        if not hasattr(p, "_ad_extra_params_result"):
            p._ad_extra_params_result = {}

        pt = "ADetailer prompt" + suffix(n)
        if pt in extra_params and extra_params[pt] != processed.all_prompts[0]:
            print(
                f"[-] ADetailer: applied {ordinal(n + 1)} ad_prompt: {processed.all_prompts[0]!r}"
            )
            p._ad_extra_params_result[pt] = processed.all_prompts[0]

        ng = "ADetailer negative prompt" + suffix(n)
        if ng in extra_params and extra_params[ng] != processed.all_negative_prompts[0]:
            print(
                f"[-] ADetailer: applied {ordinal(n + 1)} ad_negative_prompt: {processed.all_negative_prompts[0]!r}"
            )
            p._ad_extra_params_result[ng] = processed.all_negative_prompts[0]

    @staticmethod
    def get_i2i_init_image(p, pp):
        if getattr(p, "_ad_skip_img2img", False):
            return p.init_images[0]
        return pp.image

    @staticmethod
    def get_each_tap_seed(seed: int, i: int):
        use_same_seed = shared.opts.data.get("ad_same_seed_for_each_tap", False)
        return seed if use_same_seed else seed + i

    @staticmethod
    def inpaint_mask_filter(
        img2img_mask: Image.Image, ad_mask: list[Image.Image]
    ) -> list[Image.Image]:
        if ad_mask and img2img_mask.size != ad_mask[0].size:
            img2img_mask = img2img_mask.resize(ad_mask[0].size, resample=images.LANCZOS)
        return [mask for mask in ad_mask if has_intersection(img2img_mask, mask)]

    @staticmethod
    def get_image_mask(p) -> Image.Image:
        mask = p.image_mask
        if getattr(p, "inpainting_mask_invert", False):
            mask = ImageChops.invert(mask)
        mask = create_binary_mask(mask)

        if getattr(p, "_ad_skip_img2img", False):
            if hasattr(p, "init_images") and p.init_images:
                width, height = p.init_images[0].size
            else:
                msg = "[-] ADetailer: no init_images."
                raise RuntimeError(msg)
        else:
            width, height = p.width, p.height
        return images.resize_image(p.resize_mode, mask, width, height)

    @rich_traceback
    def process(self, p, *args_):
        if getattr(p, "_ad_disabled", False):
            return

        if is_img2img_inpaint(p) and is_all_black(self.get_image_mask(p)):
            p._ad_disabled = True
            msg = (
                "[-] ADetailer: img2img inpainting with no mask -- adetailer disabled."
            )
            print(msg)
            return

        if not self.is_ad_enabled(*args_):
            p._ad_disabled = True
            return

        self.set_skip_img2img(p, *args_)
        if getattr(p, "_ad_disabled", False):
            # case when img2img inpainting with skip img2img
            return

        arg_list = self.get_args(p, *args_)

        if hasattr(p, "_ad_xyz_prompt_sr"):
            replaced_positive_prompt, replaced_negative_prompt = self.get_prompt(
                p, arg_list[0]
            )
            arg_list[0].ad_prompt = replaced_positive_prompt[0]
            arg_list[0].ad_negative_prompt = replaced_negative_prompt[0]

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

        i = get_i(p)

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
            kwargs["classes"] = args.ad_model_classes

        with change_torch_load():
            pred = predictor(ad_model, pp.image, args.ad_confidence, **kwargs)

        if pred.preview is None:
            print(
                f"[-] ADetailer: nothing detected on image {i + 1} with {ordinal(n + 1)} settings."
            )
            return False

        masks = self.pred_preprocessing(p, pred, args)
        shared.state.assign_current_image(pred.preview)

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
            p2.init_images[0] = ensure_pil_image(p2.init_images[0], "RGB")
            self.i2i_prompts_replace(p2, ad_prompts, ad_negatives, j)

            if re.match(r"^\s*\[SKIP\]\s*$", p2.prompt):
                continue

            p2.seed = self.get_each_tap_seed(seed, j)
            p2.subseed = self.get_each_tap_seed(subseed, j)

            try:
                processed = process_images(p2)
            except NansException as e:
                msg = f"[-] ADetailer: 'NansException' occurred with {ordinal(n + 1)} settings.\n{e}"
                print(msg, file=sys.stderr)
                continue
            finally:
                p2.close()

            self.compare_prompt(p, p.extra_generation_params, processed, n=n)
            p2 = copy(i2i)
            p2.init_images = [processed.images[0]]

        if processed is not None:
            pp.image = processed.images[0]
            return True

        return False

    @rich_traceback
    def postprocess_image(self, p, pp, *args_):
        if getattr(p, "_ad_disabled", False) or not self.is_ad_enabled(*args_):
            return

        pp.image = self.get_i2i_init_image(p, pp)
        pp.image = ensure_pil_image(pp.image, "RGB")
        init_image = copy(pp.image)
        arg_list = self.get_args(p, *args_)
        params_txt_content = self.read_params_txt()

        if need_call_postprocess(p):
            dummy = Processed(p, [], p.seed, "")
            with preseve_prompts(p):
                p.scripts.postprocess(copy(p), dummy)

        is_processed = False
        with CNHijackRestore(), pause_total_tqdm(), cn_allow_script_control():
            for n, args in enumerate(arg_list):
                if args.ad_model == "None":
                    continue
                is_processed |= self._postprocess_image_inner(p, pp, args, n=n)

        if is_processed and not getattr(p, "_ad_skip_img2img", False):
            self.save_image(
                p, init_image, condition="ad_save_images_before", suffix="-ad-before"
            )

        if need_call_process(p):
            with preseve_prompts(p):
                copy_p = copy(p)
                if hasattr(p.scripts, "before_process"):
                    p.scripts.before_process(copy_p)
                p.scripts.process(copy_p)

        self.write_params_txt(params_txt_content)

        if hasattr(p, "_ad_extra_params_result"):
            p.extra_generation_params.update(p._ad_extra_params_result)


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
        "ad_extra_models_dir",
        shared.OptionInfo(
            default="",
            label="Extra path to scan adetailer models",
            component=gr.Textbox,
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

    shared.opts.add_option(
        "ad_same_seed_for_each_tap",
        shared.OptionInfo(
            False, "Use same seed for each tab in adetailer", section=section
        ),
    )


# xyz_grid


class PromptSR(NamedTuple):
    s: str
    r: str


def set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_ad_xyz"):
        p._ad_xyz = {}
    p._ad_xyz[field] = x


def search_and_replace_prompt(p, x: Any, xs: Any, replace_in_main_prompt: bool):
    if replace_in_main_prompt:
        p.prompt = p.prompt.replace(xs[0], x)
        p.negative_prompt = p.negative_prompt.replace(xs[0], x)

    if not hasattr(p, "_ad_xyz_prompt_sr"):
        p._ad_xyz_prompt_sr = []
    p._ad_xyz_prompt_sr.append(PromptSR(s=xs[0], r=x))


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
            "[ADetailer] Prompt S/R (AD 1st)",
            str,
            partial(search_and_replace_prompt, replace_in_main_prompt=False),
        ),
        xyz_grid.AxisOption(
            "[ADetailer] Prompt S/R (AD 1st and main prompt)",
            str,
            partial(search_and_replace_prompt, replace_in_main_prompt=True),
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
            choices=lambda: ["None", "Passthrough", *get_cn_models()],
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


# api


def add_api_endpoints(_: gr.Blocks, app: FastAPI):
    @app.get("/adetailer/v1/version")
    async def version():
        return {"version": __version__}

    @app.get("/adetailer/v1/schema")
    async def schema():
        if hasattr(ADetailerArgs, "model_json_schema"):
            return ADetailerArgs.model_json_schema()
        return ADetailerArgs.schema()

    @app.get("/adetailer/v1/ad_model")
    async def ad_model():
        return {"ad_model": list(model_mapping)}


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(on_after_component)
script_callbacks.on_app_started(add_api_endpoints)
script_callbacks.on_before_ui(on_before_ui)
