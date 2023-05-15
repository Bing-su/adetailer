from __future__ import annotations

import platform
import re
import sys
import traceback
from copy import copy, deepcopy
from pathlib import Path
from textwrap import dedent
from typing import Any

import gradio as gr
import torch

import modules  # noqa: F401
from adetailer import (
    AFTER_DETAILER,
    ALL_ARGS,
    ADetailerArgs,
    EnableChecker,
    __version__,
    get_models,
    mediapipe_predict,
    ultralytics_predict,
)
from adetailer.common import mask_preprocess
from adetailer.ui import adui, ordinal, suffix
from controlnet_ext import ControlNetExt, controlnet_exists
from modules import images, safe, script_callbacks, scripts, shared
from modules.paths import data_path, models_path
from modules.processing import (
    StableDiffusionProcessingImg2Img,
    create_infotext,
    process_images,
)
from modules.shared import cmd_opts, opts

try:
    from rich import print
    from rich.traceback import install

    install(show_locals=True)
except Exception:
    pass

no_huggingface = getattr(cmd_opts, "ad_no_huggingface", False)
adetailer_dir = Path(models_path, "adetailer")
model_mapping = get_models(adetailer_dir, huggingface=not no_huggingface)
txt2img_submit_button = img2img_submit_button = None


print(
    f"[-] ADetailer initialized. version: {__version__}, num models: {len(model_mapping)}"
)


class ChangeTorchLoad:
    def __enter__(self):
        self.orig = torch.load
        torch.load = safe.unsafe_torch_load

    def __exit__(self, *args, **kwargs):
        torch.load = self.orig


class AfterDetailerScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.controlnet_ext = None
        self.ultralytics_device = self.get_ultralytics_device()

    def title(self):
        return AFTER_DETAILER

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        num_models = opts.data.get("ad_max_models", 2)
        model_list = list(model_mapping.keys())

        components, infotext_fields = adui(
            num_models,
            is_img2img,
            model_list,
            txt2img_submit_button,
            img2img_submit_button,
        )

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
                p, args.ad_controlnet_model, args.ad_controlnet_weight
            )

    def is_ad_enabled(self, *args_) -> bool:
        if len(args_) == 0 or (len(args_) == 1 and isinstance(args_[0], bool)):
            message = f"""
                       [-] ADetailer: Not enough arguments passed to ADetailer.
                           input: {args_!r}
                       """
            raise ValueError(dedent(message))
        a0 = args_[0]
        a1 = args_[1] if len(args_) > 1 else None
        checker = EnableChecker(a0=a0, a1=a1)
        return checker.is_enabled()

    def get_args(self, *args_) -> list[ADetailerArgs]:
        """
        `args_` is at least 1 in length by `is_ad_enabled` immediately above
        """
        args = args_[1:] if isinstance(args_[0], bool) else args_

        all_inputs = []

        for n, arg_dict in enumerate(args, 1):
            try:
                inp = ADetailerArgs(**arg_dict)
            except ValueError as e:
                message = [
                    f"[-] ADetailer: ValidationError when validating {ordinal(n)} arguments: {e}\n"
                ]
                for attr in ALL_ARGS.attrs:
                    arg = arg_dict.get(attr)
                    dtype = type(arg)
                    arg = "DEFAULT" if arg is None else repr(arg)
                    message.append(f"    {attr}: {arg} ({dtype})")
                raise ValueError("\n".join(message)) from e
            except TypeError as e:
                message = f"[-] ADetailer: {ordinal(n)} - Non-mapping arguments are sent: {arg_dict!r}\n{e}"
                raise TypeError(message) from e

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
        '`device = ""` means autodetect'
        device = ""
        if platform.system() == "Darwin":
            return device

        if any(getattr(cmd_opts, vram, False) for vram in ["lowvram", "medvram"]):
            device = "cpu"

        return device

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
        return prompts

    def get_prompt(self, p, args: ADetailerArgs) -> tuple[list[str], list[str]]:
        i = p._idx

        prompt = self._get_prompt(args.ad_prompt, p.all_prompts, i, p.prompt)
        negative_prompt = self._get_prompt(
            args.ad_negative_prompt, p.all_negative_prompts, i, p.negative_prompt
        )

        return prompt, negative_prompt

    def get_seed(self, p) -> tuple[int, int]:
        i = p._idx

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
        if args.ad_use_steps:
            return args.ad_steps
        return p.steps

    def get_cfg_scale(self, p, args: ADetailerArgs) -> float:
        if args.ad_use_cfg_scale:
            return args.ad_cfg_scale
        return p.cfg_scale

    def infotext(self, p) -> str:
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

        ad_only_seleted_scripts = opts.data.get("ad_only_seleted_scripts", True)
        if not ad_only_seleted_scripts:
            return script_runner, script_args

        default = "dynamic_prompting,dynamic_thresholding,wildcards,wildcard_recursive"
        ad_script_names = opts.data.get("ad_script_names", default)
        script_names_set = {
            name
            for script_name in ad_script_names.split(",")
            for name in (script_name, script_name.strip())
        }
        if args.ad_controlnet_model != "None":
            self.disable_controlnet_units(script_args)
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
            if "controlnet" in obj.__class__.__name__.lower() and hasattr(
                obj, "enabled"
            ):
                obj.enabled = False

    def get_i2i_p(self, p, args: ADetailerArgs, image):
        seed, subseed = self.get_seed(p)
        width, height = self.get_width_height(p, args)
        steps = self.get_steps(p, args)
        cfg_scale = self.get_cfg_scale(p, args)

        sampler_name = p.sampler_name
        if sampler_name in ["PLMS", "UniPC"]:
            sampler_name = "Euler"

        i2i = StableDiffusionProcessingImg2Img(
            init_images=[image],
            resize_mode=0,
            denoising_strength=args.ad_denoising_strength,
            mask=None,
            mask_blur=args.ad_mask_blur,
            inpainting_fill=1,
            inpaint_full_res=args.ad_inpaint_full_res,
            inpaint_full_res_padding=args.ad_inpaint_full_res_padding,
            inpainting_mask_invert=0,
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
            tiling=p.tiling,
            extra_generation_params=p.extra_generation_params,
            do_not_save_samples=True,
            do_not_save_grid=True,
        )

        i2i.scripts, i2i.script_args = self.script_filter(p, args)
        i2i._disable_adetailer = True

        if args.ad_controlnet_model != "None":
            self.update_controlnet_args(i2i, args)
        return i2i

    def save_image(self, p, image, *, condition: str, suffix: str) -> None:
        i = p._idx
        seed, _ = self.get_seed(p)

        if opts.data.get(condition, False):
            images.save_image(
                image=image,
                path=p.outpath_samples,
                basename="",
                seed=seed,
                prompt=p.all_prompts[i] if i < len(p.all_prompts) else p.prompt,
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

    def i2i_prompts_replace(
        self, i2i, prompts: list[str], negative_prompts: list[str], j: int
    ):
        i1 = min(j, len(prompts) - 1)
        i2 = min(j, len(negative_prompts) - 1)
        prompt = prompts[i1]
        negative_prompt = negative_prompts[i2]
        i2i.prompt = prompt
        i2i.negative_prompt = negative_prompt

    def process(self, p, *args_):
        if getattr(p, "_disable_adetailer", False):
            return

        if self.is_ad_enabled(*args_):
            arg_list = self.get_args(*args_)
            extra_params = self.extra_params(arg_list)
            p.extra_generation_params.update(extra_params)

    def _postprocess_image(self, p, pp, args: ADetailerArgs, *, n: int = 0) -> bool:
        """
        Returns
        -------
            bool

            `True` if image was processed, `False` otherwise.
        """
        i = p._idx

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

        with ChangeTorchLoad():
            pred = predictor(ad_model, pp.image, args.ad_conf, **kwargs)

        masks = mask_preprocess(
            pred.masks,
            kernel=args.ad_dilate_erode,
            x_offset=args.ad_x_offset,
            y_offset=args.ad_y_offset,
        )

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

        if is_mediapipe:
            print(f"mediapipe: {steps} detected.")

        p2 = copy(i2i)
        for j in range(steps):
            p2.image_mask = masks[j]
            self.i2i_prompts_replace(p2, ad_prompts, ad_negatives, j)
            processed = process_images(p2)

            p2 = copy(i2i)
            p2.init_images = [processed.images[0]]

            p2.seed = seed + j + 1
            p2.subseed = subseed + j + 1

        if processed is not None:
            pp.image = processed.images[0]
            return True

        return False

    def postprocess_image(self, p, pp, *args_):
        if getattr(p, "_disable_adetailer", False):
            return

        if not self.is_ad_enabled(*args_):
            return

        p._idx = getattr(p, "_idx", -1) + 1
        init_image = copy(pp.image)
        arg_list = self.get_args(*args_)

        is_processed = False
        for n, args in enumerate(arg_list):
            if args.ad_model == "None":
                continue
            is_processed |= self._postprocess_image(p, pp, args, n=n)

        if is_processed:
            self.save_image(
                p, init_image, condition="ad_save_images_before", suffix="-ad-before"
            )

        try:
            if p._idx == len(p.all_prompts) - 1:
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
            component_args={"minimum": 1, "maximum": 5, "step": 1},
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
            default="dynamic_prompting,dynamic_thresholding,wildcards,wildcard_recursive",
            label="Script names to apply to ADetailer (separated by comma)",
            component=gr.Textbox,
            component_args=textbox_args,
            section=section,
        ),
    )


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(on_after_component)
