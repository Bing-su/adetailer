from __future__ import annotations

import platform
import sys
import traceback
from copy import copy, deepcopy
from itertools import zip_longest
from pathlib import Path
from textwrap import dedent

import gradio as gr
import torch

import modules  # noqa: F401
from adetailer import (
    ALL_ARGS,
    ADetailerArgs,
    EnableChecker,
    __version__,
    get_models,
    get_one_args,
    mediapipe_predict,
    ultralytics_predict,
)
from adetailer.common import mask_preprocess
from controlnet_ext import ControlNetExt, controlnet_exists, get_cn_inpaint_models
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

AFTER_DETAILER = "After Detailer"
adetailer_dir = Path(models_path, "adetailer")
model_mapping = get_models(adetailer_dir)

print(
    f"[-] ADetailer initialized. version: {__version__}, num models: {len(model_mapping)}"
)


class Widgets:
    def tolist(self):
        return [getattr(self, attr) for attr, *_ in ALL_ARGS[1:]]


class ChangeTorchLoad:
    def __enter__(self):
        self.orig = torch.load
        torch.load = safe.unsafe_torch_load

    def __exit__(self, *args, **kwargs):
        torch.load = self.orig


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


def suffix(n: int, c: str = " ") -> str:
    return "" if n == 0 else c + ordinal(n + 1)


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
        model_list = list(model_mapping.keys())

        num_models = opts.data.get("ad_max_models", 2)
        w = [Widgets() for _ in range(num_models)]

        with gr.Accordion(AFTER_DETAILER, open=False, elem_id="AD_main_acc"):
            with gr.Row():
                ad_enable = gr.Checkbox(
                    label="Enable ADetailer",
                    value=False,
                    visible=True,
                )

            with gr.Group(), gr.Tabs():
                for n in range(num_models):
                    with gr.Tab(ordinal(n + 1)):
                        with gr.Row():
                            model_choices = (
                                model_list if n == 0 else ["None"] + model_list
                            )

                            w[n].ad_model = gr.Dropdown(
                                label="ADetailer model" + suffix(n),
                                choices=model_choices,
                                value=model_choices[0],
                                visible=True,
                                type="value",
                            )

                        with gr.Row(elem_id="AD_toprow_prompt" + suffix(n, "_")):
                            w[n].ad_prompt = gr.Textbox(
                                label="ad_prompt" + suffix(n),
                                show_label=False,
                                lines=3,
                                placeholder="ADetailer prompt" + suffix(n),
                                elem_id="AD_prompt" + suffix(n, "_"),
                            )

                        with gr.Row(
                            elem_id="AD_toprow_negative_prompt" + suffix(n, "_")
                        ):
                            w[n].ad_negative_prompt = gr.Textbox(
                                label="ad_negative_prompt" + suffix(n),
                                show_label=False,
                                lines=2,
                                placeholder="ADetailer negative prompt" + suffix(n),
                                elem_id="AD_negative_prompt" + suffix(n, "_"),
                            )

                        with gr.Group():
                            with gr.Row():
                                w[n].ad_conf = gr.Slider(
                                    label="Detection model confidence threshold %"
                                    + suffix(n),
                                    minimum=0,
                                    maximum=100,
                                    step=1,
                                    value=30,
                                    visible=True,
                                )
                                w[n].ad_dilate_erode = gr.Slider(
                                    label="Mask erosion (-) / dilation (+)" + suffix(n),
                                    minimum=-128,
                                    maximum=128,
                                    step=4,
                                    value=32,
                                    visible=True,
                                )

                            with gr.Row():
                                w[n].ad_x_offset = gr.Slider(
                                    label="Mask x(→) offset" + suffix(n),
                                    minimum=-200,
                                    maximum=200,
                                    step=1,
                                    value=0,
                                    visible=True,
                                )
                                w[n].ad_y_offset = gr.Slider(
                                    label="Mask y(↑) offset" + suffix(n),
                                    minimum=-200,
                                    maximum=200,
                                    step=1,
                                    value=0,
                                    visible=True,
                                )

                            with gr.Row():
                                w[n].ad_mask_blur = gr.Slider(
                                    label="Inpaint mask blur" + suffix(n),
                                    minimum=0,
                                    maximum=64,
                                    step=1,
                                    value=4,
                                    visible=True,
                                )

                                w[n].ad_denoising_strength = gr.Slider(
                                    label="Inpaint denoising strength" + suffix(n),
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    value=0.4,
                                    visible=True,
                                )

                            with gr.Row():
                                w[n].ad_inpaint_full_res = gr.Checkbox(
                                    label="Inpaint at full resolution " + suffix(n),
                                    value=True,
                                    visible=True,
                                )
                                w[n].ad_inpaint_full_res_padding = gr.Slider(
                                    label="Inpaint at full resolution padding, pixels "
                                    + suffix(n),
                                    minimum=0,
                                    maximum=256,
                                    step=4,
                                    value=0,
                                    visible=True,
                                )

                            with gr.Row():
                                w[n].ad_use_inpaint_width_height = gr.Checkbox(
                                    label="Use separate width/height" + suffix(n),
                                    value=False,
                                    visible=True,
                                )

                                w[n].ad_inpaint_width = gr.Slider(
                                    label="inpaint width" + suffix(n),
                                    minimum=64,
                                    maximum=2048,
                                    step=4,
                                    value=512,
                                    visible=True,
                                )

                                w[n].ad_inpaint_height = gr.Slider(
                                    label="inpaint height" + suffix(n),
                                    minimum=64,
                                    maximum=2048,
                                    step=4,
                                    value=512,
                                    visible=True,
                                )

                            with gr.Row():
                                w[n].ad_use_cfg_scale = gr.Checkbox(
                                    label="Use separate CFG scale" + suffix(n),
                                    value=False,
                                    visible=True,
                                )

                                w[n].ad_cfg_scale = gr.Slider(
                                    label="ADetailer CFG scale" + suffix(n),
                                    minimum=0.0,
                                    maximum=30.0,
                                    step=0.5,
                                    value=7.0,
                                    visible=True,
                                )

                            cn_inpaint_models = ["None"] + get_cn_inpaint_models()

                            with gr.Group():
                                with gr.Row():
                                    w[n].ad_controlnet_model = gr.Dropdown(
                                        label="ControlNet model" + suffix(n),
                                        choices=cn_inpaint_models,
                                        value="None",
                                        visible=True,
                                        type="value",
                                        interactive=controlnet_exists,
                                    )

                                with gr.Row():
                                    w[n].ad_controlnet_weight = gr.Slider(
                                        label="ControlNet weight" + suffix(n),
                                        minimum=0.0,
                                        maximum=1.0,
                                        step=0.05,
                                        value=1.0,
                                        visible=True,
                                        interactive=controlnet_exists,
                                    )

        # Accordion end

        self.infotext_fields = [(ad_enable, ALL_ARGS[0].name)]
        self.infotext_fields += [
            (getattr(w[n], attr), name + suffix(n))
            for n in range(num_models)
            for attr, name, *_ in ALL_ARGS[1:]
        ]

        ret = [ad_enable]
        for n in range(num_models):
            ret.extend(w[n].tolist())

        return ret

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
        if len(args_) < 2:
            message = f"""
                       [-] ADetailer: Not enough arguments passed to adetailer.
                           input: {args_!r}
                       """
            raise ValueError(dedent(message))
        checker = EnableChecker(ad_enable=args_[0], ad_model=args_[1])
        return checker.is_enabled()

    def get_args(self, *args_) -> list[ADetailerArgs]:
        """
        `args_` is at least 2 in length by `is_ad_enabled` immediately above
        """
        enabled = args_[0]
        rem = args_[1:]
        length = len(ALL_ARGS) - 1

        all_inputs = []
        iter_args = (rem[i : i + length] for i in range(0, len(rem), length))

        for n, args in enumerate(iter_args, 1):
            try:
                inp = get_one_args(enabled, *args)
            except ValueError as e:
                message = [
                    f"[-] ADetailer: ValidationError when validating {ordinal(n)} arguments: {e}\n"
                ]
                for arg, (attr, *_) in zip_longest(args, ALL_ARGS[1:]):
                    dtype = type(arg)
                    arg = "MISSING" if arg is None else repr(arg)
                    message.append(f"    {attr}: {arg} ({dtype})")
                raise ValueError("\n".join(message)) from e

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

    def get_prompt(self, p, args: ADetailerArgs) -> tuple[str, str]:
        i = p._idx

        if args.ad_prompt:
            prompt = args.ad_prompt
        elif not p.all_prompts:
            prompt = p.prompt
        elif i < len(p.all_prompts):
            prompt = p.all_prompts[i]
        else:
            j = i % len(p.all_prompts)
            prompt = p.all_prompts[j]

        if args.ad_negative_prompt:
            negative_prompt = args.ad_negative_prompt
        elif not p.all_negative_prompts:
            negative_prompt = p.negative_prompt
        elif i < len(p.all_negative_prompts):
            negative_prompt = p.all_negative_prompts[i]
        else:
            j = i % len(p.all_negative_prompts)
            negative_prompt = p.all_negative_prompts[j]

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

        ad_only_seleted_scripts = opts.data.get("ad_only_seleted_scripts", True)
        if not ad_only_seleted_scripts:
            return script_runner

        default = "dynamic_prompting,dynamic_thresholding,wildcards,wildcard_recursive"
        ad_script_names = opts.data.get("ad_script_names", default)
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
        return script_runner

    def get_i2i_p(self, p, args: ADetailerArgs, image):
        prompt, negative_prompt = self.get_prompt(p, args)
        seed, subseed = self.get_seed(p)
        width, height = self.get_width_height(p, args)
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
            prompt=prompt,
            negative_prompt=negative_prompt,
            styles=p.styles,
            seed=seed,
            subseed=subseed,
            subseed_strength=p.subseed_strength,
            seed_resize_from_h=p.seed_resize_from_h,
            seed_resize_from_w=p.seed_resize_from_w,
            sampler_name=sampler_name,
            batch_size=1,
            n_iter=1,
            steps=p.steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            tiling=p.tiling,
            extra_generation_params=p.extra_generation_params,
            do_not_save_samples=True,
            do_not_save_grid=True,
        )

        i2i.scripts = self.script_filter(p, args)
        i2i.script_args = deepcopy(p.script_args)
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

        masks = mask_preprocess(pred.masks)

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
