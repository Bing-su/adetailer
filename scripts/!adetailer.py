from __future__ import annotations

import platform
import sys
from copy import copy, deepcopy
from itertools import zip_longest
from pathlib import Path

import gradio as gr
import torch

import modules  # noqa: F401
from adetailer import (
    ALL_ARGS,
    ADetailerArgs,
    __version__,
    get_args,
    get_models,
    mediapipe_predict,
    ultralytics_predict,
)
from adetailer.common import dilate_erode, is_all_black, offset
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
        return [getattr(self, attr) for attr, *_ in ALL_ARGS]


class ChangeTorchLoad:
    def __enter__(self):
        self.orig = torch.load
        torch.load = safe.unsafe_torch_load

    def __exit__(self, *args, **kwargs):
        torch.load = self.orig


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


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
        model_list = ["None"] + list(model_mapping.keys())

        w = Widgets()

        with gr.Accordion(AFTER_DETAILER, open=False, elem_id="AD_main_acc"):
            with gr.Row():
                w.ad_enable = gr.Checkbox(
                    label="Enable ADetailer",
                    value=True,
                    visible=True,
                )

            with gr.Group():
                with gr.Row():
                    w.ad_model = gr.Dropdown(
                        label="ADetailer model",
                        choices=model_list,
                        value=model_list[0],
                        visible=True,
                        type="value",
                    )

                with gr.Row():
                    w.ad_prompt = gr.Textbox(
                        label="ad_prompt",
                        show_label=False,
                        lines=3,
                        placeholder="ADetailer prompt",
                    )

                with gr.Row():
                    w.ad_negative_prompt = gr.Textbox(
                        label="ad_negative_prompt",
                        show_label=False,
                        lines=2,
                        placeholder="ADetailer negative prompt",
                    )

            with gr.Group():
                with gr.Row():
                    w.ad_conf = gr.Slider(
                        label="ADetailer confidence threshold %",
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=30,
                        visible=True,
                    )
                    w.ad_dilate_erode = gr.Slider(
                        label="ADetailer erosion (-) / dilation (+)",
                        minimum=-128,
                        maximum=128,
                        step=4,
                        value=32,
                        visible=True,
                    )

                with gr.Row():
                    w.ad_x_offset = gr.Slider(
                        label="ADetailer x(→) offset",
                        minimum=-200,
                        maximum=200,
                        step=1,
                        value=0,
                        visible=True,
                    )
                    w.ad_y_offset = gr.Slider(
                        label="ADetailer y(↑) offset",
                        minimum=-200,
                        maximum=200,
                        step=1,
                        value=0,
                        visible=True,
                    )

                with gr.Row():
                    w.ad_mask_blur = gr.Slider(
                        label="ADetailer mask blur",
                        minimum=0,
                        maximum=64,
                        step=1,
                        value=4,
                        visible=True,
                    )

                    w.ad_denoising_strength = gr.Slider(
                        label="ADetailer denoising strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.4,
                        visible=True,
                    )

                with gr.Row():
                    w.ad_inpaint_full_res = gr.Checkbox(
                        label="Inpaint at full resolution ",
                        value=True,
                        visible=True,
                    )
                    w.ad_inpaint_full_res_padding = gr.Slider(
                        label="Inpaint at full resolution padding, pixels ",
                        minimum=0,
                        maximum=256,
                        step=4,
                        value=0,
                        visible=True,
                    )

                with gr.Row():
                    w.ad_use_inpaint_width_height = gr.Checkbox(
                        label="Use inpaint width/height",
                        value=False,
                        visible=True,
                    )

                    w.ad_inpaint_width = gr.Slider(
                        label="inpaint width",
                        minimum=4,
                        maximum=1024,
                        step=4,
                        value=512,
                        visible=True,
                    )

                    w.ad_inpaint_height = gr.Slider(
                        label="inpaint height",
                        minimum=4,
                        maximum=1024,
                        step=4,
                        value=512,
                        visible=True,
                    )

                with gr.Row():
                    w.ad_cfg_scale = gr.Slider(
                        label="ADetailer CFG scale",
                        minimum=0.0,
                        maximum=30.0,
                        step=0.5,
                        value=7.0,
                        visible=True,
                    )

                cn_inpaint_models = ["None"] + get_cn_inpaint_models()

                with gr.Group():
                    with gr.Row():
                        w.ad_controlnet_model = gr.Dropdown(
                            label="ControlNet model",
                            choices=cn_inpaint_models,
                            value="None",
                            visible=True,
                            type="value",
                            interactive=controlnet_exists,
                        )

                    with gr.Row():
                        w.ad_controlnet_weight = gr.Slider(
                            label="ControlNet weight",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=1.0,
                            visible=True,
                            interactive=controlnet_exists,
                        )

        self.infotext_fields = [(getattr(w, attr), name) for attr, name, *_ in ALL_ARGS]

        return w.tolist()

    def init_controlnet_ext(self):
        if self.controlnet_ext is None:
            self.controlnet_ext = ControlNetExt()
            success = self.controlnet_ext.init_controlnet()
            if not success:
                print("[-] ADetailer: ControlNetExt init failed.", file=sys.stderr)

    def is_ad_enabled(self, args: ADetailerArgs):
        return args.ad_enable is True and args.ad_model != "None"

    def get_args(self, *args_):
        try:
            args = get_args(*args_)
        except IndexError as e:
            message = [f"[-] ADetailer: IndexError during get_args: {e}"]
            for arg, (attr, *_) in zip_longest(args_, ALL_ARGS):
                dtype = type(arg)
                arg = "MISSING" if arg is None else repr(arg)
                message.append(f"    {attr}: {arg} ({dtype})")
            raise IndexError("\n".join(message)) from e

        return args

    def extra_params(self, args: ADetailerArgs):
        params = args.extra_params()
        params["ADetailer version"] = __version__
        return params

    @staticmethod
    def get_ultralytics_device():
        '`device = ""` means autodetect'
        device = ""
        if platform.system() == "Darwin":
            return device

        if any(getattr(cmd_opts, vram, False) for vram in ["lowvram", "medvram"]):
            device = "cpu"

        return device

    def get_prompt(self, p, args: ADetailerArgs):
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

    def get_seed(self, p):
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

    def get_width_height(self, p, args: ADetailerArgs):
        if args.ad_use_inpaint_width_height:
            width = args.ad_inpaint_width
            height = args.ad_inpaint_height
        else:
            width = p.width
            height = p.height

        return width, height

    def infotext(self, p):
        return create_infotext(
            p, p.all_prompts, p.all_seeds, p.all_subseeds, None, 0, 0
        )

    def write_params_txt(self, p):
        infotext = self.infotext(p)
        params_txt = Path(data_path, "params.txt")
        params_txt.write_text(infotext, encoding="utf-8")

    def get_i2i_p(self, p, args: ADetailerArgs, image):
        prompt, negative_prompt = self.get_prompt(p, args)
        seed, subseed = self.get_seed(p)
        width, height = self.get_width_height(p, args)

        sampler_name = p.sampler_name
        if sampler_name in ["PLMS", "UniPC"]:
            sampler_name = "Euler"

        self.init_controlnet_ext()

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
            cfg_scale=args.ad_cfg_scale,
            width=width,
            height=height,
            tiling=p.tiling,
            extra_generation_params=p.extra_generation_params,
            do_not_save_samples=True,
            do_not_save_grid=True,
        )

        i2i.scripts = copy(p.scripts)
        i2i.script_args = deepcopy(p.script_args)
        i2i._disable_adetailer = True

        self.update_controlnet_args(i2i, args)
        return i2i

    def save_image(self, p, image, seed, *, condition: str, suffix: str):
        i = p._idx
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

    def update_controlnet_args(self, p, args: ADetailerArgs):
        if (
            self.controlnet_ext is not None
            and self.controlnet_ext.cn_available
            and args.ad_controlnet_model != "None"
        ):
            self.controlnet_ext.update_scripts_args(
                p, args.ad_controlnet_model, args.ad_controlnet_weight
            )

    def process(self, p, *args_):
        if getattr(p, "_disable_adetailer", False):
            return

        args = self.get_args(*args_)
        if self.is_ad_enabled(args):
            extra_params = self.extra_params(args)
            p.extra_generation_params.update(extra_params)

    def postprocess_image(self, p, pp, *args_):
        if getattr(p, "_disable_adetailer", False):
            return

        args = self.get_args(*args_)

        if not self.is_ad_enabled(args):
            return

        p._idx = getattr(p, "_idx", -1) + 1
        i = p._idx

        i2i = self.get_i2i_p(p, args, pp.image)
        seed, subseed = self.get_seed(p)

        self.save_image(
            p, pp.image, seed, condition="ad_save_images_before", suffix="-ad-before"
        )

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

        if pred.masks is None:
            print(
                f"[-] ADetailer: nothing detected on image {i + 1} with current settings."
            )
            return

        self.save_image(
            p, pred.preview, seed, condition="ad_save_previews", suffix="-ad-preview"
        )

        masks = pred.masks
        steps = len(masks)
        processed = None

        if is_mediapipe:
            print(f"mediapipe: {steps} detected.")

        p2 = copy(i2i)
        for j in range(steps):
            mask = masks[j]
            mask = dilate_erode(mask, args.ad_dilate_erode)

            if not is_all_black(mask):
                mask = offset(mask, args.ad_x_offset, args.ad_y_offset)
                p2.image_mask = mask
                processed = process_images(p2)

                p2 = copy(i2i)
                p2.init_images = [processed.images[0]]

            p2.seed = seed + j + 1
            p2.subseed = subseed + j + 1

        if processed is not None:
            pp.image = processed.images[0]

        try:
            if i == len(p.all_prompts) - 1:
                self.write_params_txt(p)
        except Exception:
            pass


def on_ui_settings():
    section = ("ADetailer", AFTER_DETAILER)
    shared.opts.add_option(
        "ad_save_previews",
        shared.OptionInfo(False, "Save mask previews", section=section),
    )

    shared.opts.add_option(
        "ad_save_images_before",
        shared.OptionInfo(False, "Save images before ADetailer", section=section),
    )


script_callbacks.on_ui_settings(on_ui_settings)
