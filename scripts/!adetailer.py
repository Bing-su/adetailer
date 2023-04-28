from __future__ import annotations

import platform
from copy import copy
from pathlib import Path

import gradio as gr
import torch

import modules
from adetailer import __version__, get_models, mediapipe_predict, ultralytics_predict
from adetailer.common import dilate_erode, is_all_black, offset
from modules import images, safe, script_callbacks, scripts, shared
from modules.paths import data_path, models_path
from modules.processing import (
    StableDiffusionProcessingImg2Img,
    create_infotext,
    process_images,
)
from modules.shared import cmd_opts, opts, state

AFTER_DETAILER = "After Detailer"
adetailer_dir = Path(models_path, "adetailer")
model_mapping = get_models(adetailer_dir)

print(
    f"[-] ADetailer initialized. version: {__version__}, num models: {len(model_mapping)}"
)


class ADetailerArgs:
    """
    manualy implemented dataclass for adetailer args, due to some issue...
    see: https://github.com/mkdocs/mkdocs/issues/3141
    """

    def __init__(self, *args):
        self.ad_model: str = args[0]
        self.ad_prompt: str = args[1]
        self.ad_negative_prompt: str = args[2]
        self.ad_conf: float = args[3] / 100.0
        self.ad_dilate_erode: int = args[4]
        self.ad_x_offset: int = args[5]
        self.ad_y_offset: int = args[6]
        self.ad_mask_blur: int = args[7]
        self.ad_denoising_strength: float = args[8]
        self.ad_inpaint_full_res: bool = args[9]
        self.ad_inpaint_full_res_padding: int = args[10]
        self.ad_cfg_scale: float = args[11]

    def asdict(self):
        return {
            "ad_model": self.ad_model,
            "ad_prompt": self.ad_prompt,
            "ad_negative_prompt": self.ad_negative_prompt,
            "ad_conf": self.ad_conf,
            "ad_dilate_erode": self.ad_dilate_erode,
            "ad_x_offset": self.ad_x_offset,
            "ad_y_offset": self.ad_y_offset,
            "ad_mask_blur": self.ad_mask_blur,
            "ad_denoising_strength": self.ad_denoising_strength,
            "ad_inpaint_full_res": self.ad_inpaint_full_res,
            "ad_inpaint_full_res_padding": self.ad_inpaint_full_res_padding,
            "ad_cfg_scale": self.ad_cfg_scale,
        }


class ChangeTorchLoad:
    def __enter__(self):
        self.orig = torch.load
        torch.load = safe.unsafe_torch_load

    def __exit__(self, *args, **kwargs):
        torch.load = self.orig


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


class AfterDetailerScript(scripts.Script):
    def title(self):
        return AFTER_DETAILER

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        model_list = ["None"] + list(model_mapping.keys())

        with gr.Accordion(AFTER_DETAILER, open=False, elem_id="AD_main_acc"):
            with gr.Group():
                with gr.Row():
                    ad_model = gr.Dropdown(
                        label="ADetailer model",
                        choices=model_list,
                        value="None",
                        visible=True,
                        type="value",
                    )

                with gr.Row():
                    ad_prompt = gr.Textbox(
                        label="ad_prompt",
                        show_label=False,
                        lines=3,
                        placeholder="ADetailer prompt",
                    )

                with gr.Row():
                    ad_negative_prompt = gr.Textbox(
                        label="ad_negative_prompt",
                        show_label=False,
                        lines=2,
                        placeholder="ADetailer negative prompt",
                    )

            with gr.Group():
                with gr.Row():
                    ad_conf = gr.Slider(
                        label="ADetailer confidence threshold %",
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=30,
                        visible=True,
                    )
                    ad_dilate_erode = gr.Slider(
                        label="ADetailer erosion (-) / dilation (+)",
                        minimum=-128,
                        maximum=128,
                        step=4,
                        value=32,
                        visible=True,
                    )

                with gr.Row():
                    ad_x_offset = gr.Slider(
                        label="ADetailer x(→) offset",
                        minimum=-200,
                        maximum=200,
                        step=1,
                        value=0,
                        visible=True,
                    )
                    ad_y_offset = gr.Slider(
                        label="ADetailer y(↑) offset",
                        minimum=-200,
                        maximum=200,
                        step=1,
                        value=0,
                        visible=True,
                    )

                with gr.Row():
                    ad_mask_blur = gr.Slider(
                        label="ADetailer mask blur",
                        minimum=0,
                        maximum=64,
                        step=1,
                        value=4,
                        visible=True,
                    )

                    ad_denoising_strength = gr.Slider(
                        label="ADetailer denoising strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.4,
                        visible=True,
                    )

                with gr.Row():
                    ad_inpaint_full_res = gr.Checkbox(
                        label="Inpaint at full resolution ",
                        value=True,
                        visible=True,
                    )
                    ad_inpaint_full_res_padding = gr.Slider(
                        label="Inpaint at full resolution padding, pixels ",
                        minimum=0,
                        maximum=256,
                        step=4,
                        value=0,
                        visible=True,
                    )

                with gr.Row():
                    ad_cfg_scale = gr.Slider(
                        label="ADetailer CFG scale",
                        minimum=0.0,
                        maximum=30.0,
                        step=0.5,
                        value=7.0,
                        visible=True,
                    )

        all_widgets = [
            ad_model,
            ad_prompt,
            ad_negative_prompt,
            ad_conf,
            ad_dilate_erode,
            ad_x_offset,
            ad_y_offset,
            ad_mask_blur,
            ad_denoising_strength,
            ad_inpaint_full_res,
            ad_inpaint_full_res_padding,
            ad_cfg_scale,
        ]

        def on_ad_model_change(model_name):
            visible = model_name != "None"
            return {widget: gr_show(visible) for widget in all_widgets[1:]}

        ad_model.change(on_ad_model_change, inputs=[ad_model], outputs=all_widgets[1:])

        self.infotext_fields = [
            (ad_model, "ADetailer model"),
            (ad_prompt, "ADetailer prompt"),
            (ad_negative_prompt, "ADetailer negative prompt"),
            (ad_conf, "ADetailer conf"),
            (ad_dilate_erode, "ADetailer dilate/erode"),
            (ad_x_offset, "ADetailer x offset"),
            (ad_y_offset, "ADetailer y offset"),
            (ad_mask_blur, "ADetailer mask blur"),
            (ad_denoising_strength, "ADetailer denoising strength"),
            (ad_inpaint_full_res, "ADetailer inpaint full"),
            (ad_inpaint_full_res_padding, "ADetailer inpaint padding"),
            (ad_cfg_scale, "ADetailer CFG scale"),
        ]

        return all_widgets

    @staticmethod
    def extra_params(
        ad_model,
        ad_prompt,
        ad_negative_prompt,
        ad_conf,
        ad_dilate_erode,
        ad_x_offset,
        ad_y_offset,
        ad_mask_blur,
        ad_denoising_strength,
        ad_inpaint_full_res,
        ad_inpaint_full_res_padding,
        ad_cfg_scale,
    ):
        params = {
            "ADetailer model": ad_model,
            "ADetailer prompt": ad_prompt,
            "ADetailer negative prompt": ad_negative_prompt,
            "ADetailer conf": int(ad_conf * 100),
            "ADetailer dilate/erode": ad_dilate_erode,
            "ADetailer x offset": ad_x_offset,
            "ADetailer y offset": ad_y_offset,
            "ADetailer mask blur": ad_mask_blur,
            "ADetailer denoising strength": ad_denoising_strength,
            "ADetailer inpaint full": ad_inpaint_full_res,
            "ADetailer inpaint padding": ad_inpaint_full_res_padding,
            "ADetailer CFG scale": ad_cfg_scale,
            "ADetailer version": __version__,
        }

        if not ad_prompt:
            params.pop("ADetailer prompt")
        if not ad_negative_prompt:
            params.pop("ADetailer negative prompt")

        return params

    @staticmethod
    def get_args(*args):
        return ADetailerArgs(*args)

    @staticmethod
    def get_ultralytics_device():
        '`device = ""` means autodetect'
        device = ""
        if platform.system() == "Darwin":
            return device

        if any(getattr(cmd_opts, vram, False) for vram in ["lowvram", "medvram"]):
            device = "cpu"

        return device

    def get_prompt(self, p, args):
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

    def infotext(self, p):
        return create_infotext(
            p, p.all_prompts, p.all_seeds, p.all_subseeds, None, 0, 0
        )

    def write_params_txt(self, p):
        infotext = self.infotext(p)
        params_txt = Path(data_path, "params.txt")
        params_txt.write_text(infotext, encoding="utf-8")

    def process(self, p, *args_):
        args = self.get_args(*args_)
        if args.ad_model != "None":
            extra_params = self.extra_params(**args.asdict())
            p.extra_generation_params.update(extra_params)

    def postprocess_image(self, p, pp, *args_):
        if getattr(p, "_disable_adetailer", False):
            return

        args = self.get_args(*args_)

        if args.ad_model == "None":
            return

        p._idx = getattr(p, "_idx", -1) + 1
        i = p._idx

        prompt, negative_prompt = self.get_prompt(p, args)
        seed, subseed = self.get_seed(p)

        sampler_name = p.sampler_name
        if sampler_name in ["PLMS", "UniPC"]:
            sampler_name = "Euler"

        i2i = StableDiffusionProcessingImg2Img(
            init_images=[pp.image],
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
            width=p.width,
            height=p.height,
            tiling=p.tiling,
            extra_generation_params=p.extra_generation_params,
            do_not_save_samples=True,
            do_not_save_grid=True,
        )

        i2i.scripts = p.scripts
        i2i.script_args = p.script_args
        i2i._disable_adetailer = True

        kwargs = {}
        if args.ad_model.lower().startswith("mediapipe"):
            predictor = mediapipe_predict
            ad_model = args.ad_model
        else:
            predictor = ultralytics_predict
            ad_model = model_mapping[args.ad_model]
            kwargs["device"] = self.get_ultralytics_device()

        with ChangeTorchLoad():
            pred = predictor(ad_model, pp.image, args.ad_conf, **kwargs)

        if pred.masks is None:
            print(
                f"[-] ADetailer: nothing detected on image {i + 1} with current settings."
            )
            return

        if opts.data.get("ad_save_previews", False):
            images.save_image(
                image=pred.preview,
                path=p.outpath_samples,
                basename="",
                seed=seed,
                prompt=p.all_prompts[i],
                extension=opts.samples_format,
                info=self.infotext(p),
                p=p,
                suffix="-ad-preview",
            )

        masks = pred.masks
        steps = len(masks)
        processed = None

        if args.ad_model.lower().startswith("mediapipe"):
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
                p2.init_images = processed.images

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


script_callbacks.on_ui_settings(on_ui_settings)
