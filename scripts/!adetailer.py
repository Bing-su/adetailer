from __future__ import annotations

from pathlib import Path

import gradio as gr
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt, confloat, validator

from adetailer import __version__, get_models, mediapipe_predict, ultralytics_predict
from adetailer.common import dilate_erode, is_all_black, offset
from modules import devices
from modules.paths import models_path
from modules.processing import StableDiffusionProcessingImg2Img, process_images
from modules.scripts import AlwaysVisible, Script
from modules.shared import opts, state  # noqa: F401

AFTER_DETAILER = "After Detailer"
adetailer_dir = Path(models_path, "adetailer")
model_mapping = get_models(adetailer_dir)


class ADetailerArgs(BaseModel):
    ad_model: str = "None"
    ad_prompt: str = ""
    ad_negative_prompt: str = ""
    ad_conf: confloat(ge=0.0, le=1.0) = 0.25
    ad_dilate_erode: int = 36
    ad_x_offset: int = 0
    ad_y_offset: int = 0
    ad_mask_blur: NonNegativeInt = 4
    ad_denoising_strength: confloat(ge=0.0, le=1.0) = 0.4
    ad_inpaint_full_res: bool = True
    ad_inpaint_full_res_padding: NonNegativeInt = 0
    ad_cfg_scale: NonNegativeFloat = 7.0

    @validator("ad_conf", pre=True)
    def check_ad_conf(cls, v):  # noqa: N805
        return v / 100.0


def with_gc(func):
    def wrapper(*args, **kwargs):
        devices.torch_gc()
        result = func(*args, **kwargs)
        devices.torch_gc()
        return result

    return wrapper


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


class AfterDetailerScript(Script):
    def title(self):
        return AFTER_DETAILER

    def show(self, is_img2img):
        return AlwaysVisible

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
                        value=25,
                        visible=True,
                    )
                    ad_dilate_erode = gr.Slider(
                        label="ADetailer erosion (-) / dilation (+)",
                        minimum=-128,
                        maximum=128,
                        step=4,
                        value=36,
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
    def args_validation(*args):
        return ADetailerArgs(
            ad_model=args[0],
            ad_prompt=args[1],
            ad_negative_prompt=args[2],
            ad_conf=args[3],
            ad_dilate_erode=args[4],
            ad_x_offset=args[5],
            ad_y_offset=args[6],
            ad_mask_blur=args[7],
            ad_denoising_strength=args[8],
            ad_inpaint_full_res=args[9],
            ad_inpaint_full_res_padding=args[10],
            ad_cfg_scale=args[11],
        )

    @with_gc
    def postprocess_image(self, p, pp, *args_):
        if getattr(p, "_disable_adetailer", False):
            return

        args = self.args_validation(*args_)

        if args.ad_model.lower() == "none":
            return

        extra_params = self.extra_params(**args.dict())
        p.extra_generation_params.update(extra_params)
        p._idx = getattr(p, "_idx", -1) + 1
        i = p._idx

        assert hasattr(p, "all_prompts")
        assert hasattr(p, "all_negative_prompts")
        assert len(p.all_prompts) == len(p.all_negative_prompts)
        assert 0 <= i < len(p.all_prompts)
        assert 0 <= i < len(p.all_negative_prompts)

        prompt = args.ad_prompt if args.ad_prompt else p.all_prompts[i]

        if args.ad_negative_prompt:
            negative_prompt = args.ad_negative_prompt
        else:
            negative_prompt = p.all_negative_prompts[i]

        seed = p.all_seeds[i]
        subseed = p.all_subseeds[i]

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

        if args.ad_model.lower().startswith("mediapipe"):
            predictor = mediapipe_predict
            ad_model = args.ad_model
        else:
            predictor = ultralytics_predict
            ad_model = model_mapping[args.ad_model]

        pred = predictor(ad_model, pp.image, args.ad_conf)
        if pred.masks is None:
            print("ADetailer: nothing detected with current settings")
            return

        masks = pred.masks
        steps = len(masks)
        processed = None

        for j in range(steps):
            mask = masks[j]
            mask = dilate_erode(mask, args.ad_dilate_erode)
            if is_all_black(mask):
                continue
            mask = offset(mask, args.ad_x_offset, args.ad_y_offset)

            i2i.image_mask = mask

            processed = process_images(i2i)
            i2i.seed = seed + j + 1
            i2i.subseed = subseed + j + 1
            i2i.init_images = processed.images

        if processed is not None:
            pp.image = processed.images[0]
