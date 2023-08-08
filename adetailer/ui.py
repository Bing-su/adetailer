from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import Any

import gradio as gr

from adetailer import AFTER_DETAILER, __version__
from adetailer.args import AD_ENABLE, ALL_ARGS, MASK_MERGE_INVERT
from controlnet_ext import controlnet_exists, get_cn_models

cn_module_choices = [
    "inpaint_global_harmonious",
    "inpaint_only",
    "inpaint_only+lama",
]


class Widgets(SimpleNamespace):
    def tolist(self):
        return [getattr(self, attr) for attr in ALL_ARGS.attrs]


def gr_interactive(value: bool = True):
    return gr.update(interactive=value)


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


def suffix(n: int, c: str = " ") -> str:
    return "" if n == 0 else c + ordinal(n + 1)


def on_widget_change(state: dict, value: Any, *, attr: str):
    state[attr] = value
    return state


def on_generate_click(state: dict, *values: Any):
    for attr, value in zip(ALL_ARGS.attrs, values):
        state[attr] = value
    state["is_api"] = ()
    return state


def on_cn_model_update(cn_model: str):
    if "inpaint" in cn_model:
        return gr.update(
            visible=True, choices=cn_module_choices, value=cn_module_choices[0]
        )
    return gr.update(visible=False, choices=["None"], value="None")


def elem_id(item_id: str, n: int, is_img2img: bool) -> str:
    tap = "img2img" if is_img2img else "txt2img"
    suf = suffix(n, "_")
    return f"script_{tap}_adetailer_{item_id}{suf}"


def adui(
    num_models: int,
    is_img2img: bool,
    model_list: list[str],
    samplers: list[str],
    t2i_button: gr.Button,
    i2i_button: gr.Button,
):
    states = []
    infotext_fields = []
    eid = partial(elem_id, n=0, is_img2img=is_img2img)

    with gr.Accordion(AFTER_DETAILER, open=False, elem_id=eid("ad_main_accordion")):
        with gr.Row():
            with gr.Column(scale=6):
                ad_enable = gr.Checkbox(
                    label="Enable ADetailer",
                    value=False,
                    visible=True,
                    elem_id=eid("ad_enable"),
                )

            with gr.Column(scale=1, min_width=180):
                gr.Markdown(
                    f"v{__version__}",
                    elem_id=eid("ad_version"),
                )

        infotext_fields.append((ad_enable, AD_ENABLE.name))

        with gr.Group(), gr.Tabs():
            for n in range(num_models):
                with gr.Tab(ordinal(n + 1)):
                    state, infofields = one_ui_group(
                        n=n,
                        is_img2img=is_img2img,
                        model_list=model_list,
                        samplers=samplers,
                        t2i_button=t2i_button,
                        i2i_button=i2i_button,
                    )

                states.append(state)
                infotext_fields.extend(infofields)

    # components: [bool, dict, dict, ...]
    components = [ad_enable, *states]
    return components, infotext_fields


def one_ui_group(
    n: int,
    is_img2img: bool,
    model_list: list[str],
    samplers: list[str],
    t2i_button: gr.Button,
    i2i_button: gr.Button,
):
    w = Widgets()
    state = gr.State({})
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    with gr.Row():
        model_choices = [*model_list, "None"] if n == 0 else ["None", *model_list]

        w.ad_model = gr.Dropdown(
            label="ADetailer model" + suffix(n),
            choices=model_choices,
            value=model_choices[0],
            visible=True,
            type="value",
            elem_id=eid("ad_model"),
        )

    with gr.Group():
        with gr.Row(elem_id=eid("ad_toprow_prompt")):
            w.ad_prompt = gr.Textbox(
                label="ad_prompt" + suffix(n),
                show_label=False,
                lines=3,
                placeholder="ADetailer prompt"
                + suffix(n)
                + "\nIf blank, the main prompt is used.",
                elem_id=eid("ad_prompt"),
            )

        with gr.Row(elem_id=eid("ad_toprow_negative_prompt")):
            w.ad_negative_prompt = gr.Textbox(
                label="ad_negative_prompt" + suffix(n),
                show_label=False,
                lines=2,
                placeholder="ADetailer negative prompt"
                + suffix(n)
                + "\nIf blank, the main negative prompt is used.",
                elem_id=eid("ad_negative_prompt"),
            )

    with gr.Group():
        with gr.Accordion(
            "Detection", open=False, elem_id=eid("ad_detection_accordion")
        ):
            detection(w, n, is_img2img)

        with gr.Accordion(
            "Mask Preprocessing",
            open=False,
            elem_id=eid("ad_mask_preprocessing_accordion"),
        ):
            mask_preprocessing(w, n, is_img2img)

        with gr.Accordion(
            "Inpainting", open=False, elem_id=eid("ad_inpainting_accordion")
        ):
            inpainting(w, n, is_img2img, samplers)

    with gr.Group():
        controlnet(w, n, is_img2img)

    all_inputs = [state, *w.tolist()]
    target_button = i2i_button if is_img2img else t2i_button
    target_button.click(
        fn=on_generate_click, inputs=all_inputs, outputs=state, queue=False
    )

    infotext_fields = [(getattr(w, attr), name + suffix(n)) for attr, name in ALL_ARGS]

    return state, infotext_fields


def detection(w: Widgets, n: int, is_img2img: bool):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    with gr.Row():
        with gr.Column():
            w.ad_confidence = gr.Slider(
                label="Detection model confidence threshold" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.3,
                visible=True,
                elem_id=eid("ad_confidence"),
            )

        with gr.Column(variant="compact"):
            w.ad_mask_min_ratio = gr.Slider(
                label="Mask min area ratio" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                value=0.0,
                visible=True,
                elem_id=eid("ad_mask_min_ratio"),
            )
            w.ad_mask_max_ratio = gr.Slider(
                label="Mask max area ratio" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                value=1.0,
                visible=True,
                elem_id=eid("ad_mask_max_ratio"),
            )


def mask_preprocessing(w: Widgets, n: int, is_img2img: bool):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    with gr.Group():
        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_x_offset = gr.Slider(
                    label="Mask x(→) offset" + suffix(n),
                    minimum=-200,
                    maximum=200,
                    step=1,
                    value=0,
                    visible=True,
                    elem_id=eid("ad_x_offset"),
                )
                w.ad_y_offset = gr.Slider(
                    label="Mask y(↑) offset" + suffix(n),
                    minimum=-200,
                    maximum=200,
                    step=1,
                    value=0,
                    visible=True,
                    elem_id=eid("ad_y_offset"),
                )

            with gr.Column(variant="compact"):
                w.ad_dilate_erode = gr.Slider(
                    label="Mask erosion (-) / dilation (+)" + suffix(n),
                    minimum=-128,
                    maximum=128,
                    step=4,
                    value=4,
                    visible=True,
                    elem_id=eid("ad_dilate_erode"),
                )

        with gr.Row():
            w.ad_mask_merge_invert = gr.Radio(
                label="Mask merge mode" + suffix(n),
                choices=MASK_MERGE_INVERT,
                value="None",
                elem_id=eid("ad_mask_merge_invert"),
            )


def inpainting(w: Widgets, n: int, is_img2img: bool, samplers: list[str]):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    with gr.Group():
        with gr.Row():
            w.ad_mask_blur = gr.Slider(
                label="Inpaint mask blur" + suffix(n),
                minimum=0,
                maximum=64,
                step=1,
                value=4,
                visible=True,
                elem_id=eid("ad_mask_blur"),
            )

            w.ad_denoising_strength = gr.Slider(
                label="Inpaint denoising strength" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.4,
                visible=True,
                elem_id=eid("ad_denoising_strength"),
            )

        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_inpaint_only_masked = gr.Checkbox(
                    label="Inpaint only masked" + suffix(n),
                    value=True,
                    visible=True,
                    elem_id=eid("ad_inpaint_only_masked"),
                )
                w.ad_inpaint_only_masked_padding = gr.Slider(
                    label="Inpaint only masked padding, pixels" + suffix(n),
                    minimum=0,
                    maximum=256,
                    step=4,
                    value=32,
                    visible=True,
                    elem_id=eid("ad_inpaint_only_masked_padding"),
                )

                w.ad_inpaint_only_masked.change(
                    gr_interactive,
                    inputs=w.ad_inpaint_only_masked,
                    outputs=w.ad_inpaint_only_masked_padding,
                    queue=False,
                )

            with gr.Column(variant="compact"):
                w.ad_use_inpaint_width_height = gr.Checkbox(
                    label="Use separate width/height" + suffix(n),
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_inpaint_width_height"),
                )

                w.ad_inpaint_width = gr.Slider(
                    label="inpaint width" + suffix(n),
                    minimum=64,
                    maximum=2048,
                    step=4,
                    value=512,
                    visible=True,
                    elem_id=eid("ad_inpaint_width"),
                )

                w.ad_inpaint_height = gr.Slider(
                    label="inpaint height" + suffix(n),
                    minimum=64,
                    maximum=2048,
                    step=4,
                    value=512,
                    visible=True,
                    elem_id=eid("ad_inpaint_height"),
                )

                w.ad_use_inpaint_width_height.change(
                    lambda value: (gr_interactive(value), gr_interactive(value)),
                    inputs=w.ad_use_inpaint_width_height,
                    outputs=[w.ad_inpaint_width, w.ad_inpaint_height],
                    queue=False,
                )

        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_use_steps = gr.Checkbox(
                    label="Use separate steps" + suffix(n),
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_steps"),
                )

                w.ad_steps = gr.Slider(
                    label="ADetailer steps" + suffix(n),
                    minimum=1,
                    maximum=150,
                    step=1,
                    value=28,
                    visible=True,
                    elem_id=eid("ad_steps"),
                )

                w.ad_use_steps.change(
                    gr_interactive,
                    inputs=w.ad_use_steps,
                    outputs=w.ad_steps,
                    queue=False,
                )

            with gr.Column(variant="compact"):
                w.ad_use_cfg_scale = gr.Checkbox(
                    label="Use separate CFG scale" + suffix(n),
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_cfg_scale"),
                )

                w.ad_cfg_scale = gr.Slider(
                    label="ADetailer CFG scale" + suffix(n),
                    minimum=0.0,
                    maximum=30.0,
                    step=0.5,
                    value=7.0,
                    visible=True,
                    elem_id=eid("ad_cfg_scale"),
                )

                w.ad_use_cfg_scale.change(
                    gr_interactive,
                    inputs=w.ad_use_cfg_scale,
                    outputs=w.ad_cfg_scale,
                    queue=False,
                )

        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_use_sampler = gr.Checkbox(
                    label="Use separate sampler" + suffix(n),
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_sampler"),
                )

                w.ad_sampler = gr.Dropdown(
                    label="ADetailer sampler" + suffix(n),
                    choices=samplers,
                    value=samplers[0],
                    visible=True,
                    elem_id=eid("ad_sampler"),
                )

                w.ad_use_sampler.change(
                    gr_interactive,
                    inputs=w.ad_use_sampler,
                    outputs=w.ad_sampler,
                    queue=False,
                )

            with gr.Column(variant="compact"):
                w.ad_use_noise_multiplier = gr.Checkbox(
                    label="Use separate noise multiplier" + suffix(n),
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_noise_multiplier"),
                )

                w.ad_noise_multiplier = gr.Slider(
                    label="Noise multiplier for img2img" + suffix(n),
                    minimum=0.5,
                    maximum=1.5,
                    step=0.01,
                    value=1.0,
                    visible=True,
                    elem_id=eid("ad_noise_multiplier"),
                )

                w.ad_use_noise_multiplier.change(
                    gr_interactive,
                    inputs=w.ad_use_noise_multiplier,
                    outputs=w.ad_noise_multiplier,
                    queue=False,
                )

        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_use_clip_skip = gr.Checkbox(
                    label="Use separate CLIP skip" + suffix(n),
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_clip_skip"),
                )

                w.ad_clip_skip = gr.Slider(
                    label="ADetailer CLIP skip" + suffix(n),
                    minimum=1,
                    maximum=12,
                    step=1,
                    value=1,
                    visible=True,
                    elem_id=eid("ad_clip_skip"),
                )

                w.ad_use_clip_skip.change(
                    gr_interactive,
                    inputs=w.ad_use_clip_skip,
                    outputs=w.ad_clip_skip,
                    queue=False,
                )

            with gr.Column(variant="compact"):
                w.ad_restore_face = gr.Checkbox(
                    label="Restore faces after ADetailer" + suffix(n),
                    value=False,
                    elem_id=eid("ad_restore_face"),
                )


def controlnet(w: Widgets, n: int, is_img2img: bool):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)
    cn_models = ["None", *get_cn_models()]

    with gr.Row(variant="panel"):
        with gr.Column(variant="compact"):
            w.ad_controlnet_model = gr.Dropdown(
                label="ControlNet model" + suffix(n),
                choices=cn_models,
                value="None",
                visible=True,
                type="value",
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_model"),
            )

            w.ad_controlnet_module = gr.Dropdown(
                label="ControlNet module" + suffix(n),
                choices=cn_module_choices,
                value="inpaint_global_harmonious",
                visible=False,
                type="value",
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_module"),
            )

            w.ad_controlnet_weight = gr.Slider(
                label="ControlNet weight" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=1.0,
                visible=True,
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_weight"),
            )

            w.ad_controlnet_model.change(
                on_cn_model_update,
                inputs=w.ad_controlnet_model,
                outputs=w.ad_controlnet_module,
                queue=False,
            )

        with gr.Column(variant="compact"):
            w.ad_controlnet_guidance_start = gr.Slider(
                label="ControlNet guidance start" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.0,
                visible=True,
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_guidance_start"),
            )

            w.ad_controlnet_guidance_end = gr.Slider(
                label="ControlNet guidance end" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=1.0,
                visible=True,
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_guidance_end"),
            )
