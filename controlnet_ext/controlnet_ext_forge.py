from __future__ import annotations

import copy

import numpy as np
from lib_controlnet import external_code, global_state
from lib_controlnet.external_code import ControlNetUnit

from modules import scripts
from modules.processing import StableDiffusionProcessing

from .common import cn_model_regex

controlnet_exists = True
controlnet_type = "forge"


def find_script(p: StableDiffusionProcessing, script_title: str) -> scripts.Script:
    script = next((s for s in p.scripts.scripts if s.title() == script_title), None)
    if not script:
        msg = f"Script not found: {script_title!r}"
        raise RuntimeError(msg)
    return script


def add_forge_script_to_adetailer_run(
    p: StableDiffusionProcessing, script_title: str, script_args: list
):
    p.scripts = copy.copy(scripts.scripts_img2img)
    p.scripts.alwayson_scripts = []
    p.script_args_value = []

    script = copy.copy(find_script(p, script_title))
    script.args_from = len(p.script_args_value)
    script.args_to = len(p.script_args_value) + len(script_args)
    p.scripts.alwayson_scripts.append(script)
    p.script_args_value.extend(script_args)


class ControlNetExt:
    def __init__(self):
        self.cn_available = False
        self.external_cn = external_code

    def init_controlnet(self):
        self.cn_available = True

    def update_scripts_args(
        self,
        p,
        model: str,
        module: str | None,
        weight: float,
        guidance_start: float,
        guidance_end: float,
    ):
        if (not self.cn_available) or model == "None":
            return

        image = np.asarray(p.init_images[0])
        mask = np.full_like(image, fill_value=255)

        cnet_image = {"image": image, "mask": mask}

        pres = external_code.pixel_perfect_resolution(
            image,
            target_H=p.height,
            target_W=p.width,
            resize_mode=external_code.resize_mode_from_value(p.resize_mode),
        )

        add_forge_script_to_adetailer_run(
            p,
            "ControlNet",
            [
                ControlNetUnit(
                    enabled=True,
                    image=cnet_image,
                    model=model,
                    module=module,
                    weight=weight,
                    guidance_start=guidance_start,
                    guidance_end=guidance_end,
                    processor_res=pres,
                )
            ],
        )


def get_cn_models() -> list[str]:
    models = global_state.get_all_controlnet_names()
    return [m for m in models if cn_model_regex.search(m)]
