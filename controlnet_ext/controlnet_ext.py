from __future__ import annotations

import copy
import importlib
import re
import sys
from functools import lru_cache
from pathlib import Path
from textwrap import dedent

import numpy as np

from modules import extensions, sd_models, shared

try:
    from modules.paths import extensions_builtin_dir, extensions_dir, models_path
except ImportError as e:
    msg = """
    [-] ADetailer: `stable-diffusion-webui < 1.1.0` is no longer supported.
        Please upgrade to stable-diffusion-webui >= 1.1.0.
        or you can use ADetailer v23.10.1 (https://github.com/Bing-su/adetailer/archive/refs/tags/v23.10.1.zip)
    """
    raise RuntimeError(dedent(msg)) from e

ext_path = Path(extensions_dir)
ext_builtin_path = Path(extensions_builtin_dir)

try:
    from lib_controlnet import external_code, global_state
    from lib_controlnet.external_code import ControlNetUnit
    from modules_forge.forge_util import numpy_to_pytorch

    from modules import extensions, scripts, sd_models, shared
    from modules.paths_internal import (
        extensions_builtin_dir,
        extensions_dir,
        models_path,
    )
    from modules.processing import (
        StableDiffusionProcessing,
        StableDiffusionProcessingImg2Img,
        StableDiffusionProcessingTxt2Img,
    )

    controlnet_exists = True
    controlnet_forge = True
except ImportError:
    controlnet_exists = False
    controlnet_forge = False
    external_code = None

controlnet_path = None
cn_base_path = ""

if not controlnet_forge:
    for extension in extensions.active():
        if not extension.enabled:
            continue
        # For cases like sd-webui-controlnet-master
        if "sd-webui-controlnet" in extension.name:
            controlnet_exists = True
            controlnet_path = Path(extension.path)
            cn_base_path = ".".join(controlnet_path.parts[-2:])
            break

    if controlnet_path is not None:
        sd_webui_controlnet_path = controlnet_path.resolve().parent
        if sd_webui_controlnet_path.stem in ("extensions", "extensions-builtin"):
            target_path = str(sd_webui_controlnet_path.parent)
            if target_path not in sys.path:
                sys.path.append(target_path)

cn_model_module = {
    "inpaint": "inpaint_global_harmonious",
    "scribble": "t2ia_sketch_pidi",
    "lineart": "lineart_coarse",
    "openpose": "openpose_full",
    "tile": "tile_resample",
    "depth": "depth_midas",
}
cn_model_regex = re.compile("|".join(cn_model_module.keys()))

def find_script(p : StableDiffusionProcessing, script_title : str) -> scripts.Script:
    script = next((s for s in p.scripts.scripts if s.title() == script_title  ), None)
    if not script:
        raise Exception("Script not found: " + script_title)
    return script

def add_forge_script_to_adetailer_run(p: StableDiffusionProcessing, script_title : str, script_args : list):
    p.scripts = copy.copy(scripts.scripts_img2img)
    p.scripts.alwayson_scripts = []
    p.script_args_value = []

    script = copy.copy(find_script(p, script_title))
    script.args_from = len(p.script_args_value)
    script.args_to = len(p.script_args_value) + len(script_args)
    p.scripts.alwayson_scripts.append(script)
    p.script_args_value.extend(script_args)

    print(f"Added script: {script.title()} with {len(script_args)} args, at positions {script.args_from}-{script.args_to} (of 0-{len(p.script_args_value)-1}.)", file=sys.stderr)

class ControlNetExt:
    def __init__(self):
        if not controlnet_forge:
            self.cn_models = ["None"]
        self.cn_available = False
        self.external_cn = external_code

    def init_controlnet(self):
        if controlnet_forge:
            self.cn_available = True
        else:
            import_path = cn_base_path + ".scripts.external_code"

            self.external_cn = importlib.import_module(import_path, "external_code")
            self.cn_available = True
            models = self.external_cn.get_models()
            self.cn_models.extend(m for m in models if cn_model_regex.search(m))

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

        if controlnet_forge:
            image = np.asarray(p.init_images[0])
            mask = np.zeros_like(image)
            mask[:] = 255

            cnet_image = {
                "image": image,
                "mask": mask
            }

            pres = external_code.pixel_perfect_resolution(
                image,
                target_H=p.height,
                target_W=p.width,
                resize_mode=external_code.resize_mode_from_value(p.resize_mode)
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
                        processor_res=pres
                    )
                ]
            )

            return

        if module is None or module == "None":
            for m, v in cn_model_module.items():
                if m in model:
                    module = v
                    break
            else:
                module = None

        cn_units = [
            self.external_cn.ControlNetUnit(
                model=model,
                weight=weight,
                control_mode=self.external_cn.ControlMode.BALANCED,
                module=module,
                guidance_start=guidance_start,
                guidance_end=guidance_end,
                pixel_perfect=True,
            )
        ]

        try:
            self.external_cn.update_cn_script_in_processing(p, cn_units)
        except AttributeError as e:
            if "script_args_value" not in str(e):
                raise
            msg = "[-] Adetailer: ControlNet option not available in WEBUI version lower than 1.6.0 due to updates in ControlNet"
            raise RuntimeError(msg) from e


def get_cn_model_dirs() -> list[Path]:
    cn_model_dir = Path(models_path, "ControlNet")
    if controlnet_path is not None:
        cn_model_dir_old = controlnet_path.joinpath("models")
    else:
        cn_model_dir_old = None
    ext_dir1 = shared.opts.data.get("control_net_models_path", "")
    ext_dir2 = getattr(shared.cmd_opts, "controlnet_dir", "")

    dirs = [cn_model_dir]
    dirs += [
        Path(ext_dir) for ext_dir in [cn_model_dir_old, ext_dir1, ext_dir2] if ext_dir
    ]

    return dirs


@lru_cache
def _get_cn_models() -> list[str]:
    """
    Since we can't import ControlNet, we use a function that does something like
    controlnet's `list(global_state.cn_models_names.values())`.
    """
    cn_model_exts = (".pt", ".pth", ".ckpt", ".safetensors")
    dirs = get_cn_model_dirs()
    name_filter = shared.opts.data.get("control_net_models_name_filter", "")
    name_filter = name_filter.strip(" ").lower()

    model_paths = []

    for base in dirs:
        if not base.exists():
            continue

        for p in base.rglob("*"):
            if (
                p.is_file()
                and p.suffix in cn_model_exts
                and cn_model_regex.search(p.name)
            ):
                if name_filter and name_filter not in p.name.lower():
                    continue
                model_paths.append(p)
    model_paths.sort(key=lambda p: p.name)

    models = []
    for p in model_paths:
        model_hash = sd_models.model_hash(p)
        name = f"{p.stem} [{model_hash}]"
        models.append(name)
    return models


def get_cn_models() -> list[str]:
    if controlnet_forge:
        models = global_state.get_all_controlnet_names()
        return [m for m in models if cn_model_regex.search(m)]
    if controlnet_exists:
        return _get_cn_models()
    return []
