from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
import re

from modules import sd_models, shared
from modules.paths import data_path, models_path, script_path

ext_path = Path(data_path, "extensions")
ext_builtin_path = Path(script_path, "extensions-builtin")
is_in_builtin = False  # compatibility for vladmandic/automatic
controlnet_exists = False
controlnet_enabled_models = {
    'inpaint': 'inpaint_global_harmonious',
    'scribble': 't2ia_sketch_pidi',
    'lineart': 'lineart_coarse',
    'openpose': 'openpose_full',
    'tile': None,
}
controlnet_model_regex = re.compile(r'.*('+('|'.join(controlnet_enabled_models.keys()))+').*')

if ext_path.exists():
    controlnet_exists = any(
        p.name == "sd-webui-controlnet" for p in ext_path.iterdir() if p.is_dir()
    )

if not controlnet_exists and ext_builtin_path.exists():
    controlnet_exists = any(
        p.name == "sd-webui-controlnet"
        for p in ext_builtin_path.iterdir()
        if p.is_dir()
    )

    if controlnet_exists:
        is_in_builtin = True


class ControlNetExt:
    def __init__(self):
        self.cn_models = ["None"]
        self.cn_available = False
        self.external_cn = None

    def init_controlnet(self):
        if is_in_builtin:
            import_path = "extensions-builtin.sd-webui-controlnet.scripts.external_code"
        else:
            import_path = "extensions.sd-webui-controlnet.scripts.external_code"

        self.external_cn = importlib.import_module(import_path, "external_code")
        self.cn_available = True
        models = self.external_cn.get_models()
        self.cn_models.extend(m for m in models if controlnet_model_regex.match(m))

    def _update_scripts_args(self, p, model: str, weight: float, guidance_end: float):
        module = None
        for m, v in controlnet_enabled_models.items():
            if m in model:
                module = v
                break

        cn_units = [
            self.external_cn.ControlNetUnit(
                model=model,
                weight=weight,
                control_mode=self.external_cn.ControlMode.BALANCED,
                module=module,
                guidance_end=guidance_end,
                pixel_perfect=True,
            )
        ]

        self.external_cn.update_cn_script_in_processing(p, cn_units)

    def update_scripts_args(self, p, model: str, weight: float, guidance_end: float):
        if self.cn_available and model != "None":
            self._update_scripts_args(p, model, weight, guidance_end)


def get_cn_model_dirs() -> list[Path]:
    cn_model_dir = Path(models_path, "ControlNet")
    if is_in_builtin:
        cn_model_dir_old = Path(ext_builtin_path, "sd-webui-controlnet", "models")
    else:
        cn_model_dir_old = Path(ext_path, "sd-webui-controlnet", "models")
    ext_dir1 = shared.opts.data.get("control_net_models_path", "")
    ext_dir2 = shared.opts.data.get("controlnet_dir", "")

    dirs = [cn_model_dir, cn_model_dir_old]
    for ext_dir in [ext_dir1, ext_dir2]:
        if ext_dir:
            dirs.append(Path(ext_dir))

    return dirs


@lru_cache
def _get_cn_inpaint_models() -> list[str]:
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
            if p.is_file() and p.suffix in cn_model_exts and controlnet_model_regex.match(p.name):
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


def get_cn_inpaint_models() -> list[str]:
    if controlnet_exists:
        return _get_cn_inpaint_models()
    return []
