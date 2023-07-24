from __future__ import annotations

import importlib
import re
from functools import lru_cache
from pathlib import Path

from modules import extensions, sd_models, shared
from modules.paths import data_path, models_path, script_path

ext_path = Path(data_path, "extensions")
ext_builtin_path = Path(script_path, "extensions-builtin")
controlnet_exists = False
controlnet_path = None
cn_base_path = ""

for extension in extensions.active():
    if not extension.enabled:
        continue
    # For cases like sd-webui-controlnet-master
    if "sd-webui-controlnet" in extension.name:
        controlnet_exists = True
        controlnet_path = Path(extension.path)
        cn_base_path = ".".join(controlnet_path.parts[-2:])
        break

cn_model_module = {
    "inpaint": "inpaint_global_harmonious",
    "scribble": "t2ia_sketch_pidi",
    "lineart": "lineart_coarse",
    "openpose": "openpose_full",
    "tile": None,
}
cn_model_regex = re.compile("|".join(cn_model_module.keys()))


class ControlNetExt:
    def __init__(self):
        self.cn_models = ["None"]
        self.cn_available = False
        self.external_cn = None

    def init_controlnet(self):
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

        if module is None:
            for m, v in cn_model_module.items():
                if m in model:
                    module = v
                    break

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

        self.external_cn.update_cn_script_in_processing(p, cn_units)


def get_cn_model_dirs() -> list[Path]:
    cn_model_dir = Path(models_path, "ControlNet")
    if controlnet_path is not None:
        cn_model_dir_old = controlnet_path.joinpath("models")
    else:
        cn_model_dir_old = None
    ext_dir1 = shared.opts.data.get("control_net_models_path", "")
    ext_dir2 = getattr(shared.cmd_opts, "controlnet_dir", "")

    dirs = [cn_model_dir]
    for ext_dir in [cn_model_dir_old, ext_dir1, ext_dir2]:
        if ext_dir:
            dirs.append(Path(ext_dir))

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
    if controlnet_exists:
        return _get_cn_models()
    return []
