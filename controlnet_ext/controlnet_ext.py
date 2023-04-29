from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path

from modules import sd_models, shared
from modules.paths import data_path, models_path

extensions_path = Path(data_path, "extensions")
controlnet_exists = any(
    p.name == "sd-webui-controlnet" for p in extensions_path.iterdir() if p.is_dir()
)


class ControlNetExt:
    def __init__(self):
        self.cn_models = ["None"]
        self.cn_available = False
        self.external_cn = None

    def init_controlnet(self) -> bool:
        try:
            self.external_cn = importlib.import_module(
                "extensions.sd-webui-controlnet.scripts.external_code", "external_code"
            )
            self.cn_available = True
            models = self.external_cn.get_models()
            self.cn_models.extend(m for m in models if "inpaint" in m)
            return True
        except ImportError:
            return False

    def _update_scripts_args(self, p, model: str, weight: float):
        cn_units = [
            self.external_cn.ControlNetUnit(
                model=model,
                weight=weight,
                control_mode=self.external_cn.ControlMode.BALANCED,
                module="inpaint_global_harmonious",
                pixel_perfect=True,
            )
        ]

        self.external_cn.update_cn_script_in_processing(p, cn_units)

    def update_scripts_args(self, p, model: str, weight: float):
        if self.cn_available and model != "None":
            self._update_scripts_args(p, model, weight)


@lru_cache
def _get_cn_inpaint_models() -> list[str]:
    """
    Since we can't import ControlNet, we use a function that does something like
    controlnet's `list(global_state.cn_models_names.values())`.
    """
    cn_model_exts = (".pt", ".pth", ".ckpt", ".safetensors")
    cn_model_dir = Path(models_path, "ControlNet")
    cn_model_dir_old = Path(extensions_path, "sd-webui-controlnet", "models")
    ext_dir1 = shared.opts.data.get("control_net_models_path", "")
    ext_dir2 = shared.opts.data.get("controlnet_dir", "")
    name_filter = shared.opts.data.get("control_net_models_name_filter", "")
    name_filter = name_filter.strip(" ").lower()

    model_paths = []

    for base in [cn_model_dir, cn_model_dir_old, ext_dir1, ext_dir2]:
        if not base:
            continue
        base = Path(base)
        if not base.exists():
            continue

        for p in base.rglob("*"):
            if p.is_file() and p.suffix in cn_model_exts and "inpaint" in p.name:
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
