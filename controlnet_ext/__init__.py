try:
    from .controlnet_ext_forge import (
        ControlNetExt,
        controlnet_exists,
        controlnet_type,
        get_cn_models,
    )
except Exception:
    # Not just ImportError: a broken/drifted Forge lib_controlnet can raise
    # NameError/AttributeError at import time (e.g. reForge). Degrade to the
    # A1111 standard ControlNet backend instead of crashing the whole
    # extension at load (universal-WebUI-compat guard).
    from .controlnet_ext import (
        ControlNetExt,
        controlnet_exists,
        controlnet_type,
        get_cn_models,
    )

from .restore import CNHijackRestore, cn_allow_script_control

__all__ = [
    "CNHijackRestore",
    "ControlNetExt",
    "cn_allow_script_control",
    "controlnet_exists",
    "controlnet_type",
    "get_cn_models",
]
