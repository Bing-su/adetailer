try:
    from .controlnet_ext_forge import (
        ControlNetExt,
        controlnet_exists,
        controlnet_type,
        get_cn_models,
    )
except ImportError:
    from .controlnet_ext import (
        ControlNetExt,
        controlnet_exists,
        controlnet_type,
        get_cn_models,
    )

from .restore import CNHijackRestore, cn_allow_script_control

__all__ = [
    "ControlNetExt",
    "CNHijackRestore",
    "cn_allow_script_control",
    "controlnet_exists",
    "controlnet_type",
    "get_cn_models",
]
