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

from .restore import cn_allow_script_control

if controlnet_type == "forge":
    from contextlib import nullcontext as CNHijackRestore  # noqa: N812
else:
    from .restore import CNHijackRestore

__all__ = [
    "ControlNetExt",
    "CNHijackRestore",
    "cn_allow_script_control",
    "controlnet_exists",
    "controlnet_type",
    "get_cn_models",
]
