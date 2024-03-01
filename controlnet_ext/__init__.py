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

__all__ = [
    "ControlNetExt",
    "controlnet_exists",
    "controlnet_type",
    "get_cn_models",
]
