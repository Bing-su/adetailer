from __future__ import annotations

msg = "[-] ADetailer: WebUI versions below 1.6.0 are not supported."

try:
    from modules.processing import create_binary_mask  # noqa: F401
except ImportError as e:
    raise RuntimeError(msg) from e


try:
    from modules.ui_components import InputAccordion  # noqa: F401
except ImportError as e:
    raise RuntimeError(msg) from e


try:
    from modules.sd_schedulers import schedulers
except ImportError:
    # webui < 1.9.0
    schedulers = []
