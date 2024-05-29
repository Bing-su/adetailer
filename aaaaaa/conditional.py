from __future__ import annotations

from PIL import Image
from rich import print

try:
    from modules.processing import create_binary_mask
except ImportError:
    msg = "[-] ADetailer: Support for webui versions below 1.6.0 will be discontinued."
    print(msg)

    def create_binary_mask(image: Image.Image):
        return image.convert("L")


try:
    from modules.sd_schedulers import schedulers
except ImportError:
    schedulers = []
