from __future__ import annotations

from PIL import Image

try:
    from modules.processing import create_binary_mask
except ImportError:

    def create_binary_mask(image: Image.Image):
        return image.convert("L")


try:
    from modules.sd_schedulers import schedulers
except ImportError:
    schedulers = []
