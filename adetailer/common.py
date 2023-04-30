from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image, ImageChops, ImageDraw


@dataclass
class PredictOutput:
    bboxes: Optional[list[list[int]]] = None
    masks: Optional[list[Image.Image]] = None
    preview: Optional[Image.Image] = None


def get_models(model_dir: str | Path) -> OrderedDict[str, str | None]:
    model_dir = Path(model_dir)
    model_paths = [
        p for p in model_dir.rglob("*") if p.is_file() and p.suffix in (".pt", ".pth")
    ]

    models = OrderedDict(
        {
            "face_yolov8n.pt": hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt"),
            "face_yolov8s.pt": hf_hub_download("Bingsu/adetailer", "face_yolov8s.pt"),
            "mediapipe_face_full": None,
            "mediapipe_face_short": None,
            "hand_yolov8n.pt": hf_hub_download("Bingsu/adetailer", "hand_yolov8n.pt"),
        }
    )

    for path in model_paths:
        if path.name in models:
            continue
        models[path.name] = str(path)

    return models


def create_mask_from_bbox(
    image: Image.Image, bboxes: list[list[float]]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        image: Image.Image
            The image to create the mask from
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


def _dilate(arr: np.ndarray, value: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    return cv2.dilate(arr, kernel, iterations=1)


def _erode(arr: np.ndarray, value: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    return cv2.erode(arr, kernel, iterations=1)


def dilate_erode(img: Image.Image, value: int) -> Image.Image:
    """
    The dilate_erode function takes an image and a value.
    If the value is positive, it dilates the image by that amount.
    If the value is negative, it erodes the image by that amount.

    Parameters
    ----------
        img: PIL.Image.Image
            the image to be processed
        value: int
            kernel size of dilation or erosion

    Returns
    -------
        PIL.Image.Image
            The image that has been dilated or eroded
    """
    if value == 0:
        return img

    arr = np.array(img)
    arr = _dilate(arr, value) if value > 0 else _erode(arr, -value)

    return Image.fromarray(arr)


def offset(img: Image.Image, x: int = 0, y: int = 0) -> Image.Image:
    """
    The offset function takes an image and offsets it by a given x(→) and y(↑) value.

    Parameters
    ----------
        mask: Image.Image
            Pass the mask image to the function
        x: int
            →
        y: int
            ↑

    Returns
    -------
        PIL.Image.Image
            A new image that is offset by x and y
    """
    return ImageChops.offset(img, x, -y)


def is_all_black(img: Image.Image) -> bool:
    arr = np.array(img)
    return cv2.countNonZero(arr) == 0
