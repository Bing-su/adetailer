from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image, ImageChops, ImageDraw

repo_id = "Bingsu/adetailer"


@dataclass
class PredictOutput:
    bboxes: Optional[list[list[int]]] = None
    masks: Optional[list[Image.Image]] = None
    preview: Optional[Image.Image] = None


class SortBy(IntEnum):
    NONE = 0
    POSITION = 1
    AREA = 2


def get_models(
    model_dir: Union[str, Path], huggingface: bool = True
) -> OrderedDict[str, Optional[str]]:
    model_dir = Path(model_dir)
    if model_dir.is_dir():
        model_paths = [
            p
            for p in model_dir.rglob("*")
            if p.is_file() and p.suffix in (".pt", ".pth")
        ]
    else:
        model_paths = []

    if huggingface:
        models = OrderedDict(
            {
                "face_yolov8n.pt": hf_hub_download(repo_id, "face_yolov8n.pt"),
                "face_yolov8s.pt": hf_hub_download(repo_id, "face_yolov8s.pt"),
                "mediapipe_face_full": None,
                "mediapipe_face_short": None,
                "hand_yolov8n.pt": hf_hub_download(repo_id, "hand_yolov8n.pt"),
                "person_yolov8n-seg.pt": hf_hub_download(
                    repo_id, "person_yolov8n-seg.pt"
                ),
                "person_yolov8s-seg.pt": hf_hub_download(
                    repo_id, "person_yolov8s-seg.pt"
                ),
            }
        )
    else:
        models = OrderedDict(
            {
                "mediapipe_face_full": None,
                "mediapipe_face_short": None,
            }
        )

    for path in model_paths:
        if path.name in models:
            continue
        models[path.name] = str(path)

    return models


def create_mask_from_bbox(
    bboxes: list[list[float]], shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
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


def mask_preprocess(
    masks: list[Image.Image] | None,
    kernel: int = 0,
    x_offset: int = 0,
    y_offset: int = 0,
) -> list[Image.Image]:
    """
    The mask_preprocess function takes a list of masks and preprocesses them.
    It dilates and erodes the masks, and offsets them by x_offset and y_offset.

    Parameters
    ----------
        masks: list[Image.Image] | None
            A list of masks
        kernel: int
            kernel size of dilation or erosion
        x_offset: int
            →
        y_offset: int
            ↑

    Returns
    -------
        list[Image.Image]
            A list of processed masks
    """
    if masks is None:
        return []

    masks = [dilate_erode(m, kernel) for m in masks]
    masks = [m for m in masks if not is_all_black(m)]
    if x_offset != 0 or y_offset != 0:
        masks = [offset(m, x_offset, y_offset) for m in masks]

    return masks


def _key_position(bbox: list[float]) -> float:
    """
    Left to right

    Parameters
    ----------
    bbox: list[float]
        list of [x1, y1, x2, y2]
    """
    return bbox[0]


def _key_area(bbox: list[float]) -> float:
    """
    Large to small

    Parameters
    ----------
    bbox: list[float]
        list of [x1, y1, x2, y2]
    """
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return -area


def sort_bboxes(
    pred: PredictOutput, order: int | SortBy = SortBy.NONE
) -> PredictOutput:
    if order == SortBy.NONE or not pred.bboxes:
        return pred

    items = len(pred.bboxes)
    key = _key_area if order == SortBy.AREA else _key_position
    idx = sorted(range(items), key=lambda i: key(pred.bboxes[i]))
    pred.bboxes = [pred.bboxes[i] for i in idx]
    pred.masks = [pred.masks[i] for i in idx]
    return pred
