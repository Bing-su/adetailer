from enum import IntEnum
from functools import partial
from math import dist

import cv2
import numpy as np
from PIL import Image, ImageChops

from adetailer.common import PredictOutput


class SortBy(IntEnum):
    NONE = 0
    LEFT_TO_RIGHT = 1
    CENTER_TO_EDGE = 2
    AREA = 3


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


def bbox_area(bbox: list[float]):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


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


# Bbox sorting
def _key_left_to_right(bbox: list[float]) -> float:
    """
    Left to right

    Parameters
    ----------
    bbox: list[float]
        list of [x1, y1, x2, y2]
    """
    return bbox[0]


def _key_center_to_edge(bbox: list[float], *, center: tuple[float, float]) -> float:
    """
    Center to edge

    Parameters
    ----------
    bbox: list[float]
        list of [x1, y1, x2, y2]
    image: Image.Image
        the image
    """
    bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return dist(center, bbox_center)


def _key_area(bbox: list[float]) -> float:
    """
    Large to small

    Parameters
    ----------
    bbox: list[float]
        list of [x1, y1, x2, y2]
    """
    return -bbox_area(bbox)


def sort_bboxes(
    pred: PredictOutput, order: int | SortBy = SortBy.NONE
) -> PredictOutput:
    if order == SortBy.NONE or not pred.bboxes:
        return pred

    if order == SortBy.LEFT_TO_RIGHT:
        key = _key_left_to_right
    elif order == SortBy.CENTER_TO_EDGE:
        width, height = pred.preview.size
        center = (width / 2, height / 2)
        key = partial(_key_center_to_edge, center=center)
    elif order == SortBy.AREA:
        key = _key_area
    else:
        raise RuntimeError

    items = len(pred.bboxes)
    idx = sorted(range(items), key=lambda i: key(pred.bboxes[i]))
    pred.bboxes = [pred.bboxes[i] for i in idx]
    pred.masks = [pred.masks[i] for i in idx]
    return pred
