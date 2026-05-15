from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from adetailer import PredictOutput
from adetailer.classes import (
    get_model_class_names,
    is_world_model,
    parse_csv,
    resolve_class_ids,
)
from adetailer.common import create_mask_from_bbox

if TYPE_CHECKING:
    import torch
    from ultralytics import YOLO, YOLOWorld


def ultralytics_predict(
    model_path: str | Path,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
    classes: str = "",
    exclude_classes: str = "",
) -> PredictOutput[float]:
    from ultralytics import YOLO

    model = YOLO(model_path)

    requested = parse_csv(classes)
    excluded = parse_csv(exclude_classes)

    if is_world_model(model_path):
        # YOLO-World open-vocab path (unchanged behavior).
        if requested:
            model.set_classes(requested)
        pred = model(image, conf=confidence, device=device)
    else:
        # Multiclass YOLO: include-by-id at inference time, exclude post-hoc.
        kw: dict = {"conf": confidence, "device": device}
        if requested:
            ids = resolve_class_ids(str(model_path), requested)
            if ids:
                kw["classes"] = ids
        try:
            pred = model(image, **kw)
        except TypeError:
            # Older ultralytics may not accept the `classes=` kwarg — fall back.
            kw.pop("classes", None)
            pred = model(image, **kw)

        if (
            excluded
            and pred[0].boxes is not None
            and pred[0].boxes.cls is not None
            and len(pred[0].boxes) > 0
        ):
            names = get_model_class_names(str(model_path))
            cls_ids = pred[0].boxes.cls.cpu().numpy().astype(int).tolist()
            keep = [
                i
                for i, cid in enumerate(cls_ids)
                if (names[cid] if 0 <= cid < len(names) else str(cid)) not in excluded
                and str(cid) not in excluded
            ]
            if not keep:
                return PredictOutput()
            try:
                pred[0] = pred[0][keep]
            except (TypeError, IndexError):
                # Older ultralytics Results may not support list indexing.
                # Subset manually after extraction below by reusing `keep`.
                pred = _SubsetWrapper(pred, keep)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return PredictOutput()
    bboxes = bboxes.tolist()

    if pred[0].masks is None:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)

    confidences = pred[0].boxes.conf.cpu().numpy().tolist()

    preview = pred[0].plot()
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(preview)

    return PredictOutput(
        bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
    )


class _SubsetWrapper:
    """Fallback for old Ultralytics: subset Results-like objects by index list."""

    def __init__(self, orig, keep: list[int]):
        self._orig = orig
        self._keep = keep
        self._wrapped = _SubsetResults(orig[0], keep)

    def __getitem__(self, idx):
        if idx == 0:
            return self._wrapped
        return self._orig[idx]


class _SubsetResults:
    def __init__(self, r, keep: list[int]):
        self._r = r
        self._keep = keep

    @property
    def boxes(self):
        return _SubsetBoxes(self._r.boxes, self._keep) if self._r.boxes is not None else None

    @property
    def masks(self):
        return _SubsetMasks(self._r.masks, self._keep) if self._r.masks is not None else None

    def plot(self, *a, **kw):
        return self._r.plot(*a, **kw)


class _SubsetBoxes:
    def __init__(self, b, keep: list[int]):
        self._b = b
        self._keep = keep

    @property
    def xyxy(self):
        return self._b.xyxy[self._keep]

    @property
    def conf(self):
        return self._b.conf[self._keep]

    @property
    def cls(self):
        return self._b.cls[self._keep]


class _SubsetMasks:
    def __init__(self, m, keep: list[int]):
        self._m = m
        self._keep = keep

    @property
    def data(self):
        return self._m.data[self._keep]


def apply_classes(model: YOLO | YOLOWorld, model_path: str | Path, classes: str):
    """Backward-compatible shim: only sets classes on YOLO-World models."""
    if not classes or not is_world_model(model_path):
        return
    parsed = parse_csv(classes)
    if parsed:
        model.set_classes(parsed)


def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32 or torch.uint8, shape=(N, H, W).
        uint8 tensor is expected to have values 0 or 1 (not 0-255).

    shape: tuple[int, int]
        (W, H) of the original image
    """
    masks = masks.float()
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]
