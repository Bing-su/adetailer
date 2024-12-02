from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from adetailer import PredictOutput
from adetailer.common import create_mask_from_bbox
import numpy as np
if TYPE_CHECKING:
    import torch
    from ultralytics import YOLO, YOLOWorld


def ultralytics_predict(
    model_path: str | Path,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
    classes: str = "",
) -> PredictOutput[float]:
    from ultralytics import YOLO
    model = YOLO(model_path)
    class_indices = []
    if classes:
        parsed = [c.strip() for c in classes.split(",") if c.strip()]
        for c in parsed:
            if c.isdigit():
                class_indices.append(int(c))
            elif c in model.names.values():
                # Find the index for the class name
                for idx, name in model.names.items():
                    if name == c:
                        class_indices.append(idx)
                        break

    pred = model(image, conf=confidence, device=device)
    
    if class_indices and len(pred[0].boxes) > 0:
        cls = pred[0].boxes.cls.cpu().numpy()
        mask = np.isin(cls, class_indices)
        
        # Apply mask to boxes
        pred[0].boxes.data = pred[0].boxes.data[mask]
        if pred[0].masks is not None:
            pred[0].masks.data = pred[0].masks.data[mask]

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


def apply_classes(model: YOLO | YOLOWorld, model_path: str | Path, classes: str):
    if not classes:
        return
    
    parsed = [c.strip() for c in classes.split(",") if c.strip()]
    if not parsed:
        return
    
    try:
        class_indices = []
        for c in parsed:
            if c.isdigit():
                class_indices.append(int(c))
            elif c in model.names.values():
                for idx, name in model.names.items():
                    if name == c:
                        class_indices.append(idx)
                        break
        
        model.classes = class_indices
    except Exception as e:
        print(f"Error setting classes: {e}")


def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (W, H) of the original image
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]
