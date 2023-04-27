from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO

from adetailer import PredictOutput
from adetailer.common import create_mask_from_bbox


def ultralytics_predict(
    model_path: str | Path,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
) -> PredictOutput:
    model_path = str(model_path)

    model = YOLO(model_path)
    pred = model(image, conf=confidence, show_labels=False, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return PredictOutput()
    bboxes = bboxes.tolist()

    masks = create_mask_from_bbox(image, bboxes)
    preview = pred[0].plot()
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(preview)

    return PredictOutput(bboxes=bboxes, masks=masks, preview=preview)
