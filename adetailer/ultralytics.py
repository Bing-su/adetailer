from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO

from adetailer import PredictOutput
from adetailer.common import create_mask_from_bbox


def ultralytics_predict(
    model_path: str | Path, image: Image.Image, confidence: float = 0.25
) -> PredictOutput:
    model_path = str(model_path)

    model = YOLO(model_path)
    pred = model(image, conf=confidence, hide_labels=True)

    bboxes = pred[0].xyxy.cpu().numpy()
    masks = create_mask_from_bbox(image, bboxes)
    example = pred[0].plot()
    example = cv2.cvtColor(example, cv2.COLOR_BGR2RGB)
    example = Image.fromarray(example)

    return PredictOutput(bboxes=bboxes, masks=masks, example=example)
