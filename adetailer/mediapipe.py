from __future__ import annotations

import numpy as np
import mediapipe as mp
from PIL import Image

from adetailer import PredictOutput
from adetailer.common import create_mask_from_bbox


def mediapipe_predict(
    model_type: int, image: Image.Image, confidence: float = 0.25
) -> PredictOutput:
    img_width, img_height = image.size

    mp_face_detection = mp.solutions.face_detection
    draw_util = mp.solutions.drawing_utils

    img_array = np.array(image)

    with mp_face_detection.FaceDetection(
        model_selection=model_type, min_detection_confidence=confidence
    ) as face_detector:
        pred = face_detector.process(img_array)

    if pred.detections is None:
        return PredictOutput()

    example_array = img_array.copy()

    bboxes = []
    for detection in pred.detections:
        draw_util.draw_detection(example_array, detection)

        bbox = detection.location_data.relative_bounding_box
        x1 = bbox.xmin * img_width
        y1 = bbox.ymin * img_height
        w = bbox.width * img_width
        h = bbox.height * img_height
        x2 = x1 + w
        y2 = y1 + h

        bboxes.append([x1, y1, x2, y2])

    masks = create_mask_from_bbox(image, bboxes)
    example = Image.fromarray(example_array)

    return PredictOutput(bboxes=bboxes, masks=masks, example=example)
