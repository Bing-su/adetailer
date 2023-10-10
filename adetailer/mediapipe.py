from __future__ import annotations

from functools import partial

import cv2
import numpy as np
from PIL import Image, ImageDraw

from adetailer import PredictOutput
from adetailer.common import create_bbox_from_mask, create_mask_from_bbox


def mediapipe_predict(
    model_type: str, image: Image.Image, confidence: float = 0.3
) -> PredictOutput:
    mapping = {
        "mediapipe_face_short": partial(mediapipe_face_detection, 0),
        "mediapipe_face_full": partial(mediapipe_face_detection, 1),
        "mediapipe_face_mesh": mediapipe_face_mesh,
        "mediapipe_face_mesh_eyes_only": mediapipe_face_mesh_eyes_only,
    }
    if model_type in mapping:
        func = mapping[model_type]
        return func(image, confidence)
    msg = f"[-] ADetailer: Invalid mediapipe model type: {model_type}, Available: {list(mapping.keys())!r}"
    raise RuntimeError(msg)


def mediapipe_face_detection(
    model_type: int, image: Image.Image, confidence: float = 0.3
) -> PredictOutput:
    import mediapipe as mp

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

    preview_array = img_array.copy()

    bboxes = []
    for detection in pred.detections:
        draw_util.draw_detection(preview_array, detection)

        bbox = detection.location_data.relative_bounding_box
        x1 = bbox.xmin * img_width
        y1 = bbox.ymin * img_height
        w = bbox.width * img_width
        h = bbox.height * img_height
        x2 = x1 + w
        y2 = y1 + h

        bboxes.append([x1, y1, x2, y2])

    masks = create_mask_from_bbox(bboxes, image.size)
    preview = Image.fromarray(preview_array)

    return PredictOutput(bboxes=bboxes, masks=masks, preview=preview)


def mediapipe_face_mesh(image: Image.Image, confidence: float = 0.3) -> PredictOutput:
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh
    draw_util = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    w, h = image.size

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=20, min_detection_confidence=confidence
    ) as face_mesh:
        arr = np.array(image)
        pred = face_mesh.process(arr)

        if pred.multi_face_landmarks is None:
            return PredictOutput()

        preview = arr.copy()
        masks = []

        for landmarks in pred.multi_face_landmarks:
            draw_util.draw_landmarks(
                image=preview,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
            )

            points = np.intp([(land.x * w, land.y * h) for land in landmarks.landmark])
            outline = cv2.convexHull(points).reshape(-1).tolist()

            mask = Image.new("L", image.size, "black")
            draw = ImageDraw.Draw(mask)
            draw.polygon(outline, fill="white")
            masks.append(mask)

        bboxes = create_bbox_from_mask(masks, image.size)
        preview = Image.fromarray(preview)
        return PredictOutput(bboxes=bboxes, masks=masks, preview=preview)


def mediapipe_face_mesh_eyes_only(
    image: Image.Image, confidence: float = 0.3
) -> PredictOutput:
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh

    left_idx = np.array(list(mp_face_mesh.FACEMESH_LEFT_EYE)).flatten()
    right_idx = np.array(list(mp_face_mesh.FACEMESH_RIGHT_EYE)).flatten()

    w, h = image.size

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=20, min_detection_confidence=confidence
    ) as face_mesh:
        arr = np.array(image)
        pred = face_mesh.process(arr)

        if pred.multi_face_landmarks is None:
            return PredictOutput()

        preview = image.copy()
        masks = []

        for landmarks in pred.multi_face_landmarks:
            points = np.intp([(land.x * w, land.y * h) for land in landmarks.landmark])
            left_eyes = points[left_idx]
            right_eyes = points[right_idx]
            left_outline = cv2.convexHull(left_eyes).reshape(-1).tolist()
            right_outline = cv2.convexHull(right_eyes).reshape(-1).tolist()

            mask = Image.new("L", image.size, "black")
            draw = ImageDraw.Draw(mask)
            for outline in (left_outline, right_outline):
                draw.polygon(outline, fill="white")
            masks.append(mask)

        bboxes = create_bbox_from_mask(masks, image.size)
        preview = draw_preview(preview, bboxes, masks)
        return PredictOutput(bboxes=bboxes, masks=masks, preview=preview)


def draw_preview(
    preview: Image.Image, bboxes: list[list[int]], masks: list[Image.Image]
) -> Image.Image:
    red = Image.new("RGB", preview.size, "red")
    for mask in masks:
        masked = Image.composite(red, preview, mask)
        preview = Image.blend(preview, masked, 0.25)

    draw = ImageDraw.Draw(preview)
    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", width=2)

    return preview
