from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

repo_id = "Bingsu/adetailer"


@dataclass
class PredictOutput:
    bboxes: Optional[list[list[int]]] = None
    masks: Optional[list[Image.Image]] = None
    preview: Optional[Image.Image] = None


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
