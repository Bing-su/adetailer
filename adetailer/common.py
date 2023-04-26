from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw


@dataclass
class PredictOutput:
    bboxes: list[list[int]] | None = None
    masks: list[Image.Image] | None = None
    preview: Image.Image | None = None


def get_models(model_dir: str | Path) -> OrderedDict[str, str | None]:
    model_dir = Path(model_dir)
    model_paths = [
        p for p in model_dir.rglob("*") if p.is_file() and p.suffix in (".pt", ".pth")
    ]

    models = OrderedDict(
        {
            "face_yolo8n.pt": hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt"),
            "face_yolo8s.pt": hf_hub_download("Bingsu/adetailer", "face_yolov8s.pt"),
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
