from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from rich import print
from torchvision.transforms.functional import to_pil_image

REPO_ID = "Bingsu/adetailer"
_download_failed = False


@dataclass
class PredictOutput:
    bboxes: list[list[int | float]] = field(default_factory=list)
    masks: list[Image.Image] = field(default_factory=list)
    preview: Optional[Image.Image] = None


def hf_download(file: str, repo_id: str = REPO_ID) -> str | None:
    global _download_failed

    if _download_failed:
        return "INVALID"

    try:
        path = hf_hub_download(repo_id, file)
    except Exception:
        msg = f"[-] ADetailer: Failed to load model {file!r} from huggingface"
        print(msg)
        path = "INVALID"
        _download_failed = True
    return path


def safe_mkdir(path: str | os.PathLike[str]) -> None:
    path = Path(path)
    if not path.exists() and path.parent.exists() and os.access(path.parent, os.W_OK):
        path.mkdir()


def scan_model_dir(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [p for p in path.rglob("*") if p.is_file() and p.suffix == ".pt"]


def get_models(
    *dirs: str | os.PathLike[str], huggingface: bool = True
) -> OrderedDict[str, str]:
    model_paths = []

    for dir_ in dirs:
        if not dir_:
            continue
        model_paths.extend(scan_model_dir(Path(dir_)))

    models = OrderedDict()
    if huggingface:
        models.update(
            {
                "face_yolov8n.pt": hf_download("face_yolov8n.pt"),
                "face_yolov8s.pt": hf_download("face_yolov8s.pt"),
                "hand_yolov8n.pt": hf_download("hand_yolov8n.pt"),
                "person_yolov8n-seg.pt": hf_download("person_yolov8n-seg.pt"),
                "person_yolov8s-seg.pt": hf_download("person_yolov8s-seg.pt"),
                "yolov8x-worldv2.pt": hf_download(
                    "yolov8x-worldv2.pt", repo_id="Bingsu/yolo-world-mirror"
                ),
            }
        )
    models.update(
        {
            "mediapipe_face_full": "mediapipe_face_full",
            "mediapipe_face_short": "mediapipe_face_short",
            "mediapipe_face_mesh": "mediapipe_face_mesh",
            "mediapipe_face_mesh_eyes_only": "mediapipe_face_mesh_eyes_only",
        }
    )

    invalid_keys = [k for k, v in models.items() if v == "INVALID"]
    for key in invalid_keys:
        models.pop(key)

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


def create_bbox_from_mask(
    masks: list[Image.Image], shape: tuple[int, int]
) -> list[list[int]]:
    """
    Parameters
    ----------
        masks: list[Image.Image]
            A list of masks
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        bboxes: list[list[float]]
        A list of bounding boxes

    """
    bboxes = []
    for mask in masks:
        mask = mask.resize(shape)
        bbox = mask.getbbox()
        if bbox is not None:
            bboxes.append(list(bbox))
    return bboxes


def ensure_pil_image(image: Any, mode: str = "RGB") -> Image.Image:
    if not isinstance(image, Image.Image):
        image = to_pil_image(image)
    if image.mode != mode:
        image = image.convert(mode)
    return image
