import pytest
from huggingface_hub import hf_hub_download
from PIL import Image

from adetailer.ultralytics import ultralytics_predict


@pytest.mark.parametrize(
    "model_name",
    [
        "face_yolov8n.pt",
        "face_yolov8n_v2.pt",
        "face_yolov8s.pt",
        "face_yolov9c.pt",
        "hand_yolov8n.pt",
        "hand_yolov8s.pt",
        "hand_yolov9c.pt",
        "person_yolov8n-seg.pt",
        "person_yolov8s-seg.pt",
        "person_yolov8m-seg.pt",
        "deepfashion2_yolov8s-seg.pt",
    ],
)
def test_ultralytics_hf_models(sample_image: Image.Image, model_name: str):
    model_path = hf_hub_download("Bingsu/adetailer", model_name)
    result = ultralytics_predict(model_path, sample_image)
    assert result.preview is not None


def test_yolo_world_default(sample_image: Image.Image):
    model_path = hf_hub_download("Bingsu/yolo-world-mirror", "yolov8x-worldv2.pt")
    result = ultralytics_predict(model_path, sample_image)
    assert result.preview is not None


@pytest.mark.parametrize(
    "klass",
    [
        "person",
        "bird",
        "yellow bird",
        "person,glasses,headphone",
        "person,bird",
        "glasses,yellow bird",
    ],
)
def test_yolo_world(sample_image2: Image.Image, klass: str):
    model_path = hf_hub_download("Bingsu/yolo-world-mirror", "yolov8x-worldv2.pt")
    result = ultralytics_predict(model_path, sample_image2, classes=klass)
    assert result.preview is not None
