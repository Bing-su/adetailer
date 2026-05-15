import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from adetailer.classes import _names_from_json, parse_csv, resolve_class_ids
from adetailer.ultralytics import mask_to_pil, ultralytics_predict


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
    assert len(result.bboxes) > 0
    assert len(result.masks) > 0
    assert len(result.confidences) > 0
    assert len(result.bboxes) == len(result.masks) == len(result.confidences)


def test_yolo_world_default(sample_image: Image.Image):
    model_path = hf_hub_download("Bingsu/yolo-world-mirror", "yolov8x-worldv2.pt")
    result = ultralytics_predict(model_path, sample_image)
    assert result.preview is not None
    assert len(result.bboxes) > 0
    assert len(result.masks) > 0
    assert len(result.confidences) > 0
    assert len(result.bboxes) == len(result.masks) == len(result.confidences)


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
    assert len(result.bboxes) > 0
    assert len(result.masks) > 0
    assert len(result.confidences) > 0
    assert len(result.bboxes) == len(result.masks) == len(result.confidences)


def test_class_filter_include_person(sample_image: Image.Image):
    """Single-class model: filtering on its only class is a no-op vs unfiltered."""
    model_path = hf_hub_download("Bingsu/adetailer", "person_yolov8n-seg.pt")
    full = ultralytics_predict(model_path, sample_image)
    filtered = ultralytics_predict(model_path, sample_image, classes="person")
    assert len(filtered.bboxes) == len(full.bboxes)
    assert len(filtered.masks) == len(full.masks)


def test_class_filter_include_unknown_falls_back(sample_image: Image.Image):
    """Unknown class names are dropped; if no valid id remains, falls back to no filter."""
    model_path = hf_hub_download("Bingsu/adetailer", "person_yolov8n-seg.pt")
    full = ultralytics_predict(model_path, sample_image)
    filtered = ultralytics_predict(model_path, sample_image, classes="nonexistent_class")
    assert len(filtered.bboxes) == len(full.bboxes)


def test_class_filter_exclude_person(sample_image: Image.Image):
    """Excluding the only class the model produces should yield zero detections."""
    model_path = hf_hub_download("Bingsu/adetailer", "person_yolov8n-seg.pt")
    result = ultralytics_predict(model_path, sample_image, exclude_classes="person")
    assert len(result.bboxes) == 0
    assert len(result.masks) == 0
    assert len(result.confidences) == 0


def test_parse_csv():
    assert parse_csv("") == []
    assert parse_csv("face") == ["face"]
    assert parse_csv("face, penis ,pussy") == ["face", "penis", "pussy"]
    assert parse_csv(",,,") == []


class TestNamesFromJson:
    def test_list(self):
        assert _names_from_json(["face", "hand"]) == ["face", "hand"]

    def test_names_list(self):
        assert _names_from_json({"names": ["face", "hand"]}) == ["face", "hand"]

    def test_names_dict(self):
        assert _names_from_json({"names": {"0": "face", "1": "hand"}}) == ["face", "hand"]

    def test_bare_int_keys(self):
        assert _names_from_json({"0": "face", "1": "hand", "2": "eye"}) == [
            "face",
            "hand",
            "eye",
        ]

    def test_civitai_helper_sidecar_ignored(self):
        """Regression: civitai_helper-style metadata JSON must not be mistaken
        for a class-names file. It has string keys and nested dicts; we should
        return [] so the caller falls through to model.names."""
        civitai_blob = {
            "description": "<h2>...</h2>",
            "sd version": "Unknown",
            "extensions": {"sd_civitai_helper": {"version": "1.8.13"}},
        }
        assert _names_from_json(civitai_blob) == []

    def test_partial_int_keys_rejected(self):
        """A dict with some int keys and some non-int keys is NOT a class map."""
        assert _names_from_json({"0": "face", "description": "x"}) == []

    def test_nested_values_rejected(self):
        """A dict with int keys whose values are dicts is NOT a class map."""
        assert _names_from_json({"0": {"name": "face"}, "1": {"name": "hand"}}) == []

    def test_empty_returns_empty(self):
        assert _names_from_json({}) == []
        assert _names_from_json([]) == []
        assert _names_from_json(None) == []
        assert _names_from_json("face") == []


def test_resolve_class_ids_with_known_model():
    """resolve_class_ids should accept names AND numeric strings."""
    model_path = hf_hub_download("Bingsu/adetailer", "person_yolov8n-seg.pt")
    ids = resolve_class_ids(model_path, ["person", "0", "unknown"])
    assert 0 in ids
    assert all(isinstance(i, int) for i in ids)


class TestMaskToPil:
    def test_mask_to_pil_float32(self):
        mask = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32)
        imgs = mask_to_pil(mask, shape=(2, 2))

        assert len(imgs) == 1
        img = imgs[0]
        assert isinstance(img, Image.Image)

        arr = np.array(img)
        assert arr.shape == (2, 2)
        assert arr.dtype == np.uint8

        expected = np.array([[0, 255], [0, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(arr, expected)

    def test_mask_to_pil_uint8(self):
        mask = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.uint8)
        imgs = mask_to_pil(mask, shape=(2, 2))

        assert len(imgs) == 1
        img = imgs[0]
        assert isinstance(img, Image.Image)

        arr = np.array(img)
        assert arr.shape == (2, 2)
        assert arr.dtype == np.uint8

        expected = np.array([[0, 255], [0, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(arr, expected)
