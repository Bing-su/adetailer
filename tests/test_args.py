from __future__ import annotations

import pytest

from adetailer.args import ALL_ARGS, ADetailerArgs


def test_all_args() -> None:
    args = ADetailerArgs()
    for attr, _ in ALL_ARGS:
        assert hasattr(args, attr), attr

    for attr, _ in args:
        if attr == "is_api":
            continue
        assert attr in ALL_ARGS.attrs, attr


@pytest.mark.parametrize(
    ("ad_model", "expect"),
    [("mediapipe_face_full", True), ("face_yolov8n.pt", False)],
)
def test_is_mediapipe(ad_model: str, expect: bool) -> None:
    args = ADetailerArgs(ad_model=ad_model)
    assert args.is_mediapipe() is expect


@pytest.mark.parametrize(
    ("ad_model", "expect"),
    [("mediapipe_face_full", False), ("face_yolov8n.pt", False), ("None", True)],
)
def test_need_skip(ad_model: str, expect: bool) -> None:
    args = ADetailerArgs(ad_model=ad_model)
    assert args.need_skip() is expect


@pytest.mark.parametrize(
    ("ad_model", "ad_tap_enable", "expect"),
    [
        ("face_yolov8n.pt", False, True),
        ("mediapipe_face_full", False, True),
        ("None", True, True),
        ("ace_yolov8s.pt", True, False),
    ],
)
def test_need_skip_tap_enable(ad_model: str, ad_tap_enable: bool, expect: bool) -> None:
    args = ADetailerArgs(ad_model=ad_model, ad_tap_enable=ad_tap_enable)
    assert args.need_skip() is expect
