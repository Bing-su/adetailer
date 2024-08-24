from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from adetailer.opts import dynamic_denoise_strength, optimal_crop_size


@pytest.mark.parametrize(
    ("denoise_power", "denoise_strength", "bbox", "image_size", "expected_result"),
    [
        (0.001, 0.5, [0, 0, 100, 100], (200, 200), 0.4998561796520339),
        (1.5, 0.3, [0, 0, 100, 100], (200, 200), 0.1948557158514987),
        (-0.001, 0.7, [0, 0, 100, 100], (1000, 1000), 0.7000070352704507),
        (-0.5, 0.5, [0, 0, 100, 100], (1000, 1000), 0.502518907629606),
    ],
)
def test_dynamic_denoise_strength(
    denoise_power: float,
    denoise_strength: float,
    bbox: list[int],
    image_size: tuple[int, int],
    expected_result: float,
):
    result = dynamic_denoise_strength(denoise_power, denoise_strength, bbox, image_size)
    assert np.isclose(result, expected_result)


@given(denoise_strength=st.floats(allow_nan=False))
def test_dynamic_denoise_strength_no_bbox(denoise_strength: float):
    with pytest.raises(ValueError, match="bbox length must be 4, got 0"):
        dynamic_denoise_strength(0.5, denoise_strength, [], (1000, 1000))


@given(denoise_strength=st.floats(allow_nan=False))
def test_dynamic_denoise_strength_zero_power(denoise_strength: float):
    result = dynamic_denoise_strength(
        0.0, denoise_strength, [0, 0, 100, 100], (1000, 1000)
    )
    assert np.isclose(result, denoise_strength)


@given(
    inpaint_width=st.integers(1),
    inpaint_height=st.integers(1),
    bbox=st.tuples(
        st.integers(0, 500),
        st.integers(0, 500),
        st.integers(501, 1000),
        st.integers(501, 1000),
    ),
)
def test_optimal_crop_size_sdxl(
    inpaint_width: int, inpaint_height: int, bbox: tuple[int, int, int, int]
):
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    assume(bbox_width > 0 and bbox_height > 0)

    result = optimal_crop_size.sdxl(inpaint_width, inpaint_height, bbox)
    assert (result in optimal_crop_size.sdxl_res) or result == (
        inpaint_width,
        inpaint_height,
    )

    if result != (inpaint_width, inpaint_height):
        assert result[0] >= bbox_width
        assert result[1] >= bbox_height
        assert result[0] >= inpaint_width or result[1] >= inpaint_height


@given(
    inpaint_width=st.integers(1),
    inpaint_height=st.integers(1),
    bbox=st.tuples(
        st.integers(0, 500),
        st.integers(0, 500),
        st.integers(501, 1000),
        st.integers(501, 1000),
    ),
)
def test_optimal_crop_size_free(
    inpaint_width: int, inpaint_height: int, bbox: tuple[int, int, int, int]
):
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    assume(bbox_width > 0 and bbox_height > 0)

    result = optimal_crop_size.free(inpaint_width, inpaint_height, bbox)
    assert result[0] % 8 == 0
    assert result[1] % 8 == 0
