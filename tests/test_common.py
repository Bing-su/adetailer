import numpy as np
from PIL import Image, ImageDraw

from adetailer.common import create_bbox_from_mask, create_mask_from_bbox


def test_create_mask_from_bbox():
    img = Image.new("L", (10, 10), color="black")
    bbox = [[1.0, 1.0, 2.0, 2.0], [7.0, 7.0, 8.0, 8.0]]
    masks = create_mask_from_bbox(bbox, img.size)
    expect1 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 255, 255, 0, 0, 0, 0, 0, 0, 0],
            [0, 255, 255, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    expect2 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 255, 255, 0],
            [0, 0, 0, 0, 0, 0, 0, 255, 255, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert len(masks) == len(bbox)
    arr1 = np.array(masks[0])
    arr2 = np.array(masks[1])
    assert arr1.shape == expect1.shape
    assert arr2.shape == expect2.shape
    assert arr1.shape == (10, 10)
    assert arr1.dtype == expect1.dtype
    assert arr2.dtype == expect2.dtype
    assert np.array_equal(arr1, expect1)
    assert np.array_equal(arr2, expect2)

    # The function correctly receives a list of masks and the shape of the image.


def test_create_bbox_from_mask():
    mask = Image.new("L", (10, 10), color="black")
    draw = ImageDraw.Draw(mask)
    draw.rectangle((2, 2, 5, 5), fill="white")

    result = create_bbox_from_mask([mask], (10, 10))

    assert isinstance(result, list)
    assert len(result) == 1
    assert all(isinstance(bbox, list) for bbox in result)
    assert all(len(bbox) == 4 for bbox in result)
    assert result[0] == [2, 2, 6, 6]

    result = create_bbox_from_mask([mask], (256, 256))
    assert result[0] == [38, 38, 166, 166]
