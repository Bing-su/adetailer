import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from adetailer.mask import dilate_erode, has_intersection, is_all_black, offset


def test_dilate_positive_value():
    img = Image.new("L", (10, 10), color="black")
    draw = ImageDraw.Draw(img)
    draw.rectangle((3, 3, 5, 5), fill="white")
    value = 3

    result = dilate_erode(img, value)

    assert isinstance(result, Image.Image)
    assert result.size == (10, 10)

    expect = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
            [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
            [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
            [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
            [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(np.array(result), expect)


def test_offset():
    img = Image.new("L", (10, 10), color="black")
    draw = ImageDraw.Draw(img)
    draw.rectangle((4, 4, 5, 5), fill="white")

    result = offset(img, x=1, y=2)

    assert isinstance(result, Image.Image)
    assert result.size == (10, 10)

    expect = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 255, 255, 0, 0, 0],
            [0, 0, 0, 0, 0, 255, 255, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(np.array(result), expect)


def test_is_all_black_1():
    img = Image.new("L", (10, 10), color="black")
    assert is_all_black(img)

    draw = ImageDraw.Draw(img)
    draw.rectangle((4, 4, 5, 5), fill="white")
    assert not is_all_black(img)


def test_is_all_black_2():
    img = np.zeros((10, 10), dtype=np.uint8)
    assert is_all_black(img)

    img[4:6, 4:6] = 255
    assert not is_all_black(img)


def test_is_all_black_rgb_image_pil():
    img = Image.new("RGB", (10, 10), color="red")
    assert not is_all_black(img)

    img = Image.new("RGBA", (10, 10), color="red")
    assert not is_all_black(img)


def test_is_all_black_rgb_image_numpy():
    img = np.full((10, 10, 4), 127, dtype=np.uint8)
    with pytest.raises(cv2.error):
        is_all_black(img)

    img = np.full((4, 10, 10), 0.5, dtype=np.float32)
    with pytest.raises(cv2.error):
        is_all_black(img)


def test_has_intersection_1():
    arr1 = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    arr2 = arr1.copy()
    assert not has_intersection(arr1, arr2)


def test_has_intersection_2():
    arr1 = np.array(
        [
            [0, 0, 0, 0],
            [0, 255, 255, 0],
            [0, 255, 255, 0],
            [0, 0, 0, 0],
        ]
    )
    arr2 = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ]
    )
    assert has_intersection(arr1, arr2)

    arr3 = np.array(
        [
            [255, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 255],
            [0, 0, 255, 255],
        ]
    )
    assert not has_intersection(arr1, arr3)


def test_has_intersection_3():
    img1 = Image.new("L", (10, 10), color="black")
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle((3, 3, 5, 5), fill="white")
    img2 = Image.new("L", (10, 10), color="black")
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle((6, 6, 8, 8), fill="white")
    assert not has_intersection(img1, img2)

    img3 = Image.new("L", (10, 10), color="black")
    draw3 = ImageDraw.Draw(img3)
    draw3.rectangle((2, 2, 8, 8), fill="white")
    assert has_intersection(img1, img3)


def test_has_intersection_4():
    img1 = Image.new("RGB", (10, 10), color="black")
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle((3, 3, 5, 5), fill="white")
    img2 = Image.new("RGBA", (10, 10), color="black")
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle((2, 2, 8, 8), fill="white")
    assert has_intersection(img1, img2)


def test_has_intersection_5():
    img1 = Image.new("RGB", (10, 10), color="black")
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle((4, 4, 5, 5), fill="white")
    img2 = np.full((10, 10, 4), 255, dtype=np.uint8)
    assert has_intersection(img1, img2)
