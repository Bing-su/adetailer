from functools import cache

import pytest
import requests
from PIL import Image


@cache
def _sample_image():
    url = "https://i.imgur.com/E5OVXvn.png"
    resp = requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
    return Image.open(resp.raw)


@cache
def _sample_image2():
    url = "https://i.imgur.com/px5UT7T.png"
    resp = requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
    return Image.open(resp.raw)


@pytest.fixture()
def sample_image():
    return _sample_image()


@pytest.fixture()
def sample_image2():
    return _sample_image2()
