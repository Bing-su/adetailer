import pytest
import requests
from PIL import Image


def get_image(url: str) -> Image.Image:
    resp = requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
    return Image.open(resp.raw)


@pytest.fixture(scope="session")
def sample_image():
    return get_image("https://i.imgur.com/E5OVXvn.png")


@pytest.fixture(scope="session")
def sample_image2():
    return get_image("https://i.imgur.com/px5UT7T.png")
