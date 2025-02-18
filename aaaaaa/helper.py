from __future__ import annotations

import os
from contextlib import contextmanager
from copy import copy
from typing import TYPE_CHECKING, Any, Union
from unittest.mock import patch

import torch
from PIL import Image
from typing_extensions import Protocol

from modules import safe
from modules.shared import cmd_opts, opts

if TYPE_CHECKING:
    # 타입 체커가 빨간 줄을 긋지 않게 하는 편법
    from types import SimpleNamespace

    StableDiffusionProcessingTxt2Img = SimpleNamespace
    StableDiffusionProcessingImg2Img = SimpleNamespace
else:
    from modules.processing import (
        StableDiffusionProcessingImg2Img,
        StableDiffusionProcessingTxt2Img,
    )

PT = Union[StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img]


@contextmanager
def change_torch_load():
    orig = torch.load
    try:
        torch.load = safe.unsafe_torch_load
        yield
    finally:
        torch.load = orig


@contextmanager
def disable_safe_unpickle():
    with (
        patch.dict(os.environ, {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}, clear=False),
        patch.object(cmd_opts, "disable_safe_unpickle", True),
    ):
        yield


@contextmanager
def pause_total_tqdm():
    with patch.dict(opts.data, {"multiple_tqdm": False}, clear=False):
        yield


@contextmanager
def preserve_prompts(p: PT):
    all_pt = copy(p.all_prompts)
    all_ng = copy(p.all_negative_prompts)
    try:
        yield
    finally:
        p.all_prompts = all_pt
        p.all_negative_prompts = all_ng


def copy_extra_params(extra_params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in extra_params.items() if not callable(v)}


class PPImage(Protocol):
    image: Image.Image
