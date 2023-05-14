from __future__ import annotations

from collections import UserList
from collections.abc import Mapping
from functools import cached_property
from typing import Any, NamedTuple

import pydantic
from pydantic import (
    BaseModel,
    Extra,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    confloat,
    validator,
)


class Arg(NamedTuple):
    attr: str
    name: str


class ArgsList(UserList):
    @cached_property
    def attrs(self) -> tuple[str]:
        return tuple(attr for attr, _ in self)

    @cached_property
    def names(self) -> tuple[str]:
        return tuple(name for _, name in self)


class ADetailerArgs(BaseModel, extra=Extra.forbid):
    ad_model: str = "None"
    ad_prompt: str = ""
    ad_negative_prompt: str = ""
    ad_conf: confloat(ge=0.0, le=1.0) = 0.3
    ad_dilate_erode: int = 32
    ad_x_offset: int = 0
    ad_y_offset: int = 0
    ad_mask_blur: NonNegativeInt = 4
    ad_denoising_strength: confloat(ge=0.0, le=1.0) = 0.4
    ad_inpaint_full_res: bool = True
    ad_inpaint_full_res_padding: NonNegativeInt = 0
    ad_use_inpaint_width_height: bool = False
    ad_inpaint_width: PositiveInt = 512
    ad_inpaint_height: PositiveInt = 512
    ad_use_steps: bool = False
    ad_steps: PositiveInt = 28
    ad_use_cfg_scale: bool = False
    ad_cfg_scale: NonNegativeFloat = 7.0
    ad_controlnet_model: str = "None"
    ad_controlnet_weight: confloat(ge=0.0, le=1.0) = 1.0

    @validator("ad_conf", pre=True)
    def check_ad_conf(cls, v: Any):  # noqa: N805
        "ad_conf가 문자열로 들어올 경우를 대비"
        if not isinstance(v, (int, float)):
            try:
                v = int(v)
            except ValueError:
                v = float(v)
        if isinstance(v, int):
            v /= 100.0
        return v

    def extra_params(self, suffix: str = ""):
        if self.ad_model == "None":
            return {}

        params = {name: getattr(self, attr) for attr, name in ALL_ARGS}
        params["ADetailer conf"] = int(params["ADetailer conf"] * 100)

        if not params["ADetailer prompt"]:
            params.pop("ADetailer prompt")
        if not params["ADetailer negative prompt"]:
            params.pop("ADetailer negative prompt")

        if params["ADetailer x offset"] == 0:
            params.pop("ADetailer x offset")
        if params["ADetailer y offset"] == 0:
            params.pop("ADetailer y offset")

        if not params["ADetailer inpaint full"]:
            params.pop("ADetailer inpaint padding")

        if not params["ADetailer use inpaint width/height"]:
            params.pop("ADetailer use inpaint width/height")
            params.pop("ADetailer inpaint width")
            params.pop("ADetailer inpaint height")

        if not params["ADetailer use separate steps"]:
            params.pop("ADetailer use separate steps")
            params.pop("ADetailer steps")

        if not params["ADetailer use separate CFG scale"]:
            params.pop("ADetailer use separate CFG scale")
            params.pop("ADetailer CFG scale")

        if params["ADetailer ControlNet model"] == "None":
            params.pop("ADetailer ControlNet model")
            params.pop("ADetailer ControlNet weight")

        if suffix:
            params = {k + suffix: v for k, v in params.items()}

        return params


def enable_check(*args: Any) -> bool:
    if not args:
        return False
    a0: bool | Mapping = args[0]
    ad_model = ALL_ARGS[0].attr

    if isinstance(a0, Mapping):
        return a0.get(ad_model, "None") != "None"
    if len(args) == 1:
        return False

    a1 = args[1]
    a1_model = a1.get(ad_model, "None")
    return a0 and a1_model != "None"


_all_args = [
    ("ad_enable", "ADetailer enable"),
    ("ad_model", "ADetailer model"),
    ("ad_prompt", "ADetailer prompt"),
    ("ad_negative_prompt", "ADetailer negative prompt"),
    ("ad_conf", "ADetailer conf"),
    ("ad_dilate_erode", "ADetailer dilate/erode"),
    ("ad_x_offset", "ADetailer x offset"),
    ("ad_y_offset", "ADetailer y offset"),
    ("ad_mask_blur", "ADetailer mask blur"),
    ("ad_denoising_strength", "ADetailer denoising strength"),
    ("ad_inpaint_full_res", "ADetailer inpaint full"),
    ("ad_inpaint_full_res_padding", "ADetailer inpaint padding"),
    ("ad_use_inpaint_width_height", "ADetailer use inpaint width/height"),
    ("ad_inpaint_width", "ADetailer inpaint width"),
    ("ad_inpaint_height", "ADetailer inpaint height"),
    ("ad_use_steps", "ADetailer use separate steps"),
    ("ad_steps", "ADetailer steps"),
    ("ad_use_cfg_scale", "ADetailer use separate CFG scale"),
    ("ad_cfg_scale", "ADetailer CFG scale"),
    ("ad_controlnet_model", "ADetailer ControlNet model"),
    ("ad_controlnet_weight", "ADetailer ControlNet weight"),
]

AD_ENABLE = Arg(*_all_args[0])
_args = [Arg(*args) for args in _all_args[1:]]
ALL_ARGS = ArgsList(_args)
