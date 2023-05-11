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
    ("ad_use_cfg_scale", "ADetailer use separate CFG scale"),
    ("ad_cfg_scale", "ADetailer CFG scale"),
    ("ad_controlnet_model", "ADetailer ControlNet model"),
    ("ad_controlnet_weight", "ADetailer ControlNet weight"),
]

ALL_ARGS = [Arg(*args) for args in _all_args]


class ADetailerArgs(BaseModel, extra=Extra.forbid):
    ad_enable: bool = False
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

        params = {name: getattr(self, attr) for attr, name in ALL_ARGS[1:]}
        params["ADetailer conf"] = int(params["ADetailer conf"] * 100)

        if not params["ADetailer prompt"]:
            params.pop("ADetailer prompt")
        if not params["ADetailer negative prompt"]:
            params.pop("ADetailer negative prompt")

        if params["ADetailer x offset"] == 0:
            params.pop("ADetailer x offset")
        if params["ADetailer y offset"] == 0:
            params.pop("ADetailer y offset")

        if not params["ADetailer use inpaint width/height"]:
            params.pop("ADetailer inpaint width")
            params.pop("ADetailer inpaint height")

        if not params["ADetailer use separate CFG scale"]:
            params.pop("ADetailer CFG scale")

        if params["ADetailer ControlNet model"] == "None":
            params.pop("ADetailer ControlNet model")
            params.pop("ADetailer ControlNet weight")

        if suffix:
            params = {k + suffix: v for k, v in params.items()}

        return params


class EnableChecker(BaseModel):
    ad_enable: bool = False
    ad_model: str = "None"

    def is_enabled(self):
        return self.ad_enable and self.ad_model != "None"


def get_one_args(*args: Any) -> ADetailerArgs:
    arg_dict = {attr: arg for arg, (attr, *_) in zip(args, ALL_ARGS)}
    return ADetailerArgs(**arg_dict)
