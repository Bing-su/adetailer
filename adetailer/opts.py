from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, TypeVar

import numpy as np

T = TypeVar("T", int, float)


def dynamic_denoise_strength(
    denoise_power: float,
    denoise_strength: float,
    bbox: Sequence[T],
    image_size: tuple[int, int],
) -> float:
    if len(bbox) != 4:
        msg = f"bbox length must be 4, got {len(bbox)}"
        raise ValueError(msg)

    if np.isclose(denoise_power, 0.0) or len(bbox) != 4:
        return denoise_strength

    width, height = image_size

    image_pixels = width * height
    bbox_pixels = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    normalized_area = bbox_pixels / image_pixels
    denoise_modifier = (1.0 - normalized_area) ** denoise_power

    return denoise_strength * denoise_modifier


class _OptimalCropSize:
    sdxl_res: ClassVar[list[tuple[int, int]]] = [
        (1024, 1024),
        (1152, 896),
        (896, 1152),
        (1216, 832),
        (832, 1216),
        (1344, 768),
        (768, 1344),
        (1536, 640),
        (640, 1536),
    ]

    def sdxl(
        self, inpaint_width: int, inpaint_height: int, bbox: Sequence[T]
    ) -> tuple[int, int]:
        if len(bbox) != 4:
            msg = f"bbox length must be 4, got {len(bbox)}"
            raise ValueError(msg)

        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_aspect_ratio = bbox_width / bbox_height

        resolutions = [
            res
            for res in self.sdxl_res
            if (res[0] >= bbox_width and res[1] >= bbox_height)
            and (res[0] >= inpaint_width or res[1] >= inpaint_height)
        ]

        if not resolutions:
            return inpaint_width, inpaint_height

        return min(
            resolutions,
            key=lambda res: abs((res[0] / res[1]) - bbox_aspect_ratio),
        )

    def free(
        self, inpaint_width: int, inpaint_height: int, bbox: Sequence[T]
    ) -> tuple[int, int]:
        if len(bbox) != 4:
            msg = f"bbox length must be 4, got {len(bbox)}"
            raise ValueError(msg)

        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_aspect_ratio = bbox_width / bbox_height

        scale_size = max(inpaint_width, inpaint_height)

        if bbox_aspect_ratio > 1:
            optimal_width = scale_size
            optimal_height = scale_size / bbox_aspect_ratio
        else:
            optimal_width = scale_size * bbox_aspect_ratio
            optimal_height = scale_size

        # Round up to the nearest multiple of 8 to make the dimensions friendly for upscaling/diffusion.
        optimal_width = ((optimal_width + 8 - 1) // 8) * 8
        optimal_height = ((optimal_height + 8 - 1) // 8) * 8

        return int(optimal_width), int(optimal_height)


optimal_crop_size = _OptimalCropSize()
