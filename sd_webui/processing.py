from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataclasses import dataclass, field
    from typing import Any, Callable

    import numpy as np
    import torch
    from PIL import Image

    def _image():
        return Image.new("L", (512, 512))

    @dataclass
    class StableDiffusionProcessing:
        sd_model: torch.nn.Module = field(default_factory=lambda: torch.nn.Linear(1, 1))
        outpath_samples: str = ""
        outpath_grids: str = ""
        prompt: str = ""
        prompt_for_display: str = ""
        negative_prompt: str = ""
        styles: list[str] = field(default_factory=list)
        seed: int = -1
        subseed: int = -1
        subseed_strength: float = 0.0
        seed_resize_from_h: int = -1
        seed_resize_from_w: int = -1
        sampler_name: str | None = None
        batch_size: int = 1
        n_iter: int = 1
        steps: int = 50
        cfg_scale: float = 7.0
        width: int = 512
        height: int = 512
        restore_faces: bool = False
        tiling: bool = False
        do_not_save_samples: bool = False
        do_not_save_grid: bool = False
        extra_generation_params: dict[str, Any] = field(default_factory=dict)
        overlay_images: list[Image.Image] = field(default_factory=list)
        eta: float = 0.0
        do_not_reload_embeddings: bool = False
        paste_to: tuple[int | float, ...] = (0, 0, 0, 0)
        color_corrections: list[np.ndarray] = field(default_factory=list)
        denoising_strength: float = 0.0
        sampler_noise_scheduler_override: Callable | None = None
        ddim_discretize: str = ""
        s_min_uncond: float = 0.0
        s_churn: float = 0.0
        s_tmin: float = 0.0
        s_tmax: float = 0.0
        s_noise: float = 0.0
        override_settings: dict[str, Any] = field(default_factory=dict)
        override_settings_restore_afterwards: bool = False
        is_using_inpainting_conditioning: bool = False
        disable_extra_networks: bool = False
        scripts: Any = None
        script_args: list[Any] = field(default_factory=list)
        all_prompts: list[str] = field(default_factory=list)
        all_negative_prompts: list[str] = field(default_factory=list)
        all_seeds: list[int] = field(default_factory=list)
        all_subseeds: list[int] = field(default_factory=list)
        iteration: int = 1
        is_hr_pass: bool = False

    @dataclass
    class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
        sampler: Callable | None = None
        enable_hr: bool = False
        denoising_strength: float = 0.75
        hr_scale: float = 2.0
        hr_upscaler: str = ""
        hr_second_pass_steps: int = 0
        hr_resize_x: int = 0
        hr_resize_y: int = 0
        hr_upscale_to_x: int = 0
        hr_upscale_to_y: int = 0
        width: int = 512
        height: int = 512
        truncate_x: int = 512
        truncate_y: int = 512
        applied_old_hires_behavior_to: tuple[int, int] = (512, 512)

    @dataclass
    class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
        sampler: Callable | None = None
        init_images: list[Image.Image] = field(default_factory=list)
        resize_mode: int = 0
        denoising_strength: float = 0.75
        image_cfg_scale: float | None = None
        init_latent: torch.Tensor | None = None
        image_mask: Image.Image = field(default_factory=_image)
        latent_mask: Image.Image = field(default_factory=_image)
        mask_for_overlay: Image.Image = field(default_factory=_image)
        mask_blur: int = 4
        inpainting_fill: int = 0
        inpaint_full_res: bool = True
        inpaint_full_res_padding: int = 0
        inpainting_mask_invert: int | bool = 0
        initial_noise_multiplier: float = 1.0
        mask: torch.Tensor | None = None
        nmask: torch.Tensor | None = None
        image_conditioning: torch.Tensor | None = None

    @dataclass
    class Processed:
        images: list[Image.Image] = field(default_factory=list)
        prompt: list[str] = field(default_factory=list)
        negative_prompt: list[str] = field(default_factory=list)
        seed: list[int] = field(default_factory=list)
        subseed: list[int] = field(default_factory=list)
        subseed_strength: float = 0.0
        info: str = ""
        comments: str = ""
        width: int = 512
        height: int = 512
        sampler_name: str = ""
        cfg_scale: float = 7.0
        image_cfg_scale: float | None = None
        steps: int = 50
        batch_size: int = 1
        restore_faces: bool = False
        face_restoration_model: str | None = None
        sd_model_hash: str = ""
        seed_resize_from_w: int = -1
        seed_resize_from_h: int = -1
        denoising_strength: float = 0.0
        extra_generation_params: dict[str, Any] = field(default_factory=dict)
        index_of_first_image: int = 0
        styles: list[str] = field(default_factory=list)
        job_timestamp: str = ""
        clip_skip: int = 1
        eta: float = 0.0
        ddim_discretize: str = ""
        s_churn: float = 0.0
        s_tmin: float = 0.0
        s_tmax: float = 0.0
        s_noise: float = 0.0
        sampler_noise_scheduler_override: Callable | None = None
        is_using_inpainting_conditioning: bool = False
        all_prompts: list[str] = field(default_factory=list)
        all_negative_prompts: list[str] = field(default_factory=list)
        all_seeds: list[int] = field(default_factory=list)
        all_subseeds: list[int] = field(default_factory=list)
        infotexts: list[str] = field(default_factory=list)

    def create_infotext(
        p: StableDiffusionProcessingTxt2Img | StableDiffusionProcessingImg2Img,
        all_prompts: list[str],
        all_seeds: list[int],
        all_subseeds: list[int],
        comments: Any,
        iteration: int = 0,
        position_in_batch: int = 0,
    ) -> str:
        pass

    def process_images(
        p: StableDiffusionProcessingTxt2Img | StableDiffusionProcessingImg2Img,
    ) -> Processed:
        pass

else:
    from modules.processing import (
        StableDiffusionProcessing,
        StableDiffusionProcessingImg2Img,
        StableDiffusionProcessingTxt2Img,
        create_infotext,
        process_images,
    )
