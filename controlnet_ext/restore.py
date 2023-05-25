from __future__ import annotations

from modules import img2img, processing


def cn_restore_unet_hook(p, cn_latest_network):
    if cn_latest_network is not None:
        unet = p.sd_model.model.diffusion_model
        cn_latest_network.restore(unet)


class CNHijackRestore:
    def __init__(self):
        self.process = hasattr(processing, "__controlnet_original_process_images_inner")
        self.img2img = hasattr(img2img, "__controlnet_original_process_batch")

    def __enter__(self):
        if self.process:
            self.orig_process = processing.process_images_inner
            processing.process_images_inner = getattr(
                processing, "__controlnet_original_process_images_inner"
            )
        if self.img2img:
            self.orig_img2img = img2img.process_batch
            img2img.process_batch = getattr(
                img2img, "__controlnet_original_process_batch"
            )

    def __exit__(self, *args, **kwargs):
        if self.process:
            processing.process_images_inner = self.orig_process
        if self.img2img:
            img2img.process_batch = self.orig_img2img
