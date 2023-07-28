from __future__ import annotations

from contextlib import contextmanager

from modules import img2img, processing, shared


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


@contextmanager
def cn_allow_script_control():
    orig = False
    if "control_net_allow_script_control" in shared.opts.data:
        try:
            orig = shared.opts.data["control_net_allow_script_control"]
            shared.opts.data["control_net_allow_script_control"] = True
            yield
        finally:
            shared.opts.data["control_net_allow_script_control"] = orig
    else:
        yield
