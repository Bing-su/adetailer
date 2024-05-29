from __future__ import annotations


def need_call_process(p) -> bool:
    if p.scripts is None:
        return False
    i = p.batch_index
    bs = p.batch_size
    return i == bs - 1


def need_call_postprocess(p) -> bool:
    if p.scripts is None:
        return False
    return p.batch_index == 0


def is_img2img_inpaint(p) -> bool:
    return hasattr(p, "image_mask") and p.image_mask is not None


def is_inpaint_only_masked(p) -> bool:
    return hasattr(p, "inpaint_full_res") and p.inpaint_full_res


def get_i(p) -> int:
    it = p.iteration
    bs = p.batch_size
    i = p.batch_index
    return it * bs + i


def is_skip_img2img(p) -> bool:
    return getattr(p, "_ad_skip_img2img", False)
