import re

cn_model_module = {
    "inpaint": "inpaint_global_harmonious",
    "scribble": "t2ia_sketch_pidi",
    "lineart": "lineart_coarse",
    "openpose": "openpose_full",
    "tile": "tile_resample",
    "depth": "depth_midas",
}
cn_model_regex = re.compile("|".join(cn_model_module.keys()), flags=re.I)
