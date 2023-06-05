from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from abc import ABC, abstractmethod
    from collections import namedtuple
    from dataclasses import dataclass
    from typing import Any

    import gradio as gr
    from PIL import Image

    from sd_webui.processing import (
        Processed,
        StableDiffusionProcessingImg2Img,
        StableDiffusionProcessingTxt2Img,
    )

    SDPType = StableDiffusionProcessingImg2Img | StableDiffusionProcessingTxt2Img
    AlwaysVisible = object()

    @dataclass
    class PostprocessImageArgs:
        image: Image.Image

    class Script(ABC):
        filename: str
        args_from: int
        args_to: int
        alwayson: bool

        is_txt2img: bool
        is_img2img: bool

        group: gr.Group
        infotext_fields: list[tuple[str, str]]
        paste_field_names: list[str]

        @abstractmethod
        def title(self):
            raise NotImplementedError

        def ui(self, is_img2img: bool):
            pass

        def show(self, is_img2img: bool):
            return True

        def run(self, p: SDPType, *args):
            pass

        def process(self, p: SDPType, *args):
            pass

        def before_process_batch(self, p: SDPType, *args, **kwargs):
            pass

        def process_batch(self, p: SDPType, *args, **kwargs):
            pass

        def postprocess_batch(self, p: SDPType, *args, **kwargs):
            pass

        def postprocess_image(self, p: SDPType, pp: PostprocessImageArgs, *args):
            pass

        def postprocess(self, p: SDPType, processed: Processed, *args):
            pass

        def before_component(self, component, **kwargs):
            pass

        def after_component(self, component, **kwargs):
            pass

        def describe(self):
            return ""

        def elem_id(self, item_id: Any) -> str:
            pass

    ScriptClassData = namedtuple(
        "ScriptClassData", ["script_class", "path", "basedir", "module"]
    )
    scripts_data: list[ScriptClassData] = []

else:
    from modules.scripts import (
        AlwaysVisible,
        PostprocessImageArgs,
        Script,
        scripts_data,
    )
