from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from dataclasses import dataclass
    from typing import Any, Callable

    import torch
    from PIL import Image

    @dataclass
    class State:
        skipped: bool = False
        interrupted: bool = False
        job: str = ""
        job_no: int = 0
        job_count: int = 0
        processing_has_refined_job_count: bool = False
        job_timestamp: str = "0"
        sampling_step: int = 0
        sampling_steps: int = 0
        current_latent: torch.Tensor | None = None
        current_image: Image.Image | None = None
        current_image_sampling_step: int = 0
        id_live_preview: int = 0
        textinfo: str | None = None
        time_start: float | None = None
        need_restart: bool = False
        server_start: float | None = None

    @dataclass
    class OptionInfo:
        default: Any = None
        label: str = ""
        component: Any = None
        component_args: Callable[[], dict] | dict[str, Any] | None = None
        onchange: Callable[[], None] | None = None
        section: tuple[str, str] | None = None
        refresh: Callable[[], None] | None = None

    class Option:
        data_labels: dict[str, OptionInfo]

        def __init__(self):
            self.data: dict[str, Any] = {}

        def add_option(self, key: str, info: OptionInfo):
            pass

        def __getattr__(self, item: str):
            if self.data is not None and item in self.data:
                return self.data[item]

            if item in self.data_labels:
                return self.data_labels[item].default

            return super().__getattribute__(item)

    opts = Option()
    cmd_opts = argparse.Namespace()
    state = State()

else:
    from modules.shared import OptionInfo, cmd_opts, opts, state
