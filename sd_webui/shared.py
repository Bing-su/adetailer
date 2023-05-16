from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from dataclasses import dataclass
    from typing import Any, Callable

    @dataclass
    class OptionInfo:
        default: Any = None
        label: str = ""
        component: Any = None
        component_args: dict[str, Any] | None = None
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

else:
    from modules.shared import *
