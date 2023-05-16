from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from dataclasses import dataclass
    from typing import Any, Callable

    @dataclass
    class OptionInfo:
        default: Any
        label: str
        component: Any
        component_args: dict[str, Any]
        onchange: Callable[[], None]
        section: tuple[str, str]
        refresh: Callable[[], None]

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
    from module.shared import *
