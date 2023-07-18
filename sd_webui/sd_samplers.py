from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, NamedTuple

    class SamplerData(NamedTuple):
        name: str
        constructor: Callable
        aliases: list[str]
        options: dict[str, Any]

    all_samplers: list[SamplerData] = []

else:
    from modules.sd_samplers import all_samplers
