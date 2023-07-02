from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:

    class NansException(Exception):  # noqa: N818
        pass

else:
    from modules.devices import NansException
