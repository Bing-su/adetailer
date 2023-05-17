from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    unsafe_torch_load = torch.load
else:
    from modules.safe import unsafe_torch_load
