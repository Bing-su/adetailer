from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os

    models_path = os.path.join(os.path.dirname(__file__), "1")
    script_path = os.path.join(os.path.dirname(__file__), "2")
    data_path = os.path.join(os.path.dirname(__file__), "3")
    extensions_dir = os.path.join(os.path.dirname(__file__), "4")
    extensions_builtin_dir = os.path.join(os.path.dirname(__file__), "5")
else:
    from modules.paths import data_path, models_path, script_path
