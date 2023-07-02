from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

    def on_app_started(callback: Callable):
        pass

    def on_ui_settings(callback: Callable):
        pass

    def on_after_component(callback: Callable):
        pass

    def on_before_ui(callback: Callable):
        pass

else:
    from modules.script_callbacks import (
        on_after_component,
        on_app_started,
        on_before_ui,
        on_ui_settings,
    )
