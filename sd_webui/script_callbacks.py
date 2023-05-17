from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

    def on_ui_settings(callback: Callable):
        pass

    def on_after_component(callback: Callable):
        pass

else:
    from modules.script_callbacks import *
