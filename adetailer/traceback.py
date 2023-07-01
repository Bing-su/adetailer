from __future__ import annotations

import io
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.traceback import Traceback


def processing(*args):
    try:
        from modules.processing import (
            StableDiffusionProcessingImg2Img,
            StableDiffusionProcessingTxt2Img,
        )
    except ImportError:
        return {}

    p = None
    for arg in args:
        if isinstance(
            arg, (StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img)
        ):
            p = arg
            break

    if not p:
        return {}

    return {
        "prompt": p.prompt,
        "negative_prompt": p.negative_prompt,
        "n_iter": p.n_iter,
        "batch_size": p.batch_size,
        "width": p.width,
        "height": p.height,
        "enable_hr": getattr(p, "enable_hr", False),
    }


def ad_args(*args):
    args = [
        arg
        for arg in args
        if isinstance(arg, dict) and arg.get("ad_model", "None") != "None"
    ]
    if not args:
        return {}

    arg0 = args[0]
    return {
        "ad_model": arg0["ad_model"],
        "ad_prompt": arg0.get("ad_prompt", ""),
        "ad_negative_prompt": arg0.get("ad_negative_prompt", ""),
        "ad_controlnet_model": arg0.get("ad_controlnet_model", "None"),
    }


def sys_info():
    import platform

    try:
        import launch

        version = launch.git_tag()
        commit = launch.commit_hash()
    except Exception:
        return {}

    return {
        "Platform": platform.platform(),
        "Python": platform.python_version(),
        "Version": version,
        "Commit": commit,
    }


def get_table(title: str, data: dict[str, Any]) -> Table:
    table = Table(title=title, highlight=True)
    for key, value in data.items():
        if not isinstance(value, str):
            value = repr(value)
        table.add_row(key, value)

    return table


def rich_traceback(func):
    def wrapper(*args, **kwargs):
        string = io.StringIO()
        width = Console().width
        width = width - 4 if width > 4 else None
        console = Console(file=string, force_terminal=True, width=width)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            tables = [
                get_table(title, data)
                for title, data in [
                    ("System info", sys_info()),
                    ("Inputs", processing(*args)),
                    ("ADetailer", ad_args(*args)),
                ]
                if data
            ]
            tables.append(Traceback())

            console.print(Panel(Group(*tables)))
            output = "\n" + string.getvalue()

            try:
                error = e.__class__(output)
            except Exception:
                error = RuntimeError(output)
            raise error from None

    return wrapper
