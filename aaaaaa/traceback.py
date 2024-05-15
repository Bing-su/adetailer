from __future__ import annotations

import io
import platform
import sys
from importlib.metadata import version
from typing import Any, Callable

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.traceback import Traceback

from adetailer.__version__ import __version__


def processing(*args: Any) -> dict[str, Any]:
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

    if p is None:
        return {}

    info = {
        "prompt": p.prompt,
        "negative_prompt": p.negative_prompt,
        "n_iter": p.n_iter,
        "batch_size": p.batch_size,
        "width": p.width,
        "height": p.height,
        "sampler_name": p.sampler_name,
        "enable_hr": getattr(p, "enable_hr", False),
        "hr_upscaler": getattr(p, "hr_upscaler", ""),
    }

    info.update(sd_models())
    return info


def sd_models() -> dict[str, str]:
    try:
        from modules import shared

        opts = shared.opts
    except Exception:
        return {}

    return {
        "checkpoint": getattr(opts, "sd_model_checkpoint", "------"),
        "vae": getattr(opts, "sd_vae", "------"),
        "unet": getattr(opts, "sd_unet", "------"),
    }


def ad_args(*args: Any) -> dict[str, Any]:
    ad_args = [
        arg
        for arg in args
        if isinstance(arg, dict) and arg.get("ad_model", "None") != "None"
    ]
    if not ad_args:
        return {}

    arg0 = ad_args[0]
    is_api = arg0.get("is_api", True)
    return {
        "version": __version__,
        "ad_model": arg0["ad_model"],
        "ad_prompt": arg0.get("ad_prompt", ""),
        "ad_negative_prompt": arg0.get("ad_negative_prompt", ""),
        "ad_controlnet_model": arg0.get("ad_controlnet_model", "None"),
        "is_api": type(is_api) is not tuple,
    }


def library_version():
    libraries = ["torch", "torchvision", "ultralytics", "mediapipe"]
    d = {}
    for lib in libraries:
        try:
            d[lib] = version(lib)
        except Exception:  # noqa: PERF203
            d[lib] = "Unknown"
    return d


def sys_info() -> dict[str, Any]:
    try:
        import launch

        version = launch.git_tag()
        commit = launch.commit_hash()
    except Exception:
        version = "Unknown (too old or vladmandic)"
        commit = "Unknown"

    return {
        "Platform": platform.platform(),
        "Python": sys.version,
        "Version": version,
        "Commit": commit,
        "Commandline": sys.argv,
        "Libraries": library_version(),
    }


def get_table(title: str, data: dict[str, Any]) -> Table:
    table = Table(title=title, highlight=True)
    table.add_column(" ", justify="right", style="dim")
    table.add_column("Value")
    for key, value in data.items():
        if not isinstance(value, str):
            value = repr(value)
        table.add_row(key, value)

    return table


def rich_traceback(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        string = io.StringIO()
        width = Console().width
        width = width - 4 if width > 4 else None
        console = Console(file=string, width=width)
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
            tables.append(Traceback(extra_lines=1))

            console.print(Panel(Group(*tables)))
            output = "\n" + string.getvalue()

            try:
                error = e.__class__(output)
            except Exception:
                error = RuntimeError(output)
            raise error from None

    return wrapper
