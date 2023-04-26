from __future__ import annotations

import importlib.util
import subprocess
import sys
from importlib.metadata import version  # python >= 3.8

from packaging.version import parse


def is_installed(package: str, min_version: str | None = None):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    if spec is None:
        return False

    if not min_version:
        return True

    try:
        return parse(version(package)) >= parse(min_version)
    except Exception:
        return False


def run_pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args])


def install():
    deps = [
        ("ultralytics", "8.0.87"),
        ("mediapipe", "0.9.3.0"),
        ("huggingface_hub", None),
    ]

    for name, ver in deps:
        if not is_installed(name, ver):
            run_pip("-U", name, "--prefer-binary")


try:
    from launch import skip_install
except ImportError:
    skip_install = False

if not skip_install:
    install()
