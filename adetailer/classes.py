from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def is_world_model(model_path: str | Path) -> bool:
    return "-world" in Path(model_path).stem


def parse_csv(csv: str) -> list[str]:
    return [c.strip() for c in (csv or "").split(",") if c.strip()]


def _scalar_list(seq: Any) -> list[str]:
    """Filter a list-like to scalars, stringified. Non-list input → []."""
    if not isinstance(seq, list):
        return []
    return [str(x) for x in seq if isinstance(x, (str, int, float))]


def _names_from_int_keyed_dict(d: Any) -> list[str]:
    """Treat `d` as `{"0": "face", "1": "hand", ...}`. Return [] if the shape
    doesn't match: requires ALL keys int-parseable AND all values scalar.
    """
    if not isinstance(d, dict):
        return []
    try:
        int_keys = [int(k) for k in d]
    except (TypeError, ValueError):
        return []
    if not int_keys or len(int_keys) != len(d):
        return []
    out: list[str] = []
    for i in sorted(int_keys):
        v = d.get(str(i))
        if not isinstance(v, (str, int, float)):
            return []
        out.append(str(v))
    return out


def _names_from_json(data: Any) -> list[str]:
    """Try to extract a class-names list from a parsed JSON blob.

    Returns [] when the blob is *not* a recognized class-names format.
    The caller is expected to fall through to another resolution path on [].
    Recognized formats:
      - ["face", "hand", ...]                          (list of names)
      - {"names": ["face", "hand", ...]}               (Ultralytics-export-style)
      - {"names": {"0": "face", "1": "hand", ...}}     (Ultralytics-dict-style)
      - {"0": "face", "1": "hand", ...}                (bare integer-keyed map)
    Anything else (e.g. civitai_helper sidecar JSONs) returns [].
    """
    if isinstance(data, list):
        return _scalar_list(data)
    if not isinstance(data, dict):
        return []

    # `{"names": ...}` — Ultralytics export shapes.
    if "names" in data:
        inner = data["names"]
        if isinstance(inner, list):
            return _scalar_list(inner)
        return _names_from_int_keyed_dict(inner)

    # Bare `{"0": "face", ...}` map. Anything else is unrelated metadata.
    return _names_from_int_keyed_dict(data)


@lru_cache(maxsize=32)
def get_model_class_names(model_path: str) -> list[str]:
    """Resolve class names for a YOLO model.

    Resolution order:
      1. Sidecar JSON file next to the .pt — only used if it parses into a
         recognized class-names format. Unrelated JSONs (e.g. civitai_helper
         metadata) are silently ignored.
      2. model.names from a transient YOLO() load.
      3. [] if unknown (YOLO-World, MediaPipe, missing file, or load failure).
    """
    p = Path(model_path)
    if is_world_model(p) or not p.exists() or p.suffix != ".pt":
        return []

    sidecar = p.with_suffix(".json")
    if sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = None
        if data is not None:
            names = _names_from_json(data)
            if names:
                return names

    try:
        from ultralytics import YOLO

        names = YOLO(str(p)).names
        if isinstance(names, dict):
            return [str(names[i]) for i in sorted(names)]
        return [str(n) for n in (names or [])]
    except Exception:
        return []


def resolve_class_ids(model_path: str, requested: list[str]) -> list[int]:
    """Convert user-provided class names (or numeric ids as strings) to int ids.
    Unknown entries are silently dropped — matches uddetailer's behavior.
    """
    names = get_model_class_names(model_path)
    out: list[int] = []
    for token in requested:
        if token.isdigit():
            i = int(token)
            if 0 <= i < max(1, len(names) or 10_000):
                out.append(i)
            continue
        if token in names:
            out.append(names.index(token))
    return out
