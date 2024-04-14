from __future__ import annotations

from adetailer.args import ALL_ARGS, ADetailerArgs


def test_all_args() -> None:
    args = ADetailerArgs()
    for attr, _ in ALL_ARGS:
        assert hasattr(args, attr), attr

    for attr, _ in args:
        if attr == "is_api":
            continue
        assert attr in ALL_ARGS.attrs, attr
