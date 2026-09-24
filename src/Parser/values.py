from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..Core.coordinates import BlockPos, ColumnPos


def require(data: Mapping, key: str):
    if key not in data:
        raise ValueError(f"missing required key {key!r}")
    return data[key]


def read_toml(path: Path, kind: str) -> Mapping:
    if not path.exists():
        raise ValueError(f"no {kind} file at {path}")
    return tomllib.loads(path.read_text())


def listed_entries(
    data: Mapping, file_key: str, entry_key: str, kind: str, base: Path
) -> tuple[Sequence[Mapping], Path]:
    # `entry_key` tables from the build, or from the file `file_key` names,
    # with the directory their paths are relative to.
    named = data.get(file_key)
    entries = data.get(entry_key, ())
    if named is not None and entries:
        raise ValueError(
            f"{kind} are listed in the build or in a file of their own, not both"
        )
    if named is None:
        return entries, base
    listing = base / named
    return read_toml(listing, kind).get(entry_key, ()), listing.parent


def read_enum(enum, value, default=None):
    if value is None:
        return default
    try:
        return enum[value.upper()]
    except KeyError:
        known = ", ".join(member.name for member in enum)
        raise ValueError(f"{value!r} is not one of {known}") from None


def read_set(value) -> frozenset[str] | None:
    return None if value is None else frozenset(value)


def read_properties(value) -> Mapping[str, frozenset[str]] | None:
    if value is None:
        return None
    return {name: frozenset(values) for name, values in value.items()}


def read_column(value: Sequence[int], name: str | None = None) -> ColumnPos:
    if len(value) != 2:
        where = f"marker {name!r}" if name else "a column"
        raise ValueError(f"{where} needs an x and a z, got {list(value)}")
    return ColumnPos.coerce(value)


def read_position(key: str) -> BlockPos:
    return BlockPos.coerce(tuple(int(part) for part in key.split(",")))
