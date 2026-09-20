from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..Core.coordinates import BlockPos, ColumnPos


def require(data: Mapping, key: str):
    if key not in data:
        raise ValueError(f"missing required key {key!r}")
    return data[key]


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
