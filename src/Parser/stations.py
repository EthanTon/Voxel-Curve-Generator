from __future__ import annotations


def validate_station(value: object, where: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < -1:
        raise ValueError(f"{where} needs a nonnegative integer or -1 for the last station")


def resolve_station(value: object, count: int, where: str) -> int:
    validate_station(value, where)
    index = count - 1 if value == -1 else value
    if not 0 <= index < count:
        raise ValueError(f"{where} must be from 0 to {count - 1}, or -1 for the last station")
    return index
