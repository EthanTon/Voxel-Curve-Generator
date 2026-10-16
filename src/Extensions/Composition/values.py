from collections.abc import Mapping
from math import isfinite


def check_keys(data: Mapping, allowed: str, context: str) -> None:
    if not isinstance(data, Mapping):
        raise ValueError(f"{context} must be a table")
    unknown = set(data) - set(allowed.split())
    if unknown:
        raise ValueError(f"{context} has unknown options: {sorted(unknown)}")


def number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def coordinates(value, size: int, name: str) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} needs {size} coordinates")
    return tuple(number(each, name) for each in value)


def stations(value, count: int, name: str) -> tuple[int, int]:
    start, stop = coordinates(value, 2, name)
    start, stop = integer(start, name), integer(stop, name)
    if not 0 <= start <= stop < count:
        raise ValueError(f"{name} needs 0 <= start <= stop < {count}")
    return start, stop


def names(value, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(v, str) or not v for v in value):
        raise ValueError(f"{name} needs a list of names")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} needs distinct names")
    return tuple(value)
