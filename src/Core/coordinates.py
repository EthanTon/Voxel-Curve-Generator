from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from math import sqrt
from operator import attrgetter
from typing import Protocol, runtime_checkable


@runtime_checkable
class Coordinate(Protocol):
    """Tuple-like behavior every immutable coordinate type provides."""

    def __iter__(self) -> Iterator[int | float]: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int | slice) -> int | float | tuple[int | float, ...]: ...

    def __contains__(self, value: object) -> bool: ...

    def __reversed__(self) -> Iterator[int | float]: ...

    def index(self, value: object) -> int: ...

    def count(self, value: object) -> int: ...

    def as_tuple(self) -> tuple[int | float, ...]: ...

    @classmethod
    def coerce(cls, value: Sequence[int | float]) -> Coordinate: ...


def _is_coordinate(value: object) -> bool:
    return getattr(type(value), "_components", None) is not None


def _tuple_getter(components: tuple[str, ...]) -> Callable[[object], tuple]:
    if len(components) == 1:
        name = components[0]
        return lambda self: (getattr(self, name),)
    return attrgetter(*components)


def _coerce(cls, value):
    if isinstance(value, cls):
        return value
    return cls(*value)


def _coordinate_methods(components: tuple[str, ...]) -> dict[str, object]:
    # Built per class so the hot dunders read the fields directly: positions
    # are hashed and compared millions of times while resolving a build.
    values = _tuple_getter(components)
    size = len(components)

    def as_tuple(self) -> tuple[int | float, ...]:
        return values(self)

    def __iter__(self) -> Iterator[int | float]:
        return iter(values(self))

    def __len__(self) -> int:
        return size

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return values(self)[index]
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError("coordinate index out of range")
        return values(self)[index]

    def __contains__(self, value: object) -> bool:
        return any(item == value for item in values(self))

    def __reversed__(self) -> Iterator[int | float]:
        return reversed(values(self))

    def index(self, value: object) -> int:
        for position, item in enumerate(values(self)):
            if item == value:
                return position
        raise ValueError(f"{value!r} is not in {self!r}")

    def count(self, value: object) -> int:
        return sum(1 for item in values(self) if item == value)

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return values(self) == values(other)
        if _is_coordinate(other):
            return False
        if isinstance(other, tuple):
            return values(self) == other
        return NotImplemented

    def __hash__(self) -> int:
        # Equal to the tuple's hash, so tuple-keyed mappings still find it.
        return hash(values(self))

    def __lt__(self, other: object) -> bool:
        if type(other) is type(self):
            return values(self) < values(other)
        if _is_coordinate(other) or not isinstance(other, tuple):
            return NotImplemented
        return values(self) < other

    return {
        "__iter__": __iter__,
        "__len__": __len__,
        "__getitem__": __getitem__,
        "__contains__": __contains__,
        "__reversed__": __reversed__,
        "index": index,
        "count": count,
        "as_tuple": as_tuple,
        "coerce": classmethod(_coerce),
        "__eq__": __eq__,
        "__hash__": __hash__,
        "__lt__": __lt__,
    }


def coordinate_methods(cls: type) -> type:
    # Gives a class with `_components` the shared methods it does not define
    # and registers it as a Sequence. Apply above @dataclass(eq=False), since
    # a slotted dataclass is rebuilt as a new class.
    for name, method in _coordinate_methods(cls._components).items():
        if name not in cls.__dict__:
            setattr(cls, name, method)
    Sequence.register(cls)
    return cls


@runtime_checkable
class Vector(Protocol):
    """The vector math every float coordinate type provides."""

    def dot(self, other: Sequence[float]) -> float: ...

    @property
    def length(self) -> float: ...

    def unit(self) -> Vector: ...

    def distance_to(self, other: Sequence[float]) -> float: ...

    def is_close(self, other: Sequence[float], tolerance: float = 1e-9) -> bool: ...


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _unit_components(vector: Sequence[float]) -> tuple[float, ...]:
    length = sqrt(_dot(vector, vector))
    if length == 0.0:
        raise ValueError(f"{vector} has no direction")
    return tuple(value / length for value in vector)


def _vector_dot(self, other: Sequence[float]) -> float:
    return _dot(self, type(self).coerce(other))


def _vector_length(self) -> float:
    return sqrt(_dot(self, self))


def _vector_unit(self):
    return type(self)(*_unit_components(self))


def _vector_distance_to(self, other: Sequence[float]) -> float:
    return _distance(self, type(self).coerce(other))


def _vector_is_close(self, other: Sequence[float], tolerance: float = 1e-9) -> bool:
    return self.distance_to(other) <= tolerance


_VECTOR_METHODS = {
    "dot": _vector_dot,
    "length": property(_vector_length),
    "unit": _vector_unit,
    "distance_to": _vector_distance_to,
    "is_close": _vector_is_close,
}


def vector_methods(cls: type) -> type:
    # Apply above @dataclass, like coordinate_methods.
    for name, method in _VECTOR_METHODS.items():
        if name not in cls.__dict__:
            setattr(cls, name, method)
    return cls


def quarter_turn(x: float, z: float, turns: int = 1) -> tuple[float, float]:
    # Clockwise seen from above: north (0, -1) becomes east (1, 0).
    for _ in range(turns % 4):
        x, z = -z, x
    return x, z


@vector_methods
@coordinate_methods
@dataclass(frozen=True, slots=True, eq=False)
class Vec2:
    _components = ("x", "z")

    x: float
    z: float

    def cross(self, other: Vec2 | Sequence[float]) -> float:
        other = Vec2.coerce(other)
        return self.x * other.z - self.z * other.x


@vector_methods
@coordinate_methods
@dataclass(frozen=True, slots=True, eq=False)
class Vec3:
    _components = ("x", "y", "z")

    x: float
    y: float
    z: float

    @property
    def horizontal(self) -> Vec2:
        return Vec2(self.x, self.z)


@coordinate_methods
@dataclass(frozen=True, slots=True, eq=False)
class BlockPos:
    _components = ("x", "y", "z")

    x: int
    y: int
    z: int

    def __add__(self, other: Sequence[int]) -> BlockPos:
        other = BlockPos.coerce(other)
        return BlockPos(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Sequence[int]) -> BlockPos:
        other = BlockPos.coerce(other)
        return BlockPos(self.x - other.x, self.y - other.y, self.z - other.z)


@coordinate_methods
@dataclass(frozen=True, slots=True, eq=False)
class ChunkPos:
    _components = ("x", "y", "z")

    x: int
    y: int
    z: int


@coordinate_methods
@dataclass(frozen=True, slots=True, eq=False)
class ColumnPos:
    _components = ("x", "z")

    x: int
    z: int
