from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import sqrt
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


# Implementations the coordinate types share. Each type binds them as its own
# methods, so a type can replace any one of them. They read the type's
# `_components`, the names of its fields in order.


def _is_coordinate(value: object) -> bool:
    return getattr(type(value), "_components", None) is not None


def _iter(self) -> Iterator[int | float]:
    for component in self._components:
        yield getattr(self, component)


def _len(self) -> int:
    return len(self._components)


def _getitem(self, index: int | slice) -> int | float | tuple[int | float, ...]:
    if isinstance(index, slice):
        return self.as_tuple()[index]
    if index < 0:
        index += len(self._components)
    if index < 0 or index >= len(self._components):
        raise IndexError("coordinate index out of range")
    return getattr(self, self._components[index])


def _contains(self, value: object) -> bool:
    return any(item == value for item in _iter(self))


def _reversed(self) -> Iterator[int | float]:
    for component in reversed(self._components):
        yield getattr(self, component)


def _index(self, value: object) -> int:
    for index, item in enumerate(_iter(self)):
        if item == value:
            return index
    raise ValueError(f"{value!r} is not in {self!r}")


def _count(self, value: object) -> int:
    return sum(1 for item in _iter(self) if item == value)


def _as_tuple(self) -> tuple[int | float, ...]:
    if len(self._components) == 2:
        return (
            getattr(self, self._components[0]),
            getattr(self, self._components[1]),
        )
    return (
        getattr(self, self._components[0]),
        getattr(self, self._components[1]),
        getattr(self, self._components[2]),
    )


def _coerce(cls, value):
    if isinstance(value, cls):
        return value
    return cls(*value)


def _eq(self, other: object) -> bool:
    if _is_coordinate(other):
        if type(self) is not type(other):
            return False
        other_values: Sequence[int | float] = other
    elif isinstance(other, tuple):
        other_values = other
    else:
        return NotImplemented
    if len(self) != len(other_values):
        return False
    for index, component in enumerate(self._components):
        if getattr(self, component) != other_values[index]:
            return False
    return True


def _hash(self) -> int:
    # Matching tuple's hash keeps legacy tuple-keyed mappings interoperable.
    return hash(self.as_tuple())


def _lt(self, other: object) -> bool:
    if _is_coordinate(other):
        if type(self) is not type(other):
            return NotImplemented
        other_values: Sequence[int | float] = other
    elif isinstance(other, tuple):
        other_values = other
    else:
        return NotImplemented
    for index, component in enumerate(self._components):
        if index >= len(other_values):
            return False
        left = getattr(self, component)
        right = other_values[index]
        if left != right:
            return left < right
    return len(self) < len(other_values)


@runtime_checkable
class Vector(Protocol):
    """The vector math every float coordinate type provides."""

    def dot(self, other: Sequence[float]) -> float: ...

    @property
    def length(self) -> float: ...

    def unit(self) -> Vector: ...

    def distance_to(self, other: Sequence[float]) -> float: ...

    def is_close(self, other: Sequence[float], tolerance: float = 1e-9) -> bool: ...


# Component-wise implementations the vector types share. Each type calls them
# from its own methods, so a type can replace any one of them.


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _unit_components(vector: Sequence[float]) -> tuple[float, ...]:
    length = sqrt(_dot(vector, vector))
    if length == 0.0:
        raise ValueError(f"{vector} has no direction")
    return tuple(value / length for value in vector)


@dataclass(frozen=True, slots=True, eq=False)
class Vec2:
    _components = ("x", "z")

    __iter__ = _iter
    __len__ = _len
    __getitem__ = _getitem
    __contains__ = _contains
    __reversed__ = _reversed
    index = _index
    count = _count
    as_tuple = _as_tuple
    coerce = classmethod(_coerce)
    __eq__ = _eq
    __hash__ = _hash
    __lt__ = _lt

    x: float
    z: float

    def dot(self, other: Vec2 | Sequence[float]) -> float:
        return _dot(self, Vec2.coerce(other))

    @property
    def length(self) -> float:
        return sqrt(_dot(self, self))

    def unit(self) -> Vec2:
        return Vec2(*_unit_components(self))

    def distance_to(self, other: Vec2 | Sequence[float]) -> float:
        return _distance(self, Vec2.coerce(other))

    def is_close(self, other: Vec2 | Sequence[float], tolerance: float = 1e-9) -> bool:
        return self.distance_to(other) <= tolerance


@dataclass(frozen=True, slots=True, eq=False)
class Vec3:
    _components = ("x", "y", "z")

    __iter__ = _iter
    __len__ = _len
    __getitem__ = _getitem
    __contains__ = _contains
    __reversed__ = _reversed
    index = _index
    count = _count
    as_tuple = _as_tuple
    coerce = classmethod(_coerce)
    __eq__ = _eq
    __hash__ = _hash
    __lt__ = _lt

    x: float
    y: float
    z: float

    def dot(self, other: Vec3 | Sequence[float]) -> float:
        return _dot(self, Vec3.coerce(other))

    @property
    def length(self) -> float:
        return sqrt(_dot(self, self))

    def unit(self) -> Vec3:
        return Vec3(*_unit_components(self))

    def distance_to(self, other: Vec3 | Sequence[float]) -> float:
        return _distance(self, Vec3.coerce(other))

    def is_close(self, other: Vec3 | Sequence[float], tolerance: float = 1e-9) -> bool:
        return self.distance_to(other) <= tolerance

    @property
    def horizontal(self) -> Vec2:
        # Projection onto the x-z plane.
        return Vec2(self.x, self.z)


@dataclass(frozen=True, slots=True, eq=False)
class BlockPos:
    _components = ("x", "y", "z")

    __iter__ = _iter
    __len__ = _len
    __getitem__ = _getitem
    __contains__ = _contains
    __reversed__ = _reversed
    index = _index
    count = _count
    as_tuple = _as_tuple
    coerce = classmethod(_coerce)
    __eq__ = _eq
    __hash__ = _hash
    __lt__ = _lt

    x: int
    y: int
    z: int

    def __add__(self, other: Sequence[int]) -> BlockPos:
        other = BlockPos.coerce(other)
        return BlockPos(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Sequence[int]) -> BlockPos:
        other = BlockPos.coerce(other)
        return BlockPos(self.x - other.x, self.y - other.y, self.z - other.z)


@dataclass(frozen=True, slots=True, eq=False)
class ChunkPos:
    _components = ("x", "y", "z")

    __iter__ = _iter
    __len__ = _len
    __getitem__ = _getitem
    __contains__ = _contains
    __reversed__ = _reversed
    index = _index
    count = _count
    as_tuple = _as_tuple
    coerce = classmethod(_coerce)
    __eq__ = _eq
    __hash__ = _hash
    __lt__ = _lt

    x: int
    y: int
    z: int


@dataclass(frozen=True, slots=True, eq=False)
class ColumnPos:
    _components = ("x", "z")

    __iter__ = _iter
    __len__ = _len
    __getitem__ = _getitem
    __contains__ = _contains
    __reversed__ = _reversed
    index = _index
    count = _count
    as_tuple = _as_tuple
    coerce = classmethod(_coerce)
    __eq__ = _eq
    __hash__ = _hash
    __lt__ = _lt

    x: int
    z: int


# Coordinates stay virtual sequences without inheriting from Sequence.
for _type in (Vec2, Vec3, BlockPos, ChunkPos, ColumnPos):
    Sequence.register(_type)
del _type
