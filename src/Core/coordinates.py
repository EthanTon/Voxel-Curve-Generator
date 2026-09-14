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


# Implementations the coordinate types share, bound by coordinate_methods. A
# type replaces any one of them by defining that method itself. They read the
# type's `_components`, the names of its fields in order.


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


_COORDINATE_METHODS = {
    "__iter__": _iter,
    "__len__": _len,
    "__getitem__": _getitem,
    "__contains__": _contains,
    "__reversed__": _reversed,
    "index": _index,
    "count": _count,
    "as_tuple": _as_tuple,
    "coerce": classmethod(_coerce),
    "__eq__": _eq,
    "__hash__": _hash,
    "__lt__": _lt,
}


def coordinate_methods(cls: type) -> type:
    # Give a class with `_components` the shared coordinate methods it does not
    # define itself, and register it as a Sequence. Apply it above
    # @dataclass(eq=False): slotted dataclasses are rebuilt as new classes, so
    # the binding and registration must happen to the final class. A class
    # that defines __eq__ must also define __hash__, as Python requires.
    for name, method in _COORDINATE_METHODS.items():
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


# Implementations the vector types share, bound by vector_methods. A type
# replaces any one of them by defining that method itself.


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
    # Give a coordinate class the shared vector methods it does not define
    # itself. Apply it above @dataclass, like coordinate_methods.
    for name, method in _VECTOR_METHODS.items():
        if name not in cls.__dict__:
            setattr(cls, name, method)
    return cls


def quarter_turn(x: float, z: float, turns: int = 1) -> tuple[float, float]:
    # Rotate a column clockwise, seen from above, by quarter turns. This is the
    # one definition of a clockwise quarter turn: north (0, -1) becomes east
    # (1, 0).
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
        # Signed area of the parallelogram the two vectors span: zero when they
        # are parallel.
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
        # Projection onto the x-z plane.
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
