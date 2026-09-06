from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Self


class _Coordinate(Sequence[int | float]):
    """Shared tuple-like behavior for immutable coordinate values."""

    __slots__ = ()

    _components: tuple[str, ...]

    def __iter__(self) -> Iterator[int | float]:
        for component in self._components:
            yield getattr(self, component)

    def __len__(self) -> int:
        return len(self._components)

    def __getitem__(self, index: int | slice) -> int | float | tuple[int | float, ...]:
        if isinstance(index, slice):
            return self.as_tuple()[index]
        if index < 0:
            index += len(self._components)
        if index < 0 or index >= len(self._components):
            raise IndexError("coordinate index out of range")
        return getattr(self, self._components[index])

    def as_tuple(self) -> tuple[int | float, ...]:
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

    @classmethod
    def coerce(cls, value: Self | Sequence[int | float]) -> Self:
        if isinstance(value, cls):
            return value
        return cls(*value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _Coordinate):
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

    def __hash__(self) -> int:
        # Matching tuple's hash keeps legacy tuple-keyed mappings interoperable.
        return hash(self.as_tuple())

    def __lt__(self, other: object) -> bool:
        if isinstance(other, _Coordinate):
            if type(self) is not type(other):
                return NotImplemented
            other_values: Sequence[int | float] = other
        if isinstance(other, tuple):
            other_values = other
        elif not isinstance(other, _Coordinate):
            return NotImplemented
        for index, component in enumerate(self._components):
            if index >= len(other_values):
                return False
            left = getattr(self, component)
            right = other_values[index]
            if left != right:
                return left < right
        return len(self) < len(other_values)


@dataclass(frozen=True, slots=True, eq=False)
class Vec2(_Coordinate):
    _components = ("x", "z")

    x: float
    z: float


@dataclass(frozen=True, slots=True, eq=False)
class Vec3(_Coordinate):
    _components = ("x", "y", "z")

    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True, eq=False)
class BlockPos(_Coordinate):
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


@dataclass(frozen=True, slots=True, eq=False)
class ChunkPos(_Coordinate):
    _components = ("x", "y", "z")

    x: int
    y: int
    z: int


@dataclass(frozen=True, slots=True, eq=False)
class ColumnPos(_Coordinate):
    _components = ("x", "z")

    x: int
    z: int
