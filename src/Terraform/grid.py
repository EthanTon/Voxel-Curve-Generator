from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from ..Core.coordinates import ColumnPos


Field = np.ndarray


@dataclass(frozen=True, slots=True)
class Grid:
    origin: ColumnPos
    width: int
    depth: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", ColumnPos.coerce(self.origin))
        if self.width <= 0 or self.depth <= 0:
            raise ValueError(f"grid must be positive, got {self.width}x{self.depth}")

    @classmethod
    def around(cls, columns: Iterable[ColumnPos], margin: int = 0) -> Grid:
        walked = [ColumnPos.coerce(each) for each in columns]
        if not walked:
            raise ValueError("a grid needs at least one column")
        least_x = min(each.x for each in walked) - margin
        least_z = min(each.z for each in walked) - margin
        most_x = max(each.x for each in walked) + margin
        most_z = max(each.z for each in walked) + margin
        return cls(ColumnPos(least_x, least_z), most_x - least_x + 1, most_z - least_z + 1)

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth, self.width

    def holds(self, x: int, z: int) -> bool:
        return (
            0 <= x - self.origin.x < self.width
            and 0 <= z - self.origin.z < self.depth
        )

    def at(self, x: int, z: int) -> tuple[int, int]:
        return z - self.origin.z, x - self.origin.x

    def column(self, row: int, cell: int) -> ColumnPos:
        return ColumnPos(self.origin.x + cell, self.origin.z + row)

    def columns(self) -> Iterator[tuple[int, int, ColumnPos]]:
        for row in range(self.depth):
            for cell in range(self.width):
                yield row, cell, self.column(row, cell)

    def filled(self, value: float, dtype=np.float64) -> Field:
        return np.full(self.shape, value, dtype=dtype)

    def mark(self, columns: Iterable[ColumnPos]) -> Field:
        marked = self.filled(False, dtype=bool)
        for each in columns:
            each = ColumnPos.coerce(each)
            if self.holds(each.x, each.z):
                marked[self.at(each.x, each.z)] = True
        return marked
