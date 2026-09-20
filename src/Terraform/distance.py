from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

import numpy as np

from .grid import Field


STEPS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, 1.41421356),
    (-1, 1, 1.41421356),
    (1, -1, 1.41421356),
    (1, 1, 1.41421356),
)


@dataclass(frozen=True, slots=True)
class Nearest:
    distance: Field
    carried: tuple[Field, ...]

    def __getitem__(self, index: int) -> Field:
        return self.carried[index]


def nearest(seeds: Field, carried: Sequence[Field], reach: float) -> Nearest:
    distance = np.where(seeds, 0.0, np.inf)
    fields = [np.where(seeds, each, 0).astype(np.float64) for each in carried]
    for _ in range(ceil(reach) + 1):
        settled = True
        for rows, cells, cost in STEPS:
            candidate = _shifted(distance, rows, cells, np.inf) + cost
            closer = candidate < distance
            if not closer.any():
                continue
            distance = np.where(closer, candidate, distance)
            fields = [
                np.where(closer, _shifted(each, rows, cells, 0.0), each)
                for each in fields
            ]
            settled = False
        if settled:
            break
    return Nearest(np.minimum(distance, reach), tuple(fields))


def _shifted(field: Field, rows: int, cells: int, absent: float) -> Field:
    moved = np.full_like(field, absent)
    source = (
        slice(max(-rows, 0), field.shape[0] - max(rows, 0)),
        slice(max(-cells, 0), field.shape[1] - max(cells, 0)),
    )
    target = (
        slice(max(rows, 0), field.shape[0] - max(-rows, 0)),
        slice(max(cells, 0), field.shape[1] - max(-cells, 0)),
    )
    moved[target] = field[source]
    return moved
