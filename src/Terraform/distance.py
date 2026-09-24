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
    candidate = np.empty_like(distance)
    shifted_field = np.empty_like(distance)
    closer = np.empty(distance.shape, dtype=bool)
    for _ in range(ceil(reach) + 1):
        settled = True
        for rows, cells, cost in STEPS:
            _shift_into(distance, rows, cells, np.inf, candidate)
            np.add(candidate, cost, out=candidate)
            np.less(candidate, distance, out=closer)
            if not closer.any():
                continue
            np.copyto(distance, candidate, where=closer)
            for each in fields:
                _shift_into(each, rows, cells, 0.0, shifted_field)
                np.copyto(each, shifted_field, where=closer)
            settled = False
        if settled:
            break
    return Nearest(np.minimum(distance, reach), tuple(fields))


def nearest_euclidean(
    seeds: Field, carried: Sequence[Field], reach: float, radii: Field | None = None
) -> Nearest:
    """Stamp exact disks; optional radii choose the strongest overlapping disk."""
    squared = np.full(seeds.shape, np.inf)
    priority = np.full(seeds.shape, np.inf)
    fields = [np.zeros(seeds.shape, dtype=np.float64) for _ in carried]
    radius = ceil(reach)
    for row, cell in zip(*np.nonzero(seeds)):
        z0, z1 = max(0, row - radius), min(seeds.shape[0], row + radius + 1)
        x0, x1 = max(0, cell - radius), min(seeds.shape[1], cell + radius + 1)
        dz = np.arange(z0, z1)[:, None] - row
        dx = np.arange(x0, x1)[None, :] - cell
        candidate = dz * dz + dx * dx
        local_radius = reach if radii is None else radii[row, cell]
        scaled = candidate / max(local_radius * local_radius, 1e-6)
        held = squared[z0:z1, x0:x1]
        ranked = priority[z0:z1, x0:x1]
        closer = (scaled < ranked) & (candidate <= local_radius * local_radius)
        held[closer] = candidate[closer]
        ranked[closer] = scaled[closer]
        for output, source in zip(fields, carried):
            output[z0:z1, x0:x1][closer] = source[row, cell]
    return Nearest(np.sqrt(np.minimum(squared, reach * reach)), tuple(fields))


def _shifted(field: Field, rows: int, cells: int, absent: float) -> Field:
    moved = np.empty_like(field)
    _shift_into(field, rows, cells, absent, moved)
    return moved


def _shift_into(
    field: Field, rows: int, cells: int, absent: float, moved: Field
) -> None:
    moved.fill(absent)
    source = (
        slice(max(-rows, 0), field.shape[0] - max(rows, 0)),
        slice(max(-cells, 0), field.shape[1] - max(cells, 0)),
    )
    target = (
        slice(max(rows, 0), field.shape[0] - max(-rows, 0)),
        slice(max(cells, 0), field.shape[1] - max(-cells, 0)),
    )
    moved[target] = field[source]
