from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.ndimage import distance_transform_edt

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
    # Each column's offset to its nearest seed, handed on from neighbour to
    # neighbour and compared by its true length, so distances are Euclidean
    # and contours round rather than octagonal. `carried` fields come along
    # from the same seed.
    if seeds.size >= 262144:
        if not seeds.any():
            return Nearest(np.full(seeds.shape, reach),
                           tuple(np.zeros(seeds.shape) for _ in carried))
        if carried:
            distance, indices = distance_transform_edt(~seeds, return_indices=True)
            fields = tuple(np.where(distance <= reach + 1.5, each[tuple(indices)], 0)
                           .astype(np.float64) for each in carried)
        else:
            distance = distance_transform_edt(~seeds)
            fields = ()
        return Nearest(np.minimum(distance, reach), fields)
    far = 1e9
    along_rows = np.where(seeds, 0.0, far)
    along_cells = np.where(seeds, 0.0, far)
    squared = np.where(seeds, 0.0, np.inf)
    fields = [np.where(seeds, each, 0).astype(np.float64) for each in carried]
    limit = (reach + 1.5) ** 2
    rows_in = np.empty_like(squared)
    cells_in = np.empty_like(squared)
    shifted_field = np.empty_like(squared)
    closer = np.empty(squared.shape, dtype=bool)
    for _ in range(2 * ceil(reach) + 2):
        settled = True
        for rows, cells, _ in STEPS:
            _shift_into(along_rows, rows, cells, far, rows_in)
            _shift_into(along_cells, rows, cells, far, cells_in)
            rows_in -= rows
            cells_in -= cells
            candidate = rows_in * rows_in + cells_in * cells_in
            np.less(candidate, squared, out=closer)
            closer &= candidate <= limit
            if not closer.any():
                continue
            np.copyto(squared, candidate, where=closer)
            np.copyto(along_rows, rows_in, where=closer)
            np.copyto(along_cells, cells_in, where=closer)
            for each in fields:
                _shift_into(each, rows, cells, 0.0, shifted_field)
                np.copyto(each, shifted_field, where=closer)
            settled = False
        if settled:
            break
    return Nearest(np.minimum(np.sqrt(squared), reach), tuple(fields))


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
