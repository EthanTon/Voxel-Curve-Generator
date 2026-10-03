"""Choose independent earthworks to the left and right of a route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .grid import Field, Grid
from .plan import CrossSection
from .terrain import ABSENT

if TYPE_CHECKING:
    from .layers import Ground
    from .terraform import Alignment


@dataclass(frozen=True, slots=True)
class StationAxes:
    x: Field
    z: Field
    right_x: Field
    right_z: Field
    height: Field


def axes(alignment: Alignment) -> StationAxes:
    count = alignment.length
    x = np.zeros(count)
    z = np.zeros(count)
    height = np.zeros(count)
    samples = np.zeros(count)
    for (cx, cz), station in alignment.station.items():
        x[station] += cx
        z[station] += cz
        height[station] += alignment.footing[(cx, cz)]
        samples[station] += 1
    valid = np.flatnonzero(samples)
    if not len(valid):
        raise ValueError("a mixed cross section needs alignment stations")
    indices = np.arange(count)
    x = np.interp(indices, valid, x[valid] / samples[valid])
    z = np.interp(indices, valid, z[valid] / samples[valid])
    # Sides are measured from the path itself: an uneven section's middle
    # column is not its centreline.
    for station, (cx, cz) in alignment.centres.items():
        if station < count:
            x[station], z[station] = cx, cz
    height = np.interp(indices, valid, height[valid] / samples[valid])
    dx = np.gradient(x) if count > 1 else np.ones(count)
    dz = np.gradient(z) if count > 1 else np.zeros(count)
    length = np.hypot(dx, dz)
    dx = np.divide(dx, length, out=np.ones_like(dx), where=length > 1e-6)
    dz = np.divide(dz, length, out=np.zeros_like(dz), where=length > 1e-6)
    return StationAxes(x, z, -dz, dx, height)


def side_modes(
    grid: Grid,
    ground: Ground,
    alignment: Alignment,
    section: CrossSection,
    natural: Field,
    water: Field,
    known: Field,
) -> dict[str, Field]:
    """Map each column to the policy on its side of the nearest station."""
    oriented = axes(alignment)
    station = np.clip(np.rint(ground.station).astype(int), 0, alignment.length - 1)
    rows = np.arange(grid.depth)[:, None] + grid.origin.z
    cells = np.arange(grid.width)[None, :] + grid.origin.x
    lateral = ((cells - oriented.x[station]) * oriented.right_x[station]
               + (rows - oriented.z[station]) * oriented.right_z[station])
    left = _stations(section.left, -1, oriented, section, grid, natural, water, known)
    right = _stations(section.right, 1, oriented, section, grid, natural, water, known)
    chosen = np.where(lateral < 0, left[station], right[station])
    return {
        mode: chosen == mode
        for mode in ("at_grade", "cutting", "embankment", "bridge")
    }


def _stations(
    setting: str,
    side: int,
    oriented: StationAxes,
    section: CrossSection,
    grid: Grid,
    natural: Field,
    water: Field,
    known: Field,
) -> np.ndarray:
    if setting != "auto":
        return np.full(len(oriented.x), setting, dtype="<U12")
    modes = []
    for station in range(len(oriented.x)):
        levels: list[float] = []
        wet = 0
        for nearby in range(max(0, station - section.sample_window),
                            min(len(oriented.x), station + section.sample_window + 1)):
            for distance in range(max(1, section.sample_distance - 1), section.sample_distance + 2):
                x = round(oriented.x[nearby] + side * distance * oriented.right_x[nearby])
                z = round(oriented.z[nearby] + side * distance * oriented.right_z[nearby])
                if not grid.holds(x, z):
                    continue
                row, cell = grid.at(x, z)
                if not known[row, cell]:
                    continue
                levels.append(float(natural[row, cell]))
                wet += bool(water[row, cell] > natural[row, cell] and water[row, cell] > ABSENT)
        if not levels:
            modes.append("at_grade")
        elif wet >= max(1, len(levels) // 3) and section.water_mode == "bridge":
            modes.append("bridge")
        else:
            difference = float(np.median(levels)) - oriented.height[station]
            if difference >= section.cut_threshold:
                modes.append("cutting")
            elif difference <= -section.bridge_threshold:
                modes.append("bridge")
            elif difference <= -section.fill_threshold:
                modes.append("embankment")
            else:
                modes.append("at_grade")
    return np.asarray(modes, dtype="<U12")
