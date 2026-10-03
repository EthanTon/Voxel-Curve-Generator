from __future__ import annotations

import numpy as np

from .grid import Field, Grid

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def _lattice(ix: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    # A value in [-1, 1] for every integer lattice point, the same wherever
    # it is asked for.
    with np.errstate(over="ignore"):
        h = (ix.astype(np.int64).view(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
             ^ iz.astype(np.int64).view(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
             ^ np.uint64(seed & 0xFFFFFFFF) * np.uint64(0x165667B19E3779F9)) & _MASK
        h ^= h >> np.uint64(31)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(29)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53) * 2.0 - 1.0


def value_noise(x: np.ndarray, z: np.ndarray, seed: int) -> np.ndarray:
    # Smoothly interpolated lattice values, in [-1, 1].
    x0, z0 = np.floor(x), np.floor(z)
    fx, fz = x - x0, z - z0
    ux, uz = fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz)
    ix, iz = x0.astype(np.int64), z0.astype(np.int64)
    top = _lattice(ix, iz, seed) * (1 - ux) + _lattice(ix + 1, iz, seed) * ux
    bottom = _lattice(ix, iz + 1, seed) * (1 - ux) + _lattice(ix + 1, iz + 1, seed) * ux
    return top * (1 - uz) + bottom * uz


def fbm(grid: Grid, seed: int, wavelength: float, octaves: int = 3, warp: float = 0.0) -> Field:
    # Octaves of value noise over the grid's world columns, in about [-1, 1].
    # A warp shifts where each column samples it, so its contours wander.
    rows, cells = np.indices((grid.depth, grid.width), dtype=np.float64)
    x = cells + grid.origin.x
    z = rows + grid.origin.z
    if warp:
        x, z = (x + warp * value_noise(x / wavelength, z / wavelength, seed + 7919),
                z + warp * value_noise(x / wavelength, z / wavelength, seed + 104729))
    total = np.zeros_like(x)
    amplitude, weight, scale = 1.0, 0.0, 1.0 / wavelength
    for octave in range(octaves):
        total += amplitude * value_noise(x * scale, z * scale, seed + octave)
        weight += amplitude
        amplitude *= 0.5
        scale *= 2.0
    return total / weight
