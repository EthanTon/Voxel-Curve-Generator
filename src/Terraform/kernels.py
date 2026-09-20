from __future__ import annotations

from math import floor, sqrt

import numpy as np

from .grid import Field


GAUSSIAN_PASSES = 3


def box(field: Field, radius: int, edge: str = "edge") -> Field:
    for axis in range(field.ndim):
        field = _running_mean(field, radius, axis, edge)
    return field


def gaussian(field: Field, sigma: float, edge: str = "edge") -> Field:
    for radius in _box_radii(sigma, GAUSSIAN_PASSES):
        field = box(field, radius, edge)
    return field


def weighted(field: Field, weight: Field, radius: int) -> Field:
    carried = box(field * weight, radius, "constant")
    share = box(weight, radius, "constant")
    return np.divide(carried, share, out=np.zeros_like(carried), where=share > 0)


def smooth_max(first: Field, second: Field, softness) -> Field:
    softness = np.maximum(np.asarray(softness, dtype=np.float64), 0.0)
    safe = np.where(softness > 0, softness, 1.0)
    blend = np.clip(0.5 + 0.5 * (first - second) / safe, 0.0, 1.0)
    eased = second * (1 - blend) + first * blend + safe * blend * (1 - blend)
    return np.where(softness > 0, eased, np.maximum(first, second))


def smooth_min(first: Field, second: Field, softness) -> Field:
    return -smooth_max(-first, -second, softness)


def local_maximum(field: Field, radius: int) -> Field:
    return _sweep(field, radius, np.maximum)


def local_minimum(field: Field, radius: int) -> Field:
    return _sweep(field, radius, np.minimum)


def opening(field: Field, radius: int) -> Field:
    return local_maximum(local_minimum(field, radius), radius)


def closing(field: Field, radius: int) -> Field:
    return local_minimum(local_maximum(field, radius), radius)


def slope(field: Field) -> Field:
    rise_z, rise_x = np.gradient(field)
    return np.hypot(rise_x, rise_z)


def _running_mean(field: Field, radius: int, axis: int, edge: str) -> Field:
    if radius < 1:
        return field
    window = 2 * radius + 1
    padding = [(0, 0)] * field.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(field.astype(np.float64), padding, mode=edge)
    zero = np.zeros_like(padded[_span(field.ndim, axis, 0, 1)])
    running = np.concatenate([zero, np.cumsum(padded, axis=axis)], axis=axis)
    length = field.shape[axis]
    ahead = running[_span(field.ndim, axis, window, window + length)]
    behind = running[_span(field.ndim, axis, 0, length)]
    return (ahead - behind) / window


def _sweep(field: Field, radius: int, combine) -> Field:
    for axis in range(field.ndim):
        gathered = field
        for shift in range(-radius, radius + 1):
            if shift:
                gathered = combine(gathered, _shifted(field, shift, axis))
        field = gathered
    return field


def _shifted(field: Field, shift: int, axis: int) -> Field:
    padding = [(0, 0)] * field.ndim
    padding[axis] = (max(shift, 0), max(-shift, 0))
    padded = np.pad(field, padding, mode="edge")
    start = max(-shift, 0)
    return padded[_span(field.ndim, axis, start, start + field.shape[axis])]


def _span(rank: int, axis: int, start: int, stop: int) -> tuple[slice, ...]:
    cuts = [slice(None)] * rank
    cuts[axis] = slice(start, stop)
    return tuple(cuts)


def _box_radii(sigma: float, passes: int) -> list[int]:
    if sigma <= 0:
        return []
    ideal = sqrt(12 * sigma * sigma / passes + 1)
    lower = int(floor(ideal))
    if lower % 2 == 0:
        lower -= 1
    lower = max(lower, 1)
    crossover = round(
        (12 * sigma * sigma - passes * lower * lower - 4 * passes * lower - 3 * passes)
        / (-4 * lower - 4)
    )
    sizes = [lower if index < crossover else lower + 2 for index in range(passes)]
    return [(size - 1) // 2 for size in sizes]
