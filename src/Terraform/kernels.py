from __future__ import annotations

from math import ceil, floor, sqrt

import numpy as np

from .grid import Field


GAUSSIAN_PASSES = 3


def box(field: Field, radius: int, edge: str = "edge") -> Field:
    for axis in range(field.ndim):
        field = _running_mean(field, radius, axis, edge)
    return field


def gaussian(field: Field, sigma: float, edge: str = "edge") -> Field:
    # A separable Gaussian responds continuously even at sub-block sizes.
    # Integer box approximations made 0.75..1.25 brushes jump between shapes.
    from scipy.ndimage import gaussian_filter
    if sigma <= 0:
        return field
    return gaussian_filter(field.astype(np.float64), sigma,
                           mode="nearest" if edge == "edge" else edge, truncate=4.0)


def wendland(field: Field, radius: float, edge: str = "edge") -> Field:
    """Radial C2 kernel with finite support, for compact contour finishing."""
    from scipy.ndimage import convolve
    if radius <= 0:
        return field
    reach = ceil(radius)
    axis = np.arange(-reach, reach + 1)
    t = np.minimum(np.hypot(axis[:, None], axis[None, :]) / radius, 1.0)
    weight = (1 - t) ** 4 * (1 + 4 * t)
    weight /= weight.sum()
    return convolve(field.astype(np.float64), weight,
                    mode="nearest" if edge == "edge" else edge)


def normalized(values: Field, known: Field, radius: float, kernel: str = "gaussian") -> Field:
    """Missing samples are absent, rather than zero-height terrain."""
    def spread(field):
        if kernel == "wendland":
            return wendland(field, 3 * radius, "constant")
        if kernel == "average":
            return disk(field, 2 * radius, "constant")
        if kernel == "both":
            return disk(gaussian(field, radius, "constant"), 2 * radius, "constant")
        return gaussian(field, radius, "constant")
    share = spread(known.astype(float))
    total = spread(np.where(known, values, 0.0))
    return np.divide(total, share, out=values.astype(float).copy(), where=share > 1e-9)


def contour(values: Field, guide: Field, domain: Field, iterations: int,
            edge_scale: float = 3.0) -> Field:
    """Diffuse displacement along native contours, through face neighbours.

    Unlike a spatial blur this cannot jump a masked strip. All updates use
    the same snapshot; native cliffs weaken diffusion across their edge.
    """
    result = values.astype(float).copy()
    edges = []
    for axis in range(values.ndim):
        first, second = [slice(None)] * values.ndim, [slice(None)] * values.ndim
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        valid = domain[first] & domain[second]
        delta = np.where(valid, guide[first] - guide[second], 0.0)
        affinity = valid * np.exp(-.5 * (delta / max(edge_scale, 1e-6)) ** 2)
        edges.append((first, second, affinity))
    for _ in range(iterations):
        total, share = result.copy(), np.ones_like(result)
        for first, second, affinity in edges:
            total[first] += affinity * result[second]
            total[second] += affinity * result[first]
            share[first] += affinity
            share[second] += affinity
        result = np.where(domain, total / share, values)
    return result


def disk(field: Field, radius: float, edge: str = "edge") -> Field:
    # The mean over a circle; rim cells count by how much of them it covers.
    if radius <= 0.5:
        return field
    reach = int(ceil(radius + 0.5))
    offsets = np.arange(-reach, reach + 1)
    weight = np.clip(radius + 0.5 - np.hypot(offsets[:, None], offsets[None, :]), 0.0, 1.0)
    weight /= weight.sum()
    padded = np.pad(field.astype(np.float64), reach, mode=edge)
    size = (padded.shape[0] + 2 * reach, padded.shape[1] + 2 * reach)
    spread = np.fft.irfft2(
        np.fft.rfft2(padded, size) * np.fft.rfft2(weight, size), size
    )
    rows, cells = field.shape
    return spread[2 * reach : 2 * reach + rows, 2 * reach : 2 * reach + cells]


def weighted(field: Field, weight: Field, radius: int) -> Field:
    carried = box(field * weight, radius, "constant")
    share = box(weight, radius, "constant")
    return np.divide(carried, share, out=np.zeros_like(carried), where=share > 0)


def smooth_max(first: Field, second: Field, softness) -> Field:
    # Polynomial smooth maximum, blending within `softness` of the crossover.
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


def square_opening(field: Field, size: int) -> Field:
    # Every ridge narrower than a `size`-wide square comes down to the
    # ground either side of it.
    return _squared(field, size, np.minimum, np.maximum)


def square_closing(field: Field, size: int) -> Field:
    # Every groove narrower than a `size`-wide square fills to the ground
    # either side of it.
    return _squared(field, size, np.maximum, np.minimum)


def _squared(field: Field, size: int, first, then) -> Field:
    # `first` over each square, then `then` over the squares holding each
    # cell; padded so a square off the edge sees the edge.
    reach = size - 1
    if reach < 1:
        return field
    padded = np.pad(field, reach, mode="edge")
    out = _from_corner(_from_corner(padded, size, first, 1), size, then, -1)
    return out[(slice(reach, -reach),) * field.ndim]


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


def _from_corner(field: Field, size: int, combine, direction: int) -> Field:
    # Each cell with the `size - 1` after it (direction 1) or before it (-1)
    # along every axis: a square anchored at a corner, so any size works.
    for axis in range(field.ndim):
        gathered = field
        for step in range(1, size):
            gathered = combine(gathered, _shifted(field, -direction * step, axis))
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
    # Box sizes whose `passes` repeated blurs match a Gaussian's variance.
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
