from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

from . import kernels
from .grid import Field, Grid
from .surface_mask import SurfaceMaskSpec


@dataclass(frozen=True, slots=True)
class Ground:
    grid: Grid
    natural: Field
    distance: Field
    seed_height: Field
    crest: Field
    station: Field
    reach: Field | None = None


class Reference(Enum):
    NATURAL = "natural"
    ACCUMULATED = "accumulated"


@runtime_checkable
class Generator(Protocol):
    def __call__(self, ground: Ground, accumulated: Field) -> Field: ...


def _base(ground: Ground, from_crest: bool) -> Field:
    # The nearest track column's highest block, or its lowest.
    return ground.crest if from_crest else ground.seed_height


def _texture(field: Field) -> Field:
    # Local unevenness: distance from a 5x5 mean.
    return np.abs(field - kernels.box(field, 2))


def _pad(
    ground: Ground, accumulated: Field, weight: Field, sigma: float, from_crest: bool
) -> Field:
    # Eases the terrain toward a blur of itself, then to the base, by `weight`.
    smooth = kernels.gaussian(accumulated, sigma)
    background = accumulated + weight * (smooth - accumulated)
    return background + weight * (_base(ground, from_crest) - background)


@dataclass(frozen=True, slots=True)
class Slope:
    ratio: float
    rising: bool = False
    from_crest: bool = False
    height_response: float = 0.0
    terrain_response: float = 0.0
    max_ratio: float = 4.5

    def __call__(self, ground: Ground, accumulated: Field) -> Field:
        ratio = max(self.ratio, 1e-6)
        if not self.rising and (self.height_response or self.terrain_response):
            # Taller fills and rougher ground get gentler slopes.
            height = np.maximum(_base(ground, self.from_crest) - ground.natural, 0.0)
            texture = _texture(ground.natural)
            ratio = np.clip(
                ratio * (1.0 + self.height_response * height
                         + self.terrain_response * texture),
                self.ratio, max(self.ratio, self.max_ratio),
            )
        step = ground.distance / ratio
        return _base(ground, self.from_crest) + (step if self.rising else -step)


@dataclass(frozen=True, slots=True)
class Level:
    from_crest: bool = False

    def __call__(self, ground: Ground, accumulated: Field) -> Field:
        return _base(ground, self.from_crest)


@dataclass(frozen=True, slots=True)
class GaussianPad:
    sigma: float
    from_crest: bool = False

    def __call__(self, ground: Ground, accumulated: Field) -> Field:
        # Gaussian weight: exactly the base at the anchor, fading outward.
        spread = max(self.sigma, 1e-6)
        weight = np.exp(-0.5 * (ground.distance / spread) ** 2)
        return _pad(ground, accumulated, weight, spread, self.from_crest)


@dataclass(frozen=True, slots=True)
class CircularPad:
    radius: float
    blur_sigma: float = 3.0
    from_crest: bool = False

    def __call__(self, ground: Ground, accumulated: Field) -> Field:
        # Smootherstep weight over the radius: level at the centre and edge.
        radius = self.radius if ground.reach is None else ground.reach
        t = np.clip(ground.distance / np.maximum(radius, 1e-6), 0.0, 1.0)
        weight = 1.0 - t**3 * (10.0 - 15.0 * t + 6.0 * t * t)
        return _pad(ground, accumulated, weight, self.blur_sigma, self.from_crest)


@dataclass(frozen=True, slots=True)
class Mound:
    height: float
    width: float

    def __call__(self, ground: Ground, accumulated: Field) -> Field:
        falloff = np.clip(1.0 - ground.distance / max(self.width, 1e-6), 0.0, 1.0)
        return accumulated + self.height * falloff * falloff * (3 - 2 * falloff)


@dataclass(frozen=True, slots=True)
class Offset:
    height: float
    from_crest: bool = False

    def __call__(self, ground: Ground, accumulated: Field) -> Field:
        return _base(ground, self.from_crest) + self.height


@dataclass(frozen=True, slots=True)
class Keep:
    def __call__(self, ground: Ground, accumulated: Field) -> Field:
        return ground.natural


class Blend(Enum):
    RAISE = "raise"
    LOWER = "lower"
    SET = "set"
    RESTORE = "restore"


def combine(
    mode: Blend,
    accumulated: Field,
    candidate: Field,
    weight: Field,
    natural: Field,
    softness: float,
) -> Field:
    if mode is Blend.RAISE:
        merged = kernels.smooth_max(accumulated, candidate, softness)
    elif mode is Blend.LOWER:
        merged = kernels.smooth_min(accumulated, candidate, softness)
    elif mode is Blend.SET:
        merged = candidate
    else:
        merged = natural
    return accumulated + (merged - accumulated) * weight


@runtime_checkable
class Filter(Protocol):
    def __call__(self, relief: Field, ground: Ground) -> Field: ...


KERNELS = ("gaussian", "average", "both")


@dataclass(frozen=True, slots=True)
class Smooth:
    sigma: float
    strength: float = 1.0
    passes: int = 1
    # "average" is the mean over a circle of radius 2 * sigma, the same spread
    # as the gaussian; "both" applies the gaussian and then the average.
    kernel: str = field(default="gaussian", kw_only=True)

    def blur(self, values: Field, size: float, edge: str) -> Field:
        if self.kernel != "average":
            values = kernels.gaussian(values, size, edge)
        if self.kernel != "gaussian":
            values = kernels.disk(values, 2.0 * size, edge)
        return values

    def apply(self, field: Field, ground: Ground, relief: Field, edge: str) -> Field:
        for _ in range(self.passes):
            field = field + self.strength * (self.blur(field, self.sigma, edge) - field)
        return field

    def __call__(self, relief: Field, ground: Ground) -> Field:
        return self.apply(relief, ground, relief, "constant")


@dataclass(frozen=True, slots=True)
class AdaptiveSmooth(Smooth):
    min_size: float = 0.75
    max_size: float = 3.5
    terrain_response: float = 1.0

    def apply(self, field: Field, ground: Ground, relief: Field, edge: str) -> Field:
        texture = _texture(ground.natural)
        activity = np.clip(np.abs(relief) / 6.0 + texture / 2.0, 0.0, 1.0)
        size = np.clip(
            self.sigma * (1.0 + self.terrain_response * activity),
            self.min_size, self.max_size,
        )
        # Blur at scales 1.5x apart across the requested sizes and blend
        # between neighbouring scales per cell.
        smallest = float(np.min(size))
        largest = float(np.max(size))
        scales = [smallest]
        while scales[-1] * 1.5 < largest:
            scales.append(scales[-1] * 1.5)
        if scales[-1] < largest:
            scales.append(largest)
        for _ in range(self.passes):
            blurred = self.blur(field, scales[0], edge)
            for lower, upper in zip(scales, scales[1:]):
                higher = self.blur(field, upper, edge)
                weight = np.clip((size - lower) / (upper - lower), 0.0, 1.0)
                blurred = blurred + weight * (higher - blurred)
            field = field + self.strength * (blurred - field)
        return field


@dataclass(frozen=True, slots=True)
class Terrace:
    step: float

    def __call__(self, relief: Field, ground: Ground) -> Field:
        if self.step <= 0:
            return relief
        return np.round(relief / self.step) * self.step


@dataclass(frozen=True, slots=True)
class Roughen:
    amplitude: float
    scale: float
    seed: int = 0

    def __call__(self, relief: Field, ground: Ground) -> Field:
        if self.amplitude <= 0:
            return relief
        grain = _noise(ground.grid, self.scale, self.seed)
        rough = grain * self.amplitude * np.clip(np.abs(relief), 0.0, 1.0)
        # Grain stays under a block.
        most = np.minimum(np.abs(relief) / 2.0, MAX_GRAIN)
        return relief + np.clip(rough, -most, most)


@dataclass(frozen=True, slots=True)
class Clean:
    radius: int = 1

    def __call__(self, relief: Field, ground: Ground) -> Field:
        return kernels.opening(kernels.closing(relief, self.radius), self.radius)


@dataclass(frozen=True, slots=True)
class Layer:
    generator: Generator
    blend: Blend = Blend.RAISE
    mask: Field | None = None
    softness: float = 0.0
    reach: float = 0.0
    feather: float | None = None
    filters: Sequence[Filter] = ()
    order: int = 0
    name: str = ""
    ground: Ground | None = None
    surface_mask: SurfaceMaskSpec | None = None


def shape(
    ground: Ground, layers: Sequence[Layer], start: Field | None = None
) -> Field:
    # Applies layers in order over `start` (default: natural). Each layer's
    # filters see only that layer's change.
    accumulated = (ground.natural if start is None else start).astype(np.float64)
    for layer in sorted(layers, key=lambda each: each.order):
        anchored = layer.ground or ground
        candidate = layer.generator(anchored, accumulated)
        weight = _weight(anchored, layer)
        crisp = combine(
            layer.blend, accumulated, candidate, weight, ground.natural, 0.0
        )
        merged = combine(
            layer.blend,
            accumulated,
            candidate,
            weight,
            ground.natural,
            _softness(crisp - accumulated, layer.softness),
        )
        change = merged - accumulated
        for each in layer.filters:
            change = each(change, anchored)
        accumulated = accumulated + change
    return accumulated


def settle(
    ground: Ground, designed: Field, filters: Sequence[Filter], hold: Field
) -> Field:
    # Applies the look filters to the relief once, faded out where `hold` is
    # 1 (the track formation). Wherever earth moves, the relief sits on the
    # ground as it was before being rounded to whole blocks, so rounding moves
    # its terrace edges instead of drawing new ones across them.
    if not filters:
        return designed
    relief = designed - ground.natural
    for each in filters:
        relief = each(relief, ground)
    derounded = kernels.disk(ground.natural, 2.0 * DEROUNDING, "edge")
    moved = np.clip(np.abs(relief) / RIM, 0.0, 1.0)
    settled = ground.natural + moved * (derounded - ground.natural) + relief
    held = np.clip(hold, 0.0, 1.0)
    return held * designed + (1.0 - held) * settled


def _softness(displaced: Field, coefficient: float) -> Field:
    if coefficient <= 0:
        return np.zeros_like(displaced)
    return coefficient * kernels.box(np.abs(displaced), 2, "constant")


def _weight(ground: Ground, layer: Layer) -> Field:
    weight = np.ones(ground.grid.shape) if layer.mask is None else layer.mask
    if layer.reach > 0:
        reach = layer.reach if ground.reach is None else ground.reach
        edge = np.maximum(reach, 1e-6) if layer.feather is None else max(layer.feather, 1e-6)
        weight = weight * np.clip((reach - ground.distance) / edge, 0.0, 1.0)
    return np.clip(weight, 0.0, 1.0)


def feathered(mask: Field, width: int) -> Field:
    if width <= 0:
        return mask.astype(np.float64)
    return np.clip(kernels.box(mask.astype(np.float64), width, "constant"), 0.0, 1.0)


# The blur that recovers the ground from before it was rounded to whole
# blocks, and the change over which the works move onto it.
DEROUNDING = 1.0
RIM = 0.5

NOISE_SPAN = 8
NOISE_LIMIT = 2.5
MAX_GRAIN = 0.65


def _noise(grid: Grid, scale: float, seed: int) -> Field:
    rows = np.arange(grid.depth)[:, None] + grid.origin.z
    cells = np.arange(grid.width)[None, :] + grid.origin.x
    mixed = (rows * 374761393 + cells * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    mixed = (mixed ^ (mixed >> 13)) * 1274126177 & 0xFFFFFFFF
    white = (mixed ^ (mixed >> 16)) / 0xFFFFFFFF - 0.5
    sigma = min(max(scale, 0.5), min(grid.width, grid.depth) / NOISE_SPAN)
    smoothed = kernels.gaussian(white, sigma)
    spread = float(smoothed.std())
    if spread < 1e-6:
        return np.zeros_like(smoothed)
    return np.clip(smoothed / spread, -NOISE_LIMIT, NOISE_LIMIT)
