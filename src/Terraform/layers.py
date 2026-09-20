from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

from . import kernels
from .grid import Field, Grid


@dataclass(frozen=True, slots=True)
class Ground:
    grid: Grid
    natural: Field
    distance: Field
    seed_height: Field
    crest: Field
    station: Field


class Reference(Enum):
    NATURAL = "natural"
    ACCUMULATED = "accumulated"


@runtime_checkable
class Generator(Protocol):
    def __call__(self, ground: Ground, accumulated: Field) -> Field: ...


@dataclass(frozen=True, slots=True)
class Slope:
    ratio: float
    rising: bool = False

    def __call__(self, ground: Ground, accumulated: Field) -> Field:
        step = ground.distance / max(self.ratio, 1e-6)
        return ground.seed_height + (step if self.rising else -step)


@dataclass(frozen=True, slots=True)
class Level:
    def __call__(self, ground: Ground, accumulated: Field) -> Field:
        return ground.seed_height


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

    def __call__(self, ground: Ground, accumulated: Field) -> Field:
        return ground.seed_height + self.height


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


@dataclass(frozen=True, slots=True)
class Smooth:
    sigma: float

    def __call__(self, relief: Field, ground: Ground) -> Field:
        return kernels.gaussian(relief, self.sigma, edge="constant")


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
        return relief + grain * self.amplitude * np.clip(np.abs(relief), 0.0, 1.0)


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


def shape(ground: Ground, layers: Sequence[Layer]) -> Field:
    accumulated = ground.natural.astype(np.float64)
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
        relief = merged - ground.natural
        for each in layer.filters:
            relief = each(relief, anchored)
        accumulated = ground.natural + relief
    return accumulated


def _softness(displaced: Field, coefficient: float) -> Field:
    if coefficient <= 0:
        return np.zeros_like(displaced)
    return coefficient * kernels.box(np.abs(displaced), 2, "constant")


def _weight(ground: Ground, layer: Layer) -> Field:
    weight = np.ones(ground.grid.shape) if layer.mask is None else layer.mask
    if layer.reach > 0:
        edge = layer.reach if layer.feather is None else max(layer.feather, 1e-6)
        weight = weight * np.clip((layer.reach - ground.distance) / edge, 0.0, 1.0)
    return np.clip(weight, 0.0, 1.0)


def feathered(mask: Field, width: int) -> Field:
    if width <= 0:
        return mask.astype(np.float64)
    return np.clip(kernels.box(mask.astype(np.float64), width, "constant"), 0.0, 1.0)


NOISE_SPAN = 8
NOISE_LIMIT = 2.5


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
