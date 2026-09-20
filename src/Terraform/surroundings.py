from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import kernels
from .grid import Field
from .terrain import ABSENT


ROUGHNESS_RADIUS = 3
SCALE_RADII = (1, 2, 4, 8, 16)
GENTLEST_FILL = 1.5
FLATTEST_FILL = 3.0
GENTLEST_CUT = 0.75
FLATTEST_CUT = 2.0


@dataclass(frozen=True, slots=True)
class Surroundings:
    roughness: float
    feature_scale: float
    steepness: float
    repose: float

    @property
    def fill_ratio(self) -> float:
        return float(np.clip(1.0 / max(self.repose, 1e-3), GENTLEST_FILL, FLATTEST_FILL))

    @property
    def cut_ratio(self) -> float:
        return float(np.clip(self.fill_ratio * 0.6, GENTLEST_CUT, FLATTEST_CUT))


def measure(height: Field, sampled: Field) -> Surroundings:
    known = sampled & (height > ABSENT)
    if not known.any():
        return Surroundings(0.0, 4.0, 0.0, 0.5)
    ground = np.where(known, height, np.nan).astype(np.float64)
    filled = _filled(ground, known)
    residual = filled - kernels.box(filled, ROUGHNESS_RADIUS)
    roughness = float(np.std(residual[known]))
    incline = kernels.slope(filled)[known]
    return Surroundings(
        roughness=roughness,
        feature_scale=_feature_scale(filled, known),
        steepness=float(np.percentile(incline, 50)),
        repose=float(np.percentile(incline, 90)),
    )


def _filled(ground: Field, known: Field) -> Field:
    average = float(np.mean(ground[known]))
    return np.where(known, ground, average)


def _feature_scale(filled: Field, known: Field) -> float:
    variances = [
        float(np.var((filled - kernels.box(filled, radius))[known]))
        for radius in SCALE_RADII
    ]
    ceiling = max(variances)
    if ceiling <= 0:
        return float(SCALE_RADII[0])
    for radius, variance in zip(SCALE_RADII, variances):
        if variance >= ceiling * 0.5:
            return float(radius)
    return float(SCALE_RADII[-1])
