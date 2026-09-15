from __future__ import annotations

from dataclasses import dataclass

from .boundary import OrthogonalBoundary


@dataclass(frozen=True, slots=True)
class ElevationStep:
    boundary: OrthogonalBoundary
    y_before: int
    y_after: int

    def __call__(self, x: int, z: int) -> int:
        return self.y_after if self.boundary.after(x, z) else self.y_before

    @property
    def span(self) -> tuple[int, int]:
        if self.y_before <= self.y_after:
            return self.y_before, self.y_after
        return self.y_after, self.y_before

    @property
    def is_constant(self) -> bool:
        return self.y_before == self.y_after


@dataclass(frozen=True, slots=True)
class MultiElevationStep:
    y_first: int
    steps: tuple[tuple[OrthogonalBoundary, int], ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("MultiElevationStep requires at least one boundary")
        object.__setattr__(self, "steps", tuple(self.steps))

    def __call__(self, x: int, z: int) -> int:
        level = self.y_first
        for boundary, y_after in self.steps:
            if boundary.after(x, z):
                level = y_after
        return level

    @property
    def span(self) -> tuple[int, int]:
        levels = [self.y_first, *(y for _, y in self.steps)]
        return min(levels), max(levels)

    @property
    def is_constant(self) -> bool:
        return all(y == self.y_first for _, y in self.steps)
