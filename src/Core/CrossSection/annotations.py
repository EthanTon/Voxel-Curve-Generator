from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..coordinates import BlockPos
from ..modes import StepMode

if TYPE_CHECKING:
    from ..Feature.feature import Feature


@dataclass(frozen=True, slots=True)
class FeatureAnnotation:
    # Where a feature sits in a cross section.

    name: str
    position: BlockPos
    feature: Feature
    step_mode: StepMode = StepMode.MANHATTAN
    # The owning section when a profile is split into one annotation per column.
    section: str | None = None
    # Stop at the span boundary instead of filling small gaps at a swap.
    clip_to_span: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", BlockPos.coerce(self.position))
