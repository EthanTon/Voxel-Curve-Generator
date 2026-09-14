from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..coordinates import BlockPos
from ..modes import StepMode

if TYPE_CHECKING:
    from ..Feature.feature import Feature


@dataclass(frozen=True, slots=True)
class FeatureAnnotation:
    # Marks where a feature sits in a cross section. The section carries no
    # blocks of its own: delivering it translates `position` to a path point,
    # and the feature builds its blocks from the positions on its path later.
    #
    # `step_mode` picks the step rule that decides which positions the
    # annotation delivers and how far it moves the points beyond it. See
    # steps.py.

    name: str
    position: BlockPos
    feature: Feature
    step_mode: StepMode = StepMode.MANHATTAN

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", BlockPos.coerce(self.position))
