from __future__ import annotations

from collections.abc import Sequence

from ..Core.CrossSection.cross_section import CrossSection
from ..Core.Feature.feature_path import FeaturePath
from ..Core.coordinates import BlockPos
from ..Core.modes import StepMode
from ..Generation.Sweep.paths import (
    DeliveredPath,
    Level,
    along_travel,
    at_their_own_level,
    dashed,
    eight_connected,
    face_connected,
    one_line,
)
from ..Parser.spec import LaneSpec, Measure
from .anchors import as_laid, column_runs


def mark_lane(
    spec: LaneSpec,
    run: DeliveredPath,
    section: CrossSection,
    level: Level,
    support: frozenset[BlockPos] | None = None,
) -> FeaturePath:
    # One position per path point, dashed, moved onto the section column the
    # lane rides, then joined up: face-connected for stepping lanes and
    # corner-connected for continuous ones.
    path = one_line(run.path)
    marked = dashed(path, spec.on, spec.off, spec.measure is Measure.INDEX)
    offset = as_laid(spec.offset)
    laid = at_their_own_level(
        along_travel(marked, section.along(spec.offset.x, spec.step)), level, offset
    )
    if spec.step is StepMode.MANHATTAN:
        return face_connected(
            laid, offset, continuous=not spec.off, level=level, support=support
        )
    if not spec.off:
        return eight_connected(laid, offset, continuous=True, level=level)
    return laid


def lane_support(
    lane: LaneSpec, centre: DeliveredPath, runs: Sequence[DeliveredPath]
) -> frozenset[BlockPos]:
    # The cells the lane's section column actually occupies, which a
    # connected lane bridges along instead of cutting across the deck.
    return frozenset(
        BlockPos(point.position.x, point.position.y + point.y_offset + lane.offset.y,
                 point.position.z)
        for run in column_runs(centre, runs, lane.offset.x, lane.offset.z)
        if any(
            local.y == lane.offset.y - 1
            and run.annotation.feature.blocks.palette.get_state(block).name != "minecraft:air"
            for local, block in run.annotation.feature.blocks.iter_block_ids()
        )
        for point in run.path
    )
