from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from ..Core.Domain import DomainRegistry
from ..Core.CrossSection.turns import (
    is_quarter,
    normalize,
    place_along as placed_column,
    quarter_of,
)
from ..Core.Feature.feature import Feature
from ..Core.Feature.feature_path import FeaturePath, FeaturePathPoint
from ..Core.Path.path_frame import PathFrame
from ..Core.World.block_world import (
    BlockTransform,
    BlockWorld,
    ColumnMask,
    PlacementId,
    YOffset,
)
from ..Core.constants import DEFAULT_BLOCK, DEFAULT_TERRAIN_MASK_MODE
from ..Core.coordinates import BlockPos, quarter_turn
from ..Core.modes import ElevationMode
from .variants import (
    DEFAULT_RESOLUTION,
    authored_for,
    Heading,
    authored_headings,
    by_authored,
    heading_of,
    step_of,
    steps_per_quarter,
)


@dataclass(frozen=True, slots=True)
class Everywhere:
    def contains(self, x: float, z: float) -> bool:
        return True


UNCLIPPED = Everywhere()

# The columns a point may be stamped into.
ClipRule = Callable[[FeaturePathPoint], ColumnMask]


def unclipped(point: FeaturePathPoint) -> ColumnMask:
    return UNCLIPPED


def in_frames(frames: Sequence[PathFrame]) -> ClipRule:
    # Clips each point to the frame containing its column.
    owned: dict[tuple[int, int], PathFrame] = {}

    def clip(point: FeaturePathPoint) -> ColumnMask:
        column = (point.position.x, point.position.z)
        if column in owned:
            return owned[column]
        for frame in frames:
            if frame.contains(point.position.x, point.position.z):
                owned[column] = frame
                return frame
        raise ValueError(
            f"{tuple(point.position)} is in no frame, so nothing says which "
            "columns it may be stamped into"
        )

    return clip


@runtime_checkable
class Stamp(Protocol):
    # `rule` picks the feature for a point and `stamp` places it.

    def rule(self, point: FeaturePathPoint) -> Feature: ...

    def stamp(
        self,
        world: BlockWorld,
        feature: Feature,
        point: FeaturePathPoint,
        *,
        clip: ColumnMask,
        y_offset: YOffset | None = None,
        sequence_index: int | None = None,
    ) -> tuple[PlacementId, ...]: ...


@dataclass(frozen=True, slots=True)
class FixedStamp:
    # One unturned feature at every point.

    feature: Feature
    domains: DomainRegistry | None = None

    def rule(self, point: FeaturePathPoint) -> Feature:
        return self.feature

    def stamp(
        self,
        world: BlockWorld,
        feature: Feature,
        point: FeaturePathPoint,
        *,
        clip: ColumnMask,
        y_offset: YOffset | None = None,
        sequence_index: int | None = None,
    ) -> tuple[PlacementId, ...]:
        return (
            place(
                world,
                feature,
                point,
                clip=clip,
                domains=self.domains,
                y_offset=y_offset,
                sequence_index=sequence_index,
            ),
        )


# The quarter turn that, with a mirror across x, reflects across the diagonal.
DIAGONAL_TURN = 1


def laid_out(quarter_turns: int) -> int:
    # Schematics are authored with local x across travel, a quarter ahead.
    return (quarter_turns - 1) % 4


def lateral_turns(turn: int) -> int:
    # Quarter turns for a Turn: west 3, north 0, east 1, south 2.
    return laid_out(normalize(turn) // 90)


@dataclass(frozen=True, slots=True)
class TurnStamp:
    # One feature turned by the layout that placed each point (each arm of a
    # diagonal join its own quarter), as the section's blocks are.

    feature: Feature
    domains: DomainRegistry | None = None

    def rule(self, point: FeaturePathPoint) -> Feature:
        return self.feature

    def layout(self, point: FeaturePathPoint) -> int:
        return point.laid_at

    def stamp(
        self,
        world: BlockWorld,
        feature: Feature,
        point: FeaturePathPoint,
        *,
        clip: ColumnMask,
        y_offset: YOffset | None = None,
        sequence_index: int | None = None,
    ) -> tuple[PlacementId, ...]:
        return (
            place(
                world,
                feature,
                point,
                clip=clip,
                quarter_turns=lateral_turns(quarter_of(self.layout(point))),
                offset_layout=self.layout(point),
                domains=self.domains,
                y_offset=y_offset,
                sequence_index=sequence_index,
            ),
        )


class HeadingStamp:
    # A variant per heading of travel in plan, turned and mirrored from the
    # authored first octant.

    def __init__(
        self,
        features: Mapping[Heading, Feature],
        resolution: float = DEFAULT_RESOLUTION,
        domains: DomainRegistry | None = None,
    ) -> None:
        self.resolution = resolution
        self.domains = domains
        self._features = by_authored(features, resolution)
        self._turns: dict[tuple[int, bool], tuple[Feature, int, bool]] = {}
        self.mirror = _mirror_axis(self._features[0])

    @property
    def headings(self) -> tuple[Heading, ...]:
        return authored_headings(self.resolution)

    def at(self, heading: Heading, layout: int | None = None) -> tuple[Feature, int, bool]:
        # (variant, quarter turns, mirrored), cached per step so the same
        # feature object is reused. A diagonal laid in the next quarter leans
        # the other way, mirrored across it to follow the section's rows.
        step = step_of(heading, self.resolution)
        per_quarter = steps_per_quarter(self.resolution)
        quarter_turns, within = divmod(step, per_quarter)
        past = (
            2 * within == per_quarter
            and layout is not None
            and is_quarter(layout)
            and normalize(layout) // 90 == (quarter_turns + 1) % 4
        )
        found = self._turns.get((step, past))
        if found is None:
            if past:
                offset, mirrored = within, True
            else:
                offset, mirrored = authored_for(within, self.resolution, self._features)
            found = (self._features[offset], quarter_turns, mirrored)
            self._turns[(step, past)] = found
        return found

    def rule(self, point: FeaturePathPoint) -> Feature:
        return self.at(heading_of(point.gradient), point.laid_at)[0]

    def stamp(
        self,
        world: BlockWorld,
        feature: Feature,
        point: FeaturePathPoint,
        *,
        clip: ColumnMask,
        y_offset: YOffset | None = None,
        sequence_index: int | None = None,
    ) -> tuple[PlacementId, ...]:
        _, quarter_turns, mirrored = self.at(heading_of(point.gradient), point.laid_at)
        return (
            place(
                world,
                feature,
                point,
                clip=clip,
                quarter_turns=laid_out(quarter_turns + (DIAGONAL_TURN if mirrored else 0)),
                mirror=self.mirror if mirrored else None,
                # The offset follows the section's layout, not the heading.
                offset_layout=point.layout,
                domains=self.domains,
                y_offset=y_offset,
                sequence_index=sequence_index,
            ),
        )


def _mirror_axis(feature: Feature) -> str:
    # A variant past the diagonal is a reflection of one before it, which
    # swaps either the sides of travel ("z", along it) or the front and back
    # ("x", across it). Keep the structure's sides unless only its front and
    # back differ.
    blocks = getattr(feature, "blocks", None)
    if blocks is None:
        return "z"
    cells = {
        (local.x, local.y, local.z): blocks.palette.get_state(block_id).name
        for local, block_id in blocks.iter_block_ids()
    }
    across = {(-x, y, z): name for (x, y, z), name in cells.items()} == cells
    along = {(x, y, -z): name for (x, y, z), name in cells.items()} == cells
    return "x" if across and not along else "z"


def place(
    world: BlockWorld,
    feature: Feature,
    point: FeaturePathPoint,
    *,
    clip: ColumnMask,
    quarter_turns: int = 0,
    offset_layout: int | None = None,
    mirror: str | None = None,
    domains: DomainRegistry | None = None,
    y_offset: YOffset | None = None,
    sequence_index: int | None = None,
) -> PlacementId:
    # Places one feature at one point. RIGID features take the point's level
    # in their origin; DEFORMABLE ones follow each column's elevation. The
    # offset turns with the blocks.
    rigid = feature.elevation_mode == ElevationMode.RIGID
    origin = BlockPos.coerce(point.position) + (
        turned_offset(feature.offset, quarter_turns)
        if offset_layout is None
        else laid_offset(feature.offset, offset_layout)
    )
    transform = BlockTransform(
        origin + BlockPos(0, point.y_offset, 0) if rigid else origin,
        quarter_turns,
        _elevation(feature, y_offset),
        mirror,
    )
    profile = None if domains is None else domains.profile(feature.domain)
    surface_mask = getattr(feature, "surface_mask", None)
    predicate = None if surface_mask is None else world.surface_reference.mask(surface_mask)
    minimum_y = getattr(feature, "surface_mask_min_y", None)
    if predicate is not None and minimum_y is not None:
        predicate = replace(predicate, minimum_y=transform.origin.y + minimum_y)
    return world.add_field(
        feature.blocks,
        placement_mode=feature.placement_mode,
        terrain_mask_mode=(
            DEFAULT_TERRAIN_MASK_MODE if profile is None else profile.terrain_mask_mode
        ),
        replace=DEFAULT_BLOCK if profile is None else profile.replace,
        block_replacement=None if profile is None else profile.replacement,
        transform=transform,
        connectivity_domain=0 if profile is None else profile.domain_id,
        sequence_index=sequence_index,
        clip=clip,
        surface_mask=predicate,
        yielding=getattr(feature, "yielding", frozenset()),
    )


def stamp_along(
    stamp: Stamp,
    world: BlockWorld,
    path: FeaturePath,
    *,
    clip: ClipRule,
    y_offset: YOffset | None = None,
    sequence_index: int | None = None,
) -> tuple[PlacementId, ...]:
    # Stamps every point of `path`; connectivity is resolved later. Each
    # placement records the path point that laid it, unless told otherwise.
    return tuple(
        placed
        for point in path
        for placed in stamp.stamp(
            world,
            stamp.rule(point),
            point,
            clip=clip(point),
            y_offset=y_offset,
            sequence_index=point.path_index if sequence_index is None else sequence_index,
        )
    )


def turned_offset(offset: BlockPos, quarter_turns: int) -> BlockPos:
    x, z = quarter_turn(offset.x, offset.z, quarter_turns)
    return BlockPos(round(x), offset.y, round(z))


def laid_offset(offset: BlockPos, layout: int) -> BlockPos:
    # Where the section places its own column at this local offset in the
    # given layout, so offsets stay on the deck's columns (rotating and
    # rounding would shorten them on diagonals). The lean never turns an
    # offset ahead into one behind.
    x, z = placed_column(layout, offset.x, offset.z)
    return BlockPos(x, offset.y, z)


def _elevation(feature: Feature, y_offset: YOffset | None) -> YOffset | None:
    mode = feature.elevation_mode
    if mode == ElevationMode.DEFORMABLE:
        return y_offset
    if mode == ElevationMode.RIGID:
        return None
    raise NotImplementedError(f"Elevation mode {mode} is not stamped yet")
