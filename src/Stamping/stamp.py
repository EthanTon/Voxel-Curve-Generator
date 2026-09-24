from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..Core.Domain import DomainRegistry
from ..Core.CrossSection.turns import (
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
                mirror="z" if point.mirrored else None,
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
        self._turns: dict[int, tuple[Feature, int, bool]] = {}

    @property
    def headings(self) -> tuple[Heading, ...]:
        return authored_headings(self.resolution)

    def at(self, heading: Heading) -> tuple[Feature, int, bool]:
        # (variant, quarter turns, mirrored), cached per step so the same
        # feature object is reused.
        step = step_of(heading, self.resolution)
        found = self._turns.get(step)
        if found is None:
            quarter_turns, within = divmod(step, steps_per_quarter(self.resolution))
            offset, mirrored = authored_for(within, self.resolution, self._features)
            found = (self._features[offset], quarter_turns, mirrored)
            self._turns[step] = found
        return found

    def rule(self, point: FeaturePathPoint) -> Feature:
        return self.at(heading_of(point.gradient))[0]

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
        _, quarter_turns, mirrored = self.at(heading_of(point.gradient))
        return (
            place(
                world,
                feature,
                point,
                clip=clip,
                quarter_turns=laid_out(quarter_turns + (DIAGONAL_TURN if mirrored else 0)),
                mirror="x" if mirrored else None,
                # The offset follows the section's layout, not the heading.
                offset_layout=point.layout,
                domains=self.domains,
                y_offset=y_offset,
                sequence_index=sequence_index,
            ),
        )


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
        else laid_offset(feature.offset, offset_layout, point.mirrored)
    )
    transform = BlockTransform(
        origin + BlockPos(0, point.y_offset, 0) if rigid else origin,
        quarter_turns,
        _elevation(feature, y_offset),
        mirror,
    )
    profile = None if domains is None else domains.profile(feature.domain)
    surface_mask = getattr(feature, "surface_mask", None)
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
        surface_mask=(
            None if surface_mask is None else
            world.surface_reference.mask(surface_mask)
        ),
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
    # Stamps every point of `path`; connectivity is resolved later.
    return tuple(
        placed
        for point in path
        for placed in stamp.stamp(
            world,
            stamp.rule(point),
            point,
            clip=clip(point),
            y_offset=y_offset,
            sequence_index=sequence_index,
        )
    )


def turned_offset(offset: BlockPos, quarter_turns: int) -> BlockPos:
    x, z = quarter_turn(offset.x, offset.z, quarter_turns)
    return BlockPos(round(x), offset.y, round(z))


def laid_offset(offset: BlockPos, layout: int, mirrored: bool = False) -> BlockPos:
    # Where the section places its own column at this local offset in the
    # given layout, so offsets stay on the deck's columns (rotating and
    # rounding would shorten them on diagonals).
    x, z = placed_column(layout, offset.x, -offset.z if mirrored else offset.z)
    return BlockPos(x, offset.y, z)


def _elevation(feature: Feature, y_offset: YOffset | None) -> YOffset | None:
    mode = feature.elevation_mode
    if mode == ElevationMode.DEFORMABLE:
        return y_offset
    if mode == ElevationMode.RIGID:
        return None
    raise NotImplementedError(f"Elevation mode {mode} is not stamped yet")
