from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..Core.Domain import DomainRegistry
from ..Core.Feature.feature import Feature
from ..Core.Feature.feature_path import FeaturePath, FeaturePathPoint
from ..Core.World.block_world import (
    BlockTransform,
    BlockWorld,
    ColumnMask,
    PlacementId,
    YOffset,
)
from ..Core.constants import DEFAULT_BLOCK, DEFAULT_TERRAIN_MASK_MODE
from ..Core.coordinates import BlockPos
from ..Core.modes import ElevationMode
from .variants import (
    DEFAULT_RESOLUTION,
    Heading,
    authored_headings,
    by_authored,
    heading_of,
    step_of,
    steps_per_quarter,
)


@dataclass(frozen=True, slots=True)
class Everywhere:
    # A mask that clips nothing. Clips are required, so this is how a caller
    # says it meant to stamp everywhere.

    def contains(self, x: float, z: float) -> bool:
        return True


UNCLIPPED = Everywhere()


@runtime_checkable
class Stamp(Protocol):
    # `rule` picks the feature for a point. `stamp` places it.
    # Implementations differ in what `rule` reads off the point.

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
    ) -> PlacementId: ...


@dataclass(frozen=True, slots=True)
class FixedStamp:
    # One feature at every point, unturned. For anything that reads the same
    # from every direction.

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
    ) -> PlacementId:
        return place(
            world,
            feature,
            point,
            clip=clip,
            domains=self.domains,
            y_offset=y_offset,
            sequence_index=sequence_index,
        )


class HeadingStamp:
    # A feature per heading, picked by the way the path travels in plan view.
    #
    # Only the first quadrant is authored; quarter turns cover the rest, so
    # 22.5 degree spacing needs four features, not sixteen. Grade is ignored:
    # a climb and a level run on the same bearing get the same feature.

    def __init__(
        self,
        features: Mapping[Heading, Feature],
        resolution: float = DEFAULT_RESOLUTION,
        domains: DomainRegistry | None = None,
    ) -> None:
        self.resolution = resolution
        self.domains = domains
        self._features = by_authored(features, resolution)
        self._turns: dict[int, tuple[Feature, int]] = {}

    @property
    def headings(self) -> tuple[Heading, ...]:
        return authored_headings(self.resolution)

    def at(self, heading: Heading) -> tuple[Feature, int]:
        # The feature to stamp and the quarter turns to stamp it with.
        # Cached so the same feature object comes back every time, keeping
        # BlockWorld's rotated-state cache warm.
        step = step_of(heading, self.resolution)
        found = self._turns.get(step)
        if found is None:
            quarter_turns, offset = divmod(step, steps_per_quarter(self.resolution))
            found = (self._features[offset], quarter_turns)
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
    ) -> PlacementId:
        _, quarter_turns = self.at(heading_of(point.gradient))
        return place(
            world,
            feature,
            point,
            clip=clip,
            quarter_turns=quarter_turns,
            domains=self.domains,
            y_offset=y_offset,
            sequence_index=sequence_index,
        )


def place(
    world: BlockWorld,
    feature: Feature,
    point: FeaturePathPoint,
    *,
    clip: ColumnMask,
    quarter_turns: int = 0,
    domains: DomainRegistry | None = None,
    y_offset: YOffset | None = None,
    sequence_index: int | None = None,
) -> PlacementId:
    # One feature into the world at one point. Every Stamp shares this: they
    # decide the feature and the turns, not how a placement is made.
    #
    # A point's y is height above the path, so a RIGID feature adds its own
    # offset here and a DEFORMABLE one leaves it to the column elevation.
    #
    # add_field rather than DomainPolicy.place, which cannot pass a clip.
    rigid = feature.elevation_mode == ElevationMode.RIGID
    origin = BlockPos.coerce(point.position) + feature.offset
    transform = BlockTransform(
        origin + BlockPos(0, point.y_offset, 0) if rigid else origin,
        quarter_turns,
        _elevation(feature, y_offset),
    )
    profile = None if domains is None else domains.profile(feature.domain)
    return world.add_field(
        feature.blocks,
        placement_mode=feature.placement_mode,
        terrain_mask_mode=(
            DEFAULT_TERRAIN_MASK_MODE if profile is None else profile.terrain_mask_mode
        ),
        replace=DEFAULT_BLOCK if profile is None else profile.replace,
        transform=transform,
        connectivity_domain=0 if profile is None else profile.domain_id,
        sequence_index=sequence_index,
        clip=clip,
    )


def stamp_along(
    stamp: Stamp,
    world: BlockWorld,
    path: FeaturePath,
    *,
    clip: ColumnMask,
    y_offset: YOffset | None = None,
    sequence_index: int | None = None,
) -> tuple[PlacementId, ...]:
    # One stamp at every point of a feature path. A path that changes heading
    # part way through changes feature with it.
    #
    # `clip` is the frame this path belongs to. Adjacent frames share their
    # boundary, so the point on it appears in both paths and the clip is what
    # keeps each stamp to its own columns. Pass UNCLIPPED for no clipping.
    #
    # Connectivity is left for resolve(), once every feature is in.
    return tuple(
        stamp.stamp(
            world,
            stamp.rule(point),
            point,
            clip=clip,
            y_offset=y_offset,
            sequence_index=sequence_index,
        )
        for point in path
    )


def _elevation(feature: Feature, y_offset: YOffset | None) -> YOffset | None:
    # DEFORMABLE takes each block's own column elevation. RIGID already has
    # its point's offset in the origin, so it takes none.
    mode = feature.elevation_mode
    if mode == ElevationMode.DEFORMABLE:
        return y_offset
    if mode == ElevationMode.RIGID:
        return None
    raise NotImplementedError(f"Elevation mode {mode} is not stamped yet")
