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
    # A mask that clips nothing. Clips are required, so this is how a caller
    # says it meant to stamp everywhere.

    def contains(self, x: float, z: float) -> bool:
        return True


UNCLIPPED = Everywhere()

# Which columns a point may be stamped into. A run crosses frames, so the
# clip is asked for per point rather than given once.
ClipRule = Callable[[FeaturePathPoint], ColumnMask]


def unclipped(point: FeaturePathPoint) -> ColumnMask:
    return UNCLIPPED


def in_frames(frames: Sequence[PathFrame]) -> ClipRule:
    # Each point clipped to the frame its own column sits in. Frames partition
    # columns, so exactly one holds any point.
    def clip(point: FeaturePathPoint) -> ColumnMask:
        for frame in frames:
            if frame.contains(point.position.x, point.position.z):
                return frame
        raise ValueError(
            f"{tuple(point.position)} is in no frame, so nothing says which "
            "columns it may be stamped into"
        )

    return clip


@runtime_checkable
class Stamp(Protocol):
    # `rule` picks the feature for a point. `stamp` places it.
    # Implementations differ in what `rule` reads off the point.

    def rule(self, point: FeaturePathPoint) -> Feature: ...

    # Every placement the stamp made at this point.
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


# Mirroring across the diagonal is a mirror across one axis and a quarter
# turn together. A reflection is not a rotation, but the two compose into
# one, and this is the turn that goes with the mirror across x.
DIAGONAL_TURN = 1


def laid_out(quarter_turns: int) -> int:
    # Schematics are authored with local x across travel, not along it, so
    # they sit a quarter ahead of the quarter they are placed at. Local x then
    # lands to the right of travel every way round.
    return (quarter_turns - 1) % 4


def lateral_turns(turn: int) -> int:
    # The quarter turns that lay a feature out for a turn counted from west:
    # west at 3, north at 0, east at 1, south at 2.
    return laid_out(normalize(turn) // 90)


@dataclass(frozen=True, slots=True)
class TurnStamp:
    # One feature, laid out at the turn its point was delivered with. A swept
    # profile is stamped by one of these: the section chose the turn, so the
    # blocks follow it rather than deciding again from the gradient.
    #
    # Blocks only turn by quarters. A diagonal is two quarter layouts
    # staggered, and the section records which of them placed each position,
    # so each half is turned its own way. Left to the turn alone both halves
    # would take the quarter below the diagonal: entering a diagonal would
    # change nothing, and leaving one would turn the profile 90 degrees in a
    # single step and tear a wedge out of the inside of the bend.

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
                # A mirrored layout mirrors its blocks too. Across local
                # depth, the section's own long edge, which is the same
                # mirror the positions took.
                mirror="z" if point.mirrored else None,
                domains=self.domains,
                y_offset=y_offset,
                sequence_index=sequence_index,
            ),
        )


class HeadingStamp:
    # A feature per heading, picked by the way the path travels in plan view.
    #
    # Only the first octant is authored. Quarter turns carry a variant into
    # the other three quadrants and a mirror carries it across the diagonal
    # into the other half of its own, so 22.5 degree spacing needs three
    # features, not sixteen and not four -- 67.5 is 22.5 mirrored. Grade is
    # ignored: a climb and a level run on the same bearing get the same one.

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
        # The feature authored for a heading, the quarter turns round its own
        # orbit, and whether it is mirrored across the diagonal to get there.
        # `stamp` shifts the turns by the authoring convention. Cached so the
        # same feature object comes back every time, keeping BlockWorld's
        # rotated-state cache warm.
        step = step_of(heading, self.resolution)
        found = self._turns.get(step)
        if found is None:
            quarter_turns, within = divmod(step, steps_per_quarter(self.resolution))
            offset, mirrored = authored_for(within, self.resolution)
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
                # Mirroring across the diagonal is a mirror across one axis
                # and a quarter turn together: a reflection cannot be a
                # rotation, but the two compose into one.
                quarter_turns=laid_out(quarter_turns + (DIAGONAL_TURN if mirrored else 0)),
                mirror="x" if mirrored else None,
                # Where it stands is the section's business, not the
                # heading's: a pole beside the deck belongs where the
                # section put the column it stands on. Rotating its offset
                # by the heading's own quarter drifts it as the heading
                # turns within that quarter.
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
    # One feature into the world at one point. Every Stamp shares this: they
    # decide the feature and the turns, not how a placement is made.
    #
    # A point's y is height above the path, so a RIGID feature adds its own
    # offset here and a DEFORMABLE one leaves it to the column elevation.
    #
    # A feature's offset is authored in the same local axes as its blocks, so
    # it turns with them: an offset to the right of travel stays there.
    #
    # add_field rather than DomainPolicy.place, which cannot pass a clip.
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
    clip: ClipRule,
    y_offset: YOffset | None = None,
    sequence_index: int | None = None,
) -> tuple[PlacementId, ...]:
    # One stamp at every point of a feature path. A path that changes heading
    # part way through changes feature with it.
    #
    # `clip` answers per point, because one run crosses several frames and a
    # stamp belongs to the columns of the frame it landed in. Pass `unclipped`
    # for no clipping, or `in_frames(frames)`.
    #
    # Connectivity is left for resolve(), once every feature is in.
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
    # An offset authored in local axes, turned with the blocks it belongs to.
    x, z = quarter_turn(offset.x, offset.z, quarter_turns)
    return BlockPos(round(x), offset.y, round(z))


def laid_offset(offset: BlockPos, layout: int, mirrored: bool = False) -> BlockPos:
    # Where an offset lands: exactly where the section puts its own column at
    # that local x, in the layout that placed this position -- and mirrored
    # the same way that layout is, since a mirror across the long edge turns
    # local depth over and leaves local x alone.
    #
    # Asked as an angle instead -- rotate the offset and round it -- a
    # diagonal shortens it, because rounding each axis on its own lands
    # inside the true point. An offset of 2 at 225 degrees is (-1.41, +1.41),
    # which rounds to (-1, +1): a rail meant to sit two to the right of
    # travel ends up one and a half, off the column it belongs on. Placing it
    # the way the section places that column puts it back on the deck.
    x, z = placed_column(layout, offset.x, -offset.z if mirrored else offset.z)
    return BlockPos(x, offset.y, z)


def _elevation(feature: Feature, y_offset: YOffset | None) -> YOffset | None:
    # DEFORMABLE takes each block's own column elevation. RIGID already has
    # its point's offset in the origin, so it takes none.
    mode = feature.elevation_mode
    if mode == ElevationMode.DEFORMABLE:
        return y_offset
    if mode == ElevationMode.RIGID:
        return None
    raise NotImplementedError(f"Elevation mode {mode} is not stamped yet")
