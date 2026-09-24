from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ..Core.CrossSection.annotations import FeatureAnnotation
from ..Core.CrossSection.cross_section import CrossSection, PointField
from ..Core.Feature.feature import Feature
from ..Core.Field.block_field import BlockField
from ..Core.coordinates import BlockPos
from ..Core.modes import StepMode
from .region import BlockRegion, RegionEntity

NAMESPACE = "vcg"
# Only the first separator splits key from value (`vcg.feature-egg-egg` is
# feature = "egg-egg"); `-` because `/tag` does not accept `=`.
SEPARATOR = "-"

# A marker carries one kind tag (`vcg.annotation`) and any number of
# key/value tags (`vcg.name-left_rail`); other namespaces are ignored.
ANCHOR = "anchor"
ANNOTATION = "annotation"
FEATURE = "feature"


@dataclass(frozen=True, slots=True)
class Marker:
    kind: str
    position: BlockPos
    properties: Mapping[str, str]
    entity: RegionEntity

    @property
    def name(self) -> str:
        return self.require("name")

    @property
    def step_mode(self) -> StepMode:
        return StepMode(self.properties.get("step", StepMode.MANHATTAN.value))

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.properties.get(key, default)

    def require(self, key: str) -> str:
        try:
            return self.properties[key]
        except KeyError:
            raise KeyError(
                f"{self.kind} marker at {tuple(self.position)} has no {key!r} tag"
            ) from None

    def rebased(self, origin: BlockPos) -> Marker:
        return Marker(self.kind, self.position - origin, self.properties, self.entity)


def parse_tags(
    tags: Iterable[str],
    *,
    namespace: str = NAMESPACE,
    separator: str = SEPARATOR,
) -> tuple[str | None, dict[str, str]]:
    # `<ns>.<word>` names the kind; `<ns>.<key><sep><value>` sets a property.
    prefix = f"{namespace}."
    kind: str | None = None
    properties: dict[str, str] = {}
    for tag in tags:
        if not tag.startswith(prefix):
            continue
        body = tag[len(prefix) :]
        key, found, value = body.partition(separator)
        if found:
            properties[key] = value
            continue
        if not body.isidentifier():
            # A property written with the wrong separator.
            raise ValueError(f"{tag!r} is neither a kind nor a {separator!r} property")
        if kind is not None and kind != body:
            raise ValueError(f"marker claims two kinds: {kind!r} and {body!r}")
        kind = body
    return kind, properties


def markers(
    region: BlockRegion,
    *,
    namespace: str = NAMESPACE,
    separator: str = SEPARATOR,
) -> list[Marker]:
    # Positions are absolute, matching the region's own frame.
    found = []
    for entity in region.entities:
        kind, properties = parse_tags(entity.tags, namespace=namespace, separator=separator)
        if kind is not None:
            found.append(Marker(kind, entity.block_pos, properties, entity))
    return found


def of_kind(found: Sequence[Marker], kind: str) -> list[Marker]:
    return [marker for marker in found if marker.kind == kind]


def anchor(found: Sequence[Marker]) -> Marker:
    # Exactly one anchor marks the path point sections are relative to.
    anchors = of_kind(found, ANCHOR)
    if len(anchors) != 1:
        raise ValueError(f"expected exactly one {ANCHOR} marker, found {len(anchors)}")
    return anchors[0]


def rebased(found: Sequence[Marker]) -> list[Marker]:
    # Every marker relative to the anchor.
    origin = anchor(found).position
    return [marker.rebased(origin) for marker in found if marker.kind != ANCHOR]


def section_from_region(
    region: BlockRegion,
    features: Mapping[str, Feature],
    *,
    namespace: str = NAMESPACE,
    separator: str = SEPARATOR,
) -> CrossSection:
    # Positions stay in the region's frame; the anchor becomes the origin.
    found = markers(region, namespace=namespace, separator=separator)
    base = region.origin
    points = PointField()
    for marker in of_kind(found, ANNOTATION):
        name = marker.require("feature")
        if name not in features:
            raise KeyError(
                f"annotation {marker.name!r} wants feature {name!r}; "
                f"supplied: {sorted(features)}"
            )
        points.add(
            FeatureAnnotation(
                marker.name,
                marker.position - base,
                features[name],
                marker.step_mode,
            )
        )
    return CrossSection(points, origin=anchor(found).position - base)


def block_field(region: BlockRegion, *, origin: BlockPos | None = None) -> BlockField:
    # The region's blocks relative to `origin` (default: its minimum corner).
    base = region.origin if origin is None else BlockPos.coerce(origin)
    field = BlockField()
    for position, state in region.iter_occupied():
        field.set_block(position - base, state)
    return field
