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
# Only the first separator in a tag splits key from value, so a value may
# contain it: `vcg.feature-egg-egg` is feature = "egg-egg". `-` is the default
# because `/tag @e add` takes an unquoted word and will not accept `=`.
SEPARATOR = "-"

# A marker entity carries one bare tag naming what it is, plus any number of
# key/value tags. `vcg.annotation` says which kind; `vcg.name-left_rail` says
# something about it. Tags outside the namespace are ignored, so an entity
# may also carry a pack's own tags.
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
            # Catches a property written with the wrong separator, which
            # would otherwise become a very strange kind.
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
    # Section-local coordinates are relative to the path point, not to the
    # schematic corner, so exactly one anchor has to say where that is.
    anchors = of_kind(found, ANCHOR)
    if len(anchors) != 1:
        raise ValueError(f"expected exactly one {ANCHOR} marker, found {len(anchors)}")
    return anchors[0]


def rebased(found: Sequence[Marker]) -> list[Marker]:
    # Every marker relative to the anchor, which the anchor defines as origin.
    origin = anchor(found).position
    return [marker.rebased(origin) for marker in found if marker.kind != ANCHOR]


def section_from_region(
    region: BlockRegion,
    features: Mapping[str, Feature],
    *,
    namespace: str = NAMESPACE,
    separator: str = SEPARATOR,
) -> CrossSection:
    # Annotation positions stay in the region's own frame and the anchor
    # becomes CrossSection.origin, which is what the section subtracts.
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
    # The authored blocks in local coordinates. A CrossSection holds none of
    # its own -- blocks belong to features -- so this is for building one.
    base = region.origin if origin is None else BlockPos.coerce(origin)
    field = BlockField()
    for position, state in region.iter_occupied():
        field.set_block(position - base, state)
    return field
