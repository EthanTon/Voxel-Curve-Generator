from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite

from ..Core.coordinates import BlockPos
from .classify import BlockClass, classify
from .terrain import Terrain


@dataclass(frozen=True, slots=True)
class SurfaceMaskSpec:
    # The boundary is inclusive: below keeps y <= surface + offset; above
    # keeps y >= surface + offset. A negative offset reaches into the ground.
    side: str
    offset: int = 0
    reference: str = "original"
    include_water: bool = False
    include_snow: bool = False
    include_carpet: bool = False
    include_foliage: bool = False

    def __post_init__(self) -> None:
        if self.side not in ("above", "below"):
            raise ValueError("surface_mask.side must be 'above' or 'below'")
        if self.reference not in ("original", "designed"):
            raise ValueError("surface_mask.reference must be 'original' or 'designed'")
        if not isinstance(self.offset, int) or isinstance(self.offset, bool):
            raise ValueError("surface_mask.offset must be a whole number of blocks")
        for name in ("include_water", "include_snow", "include_carpet", "include_foliage"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"surface_mask.{name} must be true or false")


def read_surface_mask(data: Mapping | None) -> SurfaceMaskSpec | None:
    if data is None:
        return None
    if not isinstance(data, Mapping):
        raise ValueError("surface_mask must be a table")
    keys = {"side", "offset", "reference", "include_water", "include_snow", "include_carpet", "include_foliage"}
    unknown = set(data) - keys
    if unknown:
        raise ValueError(f"unknown surface_mask option(s): {', '.join(sorted(unknown))}")
    if "side" not in data:
        raise ValueError("surface_mask.side is required ('above' or 'below')")
    return SurfaceMaskSpec(**data)


@dataclass(slots=True)
class SurfaceReference:
    terrain: Terrain
    designed: object | None = None
    _cache: dict[tuple[SurfaceMaskSpec, int, int], int | None] = field(default_factory=dict)

    def update(self, designed: object | None) -> None:
        self.designed = designed
        self._cache.clear()

    def height(self, x: int, z: int, spec: SurfaceMaskSpec) -> int | None:
        key = (spec, x, z)
        if key in self._cache:
            return self._cache[key]
        shaped = self.designed if spec.reference == "designed" else None
        if spec.reference == "designed" and shaped is None:
            # Placement is provisionally unmasked until the earthworks exist.
            return None
        selected: list[int] = []
        has_design = False
        if shaped is not None and shaped.grid.holds(x, z):
            row, cell = shaped.grid.at(x, z)
            natural = float(shaped.natural[row, cell])
            value = float(shaped.designed[row, cell])
            if isfinite(value) and natural > -32768:
                selected.append(round(value))
                has_design = True
        for y, state in self.terrain.column(x, z):
            if shaped is not None and BlockPos(x, y, z) in getattr(shaped, "cut", ()):
                continue
            name = state.name
            if name in ("minecraft:snow", "minecraft:snow_block", "minecraft:powder_snow"):
                include = spec.include_snow
            elif name.endswith("_carpet"):
                include = spec.include_carpet
            elif name in ("minecraft:water", "minecraft:bubble_column"):
                include = spec.include_water
            else:
                kind = classify(state)
                include = kind in (BlockClass.GROUND, BlockClass.STRUCTURE) or (
                    kind is BlockClass.VEGETATION and spec.include_foliage
                )
                if has_design and kind is BlockClass.GROUND:
                    # The designed ground height replaces the old ground.
                    include = False
            if include:
                selected.append(y)
        height = max(selected) if selected else None
        self._cache[key] = height
        return height

    def mask(self, spec: SurfaceMaskSpec) -> SurfacePredicate:
        return SurfacePredicate(self, spec)


@dataclass(frozen=True, slots=True)
class SurfacePredicate:
    reference: SurfaceReference
    spec: SurfaceMaskSpec

    def __call__(self, position: BlockPos) -> bool:
        if self.spec.reference == "designed" and self.reference.designed is None:
            return True
        surface = self.reference.height(position.x, position.z, self.spec)
        if surface is None:
            return False
        boundary = surface + self.spec.offset
        return position.y <= boundary if self.spec.side == "below" else position.y >= boundary
