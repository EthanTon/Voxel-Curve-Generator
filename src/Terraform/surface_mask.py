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
    # Retaining walls also hold back water exposed just behind them.
    water_reach: int = 0

    def __post_init__(self) -> None:
        if self.side not in ("above", "below"):
            raise ValueError("surface_mask.side must be 'above' or 'below'")
        if self.reference not in ("original", "designed"):
            raise ValueError("surface_mask.reference must be 'original' or 'designed'")
        if not isinstance(self.offset, int) or isinstance(self.offset, bool):
            raise ValueError("surface_mask.offset must be a whole number of blocks")
        if not isinstance(self.water_reach, int) or isinstance(self.water_reach, bool) or self.water_reach < 0:
            raise ValueError("surface_mask.water_reach must be a non-negative whole number of blocks")
        for name in ("include_water", "include_snow", "include_carpet", "include_foliage"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"surface_mask.{name} must be true or false")


def read_surface_mask(data: Mapping | None) -> SurfaceMaskSpec | None:
    if data is None:
        return None
    if not isinstance(data, Mapping):
        raise ValueError("surface_mask must be a table")
    keys = {"side", "offset", "reference", "include_water", "include_snow", "include_carpet", "include_foliage", "water_reach"}
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
    _water_cache: dict[tuple[str, int, int], int | None] = field(default_factory=dict)

    def update(self, designed: object | None) -> None:
        self.designed = designed
        self._cache.clear()
        self._water_cache.clear()

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
        if spec.include_water and spec.water_reach:
            for nx in range(x - spec.water_reach, x + spec.water_reach + 1):
                for nz in range(z - spec.water_reach, z + spec.water_reach + 1):
                    key_water = (spec.reference, nx, nz)
                    if key_water not in self._water_cache:
                        self._water_cache[key_water] = max((
                            y for y, state in self.terrain.column(nx, nz)
                            if state.name in ("minecraft:water", "minecraft:bubble_column")
                            and (shaped is None or BlockPos(nx, y, nz) not in getattr(shaped, "cut", ()))
                        ), default=None)
                    water = self._water_cache[key_water]
                    if water is not None:
                        selected.append(water)
        height = max(selected) if selected else None
        self._cache[key] = height
        return height

    def mask(self, spec: SurfaceMaskSpec) -> SurfacePredicate:
        return SurfacePredicate(self, spec)


@dataclass(frozen=True, slots=True)
class SurfacePredicate:
    reference: SurfaceReference
    spec: SurfaceMaskSpec
    # Leave the track bed below this absolute height intact when trimming walls.
    minimum_y: int | None = None

    def __call__(self, position: BlockPos) -> bool:
        if self.minimum_y is not None and position.y < self.minimum_y:
            return True
        if self.spec.reference == "designed" and self.reference.designed is None:
            return True
        surface = self.reference.height(position.x, position.z, self.spec)
        if surface is None:
            return False
        boundary = surface + self.spec.offset
        return position.y <= boundary if self.spec.side == "below" else position.y >= boundary
