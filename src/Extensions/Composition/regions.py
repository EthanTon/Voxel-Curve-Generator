from __future__ import annotations

from math import ceil, floor

from ...Core.Field.block_field import BlockField
from ...Core.block.palette import BlockPalette
from ...Core.coordinates import BlockPos
from ...Format.schematics import parse_state_key
from .spec import Polygon, RegionSpec, Triangle


def contains(boundary: Polygon, x: float, z: float) -> bool:
    inside = False
    for (ax, az), (bx, bz) in zip(boundary, (*boundary[1:], boundary[0])):
        cross = (x - ax) * (bz - az) - (z - az) * (bx - ax)
        if abs(cross) < 1e-9 and min(ax, bx) <= x <= max(ax, bx) and min(az, bz) <= z <= max(az, bz):
            return True
        if (az > z) != (bz > z) and x < ax + (z - az) * (bx - ax) / (bz - az):
            inside = not inside
    return inside


def triangle_height(triangle: Triangle, x: float, z: float) -> float | None:
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = triangle
    area = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
    first = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / area
    second = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / area
    third = 1 - first - second
    if min(first, second, third) < -1e-9:
        return None
    return first * ay + second * by + third * cy


def generate_region(spec: RegionSpec, palette: BlockPalette) -> BlockField:
    field = BlockField(palette)
    state = parse_state_key(spec.block)
    xs, zs = zip(*spec.boundary)
    for x in range(ceil(min(xs) - 0.5), floor(max(xs) - 0.5) + 1):
        for z in range(ceil(min(zs) - 0.5), floor(max(zs) - 0.5) + 1):
            # Boundaries follow block edges; sample the centre of each voxel column.
            px, pz = x + 0.5, z + 0.5
            if not contains(spec.boundary, px, pz) or any(contains(hole, px, pz) for hole in spec.holes):
                continue
            height = spec.top + spec.grade[0] * (px - xs[0]) + spec.grade[1] * (pz - zs[0])
            if spec.triangles:
                heights = [value for triangle in spec.triangles if (value := triangle_height(triangle, px, pz)) is not None]
                if not heights:
                    raise ValueError(f"region {spec.name!r} surface does not cover column {(x, z)}")
                if max(heights) - min(heights) > 1e-6:
                    raise ValueError(f"region {spec.name!r} has conflicting surface heights at {(x, z)}")
                height = heights[0]
            top = floor(height + 0.5)
            for y in range(top - spec.thickness + 1, top + 1):
                field.set_block(BlockPos(x, y, z), state)
    return field
