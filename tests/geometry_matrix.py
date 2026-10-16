"""Deterministic real builds shared by pytest and the Matplotlib audit.

Expected geometry is checked independently of the sweep: exact straight slices,
occupied centreline columns, connected footprints, bounded heights and materials.
"""

from dataclasses import dataclass, replace
from collections import defaultdict
from functools import lru_cache
from math import cos, hypot, pi, sin
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from src.Assembly import finalize_build, place_features, prepare_build
from src.Assembly.earthworks import reshape
from src.Core.Field.block_field import BlockField
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos, ColumnPos
from src.Format.region import BlockRegion
from src.Format.schematics import write_schematic
from src.Generation.Geometry.reach import required_lead
from src.Parser import parse
from src.Terraform import FieldTerrain, Intensity, Look, Plan
from src.Terraform.plan import Brush, Keepout, Slopes, TerrainMask


@dataclass(frozen=True)
class Profile:
    name: str
    cells: tuple[tuple[int, int, int], ...]
    steps: tuple[tuple[str, str], ...] = ()
    step: str = "chebyshev"


@dataclass(frozen=True)
class Route:
    name: str
    points: tuple[tuple[int, int, int], ...]
    gradients: tuple[tuple[float, float, float], ...]
    # Analytic direction for independent straight-section checks.
    straight: tuple[int, int] | None = None


def profile(name, cells, **kwargs):
    return Profile(name, tuple(sorted(set(cells))), **kwargs)


@lru_cache
def profiles():
    floor = lambda lo, hi: [(x, 0, 0) for x in range(lo, hi + 1)]
    box = lambda lo, hi, height: [(x, y, 0) for x in range(lo, hi + 1)
                                for y in range(height + 1)
                                if y in (0, height) or x in (lo, hi)]
    left = floor(-6, 2) + [(-6, y, 0) for y in range(1, 6)]
    return (
        profile("point_1", floor(0, 0)),
        profile("symmetric_flat_5", floor(-2, 2)),
        profile("even_width_8", floor(-3, 4)),
        profile("wide_deck_19", [(x, y, 0) for x in range(-9, 10) for y in (-1, 0)]),
        profile("rectangle_7x5", [(x, y, 0) for x in range(-3, 4) for y in range(-2, 3)]),
        profile("triangle_9", [(x, y, 0) for x in range(-4, 5) for y in range(5 - abs(x))]),
        profile("round_9", [(x, y, 0) for x in range(-4, 5) for y in range(9)
                             if x*x + (y-4)**2 <= 16]),
        profile("hollow_tunnel_9x7", box(-4, 4, 6)),
        profile("asymmetric_left_L", left),
        profile("asymmetric_right_L", [(-x, y, z) for x, y, z in left]),
        profile("asymmetric_steps", [(x, y, 0) for x in range(-2, 7) for y in range(1 + (x+2)//3)]),
        profile("offset_side_6", [(x, y, 0) for x in range(2, 8) for y in (0, 1)]),
        profile("depth_3", [(x, y, z) for x in range(-3, 4) for y in (-1, 0) for z in (-1, 0, 1)]),
        profile("recessed_roof_depth_3", [(x, y, z) for x, y, _ in box(-4, 4, 6)
                                        for z in ((1,) if y == 6 else (-1, 0, 1))]),
        profile("stepping_edges_9", floor(-4, 4), steps=tuple((f"{x},0", "manhattan") for x in (-4, -2, 2, 4))),
        profile("height_specific_steps", box(-4, 4, 6), steps=(("-4,0,0", "manhattan"), ("4,0,0", "manhattan"))),
        profile("all_manhattan_7", floor(-3, 3), step="manhattan"),
    )


def sampled(name, xy, gradients, heights=None):
    """Voxelise analytic inputs without consecutive duplicates or tiny jogs."""
    points, kept = [], []
    for i, ((x, z), gradient) in enumerate(zip(xy, gradients)):
        y = 64 if heights is None else round(heights[i])
        cell = (round(x), y, round(z))
        if points and cell == points[-1]:
            continue
        if (len(points) > 1 and points[-2][1] == y == points[-1][1]
                and abs(cell[0] - points[-2][0]) == 1
                and abs(cell[2] - points[-2][2]) == 1):
            points.pop()
            kept.pop()
        points.append(cell)
        kept.append(tuple(gradient))
    return Route(name, tuple(points), tuple(kept))


def arc(radius, angle=pi/2, sign=1):
    ts = np.linspace(0, angle, max(32, int(radius * angle * 16)) + 1)
    return sampled(f"arc_R{radius}_{'cw' if sign > 0 else 'ccw'}_{round(angle*180/pi)}",
                   [(radius*sin(t), sign*radius*(1-cos(t))) for t in ts],
                   [(cos(t), 0., sign*sin(t)) for t in ts])


def transformed(route, quarter=0, shift=(0, 0, 0), reverse=False, name=None):
    def rotate(x, z):
        for _ in range(quarter % 4):
            x, z = -z, x
        return x, z
    points = []
    gradients = []
    for (x, y, z), (gx, gy, gz) in zip(route.points, route.gradients):
        x, z = rotate(x, z)
        gx, gz = rotate(gx, gz)
        points.append((x+shift[0], y+shift[1], z+shift[2]))
        gradients.append((gx, gy, gz))
    direction = None if route.straight is None else rotate(*route.straight)
    if reverse:
        points.reverse()
        gradients = [tuple(-v for v in g) for g in reversed(gradients)]
        direction = None if direction is None else tuple(-v for v in direction)
    return Route(name or route.name, tuple(points), tuple(gradients), direction)


@lru_cache
def routes():
    result = []
    for dx, dz in ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)):
        result.append(Route(f"straight_{dx}_{dz}", tuple((i*dx, 64, i*dz) for i in range(49)),
                            ((dx, 0., dz),)*49, (dx, dz)))
    for radius in (3, 6, 12, 25, 60, 150):
        for sign in (-1, 1):
            result.append(arc(radius, sign=sign))
    for radius in (12, 35):
        for sign in (-1, 1):
            result.append(arc(radius, pi, sign))
    result.append(arc(30, 2*pi - .25))  # near-closed returning route
    for radius in (10, 40):
        first = arc(radius)
        second = arc(radius, sign=-1)
        # Reflect the second bend through the outgoing tangent of the first.
        second_xy = [(first.points[-1][0]+z, first.points[-1][2]+x) for x, _, z in second.points[1:]]
        second_g = [(g[2], g[1], g[0]) for g in second.gradients[1:]]
        result.append(Route(f"S_bend_R{radius}", first.points + tuple((x, 64, z) for x, z in second_xy),
                            first.gradients + tuple(second_g)))
    ts = np.linspace(0, 2*pi, 1801)
    result.append(sampled("figure_eight", [(28*sin(t), 14*sin(2*t)) for t in ts],
                          [(28*cos(t), 0., 28*cos(2*t)) for t in ts]))
    ts = np.linspace(0, 90, 1441)
    result.append(sampled("changing_radius_wave", [(t, 9*sin(t/9)) for t in ts],
                          [(1., 0., cos(t/9)) for t in ts]))
    xy = [(x, 6*abs((x % 24)-12)/12) for x in range(97)]
    result.append(sampled("sharp_zigzag", xy, [(1., 0., -.5 if x % 24 < 12 else .5) for x in range(97)]))
    base = result[0]
    for quarter in (1, 2, 3):
        result.append(transformed(arc(25), quarter, name=f"arc_R25_rotated_{quarter*90}"))
    result.append(transformed(arc(25), reverse=True, name="arc_R25_reverse"))
    result.append(transformed(arc(25), shift=(-400, -96, -700), name="negative_coordinates_and_height"))
    result.append(transformed(arc(25), shift=(300000, 180, -300000), name="distant_coordinates_and_height"))
    result.append(Route("sparse_diagonal", tuple((i*3, 64, i*3) for i in range(25)), ((1., 0., 1.),)*25))
    points = list(base.points)
    points.insert(20, points[20])
    result.append(Route("repeated_interior_station", tuple(points), ((1., 0., 0.),)*len(points), (1, 0)))
    result.append(Route("short_2_stations", ((0, 64, 0), (1, 64, 0)), ((1., 0., 0.),)*2, (1, 0)))
    result.append(Route("single_station", ((0, 64, 0),), ((1., 0., 0.),), (1, 0)))
    for kind in ("uphill", "downhill", "crest_and_sag"):
        heights = [64 + (i//8 if kind == "uphill" else -i//8 if kind == "downhill"
                         else round(3*sin(i*pi/24))) for i in range(49)]
        result.append(Route(kind, tuple((i, heights[i], 0) for i in range(49)),
                            tuple((1., (heights[min(i+1, 48)]-heights[max(i-1, 0)])/2, 0.) for i in range(49)), (1, 0)))
    return tuple(result)


MATERIALS = ("minecraft:stone", "minecraft:gold_block", "minecraft:blue_concrete")

# These profiles reach past the inside of their tiny supplied bend. The
# documented frame safety check must reject them, rather than emit an overlap.
EXPECTED_REJECTIONS = {
    "wide_deck_19": {"arc_R3_ccw_90", "arc_R3_cw_90", "arc_R6_ccw_90", "arc_R6_cw_90"},
    "asymmetric_left_L": {"arc_R3_ccw_90"},
    "asymmetric_right_L": {"arc_R3_cw_90"},
    "asymmetric_steps": {"arc_R3_cw_90"},
    "offset_side_6": {"arc_R3_cw_90"},
    "all_manhattan_7": {"arc_R3_ccw_90", "arc_R3_cw_90"},
}


def write_profiles(directory):
    directory.mkdir(parents=True, exist_ok=True)
    registry = BlockRegistry()
    for shape in profiles():
        blocks = BlockField(registry.palette)
        for x, y, z in shape.cells:
            # Distinguish sides to catch an accidental lateral mirror.
            blocks.set_block(BlockPos(x, y, z), BlockState.create(MATERIALS[0 if x == 0 else 1 if x < 0 else 2]))
        region = BlockRegion.from_field(blocks, name=shape.name)
        write_schematic(region, directory / f"{shape.name}.schem", offset=region.origin, data_version=3700)


def make_spec(directory, route, shapes, transitions=False, ridden=False):
    assignments = [{"stations": [0, len(route.points)-1], "sections": [shapes[0].name]}]
    if transitions:
        bounds = np.linspace(0, len(route.points), len(shapes)+1, dtype=int)
        assignments = [{"stations": [int(a), int(b)-1], "sections": [shape.name]}
                       for shape, a, b in zip(shapes, bounds, bounds[1:])]
    spec = parse({
        "path": {"points": route.points, "gradients": route.gradients, "lead": 0},
        "output": {"path": "audit.schem"},
        "domain": [{"name": "bed", "order": 0, "placement": {"mode": "OVERWRITE"}},
                   {"name": "terrain", "order": -1}, {"name": "fitting", "order": 1}],
        "section": [{"name": shape.name, "domain": "bed", "schematic": f"{shape.name}.schem",
                     "width": max(x for x, _, _ in shape.cells)-min(x for x, _, _ in shape.cells)+1,
                     "depth": max(z for _, _, z in shape.cells)-min(z for _, _, z in shape.cells)+1,
                     "steps": dict(shape.steps), "step": shape.step} for shape in shapes],
        "assignment": assignments,
        "lane": ([{"name": "centre_marker", "domain": "fitting", "sections": [s.name for s in shapes],
                  "block": "minecraft:red_concrete", "offset": [0, 10, 0], "on": 1, "off": 0}]
                 if ridden else []),
    }, directory)
    # Compute required padding using the actual loaded profile, not a guessed width.
    from src.Assembly.structure import build_domains, build_section
    from src.Core.Path.path import Path as VoxelPath
    sections = [build_section(each, BlockRegistry.for_vanilla_26_2(), build_domains(spec)) for each in spec.sections]
    lead = required_lead(VoxelPath(route.points, route.gradients), sections)
    return replace(spec, path=replace(spec.path, lead=lead))


def components(cells, face=False):
    if not cells:
        return 0
    array = np.array(list(cells), dtype=int)
    lo = array.min(axis=0)
    occupied = np.zeros(tuple(array.max(axis=0)-lo+1), dtype=bool)
    occupied[tuple((array-lo).T)] = True
    from scipy.ndimage import generate_binary_structure
    neighbours = generate_binary_structure(array.shape[1], 1) if face else np.ones((3,)*array.shape[1])
    return int(label(occupied, structure=neighbours)[1])


def check_l_wall(prepared, body, shape, continuous_input):
    """Check the raised wall separately: a connected floor can hide wall gaps."""
    heights = defaultdict(set)
    wrong_side = set()
    wrong_material = set()
    left = shape.name == "asymmetric_left_L"
    material = MATERIALS[1 if left else 2]
    for (x, y, z), state in body.items():
        frame = prepared.silhouette.frame_at(x, z)
        if frame is None:
            continue  # The caller already reports unframed output.
        height = y - frame.y_offset
        heights[x, z].add(height)
        if height <= 0:
            continue
        if state != material:
            wrong_material.add((x, z))
        point = prepared.path.path_points[prepared.silhouette.owner(x, z)]
        gx, _, gz = point.gradient
        across = ((x - point.position.x) * -gz + (z - point.position.z) * gx) / hypot(gx, gz)
        if (across >= -1e-9 if left else across <= 1e-9):
            wrong_side.add((x, z))
    wall = {cell for cell, levels in heights.items() if any(y > 0 for y in levels)}
    parts = components(wall)
    incomplete = {cell for cell, levels in heights.items() if levels not in ({0}, set(range(6)))}
    issues = []
    if continuous_input and parts != 1:
        issues.append(f"disconnected raised wall: {parts} components")
    if incomplete:
        issues.append(f"incomplete wall/floor stacks: {len(incomplete)}")
    if wrong_side:
        issues.append(f"raised wall on wrong side of travel: {len(wrong_side)} columns")
    if wrong_material:
        issues.append(f"raised wall material changed: {len(wrong_material)} columns")
    return {"wall_columns": len(wall), "wall_components": parts,
            "incomplete_wall_stacks": len(incomplete), "wrong_side_wall_columns": len(wrong_side),
            "wrong_material_wall_columns": len(wrong_material)}, issues


def check_build(directory, route, shapes, transitions=False, ridden=False):
    spec = make_spec(directory, route, shapes, transitions, ridden)
    expected_rejection = route.name == "single_station" or (
        len(shapes) == 1 and route.name in EXPECTED_REJECTIONS.get(shapes[0].name, ())
    )
    try:
        prepared = prepare_build(spec)
    except ValueError as error:
        matches = ("at least two points to be framed" in str(error) if route.name == "single_station"
                   else "Boundaries" in str(error) and "cross within" in str(error))
        if not expected_rejection or not matches:
            raise
        return {"route": route.name, "profiles": [s.name for s in shapes], "ridden": ridden,
                "stations": len(route.points), "lead": spec.path.lead, "blocks": 0,
                "expected_rejection": str(error), "issues": []}, None, None, {}
    built = finalize_build(place_features(prepared))
    states = {tuple(p): built.world.get_block(p).name for p in built.world.iter_source_positions()
              if built.world.get_block(p) is not None}
    body = {p: state for p, state in states.items() if state in MATERIALS}
    issues = []
    if expected_rejection:
        issues.append("unsupported input was not rejected")
    if not body:
        issues.append("empty output")
    columns = {(x, z) for x, _, z in body}
    parts = components(columns)
    # A sparse supplied route may intentionally contain jumps; measure those
    # separately instead of assuming they have continuous input stations.
    continuous_input = all(max(abs(a[0]-b[0]), abs(a[2]-b[2])) <= 1
                           for a, b in zip(route.points, route.points[1:]))
    if continuous_input and parts != 1:
        issues.append(f"disconnected footprint: {parts} components")
    if continuous_input and components(body) != 1:
        issues.append("disconnected voxel volume")
    if continuous_input and all(s.step == "manhattan" for s in shapes) and components(columns, face=True) != 1:
        issues.append("Manhattan footprint is not face-connected")
    if any(prepared.silhouette.owner(x, z) is None or prepared.silhouette.frame_at(x, z) is None for x, z in columns):
        issues.append("blocks outside the framed route")
    if not set(states.values()) <= set(MATERIALS) | {"minecraft:red_concrete"}:
        issues.append("unexpected material")
    # All profiles except the deliberately offset side profile have centre material.
    if all(any(x == z == 0 for x, _, z in shape.cells) for shape in shapes):
        missing = {(x, z) for x, _, z in route.points} - columns
        if missing:
            issues.append(f"missing centreline columns: {len(missing)}; sample={sorted(missing)[:3]}")
    ylo = min(y for _, y, _ in route.points) + min(y for s in shapes for _, y, _ in s.cells)
    yhi = max(y for _, y, _ in route.points) + max(y for s in shapes for _, y, _ in s.cells)
    if any(y < ylo or y > yhi for _, y, _ in body):
        issues.append("profile height exceeded")
    # A cardinal straight interior must reproduce the authored cross section,
    # including holes, unequal sides and side-specific materials.
    if (route.straight in ((1, 0), (-1, 0), (0, 1), (0, -1)) and not transitions
            and len(route.points) >= 40 and len({y for _, y, _ in route.points}) == 1):
        shape = shapes[0]
        x0, y0, z0 = route.points[len(route.points)//2]
        dx, dz = route.straight
        expected = {(x0-dz*x, y0+y, z0+dx*x): MATERIALS[0 if x == 0 else 1 if x < 0 else 2]
                    for x, y, _ in shape.cells}
        actual = {p: v for p, v in body.items() if (p[0]-x0)*dx + (p[2]-z0)*dz == 0}
        if actual != expected:
            issues.append(f"straight slice changed: missing={len(expected.keys()-actual.keys())}, "
                          f"extra={len(actual.keys()-expected.keys())}, wrong_material="
                          f"{sum(actual.get(p) != v for p, v in expected.items() if p in actual)}")
    wall_metrics = {}
    if len(shapes) == 1 and shapes[0].name in ("asymmetric_left_L", "asymmetric_right_L"):
        wall_metrics, wall_issues = check_l_wall(prepared, body, shapes[0], continuous_input)
        issues.extend(wall_issues)
    return {"route": route.name, "profiles": [s.name for s in shapes], "ridden": ridden,
            "stations": len(route.points), "lead": spec.path.lead, "blocks": len(body),
            "footprint_components": parts, **wall_metrics, "issues": issues}, prepared, built, body


ENVIRONMENTS = ("flat_at_grade", "low_plain_fill", "high_plain_cut", "rolling_hills",
                "left_cliff", "right_cliff", "river", "deep_water", "protected_structure", "missing_chunks")


def environment_height(name, x, z):
    return {"flat_at_grade": 63, "low_plain_fill": 57, "high_plain_cut": 72,
            "rolling_hills": round(64+5*sin(x/8)*cos(z/9)),
            "left_cliff": 74 if z < 0 else 58,
            "right_cliff": 58 if z < 0 else 74,
            "river": 57 if abs(z-10) < 6 else 63,
            "deep_water": 54, "protected_structure": 60, "missing_chunks": 61}[name]


def check_environment(directory, route, shape, name, look=Look.ENGINEERED):
    metrics, prepared, built, body = check_build(directory, route, (shape,))
    field = BlockField(built.world.registry.palette)
    xy = np.array([(x, z) for x, _, z in body])
    least, most = xy.min(axis=0)-13, xy.max(axis=0)+13
    for x in range(int(least[0]), int(most[0])+1):
        for z in range(int(least[1]), int(most[1])+1):
            if name == "missing_chunks" and x//16 == 1:
                continue
            top = environment_height(name, x, z)
            for y in range(top-4, top+1):
                field.set_block(BlockPos(x, y, z), BlockState.create("minecraft:grass_block" if y == top else "minecraft:stone"))
            wet = name == "deep_water" or name == "river" and abs(z-10) < 6
            if wet:
                for y in range(top+1, 67 if name == "deep_water" else 63):
                    field.set_block(BlockPos(x, y, z), BlockState.create("minecraft:water"))
            if name == "protected_structure" and 12 <= x <= 17 and -11 <= z <= -8:
                field.set_block(BlockPos(x, top+1, z), BlockState.create("minecraft:bricks"))
    keep = Keepout(ColumnPos(-8, -8), ColumnPos(-6, -6), "audit keepout")
    plan = Plan(look=look, default=Intensity.SHAPE, margin=12, reach=10, slopes=Slopes(seed=17),
                brush=Brush(passes=1), keepouts=(keep,),
                terrain_mask=TerrainMask(preserve_water=True, preserve_structures=True))
    spec = replace(prepared.spec, terraform=plan)
    shaped = reshape(spec, built.world, built.world.registry, prepared.domains, prepared.path, {},
                     prepared.silhouette, prepared.spans, apply=True, terrain=FieldTerrain(field))
    issues = list(metrics["issues"])
    if not np.isfinite(shaped.designed).all():
        issues.append("nonfinite designed terrain")
    if any(keep.holds(p.x, p.z) for p, _ in shaped.fill.iter_block_ids()) or any(keep.holds(p.x, p.z) for p in shaped.cut):
        issues.append("keepout modified")
    # cut describes native terrain, which can occupy the same coordinates as
    # the build. Verify applied world output, not intersection of those sets.
    if any((state := built.world.get_block(BlockPos(*p))) is None or state.name != material
           for p, material in body.items()):
        issues.append("authored structure changed by earthworks")
    if name == "low_plain_fill" and not len(shaped.fill):
        issues.append("low ground received no fill")
    if name == "high_plain_cut" and not shaped.cut:
        issues.append("high ground received no cut")
    if name == "missing_chunks" and not shaped.report.ungenerated:
        issues.append("missing chunks not reported")
    if name == "missing_chunks" and any(p.x//16 == 1 for p, _ in shaped.fill.iter_block_ids()):
        issues.append("fill placed in missing chunks")
    if name == "protected_structure":
        bricks = {p for p, block in field.iter_block_ids() if field.palette.get_state(block).name == "minecraft:bricks"}
        if bricks & shaped.cut:
            issues.append("protected structure removed")
    return {**metrics, "environment": name, "look": look.value, "filled": len(shaped.fill),
            "carved": len(shaped.cut), "ungenerated": shaped.report.ungenerated, "issues": issues}, shaped, body
