"""Swept sections keep their authored shape on every heading."""

import tempfile
import unittest
from dataclasses import replace
from math import atan2, cos, degrees, radians, sin
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes

from src.Assembly import finalize_build, place_features, prepare_build
from src.Assembly.structure import build_domains, build_section, laid_beside
from src.Core.Field.block_field import BlockField
from src.Core.Path.path import Path as VoxelPath
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos
from src.Format.region import BlockRegion
from src.Format.schematics import write_schematic
from src.Generation.Geometry.reach import lateral_reach, required_lead
from src.Parser import parse

STAIR = BlockState.create("minecraft:oak_stairs", {
    "facing": "north", "half": "bottom", "shape": "straight", "waterlogged": "false",
})


def route(*legs, start=0.0):
    # ("line", length) or ("arc", radius, degrees), one station per column.
    x = z = 0.0
    theta = radians(start)
    samples = [(x, z, theta)]
    for leg in legs:
        count = int((leg[1] if leg[0] == "line" else abs(radians(leg[2])) * leg[1]) / 0.02)
        for _ in range(count):
            if leg[0] == "arc":
                theta += 0.02 / leg[1] * (1 if leg[2] > 0 else -1)
            x, z = x + cos(theta) * 0.02, z + sin(theta) * 0.02
            samples.append((x, z, theta))
    points, gradients = [], []
    for x, z, theta in samples:
        cell = (round(x), 64, round(z))
        if points and cell == points[-1]:
            continue
        if len(points) > 1 and abs(cell[0] - points[-2][0]) == 1 and abs(cell[2] - points[-2][2]) == 1:
            points.pop()
            gradients.pop()
        points.append(cell)
        gradients.append((cos(theta), 0.0, sin(theta)))
    return points, gradients


def section(cells, steps=None, step="chebyshev"):
    registry = BlockRegistry.for_vanilla_26_2()
    with tempfile.TemporaryDirectory() as directory:
        blocks = BlockField(registry.palette)
        for (x, y, z), state in cells.items():
            blocks.set_block(BlockPos(x, y, z), state)
        region = BlockRegion.from_field(blocks, name="shape")
        write_schematic(region, Path(directory) / "shape.schem", offset=region.origin, data_version=4671)
        spec = parse({
            "path": {"points": [(0, 64, 0), (1, 64, 0)], "gradients": [(1, 0, 0)] * 2},
            "output": {"path": "shape.schem"},
            "domain": [{"name": "bed", "order": 0}],
            "section": [{"name": "shape", "domain": "bed", "schematic": "shape.schem",
                         "steps": dict(steps or {}), "step": step}],
        }, Path(directory))
        return build_section(spec.sections[0], registry, build_domains(spec))


def build(cells, points, gradients, steps=None, lane=None, beside=None):
    # `beside`: a second section's cells, laid over the same stretch.
    registry = BlockRegistry.for_vanilla_26_2()
    with tempfile.TemporaryDirectory() as directory:
        for name, laid in (("shape", cells), ("beside", beside or {})):
            blocks = BlockField(registry.palette)
            for (x, y, z), state in laid.items():
                blocks.set_block(BlockPos(x, y, z), state)
            if laid:
                region = BlockRegion.from_field(blocks, name=name)
                write_schematic(region, Path(directory) / f"{name}.schem", offset=region.origin, data_version=4671)
        spec = parse({
            "path": {"points": points, "gradients": gradients, "lead": 0},
            "output": {"path": "shape.schem"},
            "domain": [{"name": "bed", "order": 0, "placement": {"mode": "OVERWRITE"}},
                       {"name": "terrain", "order": -1}],
            "section": [{"name": "shape", "domain": "bed", "schematic": "shape.schem",
                         "steps": dict(steps or {})}]
            + ([{"name": "beside", "domain": "bed", "schematic": "beside.schem"}] if beside else []),
            "assignment": [{"stations": [0, len(points) - 1], "sections": ["shape", "beside"] if beside else ["shape"]}],
            "lane": [{"name": "rail", "domain": "bed", "sections": ["shape"],
                      "block": lane, "offset": [0, 1, 0], "on": 1, "off": 0}] if lane else [],
        }, Path(directory))
        spans = {each.name: [(0, len(points) - 1)] for each in spec.sections}
        sections = laid_beside({each.name: build_section(each, registry, build_domains(spec))
                                for each in spec.sections}, spans)
        lead = required_lead(VoxelPath(points, gradients), list(sections.values()))
        spec = replace(spec, path=replace(spec.path, lead=lead))
        world = finalize_build(place_features(prepare_build(spec))).world
        return {(p.x, p.y, p.z): world.get_block(p) for p in world.iter_source_positions()
                if world.get_block(p) is not None}


def holes(blocks, y):
    cells = np.array(sorted({(x, z) for x, h, z in blocks if h == y}))
    lowest = cells.min(axis=0)
    grid = np.zeros(tuple(cells.max(axis=0) - lowest + 1), dtype=bool)
    grid[tuple((cells - lowest).T)] = True
    return int((binary_fill_holes(grid) & ~grid).sum())


def pieces(cells):
    left, count = set(cells), 0
    while left:
        count += 1
        stack = [left.pop()]
        while stack:
            x, z = stack.pop()
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (x + dx, z + dz) in left:
                    left.remove((x + dx, z + dz))
                    stack.append((x + dx, z + dz))
    return count


def row(xs, y, name="minecraft:stone"):
    return {(x, y, 0): BlockState.create(name) for x in xs}


class SweptShapeTests(unittest.TestCase):
    def test_deep_cells_stay_on_their_own_column(self) -> None:
        # A roof whose edges are set a block forward and back is laid along
        # travel, so off the quarters no roof line is left empty.
        cells = {**row(range(-3, 4), 0), **row(range(-2, 3), 2)}
        cells[-3, 2, -1] = cells[3, 2, 1] = BlockState.create("minecraft:stone")
        for heading in (20, 45, 60):
            with self.subTest(heading=heading):
                self.assertEqual(holes(build(cells, *route(("line", 40), start=heading)), 66), 0)

    def test_a_column_stepping_at_one_height_is_a_corner_at_every_height(self) -> None:
        cells = {**row(range(-4, 5), 0), **row(range(-4, 5), 3, "minecraft:glass")}
        for key in ("1,0,0", "1,3,0"):
            blocks = build(cells, *route(("line", 40), start=30), {key: "manhattan"})
            for y in (64, 67):
                with self.subTest(step=key, y=y):
                    self.assertEqual(holes(blocks, y), 0)

    def test_a_mirrored_lean_keeps_the_authored_facing(self) -> None:
        # A stair facing travel faces travel whichever side of the quarter
        # the path heads.
        cells = {**row(range(-3, 4), 0), (-1, 1, 0): STAIR}
        for start in (-30, 30):
            points, gradients = route(("line", 40), start=start)
            gx, _, gz = gradients[0]
            heading = degrees(atan2(gx, -gz)) % 360
            facings = {state.get("facing") for state in build(cells, points, gradients).values()
                       if state.name == "minecraft:oak_stairs"}
            angle = {"north": 0, "east": 90, "south": 180, "west": 270}
            with self.subTest(heading=heading):
                self.assertTrue(facings)
                for facing in facings:
                    self.assertLessEqual(abs((angle[facing] - heading + 180) % 360 - 180), 45)

    def test_a_mirrored_lean_keeps_a_lane_facing_travel(self) -> None:
        # The same for a stair ridden as a lane at either lean, and due south.
        lane = "minecraft:oak_stairs[facing=north,half=bottom,shape=straight,waterlogged=false]"
        for start in (-30, 30, 90):
            points, gradients = route(("line", 40), start=start)
            gx, _, gz = gradients[0]
            heading = degrees(atan2(gx, -gz)) % 360
            facings = {state.get("facing") for state in build(row(range(-3, 4), 0), points, gradients, lane=lane).values()
                       if state.name == "minecraft:oak_stairs"}
            angle = {"north": 0, "east": 90, "south": 180, "west": 270}
            with self.subTest(heading=heading):
                self.assertTrue(facings)
                for facing in facings:
                    self.assertLessEqual(abs((angle[facing] - heading + 180) % 360 - 180), 45)

    def test_a_section_laid_beside_another_is_pushed_by_its_steps(self) -> None:
        # A brim beside a stepping edge lands where the same column would in
        # one section, never under the edge's corner.
        host = {**row(range(-2, 3), 0), **row((-2,), 1)}
        brim = row((-4, -3), 0, "minecraft:gold_block")
        steps = {"-2,0": "manhattan", "0,0": "manhattan"}
        for start in (20, -20, 60):
            points, gradients = route(("line", 40), ("arc", 30, 40), start=start)
            together = build({**host, **brim}, points, gradients, steps)
            apart = build(host, points, gradients, steps, beside=brim)
            with self.subTest(start=start):
                self.assertEqual({p for p, s in apart.items() if s.name == "minecraft:gold_block"},
                                 {p for p, s in together.items() if s.name == "minecraft:gold_block"})

    def test_a_path_passing_over_itself_lays_both_passes_whole(self) -> None:
        # A figure 8 lifted clear at its crossing: each pass is laid there as
        # it is when built without the other.
        points, gradients = route(("arc", 12, -105), ("line", 24), ("arc", 12, 270), ("line", 24),
                                  ("arc", 12, -105), start=20)
        first, second = min(((i, j) for i in range(len(points)) for j in range(i + 30, len(points))),
                            key=lambda pair: abs(points[pair[0]][0] - points[pair[1]][0])
                            + abs(points[pair[0]][2] - points[pair[1]][2]))
        heights = []
        for k in range(len(points)):
            wanted = 64 + round(4 * cos(3.14159 * (k - first) / (second - first)))
            heights.append(wanted if not heights else max(heights[-1] - 1, min(heights[-1] + 1, wanted)))
        points = [(x, y, z) for (x, _, z), y in zip(points, heights)]
        cells = row(range(-2, 3), 0)
        whole = build(cells, points, gradients)
        cx, _, cz = points[first]
        for station in (first, second):
            alone = build(cells, points[station - 15:station + 16], gradients[station - 15:station + 16])
            near = {p for p in alone if abs(p[0] - cx) <= 4 and abs(p[2] - cz) <= 4}
            with self.subTest(station=station):
                self.assertTrue(near)
                self.assertEqual({p for p in near if p not in whole}, set())

    def test_a_ridden_staircase_turning_over_keeps_its_deck(self) -> None:
        # A lane rides the deck while the heading crosses north and back, so
        # the stepping edges change lean; the deck keeps its full width.
        points = [(0, 64, -k) for k in range(60)]
        gradients = [(sin(radians(0.3 if 20 <= k < 40 else -0.3)), 0.0, -1.0) for k in range(60)]
        cells = row(range(-6, 7), 0)
        steps = {"-5,0": "manhattan", "5,0": "manhattan", "-3,0": "manhattan", "3,0": "manhattan"}
        blocks = build(cells, points, gradients, steps, lane="minecraft:sponge")
        self.assertTrue(any(state.name == "minecraft:sponge" for state in blocks.values()))
        widths = {}
        for x, y, z in blocks:
            if y == 64 and 5 <= -z <= 54:
                widths[z] = widths.get(z, 0) + 1
        self.assertEqual(set(widths.values()), {13})

    def test_a_section_alike_either_side_reaches_alike_either_side(self) -> None:
        # Its joins turning either way are mirror images, so a path turning
        # left is not refused where the same path turning right is laid.
        for xs in ((-2, -1, 0, 1, 2), (-2, -1, 1, 2), (-1, 0, 1)):
            for depth in (1, 3):
                cells = {(x, 0, z): BlockState.create("minecraft:stone") for x in xs for z in range(depth)}
                with self.subTest(columns=xs, depth=depth):
                    reach = lateral_reach([section(cells)])
                    self.assertAlmostEqual(reach.left, reach.right)

    def test_an_empty_centreline_does_not_step_with_the_section(self) -> None:
        # Kept only for fittings to ride, it would push one side alone.
        cells = {(x, 0, 0): BlockState.create("minecraft:stone") for x in (-2, -1, 1, 2)}
        reach = lateral_reach([section(cells, step="manhattan")])
        self.assertAlmostEqual(reach.left, reach.right)

    def test_overlapping_slices_show_the_same_slice_on_every_heading(self) -> None:
        # Deep slices land on one another along travel. The slice furthest
        # from the path shows, the one behind when two are as deep, whichever
        # way the path heads or turns.
        cells = {}
        for x in range(-2, 3):
            cells[x, 0, -1] = BlockState.create("minecraft:glass")
            cells[x, 0, 0] = BlockState.create("minecraft:stone")
            cells[x, 0, 1] = BlockState.create("minecraft:gold_block")
        for heading in (0, 45, 135, 315):
            points, gradients = route(("line", 40), start=heading)
            blocks = build(cells, points, gradients)
            middle = {(x, z) for x, _, z in points[8:-8]}
            names = {state.name for (x, y, z), state in blocks.items() if (x, z) in middle}
            with self.subTest(heading=heading):
                self.assertEqual(names, {"minecraft:gold_block"})

    def test_a_path_hooking_back_at_its_end_is_refused_rather_than_laid_empty(self) -> None:
        # The plane at the hook faces back over the stretch before it; a frame
        # holding none of its own points would clip the whole section away.
        points = [(x, 64, z) for x, z in (
            (0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1), (6, 1), (7, 2), (8, 2), (9, 3),
            (10, 3), (11, 4), (12, 4), (13, 4), (14, 4), (15, 3), (14, 2))]
        gradients = [(1.0, 0.0, 0.2)] * 14 + [(0.7, 0.0, -0.7), (-0.6, 0.0, -0.8), (-0.8, 0.0, -0.6)]
        cells = {(-5, -1, 0): BlockState.create("minecraft:stone")}
        with self.assertRaisesRegex(ValueError, "turns back on itself"):
            build(cells, points, gradients)

    def test_a_deep_staircase_turning_over_on_a_curve_keeps_each_column_whole(self) -> None:
        # Crossing north on an arc leaves no straight run as long as the
        # deepest push; every stepping column must still be face-connected.
        steps = {f"{x},0": "manhattan" for x in (1, 3, 4, 5, 6, 8)}
        cells = {(x, 0, 0): BlockState.create(name) for x, name in zip(range(0, 9), (
            "minecraft:stone", "minecraft:white_wool", "minecraft:orange_wool", "minecraft:magenta_wool",
            "minecraft:light_blue_wool", "minecraft:yellow_wool", "minecraft:lime_wool", "minecraft:pink_wool",
            "minecraft:gray_wool"))}
        points, gradients = route(("line", 10), ("arc", 40, 40), ("line", 10), start=-110)
        blocks = build(cells, points, gradients, steps)
        for x in (1, 3, 4, 5, 6, 8):
            name = cells[x, 0, 0].name
            columns = {(bx, bz) for (bx, by, bz), state in blocks.items() if state.name == name}
            with self.subTest(column=x):
                self.assertEqual(pieces(columns), 1)

    def test_a_flat_hairpin_ending_short_of_its_start_builds(self) -> None:
        points, gradients = route(("line", 50), ("arc", 12, 180), ("line", 30))
        gradients = [(-1.0, 0.0, 0.0) if k > len(gradients) - 15 else g for k, g in enumerate(gradients)]
        gradients[0] = (1.0, 0.0, 0.0)
        self.assertTrue(build(row(range(-2, 3), 0), points, gradients))


if __name__ == "__main__":
    unittest.main()
