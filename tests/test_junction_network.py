import json
from collections import Counter
from pathlib import Path
import tomllib

import pytest

from src.Core.block.info import SolidFace
from src.Core.coordinates import BlockPos
from src.Extensions.Composition import build, compose
from examples.junction_network.generate import rasterize
from examples.junction_network.variants import variant


ROOT = Path(__file__).resolve().parents[1] / "examples/junction_network"
DIRECTIONS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
OPPOSITE = {"north": "south", "east": "west", "south": "north", "west": "east"}


def ports(shape):
    if shape.startswith("ascending_"):
        direction = shape.removeprefix("ascending_")
        return {direction: 1, OPPOSITE[direction]: 0}
    return {direction: 0 for direction in shape.split("_")}


def connected(rails, first, second):
    direction = next((name for name, (dx, dz) in DIRECTIONS.items()
                      if (second[0] - first[0], second[2] - first[2]) == (dx, dz)), None)
    if direction is None:
        return False
    a, b = ports(rails[first].get("shape")), ports(rails[second].get("shape"))
    return (direction in a and OPPOSITE[direction] in b
            and first[1] + a[direction] == second[1] + b[OPPOSITE[direction]])


@pytest.fixture(scope="module", params=("main", "branch"))
def network(request):
    spec = compose(variant(request.param))
    world = build(spec).world
    expected = json.loads((ROOT / "expectations.json").read_text())
    expected.update(expected["variants"][request.param])
    rails = {tuple(position): world.get_block(position) for position in world.iter_source_positions()
             if world.get_block(position).name == "minecraft:rail"}
    return spec, world, expected, rails


def test_paths_come_from_the_bezier_tool():
    design = json.loads((ROOT / "design.json").read_text())
    assert design["tool"] == "minescript_integration.curve.generate_curve"
    for name, definition in design["paths"].items():
        generated = rasterize(definition["segments"])
        if jump := definition.get("jump"):
            for point in generated["points"]:
                if point[0] >= jump["x"]:
                    point[:] = [a + b for a, b in zip(point, jump["delta"])]
        assert tomllib.loads((ROOT / "paths" / f"{name}.toml").read_text())["path"] == generated
    elevated = rasterize(design["paths"]["merge_b1"]["segments"])
    assert any(g[1] for g in elevated["gradients"])
    assert any(g[2] for g in elevated["gradients"])


def test_junction_declarations_and_rebase(network):
    spec, _, _, _ = network
    config = tomllib.loads((ROOT / "composition.toml").read_text())
    assert len(spec.components) == 24
    assert len(spec.connections) == 20
    assert Counter(t["handler"] for t in config["transition"]) == {"merge": 4, "split": 4, "cross": 4, "continue": 4}
    branch = next(c for c in spec.components if c.name == "merge_a1")
    source = tomllib.loads((ROOT / "paths/merge_a1.toml").read_text())["path"]
    assert source["points"][100][0] == 1100
    assert branch.build.path.points[100] == BlockPos(100, *source["points"][100][1:])
    assert list(branch.build.path.gradients[100]) == source["gradients"][100]
    beds = [c for c in spec.components if c.name.endswith("bed")]
    assert all(not c.build.lanes and not c.build.wires for c in beds)
    assert all(f.source.startswith("section:") for c in beds for f in c.build.feature_paths)
    cropped = next(c for c in beds if c.name == "split_bed")
    assert cropped.build.path.points[0].x == 280
    assert cropped.build.path.points[-1].x == 300


def test_selected_turnouts_and_every_authored_rail_connection(network):
    _, _, expected, rails = network
    inactive = {frozenset(map(tuple, t["inactive_edge"])) for t in expected["turnouts"]}
    assert expected["selection"] in ("main", "branch")
    assert len(inactive) == 8
    assert {"ascending_east", "ascending_west"} <= {s.get("shape") for s in rails.values()}
    assert set(rails) == {tuple(p) for route in expected["routes"].values() for p in route}
    for route in expected["routes"].values():
        for first, second in zip(route, route[1:]):
            a, b = tuple(first), tuple(second)
            assert connected(rails, a, b) == (frozenset((a, b)) not in inactive), (a, b)
    adjacent = {p: [q for dx, dz in DIRECTIONS.values() for dy in (-1, 0, 1)
                    if (q := (p[0] + dx, p[1] + dy, p[2] + dz)) in rails and connected(rails, p, q)]
                for p in rails}
    assert Counter(map(len, adjacent.values())) == {1: 20, 2: len(rails) - 20}
    remaining, groups = set(rails), []
    while remaining:
        group, pending = set(), [remaining.pop()]
        while pending:
            p = pending.pop()
            group.add(p)
            for q in adjacent[p]:
                if q in remaining:
                    remaining.remove(q)
                    pending.append(q)
        groups.append(group)
    assert len(groups) == 10
    ends = ((-15, -9), (-9, -3)) if expected["selection"] == "main" else ((9, 3), (15, 9))
    for start, end in ends:
        assert any({(0, 81, start), (720, 81, end)} <= group for group in groups)


def test_rail_support_contact_wires_and_clearance(network):
    _, world, _, rails = network
    for x, y, z in rails:
        below = world.get_block(BlockPos(x, y - 1, z))
        assert below is not None
        assert world.registry.get_info(world.registry.register(below)).solid_faces & SolidFace.UP
        assert world.get_block(BlockPos(x, y + 6, z)).name == "minecraft:iron_bars"
        for dy in range(1, 5):
            above = world.get_block(BlockPos(x, y + dy, z))
            assert above is None or above.name == "minecraft:air", (x, y, z, dy, above)


def test_shared_decks_piers_and_empty_space_ownership(network):
    _, world, _, rails = network
    for x, zs in ((0, {-30, -15, -9, 9, 15}), (240, {-3, 3}),
                  (500, {-9, -3, 3, 9}), (600, {-15, -9, -3, 3, 9, 15}), (700, {-9, -3, 3, 9})):
        assert {z for a, _, z in rails if a == x} == zs
    for x, half, owner in ((240, 6, "common_bed"), (500, 12, "four_bed"),
                           (600, 18, "four_bed"), (700, 12, "four_bed")):
        for z in range(-half, half + 1):
            p = BlockPos(x, 78, z)
            assert world.get_block(p).name == "minecraft:smooth_stone"
            assert world.owner_at(p) == owner
        for z in (-half, half):
            assert world.get_block(BlockPos(x, 81, z)).name == "minecraft:stone_brick_wall"
        for z in (-half - 1, half + 1):
            assert world.get_block(BlockPos(x, 78, z)) is None
    for x in (200, 240):
        assert world.owner_at(BlockPos(x, 70, 0)) == "common_1"
        assert world.get_block(BlockPos(x, 70, -3)) is None
        assert world.get_block(BlockPos(x, 70, 3)) is None
    p = BlockPos(240, 82, 0)
    assert world.get_block(p) is None
    assert world.ownership.owner_at("common_2", "bed", p) == "common_bed"
    assert world.owner_at(BlockPos(581, 70, 0)) == "four_bed"
    for x, left, right in ((400, -12, 12), (601, -18, 18)):
        for z in (left, right):
            assert world.get_block(BlockPos(x, 87, z)).name == "minecraft:stone_brick_wall"
        for z in range(left, right + 1):
            assert world.get_block(BlockPos(x, 90, z)).name == "minecraft:iron_bars"
    for z in (-15, -9, -3, 3, 9, 15):
        assert world.get_block(BlockPos(601, 88, z)).name == "minecraft:iron_bars"
    for x in range(480, 721):
        for z in (-9, -3, 3, 9):
            assert (x, 81, z) in rails
            for side in (-1, 1):
                if (x, 81, z + side) not in rails:
                    assert world.get_block(BlockPos(x, 80, z + side)).name == "minecraft:anvil", (x, z, side)


def test_transition_decks_follow_the_ground_track_envelope(network):
    _, world, _, rails = network
    for name, first, stop in (("merge_bed", 110, 200), ("split_bed", 280, 300), ("fan_bed", 300, 480)):
        for x in range(first, stop):
            tracks = {z for a, y, z in rails if a == x and y == 81}
            left, right = min(tracks) - 3, max(tracks) + 3
            assert right - left + 1 <= 25
            for z in range(-19, 20):
                position = BlockPos(x, 78, z)
                assert (world.owner_at(position) == name) == (left <= z <= right), (name, x, z)
                state = world.get_block(position)
                assert (state is not None and state.name == "minecraft:smooth_stone") == (left <= z <= right), (x, z)
            for z in (left, right):
                assert world.get_block(BlockPos(x, 81, z)).name == "minecraft:stone_brick_wall"
    for x, left, right in ((110, -8, 8), (140, -6, 6), (199, -6, 6),
                           (280, -6, 6), (320, -18, 6), (400, -12, 12), (479, -12, 12)):
        assert world.get_block(BlockPos(x, 78, left)).name == "minecraft:smooth_stone"
        assert world.get_block(BlockPos(x, 78, right)).name == "minecraft:smooth_stone"


def test_gaps_preserve_separate_runs_and_bound_wires(network):
    spec, world, expected, rails = network
    service = [c for c in spec.components if c.name.startswith("service/")]
    assert [c.name for c in service] == ["service/0", "service/1"]
    assert [(c.build.path.points[0].x, c.build.path.points[-1].x) for c in service] == [(0, 39), (60, 100)]
    assert expected["routes"]["service/0"][-1] == [39, 81, -30]
    assert expected["routes"]["service/1"][0] == [60, 81, -30]
    for x in range(101):
        present = x < 40 or x >= 60
        assert ((x, 81, -30) in rails) == present
        assert (world.get_block(BlockPos(x, 87, -30)) is not None) == present
        if present:
            assert world.owner_at(BlockPos(x, 87, -30)) == f"service/{int(x >= 60)}"
        else:
            assert world.get_block(BlockPos(x, 78, -30)) is None


def test_flyovers_preserve_separate_levels_and_full_edges(network):
    spec, world, expected, rails = network
    by_name = {c.name: c for c in spec.components}
    config = tomllib.loads((ROOT / "composition.toml").read_text())
    for node in config["node"]:
        if "crossing" not in node["name"]:
            continue
        positions = [by_name[branch].build.path.points[int(station)]
                     for reference in node["connections"][0]
                     for branch, station in [reference.rsplit(":", 1)]]
        high, low = positions
        assert abs(high.x - low.x) + abs(high.z - low.z) <= 1
        assert high.y - low.y == 12
        assert (high.x, high.y + 1, high.z) in rails
        assert (low.x, low.y + 1, low.z) in rails
        names = [r.rsplit(":", 1)[0] for r in node["connections"][0]]
        high_route, low_route = [expected["routes"][name] for name in names]
        crossings = [(tuple(a), tuple(b)) for a in high_route for b in low_route
                     if abs(a[0] - high.x) <= 2 and (a[0], a[2]) == (b[0], b[2])]
        assert crossings
        for a, b in crossings:
            assert a[1] - b[1] == 12
            assert not connected(rails, a, b)
    for x, z in ((110, -9), (365, -15)):
        assert (x, 93, z) in rails
        for side in (-3, 3):
            assert world.get_block(BlockPos(x, 93, z + side)).name == "minecraft:stone_brick_wall"
    for x, y in ((140, 86), (150, 82)):
        assert (x, y + 1, -9) in rails
        for z in (-12, -6):
            assert world.get_block(BlockPos(x, y + 1, z)).name == "minecraft:stone_brick_wall"
