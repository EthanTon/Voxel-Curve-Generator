from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"sandbox/flying_junction"), str(ROOT/"sandbox")]
pytest.importorskip("minescript_integration.curve.arthimatic")
from geometry import flying_junction
from minescript_integration.curve.arthimatic import build_expression, cubic_bezier_min_radius, cubic_bezier_max_grade
from shared_bridge import shared_decks, shared_ownership, shared_floors, track_floors


@pytest.mark.parametrize("kind", ["merge", "split"])
@pytest.mark.parametrize("spacing", [6, 16])
@pytest.mark.parametrize("radius,grade", [(300, 0.03), (3000, 0.02), (10000, 0.03)])
def test_flying_junction_limits_and_continuity(kind, radius, grade, spacing):
    routes, metadata = flying_junction(kind, (100, 80, -200), radius=radius, grade=grade, spacing=spacing)
    for segments in routes.values():
        for controls in segments:
            assert np.all(np.diff(controls[:, 0]) > 0)
            if len(controls) == 4:
                expression = build_expression(controls)
                assert cubic_bezier_min_radius(expression) >= radius
                assert cubic_bezier_max_grade(expression) <= grade+1e-9
        for first, second in zip(segments[:-1], segments[1:]):
            np.testing.assert_allclose(first[-1], second[0])
            incoming, outgoing = first[-1]-first[-2], second[1]-second[0]
            np.testing.assert_allclose(incoming/np.linalg.norm(incoming), outgoing/np.linalg.norm(outgoing), atol=1e-10)
    entrance = {tuple(segments[0][0]) for segments in routes.values()}
    exit = {tuple(segments[-1][-1]) for segments in routes.values()}
    assert len(entrance if kind == "merge" else exit) == 4
    common = [routes[f"a{index}"][-1][-1] if kind == "merge" else routes[f"a{index}"][0][0] for index in (1, 2)]
    assert common[1][2]-common[0][2] == metadata["spacing"]
    assert all(point[1] == 80 for point in common)


@pytest.mark.parametrize("parameters", [{"kind": "crossing"}, {"spacing": 0}, {"pair_offset": 16}, {"radius": 0}, {"grade": 0}])
def test_invalid_junction_dimensions(parameters):
    with pytest.raises(ValueError):
        flying_junction(**parameters)


def paired_paths():
    names = [f"{side}_{pair}{index}" for side in ("merge", "split") for pair in ("a", "b") for index in (1, 2)]
    names.extend(("common_1", "common_2"))
    return {name: {"points": [[x, 80, 100*(number//2)+6*(number % 2)] for x in range(5)],
                   "gradients": [[1, 0, 0]]*5, "lead": 16} for number, name in enumerate(names)}


def test_shared_bridge_separates_at_height_changes_and_centres_piers():
    paths = paired_paths()
    paths["merge_a2"]["points"][2][1] += 1
    labels, supports = shared_decks(paths)
    assert list(labels["merge_a1"]) == ["bridge_shared_left", "bridge_shared_left", "", "bridge_shared_left", "bridge_shared_left"]
    assert list(labels["merge_a2"]) == ["bridge_shared_right", "bridge_shared_right", "", "bridge_shared_right", "bridge_shared_right"]
    stretches = [path["points"] for path in supports.values() if path["members"] == ["merge_a1", "merge_a2"]]
    assert stretches == [[[0, 80, 3], [1, 80, 3]], [[3, 80, 3], [4, 80, 3]]]


def test_shared_bridge_omits_piers_over_lower_track_clearance(tmp_path):
    from scaffolding import scaffold_components
    from src.Core.block.registry import BlockRegistry
    path = {"points": [[0, 94, 0]], "rail_offsets": [[-3, 3]]}
    lower = {"points": [[0, 80, 0]]}
    _, placements = scaffold_components({"support": path}, {"lower": lower}, tmp_path,
                                        BlockRegistry.for_vanilla_26_2(), [])
    assert {placement["kind"] for placement in placements} == {"portal"}


def test_joined_bridge_handles_a_four_track_fan():
    paths = {f"track_{i}": {"points": [[x, 80, z] for x in range(5)]}
             for i, z in enumerate((-9, -3, 3, 9))}
    labels, supports = shared_decks(paths)
    assert labels["track_0"][0] == "bridge_shared_left"
    assert labels["track_1"][0] == labels["track_2"][0] == "bridge_shared_middle"
    assert labels["track_3"][0] == "bridge_shared_right"
    assert list(supports.values())[0]["rail_offsets"][0] == [-9, -3, 3, 9]


def test_joined_ownership_reserves_air_as_well_as_material():
    from src.Extensions.Composition.ownership import Ownership
    from src.Extensions.Composition.parser import read_ownership
    from src.Core.coordinates import BlockPos
    support = {"points": [[10, 80, 0], [11, 80, 0]], "rail_offsets": [[-3, 3]]*2}
    rules = shared_ownership({"shared": support}, {"single": {}})
    ownership = Ownership(tuple(read_ownership(rule) for rule in rules))
    assert not ownership.allows("single", "bed", BlockPos(10, 79, 2))
    assert not ownership.allows("single", "pole", BlockPos(10, 85, 0))
    assert ownership.allows("single", "rail", BlockPos(10, 81, 0))
    assert ownership.allows("single", "bed", BlockPos(10, 93, 0))


def test_shared_floor_matches_the_changing_rail_span():
    from src.Extensions.Composition.regions import generate_region
    from src.Extensions.Composition.parser import read_region
    from src.Core.block.registry import BlockRegistry
    support = {"points": [[0, 80, 0], [1, 80, 1], [2, 80, 2]], "rail_offsets": [[-3, 3], [-4, 2, 5], [-5, 1, 6]]}
    field = generate_region(read_region(shared_floors({"shared": support})[0]), BlockRegistry().palette)
    actual = {(p.x, p.y, p.z) for p, _ in field.iter_block_ids()}
    expected = {(x, y-1, z+offset) for (x, y, z), rails in zip(support["points"], support["rail_offsets"])
                for offset in range(min(rails)-3, max(rails)+4)}
    assert actual == expected


def test_track_bed_underlay_closes_gaps_and_preserves_the_centre_recess():
    from src.Extensions.Composition.regions import generate_region
    from src.Extensions.Composition.parser import read_region
    from src.Core.block.registry import BlockRegistry
    paths = {"left": {"points": [[0, 80, -3], [1, 80, -2], [2, 81, -2]]},
             "right": {"points": [[0, 80, 3], [1, 80, 4], [2, 81, 4]]}}
    cells = {tuple(p) for region in track_floors(paths)
             for p, _ in generate_region(read_region(region), BlockRegistry().palette).iter_block_ids()}
    assert (0, 80, 0) not in cells
    assert (1, 80, 1) not in cells
    assert all((x, y, z+offset) in cells for path in paths.values() for x, y, z in path["points"] for offset in range(-2, 3))


@pytest.mark.parametrize("members,kind", [(["merge_a1", "merge_a2"], "poles"), (["merge_a1", "merge_b1"], "portal")])
def test_shared_pair_uses_individual_poles_and_joining_branches_use_a_portal(tmp_path, members, kind):
    from scaffolding import scaffold_components
    from src.Core.block.registry import BlockRegistry
    support = {"points": [[0, 80, 0]], "rail_offsets": [[-3, 3]], "members": members}
    _, placements = scaffold_components({"shared": support}, {}, tmp_path, BlockRegistry.for_vanilla_26_2(), [])
    assert {placement["kind"] for placement in placements} == {"pier", kind}


@pytest.mark.parametrize("angle", [-22.5, 0, 22.5])
def test_native_shared_poles_keep_mrt_variants_and_orientation_at_a_taller_wire(tmp_path, angle):
    from assets import prepare_assets
    from scaffolding import scaffold_components
    from generate import DOMAINS
    from src.Assembly.assemble import build
    from src.Assembly.structure import authored_field
    from src.Core.block.registry import BlockRegistry
    from src.Parser import parse
    prepare_assets(tmp_path/"schems", wire_height=8)
    (tmp_path/"components").mkdir()
    support = {"points": [[0, 80, 0]], "rail_offsets": [[-3, 3]], "members": ["merge_a1", "merge_a2"],
               "gradients": [[1, 0, float(np.tan(np.radians(angle)))]]}
    components, placements = scaffold_components({"shared": support}, {}, tmp_path, BlockRegistry.for_vanilla_26_2(), DOMAINS, wire_height=8)
    placement = next(p for p in placements if p["kind"] == "poles")
    world = build(parse(components[placement["name"]], tmp_path/"components")).world
    original = authored_field(tmp_path/"schems"/f"{placement['name']}.schem", world.registry, "poles", 0)
    assert placement["angle"] == angle
    assert original.bounds.maximum.y == 13
    assert all(world.get_block(p+(0, 80, 0)) == original.palette.get_state(i) for p, i in original.iter_block_ids())


def test_scalable_pier_preserves_cap_and_changes_width_and_depth():
    from scaffolding import scalable_pier
    from src.Core.block.registry import BlockRegistry
    registry = BlockRegistry.for_vanilla_26_2()
    narrow = scalable_pier(registry)
    wide = scalable_pier(registry, depth=40, column_width=5, rail_span=18)
    assert narrow.bounds.minimum.y == -28
    assert wide.bounds.minimum.y == -40
    assert narrow.bounds.maximum.z == 4
    assert wide.bounds.maximum.z == 10
    stem = [p for p, block_id in wide.iter_block_ids() if p.x == 0 and p.y == -40
            and wide.palette.get_state(block_id).name == "minecraft:bone_block"]
    assert len(stem) == 5


def test_scalable_portal_spans_four_tracks_and_moves_hangers_with_wire_height():
    from scaffolding import scalable_portal
    from src.Core.block.registry import BlockRegistry
    from src.Core.coordinates import BlockPos
    portal = scalable_portal(BlockRegistry.for_vanilla_26_2(), [-9, -3, 3, 9], wire_height=8)
    assert portal.bounds.minimum.z == -12
    assert portal.bounds.maximum.z == 12
    for z in (-9, -3, 3, 9):
        assert portal.get_block(BlockPos(0, 10, z)).name == "minecraft:white_stained_glass_pane"
        assert portal.get_block(BlockPos(0, 12, z)).name == "minecraft:iron_trapdoor"


@pytest.mark.parametrize("rails", [[-3, 3], [-3, 2], [-5, 1, 4], [-9, -3, 3, 9]])
def test_native_assembly_keeps_complete_portal_and_pier_at_one_anchor(tmp_path, rails):
    from assets import prepare_assets
    from scaffolding import scaffold_components
    from generate import DOMAINS
    from src.Assembly.assemble import build
    from src.Assembly.structure import authored_field
    from src.Parser import parse
    from src.Core.block.registry import BlockRegistry
    from src.Core.coordinates import BlockPos
    prepare_assets(tmp_path/"schems")
    (tmp_path/"components").mkdir()
    support = {"points": [[0, 80, 0]], "rail_offsets": [rails]}
    components, _ = scaffold_components({"shared": support}, {}, tmp_path, BlockRegistry.for_vanilla_26_2(), DOMAINS)
    for component in components.values():
        world = build(parse(component, tmp_path/"components")).world
        asset = tmp_path/"components"/component["structure"][0]["headings"]["0.0"]
        original = authored_field(asset, world.registry, "scaffold", 0)
        assert all(world.get_block(position+(0, 80, 0)).name == original.palette.get_state(block_id).name
                   for position, block_id in original.iter_block_ids())
        if component["structure"][0]["domain"] == "portal":
            assert world.get_block(BlockPos(0, 90, 0)).name == "minecraft:iron_trapdoor"
            for rail in rails:
                assert world.get_block(BlockPos(0, 88, rail)).name == "minecraft:white_stained_glass_pane"
            assert world.get_block(BlockPos(0, 81, max(rails)+3)).name == "minecraft:andesite_wall"
        else:
            assert world.get_block(BlockPos(0, 52, 0)).name == "minecraft:bone_block"
            assert world.get_block(BlockPos(0, 79, 4)).name == "minecraft:andesite"


@pytest.mark.parametrize("wire_height", [6, 8])
@pytest.mark.parametrize("profile", ["bridge", "tunnel"])
def test_native_profiles_keep_headroom_and_wire_connections_when_height_changes(tmp_path, wire_height, profile):
    from assets import prepare_assets
    from generate import component_build, write_toml
    from src.Assembly.assemble import build
    from src.Parser import load
    from src.Core.coordinates import BlockPos
    prepare_assets(tmp_path/"schems", wire_height=wire_height)
    write_toml(tmp_path/"paths/track.toml", {"points": [[x, 80, 0] for x in range(40)], "gradients": [[1, 0, 0]]*40, "lead": 16})
    component_build("track", [{"stations": [0, 39], "sections": [profile]}], tmp_path/"components", wire_height)
    world = build(load(tmp_path/"components/track.toml")).world
    assert world.get_block(BlockPos(20, 81, 0)).name == "minecraft:rail"
    assert all(world.get_block(BlockPos(20, 81+height, 0)) is None for height in range(1, wire_height))
    assert world.get_block(BlockPos(20, 81+wire_height, 0)).name == "minecraft:iron_bars"
    if profile == "tunnel":
        assert world.get_block(BlockPos(20, 82+wire_height, 0)).name == "minecraft:white_concrete"
    else:
        assert world.get_block(BlockPos(0, 81+wire_height, 0)).name == "minecraft:lever"
        assert world.get_block(BlockPos(1, 81+wire_height, 0)).get("west") == "true"


def test_curved_bridge_cantilevers_do_not_interrupt_contact_wires(tmp_path):
    from assets import prepare_assets
    from generate import component_build, write_toml
    from src.Assembly.assemble import build
    from src.Parser import load
    prepare_assets(tmp_path/"schems")
    write_toml(tmp_path/"paths/track.toml", {"points": [[x, 80, round(0.3*x)] for x in range(40)],
                                          "gradients": [[1, 0, 0.3]]*40, "lead": 16})
    component_build("track", [{"stations": [0, 39], "sections": ["bridge"]}], tmp_path/"components")
    world = build(load(tmp_path/"components/track.toml")).world
    rails = [p for _, cells in world.iter_resolved_chunks() for p, block_id in cells.items()
             if world.registry.palette.get_state(block_id).name == "minecraft:rail"]
    for rail in rails:
        contact = world.get_block(rail+(0, 6, 0))
        assert contact.name == "minecraft:iron_bars" or (contact.name == "minecraft:lever" and contact.get("face") == "floor")


def test_shared_bridge_halves_reproduce_original_double_track_profile(tmp_path):
    from assets import EMPTY, SANDBOX, prepare_assets
    from src.Assembly.structure import authored_field
    from src.Core.block.registry import BlockRegistry
    from src.Core.coordinates import BlockPos
    source = SANDBOX/"terrain_testing/schems/bridge_thick.schem"
    if not source.exists():
        pytest.skip("Requires the existing sandbox MRT schematic assets.")
    prepare_assets(tmp_path)
    registry = BlockRegistry.for_vanilla_26_2()
    original = authored_field(source, registry, "original", 0)
    expected = {position: original.palette.get_state(block_id) for position, block_id in original.iter_block_ids()
                if original.palette.get_state(block_id).name not in EMPTY
                and not original.palette.get_state(block_id).name.endswith(":rail")}
    combined = {}
    for side, shift in (("left", -3), ("right", 3)):
        half = authored_field(tmp_path/f"bridge_{side}.schem", registry, side, 0)
        combined.update({BlockPos(position.x+shift, position.y, position.z): half.palette.get_state(block_id)
                         for position, block_id in half.iter_block_ids()})
    assert combined == expected


def test_four_track_departure_keeps_four_routes_and_curve_limits():
    from geometry import four_track_departure
    routes, _ = flying_junction("split", spacing=6)
    extended, span = four_track_departure(routes, 6, 300, 240)
    assert span[1]-span[0] == 240
    assert sorted(segments[-1][-1][2] for segments in extended.values()) == [-9, -3, 3, 9]
    for name, segments in extended.items():
        for first, second in zip(segments[:-1], segments[1:]):
            np.testing.assert_allclose(first[-1], second[0])
        for controls in segments[len(routes[name]):]:
            if len(controls) == 4:
                assert cubic_bezier_min_radius(build_expression(controls)) >= 300


def test_shared_owner_cannot_cut_a_neighbouring_higher_single():
    from src.Extensions.Composition.ownership import Ownership
    from src.Extensions.Composition.parser import read_ownership
    from src.Core.coordinates import BlockPos
    support = {"points": [[0, 80, 0], [1, 80, 1]], "rail_offsets": [[-3, 3]]*2, "members": ["a1", "b1"]}
    rules = shared_ownership({"shared": support}, {"a1": {}, "b1": {}, "upper": {}})
    ownership = Ownership(tuple(read_ownership(rule) for rule in rules))
    assert ownership.allows("upper", "bed", BlockPos(0, 85, 3))
    assert ownership.allows("a1", "bed", BlockPos(0, 80, 7))
    assert not ownership.allows("a1", "bed", BlockPos(0, 80, 6))


def test_different_height_singles_keep_both_edges_and_inward_poles(tmp_path):
    from assets import prepare_assets
    from generate import component_build, write_toml, corridor_rules
    from domains import DOMAINS
    from shared_bridge import single_pole_paths, bridge_edges, track_anvils
    from scaffolding import scaffold_components
    from src.Extensions.Composition import build, parse
    from src.Core.block.registry import BlockRegistry
    from src.Core.coordinates import BlockPos
    prepare_assets(tmp_path/"schems")
    paths = {name: {"points": [[x, y, z] for x in range(40)], "gradients": [[1, 0, 0]]*40, "lead": 16}
             for name, y, z in (("lower", 80, 0), ("higher", 85, 6))}
    labels, supports = shared_decks(paths)
    assert not supports
    for name, path in paths.items():
        write_toml(tmp_path/f"paths/{name}.toml", path)
        component_build(name, [{"stations": [0, 39], "sections": ["bridge_junction"]}], tmp_path/"components", anchored_poles=True)
    scaffolds, placements = scaffold_components(single_pole_paths(paths, labels), paths, tmp_path,
                                                 BlockRegistry.for_vanilla_26_2(), DOMAINS)
    for name, data in scaffolds.items():
        write_toml(tmp_path/f"components/{name}.toml", data)
    components = list(paths)+list(scaffolds)
    world = build(parse({"composition": {}, "output": {"path": "out.schem"},
        "component": [{"name": name, "build": f"components/{name}.toml"} for name in components],
        "region": bridge_edges(paths)+track_anvils(paths), "ownership": corridor_rules(paths, components)}, tmp_path)).world
    assert len(placements) == 4
    for y, z, side, post_z in ((80, 0, "L", -3), (85, 6, "R", 9)):
        for x in range(40):
            for edge in (z-3, z+3):
                assert world.get_block(BlockPos(x, y+1, edge)) is not None
                if x not in (0, 31):
                    assert world.get_block(BlockPos(x, y+1, edge)).name == "minecraft:andesite_wall"
            for edge in (z-1, z+1):
                assert world.get_block(BlockPos(x, y, edge)).name == "minecraft:anvil"
        for x in (0, 31):
            assert world.get_block(BlockPos(x, y+5, post_z)).name == "minecraft:andesite_wall"
            connector = world.get_block(BlockPos(x, y+7, z))
            assert connector.name == "minecraft:lever" and connector.get("face") == "floor"
            assert world.get_block(BlockPos(x+1, y+7, z)).get("west") == "true"
        assert all(p["side"] == side for p in placements if p["position"][1] == y)


def test_stepped_anvil_and_exposed_edge_regions_have_valid_boundaries():
    from shared_bridge import bridge_edges, track_anvils
    from src.Extensions.Composition.parser import read_region
    paths = {"track": {"points": [[0, 80, 0], [1, 80, 1], [2, 80, 0], [3, 81, 0]]}}
    for region in bridge_edges(paths)+track_anvils(paths):
        read_region(region)


def turnout_routes():
    routes = {}
    for side, origin in (("merge", 0), ("split", 20)):
        for index, z in ((1, -10), (2, 10)):
            routes[f"{side}_a{index}"] = [(x, 81, z) for x in range(origin, origin+9)]
            branch = [(0, 81, z+2), (1, 81, z+2), (1, 81, z+1), (2, 81, z+1),
                      (3, 81, z+1), (3, 81, z), (4, 81, z), (5, 81, z)]
            routes[f"{side}_b{index}"] = branch if side == "merge" else [(25-x, y, lateral) for x, y, lateral in branch[::-1]]
    return routes


@pytest.mark.parametrize("selection", ["main", "branch"])
def test_rail_shapes_preserve_routes_beside_other_tracks_and_select_turnouts(selection):
    from rail_routes import route_shapes, rail_connected
    routes = turnout_routes()
    shapes, turnouts = route_shapes(routes, selection)
    inactive = {frozenset(tuple(p) for p in each["inactive_edge"]) for each in turnouts}
    for points in routes.values():
        for first, second in zip(points[:-1], points[1:]):
            assert rail_connected(first, shapes[first], second, shapes[second]) == (frozenset((first, second)) not in inactive)
    assert len(turnouts) == 4
    assert shapes[1, 81, -9] == "south_east"


def test_rail_ports_reject_a_broken_corner_and_an_incorrect_rise():
    from rail_routes import rail_connected, rail_shape
    assert not rail_connected((0, 81, 0), "east_west", (0, 81, 1), "north_east")
    assert rail_connected((0, 81, 0), "ascending_east", (1, 82, 0), "east_west")
    assert not rail_connected((0, 81, 0), "east_west", (1, 82, 0), "east_west")
    with pytest.raises(ValueError, match="turn and ascend"):
        rail_shape((0, 81, 0), [(1, 82, 0), (0, 81, 1)])


def test_bridge_edges_join_diagonal_cells_and_resolve_wall_arms():
    from shared_bridge import bridge_edges
    from domains import DOMAINS
    from src.Extensions.Composition.parser import read_region
    from src.Extensions.Composition.regions import generate_region
    from src.Core.Domain import DomainRegistry
    from src.Core.World.block_world import BlockWorld
    from src.Core.block.registry import BlockRegistry
    from src.Core.coordinates import BlockPos
    from src.Stamping import resolve
    registry, domains = BlockRegistry.for_vanilla_26_2(), DomainRegistry()
    domain = domains.create("edge")
    world = BlockWorld(registry)
    paths = {"track": {"points": [[0, 80, 0], [1, 80, 1], [2, 80, 2], [3, 81, 2]]}}
    cells = set()
    for region in bridge_edges(paths):
        field = generate_region(read_region(region), registry.palette)
        cells.update(p for p, _ in field.iter_block_ids())
        domains.place(world, field, domain)
    resolve(world, registry, domains)
    assert next(each for each in DOMAINS if each["name"] == "edge").get("connectivity", {}).get("connects_to_self", True)
    for p in cells:
        for direction, step in (("east", (1, 0, 0)), ("west", (-1, 0, 0)), ("north", (0, 0, -1)), ("south", (0, 0, 1))):
            if p+step in cells:
                assert world.get_block(p).get(direction) != "none"
    assert BlockPos(1, 81, -3) in cells
    assert BlockPos(0, 81, 4) in cells


def test_bridge_assets_can_delegate_all_parapets_to_explicit_edges(tmp_path):
    from assets import prepare_assets
    from src.Format.schematics import load_blocks
    from src.Core.block.registry import BlockRegistry
    prepare_assets(tmp_path, external_edges=True)
    registry = BlockRegistry.for_vanilla_26_2()
    for name in ("bridge", "bridge_left", "bridge_right", "bridge_middle"):
        field = load_blocks(tmp_path/f"{name}.schem", registry.palette)
        assert not any(field.palette.get_state(block_id).name.endswith("_wall") for _, block_id in field.iter_block_ids())


@pytest.mark.parametrize("selection", ["main", "branch"])
def test_native_rail_patches_cover_sparse_corners_and_crowded_straights(tmp_path, monkeypatch, selection):
    import tomli_w
    import rail_routes
    from assets import prepare_assets
    from domains import DOMAINS
    from src.Extensions.Composition import build, parse
    from src.Core.coordinates import BlockPos
    routes = turnout_routes()
    for name, points in tuple(routes.items()):
        shift = 1536 if name.startswith("merge") else 2000
        routes[name] = [(x+shift, y, z) for x, y, z in points]
    for index, z in ((1, -10), (2, 10)):
        routes[f"merge_a{index}"] = [(x, 81, z) for x in range(1536, 1587)]
        routes[f"merge_b{index}"] = ([(x, 81, z+6) for x in range(1536, 1542)]
            +[(x, 81, z+5) for x in range(1541, 1558)]
            +[(1557, 81, z+offset) for offset in (4, 3, 2)]
            +[(x, 81, z+1) for x in range(1557, 1578)]
            +[(x, 81, z) for x in range(1577, 1587)])
    monkeypatch.setattr(rail_routes, "lane_positions", lambda file: (routes[file.stem], set(routes[file.stem])))
    directory = tmp_path/"bridge"
    prepare_assets(directory/"schems")
    components, _, _ = rail_routes.route_components(routes, directory, DOMAINS, selection)
    (directory/"components").mkdir()
    for name, component in components.items():
        (directory/"components"/f"{name}.toml").write_text(tomli_w.dumps(component))
    cells = set.union(*(set(points) for points in routes.values()))
    scene = {"composition": {}, "output": {"path": "out.schem"},
             "component": [{"name": name, "build": f"components/{name}.toml"} for name in components],
             "region": [{"name": f"rail_{i}", "domain": "rail", "block": "minecraft:rail[shape=east_west,waterlogged=false]",
                         "boundary": [[x, z], [x+1, z], [x+1, z+1], [x, z+1]], "top": y}
                        for i, (x, y, z) in enumerate(sorted(cells))]}
    world = build(parse(scene, directory)).world
    expected, _ = rail_routes.route_shapes(routes, selection)
    for position, shape in expected.items():
        assert world.get_block(BlockPos(*position)).get("shape") == shape
    assert world.get_block(BlockPos(1558, 81, -10)).get("shape") == "east_west"
    assert world.get_block(BlockPos(1558, 81, -9)).get("shape") == "east_west"


def test_portal_posts_connect_to_the_adjacent_parapet(tmp_path):
    from assets import prepare_assets
    from scaffolding import scaffold_components
    from shared_bridge import shared_decks, bridge_edges
    from domains import DOMAINS
    from generate import write_toml
    from src.Core.block.registry import BlockRegistry
    from src.Core.coordinates import BlockPos
    from src.Extensions.Composition import build, parse
    prepare_assets(tmp_path/"schems", external_edges=True)
    paths = {name: {"points": [[x, 80, z] for x in range(40)]}
             for name, z in zip(("merge_a1", "merge_a2", "merge_b1", "merge_b2"), (-9, -3, 3, 9))}
    _, supports = shared_decks(paths)
    components, _ = scaffold_components(supports, paths, tmp_path, BlockRegistry.for_vanilla_26_2(), DOMAINS)
    for name, component in components.items():
        write_toml(tmp_path/"components"/f"{name}.toml", component)
    scene = {"composition": {}, "output": {"path": "out.schem"},
             "component": [{"name": name, "build": f"components/{name}.toml"} for name in components], "region": bridge_edges(paths)}
    world = build(parse(scene, tmp_path)).world
    for x in (30, 31, 32):
        for y in (80, 81):
            for z in (-12, 12):
                state = world.get_block(BlockPos(x, y, z))
                assert state.name == "minecraft:andesite_wall"
                assert state.get("east") != "none"
                assert state.get("west") != "none"
    assert world.get_block(BlockPos(31, 81, -12)).get("up") == "true"
