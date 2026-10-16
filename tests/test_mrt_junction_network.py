import json
import pytest

from sandbox.mrt_junction_network.generate import EXAMPLE, ROOT, generate, variant
from src.Assembly.export import write_outputs
from src.Core.block.info import SolidFace
from src.Core.coordinates import BlockPos
from src.Extensions.Composition import build, compose
from src.Format.schematics import load_schematic
from tests.test_junction_network import connected


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp("mrt_junction_network")
    generate(root)
    return root


@pytest.fixture(scope="module", params=("main", "branch"))
def network(request, library):
    declaration = variant(request.param, library)
    spec = compose(declaration)
    assembled = build(spec)
    world = assembled.world
    write_outputs(world, declaration, assembled.terraforming, origin=spec.origin)
    expected = json.loads((EXAMPLE / "expectations.json").read_text())
    expected.update(expected["variants"][request.param])
    expected["directory"] = declaration.output.path.parent
    expected["library"] = library
    rails = {tuple(p): world.get_block(p) for p in world.iter_source_positions()
             if world.get_block(p).name == "minecraft:rail"}
    return spec, world, expected, rails


def test_mrt_assets_preserve_authored_details(network):
    _, world, expected, _ = network
    original, _ = load_schematic(ROOT.parent / "terrain_testing/schems/bridge_thick.schem")
    fitted, _ = load_schematic(expected["library"] / "assets/pair.schem")
    for z in range(-6, 7):
        assert fitted.state_at((z, -1, 0)) == original.state_at((z, -1, 0))
        assert world.get_block(BlockPos(230, 79, z)).name == original.state_at((z, -1, 0)).name
    for centre in (-3, 3):
        assert [world.get_block(BlockPos(230, 80, centre + offset)).name for offset in range(-2, 3)] == [
            "minecraft:gray_wool", "minecraft:anvil", "minecraft:gray_wool", "minecraft:anvil", "minecraft:gray_wool"]
        for side in (-1, 1):
            assert world.get_block(BlockPos(230, 80, centre + side)).get("facing") in ("east", "west")
    mast, _ = load_schematic(ROOT.parent / "mrtsurfacemap_bpath/schems/CRThickBridgePoleL0.schem")
    for position, state in mast.iter_occupied():
        if state.name != "minecraft:air":
            assert world.get_block(position + (0, 80, -30)) == state
    assert world.get_block(BlockPos(230, 80, 0)) is None
    names = {world.get_block(p).name for p in world.iter_source_positions()}
    assert {"minecraft:andesite_slab", "minecraft:smooth_stone_slab", "minecraft:andesite_wall",
            "minecraft:bone_block", "minecraft:white_stained_glass_pane", "minecraft:pale_oak_fence_gate",
            "minecraft:lever", "minecraft:snow", "minecraft:iron_trapdoor"} <= names
    assert "minecraft:stone_brick_wall" not in names


def test_mrt_rails_and_switch_selections(network):
    spec, _, expected, rails = network
    assert len(spec.components) == 24
    assert len(spec.connections) == 20
    assert set(rails) == {tuple(p) for route in expected["routes"].values() for p in route}
    inactive = {frozenset(map(tuple, t["inactive_edge"])) for t in expected["turnouts"]}
    assert len(inactive) == 8
    for route in expected["routes"].values():
        for first, second in zip(route, route[1:]):
            a, b = tuple(first), tuple(second)
            assert connected(rails, a, b) == (frozenset((a, b)) not in inactive), (a, b)


def test_mrt_rail_support_clearance_and_contact(network):
    _, world, _, rails = network
    for x, y, z in rails:
        below = world.get_block(BlockPos(x, y - 1, z))
        assert below is not None
        assert world.registry.get_info_for_state(below).solid_faces & SolidFace.UP
        contact = world.get_block(BlockPos(x, y + 6, z))
        assert contact is not None and (contact.name == "minecraft:iron_bars" or
               (contact.name == "minecraft:lever" and contact.get("face") == "floor")), (x, y, z, contact)
        for dy in range(1, 5):
            above = world.get_block(BlockPos(x, y + dy, z))
            assert above is None or above.name == "minecraft:air", (x, y, z, dy, above)


def test_mrt_decks_follow_transition_widths_and_keep_flyover_edges(network):
    _, world, expected, rails = network
    for samples in expected["decks"].values():
        for x, left, right in samples[:-1]:
            for z in range(left, right + 1):
                state = world.get_block(BlockPos(x, 79, z))
                assert state is not None and world.registry.get_info_for_state(state).solid_faces & SolidFace.UP, (x, z, state)
    for x, half in ((240, 6), (500, 12), (600, 18), (700, 12)):
        for z in range(-half, half + 1):
            state = world.get_block(BlockPos(x, 79, z))
            assert state is not None and world.registry.get_info_for_state(state).solid_faces & SolidFace.UP
        for z in (-half, half):
            state = world.get_block(BlockPos(x, 81, z))
            assert state.name == "minecraft:andesite_wall" or (
                state.name == "minecraft:smooth_stone_slab" and state.get("type") == "double"), (x, z, state)
    for x in range(40, 60):
        for y in (79, 80, 81, 87):
            assert world.get_block(BlockPos(x, y, -30)) is None


def test_saved_mrt_schematics_and_piece_budgets(network):
    _, _, expected, _ = network
    directory = expected["directory"]
    region, _ = load_schematic(directory / "mrt_junction_network.schem")
    cells = {tuple(p): s for p, s in region.iter_occupied() if s.name != "minecraft:air"}
    assert sum(s.name == "minecraft:rail" for s in cells.values()) == 3101
    inactive = {frozenset(map(tuple, t["inactive_edge"])) for t in expected["turnouts"]}
    for route in expected["routes"].values():
        for first, second in zip(route, route[1:]):
            a, b = tuple(first), tuple(second)
            assert connected(cells, a, b) == (frozenset((a, b)) not in inactive)
    pieces = sorted((directory / "mrt_junction_network").glob("*.schem"))
    assert pieces
    for file in pieces:
        piece, _ = load_schematic(file)
        x, y, z = piece.size
        assert max(x, y, z) <= 256
        assert x * y * z <= 5_000_000


def pieces(points, diagonal=False):
    remaining, result = set(points), []
    while remaining:
        queue, count = [remaining.pop()], 0
        while queue:
            p = queue.pop()
            count += 1
            for step in ((dx, 0, dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)
                         if (dx or dz) and (diagonal or abs(dx) + abs(dz) == 1)):
                q = p + step
                if q in remaining:
                    remaining.remove(q)
                    queue.append(q)
        result.append(count)
    return sorted(result, reverse=True)


def test_mrt_track_strips_and_walls_remain_continuous_on_curves(network):
    _, world, _, _ = network
    cells = {p: world.get_block(p) for p in world.iter_source_positions()}
    for first, last in ((50, 110), (355, 415)):
        wool = {p for p, state in cells.items() if first <= p.x <= last and p.y == 92 and state.name == "minecraft:gray_wool"}
        assert len(pieces(wool)) == 3, (first, pieces(wool))
        for y, name in ((92, "minecraft:anvil"), (93, "minecraft:andesite_wall")):
            points = {p for p, state in cells.items() if first <= p.x <= last and p.y == y and state.name == name}
            if name == "minecraft:andesite_wall":
                points |= {p for p, state in cells.items() if first <= p.x <= last and p.y == y
                           and world.registry.get_info_for_state(state).solid_faces == SolidFace.ALL}
            runs = pieces(points, diagonal=True)
            assert len(runs) == 2, (first, y, runs)
            assert len(pieces(points)) > 2, (first, y)


def test_mrt_wool_has_native_manhattan_corner_positions(network):
    _, world, _, _ = network
    expected = {63: {6, 8, 9, 11}, 64: {6, 8, 10, 11}, 65: {6, 8, 10}, 66: {5, 6, 8, 10}}
    for x, columns in expected.items():
        actual = {z for z in range(0, 15) if (state := world.get_block(BlockPos(x, 92, z))) is not None
                  and state.name == "minecraft:gray_wool"}
        assert actual == columns, (x, actual)


def test_mrt_formation_preserves_flanks_and_supports_every_wall(network):
    _, world, _, rails = network
    for x, centres in ((230, (-3, 3)), (500, (-9, -3, 3, 9)), (600, (-15, -9, -3, 3, 9, 15))):
        for z in centres:
            for side in (-1, 1):
                assert world.get_block(BlockPos(x, 80, z + side)).name == "minecraft:anvil"
    for p in world.iter_source_positions():
        state = world.get_block(p)
        if state.name == "minecraft:andesite_wall":
            below = world.get_block(p + (0, -1, 0))
            assert below is not None and (below.name == state.name or
                world.registry.get_info_for_state(below).solid_faces & SolidFace.UP), (p, below)
            if below.name != state.name and world.source_at(p).domain == world.domains.id_of("formation"):
                neighbors = [world.get_block(p + step) for step in ((-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1))]
                assert any(neighbor is not None and (neighbor.name == "minecraft:gray_wool" or
                           world.registry.get_info_for_state(neighbor).solid_faces == SolidFace.ALL)
                           for neighbor in neighbors), p


def test_mrt_fittings_are_complete_and_corner_foundations_stay_near_rails(network):
    spec, world, _, rails = network
    domains = {domain.name: world.domains.id_of(domain.name) for domain in spec.components[0].build.domains}
    groups = {}
    for placement in world.instances:
        if placement.connectivity_domain in {domains[name] for name in ("pole", "connector", "support")}:
            groups.setdefault((world.owners.get(placement.placement_id), placement.transform.origin), set()).add(placement.placement_id)
    for ids in groups.values():
        for placement_id in ids:
            placement = world.get_placement(placement_id)
            for local, block_id in placement.field.iter_block_ids():
                if placement.field.palette.get_state(block_id).name == "minecraft:air":
                    continue
                p = placement.transform.apply(local)
                assert placement.occupies(p)
                assert world.source_at(p).placement_id in ids, (p, placement_id)
    for p in world.iter_source_positions():
        if world.source_at(p).domain == domains["formation"] and world.get_block(p).name.endswith("_slab"):
            assert any((p.x + dx, p.y + 2, p.z + dz) in rails for dx in range(-3, 4) for dz in range(-3, 4)), p
