from types import SimpleNamespace

import pytest

from src.Assembly.fittings import retain_assemblies
from src.Assembly.formation_paths import formation_paths, formation_section
from src.Assembly.assemble import sweep
from src.Assembly.structure import build_path, build_path_frames, build_schedules
from src.Core.CrossSection.annotations import FeatureAnnotation
from src.Core.CrossSection.cross_section import CrossSection, PointField
from src.Core.Domain import DomainRegistry
from src.Core.Field.block_field import BlockField
from src.Core.block.registry import BlockRegistry
from src.Core.coordinates import BlockPos, Vec3
from src.Core.modes import StepMode
from src.Core.Feature.feature import PointFeature
from src.Extensions.Composition.formations import lay_formation
from src.Extensions.Composition.ownership import OwnedWorld
from src.Extensions.Composition.spec import FormationRule
from src.Format.schematics import parse_state_key
from src.Parser.formations import read_formation
from src.Generation.Geometry.frames import level_near
from src.Parser.spec import LaneSpec, PathSpec, StructureSpec


DEFINITION = dict(name="track", domain="formation", rail_domains=["rail"], replace_domains=["bed"],
    foundation_blocks=["minecraft:stone_slab"], support="minecraft:gray_wool", flank="minecraft:anvil[facing=north]",
    shoulder="minecraft:gray_wool", edge="minecraft:andesite_wall")


def traces(*routes):
    combined = {role: set() for role in ("flanks", "shoulders", "edges")}
    for route in routes:
        route = list(dict.fromkeys(route))
        forward = Vec3(*(BlockPos(*route[-1]) - BlockPos(*route[0])))
        spec = SimpleNamespace(path=PathSpec(tuple(BlockPos(*p) + (0, -1, 0) for p in route),
                                            (forward,) * len(route), lead=8),
            formations=(read_formation(DEFINITION),),
            lanes=(LaneSpec("rail", "rail", ("track",), 1, 0, offset=BlockPos(0, 1, 0)),))
        path = build_path(spec)
        feature = PointFeature(BlockField(BlockRegistry.for_vanilla_26_2().palette), None, None)
        section = CrossSection(PointField(FeatureAnnotation("track" if x == 0 else str(x),
            BlockPos(x, 0, 0), feature, StepMode.CHEBYSHEV, section="track") for x in range(-3, 4)))
        sections = {"track": section}
        spans = {"track": ((path.first_index, path.last_index),)}
        schedules = build_schedules(sections, spans, len(path.path_points))
        frames = build_path_frames(spec, schedules, path)
        strips = formation_paths(spec, sweep(spec, schedules, path, frames), sections, level_near(frames), path)["track"]
        for role in combined:
            combined[role].update(strips[role])
    return combined


def test_wool_steps_push_the_chebyshev_columns_outside_them():
    feature = PointFeature(BlockField(), None, None)
    original = CrossSection(PointField((FeatureAnnotation("track", BlockPos(0, 0, 0), feature,
                                                         StepMode.CHEBYSHEV),)))
    lane = LaneSpec("rail", "rail", ("track",), 1, 0, offset=BlockPos(0, 1, 0))
    section = formation_section(original, read_formation(DEFINITION), lane)
    assert section.along(-2, StepMode.MANHATTAN) == (1, 2)
    assert section.along(2, StepMode.MANHATTAN) == (-1, 0)
    assert section.along(-1, StepMode.CHEBYSHEV) == (1,)
    assert section.along(1, StepMode.CHEBYSHEV) == (0,)
    assert section.along(-3, StepMode.CHEBYSHEV) == (2,)
    assert section.along(3, StepMode.CHEBYSHEV) == (-1,)


def test_native_manhattan_corner_positions_are_not_square_padding(scene):
    world, domains, place = scene
    points = [(x, 1, x // 3) for x in range(12)]
    laid = traces(points)
    expected = {
        "flanks": {(2, -1), (2, 1), (3, -1), (3, 2), (4, 0), (4, 2), (5, 0), (5, 2)},
        "shoulders": {(2, -2), (2, 2), (2, 3), (3, -2), (3, 3), (4, -2), (4, -1), (4, 3),
                      (5, -1), (5, 3), (5, 4)},
        "edges": {(2, -3), (2, 4), (3, -3), (3, 4), (4, -3), (4, 4), (5, -2), (5, 5)},
    }
    for role, cells in expected.items():
        assert {(p.x, p.z) for p in laid[role] if 2 <= p.x <= 5} == cells
    place("track", "bed", [(x, -1, z) for x in range(12) for z in range(-6, 10)], "minecraft:stone_slab[type=top]")
    place("track", "rail", points + [(x, 1, x // 3 - 1) for x in (3, 6, 9)], "minecraft:rail[shape=east_west]")
    lay_formation(world, domains, FormationRule(read_formation(DEFINITION), ("track",)), laid)
    for role, name in (("shoulders", "gray_wool"), ("flanks", "anvil"), ("edges", "andesite_wall")):
        for x, z in expected[role]:
            assert world.get_block(BlockPos(x, 0, z)).name == "minecraft:" + name
    assert world.get_block(BlockPos(4, 0, 4)).name == "minecraft:andesite_wall"
    assert world.get_block(BlockPos(3, 0, 4)).name == "minecraft:andesite_wall"


def test_a_formation_requires_native_paths_instead_of_inventing_a_wool_band(scene):
    world, domains, place = scene
    place("track", "rail", [(0, 1, 0)], "minecraft:rail[shape=east_west]")
    with pytest.raises(ValueError, match="native rail lane paths"):
        lay_formation(world, domains, FormationRule(read_formation(DEFINITION), ("track",)),
                      {role: set() for role in ("flanks", "shoulders", "edges")})


@pytest.fixture
def scene():
    domains = DomainRegistry()
    for name in ("bed", "rail", "pole", "wire", "formation"):
        domains.create(name)
    world = OwnedWorld(BlockRegistry.for_vanilla_26_2(), ())
    domains.bind(world)
    world.domains = domains

    def place(owner, domain, positions, state, **options):
        field = BlockField(world.registry.palette)
        for p in positions:
            field.set_block(BlockPos(*p), parse_state_key(state))
        world.component = owner
        result = world.add_field(field, connectivity_domain=domains.id_of(domain), **options)
        world.component = None
        return result

    return world, domains, place


@pytest.mark.parametrize("reverse", (False, True))
def test_curved_strips_share_the_rail_cells_without_offcuts(scene, reverse):
    world, domains, place = scene
    authored = [(x, 1, x // 3) for x in range(12)]
    points = list(authored)
    points += [(x, 1, x // 3 - 1) for x in (3, 6, 9)]
    if reverse:
        points.reverse()
        authored.reverse()
    floor = {(x, -1, z) for x, _, centre in points for z in range(centre - 3, centre + 4)}
    place("curve", "bed", floor, "minecraft:stone_slab[type=top]")
    place("curve", "bed", [(3, 0, 3)], "minecraft:anvil")
    place("curve", "rail", points, "minecraft:rail[shape=east_west]")
    laid = traces(authored)
    lay_formation(world, domains, FormationRule(read_formation(DEFINITION), ("curve",)), laid)
    bases = {(x, 0, z) for x, _, z in points}
    expected = {tuple(p) for p in laid["flanks"] - laid["shoulders"]} - bases
    expected &= {(x, 0, z) for x, _, z in floor}
    actual = {tuple(p) for p in world.iter_source_positions() if world.get_block(p).name == "minecraft:anvil"}
    assert actual == expected
    for x, _, z in points:
        assert world.get_block(BlockPos(x, 1, z)).name == "minecraft:rail"
        assert world.get_block(BlockPos(x, 0, z)).name == "minecraft:gray_wool"
    for p in laid["shoulders"]:
        if (p.x, -1, p.z) in floor:
            assert world.get_block(p).name == "minecraft:gray_wool"


def test_shared_formation_removes_interior_walls_and_keeps_fittings(scene):
    world, domains, place = scene
    for owner, centre in (("left", 0), ("right", 6)):
        place(owner, "bed", [(x, -1, z) for x in range(11) for z in range(centre - 3, centre + 4)], "minecraft:stone_slab[type=top]")
        place(owner, "bed", [(x, y, centre + 3) for x in range(11) for y in (0, 1)], "minecraft:andesite_wall")
        place(owner, "rail", [(x, 1, centre) for x in range(11)], "minecraft:rail[shape=east_west]")
    place("left", "pole", [(5, 0, -3), (5, 1, -3)], "minecraft:smooth_stone_slab[type=double]")
    lay_formation(world, domains, FormationRule(read_formation(DEFINITION), ("left", "right")),
                  traces(*([(x, 1, z) for x in range(11)] for z in (0, 6))))
    assert world.get_block(BlockPos(5, 1, 3)) is None
    assert world.get_block(BlockPos(5, 0, 3)) is None
    assert world.get_block(BlockPos(5, 1, -3)).name == "minecraft:smooth_stone_slab"
    assert world.get_block(BlockPos(5, 1, 9)).name == "minecraft:andesite_wall"
    assert world.get_block(BlockPos(5, 0, 1)).name == "minecraft:anvil"
    assert world.get_block(BlockPos(5, 0, 5)).name == "minecraft:anvil"


@pytest.mark.parametrize("curve", (False, True))
def test_wall_sits_outside_wool_even_on_a_wider_foundation(scene, curve):
    world, domains, place = scene
    points = [(x, 1, x // 3 if curve else 0) for x in range(12)]
    points += [(x, 1, x // 3 - 1) for x in (3, 6, 9)] if curve else []
    floor = {(x, -1, z) for x, _, centre in points for z in range(centre - 6, centre + 7)}
    place("track", "bed", floor, "minecraft:stone_slab[type=top]")
    place("track", "rail", points, "minecraft:rail[shape=east_west]")
    lay_formation(world, domains, FormationRule(read_formation(DEFINITION), ("track",)), traces(points[:12]))
    walls = {p for p in world.iter_source_positions() if p.y == 0 and world.get_block(p).name == "minecraft:andesite_wall"}
    assert walls
    for p in walls:
        neighbors = [world.get_block(p + step) for step in ((-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1))]
        assert any(state is not None and state.name == "minecraft:gray_wool" for state in neighbors), p
    if not curve:
        assert world.get_block(BlockPos(5, 0, 3)).name == "minecraft:andesite_wall"
        assert world.get_block(BlockPos(5, 0, 6)) is None


def test_flying_routes_and_scoped_changes_stay_separate(scene):
    world, domains, place = scene
    for owner, height in (("lower", 0), ("upper", 12)):
        place(owner, "bed", [(x, height - 1, z) for x in range(11) for z in range(-3, 4)], "minecraft:stone_slab[type=top]")
        place(owner, "rail", [(x, height + 1, 0) for x in range(11)], "minecraft:rail[shape=east_west]")
    rule = FormationRule(read_formation(DEFINITION), ("lower", "upper"), ((2, -4), (9, -4), (9, 4), (2, 4)), (0, 13))
    lay_formation(world, domains, rule, traces(*([(x, y, 0) for x in range(11)] for y in (1, 13))))
    for y in (1, 13):
        assert world.get_block(BlockPos(5, y, -3)).name == "minecraft:andesite_wall"
        assert world.get_block(BlockPos(5, y, 3)).name == "minecraft:andesite_wall"
    assert world.get_block(BlockPos(0, 0, 1)) is None


def test_mast_leg_keeps_a_foundation_when_the_wall_moves_inwards(scene):
    world, domains, place = scene
    place("track", "bed", [(x, -1, z) for x in range(11) for z in range(-6, 7)], "minecraft:stone_slab[type=top]")
    place("track", "bed", [(5, 0, 6)], "minecraft:andesite_wall")
    place("track", "rail", [(x, 1, 0) for x in range(11)], "minecraft:rail[shape=east_west]")
    place("track", "pole", [(5, 1, 6)], "minecraft:andesite_wall")
    lay_formation(world, domains, FormationRule(read_formation(DEFINITION), ("track",)),
                  traces([(x, 1, 0) for x in range(11)]))
    assert world.get_block(BlockPos(5, 0, 3)).name == "minecraft:andesite_wall"
    assert world.get_block(BlockPos(5, 0, 6)).name == "minecraft:stone_slab"
    assert world.get_block(BlockPos(5, 1, 6)).name == "minecraft:andesite_wall"


def test_sloping_formation_keeps_its_wall_foundations_and_track_clearance(scene):
    world, domains, place = scene
    points = [(x, 1 + x // 5, 0) for x in range(19)]
    place("slope", "bed", [(x, y - 2, z) for x, y, _ in points for z in range(-3, 4)], "minecraft:stone_slab[type=top]")
    place("slope", "rail", points, "minecraft:rail[shape=ascending_east]")
    lay_formation(world, domains, FormationRule(read_formation(DEFINITION), ("slope",)), traces(points))
    for x, y, _ in points:
        assert world.get_block(BlockPos(x, y - 1, 1)).name == "minecraft:anvil"
        for dy in range(1, 5):
            assert world.get_block(BlockPos(x, y + dy, 0)) is None
    for p in world.iter_source_positions():
        if world.get_block(p).name == "minecraft:andesite_wall":
            assert world.get_block(p + (0, -1, 0)).name in ("minecraft:andesite_wall", "minecraft:stone_slab")


@pytest.mark.parametrize("damage", ("clip", "collision", "none"))
def test_complete_fittings_are_kept_or_removed_together(scene, damage):
    world, _, place = scene
    mask = {"surface_mask": lambda p: p.y < 2} if damage == "clip" else {}
    body = place("curve", "pole", [(2, y, 2) for y in range(3)], "minecraft:andesite_wall", **mask)
    connector = place("curve", "pole", [(2, 2, 2)], "minecraft:lever[face=floor,facing=north,powered=false]")
    if damage == "collision":
        place("curve", "wire", [(2, 1, 2)], "minecraft:iron_bars")
    structures = tuple(StructureSpec(name, "pole", (), {}, assembly="mast") for name in ("body", "connector"))
    fittings = {"body": [body], "connector": [connector]}
    retain_assemblies(SimpleNamespace(components=(SimpleNamespace(world=world, spec=SimpleNamespace(structures=structures)),),
                                     fitting_ids=(fittings,)))
    if damage == "none":
        assert fittings == {"body": [body], "connector": [connector]}
        assert world.get_block(BlockPos(2, 0, 2)).name == "minecraft:andesite_wall"
    else:
        assert fittings == {"body": [], "connector": []}
        assert world.get_block(BlockPos(2, 0, 2)) is None
        assert world.get_block(BlockPos(2, 2, 2)) is None


@pytest.mark.parametrize("height", (0, 12))
def test_flying_foundations_reject_foreign_masts_but_keep_their_own_footings(scene, height):
    world, domains, place = scene
    place("upper", "rail", [(2, 13, 0)], "minecraft:rail[shape=east_west]")
    mast = place("mast", "pole", [(2, y, 3) for y in range(-1, 12)], "minecraft:andesite_wall",
                 offset=BlockPos(0, height, 0))
    structure = StructureSpec("mast", "pole", (), {}, assembly="mast")
    fittings = {"mast": [mast]}
    component = SimpleNamespace(world=world, domains=domains,
        spec=SimpleNamespace(structures=(structure,), formations=(read_formation(DEFINITION),)))
    retain_assemblies(SimpleNamespace(components=(component,), fitting_ids=(fittings,)))
    assert bool(fittings["mast"]) == (height == 12)


@pytest.mark.parametrize("changes", ({"clearance": True}, {"rail_domains": ["bed"]}, {"edge_height": 0}))
def test_invalid_formation_roles_are_rejected(changes):
    with pytest.raises(ValueError):
        read_formation({**DEFINITION, **changes})
