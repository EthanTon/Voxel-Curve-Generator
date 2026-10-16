from pathlib import Path
from dataclasses import replace
import subprocess
import sys
import tomllib

import pytest
import tomli_w

from src.Assembly import run
from src.Assembly.export import _in_track_space
from src.Core.Field.block_field import BlockField
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos, ColumnPos
from src.Extensions.Composition import build, parse
from src.Extensions.Composition.ownership import OwnedWorld
from src.Extensions.Composition.parser import read_region
from src.Extensions.Composition.regions import generate_region
from src.Extensions.Composition.terrain import combine_terrain
from src.Format.anvil import write_region_files
from src.Format.region import BlockRegion
from src.Format.schematics import load_schematic, write_schematic
from src.Terraform import Earthworks, Terraforming
from src.Terraform.clearance import Clearance
from src.Terraform.grid import Grid
from src.Terraform.terraform import Report


@pytest.fixture
def scene(tmp_path):
    field = BlockField(BlockRegistry().palette)
    field.set_block(BlockPos(0, 0, 0), BlockState.create("minecraft:stone"))
    write_schematic(BlockRegion.from_field(field, name="deck"), tmp_path / "deck.schem", data_version=3700)
    for name, start in (("first", 0), ("second", 7)):
        data = {
            "path": {"points": [[x, 64, 0] for x in range(start, start + 7)], "gradients": [[1, 0, 0]] * 7, "lead": 1},
            "output": {"path": f"{name}.schem"},
            "domain": [{"name": "bed", "order": 0}, {"name": "fitting", "order": 1}, {"name": "terrain", "order": 2}],
            "section": [{"name": "deck", "domain": "bed", "schematic": "deck.schem", "width": 1, "depth": 1}],
            "assignment": [{"stations": [0, 6], "sections": ["deck"]}],
            "lane": [{"name": "fence", "domain": "fitting", "sections": ["deck"], "block": "minecraft:oak_fence",
                      "offset": [0, 1, 0], "on": 1, "off": 0}],
        }
        (tmp_path / f"{name}.toml").write_text(tomli_w.dumps(data))
    return {
        "composition": {"origin": [10, 64, 20]},
        "output": {"path": "combined.schem", "mode": "absolute"},
        "component": [{"name": name, "build": f"{name}.toml"} for name in ("first", "second")],
        "connection": [{"from": "first", "from_station": 6, "to": "second", "to_station": 0}],
    }


def test_configured_run_connects_paths_and_writes_split_manifest(scene, tmp_path):
    scene["output"]["split"] = {"max_span": 8, "combined": False}
    config = tmp_path / "composition.toml"
    config.write_text(tomli_w.dumps(scene))
    result = run(config, manifest=True)
    manifest = tomllib.loads(result.primary.read_text())
    assert manifest["paste_origin"] == [10, 64, 20]
    assert manifest["piece_count"] == 2
    pieces = [load_schematic(Path(each["file"]))[0] for each in manifest["piece"]]
    assert pieces[0].state_at(BlockPos(6, 65, 0)).get("east") == "true"
    assert pieces[0].state_at(BlockPos(7, 65, 0)).get("west") == "true"
    assert result.manifest is not None
    assert set(tomllib.loads(result.manifest.read_text())["outputs"]) == {str(each) for each in result.outputs}


def test_region_ownership_clips_only_selected_heights_and_domains(scene, tmp_path):
    scene["region"] = [{"name": "shared", "domain": "bed", "boundary": [[4, 0], [10, 0], [10, 1], [4, 1]],
                         "top": 64, "block": "minecraft:gold_block"}]
    scene["ownership"] = [{"owner": "shared", "boundary": scene["region"][0]["boundary"],
                            "heights": [64, 64], "domains": ["bed"]}]
    world = build(parse(scene, tmp_path)).world
    for x in range(14):
        assert world.get_block(BlockPos(x, 64, 0)).name == ("minecraft:gold_block" if 4 <= x < 10 else "minecraft:stone")
        assert world.get_block(BlockPos(x, 65, 0)).name == "minecraft:oak_fence"
    assert world.owner_at(BlockPos(5, 64, 0)) == "shared"


def test_ownership_priority_and_ties(scene, tmp_path):
    boundary = [[0, 0], [7, 0], [7, 1], [0, 1]]
    scene["component"][1]["build"] = "first.toml"
    scene.pop("connection")
    scene["ownership"] = [{"owner": owner, "boundary": boundary} for owner in ("first", "second")]
    with pytest.raises(ValueError, match="conflicting ownership"):
        build(parse(scene, tmp_path))
    scene["ownership"][1]["priority"] = 1
    world = build(parse(scene, tmp_path)).world
    assert world.owner_at(BlockPos(3, 64, 0)) == "second"


@pytest.mark.parametrize("reverse_components", [False, True])
def test_shared_ownership_preserves_empty_space_regardless_of_component_order(scene, tmp_path, reverse_components):
    boundary = [[4, 0], [10, 0], [10, 1], [4, 1]]
    scene["region"] = [{"name": "shared", "domain": "bed", "boundary": boundary,
                        "holes": [[[6, 0], [7, 0], [7, 1], [6, 1]]], "top": 64, "block": "minecraft:gold_block"}]
    scene["ownership"] = [{"owner": "first", "boundary": boundary, "domains": ["bed"], "priority": 0},
                          {"owner": "shared", "boundary": boundary, "heights": [64, 64], "domains": ["bed"], "priority": 100}]
    if reverse_components:
        scene["component"].reverse()
    world = build(parse(scene, tmp_path)).world
    assert world.get_block(BlockPos(6, 64, 0)) is None
    assert world.get_block(BlockPos(5, 64, 0)).name == "minecraft:gold_block"
    assert world.get_block(BlockPos(6, 65, 0)).name == "minecraft:oak_fence"


def test_concave_region_holes_and_graded_surface():
    spec = read_region({"name": "surface", "domain": "bed", "block": "minecraft:stone", "top": 10,
        "grade": [1, 0], "thickness": 2, "boundary": [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]],
        "holes": [[[0, 0], [1, 0], [1, 1], [0, 1]]]})
    field = generate_region(spec, BlockRegistry().palette)
    positions = {position for position, _ in field.iter_block_ids()}
    assert len(positions) == 22
    assert BlockPos(3, 14, 1) in positions
    assert not any(position.x == position.z == 0 for position in positions)
    assert not any(position.x >= 2 and position.z >= 2 for position in positions)


def test_triangulated_surface_and_missing_coverage():
    region = {"name": "surface", "domain": "bed", "block": "minecraft:stone", "boundary": [[0, 0], [2, 0], [2, 2], [0, 2]],
              "triangles": [[[0, 10, 0], [2, 12, 0], [2, 12, 2]], [[0, 10, 0], [2, 12, 2], [0, 10, 2]]]}
    field = generate_region(read_region(region), BlockRegistry().palette)
    assert {position.y for position, _ in field.iter_block_ids()} == {11, 12}
    region["triangles"].pop()
    with pytest.raises(ValueError, match="does not cover"):
        generate_region(read_region(region), BlockRegistry().palette)


@pytest.mark.parametrize("change, message", [
    ({"composition": {"extra": True}}, "unknown options"),
    ({"component": []}, "at least one"),
    ({"ownership": [{"owner": "missing", "boundary": [[0, 0], [2, 0], [0, 2]]}]}, "unknown owner"),
    ({"region": [{"name": "first", "domain": "bed", "block": "minecraft:stone", "boundary": [[0, 0], [2, 0], [0, 2]]}]}, "unique"),
    ({"connection": [{"from": "first", "from_station": 0, "to": "second", "to_station": 0}]}, "exceeds"),
    ({"connection": [{"from": "first", "from_station": 99, "to": "second", "to_station": 0}]}, "outside"),
])
def test_invalid_configuration_is_rejected(scene, tmp_path, change, message):
    scene.update(change)
    with pytest.raises(ValueError, match=message):
        parse(scene, tmp_path)


def test_layered_clearance_preserves_ground_between_routes():
    grid = Grid(ColumnPos(0, 0), 1, 1)
    world = OwnedWorld(BlockRegistry.for_vanilla_26_2(), ())
    report = Report(None, (), (), 0, 0, 0, 0, 0)
    parts = []
    for low in (65, 75):
        clearance = Clearance(grid.filled(low), grid.filled(low + 2))
        part = Terraforming(grid, grid.filled(80), grid.filled(80), Earthworks(BlockField(world.registry.palette), frozenset()),
                            report, clearance=clearance, track_clearance=clearance)
        parts.append((str(low), 0, "terrain", part))
    shaped, _ = combine_terrain(parts, world)
    assert _in_track_space(BlockPos(0, 65, 0), shaped)
    assert _in_track_space(BlockPos(0, 75, 0), shaped)
    assert not _in_track_space(BlockPos(0, 70, 0), shaped)
    assert _in_track_space(BlockPos(0, 64, 0), shaped, floor=True)


def test_combined_terrain_protects_both_paths(scene, tmp_path):
    ground = BlockRegion("ground", BlockPos(0, 58, 0), BlockPos(16, 7, 16))
    for x in range(16):
        for z in range(16):
            for y in range(58, 65):
                ground.set_state(BlockPos(x, y, z), BlockState.create("minecraft:grass_block" if y == 64 else "minecraft:stone"))
    write_region_files(ground, tmp_path / "world" / "region", data_version=3700, merge=False)
    (tmp_path / "terrain.toml").write_text('[terraform]\nworld="world"\ndomain="terrain"\ndefault="shape"\nlook="minimal"\nmargin=4\nreach=4\n')
    scene["terraform"] = "terrain.toml"
    built = build(parse(scene, tmp_path))
    assert built.terraforming is not None
    for x in range(14):
        assert built.world.get_block(BlockPos(x, 64, 0)).name == "minecraft:stone"
        assert built.world.get_block(BlockPos(x, 65, 0)).name == "minecraft:oak_fence"
    assert {name.split(":")[0] for name, *_ in built.terraforming.report.stretches} == {"first", "second"}
    scene["output"]["split"] = {"terrain": True, "max_span": 16}
    result = run(parse(scene, tmp_path))
    assert result.primary.exists()


def test_extension_is_not_loaded_for_an_ordinary_build(scene, tmp_path):
    code = "from src.Assembly import run; import sys; run(sys.argv[1]); assert 'src.Extensions.Composition' not in sys.modules"
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "first.toml")], check=True)


def test_path_override_precedes_automatic_assignments(scene, tmp_path):
    source = tmp_path / "first.toml"
    data = tomllib.loads(source.read_text())
    data.pop("assignment")
    data["auto"] = {}
    data["selection"] = {"auto": False, "control_point": [{"point": [20, 0], "sections": ["deck"]}]}
    source.write_text(tomli_w.dumps(data))
    (tmp_path / "override.toml").write_text(tomli_w.dumps({"path": {
        "points": [[x, 64, 0] for x in range(20, 27)], "gradients": [[1, 0, 0]] * 7, "lead": 1}}))
    scene["component"] = [{"name": "first", "build": "first.toml", "path": "override.toml"}]
    scene.pop("connection")
    world = build(parse(scene, tmp_path)).world
    assert world.get_block(BlockPos(20, 64, 0)).name == "minecraft:stone"
    assert world.get_block(BlockPos(0, 64, 0)) is None


def test_combined_terrain_priorities_clearance_and_construction():
    world = OwnedWorld(BlockRegistry.for_vanilla_26_2(), ())
    constructed = BlockField(world.registry.palette)
    protected = BlockPos(0, 66, 0)
    constructed.set_block(protected, BlockState.create("minecraft:gold_block"))
    world.add_field(constructed)
    grid = Grid(ColumnPos(0, 0), 1, 1)
    report = Report(None, (), (), 0, 0, 0, 0, 0)
    parts = []
    for name, priority, block in (("higher", 10, "minecraft:dirt"), ("lower", 0, "minecraft:stone")):
        fill = BlockField(world.registry.palette)
        for y in range(64, 67):
            fill.set_block(BlockPos(0, y, 0), BlockState.create(block))
        clearance = Clearance(grid.filled(65), grid.filled(65)) if priority == 0 else None
        part = Terraforming(grid, grid.filled(63), grid.filled(66), Earthworks(fill, frozenset()), report,
                            clearance=clearance)
        parts.append((name, priority, "terrain", part))
    shaped, _ = combine_terrain(parts, world)
    assert shaped.fill.get_block(BlockPos(0, 64, 0)).name == "minecraft:dirt"
    assert shaped.cut == {BlockPos(0, 65, 0)}
    assert shaped.fill.get_block(protected) is None
    assert world.get_block(protected).name == "minecraft:gold_block"


def test_shared_designed_masks_follow_combined_terrain(scene, tmp_path):
    ground = BlockField(BlockRegistry().palette)
    for x in range(-5, 20):
        for z in range(-5, 6):
            ground.set_block(BlockPos(x, 60, z), BlockState.create("minecraft:grass_block"))
    ground_region = BlockRegion.from_field(ground, name="ground")
    write_schematic(ground_region, tmp_path / "ground.schem", data_version=3700, offset=ground_region.origin)
    mast = BlockField(BlockRegistry().palette)
    for y in range(-6, 4):
        mast.set_block(BlockPos(0, y, 0), BlockState.create("minecraft:gold_block"))
    mast_region = BlockRegion.from_field(mast, name="mast")
    write_schematic(mast_region, tmp_path / "mast.schem", data_version=3700, offset=mast_region.origin)
    for name in ("first", "second"):
        source = tmp_path / f"{name}.toml"
        data = tomllib.loads(source.read_text())
        data["structure"] = [{"name": "mast", "domain": "fitting", "sections": ["deck"],
            "headings": {"0.0": "mast.schem"}, "offset": [1, 0, 0],
            "surface_mask": {"side": "below", "reference": "designed"}}]
        source.write_text(tomli_w.dumps(data))
    (tmp_path / "terrain.toml").write_text('[terraform]\nworld="ground.schem"\ndomain="terrain"\ndefault="shape"\nlook="minimal"\nmargin=4\nreach=4\n')
    scene["terraform"] = "terrain.toml"
    original = (tmp_path / "ground.schem").read_bytes()
    built = build(parse(scene, tmp_path))
    assert len(built.terraforming.fill) > 0
    for name in ("first", "second"):
        retained = [position.y for position in built.world.iter_source_positions()
                    if built.world.owner_at(position) == name
                    and built.world.get_block(position).name == "minecraft:gold_block"]
        assert retained and 60 < max(retained) < 67
    assert (tmp_path / "ground.schem").read_bytes() == original


def test_composition_template_builds_and_exports(tmp_path):
    from src.Extensions.Composition import load

    spec = load(Path(__file__).resolve().parents[1] / "path template" / "composition.toml")
    result = run(replace(spec, output=replace(spec.output, path=tmp_path / "template.schem")))
    region, _ = load_schematic(result.primary)
    assert region.state_at(BlockPos(5, 71, 15)).name == "minecraft:smooth_stone"
    assert any(region.state_at(BlockPos(20, 71, z)) is not None for z in range(28, 37))


@pytest.mark.parametrize("module", ["src.main", "src.Extensions.Composition"])
def test_composition_cli_uses_shared_relative_origin(scene, tmp_path, module):
    scene["output"]["mode"] = "relative"
    config = tmp_path / "composition.toml"
    config.write_text(tomli_w.dumps(scene))
    completed = subprocess.run([sys.executable, "-m", module, str(config), "--manifest"],
                               check=True, capture_output=True, text=True)
    manifest = Path(completed.stdout.strip())
    assert manifest.exists()
    region, _ = load_schematic(tmp_path / "combined.schem")
    assert region.state_at(BlockPos(-10, 0, -20)).name == "minecraft:stone"
    assert region.state_at(BlockPos(3, 0, -20)).name == "minecraft:stone"
