import numpy as np
import pytest
import tomli_w
from tests.test_composition_formations import DEFINITION

from src.Assembly import assemble, run
from src.Core.Field.block_field import BlockField
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos, Vec3
from src.Extensions.Composition.rooted import compose
from src.Format.region import BlockRegion
from src.Format.schematics import load_schematic, write_schematic
from src.Parser import load


@pytest.fixture
def scene(tmp_path):
    for name, block in (("deck", "stone"), ("mast", "gold_block")):
        field = BlockField(BlockRegistry().palette)
        field.set_block(BlockPos(0, 0, 0), BlockState.create(f"minecraft:{block}"))
        write_schematic(BlockRegion.from_field(field, name=name), tmp_path / f"{name}.schem", data_version=3700)
    root = {
        "composition": "composition.toml",
        "path": {"points": [[x, 64, 0] for x in range(41)], "gradients": [[1, 0, 0]] * 41, "lead": 1},
        "output": {"path": "combined.schem", "mode": "absolute"},
        "domain": [{"name": "bed", "order": 0}, {"name": "fitting", "order": 1}],
        "section": [{"name": "deck", "domain": "bed", "schematic": "deck.schem", "width": 1, "depth": 1}],
        "assignment": [{"stations": [0, 40], "sections": ["deck"]}],
        "structure": [{"name": "mast", "domain": "fitting", "sections": ["deck"],
                       "headings": {"0.0": "mast.schem"}, "spacing": 10, "offset": [0, 2, 0]}],
    }
    config = {"branch": [{"name": "main"}]}

    def write(paths=None):
        for name, path in (paths or {}).items():
            (tmp_path / name).write_text(tomli_w.dumps({"path": path}))
        (tmp_path / "build.toml").write_text(tomli_w.dumps(root))
        (tmp_path / "composition.toml").write_text(tomli_w.dumps(config))
        return load(tmp_path / "build.toml")

    return root, config, write


def shifted_path(root, lateral):
    return {**root["path"], "points": [[x, y, z + lateral] for x, y, z in root["path"]["points"]]}


@pytest.mark.parametrize("scoped", (False, True))
def test_formation_rules_reference_root_library_and_transition_extent(scene, scoped):
    root, config, write = scene
    root["domain"] += [{"name": "rail"}, {"name": "formation"}]
    root["formation"] = [DEFINITION]
    config["branch"].append({"name": "second"})
    config["composition_rule"] = [{"name": "track", "handler": "formation", "target": "formation:track"}]
    config["rule_application"] = [{"rule": "track", "branches": ["main", "second"]}]
    if scoped:
        config["node"] = [{"name": "joint", "connections": [["main:20", "second:20"]]}]
        config["transition"] = [{"name": "joint", "node": "joint", "handler": "continue",
            "boundary": [[15, -5], [25, -5], [25, 5], [15, 5]], "heights": [60, 70]}]
        config["rule_application"][0]["scope"] = "joint"
    spec = compose(write())
    assert spec.formations[0].formation == spec.components[0].build.formation("track")
    assert spec.formations[0].components == ("main", "second")
    assert bool(spec.formations[0].boundary) == scoped
    if scoped:
        del config["transition"][0]["boundary"]
        with pytest.raises(ValueError, match="needs a boundary"):
            compose(write())


def test_root_is_entry_point_for_export_and_assembly(scene, tmp_path):
    root, config, write = scene
    config["branch"][0]["path"] = "main.toml"
    spec = write({"main.toml": shifted_path(root, 5)})
    world = assemble(spec)
    assert world.get_block(BlockPos(20, 64, 5)).name == "minecraft:stone"
    assert world.get_block(BlockPos(20, 64, 0)) is None
    result = run(tmp_path / "build.toml", manifest=True)
    region, _ = load_schematic(result.primary)
    assert region.state_at(BlockPos(20, 64, 5)).name == "minecraft:stone"
    assert result.manifest.exists()
    with pytest.raises(ValueError, match="must not replace"):
        run(spec, manifest=tmp_path / "composition.toml")


def test_cropped_branch_preserves_marker_assignments(scene):
    root, config, write = scene
    root["marker"] = {"first": [0, 0], "last": [40, 0]}
    root["assignment"] = [{"from": "first", "to": "last", "sections": ["deck"]}]
    config["branch"][0].update(stations=[10, 20])
    world = assemble(write())
    assert world.get_block(BlockPos(10, 64, 0)).name == "minecraft:stone"
    assert world.get_block(BlockPos(20, 64, 0)).name == "minecraft:stone"
    assert world.get_block(BlockPos(9, 64, 0)) is None
    assert world.get_block(BlockPos(21, 64, 0)) is None


def test_designer_supplied_path_and_gradients_are_preserved(scene):
    root, config, write = scene
    path = shifted_path(root, 2)
    path["points"] = [[x, 64, 2 + x // 10] for x in range(41)]
    path["gradients"] = [[2, 0, .2]] * 41
    config["branch"][0]["path"] = "designed.toml"
    component, = compose(write({"designed.toml": path})).components
    assert component.build.path.points == tuple(BlockPos(*point) for point in path["points"])
    assert component.build.path.gradients == tuple(Vec3(*gradient) for gradient in path["gradients"])


def test_rebase_keeps_geometry_and_gap_stays_empty(scene):
    root, config, write = scene
    for point in root["path"]["points"][20:]:
        point[2] += 20
    with pytest.raises(ValueError, match="undeclared path jump"):
        compose(write())
    config["branch"][0]["jump"] = [{"station": 20, "mode": "rebase", "delta": [0, 0, 20]}]
    world = assemble(write())
    assert world.get_block(BlockPos(20, 64, 0)).name == "minecraft:stone"
    config["branch"][0]["jump"] = [{"station": 20, "mode": "gap"}]
    world = assemble(write())
    assert world.get_block(BlockPos(19, 64, 0)).name == "minecraft:stone"
    assert world.get_block(BlockPos(20, 64, 20)).name == "minecraft:stone"
    assert world.get_block(BlockPos(20, 64, 10)) is None


def test_split_uses_declared_geometry_and_checks_endpoints(scene):
    root, config, write = scene
    config["branch"] = [{"name": "in", "stations": [0, 10]},
                        {"name": "main", "stations": [10, 40]},
                        {"name": "side", "stations": [10, 40], "path": "side.toml"}]
    side = {**root["path"], "points": [[x, 64, max(0, x - 10) // 6] for x in range(41)],
            "gradients": [[1, 0, 1 / 6]] * 41}
    config["node"] = [{"name": "fork", "connections": [["in:end", "main:start"], ["in:end", "side:start"]]}]
    config["transition"] = [{"name": "fork", "node": "fork", "handler": "split"}]
    assert len(compose(write({"side.toml": side})).connections) == 2
    side["points"][10][2] = 5
    config["branch"][2]["max_step"] = 10
    with pytest.raises(ValueError, match="endpoints exceed"):
        compose(write({"side.toml": side}))


@pytest.mark.parametrize("reverse", [False, True])
def test_priority_and_reservation_are_independent_of_branch_order(scene, reverse):
    root, config, write = scene
    root["section"].append({**root["section"][0], "name": "gold", "schematic": "mast.schem"})
    root["assignment"][0]["sections"].append("gold")
    config["branch"] = [{"name": "stone", "sections": ["deck"], "structures": [], "priority": 10},
                        {"name": "gold", "sections": ["gold"], "structures": [], "priority": 0}]
    if reverse:
        config["branch"].reverse()
    world = assemble(write())
    assert world.get_block(BlockPos(20, 64, 0)).name == "minecraft:stone"
    assert world.owner_at(BlockPos(20, 64, 0)) == "stone"
    config["composition_rule"] = [{"name": "reserve", "handler": "reserve", "owner": "gold",
                                    "boundary": [[15, -1], [25, -1], [25, 1], [15, 1]], "domains": ["bed"]}]
    config["rule_application"] = [{"rule": "reserve", "branches": ["stone", "gold"]}]
    world = assemble(write())
    assert world.get_block(BlockPos(20, 64, 0)).name == "minecraft:gold_block"
    assert world.get_block(BlockPos(10, 64, 0)).name == "minecraft:stone"


def test_cross_branch_shared_structure_is_placed_once_and_reindexed(scene):
    root, config, write = scene
    config["branch"] = [{"name": "main", "stations": [10, 30]},
                        {"name": "side", "path": "side.toml"}]
    config["feature_path"] = [{"name": "anchors", "source": "side/path", "spacing": 10}]
    config["composition_rule"] = [{"name": "shared_mast", "handler": "share", "owner": "main",
                                    "source": "feature_path:anchors", "target": "structure:mast"}]
    config["rule_application"] = [{"rule": "shared_mast", "branches": ["main", "side"]}]
    world = assemble(write({"side.toml": shifted_path(root, 6)}))
    for x in (10, 20, 30):
        assert world.get_block(BlockPos(x, 66, 6)).name == "minecraft:gold_block"
        assert world.owner_at(BlockPos(x, 66, 6)) == "main"
    assert world.get_block(BlockPos(0, 66, 6)) is None
    assert world.get_block(BlockPos(20, 66, 0)) is None


def test_shared_midpoint_uses_both_tracks(scene):
    root, config, write = scene
    config["branch"] = [{"name": "main"}, {"name": "side", "path": "side.toml"}]
    config["feature_path"] = [{"name": "middle", "source": "main/path", "spacing": 10,
                               "project": {"target": "side/path", "direction": "right", "placement": "midpoint"}}]
    config["composition_rule"] = [{"name": "shared", "handler": "share", "owner": "main",
                                    "source": "feature_path:middle", "target": "structure:mast"}]
    config["rule_application"] = [{"rule": "shared", "branches": ["main", "side"]}]
    world = assemble(write({"side.toml": shifted_path(root, 6)}))
    assert world.get_block(BlockPos(20, 66, 3)).name == "minecraft:gold_block"
    assert world.get_block(BlockPos(20, 66, 0)) is None
    assert world.get_block(BlockPos(20, 66, 6)) is None


def test_shared_source_gap_never_bridges_a_wire(scene, tmp_path):
    root, config, write = scene
    root["wire"] = [{"name": "wire", "domain": "fitting", "sections": ["deck"], "schematic": "mast.schem",
                      "offset": [0, 4, 0], "geometry": "line"}]
    for point in root["path"]["points"][20:]:
        point[0] += 20
    config["branch"][0]["jump"] = [{"station": 20, "mode": "gap"}]
    config["feature_path"] = [{"name": "anchors", "source": "main/path", "spacing": 10}]
    config["composition_rule"] = [{"name": "bind", "handler": "bind", "source": "feature_path:anchors", "target": "wire:wire"}]
    config["rule_application"] = [{"rule": "bind", "branches": ["main"]}]
    world = assemble(write())
    assert world.get_block(BlockPos(19, 68, 0)).name == "minecraft:gold_block"
    assert world.get_block(BlockPos(40, 68, 0)).name == "minecraft:gold_block"
    assert world.get_block(BlockPos(30, 68, 0)) is None


def test_transition_spans_reference_authored_stations(scene):
    root, config, write = scene
    config["branch"] = [{"name": "in", "stations": [0, 10]}, {"name": "main", "stations": [10, 40]},
                        {"name": "side", "stations": [10, 40]}]
    config["node"] = [{"name": "fork", "connections": [["in:end", "main:start"], ["in:end", "side:start"]]}]
    config["transition"] = [{"name": "fork", "node": "fork", "handler": "split", "spans": [
        {"branch": "in", "stations": [9, 10]}, {"branch": "main", "stations": [10, 30]},
        {"branch": "side", "stations": [10, 30]}]}]
    compose(write())
    config["transition"][0]["spans"][1]["stations"] = [20, 30]
    with pytest.raises(ValueError, match="must include"):
        compose(write())


def test_unknown_transition_flags_are_rejected(scene):
    root, config, write = scene
    config["branch"] = [{"name": "in", "stations": [0, 20]}, {"name": "out", "stations": [20, 40]}]
    config["node"] = [{"name": "join", "connections": [["in:end", "out:start"]]}]
    config["transition"] = [{"name": "join", "node": "join", "flags": {"rail_state": "main"}}]
    with pytest.raises(ValueError, match="unknown options"):
        compose(write())


@pytest.mark.parametrize("change,message", [
    ({"mystery": True}, "unknown options"),
    ({"constraints": {"profile": "conventional"}}, "unknown options"),
    ({"branch": [{"name": "main", "offsets": [{"at": 0, "lateral": 2}]}]}, "unknown options"),
    ({"branch": [{"name": "main", "jump": [{"station": 20, "mode": "teleport"}]}]}, "jump mode"),
    ({"node": [{"name": "node", "connections": [["main:start", "main:end"]]}]}, "every node"),
    ({"feature_path": [{"name": "missing", "source": "missing/path"}]}, "unknown feature path"),
    ({"feature_path": [{"name": "a", "source": "feature_path:b"}, {"name": "b", "source": "feature_path:a"}]}, "cycle"),
])
def test_invalid_graph_and_feature_dependencies_fail(scene, change, message):
    root, config, write = scene
    config.update(change)
    with pytest.raises(ValueError, match=message):
        assemble(write())


def test_builder_accepts_designer_spacing_and_curvature(scene):
    root, config, write = scene
    config["branch"] = [{"name": "main"}, {"name": "side", "path": "side.toml"}]
    world = assemble(write({"side.toml": shifted_path(root, 1)}))
    assert world.get_block(BlockPos(20, 64, 1)).name == "minecraft:stone"
    angles = np.arange(21) / 100
    root["path"] = {"points": [[round(100 * np.sin(a)), 64, round(100 * (1 - np.cos(a)))] for a in angles],
                    "gradients": [[np.cos(a), 0, np.sin(a)] for a in angles], "lead": 1}
    root["assignment"][0]["stations"] = [0, 20]
    config["branch"] = [{"name": "curve"}]
    world = assemble(write())
    assert world.get_block(BlockPos(*root["path"]["points"][-1])).name == "minecraft:stone"
