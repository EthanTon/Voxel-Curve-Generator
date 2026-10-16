from dataclasses import replace

import pytest

from src.Assembly import finalize_build, place_features, prepare_build
from src.Core.Field.block_field import BlockField
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos
from src.Format.region import BlockRegion
from src.Format.schematics import write_schematic
from src.Parser import parse
from src.Terraform.plan import Plan


@pytest.fixture
def build_spec(tmp_path):
    blocks = BlockField(BlockRegistry().palette)
    blocks.set_block(BlockPos(0, 0, 0), BlockState.create("minecraft:stone"))
    write_schematic(BlockRegion.from_field(blocks, name="deck"), tmp_path / "deck.schem", data_version=3700)
    return parse({
        "path": {"points": [[x, 64, 0] for x in range(7)], "gradients": [[1, 0, 0]] * 7, "lead": 1},
        "output": {"path": "out.schem"},
        "domain": [
            {"name": "bed", "order": 0, "placement": {"mode": "OVERWRITE"}},
            {"name": "fitting", "order": 1, "placement": {"mode": "OVERWRITE"}},
        ],
        "section": [{"name": "deck", "domain": "bed", "schematic": "deck.schem", "width": 1, "depth": 1}],
        "assignment": [{"stations": [0, 6], "sections": ["deck"]}],
        "lane": [{"name": "fence", "domain": "fitting", "sections": ["deck"],
                  "block": "minecraft:oak_fence", "offset": [0, 1, 0], "on": 1, "off": 0}],
    }, tmp_path)


def test_stages_defer_placement_and_neighbor_resolution(build_spec):
    stages = []
    progress = lambda stage, completed, total: stages.append(stage)
    prepared = prepare_build(build_spec, progress=progress)
    assert not prepared.world.instances
    assert "Placing features" not in stages
    placed = place_features(prepared, progress=progress)
    assert prepared.world.instances
    assert prepared.world.overlay_size == 0
    assert prepared.world.get_block(BlockPos(3, 65, 0)).get("east") is None
    assert "Shaping terrain" not in stages
    built = finalize_build(placed, progress=progress)
    assert built.world is prepared.world
    assert built.world.get_block(BlockPos(3, 65, 0)).get("east") == "true"
    assert built.terraforming is None
    assert stages[-1] == "Resolving blocks"


def test_shared_paths_connect_only_when_finalized_together(build_spec):
    first = prepare_build(build_spec)
    next_path = replace(build_spec.path, points=tuple(point + (7, 0, 0) for point in build_spec.path.points))
    second = prepare_build(replace(build_spec, path=next_path), shared=first)
    assert second.world is first.world
    assert second.domains is first.domains
    placed = place_features(first, second)
    assert len(placed.fitting_ids) == 2
    assert placed.fitting_ids[0]["fence"] != placed.fitting_ids[1]["fence"]
    built = finalize_build(placed)
    assert built.world.get_block(BlockPos(6, 65, 0)).get("east") == "true"
    assert built.world.get_block(BlockPos(7, 65, 0)).get("west") == "true"
    assert set(built.world.iter_source_positions()) == {BlockPos(x, y, 0) for x in range(14) for y in (64, 65)}


def test_domain_order_applies_across_components(build_spec):
    painted_lane = replace(build_spec.lanes[0], block="minecraft:gold_block", offset=BlockPos(0, 0, 0))
    first = prepare_build(replace(build_spec, lanes=(painted_lane,)))
    second = prepare_build(replace(build_spec, lanes=()), shared=first)
    built = finalize_build(place_features(first, second))
    assert {built.world.get_block(point).name for point in built.world.iter_source_positions()} == {"minecraft:gold_block"}


def test_unrelated_worlds_are_rejected_before_placement(build_spec):
    first, second = prepare_build(build_spec), prepare_build(build_spec)
    with pytest.raises(ValueError, match="shared="):
        place_features(first, second)
    assert not first.world.instances
    assert not second.world.instances


def test_shared_build_rejects_conflicting_domain_rules(build_spec):
    first = prepare_build(build_spec)
    changed_domain = replace(build_spec.domains[0], order=2)
    changed_spec = replace(build_spec, domains=(changed_domain, build_spec.domains[1]))
    with pytest.raises(ValueError, match="domain and interaction"):
        prepare_build(changed_spec, shared=first)


def test_shared_build_rejects_different_surface_sources(build_spec, tmp_path):
    first = prepare_build(build_spec)
    with pytest.raises(ValueError, match="surface world"):
        prepare_build(replace(build_spec, surface_world=tmp_path / "other.schem"), shared=first)


def test_combined_terrain_is_rejected_before_placement(build_spec):
    terrain_spec = replace(build_spec, terraform=Plan())
    first = prepare_build(terrain_spec)
    second = prepare_build(terrain_spec, shared=first)
    with pytest.raises(ValueError, match="combined builds must omit terraform"):
        place_features(first, second)
    assert not first.world.instances
