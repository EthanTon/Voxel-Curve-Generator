from src.Assembly.earthworks import alignment_of
from src.Core.Field.block_field import BlockField
from src.Core.Path.path import Path
from src.Core.World.block_world import BlockWorld
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos


def test_a_block_belongs_to_the_station_that_laid_it():
    # A staircase corner nearer the next point stays with the one that laid it.
    registry = BlockRegistry.for_vanilla_26_2()
    world = BlockWorld(registry=registry)
    path = Path([(x, 70, 0) for x in range(6)], [(1.0, 0.0, 0.0)] * 6)
    for station, corner in ((1, BlockPos(3, 70, 4)), (4, BlockPos(4, 70, 0))):
        field = BlockField(registry.palette)
        field.set_block(BlockPos(0, 0, 0), BlockState.create("minecraft:stone"))
        world.add_field(field, corner, sequence_index=station)
    alignment = alignment_of(world, path)
    assert alignment.station[(3, 4)] == 1
    assert alignment.station[(4, 0)] == 4


def test_a_yielding_side_foundation_does_not_hide_protruding_rail_support():
    registry = BlockRegistry.for_vanilla_26_2()
    world = BlockWorld(registry)
    path = Path([(0, 70, 0)], [(0., 0., 1.)])
    bed = BlockField(registry.palette)
    bed.set_block(BlockPos(0, 70, 0), BlockState.create("minecraft:gray_wool"))
    world.add_field(bed, connectivity_domain=1)
    side = BlockField(registry.palette)
    side.set_block(BlockPos(1, 66, 0), BlockState.create("minecraft:andesite"))
    world.add_field(side, connectivity_domain=1, yielding=frozenset({"minecraft:andesite"}))
    rail = BlockField(registry.palette)
    rail.set_block(BlockPos(1, 71, 0), BlockState.create("minecraft:rail"))
    world.add_field(rail, connectivity_domain=2)

    assert set(alignment_of(world, path, {1}).footing) == {(0, 0)}
    supported = alignment_of(world, path, {1}, include_rails=True)
    assert supported.meets[(1, 0)] == 70
    assert supported.rail_support == {(1, 0): 70}
