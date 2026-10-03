from src.Assembly.lanes import support_rail_joints
from src.Core.Field.block_field import BlockField
from src.Core.World.block_world import BlockWorld
from src.Core.block.registry import BlockRegistry
from src.Format.schematics import parse_state_key
from src.Core.coordinates import BlockPos


def build_joint(deck):
    world = BlockWorld(BlockRegistry.for_vanilla_26_2())
    bed = BlockField(world.registry.palette)
    for x in (0, 2):
        bed.set_block(BlockPos(x, 65, 0), parse_state_key('minecraft:gray_wool'))
    if deck:
        bed.set_block(BlockPos(1, 64, 0), parse_state_key('minecraft:smooth_stone_slab[type=top]'))
    world.add_field(bed, connectivity_domain=1)
    rails = BlockField(world.registry.palette)
    for x in range(3):
        rails.set_block(BlockPos(x, 66, 0), parse_state_key('minecraft:rail'))
    world.add_field(rails, connectivity_domain=2)
    return world


def test_diagonal_joint_extends_the_sleepers_over_its_existing_deck():
    world = build_joint(True)
    support_rail_joints(world, {1})
    assert world.get_block(BlockPos(1, 65, 0), resolved=False).name == 'minecraft:gray_wool'
    assert world.source_at(BlockPos(1, 65, 0)).domain == 1


def test_a_joint_does_not_make_support_beyond_the_deck():
    world = build_joint(False)
    support_rail_joints(world, {1})
    assert world.get_block(BlockPos(1, 65, 0), resolved=False) is None
