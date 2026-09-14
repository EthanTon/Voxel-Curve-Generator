import unittest

from src.Core.Field.block_field import BlockBounds, BlockField
from src.Core.World.block_world import BlockTransform
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos, ChunkPos, Vec3


class CoordinateTests(unittest.TestCase):
    def test_coordinates_expose_named_components_and_tuple_compatibility(self) -> None:
        position = BlockPos(1, 2, 3)

        self.assertEqual((position.x, position.y, position.z), (1, 2, 3))
        self.assertFalse(hasattr(position, "__dict__"))
        self.assertEqual(position.as_tuple(), (1, 2, 3))
        self.assertEqual(position, (1, 2, 3))
        self.assertEqual(Vec3.coerce((1.5, 2.5, 3.5)), Vec3(1.5, 2.5, 3.5))

    def test_field_normalizes_positions_and_returns_coordinate_structs(self) -> None:
        field = BlockField()
        field.set_block((17, -1, 0), BlockState.create("minecraft:stone"))

        position, _ = next(field.iter_block_ids())
        self.assertIsInstance(position, BlockPos)
        self.assertEqual(field.bounds, BlockBounds(BlockPos(17, -1, 0), position))
        self.assertEqual(field.occupied_chunks(), {ChunkPos(1, -1, 0)})

    def test_transform_returns_block_positions(self) -> None:
        transform = BlockTransform((10, 0, 10), quarter_turns=1)

        transformed = transform.apply((0, 0, -1))
        self.assertIsInstance(transformed, BlockPos)
        self.assertEqual(transformed, BlockPos(11, 0, 10))
        self.assertEqual(transform.inverse(transformed), BlockPos(0, 0, -1))


if __name__ == "__main__":
    unittest.main()
