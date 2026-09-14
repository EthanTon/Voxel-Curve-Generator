import unittest

from src.Core.Field.block_field import BlockBounds, BlockField
from src.Core.World.block_world import BlockTransform
from src.Core.block.states import BlockState
from collections.abc import Sequence

from src.Core.coordinates import BlockPos, ChunkPos, ColumnPos, Coordinate, Vec2, Vec3, Vector


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

    def test_coordinates_match_the_interface_without_inheriting(self) -> None:
        for value in (Vec2(1.0, 2.0), Vec3(1.0, 2.0, 3.0), BlockPos(1, 2, 3), ChunkPos(1, 2, 3), ColumnPos(1, 2)):
            self.assertEqual(type(value).__mro__, (type(value), object))
            self.assertIsInstance(value, Coordinate)
            self.assertIsInstance(value, Sequence)

    def test_sequence_behavior_is_kept(self) -> None:
        position = BlockPos(1, 2, 2)
        self.assertIn(2, position)
        self.assertEqual(list(reversed(position)), [2, 2, 1])
        self.assertEqual((position.index(2), position.count(2)), (1, 2))
        self.assertEqual(position[1:], (2, 2))
        self.assertLess(BlockPos(1, 2, 1), position)
        self.assertNotEqual(position, ChunkPos(1, 2, 2))
        self.assertEqual({(1, 2, 2): "tuple key"}[position], "tuple key")


class VectorTests(unittest.TestCase):
    def test_dot_and_length(self) -> None:
        self.assertEqual(Vec3(1.0, 2.0, 3.0).dot((4.0, -5.0, 6.0)), 12.0)
        self.assertEqual(Vec2(3.0, 4.0).length, 5.0)

    def test_unit_keeps_direction_and_type(self) -> None:
        unit = Vec2(3.0, -4.0).unit()
        self.assertIsInstance(unit, Vec2)
        self.assertTrue(unit.is_close((0.6, -0.8)))
        self.assertAlmostEqual(Vec3(2.0, 2.0, 1.0).unit().length, 1.0)

    def test_zero_vectors_have_no_unit(self) -> None:
        with self.assertRaises(ValueError):
            Vec2(0.0, 0.0).unit()

    def test_is_close_compares_by_distance(self) -> None:
        self.assertEqual(Vec2(0.0, 0.0).distance_to((3.0, 4.0)), 5.0)
        self.assertTrue(Vec2(1.0, 0.0).is_close((1.0, 0.01), tolerance=0.02))
        self.assertFalse(Vec2(1.0, 0.0).is_close((1.0, 0.03), tolerance=0.02))

    def test_vectors_match_the_interface_without_inheriting(self) -> None:
        for vector in (Vec2(1.0, 0.0), Vec3(1.0, 0.0, 0.0)):
            self.assertIsInstance(vector, Vector)
        self.assertNotIsInstance(BlockPos(1, 0, 0), Vector)

    def test_horizontal_drops_the_height(self) -> None:
        self.assertEqual(Vec3(1.0, 9.0, -2.0).horizontal, Vec2(1.0, -2.0))


if __name__ == "__main__":
    unittest.main()
