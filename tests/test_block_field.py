import unittest

from src.Core.Field.block_field import BlockBounds, BlockField
from src.Core.block.states import BlockState


class BlockFieldTests(unittest.TestCase):
    def test_bounds_iteration_and_revision_follow_edits(self) -> None:
        field = BlockField()
        field.set_block((-2, 3, 4), BlockState.create("minecraft:stone"))
        field.set_block((5, -1, 0), BlockState.create("minecraft:dirt"))

        self.assertEqual(field.bounds, BlockBounds((-2, -1, 0), (5, 3, 4)))
        self.assertEqual(field.revision, 2)
        self.assertEqual(len(tuple(field.iter_block_ids())), 2)

        field.remove_block((-2, 3, 4))

        self.assertEqual(field.bounds, BlockBounds((5, -1, 0), (5, -1, 0)))
        self.assertEqual(field.revision, 3)

    def test_frozen_field_rejects_mutation(self) -> None:
        field = BlockField()
        field.set_block((0, 0, 0), BlockState.create("minecraft:stone"))
        field.freeze()

        with self.assertRaises(RuntimeError):
            field.remove_block((0, 0, 0))

    def test_negative_positions_use_floor_division_for_chunks(self) -> None:
        field = BlockField()
        field.set_block((-1, 0, -17), BlockState.create("minecraft:stone"))

        self.assertEqual(field.occupied_chunks(), frozenset({(-1, 0, -2)}))


if __name__ == "__main__":
    unittest.main()
