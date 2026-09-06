import unittest

from src.Core.Field.block_field import BlockBounds, BlockField
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos


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

    def test_dense_chunk_storage_preserves_positions_and_values(self) -> None:
        field = BlockField()
        stone = BlockState.create("minecraft:stone")
        positions = [
            BlockPos(x, y, z)
            for y in range(4)
            for z in range(8)
            for x in range(8)
        ]

        for position in positions:
            field.set_block(position, stone)

        self.assertEqual(len(field), len(positions))
        self.assertEqual(
            {position for position, _ in field.iter_block_ids()},
            set(positions),
        )
        for position in positions:
            self.assertEqual(field.get_block(position), stone)

    def test_freeze_compacts_sparse_chunks_without_changing_lookup(self) -> None:
        field = BlockField()
        stone = BlockState.create("minecraft:stone")
        positions = (BlockPos(-17, -1, 31), BlockPos(1000, 200, -1000))
        for position in positions:
            field.set_block(position, stone)

        field.freeze()

        for position in positions:
            self.assertEqual(field.get_block(position), stone)


if __name__ == "__main__":
    unittest.main()
