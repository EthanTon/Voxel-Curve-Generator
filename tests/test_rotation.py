import unittest

from src.Core.block.info import BlockInfo, RotationRule
from src.Core.block.rotation import rotate_block_state
from src.Core.block.states import BlockState


class RotationTests(unittest.TestCase):
    def test_sparse_cardinal_property_moves_instead_of_being_copied(self) -> None:
        state = BlockState.create(
            "minecraft:test_block",
            {"north": "true", "waterlogged": "false"},
        )

        rotated = rotate_block_state(
            state,
            BlockInfo(rotation=RotationRule.CARDINAL),
            1,
        )

        self.assertIsNone(rotated.get("north"))
        self.assertEqual(rotated.get("east"), "true")
        self.assertEqual(rotated.get("waterlogged"), "false")

    def test_four_cardinal_properties_rotate_without_data_loss(self) -> None:
        state = BlockState.create(
            "minecraft:test_block",
            {
                "north": "north-value",
                "east": "east-value",
                "south": "south-value",
                "west": "west-value",
            },
        )

        rotated = rotate_block_state(
            state,
            BlockInfo(rotation=RotationRule.CARDINAL),
            1,
        )

        self.assertEqual(
            dict(rotated.properties),
            {
                "north": "west-value",
                "east": "north-value",
                "south": "east-value",
                "west": "south-value",
            },
        )


if __name__ == "__main__":
    unittest.main()
