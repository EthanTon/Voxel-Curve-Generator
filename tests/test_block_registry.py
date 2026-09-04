import unittest

from src.Core.block.info import BlockInfo, NeighborRule, RotationRule, SolidFace
from src.Core.block.palette import BlockPalette
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState


class BlockRegistryTests(unittest.TestCase):
    def test_registry_stays_aligned_with_shared_palette(self) -> None:
        palette = BlockPalette()
        stone = BlockState.create("minecraft:stone")
        stone_id = palette.add(stone)
        registry = BlockRegistry(palette)

        self.assertEqual(registry.get_state(stone_id), stone)
        self.assertEqual(registry.get_info(stone_id), BlockInfo())

        fence_info = BlockInfo(
            rotation=RotationRule.CARDINAL,
            neighbor_rule=NeighborRule.FENCE,
        )
        fence = BlockState.create("minecraft:oak_fence")
        fence_id = registry.register(fence, fence_info)

        self.assertEqual(registry.get_id(fence), fence_id)
        self.assertEqual(registry.get_info(fence_id), fence_info)
        self.assertEqual(len(registry), 2)

    def test_variant_inherits_source_behavior(self) -> None:
        registry = BlockRegistry()
        info = BlockInfo(
            neighbor_rule=NeighborRule.PANE,
            solid_faces=SolidFace.UP | SolidFace.DOWN,
        )
        source_id = registry.register(
            BlockState.create("minecraft:glass_pane"),
            info,
        )
        variant = BlockState.create(
            "minecraft:glass_pane",
            {"north": "true"},
        )

        variant_id = registry.get_or_add_variant(source_id, variant)

        self.assertEqual(registry.get_info(variant_id), info)
        self.assertEqual(registry.get_state(variant_id), variant)

    def test_frozen_palette_still_returns_existing_id(self) -> None:
        palette = BlockPalette()
        state = BlockState.create("minecraft:stone")
        block_id = palette.add(state)
        palette.freeze()

        self.assertEqual(palette.add(state), block_id)
        with self.assertRaises(RuntimeError):
            palette.add(BlockState.create("minecraft:dirt"))

    def test_invalid_registry_id_is_rejected(self) -> None:
        registry = BlockRegistry()

        with self.assertRaises(IndexError):
            registry.get_info(0)


if __name__ == "__main__":
    unittest.main()
