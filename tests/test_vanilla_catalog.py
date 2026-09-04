import unittest
from math import prod

from src.Core.block.info import NeighborRule, RotationRule, SolidFace
from src.Core.block.connectivity import NeighborResolver
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.block.vanilla import VANILLA_VERSION, vanilla_catalog


class VanillaCatalogTests(unittest.TestCase):
    def test_catalog_contains_every_reported_26_2_block(self) -> None:
        catalog = vanilla_catalog()

        self.assertEqual(VANILLA_VERSION, "26.2")
        self.assertEqual(len(catalog), 1196)
        self.assertIn("minecraft:copper_bars", catalog)
        self.assertIn("minecraft:oak_shelf", catalog)
        self.assertIn("minecraft:cinnabar_wall", catalog)

    def test_catalog_describes_every_official_26_2_block_state(self) -> None:
        catalog = vanilla_catalog()

        state_count = sum(
            prod(len(values) for values in catalog.definition(name).properties.values())
            for name in catalog.names()
        )
        self.assertEqual(state_count, 32_366)

    def test_behavior_families_select_connectivity_rules(self) -> None:
        catalog = vanilla_catalog()

        cases = {
            "minecraft:oak_fence": NeighborRule.FENCE,
            "minecraft:copper_bars": NeighborRule.PANE,
            "minecraft:cobblestone_wall": NeighborRule.WALL,
            "minecraft:oak_stairs": NeighborRule.STAIR,
            "minecraft:redstone_wire": NeighborRule.REDSTONE,
            "minecraft:pale_moss_carpet": NeighborRule.MOSS_CARPET,
            "minecraft:scaffolding": NeighborRule.SCAFFOLDING,
            "minecraft:oak_shelf": NeighborRule.SHELF,
            "minecraft:pointed_dripstone": NeighborRule.POINTED_DRIPSTONE,
            "minecraft:sulfur_spike": NeighborRule.POINTED_DRIPSTONE,
            "minecraft:oak_leaves": NeighborRule.LEAVES,
            "minecraft:lantern": NeighborRule.HANGING,
            "minecraft:oak_door": NeighborRule.MULTIBLOCK,
            "minecraft:brown_mushroom_block": NeighborRule.HUGE_MUSHROOM,
            "minecraft:grass_block": NeighborRule.SNOWY,
            "minecraft:kelp": NeighborRule.GROWING_PLANT,
            "minecraft:pale_hanging_moss": NeighborRule.HANGING_MOSS,
            "minecraft:campfire": NeighborRule.CAMPFIRE,
            "minecraft:tripwire_hook": NeighborRule.TRIPWIRE_HOOK,
            "minecraft:wall_torch": NeighborRule.SUPPORT,
            "minecraft:oak_hanging_sign": NeighborRule.SUPPORT,
            "minecraft:skeleton_wall_skull": NeighborRule.SUPPORT,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                state = catalog.default_state(name)
                self.assertEqual(catalog.info(state).neighbor_rule, expected)

    def test_rotation_metadata_is_inferred_from_official_properties(self) -> None:
        catalog = vanilla_catalog()

        log = catalog.info(catalog.default_state("minecraft:oak_log"))
        stairs = catalog.info(catalog.default_state("minecraft:oak_stairs"))
        rail = catalog.info(catalog.default_state("minecraft:rail"))

        self.assertTrue(log.rotation & RotationRule.AXIS)
        self.assertTrue(stairs.rotation & RotationRule.FACING)
        self.assertTrue(rail.rotation & RotationRule.RAIL_SHAPE)

    def test_registry_can_use_catalog_metadata_automatically(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        block_id = registry.register(BlockState.create("minecraft:stone"))

        self.assertEqual(registry.get_info(block_id).solid_faces, SolidFace.ALL)

    def test_solid_faces_come_from_the_official_state_shapes(self) -> None:
        catalog = vanilla_catalog()

        self.assertEqual(
            catalog.info(catalog.default_state("minecraft:furnace")).solid_faces,
            SolidFace.ALL,
        )
        self.assertEqual(
            catalog.info(catalog.default_state("minecraft:chest")).solid_faces,
            SolidFace.NONE,
        )
        top_slab = BlockState.create(
            "minecraft:oak_slab",
            {"type": "top", "waterlogged": "false"},
        )
        self.assertEqual(catalog.info(top_slab).solid_faces, SolidFace.UP)

    def test_extended_connectivity_metadata_comes_from_official_states(self) -> None:
        catalog = vanilla_catalog()
        redstone_block = catalog.info(
            catalog.default_state("minecraft:redstone_block")
        )
        top_slab = catalog.info(
            BlockState.create(
                "minecraft:oak_slab",
                {"type": "top", "waterlogged": "false"},
            )
        )

        self.assertTrue(redstone_block.signal_source)
        self.assertEqual(redstone_block.center_faces, SolidFace.ALL)
        self.assertEqual(top_slab.wall_cover, 0)

    def test_catalog_rejects_unknown_property_values(self) -> None:
        with self.assertRaises(ValueError):
            vanilla_catalog().validate(
                BlockState.create("minecraft:oak_fence", {"north": "maybe"})
            )

    def test_catalog_includes_flattened_official_block_tags(self) -> None:
        catalog = vanilla_catalog()

        self.assertIn(
            "logs",
            catalog.info(catalog.default_state("minecraft:oak_log")).tags,
        )
        self.assertIn(
            "wooden_fences",
            catalog.info(catalog.default_state("minecraft:oak_fence")).tags,
        )
        self.assertIn(
            "supports_chorus_plant",
            catalog.info(catalog.default_state("minecraft:end_stone")).tags,
        )
        self.assertIn(
            "wall_post_override",
            catalog.info(catalog.default_state("minecraft:torch")).tags,
        )

    def test_every_catalog_block_resolves_to_a_valid_catalog_state(self) -> None:
        catalog = vanilla_catalog()

        for name in catalog.names():
            with self.subTest(name=name):
                registry = BlockRegistry.for_vanilla_26_2()
                block_id = registry.register(catalog.default_state(name))
                blocks = {(0, 0, 0): block_id}

                NeighborResolver(registry).resolve_all(blocks)

                if blocks:
                    catalog.validate(registry.get_state(blocks[(0, 0, 0)]))


if __name__ == "__main__":
    unittest.main()
