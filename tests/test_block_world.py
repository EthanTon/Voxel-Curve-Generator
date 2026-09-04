import unittest

from src.Core.Field.block_field import BlockField
from src.Core.World.block_world import BlockTransform, BlockWorld
from src.Core.block.connectivity import NeighborResolver
from src.Core.block.connectivity_options import ConnectionOverride, ConnectivityOptions
from src.Core.block.info import BlockInfo, NeighborRule, RotationRule, SolidFace
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.modes import PlacementMode


class BlockWorldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = BlockRegistry()

    def field_with_id(self, block_id: int, position=(0, 0, 0)) -> BlockField:
        field = BlockField(self.registry.palette)
        field.set_block_id(position, block_id)
        return field

    def test_incremental_stamps_recompute_both_wall_sides(self) -> None:
        wall = self.registry.register(
            BlockState.create("minecraft:cobblestone_wall"),
            BlockInfo(neighbor_rule=NeighborRule.WALL),
        )
        field = self.field_with_id(wall)
        resolver = NeighborResolver(
            self.registry,
            options=ConnectivityOptions(
                [
                    ConnectionOverride(
                        source_blocks={"minecraft:cobblestone_wall"},
                        neighbor_blocks={"minecraft:cobblestone_wall"},
                        directions={"east"},
                        value="tall",
                        symmetric=True,
                    )
                ]
            ),
        )
        world = BlockWorld(self.registry)

        world.add_field(field, (0, 0, 0), connectivity_domain=7)
        world.resolve_dirty(resolver)
        world.add_field(field, (1, 0, 0), connectivity_domain=7)
        delta = world.resolve_dirty(resolver)

        left = self.registry.get_state(world.effective_at((0, 0, 0)).block_id)
        right = self.registry.get_state(world.effective_at((1, 0, 0)).block_id)
        self.assertEqual(left.get("east"), "tall")
        self.assertEqual(right.get("west"), "tall")
        self.assertEqual(delta.changed, {(0, 0, 0), (1, 0, 0)})

    def test_placement_domains_do_not_require_a_coordinate_domain_map(self) -> None:
        fence = self.registry.register(
            BlockState.create("minecraft:oak_fence"),
            BlockInfo(neighbor_rule=NeighborRule.FENCE),
        )
        field = self.field_with_id(fence)
        world = BlockWorld(self.registry)
        world.add_field(field, (0, 0, 0), connectivity_domain=1)
        world.add_field(field, (1, 0, 0), connectivity_domain=2)

        world.resolve_dirty(NeighborResolver(self.registry))

        left = self.registry.get_state(world.effective_at((0, 0, 0)).block_id)
        right = self.registry.get_state(world.effective_at((1, 0, 0)).block_id)
        self.assertEqual(left.get("east"), "false")
        self.assertEqual(right.get("west"), "false")

    def test_removing_an_overlap_reveals_the_lower_placement(self) -> None:
        stone = self.registry.register(BlockState.create("minecraft:stone"))
        dirt = self.registry.register(BlockState.create("minecraft:dirt"))
        world = BlockWorld(self.registry)
        world.add_field(self.field_with_id(stone))
        upper = world.add_field(self.field_with_id(dirt))

        self.assertEqual(world.source_at((0, 0, 0)).block_id, dirt)

        world.remove_placement(upper)

        self.assertEqual(world.source_at((0, 0, 0)).block_id, stone)

    def test_keep_existing_does_not_replace_a_lower_block(self) -> None:
        stone = self.registry.register(BlockState.create("minecraft:stone"))
        dirt = self.registry.register(BlockState.create("minecraft:dirt"))
        world = BlockWorld(self.registry)
        world.add_field(self.field_with_id(stone))
        world.add_field(
            self.field_with_id(dirt),
            placement_mode=PlacementMode.KEEP_EXISTING,
        )

        self.assertEqual(world.source_at((0, 0, 0)).block_id, stone)

    def test_replace_matching_treats_an_empty_cell_as_air(self) -> None:
        stone = self.registry.register(BlockState.create("minecraft:stone"))
        world = BlockWorld(self.registry)
        world.add_field(
            self.field_with_id(stone),
            placement_mode=PlacementMode.REPLACE_MATCHING,
            replace="minecraft:air",
        )

        self.assertEqual(world.source_at((0, 0, 0)).block_id, stone)

    def test_move_updates_departure_and_arrival_coordinates(self) -> None:
        stone = self.registry.register(BlockState.create("minecraft:stone"))
        world = BlockWorld(self.registry)
        placement = world.add_field(self.field_with_id(stone))
        world.drain_dirty_changes()

        changes = world.move_placement(placement, (20, 0, 0))

        self.assertIsNone(world.source_at((0, 0, 0)))
        self.assertEqual(world.source_at((20, 0, 0)).block_id, stone)
        self.assertEqual(
            {change.position for change in changes},
            {(0, 0, 0), (20, 0, 0)},
        )

    def test_moving_a_stamp_recomputes_old_and_new_neighbors(self) -> None:
        fence = self.registry.register(
            BlockState.create("minecraft:oak_fence"),
            BlockInfo(neighbor_rule=NeighborRule.FENCE),
        )
        field = self.field_with_id(fence)
        world = BlockWorld(self.registry)
        world.add_field(field)
        moving = world.add_field(field, (1, 0, 0))
        resolver = NeighborResolver(self.registry)
        world.resolve_dirty(resolver)

        world.move_placement(moving, (5, 0, 0))
        delta = world.resolve_dirty(resolver)

        left = self.registry.get_state(world.effective_at((0, 0, 0)).block_id)
        moved = self.registry.get_state(world.effective_at((5, 0, 0)).block_id)
        self.assertEqual(left.get("east"), "false")
        self.assertEqual(moved.get("west"), "false")
        self.assertIn((0, 0, 0), delta.changed)
        self.assertIn((5, 0, 0), delta.changed)

    def test_changing_placement_domain_recomputes_connectivity(self) -> None:
        fence = self.registry.register(
            BlockState.create("minecraft:oak_fence"),
            BlockInfo(neighbor_rule=NeighborRule.FENCE),
        )
        field = self.field_with_id(fence)
        world = BlockWorld(self.registry)
        world.add_field(field)
        right = world.add_field(field, (1, 0, 0))
        resolver = NeighborResolver(self.registry)
        world.resolve_dirty(resolver)

        world.set_placement_domain(right, 3)
        world.resolve_dirty(resolver)

        left = self.registry.get_state(world.effective_at((0, 0, 0)).block_id)
        changed = self.registry.get_state(world.effective_at((1, 0, 0)).block_id)
        self.assertEqual(left.get("east"), "false")
        self.assertEqual(changed.get("west"), "false")

    def test_transform_rotates_positions_and_block_state(self) -> None:
        directional = self.registry.register(
            BlockState.create("custom:directional", {"facing": "north"}),
            BlockInfo(rotation=RotationRule.FACING),
        )
        field = self.field_with_id(directional, (0, 0, -1))
        world = BlockWorld(self.registry)

        world.add_field(field, transform=BlockTransform((10, 0, 10), 1))

        cell = world.source_at((11, 0, 10))
        self.assertIsNotNone(cell)
        self.assertEqual(self.registry.get_state(cell.block_id).get("facing"), "east")

    def test_derived_deletion_revives_when_support_is_added(self) -> None:
        torch = self.registry.register(
            BlockState.create("minecraft:torch"),
            BlockInfo(neighbor_rule=NeighborRule.SUPPORT),
        )
        stone = self.registry.register(
            BlockState.create("minecraft:stone"),
            BlockInfo(solid_faces=SolidFace.ALL),
        )
        world = BlockWorld(self.registry)
        world.add_field(self.field_with_id(torch), (0, 1, 0))
        resolver = NeighborResolver(self.registry)

        world.resolve_dirty(resolver)
        self.assertIsNone(world.effective_at((0, 1, 0)))
        self.assertIsNotNone(world.source_at((0, 1, 0)))

        world.add_field(self.field_with_id(stone), (0, 0, 0))
        delta = world.resolve_dirty(resolver)

        self.assertIsNotNone(world.effective_at((0, 1, 0)))
        self.assertIn((0, 1, 0), delta.changed)

    def test_connectivity_crosses_chunk_boundaries(self) -> None:
        fence = self.registry.register(
            BlockState.create("minecraft:oak_fence"),
            BlockInfo(neighbor_rule=NeighborRule.FENCE),
        )
        field = self.field_with_id(fence)
        world = BlockWorld(self.registry)
        world.add_field(field, (15, 0, 0))
        world.add_field(field, (16, 0, 0))

        world.resolve_dirty(NeighborResolver(self.registry))

        left = self.registry.get_state(world.effective_at((15, 0, 0)).block_id)
        right = self.registry.get_state(world.effective_at((16, 0, 0)).block_id)
        self.assertEqual(left.get("east"), "true")
        self.assertEqual(right.get("west"), "true")
        self.assertEqual(len(tuple(world.iter_resolved_chunks())), 2)

    def test_diagonal_redstone_change_invalidates_the_lower_wire(self) -> None:
        wire = self.registry.register(
            BlockState.create(
                "minecraft:redstone_wire",
                {
                    "north": "none",
                    "east": "none",
                    "south": "none",
                    "west": "none",
                },
            ),
            BlockInfo(neighbor_rule=NeighborRule.REDSTONE),
        )
        stone = self.registry.register(
            BlockState.create("minecraft:stone"),
            BlockInfo(solid_faces=SolidFace.ALL),
        )
        world = BlockWorld(self.registry)
        wire_field = self.field_with_id(wire)
        world.add_field(wire_field)
        world.add_field(self.field_with_id(stone), (1, 0, 0))
        resolver = NeighborResolver(self.registry)
        world.resolve_dirty(resolver)

        world.add_field(wire_field, (1, 1, 0))
        delta = world.resolve_dirty(resolver)

        lower = self.registry.get_state(world.effective_at((0, 0, 0)).block_id)
        self.assertEqual(lower.get("east"), "side")
        self.assertIn((0, 0, 0), delta.changed)

    def test_removing_a_stamp_recomputes_the_remaining_neighbor(self) -> None:
        fence = self.registry.register(
            BlockState.create("minecraft:oak_fence"),
            BlockInfo(neighbor_rule=NeighborRule.FENCE),
        )
        field = self.field_with_id(fence)
        world = BlockWorld(self.registry)
        world.add_field(field)
        right = world.add_field(field, (1, 0, 0))
        resolver = NeighborResolver(self.registry)
        world.resolve_dirty(resolver)

        world.remove_placement(right)
        delta = world.resolve_dirty(resolver)

        left = self.registry.get_state(world.effective_at((0, 0, 0)).block_id)
        self.assertEqual(left.get("east"), "false")
        self.assertIn((0, 0, 0), delta.changed)

    def test_policy_epoch_clears_and_rebuilds_resolved_overlay(self) -> None:
        fence = self.registry.register(
            BlockState.create("minecraft:oak_fence"),
            BlockInfo(neighbor_rule=NeighborRule.FENCE),
        )
        field = self.field_with_id(fence)
        world = BlockWorld(self.registry)
        world.add_field(field)
        world.add_field(field, (1, 0, 0))
        world.resolve_dirty(NeighborResolver(self.registry))
        self.assertGreater(world.overlay_size, 0)

        resolver = NeighborResolver(
            self.registry,
            domains_can_connect=lambda left, right: False,
        )
        world.resolve_dirty(resolver)

        left = self.registry.get_state(world.effective_at((0, 0, 0)).block_id)
        right = self.registry.get_state(world.effective_at((1, 0, 0)).block_id)
        self.assertEqual(left.get("east"), "false")
        self.assertEqual(right.get("west"), "false")

    def test_explicit_air_hides_a_lower_placement_until_removed(self) -> None:
        stone = self.registry.register(BlockState.create("minecraft:stone"))
        air = self.registry.register(BlockState.create("minecraft:air"))
        world = BlockWorld(self.registry)
        world.add_field(self.field_with_id(stone))
        eraser = world.add_field(self.field_with_id(air))

        self.assertIsNone(world.source_at((0, 0, 0)))

        world.remove_placement(eraser)

        self.assertEqual(world.source_at((0, 0, 0)).block_id, stone)

    def test_local_palette_is_remapped_once_into_the_world_registry(self) -> None:
        field = BlockField()
        field.set_block((0, 0, 0), BlockState.create("minecraft:stone"))
        world = BlockWorld(self.registry)
        for x in range(100):
            world.add_field(field, (x, 0, 0))

        first = world.source_at((0, 0, 0))
        second = world.source_at((99, 0, 0))

        self.assertEqual(first.block_id, second.block_id)
        self.assertEqual(self.registry.get_state(first.block_id).name, "minecraft:stone")
        self.assertEqual(world.palette_translation_size, 1)
        self.assertEqual(len(field), 1)


if __name__ == "__main__":
    unittest.main()
