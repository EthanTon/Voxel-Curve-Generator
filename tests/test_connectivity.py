import unittest

from src.Core.block.connectivity import DependencySpec, NeighborResolver
from src.Core.block.connectivity_options import (
    ConnectionAction,
    ConnectionOverride,
    ConnectivityOptions,
)
from src.Core.block.info import BlockInfo, NeighborRule, SolidFace
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.block.vanilla import vanilla_catalog


class ConnectivityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = BlockRegistry()
        self.fence_id = self.registry.register(
            BlockState.create("minecraft:oak_fence"),
            BlockInfo(neighbor_rule=NeighborRule.FENCE),
        )
        self.resolver = NeighborResolver(self.registry)

    def test_fences_connect_to_each_other_and_solid_faces(self) -> None:
        solid_id = self.registry.register(
            BlockState.create("minecraft:stone"),
            BlockInfo(solid_faces=SolidFace.ALL),
        )
        blocks = {
            (0, 0, 0): self.fence_id,
            (1, 0, 0): self.fence_id,
            (-1, 0, 0): solid_id,
        }

        self.resolver.resolve_all(blocks)

        state = self.registry.get_state(blocks[(0, 0, 0)])
        self.assertEqual(state.get("east"), "true")
        self.assertEqual(state.get("west"), "true")
        self.assertEqual(state.get("north"), "false")
        self.assertEqual(state.get("south"), "false")

    def test_wooden_fence_does_not_connect_to_nether_brick_fence(self) -> None:
        nether_fence_id = self.registry.register(
            BlockState.create("minecraft:nether_brick_fence"),
            BlockInfo(neighbor_rule=NeighborRule.FENCE),
        )
        blocks = {(0, 0, 0): self.fence_id, (1, 0, 0): nether_fence_id}

        self.resolver.resolve_all(blocks)

        self.assertEqual(
            self.registry.get_state(blocks[(0, 0, 0)]).get("east"), "false"
        )

    def test_pane_connects_to_wall(self) -> None:
        pane_id = self.registry.register(
            BlockState.create("minecraft:iron_bars"),
            BlockInfo(neighbor_rule=NeighborRule.PANE),
        )
        wall_id = self.registry.register(
            BlockState.create("minecraft:cobblestone_wall"),
            BlockInfo(neighbor_rule=NeighborRule.WALL),
        )
        blocks = {(0, 0, 0): pane_id, (1, 0, 0): wall_id}

        self.resolver.resolve_all(blocks)

        self.assertEqual(
            self.registry.get_state(blocks[(0, 0, 0)]).get("east"), "true"
        )

    def test_connectivity_domains_prevent_connections(self) -> None:
        blocks = {
            (0, 0, 0): self.fence_id,
            (1, 0, 0): self.fence_id,
        }
        domains = {(0, 0, 0): 1, (1, 0, 0): 2}

        self.resolver.resolve_all(blocks, domains)

        left = self.registry.get_state(blocks[(0, 0, 0)])
        right = self.registry.get_state(blocks[(1, 0, 0)])
        self.assertEqual(left.get("east"), "false")
        self.assertEqual(right.get("west"), "false")

    def test_override_connects_selected_blocks_across_domains(self) -> None:
        post_id = self.registry.register(BlockState.create("custom:post"))
        options = ConnectivityOptions(
            [
                ConnectionOverride(
                    source_domains={1},
                    neighbor_domains={2},
                    source_blocks={"minecraft:oak_fence"},
                    neighbor_blocks={"custom:post"},
                    directions={"east"},
                )
            ]
        )
        resolver = NeighborResolver(self.registry, options=options)
        blocks = {(0, 0, 0): self.fence_id, (1, 0, 0): post_id}
        domains = {(0, 0, 0): 1, (1, 0, 0): 2}

        resolver.resolve_all(blocks, domains)

        self.assertEqual(
            self.registry.get_state(blocks[(0, 0, 0)]).get("east"), "true"
        )

    def test_later_override_can_force_a_disconnect(self) -> None:
        options = ConnectivityOptions(
            [
                ConnectionOverride(symmetric=True),
                ConnectionOverride(
                    action=ConnectionAction.DISCONNECT,
                    source_blocks={"minecraft:oak_fence"},
                    neighbor_blocks={"minecraft:oak_fence"},
                    directions={"east"},
                ),
            ]
        )
        resolver = NeighborResolver(self.registry, options=options)
        blocks = {(0, 0, 0): self.fence_id, (1, 0, 0): self.fence_id}

        resolver.resolve_all(blocks)

        left = self.registry.get_state(blocks[(0, 0, 0)])
        right = self.registry.get_state(blocks[(1, 0, 0)])
        self.assertEqual(left.get("east"), "false")
        self.assertEqual(right.get("west"), "true")

    def test_override_selects_nonstandard_wall_shape(self) -> None:
        wall_id = self.registry.register(
            BlockState.create("minecraft:cobblestone_wall"),
            BlockInfo(neighbor_rule=NeighborRule.WALL),
        )
        marker_id = self.registry.register(BlockState.create("custom:marker"))
        resolver = NeighborResolver(
            self.registry,
            options=ConnectivityOptions(
                [
                    ConnectionOverride(
                        source_blocks={"minecraft:cobblestone_wall"},
                        neighbor_blocks={"custom:marker"},
                        directions={"north"},
                        value="tall",
                    )
                ]
            ),
        )
        blocks = {(0, 0, 0): wall_id, (0, 0, -1): marker_id}

        resolver.resolve_all(blocks)

        self.assertEqual(
            self.registry.get_state(blocks[(0, 0, 0)]).get("north"), "tall"
        )

    def test_override_can_select_an_entire_behavior_family(self) -> None:
        wall_id = self.registry.register(
            BlockState.create("minecraft:cobblestone_wall"),
            BlockInfo(neighbor_rule=NeighborRule.WALL, behavior="wall"),
        )
        post_id = self.registry.register(
            BlockState.create("custom:post"),
            BlockInfo(behavior="decorative_post"),
        )
        resolver = NeighborResolver(
            self.registry,
            options=ConnectivityOptions(
                [
                    ConnectionOverride(
                        source_behaviors={"wall"},
                        neighbor_behaviors={"decorative_post"},
                    )
                ]
            ),
        )
        blocks = {(0, 0, 0): wall_id, (1, 0, 0): post_id}

        resolver.resolve_all(blocks)

        self.assertEqual(
            self.registry.get_state(blocks[(0, 0, 0)]).get("east"), "low"
        )

    def test_dirty_neighbor_is_updated_after_removal(self) -> None:
        blocks = {
            (0, 0, 0): self.fence_id,
            (1, 0, 0): self.fence_id,
        }
        self.resolver.resolve_all(blocks)
        blocks.pop((1, 0, 0))

        changed = self.resolver.resolve_dirty(blocks, ((1, 0, 0),))

        state = self.registry.get_state(blocks[(0, 0, 0)])
        self.assertEqual(state.get("east"), "false")
        self.assertIn((0, 0, 0), changed)

    def test_custom_handler_can_declare_a_wider_dependency_radius(self) -> None:
        source_id = self.registry.register(
            BlockState.create("custom:wide_reader", {"active": "false"})
        )
        marker_id = self.registry.register(BlockState.create("custom:marker"))
        blocks = {(0, 0, 0): source_id}

        def resolve_wide(context):
            marker_position = (context.position[0] + 2, 0, 0)
            marker = context.blocks.get(marker_position)
            same_domain = context.domains.get(marker_position, 0) == context.domain
            return BlockState.create(
                context.state.name,
                {"active": str(marker is not None and same_domain).lower()},
            )

        self.resolver.register_handler(
            NeighborRule.NONE,
            resolve_wide,
            DependencySpec(radius=(2, 0, 0), stream_halo=2),
        )
        self.resolver.resolve_all(blocks)
        blocks[(2, 0, 0)] = marker_id

        changed = self.resolver.resolve_dirty(blocks, ((2, 0, 0),))

        self.assertIn((0, 0, 0), changed)
        self.assertEqual(
            self.registry.get_state(blocks[(0, 0, 0)]).get("active"),
            "true",
        )

    def test_wall_post_and_fence_gate_in_wall_are_resolved(self) -> None:
        wall_id = self.registry.register(
            BlockState.create("minecraft:cobblestone_wall"),
            BlockInfo(neighbor_rule=NeighborRule.WALL),
        )
        gate_id = self.registry.register(
            BlockState.create(
                "minecraft:oak_fence_gate",
                {"facing": "north"},
            ),
            BlockInfo(neighbor_rule=NeighborRule.FENCE_GATE),
        )
        blocks = {(0, 0, 0): gate_id, (1, 0, 0): wall_id}

        self.resolver.resolve_all(blocks)

        gate = self.registry.get_state(blocks[(0, 0, 0)])
        wall = self.registry.get_state(blocks[(1, 0, 0)])
        self.assertEqual(gate.get("in_wall"), "true")
        self.assertEqual(wall.get("west"), "low")
        self.assertEqual(wall.get("up"), "true")

    def test_wall_uses_low_connection_next_to_a_solid_face(self) -> None:
        wall_id = self.registry.register(
            BlockState.create("minecraft:cobblestone_wall"),
            BlockInfo(neighbor_rule=NeighborRule.WALL),
        )
        solid_id = self.registry.register(
            BlockState.create("minecraft:stone"),
            BlockInfo(solid_faces=SolidFace.ALL),
        )
        blocks = {(0, 0, 0): wall_id, (0, 0, -1): solid_id}

        self.resolver.resolve_all(blocks)

        wall = self.registry.get_state(blocks[(0, 0, 0)])
        self.assertEqual(wall.get("north"), "low")
        self.assertEqual(wall.get("up"), "true")

    def test_wall_arm_becomes_tall_when_covered_from_above(self) -> None:
        wall_id = self.registry.register(
            BlockState.create("minecraft:cobblestone_wall"),
            BlockInfo(neighbor_rule=NeighborRule.WALL),
        )
        solid_id = self.registry.register(
            BlockState.create("minecraft:stone"),
            BlockInfo(solid_faces=SolidFace.ALL),
        )
        blocks = {
            (0, 0, 0): wall_id,
            (0, 0, -1): solid_id,
            (0, 1, 0): solid_id,
        }

        self.resolver.resolve_all(blocks)

        wall = self.registry.get_state(blocks[(0, 0, 0)])
        self.assertEqual(wall.get("north"), "tall")

    def test_lantern_preserves_its_selected_supported_orientation(self) -> None:
        lantern_id = self.registry.register(
            BlockState.create("minecraft:lantern", {"hanging": "true"}),
            BlockInfo(neighbor_rule=NeighborRule.HANGING),
        )
        solid_id = self.registry.register(
            BlockState.create("minecraft:stone"),
            BlockInfo(solid_faces=SolidFace.ALL),
        )
        blocks = {(0, 0, 0): lantern_id, (0, 1, 0): solid_id}

        self.resolver.resolve_all(blocks)

        self.assertEqual(
            self.registry.get_state(blocks[(0, 0, 0)]).get("hanging"), "true"
        )

    def test_pointed_dripstone_resolves_a_complete_run(self) -> None:
        dripstone_id = self.registry.register(
            BlockState.create(
                "minecraft:pointed_dripstone",
                {"vertical_direction": "up", "thickness": "tip"},
            ),
            BlockInfo(neighbor_rule=NeighborRule.POINTED_DRIPSTONE),
        )
        blocks = {(0, y, 0): dripstone_id for y in range(4)}

        self.resolver.resolve_all(blocks)

        self.assertEqual(
            [
                self.registry.get_state(blocks[(0, y, 0)]).get("thickness")
                for y in range(4)
            ],
            ["base", "middle", "frustum", "tip"],
        )

    def test_shelf_chains_are_limited_to_three_blocks(self) -> None:
        shelf_id = self.registry.register(
            BlockState.create(
                "minecraft:oak_shelf",
                {"facing": "north", "powered": "true"},
            ),
            BlockInfo(neighbor_rule=NeighborRule.SHELF),
        )
        blocks = {(x, 0, 0): shelf_id for x in range(4)}

        self.resolver.resolve_all(blocks)

        self.assertEqual(
            [
                self.registry.get_state(blocks[(x, 0, 0)]).get("side_chain")
                for x in range(4)
            ],
            ["left", "center", "right", "unconnected"],
        )

    def test_chest_runs_pair_without_creating_a_triple_chest(self) -> None:
        chest_id = self.registry.register(
            BlockState.create(
                "minecraft:chest",
                {"facing": "north", "type": "single"},
            ),
            BlockInfo(neighbor_rule=NeighborRule.CHEST),
        )
        blocks = {(x, 0, 0): chest_id for x in range(3)}

        self.resolver.resolve_all(blocks)

        self.assertEqual(
            [
                self.registry.get_state(blocks[(x, 0, 0)]).get("type")
                for x in range(3)
            ],
            ["left", "right", "single"],
        )

    def test_mature_stem_switches_to_the_attached_block(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        stem_id = registry.register(
            BlockState.create("minecraft:melon_stem", {"age": "7"})
        )
        melon_id = registry.register(BlockState.create("minecraft:melon"))
        resolver = NeighborResolver(registry)
        blocks = {(0, 0, 0): stem_id, (1, 0, 0): melon_id}

        resolver.resolve_all(blocks)

        attached_id = blocks[(0, 0, 0)]
        attached = registry.get_state(attached_id)
        self.assertEqual(attached.name, "minecraft:attached_melon_stem")
        self.assertEqual(attached.get("facing"), "east")
        self.assertEqual(
            registry.get_info(attached_id).behavior,
            "attached_stem",
        )

    def test_rail_shape_is_resolved_and_keeps_registry_info(self) -> None:
        rail_info = BlockInfo(neighbor_rule=NeighborRule.RAIL)
        rail_id = self.registry.register(
            BlockState.create("minecraft:rail", {"shape": "north_south"}),
            rail_info,
        )
        blocks = {
            (0, 0, 0): rail_id,
            (0, 0, -1): rail_id,
            (1, 0, 0): rail_id,
        }

        self.resolver.resolve_all(blocks)

        center_id = blocks[(0, 0, 0)]
        center = self.registry.get_state(center_id)
        self.assertEqual(center.get("shape"), "north_east")
        self.assertEqual(self.registry.get_info(center_id), rail_info)

    def test_powered_rail_does_not_form_a_curve(self) -> None:
        rail_id = self.registry.register(
            BlockState.create(
                "minecraft:powered_rail",
                {"shape": "north_south"},
            ),
            BlockInfo(
                neighbor_rule=NeighborRule.RAIL,
                behavior="powered_rail",
            ),
        )
        blocks = {
            (0, 0, 0): rail_id,
            (0, 0, -1): rail_id,
            (1, 0, 0): rail_id,
        }

        self.resolver.resolve_all(blocks)

        center = self.registry.get_state(blocks[(0, 0, 0)])
        self.assertEqual(center.get("shape"), "north_south")

    def test_leaf_distance_propagates_from_official_log_tag(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        log_id = registry.register(BlockState.create("minecraft:oak_log"))
        leaves_id = registry.register(BlockState.create("minecraft:oak_leaves"))
        blocks = {
            (0, 0, 0): log_id,
            (1, 0, 0): leaves_id,
            (2, 0, 0): leaves_id,
        }

        NeighborResolver(registry).resolve_all(blocks)

        self.assertEqual(
            [
                registry.get_state(blocks[(x, 0, 0)]).get("distance")
                for x in (1, 2)
            ],
            ["1", "2"],
        )

    def test_sulfur_spikes_do_not_join_pointed_dripstone(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        sulfur_id = registry.register(
            vanilla_catalog().default_state("minecraft:sulfur_spike")
        )
        dripstone_id = registry.register(
            BlockState.create(
                "minecraft:pointed_dripstone",
                {"vertical_direction": "up", "thickness": "tip"},
            )
        )
        blocks = {(0, 0, 0): sulfur_id, (0, 1, 0): dripstone_id}

        NeighborResolver(registry).resolve_all(blocks)

        self.assertEqual(
            registry.get_state(blocks[(0, 0, 0)]).get("thickness"),
            "tip",
        )
        self.assertEqual(
            registry.get_state(blocks[(0, 1, 0)]).get("thickness"),
            "tip",
        )

    def test_wall_post_override_tag_raises_a_straight_wall_post(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        wall_id = registry.register(BlockState.create("minecraft:cobblestone_wall"))
        stone_id = registry.register(BlockState.create("minecraft:stone"))
        torch_id = registry.register(BlockState.create("minecraft:torch"))
        blocks = {
            (0, 0, 0): wall_id,
            (0, 0, -1): stone_id,
            (0, 0, 1): stone_id,
            (0, 1, 0): torch_id,
        }

        NeighborResolver(registry).resolve_all(blocks)

        wall = registry.get_state(blocks[(0, 0, 0)])
        self.assertEqual(wall.get("north"), "low")
        self.assertEqual(wall.get("south"), "low")
        self.assertEqual(wall.get("up"), "true")

    def test_unsupported_block_is_removed(self) -> None:
        support_id = self.registry.register(
            BlockState.create("minecraft:torch"),
            BlockInfo(neighbor_rule=NeighborRule.SUPPORT),
        )
        blocks = {(0, 1, 0): support_id}

        changed = self.resolver.resolve_all(blocks)

        self.assertNotIn((0, 1, 0), blocks)
        self.assertEqual(changed, {(0, 1, 0)})

    def test_tripwire_connects_only_to_a_hook_facing_the_wire(self) -> None:
        tripwire_id = self.registry.register(
            BlockState.create("minecraft:tripwire"),
            BlockInfo(neighbor_rule=NeighborRule.TRIPWIRE),
        )
        hook_id = self.registry.register(
            BlockState.create("minecraft:tripwire_hook", {"facing": "west"}),
            BlockInfo(behavior="trip_wire_hook"),
        )
        blocks = {(0, 0, 0): tripwire_id, (1, 0, 0): hook_id}

        self.resolver.resolve_all(blocks)

        self.assertEqual(
            self.registry.get_state(blocks[(0, 0, 0)]).get("east"), "true"
        )

    def test_chorus_plant_connects_down_to_end_stone(self) -> None:
        chorus_id = self.registry.register(
            BlockState.create("minecraft:chorus_plant"),
            BlockInfo(neighbor_rule=NeighborRule.CHORUS),
        )
        end_stone_id = self.registry.register(
            BlockState.create("minecraft:end_stone"),
        )
        blocks = {(0, 1, 0): chorus_id, (0, 0, 0): end_stone_id}

        self.resolver.resolve_all(blocks)

        self.assertEqual(
            self.registry.get_state(blocks[(0, 1, 0)]).get("down"), "true"
        )

    def test_fire_uses_catalog_flammability(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        fire_id = registry.register(
            vanilla_catalog().default_state("minecraft:fire")
        )
        planks_id = registry.register(BlockState.create("minecraft:oak_planks"))
        stone_id = registry.register(BlockState.create("minecraft:stone"))
        blocks = {
            (0, 0, 0): fire_id,
            (1, 0, 0): planks_id,
            (-1, 0, 0): stone_id,
        }

        NeighborResolver(registry).resolve_all(blocks)

        fire = registry.get_state(blocks[(0, 0, 0)])
        self.assertEqual(fire.get("east"), "true")
        self.assertEqual(fire.get("west"), "false")

    def test_fence_respects_vanilla_connection_exceptions(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        fence_id = registry.register(
            vanilla_catalog().default_state("minecraft:oak_fence")
        )
        pumpkin_id = registry.register(BlockState.create("minecraft:pumpkin"))
        blocks = {(0, 0, 0): fence_id, (1, 0, 0): pumpkin_id}

        NeighborResolver(registry).resolve_all(blocks)

        fence = registry.get_state(blocks[(0, 0, 0)])
        self.assertEqual(fence.get("east"), "false")

    def test_huge_mushroom_culls_only_matching_internal_faces(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        mushroom = registry.register(
            vanilla_catalog().default_state("minecraft:brown_mushroom_block")
        )
        stem = registry.register(
            vanilla_catalog().default_state("minecraft:mushroom_stem")
        )
        blocks = {(0, 0, 0): mushroom, (1, 0, 0): mushroom, (-1, 0, 0): stem}

        NeighborResolver(registry).resolve_all(blocks)

        state = registry.get_state(blocks[(0, 0, 0)])
        self.assertEqual(state.get("east"), "false")
        self.assertEqual(state.get("west"), "true")

    def test_snowy_surface_uses_the_official_snow_tag(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        grass = registry.register(vanilla_catalog().default_state("minecraft:grass_block"))
        snow = registry.register(vanilla_catalog().default_state("minecraft:snow"))
        blocks = {(0, 0, 0): grass, (0, 1, 0): snow}

        NeighborResolver(registry).resolve_all(blocks)

        self.assertEqual(registry.get_state(blocks[(0, 0, 0)]).get("snowy"), "true")

    def test_growing_plant_head_and_body_are_reclassified(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        kelp = registry.register(vanilla_catalog().default_state("minecraft:kelp"))
        blocks = {(0, 0, 0): kelp, (0, 1, 0): kelp}

        NeighborResolver(registry).resolve_all(blocks)

        self.assertEqual(registry.get_state(blocks[(0, 0, 0)]).name, "minecraft:kelp_plant")
        self.assertEqual(registry.get_state(blocks[(0, 1, 0)]).name, "minecraft:kelp")

    def test_hanging_moss_resolves_tip_and_requires_support(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        moss = registry.register(
            vanilla_catalog().default_state("minecraft:pale_hanging_moss")
        )
        stone = registry.register(BlockState.create("minecraft:stone"))
        blocks = {(0, 0, 0): moss, (0, 1, 0): moss, (0, 2, 0): stone}

        NeighborResolver(registry).resolve_all(blocks)

        self.assertEqual(registry.get_state(blocks[(0, 0, 0)]).get("tip"), "true")
        self.assertEqual(registry.get_state(blocks[(0, 1, 0)]).get("tip"), "false")

    def test_campfire_signal_state_tracks_hay_below(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        campfire = registry.register(vanilla_catalog().default_state("minecraft:campfire"))
        hay = registry.register(BlockState.create("minecraft:hay_block"))
        blocks = {(0, 1, 0): campfire, (0, 0, 0): hay}

        NeighborResolver(registry).resolve_all(blocks)

        self.assertEqual(registry.get_state(blocks[(0, 1, 0)]).get("signal_fire"), "true")

    def test_multiblock_door_removes_orphans_and_syncs_upper_half(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        lower_state = vanilla_catalog().default_state("minecraft:oak_door")
        lower_properties = dict(lower_state.properties)
        lower_properties.update({"half": "lower", "open": "true"})
        upper_properties = dict(lower_properties)
        upper_properties.update({"half": "upper", "open": "false"})
        lower = registry.register(BlockState.create(lower_state.name, lower_properties))
        upper = registry.register(BlockState.create(lower_state.name, upper_properties))
        stone = registry.register(BlockState.create("minecraft:stone"))
        blocks = {
            (0, -1, 0): stone,
            (0, 0, 0): lower,
            (0, 1, 0): upper,
            (3, 1, 0): upper,
        }

        NeighborResolver(registry).resolve_all(blocks)

        self.assertEqual(registry.get_state(blocks[(0, 1, 0)]).get("open"), "true")
        self.assertNotIn((3, 1, 0), blocks)

    def test_complete_tripwire_line_updates_hooks_and_wire(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        catalog = vanilla_catalog()
        west_hook_state = catalog.default_state("minecraft:tripwire_hook")
        west_props = dict(west_hook_state.properties)
        west_props["facing"] = "east"
        east_props = dict(west_props)
        east_props["facing"] = "west"
        west_hook = registry.register(BlockState.create(west_hook_state.name, west_props))
        east_hook = registry.register(BlockState.create(west_hook_state.name, east_props))
        wire_state = catalog.default_state("minecraft:tripwire")
        wire = registry.register(wire_state)
        stone = registry.register(BlockState.create("minecraft:stone"))
        blocks = {
            (-1, 0, 0): stone,
            (0, 0, 0): west_hook,
            (1, 0, 0): wire,
            (2, 0, 0): wire,
            (3, 0, 0): east_hook,
            (4, 0, 0): stone,
        }

        resolver = NeighborResolver(registry)
        resolver.resolve_all(blocks)

        self.assertEqual(registry.get_state(blocks[(0, 0, 0)]).get("attached"), "true")
        self.assertEqual(registry.get_state(blocks[(0, 0, 0)]).get("powered"), "false")
        self.assertEqual(registry.get_state(blocks[(1, 0, 0)]).get("attached"), "true")

        powered_props = dict(registry.get_state(blocks[(1, 0, 0)]).properties)
        powered_props["powered"] = "true"
        blocks[(1, 0, 0)] = registry.register(
            BlockState.create(wire_state.name, powered_props)
        )
        resolver.resolve_dirty(blocks, ((1, 0, 0),))

        self.assertEqual(registry.get_state(blocks[(3, 0, 0)]).get("powered"), "true")

    def test_a_domain_not_joining_itself_keeps_authored_connections(self) -> None:
        # The wall was authored joined north and east; only north has a
        # neighbour. Joining itself, the wall is remade from its neighbours;
        # not, it keeps what it was authored with and adds nothing.
        wall = self.registry.register(
            BlockState.create(
                "minecraft:andesite_wall",
                {"north": "tall", "east": "tall", "south": "none", "west": "none", "up": "true"},
            ),
            BlockInfo(neighbor_rule=NeighborRule.WALL),
        )
        plain = self.registry.register(
            BlockState.create("minecraft:andesite_wall"),
            BlockInfo(neighbor_rule=NeighborRule.WALL),
        )
        joined = {(0, 0, 0): wall, (0, 0, -1): plain, (0, 0, 1): plain}
        kept = dict(joined)

        NeighborResolver(self.registry).resolve_all(joined)
        NeighborResolver(self.registry, domains_can_connect=lambda a, b: False).resolve_all(kept)

        remade = self.registry.get_state(joined[(0, 0, 0)])
        authored = self.registry.get_state(kept[(0, 0, 0)])
        self.assertEqual([remade.get(side) for side in ("north", "east", "south")], ["low", "none", "low"])
        self.assertEqual([authored.get(side) for side in ("north", "east", "south")], ["tall", "tall", "none"])

    def test_a_wall_kept_as_drawn_keeps_its_post_as_drawn(self) -> None:
        # Three sides would raise a post, but it was drawn without one.
        wall = self.registry.register(
            BlockState.create(
                "minecraft:andesite_wall",
                {"north": "tall", "east": "tall", "south": "tall", "west": "none", "up": "false"},
            ),
            BlockInfo(neighbor_rule=NeighborRule.WALL),
        )
        blocks = {(0, 0, 0): wall}

        NeighborResolver(self.registry, domains_can_connect=lambda a, b: False).resolve_all(blocks)

        self.assertEqual(self.registry.get_state(blocks[(0, 0, 0)]).get("up"), "false")

    def test_a_block_of_its_own_domain_holds_it_up(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        button = registry.register(BlockState.create(
            "minecraft:stone_button", {"face": "floor", "facing": "east", "powered": "false"},
        ))
        bars = registry.register(BlockState.create("minecraft:iron_bars"))
        blocks = {(0, 0, 0): bars, (0, 1, 0): button}

        NeighborResolver(registry, domains_can_connect=lambda a, b: False).resolve_all(blocks)

        self.assertIn((0, 1, 0), blocks)

    def test_vanilla_wall_attachable_requires_center_support(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        torch_state = vanilla_catalog().default_state("minecraft:wall_torch")
        props = dict(torch_state.properties)
        props["facing"] = "east"
        torch = registry.register(BlockState.create(torch_state.name, props))
        stone = registry.register(BlockState.create("minecraft:stone"))
        supported = {(-1, 0, 0): stone, (0, 0, 0): torch}
        orphan = {(0, 0, 0): torch}

        NeighborResolver(registry).resolve_all(supported)
        NeighborResolver(registry).resolve_all(orphan)

        self.assertIn((0, 0, 0), supported)
        self.assertNotIn((0, 0, 0), orphan)

    def test_generic_override_selects_tags_properties_and_sets_properties(self) -> None:
        source = self.registry.register(
            BlockState.create("custom:lamp", {"mode": "auto", "lit": "false"}),
            BlockInfo(tags=frozenset({"custom_lamps"})),
        )
        neighbor = self.registry.register(
            BlockState.create("custom:sensor", {"active": "true"}),
            BlockInfo(tags=frozenset({"custom_sensors"})),
        )
        resolver = NeighborResolver(
            self.registry,
            options=ConnectivityOptions(
                [
                    ConnectionOverride(
                        source_tags={"custom_lamps"},
                        neighbor_tags={"custom_sensors"},
                        source_properties={"mode": {"auto"}},
                        neighbor_properties={"active": {"true"}},
                        directions={"east"},
                        set_properties={"lit": "true"},
                    )
                ]
            ),
        )
        blocks = {(0, 0, 0): source, (1, 0, 0): neighbor}

        resolver.resolve_all(blocks)

        self.assertEqual(self.registry.get_state(blocks[(0, 0, 0)]).get("lit"), "true")

    def test_redstone_connects_to_generic_signal_source_and_completes_line(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        wire = registry.register(vanilla_catalog().default_state("minecraft:redstone_wire"))
        source = registry.register(BlockState.create("minecraft:redstone_block"))
        blocks = {(0, 0, 0): wire, (1, 0, 0): source}

        NeighborResolver(registry).resolve_all(blocks)

        state = registry.get_state(blocks[(0, 0, 0)])
        self.assertEqual(state.get("east"), "side")
        self.assertEqual(state.get("west"), "side")

    def test_isolated_redstone_preserves_selected_dot_or_cross(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        dot_state = vanilla_catalog().default_state("minecraft:redstone_wire")
        cross_properties = dict(dot_state.properties)
        cross_properties.update(
            {direction: "side" for direction in ("north", "east", "south", "west")}
        )
        dot = registry.register(dot_state)
        cross = registry.register(BlockState.create(dot_state.name, cross_properties))
        dot_blocks = {(0, 0, 0): dot}
        cross_blocks = {(0, 0, 0): cross}

        NeighborResolver(registry).resolve_all(dot_blocks)
        NeighborResolver(registry).resolve_all(cross_blocks)

        self.assertTrue(
            all(
                registry.get_state(dot_blocks[(0, 0, 0)]).get(direction) == "none"
                for direction in ("north", "east", "south", "west")
            )
        )
        self.assertTrue(
            all(
                registry.get_state(cross_blocks[(0, 0, 0)]).get(direction) == "side"
                for direction in ("north", "east", "south", "west")
            )
        )

    def test_rail_t_junction_keeps_at_most_two_connections(self) -> None:
        rail = self.registry.register(
            BlockState.create("minecraft:rail", {"shape": "north_south"}),
            BlockInfo(neighbor_rule=NeighborRule.RAIL),
        )
        blocks = {
            (0, 0, 0): rail,
            (0, 0, -1): rail,
            (0, 0, 1): rail,
            (1, 0, 0): rail,
        }

        self.resolver.resolve_all(blocks)

        self.assertEqual(self.registry.get_state(blocks[(0, 0, 0)]).get("shape"), "north_south")

    def test_wall_uses_collision_coverage_not_only_sturdy_face(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        wall = registry.register(vanilla_catalog().default_state("minecraft:cobblestone_wall"))
        stone = registry.register(BlockState.create("minecraft:stone"))
        slab = registry.register(
            BlockState.create("minecraft:oak_slab", {"type": "top", "waterlogged": "false"})
        )
        blocks = {(0, 0, 0): wall, (0, 0, -1): stone, (0, 1, 0): slab}

        NeighborResolver(registry).resolve_all(blocks)

        self.assertEqual(registry.get_state(blocks[(0, 0, 0)]).get("north"), "low")

    def test_unsupported_scaffolding_is_removed(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        scaffold = registry.register(vanilla_catalog().default_state("minecraft:scaffolding"))
        blocks = {(0, 0, 0): scaffold}

        NeighborResolver(registry).resolve_all(blocks)

        self.assertFalse(blocks)

    def test_pale_moss_carpet_keeps_base_and_vertical_side_continuity(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        catalog = vanilla_catalog()
        base_state = catalog.default_state("minecraft:pale_moss_carpet")
        base = registry.register(base_state)
        topper_props = dict(base_state.properties)
        topper_props.update({"bottom": "false", "east": "low"})
        topper = registry.register(BlockState.create(base_state.name, topper_props))
        stone = registry.register(BlockState.create("minecraft:stone"))
        blocks = {
            (0, -1, 0): stone,
            (0, 0, 0): base,
            (0, 1, 0): topper,
            (1, 0, 0): stone,
            (1, 1, 0): stone,
        }

        NeighborResolver(registry).resolve_all(blocks)

        lower = registry.get_state(blocks[(0, 0, 0)])
        upper = registry.get_state(blocks[(0, 1, 0)])
        self.assertEqual(lower.get("bottom"), "true")
        self.assertEqual(lower.get("east"), "tall")
        self.assertEqual(upper.get("bottom"), "false")
        self.assertEqual(upper.get("east"), "low")

    def test_bed_requires_matching_other_half(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        state = vanilla_catalog().default_state("minecraft:red_bed")
        foot_props = dict(state.properties)
        foot_props.update({"part": "foot", "facing": "east", "occupied": "true"})
        head_props = dict(foot_props)
        head_props.update({"part": "head", "occupied": "false"})
        foot = registry.register(BlockState.create(state.name, foot_props))
        head = registry.register(BlockState.create(state.name, head_props))
        blocks = {(0, 0, 0): foot, (1, 0, 0): head, (4, 0, 0): head}

        NeighborResolver(registry).resolve_all(blocks)

        self.assertEqual(registry.get_state(blocks[(1, 0, 0)]).get("occupied"), "true")
        self.assertNotIn((4, 0, 0), blocks)

    def test_unsupported_lantern_is_removed_instead_of_flipped(self) -> None:
        registry = BlockRegistry.for_vanilla_26_2()
        state = vanilla_catalog().default_state("minecraft:lantern")
        properties = dict(state.properties)
        properties["hanging"] = "false"
        lantern = registry.register(BlockState.create(state.name, properties))
        stone = registry.register(BlockState.create("minecraft:stone"))
        blocks = {(0, 0, 0): lantern, (0, 1, 0): stone}

        NeighborResolver(registry).resolve_all(blocks)

        self.assertNotIn((0, 0, 0), blocks)


if __name__ == "__main__":
    unittest.main()


class DiagonalStairTests(unittest.TestCase):
    # A kerb of east-facing stairs running north (-z) that steps sideways one
    # block. Vanilla leaves both stairs either side of the step straight; a
    # domain that joins stairs diagonally turns them into a corner, with no
    # block added.

    def setUp(self) -> None:
        self.registry = BlockRegistry.for_vanilla_26_2()

    def stair(self):
        return self.registry.register(BlockState.create(
            "minecraft:stone_brick_stairs",
            {"facing": "east", "half": "bottom", "shape": "straight", "waterlogged": "false"},
        ))

    def shapes(self, cells, joins):
        blocks = {cell: self.stair() for cell in cells}
        NeighborResolver(
            self.registry, stairs_join_diagonals=(lambda domain: True) if joins else None
        ).resolve_all(blocks)
        return {cell: self.registry.get_state(blocks[cell]).get("shape") for cell in cells}

    def test_vanilla_leaves_a_sideways_step_straight(self) -> None:
        cells = [(0, 0, 0), (0, 0, -1), (1, 0, -2), (1, 0, -3)]
        self.assertEqual(set(self.shapes(cells, joins=False).values()), {"straight"})

    def test_a_step_outwards_is_an_outer_corner_then_an_inner_one(self) -> None:
        # Outwards is the way the stairs face: the one before the step turns
        # out towards the next, the one after turns in from the last.
        shapes = self.shapes([(0, 0, 0), (0, 0, -1), (1, 0, -2), (1, 0, -3)], joins=True)
        self.assertEqual(shapes[(0, 0, -1)], "outer_left")
        self.assertEqual(shapes[(1, 0, -2)], "inner_right")
        self.assertEqual(shapes[(0, 0, 0)], "straight")
        self.assertEqual(shapes[(1, 0, -3)], "straight")

    def test_a_step_inwards_is_an_inner_corner_then_an_outer_one(self) -> None:
        shapes = self.shapes([(0, 0, 0), (0, 0, -1), (-1, 0, -2), (-1, 0, -3)], joins=True)
        self.assertEqual(shapes[(0, 0, -1)], "inner_left")
        self.assertEqual(shapes[(-1, 0, -2)], "outer_right")

    def test_a_straight_run_stays_straight(self) -> None:
        shapes = self.shapes([(0, 0, z) for z in range(0, -4, -1)], joins=True)
        self.assertEqual(set(shapes.values()), {"straight"})

    def test_a_stray_in_a_diagonal_run_takes_the_run_s_facing(self) -> None:
        # A diagonal run of west-facing stairs with one south-facing stair in
        # it, as a join leaves: the stray joins the run.
        blocks = {}
        for step in (-2, -1, 1, 2):
            blocks[(step, 0, step)] = self.registry.register(BlockState.create(
                "minecraft:stone_brick_stairs",
                {"facing": "west", "half": "bottom", "shape": "straight", "waterlogged": "false"},
            ))
        blocks[(0, 0, 0)] = self.registry.register(BlockState.create(
            "minecraft:stone_brick_stairs",
            {"facing": "south", "half": "bottom", "shape": "straight", "waterlogged": "false"},
        ))
        NeighborResolver(
            self.registry, stairs_join_diagonals=lambda domain: True
        ).resolve_all(blocks)
        stray = self.registry.get_state(blocks[(0, 0, 0)])
        before = self.registry.get_state(blocks[(-1, 0, -1)])
        self.assertEqual(stray.get("facing"), "west")
        self.assertEqual(stray.get("shape"), before.get("shape"))


class EdgeOverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = BlockRegistry()
        self.fence = self.registry.register(
            BlockState.create("minecraft:oak_fence"),
            BlockInfo(neighbor_rule=NeighborRule.FENCE),
        )
        self.marker = self.registry.register(BlockState.create("custom:marker"))

    def test_exact_edge_offset_connects_selected_side_only(self) -> None:
        resolver = NeighborResolver(self.registry, options=ConnectivityOptions([
            ConnectionOverride(
                source_blocks={"minecraft:oak_fence"},
                neighbor_blocks={"custom:marker"},
                directions={"east"},
                offsets={(1, 0, -1)},
                max_distance=2 ** 0.5,
            ),
        ]))
        blocks = {(0, 0, 0): self.fence, (1, 0, -1): self.marker}
        resolver.resolve_all(blocks)
        state = self.registry.get_state(blocks[(0, 0, 0)])
        self.assertEqual(state.get("east"), "true")
        self.assertEqual(state.get("north"), "false")

        blocks.pop((1, 0, -1))
        resolver.resolve_dirty(blocks, [(1, 0, -1)])
        self.assertEqual(self.registry.get_state(blocks[(0, 0, 0)]).get("east"), "false")

    def test_face_default_does_not_reach_edge_or_corner(self) -> None:
        resolver = NeighborResolver(self.registry, options=ConnectivityOptions([
            ConnectionOverride(
                source_blocks={"minecraft:oak_fence"},
                neighbor_blocks={"custom:marker"},
                directions={"east"},
            ),
        ]))
        for offset in ((1, 0, -1), (1, 1, 1)):
            blocks = {(0, 0, 0): self.fence, offset: self.marker}
            resolver.resolve_all(blocks)
            self.assertEqual(self.registry.get_state(blocks[(0, 0, 0)]).get("east"), "false")

    def test_vertical_edge_can_patch_a_custom_state(self) -> None:
        lamp = self.registry.register(BlockState.create("custom:lamp", {"lit": "false"}))
        resolver = NeighborResolver(self.registry, options=ConnectivityOptions([
            ConnectionOverride(
                source_blocks={"custom:lamp"},
                neighbor_blocks={"custom:marker"},
                directions={"up"},
                offsets={(1, 1, 0)},
                max_distance=2 ** 0.5,
                set_properties={"lit": "true"},
            ),
        ]))
        blocks = {(0, 0, 0): lamp, (1, 1, 0): self.marker}
        resolver.resolve_all(blocks)
        self.assertEqual(self.registry.get_state(blocks[(0, 0, 0)]).get("lit"), "true")

    def test_invalid_offset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "offsets"):
            ConnectionOverride(offsets={(1, 1, 1)}, max_distance=2 ** 0.5)

    def test_wall_edge_override_uses_requested_height(self) -> None:
        wall = self.registry.register(
            BlockState.create("minecraft:cobblestone_wall"),
            BlockInfo(neighbor_rule=NeighborRule.WALL),
        )
        resolver = NeighborResolver(self.registry, options=ConnectivityOptions([
            ConnectionOverride(
                source_blocks={"minecraft:cobblestone_wall"},
                neighbor_blocks={"custom:marker"},
                directions={"east"},
                max_distance=2 ** 0.5,
                value="tall",
            ),
        ]))
        blocks = {(0, 0, 0): wall, (1, 0, -1): self.marker}
        resolver.resolve_all(blocks)
        self.assertEqual(self.registry.get_state(blocks[(0, 0, 0)]).get("east"), "tall")

    def test_symmetric_edge_rule_matches_reverse_offset(self) -> None:
        resolver = NeighborResolver(self.registry, options=ConnectivityOptions([
            ConnectionOverride(
                source_blocks={"custom:marker"},
                neighbor_blocks={"minecraft:oak_fence"},
                directions={"west"},
                offsets={(-1, 0, 1)},
                max_distance=2 ** 0.5,
                symmetric=True,
            ),
        ]))
        blocks = {(0, 0, 0): self.fence, (1, 0, -1): self.marker}
        resolver.resolve_all(blocks)
        self.assertEqual(self.registry.get_state(blocks[(0, 0, 0)]).get("east"), "true")
