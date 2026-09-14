import unittest

from src.Core.Domain import (
    ConnectivityRules,
    DomainInteraction,
    DomainRegistry,
    PlacementRules,
    TerrainRules,
)
from src.Core.Field.block_field import BlockField
from src.Core.World.block_world import BlockWorld
from src.Core.block.connectivity_options import ConnectionOverride
from src.Core.block.info import BlockInfo, NeighborRule, SolidFace
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.modes import PlacementMode, TerrainMaskMode


class DomainInheritanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.domains = DomainRegistry()

    def test_unset_rules_are_inherited_from_the_nearest_parent(self) -> None:
        structure = self.domains.create(
            "structure",
            placement=PlacementRules(mode=PlacementMode.KEEP_EXISTING, protected=True),
            connectivity=ConnectivityRules(connects_to_self=False),
        )
        house = self.domains.create(
            "house",
            parents=[structure],
            placement=PlacementRules(protected=False),
        )

        profile = self.domains.profile(house)

        self.assertEqual(profile.lineage, (house.domain_id, structure.domain_id))
        self.assertEqual(profile.placement_mode, PlacementMode.KEEP_EXISTING)
        self.assertFalse(profile.protected)
        self.assertFalse(profile.connects_to_self)

    def test_unset_rules_fall_back_to_world_defaults(self) -> None:
        profile = self.domains.profile(self.domains.create("plain"))

        self.assertEqual(profile.placement_mode, PlacementMode.OVERWRITE)
        self.assertEqual(profile.replace, "minecraft:air")
        self.assertEqual(profile.terrain_mask_mode, TerrainMaskMode.NONE)
        self.assertTrue(profile.connects_to_self)
        self.assertFalse(profile.connects_to_any)

    def test_breadth_first_lineage_puts_parents_before_grandparents(self) -> None:
        base = self.domains.create("base", properties={"tier": "base"})
        left = self.domains.create("left", parents=["base"])
        right = self.domains.create("right", parents=["base"], properties={"tier": "right"})
        child = self.domains.create("child", parents=[left, right])

        self.assertEqual(
            self.domains.compile().lineage("child"),
            (child.domain_id, left.domain_id, right.domain_id, base.domain_id),
        )
        self.assertEqual(self.domains.profile(child).properties["tier"], "right")

    def test_properties_grow_through_interactions(self) -> None:
        rail = self.domains.create("rail", properties={"speed": 8})
        redstone = self.domains.create("redstone")
        self.domains.interact(
            rail,
            redstone,
            grants={"powered": True, "speed": 1},
            properties={"signal": "strong"},
        )

        rail_profile = self.domains.profile(rail)
        redstone_profile = self.domains.profile(redstone)

        # A domain's own properties outrank anything it is granted.
        self.assertEqual(dict(rail_profile.properties), {"powered": True, "speed": 8})
        self.assertTrue(redstone_profile.properties["powered"])
        self.assertEqual(
            dict(self.domains.compile().interaction_properties("redstone", "rail")),
            {"signal": "strong"},
        )

    def test_one_way_grants_only_reach_the_source(self) -> None:
        road = self.domains.create("road")
        town = self.domains.create("town")
        self.domains.interact(road, town, grants={"paved": True}, symmetric=False)

        self.assertIn("paved", self.domains.profile(road).properties)
        self.assertNotIn("paved", self.domains.profile(town).properties)

    def test_registry_validates_names_parents_and_cycles(self) -> None:
        first = self.domains.create("first")
        second = self.domains.create("second", parents=[first])

        with self.assertRaises(ValueError):
            self.domains.create("first")
        with self.assertRaises(KeyError):
            self.domains.create("orphan", parents=["missing"])
        with self.assertRaises(ValueError):
            self.domains.update(first, parents=[second])
        self.assertEqual(self.domains.get("first").parents, ())

    def test_automatic_ids_leave_the_world_default_domain_free(self) -> None:
        self.assertEqual(self.domains.create("first").domain_id, 1)
        self.assertEqual(self.domains.create("zero", domain_id=0).domain_id, 0)
        self.assertEqual(self.domains.create("next").domain_id, 2)

    def test_scoped_overrides_must_not_name_domains(self) -> None:
        with self.assertRaises(ValueError):
            DomainInteraction(1, 2, overrides=[ConnectionOverride(source_domains={1})])
        with self.assertRaises(ValueError):
            ConnectivityRules(overrides=[ConnectionOverride(source_domains={1})])

    def test_compile_is_cached_until_the_registry_changes(self) -> None:
        self.domains.create("first")
        policy = self.domains.compile()

        self.assertIs(self.domains.compile(), policy)
        self.domains.create("second")
        self.assertIsNot(self.domains.compile(), policy)
        self.assertNotIn("second", policy)


class DomainConnectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.domains = DomainRegistry()

    def test_same_domain_connects_unless_isolated(self) -> None:
        plain = self.domains.create("plain")
        isolated = self.domains.create(
            "isolated", connectivity=ConnectivityRules(connects_to_self=False)
        )
        policy = self.domains.compile()

        self.assertTrue(policy.can_connect(plain.domain_id, plain.domain_id))
        self.assertFalse(policy.can_connect(isolated.domain_id, isolated.domain_id))
        self.assertFalse(policy.can_connect(plain.domain_id, isolated.domain_id))

    def test_unregistered_domains_fall_back_to_equality(self) -> None:
        policy = self.domains.compile()

        self.assertTrue(policy.can_connect(0, 0))
        self.assertFalse(policy.can_connect(0, 5))

    def test_parent_interaction_joins_every_child(self) -> None:
        transport = self.domains.create("transport")
        rail = self.domains.create("rail", parents=[transport])
        road = self.domains.create("road", parents=[transport])
        self.domains.interact(transport, transport, connect=True)
        policy = self.domains.compile()

        self.assertTrue(policy.can_connect(rail.domain_id, road.domain_id))

    def test_more_specific_interaction_wins(self) -> None:
        transport = self.domains.create("transport")
        rail = self.domains.create("rail", parents=[transport])
        road = self.domains.create("road", parents=[transport])
        self.domains.interact(rail, road, connect=False)
        self.domains.interact(transport, transport, connect=True)
        policy = self.domains.compile()

        self.assertFalse(policy.can_connect(rail.domain_id, road.domain_id))
        self.assertFalse(policy.can_connect(road.domain_id, rail.domain_id))
        self.assertTrue(policy.can_connect(rail.domain_id, rail.domain_id))

    def test_directional_interaction_only_applies_one_way(self) -> None:
        vine = self.domains.create("vine")
        wall = self.domains.create("wall")
        self.domains.interact(vine, wall, connect=True, symmetric=False)
        policy = self.domains.compile()

        self.assertTrue(policy.can_connect(vine.domain_id, wall.domain_id))
        self.assertFalse(policy.can_connect(wall.domain_id, vine.domain_id))

    def test_later_interaction_breaks_a_tie(self) -> None:
        left = self.domains.create("left")
        right = self.domains.create("right")
        self.domains.interact(left, right, connect=True)
        self.domains.interact(right, left, connect=False)

        self.assertFalse(self.domains.compile().can_connect(left.domain_id, right.domain_id))

    def test_connects_to_any_is_overruled_by_an_explicit_interaction(self) -> None:
        terrain = self.domains.create(
            "terrain", connectivity=ConnectivityRules(connects_to_any=True)
        )
        fence = self.domains.create("fence")
        glass = self.domains.create("glass")
        self.domains.interact(glass, terrain, connect=False)
        policy = self.domains.compile()

        self.assertTrue(policy.can_connect(fence.domain_id, terrain.domain_id))
        self.assertFalse(policy.can_connect(glass.domain_id, terrain.domain_id))


class DomainWorldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = BlockRegistry()
        self.domains = DomainRegistry()
        self.fence = self.registry.register(
            BlockState.create("minecraft:oak_fence"),
            BlockInfo(neighbor_rule=NeighborRule.FENCE),
        )
        self.stone = self.registry.register(
            BlockState.create("minecraft:stone"),
            BlockInfo(solid_faces=SolidFace.ALL),
        )
        self.dirt = self.registry.register(BlockState.create("minecraft:dirt"))

    def field_with_id(self, block_id: int, position=(0, 0, 0)) -> BlockField:
        field = BlockField(self.registry.palette)
        field.set_block_id(position, block_id)
        return field

    def state_at(self, world: BlockWorld, position) -> BlockState:
        return self.registry.get_state(world.effective_at(position).block_id)

    def test_interaction_connects_fences_across_domains(self) -> None:
        garden = self.domains.create("garden")
        farm = self.domains.create("farm")
        self.domains.interact(garden, farm, connect=True)
        world = BlockWorld(self.registry)
        self.domains.place(world, self.field_with_id(self.fence), garden, (0, 0, 0))
        self.domains.place(world, self.field_with_id(self.fence), farm, (1, 0, 0))

        world.resolve_dirty(self.domains.resolver(self.registry))

        self.assertEqual(self.state_at(world, (0, 0, 0)).get("east"), "true")
        self.assertEqual(self.state_at(world, (1, 0, 0)).get("west"), "true")

    def test_boundary_patch_marks_only_the_source_side(self) -> None:
        path = self.domains.create("path")
        field_domain = self.domains.create("field")
        self.domains.interact(
            path,
            field_domain,
            set_properties={"edge": "true"},
            symmetric=False,
        )
        world = BlockWorld(self.registry)
        self.domains.place(world, self.field_with_id(self.stone), path, (0, 0, 0))
        self.domains.place(world, self.field_with_id(self.dirt), field_domain, (1, 0, 0))
        self.domains.place(world, self.field_with_id(self.stone), path, (-5, 0, 0))

        world.resolve_dirty(self.domains.resolver(self.registry))

        self.assertEqual(self.state_at(world, (0, 0, 0)).get("edge"), "true")
        self.assertIsNone(self.state_at(world, (1, 0, 0)).get("edge"))
        self.assertIsNone(self.state_at(world, (-5, 0, 0)).get("edge"))

    def test_domain_override_is_scoped_to_its_members(self) -> None:
        wall_domain = self.domains.create(
            "walls",
            connectivity=ConnectivityRules(
                overrides=[
                    ConnectionOverride(
                        source_blocks={"minecraft:stone"},
                        set_properties={"tagged": "true"},
                    )
                ]
            ),
        )
        tower = self.domains.create("tower", parents=[wall_domain])
        other = self.domains.create("other")
        world = BlockWorld(self.registry)
        self.domains.place(world, self.field_with_id(self.stone), tower, (0, 0, 0))
        self.domains.place(world, self.field_with_id(self.stone), tower, (1, 0, 0))
        self.domains.place(world, self.field_with_id(self.stone), other, (0, 5, 0))
        self.domains.place(world, self.field_with_id(self.stone), other, (1, 5, 0))

        world.resolve_dirty(self.domains.resolver(self.registry))

        self.assertEqual(self.state_at(world, (0, 0, 0)).get("tagged"), "true")
        self.assertIsNone(self.state_at(world, (0, 5, 0)).get("tagged"))

    def test_place_uses_domain_defaults_and_accepts_overrides(self) -> None:
        decoration = self.domains.create(
            "decoration",
            placement=PlacementRules(mode=PlacementMode.KEEP_EXISTING),
        )
        world = BlockWorld(self.registry)
        world.add_field(self.field_with_id(self.stone))

        kept = self.domains.place(world, self.field_with_id(self.dirt), "decoration")
        self.assertEqual(world.source_at((0, 0, 0)).block_id, self.stone)
        self.assertEqual(world.get_placement(kept).connectivity_domain, decoration.domain_id)

        self.domains.place(
            world,
            self.field_with_id(self.dirt),
            decoration,
            placement_mode=PlacementMode.OVERWRITE,
        )
        self.assertEqual(world.source_at((0, 0, 0)).block_id, self.dirt)

    def test_protected_domain_keeps_its_blocks(self) -> None:
        building = self.domains.create(
            "building", placement=PlacementRules(protected=True)
        )
        road = self.domains.create("road")
        world = BlockWorld(self.registry)
        self.domains.bind(world)
        self.domains.place(world, self.field_with_id(self.stone), building)
        self.domains.place(world, self.field_with_id(self.dirt), road)
        self.assertEqual(world.source_at((0, 0, 0)).block_id, self.stone)

        # Its own later placements still replace it.
        self.domains.place(world, self.field_with_id(self.fence), building)
        self.assertEqual(world.source_at((0, 0, 0)).block_id, self.fence)

    def test_overwrite_interaction_breaks_protection_after_rebinding(self) -> None:
        building = self.domains.create(
            "building", placement=PlacementRules(protected=True)
        )
        road = self.domains.create("road")
        world = BlockWorld(self.registry)
        self.domains.bind(world)
        self.domains.place(world, self.field_with_id(self.stone), building)
        self.domains.place(world, self.field_with_id(self.dirt), road)
        world.drain_dirty_changes()

        self.domains.interact(road, building, overwrite=True)
        self.domains.bind(world)

        self.assertEqual(world.source_at((0, 0, 0)).block_id, self.dirt)
        self.assertEqual(list(world.drain_dirty_changes()), [(0, 0, 0)])

    def test_overwrite_false_blocks_placement_without_protection(self) -> None:
        water = self.domains.create("water")
        sand = self.domains.create("sand")
        self.domains.interact(sand, water, overwrite=False)
        world = BlockWorld(self.registry)
        self.domains.bind(world)
        self.domains.place(world, self.field_with_id(self.stone), water)
        self.domains.place(world, self.field_with_id(self.dirt), sand)

        self.assertEqual(world.source_at((0, 0, 0)).block_id, self.stone)
        self.assertTrue(self.domains.compile().can_overwrite(water.domain_id, sand.domain_id))

    def test_bind_sets_the_terrain_domain_for_base_blocks(self) -> None:
        terrain = self.domains.create(
            "terrain",
            terrain=TerrainRules(is_terrain=True),
            connectivity=ConnectivityRules(connects_to_any=True),
        )
        self.domains.create("cliffs", parents=[terrain])
        fences = self.domains.create("fences")
        world = BlockWorld(self.registry, base_blocks={(1, 0, 0): self.stone})
        self.domains.bind(world)
        self.domains.place(world, self.field_with_id(self.fence), fences)

        world.resolve_dirty(self.domains.resolver(self.registry))

        self.assertEqual(world.base_domain, terrain.domain_id)
        self.assertEqual(self.state_at(world, (0, 0, 0)).get("east"), "true")

    def test_several_terrain_roots_require_an_explicit_choice(self) -> None:
        self.domains.create("land", terrain=TerrainRules(is_terrain=True))
        sea = self.domains.create("sea", terrain=TerrainRules(is_terrain=True))
        world = BlockWorld(self.registry)

        with self.assertRaises(ValueError):
            self.domains.bind(world)
        self.domains.bind(world, terrain="sea")
        self.assertEqual(world.base_domain, sea.domain_id)

    def test_terrain_mask_mode_comes_from_the_domain(self) -> None:
        seen: list[TerrainMaskMode] = []

        def above_ground(position, mode) -> bool:
            seen.append(mode)
            return position.y > 0

        self.domains.create(
            "grass",
            placement=PlacementRules(mode=PlacementMode.TERRAIN_MASK),
            terrain=TerrainRules(mask_mode=TerrainMaskMode.ABOVE),
        )
        world = BlockWorld(self.registry, terrain_mask=above_ground)
        field = self.field_with_id(self.dirt)
        field.set_block_id((0, 1, 0), self.dirt)
        self.domains.place(world, field, "grass")

        self.assertIsNone(world.source_at((0, 0, 0)))
        self.assertEqual(world.source_at((0, 1, 0)).block_id, self.dirt)
        self.assertEqual(set(seen), {TerrainMaskMode.ABOVE})

    def test_recompiling_equivalent_rules_keeps_resolved_state(self) -> None:
        self.domains.create("first")
        world = BlockWorld(self.registry)
        self.domains.bind(world)
        self.domains.place(world, self.field_with_id(self.fence), "first")
        world.resolve_dirty(self.domains.resolver(self.registry))
        epoch = world.policy_epoch

        # Property edits do not affect connectivity or placement rules.
        self.domains.update("first", properties={"label": "renamed"})
        self.domains.bind(world)
        world.resolve_dirty(self.domains.resolver(self.registry))
        self.assertEqual(world.policy_epoch, epoch)

        self.domains.update(
            "first", connectivity=ConnectivityRules(connects_to_self=False)
        )
        world.resolve_dirty(self.domains.resolver(self.registry))
        self.assertEqual(world.policy_epoch, epoch + 1)


if __name__ == "__main__":
    unittest.main()
