import unittest

from src.Core.Field.block_field import BlockBounds, BlockField
from src.Core.World.block_world import BlockWorld
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos
from src.Format.region import (
    AIR,
    RegionEntity,
    Axis,
    BlockRegion,
    Overflow,
    SplitBudget,
    enclosing_bounds,
    split_by_budget,
)

STONE = BlockState.create("minecraft:stone")
RAIL = BlockState.create("minecraft:rail", {"shape": "north_south"})
GLASS = BlockState.create("minecraft:glass")


def region_of(name, positions, state=STONE, origin=None, size=None):
    positions = list(positions)
    if origin is None or size is None:
        low = [min(position[axis] for position in positions) for axis in range(3)]
        high = [max(position[axis] for position in positions) for axis in range(3)]
        origin = BlockPos(*low)
        size = BlockPos(*(high[axis] - low[axis] + 1 for axis in range(3)))
    region = BlockRegion(name, origin, size)
    for position in positions:
        region.set_state(position, state)
    return region


class BlockRegionTests(unittest.TestCase):
    def test_cells_store_x_fastest_then_z_then_y(self) -> None:
        region = BlockRegion("box", (0, 0, 0), (2, 3, 4))

        self.assertEqual(region.linear_index((1, 0, 0)), 1)
        self.assertEqual(region.linear_index((0, 0, 1)), 2)
        self.assertEqual(region.linear_index((0, 1, 0)), 8)
        for index in range(region.volume):
            self.assertEqual(region.linear_index(region.position_at(index)), index)

    def test_linear_index_is_relative_to_the_origin(self) -> None:
        region = BlockRegion("box", (-5, 64, -5), (2, 2, 2))

        self.assertEqual(region.linear_index((-5, 64, -5)), 0)
        self.assertEqual(region.position_at(7), BlockPos(-4, 65, -4))
        with self.assertRaises(IndexError):
            region.linear_index((-3, 64, -5))

    def test_untouched_cells_read_as_air(self) -> None:
        region = BlockRegion("box", (0, 0, 0), (2, 1, 1))
        region.set_state((0, 0, 0), STONE)

        self.assertEqual(region.state_at((1, 0, 0)), AIR)
        self.assertFalse(region.is_occupied((1, 0, 0)))
        self.assertEqual(list(region.iter_cells()), [1, 0])

    def test_explicit_air_is_occupied_but_not_counted(self) -> None:
        region = BlockRegion("box", (0, 0, 0), (2, 1, 1))
        region.set_state((0, 0, 0), STONE)
        region.set_state((1, 0, 0), AIR)

        self.assertTrue(region.is_occupied((1, 0, 0)))
        self.assertEqual(region.filled_count, 2)
        self.assertEqual(region.block_count, 1)
        self.assertTrue(region.has_content)

    def test_from_field_places_content_at_the_field_offset(self) -> None:
        field = BlockField()
        field.set_block((0, 0, 0), STONE)
        field.set_block((2, 0, 0), RAIL)

        region = BlockRegion.from_field(field, name="piece", origin=(10, 64, -3))

        self.assertEqual(region.origin, BlockPos(10, 64, -3))
        self.assertEqual(region.size, BlockPos(3, 1, 1))
        self.assertEqual(region.state_at((10, 64, -3)), STONE)
        self.assertEqual(region.state_at((12, 64, -3)), RAIL)
        self.assertEqual(region.block_count, 2)

    def test_from_world_covers_composed_placements(self) -> None:
        registry = BlockRegistry()
        field = BlockField(registry.palette)
        field.set_block_id((0, 0, 0), registry.register(STONE))
        world = BlockWorld(registry)
        world.add_field(field, (0, 0, 0))
        world.add_field(field, (4, 0, 0))

        region = BlockRegion.from_world(world, name="build")

        self.assertEqual(region.origin, BlockPos(0, 0, 0))
        self.assertEqual(region.size, BlockPos(5, 1, 1))
        self.assertEqual(region.block_count, 2)
        self.assertEqual(region.state_at((4, 0, 0)), STONE)

    def test_sliced_pads_beyond_the_source_with_untouched_air(self) -> None:
        region = region_of("box", [(0, 0, 0)])

        piece = region.sliced(BlockBounds((-2, 0, 0), (2, 0, 0)), name="wide")

        self.assertEqual(piece.name, "wide")
        self.assertEqual(piece.origin, BlockPos(-2, 0, 0))
        self.assertEqual(piece.size, BlockPos(5, 1, 1))
        self.assertEqual(piece.filled_count, 1)
        self.assertEqual(piece.state_at((0, 0, 0)), STONE)

    def test_sliced_builds_its_palette_from_what_it_carries(self) -> None:
        region = BlockRegion("box", (0, 0, 0), (2, 1, 1))
        region.set_state((0, 0, 0), STONE)
        region.set_state((1, 0, 0), RAIL)

        piece = region.sliced(BlockBounds((0, 0, 0), (0, 0, 0)))

        self.assertEqual(len(piece.palette), 2)
        self.assertEqual(piece.palette.get_state(1), STONE)

    def test_iter_sections_aligns_to_the_world_grid_through_negatives(self) -> None:
        region = region_of("box", [(-1, -1, -1), (0, 0, 0)])

        sections = dict(region.iter_sections())

        self.assertEqual(
            sorted(tuple(key) for key in sections),
            [(-1, -1, -1), (0, 0, 0)],
        )
        low = sections[(-1, -1, -1)]
        self.assertEqual(low.origin, BlockPos(-16, -16, -16))
        self.assertEqual(low.size, BlockPos(16, 16, 16))
        self.assertEqual(low.state_at((-1, -1, -1)), STONE)

    def test_iter_sections_gives_each_section_its_own_palette(self) -> None:
        region = BlockRegion("box", (0, 0, 0), (20, 1, 1))
        region.set_state((0, 0, 0), STONE)
        region.set_state((17, 0, 0), RAIL)

        sections = dict(region.iter_sections())

        self.assertEqual(len(sections), 2)
        self.assertEqual(len(sections[(0, 0, 0)].palette), 2)
        self.assertEqual(sections[(0, 0, 0)].palette.get_state(1), STONE)
        self.assertEqual(sections[(1, 0, 0)].palette.get_state(1), RAIL)

    def test_iter_sections_skips_sections_nothing_touched(self) -> None:
        region = BlockRegion("box", (0, 0, 0), (40, 1, 1))
        region.set_state((0, 0, 0), STONE)
        region.set_state((33, 0, 0), STONE)

        self.assertEqual(
            sorted(tuple(key) for key, _ in region.iter_sections()),
            [(0, 0, 0), (2, 0, 0)],
        )

    def test_enclosing_bounds_spans_every_region(self) -> None:
        left = region_of("left", [(0, 0, 0)], STONE)
        right = region_of("right", [(2, 1, 0), (3, 0, 0)], RAIL)

        self.assertEqual(
            enclosing_bounds([left, right]),
            BlockBounds(BlockPos(0, 0, 0), BlockPos(3, 1, 0)),
        )


class EntityTests(unittest.TestCase):
    def marker(self, pos, *tags):
        return RegionEntity("minecraft:block_display", pos, tags)

    def test_an_entity_takes_the_floored_cell(self) -> None:
        self.assertEqual(self.marker((2.5, -0.5, 3.0)).block_pos, BlockPos(2, -1, 3))

    def test_slicing_keeps_only_the_entities_inside_the_box(self) -> None:
        region = BlockRegion("build", (0, 0, 0), (6, 1, 1))
        region.add_entity(self.marker((1.5, 0.0, 0.0), "vcg.anchor"))
        region.add_entity(self.marker((4.5, 0.0, 0.0), "vcg.anchor"))

        piece = region.sliced(BlockBounds((0, 0, 0), (2, 0, 0)))

        self.assertEqual([e.block_pos.x for e in piece.entities], [1])

    def test_translating_moves_entities_with_the_blocks(self) -> None:
        region = BlockRegion("build", (0, 0, 0), (2, 1, 1))
        region.set_state((0, 0, 0), STONE)
        region.add_entity(self.marker((0.5, 0.25, 0.5)))

        moved = region.translated((10, 64, -3))

        self.assertEqual(moved.entities[0].pos, (10.5, 64.25, -2.5))
        self.assertEqual(moved.state_at((10, 64, -3)), STONE)

    def test_a_split_routes_every_entity_to_exactly_one_piece(self) -> None:
        region = BlockRegion("build", (0, 0, 0), (4, 1, 1))
        for x in range(4):
            region.set_state((x, 0, 0), STONE)
        region.add_entity(self.marker((1.9, 0.0, 0.0), "vcg.anchor"))
        region.add_entity(self.marker((2.0, 0.0, 0.0), "vcg.anchor"))

        pieces = split_by_budget(region, SplitBudget(max_blocks=2, axis=Axis.X))

        self.assertEqual([piece.origin.x for piece in pieces], [0, 2])
        self.assertEqual([len(piece.entities) for piece in pieces], [1, 1])

    def test_an_entity_in_a_gap_is_not_stranded(self) -> None:
        # Its cell joins the slab profile, so a piece stretches to cover it.
        region = BlockRegion("build", (0, 0, 0), (11, 1, 1))
        region.set_state((0, 0, 0), STONE)
        region.set_state((10, 0, 0), STONE)
        region.add_entity(self.marker((5.0, 0.0, 0.0), "vcg.anchor"))

        pieces = split_by_budget(region, SplitBudget(max_blocks=1, axis=Axis.X))

        self.assertEqual(sum(len(piece.entities) for piece in pieces), 1)


class SplitTests(unittest.TestCase):
    def occupied(self, regions):
        found = {}
        for region in regions:
            for position, state in region.iter_occupied():
                self.assertNotIn(position, found, f"{position} lands in two pieces")
                found[position] = state
        return found

    def test_split_covers_every_block_exactly_once(self) -> None:
        positions = [(index, index % 3, index * 2) for index in range(40)]
        region = region_of("curve", positions)

        pieces = split_by_budget(region, SplitBudget(max_blocks=7))

        self.assertGreater(len(pieces), 1)
        self.assertEqual(
            set(self.occupied(pieces)),
            {BlockPos(*position) for position in positions},
        )
        for piece in pieces:
            self.assertLessEqual(piece.block_count, 7)

    def test_pieces_are_named_in_paste_order(self) -> None:
        region = region_of("curve", [(index, 0, 0) for index in range(10)])

        pieces = split_by_budget(region, SplitBudget(max_blocks=4))

        self.assertEqual(
            [piece.name for piece in pieces],
            ["curve_000", "curve_001", "curve_002"],
        )
        self.assertEqual([piece.origin.x for piece in pieces], [0, 4, 8])

    def test_split_trims_pieces_flush_to_content(self) -> None:
        region = BlockRegion("build", (0, 0, 0), (10, 4, 1))
        region.set_state((0, 0, 0), STONE)
        region.set_state((1, 3, 0), STONE)

        pieces = split_by_budget(region, SplitBudget(max_blocks=1, axis=Axis.X))

        self.assertEqual([tuple(piece.origin) for piece in pieces], [(0, 0, 0), (1, 3, 0)])
        self.assertEqual([tuple(piece.size) for piece in pieces], [(1, 1, 1), (1, 1, 1)])

    def test_span_cap_breaks_a_run_a_block_budget_would_glue(self) -> None:
        # A count budget alone glues two clusters into one huge box.
        positions = [(0, 0, z) for z in range(5)] + [(0, 0, z) for z in range(100, 105)]
        region = region_of("curve", positions)

        glued = split_by_budget(region, SplitBudget(max_blocks=1000, axis=Axis.Z))
        self.assertEqual(len(glued), 1)
        self.assertEqual(glued[0].size.z, 105)

        split = split_by_budget(
            region, SplitBudget(max_blocks=1000, axis=Axis.Z, max_span=20)
        )
        self.assertEqual([tuple(piece.size) for piece in split], [(1, 1, 5), (1, 1, 5)])
        self.assertEqual([piece.origin.z for piece in split], [0, 100])

    def test_count_air_measures_the_trimmed_box(self) -> None:
        # Untrimmed the second piece would fill the budget on its own.
        region = BlockRegion("build", (0, 0, 0), (10, 1, 2))
        for x in range(10):
            region.set_state((x, 0, 0), STONE)
        region.set_state((0, 0, 1), STONE)

        pieces = split_by_budget(
            region, SplitBudget(max_blocks=10, count_air=True, axis=Axis.Z)
        )

        self.assertEqual([tuple(piece.size) for piece in pieces], [(10, 1, 1), (1, 1, 1)])

    def test_cuts_start_new_pieces(self) -> None:
        region = region_of("curve", [(0, 0, z) for z in range(10)])

        pieces = split_by_budget(
            region, SplitBudget(max_blocks=1000, axis=Axis.Z, cuts=(5,))
        )

        self.assertEqual([piece.origin.z for piece in pieces], [0, 5])
        self.assertEqual([piece.size.z for piece in pieces], [5, 5])

    def test_cuts_still_subdivide_under_the_budget(self) -> None:
        region = region_of("curve", [(0, 0, z) for z in range(10)])

        pieces = split_by_budget(
            region, SplitBudget(max_blocks=3, axis=Axis.Z, cuts=(5,))
        )

        self.assertEqual([piece.origin.z for piece in pieces], [0, 3, 5, 8])

    def test_extent_tiles_the_footprint_contiguously(self) -> None:
        extent = BlockBounds((0, 0, 0), (3, 0, 9))
        region = region_of("build", [(0, 0, 0), (2, 0, 9)])

        pieces = split_by_budget(
            region,
            SplitBudget(max_blocks=20, count_air=True, axis=Axis.Z, extent=extent),
        )

        self.assertEqual([tuple(piece.origin) for piece in pieces], [(0, 0, 0), (0, 0, 5)])
        self.assertEqual([tuple(piece.size) for piece in pieces], [(4, 1, 5), (4, 1, 5)])

    def test_extent_holds_pieces_out_to_the_footprint(self) -> None:
        # Without the extent each piece would hug its single block.
        extent = BlockBounds((-2, 0, 0), (2, 0, 3))
        region = region_of("build", [(0, 0, 0), (0, 0, 3)])

        pieces = split_by_budget(
            region,
            SplitBudget(max_blocks=10, count_air=True, axis=Axis.Z, extent=extent),
        )

        for piece in pieces:
            self.assertEqual(piece.origin.x, -2)
            self.assertEqual(piece.size.x, 5)

    def test_extent_rejects_content_outside_it(self) -> None:
        region = region_of("build", [(0, 0, 0), (9, 0, 0)])
        budget = SplitBudget(max_blocks=10, extent=BlockBounds((0, 0, 0), (4, 0, 0)))

        with self.assertRaises(ValueError) as caught:
            split_by_budget(region, budget)

        self.assertIn("(9, 0, 0)", str(caught.exception))

    def test_overflow_subdivides_on_another_axis(self) -> None:
        region = BlockRegion("slab", (0, 0, 0), (4, 1, 4))
        for z in range(4):
            for x in range(4):
                region.set_state((x, 0, z), STONE)

        pieces = split_by_budget(region, SplitBudget(max_blocks=2, axis=Axis.Z))

        self.assertEqual(len(pieces), 8)
        for piece in pieces:
            self.assertEqual(piece.block_count, 2)
        self.assertEqual(len(self.occupied(pieces)), 16)

    def test_overflow_can_refuse_instead(self) -> None:
        region = BlockRegion("slab", (0, 0, 0), (4, 1, 4))
        for z in range(4):
            for x in range(4):
                region.set_state((x, 0, z), STONE)
        budget = SplitBudget(max_blocks=2, axis=Axis.Z, overflow=Overflow.RAISE)

        with self.assertRaises(ValueError) as caught:
            split_by_budget(region, budget)

        self.assertIn("does not fit the budget", str(caught.exception))

    def test_a_region_with_nothing_in_it_splits_to_nothing(self) -> None:
        region = BlockRegion("empty", (0, 0, 0), (4, 4, 4))

        self.assertEqual(split_by_budget(region, SplitBudget(max_blocks=10)), [])

    def test_longest_axis_is_the_default(self) -> None:
        region = region_of("build", [(0, y, 0) for y in range(10)])

        pieces = split_by_budget(region, SplitBudget(max_blocks=4))

        self.assertEqual([piece.origin.y for piece in pieces], [0, 4, 8])

    def test_capped_keeps_the_stricter_span(self) -> None:
        self.assertEqual(SplitBudget(max_blocks=1).capped(100).max_span, 100)
        self.assertEqual(SplitBudget(max_blocks=1, max_span=10).capped(100).max_span, 10)
        self.assertEqual(SplitBudget(max_blocks=1, max_span=500).capped(100).max_span, 100)


if __name__ == "__main__":
    unittest.main()
