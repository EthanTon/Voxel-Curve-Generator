import unittest

import numpy as np

from src.Core.Field.block_field import BlockField
from src.Core.block.palette import BlockPalette
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos, ColumnPos
from src.Terraform.classify import DEFAULT_LOOSE, DEFAULT_TOPS, BlockNames, is_loose
from src.Terraform.grid import Grid
from src.Terraform.materialize import Strata, build
from src.Terraform.plan import read as read_plan
from src.Terraform.subsurface import RockRule, Subsurface, stony
from src.Terraform.terrain import FieldTerrain, survey
from src.Terraform.terraform import _tidied_levels

GRASS = BlockState.create("minecraft:grass_block")
DIRT = BlockState.create("minecraft:dirt")
STONE = BlockState.create("minecraft:stone")
SAND = BlockState.create("minecraft:sand")
SANDSTONE = BlockState.create("minecraft:sandstone")
GRAVEL = BlockState.create("minecraft:gravel")
SMOOTH = BlockState.create("minecraft:smooth_stone")
WATER = BlockState.create("minecraft:water")

PLAINS = [GRASS, DIRT, DIRT, DIRT] + [STONE] * 12
DESERT = [SAND] * 4 + [SANDSTONE] * 4 + [STONE] * 8
CUSTOM = [GRASS, DIRT] + [SMOOTH] * 14
STRATA = Strata(GRASS, DIRT, STONE)


def land(width, depth, column, top=64, water=None):
    # Every column the same from its top down, as `column(x, z)` gives it.
    blocks = BlockField(BlockPalette())
    for x in range(width):
        for z in range(depth):
            for below, state in enumerate(column(x, z)):
                if state is not None:
                    blocks.set_block(BlockPos(x, top - below, z), state)
            if water is not None:
                for y in range(top + 1, water + 1):
                    blocks.set_block(BlockPos(x, y, z), WATER)
    grid = Grid(ColumnPos(0, 0), width, depth)
    found = survey(FieldTerrain(blocks), grid)
    below = Subsurface(found.layers, found.layer_states, found.height.astype(np.float64))
    return grid, found, below


class SurveyTests(unittest.TestCase):
    def test_the_survey_keeps_what_lies_under_each_surface(self) -> None:
        column = list(PLAINS)
        column[6] = None                    # a cave
        grid, found, _ = land(1, 1, lambda x, z: column)
        states = found.layer_states
        self.assertEqual([states[i] if i >= 0 else None for i in found.layers[:8, 0, 0]],
                         [GRASS, DIRT, DIRT, DIRT, STONE, STONE, None, STONE])


class LayeringTests(unittest.TestCase):
    def test_the_land_round_a_column_gives_its_layering(self) -> None:
        odd = {(5, 5)}

        def desert(x, z):
            column = list(DESERT)
            if (x, z) in odd:
                column[1] = GRAVEL          # a pocket that is not the land's
            return column

        grid, _, below = land(12, 12, desert)
        layering = below.layering(np.ones((12, 12), dtype=bool))
        self.assertEqual([layering.at(5, 5, depth) for depth in range(6)],
                         [SAND, SAND, SAND, SAND, SANDSTONE, SANDSTONE])
        # Below what the survey kept, the deepest layer carries on.
        self.assertEqual(layering.at(5, 5, 40), STONE)


    def test_the_reach_and_the_tile_set_how_much_land_is_read(self) -> None:
        # A 4 by 4 patch of desert in grassland: read on its own it is sand,
        # read with the land round it, grass.
        def patch(x, z):
            return DESERT if 4 <= x < 8 and 4 <= z < 8 else PLAINS

        grid, _, below = land(16, 16, patch)
        known = np.ones((16, 16), dtype=bool)
        self.assertEqual(below.layering(known, tile=4, reach=0).at(5, 5, 0), SAND)
        self.assertEqual(below.layering(known).at(5, 5, 0), GRASS)
        # Column by column, a narrow reach still sees mostly the patch.
        self.assertEqual(below.layering(known, tile=1, reach=1).at(5, 5, 0), SAND)

    def test_the_plan_reads_the_layering_and_checks_it(self) -> None:
        embankment = read_plan({"terraform": {"embankment": {"layering_reach": 20, "layering_tile": 2}}}).embankment
        self.assertEqual((embankment.layering_reach, embankment.layering_tile), (20, 2))
        for key, wrong in (("layering_reach", -1), ("layering_reach", 2.5), ("layering_tile", 0), ("layering_tile", True)):
            with self.assertRaisesRegex(ValueError, f"embankment.{key}"):
                read_plan({"terraform": {"embankment": {key: wrong}}})


class RockTests(unittest.TestCase):
    def test_rock_is_whatever_under_the_soil_is_not_loose(self) -> None:
        for column, top in ((PLAINS, 60), (DESERT, 60), (CUSTOM, 62)):
            _, _, below = land(1, 1, lambda x, z, column=column: column)
            self.assertEqual(below.rock_tops()[0, 0], top)
        _, _, below = land(1, 1, lambda x, z: [GRASS] + [DIRT] * 15)
        self.assertEqual(below.rock_tops()[0, 0], -np.inf)

    def test_a_thin_band_over_a_pocket_of_soil_is_not_the_rock(self) -> None:
        coal = BlockState.create("minecraft:coal_ore")
        column = [GRASS] + [DIRT] * 4 + [STONE, coal, coal] + [DIRT] * 4 + [STONE] * 4
        _, _, below = land(1, 1, lambda x, z: column)
        self.assertEqual(below.rock_tops()[0, 0], 64 - 12)

    def test_a_face_is_rock_where_most_of_it_is(self) -> None:
        self.assertTrue(stony(["minecraft:stone", "minecraft:sandstone", "minecraft:dirt", None]))
        self.assertFalse(stony(["minecraft:stone", "minecraft:sand", "minecraft:gravel"]))
        self.assertTrue(is_loose("minecraft:red_sand") and not is_loose("minecraft:red_sandstone"))
        _, _, below = land(1, 1, lambda x, z: PLAINS)
        self.assertEqual(list(below.face([0, 0], [0, 0], [55, 58], [60, 63])), [True, False])


class LooseTests(unittest.TestCase):
    def test_names_and_patterns_say_what_is_soil(self) -> None:
        loose = BlockNames(("minecraft:dirt", "minecraft:*sand"))
        self.assertTrue(loose("minecraft:dirt"))
        self.assertTrue(loose("minecraft:red_sand") and loose("minecraft:soul_sand"))
        self.assertFalse(loose("minecraft:sandstone") or loose("minecraft:stone"))

    def test_a_custom_list_decides_the_rock(self) -> None:
        soft = BlockNames(("minecraft:grass_block", "minecraft:dirt", "minecraft:smooth_stone"))
        _, found, below = land(1, 1, lambda x, z: CUSTOM)
        self.assertEqual(below.rock_tops()[0, 0], 62)
        custom = Subsurface(below.layers, below.states, below.height, RockRule(loose=soft))
        self.assertEqual(custom.rock_tops()[0, 0], -np.inf)
        self.assertFalse(custom.face([0], [0], [55], [60])[0])

    def test_the_rule_says_how_much_rock_a_body_and_a_face_need(self) -> None:
        coal = BlockState.create("minecraft:coal_ore")
        column = [GRASS] + [DIRT] * 4 + [STONE, coal, coal] + [DIRT] * 4 + [STONE] * 4
        _, _, below = land(1, 1, lambda x, z: column)
        self.assertEqual(below.rock_tops()[0, 0], 64 - 12)
        # A body of three is enough for the band over the pocket.
        band = Subsurface(below.layers, below.states, below.height, RockRule(body=3))
        self.assertEqual(band.rock_tops()[0, 0], 64 - 5)
        face = ["minecraft:stone", "minecraft:stone", "minecraft:dirt"]
        self.assertTrue(stony(face))
        self.assertFalse(stony(face, RockRule(share=0.9)))

    def test_the_plan_reads_the_rock_rule_and_checks_it(self) -> None:
        cutting = read_plan({}).cutting
        self.assertEqual((cutting.rock_body, cutting.stone_share), (6, 0.66))
        cutting = read_plan({"terraform": {"cutting": {"rock_body": 4, "stone_share": 0.75}}}).cutting
        self.assertEqual(cutting.rock_rule, RockRule(DEFAULT_LOOSE, 4, 0.75))
        for key, wrong in (("rock_body", 0), ("rock_body", 17), ("rock_body", 2.5), ("rock_body", True),
                           ("stone_share", 0), ("stone_share", 1.5), ("stone_share", "most")):
            with self.assertRaisesRegex(ValueError, f"cutting.{key}"):
                read_plan({"terraform": {"cutting": {key: wrong}}})

    def test_the_plan_reads_the_slope_settings_and_checks_them(self) -> None:
        slopes = read_plan({"terraform": {"slopes": {"portal_reach": 6, "repair_passes": 50}}}).slopes
        self.assertEqual((slopes.portal_reach, slopes.repair_passes), (6.0, 50))
        self.assertEqual((read_plan({}).slopes.portal_reach, read_plan({}).slopes.repair_passes), (4.0, 400))
        for key, wrong in (("portal_reach", -1), ("portal_reach", True), ("repair_passes", -1),
                           ("repair_passes", 2.5), ("pocket", -1), ("pocket", 1.5)):
            with self.assertRaisesRegex(ValueError, f"slopes.{key}"):
                read_plan({"terraform": {"slopes": {key: wrong}}})
        plan = read_plan({"terraform": {"slopes": {"pocket": 8}, "cutting": {"rock_band": 3}}})
        self.assertEqual((plan.slopes.pocket, plan.cutting.rock_band), (8, 3.0))
        for wrong in (-1, True, "wide"):
            with self.assertRaisesRegex(ValueError, "cutting.rock_band"):
                read_plan({"terraform": {"cutting": {"rock_band": wrong}}})

    def test_the_plan_reads_the_ridge_width_and_checks_it(self) -> None:
        self.assertEqual(read_plan({}).slopes.ridge_width, 2)
        for width in (0, 3):
            self.assertEqual(read_plan({"terraform": {"slopes": {"ridge_width": width}}}).slopes.ridge_width, width)
        for wrong in (-1, 1.5, True, "wide"):
            with self.assertRaisesRegex(ValueError, "slopes.ridge_width"):
                read_plan({"terraform": {"slopes": {"ridge_width": wrong}}})

    def test_the_plan_reads_the_cone_passes_and_wall_band_and_checks_them(self) -> None:
        plan = read_plan({"terraform": {"slopes": {"cone_passes": 200}, "cutting": {"wall_band": 3}}})
        self.assertEqual((plan.slopes.cone_passes, plan.cutting.wall_band), (200, 3.0))
        self.assertEqual((read_plan({}).slopes.cone_passes, read_plan({}).cutting.wall_band), (1000, 2.0))
        for section, key, wrong in (("slopes", "cone_passes", -1), ("slopes", "cone_passes", 2.5),
                                    ("slopes", "cone_passes", True), ("cutting", "wall_band", -1),
                                    ("cutting", "wall_band", "wide")):
            with self.assertRaisesRegex(ValueError, f"{section}.{key}"):
                read_plan({"terraform": {section: {key: wrong}}})

    def test_the_plan_reads_the_list_and_keeps_the_default_without_one(self) -> None:
        self.assertEqual(read_plan({}).cutting.loose, DEFAULT_LOOSE)
        plan = read_plan({"terraform": {"cutting": {"loose": ["minecraft:dirt", "minecraft:*sand"]}}})
        self.assertEqual(plan.cutting.loose, BlockNames(("minecraft:dirt", "minecraft:*sand")))
        for wrong in ("minecraft:dirt", ["dirt"], [3]):
            with self.assertRaisesRegex(ValueError, "cutting.loose"):
                read_plan({"terraform": {"cutting": {"loose": wrong}}})


class MaterialTests(unittest.TestCase):
    def raised(self, column, to, water=None):
        grid, _, below = land(3, 3, lambda x, z: column, top=64)
        layering = below.layering(np.ones((3, 3), dtype=bool))
        designed = grid.filled(64.0)
        designed[1, 1] = to
        waterline = None if water is None else grid.filled(float(water))
        return build(grid, grid.filled(64.0), designed, STRATA, BlockPalette(), layering, waterline).fill

    def test_a_fill_is_built_the_way_the_land_round_it_lies(self) -> None:
        for column in (PLAINS, DESERT, CUSTOM):
            fill = self.raised(column, 70)
            self.assertEqual([fill.get_block(BlockPos(1, y, 1)) for y in range(70, 64, -1)], column[:6])

    def test_a_cut_takes_the_lands_surface_and_leaves_its_face(self) -> None:
        grid, _, below = land(3, 1, lambda x, z: DESERT, top=64)
        layering = below.layering(np.ones((1, 3), dtype=bool))
        designed = grid.filled(64.0)
        designed[0, 1] = 58
        earthworks = build(grid, grid.filled(64.0), designed, STRATA, BlockPalette(), layering)
        self.assertEqual(earthworks.fill.get_block(BlockPos(1, 58, 0)), SAND)
        self.assertIn(BlockPos(1, 60, 0), earthworks.cut)
        self.assertIsNone(earthworks.fill.get_block(BlockPos(1, 59, 0)))

    def test_under_water_a_fill_shows_what_lies_beneath_the_lands_top(self) -> None:
        fill = self.raised(PLAINS, 66, water=66)
        self.assertEqual(fill.get_block(BlockPos(1, 66, 1)), DIRT)
        self.assertEqual(fill.get_block(BlockPos(1, 65, 1)), DIRT)

    def test_the_plans_top_only_blocks_give_way_under_water(self) -> None:
        # With sand taken as only ever topping the ground, a drowned desert
        # fill shows the sandstone under it.
        grid, _, below = land(3, 3, lambda x, z: DESERT, top=64)
        layering = below.layering(np.ones((3, 3), dtype=bool))
        designed = grid.filled(64.0)
        designed[1, 1] = 67
        tops = BlockNames(("minecraft:sand",))
        waterline = grid.filled(67.0)
        fill = build(grid, grid.filled(64.0), designed, STRATA, BlockPalette(), layering, waterline, None, tops).fill
        self.assertEqual(fill.get_block(BlockPos(1, 67, 1)), SANDSTONE)
        fill = build(grid, grid.filled(64.0), designed, STRATA, BlockPalette(), layering, waterline).fill
        self.assertEqual(fill.get_block(BlockPos(1, 67, 1)), SAND)

    def test_the_plan_reads_its_top_only_blocks(self) -> None:
        self.assertEqual(read_plan({}).embankment.tops, DEFAULT_TOPS)
        plan = read_plan({"terraform": {"embankment": {"tops": ["minecraft:grass_block"]}}})
        self.assertEqual(plan.embankment.tops, BlockNames(("minecraft:grass_block",)))
        with self.assertRaisesRegex(ValueError, "embankment.tops"):
            read_plan({"terraform": {"embankment": {"tops": ["grass_block"]}}})

    def test_without_a_survey_the_plans_strata_build_it(self) -> None:
        grid = Grid(ColumnPos(0, 0), 1, 1)
        fill = build(grid, grid.filled(60.0), grid.filled(65.0), STRATA, BlockPalette()).fill
        self.assertEqual([fill.get_block(BlockPos(0, y, 0)) for y in range(65, 60, -1)],
                         [GRASS, DIRT, DIRT, DIRT, STONE])


class LevelTidyTests(unittest.TestCase):
    def test_teeth_come_down_and_notches_fill_but_straight_edges_stay(self) -> None:
        designed = np.zeros((9, 9))
        designed[:, 5:] = 1.0
        designed[4, 4] = 1.0          # a tooth sticking out of the edge
        designed[2, 6] = 0.0          # a notch inside the level
        tidied = _tidied_levels(designed, np.ones((9, 9), bool))
        self.assertEqual(tidied[4, 4], 0.0)
        self.assertEqual(tidied[2, 6], 1.0)
        straight = np.zeros((9, 9))
        straight[:, 5:] = 1.0
        self.assertTrue((_tidied_levels(straight, np.ones((9, 9), bool)) == straight).all())

    def test_held_columns_are_not_tidied(self) -> None:
        designed = np.zeros((5, 5))
        designed[2, 2] = 1.0
        self.assertEqual(_tidied_levels(designed, np.zeros((5, 5), bool))[2, 2], 1.0)


if __name__ == "__main__":
    unittest.main()
