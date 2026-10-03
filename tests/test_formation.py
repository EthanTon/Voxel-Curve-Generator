import unittest

import numpy as np

from src.Core.coordinates import BlockPos
from src.Parser.sections import read_section
from src.Terraform.layers import Ground, Slope
from src.Terraform.plan import read
from src.Terraform.terraform import Alignment


def section(**extra):
    return read_section(
        {"name": "bank", "domain": "bed", "schematic": "bank.schem", **extra}, {}
    )


class FormationTests(unittest.TestCase):
    def test_a_linear_embankment_and_water_spill_are_explicit_options(self) -> None:
        spec = read({"terraform": {"embankment": {"ratio": 2, "shape": "linear", "water_spill": True}}})
        self.assertEqual(spec.embankment.shape, "linear")
        self.assertTrue(spec.embankment.water_spill)
        self.assertEqual(read({"terraform": {}}).embankment.shape, "eased")
        self.assertFalse(read({"terraform": {}}).embankment.water_spill)
        for data in ({"shape": "steps"}, {"water_spill": "true"}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                read({"terraform": {"embankment": data}})

    def test_water_slopes_require_preserved_water_and_a_boolean(self) -> None:
        self.assertFalse(read({"terraform": {}}).terrain_mask.slope_to_water)
        self.assertTrue(read({"terraform": {"terrain_mask": {
            "preserve_water": True, "slope_to_water": True,
        }}}).terrain_mask.slope_to_water)
        for mask in ({"slope_to_water": True}, {"preserve_water": True, "slope_to_water": "true"}):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                read({"terraform": {"terrain_mask": mask}})

    def test_rails_are_included_only_when_requested(self) -> None:
        self.assertFalse(read({"terraform": {}}).include_rails)
        self.assertTrue(read({"terraform": {"include_rails": True}}).include_rails)
        with self.assertRaises(ValueError):
            read({"terraform": {"include_rails": "true"}})

    def test_a_section_has_no_formation_unless_it_sets_one(self) -> None:
        self.assertIsNone(section().formation)
        self.assertEqual(section(formation=-4).formation, -4)

    def test_a_formation_is_a_whole_height(self) -> None:
        for bad in (-4.5, "top", True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                section(formation=bad)

    def test_the_ground_meets_the_formation_where_one_is_set(self) -> None:
        # Two columns of a base five deep; only the first is at a formation.
        blocks = [BlockPos(x, y, 0) for x in (0, 1) for y in (60, 61, 65)]
        alignment = Alignment.of(
            blocks, [0] * len(blocks),
            lambda x, z, station: 61 if x == 0 else None,
        )
        self.assertEqual(alignment.footing, {(0, 0): 60, (1, 0): 60})
        self.assertEqual(alignment.meets, {(0, 0): 61, (1, 0): 60})


    def test_blocks_the_ground_covers_mark_the_track_but_not_its_footing(self) -> None:
        # Column 0: a covered base under a bed; column 1: only a covered rail.
        blocks = [BlockPos(0, 60, 0), BlockPos(0, 64, 0), BlockPos(1, 61, 0)]
        alignment = Alignment.of(blocks, [0, 0, 0], covered=[True, False, True])
        self.assertEqual(alignment.footing, {(0, 0): 64, (1, 0): 61})
        self.assertEqual(set(alignment.station), {(0, 0), (1, 0)})

    def test_terrain_overrides_are_block_names(self) -> None:
        self.assertEqual(
            section(terrain_overrides=["minecraft:andesite"]).terrain_overrides,
            frozenset({"minecraft:andesite"}),
        )
        with self.assertRaises(ValueError):
            section(terrain_overrides="minecraft:andesite")
        with self.assertRaises(ValueError):
            section(terrain_overrides=["andesite"])

    def test_override_columns_protect_the_bed_even_with_shared_materials(self) -> None:
        self.assertIsNone(section().terrain_overrides_columns)
        self.assertEqual(section(terrain_overrides_columns=[-9, 9]).terrain_overrides_columns,
                         frozenset({-9, 9}))
        for value in ("edges", [True], [-7.5], ["9"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                section(terrain_overrides_columns=value)


class ShoulderTests(unittest.TestCase):
    def ground(self) -> Ground:
        # Ground at 0 either side of a track at 10.
        distance = np.arange(60.0)[None, :].repeat(3, 0)
        level = np.full(distance.shape, 10.0)
        return Ground(
            grid=None, natural=np.zeros(distance.shape), distance=distance,
            seed_height=level, crest=level, station=np.zeros(distance.shape),
        )

    def test_a_slope_eases_out_past_the_shoulder(self) -> None:
        # Level across the shoulder, then down as a smooth curve, flat at its
        # top and its foot, and never steeper than 1 in 2, onto the ground.
        falling = Slope(2.0, shoulder=2.0)(self.ground(), np.zeros((3, 60)))[1]
        self.assertTrue((falling[:3] == 10).all())
        steps = np.diff(falling)
        self.assertTrue((steps <= 1e-9).all())
        self.assertLess(-steps.min(), 0.5 + 0.01)
        self.assertLess(-steps[2], 0.01)
        self.assertTrue((falling[2 + 38:] == 0).all())

    def test_a_cut_eases_up_the_same_way(self) -> None:
        ground = self.ground()
        rising = Slope(2.0, rising=True, shoulder=2.0)(ground, np.full((3, 60), 20.0))[1]
        self.assertTrue((rising[:3] == 10).all())
        self.assertTrue((np.diff(rising) >= -1e-9).all())
        self.assertTrue((rising[2 + 38:] == 20).all())

    def test_the_plan_sets_the_shoulder_and_a_step_may_override_it(self) -> None:
        plan = read({
            "terraform": {"shoulder": 2.0},
            "profile": {"bench": {"action": "slope_up", "shoulder": 4.0}},
        })
        self.assertEqual(plan.shoulder, 2.0)
        self.assertEqual(plan.profile("bench").steps[0].shoulder, 4.0)
        with self.assertRaises(ValueError):
            read({"terraform": {"shoulder": -1.0}})


if __name__ == "__main__":
    unittest.main()
