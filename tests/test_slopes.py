import unittest

import numpy as np

from src.Terraform.slopes import HELD, SHAPED, SPILL, cone, level, limits



def drops(heights: np.ndarray, mask: np.ndarray) -> int:
    # Side neighbours inside `mask` two or more apart.
    rows = (np.abs(np.diff(heights, axis=0)) >= 2) & mask[1:] & mask[:-1]
    cells = (np.abs(np.diff(heights, axis=1)) >= 2) & mask[:, 1:] & mask[:, :-1]
    return int(rows.sum() + cells.sum())


class Rock:
    # Soil `soil` deep over stone everywhere.
    def __init__(self, natural, soil):
        self.natural, self.soil = natural, soil

    def rock_tops(self):
        return self.natural - self.soil

    def face(self, rows, cells, lows, highs):
        return self.rock_tops()[rows, cells] >= np.asarray(highs)


def straight_track(width=41, depth=21, formation=66.0, ground=60.0, role=SHAPED):
    # A 3-wide track along the middle row band, the ground flat beside it.
    natural = np.full((depth, width), ground)
    track = np.zeros((depth, width), dtype=bool)
    track[depth // 2 - 1:depth // 2 + 2, :] = True
    meets = np.where(track, formation, 0.0)
    roles = np.full((depth, width), role, dtype=np.int8)
    return natural, track, roles, meets


def built(natural, track, roles, meets, blocked=None, rock=None, shoulder=0.0, run=2.0, passes=80,
          retained=None, **settings):
    blocked = np.zeros(natural.shape, dtype=bool) if blocked is None else blocked
    retained = np.zeros(natural.shape, dtype=bool) if retained is None else retained
    return limits(
        natural, track, roles, meets, retained, blocked,
        np.full(natural.shape, np.inf), shoulder, run, run, passes, rock, **settings,
    )


class ConeTests(unittest.TestCase):
    def test_a_cone_falls_a_block_per_run_and_is_nearly_round(self) -> None:
        source = np.zeros((41, 41), dtype=bool)
        source[20, 20] = True
        heights = np.where(source, 30.0, 0.0)
        found = cone(source, heights, np.ones_like(source), 2.0, rising=False, passes=60)
        rows, cells = np.indices(source.shape)
        exact = 30.0 - np.hypot(rows - 20, cells - 20) / 2.0
        self.assertLess(np.max(np.abs(found - exact) / np.maximum(30.0 - exact, 1e-9)), 0.03)

    def test_a_cone_goes_round_a_wall_not_through_it(self) -> None:
        source = np.zeros((21, 21), dtype=bool)
        source[10, 2] = True
        open_ground = np.ones_like(source)
        open_ground[:18, 6] = False          # a wall with a gap at the bottom
        found = cone(source, np.where(source, 20.0, 0.0), open_ground, 1.0, rising=False, passes=80)
        # Straight across is 6 columns; round the wall's end it is much further.
        self.assertLess(found[10, 8], 20.0 - 15.0)
        self.assertEqual(found[5, 6], -np.inf)


class LimitTests(unittest.TestCase):
    def test_retaining_wall_floor_survives_a_lower_neighbouring_bed_control(self):
        natural = np.full((21, 21), 92.)
        track = np.zeros_like(natural, bool)
        track[10, 2:19] = True
        retained = track.copy()
        retained[10, :10] = False
        meets = np.where(retained, 86., 76.)
        made = built(natural, track, np.full(track.shape, SHAPED), meets,
                     retained=retained, shoulder=2, run=1, hold_walls=True)
        # At the handover the nearest shoulder can inherit the lower bed.
        # Ground beside the real wall still sweeps to its height, rather
        # than following that lower cutting cone into a moat.
        self.assertGreaterEqual(made.least[9, 9], 86 - np.sqrt(2))
        self.assertTrue((made.cut[made.free] >= made.fill[made.free]).all())
        shaped = made.shape(natural.copy(), natural, np.zeros_like(track))
        final, _ = made.enforce(level(shaped), natural)
        self.assertGreaterEqual(final[9, 9], 84)
        np.testing.assert_array_equal(final[made.bed], meets[made.bed])
        self.assertTrue(np.isneginf(made.least[track]).all())

    def test_wall_contact_floor_obeys_a_blocked_bank_and_section_mask(self):
        natural, track, roles, meets = straight_track(depth=21, formation=86., ground=92.)
        rows, _ = np.indices(track.shape)
        area = rows >= 9
        blocked = rows == 15
        made = built(natural, track, roles, meets, retained=track, blocked=blocked,
                     retained_area=area, hold_walls=True, run=1)
        self.assertTrue(np.isneginf(made.least[~area | blocked]).all())
        self.assertTrue(np.isneginf(made.least[16:]).all())

    def test_a_bank_runs_out_at_its_ratio_as_far_as_its_height_needs(self) -> None:
        natural, track, roles, meets = straight_track(depth=41)
        made = built(natural, track, roles, meets)
        shaped = level(made.shape(natural.copy(), natural, np.zeros_like(track)))
        # Beside the 3-wide track (rows 19..21) the bank falls a block per two
        # columns from the formation to the ground six blocks down.
        self.assertEqual(list(shaped[22:36, 20]), [66, 65, 65, 64, 64, 63, 63, 62, 62, 61, 61, 60, 60, 60])
        self.assertEqual(drops(shaped, made.free), 0)

    def test_a_bank_stops_where_it_meets_the_ground(self) -> None:
        natural, track, roles, meets = straight_track(depth=41)
        natural[26:28, :] = 64.0             # a rise the bank's toe meets
        natural[30, :] = 58.0                # and a ditch beyond it
        made = built(natural, track, roles, meets)
        shaped = level(made.shape(natural.copy(), natural, np.zeros_like(track)))
        self.assertEqual(shaped[30, 20], 58.0)

    def test_a_cutting_leaves_a_hill_beyond_its_top_alone(self) -> None:
        natural, track, roles, meets = straight_track(depth=41, formation=60.0)
        natural[30, :] = 66.0                # a spike past the level ground
        made = built(natural, track, roles, meets)
        shaped = level(made.shape(natural.copy(), natural, np.zeros_like(track)))
        self.assertEqual(shaped[30, 20], 66.0)

    def test_a_cut_through_stone_stands_as_a_face_with_the_soil_set_back(self) -> None:
        natural, track, roles, meets = straight_track(depth=41, formation=60.0, ground=72.0)
        made = built(natural, track, roles, meets, rock=Rock(natural, soil=3), shoulder=1.0)
        shaped = level(made.shape(natural.copy(), natural, np.zeros_like(track)))
        column = shaped[:, 20]
        # The track rows 19..21, a level strip at 60, then stone to 69 and the
        # soil above set back a block per two columns.
        self.assertEqual(column[22], 60.0)
        self.assertEqual(column[23], 69.0)
        self.assertGreaterEqual(column[26], 70.0)
        levelled, left = made.enforce(shaped, natural)
        self.assertEqual(left, 0)
        self.assertEqual(levelled[23, 20], 69.0)

    def test_without_a_rock_band_the_rock_stands_under_its_soil(self) -> None:
        natural, track, roles, meets = straight_track(depth=41, formation=60.0, ground=72.0)
        rock = Rock(natural, soil=3)
        for band, beside in ((2.0, 69.0), (0.0, 72.0)):
            made = built(natural, track, roles, meets, rock=rock, shoulder=1.0, rock_band=band)
            shaped = level(made.shape(natural.copy(), natural, np.zeros_like(track)))
            self.assertEqual(shaped[23, 20], beside)

    def test_the_cone_passes_set_how_far_a_slope_spreads(self) -> None:
        natural, track, roles, meets = straight_track(depth=41)
        reach = [int(np.isfinite(built(natural, track, roles, meets, passes=passes).fill[22:, 20]).sum())
                 for passes in (0, 1, 80)]
        self.assertEqual(reach, [0, 9, 13])
        # Without any the bank is left to the repair, a block a column.
        idle = built(natural, track, roles, meets, passes=0)
        shaped = level(idle.shape(natural.copy(), natural, np.zeros_like(track)))
        fixed, left = idle.enforce(shaped, natural)
        self.assertEqual(left, 0)
        self.assertEqual(list(fixed[22:28, 20]), [65, 64, 63, 62, 61, 60])

    def test_the_wall_band_cuts_the_rock_behind_a_trenchs_walls(self) -> None:
        # Walls to 69 through stone under three blocks of soil: within the
        # band the ground is cut down to them, rock or not, and never raised
        # past their top; without one the rock stands as it is.
        natural, track, roles, meets = straight_track(depth=41, formation=69.0, ground=75.0)
        rock = Rock(natural, soil=3)
        for band, behind, cap in ((2.0, [70.0, 70.0], 69.0), (0.0, [75.0, 75.0], np.inf)):
            made = built(natural, track, roles, meets, rock=rock, retained=track, wall_band=band)
            shaped = level(made.shape(natural.copy(), natural, np.zeros_like(track)))
            self.assertEqual(list(shaped[22:24, 20]), behind)
            self.assertEqual(made.cap[22, 20], cap)

    def test_the_pocket_reaches_the_repair(self) -> None:
        natural, track, roles, meets = straight_track()
        self.assertEqual(built(natural, track, roles, meets, pocket=5).pocket, 5)

    def test_without_stone_the_same_cut_is_terraced(self) -> None:
        natural, track, roles, meets = straight_track(depth=41, formation=60.0, ground=72.0)
        made = built(natural, track, roles, meets, shoulder=1.0)
        shaped = level(made.shape(natural.copy(), natural, np.zeros_like(track)))
        self.assertEqual(drops(shaped, made.free), 0)
        self.assertEqual(shaped[23, 20], 61.0)

    def test_a_bank_spills_onto_a_bridges_ground_but_not_a_tunnels(self) -> None:
        natural, track, roles, meets = straight_track(width=61)
        roles[:, 30:] = SPILL
        made = built(natural, track, roles, meets)
        shaped = level(made.shape(natural.copy(), natural, np.zeros_like(track)))
        self.assertGreater(shaped[10, 32], 62.0)      # under the first span
        self.assertEqual(shaped[10, 50], 60.0)        # the far span untouched
        roles[:, 30:] = HELD
        made = built(natural, track, roles, meets)
        shaped = level(made.shape(natural.copy(), natural, np.zeros_like(track)))
        self.assertEqual(shaped[10, 32], 60.0)


class EnforceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.natural, self.track, self.roles, self.meets = straight_track()
        self.made = built(self.natural, self.track, self.roles, self.meets)

    def test_a_drop_in_the_works_is_closed(self) -> None:
        heights = level(self.made.shape(self.natural.copy(), self.natural, np.zeros_like(self.track)))
        heights[15:, 20] = 60.0              # a notch cut down the bank
        fixed, left = self.made.enforce(heights, self.natural)
        self.assertEqual(left, 0)
        self.assertEqual(drops(fixed, self.made.free), 0)

    def test_without_repair_passes_the_drops_stay(self) -> None:
        heights = level(self.made.shape(self.natural.copy(), self.natural, np.zeros_like(self.track)))
        heights[15:, 20] = 60.0
        idle = built(self.natural, self.track, self.roles, self.meets, repair_passes=0, ridge_width=0)
        self.assertGreater(idle.enforce(heights, self.natural)[1], 0)

    def test_a_portal_face_stands_as_far_out_as_its_reach(self) -> None:
        # The track's middle third is a tunnel; the ground beside its lining
        # may keep a face out to the portal's reach.
        natural, track, roles, meets = straight_track(width=61, depth=41)
        roles[:, 20:40] = HELD
        column = np.arange(41)
        for reach, widest in ((4.0, 4), (0.0, 0)):
            made = built(natural, track, roles, meets, portal=reach)
            faced = made.exempt[:, 30] & ~track[:, 30]
            self.assertEqual(int(np.abs(column[faced] - 20).max(initial=1) - 1), widest)

    def test_a_groove_behind_a_cuttings_crest_fills(self) -> None:
        # The two columns beyond the crest sunk a block below it, as a brush
        # can leave them: they fill to it, unless ridges are left alone.
        natural, track, roles, meets = straight_track(depth=41, formation=60.0, ground=72.0)
        heights = level(built(natural, track, roles, meets).shape(natural.copy(), natural, np.zeros_like(track)))
        heights[22:30, :] = np.array([61.0, 61.0, 60.0, 60.0, 61.0, 62.0, 63.0, 64.0])[:, None]
        for width, profile in ((2, [61, 61, 61, 61, 61, 62, 63, 64]), (0, [61, 61, 60, 60, 61, 62, 63, 64])):
            fixed, left = built(natural, track, roles, meets, ridge_width=width).enforce(heights, natural)
            self.assertEqual(left, 0)
            self.assertEqual(list(fixed[22:30, 20]), profile)

    def test_a_fin_of_rock_left_beside_the_track_comes_down(self) -> None:
        # Rock under a hump beside the track stands as a face with lower
        # ground beyond it: a fin two columns across, which comes down to
        # that ground.
        natural, track, roles, meets = straight_track(depth=41, formation=60.0, ground=64.0)
        natural[22:24, :] = 72.0
        rock = Rock(natural, soil=3)
        for width, beside in ((2, [64.0, 64.0]), (1, [69.0, 69.0]), (0, [69.0, 69.0])):
            made = built(natural, track, roles, meets, rock=rock, ridge_width=width)
            shaped = level(made.shape(natural.copy(), natural, np.zeros_like(track)))
            fixed, left = made.enforce(shaped, natural)
            self.assertEqual(left, 0)
            self.assertEqual(list(fixed[22:24, 20]), beside)
            self.assertEqual(fixed[25, 20], 64.0)

    def test_a_natural_cliff_the_bank_runs_into_stays(self) -> None:
        natural, track, roles, meets = straight_track(depth=41)
        natural[30:, :] = 70.0               # a cliff where the bank's toe lands
        made = built(natural, track, roles, meets)
        heights = level(made.shape(natural.copy(), natural, np.zeros_like(track)))
        fixed, left = made.enforce(heights, natural)
        self.assertEqual(left, 0)
        self.assertEqual(fixed[30, 20], 70.0)
        # The bank's last column, a groove against the cliff, fills to the
        # terrace above it.
        self.assertEqual(list(fixed[26:30, 20]), [64.0, 63.0, 63.0, 63.0])

    def test_water_keeps_its_edge(self) -> None:
        blocked = np.zeros_like(self.track)
        blocked[16:, :] = True               # a river past the bank's toe
        made = built(self.natural, self.track, self.roles, self.meets, blocked=blocked)
        heights = level(made.shape(self.natural.copy(), self.natural, np.zeros_like(self.track)))
        fixed, left = made.enforce(heights, self.natural)
        self.assertEqual(left, 0)
        self.assertEqual(fixed[16, 20], 60.0)
        self.assertGreater(fixed[15, 20], 61.0)

    def test_a_bank_steepened_to_uneven_held_ground_has_no_ribs(self) -> None:
        # Cut back a block a column from held ground that alternates a block,
        # the slope carried that step up its face as ribs; they are cut whole.
        natural, track, roles, meets = straight_track(depth=41, formation=72.0)
        rows, cells = np.indices(natural.shape)
        natural = natural + cells % 2
        roles = np.where(rows >= 34, HELD, roles).astype(np.int8)
        made = built(natural, track, roles, meets)
        heights = level(made.shape(natural.copy(), natural, np.zeros_like(track)))
        fixed, left = made.enforce(heights, natural)
        self.assertEqual(left, 0)
        self.assertEqual(drops(fixed, made.free), 0)
        self.assertTrue((fixed[22:34, 1:] == fixed[22:34, :-1]).all())

    def test_a_brushed_ridge_in_held_ground_is_levelled(self) -> None:
        natural = np.full((9, 9), 60.0)
        track = np.zeros(natural.shape, dtype=bool)
        roles = np.full(natural.shape, HELD, dtype=np.int8)
        made = built(natural, track, roles, np.zeros_like(natural))
        heights = natural.copy()
        heights[:, 4] += 1.0
        # The brush may have edited held ground just beyond a bank's toe.
        # Those edits can be tidied without moving any untouched column.
        fixed, left = made.enforce(heights, natural)
        self.assertEqual(left, 0)
        np.testing.assert_array_equal(fixed, natural)

    def test_a_shore_takes_a_bank_but_no_cut(self) -> None:
        # Dry ground beside a river the bank reaches: filled down to the
        # water, never cut below itself.
        blocked = np.zeros_like(self.track)
        blocked[18:, :] = True
        shore = np.zeros_like(self.track)
        shore[16:18, :] = True
        made = built(self.natural, self.track, self.roles, self.meets, blocked=blocked, shore=shore)
        heights = level(made.shape(self.natural.copy(), self.natural, np.zeros_like(self.track)))
        fixed, left = made.enforce(heights, self.natural)
        self.assertEqual(left, 0)
        self.assertEqual(drops(fixed[:18], made.free[:18]), 0)
        self.assertTrue((fixed[16:18] >= self.natural[16:18]).all())
        self.assertGreater(fixed[17, 20], 61.0)

    def test_rounding_is_half_up_everywhere(self) -> None:
        self.assertEqual(list(level(np.array([0.5, 1.5, 2.5, -0.5]))), [1.0, 2.0, 3.0, 0.0])


if __name__ == "__main__":
    unittest.main()
