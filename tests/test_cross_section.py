import unittest
from math import cos, radians, sin

from src.Core.CrossSection.annotations import FeatureAnnotation
from src.Core.CrossSection.cross_section import (
    CrossSection,
    PointField,
    orient_annotation,
)
from src.Core.CrossSection.steps import (
    STEP_RULES,
    column_pushes,
    lead,
    manhattan_step,
)
from src.Core.CrossSection.turns import (
    CLOCKWISE,
    COUNTER_CLOCKWISE,
    angle_between,
    heading,
    sense,
    snap,
    travel,
    turn_from,
    turn_of,
    turns_along,
)
from src.Core.Domain import Domain
from src.Core.Feature.feature import (
    ContinuousFeature,
    Feature,
    PointFeature,
    ProfileFeature,
    feature_class,
)
from src.Core.Feature.feature_path import DEFAULT_RULES, FeaturePath
from src.Core.Field.block_field import BlockField
from src.Core.coordinates import BlockPos, Vec3
from src.Core.modes import ElevationMode, PlacementMode, StepMode

NAMES = "abcodef"
GRADIENT = Vec3(0.0, 0.0, -1.0)


def feature():
    return PointFeature(BlockField(), PlacementMode.OVERWRITE, Domain(1, "test"))


def annotation(position, step_mode=StepMode.CHEBYSHEV, name="a"):
    return FeatureAnnotation(name, position, feature(), step_mode)


def row(step_mode=StepMode.CHEBYSHEV):
    # a b c o d e f, with o on the origin.
    return CrossSection(
        PointField(
            annotation((index - 3, 0, 0), step_mode, name)
            for index, name in enumerate(NAMES)
        )
    )


def layout(section, turn, direction=CLOCKWISE):
    return [
        (o.annotation.name, o.offsets[0].x, o.offsets[0].z)
        for o in section.orient(turn, direction)
    ]


def line(names, start, step):
    return [
        (name, start[0] + step[0] * i, start[1] + step[1] * i)
        for i, name in enumerate(names)
    ]


class PointFieldTests(unittest.TestCase):
    def test_names_are_unique(self) -> None:
        with self.assertRaises(ValueError):
            PointField((annotation((1, 0, 0)), annotation((-1, 0, 0))))


class QuarterLayoutTests(unittest.TestCase):
    # Turns are world directions clockwise from west: 0 west, 90 north,
    # 180 east, 270 south. Local x always lands to the right of travel.

    def test_zero_travels_west_as_authored(self) -> None:
        # a (left of travel) to the south, f (right) to the north.
        self.assertEqual(layout(row(), 0), line(NAMES, (0, 3), (0, -1)))

    def test_ninety_travels_north(self) -> None:
        self.assertEqual(layout(row(), 90), line(NAMES, (-3, 0), (1, 0)))

    def test_one_eighty_travels_east(self) -> None:
        self.assertEqual(layout(row(), 180), line(NAMES, (0, -3), (0, 1)))

    def test_two_seventy_travels_south(self) -> None:
        self.assertEqual(layout(row(), 270), line(NAMES, (3, 0), (-1, 0)))

    def test_depth_follows_travel_at_even_quarters_and_mirrors_at_odd(self) -> None:
        # One right of the origin and one ahead of it, as authored.
        section = CrossSection(PointField((annotation((1, 0, -1)),)))
        self.assertEqual(layout(section, 0), [("a", -1, -1)])    # north, and ahead (west)
        self.assertEqual(layout(section, 90), [("a", 1, 1)])     # east, and behind (south)
        self.assertEqual(layout(section, 180), [("a", 1, 1)])    # south, and ahead (east)
        self.assertEqual(layout(section, 270), [("a", -1, -1)])  # west, and behind (north)

    def test_positions_are_relative_to_the_origin(self) -> None:
        section = CrossSection(PointField((annotation((2, 1, 0)),)), origin=(2, 1, 0))
        offset = section.orient(90)[0].offsets[0]
        self.assertEqual(offset, BlockPos(0, 0, 0))

    def test_each_turn_is_oriented_once_and_stored(self) -> None:
        section = row()
        first = section.orient(45)
        self.assertIs(section.orient(45), first)
        self.assertIs(section.orient(405), first)
        self.assertIsNot(section.orient(90), first)

    def test_adding_points_refreshes_stored_orientations(self) -> None:
        section = row()
        before = section.orient(90)
        section.points.add(annotation((0, 1, 0), name="g"))
        after = section.orient(90)
        self.assertEqual(len(after), len(before) + 1)
        self.assertIs(section.orient(90), after)

    def test_turns_must_be_multiples_of_45(self) -> None:
        with self.assertRaises(ValueError):
            row().orient(30)


class DiagonalLayoutTests(unittest.TestCase):
    def test_forty_five_joins_zero_and_ninety(self) -> None:
        # The west layout moves forward 3 and the north layout, joined at f,
        # backtracks 3; they meet at the north-west corner.
        self.assertEqual(
            layout(row(), 45),
            line(NAMES, (-3, 3), (0, -1)) + line(NAMES, (-3, -3), (1, 0)),
        )

    def test_one_thirty_five_joins_ninety_and_one_eighty(self) -> None:
        self.assertEqual(
            layout(row(), 135),
            line(NAMES, (-3, -3), (1, 0)) + line(NAMES, (3, -3), (0, 1)),
        )

    def test_two_twenty_five_joins_one_eighty_and_two_seventy(self) -> None:
        self.assertEqual(
            layout(row(), 225),
            line(NAMES, (3, -3), (0, 1)) + line(NAMES, (3, 3), (-1, 0)),
        )

    def test_three_fifteen_joins_two_seventy_and_zero(self) -> None:
        self.assertEqual(
            layout(row(), 315),
            line(NAMES, (3, 3), (-1, 0)) + line(NAMES, (-3, 3), (0, -1)),
        )

    def test_half_the_long_length_rounds_down(self) -> None:
        section = CrossSection(
            PointField(annotation((x, 0, 0), name=str(x)) for x in range(-3, 5))
        )
        self.assertEqual(section.long_length, 8)
        offsets = section.orient(135)
        # The north layout moves forward 4; the east layout starts one below it.
        self.assertEqual({o.offsets[0].z for o in offsets[:8]}, {-4})
        self.assertEqual(offsets[8].offsets[0], BlockPos(4, 0, -3))

    def test_each_feature_point_exists_once_on_its_path(self) -> None:
        # A single point has nothing to shift, so both layouts land on it; the
        # feature path keeps it once.
        section = CrossSection(PointField((annotation((0, 0, 0)),)))
        self.assertEqual(layout(section, 45), [("a", 0, 0), ("a", 0, 0)])
        feature_path = FeaturePath()
        for offset in section.orient(45):
            for position in offset.offsets:
                feature_path.add(position, GRADIENT, 45, 0, 64)
        self.assertEqual(len(feature_path), 1)

    def test_empty_sections_orient_to_nothing(self) -> None:
        self.assertEqual(CrossSection().orient(45), ())


class HeadingTests(unittest.TestCase):
    def test_quarters_run_clockwise_from_west(self) -> None:
        self.assertEqual(heading(0), (-1, 0))
        self.assertEqual(heading(90), (0, -1))
        self.assertEqual(heading(180), (1, 0))
        self.assertEqual(heading(270), (0, 1))

    def test_diagonals_head_between_their_quarters(self) -> None:
        north_west = heading(45)
        self.assertAlmostEqual(north_west.x, -(2 ** -0.5))
        self.assertAlmostEqual(north_west.z, -(2 ** -0.5))
        self.assertEqual(heading(-45), heading(315))

    def test_turns_off_the_45_grid_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            heading(30)


class ReachTests(unittest.TestCase):
    def test_a_straight_row_at_a_quarter_sits_on_the_point(self) -> None:
        self.assertEqual(row().reach(0), 0.0)
        self.assertEqual(row().reach(90), 0.0)

    def test_diagonals_reach_by_their_shifted_corners(self) -> None:
        # the 45 layout's far corner sits at (-3, -3), three diagonal steps ahead
        self.assertAlmostEqual(row().reach(45), 3 * 2 ** 0.5)

    def test_manhattan_leads_reach_along_travel(self) -> None:
        section = ManhattanLeadTests().section("de")
        self.assertEqual(section.reach(0), 2.0)

    def test_without_a_turn_it_is_the_worst_turn(self) -> None:
        section = row()
        self.assertEqual(section.reach(), max(section.reach(t) for t in range(0, 360, 45)))

    def test_empty_sections_reach_nothing(self) -> None:
        self.assertEqual(CrossSection().reach(), 0.0)


class TurnSenseTests(unittest.TestCase):
    # Which way a path turned is the sign of the cross product of the heading
    # before it with the heading at it.

    def test_the_sign_of_the_cross_product_is_the_direction(self) -> None:
        self.assertEqual(sense((-1, 0, 0), (0, 0, -1)), CLOCKWISE)
        self.assertEqual(sense((-1, 0, 0), (0, 0, 1)), COUNTER_CLOCKWISE)

    def test_headings_along_one_line_keep_the_direction(self) -> None:
        for ahead in ((-1, 0, 0), (1, 0, 0)):
            with self.subTest(ahead=ahead):
                self.assertEqual(sense((-1, 0, 0), ahead, CLOCKWISE), CLOCKWISE)
                self.assertEqual(
                    sense((-1, 0, 0), ahead, COUNTER_CLOCKWISE), COUNTER_CLOCKWISE
                )

    def test_a_bend_too_slight_to_change_the_turn_still_has_a_direction(self) -> None:
        bend = (-1.0, 0.0, 0.05)
        self.assertEqual(snap(bend), snap((-1.0, 0.0, 0.0)))
        self.assertEqual(sense((-1, 0, 0), bend), COUNTER_CLOCKWISE)

    def test_length_does_not_change_the_sign(self) -> None:
        self.assertEqual(sense((-3, 0, 0), (0, 0, -7)), CLOCKWISE)

    def test_a_heading_works_as_well_as_a_gradient(self) -> None:
        # Two components rather than three: what SectionTurn.travel gives.
        self.assertEqual(sense((-1, 0), (0, 1)), COUNTER_CLOCKWISE)


WEST = (-1.0, 0.0, 0.0)
NORTH = (0.0, 0.0, -1.0)
SOUTH = (0.0, 0.0, 1.0)
EAST = (1.0, 0.0, 0.0)
SOUTHWEST = (-1.0, 0.0, 1.0)


class TurnOfTests(unittest.TestCase):
    def test_gradients_snap_to_a_world_direction(self) -> None:
        self.assertEqual(snap(WEST), 0)
        self.assertEqual(snap(NORTH), 90)
        self.assertEqual(snap(SOUTH), 270)
        self.assertEqual(snap(SOUTHWEST), 315)

    def test_only_a_near_exact_diagonal_lays_out_as_one(self) -> None:
        # The tolerance is a little over a degree, so a heading a few degrees
        # off the diagonal takes the quarter it is nearest instead.
        self.assertEqual(snap((-1.0, 0.0, 0.99)), 315)
        self.assertEqual(snap((-1.0, 0.0, 0.9)), 0)
        self.assertEqual(snap((-1.0, 0.0, 0.1)), 0)

    def test_west_to_north_is_clockwise_and_west_to_south_counter(self) -> None:
        north = turn_of(NORTH, WEST, turn_from(WEST))
        south = turn_of(SOUTH, WEST, turn_from(WEST))
        self.assertEqual((north.turn, north.direction), (90, CLOCKWISE))
        self.assertEqual((south.turn, south.direction), (90, COUNTER_CLOCKWISE))

    def test_both_ways_round_reach_the_same_heading(self) -> None:
        left = turn_of(SOUTH, WEST, turn_from(WEST))
        right = turn_of(SOUTH, EAST, turn_from(EAST))
        self.assertEqual(left.absolute, right.absolute)
        self.assertEqual(left.travel, right.travel)
        self.assertEqual((left.turn, left.direction), (90, COUNTER_CLOCKWISE))
        self.assertEqual((right.turn, right.direction), (270, CLOCKWISE))

    def test_a_turn_keeps_counting_the_way_it_was_going(self) -> None:
        # West, then round to the left: the turn grows and stays counter.
        self.assertEqual(
            [(t.turn, t.direction) for t in turns_along([WEST, SOUTHWEST, SOUTH, EAST])],
            [(0, CLOCKWISE), (45, COUNTER_CLOCKWISE), (90, COUNTER_CLOCKWISE), (180, COUNTER_CLOCKWISE)],
        )

    def test_the_first_gradient_is_the_one_it_was_authored_for(self) -> None:
        # A path that sets off south and stays there never turned, so it is
        # laid out as authored rather than mirrored.
        self.assertEqual(
            [(t.turn, t.direction) for t in turns_along([SOUTH, SOUTH, SOUTH])],
            [(270, CLOCKWISE)] * 3,
        )

    def test_a_turn_lays_its_section_out_travelling_the_way_it_points(self) -> None:
        for gradient in (WEST, NORTH, SOUTH, EAST, SOUTHWEST):
            with self.subTest(gradient=gradient):
                section_turn = turn_of(gradient, WEST, turn_from(WEST))
                self.assertEqual(
                    tuple(round(v, 6) for v in section_turn.travel),
                    tuple(round(v, 6) for v in Vec3(*gradient).horizontal.unit()),
                )


def toward(turn):
    # A gradient pointing `turn` degrees clockwise from west, and negative
    # degrees counter-clockwise from it.
    return (-cos(radians(turn)), 0.0, -sin(radians(turn)))


class TurnsAlongTests(unittest.TestCase):
    # The turn at each point comes from one angle carried along the path, so
    # it is the trajectory that decides the layout rather than the last two
    # gradients on their own.

    def turns(self, *gradients):
        return [(t.turn, t.direction) for t in turns_along(gradients)]

    def test_an_angle_is_signed_the_way_a_turn_is(self) -> None:
        self.assertAlmostEqual(angle_between(toward(0), toward(90)), 90.0)
        self.assertAlmostEqual(angle_between(toward(0), toward(-90)), -90.0)
        self.assertAlmostEqual(angle_between(toward(30), toward(45)), 15.0)

    def test_a_reversal_reads_as_half_a_turn_clockwise(self) -> None:
        # It has no side to be on, so the sign is the one atan2 hands back.
        self.assertAlmostEqual(angle_between(toward(0), toward(180)), 180.0)

    def test_a_wobble_inside_a_curve_does_not_turn_the_section_round(self) -> None:
        # The two degrees back the wrong way take the angle from 10 to 8, and
        # nothing else: the section stays laid out the way the curve is going.
        self.assertEqual(
            self.turns(toward(0), toward(10), toward(8), toward(30), toward(45)),
            [(0, CLOCKWISE), (0, CLOCKWISE), (0, CLOCKWISE), (0, CLOCKWISE), (45, CLOCKWISE)],
        )

    def test_bends_too_slight_to_be_a_turn_add_up_to_one(self) -> None:
        bends = [toward(-12 * step) for step in range(9)]
        # Each bend on its own is a heading that lays out as authored.
        self.assertEqual(snap(bends[1]), 0)
        self.assertEqual(self.turns(*bends[:4])[-1], (0, CLOCKWISE))
        # Eight of them are a quarter turn counter-clockwise.
        self.assertEqual(self.turns(*bends)[-1], (90, COUNTER_CLOCKWISE))

    def test_a_path_can_turn_past_a_full_circle(self) -> None:
        loop = [toward(-45 * step) for step in range(10)]
        self.assertEqual(
            self.turns(*loop)[-4:],
            [(270, COUNTER_CLOCKWISE), (315, COUNTER_CLOCKWISE), (0, COUNTER_CLOCKWISE), (45, COUNTER_CLOCKWISE)],
        )


class CounterClockwiseTests(unittest.TestCase):
    # A turn says how far round from the authored heading a section is laid
    # out; the direction says which way round it was counted. Counting the
    # other way mirrors the whole layout about the x axis, travel included.

    def section(self, step_mode=StepMode.CHEBYSHEV):
        # a and b either side of the origin, with f one ahead of it.
        return CrossSection(
            PointField(
                (
                    annotation((-2, 0, 0), step_mode, "a"),
                    annotation((0, 0, 0), step_mode, "o"),
                    annotation((3, 0, 0), step_mode, "b"),
                    annotation((0, 0, -1), step_mode, "f"),
                )
            )
        )

    def offsets(self, section, turn, direction=CLOCKWISE):
        return [
            (o.annotation.name, [(p.x, p.z) for p in o.offsets])
            for o in section.orient(turn, direction)
        ]

    def test_every_turn_is_its_clockwise_twin_mirrored_about_the_x_axis(self) -> None:
        for step_mode in (StepMode.CHEBYSHEV, StepMode.MANHATTAN):
            for turn in range(0, 360, 45):
                with self.subTest(step_mode=step_mode, turn=turn):
                    section = self.section(step_mode)
                    self.assertEqual(
                        self.offsets(section, turn, COUNTER_CLOCKWISE),
                        [
                            (name, [(x, -z) for x, z in positions])
                            for name, positions in self.offsets(section, turn)
                        ],
                    )

    def test_the_turns_that_travel_the_same_either_way_still_mirror(self) -> None:
        # Nothing and a half turn travel the same both ways round, but which
        # way the path came round to them is not the same, so a direction
        # carried into them still lays the section out mirrored.
        for turn in (0, 180):
            with self.subTest(turn=turn):
                self.assertEqual(travel(turn, COUNTER_CLOCKWISE), travel(turn))
                self.assertEqual(
                    layout(row(), turn, COUNTER_CLOCKWISE),
                    [(name, x, -z) for name, x, z in layout(row(), turn)],
                )
        self.assertEqual(layout(row(), 0), line(NAMES, (0, 3), (0, -1)))
        self.assertEqual(
            layout(row(), 0, COUNTER_CLOCKWISE), line(NAMES, (0, -3), (0, 1))
        )

    def test_a_quarter_counted_the_other_way_travels_the_other_way(self) -> None:
        # 90 clockwise from west is north; counted counter-clockwise it is
        # south, and the section that faced north now faces south.
        self.assertEqual(tuple(travel(90)), (0, -1))
        self.assertEqual(tuple(travel(90, COUNTER_CLOCKWISE)), (0, 1))
        self.assertEqual(self.offsets(self.section(), 90)[-1], ("f", [(0, 1)]))
        self.assertEqual(
            self.offsets(self.section(), 90, COUNTER_CLOCKWISE)[-1], ("f", [(0, -1)])
        )

    def test_a_diagonal_staggers_from_the_other_half(self) -> None:
        # The 45 layout leads with its west half and lays the north half
        # behind it; counted counter-clockwise the trailing half is the south
        # one, so the stagger leans the other way.
        section = self.section()
        self.assertEqual(
            [positions for _, positions in self.offsets(section, 45)[4:]],
            [[(-2, -3)], [(0, -3)], [(3, -3)], [(0, -2)]],
        )
        self.assertEqual(
            [
                positions
                for _, positions in self.offsets(section, 45, COUNTER_CLOCKWISE)[4:]
            ],
            [[(-2, 3)], [(0, 3)], [(3, 3)], [(0, 2)]],
        )

    def test_each_direction_is_laid_out_and_kept_on_its_own(self) -> None:
        section = self.section()
        clockwise = section.orient(90)
        self.assertIsNot(clockwise, section.orient(90, COUNTER_CLOCKWISE))
        self.assertIs(clockwise, section.orient(90))
        self.assertIs(
            section.orient(90, COUNTER_CLOCKWISE),
            section.orient(90, COUNTER_CLOCKWISE),
        )

    def test_a_mirrored_layout_reaches_the_same_distance(self) -> None:
        section = self.section(StepMode.MANHATTAN)
        for turn in range(0, 360, 45):
            with self.subTest(turn=turn):
                furthest = max(
                    abs(x * travel(turn, COUNTER_CLOCKWISE).x + z * travel(turn, COUNTER_CLOCKWISE).z)
                    for _, positions in self.offsets(section, turn, COUNTER_CLOCKWISE)
                    for x, z in positions
                )
                self.assertAlmostEqual(furthest, section.reach(turn))


class StepRuleTests(unittest.TestCase):
    def test_orient_annotation_takes_its_lead_and_steps_explicitly(self) -> None:
        offset = orient_annotation(
            annotation((2, 0, 0)), BlockPos(0, 0, 0), 90, lead=-1, steps=manhattan_step(2)
        )
        self.assertEqual(offset.offsets, (BlockPos(2, 0, 1), BlockPos(2, 0, 2)))

    def test_columns_push_the_points_beyond_them_on_their_side(self) -> None:
        pushes = column_pushes([(-1, (0, 1)), (0, (0,)), (1, (0, -1)), (1, (0,))])
        self.assertEqual(pushes, {-1: 1, 1: -1})
        self.assertEqual([lead(column, pushes) for column in range(-3, 4)], [1, 1, 0, 0, 0, -1, -1])

    def test_sections_accept_custom_step_rules(self) -> None:
        def double_step(column):
            return (0, 1, 2)

        rules = {**STEP_RULES, StepMode.MANHATTAN: double_step}
        section = CrossSection(
            PointField(
                (
                    annotation((0, 0, 0), StepMode.MANHATTAN, "inner"),
                    annotation((-1, 0, 0), name="outer"),
                )
            ),
            step_rules=rules,
        )
        offsets = {o.annotation.name: o.offsets for o in section.orient(90)}
        self.assertEqual([o.z for o in offsets["inner"]], [0, -1, -2])
        # The last step, two blocks, moves the point beyond it.
        self.assertEqual(offsets["outer"], (BlockPos(-1, 0, -2),))


class ManhattanLeadTests(unittest.TestCase):
    def section(self, manhattan):
        return CrossSection(
            PointField(
                annotation(
                    (index - 3, 0, 0),
                    StepMode.MANHATTAN if name in manhattan else StepMode.CHEBYSHEV,
                    name,
                )
                for index, name in enumerate(NAMES)
            )
        )

    def depths(self, section, turn=90):
        return {
            o.annotation.name: (o.offsets[0].x, o.offsets[0].z)
            for o in section.orient(turn)
        }

    def test_points_right_of_a_right_manhattan_point_move_backward(self) -> None:
        depths = self.depths(self.section("d"))
        self.assertEqual([depths[n][1] for n in NAMES], [0, 0, 0, 0, 0, 1, 1])

    def test_points_left_of_a_left_manhattan_point_move_forward(self) -> None:
        depths = self.depths(self.section("c"))
        self.assertEqual([depths[n][1] for n in NAMES], [-1, -1, 0, 0, 0, 0, 0])

    def test_moves_add_up_across_manhattan_points(self) -> None:
        depths = self.depths(self.section("bcde"))
        self.assertEqual([depths[n][1] for n in NAMES], [-2, -1, 0, 0, 0, 1, 2])

    def test_the_origin_column_moves_the_left_side(self) -> None:
        depths = self.depths(self.section("o"))
        self.assertEqual([depths[n][1] for n in NAMES], [-1, -1, -1, 0, 0, 0, 0])

    def test_manhattan_points_in_one_column_move_the_points_beyond_once(self) -> None:
        section = CrossSection(
            PointField(
                (
                    annotation((1, 0, 0), StepMode.MANHATTAN, "low"),
                    annotation((1, 2, 0), StepMode.MANHATTAN, "high"),
                    annotation((2, 0, 0), name="out"),
                )
            )
        )
        self.assertEqual(self.depths(section)["out"], (2, 1))

    def test_moves_follow_the_quarter_layout(self) -> None:
        # At 180, forward is east, so the right side moves west.
        depths = self.depths(self.section("d"), turn=180)
        self.assertEqual(depths["e"], (-1, 2))
        self.assertEqual(depths["d"], (0, 1))

    def test_moved_manhattan_points_still_step_from_where_they_moved(self) -> None:
        offsets = {o.annotation.name: o for o in self.section("de").orient(90)}
        self.assertEqual([tuple(p) for p in offsets["e"].offsets], [(2, 0, 1), (2, 0, 2)])


class FeaturePathTests(unittest.TestCase):
    def add(self, feature_path, path_index, *positions):
        return [
            feature_path.add(position, GRADIENT, 0, path_index, 64) is not None
            for position in positions
        ]

    def positions(self, feature_path):
        return [tuple(p.position) for p in feature_path]

    def test_points_from_the_previous_delivery_are_not_duplicated(self) -> None:
        feature_path = FeaturePath()
        self.add(feature_path, 0, (0, 0, 0), (0, 0, -1))
        self.assertEqual(self.add(feature_path, 1, (0, 0, -1), (0, 0, -2)), [False, True])
        self.assertEqual(self.positions(feature_path), [(0, 0, 0), (0, 0, -1), (0, 0, -2)])
        self.assertEqual([p.point_index for p in feature_path], [0, 1, 2])

    def test_points_within_a_delivery_are_not_duplicated(self) -> None:
        feature_path = FeaturePath()
        self.assertEqual(self.add(feature_path, 0, (1, 0, 0), (2, 0, 0), (1, 0, 0)), [True, True, False])

    def test_paths_may_loop_back_over_older_deliveries(self) -> None:
        feature_path = FeaturePath()
        square = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 0, 0)]
        for path_index, position in enumerate(square):
            self.add(feature_path, path_index, position)
        self.assertEqual(self.positions(feature_path), square)

    def test_the_previous_delivery_skips_deliveries_that_kept_nothing(self) -> None:
        feature_path = FeaturePath()
        self.add(feature_path, 0, (0, 0, 0))
        self.assertEqual(self.add(feature_path, 1, (0, 0, 0)), [False])
        self.assertEqual(self.add(feature_path, 2, (0, 0, 0)), [False])
        self.assertEqual(feature_path.previous_delivery[0].path_index, 0)

    def test_rules_are_supplied_not_inherited(self) -> None:
        def below_y_ten(path, point):
            return point.position.y < 10

        feature_path = FeaturePath(rules=(below_y_ten,))
        self.assertEqual(self.add(feature_path, 0, (0, 0, 0), (0, 0, 0), (0, 20, 0)), [True, True, False])

        with_defaults = FeaturePath(rules=(*DEFAULT_RULES, below_y_ten))
        self.assertEqual(self.add(with_defaults, 0, (0, 0, 0), (0, 0, 0), (0, 20, 0)), [True, False, False])

    def test_points_keep_the_direction_they_were_laid_out_in(self) -> None:
        feature_path = FeaturePath()
        feature_path.add((0, 0, 0), GRADIENT, 0, 0, 64)
        feature_path.add((1, 0, 0), GRADIENT, 315, 1, 64, COUNTER_CLOCKWISE)
        self.assertEqual(
            [(p.turn, p.direction) for p in feature_path],
            [(0, CLOCKWISE), (315, COUNTER_CLOCKWISE)],
        )

    def test_rules_see_the_path_before_the_point_is_added(self) -> None:
        seen = []

        def record(path, point):
            seen.append((len(path), len(path.current_delivery), point.point_index))
            return True

        feature_path = FeaturePath(rules=(record,))
        self.add(feature_path, 0, (0, 0, 0), (1, 0, 0))
        self.add(feature_path, 1, (2, 0, 0))
        self.assertEqual(seen, [(0, 0, 0), (1, 1, 1), (2, 0, 2)])
        self.assertEqual(feature_path[0].y_offset, 64)


class FeatureTests(unittest.TestCase):
    def test_every_feature_matches_the_interface_without_inheriting(self) -> None:
        for cls in (PointFeature, ProfileFeature, ContinuousFeature):
            instance = cls(BlockField(), PlacementMode.OVERWRITE, Domain(7, "test"))
            self.assertIsInstance(instance, Feature)
            self.assertEqual(cls.__mro__, (cls, object))
            self.assertEqual(instance.offset, BlockPos(0, 0, 0))
            self.assertIs(instance.elevation_mode, ElevationMode.DEFORMABLE)

    def test_custom_features_only_need_the_same_shape(self) -> None:
        class Custom:
            blocks = BlockField()
            placement_mode = PlacementMode.OVERWRITE
            domain = Domain(1, "custom")
            offset = BlockPos(0, 0, 0)
            elevation_mode = ElevationMode.RIGID

        self.assertIsInstance(Custom(), Feature)

    def test_feature_classes_get_the_shared_fields_ahead_of_their_own(self) -> None:
        @feature_class
        class Lamp:
            spacing: int = 4

        lamp = Lamp(BlockField(), PlacementMode.OVERWRITE, Domain(1, "lamp"), (0, 1, 0))
        self.assertIsInstance(lamp, Feature)
        self.assertEqual(lamp.offset, BlockPos(0, 1, 0))
        self.assertEqual(lamp.spacing, 4)
        self.assertEqual(Lamp.__mro__, (Lamp, object))

    def test_feature_classes_keep_what_they_declare_themselves(self) -> None:
        @feature_class
        class Rigid:
            elevation_mode: ElevationMode = ElevationMode.RIGID

            def __post_init__(self) -> None:
                self.offset = BlockPos(9, 9, 9)

        rigid = Rigid(BlockField(), PlacementMode.OVERWRITE, Domain(1, "rigid"))
        self.assertIs(rigid.elevation_mode, ElevationMode.RIGID)
        self.assertEqual(rigid.offset, BlockPos(9, 9, 9))


if __name__ == "__main__":
    unittest.main()
