import unittest

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
from src.Core.CrossSection.turns import heading
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


def layout(section, turn):
    return [
        (o.annotation.name, o.offsets[0].x, o.offsets[0].z)
        for o in section.orient(turn)
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
                feature_path.add(position, GRADIENT, 0, 64)
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
            feature_path.add(position, GRADIENT, path_index, 64) is not None
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
