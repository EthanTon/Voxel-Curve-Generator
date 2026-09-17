import unittest

from src.Core.CrossSection.annotations import FeatureAnnotation
from src.Core.CrossSection.cross_section import (
    CrossSection,
    PointField,
    orient_annotation,
)
from src.Core.CrossSection.delivery import deliver, deliver_offset
from src.Core.CrossSection.steps import (
    STEP_RULES,
    column_pushes,
    lead,
    manhattan_step,
)
from src.Core.CrossSection.turns import turn_of
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
from src.Core.Path.path import Path, PathPoint
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


def point(x=0, y=0, z=0, gradient=(0.0, 0.0, -1.0)):
    return PathPoint(BlockPos(x, y, z), Vec3(*gradient))


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


class TurnTests(unittest.TestCase):
    NORTH = Vec3(0.0, 0.0, -1.0)
    EAST = Vec3(1.0, 0.0, 0.0)

    def test_the_initial_gradient_is_turn_zero(self) -> None:
        self.assertEqual(turn_of(self.NORTH, self.NORTH), 0)
        self.assertEqual(turn_of(self.EAST, self.EAST), 0)

    def test_turns_run_clockwise_from_the_initial_gradient(self) -> None:
        self.assertEqual(turn_of((1.0, 0.0, -1.0), self.NORTH), 45)
        self.assertEqual(turn_of((1.0, 0.0, 0.0), self.NORTH), 90)
        self.assertEqual(turn_of((1.0, 0.0, 1.0), self.NORTH), 135)
        self.assertEqual(turn_of((0.0, 0.0, 1.0), self.NORTH), 180)
        self.assertEqual(turn_of((-1.0, 0.0, 1.0), self.NORTH), 225)
        self.assertEqual(turn_of((-1.0, 0.0, 0.0), self.NORTH), 270)
        self.assertEqual(turn_of((-1.0, 0.0, -1.0), self.NORTH), 315)
        # Starting east, heading south is a quarter turn clockwise.
        self.assertEqual(turn_of((0.0, 0.0, 1.0), self.EAST), 90)
        self.assertEqual(turn_of((1.0, 0.0, 1.0), self.EAST), 45)

    def test_magnitude_and_grade_do_not_matter(self) -> None:
        self.assertEqual(turn_of((3.0, 7.0, -3.0), (0.0, -2.0, -5.0)), 45)

    def test_diagonals_need_a_heading_very_close_to_45(self) -> None:
        self.assertEqual(turn_of((1.0, 0.0, -1.01), self.NORTH), 45)
        self.assertEqual(turn_of((1.0, 0.0, -1.2), self.NORTH), 0)
        self.assertEqual(turn_of((1.0, 0.0, -0.8), self.NORTH), 90)
        self.assertEqual(turn_of((-1.0, 0.0, 0.8), self.NORTH), 270)

    def test_tolerance_can_be_widened(self) -> None:
        self.assertEqual(turn_of((1.0, 0.0, -1.2), self.NORTH, tolerance=0.2), 45)

    def test_vertical_gradients_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            turn_of((0.0, 1.0, 0.0), self.NORTH)
        with self.assertRaises(ValueError):
            turn_of(self.NORTH, (0.0, -1.0, 0.0))


class QuarterLayoutTests(unittest.TestCase):
    def test_zero_uses_the_section_as_authored(self) -> None:
        self.assertEqual(layout(row(), 0), line(NAMES, (-3, 0), (1, 0)))

    def test_ninety_rotates_then_mirrors(self) -> None:
        # a (left) at the top, f (right) at the bottom.
        self.assertEqual(layout(row(), 90), line(NAMES, (0, -3), (0, 1)))

    def test_one_eighty(self) -> None:
        self.assertEqual(layout(row(), 180), line(NAMES, (3, 0), (-1, 0)))

    def test_two_seventy(self) -> None:
        self.assertEqual(layout(row(), 270), line(NAMES, (0, 3), (0, -1)))

    def test_mirroring_flips_depth_on_odd_quarters(self) -> None:
        section = CrossSection(PointField((annotation((1, 0, -1)),)))
        self.assertEqual(layout(section, 90), [("a", -1, 1)])
        self.assertEqual(layout(section, 180), [("a", -1, 1)])
        self.assertEqual(layout(section, 270), [("a", 1, -1)])

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
        # The 0 layout moves forward (north) 3 and the 90 layout, joined at
        # f, backtracks 3; they meet at the north-east corner.
        self.assertEqual(
            layout(row(), 45),
            line(NAMES, (-3, -3), (1, 0)) + line(NAMES, (3, -3), (0, 1)),
        )

    def test_one_thirty_five_joins_ninety_and_one_eighty(self) -> None:
        self.assertEqual(
            layout(row(), 135),
            line(NAMES, (3, -3), (0, 1)) + line(NAMES, (3, 3), (-1, 0)),
        )

    def test_two_twenty_five_joins_one_eighty_and_two_seventy(self) -> None:
        self.assertEqual(
            layout(row(), 225),
            line(NAMES, (3, 3), (-1, 0)) + line(NAMES, (-3, 3), (0, -1)),
        )

    def test_three_fifteen_joins_two_seventy_and_zero(self) -> None:
        self.assertEqual(
            layout(row(), 315),
            line(NAMES, (-3, 3), (0, -1)) + line(NAMES, (-3, -3), (1, 0)),
        )

    def test_half_the_long_length_rounds_down(self) -> None:
        section = CrossSection(
            PointField(annotation((x, 0, 0), name=str(x)) for x in range(-3, 5))
        )
        self.assertEqual(section.long_length, 8)
        offsets = section.orient(45)
        # 0 layout moves forward 4; 90 layout starts one below it.
        self.assertEqual({o.offsets[0].z for o in offsets[:8]}, {-4})
        self.assertEqual(offsets[8].offsets[0], BlockPos(4, 0, -3))

    def test_each_feature_point_exists_once_on_its_path(self) -> None:
        # A single point has nothing to shift, so both layouts land on it; the
        # feature path keeps it once.
        section = CrossSection(PointField((annotation((0, 0, 0)),)))
        self.assertEqual(layout(section, 45), [("a", 0, 0), ("a", 0, 0)])
        feature_path = FeaturePath()
        for delivery in deliver(point(), section.orient(45)):
            for position in delivery.positions:
                feature_path.add(position, GRADIENT, 45, 0, 64)
        self.assertEqual(len(feature_path), 1)

    def test_empty_sections_orient_to_nothing(self) -> None:
        self.assertEqual(CrossSection().orient(45), ())


class StepRuleTests(unittest.TestCase):
    def test_orient_annotation_takes_its_lead_and_steps_explicitly(self) -> None:
        offset = orient_annotation(
            annotation((2, 0, 0)), BlockPos(0, 0, 0), 0, lead=-1, steps=manhattan_step(2)
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
        offsets = {o.annotation.name: o.offsets for o in section.orient(0)}
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

    def depths(self, section, turn=0):
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
        # At 90, forward is east, so the right side moves west.
        depths = self.depths(self.section("d"), turn=90)
        self.assertEqual(depths["e"], (-1, 2))
        self.assertEqual(depths["d"], (0, 1))

    def test_moved_manhattan_points_still_step_from_where_they_moved(self) -> None:
        offsets = {o.annotation.name: o for o in self.section("de").orient(0)}
        delivery = deliver_offset(point(), offsets["e"])
        self.assertEqual([tuple(p) for p in delivery.positions], [(2, 0, 1), (2, 0, 2)])


class DeliverTests(unittest.TestCase):
    def deliver(self, position, turn=0, at=point()):
        section = CrossSection(PointField((annotation(position, StepMode.MANHATTAN),)))
        (offset,) = section.orient(turn)
        return [tuple(p) for p in deliver_offset(at, offset).positions]

    def test_positions_keep_the_local_height(self) -> None:
        offset = orient_annotation(annotation((-1, 2, 0)), BlockPos(0, 0, 0), 0, lead=0)
        delivery = deliver_offset(point(10, 64, 5), offset)
        self.assertEqual([tuple(p) for p in delivery.positions], [(9, 2, 5)])

    def test_manhattan_left_steps_forward(self) -> None:
        self.assertEqual(self.deliver((-1, 0, 0)), [(-1, 0, 0), (-1, 0, -1)])

    def test_manhattan_right_steps_backward(self) -> None:
        self.assertEqual(self.deliver((1, 0, 0)), [(1, 0, 0), (1, 0, 1)])

    def test_manhattan_on_the_origin_steps_forward(self) -> None:
        self.assertEqual(self.deliver((0, 0, 0)), [(0, 0, 0), (0, 0, -1)])

    def test_forward_follows_the_quarter_layout(self) -> None:
        # At 90 the left point sits north of the path and forward is east.
        self.assertEqual(self.deliver((-1, 0, 0), turn=90), [(0, 0, -1), (1, 0, -1)])

    def test_deliver_covers_the_whole_section(self) -> None:
        deliveries = deliver(point(5, 0, 5), row().orient(0))
        self.assertEqual([d.annotation.name for d in deliveries], list(NAMES))


class FeaturePathTests(unittest.TestCase):
    def test_deliveries_along_a_path_extend_the_feature_path(self) -> None:
        path = Path([(0, 0, 0), (0, 0, -1), (0, 0, -2)], [(0.0, 0.0, -1.0)] * 3)
        section = CrossSection(PointField((annotation((-1, 0, 0), StepMode.MANHATTAN),)))
        feature_path = FeaturePath()
        for path_index, path_point in enumerate(path.path_points):
            turn = turn_of(path_point.gradient, path.path_points[0].gradient)
            oriented = section.orient(turn)
            for delivery in deliver(path_point, oriented):
                for position in delivery.positions:
                    feature_path.add(position, path_point.gradient, turn, path_index, 64)

        self.assertEqual([p.position.z for p in feature_path], [0, -1, -2, -3])
        self.assertEqual([p.path_index for p in feature_path], [0, 0, 1, 2])
        self.assertEqual([p.point_index for p in feature_path], [0, 1, 2, 3])

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
