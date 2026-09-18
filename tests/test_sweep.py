import unittest
from math import cos, radians, sin

from src.Core.CrossSection.annotations import FeatureAnnotation
from src.Core.CrossSection.cross_section import CrossSection, PointField, orient_annotation
from src.Core.CrossSection.schedule import SectionSchedule, SectionSpan
from src.Core.CrossSection.turns import heading, snap
from src.Core.Domain import Domain
from src.Core.Feature.feature import PointFeature
from src.Core.Feature.feature_path import FeaturePath
from src.Core.Field.block_field import BlockField
from src.Core.Path.boundary import Boundary
from src.Core.Path.path import Path, PathPoint
from src.Core.Path.path_frame import PathFrame
from src.Core.coordinates import BlockPos, Vec2, Vec3
from src.Core.modes import PlacementMode, StepMode
from src.Generation.Sweep.delivery import FrameCover, cover, deliver, deliver_offset
from src.Generation.Sweep.schedule import lead_for, walk_schedule
from src.Generation.Geometry.boundaries import orthogonal, path_bounds
from src.Generation.Geometry.orientation import SectionOrientation
from src.Generation.Geometry.frames import (
    ElevationBreak,
    build_frames,
    build_section_frames,
    elevation_from_frames,
    span_stations,
)

NAMES = "abcodef"


def feature():
    return PointFeature(BlockField(), PlacementMode.OVERWRITE, Domain(1, "test"))


def annotation(position, step_mode=StepMode.CHEBYSHEV, name="a"):
    return FeatureAnnotation(name, position, feature(), step_mode)


def row(step_mode=StepMode.CHEBYSHEV):
    return CrossSection(
        PointField(
            annotation((index - 3, 0, 0), step_mode, name)
            for index, name in enumerate(NAMES)
        )
    )


def point(x=0, y=0, z=0, gradient=(0.0, 0.0, -1.0)):
    return PathPoint(BlockPos(x, y, z), Vec3(*gradient))


def north(count: int, **padding) -> Path:
    # Travelling north (-z) from z=0, one block per point.
    return Path([(0, 64, -i) for i in range(count)], [(0.0, 0.0, -1.0)] * count, **padding)


class TurnTests(unittest.TestCase):
    def test_quarters_are_world_directions_clockwise_from_west(self) -> None:
        self.assertEqual(snap((-1.0, 0.0, 0.0)), 0)
        self.assertEqual(snap((0.0, 0.0, -1.0)), 90)
        self.assertEqual(snap((1.0, 0.0, 0.0)), 180)
        self.assertEqual(snap((0.0, 0.0, 1.0)), 270)

    def test_diagonals_sit_between_their_quarters(self) -> None:
        self.assertEqual(snap((-1.0, 0.0, -1.0)), 45)   # north-west
        self.assertEqual(snap((1.0, 0.0, -1.0)), 135)   # north-east
        self.assertEqual(snap((1.0, 0.0, 1.0)), 225)    # south-east
        self.assertEqual(snap((-1.0, 0.0, 1.0)), 315)   # south-west

    def test_turns_do_not_depend_on_where_the_path_started(self) -> None:
        # The same heading is the same turn on any path.
        self.assertEqual(snap(Vec3(0.0, 0.0, 1.0)), snap(Vec3(0.0, 0.0, 5.0)))

    def test_magnitude_and_grade_do_not_matter(self) -> None:
        self.assertEqual(snap((3.0, 7.0, -3.0)), 135)

    def test_diagonals_need_a_heading_very_close_to_45(self) -> None:
        self.assertEqual(snap((1.0, 0.0, -1.01)), 135)
        self.assertEqual(snap((1.0, 0.0, -1.2)), 90)
        self.assertEqual(snap((1.0, 0.0, -0.8)), 180)
        self.assertEqual(snap((-1.0, 0.0, 0.8)), 0)

    def test_tolerance_can_be_widened(self) -> None:
        self.assertEqual(snap((1.0, 0.0, -1.2), tolerance=0.2), 135)

    def test_vertical_gradients_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            snap((0.0, 1.0, 0.0))


class SectionOrientationTests(unittest.TestCase):
    def setUp(self) -> None:
        # Starts heading north, then bends 30 degrees east of north.
        bend = (sin(radians(30)), 0.0, -cos(radians(30)))
        self.path = Path(
            [(0, 64, 0), (0, 64, -1), (1, 64, -2), (1, 64, -3)],
            [(0.0, 0.0, -1.0), (0.0, 0.0, -1.0), bend, bend],
            lead_in=1,
        )

    def test_turns_snap_each_point_to_a_world_direction(self) -> None:
        orientation = SectionOrientation()
        self.assertEqual([t.turn for t in orientation.turns(self.path)], [90, 90, 90, 90])

    def test_a_path_starting_east_lays_sections_out_for_east(self) -> None:
        east = Path([(0, 64, 0), (1, 64, 0)], [(1.0, 0.0, 0.0)] * 2)
        section = CrossSection(PointField((annotation((-1, 0, 0), name="left"),)))
        turn = SectionOrientation().turns(east)[0]
        [delivery] = deliver(east.path_points[0], section.orient(turn.turn, turn.direction))
        # left of travel east is north
        self.assertEqual([tuple(p) for p in delivery.positions], [(0, 0, -1)])

    def test_boundaries_stay_orthogonal_by_default(self) -> None:
        orientation = SectionOrientation()
        self.assertEqual(orientation(self.path, 2), orthogonal(self.path, 2))

    def test_snapped_boundaries_face_the_turn_the_section_was_laid_out_for(self) -> None:
        orientation = SectionOrientation(snap_boundaries=True)
        turn = orientation.turns(self.path)[2]
        self.assertEqual(orientation(self.path, 2), turn.travel)
        self.assertNotEqual(orientation(self.path, 2), orthogonal(self.path, 2))

    def test_it_is_a_geometry_orientation(self) -> None:
        orientation = SectionOrientation(snap_boundaries=True)
        [frame] = build_frames(self.path, 64, orientation=orientation)
        self.assertEqual(frame.end_boundary.normal, Vec2(0.0, -1.0))


class DeliverTests(unittest.TestCase):
    def deliver(self, position, turn=90, at=point()):
        section = CrossSection(PointField((annotation(position, StepMode.MANHATTAN),)))
        (offset,) = section.orient(turn)
        return [tuple(p) for p in deliver_offset(at, offset).positions]

    def test_positions_keep_the_local_height(self) -> None:
        offset = orient_annotation(annotation((-1, 2, 0)), BlockPos(0, 0, 0), 90, lead=0)
        delivery = deliver_offset(point(10, 64, 5), offset)
        self.assertEqual([tuple(p) for p in delivery.positions], [(9, 2, 5)])

    def test_manhattan_left_steps_forward(self) -> None:
        self.assertEqual(self.deliver((-1, 0, 0)), [(-1, 0, 0), (-1, 0, -1)])

    def test_manhattan_right_steps_backward(self) -> None:
        self.assertEqual(self.deliver((1, 0, 0)), [(1, 0, 0), (1, 0, 1)])

    def test_manhattan_on_the_origin_steps_forward(self) -> None:
        self.assertEqual(self.deliver((0, 0, 0)), [(0, 0, 0), (0, 0, -1)])

    def test_forward_follows_the_quarter_layout(self) -> None:
        # At 180 the left point sits north of the path and forward is east.
        self.assertEqual(self.deliver((-1, 0, 0), turn=180), [(0, 0, -1), (1, 0, -1)])

    def test_deliver_covers_the_whole_section(self) -> None:
        deliveries = deliver(point(5, 0, 5), row().orient(0))
        self.assertEqual([d.annotation.name for d in deliveries], list(NAMES))

    def test_a_mask_drops_columns_outside_it_and_keeps_the_order(self) -> None:
        section = CrossSection(PointField((annotation((-1, 0, 0), StepMode.MANHATTAN),)))
        # Travelling north, this frame holds only the row at z=0.
        frame = PathFrame(
            0, 0, 1,
            Boundary.from_direction((0.0, 0.0), (0.0, -1.0)),
            Boundary.from_direction((0.0, -1.0), (0.0, -1.0)),
            64,
        )
        [full] = deliver(point(), section.orient(90))
        [clipped] = deliver(point(), section.orient(90), frame)
        self.assertEqual([tuple(p) for p in full.positions], [(-1, 0, 0), (-1, 0, -1)])
        self.assertEqual([tuple(p) for p in clipped.positions], [(-1, 0, 0)])

    def test_annotations_emptied_by_a_mask_still_deliver(self) -> None:
        nowhere = PathFrame(
            0, 0, 1,
            Boundary.from_direction((0.0, 100.0), (0.0, 1.0)),
            Boundary.from_direction((0.0, 101.0), (0.0, 1.0)),
            64,
        )
        deliveries = deliver(point(), row().orient(0), nowhere)
        self.assertEqual([d.positions for d in deliveries], [()] * len(NAMES))


class FrameCoverTests(unittest.TestCase):
    def test_covers_any_of_its_frames(self) -> None:
        frames = build_frames(north(21), 64, [ElevationBreak(10, 65)])
        mask = FrameCover(tuple(frames))
        self.assertTrue(mask.contains(0, -3))
        self.assertTrue(mask.contains(0, -15))
        self.assertFalse(mask.contains(0, 1))

    def test_cover_of_one_frame_is_the_frame(self) -> None:
        [frame] = build_frames(north(5), 64)
        self.assertIs(cover([frame]), frame)
        self.assertIsInstance(cover([frame, frame]), FrameCover)


class WalkScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.road = CrossSection()
        self.bridge = CrossSection()

    def test_walk_skips_gaps_and_yields_path_points(self) -> None:
        path = north(10)
        schedule = SectionSchedule(
            [SectionSpan(6, 9, self.bridge), SectionSpan(0, 3, self.road)],
            path_length=10,
        )
        walked = list(walk_schedule(schedule, path))
        self.assertEqual([step.index for step in walked], [0, 1, 2, 3, 6, 7, 8, 9])
        for step in walked:
            self.assertIs(step.point, path.path_points[step.index])
            self.assertIs(step.section, self.road if step.index <= 3 else self.bridge)
            self.assertEqual(step.frames, ())

    def test_an_empty_schedule_walks_nothing(self) -> None:
        self.assertEqual(list(walk_schedule(SectionSchedule([], 5), north(5))), [])

    def test_frames_come_with_each_point_and_unreached_padding_is_skipped(self) -> None:
        path = north(25, lead_in=2, lead_out=3)
        frames = build_frames(path, 64, [ElevationBreak(10, 65)])
        schedule = SectionSchedule([SectionSpan(0, 24, self.road)], 25)
        walked = {step.index: [f.index for f in step.frames] for step in walk_schedule(schedule, path, frames, 1, 1)}
        self.assertEqual(min(walked), 1)
        self.assertEqual(max(walked), 22)
        self.assertEqual(walked[5], [0])
        self.assertEqual(walked[9], [0, 1])
        self.assertEqual(walked[11], [0, 1])
        self.assertEqual(walked[12], [1])


class SpanStationsTests(unittest.TestCase):
    def test_section_changes_and_gaps_are_stations(self) -> None:
        road, bridge = CrossSection(), CrossSection()
        path = north(25, lead_in=2, lead_out=3)
        schedule = SectionSchedule(
            [SectionSpan(0, 7, road), SectionSpan(8, 13, bridge), SectionSpan(16, 24, road)],
            25,
        )
        self.assertEqual(span_stations(schedule, path), [8, 14, 16])

    def test_changes_off_the_path_proper_are_not_stations(self) -> None:
        path = north(25, lead_in=2, lead_out=3)
        schedule = SectionSchedule([SectionSpan(0, 1, CrossSection()), SectionSpan(2, 24, CrossSection())], 25)
        self.assertEqual(span_stations(schedule, path), [])


class LeadForTests(unittest.TestCase):
    def test_reach_over_the_tightest_spacing(self) -> None:
        path = Path([(0, 0, 0), (0, 0, -3), (0, 0, -5), (0, 0, -9)], [(0.0, 0.0, -1.0)] * 4)
        self.assertEqual(lead_for(path, 4.0), 2)
        self.assertEqual(lead_for(path, 4.5), 3)
        self.assertEqual(lead_for(path, 0.0), 0)

    def test_repeated_points_do_not_count_as_spacing(self) -> None:
        path = Path([(0, 0, 0), (0, 0, 0), (0, 0, -2)], [(0.0, 0.0, -1.0)] * 3)
        self.assertEqual(lead_for(path, 3.0), 2)

    def test_negative_reach(self) -> None:
        with self.assertRaises(ValueError):
            lead_for(north(3), -1.0)


class FeaturePathAlongFramesTests(unittest.TestCase):
    def test_deliveries_along_a_path_extend_the_feature_path(self) -> None:
        path = Path([(0, 0, 0), (0, 0, -1), (0, 0, -2)], [(0.0, 0.0, -1.0)] * 3)
        section = CrossSection(PointField((annotation((-1, 0, 0), StepMode.MANHATTAN),)))
        orientation = SectionOrientation()
        feature_path = FeaturePath()
        turns = orientation.turns(path)
        for path_index, path_point in enumerate(path.path_points):
            turn = turns[path_index]
            oriented = section.orient(turn.turn, turn.direction)
            for delivery in deliver(path_point, oriented):
                for position in delivery.positions:
                    feature_path.add(
                        position, path_point.gradient, turn.turn, path_index, 64, turn.direction
                    )

        self.assertEqual([p.position.z for p in feature_path], [0, -1, -2, -3])
        self.assertEqual([p.path_index for p in feature_path], [0, 0, 1, 2])
        self.assertEqual([p.point_index for p in feature_path], [0, 1, 2, 3])

    def test_a_padded_scheduled_path_delivers_only_inside_its_frames(self) -> None:
        # Path proper is points 2-21 (z -2..-21), stepping up at point 8, with
        # the section changing at point 12. Padding carries sections too.
        path = north(25, lead_in=2, lead_out=3)
        left = lambda: CrossSection(PointField((annotation((-1, 0, 0)),)))
        schedule = SectionSchedule([SectionSpan(0, 11, left()), SectionSpan(12, 24, left())], 25)
        orientation = SectionOrientation()

        elevation = build_frames(path, 64, [ElevationBreak(8, 65)], orientation=orientation)
        stations = sorted({*span_stations(schedule, path), *(f.path_start for f in elevation[1:])})
        sections = build_section_frames(path, elevation, stations, orientation=orientation)
        self.assertEqual([f.path_start for f in sections], [2, 8, 12])

        reach = max(span.section.reach() for span in schedule.spans)
        lead = lead_for(path, reach)
        level = elevation_from_frames(elevation)
        feature_path = FeaturePath()
        turns = orientation.turns(path)
        for step in walk_schedule(schedule, path, sections, lead, lead):
            turn = turns[step.index]
            oriented = step.section.orient(turn.turn, turn.direction)
            for delivery in deliver(step.point, oriented, cover(step.frames)):
                for position in delivery.positions:
                    feature_path.add(
                        position,
                        step.point.gradient,
                        turn.turn,
                        step.index,
                        level(position.x, position.z),
                        turn.direction,
                    )

        bounds = path_bounds(path)
        self.assertTrue(all(bounds.contains(p.position.x, p.position.z) for p in feature_path))
        self.assertEqual([tuple(p.position) for p in feature_path], [(-1, 0, -z) for z in range(2, 21)])
        self.assertEqual([p.y_offset for p in feature_path], [64 if z < 8 else 65 for z in range(2, 21)])


if __name__ == "__main__":
    unittest.main()
