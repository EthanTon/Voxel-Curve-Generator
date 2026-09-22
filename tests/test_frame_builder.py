import unittest
from math import cos, radians, sin

from src.Core.Field.block_field import BlockField
from src.Core.Path.elevation import ConstantElevation, MultiElevationStep
from src.Core.Path.path import Path
from src.Core.World.block_world import BlockTransform, BlockWorld, ColumnMask, YOffset
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.coordinates import Vec2
from src.Generation.Geometry.boundaries import path_bounds
from src.Generation.Geometry.frames import (
    ElevationBreak,
    build_frames,
    build_section_frames,
    check_padding,
    elevation_from_frames,
    frames_for_index,
)


def straight(count: int, **padding) -> Path:
    return Path([(0, 64, z) for z in range(count)], [(0.0, 0.0, 1.0)] * count, **padding)


def arc(radius: float, degrees: float, step: float, lead: int) -> Path:
    # Points on a circle about the origin, `lead` extra steps at either end.
    angles = [radians(-lead * step + i * step) for i in range(int(degrees / step) + 2 * lead + 1)]
    return Path(
        [(round(radius * cos(a)), 64, round(radius * sin(a))) for a in angles],
        [(-sin(a), 0.0, cos(a)) for a in angles],
        lead_in=lead,
        lead_out=lead,
    )


def square(registry: BlockRegistry, half: int) -> BlockField:
    stone = registry.register(BlockState.create("minecraft:stone"))
    field = BlockField(registry.palette)
    for x in range(-half, half + 1):
        for z in range(-half, half + 1):
            field.set_block_id((x, 0, z), stone)
    return field


class BuildFramesTests(unittest.TestCase):
    def test_no_breaks_is_one_frame_over_the_whole_path(self) -> None:
        path = straight(11)
        [frame] = build_frames(path, 64)
        self.assertEqual((frame.index, frame.path_start, frame.path_stop), (0, 0, 10))
        self.assertEqual(frame.y_offset, 64)

    def test_breaks_split_the_path_on_shared_boundaries(self) -> None:
        path = straight(21)
        frames = build_frames(
            path, 64, [ElevationBreak(5, 65), ElevationBreak(12, 63)]
        )
        self.assertEqual(
            [(f.path_start, f.path_stop, f.y_offset) for f in frames],
            [(0, 5, 64), (5, 12, 65), (12, 20, 63)],
        )
        self.assertEqual([f.index for f in frames], [0, 1, 2])
        for before, after in zip(frames, frames[1:]):
            self.assertIs(before.end_boundary, after.start_boundary)

    def test_outer_boundaries_are_the_path_bounds(self) -> None:
        path = straight(25, lead_in=2, lead_out=3)
        frames = build_frames(path, 64, [ElevationBreak(10, 65)])
        bounds = path_bounds(path)
        self.assertEqual(frames[0].start_boundary, bounds.start)
        self.assertEqual(frames[-1].end_boundary, bounds.end)
        self.assertEqual((frames[0].path_start, frames[-1].path_stop), (2, 21))

    def test_frames_tile_the_path_proper(self) -> None:
        path = straight(25, lead_in=2, lead_out=3)
        frames = build_frames(path, 64, [ElevationBreak(8, 65), ElevationBreak(15, 66)])
        bounds = path_bounds(path)
        for x in range(-6, 7):
            for z in range(-3, 28):
                owners = [f for f in frames if f.contains(x, z)]
                self.assertEqual(len(owners), 1 if bounds.contains(x, z) else 0, (x, z))

    def test_a_break_that_keeps_the_level_adds_no_boundary(self) -> None:
        frames = build_frames(
            straight(21), 64, [ElevationBreak(5, 64), ElevationBreak(10, 65)]
        )
        self.assertEqual([(f.path_start, f.y_offset) for f in frames], [(0, 64), (10, 65)])

    def test_breaks_must_increase(self) -> None:
        with self.assertRaises(ValueError):
            build_frames(straight(21), 64, [ElevationBreak(10, 65), ElevationBreak(10, 66)])
        with self.assertRaises(ValueError):
            build_frames(straight(21), 64, [ElevationBreak(10, 65), ElevationBreak(4, 66)])

    def test_skipped_breaks_still_count_for_ordering(self) -> None:
        with self.assertRaises(ValueError):
            build_frames(straight(21), 64, [ElevationBreak(10, 64), ElevationBreak(4, 66)])

    def test_breaks_must_fall_inside_the_path_proper(self) -> None:
        path = straight(25, lead_in=2, lead_out=3)
        for index in (0, 2, 21, 23):
            with self.assertRaises(ValueError, msg=index):
                build_frames(path, 64, [ElevationBreak(index, 65)])

    def test_paths_that_cannot_be_framed(self) -> None:
        with self.assertRaises(ValueError):
            build_frames(Path([], []), 64)
        with self.assertRaises(ValueError):
            build_frames(Path([(0, 0, 0)], [(0.0, 0.0, 1.0)]), 64)

    def test_orientation_is_passed_through(self) -> None:
        skew = lambda path, index: Vec2(1.0, 1.0)
        [frame] = build_frames(straight(11), 64, orientation=skew)
        self.assertAlmostEqual(frame.start_boundary.normal.x, frame.start_boundary.normal.z)

    def test_crossing_check_is_passed_through(self) -> None:
        # Quarter circle of radius 10: orthogonal planes meet near its centre.
        angles = [radians(a) for a in (0.0, 45.0, 90.0)]
        path = Path(
            [(round(10 * cos(a)), 64, round(10 * sin(a))) for a in angles],
            [(-sin(a), 0.0, cos(a)) for a in angles],
        )
        breaks = [ElevationBreak(1, 65)]
        build_frames(path, 64, breaks, half_width=5.0)
        with self.assertRaises(ValueError):
            build_frames(path, 64, breaks, half_width=12.0)

    def test_lead_must_fit_in_the_padding(self) -> None:
        path = straight(25, lead_in=2, lead_out=3)
        build_frames(path, 64, lead_in=2, lead_out=3)
        with self.assertRaises(ValueError):
            build_frames(path, 64, lead_in=3)
        with self.assertRaises(ValueError):
            build_frames(path, 64, lead_out=4)

    def test_check_padding_rejects_negative_leads(self) -> None:
        with self.assertRaises(ValueError):
            check_padding(straight(5), -1, 0)


class FramesForIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        # path proper 2-21, frames 2-8, 8-15, 15-21
        self.path = straight(25, lead_in=2, lead_out=3)
        self.frames = build_frames(self.path, 64, [ElevationBreak(8, 65), ElevationBreak(15, 66)])

    def positions(self, index: int, lead_in: int = 0, lead_out: int = 0) -> list[int]:
        return [f.index for f in frames_for_index(self.frames, index, lead_in, lead_out)]

    def test_without_a_lead_a_point_on_a_break_goes_to_both_frames(self) -> None:
        self.assertEqual(self.positions(7), [0])
        self.assertEqual(self.positions(8), [0, 1])
        self.assertEqual(self.positions(9), [1])
        self.assertEqual(self.positions(15), [1, 2])

    def test_a_lead_reaches_into_the_neighbouring_frames(self) -> None:
        # frame 1 (8-15) with 2 in and 1 out takes points 6-16
        self.assertEqual(self.positions(6, lead_in=2, lead_out=1), [0, 1])
        self.assertEqual(self.positions(5, lead_in=2, lead_out=1), [0])
        self.assertEqual(self.positions(16, lead_in=2, lead_out=1), [1, 2])
        self.assertEqual(self.positions(17, lead_in=2, lead_out=1), [2])

    def test_padding_is_pasted_only_as_far_as_the_lead(self) -> None:
        self.assertEqual(self.positions(1), [])
        self.assertEqual(self.positions(1, lead_in=1), [0])
        self.assertEqual(self.positions(0, lead_in=1), [])
        self.assertEqual(self.positions(24, lead_out=3), [2])

    def test_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            frames_for_index([], 0)
        with self.assertRaises(ValueError):
            frames_for_index(self.frames, 0, lead_in=-1)


class ElevationFromFramesTests(unittest.TestCase):
    def test_a_single_frame_is_constant(self) -> None:
        [frame] = build_frames(straight(11), 70)
        elevation = elevation_from_frames([frame])
        self.assertEqual(elevation, ConstantElevation(70))
        self.assertIsInstance(elevation, YOffset)

    def test_columns_take_the_level_of_the_frame_they_are_in(self) -> None:
        path = arc(40, 90, 5, lead=2)
        breaks = [ElevationBreak(6, 66), ElevationBreak(12, 62)]
        frames = build_frames(path, 64, breaks)
        elevation = elevation_from_frames(frames)
        self.assertIsInstance(elevation, MultiElevationStep)
        self.assertEqual(elevation.span, (62, 66))
        for x in range(-5, 46):
            for z in range(-5, 46):
                owners = [f for f in frames if f.contains(x, z)]
                if len(owners) == 1:
                    self.assertEqual(elevation(x, z), owners[0].y_offset, (x, z))

    def test_uses_the_frames_boundaries_as_they_are(self) -> None:
        frames = build_frames(straight(21), 64, [ElevationBreak(10, 65)])
        [(boundary, level)] = elevation_from_frames(frames).steps
        self.assertIs(boundary, frames[1].start_boundary)
        self.assertEqual(level, 65)

    def test_empty(self) -> None:
        with self.assertRaises(ValueError):
            elevation_from_frames([])

    def test_shears_one_field_across_frames_when_clipped_to_the_path(self) -> None:
        registry = BlockRegistry()
        path = straight(25, lead_in=2, lead_out=3)
        frames = build_frames(path, 0, [ElevationBreak(12, 1)])
        world = BlockWorld(registry)
        strip = square(registry, 12)  # one field over z -12..12, placed at z=12
        world.add_field(
            strip,
            transform=BlockTransform((0, 64, 12), 0, elevation_from_frames(frames)),
            clip=path_bounds(path),
        )
        levels = {(p.x, p.z): p.y for p in world.iter_source_positions()}
        self.assertEqual({z for _, z in levels}, set(range(2, 22)))
        for (x, z), y in levels.items():
            self.assertEqual(y, 64 if z < 12 else 65, (x, z))


class BuildSectionFramesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = straight(25, lead_in=2, lead_out=3)  # path proper 2-21
        self.elevation = build_frames(self.path, 64, [ElevationBreak(12, 65)])

    def test_defaults_to_a_boundary_at_every_point(self) -> None:
        frames = build_section_frames(self.path, self.elevation)
        self.assertEqual([(f.path_start, f.path_stop) for f in frames], [(i, i + 1) for i in range(2, 21)])
        self.assertEqual([f.index for f in frames], list(range(19)))
        for before, after in zip(frames, frames[1:]):
            self.assertIs(before.end_boundary, after.start_boundary)

    def test_levels_follow_the_elevation_frames(self) -> None:
        frames = build_section_frames(self.path, self.elevation)
        for frame in frames:
            self.assertEqual(frame.y_offset, 64 if frame.path_start < 12 else 65)

    def test_elevation_boundaries_are_reused(self) -> None:
        frames = build_section_frames(self.path, self.elevation, [4, 8, 12, 16])
        self.assertEqual([f.path_start for f in frames], [2, 4, 8, 12, 16])
        self.assertIs(frames[0].start_boundary, self.elevation[0].start_boundary)
        self.assertIs(frames[3].start_boundary, self.elevation[1].start_boundary)
        self.assertIs(frames[-1].end_boundary, self.elevation[-1].end_boundary)

    def test_each_section_frame_lies_inside_one_elevation_frame(self) -> None:
        path = arc(40, 90, 3, lead=2)
        elevation = build_frames(path, 64, [ElevationBreak(12, 65), ElevationBreak(20, 63)])
        sections = build_section_frames(path, elevation, range(4, 30, 4), half_width=8.0)
        centre = [p.position for p in path.path_points]
        for x in range(-5, 46):
            for z in range(-5, 46):
                # Planes are unbounded; they only have to agree within the section.
                if min((x - c.x) ** 2 + (z - c.z) ** 2 for c in centre) > 64:
                    continue
                owner = [s for s in sections if s.contains(x, z)]
                levels = [e for e in elevation if e.contains(x, z)]
                self.assertEqual(len(owner), len(levels), (x, z))
                if owner:
                    self.assertEqual(owner[0].y_offset, levels[0].y_offset, (x, z))

    def test_every_elevation_break_must_be_a_station(self) -> None:
        with self.assertRaises(ValueError):
            build_section_frames(self.path, self.elevation, [5, 10, 15])

    def test_stations_must_increase_and_stay_on_the_path(self) -> None:
        with self.assertRaises(ValueError):
            build_section_frames(self.path, self.elevation, [12, 8])
        with self.assertRaises(ValueError):
            build_section_frames(self.path, self.elevation, [1, 12])
        with self.assertRaises(ValueError):
            build_section_frames(self.path, self.elevation, [12, 22])

    def test_elevation_frames_must_cover_the_path(self) -> None:
        with self.assertRaises(ValueError):
            build_section_frames(self.path, self.elevation[:1])

    def test_orientation_applies_to_new_stations_only(self) -> None:
        skew = lambda path, index: Vec2(1.0, 2.0)
        frames = build_section_frames(self.path, self.elevation, [6, 12], orientation=skew)
        self.assertIs(frames[1].end_boundary, self.elevation[1].start_boundary)
        self.assertAlmostEqual(frames[1].start_boundary.normal.z, 2 * frames[1].start_boundary.normal.x)


class PastedSectionsTests(unittest.TestCase):
    # Generation pastes a section at every point, once into each frame that
    # point belongs to, at that frame's level and clipped to it. What survives
    # is what the world flattens.

    def paste(self, path: Path, frames, half: int, lead: int) -> BlockWorld:
        registry = BlockRegistry()
        section = square(registry, half)
        world = BlockWorld(registry)
        for index, point in enumerate(path.path_points):
            for frame in frames_for_index(frames, index, lead, lead):
                world.add_field(
                    section,
                    (point.position.x, frame.y_offset, point.position.z),
                    clip=frame,
                )
        return world

    def test_frames_satisfy_the_mask_protocol(self) -> None:
        self.assertIsInstance(build_frames(straight(3), 64)[0], ColumnMask)

    def test_every_column_on_the_path_has_one_block_at_its_frame_level(self) -> None:
        # z 2-21 path proper, its last point included; steps up at z=12.
        path = straight(25, lead_in=2, lead_out=3)
        frames = build_frames(path, 64, [ElevationBreak(12, 65)], lead_in=2, lead_out=2)
        world = self.paste(path, frames, half=2, lead=2)
        flattened = {p for _, blocks in world.iter_resolved_chunks() for p in blocks}
        expected = {(x, 64 if z < 12 else 65, z) for x in range(-2, 3) for z in range(2, 22)}
        self.assertEqual(flattened, expected)

    def reaching_gaps(self, path: Path, frames, lead: int, half: int = 3, reach: int = 8):
        # Paste a section reaching `reach` blocks ahead of each point along its
        # heading, as a turned cross section does, and list the uncovered
        # columns within `half` of the path.
        registry = BlockRegistry()
        stone = registry.register(BlockState.create("minecraft:stone"))
        world = BlockWorld(registry)
        for index, point in enumerate(path.path_points):
            heading = point.heading.unit()
            field = BlockField(registry.palette)
            for ahead in range(reach + 1):
                for side in range(-half, half + 1):
                    field.set_block_id(
                        (
                            round(heading.x * ahead - heading.z * side),
                            0,
                            round(heading.z * ahead + heading.x * side),
                        ),
                        stone,
                    )
            for frame in frames_for_index(frames, index, lead, lead):
                world.add_field(
                    field, (point.position.x, frame.y_offset, point.position.z), clip=frame
                )
        bounds = path_bounds(path)
        centre = [p.position for p in path.bounded_points]
        gaps = set()
        for x in range(-15, 55):
            for z in range(-15, 55):
                if not bounds.contains(x, z):
                    continue
                if min((x - c.x) ** 2 + (z - c.z) ** 2 for c in centre) > half * half:
                    continue
                [frame] = [f for f in frames if f.contains(x, z)]
                if world.source_at((x, frame.y_offset, z)) is None:
                    gaps.add((x, z))
        return gaps

    def test_a_lead_fills_what_sections_reaching_across_a_break_leave(self) -> None:
        path = arc(40, 90, 4, lead=4)
        breaks = [
            ElevationBreak(path.first_index + k, 65 + k % 2)
            for k in range(2, len(path.bounded_points) - 1, 3)
        ]
        plain = build_frames(path, 64)
        stepped = build_frames(path, 64, breaks, lead_in=2, lead_out=2)
        # Measured against the same paste with no breaks, so only gaps the
        # breaks cause count.
        caused = lambda lead: self.reaching_gaps(path, stepped, lead) - self.reaching_gaps(path, plain, lead)
        self.assertTrue(caused(0))
        self.assertEqual(caused(2), set())

    def test_section_frames_clip_neighbouring_sections_to_orthogonal_seams(self) -> None:
        path = straight(25, lead_in=2, lead_out=3)
        elevation = build_frames(path, 64, [ElevationBreak(12, 65)])
        sections = build_section_frames(path, elevation, [6, 12, 16])
        world = self.paste(path, sections, half=2, lead=2)
        flattened = {p for _, blocks in world.iter_resolved_chunks() for p in blocks}
        expected = {(x, 64 if z < 12 else 65, z) for x in range(-2, 3) for z in range(2, 22)}
        self.assertEqual(flattened, expected)


if __name__ == "__main__":
    unittest.main()
