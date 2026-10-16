import unittest
from math import cos, pi, radians, sin

from src.Core.CrossSection.annotations import FeatureAnnotation
from src.Core.CrossSection.cross_section import CrossSection, PointField
from src.Core.CrossSection.schedule import SectionSchedule, SectionSpan
from src.Core.CrossSection.turns import JOIN_TOLERANCE
from src.Core.Domain import Domain
from src.Core.Feature.feature import PointFeature
from src.Core.Field.block_field import BlockField
from src.Core.Path.boundary import Boundary
from src.Core.Path.path_frame import PathFrame
from src.Core.coordinates import Vec3
from src.Core.modes import PlacementMode, StepMode
from src.Core.plan_index import NearestInPlan
from src.Generation.Geometry.orientation import SectionOrientation
from src.Generation.Geometry.padding import padded_path
from src.Generation.Geometry.silhouette import PathSilhouette, annotation_column_owners
from src.Generation.Sweep.corners import corners_of, settle_corners
from src.Generation.Sweep.paths import deliver_paths

LEAD = 12
STEPPING = (-2, 1, 3)
FOUR = ((1, 0), (-1, 0), (0, 1), (0, -1))
EIGHT = FOUR + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def road():
    # Nine columns; the stepping ones push those outboard of them along
    # travel, so each quarter's row is a staircase.
    return CrossSection(PointField(
        FeatureAnnotation(
            f"c{x}", (x, 0, 0),
            PointFeature(BlockField(), PlacementMode.OVERWRITE, Domain(1, "test")),
            StepMode.MANHATTAN if x in STEPPING else StepMode.CHEBYSHEV,
            section="road",
        )
        for x in range(-4, 5)
    ))


def arc(radius=40):
    # West, then a quarter turn clockwise to north, one block at a time and
    # diagonally where it moves along both axes, as a drawn curve does.
    points, gradients = [], []
    for sample in range(radius * 16 + 1):
        angle = sample / (radius * 16) * pi / 2
        cell = (round(-radius * sin(angle)), 64, round(radius * cos(angle) - radius))
        if points and cell == points[-1]:
            continue
        if len(points) > 1 and abs(cell[0] - points[-2][0]) == 1 and abs(cell[2] - points[-2][2]) == 1:
            points.pop()
            gradients.pop()
        points.append(cell)
        gradients.append(Vec3(-cos(angle), 0.0, -sin(angle)))
    return padded_path(points, gradients, LEAD)


def swept(path, section, solve_corners):
    schedule = SectionSchedule(
        (SectionSpan(path.first_index, path.last_index, section),), len(path.path_points)
    )
    frame = PathFrame(
        0, 0, len(path.path_points) - 1,
        Boundary.from_direction((100, 0), (-1, 0)),
        Boundary.from_direction((-200, 0), (-1, 0)), 64,
    )
    turns = SectionOrientation(tolerance=JOIN_TOLERANCE).turns(path)
    return deliver_paths(
        schedule, path, (frame,), turns, LEAD, LEAD,
        silhouette=PathSilhouette(path, 12), solve_corners=solve_corners,
    )


def pieces(cells, steps):
    left, count = set(cells), 0
    while left:
        count += 1
        stack = [left.pop()]
        while stack:
            x, z = stack.pop()
            for dx, dz in steps:
                if (x + dx, z + dz) in left:
                    left.remove((x + dx, z + dz))
                    stack.append((x + dx, z + dz))
    return count


class CornerTests(unittest.TestCase):
    def test_quarters_hand_over_between_their_own_steps(self) -> None:
        # The last step west and the first north bound the diagonal steps a
        # row of either quarter can be laid across.
        path = arc()
        [corner] = corners_of(
            SectionOrientation(tolerance=JOIN_TOLERANCE).turns(path), path, road(),
            path.first_index, path.last_index,
        )
        points = path.path_points

        def step(k):
            return (points[k].position.x - points[k - 1].position.x,
                    points[k].position.z - points[k - 1].position.z)

        west = max(k for k in range(1, corner.switch + 1) if step(k) == (-1, 0))
        north = min(k for k in range(corner.switch + 1, len(points)) if step(k) == (0, -1))
        self.assertTrue(all(step(k) == (-1, -1) for k in range(west + 1, north)))
        self.assertEqual(corner.switch, (west + north - 1) // 2)

    def test_every_column_holds_one_run_round_the_corner(self) -> None:
        # Where the world columns go to the section columns laid there, each
        # section column is one run through the turn, face-connected if it
        # steps: no lane line breaks or swaps with its neighbour.
        for radius in (25, 60, 150):
            path = arc(radius)
            runs = swept(path, road(), solve_corners=True)
            owners = annotation_column_owners(runs, path)
            settle_corners(owners, runs, path)
            held = {}
            for cell, name in owners["road", 0].items():
                held.setdefault(name, set()).add(cell)
            for run in runs:
                name = run.annotation.name
                steps = FOUR if run.annotation.step_mode is StepMode.MANHATTAN else EIGHT
                with self.subTest(radius=radius, column=name):
                    self.assertEqual(pieces(held[name], steps), 1)


if __name__ == "__main__":
    unittest.main()


def deep_road():
    # A staircase pushing its right edge four blocks along travel.
    return CrossSection(PointField(
        FeatureAnnotation(
            f"c{x}", (x, 0, 0),
            PointFeature(BlockField(), PlacementMode.OVERWRITE, Domain(1, "test")),
            StepMode.MANHATTAN if x in (1, 3, 5, 7) else StepMode.CHEBYSHEV,
            section="road",
        )
        for x in range(-1, 10)
    ))


def across_west(jog=25, count=60):
    # Heading west, bending from just south of west to just north of it, with
    # one step sideways a few stations before the heading crosses west.
    points = [(-k, 64, 1 if k >= jog else 0) for k in range(count)]
    gradients = [Vec3(-cos(radians(a)), 0.0, -sin(radians(a)))
                 for a in (-5 + 10 * k / (count - 1) for k in range(count))]
    return padded_path(points, gradients, LEAD)


class FlipTests(unittest.TestCase):
    def test_a_staircase_turning_over_keeps_every_column_one_run(self) -> None:
        # The flip is moved clear of the sideways step and every column
        # changes lean at one place, so none lands on its neighbour's line.
        path = across_west()
        runs = swept(path, deep_road(), solve_corners=True)
        owners = annotation_column_owners(runs, path)
        held = {}
        for cell, name in owners["road", 0].items():
            held.setdefault(name, set()).add(cell)
        for run in runs:
            name = run.annotation.name
            steps = FOUR if run.annotation.step_mode is StepMode.MANHATTAN else EIGHT
            with self.subTest(column=name):
                self.assertEqual(pieces(held[name], steps), 1)


def wobble(period=40, amplitude=3, length=60):
    # South-east by diagonal steps only, the heading swinging `amplitude`
    # degrees either side of the diagonal so the quarters change back and
    # forth.
    points, gradients = [], []
    x = z = travelled = 0.0
    while travelled < length:
        angle = radians(45 + amplitude * sin(2 * pi * travelled / period))
        x, z, travelled = x + cos(angle) * 0.02, z + sin(angle) * 0.02, travelled + 0.02
        cell = (round(x), 64, round(z))
        if points and cell == points[-1]:
            continue
        if len(points) > 1 and abs(cell[0] - points[-2][0]) == 1 and abs(cell[2] - points[-2][2]) == 1:
            points.pop()
            gradients.pop()
        points.append(cell)
        gradients.append(Vec3(cos(angle), 0.0, sin(angle)))
    return padded_path(points, gradients, LEAD)


def deck():
    # Fifteen columns, none stepping.
    return CrossSection(PointField(
        FeatureAnnotation(
            f"c{x}", (x, 0, 0),
            PointFeature(BlockField(), PlacementMode.OVERWRITE, Domain(1, "test")),
            StepMode.CHEBYSHEV, section="road",
        )
        for x in range(-7, 8)
    ))


def diagonals(path, turns):
    return sum(
        1 for k in range(path.first_index, path.last_index + 1)
        if turns[k].absolute % 90 and (k == path.first_index or turns[k - 1].absolute != turns[k].absolute)
    )


def held_columns(path, runs):
    # Each column's settled cells beside the path; the frame runs over the
    # padding, which a build's end planes cut off.
    owners = annotation_column_owners(runs, path)
    settle_corners(owners, runs, path)
    nearest = NearestInPlan(
        (point.position.x, point.position.z, index) for index, point in enumerate(path.path_points)
    )
    held = {}
    for cell, name in owners["road", 0].items():
        if path.first_index <= nearest.nearest(*cell) <= path.last_index:
            held.setdefault(name, set()).add(cell)
    return held


class DiagonalRunTests(unittest.TestCase):
    def test_a_heading_swinging_about_a_diagonal_keeps_every_column_one_run(self) -> None:
        path = wobble()
        turns = SectionOrientation(tolerance=JOIN_TOLERANCE).turns(path)
        corners = corners_of(turns, path, road(), path.first_index, path.last_index)
        self.assertEqual(len(corners), diagonals(path, turns))
        runs = swept(path, road(), solve_corners=True)
        held = held_columns(path, runs)
        for run in runs:
            name = run.annotation.name
            steps = FOUR if run.annotation.step_mode is StepMode.MANHATTAN else EIGHT
            with self.subTest(column=name):
                self.assertEqual(pieces(held[name], steps), 1)

    def test_neighbouring_corners_hand_over_in_turn(self) -> None:
        # Diagonals a few stations apart reach into one quarter stretch: each
        # hands over after the one before it, leaving the stretch long enough
        # for the next one's wedge, or the two keep the quarter they turn
        # back to. Every diagonal station is laid in a quarter.
        for period, amplitude, length in ((20, 10, 120), (40, 6, 124)):
            path = wobble(period, amplitude, length)
            turns = SectionOrientation(tolerance=JOIN_TOLERANCE).turns(path)
            corners = corners_of(turns, path, deck(), path.first_index, path.last_index)
            switches = [corner.switch for corner in corners]
            relaid = {k for corner in corners for k in corner.turns}
            with self.subTest(wobble=(period, amplitude)):
                self.assertEqual(switches, sorted(set(switches)))
                self.assertFalse([
                    k for k in range(path.first_index, path.last_index + 1)
                    if turns[k].absolute % 90 and k not in relaid
                ])
            runs = swept(path, deck(), solve_corners=True)
            held = held_columns(path, runs)
            for run in runs:
                with self.subTest(wobble=(period, amplitude), column=run.annotation.name):
                    self.assertEqual(pieces(held[run.annotation.name], EIGHT), 1)
