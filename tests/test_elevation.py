import unittest

from src.Core.Path.elevation import ConstantElevation, ElevationStep, MultiElevationStep
from src.Core.Path.boundary import Boundary


def boundary(index=0, origin=(0.0, 0.0), tangent=(0.0, 1.0)):
    return Boundary.from_direction(origin, tangent, path_index=index)


class ConstantElevationTests(unittest.TestCase):
    def test_one_level_everywhere(self) -> None:
        flat = ConstantElevation(70)
        self.assertEqual({flat(x, z) for x in range(-3, 4) for z in range(-3, 4)}, {70})
        self.assertEqual(flat.span, (70, 70))
        self.assertTrue(flat.is_constant)


class ElevationStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.step = ElevationStep(boundary(origin=(0.0, 10.0)), y_before=64, y_after=65)

    def test_selects_by_side_of_the_plane(self) -> None:
        self.assertEqual(self.step(0, 9), 64)
        self.assertEqual(self.step(0, 10), 65)
        self.assertEqual(self.step(0, 11), 65)

    def test_span_is_ordered_regardless_of_direction(self) -> None:
        self.assertEqual(self.step.span, (64, 65))
        descending = ElevationStep(boundary(), y_before=70, y_after=61)
        self.assertEqual(descending.span, (61, 70))

    def test_is_constant(self) -> None:
        self.assertFalse(self.step.is_constant)
        self.assertTrue(ElevationStep(boundary(), 64, 64).is_constant)

    def test_frozen_and_hashable(self) -> None:
        # placements share these, and source_token needs them stable
        self.assertEqual(hash(self.step), hash(self.step))
        with self.assertRaises(Exception):
            self.step.y_after = 99


class MultiElevationStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.step = MultiElevationStep(
            y_first=64,
            steps=(
                (boundary(1, origin=(0.0, 10.0)), 65),
                (boundary(2, origin=(0.0, 20.0)), 66),
            ),
        )

    def test_takes_the_level_of_the_last_boundary_crossed(self) -> None:
        self.assertEqual(self.step(0, 5), 64)
        self.assertEqual(self.step(0, 15), 65)
        self.assertEqual(self.step(0, 25), 66)

    def test_every_boundary_is_tested_so_non_monotone_still_resolves(self) -> None:
        # second plane faces the other way: a column can be after it but before
        # the first. Scanning without short-circuit gives a defined answer.
        odd = MultiElevationStep(
            y_first=64,
            steps=(
                (boundary(1, origin=(0.0, 10.0), tangent=(0.0, 1.0)), 65),
                (boundary(2, origin=(0.0, 0.0), tangent=(0.0, -1.0)), 66),
            ),
        )
        self.assertEqual(odd(0, -5), 66)

    def test_span_covers_every_level(self) -> None:
        self.assertEqual(self.step.span, (64, 66))

    def test_empty_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MultiElevationStep(y_first=64, steps=())

    def test_is_constant(self) -> None:
        self.assertFalse(self.step.is_constant)
        flat = MultiElevationStep(64, ((boundary(), 64), (boundary(1, (0.0, 5.0)), 64)))
        self.assertTrue(flat.is_constant)


if __name__ == "__main__":
    unittest.main()
