import unittest

from src.Core.Path.boundary import Boundary, PathBounds, spans_nothing
from src.Core.coordinates import Vec2


def boundary(index=None, origin=(0.0, 0.0), direction=(0.0, 1.0)):
    return Boundary.from_direction(origin, direction, path_index=index)


class BoundaryTests(unittest.TestCase):
    def test_side_is_signed_along_the_normal(self) -> None:
        b = boundary(origin=(0.0, 10.0))
        self.assertGreater(b.side(0, 11), 0.0)
        self.assertLess(b.side(0, 9), 0.0)
        self.assertEqual(b.side(0, 10), 0.0)

    def test_the_plane_itself_belongs_to_the_side_after_it(self) -> None:
        self.assertTrue(boundary(origin=(0.0, 10.0)).after(0, 10))

    def test_from_direction_normalises(self) -> None:
        self.assertAlmostEqual(boundary(direction=(0.0, 7.0)).normal.z, 1.0)

    def test_any_direction_is_accepted(self) -> None:
        # Core no longer assumes orthogonality to anything.
        b = Boundary(Vec2(1.0, 2.0), Vec2(3.0, -4.0))
        self.assertEqual(b.normal, Vec2(3.0, -4.0))
        self.assertIsNone(b.path_index)

    def test_zero_normal_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Boundary(Vec2(0.0, 0.0), Vec2(0.0, 0.0))
        with self.assertRaises(ValueError):
            Boundary.from_direction((0.0, 0.0), (0.0, 0.0))

    def test_parallel_detection(self) -> None:
        a = boundary(direction=(1.0, 0.0))
        b = boundary(origin=(3.0, 0.0), direction=(-1.0, 0.0))
        c = boundary(direction=(0.0, 1.0))
        self.assertTrue(a.is_parallel_to(b))
        self.assertFalse(a.is_parallel_to(c))

    def test_spans_nothing(self) -> None:
        start = boundary(origin=(0.0, 10.0))
        self.assertTrue(spans_nothing(start, boundary(origin=(0.0, 4.0))))
        self.assertTrue(spans_nothing(start, start))
        self.assertFalse(spans_nothing(start, boundary(origin=(0.0, 11.0))))
        self.assertFalse(spans_nothing(start, boundary(direction=(1.0, 1.0))))


class PathBoundsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = PathBounds(boundary(origin=(0.0, 2.0)), boundary(origin=(0.0, 8.0)))

    def test_ends_are_half_open_like_any_boundary(self) -> None:
        self.assertFalse(self.bounds.contains(0, 1))
        self.assertTrue(self.bounds.contains(0, 2))
        self.assertTrue(self.bounds.contains(0, 7))
        self.assertFalse(self.bounds.contains(0, 8))

    def test_degenerate(self) -> None:
        self.assertFalse(self.bounds.is_degenerate)
        flat = boundary(origin=(0.0, 2.0))
        self.assertTrue(PathBounds(flat, flat).is_degenerate)


if __name__ == "__main__":
    unittest.main()
