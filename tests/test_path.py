import unittest

from src.Core.Path.path import Path


def straight(count: int, **padding) -> Path:
    # Points along +z, one block apart, all heading +z.
    return Path(
        [(0, 64, z) for z in range(count)],
        [(0.0, 0.0, 1.0)] * count,
        **padding,
    )


class PathPaddingTests(unittest.TestCase):
    def test_unpadded_path_is_all_path(self) -> None:
        path = straight(5)
        self.assertEqual((path.first_index, path.last_index), (0, 4))
        self.assertEqual(path.bounded_points, path.path_points)
        self.assertFalse(any(path.is_padding(i) for i in range(5)))

    def test_padding_keeps_every_point_but_narrows_the_path(self) -> None:
        path = straight(10, lead_in=2, lead_out=3)
        self.assertEqual(len(path.path_points), 10)
        self.assertEqual((path.first_index, path.last_index), (2, 6))
        self.assertEqual(
            [p.position.z for p in path.bounded_points], [2, 3, 4, 5, 6]
        )
        self.assertEqual(
            [i for i in range(10) if path.is_padding(i)], [0, 1, 7, 8, 9]
        )

    def test_length_excludes_padding(self) -> None:
        self.assertAlmostEqual(straight(10, lead_in=2, lead_out=3).length(), 4.0)
        self.assertAlmostEqual(straight(5).length(), 4.0)

    def test_negative_padding_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            straight(5, lead_in=-1)
        with self.assertRaises(ValueError):
            straight(5, lead_out=-1)

    def test_padding_must_leave_a_path_point(self) -> None:
        with self.assertRaises(ValueError):
            straight(4, lead_in=2, lead_out=2)
        straight(5, lead_in=2, lead_out=2)

    def test_mismatched_gradients_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Path([(0, 0, 0), (0, 0, 1)], [(0.0, 0.0, 1.0)])


if __name__ == "__main__":
    unittest.main()
