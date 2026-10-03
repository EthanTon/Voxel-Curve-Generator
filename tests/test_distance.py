import unittest

import numpy as np

from src.Terraform.distance import nearest


def exact(seeds, reach):
    rows, cells = np.indices(seeds.shape)
    found = np.full(seeds.shape, np.inf)
    for row, cell in zip(*np.nonzero(seeds)):
        found = np.minimum(found, np.hypot(rows - row, cells - cell))
    return np.minimum(found, reach)


class NearestTests(unittest.TestCase):
    def test_distances_are_euclidean_so_contours_are_round(self) -> None:
        seeds = np.zeros((41, 41), dtype=bool)
        seeds[20, 20] = True
        found = nearest(seeds, (), 30.0)
        self.assertTrue(np.allclose(found.distance, exact(seeds, 30.0)))

    def test_a_bent_track_and_what_it_carries(self) -> None:
        seeds = np.zeros((60, 60), dtype=bool)
        seeds[10:50, 30] = True
        seeds[30, 5:55] = True
        labels = np.where(seeds, np.arange(60)[None, :].repeat(60, 0), 0).astype(float)
        found = nearest(seeds, (labels,), 40.0)
        self.assertTrue(np.allclose(found.distance, exact(seeds, 40.0)))
        # Beside the vertical arm, the label is the arm's own column.
        self.assertEqual(found[0][20, 36], 30)


if __name__ == "__main__":
    unittest.main()


def test_world_sized_distance_field_carries_the_exact_nearest_source():
    seeds = np.zeros((512, 512), dtype=bool)
    seeds[40, 20] = seeds[470, 480] = True
    heights = np.zeros(seeds.shape)
    heights[40, 20], heights[470, 480] = 71, 109
    found = nearest(seeds, (heights,), 700)
    assert np.allclose(found.distance, exact(seeds, 700))
    assert found[0][100, 100] == 71
    assert found[0][400, 400] == 109
    empty = nearest(np.zeros_like(seeds), (heights,), 32)
    assert np.all(empty.distance == 32)
    assert np.all(empty[0] == 0)
