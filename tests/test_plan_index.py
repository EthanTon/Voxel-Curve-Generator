from itertools import product

from src.Core.plan_index import NearestInPlan


def test_nearest_in_plan_matches_ordered_scan_including_ties_and_duplicates():
    points = [
        (-2, 0, "first"),
        (2, 0, "second"),
        (0, 2, "third"),
        (0, 2, "duplicate"),
        (4, -3, "far"),
    ]
    index = NearestInPlan(points)

    for x, z in product(range(-5, 6), repeat=2):
        expected = min(
            points, key=lambda each: (each[0] - x) ** 2 + (each[1] - z) ** 2
        )[2]
        assert index.nearest(x, z) == expected
