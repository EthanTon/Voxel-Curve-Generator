import unittest
from types import SimpleNamespace

import numpy as np

from src.Core.CrossSection.annotations import FeatureAnnotation
from src.Core.Domain import Domain
from src.Core.Feature.feature import PointFeature
from src.Core.Feature.feature_path import FeaturePath
from src.Core.Field.block_field import BlockField
from src.Core.coordinates import BlockPos, Vec3
from src.Core.modes import PlacementMode, StepMode
from src.Generation.Geometry.silhouette import annotation_column_owners
from src.Generation.Sweep.paths import DeliveredPath
from src.Terraform.terraform import _tidy_fringe

EAST = Vec3(1, 0, 0)
# Turn 180 travels east, turn 90 north; local x is right of travel.
ALONG, ACROSS = 180, 90


def run(column, cells, layout, start=0, step=StepMode.CHEBYSHEV):
    annotation = FeatureAnnotation(
        f"bank[{column},0]",
        BlockPos(column, 0, 0),
        PointFeature(BlockField(), PlacementMode.OVERWRITE, Domain(1, "bed")),
        step,
        section="bank",
    )
    path = FeaturePath()
    for index, cell in cells:
        path.add(BlockPos(cell[0], 0, cell[1]), EAST, layout, index, 64, layout=layout)
    return DeliveredPath(annotation, 0, path, start)


# Three points heading east along z = 0.
PATH = SimpleNamespace(path_points=[
    SimpleNamespace(position=BlockPos(x, 64, 0), gradient=EAST) for x in range(3)
])


class ColumnOwnershipTests(unittest.TestCase):
    def owner(self, *runs):
        return annotation_column_owners(runs, PATH)[("bank", 0)][(1, 2)]

    def test_a_contested_column_goes_to_the_column_that_belongs_there(self) -> None:
        # Two blocks south of the path is column 2, whichever was laid last.
        anvil = run(2, [(1, (1, 2))], ALONG)
        wool = run(3, [(1, (1, 2))], ALONG, start=1)
        self.assertEqual(self.owner(anvil, wool), "bank[2,0]")
        self.assertEqual(self.owner(wool, anvil), "bank[2,0]")

    def test_the_nearest_points_layout_wins_over_a_joins_other_arm(self) -> None:
        # The point beside it lays along the path; an arm of the other
        # quarter reaching over it does not take its column, stepping or not.
        own = run(2, [(1, (1, 2))], ALONG)
        arm = run(3, [(2, (1, 2))], ACROSS, start=1, step=StepMode.MANHATTAN)
        self.assertEqual(self.owner(own, arm), "bank[2,0]")

    def test_a_stepping_column_keeps_its_step_in_its_own_layout(self) -> None:
        plain = run(2, [(1, (1, 2))], ALONG, start=1)
        stepping = run(3, [(1, (1, 2))], ALONG, step=StepMode.MANHATTAN)
        self.assertEqual(self.owner(plain, stepping), "bank[3,0]")


class FringeTests(unittest.TestCase):
    def test_specks_go_and_notches_fill_but_edges_stay(self) -> None:
        natural = np.zeros((7, 7))
        designed = np.zeros((7, 7))
        designed[3, 1] = 1.0          # a lone speck
        designed[:, 4:] = 1.0         # a fill with a straight edge...
        designed[3, 5] = 0.0          # ...and a notch in it
        tidied = np.rint(_tidy_fringe(designed, natural, np.ones((7, 7), bool)))
        self.assertEqual(tidied[3, 1], 0)
        self.assertEqual(tidied[3, 5], 1)
        self.assertTrue((tidied[:, 4:] == 1).all())
        self.assertTrue((tidied[:, :4] == 0).all())


if __name__ == "__main__":
    unittest.main()


class BrushTests(unittest.TestCase):
    def test_the_brush_smooths_a_step_and_holds_what_it_does_not_cover(self) -> None:
        from src.Terraform.plan import Brush, Plan
        from src.Terraform.terraform import _brushed

        plan = Plan(brush=Brush(radius=1.5, passes=4))
        # A fill ending in a four-block step, the brush over its middle only.
        designed = np.zeros((5, 20))
        designed[:, :10] = 4.0
        area = np.zeros((5, 20))
        area[:, 4:16] = 1.0
        brushed = _brushed(plan, designed, area)
        middle = brushed[2]
        self.assertTrue((np.diff(middle) <= 1e-9).all())
        # Stable's default one-block budget reduces this four-block step
        # to two without eroding or growing either side further.
        self.assertLessEqual(-np.diff(middle).min(), 2.0)
        self.assertLessEqual(np.max(np.abs(brushed - designed)), 1.0)
        self.assertTrue((brushed[:, :4] == 4.0).all())
        self.assertTrue((brushed[:, 16:] == 0.0).all())

    def test_no_passes_leaves_the_ground_as_designed(self) -> None:
        from src.Terraform.plan import Brush, Plan
        from src.Terraform.terraform import _brushed

        designed = np.arange(20.0)[None, :].repeat(3, 0)
        area = np.ones((3, 20))
        self.assertTrue((_brushed(Plan(brush=Brush(passes=0)), designed, area) == designed).all())


class ShoulderBrushTests(unittest.TestCase):
    def test_the_shoulder_is_brushed_but_the_footprint_holds(self) -> None:
        from src.Core.coordinates import ColumnPos
        from src.Terraform.grid import Grid
        from src.Terraform.layers import Ground
        from src.Terraform.plan import Plan
        from src.Terraform.terraform import _brush_area

        grid = Grid(ColumnPos(0, 0), 20, 1)
        distance = np.abs(np.arange(20.0) - 3.0)[None, :]
        distance[0, :4] = 0.0
        natural = grid.filled(60.0)
        # A fill standing level to the end of a four-wide shoulder.
        shaped = np.where(distance <= 4.0, 66.0, 60.0)
        ground = Ground(grid, natural, distance, shaped, shaped, distance)
        area = _brush_area(Plan(shoulder=4.0), ground, (), (), shaped, np.ones((1, 20), bool))
        self.assertTrue((area[0, :4] == 0.0).all())
        self.assertTrue((area[0, 5:8] > 0.0).all())
