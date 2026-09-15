import unittest

from src.Core.Field.block_field import BlockBounds
from src.Core.Path.elevation import ElevationStep, MultiElevationStep
from src.Core.Path.boundary import Boundary
from src.Core.World.block_world import BlockTransform, YOffset
from src.Core.coordinates import BlockPos


def boundary(index=0, origin=(0.0, 0.0), tangent=(0.0, 1.0)):
    return Boundary.from_direction(origin, tangent, path_index=index)


GRID = [
    BlockPos(x, y, z)
    for x in range(-6, 7)
    for y in range(-3, 4)
    for z in range(-6, 7)
]


class YOffsetProtocolTests(unittest.TestCase):
    def test_elevation_steps_satisfy_the_protocol(self) -> None:
        self.assertIsInstance(ElevationStep(boundary(), 64, 65), YOffset)
        self.assertIsInstance(MultiElevationStep(64, ((boundary(), 65),)), YOffset)


class RoundTripTests(unittest.TestCase):
    def assert_round_trips(self, transform: BlockTransform) -> None:
        for local in GRID:
            self.assertEqual(transform.inverse(transform.apply(local)), local)

    def test_without_offset_matches_previous_behaviour(self) -> None:
        for turns in range(4):
            self.assert_round_trips(BlockTransform(BlockPos(3, 64, -7), turns))

    def test_with_an_elevation_step(self) -> None:
        step = ElevationStep(boundary(origin=(2.0, 2.0), tangent=(1.0, 1.0)), 64, 65)
        for turns in range(4):
            self.assert_round_trips(BlockTransform(BlockPos(3, 0, -7), turns, step))

    def test_with_a_multi_step(self) -> None:
        step = MultiElevationStep(
            60,
            (
                (boundary(1, (0.0, 0.0), (1.0, 2.0)), 63),
                (boundary(2, (4.0, 4.0), (1.0, 2.0)), 67),
            ),
        )
        for turns in range(4):
            self.assert_round_trips(BlockTransform(BlockPos(-2, 0, 5), turns, step))

    def test_offset_is_applied_in_world_coordinates(self) -> None:
        # plane at world z = 10; the origin translate must happen first
        step = ElevationStep(boundary(origin=(0.0, 10.0)), 0, 5)
        transform = BlockTransform(BlockPos(0, 64, 10), 0, step)
        self.assertEqual(transform.apply(BlockPos(0, 0, 0)).y, 69)   # world z=10 -> after
        self.assertEqual(transform.apply(BlockPos(0, 0, -1)).y, 64)  # world z=9  -> before

    def test_injective_over_a_footprint(self) -> None:
        step = ElevationStep(boundary(origin=(0.0, 0.0), tangent=(1.0, 1.0)), 64, 65)
        transform = BlockTransform(BlockPos(0, 0, 0), 0, step)
        seen = {transform.apply(p) for p in GRID}
        self.assertEqual(len(seen), len(GRID))


class BoundsTests(unittest.TestCase):
    def test_bounds_contain_every_transformed_cell(self) -> None:
        step = ElevationStep(boundary(origin=(1.0, 1.0), tangent=(2.0, -1.0)), 64, 70)
        for turns in range(4):
            transform = BlockTransform(BlockPos(5, 0, -3), turns, step)
            local = BlockBounds(BlockPos(-6, -3, -6), BlockPos(6, 3, 6))
            world = transform.apply_bounds(local)
            for p in GRID:
                self.assertTrue(world.contains(transform.apply(p)), f"{p} escaped {world}")

    def test_bounds_widen_by_the_offset_span(self) -> None:
        local = BlockBounds(BlockPos(0, 0, 0), BlockPos(1, 1, 1))
        plain = BlockTransform(BlockPos(0, 0, 0)).apply_bounds(local)
        stepped = BlockTransform(
            BlockPos(0, 0, 0), 0, ElevationStep(boundary(), 64, 70)
        ).apply_bounds(local)
        self.assertEqual(stepped.minimum.y, plain.minimum.y + 64)
        self.assertEqual(stepped.maximum.y, plain.maximum.y + 70)


if __name__ == "__main__":
    unittest.main()
