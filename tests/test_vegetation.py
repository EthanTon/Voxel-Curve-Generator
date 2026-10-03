from src.Core.coordinates import BlockPos
from src.Terraform.vegetation import clearing


def forest():
    first = {BlockPos(0, y, 0) for y in range(1, 5)}
    second = {BlockPos(6, y, 0) for y in range(1, 5)}
    leaves = {BlockPos(x, 5, z) for x in range(-2, 9) for z in range(-1, 2)}
    return first, second, leaves


def test_touching_crowns_do_not_clear_the_neighbouring_tree():
    first, second, leaves = forest()
    growth = first | second | leaves
    removed = clearing(growth, lambda p: p == BlockPos(0, 1, 0), growth, first | second)
    assert first <= removed
    assert not second & removed
    assert BlockPos(0, 5, 0) in removed
    assert BlockPos(6, 5, 0) not in removed
    assert BlockPos(3, 5, 0) not in removed  # shared crown stays with live tree


def test_overhanging_leaves_are_trimmed_without_removing_any_trunk():
    first, second, leaves = forest()
    growth = first | second | leaves
    obstructing = BlockPos(2, 5, 0)
    assert clearing(growth, lambda p: p == obstructing, growth, first | second) == {obstructing}


def test_removed_trunk_clears_diagonal_branches_and_corner_connected_leaves():
    base = BlockPos(0, 1, 0)
    trunk = {base, BlockPos(0, 2, 0), BlockPos(1, 3, 1), BlockPos(2, 4, 2)}
    leaves = {BlockPos(3, 5, 3), BlockPos(4, 5, 4)}
    growth = trunk | leaves
    assert clearing(growth, lambda p: p == base, growth, trunk) == growth


def test_diagonal_canopy_contact_preserves_a_live_neighbour():
    first = {BlockPos(0, y, 0) for y in range(1, 4)}
    second = {BlockPos(6, y, 6) for y in range(1, 4)}
    leaves = {BlockPos(i, 4, i) for i in range(7)}
    growth = first | second | leaves
    removed = clearing(growth, lambda p: p == BlockPos(0, 1, 0), growth, first | second)
    assert first <= removed
    assert not second & removed
    assert BlockPos(0, 4, 0) in removed
    assert BlockPos(3, 4, 3) not in removed
    assert BlockPos(6, 4, 6) not in removed
