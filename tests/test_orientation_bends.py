"""A join's direction comes from the bend, independently of world heading."""

from math import cos, radians, sin

import pytest

from src.Core.Path.path import Path
from src.Generation.Geometry.orientation import SectionOrientation


def orientations(angles):
    gradients = [(-cos(radians(a)), 0, -sin(radians(a))) for a in angles]
    return SectionOrientation().turns(Path([(i, 64, 0) for i in range(len(angles))], gradients))


@pytest.mark.parametrize("start", (0, 90, 180, 270))
@pytest.mark.parametrize("direction", (-1, 1))
def test_join_arms_follow_the_local_bend_from_every_quadrant(start, direction):
    turns = orientations([start + direction*i for i in range(91)])
    joins = [turn for turn in turns if turn.absolute % 90]
    assert len(joins) == 1
    assert joins[0].direction == direction
    assert turns[0].absolute == start
    assert turns[-1].absolute == (start+90*direction) % 360


@pytest.mark.parametrize("quarter", (0, 90, 180, 270))
@pytest.mark.parametrize("side", (-1, 1))
def test_touching_a_diagonal_and_returning_does_not_invent_a_handover(quarter, side):
    turns = orientations([quarter+side*i for i in (0, 15, 30, 43, 45, 43, 30, 15, 0)])
    assert {turn.absolute for turn in turns} == {quarter}


@pytest.mark.parametrize("diagonal", (45, 135, 225, 315))
@pytest.mark.parametrize("off", (-1.5, -0.5, 0.5, 1.5))
def test_a_path_along_a_diagonal_keeps_the_quarter_beside_it(diagonal, off):
    turns = orientations([diagonal + off] * 20)
    assert {turn.absolute for turn in turns} == {(diagonal + (45 if off > 0 else -45)) % 360}


@pytest.mark.parametrize("off", (-1.5, 1.5))
def test_a_path_ending_beside_a_diagonal_ends_in_the_quarter_beside_it(off):
    turns = orientations([90 + i for i in range(45)] + [135 + off] * 30)
    assert turns[-1].absolute == (180 if off > 0 else 90)
    assert len([turn for turn in turns if turn.absolute % 90]) == (1 if off > 0 else 0)


def test_a_path_setting_off_beside_a_diagonal_joins_where_it_crosses():
    turns = orientations([133.5 + 0.2 * i for i in range(16)] + [137 + i for i in range(44)])
    joins = [index for index, turn in enumerate(turns) if turn.absolute % 90]
    assert {turn.absolute for turn in turns[:8]} == {90}
    assert joins == [8] and turns[8].direction == 1
    assert turns[-1].absolute == 180


def test_ending_exactly_on_a_diagonal_turns_with_the_path():
    laid = [[(turn.absolute - start) % 360 for turn in orientations([start + i for i in range(45)] + [start + 45] * 20)]
            for start in (0, 90, 180, 270)]
    assert all(each == laid[0] for each in laid)
