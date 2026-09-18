from __future__ import annotations

from collections.abc import Sequence

from ...Core.Path.path import Path
from ...Core.coordinates import BlockPos, Vec3
from .boundaries import Orientation, orthogonal


def confirm_orthogonal_ends(
    points: Sequence[BlockPos | Sequence[int]],
    gradients: Sequence[Vec3 | Sequence[float]],
    orientation: Orientation = orthogonal,
) -> None:

    #Determines if there exists an orthogonal end
    
    ends = Path(points, gradients)
    if not ends.path_points:
        raise ValueError("An empty path has no ends to pad")
    for index, end in ((ends.first_index, "first"), (ends.last_index, "last")):
        lead = "lead-in" if end == "first" else "lead-out"
        try:
            orientation(ends, index)
        except ValueError as error:
            raise ValueError(
                f"The path's {end} point has no horizontal heading, so its "
                f"{lead} cannot be extended orthogonally"
            ) from error


def extend_ends(
    points: Sequence[BlockPos | Sequence[int]],
    gradients: Sequence[Vec3 | Sequence[float]],
    lead: int,
) -> tuple[list[BlockPos], list[Vec3]]:
    #Extends ends in a line such that there can be lead-in/lead-out
    if lead < 0:
        raise ValueError(f"lead must be non-negative, got {lead}")

    positions = [BlockPos.coerce(point) for point in points]
    headings = [Vec3.coerce(gradient) for gradient in gradients]
    if not lead:
        return positions, headings
    if len(positions) < 2:
        raise ValueError(
            f"A path needs two points to be extended; {len(positions)} cannot "
            "say which way its ends run"
        )

    step_in = positions[0] - positions[1]
    step_out = positions[-1] - positions[-2]
    for step, end in ((step_in, "first"), (step_out, "last")):
        if step == BlockPos(0, 0, 0):
            raise ValueError(
                f"The path's {end} two points share a position, so there is no "
                "interval to extend its lead by"
            )

    before = _run(positions[0], step_in, lead)
    before.reverse()
    after = _run(positions[-1], step_out, lead)
    return (
        [*before, *positions, *after],
        [*[headings[0]] * lead, *headings, *[headings[-1]] * lead],
    )


def padded_path(
    points: Sequence[BlockPos | Sequence[int]],
    gradients: Sequence[Vec3 | Sequence[float]],
    lead: int,
    orientation: Orientation = orthogonal,
) -> Path:
    # A path extended by `lead` points at each end and marked as padded there.
    confirm_orthogonal_ends(points, gradients, orientation)
    positions, headings = extend_ends(points, gradients, lead)
    return Path(positions, headings, lead_in=lead, lead_out=lead)


def _run(start: BlockPos, step: BlockPos, count: int) -> list[BlockPos]:
    # `count` points on from `start`, a step apart, nearest first.
    run = []
    position = start
    for _ in range(count):
        position = position + step
        run.append(position)
    return run
