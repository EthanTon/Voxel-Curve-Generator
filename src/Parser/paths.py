from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path

from ..Core.coordinates import BlockPos, ColumnPos, Vec3
from ..Core.CrossSection.turns import JOIN_DEGREES
from .spec import OutputSpec, PathSpec
from .values import read_column, require


def read_path(data: Mapping, base: Path = Path(".")) -> PathSpec:
    source = {**read_path_file(base / data["path"]), **data} if "path" in data else data
    points = tuple(BlockPos.coerce(each) for each in require(source, "points"))
    gradients = tuple(Vec3.coerce(each) for each in require(source, "gradients"))
    if len(points) != len(gradients):
        raise ValueError(
            f"path has {len(points)} points and {len(gradients)} gradients"
        )
    lead = source.get("lead", 0)
    if lead < 0:
        raise ValueError(f"lead must be non-negative, got {lead}")
    if lead and lead * 2 >= len(points):
        raise ValueError(
            f"a lead of {lead} at each end needs more than {lead * 2} points, "
            f"and the path has {len(points)}"
        )
    return PathSpec(points, gradients, lead, read_diagonal(source))


def read_diagonal(source: Mapping) -> float:
    degrees = source.get("diagonal_degrees", JOIN_DEGREES)
    if not 0 <= degrees <= 45:
        raise ValueError(
            f"diagonal_degrees is how near a diagonal a heading has to be "
            f"for the join to be laid there, between 0 and 45 degrees, and "
            f"the path gives {degrees}"
        )
    return degrees


def read_path_file(path: Path) -> Mapping:
    if not path.exists():
        raise ValueError(f"no path file at {path}")
    data = tomllib.loads(path.read_text())
    return data.get("path", data)


def read_markers(data: Mapping) -> dict[str, ColumnPos]:
    return {name: read_column(value, name) for name, value in data.items()}


def read_output(data: Mapping, base: Path = Path(".")) -> OutputSpec:
    # Written beside the build that asked for it, like every other path in
    # the file, so where the build is run from does not move its output. An
    # absolute path is left where it was written.
    return OutputSpec(
        path=base / require(data, "path"),
        name=data.get("name", "build"),
        data_version=data.get("data_version", 3700),
    )
