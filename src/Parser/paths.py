from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..Core.coordinates import BlockPos, ColumnPos, Vec3
from ..Core.CrossSection.turns import JOIN_DEGREES
from .spec import OutputSpec, PathSpec, SplitSpec
from .values import read_column, read_toml, require


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
    if lead and len(points) < 2:
        raise ValueError(
            f"a lead of {lead} needs at least two authored points to extend, "
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
    data = read_toml(path, "path")
    return data.get("path", data)


def read_markers(data: Mapping) -> dict[str, ColumnPos]:
    return {name: read_column(value, name) for name, value in data.items()}


def read_output(data: Mapping, base: Path = Path(".")) -> OutputSpec:
    # Relative to the build file, like every other path in it.
    mode = data.get("mode")
    if mode is not None and mode not in ("absolute", "relative"):
        raise ValueError("output mode must be 'absolute' or 'relative'")
    path = base / require(data, "path")
    return OutputSpec(
        path=path,
        name=data.get("name", "build"),
        data_version=data.get("data_version", 3700),
        mode=mode,
        split=read_split(data.get("split"), path),
    )


def read_split(data: Mapping | None, path: Path) -> SplitSpec | None:
    # Pieces default to a folder beside the output, named after it.
    if data is None:
        return None
    directory = data.get("directory")
    split = SplitSpec(
        directory=path.with_suffix("") if directory is None else path.parent / directory,
        max_blocks=data.get("max_blocks", 4_000_000),
        max_span=data.get("max_span", 128),
        terrain=data.get("terrain", False),
        terrain_depth=data.get("terrain_depth", 16),
        combined=data.get("combined", True),
        count_air=data.get("count_air", True),
    )
    if split.max_blocks <= 0 or split.max_span <= 0:
        raise ValueError("a split needs a positive max_blocks and max_span")
    if type(split.terrain_depth) is not int or split.terrain_depth < 0:
        raise ValueError("split.terrain_depth must be a nonnegative integer")
    if type(split.combined) is not bool:
        raise ValueError("split.combined must be true or false")
    if type(split.count_air) is not bool:
        raise ValueError("split.count_air must be true or false")
    if split.terrain and not split.count_air:
        raise ValueError("split.count_air = false requires terrain = false")
    return split
