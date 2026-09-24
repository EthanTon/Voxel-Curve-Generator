from __future__ import annotations

# A build written as non-overlapping schematic pieces tiling it in plan. With
# terrain, each piece is the world as the build leaves it -- the build's
# blocks, air where it clears ground, and the save's own blocks elsewhere --
# so pasting it with air changes exactly what the build changes.

from dataclasses import dataclass
from pathlib import Path as FilePath

import numpy as np

from ..Core.World.block_world import BlockWorld
from ..Core.block.palette import BlockPalette
from ..Core.block.states import BlockState
from ..Core.coordinates import BlockPos
from ..Format.litematic import write_litematic
from ..Format.region import AIR, AIR_NAMES, BlockRegion
from ..Format.schematics import SCHEM_SUFFIX, write_schematic
from ..Parser.spec import BuildSpec
from ..Terraform import Terraforming
from ..Terraform.clearance import OPEN
from ..Terraform.region import WorldBlocks
from ..Terraform.terrain import region_directory


@dataclass(frozen=True, slots=True)
class Piece:
    name: str
    path: FilePath
    origin: BlockPos
    size: BlockPos
    placed: int
    cleared: int
    copied: int


def write_pieces(
    world: BlockWorld,
    spec: BuildSpec,
    terraforming: Terraforming | None,
) -> list[Piece]:
    split = spec.output.split
    if split is None:
        return []
    ground = split.terrain and terraforming is not None and spec.terraform is not None
    if split.terrain and not ground:
        raise ValueError(
            "the split asks for terrain, but the build has no terraform world to copy it from"
        )
    states, positions, codes = _changed_cells(world, terraforming)
    source = (
        WorldBlocks(region_directory(FilePath(spec.terraform.world).resolve()))
        if ground and spec.terraform.world is not None else None
    )
    directory = FilePath(split.directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Remove the previous run's pieces so none are left over.
    for stale in directory.glob(f"{spec.output.name}_[0-9][0-9][0-9]{SCHEM_SUFFIX}"):
        stale.unlink()
    first = BlockPos.coerce(spec.path.points[0])
    pieces: list[Piece] = []
    for number, (low, high) in enumerate(_boxes(positions, split.max_span, split.max_blocks)):
        name = f"{spec.output.name}_{number:03d}"
        region, placed, cleared, copied = _build_piece(
            name, low, high, states, positions, codes, source, terraforming
        )
        path = directory / f"{name}{SCHEM_SUFFIX}"
        relative = spec.output.mode == "relative"
        write_schematic(
            region,
            path,
            data_version=spec.output.data_version,
            offset=region.origin - first if relative else region.origin,
            paste_origin=BlockPos(0, 0, 0) if relative else first,
            metadata={"Part": str(number), "Of": spec.output.name},
        )
        pieces.append(Piece(name, path, region.origin, region.size, placed, cleared, copied))
    _manifest(directory / f"{spec.output.name}.toml", spec, pieces, source is not None)
    return pieces


def _changed_cells(world: BlockWorld, terraforming: Terraforming | None):
    # Every changed cell as (positions, codes into `states`): laid blocks and
    # air where ground is cut and nothing is laid.
    states: list[BlockState] = [AIR]
    index: dict[BlockState, int] = {AIR: 0}
    xs, ys, zs, codes = [], [], [], []
    laid: set[tuple[int, int, int]] = set()
    for _, blocks in world.iter_resolved_chunks(None):
        for position, block_id in blocks.items():
            state = world.registry.get_state(block_id)
            code = index.get(state)
            if code is None:
                code = index[state] = len(states)
                states.append(state)
            xs.append(position.x)
            ys.append(position.y)
            zs.append(position.z)
            codes.append(code)
            laid.add((position.x, position.y, position.z))
    if terraforming is not None:
        for position in terraforming.cut:
            key = (position.x, position.y, position.z)
            if key in laid:
                continue
            xs.append(position.x)
            ys.append(position.y)
            zs.append(position.z)
            codes.append(0)
    if not xs:
        raise ValueError("the build changes nothing to export")
    positions = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=1).astype(np.int64)
    return states, positions, np.array(codes, dtype=np.int64)


def _boxes(positions: np.ndarray, span: int, budget: int):
    # A trimmed box per occupied `span` x `span` tile, layered to fit `budget`.
    tiles = np.stack([positions[:, 0] // span, positions[:, 2] // span], axis=1)
    order = np.lexsort((tiles[:, 1], tiles[:, 0]))
    tiles, sorted_positions = tiles[order], positions[order]
    breaks = np.flatnonzero(np.any(np.diff(tiles, axis=0) != 0, axis=1)) + 1
    for group in np.split(sorted_positions, breaks):
        low, high = group.min(axis=0), group.max(axis=0)
        width, depth = high[0] - low[0] + 1, high[2] - low[2] + 1
        layers = max(1, budget // max(1, width * depth))
        for bottom in range(int(low[1]), int(high[1]) + 1, layers):
            top = min(int(high[1]), bottom + layers - 1)
            yield (
                BlockPos(int(low[0]), bottom, int(low[2])),
                BlockPos(int(high[0]), top, int(high[2])),
            )


def _build_piece(name, low, high, states, positions, codes, source, terraforming):
    size = high - low + (1, 1, 1)
    palette = BlockPalette()
    region = BlockRegion(name, low, size, palette)
    cells = np.frombuffer(region.cells, dtype=np.uint32).reshape(size.y, size.z, size.x)
    occupancy = np.frombuffer(region.occupancy, dtype=np.uint8).reshape(size.y, size.z, size.x)
    inside = np.all((positions >= np.array(low.as_tuple())) & (positions <= np.array(high.as_tuple())), axis=1)
    here, code = positions[inside] - np.array(low.as_tuple()), codes[inside]
    lookup = np.array([palette.add(state) for state in states], dtype=np.uint32)
    cells[here[:, 1], here[:, 2], here[:, 0]] = lookup[code]
    occupancy[here[:, 1], here[:, 2], here[:, 0]] = 1
    placed = int(np.count_nonzero(code))
    cleared = int(np.count_nonzero(code == 0))
    copied = 0
    if source is not None:
        copied = _copy_ground(region, cells, occupancy, source, terraforming)
    return region, placed, cleared, copied


def _copy_ground(region, cells, occupancy, source: WorldBlocks, terraforming: Terraforming) -> int:
    # Unchanged cells take the save's block, or air inside the clearance.
    low, size = region.origin, region.size
    floor, ceiling = _clearance_box(terraforming, low, size)
    copied = 0
    lookups: dict[int, np.ndarray] = {}
    for chunk_x in range(low.x >> 4, (low.x + size.x - 1 >> 4) + 1):
        for chunk_z in range(low.z >> 4, (low.z + size.z - 1 >> 4) + 1):
            x0, x1 = max(low.x, chunk_x * 16), min(low.x + size.x, chunk_x * 16 + 16)
            z0, z1 = max(low.z, chunk_z * 16), min(low.z + size.z, chunk_z * 16 + 16)
            for section_y in range(low.y >> 4, (low.y + size.y - 1 >> 4) + 1):
                found = source.section(chunk_x, section_y, chunk_z)
                if found is None:
                    continue
                states, indices = found
                y0, y1 = max(low.y, section_y * 16), min(low.y + size.y, section_y * 16 + 16)
                block = indices.reshape(16, 16, 16)[
                    y0 - section_y * 16:y1 - section_y * 16,
                    z0 - chunk_z * 16:z1 - chunk_z * 16,
                    x0 - chunk_x * 16:x1 - chunk_x * 16,
                ]
                key = id(states)
                mapped = lookups.get(key)
                if mapped is None:
                    mapped = lookups[key] = np.array(
                        [0 if state.name in AIR_NAMES else region.palette.add(state) for state in states],
                        dtype=np.uint32,
                    )
                natural = mapped[block]
                ys = np.arange(y0, y1)[:, None, None]
                keep = (
                    (ys < floor[None, z0 - low.z:z1 - low.z, x0 - low.x:x1 - low.x])
                    | (ys > ceiling[None, z0 - low.z:z1 - low.z, x0 - low.x:x1 - low.x])
                )
                natural = np.where(keep, natural, 0).astype(np.uint32)
                window = (
                    slice(y0 - low.y, y1 - low.y),
                    slice(z0 - low.z, z1 - low.z),
                    slice(x0 - low.x, x1 - low.x),
                )
                free = occupancy[window] == 0
                cells[window] = np.where(free, natural, cells[window])
                copied += int(np.count_nonzero(free & (natural != 0)))
    return copied


def _clearance_box(terraforming: Terraforming, low: BlockPos, size: BlockPos):
    # Clearance floor/ceiling over the box; nothing is cleared off the grid.
    floor = np.full((size.z, size.x), OPEN, dtype=np.int64)
    ceiling = np.full((size.z, size.x), -OPEN, dtype=np.int64)
    clearance, grid = terraforming.clearance, terraforming.grid
    if clearance is None:
        return floor, ceiling
    rows = np.arange(size.z) + low.z - grid.origin.z
    cols = np.arange(size.x) + low.x - grid.origin.x
    row_ok = (rows >= 0) & (rows < grid.depth)
    col_ok = (cols >= 0) & (cols < grid.width)
    r, c = np.ix_(rows[row_ok], cols[col_ok])
    floor[np.ix_(row_ok, col_ok)] = clearance.floor[r, c]
    ceiling[np.ix_(row_ok, col_ok)] = clearance.ceiling[r, c]
    return floor, ceiling


def _manifest(path: FilePath, spec: BuildSpec, pieces: list[Piece], ground: bool) -> None:
    first = BlockPos.coerce(spec.path.points[0])
    how = (
        "Each piece is the world as the build leaves it: paste it as it is, air\n"
        "# and all -- //schem load <file>, then //paste -o -- and it changes what the\n"
        "# build changes and nothing else. The pieces do not overlap; any order."
        if ground else
        "The pieces hold the build's own blocks only: paste them with -a so their\n"
        "# air leaves the world alone -- //schem load <file>, then //paste -o -a."
    )
    lines = [
        f"# {spec.output.name}: {len(pieces)} pieces, {'with' if ground else 'without'} terrain.",
        f"# {how}",
        f"# Coordinates are absolute; the paste origin is the first path point {list(first.as_tuple())}.",
        "",
        f'name = "{spec.output.name}"',
        f"terrain = {'true' if ground else 'false'}",
        f"data_version = {spec.output.data_version}",
        f"paste_origin = {list(first.as_tuple())}",
        f"piece_count = {len(pieces)}",
        f"placed = {sum(piece.placed for piece in pieces)}",
        f"cleared = {sum(piece.cleared for piece in pieces)}",
        f"copied = {sum(piece.copied for piece in pieces)}",
    ]
    for piece in pieces:
        lines += [
            "",
            "[[piece]]",
            f'file = "{piece.path.name}"',
            f"origin = {list(piece.origin.as_tuple())}",
            f"size = {list(piece.size.as_tuple())}",
            f"placed = {piece.placed}",
            f"cleared = {piece.cleared}",
            f"copied = {piece.copied}",
        ]
    path.write_text("\n".join(lines) + "\n")


def write(
    world: BlockWorld, spec: BuildSpec, terraforming: Terraforming | None = None
) -> FilePath:
    # The output suffix picks .schem or .litematic; a split also writes pieces.
    if spec.output.split is not None:
        write_pieces(world, spec, terraforming)
    region = BlockRegion.from_world(world, name=spec.output.name)
    first_point = spec.path.points[0]
    if spec.output.path.suffix == SCHEM_SUFFIX:
        # Offset is the minimum corner; relative mode rebases it on the first
        # path point so a plain //paste puts that point under the player.
        offset = (
            region.origin - first_point
            if spec.output.mode == "relative"
            else region.origin
        )
        paste_origin = (
            BlockPos(0, 0, 0)
            if spec.output.mode == "relative"
            else first_point
        )
        return write_schematic(
            region,
            spec.output.path,
            data_version=spec.output.data_version,
            offset=offset,
            paste_origin=paste_origin,
        )
    return write_litematic(
        [region],
        spec.output.path,
        data_version=spec.output.data_version,
        name=spec.output.name,
        origin=(
            BlockPos(0, 0, 0) if spec.output.mode == "absolute"
            else first_point if spec.output.mode == "relative"
            else None
        ),
    )
