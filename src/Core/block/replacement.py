from __future__ import annotations

from dataclasses import dataclass
from math import erf, floor, isfinite, sqrt

from ..constants import AIR_NAMES
from ..coordinates import BlockPos
from .palette import BlockId
from .registry import BlockRegistry
from .states import BlockState


def _random(x: int, y: int, z: int, seed: int) -> float:
    value = (
        x * 0x9E3779B1 ^ y * 0x85EBCA77 ^ z * 0xC2B2AE3D ^ seed
    ) & 0xFFFFFFFF
    value ^= value >> 16
    value = value * 0x7FEB352D & 0xFFFFFFFF
    value ^= value >> 15
    value = value * 0x846CA68B & 0xFFFFFFFF
    return (value ^ value >> 16) / 0x100000000


def _noise(position: BlockPos, scale: float, seed: int) -> float:
    x, y, z = (axis / scale for axis in position)
    ix, iy, iz = floor(x), floor(y), floor(z)
    fx, fy, fz = (value - floor(value) for value in (x, y, z))
    fx, fy, fz = (value * value * (3 - 2 * value) for value in (fx, fy, fz))
    total = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weight = (
                    (fx if dx else 1 - fx)
                    * (fy if dy else 1 - fy)
                    * (fz if dz else 1 - fz)
                )
                total += weight * _random(ix + dx, iy + dy, iz + dz, seed)
    return (1 + erf((total - 0.5) / (0.185 * sqrt(2)))) / 2


@dataclass(frozen=True, slots=True)
class ReplacementRules:
    blocks: tuple[tuple[BlockState, float], ...]
    pattern: str = "random"
    scale: float = 8.0
    seed: int = 0
    source_blocks: frozenset[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if self.source_blocks is not None:
            object.__setattr__(self, "source_blocks", frozenset(self.source_blocks))
        if not self.blocks or any(
            not isinstance(amount, (int, float))
            or isinstance(amount, bool)
            or not isfinite(amount)
            or amount <= 0
            for _, amount in self.blocks
        ):
            raise ValueError("Replacement blocks need positive amounts")
        if self.pattern not in ("random", "noise"):
            raise ValueError("Replacement pattern must be 'random' or 'noise'")
        if (
            not isinstance(self.scale, (int, float))
            or isinstance(self.scale, bool)
            or not isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("Replacement scale must be positive")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("Replacement seed must be an integer")

    def index_at(self, position: BlockPos) -> int:
        position = BlockPos.coerce(position)
        if len(self.blocks) == 1:
            return 0
        value = (
            _random(position.x, position.y, position.z, self.seed)
            if self.pattern == "random"
            else _noise(position, self.scale, self.seed)
        )
        threshold = value * sum(amount for _, amount in self.blocks)
        for index, (_, amount) in enumerate(self.blocks):
            threshold -= amount
            if threshold < 0:
                return index
        return len(self.blocks) - 1


@dataclass(frozen=True, slots=True)
class BlockReplacement:
    rules: ReplacementRules
    block_ids: tuple[BlockId, ...]

    @classmethod
    def create(cls, rules: ReplacementRules, registry: BlockRegistry) -> BlockReplacement:
        block_ids = []
        for state, _ in rules.blocks:
            try:
                block_ids.append(registry.get_id(state))
            except KeyError:
                block_ids.append(registry.register(state))
        return cls(rules, tuple(block_ids))

    def at(self, source_name: str, position: BlockPos) -> BlockId | None:
        if source_name in AIR_NAMES or (
            self.rules.source_blocks is not None
            and source_name not in self.rules.source_blocks
        ):
            return None
        return self.block_ids[self.rules.index_at(position)]
