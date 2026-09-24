from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar


Value = TypeVar("Value")


@dataclass(slots=True)
class _Node(Generic[Value]):
    x: int
    z: int
    index: int
    value: Value
    least_x: int
    most_x: int
    least_z: int
    most_z: int
    left: _Node[Value] | None = None
    right: _Node[Value] | None = None


class NearestInPlan(Generic[Value]):
    """Exact nearest column, keeping the first input on equal distances."""

    def __init__(self, positions: Iterable[tuple[int, int, Value]]) -> None:
        indexed = [
            (x, z, index, value)
            for index, (x, z, value) in enumerate(positions)
        ]
        if not indexed:
            raise ValueError("min() arg is an empty sequence")
        self._root = self._build(indexed, 0)

    @classmethod
    def _build(
        cls, positions: list[tuple[int, int, int, Value]], depth: int
    ) -> _Node[Value]:
        axis = depth % 2
        positions.sort(key=lambda each: (each[axis], each[1 - axis], each[2]))
        middle = len(positions) // 2
        x, z, index, value = positions[middle]
        left = cls._build(positions[:middle], depth + 1) if middle else None
        right = (
            cls._build(positions[middle + 1 :], depth + 1)
            if middle + 1 < len(positions)
            else None
        )
        children = [child for child in (left, right) if child is not None]
        return _Node(
            x, z, index, value,
            min([x, *(child.least_x for child in children)]),
            max([x, *(child.most_x for child in children)]),
            min([z, *(child.least_z for child in children)]),
            max([z, *(child.most_z for child in children)]),
            left, right,
        )

    @staticmethod
    def _lower_bound(node: _Node[Value], x: int, z: int) -> int:
        dx = max(node.least_x - x, 0, x - node.most_x)
        dz = max(node.least_z - z, 0, z - node.most_z)
        return dx * dx + dz * dz

    def nearest(self, x: int, z: int) -> Value:
        best: _Node[Value] | None = None
        best_distance = float("inf")
        stack = [self._root]
        while stack:
            node = stack.pop()
            if self._lower_bound(node, x, z) > best_distance:
                continue
            distance = (node.x - x) ** 2 + (node.z - z) ** 2
            if (
                distance < best_distance
                or (distance == best_distance and best is not None and node.index < best.index)
            ):
                best = node
                best_distance = distance
            left, right = node.left, node.right
            if left is not None and right is not None:
                left_distance = self._lower_bound(left, x, z)
                right_distance = self._lower_bound(right, x, z)
                if left_distance <= right_distance:
                    stack.extend((right, left))
                else:
                    stack.extend((left, right))
            elif left is not None:
                stack.append(left)
            elif right is not None:
                stack.append(right)
        assert best is not None
        return best.value
