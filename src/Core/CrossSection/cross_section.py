from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

from ..coordinates import BlockPos
from ..modes import StepMode
from .annotations import FeatureAnnotation
from .steps import STEP_RULES, StepRule, column_pushes, lead
from .turns import (
    CLOCKWISE,
    Rotation,
    Turn,
    forward,
    heading,
    is_quarter,
    neighbors,
    normalize,
    place,
)

# Section-local axes, as authored: x is lateral (positive to the right of
# travel), y is height above the path, and z is depth, with forward at -z.
LocalPoint = BlockPos


class PointField:
    # The annotation counterpart of BlockField: named feature locations in
    # local coordinates, with no blocks.

    def __init__(self, annotations: Iterable[FeatureAnnotation] = ()) -> None:
        self._annotations: dict[str, FeatureAnnotation] = {}
        self._revision = 0
        for annotation in annotations:
            self.add(annotation)

    @property
    def revision(self) -> int:
        return self._revision

    def add(self, annotation: FeatureAnnotation) -> None:
        if annotation.name in self._annotations:
            raise ValueError(f"Annotation name {annotation.name!r} is already used")
        self._annotations[annotation.name] = annotation
        self._revision += 1

    def get(self, name: str) -> FeatureAnnotation:
        return self._annotations[name]

    def __contains__(self, name: object) -> bool:
        return name in self._annotations

    def __iter__(self) -> Iterator[FeatureAnnotation]:
        return iter(tuple(self._annotations.values()))

    def __len__(self) -> int:
        return len(self._annotations)


class SectionOffset(NamedTuple):
    # Where one annotation delivers, relative to a path point, in the order
    # the positions belong on its feature's path. x and z are world axes; y is
    # the local height above the path.
    annotation: FeatureAnnotation
    offsets: tuple[BlockPos, ...]


OrientedSection = tuple[SectionOffset, ...]


def orient_annotation(
    annotation: FeatureAnnotation,
    origin: LocalPoint,
    quarter: Turn,
    lead: int,
    steps: tuple[int, ...] = (0,),
    shift: tuple[int, int] = (0, 0),
) -> SectionOffset:
    # Place one annotation in a quarter-turn layout. `lead` is how far the
    # rest of its section moves it, and `steps` are its own positions, both in
    # blocks forward. `shift` is added last, in world axes.
    lx, ly, lz = annotation.position - origin
    fx, fz = forward(quarter)
    x, z = place(quarter, lx, lz)
    x += shift[0]
    z += shift[1]
    return SectionOffset(
        annotation,
        tuple(
            BlockPos(x + fx * (lead + step), ly, z + fz * (lead + step))
            for step in steps
        ),
    )


class _Profile(NamedTuple):
    # What orienting needs from a section's points, gathered once per
    # revision: each annotation with its steps and lead, the rightmost local
    # column, and the long length.
    entries: tuple[tuple[FeatureAnnotation, tuple[int, ...], int], ...]
    right: int
    long_length: int


def _profile(
    points: PointField,
    origin: LocalPoint,
    rules: Mapping[StepMode, StepRule],
) -> _Profile:
    annotations = tuple(points)
    if not annotations:
        return _Profile((), 0, 0)
    first = annotations[0].position
    min_x = max_x = first.x
    min_z = max_z = first.z
    stepped = []
    for annotation in annotations:
        position = annotation.position
        column = position.x - origin.x
        stepped.append((annotation, column, rules[annotation.step_mode](column)))
        min_x, max_x = min(min_x, position.x), max(max_x, position.x)
        min_z, max_z = min(min_z, position.z), max(max_z, position.z)
    pushes = column_pushes((column, steps) for _, column, steps in stepped)
    return _Profile(
        entries=tuple(
            (annotation, steps, lead(column, pushes))
            for annotation, column, steps in stepped
        ),
        right=max_x - origin.x,
        long_length=max(max_x - min_x, max_z - min_z) + 1,
    )


def _mirrored(section: OrientedSection) -> OrientedSection:
    # A counter-clockwise turn is its clockwise twin mirrored about the x axis,
    # the heading the section was authored along. Travel mirrors with it, and a
    # diagonal's two halves trade places, so the stagger leans the other way
    # without the join having to know which way round the path is turning.
    return tuple(
        SectionOffset(
            offset.annotation,
            tuple(BlockPos(p.x, p.y, -p.z) for p in offset.offsets),
        )
        for offset in section
    )


class _OrientationCache:
    # Layouts are keyed by turn and direction together: the same turn laid out
    # the other way round is a different layout.
    __slots__ = ("layouts", "profile", "revision")

    def __init__(self) -> None:
        self.layouts: dict[tuple[Turn, Rotation], OrientedSection] = {}
        self.profile: _Profile | None = None
        self.revision: int | None = None


@dataclass(frozen=True, slots=True)
class CrossSection:
    # A virtual cross section. The physical profile, and every other set of
    # blocks, belongs to a feature; the section only records where each
    # feature sits. Delivering it at a path point yields those locations.

    points: PointField = field(default_factory=PointField)
    # The local point that sits on the path centerline.
    origin: LocalPoint = BlockPos(0, 0, 0)
    # The rule each step mode follows. See steps.py.
    step_rules: Mapping[StepMode, StepRule] = field(default_factory=lambda: STEP_RULES)
    # Each turn is oriented once and stored until the points change.
    _orientations: _OrientationCache = field(
        default_factory=_OrientationCache,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", BlockPos.coerce(self.origin))

    @property
    def long_length(self) -> int:
        # Chebyshev length of the section in plan: its widest horizontal extent.
        return self._current().profile.long_length

    def reach(self, turn: Turn | None = None) -> float:
        # How far along travel any oriented position sits from the path point,
        # ahead or behind, at `turn` or at the worst of every turn. Generation
        # sizes the lead a section is pasted with from this. Mirroring a layout
        # mirrors travel with it, so a turn reaches the same either way round.
        turns = range(0, 360, 45) if turn is None else (turn,)
        furthest = 0.0
        for each in turns:
            direction = heading(each)
            for offset in self.orient(each):
                for position in offset.offsets:
                    along = position.x * direction.x + position.z * direction.z
                    furthest = max(furthest, float(abs(along)))
        return furthest

    def orient(
        self, turn: Turn, direction: Rotation = CLOCKWISE
    ) -> OrientedSection:
        # `turn` is how far round from the authored heading the section is laid
        # out, and `direction` is which way round it was counted, so the same
        # turn counted the other way travels the other way too. Every turn is
        # counted, including the ones that travel the same either way round: a
        # path carrying a direction through them lays them out mirrored.
        turn = normalize(turn)
        key = (turn, direction)
        cache = self._current()
        oriented = cache.layouts.get(key)
        if oriented is None:
            clockwise = self._orient(turn, cache.profile)
            oriented = (
                clockwise
                if direction == CLOCKWISE
                else _mirrored(clockwise)
            )
            cache.layouts[key] = oriented
        return oriented

    def _current(self) -> _OrientationCache:
        cache = self._orientations
        if cache.profile is None or cache.revision != self.points.revision:
            cache.layouts.clear()
            cache.profile = _profile(self.points, self.origin, self.step_rules)
            cache.revision = self.points.revision
        return cache

    def _orient(self, turn: Turn, profile: _Profile) -> OrientedSection:
        if is_quarter(turn):
            return self._layout(turn, profile, (0, 0))

        # A diagonal joins the quarter layouts on either side. The lower
        # layout's rightmost point meets the same point in the upper layout,
        # then the upper layout backtracks half the long length along the
        # lower heading while the lower layout moves forward by the same.
        # Points the two layouts share are left for the feature path's rules.
        lower, upper = neighbors(turn)
        lower_x, lower_z = place(lower, profile.right, 0)
        upper_x, upper_z = place(upper, profile.right, 0)
        fx, fz = forward(lower)
        half = profile.long_length // 2
        return self._layout(lower, profile, (fx * half, fz * half)) + self._layout(
            upper,
            profile,
            (lower_x - upper_x - fx * half, lower_z - upper_z - fz * half),
        )

    def _layout(
        self, quarter: Turn, profile: _Profile, shift: tuple[int, int]
    ) -> OrientedSection:
        return tuple(
            orient_annotation(annotation, self.origin, quarter, lead, steps, shift)
            for annotation, steps, lead in profile.entries
        )
