from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import NamedTuple

from ..coordinates import BlockPos
from ..modes import StepMode
from .annotations import FeatureAnnotation
from .steps import STEP_RULES, StepRule, column_pushes, lead
from .turns import (
    CLOCKWISE,
    Rotation,
    Turn,
    heading,
    is_quarter,
    neighbors,
    normalize,
    place_along,
    step_along,
)

# Section-local axes, as authored: x is lateral (positive to the right of
# travel), y is height above the path, and z is depth, with forward at -z.
LocalPoint = BlockPos

# A mirror across the section's own long edge. It turns depth over -- local z
# and everything measured along travel, which is what the staircase of a
# stepping column is made of -- and leaves local x alone, so a column stays on
# its own side of the path.
MIRRORED = -1
PLAIN = 1


def mirror_of(quarter: Turn) -> int:
    # Every quarter turn adds a mirror to the one before it, so the mirror
    # alternates round the four: the authored section at 0, mirrored at 90,
    # mirrored back at 180, mirrored again at 270.
    return MIRRORED if (normalize(quarter) // 90) % 2 else PLAIN


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
    #
    # `layout` is the one of the eight these offsets were placed in. Where a
    # point is laid out more than once -- to cover the seam between a run of
    # one layout and a run of another -- it is not the turn the path point is
    # at, and whatever paints blocks here has to be told which.
    annotation: FeatureAnnotation
    offsets: tuple[BlockPos, ...]
    layout: Turn = 0
    # Whether that layout is the mirrored one. The quarters alternate and a
    # join's two arms take opposite mirrors, so the layout alone does not say
    # -- and anything that turns with a position has to know, or it turns the
    # wrong way at half the layouts.
    mirrored: bool = False


OrientedSection = tuple[SectionOffset, ...]


def orient_annotation(
    annotation: FeatureAnnotation,
    origin: LocalPoint,
    turn: Turn,
    lead: int,
    steps: tuple[int, ...] = (0,),
    shift: tuple[int, int] = (0, 0),
    depth: int = 1,
) -> SectionOffset:
    # Place one annotation in a layout. `lead` is how far the rest of its
    # section moves it, and `steps` are its own positions, both in blocks
    # forward. `shift` is added last, in world axes.
    #
    # `depth` of -1 mirrors the section across its own long edge: local z
    # changes sign and so does everything measured along travel, while local
    # x is untouched. The layout is unchanged by that, since a mirror across
    # the long edge moves nothing sideways -- which is what lets the arm keep
    # its own quarter, and anything riding it land on the right side.
    turn = normalize(turn)
    lx, ly, lz = annotation.position - origin
    fx, fz = step_along(turn)
    x, z = place_along(turn, lx, depth * lz)
    x += shift[0]
    z += shift[1]
    return SectionOffset(
        annotation,
        tuple(
            BlockPos(
                x + fx * depth * (lead + step), ly, z + fz * depth * (lead + step)
            )
            for step in steps
        ),
        turn,
        depth < 0,
    )


class _Profile(NamedTuple):
    # What orienting needs from a section's points, gathered once per
    # revision: each annotation with its steps and lead, the rightmost local
    # column, and the long length.
    entries: tuple[tuple[FeatureAnnotation, tuple[int, ...], int], ...]
    right: int
    long_length: int
    depth: int
    # How far each column moves what is outboard of it. Kept so anything
    # sitting beside the section can be placed the way the section would
    # place a column there.
    pushes: Mapping[int, int] = MappingProxyType({})


def _profile(
    points: PointField,
    origin: LocalPoint,
    rules: Mapping[StepMode, StepRule],
) -> _Profile:
    annotations = tuple(points)
    if not annotations:
        return _Profile((), 0, 0, 0)
    first = annotations[0].position
    min_x = max_x = first.x
    min_z = max_z = first.z
    stepped = []
    for annotation in annotations:
        position = annotation.position
        column = position.x - origin.x
        stepped.append((annotation, column, rules[annotation.step_mode](column)))
        left, right_, back, front = _covered(annotation)
        min_x, max_x = min(min_x, left), max(max_x, right_)
        min_z, max_z = min(min_z, back), max(max_z, front)
    pushes = column_pushes((column, steps) for _, column, steps in stepped)
    return _Profile(
        entries=tuple(
            (annotation, steps, lead(column, pushes))
            for annotation, column, steps in stepped
        ),
        right=max_x - origin.x,
        long_length=max(max_x - min_x, max_z - min_z) + 1,
        depth=max_z - min_z + 1,
        pushes=MappingProxyType(pushes),
    )


def _covered(annotation: FeatureAnnotation) -> tuple[int, int, int, int]:
    # The local columns an annotation covers: where it sits, widened by the
    # blocks its feature paints around it, as (left, right, back, front).
    #
    # A whole profile carried by one annotation is the reason this is not
    # just the position. Measured by position alone such a section is one
    # column wide and one long, so a diagonal has nothing to stagger and
    # collapses onto the quarter below it.
    position = annotation.position
    blocks = getattr(annotation.feature, "blocks", None)
    bounds = None if blocks is None else blocks.bounds
    if bounds is None:
        return position.x, position.x, position.z, position.z
    return (
        position.x + bounds.minimum.x,
        position.x + bounds.maximum.x,
        position.z + bounds.minimum.z,
        position.z + bounds.maximum.z,
    )


def _in_travel_order(
    section: OrientedSection, ahead: tuple[int, int]
) -> OrientedSection:
    # Every annotation's positions sorted down the path. A stepping column
    # delivers its own position and one along, and on the right of the
    # section that one is behind -- so shift by shift the pairs come out
    # against travel even when the shifts themselves are in order. What
    # rides a join reads these in order, so they have to be in one.
    return tuple(
        SectionOffset(
            offset.annotation,
            tuple(sorted(
                offset.offsets, key=lambda p: p.x * ahead[0] + p.z * ahead[1]
            )),
            offset.layout,
            offset.mirrored,
        )
        for offset in section
    )


def _centred(section: OrientedSection) -> OrientedSection:
    # The join moved so the path point sits in the middle of it rather than
    # at the end of the arm that happened to be laid first.
    #
    # A join is one piece covering a stretch of path either side of where it
    # is stamped, so the point it is stamped at belongs in the middle of it.
    # Built from the arm being left outwards, the origin lands on that arm's
    # own row and the whole piece reaches off one way, which walks it half a
    # section down the path from where it is meant to be.
    places = [(p.x, p.z) for offset in section for p in offset.offsets]
    if not places:
        return section
    middle_x = (max(x for x, _ in places) + min(x for x, _ in places)) // 2
    middle_z = (max(z for _, z in places) + min(z for _, z in places)) // 2
    if not (middle_x or middle_z):
        return section
    return _repeated(section, ((-middle_x, -middle_z),))


def _repeated(section: OrientedSection, shifts: Sequence[tuple[int, int]]) -> OrientedSection:
    # One arm: the same layout laid at every shift, gathered back into one
    # offset per annotation so a column stays a single run rather than half a
    # dozen one-block ones. Shift-major, so the positions come out in the
    # order they lie along the walk.
    return tuple(
        SectionOffset(
            offset.annotation,
            tuple(dict.fromkeys(
                BlockPos(position.x + x, position.y, position.z + z)
                for x, z in shifts
                for position in offset.offsets
            )),
            offset.layout,
            offset.mirrored,
        )
        for offset in section
    )


def _onto(
    section: OrientedSection,
    other: OrientedSection,
    origin: LocalPoint,
    edge: int,
) -> tuple[int, int]:
    # What moves `other` so that its outermost column sits on the same cell
    # of `section`. Where the column is laid, not where it was authored: a
    # column outboard of a stepping one is carried along travel.
    here = _column(section, origin, edge)
    there = _column(other, origin, edge)
    return here.x - there.x, here.z - there.z


def _column(section: OrientedSection, origin: LocalPoint, edge: int) -> BlockPos:
    for offset in section:
        if offset.annotation.position.x - origin.x == edge:
            return offset.offsets[0]
    raise ValueError(f"The section has no point in column {edge}")


def _mirrored(section: OrientedSection) -> OrientedSection:
    # A counter-clockwise turn is its clockwise twin mirrored about the x axis,
    # the heading the section was authored along. Travel mirrors with it, and a
    # diagonal's two halves trade places, so the stagger leans the other way
    # without the join having to know which way round the path is turning.
    # The quarter mirrors with the layout: reflected about the authored
    # heading, a layout laid out for north is one laid out for south.
    return tuple(
        SectionOffset(
            offset.annotation,
            tuple(BlockPos(p.x, p.y, -p.z) for p in offset.offsets),
            normalize(-offset.layout),
            offset.mirrored,
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
    def depth(self) -> int:
        # How far the section runs along travel, in blocks.
        return self._current().profile.depth

    @property
    def long_length(self) -> int:
        # Chebyshev length of the section in plan: its widest horizontal extent.
        return self._current().profile.long_length

    def along(
        self, column: int, step_mode: StepMode = StepMode.CHEBYSHEV
    ) -> tuple[int, ...]:
        # Where something at a local column belongs along travel, in blocks
        # forward, placed the way the section places its own column there:
        # the lead the stepping columns inboard of it push it by, plus its
        # own step.
        #
        # A lane is the reason this is public. A rail two to the right of
        # travel sits on the column two to the right of travel, and that
        # column is not where it was authored -- it is a block along if it
        # steps, and further still if something inboard of it steps. Placed
        # by the authored offset alone a lane lands beside the column it
        # belongs on, which is what put a fifth of the rails on smooth stone
        # and left the kerb line a block off the kerb.
        #
        # In travel order, nearest the back first. A step rule puts the
        # column's own position first because that is the one the rest of
        # the section is measured from, and on the right of the section the
        # step is backwards -- so laid in that order a run walks to the point
        # and then back a block, which reads as a gap and gets bridged into a
        # second line beside the first.
        profile = self._current().profile
        carried = lead(column, profile.pushes)
        return tuple(
            sorted(carried + step for step in self.step_rules[step_mode](column))
        )

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
        # A quarter is one row. A diagonal is the join between two of them,
        # and a section with no points has neither.
        if not profile.entries:
            return ()
        if is_quarter(turn):
            return self._layout(turn, profile, (0, 0), mirror_of(turn))
        return self._joined(turn, profile)

    def _joined(self, diagonal: Turn, profile: _Profile) -> OrientedSection:
        # A diagonal laid out as one piece: the quarter being left, stamped
        # forward of the joint, and the quarter taking up, stamped back
        # behind it, half the section's width each and a block apart along
        # the diagonal.
        #
        # The two arms cross rather than meet. Each is laid through where the
        # other's rows go, so the wedge one opens on the inside of the turn is
        # already filled by the other, and neither run has to be widened or
        # trimmed at its end.
        #
        # This is stamped once, where the diagonal begins. It reaches half a
        # section either side of itself, so stamping it at every point that
        # is near the diagonal would smear the join along the path and lay
        # several times the deck.
        lower, upper = neighbors(diagonal)
        forward_x, forward_z = step_along(diagonal)
        half = max(1, profile.long_length // 2)

        # The quarter taking up is mirrored across the section's long edge
        # first. A mirror, not a turn: turning it leaves the two rows facing
        # the same way round the corner, and only one of the section's two
        # edges carries on through the joint. Mirrored, both of them do --
        # the outer kerb runs unbroken round the outside and the edge line
        # round the inside, which is what a corner has to be.
        # Within a join the mirror is relative, not absolute: the arm being
        # left is taken plain and the arm taking up is mirrored against it,
        # whichever quarters the two happen to be. Measured over every
        # combination of the two mirrors, the two edges to line up on and the
        # two walk directions, that one pairing is the only one that leaves
        # both of the section's edges unbroken round the bend -- and it is
        # the same pairing at all four joins.
        taking_up = self._layout(upper, profile, (0, 0), MIRRORED)
        # Then the two are lined up on that inside edge, before either is
        # walked: the edge of the one being left on the same cell of the one
        # taking up. That is the corner the two rows pivot about.
        being_left = self._layout(lower, profile, (0, 0), PLAIN)
        shift = _onto(being_left, taking_up, self.origin, profile.right)

        # Both arms are laid in the order they lie along travel, rearmost
        # first, so every annotation's positions come out down the path
        # rather than up it. What rides a join reads them in that order.
        ahead = tuple((forward_x * k, forward_z * k) for k in range(half))
        behind = tuple(
            (shift[0] - forward_x * k, shift[1] - forward_z * k)
            for k in reversed(range(half))
        )
        return _centred(_in_travel_order(
            _repeated(being_left, ahead) + _repeated(taking_up, behind),
            (forward_x, forward_z),
        ))

    def _carried(
        self,
        quarter: Turn,
        profile: _Profile,
        shifts: tuple[tuple[int, int], ...],
    ) -> OrientedSection:
        # One arm of a join: the same quarter laid at every shift, gathered
        # back into one offset per annotation so a column stays a single run
        # rather than half a dozen one-block ones.
        laid = [self._layout(quarter, profile, shift) for shift in shifts]
        return tuple(
            SectionOffset(
                stamps[0].annotation,
                tuple(dict.fromkeys(p for stamp in stamps for p in stamp.offsets)),
                quarter,
            )
            for stamps in zip(*laid)
        )

    def _layout(
        self,
        turn: Turn,
        profile: _Profile,
        shift: tuple[int, int],
        depth: int = 1,
    ) -> OrientedSection:
        return tuple(
            orient_annotation(
                annotation, self.origin, turn, lead, steps, shift, depth
            )
            for annotation, steps, lead in profile.entries
        )
