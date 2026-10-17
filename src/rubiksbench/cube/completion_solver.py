"""Existential completion: is there a reachable cube state agreeing with the observations?

For a fixed color scheme, a full state is reachable from solved by face turns
iff (standard group theory of the 3x3 cube):

* every corner and edge cubie appears exactly once,
* the corner orientations sum to 0 mod 3,
* the edge orientations sum to 0 mod 2,
* the corner and edge permutations have equal parity.

Corners and edges interact only through parity, so each is searched on its
own. The search is exact: a depth-first search over slots in index order with
memoised dead states ``(used cubies, orientation sum, permutation parity)``,
which is a complete enumeration of the constrained assignments. Hidden slots
simply have every placement as a candidate.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..evaluation.taxonomy import Flag
from .color_scheme import ColorScheme
from .facelets import CORNER_FACELETS, CORNER_NAMES, EDGE_FACELETS, EDGE_NAMES, facelet_name
from .observation import Observation
from .pieces import CubieState, Placement, corner_candidates, edge_candidates


def _search(
    candidates: Sequence[Sequence[Placement]],
    modulus: int,
    orientation_target: int | None,
    parity_target: int | None,
) -> tuple[Placement, ...] | None:
    """Find an assignment of distinct cubies to all slots meeting the targets."""
    n = len(candidates)
    dead: set[tuple[int, int, int]] = set()
    path: list[Placement] = []

    def dfs(used: int, orientation: int, parity: int) -> bool:
        slot = used.bit_count()
        if slot == n:
            return (orientation_target is None or orientation == orientation_target) and (
                parity_target is None or parity == parity_target
            )
        state = (used, orientation, parity)
        if state in dead:
            return False
        for placement in candidates[slot]:
            bit = 1 << placement.cubie
            if used & bit:
                continue
            # Inversions added: earlier slots holding a higher-numbered cubie.
            inversions = (used >> (placement.cubie + 1)).bit_count()
            path.append(placement)
            if dfs(used | bit, (orientation + placement.orientation) % modulus, parity ^ (inversions & 1)):
                return True
            path.pop()
        dead.add(state)
        return False

    return tuple(path) if dfs(0, 0, 0) else None


@dataclass(frozen=True)
class Culprit:
    """Observed stickers that explain a failure, e.g. the stickers of an impossible corner."""

    reason: str
    facelets: tuple[int, ...]


@dataclass
class _PieceAnalysis:
    by_parity: dict[int, tuple[Placement, ...]]
    flags: list[Flag]
    reasons: list[str]
    culprits: list[Culprit] = field(default_factory=list)


def _analyse(
    kind: str,
    candidates: list[list[Placement]],
    observation: Observation,
    slot_facelets: tuple[tuple[int, ...], ...],
    slot_names: tuple[str, ...],
    modulus: int,
) -> _PieceAnalysis:
    impossible = Flag.IMPOSSIBLE_CORNER if kind == "corner" else Flag.IMPOSSIBLE_EDGE

    def seen(slot: int) -> tuple[int, ...]:
        return tuple(i for i in slot_facelets[slot] if observation.stickers[i] is not None)

    def colors(slot: int) -> str:
        return "-".join(observation.stickers[i] or "" for i in seen(slot))

    reasons, culprits = [], []
    for slot, options in enumerate(candidates):
        if not options:
            shown = ", ".join(
                f"{facelet_name(i)}={observation.stickers[i] or '?'}" for i in slot_facelets[slot]
            )
            reasons.append(f"{kind} slot {slot_names[slot]} ({shown}) matches no {kind} cubie")
            culprits.append(Culprit(f"impossible {kind} {slot_names[slot]} ({colors(slot)})", seen(slot)))
    if reasons:
        return _PieceAnalysis({}, [impossible], reasons, culprits)

    if _search(candidates, modulus, None, None) is None:
        # Name cubies that several slots can only be filled by, when that is the cause.
        forced: dict[int, list[str]] = {}
        for slot, options in enumerate(candidates):
            if len(cubies := {p.cubie for p in options}) == 1:
                forced.setdefault(cubies.pop(), []).append(slot_names[slot])
        clashes = {cubie: slots for cubie, slots in forced.items() if len(slots) > 1}
        return _PieceAnalysis(
            {},
            [Flag.DUPLICATE_PIECE],
            [
                f"{kind} cubie {slot_names[cubie]} is required in slots {', '.join(slots)}"
                for cubie, slots in clashes.items()
            ]
            or [f"no assignment uses each {kind} cubie exactly once"],
            [
                Culprit(
                    f"duplicate {kind} in {', '.join(slots)}",
                    tuple(i for s in slots for i in seen(slot_names.index(s))),
                )
                for slots in clashes.values()
            ],
        )

    by_parity = {
        p: found for p in (0, 1) if (found := _search(candidates, modulus, 0, p)) is not None
    }
    if not by_parity:
        invariant = "twist (sum mod 3)" if kind == "corner" else "flip (sum mod 2)"
        return _PieceAnalysis(
            {},
            [Flag.ORIENTATION_INVARIANT_FAILURE],
            [f"every {kind} assignment violates the {kind} {invariant} invariant"],
        )
    return _PieceAnalysis(by_parity, [], [])


@dataclass
class SchemeAttempt:
    """The outcome of searching for a completion under one color scheme."""

    scheme: ColorScheme
    state: CubieState | None
    flags: list[Flag] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    # The stickers behind a piece-level failure; empty for invariant and parity failures,
    # which belong to the whole cube rather than to any one piece.
    culprits: list[Culprit] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.state is not None

    @property
    def stage(self) -> int:
        """How many piece-level stages passed: 1 every visible piece is a real cubie,
        2 no cubie is needed twice, 3 a legal completion exists."""
        if self.ok:
            return 3
        if {Flag.IMPOSSIBLE_CORNER, Flag.IMPOSSIBLE_EDGE} & set(self.flags):
            return 0
        if Flag.DUPLICATE_PIECE in self.flags:
            return 1
        return 2


def complete(observation: Observation, scheme: ColorScheme) -> SchemeAttempt:
    """Search for a reachable state under ``scheme`` that agrees with every observed sticker."""
    for face_index, color in enumerate(scheme.colors):
        center = observation.stickers[9 * face_index + 4]
        if center is not None and center != color:
            raise ValueError(f"scheme {scheme} contradicts observed center {center}")

    corners = _analyse(
        "corner", corner_candidates(observation, scheme), observation, CORNER_FACELETS, CORNER_NAMES, 3
    )
    edges = _analyse(
        "edge", edge_candidates(observation, scheme), observation, EDGE_FACELETS, EDGE_NAMES, 2
    )
    flags = corners.flags + edges.flags
    reasons = corners.reasons + edges.reasons
    if flags:
        return SchemeAttempt(scheme, None, flags, reasons, corners.culprits + edges.culprits)

    for parity in (0, 1):
        if parity in corners.by_parity and parity in edges.by_parity:
            state = CubieState(corners.by_parity[parity], edges.by_parity[parity])
            return SchemeAttempt(scheme, state)
    return SchemeAttempt(
        scheme,
        None,
        [Flag.PERMUTATION_PARITY_FAILURE],
        [
            f"corner permutation parity must be {sorted(corners.by_parity)} but edge "
            f"permutation parity must be {sorted(edges.by_parity)}"
        ],
    )


def first_completion(
    observation: Observation, schemes: Iterable[ColorScheme]
) -> tuple[SchemeAttempt | None, list[SchemeAttempt]]:
    """Try schemes in order; return the first success and all failed attempts before it."""
    failures = []
    for scheme in schemes:
        attempt = complete(observation, scheme)
        if attempt.ok:
            return attempt, failures
        failures.append(attempt)
    return None, failures
