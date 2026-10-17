"""Repair distance: the fewest observed stickers to recolor for a valid standard cube.

For each of the 24 orientations of the standard color scheme, every corner and
edge slot is given a cubie and an orientation, at a cost of the observed
stickers that disagree with it. A dynamic program over the slots, with states
``(used cubies, orientation sum, permutation parity)``, finds the cheapest
assignment that is a reachable cube (see ``completion_solver``); corners and
edges are joined on equal parity. Hidden stickers cost nothing, so they take
whatever the cheapest valid cube needs. The result is exact.

Because three faces stay hidden, even random colors are only a few stickers
from a valid cube; ``CHANCE_DISTANCE`` is that average, the zero point of the
validity score.
"""

import random
from dataclasses import dataclass
from functools import cache

import numpy as np

from .color_scheme import STANDARD_COLORS, STANDARD_ORIENTATIONS, ColorScheme
from .facelets import CORNER_FACELETS, CORNER_FACES, EDGE_FACELETS, EDGE_FACES, FACES, facelet_name
from .observation import Observation
from .pieces import CubieState, Placement

# Mean repair distance of uniformly random colors on the 27 stickers of a three-face
# view (U, F, R), from ``chance_distance(2000, seed=0)``.
CHANCE_DISTANCE = 4.447

_INF = 10**9


@dataclass(frozen=True)
class Fix:
    facelet: int
    seen: str
    color: str

    def to_dict(self) -> dict[str, str]:
        return {"facelet": facelet_name(self.facelet), "seen": self.seen, "color": self.color}


@dataclass(frozen=True)
class Repair:
    distance: int
    # The recolorings that reach the nearest valid cube (one of them, when several tie).
    fixes: tuple[Fix, ...]
    scheme: ColorScheme


def _costs(
    stickers: tuple[str | None, ...], scheme: ColorScheme, slots: tuple[tuple[int, ...], ...],
    faces: tuple[str, ...],
) -> list[list[int]]:
    """cost[slot][cubie * n + orientation]: observed stickers that disagree with that placement."""
    out = []
    for facelets in slots:
        n = len(facelets)
        seen = [stickers[i] for i in facelets]
        row = []
        for cubie_faces in faces:
            colors = [scheme.color(f) for f in cubie_faces]
            for o in range(n):
                row.append(sum(1 for k in range(n) if (s := seen[(k + o) % n]) is not None and s != colors[k]))
        out.append(row)
    return out


@cache
def _tables(pieces: int) -> tuple[np.ndarray, np.ndarray]:
    """Cubie-set bitmasks grouped by size, and the inversions (mod 2) placing cubie c after mask adds."""
    masks = np.arange(1 << pieces)
    size = np.array([m.bit_count() for m in range(1 << pieces)])
    flips = np.array([[(m >> (c + 1)).bit_count() & 1 for c in range(pieces)] for m in range(1 << pieces)])
    return np.stack([masks, size]), flips


class _Search:
    """The cheapest legal assignment of all cubies of one kind, for many color schemes at once.

    Slots are filled in order. ``best[scheme, used cubies, orientation sum, parity]`` holds the
    cheapest way to fill the first popcount(used) slots, and ``move`` the placement that achieved
    it; all schemes are handled in the same array operations.
    """

    def __init__(self, costs: np.ndarray, n: int, modulus: int):
        schemes, pieces, _ = costs.shape
        (masks, size), flips = _tables(pieces)
        best = np.full((schemes, 1 << pieces, modulus, 2), _INF, dtype=np.int64)
        move = np.full(best.shape, -1, dtype=np.int64)
        best[:, 0, 0, 0] = 0
        for slot in range(pieces):
            filled = masks[size == slot]
            for cubie in range(pieces):
                src = filled[(filled >> cubie) & 1 == 0]
                dst = src | (1 << cubie)
                flip = flips[src, cubie].astype(bool)[None, :, None, None]
                before = best[:, src]
                for o in range(n):
                    # Orientation sum shifts by o; parity flips where the cubie adds an odd inversion.
                    cand = np.roll(before + costs[:, slot, cubie * n + o][:, None, None, None], o, axis=2)
                    cand = np.where(flip, cand[..., ::-1], cand)
                    current = best[:, dst]
                    better = cand < current
                    best[:, dst] = np.where(better, cand, current)
                    move[:, dst] = np.where(better, cubie * n + o, move[:, dst])
        self.n, self.modulus, self.flips, self.move = n, modulus, flips, move
        self.full = (1 << pieces) - 1
        self.cost = best[:, self.full, 0, :]  # [scheme, parity]

    def placements(self, scheme: int, parity: int) -> tuple[Placement, ...]:
        path, used, orientation, par = [], self.full, 0, parity
        while used:
            cubie, o = divmod(int(self.move[scheme, used, orientation, par]), self.n)
            used ^= 1 << cubie
            par ^= int(self.flips[used, cubie])
            orientation = (orientation - o) % self.modulus
            path.append(Placement(cubie, o))
        return tuple(reversed(path))


def repair(observation: Observation) -> Repair:
    """The nearest valid standard cube to what was observed."""
    stickers = observation.stickers
    schemes = STANDARD_ORIENTATIONS
    centers = np.array([
        sum(1 for f, face in enumerate(FACES) if (c := stickers[9 * f + 4]) is not None and c != s.color(face))
        for s in schemes
    ])
    corners = _Search(np.array([_costs(stickers, s, CORNER_FACELETS, CORNER_FACES) for s in schemes]), 3, 3)
    edges = _Search(np.array([_costs(stickers, s, EDGE_FACELETS, EDGE_FACES) for s in schemes]), 2, 2)
    # Corners and edges must have equal permutation parity.
    total = centers[:, None] + corners.cost + edges.cost
    index, parity = np.unravel_index(int(np.argmin(total)), total.shape)
    scheme = schemes[index]
    state = CubieState(corners.placements(index, parity), edges.placements(index, parity))
    colors = state.facelets(scheme)
    fixes = tuple(
        Fix(i, seen, colors[i]) for i, seen in enumerate(stickers) if seen is not None and seen != colors[i]
    )
    assert len(fixes) == int(total[index, parity])
    return Repair(len(fixes), fixes, scheme)


def chance_distance(samples: int, seed: int = 0) -> float:
    """Mean repair distance of uniformly random colors on a three-face view (U, F, R)."""
    rng = random.Random(seed)
    colors = sorted(STANDARD_COLORS)
    total = 0
    for _ in range(samples):
        faces = {f: [rng.choice(colors) for _ in range(9)] for f in "UFR"}
        total += repair(Observation.from_faces(faces)).distance
    return total / samples
