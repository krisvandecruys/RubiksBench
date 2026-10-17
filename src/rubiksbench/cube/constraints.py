"""Color-level constraints: color count, centers, and candidate color schemes."""

import itertools
from collections import Counter
from collections.abc import Iterator

from ..evaluation.taxonomy import Flag
from .color_scheme import STANDARD_COLORS, STANDARD_OPPOSITE_COLOR, STANDARD_ORIENTATIONS, ColorScheme
from .completion_solver import Culprit
from .facelets import CORNER_FACELETS, CORNER_NAMES, EDGE_FACELETS, EDGE_NAMES, FACES, OPPOSITE_FACE
from .observation import Observation


def observed_centers(observation: Observation) -> dict[str, str]:
    return {f: c for f in FACES if (c := observation.center(f)) is not None}


def color_flags(observation: Observation) -> tuple[list[Flag], list[str]]:
    """Scheme-independent failures: more than six colors or a repeated center color."""
    flags, reasons = [], []
    if len(observation.labels) > 6:
        flags.append(Flag.TOO_MANY_COLORS)
        reasons.append(f"{len(observation.labels)} distinct colors: {sorted(observation.labels)}")
    counts = Counter(observed_centers(observation).values())
    if duplicated := sorted(c for c, n in counts.items() if n > 1):
        flags.append(Flag.DUPLICATE_CENTERS)
        reasons.append(f"color(s) {duplicated} appear on more than one center")
    return flags, reasons


def scheme_free_piece_failures(observation: Observation) -> list[tuple[Flag, Culprit]]:
    """Piece failures that hold under every color scheme, so they need no valid centers.

    A piece showing one color twice cannot exist, and two fully visible pieces with the
    same set of colors would need the same cubie. These are found even when the centers
    rule out every scheme (e.g. two white centers), which the completion search cannot do.
    """
    found: list[tuple[Flag, Culprit]] = []
    complete: dict[tuple[str, frozenset[str]], list[tuple[str, tuple[int, ...]]]] = {}
    for kind, names, slots in (("corner", CORNER_NAMES, CORNER_FACELETS), ("edge", EDGE_NAMES, EDGE_FACELETS)):
        impossible = Flag.IMPOSSIBLE_CORNER if kind == "corner" else Flag.IMPOSSIBLE_EDGE
        for name, facelets in zip(names, slots):
            shown = tuple(i for i in facelets if observation.stickers[i] is not None)
            colors = [observation.stickers[i] for i in shown]
            if len(set(colors)) < len(colors):
                found.append((impossible, Culprit(f"impossible {kind} {name} ({'-'.join(colors)})", shown)))
            elif len(shown) == len(facelets):
                complete.setdefault((kind, frozenset(colors)), []).append((name, shown))
    for (kind, _), group in complete.items():
        if len(group) > 1:
            where = ", ".join(name for name, _ in group)
            found.append((Flag.DUPLICATE_PIECE, Culprit(f"duplicate {kind} in {where}", sum((f for _, f in group), ()))))
    return found


def abstract_schemes(observation: Observation) -> Iterator[ColorScheme]:
    """Every scheme consistent with the observed centers that uses all observed colors.

    Faces left over after placing every observed color get placeholder colors
    (``?1``, ``?2``, ...), which stand for colors never seen in the image.
    Assumes ``color_flags`` found nothing.
    """
    centers = observed_centers(observation)
    free_faces = [f for f in FACES if f not in centers]
    loose = sorted(observation.labels - set(centers.values()))
    placeholders = [
        p for p in (f"?{i}" for i in range(1, 13)) if p not in observation.labels
    ][: len(free_faces) - len(loose)]
    for chosen in itertools.permutations(free_faces, len(loose)):
        mapping = centers | dict(zip(chosen, loose))
        rest = [f for f in free_faces if f not in mapping]
        yield ColorScheme.from_mapping(mapping | dict(zip(rest, placeholders)))


def standard_schemes(observation: Observation) -> tuple[list[ColorScheme], list[Flag], list[str]]:
    """Orientations of the standard scheme matching the observed centers.

    When none match, explain why: opposite colors on adjacent faces, opposite
    faces without an opposite color pair, or otherwise a mirrored (chirally
    wrong) arrangement.
    """
    if not observation.labels <= STANDARD_COLORS:
        extra = sorted(observation.labels - STANDARD_COLORS)
        return [], [Flag.NONSTANDARD_COLOR_LABELS], [f"labels {extra} are not standard colors"]

    centers = observed_centers(observation)
    schemes = [
        s for s in STANDARD_ORIENTATIONS if all(s.color(f) == c for f, c in centers.items())
    ]
    if schemes:
        return schemes, [], []

    flags, reasons = [], []
    for a, b in itertools.combinations(centers, 2):
        ca, cb = centers[a], centers[b]
        if OPPOSITE_FACE[a] == b:
            if STANDARD_OPPOSITE_COLOR[ca] != cb:
                flags.append(Flag.STANDARD_OPPOSITE_PAIR_INVALID)
                reasons.append(f"opposite faces {a}={ca} and {b}={cb} are not a standard opposite pair")
        elif STANDARD_OPPOSITE_COLOR[ca] == cb:
            flags.append(Flag.STANDARD_OPPOSITE_CENTERS_ADJACENT)
            reasons.append(f"adjacent faces {a}={ca} and {b}={cb} carry standard opposite colors")
    if not flags:
        flags.append(Flag.STANDARD_CHIRALITY_INVALID)
        shown = " ".join(f"{f}={c}" for f, c in centers.items())
        reasons.append(f"centers {shown} are a mirror image of the standard scheme")
    return [], sorted(set(flags)), reasons
