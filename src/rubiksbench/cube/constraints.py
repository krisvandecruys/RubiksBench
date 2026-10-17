"""Color-level constraints: color count, centers, and candidate color schemes."""

import itertools
from collections import Counter
from collections.abc import Iterator

from ..evaluation.taxonomy import Flag
from .color_scheme import STANDARD_COLORS, STANDARD_OPPOSITE_COLOR, STANDARD_ORIENTATIONS, ColorScheme
from .facelets import FACES, OPPOSITE_FACE
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
