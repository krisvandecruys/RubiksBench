"""Graded rubric: one point per check passed, half a point when inconclusive.

Every mistake costs its point once. A check is "blocked" (scoring 0, but not
counted as a mistake of its own) when it cannot be decided, or when its failure
would only be a consequence of an earlier failed check:

* sticker checks need the image read as a 3x3 cube (a box-shaped cube stops at
  the 3x3 check), color checks also need confidently named colors;
* a duplicate center already rules out the standard center order, a seventh
  color a standard palette, an impossible piece a solvable cube, and so on.

Independent mistakes still count, whatever failed before: a corner showing one
color twice fails "possible pieces" even next to two white centers, and opposite
colors on side-by-side centers fail "center order" even when another center is
duplicated. So the same kind of error is always reported the same way.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Any

from ..cube.color_scheme import STANDARD_OPPOSITE_COLOR, STANDARD_ORIENTATIONS
from ..cube.constraints import observed_centers, scheme_free_piece_failures
from ..cube.facelets import (
    CORNER_FACELETS, CORNER_NAMES, EDGE_FACELETS, EDGE_NAMES, FACES, OPPOSITE_FACE, center_of, facelet_name,
)
from ..cube.observation import Observation
from ..vision.extraction import Extraction
from .taxonomy import Flag

PASS, FAIL, INCONCLUSIVE, BLOCKED = "pass", "fail", "inconclusive", "blocked"
POINTS = {PASS: 1.0, INCONCLUSIVE: 0.5, FAIL: 0.0, BLOCKED: 0.0}
STANDARD = frozenset("WYROGB")


# Who would notice a failed check, from anyone at a glance to only a solver. An image fools
# everyone up to the first tier that catches one of its failures.
GLANCE, HELD, CUBER, SOLVER = range(4)
FOOLS = ("no one", "a casual viewer", "a cube owner", "a cuber", "everyone")
NOTICED_BY = ("anyone, at a glance", "anyone who has held a cube", "a cuber looking carefully", "only a solver")


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    description: str
    # What must have been read for the check to be decided; it is blocked otherwise.
    needs: str = ""
    # Who would notice the check failing (GLANCE ... SOLVER).
    noticed_by: int = GLANCE


CUBE, COLORS = "the image read as a 3x3 cube", "a 3x3 cube with confidently named colors"
CHECKS: tuple[Check, ...] = (
    Check("cube_detected", "Cube detected", "A cube-shaped object is found in the image."),
    Check("cube_shape", "Cube shape", "Three faces form a consistent corner view of a cube.", "a cube outline"),
    Check("grid_3x3", "3x3 grid", "Every visible face has 3 columns and 3 rows of cubies.", "three faces"),
    Check("plain_stickers", "Plain stickers",
          "Stickers carry no letters or other markings (a logo on a center is allowed).", CUBE),
    Check("single_color", "Single-colored stickers",
          "No sticker shows two clearly different colors (a logo on a center is allowed; a split between "
          "neighbouring hues such as red and orange is not detected).", CUBE),
    Check("six_colors", "At most six colors",
          "No more than six distinct sticker colors. Fewer is allowed: every sticker of a color can sit on the "
          "three hidden faces, and the solver checks that the missing colors really fit there.", CUBE),
    Check("one_shade", "One shade per color",
          "On each face, all stickers of one color share a shade (no lime next to green). Real cubes stay "
          "within 3 degrees of hue; a sticker more than 15 degrees off is a second shade.", COLORS, noticed_by=HELD),
    Check("standard_palette", "Standard palette",
          "Every visible color is white, yellow, red, orange, green or blue.", COLORS),
    Check("standard_pairs", "Standard color pairs",
          "No visible corner or edge combines colors that sit on opposite sides of a standard cube "
          "(white-yellow, red-orange, blue-green).", COLORS, noticed_by=CUBER),
    Check("scrambled", "Scrambled",
          "Looks like a random scramble: at least 15 of the 24 visible stickers differ from their face's center, "
          "and at most one face shows a symmetric pattern. Fewer than 1% of random scrambles fail this; a cube "
          "one move from solved, or a designed pattern, does.", COLORS),
    Check("distinct_centers", "Distinct centers", "No two visible centers share a color.", COLORS, noticed_by=HELD),
    Check("center_order", "Standard center order",
          "Centers fit the standard scheme: standard colors, no opposite colors side by side, correct handedness.",
          COLORS, noticed_by=CUBER),
    Check("possible_pieces", "Possible pieces",
          "Every visible corner and edge is a real cubie under some color scheme. Without a scheme (e.g. two "
          "centers share a color) only pieces impossible under every scheme, such as one color twice, are known.",
          COLORS, noticed_by=CUBER),
    Check("unique_pieces", "Unique pieces",
          "No cubie is needed twice. Decided once every piece is possible, or when two pieces show the same colors.",
          COLORS, noticed_by=CUBER),
    Check("solvable", "Solvable", "A legal, reachable cube state matches everything visible.", COLORS, noticed_by=SOLVER),
    Check("solvable_standard", "Solvable (standard)",
          "A legal state matches using the standard color scheme.", COLORS, noticed_by=SOLVER),
)
MAX_POINTS = float(len(CHECKS))
# Until these pass the stickers are not read as a cube, so sticker checks cannot be decided.
READ_AS_CUBE = ("cube_detected", "cube_shape", "grid_3x3")


# A face pattern equal to its mirror image (either axis or diagonal) or its half turn.
_SYMMETRIES = (
    (2, 1, 0, 5, 4, 3, 8, 7, 6), (6, 7, 8, 3, 4, 5, 0, 1, 2), (0, 3, 6, 1, 4, 7, 2, 5, 8),
    (8, 5, 2, 7, 4, 1, 6, 3, 0), (8, 7, 6, 5, 4, 3, 2, 1, 0),
)
# Of 3,000 random scrambles, 99% show at least 15 of the 24 visible non-center stickers off
# their face's center color (median 20) and 99.9% at most one symmetric face. A one-move
# scramble shows 6 off-center stickers and symmetric faces.
MIN_OFF_CENTER = 15 / 24
MAX_SYMMETRIC_FACES = 1


def scramble(observation: Observation) -> dict[str, Any]:
    """How scrambled the visible faces look: stickers off their face's center color, faces
    whose pattern is symmetric, and a score from 0 (barely scrambled, or a designed pattern)
    to 1 (looks like a random scramble)."""
    off = seen = 0
    symmetric = []
    for f, face in enumerate(FACES):
        grid = observation.stickers[9 * f : 9 * f + 9]
        if grid[4] is None:
            continue
        shown = [s for i, s in enumerate(grid) if i != 4 and s is not None]
        seen += len(shown)
        off += sum(s != grid[4] for s in shown)
        if None not in grid and any(all(grid[i] == grid[m[i]] for i in range(9)) for m in _SYMMETRIES):
            symmetric.append(face)
    # Enough stickers off-center (as in 99% of random scrambles) counts fully; fewer count
    # proportionally. Each symmetric face beyond the one a random scramble may show halves it.
    spread = min(1.0, off / (MIN_OFF_CENTER * seen)) if seen else 0.0
    extra = max(0, len(symmetric) - MAX_SYMMETRIC_FACES)
    return {"off_center": off, "of": seen, "symmetric_faces": symmetric, "score": round(spread * 0.5**extra, 3)}


def _scrambled(info: dict[str, Any]) -> str | None:
    if not info["of"]:
        return None
    random_like = info["off_center"] >= MIN_OFF_CENTER * info["of"]
    return PASS if random_like and len(info["symmetric_faces"]) <= MAX_SYMMETRIC_FACES else FAIL


def _opposite_pieces(observation: Observation) -> list[tuple[str, list[str]]]:
    """Visible pieces that carry two standard-opposite colors, as ("edge UF: R-O", ["U8", "F2"])."""
    found = []
    for kind, names, slots in (("corner", CORNER_NAMES, CORNER_FACELETS), ("edge", EDGE_NAMES, EDGE_FACELETS)):
        for name, facelets in zip(names, slots):
            seen = [i for i in facelets if observation.stickers[i] is not None]
            for a, b in combinations(seen, 2):
                ca, cb = observation.stickers[a], observation.stickers[b]
                if STANDARD_OPPOSITE_COLOR.get(ca) == cb:
                    found.append((f"{kind} {name}: {ca}-{cb}", [facelet_name(a), facelet_name(b)]))
    return found


def _opposite_pairs(observation: Observation) -> list[str]:
    """Visible pieces that carry two standard-opposite colors, e.g. ["edge UF: R-O"]."""
    return [label for label, _ in _opposite_pieces(observation)]


def _mark(check: str, label: str, facelets: list[str] = (), faces: list[str] = ()) -> dict[str, Any]:
    return {"check": check, "label": label, "facelets": list(facelets), "faces": list(faces)}


def _center_marks(observation: Observation) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Marks for duplicate centers, and for centers out of the standard order."""
    centers = observed_centers(observation)
    duplicates = [
        _mark("distinct_centers", f"duplicate center: {color}",
              [facelet_name(center_of(f)) for f in centers if centers[f] == color])
        for color in sorted(set(centers.values())) if list(centers.values()).count(color) > 1
    ]
    order = [
        _mark("center_order", f"opposite colors on adjacent centers {a}5 {centers[a]} and {b}5 {centers[b]}",
              [facelet_name(center_of(a)), facelet_name(center_of(b))])
        for a, b in _adjacent_opposite_centers(observation)
    ] or [_mark("center_order", "centers are a mirror image of the standard scheme",
                [facelet_name(center_of(f)) for f in centers])]
    return duplicates, order


def marks(extraction: Extraction, validation: dict[str, Any] | None, results: dict[str, str]) -> list[dict[str, Any]]:
    """What to outline on the image for each failed check: facelets (e.g. "U9") or whole faces.

    Only failed checks are marked, never blocked ones. Failures of the whole cube
    (twist, flip, parity) have no single culprit and get no mark.
    """
    failed = {c for c, r in results.items() if r == FAIL}
    geometry = extraction.details.get("geometry", {})
    out = []
    if "grid_3x3" in failed:
        out += [_mark("grid_3x3", f"{face}: {cols}x{rows} grid", faces=[face])
                for face, (cols, rows) in geometry.get("grids", {}).items() if (cols, rows) != (3, 3)]
    if "plain_stickers" in failed and (marked := geometry.get("marked")):
        out.append(_mark("plain_stickers", "letters or markings", marked))
    if "single_color" in failed:
        out += [_mark("single_color", f"two colors on sticker {name}", [name]) for name in geometry.get("split", [])]
    if "one_shade" in failed:
        out += [_mark("one_shade", f"a second shade of its color on sticker {name}", [name])
                for name in geometry.get("off_shade", [])]
    obs = extraction.observation
    if obs is None:
        return out
    if "scrambled" in failed:
        info = scramble(obs)
        if len(info["symmetric_faces"]) > MAX_SYMMETRIC_FACES:
            out += [_mark("scrambled", f"{face}: a symmetric pattern", faces=[face]) for face in info["symmetric_faces"]]
    if "standard_palette" in failed:
        odd = [i for i, s in enumerate(obs.stickers) if s is not None and s not in STANDARD]
        out.append(_mark("standard_palette", "non-standard color", [facelet_name(i) for i in odd]))
    if "standard_pairs" in failed:
        out += [_mark("standard_pairs", f"opposite colors on {label}", facelets)
                for label, facelets in _opposite_pieces(obs)]
    duplicates, order = _center_marks(obs)
    if "distinct_centers" in failed:
        out += duplicates
    if "center_order" in failed:
        out += order
    if validation is not None:
        culprits = {side: validation[side].get("culprits", []) for side in ("abstract", "standard")}
        for check, side, kind in (("possible_pieces", "abstract", "impossible"), ("unique_pieces", "abstract", "duplicate"),
                                  ("solvable_standard", "standard", "")):
            if check in failed:
                out += [_mark(check, c["reason"], c["facelets"]) for c in culprits[side] if c["reason"].startswith(kind)]
        if "unique_pieces" in failed:  # also when an impossible piece stopped the search first
            out += [_mark("unique_pieces", c.reason, [facelet_name(i) for i in c.facelets])
                    for flag, c in scheme_free_piece_failures(obs) if flag == Flag.DUPLICATE_PIECE]
    # One outline per set of stickers: a piece already marked (e.g. for opposite colors)
    # is not outlined again when it also breaks the standard scheme.
    seen: set[tuple[frozenset[str], frozenset[str]]] = set()
    unique = []
    for m in out:
        key = (frozenset(m["facelets"]), frozenset(m["faces"]))
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def _outcomes(extraction: Extraction, validation: dict[str, Any] | None) -> dict[str, str]:
    """Each check's outcome where it can be decided from what was read; missing means undecidable."""
    q = extraction.quality
    obs = extraction.observation
    out = {k: v for k, v in (("cube_detected", q.get("cube", INCONCLUSIVE)), ("cube_shape", q.get("shape")),
                             ("grid_3x3", q.get("grid"))) if v is not None}
    if not all(out.get(c) == PASS for c in READ_AS_CUBE):
        return out
    out["plain_stickers"] = q.get("stickers", INCONCLUSIVE)
    out["single_color"] = q.get("single_color", INCONCLUSIVE)
    if obs is None:
        return out
    if q.get("colors") != PASS:
        out["six_colors"] = INCONCLUSIVE
        return out
    out["six_colors"] = FAIL if len(obs.labels) > 6 else PASS
    out["one_shade"] = q.get("shades", INCONCLUSIVE)
    if len(obs.labels) <= 6:  # a seventh color cannot be a standard one: already counted
        out["standard_palette"] = PASS if obs.labels <= STANDARD else FAIL
    out["standard_pairs"] = FAIL if _opposite_pairs(obs) else PASS
    if (scrambled := _scrambled(scramble(obs))) is not None:
        out["scrambled"] = scrambled
    centers = observed_centers(obs)
    distinct = len(centers) == len(set(centers.values()))
    out["distinct_centers"] = PASS if distinct else FAIL
    if distinct and set(centers.values()) <= STANDARD:
        fits = any(all(s.color(f) == c for f, c in centers.items()) for s in STANDARD_ORIENTATIONS)
        out["center_order"] = PASS if fits else FAIL
    elif _adjacent_opposite_centers(obs):
        # Duplicate or non-standard centers rule the order out anyway; only a separate
        # mistake, opposite colors side by side, counts against it.
        out["center_order"] = FAIL
    if validation is None:
        return out
    abstract = validation["abstract"]
    # Piece failures under every color scheme; they decide the piece checks when the
    # completion search could not (no scheme fits the centers, or an impossible piece came first).
    unfixable = {flag for flag, _ in scheme_free_piece_failures(obs)}
    if abstract["schemes_tried"]:
        stage = abstract["furthest_stage"]
        out["possible_pieces"] = PASS if stage >= 1 else FAIL
        if stage >= 1:
            out["unique_pieces"] = PASS if stage >= 2 else FAIL
    elif {Flag.IMPOSSIBLE_CORNER, Flag.IMPOSSIBLE_EDGE} & unfixable:
        out["possible_pieces"] = FAIL
    if "unique_pieces" not in out and Flag.DUPLICATE_PIECE in unfixable:
        out["unique_pieces"] = FAIL
    # Unsolvable is only a mistake of its own (twist, flip, parity) once every piece is
    # possible and unique; otherwise it follows from the piece or center failures.
    if out.get("unique_pieces") == PASS:
        out["solvable"] = PASS if abstract["valid"] else FAIL
    if all(out.get(c) == PASS for c in ("solvable", "standard_palette", "standard_pairs", "center_order")):
        out["solvable_standard"] = PASS if validation["standard"]["valid"] else FAIL
    return out


def _adjacent_opposite_centers(observation: Observation) -> list[tuple[str, str]]:
    """Pairs of neighbouring faces whose centers carry standard-opposite colors."""
    centers = observed_centers(observation)
    return [(a, b) for a, b in combinations(centers, 2)
            if OPPOSITE_FACE[a] != b and STANDARD_OPPOSITE_COLOR.get(centers[a]) == centers[b]]


def fools(checks: dict[str, str], observation: Observation | None) -> int | None:
    """Who the image would fool, as an index into FOOLS: everyone below the first tier that
    notices one of its failures, or everyone when it passes every check. None when nothing
    failed but not everything could be checked."""
    tiers = {c.id: c.noticed_by for c in CHECKS}
    # A designed or shallow scramble is still a valid cube: nothing to see through.
    failed = [tiers[c] for c, r in checks.items() if r == FAIL and c != "scrambled"]
    # A piece showing one color twice is plain to anyone who has held a cube.
    if checks.get("possible_pieces") == FAIL and observation is not None and any(
        flag in (Flag.IMPOSSIBLE_CORNER, Flag.IMPOSSIBLE_EDGE) for flag, _ in scheme_free_piece_failures(observation)
    ):
        failed.append(HELD)
    if failed:
        return min(failed)
    return len(FOOLS) - 1 if all(r == PASS for c, r in checks.items() if c != "scrambled") else None


def grade(extraction: Extraction, validation: dict[str, Any] | None) -> dict[str, Any]:
    """Per-check results and total points for one image."""
    outcomes = _outcomes(extraction, validation)
    results = {check.id: outcomes.get(check.id, BLOCKED) for check in CHECKS}
    points = sum(POINTS[r] for r in results.values())
    graded: dict[str, Any] = {"checks": results, "points": points, "max_points": MAX_POINTS,
                              "fools": fools(results, extraction.observation)}
    if found := marks(extraction, validation, results):
        graded["marks"] = found
    if (obs := extraction.observation) is not None:
        graded["visible_colors"] = len(obs.labels)
        # Standard colors not seen at all: legal only if they all fit on the hidden faces.
        if obs.labels <= set("WYROGB") and (missing := sorted(set("WYROGB") - obs.labels)):
            graded["hidden_colors"] = missing
        if pairs := _opposite_pairs(obs):
            graded["opposite_pairs"] = pairs
        graded["scramble"] = scramble(obs)
    return graded

