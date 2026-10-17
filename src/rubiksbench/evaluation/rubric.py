"""Graded rubric: one point per check passed, half a point when inconclusive.

Checks are ordered and gated: a check whose prerequisites did not all pass is
"blocked" and scores 0. A box-shaped cube therefore stops at the 3x3 check
instead of being tested for solvability.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Any

from ..cube.color_scheme import STANDARD_OPPOSITE_COLOR
from ..cube.constraints import observed_centers, standard_schemes
from ..cube.facelets import CORNER_FACELETS, CORNER_NAMES, EDGE_FACELETS, EDGE_NAMES, FACES
from ..cube.observation import Observation
from ..vision.extraction import Extraction

PASS, FAIL, INCONCLUSIVE, BLOCKED = "pass", "fail", "inconclusive", "blocked"
POINTS = {PASS: 1.0, INCONCLUSIVE: 0.5, FAIL: 0.0, BLOCKED: 0.0}


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    description: str
    requires: tuple[str, ...] = ()


CHECKS: tuple[Check, ...] = (
    Check("cube_detected", "Cube detected", "A cube-shaped object is found in the image."),
    Check("cube_shape", "Cube shape", "Three faces form a consistent corner view of a cube.", ("cube_detected",)),
    Check("grid_3x3", "3x3 grid", "Every visible face has 3 columns and 3 rows of cubies.", ("cube_shape",)),
    Check(
        "plain_stickers", "Plain stickers",
        "Stickers carry no letters or other markings (a logo on a center is allowed).", ("grid_3x3",),
    ),
    Check(
        "six_colors", "At most six colors",
        "No more than six distinct sticker colors. Fewer is allowed: every sticker of a color can sit on the "
        "three hidden faces, and the solver checks later that the missing colors really fit there.",
        ("grid_3x3",),
    ),
    Check(
        "standard_palette", "Standard palette",
        "Every visible color is white, yellow, red, orange, green or blue.", ("six_colors",),
    ),
    Check(
        "standard_pairs", "Standard color pairs",
        "No visible corner or edge combines colors that sit on opposite sides of a standard cube "
        "(white-yellow, red-orange, blue-green).",
        ("standard_palette",),
    ),
    Check("scrambled", "Scrambled", "Not every visible face is a single color.", ("grid_3x3",)),
    Check("distinct_centers", "Distinct centers", "No two visible centers share a color.", ("six_colors",)),
    Check(
        "center_order", "Standard center order",
        "Centers fit the standard scheme: no opposite colors side by side, correct handedness.",
        ("standard_palette", "distinct_centers"),
    ),
    Check(
        "possible_pieces", "Possible pieces",
        "Every visible corner and edge is a real cubie under some color scheme.", ("distinct_centers",),
    ),
    Check("unique_pieces", "Unique pieces", "No cubie is needed twice.", ("possible_pieces",)),
    Check("solvable", "Solvable", "A legal, reachable cube state matches everything visible.", ("unique_pieces",)),
    Check(
        "solvable_standard", "Solvable (standard)",
        "A legal state matches using the standard color scheme.", ("solvable", "standard_pairs", "center_order"),
    ),
)
MAX_POINTS = float(len(CHECKS))


def _scrambled(observation: Observation) -> str:
    for f in range(len(FACES)):
        seen = {s for s in observation.stickers[9 * f : 9 * f + 9] if s is not None}
        if len(seen) > 1:
            return PASS
    return FAIL


def _opposite_pairs(observation: Observation) -> list[str]:
    """Visible pieces that carry two standard-opposite colors, e.g. ["edge UF: R-O"]."""
    found = []
    for kind, names, slots in (("corner", CORNER_NAMES, CORNER_FACELETS), ("edge", EDGE_NAMES, EDGE_FACELETS)):
        for name, facelets in zip(names, slots):
            colors = [observation.stickers[i] for i in facelets if observation.stickers[i] is not None]
            for a, b in combinations(colors, 2):
                if STANDARD_OPPOSITE_COLOR.get(a) == b:
                    found.append(f"{kind} {name}: {a}-{b}")
    return found


def _outcomes(extraction: Extraction, validation: dict[str, Any] | None) -> dict[str, str]:
    """Each check's own outcome, before gating."""
    q = extraction.quality
    obs = extraction.observation
    out = {
        "cube_detected": q.get("cube", INCONCLUSIVE),
        "cube_shape": q.get("shape", INCONCLUSIVE),
        "grid_3x3": q.get("grid", INCONCLUSIVE),
        "plain_stickers": q.get("stickers", INCONCLUSIVE),
    }
    if obs is None:
        return out
    colors_known = q.get("colors") == PASS
    out["six_colors"] = (FAIL if len(obs.labels) > 6 else PASS) if colors_known else INCONCLUSIVE
    out["standard_palette"] = PASS if obs.labels <= set("WYROGB") else FAIL
    out["standard_pairs"] = FAIL if _opposite_pairs(obs) else PASS
    out["scrambled"] = _scrambled(obs)
    centers = list(observed_centers(obs).values())
    out["distinct_centers"] = PASS if len(centers) == len(set(centers)) else FAIL
    out["center_order"] = PASS if standard_schemes(obs)[0] else FAIL
    if validation is not None:
        stage = validation["abstract"]["furthest_stage"]
        out["possible_pieces"] = PASS if stage >= 1 else FAIL
        out["unique_pieces"] = PASS if stage >= 2 else FAIL
        out["solvable"] = PASS if stage >= 3 else FAIL
        out["solvable_standard"] = PASS if validation["standard"]["valid"] else FAIL
    return out


def grade(extraction: Extraction, validation: dict[str, Any] | None) -> dict[str, Any]:
    """Per-check results and total points for one image."""
    outcomes = _outcomes(extraction, validation)
    results: dict[str, str] = {}
    for check in CHECKS:
        if any(results.get(r) != PASS for r in check.requires):
            results[check.id] = BLOCKED
        else:
            results[check.id] = outcomes.get(check.id, INCONCLUSIVE)
    points = sum(POINTS[r] for r in results.values())
    graded: dict[str, Any] = {"checks": results, "points": points, "max_points": MAX_POINTS}
    if (obs := extraction.observation) is not None:
        graded["visible_colors"] = len(obs.labels)
        # Standard colors not seen at all: legal only if they all fit on the hidden faces.
        if obs.labels <= set("WYROGB") and (missing := sorted(set("WYROGB") - obs.labels)):
            graded["hidden_colors"] = missing
        if pairs := _opposite_pairs(obs):
            graded["opposite_pairs"] = pairs
    return graded

