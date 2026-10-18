"""Validity: how close an image comes to a valid, scrambled, standard 3x3 cube, from 0% to 100%.

    validity = (1 - (repairs + defects) / chance) * scramble factor, clipped to [0, 1]

``repairs`` is the fewest observed stickers to recolor for a valid standard cube
(``cube.repair``), ``defects`` the stickers no recoloring fixes (letters, two
colors on one sticker, a second shade of a color, a cubie out of place) and
``chance`` the repair distance of random colors. So
100% is a valid cube and 0% no better than coloring the stickers at random.
The scramble factor holds a barely scrambled cube or a designed pattern in slightly
less regard: from 100% for a random-looking scramble down to 80%. An image that is
not read as a 3x3 cube, or not scrambled at all (every face one color), scores 0%.

The rubric explains what is wrong; this measures how much.
"""

from typing import Any

from ..cube.repair import CHANCE_DISTANCE, repair
from ..vision.extraction import Extraction
from .rubric import FAIL, PASS, READ_AS_CUBE, scramble

# How much a barely scrambled cube (scramble score 0) loses against a random-looking one.
SCRAMBLE_WEIGHT = 0.2


def validity(extraction: Extraction, checks: dict[str, str]) -> dict[str, Any]:
    """Score one image from its extraction and its rubric results."""
    if not all(checks.get(c) == PASS for c in READ_AS_CUBE) or extraction.observation is None:
        return {"score": 0.0, "reason": "not read as a 3x3 cube"}
    mixed = scramble(extraction.observation)
    if mixed["of"] and not mixed["off_center"]:
        return {"score": 0.0, "reason": "not scrambled"}
    factor = 1 - SCRAMBLE_WEIGHT * (1 - mixed["score"])
    geometry = extraction.details.get("geometry", {})
    defects = sorted(
        set(geometry.get("marked", []) if checks.get("plain_stickers") == FAIL else [])
        | set(geometry.get("split", []) if checks.get("single_color") == FAIL else [])
        | set(geometry.get("off_shade", []) if checks.get("one_shade") == FAIL else [])
        | set(geometry.get("out_of_place", []) if checks.get("cubies_in_place") == FAIL else [])
    )
    nearest = repair(extraction.observation)
    cost = nearest.distance + len(defects)
    out: dict[str, Any] = {
        "score": max(0.0, 1 - cost / CHANCE_DISTANCE) * factor,
        "repairs": nearest.distance,
        "defects": len(defects),
        "chance": CHANCE_DISTANCE,
        "scramble": mixed["score"],
        "scramble_factor": round(factor, 3),
        "fixes": [f.to_dict() for f in nearest.fixes],
        "defective": defects,
    }
    return out

