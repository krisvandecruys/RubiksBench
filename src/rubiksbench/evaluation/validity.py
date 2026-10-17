"""Validity: how close an image comes to a valid, scrambled, standard 3x3 cube, from 0% to 100%.

    validity = 1 - (repairs + defects) / chance, clipped to [0, 1]

``repairs`` is the fewest observed stickers to recolor for a valid standard cube
(``cube.repair``), ``defects`` the stickers no recoloring fixes (letters, two
colors on one sticker) and ``chance`` the repair distance of random colors. So
100% is a valid cube and 0% no better than coloring the stickers at random.
An image that is not read as a 3x3 cube, or is not scrambled, scores 0%.

The rubric explains what is wrong; this measures how much.
"""

from typing import Any

from ..cube.repair import CHANCE_DISTANCE, repair
from ..vision.extraction import Extraction
from .rubric import FAIL, INCONCLUSIVE, PASS, READ_AS_CUBE


def validity(extraction: Extraction, checks: dict[str, str]) -> dict[str, Any]:
    """Score one image from its extraction and its rubric results."""
    if not all(checks.get(c) == PASS for c in READ_AS_CUBE) or extraction.observation is None:
        return {"score": 0.0, "reason": "not read as a 3x3 cube"}
    if checks.get("scrambled") == FAIL:
        return {"score": 0.0, "reason": "not scrambled"}
    geometry = extraction.details.get("geometry", {})
    defects = sorted(
        set(geometry.get("marked", []) if checks.get("plain_stickers") == FAIL else [])
        | set(geometry.get("split", []) if checks.get("single_color") == FAIL else [])
    )
    nearest = repair(extraction.observation)
    cost = nearest.distance + len(defects)
    out: dict[str, Any] = {
        "score": max(0.0, 1 - cost / CHANCE_DISTANCE),
        "repairs": nearest.distance,
        "defects": len(defects),
        "chance": CHANCE_DISTANCE,
        "fixes": [f.to_dict() for f in nearest.fixes],
        "defective": defects,
    }
    if checks.get("six_colors") == INCONCLUSIVE:
        out["uncertain"] = "some colors were read with low confidence"
    return out

