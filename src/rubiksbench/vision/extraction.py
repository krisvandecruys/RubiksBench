"""What an extractor returns, shared by all extractors."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..cube.observation import Observation
from ..evaluation.taxonomy import Flag

PASS, FAIL, INCONCLUSIVE = "pass", "fail", "inconclusive"


@dataclass
class Extraction:
    extractor: str
    # None when nothing could be read; the sample is then INCONCLUSIVE.
    observation: Observation | None
    confidence: float | None = None
    # Geometry / confidence flags, e.g. wrong_grid_size or vision_low_confidence.
    flags: list[Flag] = field(default_factory=list)
    # Raw extractor output, kept so failures can be inspected later.
    details: dict[str, Any] = field(default_factory=dict)
    # What the extractor established about the image, per aspect ("cube", "shape",
    # "grid", "colors"): "pass", "fail" or "inconclusive". Feeds the rubric.
    quality: dict[str, str] = field(default_factory=dict)


def geometry(
    size: tuple[int, int],
    faces: dict[str, Any],
    stickers: dict[str, Any],
    grids: dict[str, tuple[int, int]],
    marked: list[str],
    split: list[str],
) -> dict[str, Any]:
    """Where things are in the image, so a report can outline them (``details["geometry"]``).

    ``size`` is (width, height); ``faces`` maps a face to its outline and ``stickers`` a
    facelet name ("U1") to its quad, both as [[x, y], ...] in pixels; ``grids`` is each
    face's measured (columns, rows); ``marked`` names the stickers carrying letters and
    ``split`` those showing two colors.
    """
    def points(quad: Any) -> list[list[float]]:
        return [[round(float(x), 1), round(float(y), 1)] for x, y in quad]

    return {
        "size": list(size),
        "faces": {f: points(q) for f, q in faces.items()},
        "stickers": {name: points(q) for name, q in stickers.items()},
        "grids": {f: list(g) for f, g in grids.items()},
        "marked": marked,
        "split": split,
    }


class Extractor(Protocol):
    name: str

    def extract(self, image_path: Path) -> Extraction: ...
