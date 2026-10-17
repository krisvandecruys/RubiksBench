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


class Extractor(Protocol):
    name: str

    def extract(self, image_path: Path) -> Extraction: ...
