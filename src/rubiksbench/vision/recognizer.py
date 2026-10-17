"""Is there a Rubik's cube in the image at all? Zero-shot CLIP (PyTorch, via transformers).

CLIP scores the image against "a Rubik's cube" and a few everyday alternatives
(a plain box, a die, an animal, ...). Needs the optional ``ml`` dependencies;
``available()`` says whether they are installed.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "openai/clip-vit-base-patch32"
RUBIKS = "a Rubik's cube"
LABELS = (
    RUBIKS, "a cube-shaped box", "a die", "an animal", "a person", "a household object", "food",
    "a building or landscape", "a toy",
)
# Below this probability, with another label on top, the image is judged not to show a cube.
# Cube renders scored 0.81-0.96 when testing; a rabbit, die, gift box and teapot <= 0.006.
MAX_NOT_A_CUBE = 0.05


@dataclass(frozen=True)
class Recognition:
    result: str  # "pass", "fail" or "inconclusive"
    rubiks_probability: float
    top_label: str
    scores: dict[str, float]


def available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


class CubeRecognizer:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._classifier = None

    def recognize(self, image_path: Path) -> Recognition:
        if self._classifier is None:
            from transformers import pipeline

            self._classifier = pipeline("zero-shot-image-classification", model=self.model_id, device="cpu")
        from PIL import Image

        ranked = self._classifier(
            Image.open(image_path).convert("RGB"), candidate_labels=list(LABELS), hypothesis_template="a photo of {}."
        )
        scores = {r["label"]: round(float(r["score"]), 4) for r in ranked}
        top = ranked[0]["label"]
        p = scores[RUBIKS]
        if top == RUBIKS:
            result = "pass"
        elif p < MAX_NOT_A_CUBE:
            result = "fail"
        else:
            result = "inconclusive"
        return Recognition(result, p, top, scores)
