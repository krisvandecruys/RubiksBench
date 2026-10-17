"""Benchmark configuration, loaded from a TOML file.

    prompt = "A valid scrambled Rubik's Cube"
    iterations = 20

    [models]
    "FLUX 3 Image" = "bfl:flux@3-image"

Only ``[models]`` is required. Without ``seed`` every image gets a fresh random
seed; with ``seed = N`` image i uses N + i. Seeds are always recorded.
"""

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = "A valid scrambled Rubik's Cube"


@dataclass(frozen=True)
class BenchmarkConfig:
    # Display name -> backend model id.
    models: dict[str, str]
    prompt: str = DEFAULT_PROMPT
    iterations: int = 20
    seed: int | None = None
    backend: str = "runware"
    # How stickers are read from each image: "vision", "sam" or "annotation".
    extractor: str = "vision"
    # Directory of real cube photos scored alongside every run as a calibration row.
    references: str | None = None
    width: int = 1024
    height: int = 1024
    negative_prompt: str | None = None
    steps: int | None = None
    guidance: float | None = None
    # Provider-specific parameters sent with every request.
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "BenchmarkConfig":
        with path.open("rb") as f:
            data = tomllib.load(f)
        known = {f.name for f in fields(cls)}
        if unknown := sorted(set(data) - known):
            raise ValueError(f"{path}: unknown setting(s) {unknown}; known: {sorted(known)}")
        if not data.get("models"):
            raise ValueError(f"{path}: needs a non-empty [models] table")
        if not all(isinstance(v, str) for v in data["models"].values()):
            raise ValueError(f'{path}: [models] entries must be name = "model id"')
        from .vision.extractor import EXTRACTORS

        if data.get("extractor", "vision") not in EXTRACTORS:
            raise ValueError(f"{path}: extractor must be one of {sorted(EXTRACTORS)}")
        if data.get("iterations", 1) < 1:
            raise ValueError(f"{path}: iterations must be at least 1")
        return cls(**data)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
