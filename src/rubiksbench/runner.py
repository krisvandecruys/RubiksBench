"""Run a benchmark: generate serially, evaluate every image, score per model."""

import random
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import EVALUATOR_VERSION, __version__
from .backends import GenerationRequest, get_backend
from .backends.base import utc_now
from .config import BenchmarkConfig
from .evaluation.evaluator import evaluate_extraction
from .evaluation.report import write_results
from .evaluation.scoring import score_run
from .storage.results import RunStore
from .vision.extractor import Extractor


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _seeds(config: BenchmarkConfig) -> list[int]:
    if config.seed is not None:
        return [config.seed + i for i in range(config.iterations)]
    rng = random.SystemRandom()
    return [rng.randrange(1, 2**31) for _ in range(config.iterations)]


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


async def generate_run(config: BenchmarkConfig, store: RunStore, extractor: Extractor) -> None:
    """Generate every image one request at a time, model by model, scoring each as it arrives."""
    store.write_config(
        {
            "benchmark_version": __version__,
            "evaluator_version": EVALUATOR_VERSION,
            "started_at": datetime.now(UTC).isoformat(),
            "config": config.as_dict(),
        }
    )
    async with get_backend(config.backend) as backend:
        for name, model in config.models.items():
            for index, seed in enumerate(_seeds(config)):
                request = GenerationRequest(
                    model=model,
                    prompt=config.prompt,
                    seed=seed,
                    width=config.width,
                    height=config.height,
                    negative_prompt=config.negative_prompt,
                    steps=config.steps,
                    guidance=config.guidance,
                    extra=config.extra,
                )
                result = await backend.generate(request)
                sample_id = f"{_slug(name)}__{index:04d}"
                record = store.add_generation(sample_id, result, model_name=name, iteration=index)
                if result.ok:
                    evaluation = evaluate_sample(store, record, extractor)
                    flags = ", ".join(evaluation.get("flags", []))
                    status = evaluation["verdict"] + (f" ({flags})" if flags else "")
                else:
                    status = f"GENERATION FAILED {result.error.code}: {result.error.message}"
                _log(f"[{name}] {index + 1}/{config.iterations} seed={seed} {result.latency_s:.1f}s {status}")


async def backfill_run(store: RunStore, extractor: Extractor) -> int:
    """Resend every failed request of a run with its original settings, and score the results.

    The new record replaces the failure for scoring but keeps its own timestamp and
    a ``backfill_of`` entry with the original attempt, so nothing is hidden.
    """
    run = store.read_config()
    failed = [g for g in store.generations() if g.get("error")]
    if not failed:
        return 0
    async with get_backend(run["config"]["backend"]) as backend:
        for record in failed:
            request = GenerationRequest(**record["request"])
            result = await backend.generate(request)
            new = store.add_generation(
                record["sample_id"], result, model_name=record["model_name"], iteration=record["iteration"],
                backfill_of={"started_at": record["started_at"], "error": record["error"]},
            )
            if result.ok:
                evaluation = evaluate_sample(store, new, extractor)
                status = evaluation["verdict"]
            else:
                status = f"GENERATION FAILED {result.error.code}: {result.error.message}"
            _log(f"[{record['model_name']}] backfill #{record['iteration']} seed={request.seed} "
                 f"{result.latency_s:.1f}s {status}")
    return len(failed)


def evaluate_sample(store: RunStore, record: dict[str, Any], extractor: Extractor) -> dict[str, Any]:
    evaluation = evaluate_extraction(extractor.extract(store.root / record["image_path"]))
    store.add_evaluation(record["sample_id"], evaluation)
    return evaluation


REFERENCE_MODEL = "Reference photos (real cubes)"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def import_references(store: RunStore, directory: Path, extractor: Extractor) -> int:
    """Copy real cube photos into the run and score them like generated images.

    They are kept apart from the generation log and shown as a calibration row:
    every real cube should score full points.
    """
    target = store.root / "references"
    target.mkdir(exist_ok=True)
    photos = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in _IMAGE_SUFFIXES and ".extraction." not in p.name
    )
    records = []
    for index, photo in enumerate(photos):
        shutil.copyfile(photo, target / photo.name)
        if (annotation := photo.with_suffix(".json")).exists():  # for the annotation extractor
            shutil.copyfile(annotation, target / annotation.name)
        records.append({
            "sample_id": f"reference__{photo.stem}",
            "model_name": REFERENCE_MODEL,
            "iteration": index,
            "reference": True,
            "request": {"model": "real photo", "prompt": None, "seed": None},
            "seed_supported": False,
            "image_path": f"references/{photo.name}",
            "error": None,
            "cost": None,
            "latency_s": 0.0,
            "started_at": utc_now(),
        })
    store.set_references(records)
    for record in records:
        evaluate_sample(store, record, extractor)
    return len(records)


def report_run(store: RunStore) -> tuple[dict[str, Any], dict[str, Path]]:
    """Score the evaluations already stored for a run; write timestamped result files."""
    samples, evaluations = store.references() + store.generations(), store.evaluations()
    scores = score_run(samples, evaluations)
    return scores, write_results(store.root, store.read_config(), samples, evaluations, scores)


def evaluate_run(
    store: RunStore, extractor: Extractor, references: Path | None = None
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Re-evaluate every image of a run (e.g. with a newer extractor), then report.

    ``references`` (re)imports a directory of real cube photos; otherwise the
    run's existing reference photos are re-evaluated.
    """
    store.clear_evaluations()
    if references is not None:
        import_references(store, references, extractor)
    for record in store.references() + store.generations():
        if record.get("image_path") and not (references is not None and record.get("reference")):
            evaluate_sample(store, record, extractor)
    return report_run(store)


def new_run_dir(out: Path, config_path: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return out / f"{stamp}-{_slug(config_path.stem)}"
