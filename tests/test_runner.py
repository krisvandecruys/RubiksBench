import asyncio
import json
from pathlib import Path

import pytest

from rubiksbench.backends.base import (
    GenerationError,
    GenerationRequest,
    GenerationResult,
    ImageBackend,
    register_backend,
    utc_now,
)
from rubiksbench.backends import available_backends
from rubiksbench.config import DEFAULT_PROMPT, BenchmarkConfig
from rubiksbench.runner import REFERENCE_MODEL, backfill_run, evaluate_run, generate_run, report_run
from rubiksbench.storage.results import RunStore
from rubiksbench.vision.extractor import AnnotationExtractor

from .test_validator import REAL_CUBE

REPO = Path(__file__).parent.parent


@register_backend
class FakeBackend(ImageBackend):
    name = "fake"
    requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        result = GenerationResult(request, self.name, utc_now(), 0.1, cost=0.01)
        if request.model == "broken":
            result.error = GenerationError("provider", "model unavailable")
        else:
            result.image, result.image_format = b"img", "png"
        return result


def write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


@pytest.mark.parametrize("path", sorted((REPO / "configs").glob("*.toml")), ids=lambda p: p.name)
def test_shipped_configs_load(path):
    config = BenchmarkConfig.load(path)
    assert config.backend in available_backends() and config.iterations >= 1 and config.models


def test_defaults(tmp_path):
    config = BenchmarkConfig.load(write(tmp_path / "b.toml", '[models]\na = "x"\n'))
    assert config.iterations == 20 and config.prompt == DEFAULT_PROMPT and config.seed is None
    assert config.extractor == "vision"


def test_extractor_setting(tmp_path):
    config = BenchmarkConfig.load(write(tmp_path / "b.toml", 'extractor = "sam"\n[models]\na = "x"\n'))
    assert config.extractor == "sam"
    with pytest.raises(ValueError, match="extractor"):
        BenchmarkConfig.load(write(tmp_path / "b.toml", 'extractor = "yolo"\n[models]\na = "x"\n'))


def test_config_rejects_unknown_settings(tmp_path):
    with pytest.raises(ValueError, match="iteration"):
        BenchmarkConfig.load(write(tmp_path / "b.toml", 'iteration = 3\n[models]\na = "x"\n'))
    with pytest.raises(ValueError, match="models"):
        BenchmarkConfig.load(write(tmp_path / "b.toml", "iterations = 3\n"))


def test_run_generates_serially_and_scores(tmp_path):
    config = BenchmarkConfig.load(
        write(tmp_path / "b.toml", 'backend = "fake"\niterations = 2\n[models]\n"Good" = "good"\n"Broken" = "broken"\n')
    )
    store = RunStore(tmp_path / "run")
    FakeBackend.requests.clear()
    # The first image is annotated before the run starts, so it is scored as soon as it is generated.
    write(store.images / "good__0000.json", json.dumps(REAL_CUBE))
    asyncio.run(generate_run(config, store, AnnotationExtractor()))
    evaluations = store.evaluations()
    assert evaluations["good__0000"]["verdict"] == "VALID_STANDARD"
    assert evaluations["good__0001"]["verdict"] == "INCONCLUSIVE"
    assert set(evaluations) == {"good__0000", "good__0001"}

    assert [r.model for r in FakeBackend.requests] == ["good", "good", "broken", "broken"]
    seeds = [r.seed for r in FakeBackend.requests]
    assert len(set(seeds)) == 4 and all(seeds)

    scores, paths = report_run(store)
    assert evaluate_run(store, AnnotationExtractor())[0] == scores  # re-scoring gives the same result
    good, broken = scores["Good"], scores["Broken"]
    assert good["verdicts"]["VALID_STANDARD"] == 1 and good["verdicts"]["INCONCLUSIVE"] == 1
    assert good["standard_valid_rate"] == 0.5 and good["total_cost"] == pytest.approx(0.02)
    assert broken["generation_failures"] == 2 and broken["images"] == 0
    assert broken["standard_valid_rate"] is None

    generations = store.generations()
    assert all("seed" in g["request"] for g in generations)
    assert json.loads(paths["json"].read_text())["scores"] == scores
    # One valid image (16 points) and one unannotated, so inconclusive at the first check (0.5).
    assert good["rubric_mean"] == pytest.approx(8.25)
    # Validity: the valid image is 100%, the unreadable one 0%.
    assert good["validity_mean"] == pytest.approx(0.5)
    assert (good["passed"], good["failed"], good["generation_failures"]) == (1, 1, 0)
    assert (broken["passed"], broken["failed"], broken["generation_failures"]) == (0, 0, 2)
    report = paths["html"].read_text()
    assert report.startswith("<!doctype html>") and "<b>50%</b>" in report
    assert "images/good__0000.png" in report and "unseeded" not in report


def test_backfill_resends_failures_and_keeps_the_history(tmp_path):
    config = BenchmarkConfig.load(
        write(tmp_path / "b.toml", 'backend = "fake"\niterations = 2\n[models]\n"Broken" = "broken"\n')
    )
    store = RunStore(tmp_path / "run")
    asyncio.run(generate_run(config, store, AnnotationExtractor()))
    first_seeds = [g["request"]["seed"] for g in store.generations()]

    FakeBackend.requests.clear()
    store.write_config(store.read_config() | {"config": config.as_dict() | {"models": {"Broken": "fixed"}}})
    records = store.generations()
    for g in records:  # the provider has since fixed the model
        g["request"]["model"] = "fixed"
    (store.root / "generations.jsonl").write_text("".join(json.dumps(g) + "\n" for g in records))

    assert asyncio.run(backfill_run(store, AnnotationExtractor())) == 2
    assert [r.seed for r in FakeBackend.requests] == first_seeds  # same settings, same seeds
    current = store.generations()
    assert len(current) == 2 and all(g.get("image_path") and "backfill_of" in g for g in current)
    assert len(store.generation_attempts()) == 4  # the failures stay on record
    assert asyncio.run(backfill_run(store, AnnotationExtractor())) == 0


def test_reference_photos_are_scored_as_a_separate_row(tmp_path):
    config = BenchmarkConfig.load(write(tmp_path / "b.toml", 'backend = "fake"\niterations = 1\n[models]\n"Good" = "good"\n'))
    store = RunStore(tmp_path / "run")
    asyncio.run(generate_run(config, store, AnnotationExtractor()))
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "reference_1.png").write_bytes(b"img")
    write(refs / "reference_1.json", json.dumps(REAL_CUBE))  # its hand annotation

    scores, paths = evaluate_run(store, AnnotationExtractor(), references=refs)
    ref = scores[REFERENCE_MODEL]
    assert ref["reference"] and ref["images"] == 1 and ref["passed"] == 1
    assert (store.root / "references" / "reference_1.png").exists()
    assert len(store.generations()) == 1  # the generation log is untouched
    html = paths["html"].read_text()
    assert "references/reference_1.png" in html and "real photo" in html


def test_site_has_sharing_tags_and_a_framed_preview(tmp_path):
    import cv2
    import numpy as np

    from rubiksbench.site import SHARE_SIZE, build_site

    config = BenchmarkConfig.load(write(tmp_path / "b.toml", 'backend = "fake"\niterations = 1\n[models]\n"Good" = "good"\n'))
    store = RunStore(tmp_path / "run")
    write(store.images / "good__0000.json", json.dumps(REAL_CUBE))
    asyncio.run(generate_run(config, store, AnnotationExtractor()))
    # A real picture: a colored square (the "cube") low on a gray background.
    picture = np.full((200, 200, 3), 127, np.uint8)
    picture[110:180, 70:140] = (40, 40, 220)
    cv2.imwrite(str(store.images / "good__0000.png"), picture)
    report_run(store)

    index = build_site(store.root, tmp_path / "site", url="https://example.org/bench/", title="RubiksBench",
                       description="How good are models at cubes?", share_sample="good__0000", share_alt="A cube")
    html = index.read_text()
    assert "<title>RubiksBench</title>" in html
    assert "<meta property='og:image' content='https://example.org/bench/share.jpg'>" in html
    assert "<meta name='twitter:card' content='summary_large_image'>" in html
    assert "images/good__0000.jpg" in html and (index.parent / "images" / "good__0000.jpg").exists()
    share = cv2.imread(str(index.parent / "share.jpg"))
    assert share.shape[1::-1] == SHARE_SIZE
    # The square is framed in the middle, not cut off at the bottom.
    red = np.argwhere(share[:, :, 2] > 180)
    assert 0.25 < red[:, 0].mean() / SHARE_SIZE[1] < 0.75
