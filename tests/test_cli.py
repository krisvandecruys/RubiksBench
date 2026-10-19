import json

import pytest

from rubiksbench import cli
from rubiksbench.vision.extraction import Extraction
from rubiksbench.vision.extractor import AnnotationExtractor

from .test_validator import REAL_CUBE


def test_validate_image_scores_annotation(tmp_path, capsys):
    image = tmp_path / "cube.png"
    image.touch()
    image.with_suffix(".json").write_text(json.dumps(REAL_CUBE))

    assert cli.main(["validate-image", str(image), "--extractor", "annotation", "--no-scramble"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["extractor"] == "annotation"
    assert report["verdict"] == "VALID_STANDARD"
    assert "rubric" in report and "validity" in report


@pytest.mark.parametrize("no_scramble", [False, True])
def test_validate_image_defaults_to_vision_and_forwards_scramble(tmp_path, monkeypatch, capsys, no_scramble):
    image = tmp_path / "cube.png"
    image.touch()
    extraction = Extraction("vision", None)

    class FakeVision:
        def extract(self, path):
            assert path == image
            return extraction

    def evaluate(value, *, with_scramble):
        assert value is extraction
        assert with_scramble is not no_scramble
        return {"verdict": "INCONCLUSIVE"}

    monkeypatch.setitem(cli.EXTRACTORS, "vision", FakeVision)
    monkeypatch.setattr(cli, "evaluate_extraction", evaluate)
    args = ["validate-image", str(image)] + (["--no-scramble"] if no_scramble else [])
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "INCONCLUSIVE"


def test_validate_image_without_extraction_is_inconclusive(tmp_path, monkeypatch, capsys):
    image = tmp_path / "cube.png"
    image.touch()
    monkeypatch.setitem(cli.EXTRACTORS, "vision", AnnotationExtractor)
    assert cli.main(["validate-image", str(image)]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "INCONCLUSIVE"


@pytest.mark.parametrize("name", ["missing.png", "."])
def test_validate_image_rejects_missing_files_and_directories(tmp_path, capsys, name):
    assert cli.main(["validate-image", str(tmp_path / name)]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "not a regular file" in captured.err
