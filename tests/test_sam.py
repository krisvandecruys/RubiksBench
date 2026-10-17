"""SAM 2.1 extractor; skipped unless the optional dependencies are installed (`uv sync --extra ml`)."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from rubiksbench.cube.observation import Observation  # noqa: E402
from rubiksbench.vision.sam_extractor import SamExtractor  # noqa: E402

from .test_vision import FIXTURES, LABELLED  # noqa: E402


@pytest.fixture(scope="module")
def extractor():
    return SamExtractor(write_overlay=False)


@pytest.mark.parametrize("name", ["qwen-image-3-0__0000", "qwen-image-3-0-pro__0000"])
def test_every_sticker_read_matches_the_hand_label(extractor, name):
    extraction = extractor.extract(FIXTURES / f"{name}.jpg")
    expected = Observation.from_faces(LABELLED[name][0]).stickers
    read = [(got, want) for got, want in zip(extraction.observation.stickers, expected) if got is not None]
    assert len(read) >= 24 and all(got == want for got, want in read)


def test_box_shaped_front_face_is_seen_as_3_rows_by_4_columns(extractor):
    extraction = extractor.extract(FIXTURES / "qwen-image-3-0-pro__0001.jpg")
    assert tuple(extraction.details["bins_per_face"]["F"]) == (3, 4)
