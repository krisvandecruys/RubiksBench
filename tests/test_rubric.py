from rubiksbench.cube.observation import Observation
from rubiksbench.evaluation.evaluator import validate
from rubiksbench.evaluation.rubric import MAX_POINTS, grade
from rubiksbench.vision.extraction import Extraction

from .test_validator import REAL_CUBE

QUALITY = {"cube": "pass", "shape": "pass", "grid": "pass", "stickers": "pass", "colors": "pass"}


def graded(faces):
    observation = Observation.from_faces(faces)
    extraction = Extraction("annotation", observation, quality=QUALITY)
    return grade(extraction, validate(observation, with_scramble=False).to_dict())


def test_real_cube_earns_every_point():
    result = graded(REAL_CUBE)
    assert result["points"] == MAX_POINTS == 14
    assert "opposite_pairs" not in result


def test_fewer_than_six_colors_is_allowed_and_recorded():
    # Five visible colors can be legal: every sticker of the sixth may be on a hidden face.
    five = {face: grid.replace("B", "G") for face, grid in REAL_CUBE.items()}
    result = graded(five)
    assert result["checks"]["six_colors"] == "pass" and result["checks"]["standard_palette"] == "pass"
    assert result["visible_colors"] == 5 and result["hidden_colors"] == ["B"]


def test_a_seventh_color_fails():
    seven = dict(REAL_CUBE, R="PRG WYG OYB")  # purple ("P") on the right face
    result = graded(seven)
    assert result["checks"]["six_colors"] == "fail" and result["visible_colors"] == 7


def test_red_orange_edge_fails_standard_pairs():
    # Make the U-F edge (U8, F2) red-orange: opposite colors on one piece.
    faces = dict(REAL_CUBE)
    faces["U"] = faces["U"].replace(" ", "")
    faces["U"] = faces["U"][:7] + "R" + faces["U"][8]
    faces["F"] = faces["F"].replace(" ", "")
    faces["F"] = faces["F"][0] + "O" + faces["F"][2:]
    result = graded(faces)
    assert result["checks"]["standard_pairs"] == "fail"
    assert "edge UF: R-O" in result["opposite_pairs"]
    assert result["checks"]["solvable_standard"] == "blocked"


def test_gallery_is_sorted_purely_by_score():
    from rubiksbench.evaluation.report import _gallery_order

    scores = {"A": {"rubric_mean": 9.0}, "B": {"rubric_mean": 5.0}}
    samples = [
        {"model_name": "A", "iteration": 0, "points": 6.0},
        {"model_name": "B", "iteration": 0, "points": 12.0},
        {"model_name": "A", "iteration": 1, "points": None},  # generation failed
        {"model_name": "B", "iteration": 1, "points": 6.0},
    ]
    order = [(s["model_name"], s["iteration"]) for s in _gallery_order(samples, scores)]
    # Model B's 12-pointer leads; the 6-point tie goes to the better-ranked model A.
    assert order == [("B", 0), ("A", 0), ("B", 1), ("A", 1)]
