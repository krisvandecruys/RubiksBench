from rubiksbench.cube.observation import Observation
from rubiksbench.evaluation.evaluator import validate
from rubiksbench.evaluation.rubric import MAX_POINTS, grade
from rubiksbench.vision.extraction import Extraction

from .test_validator import REAL_CUBE

QUALITY = {"cube": "pass", "shape": "pass", "grid": "pass", "stickers": "pass", "single_color": "pass", "colors": "pass"}


def graded(faces):
    observation = Observation.from_faces(faces)
    extraction = Extraction("annotation", observation, quality=QUALITY)
    return grade(extraction, validate(observation, with_scramble=False).to_dict())


def test_real_cube_earns_every_point():
    result = graded(REAL_CUBE)
    assert result["points"] == MAX_POINTS == 15
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
    # It cannot exist on a standard cube either, but that is the same mistake.
    assert result["checks"]["solvable_standard"] == "blocked"
    assert [m for m in result["marks"] if m["facelets"] == ["U8", "F2"]] == [
        {"check": "standard_pairs", "label": "opposite colors on edge UF: R-O", "facelets": ["U8", "F2"], "faces": []}
    ]


def test_duplicate_centers_are_marked_and_blocked_checks_are_not():
    faces = dict(REAL_CUBE, R="ORG WGG OYB")  # R's center becomes green, like U's
    result = graded(faces)
    assert result["checks"]["distinct_centers"] == "fail"
    assert [(m["check"], m["facelets"]) for m in result["marks"]] == [("distinct_centers", ["U5", "R5"])]


def test_a_clean_pass_has_no_marks():
    assert "marks" not in graded(REAL_CUBE)


def test_gallery_is_sorted_by_validity_then_points():
    from rubiksbench.evaluation.report import _gallery_order

    scores = {"A": {"validity_mean": 0.6, "rubric_mean": 9.0}, "B": {"validity_mean": 0.2, "rubric_mean": 5.0}}
    samples = [
        {"model_name": "A", "iteration": 0, "points": 12.0, "validity": {"score": 0.3}},
        {"model_name": "B", "iteration": 0, "points": 9.0, "validity": {"score": 0.8}},
        {"model_name": "A", "iteration": 1, "points": None, "validity": None},  # generation failed
        {"model_name": "B", "iteration": 1, "points": 13.0, "validity": {"score": 0.3}},
        {"model_name": "A", "iteration": 2, "points": 13.0, "validity": {"score": 0.3}},
    ]
    order = [(s["model_name"], s["iteration"]) for s in _gallery_order(samples, scores)]
    # Most valid first across models; equal validity by points, then by the better-ranked model.
    assert order == [("B", 0), ("A", 2), ("B", 1), ("A", 0), ("A", 1)]


def test_gallery_outlines_marked_stickers_over_the_image():
    from rubiksbench.evaluation.report import _marks_svg

    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    sample = {"image_size": [768, 512], "marks": [
        {"label": "duplicate center: G", "polygons": [square, square]},
        {"label": "every corner twisted", "polygons": []},  # nothing to outline
    ]}
    svg = _marks_svg(sample)
    assert "viewBox='0 0 768 512'" in svg and "<title>duplicate center: G</title>" in svg
    assert svg.count("class='line'") == 2 and "twisted" not in svg
    assert _marks_svg({"marks": []}) == ""


def test_impossible_piece_is_found_despite_duplicate_centers():
    # GPT-Image-2.5 Sunburst #1: two white centers rule out every color scheme, yet its
    # blue-red-blue corner is impossible under any scheme and is reported like any other.
    faces = {"U": "YRG BWY OGB", "R": "ROW BYG WRB", "F": "GRB YWO OYG"}
    result = graded(faces)
    checks = result["checks"]
    assert checks["distinct_centers"] == "fail" and checks["possible_pieces"] == "fail"
    assert checks["unique_pieces"] == "blocked"  # no two pieces show the same colors; undecidable without a scheme
    # Unsolvable follows from the bad centers and corner: a consequence, not a mistake of its own.
    assert checks["solvable"] == checks["solvable_standard"] == "blocked"
    # White and yellow side by side is a separate mistake from the duplicate white center.
    assert checks["center_order"] == "fail"
    assert {"check": "possible_pieces", "label": "impossible corner URF (B-R-B)",
            "facelets": ["U9", "R1", "F3"], "faces": []} in result["marks"]
    assert "impossible_corner" in validate(Observation.from_faces(faces), with_scramble=False).to_dict()["flags"]


def test_a_consequence_of_an_earlier_failure_is_not_counted_again():
    # Seedream 5.0 Pro #0: every visible center is white. That rules out the standard
    # center order too, but it is one mistake, so the order check is blocked.
    faces = {"U": "WRO YWB WRY", "R": "ORG GWG OWB", "F": "YGB WWY OGB"}
    checks = graded(faces)["checks"]
    assert checks["distinct_centers"] == "fail" and checks["center_order"] == "blocked"


def test_duplicate_piece_is_found_without_a_color_scheme():
    from rubiksbench.cube.constraints import scheme_free_piece_failures
    from rubiksbench.evaluation.taxonomy import Flag

    # Edges UF (U8, F2) and UR (U6, R2) both show white and blue.
    observation = Observation.from_faces({"U": ".....W.W.", "F": ".B.......", "R": ".B......."})
    found = scheme_free_piece_failures(observation)
    assert [(flag, c.reason) for flag, c in found] == [(Flag.DUPLICATE_PIECE, "duplicate edge in UR, UF")]
