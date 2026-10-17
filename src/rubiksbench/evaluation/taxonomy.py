"""Result verdicts and failure flags."""

from enum import StrEnum


class Verdict(StrEnum):
    VALID_STANDARD = "VALID_STANDARD"
    VALID_NONSTANDARD = "VALID_NONSTANDARD"
    INVALID = "INVALID"
    INCONCLUSIVE = "INCONCLUSIVE"


class Flag(StrEnum):
    # Image / geometry (set by the vision layer).
    NOT_A_CUBE = "not_a_cube"
    GEOMETRY_INVALID = "geometry_invalid"
    WRONG_GRID_SIZE = "wrong_grid_size"
    MARKED_STICKERS = "marked_stickers"
    SPLIT_STICKERS = "split_stickers"
    VISION_LOW_CONFIDENCE = "vision_low_confidence"
    INSUFFICIENT_VISIBILITY = "insufficient_visibility"
    NO_EXTRACTION = "no_extraction"
    # Colors and centers.
    TOO_MANY_COLORS = "too_many_colors"
    DUPLICATE_CENTERS = "duplicate_centers"
    NONSTANDARD_COLOR_LABELS = "nonstandard_color_labels"
    STANDARD_OPPOSITE_CENTERS_ADJACENT = "standard_opposite_centers_adjacent"
    STANDARD_OPPOSITE_PAIR_INVALID = "standard_opposite_pair_invalid"
    STANDARD_CHIRALITY_INVALID = "standard_chirality_invalid"
    # Pieces and reachability.
    IMPOSSIBLE_EDGE = "impossible_edge"
    IMPOSSIBLE_CORNER = "impossible_corner"
    DUPLICATE_PIECE = "duplicate_piece"
    ORIENTATION_INVARIANT_FAILURE = "orientation_invariant_failure"
    PERMUTATION_PARITY_FAILURE = "permutation_parity_failure"
    NO_LEGAL_COMPLETION = "no_legal_completion"


# Image-level flags that settle the verdict before any cube reasoning.
GEOMETRY_FAILURES = frozenset({Flag.NOT_A_CUBE, Flag.GEOMETRY_INVALID, Flag.WRONG_GRID_SIZE})
UNREADABLE = frozenset({Flag.VISION_LOW_CONFIDENCE, Flag.INSUFFICIENT_VISIBILITY, Flag.NO_EXTRACTION})
# Defects of the stickers themselves: the image is invalid however the colors read.
STICKER_DEFECTS = frozenset({Flag.MARKED_STICKERS, Flag.SPLIT_STICKERS})
