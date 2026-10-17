"""Facelet and cubie conventions.

These follow Kociemba's conventions exactly, so states can be handed to the
``kociemba`` solver and to ``magiccube`` without translation.

Faces are ordered U R F D L B. Each face has 9 facelets numbered 1-9 row by
row as seen from outside the cube, giving the global index ``9 * face + n - 1``.
Orientation of each face's grid:

* U: top row borders B, bottom row borders F.
* D: top row borders F, bottom row borders B.
* R, F, L, B: top row borders U.

For a typical image showing three faces around one corner, the top face is U,
the front-left face is F and the front-right face is R.
"""

from typing import Final

FACES: Final = "URFDLB"
CENTER_INDEX: Final = 4

OPPOSITE_FACE: Final = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}


def facelet_index(name: str) -> int:
    """Convert a facelet name such as ``"F3"`` to its index 0-53."""
    face, number = name[0], int(name[1:])
    if face not in FACES or not 1 <= number <= 9:
        raise ValueError(f"invalid facelet name {name!r}")
    return 9 * FACES.index(face) + number - 1


def facelet_name(index: int) -> str:
    return f"{FACES[index // 9]}{index % 9 + 1}"


def center_of(face: str) -> int:
    return 9 * FACES.index(face) + CENTER_INDEX


def _table(spec: str) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(facelet_index(f) for f in group.split()) for group in spec.split(","))


# Corner slots/cubies. Facelets of each corner are listed clockwise when
# looking at the corner from outside, starting with its U or D facelet.
CORNER_NAMES: Final = ("URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB")
CORNER_FACELETS: Final = _table(
    "U9 R1 F3, U7 F1 L3, U1 L1 B3, U3 B1 R3, D3 F9 R7, D1 L9 F7, D7 B9 L7, D9 R9 B7"
)

# Edge slots/cubies, listed with the U/D (or F/B for middle-layer edges) facelet first.
EDGE_NAMES: Final = ("UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR")
EDGE_FACELETS: Final = _table(
    "U6 R2, U8 F2, U4 L2, U2 B2, D6 R8, D2 F8, D4 L8, D8 B8, F6 R4, F4 L6, B6 L4, B4 R6"
)

# The home face of each sticker of each cubie, in the same order as the facelets.
CORNER_FACES: Final = CORNER_NAMES
EDGE_FACES: Final = EDGE_NAMES
