"""Witness scrambles: a constructive proof that a completed state is reachable.

The scramble comes from ``kociemba`` (two-phase solver): solve the witness
state, then invert the solution. It is then replayed on a solved cube in an
independent implementation (``magiccube``) and checked to reproduce the
witness exactly.
"""

from dataclasses import dataclass

import kociemba
import magiccube

from .color_scheme import IDENTITY_SCHEME, ColorScheme
from .facelets import CENTER_INDEX
from .pieces import CubieState

SOLVED = "".join(face * 9 for face in IDENTITY_SCHEME.colors)


@dataclass(frozen=True)
class Witness:
    scheme: ColorScheme
    state: CubieState
    facelets: str  # Kociemba face letters (URFDLB), independent of colors
    colors: tuple[str, ...]  # the witness rendered in the scheme's colors
    scramble: str | None
    scramble_verified: bool


def invert(moves: str) -> str:
    def inverse(move: str) -> str:
        if move.endswith("2"):
            return move
        return move[:-1] if move.endswith("'") else move + "'"

    return " ".join(inverse(m) for m in reversed(moves.split()))


def apply_moves(moves: str) -> str:
    """Kociemba facelet string after applying ``moves`` to a solved cube (via magiccube)."""
    cube = magiccube.Cube(3)
    if moves:
        cube.rotate(moves)
    colors = cube.get_kociemba_facelet_colors()
    face_of = {colors[9 * f + CENTER_INDEX]: face for f, face in enumerate(IDENTITY_SCHEME.colors)}
    return "".join(face_of[c] for c in colors)


def build_witness(scheme: ColorScheme, state: CubieState, with_scramble: bool = True) -> Witness:
    facelets = "".join(state.facelets(IDENTITY_SCHEME))
    scramble = None
    verified = False
    if with_scramble:
        scramble = "" if facelets == SOLVED else invert(kociemba.solve(facelets))
        verified = apply_moves(scramble) == facelets
    return Witness(scheme, state, facelets, state.facelets(scheme), scramble, verified)
