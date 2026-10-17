"""Cubie-level state and the cubies compatible with observed stickers.

A placement puts cubie ``cubie`` into a slot with ``orientation`` o, meaning the
k-th sticker of the cubie lies on facelet ``FACELETS[slot][(k + o) % n]``.
This is Kociemba's orientation convention.
"""

from dataclasses import dataclass

from .color_scheme import ColorScheme
from .facelets import CENTER_INDEX, CORNER_FACELETS, CORNER_FACES, EDGE_FACELETS, EDGE_FACES, FACES
from .observation import Observation


@dataclass(frozen=True)
class Placement:
    cubie: int
    orientation: int


@dataclass(frozen=True)
class CubieState:
    """A full cube state: which cubie sits in each slot and how it is turned."""

    corners: tuple[Placement, ...]
    edges: tuple[Placement, ...]

    def facelets(self, scheme: ColorScheme) -> tuple[str, ...]:
        out: list[str] = [""] * 54
        for f, face in enumerate(FACES):
            out[9 * f + CENTER_INDEX] = scheme.color(face)
        for slots, faces, placements in (
            (CORNER_FACELETS, CORNER_FACES, self.corners),
            (EDGE_FACELETS, EDGE_FACES, self.edges),
        ):
            for slot, placement in enumerate(placements):
                n = len(slots[slot])
                for k in range(n):
                    out[slots[slot][(k + placement.orientation) % n]] = scheme.color(
                        faces[placement.cubie][k]
                    )
        return tuple(out)


def _candidates(
    observation: Observation,
    scheme: ColorScheme,
    slot_facelets: tuple[int, ...],
    cubie_faces: tuple[str, ...],
) -> list[Placement]:
    n = len(slot_facelets)
    observed = [observation.stickers[i] for i in slot_facelets]
    result = []
    for cubie, faces in enumerate(cubie_faces):
        colors = [scheme.color(f) for f in faces]
        for orientation in range(n):
            if all(
                observed[(k + orientation) % n] in (None, colors[k]) for k in range(n)
            ):
                result.append(Placement(cubie, orientation))
    return result


def corner_candidates(observation: Observation, scheme: ColorScheme) -> list[list[Placement]]:
    return [_candidates(observation, scheme, f, CORNER_FACES) for f in CORNER_FACELETS]


def edge_candidates(observation: Observation, scheme: ColorScheme) -> list[list[Placement]]:
    return [_candidates(observation, scheme, f, EDGE_FACES) for f in EDGE_FACELETS]
