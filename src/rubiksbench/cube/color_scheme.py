"""Assignments of colors (sticker labels) to the six faces."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from .facelets import FACES


@dataclass(frozen=True)
class ColorScheme:
    """The color of each face's center, in ``FACES`` order."""

    colors: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.colors) != 6 or len(set(self.colors)) != 6:
            raise ValueError(f"a color scheme needs six distinct colors, got {self.colors}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> "ColorScheme":
        return cls(tuple(mapping[f] for f in FACES))

    def color(self, face: str) -> str:
        return self.colors[FACES.index(face)]

    def face_of(self, color: str) -> str:
        return FACES[self.colors.index(color)]

    def as_dict(self) -> dict[str, str]:
        return dict(zip(FACES, self.colors))

    def __str__(self) -> str:
        return " ".join(f"{f}={c}" for f, c in zip(FACES, self.colors))


# Faces labelled by themselves: renders a state as a Kociemba facelet string.
IDENTITY_SCHEME: Final = ColorScheme(tuple(FACES))

# The conventional (Western) scheme: white top, green front, red right.
STANDARD_SCHEME: Final = ColorScheme.from_mapping(
    {"U": "W", "R": "R", "F": "G", "D": "Y", "L": "O", "B": "B"}
)
STANDARD_COLORS: Final = frozenset(STANDARD_SCHEME.colors)
STANDARD_OPPOSITE_COLOR: Final = {"W": "Y", "Y": "W", "R": "O", "O": "R", "G": "B", "B": "G"}

# Whole-cube rotations as "new face <- old face" maps.
_ROTATION_X: Final = {"U": "F", "F": "D", "D": "B", "B": "U", "R": "R", "L": "L"}
_ROTATION_Y: Final = {"F": "R", "R": "B", "B": "L", "L": "F", "U": "U", "D": "D"}


def rotate(scheme: ColorScheme, rotation: Mapping[str, str]) -> ColorScheme:
    return ColorScheme.from_mapping({new: scheme.color(old) for new, old in rotation.items()})


def orientations(scheme: ColorScheme) -> tuple[ColorScheme, ...]:
    """All 24 schemes reachable from ``scheme`` by rotating the whole cube.

    Rotations preserve handedness, so a mirrored scheme is never among them.
    """
    seen = {scheme}
    frontier = [scheme]
    while frontier:
        current = frontier.pop()
        for rotation in (_ROTATION_X, _ROTATION_Y):
            rotated = rotate(current, rotation)
            if rotated not in seen:
                seen.add(rotated)
                frontier.append(rotated)
    return tuple(sorted(seen, key=lambda s: s.colors))


STANDARD_ORIENTATIONS: Final = orientations(STANDARD_SCHEME)
