"""Partial sticker observations of a cube."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .facelets import FACES, center_of

UNKNOWN = "."


@dataclass(frozen=True)
class Observation:
    """Confidently observed sticker labels, indexed by Kociemba facelet index.

    ``None`` marks a sticker that is hidden or was not observed confidently.
    Labels are arbitrary strings; the standard-scheme check additionally needs
    them to be the standard color letters W Y R O G B.
    """

    stickers: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if len(self.stickers) != 54:
            raise ValueError(f"expected 54 stickers, got {len(self.stickers)}")

    @classmethod
    def from_faces(cls, faces: Mapping[str, str | Sequence[str | None]]) -> "Observation":
        """Build from per-face grids, e.g. ``{"U": "WWG RWO BBY", ...}``.

        A string grid uses one character per sticker (whitespace ignored) and
        ``.`` for unknown; a sequence grid uses ``None`` or ``"."`` for unknown.
        Faces that are not given are entirely unknown.
        """
        stickers: list[str | None] = [None] * 54
        for face, grid in faces.items():
            if face not in FACES:
                raise ValueError(f"unknown face {face!r}")
            cells = list("".join(grid.split())) if isinstance(grid, str) else list(grid)
            if len(cells) != 9:
                raise ValueError(f"face {face} needs 9 stickers, got {len(cells)}")
            offset = 9 * FACES.index(face)
            for i, cell in enumerate(cells):
                stickers[offset + i] = None if cell in (None, UNKNOWN) else cell
        return cls(tuple(stickers))

    @classmethod
    def from_facelet_string(cls, facelets: str) -> "Observation":
        """Build from a 54-character string in Kociemba face order U R F D L B."""
        return cls(tuple(None if c == UNKNOWN else c for c in facelets))

    def restricted_to(self, faces: str) -> "Observation":
        """Keep only the stickers on ``faces``, e.g. ``"UFR"`` for a corner view."""
        keep = {FACES.index(f) for f in faces}
        return Observation(tuple(s if i // 9 in keep else None for i, s in enumerate(self.stickers)))

    def with_stickers(self, changes: Mapping[int, str | None]) -> "Observation":
        stickers = list(self.stickers)
        for index, label in changes.items():
            stickers[index] = label
        return Observation(tuple(stickers))

    def center(self, face: str) -> str | None:
        return self.stickers[center_of(face)]

    @property
    def labels(self) -> frozenset[str]:
        return frozenset(s for s in self.stickers if s is not None)

    @property
    def observed_count(self) -> int:
        return sum(s is not None for s in self.stickers)

    def to_faces(self) -> dict[str, str]:
        out = {}
        for f, face in enumerate(FACES):
            cells = self.stickers[9 * f : 9 * f + 9]
            if any(c is not None for c in cells):
                out[face] = "".join(UNKNOWN if c is None else c for c in cells)
        return out
