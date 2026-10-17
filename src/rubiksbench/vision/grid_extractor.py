"""Flatten a face with a perspective transform, check it is 3x3, and sample its stickers."""

from dataclasses import dataclass

import cv2
import numpy as np

SIZE = 300  # flattened face, pixels per side
# Sample the middle 40% of each cell, away from borders and rounded corners.
_CELL_LO, _CELL_HI = 0.3, 0.7


@dataclass
class FaceGrid:
    image: np.ndarray  # flattened face (BGR)
    columns: int  # dominant number of cells across the face
    rows: int  # dominant number of cells down the face
    # How much stronger the winning cell count is than the runner-up (>= 1).
    column_dominance: float
    row_dominance: float
    rgb: list[tuple[float, float, float]]  # 9 sticker colors, row-major
    # Per sticker, row-major: share of sharp-edge pixels in its centre (letters, logos).
    marking: list[float]


def flatten(image_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    target = np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])
    transform = cv2.getPerspectiveTransform(np.float32(quad), target)
    return cv2.warpPerspective(image_bgr, transform, (SIZE, SIZE))


def _period(profile: np.ndarray) -> tuple[int, float]:
    """Strongest cell count (2-7) in an edge profile, and its lead over the runner-up."""
    power = np.abs(np.fft.rfft(profile - profile.mean()))[2:8]
    best = int(np.argmax(power))
    runner_up = float(np.partition(power, -2)[-2]) or 1e-9
    return best + 2, float(power[best] / runner_up)


def grid_period(face_bgr: np.ndarray) -> tuple[tuple[int, float], tuple[int, float]]:
    """Cells across and down the face, each with its dominance, from edge periodicity.

    Sticker borders produce edge energy at every cell boundary, so the strongest
    frequency of the column (resp. row) edge profile is the number of columns
    (resp. rows). Measured separately, so a 4x3 face is caught.

    Letters inside every sticker add edges at twice the cell rate, so 6 can beat
    3. Gaps between stickers span the whole face while letters do not, so the
    median profile (instead of the mean) settles that case.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    margin = int(SIZE * 0.04)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0))[margin:-margin, :]
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))[:, margin:-margin]
    periods = []
    for edges, axis in ((gx, 0), (gy, 1)):
        period = _period(edges.mean(axis=axis))
        if period[0] == 6 and (median := _period(np.median(edges, axis=axis)))[0] == 3:
            period = median
        periods.append(period)
    return periods[0], periods[1]


# Gradient magnitude above which a pixel counts as part of a sharp stroke.
_STROKE = 120.0


def sticker_marking(face_bgr: np.ndarray) -> list[float]:
    """Share of sharp-edge pixels in the middle of each sticker, row-major.

    Plain stickers (even glossy ones) are smooth there; printed letters or logos
    are not.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    magnitude = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    step = SIZE / 3
    out = []
    for row in range(3):
        for col in range(3):
            cell = magnitude[
                int((row + _CELL_LO) * step) : int((row + _CELL_HI) * step),
                int((col + _CELL_LO) * step) : int((col + _CELL_HI) * step),
            ]
            out.append(float((cell > _STROKE).mean()))
    return out


def sample_cells(face_bgr: np.ndarray) -> list[tuple[float, float, float]]:
    rgb = []
    step = SIZE / 3
    for row in range(3):
        for col in range(3):
            cell = face_bgr[
                int((row + _CELL_LO) * step) : int((row + _CELL_HI) * step),
                int((col + _CELL_LO) * step) : int((col + _CELL_HI) * step),
            ]
            b, g, r = np.median(cell.reshape(-1, 3), axis=0)
            rgb.append((float(r), float(g), float(b)))
    return rgb


def read_face(image_bgr: np.ndarray, quad: np.ndarray) -> FaceGrid:
    face = flatten(image_bgr, quad)
    (columns, column_dominance), (rows, row_dominance) = grid_period(face)
    return FaceGrid(face, columns, rows, column_dominance, row_dominance, sample_cells(face), sticker_marking(face))
