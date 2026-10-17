"""Name sticker colors from their hue and chroma in CIELAB.

Shading changes a sticker's lightness far more than its hue, so colors are
told apart by hue angle, and white by low chroma. Hue bands were set from
hand-labelled images (red 27-40°, orange 47-66°, yellow 79-105°, green
133-155°, blue 251-288°), with boundaries between them. Hues outside every
standard band (purple, pink, ...) get the label ``X``, a seventh color.

Each sticker's margin is its distance to the nearest decision boundary
(degrees of hue, or chroma units for white), so a small margin means an
uncertain call.
"""

from dataclasses import dataclass

import cv2
import numpy as np

WHITE_MAX_CHROMA = 22.0
# Warm light and cream-colored plastic give white a yellow cast, so within the
# yellow hue band white extends further: cream white measured up to ~23 chroma,
# yellow stickers from ~49. (Faint blues reach ~23-35, so this is not global.)
WHITE_MAX_CHROMA_YELLOWISH = 36.0
_YELLOW_BAND = (72.0, 119.0)
# (lower hue bound in degrees, label); each band runs to the next bound.
HUE_BANDS: tuple[tuple[float, str], ...] = (
    (10.0, "R"),
    (43.5, "O"),
    (72.0, "Y"),
    (119.0, "G"),
    (200.0, "B"),
    (300.0, "X"),
)


@dataclass(frozen=True)
class StickerColor:
    label: str
    margin: float
    lab: tuple[float, float, float]
    hue: float
    chroma: float


def to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    pixel = np.float32(rgb).reshape(1, 1, 3) / 255
    l, a, b = cv2.cvtColor(pixel, cv2.COLOR_RGB2Lab).reshape(3)
    return float(l), float(a), float(b)


# A second color counts when it is this saturated and this far round the hue circle from the
# sticker's own color. Neighbouring hues (red/orange, orange/yellow) are not told apart from
# shading across a glossy sticker, so a split between them goes unnoticed.
SECOND_COLOR_CHROMA = 35.0
SECOND_COLOR_HUE = 60.0


def second_color_share(pixels_bgr: np.ndarray) -> float:
    """Share of a sticker's pixels that show a clearly different color (a two-colored sticker).

    White, grey and black pixels (highlights, plastic, letters) never count; on a white
    sticker every saturated pixel does.
    """
    lab = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3).astype(np.float32) / 255, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    a, b = lab[:, 1], lab[:, 2]
    main_a, main_b = float(np.median(a)), float(np.median(b))
    saturated = np.hypot(a, b) > SECOND_COLOR_CHROMA
    if np.hypot(main_a, main_b) < WHITE_MAX_CHROMA:
        return float(saturated.mean())
    turn = np.abs((np.degrees(np.arctan2(b, a) - np.arctan2(main_b, main_a)) + 180) % 360 - 180)
    return float((saturated & (turn > SECOND_COLOR_HUE)).mean())


def classify(rgb: tuple[float, float, float]) -> StickerColor:
    lab = to_lab(rgb)
    _, a, b = lab
    chroma = float(np.hypot(a, b))
    hue = float(np.degrees(np.arctan2(b, a)) % 360)
    white_limit = WHITE_MAX_CHROMA_YELLOWISH if _YELLOW_BAND[0] <= hue < _YELLOW_BAND[1] else WHITE_MAX_CHROMA
    if chroma < white_limit:
        return StickerColor("W", white_limit - chroma, lab, hue, chroma)
    for i, (low, label) in enumerate(HUE_BANDS):
        high = HUE_BANDS[(i + 1) % len(HUE_BANDS)][0]
        if high <= low:
            high += 360
        h = hue + 360 if hue < low else hue
        if low <= h < high:
            margin = min(h - low, high - h, chroma - white_limit)
            return StickerColor(label, margin, lab, hue, chroma)
    raise AssertionError("hue bands cover the full circle")
