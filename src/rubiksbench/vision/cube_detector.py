"""Locate the cube and its three visible faces in an image.

The cube is cut out with ``rembg`` (open-source background removal, U2-Net).
Seen from above one corner, its silhouette is a hexagon with vertices, clockwise
from the top: T (back), UR, LR, B (front-bottom), LL, UL. The three visible
faces meet at an inner vertex C, found from the vanishing points of the cube's
edges, giving faces U = T-UR-C-UL, F = UL-C-B-LL and R = C-UR-LR-B.

Faces are named by position: top = U, front-left = F, front-right = R. Each
quad lists its corners in Kociemba facelet order (facelets 1, 3, 9, 7).
"""

from dataclasses import dataclass

import cv2
import numpy as np
from rembg import new_session, remove

# Silhouettes simplified to six vertices; tolerance tried from tight to loose,
# as a fraction of the hull perimeter.
_SIMPLIFY_STEPS = np.linspace(0.005, 0.08, 60)


@dataclass
class FaceLayout:
    hexagon: np.ndarray  # 6x2, clockwise from the top vertex
    inner: np.ndarray  # estimated meeting point of the three faces
    quads: dict[str, np.ndarray]  # face -> 4x2 corners (facelets 1, 3, 9, 7)
    # How far C is from where the vertical edge says it should be, relative to
    # cube size; large values mean the silhouette is not a clean three-face view.
    inner_spread: float
    # Overlap (IoU) of the object's silhouette with the hexagon: ~0.95-0.99 for a
    # cube; lower when it is not a cube, or a hand or the frame edge cuts into it.
    hexagon_fit: float = 1.0

    def to_dict(self) -> dict:
        return {
            "hexagon": self.hexagon.round(1).tolist(),
            "inner": self.inner.round(1).tolist(),
            "inner_spread": round(self.inner_spread, 4),
            "hexagon_fit": round(self.hexagon_fit, 4),
            "quads": {f: q.round(1).tolist() for f, q in self.quads.items()},
        }


def _overlap(shape: tuple[int, ...], contour: np.ndarray, hexagon: np.ndarray) -> float:
    """IoU of the filled object contour and the filled hexagon."""
    obj, hexa = np.zeros(shape[:2], np.uint8), np.zeros(shape[:2], np.uint8)
    cv2.drawContours(obj, [contour], -1, 1, cv2.FILLED)
    cv2.fillPoly(hexa, [hexagon.astype(np.int32)], 1)
    union = np.count_nonzero(obj | hexa)
    return float(np.count_nonzero(obj & hexa) / union) if union else 0.0


def _h(p: np.ndarray) -> np.ndarray:
    return np.array([p[0], p[1], 1.0])


def _inner_vertex(t, ur, lr, b, ll, ul) -> tuple[np.ndarray | None, float]:
    """Where the three faces meet, by perspective geometry.

    Cube edges parallel in space meet at a vanishing point in the image (at
    infinity for an orthographic-looking render; homogeneous coordinates cover
    both). The edge UL-C is parallel to T-UR and LL-B, and UR-C to T-UL and
    LR-B, so C is the intersection of the lines from UL and UR towards those
    vanishing points. The vertical edge C-B must point at the vanishing point of
    UL-LL and UR-LR; the distance from C to that line (relative to cube size) is
    returned as a consistency measure.
    """
    vp_right = np.cross(np.cross(_h(t), _h(ur)), np.cross(_h(ll), _h(b)))
    vp_left = np.cross(np.cross(_h(t), _h(ul)), np.cross(_h(lr), _h(b)))
    vp_down = np.cross(np.cross(_h(ul), _h(ll)), np.cross(_h(ur), _h(lr)))
    c = np.cross(np.cross(_h(ul), vp_right), np.cross(_h(ur), vp_left))
    if abs(c[2]) < 1e-9:
        return None, 1.0
    inner = (c[:2] / c[2]).astype(np.float32)
    vertical = np.cross(_h(b), vp_down)
    size = float(np.linalg.norm(b - t)) or 1.0
    norm = float(np.hypot(vertical[0], vertical[1])) or 1.0
    spread = abs(float(vertical @ _h(inner))) / norm / size
    return inner, spread


class CubeDetector:
    def __init__(self, model: str = "u2net"):
        self.model = model
        self._session = None

    def mask(self, image_bgr: np.ndarray) -> np.ndarray:
        if self._session is None:
            self._session = new_session(self.model)
        rgba = np.asarray(remove(image_bgr[:, :, ::-1], session=self._session))
        return (rgba[:, :, 3] > 128).astype(np.uint8) * 255

    def layout(self, mask: np.ndarray) -> FaceLayout | None:
        """The three-face layout of the largest object, or None if it is not hexagonal."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        perimeter = cv2.arcLength(hull, True)
        approx = hull
        for step in _SIMPLIFY_STEPS:
            approx = cv2.approxPolyDP(hull, step * perimeter, True)
            if len(approx) <= 6:
                break
        if len(approx) != 6:
            return None

        points = approx.reshape(-1, 2).astype(np.float32)
        centre = points.mean(axis=0)
        # Image y points down, so increasing angle runs clockwise on screen.
        points = points[np.argsort(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))]
        top = int(np.argmin(points[:, 1]))
        t, ur, lr, b, ll, ul = (points[(top + i) % 6] for i in range(6))
        inner, spread = _inner_vertex(t, ur, lr, b, ll, ul)
        if inner is None:
            return None
        quads = {
            "U": np.array([t, ur, inner, ul]),
            "F": np.array([ul, inner, b, ll]),
            "R": np.array([inner, ur, lr, b]),
        }
        hexagon = np.array([t, ur, lr, b, ll, ul])
        return FaceLayout(hexagon, inner, quads, spread, _overlap(mask.shape, largest, hexagon))
