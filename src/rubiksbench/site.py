"""A shareable static site of one run's results, e.g. for GitHub Pages.

The newest ``results-*.html`` of the run becomes ``index.html``, with every image it
shows copied next to it (PNG converted to JPEG, so the page loads faster), the raw
results as ``results.json``, and social sharing tags (Open Graph and Twitter cards)
with a 1200x630 preview image framed around one of the run's images.
"""

import re
import shutil
from html import escape
from pathlib import Path

import cv2
import numpy as np

SHARE_SIZE = (1200, 630)  # the size Open Graph and Twitter large-image cards expect


def share_image(image_path: Path, out: Path) -> None:
    """A 1200x630 preview with the image's subject (the cube) framed in the middle.

    The subject is found as the pixels that differ from the background (the image border's
    median color). A band around it, with some margin, is widened to the preview's shape by
    repeating its left and right edge pixels, so a plain studio background stays seamless.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"cannot read {image_path}")
    height, width = image.shape[:2]
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    subject = np.linalg.norm(image.astype(np.float32) - np.median(border, axis=0), axis=2) > 40
    rows = np.flatnonzero(subject.mean(axis=1) > 0.02)
    top, bottom = (rows[0], rows[-1]) if rows.size else (0, height - 1)
    band = min(height, max(int((bottom - top) * 1.2), height // 2))
    start = int(np.clip((top + bottom) / 2 - band / 2, 0, height - band))
    crop = image[start : start + band]
    pad = max(0, round(band * SHARE_SIZE[0] / SHARE_SIZE[1]) - width) // 2
    wide = _widen(crop, pad)
    cv2.imwrite(str(out), cv2.resize(wide, SHARE_SIZE, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])


def _widen(crop: np.ndarray, pad: int, edge: int = 48) -> np.ndarray:
    """Add ``pad`` columns each side, continuing the background: each row's mean edge color,
    smoothed down the image, blended into the picture over ``edge`` columns."""
    if pad <= 0:
        return crop
    height, width = crop.shape[:2]
    out = np.empty((height, width + 2 * pad, 3), np.float32)
    for side, strip in ((slice(0, pad), crop[:, :edge]), (slice(width + pad, None), crop[:, -edge:])):
        profile = cv2.GaussianBlur(strip.astype(np.float32).mean(axis=1, keepdims=True), (1, 0), sigmaX=1, sigmaY=height / 60)
        out[:, side] = profile
    out[:, pad : pad + width] = crop
    ramp = np.linspace(0, 1, edge, dtype=np.float32)[None, :, None]
    for columns, weight in ((slice(pad, pad + edge), ramp), (slice(pad + width - edge, pad + width), ramp[:, ::-1])):
        fill = out[:, pad - 1 : pad] if columns.start == pad else out[:, pad + width : pad + width + 1]
        out[:, columns] = out[:, columns] * weight + fill * (1 - weight)
    return out.clip(0, 255).astype(np.uint8)


def _meta(url: str, title: str, description: str, image_alt: str) -> str:
    image = url.rstrip("/") + "/share.jpg"
    tags = [
        ("name", "description", description),
        ("property", "og:type", "website"),
        ("property", "og:site_name", title),
        ("property", "og:title", title),
        ("property", "og:description", description),
        ("property", "og:url", url),
        ("property", "og:image", image),
        ("property", "og:image:width", str(SHARE_SIZE[0])),
        ("property", "og:image:height", str(SHARE_SIZE[1])),
        ("property", "og:image:alt", image_alt),
        ("name", "twitter:card", "summary_large_image"),
        ("name", "twitter:title", title),
        ("name", "twitter:description", description),
        ("name", "twitter:image", image),
        ("name", "twitter:image:alt", image_alt),
    ]
    return "".join(f"<meta {kind}='{key}' content='{escape(value, quote=True)}'>" for kind, key, value in tags)


def build_site(
    run: Path, out: Path, *, url: str, title: str, description: str, share_sample: str, share_alt: str
) -> Path:
    """Write the site for ``run`` into ``out`` (which is replaced) and return its index.html."""
    report = max(run.glob("results-*Z.html"))
    html = report.read_text()
    if out.exists():
        for child in out.iterdir():
            if child.name != ".git":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    for ref in sorted({m for m in re.findall(r"(?:src|href)=['\"]([^'\"#]+)['\"]", html)
                       if not m.startswith(("http", "data:"))}):
        source, target = run / ref, ref
        if source.suffix.lower() == ".png":
            target = Path(ref).with_suffix(".jpg").as_posix()
            (out / target).parent.mkdir(parents=True, exist_ok=True)
            if (picture := cv2.imread(str(source))) is None:
                raise ValueError(f"cannot read {source}")
            cv2.imwrite(str(out / target), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
            html = html.replace(f"'{ref}'", f"'{target}'").replace(f'"{ref}"', f'"{target}"')
        else:
            (out / target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, out / target)
    image = next(p for p in (run / "images").glob(f"{share_sample}.*") if p.suffix.lower() in (".png", ".jpg", ".webp")
                 and ".extraction" not in p.name)
    share_image(image, out / "share.jpg")
    html = re.sub(r"<title>[^<]*</title>", f"<title>{escape(title)}</title>", html, count=1)
    html = html.replace("</head>", _meta(url, title, description, share_alt) + "</head>", 1)
    (out / "index.html").write_text(html)
    shutil.copy(report.with_suffix(".json"), out / "results.json")
    (out / ".nojekyll").write_text("")
    return out / "index.html"

