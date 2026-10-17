"""Timestamped result files for a scored run.

    results-<UTC timestamp>.json   everything needed to recompute the report
    results-<UTC timestamp>.html   self-contained report: tables, inline SVG, sample gallery

The HTML follows the dataviz guidance: colors as light/dark tokens, one
sequential hue for magnitudes, reserved status colors for check results, hover
titles on every mark, and tables as the accessible view of every chart.
"""

import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from .. import EVALUATOR_VERSION
from .rubric import CHECKS, MAX_POINTS
from .taxonomy import Verdict

# Outcome colors (dataviz reference palette, validated as an adjacent set in this order).
OUTCOMES: tuple[tuple[str, str, str], ...] = (
    (Verdict.VALID_STANDARD, "Valid (standard)", "--outcome-1"),
    (Verdict.VALID_NONSTANDARD, "Valid (non-standard)", "--outcome-2"),
    (Verdict.INVALID, "Invalid", "--outcome-3"),
    (Verdict.INCONCLUSIVE, "Inconclusive", "--outcome-4"),
    ("generation_failed", "Generation failed", "--outcome-5"),
)
CHECK_RESULTS = {
    "pass": ("Pass", "--status-good"),
    "inconclusive": ("Inconclusive", "--status-warning"),
    "fail": ("Fail", "--status-critical"),
    "blocked": ("Blocked", "--blocked"),
}
# Heatmap buckets 0..10 (0%..100%): light steps run the blue ramp 100 -> 700;
# dark steps start near the dark surface and climb to a light blue.
_HEAT_LIGHT = ["#e6f0fc", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
               "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_HEAT_DARK = ["#1d2430", "#1c2c44", "#1b3352", "#1c3d63", "#1d4775", "#1f5389",
              "#2461a0", "#2d71b9", "#3987e5", "#5598e7", "#86b6ef"]
_HEAT_INK_LIGHT = ["#0b0b0b"] * 7 + ["#ffffff"] * 4
_HEAT_INK_DARK = ["#c3c2b7"] * 7 + ["#0b0b0b"] * 4


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def _ranked(scores: dict[str, Any]) -> list[str]:
    # Reference photos first (a calibration line), then best rubric score first;
    # models without any images last.
    def key(m: str) -> tuple[bool, bool, float, str]:
        mean = scores[m]["rubric_mean"]
        return not scores[m].get("reference"), mean is None, -(mean or 0), m

    return sorted(scores, key=key)


def _outcome_counts(s: dict[str, Any]) -> list[int]:
    return [s["verdicts"].get(str(v), 0) for v, _, _ in OUTCOMES[:-1]] + [s["generation_failures"]]


# ----------------------------------------------------------------------------- inline SVG

STRIP_W, STRIP_H, PAD = 200, 26, 8


def _x(points: float) -> float:
    return PAD + (STRIP_W - 2 * PAD) * points / MAX_POINTS


def _score_axis() -> str:
    ticks = "".join(
        f'<line x1="{_x(t):.1f}" x2="{_x(t):.1f}" y1="14" y2="18" class="axis"/>'
        f'<text x="{_x(t):.1f}" y="11" class="tick">{t}</text>'
        for t in range(0, int(MAX_POINTS) + 1, 3)
    )
    return (
        f'<svg class="strip" width="{STRIP_W}" height="20" viewBox="0 0 {STRIP_W} 20" aria-hidden="true">'
        f'<line x1="{_x(0)}" x2="{_x(MAX_POINTS)}" y1="18" y2="18" class="axis"/>{ticks}</svg>'
    )


def _score_strip(s: dict[str, Any]) -> str:
    if s["rubric_mean"] is None:
        return '<span class="muted">no images</span>'
    mean, ci = s["rubric_mean"], s["rubric_ci95"]
    grid = "".join(
        f'<line x1="{_x(t):.1f}" x2="{_x(t):.1f}" y1="3" y2="{STRIP_H - 3}" class="grid"/>'
        for t in range(0, int(MAX_POINTS) + 1, 3)
    )
    interval = (
        f'<line x1="{_x(ci[0]):.1f}" x2="{_x(ci[1]):.1f}" y1="{STRIP_H / 2}" y2="{STRIP_H / 2}" class="ci"/>'
        if ci else ""
    )
    label = f"{mean:.1f} of {MAX_POINTS:.0f}" + (f", 95% CI {ci[0]:.1f}–{ci[1]:.1f}" if ci else "")
    return (
        f'<svg class="strip" width="{STRIP_W}" height="{STRIP_H}" viewBox="0 0 {STRIP_W} {STRIP_H}" role="img" '
        f'aria-label="{escape(label)}"><title>{escape(label)}</title>{grid}{interval}'
        f'<circle cx="{_x(mean):.1f}" cy="{STRIP_H / 2}" r="5" class="dot"/></svg>'
    )


def _outcome_bar(s: dict[str, Any]) -> str:
    counts = _outcome_counts(s)
    total = sum(counts)
    if not total:
        return ""
    width, x, parts = 120, 0.0, []
    for count, (_, label, token) in zip(counts, OUTCOMES):
        if count:
            w = width * count / total
            parts.append(
                f'<rect x="{x:.1f}" y="2" width="{max(w - 2, 1):.1f}" height="12" rx="2" style="fill:var({token})">'
                f"<title>{escape(label)}: {count} of {total}</title></rect>"
            )
            x += w
    return f'<svg class="bar" width="{width}" height="16" viewBox="0 0 {width} 16">{"".join(parts)}</svg>'


def _check_strip(checks: dict[str, str] | None) -> str:
    if not checks:
        return ""
    cells = []
    for i, c in enumerate(CHECKS):
        label, token = CHECK_RESULTS[checks.get(c.id, "blocked")]
        cells.append(
            f'<rect x="{i * 13}" y="0" width="11" height="11" rx="2" style="fill:var({token})">'
            f"<title>{i + 1}. {escape(c.title)}: {label}</title></rect>"
        )
    return f'<svg class="checks" width="{len(CHECKS) * 13}" height="11">{"".join(cells)}</svg>'


def _legend(items: list[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="background:var({token})"></i>{escape(label)}</span>' for label, token in items
    ) + "</div>"


# ----------------------------------------------------------------------------- page

CSS = """
:root { color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781; --grid:#e1e0d9;
  --axis:#c3c2b7; --border:rgba(11,11,11,.10); --series:#2a78d6;
  --outcome-1:#1baf7a; --outcome-2:#2a78d6; --outcome-3:#e34948; --outcome-4:#eda100; --outcome-5:#6250d6;
  --status-good:#0ca30c; --status-warning:#fab219; --status-critical:#d03b3b; --blocked:#e1e0d9; LIGHT_HEAT }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { DARK } }
:root[data-theme="dark"] { DARK }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 40px 0 8px; }
p, .meta { color: var(--ink-2); } .meta { margin: 0 0 8px; } .muted { color: var(--muted); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px;
  overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 8px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 600; font-size: 12px; vertical-align: bottom; }
td.wrap, th.wrap { white-space: normal; text-align: left; }
.strip line.grid { stroke: var(--grid); stroke-width: 1; } .strip line.axis { stroke: var(--axis); }
.strip line.ci { stroke: var(--axis); stroke-width: 2; stroke-linecap: round; }
.strip circle.dot { fill: var(--series); stroke: var(--surface); stroke-width: 2; }
.strip text.tick { fill: var(--muted); font-size: 10px; text-anchor: middle; }
.heat td.h { text-align: center; min-width: 52px; font-size: 12px; border: 2px solid var(--surface); }
.heat th.c { text-align: center; white-space: normal; min-width: 52px; max-width: 72px; }
HEAT_CLASSES
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--ink-2); font-size: 12px; margin: 8px 0; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px;
  vertical-align: -1px; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 12px; }
.sample { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.sample img { display: block; width: 100%; aspect-ratio: 1; object-fit: cover; background: var(--grid); }
.sample .body { padding: 8px 10px 10px; font-size: 12px; color: var(--ink-2); }
.sample .head { display: flex; justify-content: space-between; color: var(--ink); font-weight: 600; font-size: 13px; }
.sample svg.checks { display: block; margin: 6px 0; }
.sample .flags { margin-top: 4px; word-break: break-word; }
.noimg { aspect-ratio: 1; display: grid; place-items: center; color: var(--muted); background: var(--grid); }
a { color: var(--series); }
code { font-size: 12px; }
.badge { font-size: 11px; color: var(--ink-2); border: 1px solid var(--border); border-radius: 4px; padding: 0 5px; }
tr.reference td { background: color-mix(in srgb, var(--surface) 92%, var(--series)); }
"""
DARK_TOKENS = (
    "color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781;"
    " --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10); --series:#3987e5;"
    " --outcome-1:#199e70; --outcome-2:#3987e5; --outcome-3:#e66767; --outcome-4:#c98500; --outcome-5:#9085e9;"
    " --blocked:#383835;"
)


def _css() -> str:
    light_heat = " ".join(f"--h{i}:{c}; --h{i}-ink:{k};" for i, (c, k) in enumerate(zip(_HEAT_LIGHT, _HEAT_INK_LIGHT)))
    dark_heat = " ".join(f"--h{i}:{c}; --h{i}-ink:{k};" for i, (c, k) in enumerate(zip(_HEAT_DARK, _HEAT_INK_DARK)))
    classes = "\n".join(f".h{i} {{ background: var(--h{i}); color: var(--h{i}-ink); }}" for i in range(11))
    return (
        CSS.replace("LIGHT_HEAT", light_heat)
        .replace("DARK", DARK_TOKENS + " " + dark_heat)
        .replace("HEAT_CLASSES", classes)
    )


def _scores_section(scores: dict[str, Any]) -> str:
    rows = []
    for model in _ranked(scores):
        s = scores[model]
        ci = s["rubric_ci95"]
        mean = "–" if s["rubric_mean"] is None else f"{s['rubric_mean']:.1f}"
        cost = "–" if s["total_cost"] is None else f"${s['total_cost']:.3f}"
        latency = "–" if s["mean_latency_s"] is None else f"{s['mean_latency_s']:.1f}s"
        rows.append(
            f"<tr{' class=reference' if s.get('reference') else ''}><td>{escape(model)}"
            + (" <span class='badge'>reference</span>" if s.get("reference") else "")
            + f"<div class='muted'><code>{escape(s['model'])}</code></div></td>"
            f"<td><b>{mean}</b></td><td>{'–' if ci is None else f'{ci[0]:.1f}–{ci[1]:.1f}'}</td>"
            f"<td>{_score_strip(s)}</td><td>{s['images']}</td>"
            f"<td><b>{s['passed']}</b> <span class='muted'>({_pct(s['pass_rate'])})</span></td><td>{s['failed']}</td>"
            f"<td>{s['generation_failures']}</td><td>{_outcome_bar(s)}</td><td>{cost}</td><td>{latency}</td></tr>"
        )
    return (
        f"<h2>Scores</h2><p>Each image earns up to {MAX_POINTS:.0f} points: 1 per check passed, ½ when "
        "inconclusive, 0 when failed or blocked by an earlier check (see <a href='#checks'>Checks</a>). "
        "An image <b>passes</b> only if it passes all checks. Scores are over generated images; requests that "
        "returned no image are counted separately as generation errors.</p>"
        "<div class='card'><table><thead><tr><th>Model</th><th>Score</th><th>95% CI</th>"
        f"<th>{_score_axis()}</th><th>Images</th>"
        f"<th title='Images passing all {MAX_POINTS:.0f} checks'>Passed</th>"
        "<th title='Images missing at least one check'>Failed</th>"
        "<th title='Requests that returned no image (API or model errors)'>Gen. errors</th>"
        "<th>Outcomes</th><th>Cost</th><th>Latency</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
        + _legend([(label, token) for _, label, token in OUTCOMES]) + "</div>"
    )


def _heatmap_section(scores: dict[str, Any]) -> str:
    head = "".join(
        f"<th class='c' title='{escape(c.description)}'>{i}. {escape(c.title)}</th>" for i, c in enumerate(CHECKS, 1)
    )
    rows = []
    for model in _ranked(scores):
        by_check = scores[model]["rubric_by_check"]
        cells = []
        for c in CHECKS:
            value = by_check.get(c.id)
            if value is None:
                cells.append("<td class='h muted'>–</td>")
            else:
                cells.append(
                    f"<td class='h h{round(value * 10)}' title='{escape(model)} · {escape(c.title)}: "
                    f"{_pct(value)} of points'>{_pct(value)}</td>"
                )
        rows.append(f"<tr><td>{escape(model)}</td>{''.join(cells)}</tr>")
    return (
        "<h2>Points per check</h2><p>Share of the available points each model earned on each check. "
        "Checks unlock left to right: a check whose prerequisites did not pass scores 0.</p>"
        f"<div class='card heat'><table><thead><tr><th>Model</th>{head}</tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


def _checks_section() -> str:
    titles = {c.id: c.title for c in CHECKS}
    rows = "".join(
        f"<tr><td>{i}</td><td class='wrap'>{escape(c.title)}</td><td class='wrap'>{escape(c.description)}</td>"
        f"<td class='wrap'>{escape(', '.join(titles[r] for r in c.requires)) or '–'}</td></tr>"
        for i, c in enumerate(CHECKS, 1)
    )
    return (
        "<h2 id='checks'>Checks</h2><div class='card'><table><thead><tr><th>#</th><th class='wrap'>Check</th>"
        "<th class='wrap'>Passes when</th><th class='wrap'>Requires</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _details_section(scores: dict[str, Any]) -> str:
    parts = []
    flagged = [(m, scores[m]["flags"]) for m in _ranked(scores) if scores[m]["flags"]]
    if flagged:
        items = "".join(
            f"<li><b>{escape(m)}:</b> "
            + ", ".join(f"<code>{escape(f)}</code> × {n}" for f, n in flags.items()) + "</li>"
            for m, flags in flagged
        )
        parts.append(f"<h2>Failure flags</h2><p>Images carrying each flag (one image can carry several).</p><ul>{items}</ul>")
    failed = [(m, scores[m]["generation_failure_codes"]) for m in _ranked(scores) if scores[m]["generation_failure_codes"]]
    if failed:
        items = "".join(
            f"<li><b>{escape(m)}:</b> "
            + ", ".join(f"<code>{escape(c)}</code> × {n}" for c, n in codes.items()) + "</li>"
            for m, codes in failed
        )
        parts.append(f"<h2>Generation failures</h2><ul>{items}</ul>")
    return "".join(parts)


def _gallery_order(samples: list[dict[str, Any]], scores: dict[str, Any]) -> list[dict[str, Any]]:
    """Highest score first across all models; ties by leaderboard rank, failed generations last."""
    rank = {model: i for i, model in enumerate(_ranked(scores))}
    return sorted(
        samples,
        key=lambda s: (-(s["points"] if s["points"] is not None else -1), rank.get(s["model_name"], len(rank)),
                       s["iteration"]),
    )


def _samples_section(samples: list[dict[str, Any]], scores: dict[str, Any]) -> str:
    cards = []
    for s in _gallery_order(samples, scores):
        if s["image_path"]:
            src = escape(s["image_path"])
            image = f"<a href='{src}'><img loading='lazy' src='{src}' alt='{escape(s['sample_id'])}'></a>"
        else:
            image = f"<div class='noimg'>{escape(s.get('error') or 'no image')}</div>"
        if s.get("reference"):
            seed = "real photo"
        else:
            seed = f"seed {escape(str(s['seed']))}" if s["seed_supported"] else "unseeded"
        if s.get("backfilled"):
            seed += " · <span title='Generated after the run, when the original request failed'>backfilled</span>"
        overlay = f" · <a href='{escape(s['overlay_path'])}'>extraction</a>" if s.get("overlay_path") else ""
        points = "–" if s["points"] is None else f"{s['points']:g}/{MAX_POINTS:.0f}"
        flags = ", ".join(escape(f) for f in s["flags"]) or "–"
        cards.append(
            f"<div class='sample'>{image}<div class='body'><div class='head'><span>{escape(s['model_name'])} "
            f"#{s['iteration']}</span><span>{points}</span></div>{_check_strip(s.get('checks'))}"
            f"<div>{escape(s['verdict'])} · {seed}{overlay}</div><div class='flags muted'>{flags}</div></div></div>"
        )
    legend = _legend([(label, token) for label, token in CHECK_RESULTS.values()])
    return (
        f"<h2>Samples</h2><p>Every attempted image, highest score first. The squares are the {len(CHECKS)} checks in order; "
        "hover for names.</p>"
        f"{legend}<div class='gallery'>{''.join(cards)}</div>"
    )


def render_html(stamp: str, run: dict[str, Any], scores: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    config = run.get("config", {})
    seeds = "random" if config.get("seed") is None else f"from {config['seed']}"
    extractor = next((s["extractor"] for s in samples if s["extractor"]), config.get("extractor", "–"))
    meta = (
        f"<p class='meta'>Prompt: “{escape(str(config.get('prompt', '?')))}”</p>"
        f"<p class='meta'>Backend {escape(str(config.get('backend', '?')))} · "
        f"{escape(str(config.get('iterations', '?')))} images per model · seeds {seeds} · "
        f"extractor {escape(str(extractor))} · evaluator {EVALUATOR_VERSION} · "
        f"run started {escape(str(run.get('started_at', '?')))} · scored {escape(stamp)}</p>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>RubiksBench results</title><style>{_css()}</style></head><body><main>"
        f"<h1>Rubik's Cube benchmark results</h1>{meta}"
        + _scores_section(scores) + _heatmap_section(scores) + _samples_section(samples, scores)
        + _details_section(scores) + _checks_section()
        + "</main></body></html>\n"
    )


def write_results(
    root: Path, run: dict[str, Any], generations: list[dict[str, Any]], evaluations: dict[str, dict[str, Any]],
    scores: dict[str, Any],
) -> dict[str, Path]:
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    samples = []
    for g in generations:
        e = evaluations.get(g["sample_id"], {})
        samples.append(
            {
                "sample_id": g["sample_id"],
                "model_name": g["model_name"],
                "model": g["request"]["model"],
                "iteration": g["iteration"],
                "seed": g["request"]["seed"],
                "seed_supported": g.get("seed_supported", True),
                "backfilled": "backfill_of" in g,
                "image_path": g.get("image_path"),
                "error": g["error"]["code"] if g.get("error") else None,
                "verdict": e.get("verdict", "generation_failed"),
                "flags": e.get("flags", []),
                "extractor": e.get("extractor"),
                "points": e.get("rubric", {}).get("points"),
                "checks": e.get("rubric", {}).get("checks"),
                "reference": bool(g.get("reference")),
                "overlay_path": (
                    str(Path(g["image_path"]).parent / o)
                    if g.get("image_path") and (o := e.get("extraction_details", {}).get("overlay")) else None
                ),
            }
        )
    paths = {ext: root / f"results-{stamp}.{ext}" for ext in ("json", "html")}
    raw = {
        "scored_at": now.isoformat(),
        "evaluator_version": EVALUATOR_VERSION,
        "run": run,
        "scores": scores,
        "samples": samples,
        "evaluations": evaluations,
    }
    paths["json"].write_text(json.dumps(raw, indent=2))
    paths["html"].write_text(render_html(now.strftime("%Y-%m-%d %H:%M:%S UTC"), run, scores, samples))
    return paths
