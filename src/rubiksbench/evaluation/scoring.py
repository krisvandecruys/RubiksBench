"""Aggregate per-image results into per-model scores.

The headline is the mean validity (0-100%, see ``validity``); the rubric
explains it. Rates are over successfully generated images; generation failures are counted
separately and inconclusive samples are reported, never folded into valid or
invalid.
"""

import math
from collections import Counter
from typing import Any

from .rubric import CHECKS, MAX_POINTS
from .taxonomy import Verdict


def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def mean_interval(values: list[float], lo: float, hi: float, z: float = 1.96) -> list[float] | None:
    """Normal-approximation 95% interval for a mean, clipped to [lo, hi]."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = z * sd / math.sqrt(n)
    return [max(lo, mean - half), min(hi, mean + half)]


def score_model(generations: list[dict[str, Any]], evaluations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ok = [g for g in generations if g.get("error") is None and g.get("image_path")]
    verdicts = Counter(evaluations[g["sample_id"]]["verdict"] for g in ok if g["sample_id"] in evaluations)
    flags = Counter(
        f for g in ok if g["sample_id"] in evaluations for f in evaluations[g["sample_id"]].get("flags", [])
    )
    n = len(ok)
    standard = verdicts[Verdict.VALID_STANDARD]
    abstract = standard + verdicts[Verdict.VALID_NONSTANDARD]

    def rate(count: int) -> float | None:
        return count / n if n else None

    graded = [evaluations[g["sample_id"]]["rubric"] for g in ok if "rubric" in evaluations.get(g["sample_id"], {})]
    points = [r["points"] for r in graded]
    # An image passes only when it earns every point; anything less is a fail.
    passed = sum(1 for p in points if p == MAX_POINTS)
    by_check = {
        c.id: sum({"pass": 1.0, "inconclusive": 0.5}.get(r["checks"][c.id], 0.0) for r in graded) / len(graded)
        for c in CHECKS
    } if graded else {}
    validities = [evaluations[g["sample_id"]]["validity"]["score"] for g in ok
                  if "validity" in evaluations.get(g["sample_id"], {})]
    costs = [g["cost"] for g in generations if g.get("cost") is not None]
    latencies = [g["latency_s"] for g in ok]
    return {
        "attempted": len(generations),
        "generation_failures": len(generations) - n,
        "generation_failure_codes": dict(Counter(g["error"]["code"] for g in generations if g.get("error"))),
        "images": n,
        "passed": passed,
        "failed": n - passed,
        "pass_rate": rate(passed),
        "pass_rate_ci95": wilson_interval(passed, n),
        "validity_mean": sum(validities) / len(validities) if validities else None,
        "validity_ci95": mean_interval(validities, 0.0, 1.0),
        "rubric_mean": sum(points) / len(points) if points else None,
        "rubric_ci95": mean_interval(points, 0.0, MAX_POINTS),
        "rubric_max": MAX_POINTS,
        "rubric_by_check": by_check,
        "verdicts": {v.value: verdicts[v] for v in Verdict},
        "standard_valid_rate": rate(standard),
        "standard_valid_rate_ci95": wilson_interval(standard, n),
        "abstract_valid_rate": rate(abstract),
        "invalid_rate": rate(verdicts[Verdict.INVALID]),
        "inconclusive_rate": rate(verdicts[Verdict.INCONCLUSIVE]),
        "flags": dict(flags.most_common()),
        "total_cost": sum(costs) if costs else None,
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
    }


def score_run(generations: list[dict[str, Any]], evaluations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for g in generations:
        by_model.setdefault(g["model_name"], []).append(g)
    return {
        name: {"model": gens[0]["request"]["model"], "reference": bool(gens[0].get("reference"))}
        | score_model(gens, evaluations)
        for name, gens in by_model.items()
    }


def ranked(scores: dict[str, Any]) -> list[str]:
    """Reference photos first (a calibration line), then by mean validity, then by rubric
    score; models without any images last."""
    def key(m: str) -> tuple[bool, bool, float, float, str]:
        s = scores[m]
        return (not s.get("reference"), s["validity_mean"] is None, -(s["validity_mean"] or 0),
                -(s["rubric_mean"] or 0), m)

    return sorted(scores, key=key)


def format_table(scores: dict[str, Any]) -> str:
    def pct(x: float | None) -> str:
        return "-" if x is None else f"{100 * x:.0f}%"

    def pts(x: float | None) -> str:
        return "-" if x is None else f"{x:.1f}"

    header: tuple[str, ...] = (
        "model", "images", "validity", "95% CI", "passed", "failed", "gen. errors", "rubric", "cost"
    )
    rows = [header]
    for name in ranked(scores):
        s = scores[name]
        ci = s["validity_ci95"]
        rows.append(
            (
                name,
                str(s["images"]),
                pct(s["validity_mean"]),
                "-" if ci is None else f"{pct(ci[0])}-{pct(ci[1])}",
                f"{s['passed']} ({pct(s['pass_rate'])})",
                str(s["failed"]),
                str(s["generation_failures"]),
                f"{pts(s['rubric_mean'])}/{s['rubric_max']:.0f}",
                "-" if s["total_cost"] is None else f"${s['total_cost']:.3f}",
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in enumerate(zip(row, widths)))
        for row in rows
    )
