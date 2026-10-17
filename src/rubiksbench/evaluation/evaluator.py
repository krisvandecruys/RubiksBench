"""Turn an observation into a benchmark verdict with flags and a witness."""

from dataclasses import dataclass, field
from typing import Any

from .. import EVALUATOR_VERSION
from ..cube.completion_solver import Culprit, SchemeAttempt, first_completion
from ..cube.constraints import abstract_schemes, color_flags, scheme_free_piece_failures, standard_schemes
from ..cube.facelets import facelet_name
from ..cube.observation import Observation
from ..cube.witness import Witness, build_witness
from ..vision.extraction import Extraction
from .rubric import grade
from .validity import validity
from .taxonomy import GEOMETRY_FAILURES, STICKER_DEFECTS, UNREADABLE, Flag, Verdict


@dataclass
class ValidityCheck:
    """Abstract or standard validity: a witness, or the reasons none exists."""

    valid: bool
    flags: list[Flag] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    schemes_tried: int = 0
    # Best piece-level stage reached under any color scheme (see SchemeAttempt.stage).
    furthest_stage: int = 0
    # Stickers behind the failure under the scheme that got furthest (see _failed).
    culprits: list[Culprit] = field(default_factory=list)
    witness: Witness | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "valid": self.valid,
            "flags": [str(f) for f in self.flags],
            "reasons": self.reasons,
            "schemes_tried": self.schemes_tried,
            "furthest_stage": self.furthest_stage,
            "culprits": [
                {"reason": c.reason, "facelets": [facelet_name(i) for i in c.facelets]} for c in self.culprits
            ],
        }
        if w := self.witness:
            out["witness"] = {
                "scheme": w.scheme.as_dict(),
                "facelets": w.facelets,
                "colors": "".join(c if len(c) == 1 else f"[{c}]" for c in w.colors),
                "scramble": w.scramble,
                "scramble_verified": w.scramble_verified,
            }
        return out


@dataclass
class ValidationReport:
    verdict: Verdict
    flags: list[Flag]
    abstract: ValidityCheck
    standard: ValidityCheck
    observation: Observation

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "verdict": str(self.verdict),
            "flags": [str(f) for f in self.flags],
            "observation": self.observation.to_faces(),
            "abstract": self.abstract.to_dict(),
            "standard": self.standard.to_dict(),
        }


def _check(
    observation: Observation, schemes, pre_flags: list[Flag], pre_reasons: list[str], with_scramble: bool
) -> ValidityCheck:
    if pre_flags:
        # No color scheme fits, so the completion search cannot run; still report the piece
        # failures that need no scheme, so they are flagged as they would be otherwise.
        pieces = scheme_free_piece_failures(observation)
        flags = pre_flags + sorted({f for f, _ in pieces} - set(pre_flags))
        return ValidityCheck(
            False, flags, pre_reasons + [c.reason for _, c in pieces], culprits=[c for _, c in pieces]
        )
    success, failures = first_completion(observation, schemes)
    tried = len(failures) + (success is not None)
    if success is not None:
        assert success.state is not None
        return ValidityCheck(
            True, schemes_tried=tried, furthest_stage=3,
            witness=build_witness(success.scheme, success.state, with_scramble),
        )
    return _failed(failures, tried)


def _failed(failures: list[SchemeAttempt], tried: int) -> ValidityCheck:
    # A flag is reported if it explains the failure under at least one scheme;
    # the per-scheme reasons say which.
    flags = sorted({f for a in failures for f in a.flags} | {Flag.NO_LEGAL_COMPLETION})
    reasons = [f"[{a.scheme}] {r}" for a in failures for r in a.reasons]
    stage = max((a.stage for a in failures), default=0)
    # Which pieces are "wrong" depends on the scheme, so name the fewest culprits among
    # the schemes that got furthest: the smallest explanation, not the only one.
    explained = [a for a in failures if a.stage == stage and a.culprits]
    culprits = min(explained, key=lambda a: len(a.culprits)).culprits if explained else []
    return ValidityCheck(False, flags, reasons, tried, stage, culprits)


def evaluate_extraction(extraction: Extraction, *, with_scramble: bool = True) -> dict[str, Any]:
    """Verdict for one image from its extraction, as a JSON-ready record."""
    record: dict[str, Any] = {
        "evaluator_version": EVALUATOR_VERSION,
        "extractor": extraction.extractor,
        "extraction_confidence": extraction.confidence,
        "extraction_flags": [str(f) for f in extraction.flags],
        "extraction_details": extraction.details,
    }
    flags = set(extraction.flags)
    validation: dict[str, Any] | None = None
    if geometry := sorted(flags & GEOMETRY_FAILURES):
        record |= {"verdict": str(Verdict.INVALID), "flags": [str(f) for f in geometry]}
    else:
        if extraction.observation is None:
            flags.add(Flag.NO_EXTRACTION)
        # Letters, or two colors on one sticker, make the image invalid even when the colors form a
        # legal cube or are partly unsure; the checks below still show how far it got.
        defects = sorted(flags & STICKER_DEFECTS)
        if unreadable := sorted(flags & UNREADABLE):
            verdict = Verdict.INVALID if defects else Verdict.INCONCLUSIVE
            record |= {"verdict": str(verdict), "flags": [str(f) for f in defects + unreadable]}
        else:
            assert extraction.observation is not None
            validation = validate(extraction.observation, with_scramble=with_scramble).to_dict()
            record |= validation
            if defects:
                record |= {"verdict": str(Verdict.INVALID), "flags": [str(f) for f in defects] + record["flags"]}
    rubric = grade(extraction, validation)
    return record | {"rubric": rubric, "validity": validity(extraction, rubric["checks"])}


def validate(
    observation: Observation, *, min_observed_stickers: int = 1, with_scramble: bool = True
) -> ValidationReport:
    """Decide abstract and standard validity of a partial observation.

    INCONCLUSIVE is only returned here when too few stickers are observed;
    the vision layer adds its own low-confidence handling on top.
    """
    if observation.observed_count < min_observed_stickers:
        empty = ValidityCheck(False)
        return ValidationReport(Verdict.INCONCLUSIVE, [Flag.INSUFFICIENT_VISIBILITY], empty, empty, observation)

    pre_flags, pre_reasons = color_flags(observation)
    abstract = _check(observation, abstract_schemes(observation), pre_flags, pre_reasons, with_scramble)

    # Run independently of the abstract result so its reasons are specific to the standard scheme.
    schemes, std_flags, std_reasons = standard_schemes(observation)
    standard = _check(observation, schemes, pre_flags + std_flags, pre_reasons + std_reasons, with_scramble)

    if standard.valid:
        verdict = Verdict.VALID_STANDARD
        flags = []
    elif abstract.valid:
        verdict = Verdict.VALID_NONSTANDARD
        flags = standard.flags
    else:
        verdict = Verdict.INVALID
        flags = abstract.flags
    return ValidationReport(verdict, flags, abstract, standard, observation)
