# RubiksBench

[![tests](https://github.com/kvlxai/RubiksBench/actions/workflows/tests.yml/badge.svg)](https://github.com/kvlxai/RubiksBench/actions/workflows/tests.yml)

Can image models draw a **valid scrambled 3×3 Rubik's Cube**? RubiksBench generates images from a
prompt, reads the visible stickers, and checks whether a real, reachable cube state could look like
that.

**Latest results:** [kvlxai.github.io/RubiksBench](https://kvlxai.github.io/RubiksBench/) (14 models, 3 images each, plus real-cube reference photos).

[![RubiksBench results](https://kvlxai.github.io/RubiksBench/preview.png)](https://kvlxai.github.io/RubiksBench/)

```sh
uv sync
uv run pytest
uv run rubiksbench validate observation.json          # check a hand-read cube (see Observation format)
uv run --env-file .env rubiksbench models "Flux 3"   # find model ids (--all includes LoRAs/video)
uv run rubiksbench models flux --backend huggingface  # Hugging Face models served by Inference Providers
uv run --env-file .env rubiksbench run configs/benchmark_runware.toml   # .env: RUNWARE_API_KEY / HF_TOKEN / FAL_KEY
uv run rubiksbench score runs/<run>                     # re-evaluate + re-score an existing run (no API calls)
uv run --env-file .env rubiksbench backfill runs/<run>  # resend a run's failed requests (same settings and seeds)
```

## Scoring

Every image is graded on 14 checks, in order. Each check earns 1 point when passed, ½ when
inconclusive and 0 when failed. A check also scores 0 ("blocked") when an earlier check it depends
on did not pass, so a box-shaped cube stops at the 3×3 check and isn't tested for solvability. The
checks and their dependencies live in `evaluation/rubric.py`, and every report lists them.

1. cube detected
2. cube shape
3. 3×3 grid
4. plain stickers (no letters or markings; a logo on a center is allowed)
5. at most six colors (fewer is fine: a color can be entirely on the hidden faces)
6. standard palette (every visible color is white, yellow, red, orange, green or blue)
7. standard color pairs (no piece joins white–yellow, red–orange or blue–green)
8. scrambled
9. distinct centers
10. standard center order
11. possible pieces
12. unique pieces
13. solvable
14. solvable (standard)

The HTML report shows each model's mean score with a 95% CI, plus the share of points earned per
check. The verdicts (`VALID_STANDARD`, …) are still reported next to the score.

## Benchmark file

```toml
backend = "runware"      # or "huggingface" (HF_TOKEN) or "fal" (FAL_KEY); see configs/
extractor = "vision"     # or "sam" / "annotation"; `score` reuses it, `--extractor` overrides
prompt = "A valid scrambled Rubik's Cube"
iterations = 20          # images per model; random seed per image unless `seed = N` is set

[models]
"FLUX 3 Image" = "bfl:flux@3-image"   # display name = model id
```

Models run one after another, one request at a time, and each image is scored as soon as it is
generated (the verdict is printed on its log line). Ctrl-C stops the run and still reports what was
generated so far. Each run directory holds `run.json`,
`generations.jsonl` (every attempt, including failures, with seed and the exact request),
`images/` and `evaluations.jsonl`. Every scoring pass (at the end of `run`, or via `score`) also writes
timestamped results:
- `results-<UTC>.json` holds the raw data.
- `results-<UTC>.html` is a self-contained report with light and dark themes. It has:
  - the per-model scores, each with an inline 95% CI strip and an outcome bar;
  - a points-per-check heatmap;
  - a gallery of every image with its per-check results and its extraction overlay;
  - the failure flags and the list of checks.

Models that reject a `seed` (e.g. OpenAI, FLUX 3 Image) are sent without one. Their samples are
marked `seed_supported: false`.

Sticker extraction is pluggable (`extractor` in the TOML, or `--extractor` to override):

- `vision` (default) reads three-face corner views automatically:
  1. `rembg` (U2-Net) cuts out the cube; the first use downloads its ~176 MB model.
  2. The silhouette's hexagon and its vanishing points give the three faces.
  3. OpenCV flattens each face and checks it is a 3×3 grid.
  4. Each sticker's color is named by its CIELAB hue (white by low chroma).

  Whether the image shows a Rubik's cube at all is decided first:
  - **With the `ml` extra** (`uv sync --extra ml`), CLIP zero-shot recognition decides. An image that
    clearly shows something else (an animal, a die, a plain box) is `INVALID` with `not_a_cube` and
    scores 0.
  - **Without it**, the cube's silhouette has to fill its hexagonal outline (`hexagon_fit` ≥ 0.93).
    Anything else is inconclusive.

  Anything uncertain gives `INCONCLUSIVE` (`vision_low_confidence`), never a guess. Examples: no
  hexagonal silhouette, an inconsistent layout, one odd face, or a color near a hue boundary.
  The extractor writes `images/<sample>.extraction.jpg` with every label it read.
- `sam` is a sticker-first reader built on Meta's SAM 2.1 (PyTorch, via `transformers`).
  1. SAM proposes a mask for every sticker.
  2. Each sticker's edge directions say which face it is on.
  3. Its position along those directions gives its row and column.

  It doesn't use the silhouette, so it is an independent cross-check. It is optional
  (`uv sync --extra ml`) and takes ~15 s per image.
- `annotation` reads a hand annotation `images/<sample>.json` (same format as below), for checking
  the vision extractor or scoring images it cannot read.

## Layout

- `cube/`: the partial-state validator. Kociemba facelet conventions; an exact search for a
  reachable completion under each candidate color scheme; witness scrambles from `kociemba`,
  replayed independently in `magiccube`.
- `evaluation/`: verdicts (`VALID_STANDARD`, `VALID_NONSTANDARD`, `INVALID`, `INCONCLUSIVE`) and flags.
- `backends/`: `base.py` defines `ImageBackend` and a registry. `runware.py` uses the official
  `runware-sdk`. `huggingface.py` uses `huggingface_hub` (Inference Providers). `fal.py` uses `fal-client` and reads each
  endpoint's published input schema, so it sends only the fields that model accepts (size, seed, steps). To add a provider, subclass `ImageBackend`, decorate it with `@register_backend`
  and import it in `backends/__init__.py`.
- `storage/`: run directories.
- `vision/`: extractors (image → observation): `cube_detector.py`, `grid_extractor.py`,
  `color_classifier.py`, `extractor.py`, `sam_extractor.py`.
- `runner.py` / `evaluation/scoring.py`: serial generation, evaluation and per-model scores
  (standard-valid rate with 95% Wilson CI, abstract/invalid/inconclusive rates, flags, cost).

## Observation format

A JSON object mapping faces to 9-sticker grids, read row by row, with `.` for unknown. For a
three-face view, the top face is `U`, front-left is `F` and front-right is `R`. Each grid's top
row is the edge nearest the U face (for U itself, the edge nearest the back).

## License

MIT, see [LICENSE](LICENSE).
