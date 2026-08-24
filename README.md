# SfM Feature Benchmark: Classical SIFT vs. Learned Features

Benchmarking classical (SIFT) vs. learned (SuperPoint + LightGlue) local
features for Structure-from-Motion reconstruction accuracy, evaluated against
laser-scanned ground truth on the [Tanks & Temples](https://www.tanksandtemples.org/)
benchmark.

> **TODO before publishing:** add a result image/GIF here (reconstruction
> screenshot or SIFT-vs-retrieval-SPLG comparison) — see
> [`results/images/TODO.md`](results/images/TODO.md).

## Method

Three pipeline configurations reconstruct the same scene (Tanks & Temples
`Truck`) through [COLMAP](https://colmap.github.io/)'s incremental SfM, differing
only in feature extraction and matching: classical SIFT with exhaustive
matching (COLMAP's default), SuperPoint+LightGlue with exhaustive matching,
and SuperPoint+LightGlue with NetVLAD-retrieval-based pair selection. Local
features and learned matching are provided by
[hloc](https://github.com/cvg/Hierarchical-Localization); the resulting dense
point clouds are scored with the official Tanks & Temples evaluation
toolbox, which registers the reconstruction against a laser scan and reports
precision, recall, and F-score at a scene-specific distance threshold.

## Results (Truck scene)

| Pipeline | F-score |
|---|---|
| Classical SIFT (baseline) | 21.34% |
| SuperPoint+LightGlue, exhaustive, strict filtering | 22.31% |
| SuperPoint+LightGlue, exhaustive, lenient filtering | 18.84% |
| NetVLAD retrieval + SuperPoint+LightGlue, strict filtering | **23.12%** (winner) |

Machine-readable version: [`results/results.csv`](results/results.csv).

### Why more matching made things worse

The lenient-filtering row is the interesting result. Relaxing LightGlue's
confidence thresholds (`filter_threshold=0.01`, `depth_confidence=-1`,
`width_confidence=-1`) forces more matches through instead of rejecting
ambiguous ones — and reconstruction quality *drops below the classical SIFT
baseline*, to 18.84%. Forcing matches on low-confidence pairs poisons the
view graph with false correspondences, which bundle adjustment then has to
reconcile by distorting the geometry.

Fixing this needs outlier rejection at two levels, not one: strict local
filtering (LightGlue's own confidence thresholds) recovers most of the loss
(22.31%), but the best result comes from also filtering *globally* —
NetVLAD retrieval prunes visually-dissimilar image pairs before matching is
even attempted, so LightGlue never sees the ambiguous pairs that caused the
hallucinations in the first place (23.12%). Reproduce the failure mode
directly with `--lenient` on `run_exhaustive_splg.py` (see below).

## Tech stack

COLMAP / pycolmap, [hloc](https://github.com/cvg/Hierarchical-Localization),
SuperPoint, LightGlue, NetVLAD, Open3D, Python.

## Repo structure

```
src/                   authored pipeline scripts (this project's own code)
  common.py              shared output-dir / COLMAP dense-reconstruction helpers
  run_baseline_sift.py   classical SIFT + exhaustive matching
  run_exhaustive_splg.py SuperPoint+LightGlue + exhaustive matching (--lenient flag)
  run_retrieval_splg.py  SuperPoint+LightGlue + NetVLAD retrieval (winning config)
results/               results.csv + result images
third_party/SETUP.md   how to get hloc and the Tanks & Temples toolbox
```

`run.py`/evaluation and dataset-download scripts are **not** duplicated in
this repo — they're part of the official Tanks & Temples toolbox and are run
directly from a local clone of it, see `third_party/SETUP.md`.

## Setup & running

```bash
pip install -r requirements.txt
```

Then follow [`third_party/SETUP.md`](third_party/SETUP.md) to install hloc
and clone the Tanks & Temples toolbox (needed for the dataset and for
evaluation) — it also has a full worked example. Once set up, each pipeline
runs as:

```bash
python src/run_baseline_sift.py    --images path/to/Truck/images --colmap-exe path/to/COLMAP.bat
python src/run_exhaustive_splg.py  --images path/to/Truck/images --colmap-exe path/to/COLMAP.bat [--lenient]
python src/run_retrieval_splg.py   --images path/to/Truck/images --colmap-exe path/to/COLMAP.bat
```

Each writes sparse + dense reconstructions to `outputs_<pipeline>_<scene>/`.
Pass `--skip-dense` to stop after sparse reconstruction (no COLMAP binary
needed). Run `--help` on any script for the full flag list.

## Acknowledgments

- [hloc / Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization) (Sarlin et al.) — feature extraction, learned matching, and the hloc→COLMAP reconstruction glue. Apache-2.0.
- [Tanks & Temples](https://www.tanksandtemples.org/) evaluation toolbox (Knapitsch et al.) — dataset, ground-truth alignment, and precision/recall/F-score evaluation. MIT.
- [COLMAP](https://colmap.github.io/) (Schönberger & Frahm) — the underlying SfM/MVS engine driving all three pipelines.

## License

MIT — see [LICENSE](LICENSE). Vendored dependencies (hloc, Tanks & Temples
toolbox) are not included in this repo and carry their own licenses, linked
above.
