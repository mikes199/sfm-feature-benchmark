# Third-party dependencies

This benchmark relies on two external toolboxes that are **not vendored** in
this repo. Clone them locally (anywhere outside this repo, or here — both are
already git-ignored) before running anything in `src/`.

## 1. hloc (Hierarchical Localization) — SuperPoint, LightGlue, NetVLAD, SfM glue

hloc provides the learned feature extractors/matchers used by
`run_exhaustive_splg.py` and `run_retrieval_splg.py`, plus the glue code that
feeds them into COLMAP's SfM reconstruction (`hloc.reconstruction`).

```bash
git clone --recursive https://github.com/cvg/Hierarchical-Localization.git
cd Hierarchical-Localization
python -m pip install -e .
```

`torch`/`torchvision` are transitive dependencies of hloc and are
GPU/CUDA-version specific — install the right build from
https://pytorch.org/get-started/locally/ *before* the `pip install -e .` step
if you need CUDA support. `--recursive` matters: hloc vendors its own
matcher/extractor submodules under `third_party/`.

**If you're using `pycolmap-cuda12` (see README), pin torch to a CUDA 12.x
build explicitly** -- a plain `pip install torch torchvision` resolves to the
newest CUDA line available (CUDA 13 as of this writing), which drags in
`cuda-toolkit>=13` as a transitive dependency and silently *downgrades* the
CUDA 12.x toolkit `pycolmap-cuda12` needs (`cuda-toolkit<13,>=12`), breaking
it:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install --force-reinstall --no-deps pycolmap-cuda12   # restore cuda-toolkit<13 if it got bumped
```
Verify both after any torch/pycolmap install or reinstall:
```bash
python3 -c "import torch, pycolmap; print(torch.cuda.is_available(), pycolmap.has_cuda)"
```

Once installed (editable install via `-e .`), `from hloc import ...` resolves
from any Python environment where it was installed — no need to manually add
it to `PYTHONPATH`.

Licensed under Apache-2.0 — see the hloc repo for full terms.

## 2. Tanks & Temples evaluation toolbox — dataset download + ground-truth evaluation

```bash
git clone https://github.com/isl-org/TanksAndTemples.git
```

This provides everything needed outside of the SfM pipeline itself:

- `python_toolbox/download_t2_dataset.py` — downloads the benchmark scenes
  (training/intermediate/advanced).
- `python_toolbox/convert_to_logfile.py` — converts a COLMAP sparse
  reconstruction into the trajectory-log format the evaluator expects:
  ```bash
  python convert_to_logfile.py <sfm_dir>/0/ <scene>.log <images_dir>/ COLMAP jpg
  ```
- `python_toolbox/evaluation/run.py` — computes precision/recall/F-score of a
  reconstruction against the laser-scanned ground truth. This is what
  produced the numbers in `results/results.csv`. It needs, alongside your
  reconstruction, the official per-scene reference files (`<scene>_COLMAP_SfM.log`,
  `<scene>_trans.txt`, `<scene>.ply`, `<scene>.json`) that ship inside the
  toolbox under `python_toolbox/evaluation/<scene>/` — see the toolbox's own
  README for the exact layout it expects in `--dataset-dir`.
  ```bash
  python python_toolbox/evaluation/run.py \
      --dataset-dir path/to/evaluation/Truck \
      --traj-path path/to/Truck.log \
      --ply-path path/to/outputs_retrieval_splg_Truck/mvs/fused.ply \
      --out-dir path/to/outputs_retrieval_splg_Truck/evaluation
  ```

Licensed under MIT — see the toolbox repo for full terms.

## End-to-end example (retrieval pipeline, Truck scene)

```bash
# from this repo's root, with both toolboxes cloned as siblings and installed
python src/run_retrieval_splg.py \
    --images path/to/TanksAndTemples/Truck/images \
    --colmap-exe path/to/COLMAP.bat   # or colmap, if on PATH

python path/to/TanksAndTemples/python_toolbox/convert_to_logfile.py \
    outputs_retrieval_splg_Truck/sfm/0/ Truck.log \
    path/to/TanksAndTemples/Truck/images/ COLMAP jpg

python path/to/TanksAndTemples/python_toolbox/evaluation/run.py \
    --dataset-dir path/to/evaluation/Truck \
    --traj-path Truck.log \
    --ply-path outputs_retrieval_splg_Truck/mvs/fused.ply \
    --out-dir outputs_retrieval_splg_Truck/evaluation
```
