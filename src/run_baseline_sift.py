"""Baseline SfM pipeline: classical SIFT features + exhaustive matching, via pycolmap.

Reference pipeline for the classical-vs-learned-features benchmark (see README.md).
"""
import argparse
from pathlib import Path

import pycolmap

from common import setup_output_dir, run_dense_reconstruction, resolve_colmap_exe


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True,
                         help="Path to the input image directory (a single scene's photos)")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Output directory (default: outputs_baseline_sift_<scene_name>)")
    parser.add_argument("--colmap-exe", type=Path, default=None,
                         help="Path to the COLMAP executable, needed for the dense reconstruction "
                              "step (default: resolved from PATH)")
    parser.add_argument("--skip-dense", action="store_true",
                         help="Stop after sparse reconstruction, skip COLMAP dense reconstruction")
    return parser.parse_args()


def main():
    args = parse_args()
    images_path = args.images
    dataset_name = images_path.name
    outputs = setup_output_dir(args.output_dir or f"outputs_baseline_sift_{dataset_name}")

    database_path = outputs / "database.db"
    sfm_dir = outputs / "sfm"
    sfm_dir.mkdir(parents=True, exist_ok=True)

    print("--- 1. Extracting SIFT Features ---")
    pycolmap.extract_features(database_path, images_path)

    print("--- 2. Exhaustive Matching ---")
    pycolmap.match_exhaustive(database_path)

    print("--- 3. Reconstruction ---")
    pipeline_options = pycolmap.IncrementalPipelineOptions()
    pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=images_path,
        output_path=sfm_dir,
        options=pipeline_options,
    )
    print("Sparse reconstruction finished!")

    if args.skip_dense:
        return

    colmap_exe = resolve_colmap_exe(args.colmap_exe)
    if colmap_exe is None:
        raise SystemExit("COLMAP executable not found on PATH. Pass --colmap-exe or use --skip-dense.")

    best_model_path = sfm_dir / "0"
    run_dense_reconstruction(colmap_exe, images_path, best_model_path, outputs)


if __name__ == "__main__":
    main()
