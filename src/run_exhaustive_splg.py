"""SuperPoint + LightGlue SfM pipeline, exhaustive matching, via hloc.

Two matching-filter presets are available:
  strict (default) - LightGlue's default filtering thresholds.
  --lenient        - relaxed filtering (filter_threshold=0.01, depth/width_confidence=-1).
                      This is the configuration that produced the "hallucinated matches"
                      negative result documented in README.md (F-score drops vs. strict).

Reference pipeline for the classical-vs-learned-features benchmark (see README.md).
"""
import argparse
from pathlib import Path

import pycolmap
from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive

from common import setup_output_dir, run_dense_reconstruction, resolve_colmap_exe


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True,
                         help="Path to the input image directory (a single scene's photos)")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Output directory (default: outputs_exhaustive_splg_<scene_name>)")
    parser.add_argument("--colmap-exe", type=Path, default=None,
                         help="Path to the COLMAP executable, needed for the dense reconstruction "
                              "step (default: resolved from PATH)")
    parser.add_argument("--skip-dense", action="store_true",
                         help="Stop after sparse reconstruction, skip COLMAP dense reconstruction")
    parser.add_argument("--lenient", action="store_true",
                         help="Use relaxed LightGlue filtering instead of strict (reproduces the "
                              "negative 'hallucinated matches' result, see README.md)")
    return parser.parse_args()


def main():
    args = parse_args()
    images_path = args.images
    dataset_name = images_path.name
    outputs = setup_output_dir(args.output_dir or f"outputs_exhaustive_splg_{dataset_name}")

    sfm_pairs = outputs / "pairs-exhaustive.txt"
    sfm_dir = outputs / "sfm"
    features = outputs / "feats.h5"
    matches = outputs / "matches.h5"

    print("--- 1. Extracting SuperPoint Features ---")
    feature_conf = extract_features.confs['superpoint_aachen']
    feature_conf['preprocessing']['num_workers'] = 0
    feature_conf['preprocessing']['resize_max'] = 2048
    feature_conf['model']['max_keypoints'] = 8192
    extract_features.main(feature_conf, images_path, feature_path=features)

    print("--- 2. Generating Exhaustive Pairs ---")
    pairs_from_exhaustive.main(sfm_pairs, features=features)

    print(f"--- 3. Matching with LightGlue ({'lenient' if args.lenient else 'strict'} filtering) ---")
    matcher_conf = match_features.confs['superpoint+lightglue']
    matcher_conf['preprocessing'] = {'num_workers': 0}
    if args.lenient:
        matcher_conf['model']['filter_threshold'] = 0.01
        matcher_conf['model']['depth_confidence'] = -1
        matcher_conf['model']['width_confidence'] = -1
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    print("--- 4. Reconstruction ---")
    reconstruction.main(
        sfm_dir, images_path, sfm_pairs, features, matches,
        camera_mode=pycolmap.CameraMode.SINGLE,
    )
    print("Sparse reconstruction finished!")

    if args.skip_dense:
        return

    colmap_exe = resolve_colmap_exe(args.colmap_exe)
    if colmap_exe is None:
        raise SystemExit("COLMAP executable not found on PATH. Pass --colmap-exe or use --skip-dense.")

    run_dense_reconstruction(colmap_exe, images_path, sfm_dir, outputs)


if __name__ == "__main__":
    main()
