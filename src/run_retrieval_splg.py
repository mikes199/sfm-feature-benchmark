"""SuperPoint + LightGlue SfM pipeline, NetVLAD-retrieval-based pair selection, via hloc.

The winning configuration in the classical-vs-learned-features benchmark (see
README.md): global NetVLAD retrieval prunes candidate image pairs before
matching, then strict LightGlue filtering rejects bad local matches.
"""
import argparse
from pathlib import Path

import pycolmap
from hloc import extract_features, match_features, reconstruction, pairs_from_retrieval

from common import setup_output_dir, run_dense_reconstruction, resolve_colmap_exe


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True,
                         help="Path to the input image directory (a single scene's photos)")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Output directory (default: outputs_retrieval_splg_<scene_name>)")
    parser.add_argument("--colmap-exe", type=Path, default=None,
                         help="Path to the COLMAP executable, needed for the dense reconstruction "
                              "step (default: resolved from PATH)")
    parser.add_argument("--skip-dense", action="store_true",
                         help="Stop after sparse reconstruction, skip COLMAP dense reconstruction")
    parser.add_argument("--num-matched", type=int, default=50,
                         help="Number of top visually-similar images to retrieve per image (default: 50)")
    return parser.parse_args()


def main():
    args = parse_args()
    images_path = args.images
    dataset_name = images_path.name
    outputs = setup_output_dir(args.output_dir or f"outputs_retrieval_splg_{dataset_name}")

    sfm_pairs = outputs / "pairs-retrieval.txt"
    sfm_dir = outputs / "sfm"
    features = outputs / "feats.h5"
    matches = outputs / "matches.h5"
    global_feats = outputs / "global-feats.h5"

    print("--- 1. Extracting Local (SuperPoint) & Global (NetVLAD) Features ---")
    feature_conf = extract_features.confs['superpoint_aachen']
    feature_conf['preprocessing']['num_workers'] = 0
    feature_conf['preprocessing']['resize_max'] = 2048
    feature_conf['model']['max_keypoints'] = 8192
    extract_features.main(feature_conf, images_path, feature_path=features)

    retrieval_conf = extract_features.confs['netvlad']
    extract_features.main(retrieval_conf, images_path, feature_path=global_feats)

    print("--- 2. Generating Pairs via Retrieval ---")
    num_images = len([p for p in images_path.iterdir() if p.is_file()])
    actual_num_matched = min(args.num_matched, num_images - 1)
    pairs_from_retrieval.main(
        descriptors=global_feats,
        output=sfm_pairs,
        num_matched=actual_num_matched,
    )

    print("--- 3. Matching with LightGlue (strict filtering) ---")
    matcher_conf = match_features.confs['superpoint+lightglue']
    matcher_conf['preprocessing'] = {'num_workers': 0}
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
