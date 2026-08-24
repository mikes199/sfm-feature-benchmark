"""Shared helpers for the SfM pipeline scripts in this repo."""
import shutil
import subprocess
from pathlib import Path


def setup_output_dir(name) -> Path:
    """Create a fresh (emptied) output directory for a pipeline run."""
    outputs = Path(name)
    if outputs.exists():
        shutil.rmtree(outputs, ignore_errors=True)
    outputs.mkdir(parents=True, exist_ok=True)
    return outputs


def resolve_colmap_exe(colmap_exe_arg):
    """Resolve the COLMAP executable: explicit --colmap-exe arg, else PATH lookup."""
    if colmap_exe_arg is not None:
        return colmap_exe_arg
    found = shutil.which("colmap") or shutil.which("COLMAP")
    return Path(found) if found else None


def run_dense_reconstruction(colmap_exe: Path, images_path: Path, sparse_model_path: Path, outputs: Path) -> Path:
    """Run COLMAP's dense reconstruction (undistort -> PatchMatch stereo -> stereo fusion).

    Returns the path to the fused dense point cloud (fused.ply).
    """
    mvs_workspace = outputs / "mvs"
    mvs_workspace.mkdir(parents=True, exist_ok=True)
    fused_ply_path = mvs_workspace / "fused.ply"

    print("--- Undistorting Images ---")
    subprocess.run([
        str(colmap_exe), "image_undistorter",
        "--image_path", str(images_path),
        "--input_path", str(sparse_model_path),
        "--output_path", str(mvs_workspace),
    ], check=True)

    print("--- PatchMatch Stereo ---")
    subprocess.run([
        str(colmap_exe), "patch_match_stereo",
        "--workspace_path", str(mvs_workspace),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
    ], check=True)

    print("--- Stereo Fusion ---")
    subprocess.run([
        str(colmap_exe), "stereo_fusion",
        "--workspace_path", str(mvs_workspace),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(fused_ply_path),
    ], check=True)

    print(f"SUCCESS! Dense point cloud saved to: {fused_ply_path}")
    return fused_ply_path
