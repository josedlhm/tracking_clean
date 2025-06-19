from pathlib import Path
import subprocess

from new_tracks import run as build_tracks
from clean_and_flag_tracks import run as clean_tracks
from prepare_ceres_input import run as prepare_ceres
from track_utils import convert_ceres_poses
from deduplicate import run as deduplicate

# ─── DEFAULT PARAMETERS ─────────────────────────────────────────────────────
IOU_THRESH      = 0.25
OUTLIER_DELTA   = 60
MIN_TRACK_LEN   = 4
MIN_CONSECUTIVE = 4
VOXEL_SIZE      = 5
DBSCAN_EPS      = 60
DBSCAN_SAMPLES  = 10
INTRINSICS      = dict(fx=1272.44, fy=1272.67, cx=920.062, cy=618.949)

# Path to your Ceres solver binary
CERES_SOLVER_BIN = Path("/Users/josedlh/ceres_ba/build/bundle_adjustment")


def pipeline(
    depth_folder: Path,
    detections_json: Path,
    poses_file: Path,
    output_dir: Path = Path("output")
) -> Path:
    """
    Run full pipeline: 1) 3D tracking  2) cleanup  3) prepare Ceres  
    4) bundle-adjustment  5) deduplication.
    Returns path to merged-tracks JSON.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    #1) raw 3D tracks
    tracks_json = build_tracks(
        detections_json=detections_json,
        depth_dir=depth_folder,
        poses_file=poses_file,
        output_json=output_dir / "tracks.json",
        iou_thresh=IOU_THRESH,
        outlier_delta=OUTLIER_DELTA,
        min_track_len=MIN_TRACK_LEN,
        min_consecutive=MIN_CONSECUTIVE,
        **INTRINSICS
    )

    # 2) clean & flag
    cleaned_json = clean_tracks(
        tracks_json=tracks_json,
        poses_file=poses_file,
        output_json=output_dir / "tracks_cleaned.json",
        voxel_size=VOXEL_SIZE,
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_SAMPLES
    )


    # 3) prepare Ceres inputs
    ceres_dir = output_dir/"ceres"
    prepare_ceres(
        tracks_json=cleaned_json,
        poses_txt=poses_file,
        out_dir=ceres_dir
    )


    # 4) run Ceres bundle-adjustment
    refined_cams = ceres_dir/"refined_cameras.txt"
    subprocess.run(
    [
        str(CERES_SOLVER_BIN),
        "cameras.txt",
        "points.txt",
        "observations.txt",
        # optionally: str(d_mm) if you want to override the default 150.0
    ],
    cwd=str(ceres_dir),
    check=True
)
    print(f"✅ Ceres BA finished. Refined cameras at {ceres_dir/'refined_cameras.txt'}")

    # 5) deduplicate (conversion happens inside run())
    merged = deduplicate(
        tracks_json=cleaned_json,
        ceres_poses=refined_cams,
        converted_ceres_poses=poses_file.with_name("refined_poses_mm_yup.txt"),
        ceres_points=ceres_dir/"refined_points.txt",
        output_json=output_dir/"merged_tracks.json"
    )

    return merged

if __name__ == "__main__":
    # Define input paths here (no CLI for now)
    depth_folder     = Path("data/depth")
    detections_json  = Path("data/detections.json")
    poses_file       = Path("data/poses/poses_mm_yup.txt")
    output_dir       = Path("output")

    merged_json = pipeline(
        depth_folder=depth_folder,
        detections_json=detections_json,
        poses_file=poses_file,
        output_dir=output_dir
    )
    print(f"Pipeline complete. Merged tracks JSON at: {merged_json}")
