from pathlib import Path
from new_tracks import run as build_tracks
from clean_and_flag_tracks import run as clean_tracks
from prepare_ceres_input import run as prepare_ceres

# ─── DEFAULT PARAMETERS ─────────────────────────────────────────────────────
IOU_THRESH      = 0.25
OUTLIER_DELTA   = 60
MIN_TRACK_LEN   = 4
MIN_CONSECUTIVE = 4
VOXEL_SIZE      = 5
DBSCAN_EPS      = 60
DBSCAN_SAMPLES  = 10
INTRINSICS      = dict(fx=1272.44, fy=1272.67, cx=920.062, cy=618.949)


def pipeline(
    depth_folder: Path,
    detections_json: Path,
    poses_file: Path,
    output_dir: Path = Path("output")
) -> Path:
    """
    Run full pipeline: 3D tracking → cleanup → Ceres prep.
    Returns path to cleaned-tracks JSON.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # # 1) raw 3D tracks
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
        min_samples=DBSCAN_SAMPLES,
    )

    # 3) prepare Ceres inputs
    ceres_dir = output_dir / "ceres_input"
    prepare_ceres(
        tracks_json=cleaned_json,
        poses_txt=poses_file,
        out_dir=ceres_dir
    )

    return cleaned_json

if __name__ == "__main__":
    # Define input paths here (no CLI for now)
    depth_folder     = Path("data/depth")
    detections_json  = Path("data/detections.json")
    poses_file       = Path("data/poses/poses_mm_yup.txt")
    output_dir       = Path("output")

    cleaned_json = pipeline(
        depth_folder=depth_folder,
        detections_json=detections_json,
        poses_file=poses_file,
        output_dir=output_dir
    )
    print(f"Pipeline complete. Cleaned tracks JSON at: {cleaned_json}")

