from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any
from track_utils import load_poses, rotation_matrix_to_angle_axis



def run(
    tracks_json: Path,
    poses_txt:    Path,
    out_dir:      Path     = Path("output/ceres_input"),
) -> Path:
    """
    Prepare Ceres solver input files (cameras.txt, points.txt, observations.txt)
    from cleaned tracks and camera poses.
    Returns the out_dir path.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    cameras_txt = out_dir / "cameras.txt"
    points_txt = out_dir / "points.txt"
    observations_txt = out_dir / "observations.txt"

    # load tracks
    with open(tracks_json, 'r') as f:
        tracks: Dict[str, List[Dict[str, Any]]] = json.load(f)

    # load extrinsics (camera-to-world)
    poses_c2w = load_poses(poses_txt)
    # convert to world-to-camera
    poses_w2c = [np.linalg.inv(T) for T in poses_c2w]

    bundle = []  # point observations
    for track_id, dets in tracks.items():
        centroids = []
        observations = []
        for det in dets:
            if "centroid_3d" not in det:
                continue
            centroids.append(det["centroid_3d"])
            x1,y1,x2,y2 = det["bbox"]
            u = (x1 + x2) / 2
            v = (y1 + y2) / 2
            observations.append({"frame": det["frame"], "uv": [u, v]})
        if not centroids or not observations:
            continue
        mean_xyz = np.mean(np.array(centroids), axis=0).tolist()
        bundle.append({"point_id": int(track_id),
                       "xyz": mean_xyz,
                       "observations": observations})

    # write cameras
    with open(cameras_txt, 'w') as f_cam:
        for T in poses_w2c:
            R = T[:3,:3]
            t = T[:3,3]
            aa = rotation_matrix_to_angle_axis(R)
            vec = list(aa) + list(t)
            f_cam.write(" ".join(map(str, vec)) + "\n")

    # write points
    with open(points_txt, 'w') as f_pts:
        for item in bundle:
            f_pts.write(" ".join(map(str, item["xyz"])) + "\n")

    # write observations
    with open(observations_txt, 'w') as f_obs:
        for pid, item in enumerate(bundle):
            for obs in item["observations"]:
                frame = obs["frame"]
                u,v = obs["uv"]
                f_obs.write(f"{frame} {pid} {u} {v}\n")

    print(f"✅ prepare_ceres_input wrote files to {out_dir}")
    return out_dir

# No CLI; import and call run() from your pipeline module
