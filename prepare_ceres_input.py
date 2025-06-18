#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as SciRot

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TRACKS_JSON = Path("output/tracks_cleaned.json")
POSES_TXT = Path("data/poses/poses_mm_yup.txt")  # camera-to-world
OUT_DIR = Path("output/ceres_input/")
OUT_DIR.mkdir(exist_ok=True)

CAMERAS_TXT = OUT_DIR / "cameras.txt"
POINTS_TXT = OUT_DIR / "points.txt"
OBSERVATIONS_TXT = OUT_DIR / "observations.txt"

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def load_camera_extrinsics(path):
    Ts = []
    with open(path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            R = np.array([
                vals[0:3],
                vals[4:7],
                vals[8:11]
            ])
            t = np.array([vals[3], vals[7], vals[11]])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            Ts.append(T)
    return Ts

def rotation_matrix_to_angle_axis(R):
    return SciRot.from_matrix(R).as_rotvec()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    with open(TRACKS_JSON) as f:
        tracks = json.load(f)

    # Load camera-to-world and convert to world-to-camera
    poses_c2w = load_camera_extrinsics(POSES_TXT)
    poses_w2c = []
    for T_c2w in poses_c2w:
        T_w2c = np.linalg.inv(T_c2w)
        poses_w2c.append(T_w2c)

    bundle_data = []

    for track_id, detections in tracks.items():
        centroids = []
        observations = []

        for det in detections:
            if "centroid_3d" not in det:
                continue
            centroids.append(det["centroid_3d"])

            x1, y1, x2, y2 = det["bbox"]
            u = (x1 + x2) / 2
            v = (y1 + y2) / 2

            observations.append({
                "frame": det["frame"],
                "uv": [u, v]
            })

        if not centroids or not observations:
            continue

        centroid_mean = np.mean(np.array(centroids), axis=0)
        bundle_data.append({
            "point_id": int(track_id),
            "xyz": centroid_mean.tolist(),
            "observations": observations
        })

    # ─── Write Ceres-Compatible Files ─────────────────────────────────────────
    with open(CAMERAS_TXT, "w") as f:
        for T in poses_w2c:
            R = T[:3, :3]
            t = T[:3, 3]
            aa = rotation_matrix_to_angle_axis(R)
            f.write(" ".join(map(str, list(aa) + list(t))) + "\n")

    with open(POINTS_TXT, "w") as f:
        for item in bundle_data:
            f.write(" ".join(map(str, item["xyz"])) + "\n")

    with open(OBSERVATIONS_TXT, "w") as f:
        for i, item in enumerate(bundle_data):
            for obs in item["observations"]:
                frame = obs["frame"]
                u, v = obs["uv"]
                f.write(f"{frame} {i} {u} {v}\n")

    print(f"✅ Wrote Ceres input files to {OUT_DIR}/")

if __name__ == "__main__":
    main()
