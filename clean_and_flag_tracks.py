#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TRACKS_JSON = Path("output/tracks.json")
POSES_FILE = Path("data/poses/poses_mm_yup.txt")
OUTPUT_JSON = Path("output/tracks_cleaned.json")

# ─── INTRINSICS ───────────────────────────────────────────────────────────────
FX, FY = 1272.44, 1272.67
CX, CY = 920.062, 618.949

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def load_poses(path):
    Ts = []
    with open(path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            R = np.array([vals[0:3], vals[4:7], vals[8:11]])
            t = np.array([vals[3], vals[7], vals[11]])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            Ts.append(T)
    return Ts

def world_to_camera(points_world, T_wc):
    T_cw = np.linalg.inv(T_wc)
    R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
    return (R_cw @ points_world.T).T + t_cw

def voxel_downsample(points, voxel_size=5):
    if len(points) == 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int32)
    unique, inverse = np.unique(coords, axis=0, return_inverse=True)
    down_pts = np.zeros((len(unique), 3), dtype=np.float32)
    for i in range(len(unique)):
        down_pts[i] = points[inverse == i].mean(axis=0)
    return down_pts

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    poses = load_poses(POSES_FILE)
    with open(TRACKS_JSON) as f:
        tracks = json.load(f)

    new_tracks = {}
    flagged = []

    for tid, detections in tracks.items():
        all_pts_world = []
        detection_meta = []

        for d in detections:
            if "mask_points_3d" in d:
                pts_world = np.array(d["mask_points_3d"])
                if pts_world.shape[0] > 0:
                    all_pts_world.append(pts_world)
                    detection_meta.append((d, pts_world))

        if not all_pts_world:
            continue

        all_pts_world = np.vstack(all_pts_world)
        if all_pts_world.shape[0] < 10:
            new_tracks[tid] = detections
            continue

        all_pts_world = voxel_downsample(all_pts_world, voxel_size=5)
        db = DBSCAN(eps=60, min_samples=10)
        labels = db.fit_predict(all_pts_world)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            flagged.append((tid, n_clusters, len(all_pts_world)))

        if np.max(labels) < 0 or n_clusters <= 1:
            new_tracks[tid] = detections
            continue

        # Compute cluster centers
        cluster_means = {}
        for lbl in np.unique(labels[labels >= 0]):
            cluster_pts = all_pts_world[labels == lbl]
            cluster_means[lbl] = cluster_pts.mean(axis=0)

        # Assign detections to nearest cluster mean
        cluster_to_dets = {}
        for d, pts in detection_meta:
            centroid = pts.mean(axis=0)
            best_lbl = min(cluster_means, key=lambda l: np.linalg.norm(centroid - cluster_means[l]))
            cluster_to_dets.setdefault(best_lbl, []).append(d)

        # Select best cluster
        best_lbl = None
        best_z = float("inf")
        for lbl, dets in cluster_to_dets.items():
            if len(dets) >= 4:
                frame = dets[0]["frame"]
                cam_pts = world_to_camera(np.array([d["centroid_3d"] for d in dets]), poses[frame])
                z_mean = cam_pts[:, 2].mean()
                if z_mean < best_z:
                    best_z = z_mean
                    best_lbl = lbl

        if best_lbl is not None:
            new_tracks[tid] = cluster_to_dets[best_lbl]

    contiguous_tracks = {}
    for new_id, dets in enumerate(new_tracks.values()):
        for d in dets:
            d["det"] = new_id  # Optional: update detection's internal track label
        contiguous_tracks[str(new_id)] = dets

    with open(OUTPUT_JSON, "w") as f:
        json.dump(contiguous_tracks, f, indent=2)

    print(f"\nSaved cleaned tracks to {OUTPUT_JSON}")
    print(f"Final track count: {len(contiguous_tracks)}")
    print(f"Found {len(flagged)} tracks with multiple clusters:")
    for tid, n_clusters, num_pts in flagged:
        print(f" • Track {tid:>4}: {n_clusters} clusters, {num_pts} downsampled points")

if __name__ == "__main__":
    main()
