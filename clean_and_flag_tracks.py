from pathlib import Path
import json
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Dict, Any
from track_utils import load_poses, world_to_camera, voxel_downsample

def run(
    tracks_json: Path,
    poses_file:  Path,
    output_json: Path,
    voxel_size:  float = 5.0,
    eps:         float = 60.0,
    min_samples: int   = 10
) -> Path:
    """
    Clean and flag 3D tracks via DBSCAN.
    Returns path to cleaned tracks JSON.
    """
    poses = load_poses(poses_file)
    with open(tracks_json, 'r') as f:
        raw_tracks: Dict[str, List[Dict[str, Any]]] = json.load(f)

    new_tracks: Dict[str, Any] = {}
    flagged: List[Tuple[str, int, int]] = []

    for tid, dets in raw_tracks.items():
        all_pts = []
        meta: List[Tuple[Dict[str, Any], np.ndarray]] = []

        for d in dets:
            pts = np.array(d.get("mask_points_3d", []), dtype=float)
            if pts.size:
                all_pts.append(pts)
                meta.append((d, pts))

        if not all_pts:
            continue

        stacked = np.vstack(all_pts)
        if stacked.shape[0] < 10:
            new_tracks[tid] = dets
            continue

        down = voxel_downsample(stacked, voxel_size)
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(down)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            flagged.append((tid, n_clusters, down.shape[0]))

        if np.max(labels) < 0 or n_clusters <= 1:
            new_tracks[tid] = dets
            continue

        # compute cluster means
        means = {lbl: down[labels == lbl].mean(axis=0)
                 for lbl in set(labels) if lbl >= 0}

        # assign each detection to nearest cluster
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        for d, pts in meta:
            center = pts.mean(axis=0)
            best = min(means, key=lambda l: np.linalg.norm(center - means[l]))
            clusters.setdefault(best, []).append(d)

        # choose cluster with closest mean Z
        best_lbl = None
        best_z = float('inf')
        for lbl, group in clusters.items():
            if len(group) >= min_samples:
                frame0 = group[0]["frame"]
                pts_cam = world_to_camera(
                    np.vstack([g.get("centroid_3d", [0,0,0]) for g in group]),
                    poses[frame0]
                )
                zmean = pts_cam[:, 2].mean()
                if zmean < best_z:
                    best_z, best_lbl = zmean, lbl

        if best_lbl is not None:
            new_tracks[tid] = clusters[best_lbl]

    # reassign contiguous IDs
    final: Dict[str, List[Dict[str, Any]]] = {}
    for new_id, dets in enumerate(new_tracks.values()):
        for d in dets:
            d["det"] = new_id
        final[str(new_id)] = dets

    with open(output_json, 'w') as f:
        json.dump(final, f, indent=2)

    print(f"✅ clean_and_flag_tracks wrote {output_json}")
    if flagged:
        print(f"Flagged {len(flagged)} tracks with multiple clusters:")
        for tid, n, pts in flagged:
            print(f" • Track {tid}: {n} clusters, {pts} points")

    return output_json

# No CLI; import and call run() from your pipeline
