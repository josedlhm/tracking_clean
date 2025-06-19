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
    Criteria exactly matches the standalone script:
    - Drop tracks with no mask_points_3d
    - Keep tracks with <10 total points unchanged
    - Voxel-downsample points then DBSCAN(eps, min_samples)
    - Flag tracks with more than one cluster
    - If noise-only or single cluster, keep original detections
    - Otherwise assign detections to nearest cluster, select clusters with >=4 detections,
      pick the one with lowest mean Z in camera space
    - Drop tracks if no cluster has >=4 detections
    - Reassign contiguous IDs and write output
    """
    poses = load_poses(poses_file)
    with open(tracks_json, 'r') as f:
        tracks: Dict[str, List[Dict[str, Any]]] = json.load(f)

    new_tracks: Dict[str, List[Dict[str, Any]]] = {}
    flagged: List[Tuple[str, int, int]] = []

    for tid, detections in tracks.items():
        # Collect all world points and meta
        all_pts_world = []
        detection_meta: List[Tuple[Dict[str, Any], np.ndarray]] = []
        for d in detections:
            pts_world = np.array(d.get("mask_points_3d", []), dtype=float)
            if pts_world.size > 0:
                all_pts_world.append(pts_world)
                detection_meta.append((d, pts_world))

        # Drop if no 3D points
        if not all_pts_world:
            continue

        # Stack and check small tracks
        all_pts_world = np.vstack(all_pts_world)
        if all_pts_world.shape[0] < 10:
            new_tracks[tid] = detections
            continue

        # Downsample then DBSCAN
        all_pts_world = voxel_downsample(all_pts_world, voxel_size=voxel_size)
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(all_pts_world)

        # Count clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            flagged.append((tid, n_clusters, all_pts_world.shape[0]))

        # Keep original if only noise or one cluster
        if np.max(labels) < 0 or n_clusters <= 1:
            new_tracks[tid] = detections
            continue

        # Compute means of each labeled cluster
        cluster_means: Dict[int, np.ndarray] = {}
        for lbl in np.unique(labels[labels >= 0]):
            pts_lbl = all_pts_world[labels == lbl]
            cluster_means[lbl] = pts_lbl.mean(axis=0)

        # Assign detections to nearest cluster mean
        cluster_to_dets: Dict[int, List[Dict[str, Any]]] = {}
        for d, pts in detection_meta:
            centroid = pts.mean(axis=0)
            best_lbl = min(cluster_means, key=lambda l: np.linalg.norm(centroid - cluster_means[l]))
            cluster_to_dets.setdefault(best_lbl, []).append(d)

        # Select the best cluster: len >=4 and lowest mean Z
        best_lbl = None
        best_z = float('inf')
        for lbl, dets in cluster_to_dets.items():
            if len(dets) >= 4:
                frame0 = dets[0]["frame"]
                cam_pts = world_to_camera(
                    np.vstack([d.get("centroid_3d", [0,0,0]) for d in dets]),
                    poses[frame0]
                )
                zmean = cam_pts[:, 2].mean()
                if zmean < best_z:
                    best_z = zmean
                    best_lbl = lbl

        # Keep only if a valid cluster found
        if best_lbl is not None:
            new_tracks[tid] = cluster_to_dets[best_lbl]

    # Reassign contiguous track IDs
    contiguous: Dict[str, List[Dict[str, Any]]] = {}
    for new_id, dets in enumerate(new_tracks.values()):
        for d in dets:
            d["det"] = new_id
        contiguous[str(new_id)] = dets

    # Write output and print summary
    with open(output_json, 'w') as f:
        json.dump(contiguous, f, indent=2)

    print(f"\nSaved cleaned tracks to {output_json}")
    print(f"Final track count: {len(contiguous)}")
    print(f"Found {len(flagged)} tracks with multiple clusters:")
    for tid, n_clusters, num_pts in flagged:
        print(f" • Track {tid:>4}: {n_clusters} clusters, {num_pts} downsampled points")

    return output_json
