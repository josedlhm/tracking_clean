#!/usr/bin/env python3
"""
deduplicate.py

Merge duplicate fruit tracks **after bundle-adjustment** when they refer to the
same physical fruit, with correct v-axis sign inversion in projection.
"""

from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any, Set, Tuple
from tqdm import tqdm

from track_utils import (
    load_poses,
    world_to_camera,
    draw_sphere_mask,
    mask_iou,
    convert_ceres_poses
)

# ── THRESHOLDS & INTRINSICS ─────────────────────────────────────────────────
DIST_MM    = 50.0      # max centroid distance (mm)
IOU_MIN    = 0.1       # min mask IoU to accept merge
FRUIT_R_MM = 40.0      # synthetic fruit radius (mm)

FX, FY     = 1272.44, 1272.67
CX, CY     = 920.062, 618.949
IMG_W, IMG_H = 1920, 1200
# ────────────────────────────────────────────────────────────────────────────

def project_point(
    point_w: np.ndarray,
    T_wc: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> Tuple[float, float, float]:
    """
    Project one 3D world point into pixel coords (u,v) with the correct
    v-axis flip: v = -fy*(y_cam/z)+cy.
    Returns (u, v, z_cam).
    """
    x_cam, y_cam, z_cam = world_to_camera(point_w[None, :], T_wc)[0]
    u = fx * (x_cam / z_cam) + cx
    v = -fy * (y_cam / z_cam) + cy
    return u, v, z_cam


def run(
    tracks_json: Path,
    ceres_poses: Path,
    converted_ceres_poses: Path,
    ceres_points: Path,
    output_json: Path
) -> Path:
    """
    1) Convert and load camera-to-world poses from Ceres output.
    2) Load refined 3D centroids.
    3) Load and merge tracks based on 3D distance, temporal disjointness,
       and mask IoU (with v-axis flip).
    4) Write merged JSON and return its path.

    Parameters:
      converted_ceres_poses: path where convert_ceres_poses writes the intermediate poses.
    """
    # Convert & load poses
    convert_ceres_poses(ceres_poses, converted_ceres_poses)
    Ts = load_poses(converted_ceres_poses)  # list of 4×4 numpy arrays

    # Load refined points
    points_refined = np.loadtxt(ceres_points)  # shape (P,3)

    # Load raw tracks
    with open(tracks_json, 'r') as f:
        raw = json.load(f)

    # Build track objects
    tracks: List[Dict[str, Any]] = []
    for tid_str, dets in raw.items():
        tid = int(tid_str)
        centroid_w = points_refined[tid]
        frames: List[int] = []
        masks_xy: Dict[int, np.ndarray] = {}
        for d in dets:
            f = d['frame']
            frames.append(f)
            mask = np.zeros((IMG_H, IMG_W), dtype=bool)
            for x, y in d['mask_coords']:
                mask[y, x] = True
            masks_xy[f] = mask
        tracks.append({
            'id': tid,
            'centroid_w': centroid_w,
            'frames': frames,
            'masks_xy': masks_xy,
            'detections': dets
        })

    # Deduplicate
    used: Set[int] = set()
    merged: List[Dict[str, Any]] = []
    for i, ti in enumerate(tqdm(tracks, desc='Merging tracks')):
        if i in used:
            continue
        for j in range(i + 1, len(tracks)):
            if j in used:
                continue
            tj = tracks[j]
            # 3D distance check
            if np.linalg.norm(ti['centroid_w'] - tj['centroid_w']) > DIST_MM:
                continue
            # temporal disjointness
            if set(ti['frames']) & set(tj['frames']):
                continue
            # IoU in first frame
            f0 = ti['frames'][0]
            T_wc = Ts[f0]
            u_i, v_i, _ = project_point(ti['centroid_w'], T_wc, FX, FY, CX, CY)
            u_j, v_j, z_j = project_point(tj['centroid_w'], T_wc, FX, FY, CX, CY)
            # bounds check
            if not (0 <= u_i < IMG_W and 0 <= v_i < IMG_H and
                    0 <= u_j < IMG_W and 0 <= v_j < IMG_H):
                continue
            mask_i = ti['masks_xy'][f0]
            mask_j = draw_sphere_mask(
                u_j, v_j, z_j,
                FRUIT_R_MM,
                IMG_H, IMG_W,
                FX, FY, CX, CY
            )
            if mask_iou(mask_i, mask_j) < IOU_MIN:
                continue
            # log merge
            print(f"Merging tracks {ti['id']} and {tj['id']}")
            # merge in-memory
            ti['frames'] = sorted(set(ti['frames'] + tj['frames']))
            ti['masks_xy'].update(tj['masks_xy'])
            ti['detections'].extend(tj['detections'])
            pts = [d['centroid_3d'] for d in ti['detections'] if d.get('centroid_3d')]
            if pts:
                ti['centroid_w'] = np.mean(pts, axis=0)
            used.add(j)
        merged.append(ti)

    # summary
    print(f"Merged {len(tracks)} → {len(merged)} tracks")

    # Write output
    final: Dict[str, List[Dict[str, Any]]] = {}
    for new_id, tr in enumerate(merged):
        for d in tr['detections']:
            d['det'] = new_id
            d['centroid_3d'] = tr['centroid_w'].tolist()
        final[str(new_id)] = tr['detections']

    output_json.parent.mkdir(exist_ok=True, parents=True)
    with open(output_json, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"✅ deduplicate wrote {output_json}")
    return output_json

if __name__ == "__main__":
    # Example usage; adjust paths as needed
    run(
        Path("output/tracks_cleaned.json"),
        Path("output/ceres/refined_cameras.txt"),
        Path("output/converted_poses.txt"),
        Path("output/ceres/refined_points.txt"),
        Path("output/merged_tracks.json")
    )
