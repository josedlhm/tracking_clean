#!/usr/bin/env python3
"""
deduplicate_tracks.py

Fuse duplicate fruit tracks **after bundle-adjustment** when they refer to the
same physical fruit, with correct v-axis sign inversion in projection.

Configure file paths below so you don’t need CLI arguments.
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

# ── CONFIGURE PATHS & PARAMETERS HERE ────────────────────────────────────────
TRACKS_JSON           = Path("output/tracks_cleaned.json")
CERES_POSES           = Path("output/ceres/refined_cameras.txt")
CONVERTED_POSES       = Path("output/converted_poses.txt")
CERES_POINTS          = Path("output/ceres/refined_points.txt")
OUT_JSON              = Path("output/merged_tracks.json")

DIST_MM               = 50.0      # max centroid distance (mm) to consider merging
IOU_MIN               = 0.1       # min mask IoU to accept merge
FRUIT_R_MM            = 40.0      # synthetic fruit radius (mm)

FX, FY                = 1272.44, 1272.67
CX, CY                = 920.062, 618.949
IMG_W, IMG_H          = 1920, 1200
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
    converted_poses: Path,
    ceres_points: Path,
    output_json: Path,
    dist_mm: float,
    iou_min: float,
    fruit_radius_mm: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int
) -> None:
    # 1) Convert & load camera-to-world poses
    convert_ceres_poses(ceres_poses, converted_poses)
    Ts = load_poses(converted_poses)  # list of 4×4 numpy arrays

    # 2) Load refined 3D centroids
    points_refined = np.loadtxt(ceres_points)  # shape (P,3)

    # 3) Load cleaned tracks
    with open(tracks_json, 'r') as f:
        raw = json.load(f)  # { track_id: [detections...] }

    # 4) Build track objects
    tracks: List[Dict[str, Any]] = []
    for tid_str, dets in raw.items():
        tid = int(tid_str)
        centroid_w = points_refined[tid]
        frames: List[int] = []
        masks_xy: Dict[int, np.ndarray] = {}
        for d in dets:
            f = d['frame']
            frames.append(f)
            mask = np.zeros((img_h, img_w), dtype=bool)
            for x, y in d['mask_coords']:
                mask[y, x] = True
            masks_xy[f] = mask

        tracks.append({
            'id'         : tid,
            'centroid_w' : centroid_w,
            'frames'     : frames,
            'masks_xy'   : masks_xy,
            'detections' : dets
        })

    # 5) Pairwise merge
    used: Set[int] = set()
    merged: List[Dict[str, Any]] = []
    for i, ti in enumerate(tqdm(tracks, desc='Merging tracks')):
        if i in used:
            continue
        for j in range(i + 1, len(tracks)):
            if j in used:
                continue
            tj = tracks[j]

            # a) 3D centroid distance
            if np.linalg.norm(ti['centroid_w'] - tj['centroid_w']) > dist_mm:
                continue

            # b) no overlapping frames
            if set(ti['frames']) & set(tj['frames']):
                continue

            # c) mask IoU in the first frame
            f0 = ti['frames'][0]
            T_wc = Ts[f0]

            u_i, v_i, z_i = project_point(ti['centroid_w'], T_wc, fx, fy, cx, cy)
            u_j, v_j, z_j = project_point(tj['centroid_w'], T_wc, fx, fy, cx, cy)

            # ensure projections are inside the image
            if not (0 <= u_i < img_w and 0 <= v_i < img_h and
                    0 <= u_j < img_w and 0 <= v_j < img_h):
                continue

            mask_i = ti['masks_xy'][f0]
            mask_j = draw_sphere_mask(
                u_j, v_j, z_j,
                fruit_radius_mm,
                img_h, img_w,
                fx, fy, cx, cy
            )
            if mask_iou(mask_i, mask_j) < iou_min:
                continue

            # --- log the actual merge ---
            print(f"Merging tracks {ti['id']} and {tj['id']}")

            # merge tj into ti
            ti['frames'] = sorted(set(ti['frames'] + tj['frames']))
            ti['masks_xy'].update(tj['masks_xy'])
            ti['detections'].extend(tj['detections'])
            # recompute centroid
            pts = [d['centroid_3d'] for d in ti['detections'] if d.get('centroid_3d')]
            if pts:
                ti['centroid_w'] = np.mean(pts, axis=0)
            used.add(j)

        merged.append(ti)

    # report merge summary
    print(f"Merged {len(tracks)} → {len(merged)} tracks")

    # 6) Write output JSON
    final: Dict[str, List[Dict[str, Any]]] = {}
    for new_id, tr in enumerate(merged):
        for d in tr['detections']:
            d['det'] = new_id
            d['centroid_3d'] = tr['centroid_w'].tolist()
        final[str(new_id)] = tr['detections']

    output_json.parent.mkdir(exist_ok=True, parents=True)
    with open(output_json, 'w') as f:
        json.dump(final, f, indent=2)

    print(f"Wrote {output_json}")

def main():
    run(
        tracks_json=TRACKS_JSON,
        ceres_poses=CERES_POSES,
        converted_poses=CONVERTED_POSES,
        ceres_points=CERES_POINTS,
        output_json=OUT_JSON,
        dist_mm=DIST_MM,
        iou_min=IOU_MIN,
        fruit_radius_mm=FRUIT_R_MM,
        fx=FX, fy=FY, cx=CX, cy=CY,
        img_w=IMG_W, img_h=IMG_H
    )

if __name__ == "__main__":
    main()
