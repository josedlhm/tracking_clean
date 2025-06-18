#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from scipy.optimize import linear_sum_assignment
from track_utils import (
    load_poses, invert_T,
    backproject_bbox_dense, filter_highest_density_cluster,
    reject_outliers_by_mode_z, bbox_to_mask,
    compute_iou, transform_cam_to_world, project_cloud_to_mask
)

# ─── USER CONFIG ─────────────────────────────────────────────────────────────
DETECTIONS_JSON  = Path("data/detections.json")
DEPTH_DIR        = Path("data/depth")
POSES_FILE       = Path("data/poses/poses_mm_yup.txt")
OUTPUT_JSON      = Path("output/tracks.json")

FX, FY           = 1272.44, 1272.67
CX, CY           = 920.062, 618.949
IOU_THRESH       = 0.25
OUTLIER_DELTA    = 60
MIN_TRACK_LEN    = 4
MIN_CONSECUTIVE  = 4

@dataclass
class Detection:
    frame: int
    det_id: int
    bbox: List[int]
    mask_coords: List[List[int]]        = field(default_factory=list)
    mask_points_3d: List[List[float]]   = field(default_factory=list)
    centroid_3d: Optional[List[float]]  = None

def get_filtered_cloud(frame_idx: int, bb: List[float]) -> np.ndarray:
    depth   = np.load(str(DEPTH_DIR / f"depth_{frame_idx:06d}.npy"))
    dense   = backproject_bbox_dense(bb, depth)
    cluster = filter_highest_density_cluster(dense)
    return reject_outliers_by_mode_z(cluster, delta=OUTLIER_DELTA)

def transform_and_project(cloud: np.ndarray,
                          R: np.ndarray, t: np.ndarray,
                          H: int, W: int) -> np.ndarray:
    moved = (R @ cloud.T).T + t
    return project_cloud_to_mask(moved, H, W)

def build_cost_matrix(masks0: List[np.ndarray],
                      bbs1: List[List[float]],
                      frame1: int) -> np.ndarray:
    depth = np.load(str(DEPTH_DIR / f"depth_{frame1:06d}.npy"))
    H, W  = depth.shape
    gt_masks = [bbox_to_mask(bb, H, W) for bb in bbs1]

    C = np.array([[1.0 - compute_iou(m0, gm) for gm in gt_masks]
                  for m0 in masks0])
    n = C.shape[0]
    pad = np.full((n, n), 1.0 - IOU_THRESH)
    return np.hstack([C, pad])

def main():
    dets  = json.load(open(DETECTIONS_JSON))
    poses = load_poses(POSES_FILE)
    N     = len(poses)

    tracks   = {}      # tid -> List[Detection]
    next_id  = 0
    prev_map = {}

    for ℓ in range(N - 1):
        k0, k1 = f"img_{ℓ:06d}.png", f"img_{ℓ+1:06d}.png"
        if k0 not in dets or k1 not in dets:
            prev_map = {}
            continue

        bbs0 = [o["bbox"] for o in dets[k0]]
        bbs1 = [o["bbox"] for o in dets[k1]]
        depth1 = np.load(str(DEPTH_DIR / f"depth_{ℓ+1:06d}.npy"))
        H1, W1 = depth1.shape

        # Precompute filtered clouds
        clouds0 = [get_filtered_cloud(ℓ,   bb) for bb in bbs0]
        clouds1 = [get_filtered_cloud(ℓ+1, bb) for bb in bbs1]

        # Compute relative pose ℓ→ℓ+1
        Trel = invert_T(poses[ℓ+1]) @ poses[ℓ]
        Rrel, trel = Trel[:3,:3], Trel[:3,3]

        # Project each cloud into frame ℓ+1
        masks0 = [transform_and_project(c, Rrel, trel, H1, W1)
                  for c in clouds0]

        # Solve assignment
        Cfull = build_cost_matrix(masks0, bbs1, ℓ+1)
        rows, cols = linear_sum_assignment(Cfull)

        new_map, matched = {}, set()
        for i, j in zip(rows, cols):
            if j < len(bbs1) and Cfull[i,j] <= (1.0 - IOU_THRESH):
                # get or create track ID
                if i in prev_map:
                    tid = prev_map[i]
                else:
                    tid = next_id
                    next_id += 1
                    tracks[tid] = []
                    # enrich the initial detection at frame ℓ
                    det0_cloud = clouds0[i]
                    det0_mask  = project_cloud_to_mask(det0_cloud, H1, W1)
                    det0_pts_w = (
                        transform_cam_to_world(det0_cloud, poses[ℓ])
                        if det0_cloud.size else np.empty((0,3))
                    )
                    det0_cent = (
                        det0_pts_w.mean(axis=0).tolist()
                        if det0_pts_w.size else None
                    )
                    coords0 = [
                        [int(x), int(y)]
                        for y, x in zip(*np.where(det0_mask))
                    ]
                    tracks[tid].append(
                        Detection(
                            frame        = ℓ,
                            det_id       = i,
                            bbox         = list(map(int, bbs0[i])),
                            mask_coords  = coords0,
                            mask_points_3d = det0_pts_w.tolist(),
                            centroid_3d  = det0_cent
                        )
                    )

                # enrich the matched detection at ℓ+1
                det1_cloud = clouds1[j]
                det1_mask  = project_cloud_to_mask(det1_cloud, H1, W1)
                det1_pts_w = (
                    transform_cam_to_world(det1_cloud, poses[ℓ+1])
                    if det1_cloud.size else np.empty((0,3))
                )
                det1_cent = (
                    det1_pts_w.mean(axis=0).tolist()
                    if det1_pts_w.size else None
                )
                coords1 = [
                    [int(x), int(y)]
                    for y, x in zip(*np.where(det1_mask))
                ]
                tracks[tid].append(
                    Detection(
                        frame        = ℓ+1,
                        det_id       = j,
                        bbox         = list(map(int, bbs1[j])),
                        mask_coords  = coords1,
                        mask_points_3d = det1_pts_w.tolist(),
                        centroid_3d  = det1_cent
                    )
                )

                new_map[j] = tid
                matched.add(j)

        # spawn brand-new tracks for any unmatched detections in ℓ+1
        for j in range(len(bbs1)):
            if j not in matched:
                tid = next_id
                next_id += 1
                det_cloud = clouds1[j]
                det_mask  = project_cloud_to_mask(det_cloud, H1, W1)
                det_pts_w = (
                    transform_cam_to_world(det_cloud, poses[ℓ+1])
                    if det_cloud.size else np.empty((0,3))
                )
                det_cent  = (
                    det_pts_w.mean(axis=0).tolist()
                    if det_pts_w.size else None
                )
                coords   = [
                    [int(x), int(y)]
                    for y, x in zip(*np.where(det_mask))
                ]
                tracks[tid] = [
                    Detection(
                        frame        = ℓ+1,
                        det_id       = j,
                        bbox         = list(map(int, bbs1[j])),
                        mask_coords  = coords,
                        mask_points_3d = det_pts_w.tolist(),
                        centroid_3d  = det_cent
                    )
                ]
                new_map[j] = tid

        prev_map = new_map

    # ─── FINAL PRUNING & SERIALIZATION ────────────────────────────────────────
    final = {}
    new_id = 0
    for tid, dets_list in tracks.items():
        frames = [d.frame for d in dets_list]
        # keep only tracks long enough and with enough consecutiveness
        if (len(frames) >= MIN_TRACK_LEN and
            max(np.diff(sorted(frames))) <= (len(frames) - MIN_CONSECUTIVE + 1)):
            for d in dets_list:
                d.det_id = new_id
            final[new_id] = [
                {
                    "frame":         d.frame,
                    "det":           d.det_id,
                    "bbox":          d.bbox,
                    "mask_coords":   d.mask_coords,
                    "mask_points_3d": d.mask_points_3d,
                    "centroid_3d":   d.centroid_3d
                }
                for d in dets_list
            ]
            new_id += 1

    # now everything is pure Python types and serializable
    with open(OUTPUT_JSON, "w") as f:
        json.dump(final, f, indent=2)

    print(f"Total enriched tracks: {len(final)}")
    print(f"✅ Saved refined 3D tracks to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
