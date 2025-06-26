#!/usr/bin/env python3
"""
track_and_match_masks.py
Build 3D tracks from per-pixel mask detections + depth.
"""

from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.optimize import linear_sum_assignment

from track_utils import (
    load_poses, invert_T,
    backproject_mask_dense, backproject_pixels,expand_mask_to_frame,
    filter_highest_density_cluster,
    reject_outliers_by_mode_z, compute_iou,
    transform_cam_to_world, project_cloud_to_mask
)

@dataclass
class Detection:
    frame: int
    det_id: int
    mask_coords: List[List[int]]      = field(default_factory=list)
    mask_points_3d: List[List[float]] = field(default_factory=list)
    centroid_3d: Optional[List[float]] = None

def _get_cloud_from_mask(frame_idx: int,
                         mask: np.ndarray,
                         depth_dir: Path,
                         outlier_delta: float) -> np.ndarray:
    """
    Load depth for frame_idx, back-project all pixels under 'mask',
    cluster densest mode, and reject Z outliers.
    """
    depth = np.load(str(depth_dir / f"depth_{frame_idx:06d}.npy"))
    pts_cam = backproject_mask_dense(mask, depth, outlier_delta)
    return pts_cam

def _build_cost_matrix(
    masks0: List[np.ndarray],
    masks1: List[np.ndarray],
    iou_thresh: float
) -> np.ndarray:
    """
    Compute the Hungarian cost matrix for matching projected masks0 -> masks1.
    """
    C = np.array([
        [1.0 - compute_iou(m0, m1) for m1 in masks1]
        for m0 in masks0
    ])
    n = C.shape[0]
    pad = np.full((n, n), 1.0 - iou_thresh)
    return np.hstack([C, pad])

def run(
    detections_json: Path,
    depth_dir:        Path,
    poses_file:       Path,
    output_json:      Path,
    # thresholds & pruning
    iou_thresh:       float,
    outlier_delta:    float,
    min_track_len:    int,
    min_consecutive:  int,
    # intrinsics
    fx:               float,
    fy:               float,
    cx:               float,
    cy:               float
) -> Path:
    """
    Build 3D tracks from mask detections + depth.
    Writes results to output_json.
    """
    # patch intrinsics into track_utils
    import track_utils
    for name, val in (("FX", fx), ("FY", fy), ("CX", cx), ("CY", cy)):
        setattr(track_utils, name, val)

    # load data
    dets  = json.load(open(detections_json))
    poses = load_poses(poses_file)
    N     = len(poses)

    tracks   = {}      # tid -> List[Detection]
    next_id  = 0
    prev_map = {}

    for frame in range(N - 1):
        key0 = f"img_{frame:06d}.png"
        key1 = f"img_{frame+1:06d}.png"
        if key0 not in dets or key1 not in dets:
            prev_map = {}
            continue

        # 1) load detection masks for both frames
        depth0 = np.load(str(depth_dir / f"depth_{frame:06d}.npy"))
        H0, W0 = depth0.shape
        depth1 = np.load(str(depth_dir / f"depth_{frame+1:06d}.npy"))
        H1, W1 = depth1.shape

        # expand each small mask to full-frame
        det_masks0 = [
            expand_mask_to_frame(np.array(o["mask"], dtype=bool),
                                o["bbox"], H0, W0)
            for o in dets[key0]
        ]
        det_masks1 = [
            expand_mask_to_frame(np.array(o["mask"], dtype=bool),
                                o["bbox"], H1, W1)
            for o in dets[key1]
]

        # 2) back-project each detection mask into 3D clouds
        clouds0 = [
            _get_cloud_from_mask(frame,   mask, depth_dir, outlier_delta)
            for mask in det_masks0
        ]
        clouds1 = [
            _get_cloud_from_mask(frame+1, mask, depth_dir, outlier_delta)
            for mask in det_masks1
        ]

        # 3) compute relative pose and project clouds0 into frame+1
        Trel = invert_T(poses[frame+1]) @ poses[frame]
        Rrel, trel = Trel[:3, :3], Trel[:3, 3]
        depth1 = np.load(str(depth_dir / f"depth_{frame+1:06d}.npy"))
        H1, W1 = depth1.shape

        proj_masks0 = [
            project_cloud_to_mask((Rrel @ c.T).T + trel, H1, W1)
            for c in clouds0
        ]

        # 4) match projected_masks0 -> det_masks1 via IoU
        Cfull = _build_cost_matrix(proj_masks0, det_masks1, iou_thresh)
        rows, cols = linear_sum_assignment(Cfull)
        new_map, matched = {}, set()

        for i, j in zip(rows, cols):
            # only accept matches below cost threshold
            if j < len(det_masks1) and Cfull[i, j] <= (1.0 - iou_thresh):
                # assign or start a track
                tid = prev_map.get(i, next_id)
                if i not in prev_map:
                    # new track
                    tracks[tid] = []
                    next_id += 1
                    # record first detection (frame)
                    pts0 = (transform_cam_to_world(clouds0[i], poses[frame])
                            if clouds0[i].size else np.empty((0, 3)))
                    cent0 = pts0.mean(axis=0).tolist() if pts0.size else None
                    ys, xs = np.where(det_masks0[i])
                    coords0 = np.column_stack((xs, ys)).tolist()
                    tracks[tid].append(
                        Detection(frame, i, coords0, pts0.tolist(), cent0)
                    )

                # record matched detection (frame+1)
                pts1 = (transform_cam_to_world(clouds1[j], poses[frame+1])
                        if clouds1[j].size else np.empty((0, 3)))
                cent1 = pts1.mean(axis=0).tolist() if pts1.size else None
                ys, xs = np.where(det_masks1[j])
                coords1 = np.column_stack((xs, ys)).tolist()
                tracks[tid].append(
                    Detection(frame+1, j, coords1, pts1.tolist(), cent1)
                )

                new_map[i] = tid
                matched.add(j)

        # 5) any unmatched detections in frame+1 start new tracks
        for j in set(range(len(det_masks1))) - matched:
            tid = next_id
            next_id += 1
            pts1 = (transform_cam_to_world(clouds1[j], poses[frame+1])
                    if clouds1[j].size else np.empty((0, 3)))
            cent1 = pts1.mean(axis=0).tolist() if pts1.size else None
            coords1 = [[int(x), int(y)]
                       for y, x in zip(*np.where(det_masks1[j]))]
            tracks[tid] = [
                Detection(frame+1, j, coords1, pts1.tolist(), cent1)
            ]
            new_map[j] = tid

        prev_map = new_map

    # 6) final pruning & write out
    final = {}
    new_id = 0
    for tid, dets_list in tracks.items():
        frames = [d.frame for d in dets_list]
        if (len(frames) >= min_track_len and
            max(np.diff(sorted(frames))) <= (len(frames) - min_consecutive + 1)):
            for d in dets_list:
                d.det_id = new_id
            final[new_id] = [
                {
                  "frame": d.frame,
                  "det":   d.det_id,
                  "mask_coords":      d.mask_coords,
                  "mask_points_3d":   d.mask_points_3d,
                  "centroid_3d":      d.centroid_3d
                }
                for d in dets_list
            ]
            new_id += 1

    with open(output_json, "w") as f:
        json.dump(final, f, indent=2)

    print(f"Total tracks: {len(final)}")
    print(f"✅ wrote {output_json}")
    return output_json
