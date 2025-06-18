from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.optimize import linear_sum_assignment

from track_utils import (
    load_poses, invert_T,
    backproject_bbox_dense, filter_highest_density_cluster,
    reject_outliers_by_mode_z, bbox_to_mask,
    compute_iou, transform_cam_to_world, project_cloud_to_mask
)

@dataclass
class Detection:
    frame: int
    det_id: int
    bbox: List[int]
    mask_coords: List[List[int]]      = field(default_factory=list)
    mask_points_3d: List[List[float]] = field(default_factory=list)
    centroid_3d: Optional[List[float]] = None


def _get_cloud(frame_idx: int,
               bb: List[float],
               depth_dir: Path,
               outlier_delta: float) -> np.ndarray:
    """
    Load depth for a frame, back-project bbox, cluster, and reject outliers.
    """
    depth = np.load(str(depth_dir / f"depth_{frame_idx:06d}.npy"))
    dense = backproject_bbox_dense(bb, depth)
    cluster = filter_highest_density_cluster(dense)
    return reject_outliers_by_mode_z(cluster, delta=outlier_delta)


def _build_cost_matrix(masks0: List[np.ndarray],
                       bbs1: List[List[float]],
                       frame1: int,
                       depth_dir: Path,
                       iou_thresh: float) -> np.ndarray:
    """
    Compute the Hungarian cost matrix for matching masks0 -> bbox-based ground truth.
    """
    depth = np.load(str(depth_dir / f"depth_{frame1:06d}.npy"))
    H, W  = depth.shape
    gt_masks = [bbox_to_mask(bb, H, W) for bb in bbs1]

    C = np.array([[1.0 - compute_iou(m0, gm) for gm in gt_masks]
                  for m0 in masks0])
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
    Build 3D tracks from 2D detections + depth.
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
        key0, key1 = f"img_{frame:06d}.png", f"img_{frame+1:06d}.png"
        if key0 not in dets or key1 not in dets:
            prev_map = {}
            continue

        bbs0 = [o["bbox"] for o in dets[key0]]
        bbs1 = [o["bbox"] for o in dets[key1]]
        depth1 = np.load(str(depth_dir / f"depth_{frame+1:06d}.npy"))
        H1, W1 = depth1.shape

        # get filtered clouds
        clouds0 = [_get_cloud(frame,   bb, depth_dir, outlier_delta) for bb in bbs0]
        clouds1 = [_get_cloud(frame+1, bb, depth_dir, outlier_delta) for bb in bbs1]

        # relative pose
        Trel = invert_T(poses[frame+1]) @ poses[frame]
        Rrel, trel = Trel[:3, :3], Trel[:3, 3]

        # project clouds
        masks0 = [project_cloud_to_mask((Rrel @ c.T).T + trel, H1, W1)
                  for c in clouds0]

        # matching
        Cfull = _build_cost_matrix(masks0, bbs1, frame+1,
                                   depth_dir, iou_thresh)
        rows, cols = linear_sum_assignment(Cfull)
        new_map, matched = {}, set()

        # assign and enrich
        for i, j in zip(rows, cols):
            if j < len(bbs1) and Cfull[i, j] <= (1.0 - iou_thresh):
                tid = prev_map.get(i, next_id)
                if i not in prev_map:
                    next_id += 1
                    tracks[tid] = []
                    pts0 = (transform_cam_to_world(clouds0[i], poses[frame])
                            if clouds0[i].size else np.empty((0, 3)))
                    cent0 = pts0.mean(axis=0).tolist() if pts0.size else None
                    mask0 = project_cloud_to_mask(clouds0[i], H1, W1)
                    coords0 = [[int(x), int(y)] for y, x in zip(*np.where(mask0))]
                    tracks[tid].append(
                        Detection(frame, i, list(map(int, bbs0[i])),
                                  coords0, pts0.tolist(), cent0)
                    )

                pts1 = (transform_cam_to_world(clouds1[j], poses[frame+1])
                        if clouds1[j].size else np.empty((0, 3)))
                cent1 = pts1.mean(axis=0).tolist() if pts1.size else None
                mask1 = project_cloud_to_mask(clouds1[j], H1, W1)
                coords1 = [[int(x), int(y)] for y, x in zip(*np.where(mask1))]
                tracks[tid].append(
                    Detection(frame+1, j, list(map(int, bbs1[j])),
                              coords1, pts1.tolist(), cent1)
                )

                new_map[j] = tid
                matched.add(j)

        # new tracks for unmatched
        for j in set(range(len(bbs1))) - matched:
            tid = next_id
            next_id += 1
            pts1 = (transform_cam_to_world(clouds1[j], poses[frame+1])
                    if clouds1[j].size else np.empty((0, 3)))
            cent1 = pts1.mean(axis=0).tolist() if pts1.size else None
            mask1 = project_cloud_to_mask(clouds1[j], H1, W1)
            coords1 = [[int(x), int(y)] for y, x in zip(*np.where(mask1))]
            tracks[tid] = [
                Detection(frame+1, j, list(map(int, bbs1[j])),
                          coords1, pts1.tolist(), cent1)
            ]
            new_map[j] = tid

        prev_map = new_map

    # final pruning
    final = {}
    new_id = 0
    for tid, dets_list in tracks.items():
        frames = [d.frame for d in dets_list]
        if (len(frames) >= min_track_len and
            max(np.diff(sorted(frames))) <= (len(frames) - min_consecutive + 1)):
            for d in dets_list:
                d.det_id = new_id
            final[new_id] = [
                {"frame": d.frame,
                 "det":   d.det_id,
                 "bbox":  d.bbox,
                 "mask_coords": d.mask_coords,
                 "mask_points_3d": d.mask_points_3d,
                 "centroid_3d":   d.centroid_3d}
                for d in dets_list
            ]
            new_id += 1

    with open(output_json, "w") as f:
        json.dump(final, f, indent=2)

    print(f"Total tracks: {len(final)}")
    print(f"✅ new_tracks wrote {output_json}")
    return output_json

