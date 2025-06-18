#!/usr/bin/env python3
"""
deduplicate_tracks.py
─────────────────────
Fuse duplicate fruit tracks **after bundle-adjustment** when they refer to the
same physical fruit.

Inputs
------
tracks_cleaned.json         # your per-track detections (see schema below)
refined_poses_mm_yup.txt    # F×12 camera→world 3×4 poses, row-major

Output
------
merged_tracks.json          # same JSON schema, duplicates merged

tracks_cleaned.json schema
--------------------------
{
  "0": [                    ← track-ID (string)
    {
      "frame": 0,
      "det"  : 1,
      "bbox" : [x1,y1,x2,y2],
      "mask_coords": [[x,y], …],   # *all* foreground pixels
      "centroid_3d": [X,Y,Z],      # (mm, world)
      "mask_points_3d": [...]      # (unused here)
    },
    { … }                         # other frames for this track
  ],
  "1": [ … ],
  …
}
"""

# ── USER CONFIG ────────────────────────────────────────────────────────────
TRACKS_JSON        = 'output/tracks_cleaned.json'
REFINED_POSES_TXT  = "output/refined_poses_mm_yup.txt"
OUT_JSON           = "output/merged_tracks.json"

DIST_MM      = 50.0      # 3-D centroid distance to consider merge (mm)
IOU_MIN      = 0.1      # mask IoU threshold to accept merge
FRUIT_R_MM   = 40.0      # phys. fruit radius for synthetic mask (mm)

FX, FY = 1272.44, 1272.67
CX, CY = 920.062, 618.949
IMG_W, IMG_H = 1920, 1200
# ────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set

import numpy as np
import cv2
from tqdm import tqdm


# ─── CAMERA POSES ───────────────────────────────────────────────────────────
def load_cameras(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays (F,3,3) R_wc and (F,3) t_wc."""
    mats = np.loadtxt(path, dtype=np.float64).reshape(-1, 3, 4)  # (F,3,4)
    R_cw = mats[:, :, :3]                    # camera→world
    t_cw = mats[:, :,  3]                    # camera→world
    R_wc = np.transpose(R_cw, (0, 2, 1))     # world→camera
    t_wc = -np.einsum("fij,fj->fi", R_wc, t_cw)
    return R_wc, t_wc


# ─── TRACKS ─────────────────────────────────────────────────────────────────
def raster_mask(coords: np.ndarray) -> np.ndarray:
    """coords (N,2) → bool mask (IMG_H, IMG_W)."""
    mask = np.zeros((IMG_H, IMG_W), np.bool_)
    mask[coords[:, 1], coords[:, 0]] = True
    return mask


def load_tracks(path: str) -> List[Dict[str, Any]]:
    """
    Returns list of dicts:
        id           – int
        centroid_w   – (3,) float64 (from refined_points.txt)
        frames       – [int, …]
        masks_xy     – {frame: bool H×W}
        detections   – original detection dicts (for writing back)
    """
    with open(path) as f:
        raw = json.load(f)

    refined_pts = np.loadtxt("output/ceres_output/refined_points.txt")  # (N, 3)

    tracks = []
    for track_id_str, dets in raw.items():
        track_id = int(track_id_str)
        dets_sorted = sorted(dets, key=lambda d: d["frame"])

        # Use refined centroid from BA
        centroid = refined_pts[track_id]

        frames, masks_xy = [], {}
        for d in dets_sorted:
            f_idx = d["frame"]
            frames.append(f_idx)
            xy = np.asarray(d["mask_coords"], dtype=np.int32)
            masks_xy[f_idx] = raster_mask(xy)

        tracks.append({
            "id"         : track_id,
            "centroid_w" : centroid,
            "frames"     : frames,
            "masks_xy"   : masks_xy,
            "detections" : dets_sorted
        })
    return tracks


# ─── GEOMETRY ───────────────────────────────────────────────────────────────
def project_point(X_w: np.ndarray, R_wc: np.ndarray,
                  t_wc: np.ndarray) -> Tuple[float, float, float]:
    """Return (u,v,z_mm) in camera frame."""
    p = R_wc @ X_w + t_wc
    u = FX * (p[0] / p[2]) + CX
    v = -FY * (p[1] / p[2]) + CY
    return u, v, p[2]


def draw_sphere_mask(u: float, v: float, z_mm: float,
                     radius_mm: float = FRUIT_R_MM) -> np.ndarray:
    """
    Returns a boolean mask (H×W) of a filled circle at (u,v). Internally builds
    a uint8 image so that cv2.circle() works, then thresholds to bool.
    """
    # 1) Create a byte‐type image
    mask_uint8 = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    # compute pixel radius
    r_px = int(round(FX * radius_mm / z_mm))
    # draw in white (255)
    cv2.circle(mask_uint8,
               (int(round(u)), int(round(v))),
               r_px,
               255,  # color must fit uint8
               -1)
    # 2) convert to boolean for IoU
    return mask_uint8 > 0


def mask_iou(A: np.ndarray, B: np.ndarray) -> float:
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return inter / union if union else 0.0


# ─── MERGE LOGIC ────────────────────────────────────────────────────────────
def frames_disjoint(fr1: List[int], fr2: List[int]) -> bool:
    return not set(fr1).intersection(fr2)


def try_merge(ti: Dict, tj: Dict,
              R_wc_all: np.ndarray, t_wc_all: np.ndarray) -> bool:
    # 1) 3-D distance
    if np.linalg.norm(ti["centroid_w"] - tj["centroid_w"]) > DIST_MM:
        return False
    # 2) temporal disjointness
    if not frames_disjoint(ti["frames"], tj["frames"]):
        return False
    # 3) IoU check in first frame of ti
    f = ti["frames"][0]
    R_wc, t_wc = R_wc_all[f], t_wc_all[f]

    ui, vi, zi = project_point(ti["centroid_w"], R_wc, t_wc)
    uj, vj, zj = project_point(tj["centroid_w"], R_wc, t_wc)

    if not (0 <= ui < IMG_W and 0 <= vi < IMG_H
            and 0 <= uj < IMG_W and 0 <= vj < IMG_H):
        return False

    mask_i = ti["masks_xy"][f]
    mask_j = draw_sphere_mask(uj, vj, zj)

    return mask_iou(mask_i, mask_j) >= IOU_MIN


def merge_into(ti: Dict, tj: Dict) -> Dict:
    ti["frames"].extend(tj["frames"])
    ti["frames"] = sorted(set(ti["frames"]))
    ti["masks_xy"].update(tj["masks_xy"])
    ti["detections"].extend(tj["detections"])

    # Recompute centroid from merged detections
    pts = [d["centroid_3d"] for d in ti["detections"] if "centroid_3d" in d]
    if pts:
        ti["centroid_w"] = np.mean(pts, axis=0)
    else:
        ti["centroid_w"] = np.zeros(3)

    return ti


def deduplicate(tracks: List[Dict],
                R_wc_all: np.ndarray,
                t_wc_all: np.ndarray) -> List[Dict]:
    used: Set[int] = set()
    merged = []
    for i, ti in enumerate(tqdm(tracks, desc="Merging")):
        if i in used:
            continue
        for j in range(i + 1, len(tracks)):
            if j in used:
                continue
            tj = tracks[j]
            if try_merge(ti, tj, R_wc_all, t_wc_all):
                print(f"Merging tracks {ti['id']} and {tj['id']}")  # Log the merge
                ti = merge_into(ti, tj)
                used.add(j)
        merged.append(ti)
    return merged

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    R_wc_all, t_wc_all = load_cameras(REFINED_POSES_TXT)
    tracks = load_tracks(TRACKS_JSON)

    merged = deduplicate(tracks, R_wc_all, t_wc_all)
    print(f"Merged {len(tracks)} → {len(merged)} tracks")

    # tidy IDs and rebuild JSON structure identical to input
    out: Dict[str, List[Dict[str, Any]]] = {}
    for new_id, tr in enumerate(merged):
        for det in tr["detections"]:
            det["det"] = new_id
            det["centroid_3d"] = tr["centroid_w"].tolist()
        out[str(new_id)] = tr["detections"]

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote", OUT_JSON)


if __name__ == "__main__":
    main()
