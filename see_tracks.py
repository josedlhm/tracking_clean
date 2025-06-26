#!/usr/bin/env python3
"""
visualize_tracks.py

1) Reads your saved `tracks_full.json`
2) Assigns each track a distinct color
3) For each frame:
     • loads the image
     • rotates it by 90° clockwise
     • draws every bbox (rotated) for tracks active in that frame
     • labels it with the track ID
   and writes it out to out_tracks_viz/
4) Assembles all viz frames into tracks.mp4 at 10 FPS
"""

import json
import cv2
import numpy as np
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TRACKS_JSON = Path("output/merged_tracks.json")
IMAGES_DIR  = Path("data/images")
OUT_DIR     = Path("output/out_tracks_viz")
VIDEO_FILE  = "output/tracks_refined.mp4"
FPS         = 6
ROTATE_CODE = cv2.ROTATE_90_COUNTERCLOCKWISE  # rotate 90° clockwise
# ──────────────────────────────────────────────────────────────────────────────

def distinct_color(idx):
    """Generate a consistent color for each track ID."""
    np.random.seed(idx)
    return tuple(int(x) for x in np.random.randint(50, 255, size=3))

def rotate_bbox(bb, img_w, img_h):
    """
    Rotate a bbox [x1,y1,x2,y2] for a 90° clockwise rotation of an image of width img_w, height img_h.
    New coords: x' = y, y' = (img_w - 1) - x
    """
    x1,y1,x2,y2 = bb
    # corners
    pts = [(x1,y1), (x2,y2)]
    rotated = []
    for x,y in pts:
        xr = y
        yr = (img_w - 1) - x
        rotated.append((xr, yr))
    (nx1, ny1), (nx2, ny2) = rotated
    # ensure proper ordering
    x1n, x2n = min(nx1, nx2), max(nx1, nx2)
    y1n, y2n = min(ny1, ny2), max(ny1, ny2)
    return [x1n, y1n, x2n, y2n]

def main():
    # prepare output
    OUT_DIR.mkdir(exist_ok=True)

    # load tracks
    with open(TRACKS_JSON) as f:
        raw = json.load(f)

    # build frame -> list of (track_id, bbox)
    frame_map = {}
    for tid_str, rec in raw.items():
        tid = int(tid_str)
        for entry in rec:
            fidx = entry["frame"]
            bbox = entry["bbox"]
            frame_map.setdefault(fidx, []).append((tid, bbox))

    # precompute colors
    colors = {tid: distinct_color(tid) for tid in map(int, raw.keys())}

    # determine frame range
    frames = sorted(frame_map.keys())
    if not frames:
        print("No frames found in tracks_full.json!")
        return

    viz_paths = []
    for fidx in frames:
        img_path = IMAGES_DIR / f"img_{fidx:06d}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: image missing {img_path}, skipping")
            continue
        # rotate image
        img_rot = cv2.rotate(img, ROTATE_CODE)
        H, W = img_rot.shape[:2]

        for tid, bb in frame_map[fidx]:
            # rotate bbox coords
            rb = rotate_bbox(bb, img.shape[1], img.shape[0])
            x1,y1,x2,y2 = map(int, rb)
            col = colors[tid]
            cv2.rectangle(img_rot, (x1, y1), (x2, y2), col, 2)
            cv2.putText(
                img_rot,
                f"#{tid}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                col,
                1,
                cv2.LINE_AA
            )

        out_file = OUT_DIR / f"viz_{fidx:06d}.png"
        cv2.imwrite(str(out_file), img_rot)
        viz_paths.append(str(out_file))

    # assemble video
    if viz_paths:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # assume all frames same size
        sample = cv2.imread(viz_paths[0])
        h, w = sample.shape[:2]
        writer = cv2.VideoWriter(VIDEO_FILE, fourcc, FPS, (w, h))
        for p in viz_paths:
            frame = cv2.imread(p)
            writer.write(frame)
        writer.release()
        print(f"Wrote video {VIDEO_FILE}")

    print(f"Wrote {len(viz_paths)} annotated frames to {OUT_DIR}")

if __name__ == "__main__":
    main()
