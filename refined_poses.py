#!/usr/bin/env python3
"""
Convert refined_cameras.txt (angle-axis rx ry rz  tx ty tz)  ➜
refined_poses_mm_yup.txt in camera-to-world 3×4 row-major format
(r11 r12 r13 tx  r21 r22 r23 ty  r31 r32 r33 tz)
"""

from pathlib import Path
import numpy as np
import cv2

IN_TXT  = Path("output/ceres_output/refined_cameras.txt")        # output of bundle adjustment
OUT_TXT = Path("output/refined_poses_mm_yup.txt")   # matches poses_mm_yup.txt

rows = []
with open(IN_TXT) as f:
    for ln, line in enumerate(f, 1):
        vals = list(map(float, line.split()))
        if len(vals) != 6:
            raise ValueError(f"Line {ln}: need 6 numbers, got {len(vals)}")

        rvec = np.array(vals[:3], dtype=float)
        tvec = np.array(vals[3:], dtype=float).reshape(3, 1)   # column

        R_wc, _ = cv2.Rodrigues(rvec)     # world→camera
        R_cw = R_wc.T                     # camera→world
        t_cw = -R_cw @ tvec               # camera centre in world coords

        row = np.concatenate([R_cw[0], t_cw[0],
                              R_cw[1], t_cw[1],
                              R_cw[2], t_cw[2]])
        rows.append(row)

with open(OUT_TXT, "w") as f:
    for r in rows:
        f.write(" ".join(f"{x:.6f}" for x in r) + "\n")

print(f"Wrote {len(rows)} camera-to-world poses to {OUT_TXT}")


import numpy as np

old = np.loadtxt("data/poses/poses_mm_yup.txt")[0].reshape(3,4)
new = np.loadtxt("output/refined_poses_mm_yup.txt")[0].reshape(3,4)

print("First camera centre shift (mm):",
      np.linalg.norm(old[:,3] - new[:,3]))