#!/usr/bin/env python3
"""Compute the pixel to table homography from markers at known positions.

1. Put 4+ markers on the table. Measure each one in centimeters from the robot base axis: x forward, y left.
2. Run, listing the points in the order you will click them: 
    ./calibrate_homography.py --device 0 --points "10,-25 10,25 30,-12 30,12"
3. space freezes the live view. Click each marker centre in order. enter computes and saves. 'r' resets the clicks, esc quits.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from arm_vision.detector import load_camera, pixel_to_table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True, help='cm, e.g. "10,-25 10,25 30,-12 30,12"')
    ap.add_argument("--device", default="0")
    ap.add_argument("--image", help="use a saved image instead of the camera")
    ap.add_argument("--camera", help="camera.yaml with camera_matrix/dist_coeffs (optional)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "homography.yaml"))
    a = ap.parse_args()

    table_m = np.array([[float(v) / 100.0 for v in p.split(",")] for p in a.points.split()], dtype=np.float64)
    if len(table_m) < 4:
        sys.exit("need at least 4 points")
    K, dist = load_camera(a.camera) if a.camera else (None, None)

    def prep(f):
        return cv2.undistort(f, K, dist) if K is not None else f

    cap = None
    if a.image:
        frame = prep(cv2.imread(a.image))
        live = False
    else:
        dev = int(a.device) if a.device.isdigit() else a.device
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        frame, live = None, True

    clicks = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and not live and len(clicks) < len(table_m):
            clicks.append((x, y))

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)

    if cap is not None and not cap.isOpened():
        sys.exit(f"cannot open camera {a.device!r}")

    fails = 0
    while True:
        if live and cap:
            ok, f = cap.read()
            if ok:
                frame = prep(f)
                fails = 0
            else:
                fails += 1
                if fails % 100 == 1:
                    print("camera read() failed, no frames coming in")
                    
        if frame is None:
            cv2.waitKey(30)
            continue
        
        view = frame.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(view, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(view, f"{i+1}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
            
        if live:
            msg = "SPACE: freeze frame"
        elif len(clicks) < len(table_m):
            tx, ty = table_m[len(clicks)] * 100
            msg = f"click point {len(clicks)+1}/{len(table_m)}  (x={tx:.0f}, y={ty:.0f} cm)"
        else:
            msg = "ENTER: compute + save | r: reset"
            
        cv2.putText(view, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("calibrate", view)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        if k == ord(" ") and live:
            live = False
        if k == ord("r"):
            clicks.clear()
            live = cap is not None
        if k in (13, 10) and len(clicks) == len(table_m):
            px = np.array(clicks, dtype=np.float64)
            H, _ = cv2.findHomography(px, table_m, 0)
            errs = [np.hypot(*(np.array(pixel_to_table(H, u, v)) - t)) * 1000 for (u, v), t in zip(px, table_m)]
            
            print("per-point error (mm):", [f"{e:.1f}" for e in errs])
            print(f"mean error: {np.mean(errs):.1f} mm   max: {np.max(errs):.1f} mm")
            
            with open(a.out, "w") as f:
                yaml.safe_dump({"homography": H.tolist(),
                                "points_px": px.tolist(),
                                "points_table_m": table_m.tolist()}, f)
                
            print("saved", os.path.abspath(a.out))
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
