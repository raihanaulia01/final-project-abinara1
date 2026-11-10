#!/usr/bin/env python3
"""Interactive HSV tuning. Move the sliders until only one block colour is white in the mask.
Press 'p' to print a line you can paste into vision.yaml, ESC to quit.
  ./tune_hsv.py --device 0      (or --image frame.png)
Tune under the real lighting, with the real blocks on the real arena surface.
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    ap.add_argument("--image")
    a = ap.parse_args()

    cap = None
    img = None
    if a.image:
        img = cv2.imread(a.image)
    else:
        cap = cv2.VideoCapture(int(a.device) if a.device.isdigit() else a.device, cv2.CAP_V4L2)

    cv2.namedWindow("tune")
    for name, init, mx in [("H lo", 0, 179), ("S lo", 0, 255), ("V lo", 0, 255),
                           ("H hi", 179, 179), ("S hi", 255, 255), ("V hi", 255, 255)]:
        cv2.createTrackbar(name, "tune", init, mx, lambda _: None)

    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                continue
        else:
            frame = img
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        lo = [cv2.getTrackbarPos(n, "tune") for n in ("H lo", "S lo", "V lo")]
        hi = [cv2.getTrackbarPos(n, "tune") for n in ("H hi", "S hi", "V hi")]
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        both = np.hstack([frame, cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), cv2.bitwise_and(frame, frame, mask=m)])
        cv2.imshow("tune", cv2.resize(both, None, fx=0.6, fy=0.6))
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        if k == ord("p"):
            print(f"{{lo: {lo}, hi: {hi}}}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
