from dataclasses import dataclass
import cv2
import numpy as np
import yaml


@dataclass
class Detection:
    color: str
    u: float # pixel positions
    v: float
    x: float # real world positions
    y: float 
    area: float

def pixel_to_table(H, u, v):
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])

def load_homography(path):
    with open(path) as f:
        return np.array(yaml.safe_load(f)["homography"], dtype=np.float64)
    
def load_camera(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    return (np.array(d["camera_matrix"], dtype=np.float64),
            np.array(d["dist_coeffs"], dtype=np.float64))

def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1

class BlockDetector:
    def __init__(self, colors, H, min_area=400, max_area=40000, blur=5, morph=5,
                 camera_matrix=None, dist_coeffs=None):
        # colors: {name: {"lo": [h,s,v], "hi": [h,s,v]}}
        self.colors = {
            name: (np.array(c["lo"], dtype=np.uint8), np.array(c["hi"], dtype=np.uint8))
            for name, c in colors.items()
        }
        self.H = np.asarray(H, dtype=np.float64)
        self.min_area = min_area
        self.max_area = max_area
        self.blur = _odd(blur)
        k = _odd(morph)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def undistort(self, frame):
        if self.camera_matrix is None:
            return frame
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def mask(self, hsv, name):
        lo, hi = self.colors[name]
        m = cv2.inRange(hsv, lo, hi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel)
        return m

    def detect(self, frame_bgr):
        frame = self.undistort(frame_bgr)
        blurred = cv2.GaussianBlur(frame, (self.blur, self.blur), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        out = []
        for name in self.colors:
            contours, _ = cv2.findContours(self.mask(hsv, name), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if not (self.min_area <= area <= self.max_area):
                    continue
                (u, v), _, _ = cv2.minAreaRect(c)
                x, y = pixel_to_table(self.H, u, v)

                out.append(Detection(name, u, v, x, y, area))
        return out

    def debug_image(self, frame_bgr, detections):
        img = self.undistort(frame_bgr).copy()
        for d in detections:
            cv2.circle(img, (int(d.u), int(d.v)), 5, (255, 255, 255), -1)
            label = f"{d.color} ({d.x*100:.1f},{d.y*100:.1f})cm"
            cv2.putText(img, label, (int(d.u) + 8, int(d.v) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (int(d.u) + 8, int(d.v) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img
