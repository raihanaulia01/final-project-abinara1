#!/usr/bin/env python3
import threading
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from arm_msgs.msg import Block, BlockArray
from arm_vision.detector import BlockDetector, load_camera, load_homography


class LatestFrameReader:
    def __init__(self, device, width, height, fps):
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {device!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running and not rospy.is_shutdown():
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def close(self):
        self.running = False
        self.cap.release()


def main():
    rospy.init_node("detector")
    cfg = rospy.get_param("/vision")
    device = rospy.get_param("~device", cfg["device"])
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    H = load_homography(rospy.get_param("~homography_file"))
    cam_file = rospy.get_param("~camera_file", "")
    K, dist = load_camera(cam_file) if cam_file else (None, None)

    detector = BlockDetector(cfg["colors"], H,
                             min_area=cfg["min_area_px"], max_area=cfg["max_area_px"],
                             blur=cfg["blur"], morph=cfg["morph"],
                             camera_matrix=K, dist_coeffs=dist)

    reader = LatestFrameReader(device, cfg["width"], cfg["height"], cfg["fps"])
    rospy.on_shutdown(reader.close)

    pub = rospy.Publisher("blocks", BlockArray, queue_size=1)
    publish_debug = rospy.get_param("~publish_debug", True)
    dbg_pub = rospy.Publisher("debug_image", Image, queue_size=1) if publish_debug else None
    bridge = CvBridge()
    frame_id = cfg.get("frame_id", "base_link")

    rate = rospy.Rate(cfg["publish_rate"])
    while not rospy.is_shutdown():
        frame = reader.latest()
        if frame is not None:
            dets = detector.detect(frame)
            msg = BlockArray()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = frame_id
            msg.blocks = [Block(color=d.color, x=d.x, y=d.y, yaw=d.yaw, u=d.u, v=d.v, area=d.area) for d in dets]
            pub.publish(msg)
            if dbg_pub is not None:
                dbg_pub.publish(bridge.cv2_to_imgmsg(detector.debug_image(frame, dets), "bgr8"))
        rate.sleep()


if __name__ == "__main__":
    main()
