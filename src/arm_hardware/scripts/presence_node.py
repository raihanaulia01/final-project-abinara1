#!/usr/bin/env python3
"""Gripper ultrasonic range -> debounced "block in gripper" flag.

  /ultrasonic (sensor_msgs/Range)  ->  /gripper_presence (std_msgs/Bool)

task_node already subscribes to /gripper_presence and, with /task/use_presence true, treats
"not holding" at the end of the pick sequence as a missed grasp.

Private params:
  ~near_m       0.06   median range <= this  => block present
  ~far_m        0.09   median range >= this (or no echo) => block absent
                       between the two the previous state is kept (hysteresis)
  ~window_s     0.25   only samples newer than this are used
  ~min_samples  3      fewer fresh samples than this => "not holding" (fails safe)
  ~rate_hz      20.0   publish rate

near_m / far_m MUST be calibrated on the real arm: echo /ultrasonic with the empty
gripper and with a block held in the carry pose, then put the thresholds between the two.
"""
import collections
import statistics

import rospy
from sensor_msgs.msg import Range
from std_msgs.msg import Bool


class PresenceNode:
    def __init__(self):
        self.near = float(rospy.get_param("~near_m", 0.06))
        self.far = float(rospy.get_param("~far_m", 0.09))
        if self.far <= self.near:
            rospy.logfatal("far_m (%.3f) must be larger than near_m (%.3f)", self.far, self.near)
            raise SystemExit(1)
        self.window_s = float(rospy.get_param("~window_s", 0.25))
        self.min_samples = int(rospy.get_param("~min_samples", 3))

        self.samples = collections.deque(maxlen=50)  # (receive time [s], range [m])
        self.present = False

        self.pub = rospy.Publisher("/gripper_presence", Bool, queue_size=1)
        rospy.Subscriber("/ultrasonic", Range, self.on_range, queue_size=5)
        rate = float(rospy.get_param("~rate_hz", 20.0))
        rospy.Timer(rospy.Duration(1.0 / rate), self.tick)

    def on_range(self, msg):
        r = msg.range
        if r != r:  # NaN
            return
        # bridge_node publishes +inf when the MCU reports no echo; keep it as "far"
        self.samples.append((rospy.Time.now().to_sec(), r))

    def tick(self, _event):
        cutoff = rospy.Time.now().to_sec() - self.window_s
        fresh = [r for t, r in self.samples if t >= cutoff]

        if len(fresh) < self.min_samples:
            present = False  # no data (sensor or bridge down) must never look like a held block
        else:
            median = statistics.median(fresh)
            if median <= self.near:
                present = True
            elif median >= self.far:
                present = False
            else:
                present = self.present

        if present != self.present:
            rospy.loginfo("gripper presence: %s", "BLOCK" if present else "empty")
        self.present = present
        self.pub.publish(Bool(data=present))


if __name__ == "__main__":
    rospy.init_node("gripper_presence")
    PresenceNode()
    rospy.spin()