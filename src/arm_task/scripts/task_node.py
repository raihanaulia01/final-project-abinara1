#!/usr/bin/env python3
"""Pyramid-stacking sequencer.

For every block in plan.ORDER: look (arm at home) -> select -> solve IK for EVERY waypoint of
the pick and the place before moving anything -> execute -> back home.
"""
import threading

import rospy
from arm_msgs.msg import BlockArray
from arm_msgs.srv import SolveIK
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from arm_task import plan


class TaskNode:
    def __init__(self):
        self.cfg = dict(rospy.get_param("/task"))
        cfg = self.cfg
        self.joint_names = list(cfg["joint_names"])
        self.home_q = [float(v) for v in cfg["home_q"]]
        if len(self.joint_names) != 4 or len(self.home_q) != 4:
            raise ValueError("task joint_names and home_q must contain four values")
        self.slots = plan.build_slots(cfg)
        self.allow_tilt = bool(cfg["allow_tilt"])
        if self.allow_tilt:
            rospy.logwarn("allow_tilt is true: the gripper pitch is NOT held at grasp_pitch_deg")

        self.command_pub = rospy.Publisher("/joint_command", JointState, queue_size=1)
        self.gripper_pub = rospy.Publisher("/gripper_command", String, queue_size=1)
        rospy.Subscriber("/blocks", BlockArray, self.blocks_callback, queue_size=1)
        rospy.Subscriber("/gripper_presence", Bool, self.presence_callback, queue_size=1)
        self.ik = rospy.ServiceProxy("/solve_ik", SolveIK)

        self.lock = threading.Lock()
        self.latest_blocks = []
        self.latest_recv = rospy.Time(0)
        self.holding = None
        self.last_q = list(self.home_q)
        self.state = "init"

    # callbacks
    def blocks_callback(self, message):
        with self.lock:
            self.latest_blocks = list(message.blocks)
            self.latest_recv = rospy.Time.now()

    def presence_callback(self, message):
        self.holding = bool(message.data)

    # helpers 
    def set_state(self, state):
        self.state = state
        rospy.loginfo("arm_task state: %s", state)

    def publish_joint_command(self, q):
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = self.joint_names
        message.position = [float(v) for v in q]
        self.command_pub.publish(message)

    def publish_gripper(self, command):
        self.gripper_pub.publish(String(data=command))

    def move_to(self, q, extra_wait=0.0):
        """Command q and wait long enough for the slowest joint to arrive."""
        step = max(abs(a - b) for a, b in zip(q, self.last_q))
        self.publish_joint_command(q)
        self.last_q = [float(v) for v in q]
        rospy.sleep(step / float(self.cfg["joint_speed"]) + extra_wait)

    def go_home(self):
        self.set_state("home")
        self.move_to(self.home_q, extra_wait=float(self.cfg["settle_time"]))

    def look(self):
        """Blocks received after the arm settled; [] if none arrive within look_timeout."""
        self.set_state("look")
        after = rospy.Time.now() + rospy.Duration(float(self.cfg.get("camera_latency", 0.3)))
        deadline = rospy.Time.now() + rospy.Duration(float(self.cfg["look_timeout"]))
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            with self.lock:
                fresh = self.latest_recv >= after
                blocks = list(self.latest_blocks)
            if fresh and blocks:
                return blocks
            rospy.sleep(0.1)
        return []

    def solve(self, waypoint):
        try:
            response = self.ik(x=float(waypoint.x), y=float(waypoint.y),
                               z=float(waypoint.z), allow_tilt=self.allow_tilt)
        except rospy.ServiceException as error:
            rospy.logerr("solve_ik call failed: %s", error)
            return None
        if not response.success:
            rospy.logwarn("IK rejected %s (%.3f, %.3f, %.3f): %s", waypoint.name,
                          waypoint.x, waypoint.y, waypoint.z, response.message)
            return None
        return [float(v) for v in response.q_rad]

    def solve_all(self, waypoints):
        """(qs, None) if every waypoint is reachable, else (None, index of the first failure)."""
        qs = []
        for index, waypoint in enumerate(waypoints):
            q = self.solve(waypoint)
            if q is None:
                return None, index
            qs.append(q)
        return qs, None

    def execute(self, waypoints, qs):
        for waypoint, q in zip(waypoints, qs):
            if rospy.is_shutdown():
                return
            rospy.loginfo("  -> %s (%.3f, %.3f, %.3f)", waypoint.name,
                          waypoint.x, waypoint.y, waypoint.z)
            self.move_to(q)
            if waypoint.gripper:
                self.publish_gripper(waypoint.gripper)
            if waypoint.wait > 0:
                rospy.sleep(waypoint.wait)

    # one block 
    def pick_and_place(self, color, slot):
        cfg = self.cfg
        blacklist = []
        misses = {}
        while not rospy.is_shutdown():
            blocks = self.look()
            block = plan.select_block(blocks, color, cfg, blacklist)
            if block is None:
                rospy.logerr("no pickable %s block (visible, reachable, clear corridor)", color)
                return False
            key = (round(block.x, 2), round(block.y, 2))
            self.set_state("select")
            rospy.loginfo("selected %s block at (%.3f, %.3f)", color, block.x, block.y)

            # solve the entire pick + place sequence before moving. fall back to shorter run-ins
            qs = failed = pick_wps = place_wps = None
            for distance in plan.approach_distances(cfg):
                pick_wps = plan.pick_waypoints(block.x, block.y, cfg, distance)
                place_wps = plan.place_waypoints(slot, cfg, distance)
                qs, failed = self.solve_all(pick_wps + place_wps)
                if qs is not None:
                    rospy.loginfo("all waypoints reachable with a %.2f m run-in", distance)
                    break
            if qs is None:
                if failed >= len(pick_wps):
                    rospy.logerr("pyramid slot unreachable (%s): stopping",
                                 place_wps[failed - len(pick_wps)].name)
                    return False
                rospy.logwarn("block unreachable at %s: blacklisting it", pick_wps[failed].name)
                blacklist.append((block.x, block.y))
                continue

            self.set_state("pick")
            self.holding = None
            self.execute(pick_wps, qs[:len(pick_wps)])
            if cfg["use_presence"] and not self.holding:
                rospy.logwarn("grasp missed (no presence signal)")
                self.publish_gripper("open")
                self.go_home()
                misses[key] = misses.get(key, 0) + 1
                if misses[key] >= int(cfg["max_attempts"]):
                    blacklist.append((block.x, block.y))
                continue

            self.set_state("place")
            self.execute(place_wps, qs[len(pick_wps):])
            self.go_home()
            return True
        return False

    # whole task 
    def fail(self, message):
        self.set_state("error")
        rospy.logerr("arm_task stopped: %s", message)
        self.publish_gripper("open")
        self.go_home()
        self.set_state("error")

    def run(self):
        rospy.loginfo("waiting for /solve_ik ...")
        rospy.wait_for_service("/solve_ik")
        rospy.sleep(0.5)  # let the publishers connect
        self.publish_gripper("open")
        self.go_home()

        placed = {}
        for color in plan.ORDER:
            if rospy.is_shutdown():
                return
            index = placed.get(color, 0)
            slots = self.slots.get(color, [])
            if index >= len(slots):
                self.fail("no slot configured for %s #%d" % (color, index + 1))
                return
            rospy.loginfo("=== %s #%d -> slot (%.3f, %.3f, layer %d)", color, index + 1,
                          slots[index].x, slots[index].y, slots[index].layer)
            if not self.pick_and_place(color, slots[index]):
                self.fail("could not place %s #%d" % (color, index + 1))
                return
            placed[color] = index + 1

        self.set_state("done")
        rospy.loginfo("stacking sequence complete")


def main():
    rospy.init_node("arm_task")
    try:
        TaskNode().run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()