#!/usr/bin/env python3
"""ROS <-> STM32 bridge.

  /joint_command (JointState, MODEL-frame radians) -> clamp -> servo rad -> SET_ANGLES (12B)
  /gripper_command (Float64, servo radians)        -> SET_JOINT for GRIP (joint 4)
  PING keepalive at ~ping_hz (>= 10 Hz required to avoid 500 ms STM32 failsafe)
  STATUS frames (0x82) from MCU                    -> /ultrasonic (sensor_msgs/Range) + failsafe check
  ACK frames (0x81) from MCU                       -> version check, clamp warnings, error logging

Params: ~port, ~baud, ~ping_hz (default 20.0), ~stream_hz (default 20.0), ~move_duration_ms (default 100)
"""
import math

import rospy
from sensor_msgs.msg import JointState, Range
from std_msgs.msg import Float64
from std_msgs.msg import String

from arm_hardware import protocol
from arm_hardware.serial_link import SerialLink
from arm_kinematics.ik import JointConfig


class Bridge:
    def __init__(self):
        cfg = rospy.get_param("/arm")
        self.joints = JointConfig.from_params(cfg["joints"])
        self.names = list(cfg["joints"].get("names") or rospy.get_param(
            "/task/joint_names", ["base", "shoulder", "elbow", "wrist"]))
        if len(self.names) != len(self.joints.limits):
            rospy.logfatal("joint names %s do not match %d joint limits", self.names,
                           len(self.joints.limits))
            raise SystemExit(1)
        if len(self.names) > protocol.NUM_JOINTS:
            rospy.logfatal("configured %d joints, firmware supports max %d",
                           len(self.names), protocol.NUM_JOINTS)
            raise SystemExit(1)

        self.min_deg = [math.degrees(lo) for lo, _ in self.joints.limits]
        self.max_deg = [math.degrees(hi) for _, hi in self.joints.limits]
        self.grip_rad = 0.0
        self.move_duration_ms = int(rospy.get_param("~move_duration_ms", 100))

        self.pub_ultrasonic = rospy.Publisher("ultrasonic", Range, queue_size=1)
        self.last_rx_time = None
        self._version_logged = False
        self._in_failsafe = False

        port = rospy.get_param("~port", "/dev/arm_controller")
        baud = rospy.get_param("~baud", 115200)
        try:
            self.link = SerialLink(port, baud, self.on_frame, self.on_error)
        except Exception as e:
            rospy.logfatal("cannot open %s: %s", port, e)
            raise SystemExit(1)
        rospy.loginfo("serial link open: %s @ %d", port, baud)
        rospy.on_shutdown(self.shutdown)

        # Opening USB-serial often triggers DTR reset on the Blue Pill; wait and flush
        rospy.sleep(1.0)
        self.link.reset_input_buffer()
        self.start = rospy.Time.now()

        # Initial handshake & enable periodic STATUS telemetry
        self.link.send(protocol.encode_ping())
        stream_hz = float(rospy.get_param("~stream_hz", 20.0))
        if stream_hz > 0:
            stream_period_ms = max(20, int(round(1000.0 / stream_hz)))
            self.link.send(protocol.encode_stream(True, stream_period_ms))

        rospy.Subscriber("joint_command", JointState, self.on_command, queue_size=1)
        self.gripper_open_rad = float(rospy.get_param("~gripper_open_rad", 0.3))
        self.gripper_close_rad = float(rospy.get_param("~gripper_close_rad", -0.3))
        rospy.Subscriber("gripper_command", String, self.on_gripper_command, queue_size=1)

        ping_hz = float(rospy.get_param("~ping_hz", 20.0))
        if ping_hz <= 0:
            rospy.logwarn("ping_hz <= 0: STM32 will enter failsafe after 500 ms without commands!")
        else:
            if ping_hz < 10.0:
                rospy.logwarn("ping_hz=%.1f Hz is below the 10 Hz minimum; risk of 500 ms failsafe", ping_hz)
            self.ping_period = 1.0 / ping_hz
            rospy.Timer(rospy.Duration(self.ping_period), self.ping)

    # ROS -> MCU
    def on_command(self, msg):
        pos = list(msg.position)
        if msg.name:  # reorder by name if names are given
            try:
                pos = [msg.position[msg.name.index(n)] for n in self.names]
            except ValueError:
                rospy.logwarn("joint_command names %s don't match %s", msg.name, self.names)
                return
        if len(pos) != len(self.names):
            rospy.logwarn("joint_command needs %d positions, got %d", len(self.names), len(pos))
            return

        q = []
        for i, p in enumerate(pos):  # last line of defence: clamp to model limits
            deg = math.degrees(p)
            clamped = min(max(deg, self.min_deg[i]), self.max_deg[i])
            if abs(clamped - deg) > 1e-6:
                rospy.logwarn_throttle(1, "%s command %.1f deg clamped to %.1f",
                                       self.names[i], deg, clamped)
            q.append(math.radians(clamped))

        # Apply direction & zero offsets via JointConfig, then convert degrees -> radians
        servo_rad = [math.radians(d) for d in self.joints.to_servo_deg(q)]
        if len(servo_rad) == protocol.NUM_JOINTS:
            self.grip_rad = servo_rad[protocol.JOINT_GRIP]
        else:
            # Pad missing trailing joints (e.g., 4-DOF arm config + separate gripper)
            while len(servo_rad) < protocol.NUM_JOINTS - 1:
                servo_rad.append(0.0)
            servo_rad.append(self.grip_rad)

        self.link.send(protocol.encode_set_angles(servo_rad, self.move_duration_ms))

    def on_gripper_command(self, msg):
        cmd = msg.data.strip().lower()
        if cmd == "open":
            self.grip_rad = self.gripper_open_rad
        elif cmd == "close":
            self.grip_rad = self.gripper_close_rad
        else:
            try:
                self.grip_rad = float(cmd)
            except ValueError:
                rospy.logwarn("Invalid gripper_command '%s' (expected 'open', 'close', or float rad)", msg.data)
                return

        self.link.send(protocol.encode_set_joint(
            protocol.JOINT_GRIP, self.grip_rad, self.move_duration_ms
        ))

    def ping(self, _event):
        self.link.send(protocol.encode_ping())
        now = rospy.Time.now()
        last = self.last_rx_time or self.start
        if (now - last).to_sec() > max(0.6, 3 * self.ping_period):
            rospy.logwarn_throttle(5, "STM32 is not responding (no frame for %.2fs)",
                                   (now - last).to_sec())

    # MCU -> ROS (called from the reader thread)
    def on_frame(self, cmd, payload):
        self.last_rx_time = rospy.Time.now()

        if cmd == protocol.CMD_ACK:
            ack = protocol.decode_ack(payload)
            if ack is None:
                return
            orig_cmd, err_code, extra = ack
            if orig_cmd == protocol.CMD_PING and err_code == protocol.ERR_OK and len(extra) >= 3:
                if not self._version_logged:
                    self._version_logged = True
                    rospy.loginfo("STM32 connected: firmware v%d.%d, protocol v%d",
                                  extra[0], extra[1], extra[2])
                    if extra[2] != protocol.PROTOCOL_VERSION:
                        rospy.logerr("Protocol version mismatch: host expects v%d, MCU reports v%d",
                                     protocol.PROTOCOL_VERSION, extra[2])
            elif err_code == protocol.ERR_OK_CLAMPED:
                mask = extra[0] if extra else 0
                rospy.logwarn_throttle(1, "STM32 clamped cmd 0x%02X (joint bitmask=0x%02X)",
                                       orig_cmd, mask)
            elif err_code != protocol.ERR_OK:
                err_name = protocol.ERR_NAMES.get(err_code, "0x%02X" % err_code)
                rospy.logerr_throttle(1, "STM32 rejected cmd 0x%02X with %s", orig_cmd, err_name)

        elif cmd == protocol.CMD_STATUS:
            status = protocol.decode_status(payload)
            if status is None:
                return
            _angles_rad, dist_mm, flags = status

            failsafe_active = bool(flags & protocol.FLAG_FAILSAFE)
            if failsafe_active and not self._in_failsafe:
                rospy.logwarn("STM32 entered FAILSAFE (host silence > 500 ms); re-send target to resume")
            self._in_failsafe = failsafe_active

            msg = Range(
                radiation_type=Range.ULTRASOUND,
                field_of_view=0.26,
                min_range=0.02,
                max_range=4.0,
                range=dist_mm / 1000.0 if dist_mm >= 0 else float("inf"),
            )
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "ultrasonic"
            self.pub_ultrasonic.publish(msg)

        elif cmd == protocol.CMD_DIAG:
            diag = protocol.decode_diag(payload)
            if diag is not None:
                rospy.loginfo("STM32 DIAG: %s", diag)

    def on_error(self, e):
        rospy.logerr("serial link lost: %s", e)
        rospy.signal_shutdown("serial link lost")

    def shutdown(self):
        try:
            self.link.send(protocol.encode_stream(False, 50))
            self.link.send(protocol.encode_stop())
            rospy.sleep(0.05)
        except Exception:
            pass
        self.link.close()


if __name__ == "__main__":
    rospy.init_node("bridge")
    Bridge()
    rospy.spin()