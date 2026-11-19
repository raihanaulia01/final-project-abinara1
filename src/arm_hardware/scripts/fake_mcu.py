#!/usr/bin/env python3
"""Pretends to be the STM32 firmware at port /tmp/fake_mcu
roslaunch arm_hardware hardware.launch port:=/tmp/fake_mcu
"""
import argparse
import math
import os
import pty
import random
import select
import struct
import sys
import time
import tty

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from arm_hardware import protocol as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", default="/tmp/fake_mcu")
    ap.add_argument("--limit-rad", type=float, default=3.14,
                    help="Simulated physical joint limit in +/- rad (default: 1.57)")
    a = ap.parse_args()

    master, slave = pty.openpty()
    tty.setraw(slave)
    os.set_blocking(master, False)
    if os.path.lexists(a.link):
        os.remove(a.link)
    os.symlink(os.ttyname(slave), a.link)
    print(f"fake MCU ready on {a.link} (-> {os.ttyname(slave)}), Ctrl+C to stop", flush=True)

    parser = P.FrameParser()
    diag = {k: 0 for k in P.DIAG_FIELDS}

    # Simulated servo state (5 joints in radians)
    cur_rad = [0.0] * P.NUM_JOINTS
    start_rad = [0.0] * P.NUM_JOINTS
    target_rad = [0.0] * P.NUM_JOINTS
    move_start_t = [0.0] * P.NUM_JOINTS
    move_dur_s = [0.0] * P.NUM_JOINTS
    moving = [False] * P.NUM_JOINTS

    stream_on = False
    stream_period_s = 0.05
    next_stream_t = 0.0

    failsafe = False
    boot_fresh = True
    clamped_last = False
    last_valid_frame_t = time.monotonic()

    def write(frame: bytes):
        try:
            os.write(master, frame)
        except (BlockingIOError, OSError):
            diag["tx_drops"] = (diag["tx_drops"] + 1) & 0xFFFF

    def send_ack(orig_cmd: int, err_code: int, extra: bytes = b""):
        write(P.build_frame(P.CMD_ACK, bytes([orig_cmd, err_code]) + extra))

    def send_status():
        dist_mm = 40 if cur_rad[P.JOINT_GRIP] < -0.15 else 180
        dist_mm = int(dist_mm + random.uniform(-3, 3))
        flags = P.FLAG_ECHO_VALID
        if failsafe:
            flags |= P.FLAG_FAILSAFE
        if any(moving):
            flags |= P.FLAG_MOVING
        if clamped_last:
            flags |= P.FLAG_CLAMPED_LAST
        if boot_fresh:
            flags |= P.FLAG_BOOT_FRESH
        mrad_vals = [P.rad_to_mrad(r) for r in cur_rad]
        write(P.build_frame(P.CMD_STATUS, struct.pack("<5hHH", *mrad_vals, dist_mm, flags)))

    def send_diag():
        vals = [diag[k] for k in P.DIAG_FIELDS]
        write(P.build_frame(P.CMD_DIAG, struct.pack("<12H", *vals)))

    def set_joint_target(idx: int, rad: float, dur_ms: int, now: float) -> bool:
        clamped = max(-a.limit_rad, min(a.limit_rad, rad))
        was_clamped = abs(clamped - rad) > 1e-6
        start_rad[idx] = cur_rad[idx]
        target_rad[idx] = clamped
        move_start_t[idx] = now
        move_dur_s[idx] = dur_ms / 1000.0
        if move_dur_s[idx] <= 0.0:
            cur_rad[idx] = clamped
            moving[idx] = False
        else:
            moving[idx] = True
        return was_clamped

    def stop_all():
        for i in range(P.NUM_JOINTS):
            target_rad[i] = cur_rad[i]
            moving[i] = False

    try:
        while True:
            ready, _, _ = select.select([master], [], [], 0.005)
            now = time.monotonic()

            # Update simulated joint trajectories
            for i in range(P.NUM_JOINTS):
                if moving[i]:
                    elapsed = now - move_start_t[i]
                    if elapsed >= move_dur_s[i]:
                        cur_rad[i] = target_rad[i]
                        moving[i] = False
                    else:
                        alpha = elapsed / move_dur_s[i]
                        cur_rad[i] = start_rad[i] + alpha * (target_rad[i] - start_rad[i])

            # Check 500 ms failsafe timeout
            if not failsafe and not boot_fresh and (now - last_valid_frame_t) >= 0.5:
                failsafe = True
                diag["failsafe_events"] = (diag["failsafe_events"] + 1) & 0xFFFF
                stop_all()
                print("FAILSAFE triggered (no valid frame for 500 ms)", flush=True)

            frames = []
            if ready:
                try:
                    raw = os.read(master, 1024)
                    diag["rx_bytes"] = (diag["rx_bytes"] + len(raw)) & 0xFFFF
                    prev_crc_err = parser.crc_errors
                    frames = parser.feed(raw)
                    if parser.crc_errors != prev_crc_err:
                        diff = parser.crc_errors - prev_crc_err
                        diag["err_crc"] = (diag["err_crc"] + diff) & 0xFFFF
                        print(f"bad CRC frame dropped (total {diag['err_crc']})", flush=True)
                except (BlockingIOError, OSError):
                    pass

            for cmd, payload in frames:
                diag["frames_ok"] = (diag["frames_ok"] + 1) & 0xFFFF
                last_valid_frame_t = now
                if failsafe:
                    print("FAILSAFE cleared by valid frame", flush=True)
                failsafe = False
                boot_fresh = False

                if cmd == P.CMD_PING:
                    if len(payload) != 0:
                        send_ack(cmd, P.ERR_BAD_LEN)
                    else:
                        send_ack(cmd, P.ERR_OK, bytes([0, 1, P.PROTOCOL_VERSION]))

                elif cmd == P.CMD_SET_ANGLES:
                    decoded = P.decode_set_angles(payload)
                    if decoded is None:
                        send_ack(cmd, P.ERR_BAD_LEN)
                    else:
                        angles_rad, dur_ms = decoded
                        mask = 0
                        for j in range(P.NUM_JOINTS):
                            if set_joint_target(j, angles_rad[j], dur_ms, now):
                                mask |= (1 << j)
                        if mask:
                            clamped_last = True
                            diag["clamp_events"] = (diag["clamp_events"] + 1) & 0xFFFF
                            send_ack(cmd, P.ERR_OK_CLAMPED, bytes([mask]))
                        else:
                            clamped_last = False
                            send_ack(cmd, P.ERR_OK)
                        deg_str = " ".join(f"{math.degrees(r):.1f}" for r in angles_rad)
                        print(f"SET_ANGLES deg=[{deg_str}] dur={dur_ms}ms mask=0x{mask:02X}", flush=True)

                elif cmd == P.CMD_HOME:
                    if len(payload) != 2:
                        send_ack(cmd, P.ERR_BAD_LEN)
                    else:
                        (dur_ms,) = struct.unpack("<H", payload)
                        for j in range(P.NUM_JOINTS):
                            set_joint_target(j, 0.0, dur_ms, now)
                        clamped_last = False
                        send_ack(cmd, P.ERR_OK)
                        print(f"HOME dur={dur_ms}ms", flush=True)

                elif cmd == P.CMD_STOP:
                    if len(payload) != 0:
                        send_ack(cmd, P.ERR_BAD_LEN)
                    else:
                        stop_all()
                        send_ack(cmd, P.ERR_OK)
                        print("STOP", flush=True)

                elif cmd == P.CMD_GET_STATUS:
                    if len(payload) != 0:
                        send_ack(cmd, P.ERR_BAD_LEN)
                    else:
                        send_status()

                elif cmd == P.CMD_STREAM:
                    if len(payload) != 3:
                        send_ack(cmd, P.ERR_BAD_LEN)
                    else:
                        en, period_ms = struct.unpack("<BH", payload)
                        if en > 1:
                            send_ack(cmd, P.ERR_BAD_ARG)
                        else:
                            period_ms = max(20, period_ms)
                            stream_on = bool(en)
                            stream_period_s = period_ms / 1000.0
                            next_stream_t = now + stream_period_s
                            send_ack(cmd, P.ERR_OK)
                            print(f"STREAM en={en} period={period_ms}ms", flush=True)

                elif cmd == P.CMD_GET_DIAG:
                    if len(payload) != 0:
                        send_ack(cmd, P.ERR_BAD_LEN)
                    else:
                        send_diag()

                elif cmd == P.CMD_SET_JOINT:
                    decoded = P.decode_set_joint(payload)
                    if decoded is None:
                        send_ack(cmd, P.ERR_BAD_LEN)
                    else:
                        joint_id, rad, dur_ms = decoded
                        if not (0 <= joint_id < P.NUM_JOINTS):
                            send_ack(cmd, P.ERR_BAD_ARG)
                        else:
                            was_clamped = set_joint_target(joint_id, rad, dur_ms, now)
                            if was_clamped:
                                clamped_last = True
                                diag["clamp_events"] = (diag["clamp_events"] + 1) & 0xFFFF
                                send_ack(cmd, P.ERR_OK_CLAMPED, bytes([1 << joint_id]))
                            else:
                                clamped_last = False
                                send_ack(cmd, P.ERR_OK)
                            print(f"SET_JOINT j={joint_id} deg={math.degrees(rad):.1f} dur={dur_ms}ms", flush=True)

                else:
                    diag["err_unknown_cmd"] = (diag["err_unknown_cmd"] + 1) & 0xFFFF
                    send_ack(cmd, P.ERR_UNKNOWN_CMD)
                    print(f"unknown command 0x{cmd:02X} ({len(payload)} bytes)", flush=True)

            if stream_on and now >= next_stream_t:
                send_status()
                next_stream_t = now + stream_period_s

    except KeyboardInterrupt:
        pass
    finally:
        if os.path.lexists(a.link):
            os.remove(a.link)


if __name__ == "__main__":
    main()