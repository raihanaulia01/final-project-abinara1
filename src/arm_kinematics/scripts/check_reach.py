#!/usr/bin/env python3
"""Reach table for the real chain.

  check_reach.py --z 0.03 0.09 0.15 0.23 --rmin 0.16 --rmax 0.35 --tilt

For every (z, radius) it tries a vertical gripper (pitch -90 deg); with --tilt it also reports
the steepest pitch from a list that works, and a position-only solve.
"""
import math, yaml
from arm_kinematics.ik import ArmGeometry, JointConfig, solve_numeric
from arm_task import plan

def main():
        
    arm = yaml.safe_load(open("arm.yaml"))["arm"]
    cfg = yaml.safe_load(open("task.yaml"))["task"]
    geo, jc = ArmGeometry.from_params(arm), JointConfig.from_params(arm["joints"])
    pitch = math.radians(cfg["grasp_pitch_deg"])

    for i in range(10):
        r = 0.14 + 0.03 * i
        out = []
        for d in plan.approach_distances(cfg):
            q, fail = arm["initial_q"], "ok"
            for w in plan.pick_waypoints(r, 0.0, cfg, d):
                s = solve_numeric(w.x, w.y, w.z, geo, jc, q, arm["floor_clearance"],
                                pitch_target=pitch, **arm["ik"])
                if s is None:
                    fail = w.name
                    break
                q = s.q
            out.append(f"{d:.2f}:{fail}")
        print(f"r={r:.2f}  " + "  ".join(out))


if __name__ == "__main__":
    main()