#!/usr/bin/env python3
"""Reach table for the real chain.

  check_reach.py --z 0.03 0.09 0.15 0.23 --rmin 0.16 --rmax 0.35 --tilt

For every (z, radius) it tries a vertical gripper (pitch -90 deg); with --tilt it also reports
the steepest pitch from a list that works, and a position-only solve.
"""
import argparse
import math
import os

import yaml

from arm_kinematics.ik import ArmGeometry, JointConfig, solve_numeric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--z", type=float, nargs="+", default=[0.03, 0.09, 0.15, 0.23])
    parser.add_argument("--rmin", type=float, default=0.16)
    parser.add_argument("--rmax", type=float, default=0.35)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--tilt", action="store_true",
                        help="also report the tilted-gripper pitches and position-only reach")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    path = args.config or os.path.join(here, "..", "config", "arm.yaml")
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["arm"]
    geometry = ArmGeometry.from_params(config)
    joints = JointConfig.from_params(config["joints"])
    floor = float(config.get("floor_clearance", 0.01))
    initial_q = config.get("initial_q", [0.0, 1.0, -1.5, -1.0])
    ik = config.get("ik", {})
    kwargs = dict(max_iterations=int(ik.get("max_iterations", 500)),
                  tolerance=float(ik.get("tolerance", 1e-5)),
                  damping=float(ik.get("damping", 0.1)))

    count = int(round((args.rmax - args.rmin) / args.step))
    radii = [args.rmin + i * args.step for i in range(count + 1)]
    pitches = [-90, -75, -60, -45] if args.tilt else [-90]
    for z in args.z:
        print(f"z = {z:.2f} m")
        for radius in radii:
            cells = []
            for pitch in pitches:
                sol = solve_numeric(radius, 0.0, z, geometry, joints, initial_q, floor, pitch_target=math.radians(pitch), **kwargs)
                cells.append("ok" if sol else "--")
            if args.tilt:
                sol = solve_numeric(radius, 0.0, z, geometry, joints, initial_q, floor, **kwargs)
                cells.append("ok" if sol else "--")
            print(f"  r={radius:.2f}  " + "  ".join(
                f"{label}:{cell}" for label, cell in zip(
                    [f"{p}" for p in pitches] + (["free"] if args.tilt else []), cells)))


if __name__ == "__main__":
    main()