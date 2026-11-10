#!/usr/bin/env python3
import os

import yaml

from arm_kinematics.ik import ArmGeometry, JointConfig, solve_numeric


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(here, "..", "config", "arm.yaml")
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["arm"]
    geometry = ArmGeometry.from_params(config["links"])
    joints = JointConfig.from_params(config["joints"])
    floor_clearance = float(config.get("floor_clearance", 0.01))
    initial_q = config.get("initial_q", [0.0, 1.0, -1.5, -1.0])

    for radius in (0.10, 0.20, 0.30, 0.35):
        solution = solve_numeric(
            radius, 0.0, 0.02, geometry, joints, initial_q, floor_clearance,
            pitch_target=0,
        )
        result = "reachable" if solution else "unreachable"
        print(f"{radius:.2f} m: {result}")


if __name__ == "__main__":
    main()
