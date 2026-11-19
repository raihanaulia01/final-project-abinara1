#!/usr/bin/env python3
import math

import rospy
from arm_msgs.srv import SolveIK, SolveIKResponse

from arm_kinematics.ik import ArmGeometry, JointConfig, solve_numeric


def main():
    rospy.init_node("ik")
    config = rospy.get_param("/arm")
    geometry = ArmGeometry.from_params(config)
    joints = JointConfig.from_params(config["joints"])
    preferred = JointConfig(((-1.5708, 1.5708),) * 4, joints.zero_deg, joints.direction)
    floor_clearance = float(config.get("floor_clearance", 0.01))
    initial_q = config.get("initial_q", [0.0, 1.0, -1.5, -1.0])
    ik_config = config.get("ik", {})
    max_iterations = int(ik_config.get("max_iterations", 500))
    tolerance = float(ik_config.get("tolerance", 1.0e-5))
    damping = float(ik_config.get("damping", 0.1))
    current_q = [float(value) for value in initial_q]
    grasp_pitch_deg = float(rospy.get_param("~grasp_pitch_deg", -20.0))
    rospy.loginfo("Using numeric damped-least-squares inverse-kinematics solver")

    def handle(request):
        solution = None
        for limits in (preferred, joints):
            solution = solve_numeric(
                request.x, request.y, request.z, geometry, limits,
                current_q, floor_clearance,
                max_iterations=max_iterations,
                tolerance=tolerance,
                damping=damping,
                pitch_target=None if request.allow_tilt else math.radians(grasp_pitch_deg),
            )
            if solution is not None:
                break
            
        if solution is None:
            return SolveIKResponse(
                success=False,
                message="no solution: out of reach, outside joint limits, "
                "or arm would hit the floor",
            )
        current_q[:] = solution.q
        return SolveIKResponse(
            success=True,
            q_rad=list(solution.q),
            servo_deg=joints.to_servo_deg(solution.q),
            pitch_deg=math.degrees(solution.pitch),
            message="ok",
        )

    rospy.Service("solve_ik", SolveIK, handle)
    rospy.loginfo("IK service ready: /solve_ik")
    rospy.spin()


if __name__ == "__main__":
    main()