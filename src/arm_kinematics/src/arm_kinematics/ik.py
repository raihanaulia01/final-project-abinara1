import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ArmGeometry:
    base_height: float
    upper_arm: float
    forearm: float
    tool_length: float

    @classmethod
    def from_params(cls, params: Dict[str, float]) -> "ArmGeometry":
        return cls(
            base_height=float(params["base_height"]),
            upper_arm=float(params["upper_arm"]),
            forearm=float(params["forearm"]),
            tool_length=float(params["tool_length"]),
        )


@dataclass(frozen=True)
class JointConfig:
    limits: Tuple[Tuple[float, float], ...]
    zero_deg: Tuple[float, ...]
    direction: Tuple[float, ...]

    @classmethod
    def from_params(cls, params: Dict[str, Sequence[float]]) -> "JointConfig":
        limits = tuple(
            (float(limit[0]), float(limit[1])) for limit in params["limits"]
        )
        zero_deg = tuple(float(value) for value in params.get("zero_deg", [0] * 4))
        direction = tuple(
            float(value) for value in params.get("direction", [1] * 4)
        )
        
        if len(limits) != 4 or len(zero_deg) != 4 or len(direction) != 4:
            raise ValueError("joint configuration must contain exactly four joints")
        return cls(limits, zero_deg, direction)

    def within_limits(self, q: Sequence[float], tolerance: float = 1e-9) -> bool:
        return all(
            lower - tolerance <= value <= upper + tolerance
            for value, (lower, upper) in zip(q, self.limits)
        )

    def to_servo_deg(self, q: Sequence[float]) -> list:
        return [
            zero + direction * math.degrees(value)
            for value, zero, direction in zip(q, self.zero_deg, self.direction)
        ]


@dataclass(frozen=True)
class Solution:
    q: Tuple[float, float, float, float]
    pitch: float


def forward_position(q: Sequence[float], geometry: ArmGeometry) -> Tuple[float, float, float]:
    return tuple(_forward_transforms(q, geometry)[-1][:3, 3])

def solve_numeric(
    x: float,
    y: float,
    z: float,
    geometry: ArmGeometry,
    joints: JointConfig,
    initial_q: Sequence[float],
    floor_clearance: float,
    max_iterations: int = 100,
    tolerance: float = 1e-5,
    damping: float = 0.1,
    pitch_target: Optional[float] = None,
) -> Optional[Solution]:
    """Solve position IK with damped-least-squares Jacobian iterations."""
    if len(initial_q) != 4:
        raise ValueError("initial_q must contain exactly four joint angles")

    q = np.array(initial_q, dtype=float)
    target = np.array([x, y, z], dtype=float)
    lower = np.array([limit[0] for limit in joints.limits], dtype=float)
    upper = np.array([limit[1] for limit in joints.limits], dtype=float)
    q = np.clip(q, lower, upper)

    for _ in range(max_iterations):
        current = np.array(forward_position(q, geometry), dtype=float)
        error = target - current
        if pitch_target is not None:
            error = np.append(error, pitch_target - sum(q[1:]))
        if np.linalg.norm(error) < tolerance:
            result = tuple(float(value) for value in q)
            if min(_segment_heights(result, geometry)) >= floor_clearance:
                return Solution(q=result, pitch=sum(result[1:]))
            return None

        jacobian = _position_jacobian(q, geometry)
        if pitch_target is not None:
            jacobian = np.vstack((jacobian, [0.0, 1.0, 1.0, 1.0]))
        identity = np.eye(len(error))
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * identity,
            error,
        )
        q = np.clip(q + delta, lower, upper)

    return None


def _position_jacobian(q: Sequence[float], geometry: ArmGeometry) -> np.ndarray:
    transforms = _forward_transforms(q, geometry)
    tool_position = transforms[-1][:3, 3]
    joint_origins = [transform[:3, 3] for transform in transforms[:4]]
    joint_axes = [
        transforms[0][:3, :3] @ np.array([0.0, 0.0, 1.0]),
        transforms[1][:3, :3] @ np.array([0.0, -1.0, 0.0]),
        transforms[2][:3, :3] @ np.array([0.0, -1.0, 0.0]),
        transforms[3][:3, :3] @ np.array([0.0, -1.0, 0.0]),
    ]

    jacobian = np.zeros((3, 4), dtype=float)
    for index, (axis, origin) in enumerate(zip(joint_axes, joint_origins)):
        jacobian[:, index] = np.cross(axis, tool_position - origin)
    return jacobian


def _forward_transforms(q: Sequence[float], geometry: ArmGeometry) -> Tuple[np.ndarray, ...]:
    """Return transforms at the four joint origins and the tool tip.

    The temporary model uses a vertical base yaw followed by three relative
    pitch joints. Pitch is represented as a rotation about the local -Y axis.
    """
    base, shoulder, elbow, wrist = q
    transforms = [np.eye(4)]
    transforms.append(
        transforms[-1]
        @ _rotation_z(base)
        @ _translation(0.0, 0.0, geometry.base_height)
    )
    transforms.append(
        transforms[-1]
        @ _rotation_y(-shoulder)
        @ _translation(geometry.upper_arm, 0.0, 0.0)
    )
    transforms.append(
        transforms[-1]
        @ _rotation_y(-elbow)
        @ _translation(geometry.forearm, 0.0, 0.0)
    )
    transforms.append(
        transforms[-1]
        @ _rotation_y(-wrist)
        @ _translation(geometry.tool_length, 0.0, 0.0)
    )
    return tuple(transforms)


def _translation(x: float, y: float, z: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = [x, y, z]
    return transform


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([
        [cosine, 0.0, sine, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sine, 0.0, cosine, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([
        [cosine, -sine, 0.0, 0.0],
        [sine, cosine, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _segment_heights(q: Sequence[float], geometry: ArmGeometry) -> Tuple[float, ...]:
    _, shoulder, elbow, wrist = q
    p2 = shoulder + elbow
    p3 = p2 + wrist
    return (
        geometry.base_height,
        geometry.base_height + geometry.upper_arm * math.sin(shoulder),
        geometry.base_height
        + geometry.upper_arm * math.sin(shoulder)
        + geometry.forearm * math.sin(p2),
        geometry.base_height
        + geometry.upper_arm * math.sin(shoulder)
        + geometry.forearm * math.sin(p2)
        + geometry.tool_length * math.sin(p3),
    )
