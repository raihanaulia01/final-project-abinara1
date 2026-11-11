import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, cast

import numpy as np


@dataclass(frozen=True)
class ArmGeometry:
    joint_origins: Tuple[Tuple[float, float, float], ...]
    joint_rpys: Tuple[Tuple[float, float, float], ...]
    joint_axes: Tuple[Tuple[float, float, float], ...]
    tcp_origin: Tuple[float, float, float]
    tcp_rpy: Tuple[float, float, float]

    @classmethod
    def from_params(cls, params: Dict[str, object]) -> "ArmGeometry":
        chain = cast(Sequence[Mapping[str, Sequence[float]]], params["chain"])
        if len(chain) != 4:
            raise ValueError("kinematic chain must contain exactly four joints")
        tcp = cast(Mapping[str, Sequence[float]], params["tcp"])
        return cls(
            joint_origins=tuple(
                _vector3(joint["origin"])
                for joint in chain
            ),
            joint_rpys=tuple(
                _vector3(joint["rpy"])
                for joint in chain
            ),
            joint_axes=tuple(
                _vector3(joint["axis"])
                for joint in chain
            ),
            tcp_origin=_vector3(tcp["origin"]),
            tcp_rpy=_vector3(tcp["rpy"]),
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


def _vector3(values: Sequence[float]) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("kinematic vectors must contain exactly three values")
    x, y, z = values
    return float(x), float(y), float(z)


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
    joint_transforms = transforms[1:5]
    joint_origins = [transform[:3, 3] for transform in joint_transforms]
    joint_axes = [
        transform[:3, :3] @ (np.array(axis) / np.linalg.norm(axis))
        for transform, axis in zip(joint_transforms, geometry.joint_axes)
    ]

    jacobian = np.zeros((3, 4), dtype=float)
    for index, (axis, origin) in enumerate(zip(joint_axes, joint_origins)):
        jacobian[:, index] = np.cross(axis, tool_position - origin)
    return jacobian


def _forward_transforms(q: Sequence[float], geometry: ArmGeometry) -> Tuple[np.ndarray, ...]:
    """Return transforms at joint origins and the configured TCP."""
    transforms = [np.eye(4)]
    current = transforms[0]
    
    for angle, origin, rpy, axis in zip(q, geometry.joint_origins, geometry.joint_rpys, geometry.joint_axes):
        joint_origin = current @ _origin_transform(origin, rpy)
        transforms.append(joint_origin)
        current = joint_origin @ _axis_rotation(axis, angle)
        
    transforms.append(current @ _origin_transform(geometry.tcp_origin, geometry.tcp_rpy))
    return tuple(transforms)


# def _translation(x: float, y: float, z: float) -> np.ndarray:
#     transform = np.eye(4)
#     transform[:3, 3] = [x, y, z]
#     return transform


def _rpy_transform(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    return _homogeneous(
        _rotation_x(roll)[:3, :3]
        @ _rotation_y(pitch)[:3, :3]
        @ _rotation_z(yaw)[:3, :3]
    )


def _origin_transform(origin: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    transform = _rpy_transform(rpy)
    transform[:3, 3] = origin
    return transform


def _axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    unit_axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(unit_axis)
    if norm == 0:
        raise ValueError("revolute joint axis must be non-zero")
    x, y, z = unit_axis / norm
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([
        [cosine + x * x * (1 - cosine), x * y * (1 - cosine) - z * sine,
         x * z * (1 - cosine) + y * sine],
        [y * x * (1 - cosine) + z * sine, cosine + y * y * (1 - cosine),
         y * z * (1 - cosine) - x * sine],
        [z * x * (1 - cosine) - y * sine, z * y * (1 - cosine) + x * sine,
         cosine + z * z * (1 - cosine)],
    ])
    return _homogeneous(rotation)


def _homogeneous(rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return _homogeneous(np.array([
        [1.0, 0.0, 0.0],
        [0.0, cosine, -sine],
        [0.0, sine, cosine],
    ]))


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return _homogeneous(np.array([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ]))


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return _homogeneous(np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]))


def _segment_heights(q: Sequence[float], geometry: ArmGeometry) -> Tuple[float, ...]:
    transforms = _forward_transforms(q, geometry)
    return tuple(float(transform[2, 3]) for transform in transforms[1:])
