import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ORDER = ("orange", "orange", "orange", "black", "black", "beige")


@dataclass(frozen=True)
class Waypoint:
    name: str
    x: float
    y: float
    z: float
    gripper: Optional[str] = None 
    wait: float = 0.0


@dataclass(frozen=True)
class Slot:
    color: str
    layer: int
    x: float
    y: float
    z: float  # tool height for the pillar centre when the block rests in the slot


def radial_unit(x: float, y: float) -> Tuple[float, float]:
    r = math.hypot(x, y)
    if r < 1e-6:
        raise ValueError("target is on the base axis, radial direction undefined")
    return x / r, y / r


def backoff(x: float, y: float, z: float, distance: float,
            pitch_rad: float) -> Tuple[float, float, float]:
    """Point distance behind (x, y, z) along the finger axis
    The finger axis points outward and down by abs(pitch), so backing off moves the tool
    inward and up
    """
    ux, uy = radial_unit(x, y)
    horizontal = distance * math.cos(pitch_rad)
    vertical = distance * math.sin(-pitch_rad)
    return x - ux * horizontal, y - uy * horizontal, z + vertical


def yaw_deg(x: float, y: float) -> float:
    return math.degrees(math.atan2(y, x))


def build_slots(cfg: Dict) -> Dict[str, List[Slot]]:
    yaw = math.radians(cfg["stack_yaw_deg"])
    radius = cfg["stack_radius"]
    pitch = cfg["block_width"] + cfg["row_gap"]
    layers = {
        "orange": (0, (1.0, 0.0, -1.0)),
        "black": (1, (0.5, -0.5)),
        "beige": (2, (0.0,)),
    }
    slots = {}
    for color, (layer, offsets) in layers.items():
        z = cfg["web_center_z"] + layer * cfg["block_height"]
        slots[color] = [
            Slot(color, layer,
                 radius * math.cos(yaw) - o * pitch * math.sin(yaw),
                 radius * math.sin(yaw) + o * pitch * math.cos(yaw), z)
            for o in offsets
        ]
    return slots


def approach_distances(cfg: Dict) -> List[float]:
    """Run-in distances to try, longest first (shorter ones help near the base, where the
    pre-point 8 cm back needs more wrist/elbow travel than the servos have)."""
    return [float(cfg["approach_distance"])] + [float(v) for v in cfg.get("approach_fallbacks", [])]


def pick_waypoints(x: float, y: float, cfg: Dict, distance: Optional[float] = None) -> List[Waypoint]:
    pitch = math.radians(cfg["grasp_pitch_deg"])
    d = cfg["approach_distance"] if distance is None else distance
    z = cfg["web_center_z"]
    travel, lift, settle = cfg["travel_z"], cfg["lift"], cfg["settle_time"]
    pre = backoff(x, y, z, d, pitch)
    carry = backoff(x, y, z + lift, d, pitch)
    return [
        Waypoint("above_pregrasp", pre[0], pre[1], travel, gripper="open"),
        Waypoint("pregrasp", pre[0], pre[1], pre[2], wait=settle),
        Waypoint("grasp", x, y, z, gripper="close", wait=cfg["gripper_close_time"]),
        Waypoint("lift", x, y, z + lift, wait=settle),
        Waypoint("carry_back", carry[0], carry[1], carry[2]),
        Waypoint("carry_up", carry[0], carry[1], travel),
    ]


def place_waypoints(slot: Slot, cfg: Dict, distance: Optional[float] = None) -> List[Waypoint]:
    pitch = math.radians(cfg["grasp_pitch_deg"])
    d = cfg["approach_distance"] if distance is None else distance
    travel, settle = cfg["travel_z"], cfg["settle_time"]
    z = slot.z + cfg["release_margin"]  # release a few mm above contact
    pre = backoff(slot.x, slot.y, z, d, pitch)
    return [
        Waypoint("above_preplace", pre[0], pre[1], travel),
        Waypoint("preplace", pre[0], pre[1], pre[2], wait=settle),
        Waypoint("place", slot.x, slot.y, z, gripper="open", wait=cfg["gripper_open_time"]),
        Waypoint("withdraw", pre[0], pre[1], pre[2], wait=settle),
        Waypoint("withdraw_up", pre[0], pre[1], travel),
    ]


def select_block(blocks: Iterable, color: str, cfg: Dict,
                 blacklist: Sequence[Tuple[float, float]] = ()):
    """Most isolated reachable block of `color` with a clear radial approach corridor.

    * detections whose yaw is below `ignore_yaw_below_deg` are ignored, which hides blocks
      already placed in the stacking wedge (and their parallax-shifted images)
    * the corridor is the strip inward of the block (width +-corridor_half_width, length
      approach_distance + corridor_extra) in which no other block may stand
    * blocks closer than `min_clearance` to another block are skipped (they would merge in
      the detector and the fingers would not fit)
    """
    live = [b for b in blocks if yaw_deg(b.x, b.y) >= cfg["ignore_yaw_below_deg"]]
    corridor_len = cfg["approach_distance"] + cfg["corridor_extra"]
    best, best_isolation = None, -1.0
    for block in live:
        if block.color != color:
            continue
        r = math.hypot(block.x, block.y)
        if not (cfg["pick_r_min"] <= r <= cfg["pick_r_max"]):
            continue
        if any(math.hypot(block.x - bx, block.y - by) < cfg["blacklist_radius"]
               for bx, by in blacklist):
            continue
        ux, uy = radial_unit(block.x, block.y)
        blocked, nearest = False, float("inf")
        for other in live:
            if other is block:
                continue
            dx, dy = other.x - block.x, other.y - block.y
            dist = math.hypot(dx, dy)
            nearest = min(nearest, dist)
            s = dx * ux + dy * uy      # + outward
            t = -dx * uy + dy * ux     # lateral
            if dist < cfg["min_clearance"] or (
                    -corridor_len <= s <= 0.0 and abs(t) <= cfg["corridor_half_width"]):
                blocked = True
                break
        if blocked:
            continue
        if nearest > best_isolation:
            best, best_isolation = block, nearest
    return best