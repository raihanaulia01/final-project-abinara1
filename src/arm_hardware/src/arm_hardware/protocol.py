"""Binary frame protocol between the PC and the STM32.
AA 55 | LEN | CMD | PAYLOAD (LEN bytes, 0..32) | CRC          frame length = LEN + 5

CRC = CRC-8 (poly 0x07, init 0, no reflection, no final xor) over LEN, CMD and PAYLOAD.
"""
import struct
from typing import Dict, List, Optional, Sequence, Tuple

HEADER = b"\xAA\x55"
MAX_PAYLOAD = 32

# Host to MCU commands
CMD_PING = 0x01          # LEN 0: keepalive & version query
CMD_SET_ANGLES = 0x02    # LEN 12: <5hH (5 x int16 mrad + uint16 duration_ms)
CMD_HOME = 0x03          # LEN 2:  <H   (uint16 duration_ms)
CMD_STOP = 0x04          # LEN 0:  freeze motion immediately (PWM stays active)
CMD_GET_STATUS = 0x05    # LEN 0:  request single STATUS frame
CMD_STREAM = 0x06        # LEN 3:  <BH  (uint8 enable 0/1 + uint16 period_ms)
CMD_GET_DIAG = 0x07      # LEN 0:  request single DIAG frame
CMD_SET_JOINT = 0x08     # LEN 5:  <BhH (uint8 joint_id + int16 mrad + uint16 duration_ms)

# MCU to Host responses
CMD_ACK = 0x81           # LEN 2..5: [orig_cmd, err_code, ...extra]
CMD_STATUS = 0x82        # LEN 14:   <5hHH (5 x int16 mrad + uint16 dist_mm + uint16 flags)
CMD_DIAG = 0x83          # LEN 24:   <12H  (12 cumulative uint16 diagnostic counters)

# ACK error codes
ERR_OK = 0x00
ERR_OK_CLAMPED = 0x01    # extra[0] = bitmask of clamped joints (bit 0 = BASE .. bit 4 = GRIP)
ERR_BAD_LEN = 0x10
ERR_UNKNOWN_CMD = 0x11
ERR_BAD_ARG = 0x12

ERR_NAMES = {
    ERR_OK: "OK",
    ERR_OK_CLAMPED: "OK_CLAMPED",
    ERR_BAD_LEN: "BAD_LEN",
    ERR_UNKNOWN_CMD: "UNKNOWN_CMD",
    ERR_BAD_ARG: "BAD_ARG",
}

# STATUS bit flags
FLAG_FAILSAFE = 1 << 0
FLAG_MOVING = 1 << 1
FLAG_CLAMPED_LAST = 1 << 2
FLAG_ECHO_VALID = 1 << 3
FLAG_RX_OVERFLOW_SEEN = 1 << 4
FLAG_BOOT_FRESH = 1 << 5

# Joint indices & constants
JOINT_BASE = 0
JOINT_ARM1 = 1
JOINT_ARM2 = 2
JOINT_ARM3 = 3
JOINT_GRIP = 4

NUM_JOINTS = 5
PROTOCOL_VERSION = 1
NO_ECHO = 0xFFFF

DIAG_FIELDS = (
    "rx_bytes",
    "frames_ok",
    "err_crc",
    "err_len",
    "err_unknown_cmd",
    "rx_overflow",
    "uart_errors",
    "tx_drops",
    "parser_timeouts",
    "echo_timeouts",
    "failsafe_events",
    "clamp_events",
)


def crc8(data: bytes, poly: int = 0x07, init: int = 0x00) -> int:
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def build_frame(cmd: int, payload: bytes = b"") -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("payload is %d bytes, max %d" % (len(payload), MAX_PAYLOAD))
    body = bytes([len(payload), cmd]) + payload
    return HEADER + body + bytes([crc8(body)])


def rad_to_mrad(rad: float) -> int:
    return max(-32768, min(32767, int(round(rad * 1000.0))))


def mrad_to_rad(mrad: int) -> float:
    return float(mrad) / 1000.0


def clamp_u16(value: int) -> int:
    return max(0, min(65535, int(value)))


class FrameParser:
    """Feed it raw bytes in any chunking; it returns complete, CRC-valid (cmd, payload) frames.
    Garbage and corrupted frames are skipped by re-synchronising on the next AA 55."""

    def __init__(self):
        self.buf = bytearray()
        self.crc_errors = 0

    def feed(self, data: bytes) -> List[Tuple[int, bytes]]:
        self.buf.extend(data)
        frames = []
        while True:
            start = self.buf.find(HEADER)
            if start < 0:
                keep = 1 if self.buf.endswith(HEADER[:1]) else 0  # trailing AA may start a header
                del self.buf[:len(self.buf) - keep]
                break
            del self.buf[:start]
            if len(self.buf) < 3:
                break
            length = self.buf[2]
            if length > MAX_PAYLOAD:
                del self.buf[:1]
                continue
            total = length + 5
            if len(self.buf) < total:
                break
            if crc8(bytes(self.buf[2:total - 1])) == self.buf[total - 1]:
                frames.append((self.buf[3], bytes(self.buf[4:4 + length])))
                del self.buf[:total]
            else:
                self.crc_errors += 1
                del self.buf[:1]
        return frames


# Host -> MCU encoders

def encode_ping() -> bytes:
    return build_frame(CMD_PING)


def encode_set_angles(angles_rad: Sequence[float], duration_ms: int) -> bytes:
    if len(angles_rad) != NUM_JOINTS:
        raise ValueError("need %d angles, got %d" % (NUM_JOINTS, len(angles_rad)))
    mrad_vals = [rad_to_mrad(r) for r in angles_rad]
    payload = struct.pack("<5hH", *mrad_vals, clamp_u16(duration_ms))
    return build_frame(CMD_SET_ANGLES, payload)


def encode_home(duration_ms: int) -> bytes:
    return build_frame(CMD_HOME, struct.pack("<H", clamp_u16(duration_ms)))


def encode_stop() -> bytes:
    return build_frame(CMD_STOP)


def encode_get_status() -> bytes:
    return build_frame(CMD_GET_STATUS)


def encode_stream(enable: bool, period_ms: int = 50) -> bytes:
    return build_frame(CMD_STREAM, struct.pack("<BH", 1 if enable else 0, clamp_u16(period_ms)))


def encode_get_diag() -> bytes:
    return build_frame(CMD_GET_DIAG)


def encode_set_joint(joint_id: int, angle_rad: float, duration_ms: int) -> bytes:
    if not (0 <= joint_id < NUM_JOINTS):
        raise ValueError("joint_id must be 0..%d, got %d" % (NUM_JOINTS - 1, joint_id))
    payload = struct.pack("<BhH", joint_id, rad_to_mrad(angle_rad), clamp_u16(duration_ms))
    return build_frame(CMD_SET_JOINT, payload)

def decode_set_angles(payload: bytes) -> Optional[Tuple[List[float], int]]:
    """Returns (angles_rad[5], duration_ms) or None if malformed."""
    if len(payload) != 12:
        return None
    unpacked = struct.unpack("<5hH", payload)
    return [mrad_to_rad(v) for v in unpacked[:5]], unpacked[5]


def decode_set_joint(payload: bytes) -> Optional[Tuple[int, float, int]]:
    """Returns (joint_id, angle_rad, duration_ms) or None if malformed."""
    if len(payload) != 5:
        return None
    joint_id, mrad, dur_ms = struct.unpack("<BhH", payload)
    return joint_id, mrad_to_rad(mrad), dur_ms


def decode_ack(payload: bytes) -> Optional[Tuple[int, int, bytes]]:
    """Returns (orig_cmd, err_code, extra_bytes) or None if malformed."""
    if len(payload) < 2:
        return None
    return payload[0], payload[1], payload[2:]


def decode_status(payload: bytes) -> Optional[Tuple[List[float], int, int]]:
    """Returns (commanded_angles_rad[5], dist_mm_or_minus_1, flags) or None if malformed."""
    if len(payload) != 14:
        return None
    unpacked = struct.unpack("<5hHH", payload)
    angles_rad = [mrad_to_rad(v) for v in unpacked[:5]]
    dist_mm = -1 if unpacked[5] == NO_ECHO else unpacked[5]
    flags = unpacked[6]
    return angles_rad, dist_mm, flags


def decode_diag(payload: bytes) -> Optional[Dict[str, int]]:
    """Returns dict of 12 diagnostic counters or None if malformed."""
    if len(payload) != 24:
        return None
    values = struct.unpack("<12H", payload)
    return dict(zip(DIAG_FIELDS, values))