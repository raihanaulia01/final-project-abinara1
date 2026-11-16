"""Run with: PYTHONPATH=src python3 -m pytest test/   (no ROS needed; uses the fake MCU over a pty)"""
import os
import select
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from arm_hardware import protocol as P  # noqa: E402
from arm_hardware.serial_link import SerialLink  # noqa: E402


class FakeMcu:
    def __init__(self, link):
        self.link = link
        self.proc = subprocess.Popen([sys.executable, os.path.join(HERE, "..", "scripts", "fake_mcu.py"),
                                      "--link", link], stdout=subprocess.PIPE, bufsize=0)
        self.fd = self.proc.stdout.fileno()
        self.pending = b""
        for _ in range(100):
            if os.path.lexists(link):
                break
            time.sleep(0.05)

    def lines(self, seconds=0.5):
        """Lines the fake printed during the next `seconds`. Reads the raw fd: select() on a
        buffered file object can miss lines that are already sitting in Python's buffer."""
        out, end = [], time.time() + seconds
        while time.time() < end:
            if select.select([self.fd], [], [], 0.05)[0]:
                self.pending += os.read(self.fd, 4096)
            while b"\n" in self.pending:
                line, self.pending = self.pending.split(b"\n", 1)
                out.append(line.decode().strip())
        return out

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=3)


@pytest.fixture
def mcu(tmp_path):
    m = FakeMcu(str(tmp_path / "mcu"))
    yield m
    m.close()


def wait_for(cond, seconds=3.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_ping_pong_and_ultrasonic_stream(mcu):
    got = []
    link = SerialLink(mcu.link, 115200, lambda cmd, payload: got.append((cmd, payload)))
    try:
        link.send(P.encode_ping())
        assert wait_for(lambda: any(c == P.CMD_PONG for c, _ in got)), "no PONG"
        assert wait_for(lambda: sum(c == P.CMD_ULTRASONIC for c, _ in got) >= 3), "no ultrasonic frames"
        mm = P.decode_ultrasonic([p for c, p in got if c == P.CMD_ULTRASONIC][-1])
        assert 170 <= mm <= 190
    finally:
        link.close()


def test_set_angles_arrive_and_corrupt_frames_are_dropped(mcu):
    link = SerialLink(mcu.link, 115200, lambda *a: None)
    try:
        bad = bytearray(P.encode_set_angles([150, 150, 150, 150]))
        bad[5] ^= 0x40                                          # corrupt a payload byte
        link.send(bytes(bad))
        link.send(P.encode_set_angles([90, 45, -60, 0]))
        lines = mcu.lines(0.6)
        assert any(l.startswith("bad CRC") for l in lines), lines
        assert "angles 90.0 45.0 -60.0 0.0" in lines, lines
        assert not any("150" in l for l in lines), lines
    finally:
        link.close()