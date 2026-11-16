import threading

import serial

from arm_hardware.protocol import FrameParser


class SerialLink:
    def __init__(self, port, baud, on_frame, on_error=None):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.on_frame = on_frame
        self.on_error = on_error
        self.parser = FrameParser()
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def send(self, frame: bytes):
        with self._lock:
            self.ser.write(frame)

    def reset_input_buffer(self):
        with self._lock:
            self.ser.reset_input_buffer()
            self.parser.buf.clear()

    def close(self):
        self._running = False
        self._thread.join(timeout=0.5)
        try:
            self.ser.close()
        except Exception:
            pass

    def _read_loop(self):
        while self._running:
            try:
                chunk = self.ser.read(max(1, self.ser.in_waiting))
            except (serial.SerialException, OSError) as e:
                if self._running and self.on_error:
                    self.on_error(e)
                return
            for cmd, payload in self.parser.feed(chunk):
                self.on_frame(cmd, payload)