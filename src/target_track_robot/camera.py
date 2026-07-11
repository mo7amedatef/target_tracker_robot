"""
camera: Reads the camera in a dedicated background thread so the main loop
is never blocked waiting on I/O, and always receives the most recent
available frame (older buffered frames are discarded).
This is critical on Raspberry Pi to minimise tracking latency.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import settings
from target_track_robot.utils.logger import get_logger

logger = get_logger("camera")


class CameraStream:
    def __init__(
        self,
        index: int = settings.CAMERA_INDEX,
        width: int = settings.CAMERA_WIDTH,
        height: int = settings.CAMERA_HEIGHT,
        fps: int = settings.CAMERA_FPS,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, settings.CAMERA_BUFFER_SIZE)
        except Exception:  # noqa: BLE001 - not all drivers support this property
            pass

        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        logger.info("Camera opened index=%s (%dx%d)", self.index, self.width, self.height)

    def _update_loop(self) -> None:
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        logger.info("Camera closed")

    def __enter__(self) -> "CameraStream":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
