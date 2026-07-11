"""
SerialCommander: Responsible for sending movement commands (F/B/L/R/S) to
the motor driver via the Serial Port.

Features:
  - Only sends a command when it differs from the previous one (no port flooding).
  - Enforces a minimum time gap between two transmissions (throttle).
  - If the connection drops, automatically attempts to reconnect without stopping the program.
  - Accepts any SerialLike object via dependency injection so unit tests can run
    without real hardware.
"""

from __future__ import annotations

import time
from typing import Optional, Protocol

from config import settings
from target_track_robot.utils.logger import get_logger

logger = get_logger("serial_comm")


class SerialLike(Protocol):
    """Any object that provides this interface can be injected in place of pyserial.Serial."""

    is_open: bool

    def write(self, data: bytes) -> int: ...
    def close(self) -> None: ...


class SerialCommander:
    def __init__(
        self,
        port: str = settings.SERIAL_PORT,
        baudrate: int = settings.SERIAL_BAUDRATE,
        timeout: float = settings.SERIAL_TIMEOUT,
        min_write_interval: float = settings.SERIAL_WRITE_MIN_INTERVAL,
        reconnect_interval: float = settings.SERIAL_RECONNECT_INTERVAL,
        serial_factory=None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.min_write_interval = min_write_interval
        self.reconnect_interval = reconnect_interval

        # serial_factory: a callable returning a SerialLike object — used for test injection
        self._serial_factory = serial_factory or self._default_serial_factory

        self._conn: Optional[SerialLike] = None
        self._last_command: Optional[str] = None
        self._last_write_time: float = 0.0
        self._last_connect_attempt: float = 0.0

        self.connect()

    # ------------------------------------------------------------------
    @staticmethod
    def _default_serial_factory(port: str, baudrate: int, timeout: float):
        import serial  # pyserial — deferred import so tests don't require hardware

        return serial.Serial(port=port, baudrate=baudrate, timeout=timeout)

    def connect(self) -> bool:
        self._last_connect_attempt = time.monotonic()
        try:
            self._conn = self._serial_factory(self.port, self.baudrate, self.timeout)
            logger.info("Serial port connected: %s @ %d", self.port, self.baudrate)
            return True
        except Exception as exc:  # noqa: BLE001 - catch any hardware error
            logger.warning("Failed to connect to serial port (%s): %s", self.port, exc)
            self._conn = None
            return False

    @property
    def is_connected(self) -> bool:
        return self._conn is not None and getattr(self._conn, "is_open", True)

    # ------------------------------------------------------------------
    def send_command(self, command: str, force: bool = False) -> bool:
        """Send a single movement command if it differs from the last one (or if
        force=True) and the minimum write interval has elapsed.

        Returns True if the command was actually transmitted, False if skipped or failed.
        """
        if command not in settings.VALID_COMMANDS:
            raise ValueError(f"Unknown command: {command!r}")

        now = time.monotonic()

        if not force and command == self._last_command:
            return False

        if not force and (now - self._last_write_time) < self.min_write_interval:
            return False

        if not self.is_connected:
            if (now - self._last_connect_attempt) >= self.reconnect_interval:
                self.connect()
            if not self.is_connected:
                return False

        try:
            self._conn.write(command.encode("ascii"))  # type: ignore[union-attr]
            self._last_command = command
            self._last_write_time = now
            logger.debug("Command sent: %s", command)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to send command %s: %s", command, exc)
            self._conn = None
            return False

    def stop(self) -> None:
        """Send a forced stop command (used on shutdown or in emergencies)."""
        self.send_command(settings.CMD_STOP, force=True)

    def close(self) -> None:
        try:
            self.stop()
        finally:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:  # noqa: BLE001
                    pass
            self._conn = None

    def __enter__(self) -> "SerialCommander":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
