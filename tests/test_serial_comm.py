import time

import pytest

from config import settings
from target_track_robot.serial_comm import SerialCommander


class FakeSerial:
    """Fake drop-in replacement for pyserial.Serial so tests run without real hardware."""

    def __init__(self, fail_to_open: bool = False):
        if fail_to_open:
            raise RuntimeError("Simulated port-open failure")
        self.is_open = True
        self.written: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.written.append(data)
        return len(data)

    def close(self) -> None:
        self.is_open = False


@pytest.fixture
def fake_factory():
    created = {}

    def factory(port, baudrate, timeout):
        s = FakeSerial()
        created["instance"] = s
        return s

    factory.created = created
    return factory


def test_connects_on_init(fake_factory):
    commander = SerialCommander(serial_factory=fake_factory, min_write_interval=0.0)
    assert commander.is_connected is True


def test_sends_command_when_changed(fake_factory):
    commander = SerialCommander(serial_factory=fake_factory, min_write_interval=0.0)
    sent = commander.send_command(settings.CMD_FORWARD)
    assert sent is True
    assert fake_factory.created["instance"].written == [b"F"]


def test_does_not_resend_same_command(fake_factory):
    commander = SerialCommander(serial_factory=fake_factory, min_write_interval=0.0)
    commander.send_command(settings.CMD_FORWARD)
    sent_again = commander.send_command(settings.CMD_FORWARD)
    assert sent_again is False
    assert fake_factory.created["instance"].written == [b"F"]


def test_resends_when_command_changes(fake_factory):
    commander = SerialCommander(serial_factory=fake_factory, min_write_interval=0.0)
    commander.send_command(settings.CMD_FORWARD)
    commander.send_command(settings.CMD_LEFT)
    assert fake_factory.created["instance"].written == [b"F", b"L"]


def test_throttles_rapid_writes(fake_factory):
    commander = SerialCommander(serial_factory=fake_factory, min_write_interval=1.0)
    commander.send_command(settings.CMD_FORWARD)
    sent = commander.send_command(settings.CMD_LEFT)  # command changed but too soon (throttle)
    assert sent is False


def test_force_bypasses_dedupe(fake_factory):
    commander = SerialCommander(serial_factory=fake_factory, min_write_interval=0.0)
    commander.send_command(settings.CMD_STOP)
    sent = commander.send_command(settings.CMD_STOP, force=True)
    assert sent is True


def test_invalid_command_raises(fake_factory):
    commander = SerialCommander(serial_factory=fake_factory, min_write_interval=0.0)
    with pytest.raises(ValueError):
        commander.send_command("X")


def test_reconnects_when_disconnected():
    attempts = {"count": 0}

    def flaky_factory(port, baudrate, timeout):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("First connection failed")
        return FakeSerial()

    commander = SerialCommander(
        serial_factory=flaky_factory, min_write_interval=0.0, reconnect_interval=0.0
    )
    assert commander.is_connected is False

    sent = commander.send_command(settings.CMD_FORWARD)
    assert sent is True
    assert commander.is_connected is True


def test_stop_on_close(fake_factory):
    commander = SerialCommander(serial_factory=fake_factory, min_write_interval=0.0)
    commander.send_command(settings.CMD_FORWARD)
    commander.close()
    assert fake_factory.created["instance"].written[-1] == b"S"


def test_context_manager_sends_stop_on_exit(fake_factory):
    with SerialCommander(serial_factory=fake_factory, min_write_interval=0.0) as commander:
        commander.send_command(settings.CMD_RIGHT)
    assert fake_factory.created["instance"].written[-1] == b"S"
