import pytest

from config import settings
from target_track_robot.motion_planner import MotionPlanner

FRAME_W, FRAME_H = 640, 480


@pytest.fixture
def planner():
    return MotionPlanner()


def _box(cx_ratio: float, height_ratio: float) -> tuple[int, int, int, int]:
    """Build a box whose centre is at the given fraction of the frame width, with height as a fraction of frame height."""
    h = int(FRAME_H * height_ratio)
    w = int(h * 0.5)
    cx = int(FRAME_W * cx_ratio)
    x = cx - w // 2
    y = FRAME_H - h  # box base sits at the bottom of the frame (natural for a standing person)
    return (x, y, w, h)


def test_no_target_and_no_search_mode_returns_stop(planner, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_SEARCH_MODE", False)
    cmd = planner.decide(None, FRAME_W, FRAME_H, target_locked=False)
    assert cmd == settings.CMD_STOP


def test_no_target_triggers_search_after_timeout(planner, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_SEARCH_MODE", True)
    monkeypatch.setattr(settings, "SEARCH_MODE_AFTER_LOST_SECONDS", 1.0)
    cmd = planner.decide(
        None, FRAME_W, FRAME_H, target_locked=False, seconds_since_lost=5.0
    )
    assert cmd == settings.SEARCH_ROTATE_COMMAND


def test_target_centered_far_moves_forward(planner):
    box = _box(cx_ratio=0.5, height_ratio=settings.TARGET_BOX_HEIGHT_RATIO_MIN - 0.05)
    cmd = planner.decide(box, FRAME_W, FRAME_H, target_locked=True)
    assert cmd == settings.CMD_FORWARD


def test_target_centered_too_close_moves_backward(planner):
    box = _box(cx_ratio=0.5, height_ratio=settings.TARGET_BOX_HEIGHT_RATIO_MAX + 0.05)
    cmd = planner.decide(box, FRAME_W, FRAME_H, target_locked=True)
    assert cmd == settings.CMD_BACKWARD


def test_target_centered_safe_distance_stops(planner):
    mid_ratio = (settings.TARGET_BOX_HEIGHT_RATIO_MIN + settings.TARGET_BOX_HEIGHT_RATIO_MAX) / 2
    box = _box(cx_ratio=0.5, height_ratio=mid_ratio)
    cmd = planner.decide(box, FRAME_W, FRAME_H, target_locked=True)
    assert cmd == settings.CMD_STOP


def test_target_to_the_right_turns_right(planner):
    mid_ratio = (settings.TARGET_BOX_HEIGHT_RATIO_MIN + settings.TARGET_BOX_HEIGHT_RATIO_MAX) / 2
    box = _box(cx_ratio=0.9, height_ratio=mid_ratio)
    cmd = planner.decide(box, FRAME_W, FRAME_H, target_locked=True)
    assert cmd == settings.CMD_RIGHT


def test_target_to_the_left_turns_left(planner):
    mid_ratio = (settings.TARGET_BOX_HEIGHT_RATIO_MIN + settings.TARGET_BOX_HEIGHT_RATIO_MAX) / 2
    box = _box(cx_ratio=0.1, height_ratio=mid_ratio)
    cmd = planner.decide(box, FRAME_W, FRAME_H, target_locked=True)
    assert cmd == settings.CMD_LEFT


def test_obstacle_in_path_triggers_avoidance_instead_of_forward(planner):
    # Target is far (robot intends to go forward) but there is an obstacle dead centre and close
    target_box = _box(cx_ratio=0.5, height_ratio=settings.TARGET_BOX_HEIGHT_RATIO_MIN - 0.1)
    obstacle_box = _box(cx_ratio=0.5, height_ratio=settings.OBSTACLE_BOX_HEIGHT_RATIO_TRIGGER + 0.1)

    cmd = planner.decide(
        target_box, FRAME_W, FRAME_H, obstacle_boxes=[obstacle_box], target_locked=True
    )
    assert cmd in (settings.CMD_LEFT, settings.CMD_RIGHT)


def test_avoidance_persists_for_configured_frames_then_resumes(planner, monkeypatch):
    monkeypatch.setattr(settings, "OBSTACLE_AVOID_STEER_FRAMES", 3)
    target_box = _box(cx_ratio=0.5, height_ratio=settings.TARGET_BOX_HEIGHT_RATIO_MIN - 0.1)
    obstacle_box = _box(cx_ratio=0.5, height_ratio=settings.OBSTACLE_BOX_HEIGHT_RATIO_TRIGGER + 0.1)

    first_cmd = planner.decide(
        target_box, FRAME_W, FRAME_H, obstacle_boxes=[obstacle_box], target_locked=True
    )
    assert first_cmd in (settings.CMD_LEFT, settings.CMD_RIGHT)

    # In subsequent frames (no new obstacle) the planner should keep the same avoidance direction
    second_cmd = planner.decide(target_box, FRAME_W, FRAME_H, obstacle_boxes=[], target_locked=True)
    third_cmd = planner.decide(target_box, FRAME_W, FRAME_H, obstacle_boxes=[], target_locked=True)
    assert second_cmd == first_cmd
    assert third_cmd == first_cmd

    # After the avoidance frames are exhausted, the robot should resume tracking (forward — still far)
    fourth_cmd = planner.decide(target_box, FRAME_W, FRAME_H, obstacle_boxes=[], target_locked=True)
    assert fourth_cmd == settings.CMD_FORWARD


def test_obstacle_far_away_does_not_trigger_avoidance(planner):
    target_box = _box(cx_ratio=0.5, height_ratio=settings.TARGET_BOX_HEIGHT_RATIO_MIN - 0.1)
    far_obstacle = _box(cx_ratio=0.5, height_ratio=0.05)  # very small box = far away obstacle

    cmd = planner.decide(
        target_box, FRAME_W, FRAME_H, obstacle_boxes=[far_obstacle], target_locked=True
    )
    assert cmd == settings.CMD_FORWARD


def test_invalid_command_never_produced(planner):
    box = _box(0.5, 0.4)
    cmd = planner.decide(box, FRAME_W, FRAME_H, target_locked=True)
    assert cmd in settings.VALID_COMMANDS
