"""
MotionPlanner: The core of the driving logic.

Converts:
  - The position and size of the target person's bounding box
  - A list of obstacle bounding boxes
into a single movement command: F / B / L / R / S

Priority order:
  1) If obstacle avoidance is currently active, continue the avoidance
     direction for the configured number of frames.
  2) If an obstacle is blocking the path and we intend to move forward,
     start avoidance (R or L depending on obstacle position) instead of
     driving into it.
  3) If there is no locked target, enter Search Mode or stop.
  4) If the target is locked, correct horizontal alignment first
     (left/right), then adjust distance (forward/backward) once centred.

This class is pure logic (no camera, no serial) so it is fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from config import settings
from target_track_robot.utils import geometry
from target_track_robot.utils.logger import get_logger

logger = get_logger("motion_planner")

BoxT = tuple[int, int, int, int]


@dataclass
class PlannerState:
    """Internal planner state preserved between frames."""

    avoiding: bool = False
    avoid_direction: Optional[str] = None
    avoid_frames_left: int = 0
    lost_frames: int = 0
    last_command: str = settings.CMD_STOP


class MotionPlanner:
    def __init__(self) -> None:
        self.state = PlannerState()

    # ------------------------------------------------------------------
    # Main decision function
    # ------------------------------------------------------------------
    def decide(
        self,
        target_box: Optional[BoxT],
        frame_width: int,
        frame_height: int,
        obstacle_boxes: Optional[Iterable[BoxT]] = None,
        target_locked: bool = False,
        seconds_since_lost: float = 0.0,
    ) -> str:
        obstacle_boxes = list(obstacle_boxes or [])

        # 1) Currently avoiding an obstacle — keep going
        if self.state.avoiding:
            self.state.avoid_frames_left -= 1
            if self.state.avoid_frames_left <= 0:
                self.state.avoiding = False
                direction = self.state.avoid_direction
                self.state.avoid_direction = None
                logger.debug("Obstacle avoidance complete")
            else:
                return self._commit(self.state.avoid_direction or settings.CMD_STOP)

        # 2) Do we intend to move forward? (target is far or missing and we want to advance)
        intends_forward = self._intends_forward(target_box, frame_height, target_locked)

        if intends_forward:
            blocking_obstacle = self._find_blocking_obstacle(
                obstacle_boxes, frame_width, frame_height
            )
            if blocking_obstacle is not None:
                direction = self._choose_avoid_direction(blocking_obstacle, frame_width)
                self.state.avoiding = True
                self.state.avoid_direction = direction
                self.state.avoid_frames_left = settings.OBSTACLE_AVOID_STEER_FRAMES
                logger.info("Obstacle in path -> starting avoidance towards %s", direction)
                return self._commit(direction)

        # 3) No locked target
        if target_box is None or not target_locked:
            self.state.lost_frames += 1
            if (
                settings.ENABLE_SEARCH_MODE
                and seconds_since_lost >= settings.SEARCH_MODE_AFTER_LOST_SECONDS
            ):
                return self._commit(settings.SEARCH_ROTATE_COMMAND)
            return self._commit(settings.CMD_STOP)

        self.state.lost_frames = 0

        # 4) Target is locked — compute the appropriate command
        return self._commit(self._track_command(target_box, frame_width, frame_height))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _commit(self, command: str) -> str:
        self.state.last_command = command
        return command

    @staticmethod
    def _intends_forward(
        target_box: Optional[BoxT], frame_height: int, target_locked: bool
    ) -> bool:
        if target_box is None or not target_locked:
            return False
        ratio = geometry.box_height_ratio(target_box, frame_height)
        return ratio < settings.TARGET_BOX_HEIGHT_RATIO_MIN

    @staticmethod
    def _find_blocking_obstacle(
        obstacle_boxes: list[BoxT], frame_width: int, frame_height: int
    ) -> Optional[BoxT]:
        for box in obstacle_boxes:
            height_ratio = geometry.box_height_ratio(box, frame_height)
            if height_ratio < settings.OBSTACLE_BOX_HEIGHT_RATIO_TRIGGER:
                continue  # still too far away, no need to dodge yet
            if not geometry.is_box_in_lower_region(
                box, frame_height, settings.OBSTACLE_PATH_ZONE_Y_RATIO
            ):
                continue
            if not geometry.is_box_in_path_zone(
                box, frame_width, settings.OBSTACLE_PATH_ZONE_X_MARGIN_RATIO
            ):
                continue
            return box
        return None

    @staticmethod
    def _choose_avoid_direction(obstacle_box: BoxT, frame_width: int) -> str:
        """If the obstacle is left of centre -> dodge right. If right of centre -> dodge left."""
        offset = geometry.horizontal_offset_ratio(obstacle_box, frame_width)
        if offset <= 0:
            return settings.CMD_RIGHT
        return settings.CMD_LEFT

    @staticmethod
    def _track_command(target_box: BoxT, frame_width: int, frame_height: int) -> str:
        offset_ratio = geometry.horizontal_offset_ratio(target_box, frame_width)

        # Correct horizontal alignment first
        if offset_ratio > settings.CENTER_DEADZONE_RATIO:
            return settings.CMD_RIGHT
        if offset_ratio < -settings.CENTER_DEADZONE_RATIO:
            return settings.CMD_LEFT

        # Target is horizontally centred — adjust distance (box height)
        height_ratio = geometry.box_height_ratio(target_box, frame_height)
        if height_ratio < settings.TARGET_BOX_HEIGHT_RATIO_MIN:
            return settings.CMD_FORWARD
        if height_ratio > settings.TARGET_BOX_HEIGHT_RATIO_MAX:
            return settings.CMD_BACKWARD
        return settings.CMD_STOP

    def reset(self) -> None:
        self.state = PlannerState()
