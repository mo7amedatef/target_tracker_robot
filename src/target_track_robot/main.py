"""
main: The robot's primary entry point.

Per-frame work is split into two modes depending on whether the target is
currently locked:

  SEARCHING (not locked)
  ──────────────────────
  Every DETECTION_EVERY_N_FRAMES:
    • Run MobileNet-SSD ONCE  →  persons list  +  obstacle boxes
    • Pass persons to FaceTracker.process()
    • Face DNN + LBPH run inside FaceTracker only on those frames
  On all other frames: only the lightweight motion planner runs (instant).

  LOCKED (CSRT/KCF tracker active)
  ─────────────────────────────────
  Every frame:     KCF/CSRT tracker update (fast, ~5 ms)
  Every N frames:  MobileNet-SSD for obstacles only
  Every M frames:  Face DNN re-verify (inside FaceTracker, no MobileNet-SSD)

This design ensures MobileNet-SSD is NEVER called more than once per frame,
cutting the two-DNN-call bottleneck that caused slowness on the Pi 5.
"""

from __future__ import annotations

import signal
import time
from typing import List

import cv2

from config import settings
from target_track_robot.camera import CameraStream
from target_track_robot.face_tracker import FaceTracker, TrackResult
from target_track_robot.motion_planner import MotionPlanner
from target_track_robot.object_detector import Detection, ObjectDetector
from target_track_robot.serial_comm import SerialCommander
from target_track_robot.utils.logger import get_logger

logger = get_logger("main")

_running = True


def _handle_shutdown_signal(signum, frame) -> None:  # noqa: ARG001
    global _running
    logger.info("Shutdown signal received — stopping robot safely...")
    _running = False


def run() -> None:
    global _running
    _running = True
    signal.signal(signal.SIGINT,  _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    logger.info("=== target_track_robot starting ===")
    logger.info(
        "Config: %d fps target | camera %dx%d | tracker %s | "
        "detection every %d frames | obstacle check every %d frames",
        settings.MAIN_LOOP_TARGET_FPS,
        settings.CAMERA_WIDTH, settings.CAMERA_HEIGHT,
        settings.TRACKER_TYPE,
        settings.DETECTION_EVERY_N_FRAMES,
        settings.OBSTACLE_DETECT_EVERY_N_FRAMES,
    )

    object_detector = ObjectDetector()
    face_tracker    = FaceTracker(object_detector=object_detector)
    planner         = MotionPlanner()

    frame_interval = 1.0 / settings.MAIN_LOOP_TARGET_FPS
    frame_count    = 0

    # Carry-over state between frames
    obstacle_boxes: list[tuple[int, int, int, int]] = []
    persons:        List[Detection]                  = []
    track_result    = TrackResult(box=None, locked=False, seconds_since_lost=0.0)
    command         = settings.CMD_STOP

    with CameraStream() as camera, SerialCommander() as serial:
        # Wait for the first valid frame before entering the main loop
        for _ in range(100):
            if camera.read() is not None:
                break
            time.sleep(0.05)

        perf_frame_count  = 0
        perf_window_start = time.monotonic()

        while _running:
            loop_start = time.monotonic()

            frame = camera.read()
            if frame is None:
                time.sleep(0.02)
                continue

            frame_count      += 1
            perf_frame_count += 1
            frame_h, frame_w  = frame.shape[:2]

            # ── Detection scheduling ──────────────────────────────────────
            if not track_result.locked:
                # SEARCHING: run MobileNet-SSD once → persons + obstacles together
                if frame_count % settings.DETECTION_EVERY_N_FRAMES == 0:
                    all_dets = object_detector.detect(frame)
                    persons = [
                        d for d in all_dets
                        if d.label == "person"
                        and d.confidence >= settings.PERSON_DETECTION_MIN_CONFIDENCE
                    ]
                    obstacle_boxes = [
                        d.box for d in all_dets
                        if d.label in settings.OBSTACLE_CLASSES
                        and d.confidence >= settings.OBSTACLE_DETECTION_CONFIDENCE
                    ]
                else:
                    # Non-detection frame: pass nothing → tracker skips acquisition
                    persons = None  # type: ignore[assignment]
            else:
                # LOCKED: no person detection needed (tracker handles it),
                # only update obstacle boxes periodically
                persons = None  # type: ignore[assignment]
                if frame_count % settings.OBSTACLE_DETECT_EVERY_N_FRAMES == 0:
                    obstacle_boxes = [
                        d.box for d in object_detector.detect_obstacles(frame)
                    ]

            # ── Face tracker ──────────────────────────────────────────────
            track_result = face_tracker.process(frame, precomputed_persons=persons)

            # ── Motion planning ───────────────────────────────────────────
            command = planner.decide(
                target_box=track_result.box,
                frame_width=frame_w,
                frame_height=frame_h,
                obstacle_boxes=obstacle_boxes,
                target_locked=track_result.locked,
                seconds_since_lost=track_result.seconds_since_lost,
            )

            # ── Serial output ─────────────────────────────────────────────
            serial.send_command(command)

            # ── Optional debug window ─────────────────────────────────────
            if settings.SHOW_DEBUG_WINDOW:
                _draw_debug(frame, track_result.box, obstacle_boxes, command,
                            track_result.locked)
                cv2.imshow("target_track_robot", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ── FPS performance logger ────────────────────────────────────
            if settings.PERF_LOG_INTERVAL_SECONDS > 0:
                elapsed_window = time.monotonic() - perf_window_start
                if elapsed_window >= settings.PERF_LOG_INTERVAL_SECONDS:
                    actual_fps = perf_frame_count / elapsed_window
                    loop_ms    = (time.monotonic() - loop_start) * 1000
                    status     = "LOCKED" if track_result.locked else "searching"
                    logger.info(
                        "PERF | %.1f fps (target %d) | loop %.1f ms | %s | cmd: %s",
                        actual_fps, settings.MAIN_LOOP_TARGET_FPS,
                        loop_ms, status, command,
                    )
                    perf_frame_count  = 0
                    perf_window_start = time.monotonic()
            # ─────────────────────────────────────────────────────────────

            # Sleep just enough to hit the target FPS
            elapsed    = time.monotonic() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    logger.info("=== target_track_robot stopped safely ===")
    if settings.SHOW_DEBUG_WINDOW:
        cv2.destroyAllWindows()


def _draw_debug(frame, target_box, obstacle_boxes, command, locked: bool) -> None:
    colour = (0, 255, 0) if locked else (0, 165, 255)  # green = locked, orange = searching
    if target_box is not None:
        x, y, w, h = target_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        status_text = "LOCKED" if locked else "SEARCHING"
        cv2.putText(frame, status_text, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    for ox, oy, ow, oh in obstacle_boxes:
        cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 2)
    cv2.putText(frame, f"CMD: {command}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


if __name__ == "__main__":
    run()
