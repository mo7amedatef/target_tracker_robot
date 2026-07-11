"""
FaceTracker: Responsible for "locking" the robot onto the target person only
(not just anyone in front of the camera) and maintaining that lock even when
the face is temporarily hidden (e.g. while the person is walking sideways).

Strategy:
  1) While not yet locked: the main loop runs MobileNet-SSD once and passes
     the pre-computed person list here.  For each person we inspect the head
     region with the face DNN and verify identity with LBPH.  After
     MIN_LOCK_CONFIDENCE_FRAMES consecutive matches we acquire a CSRT/KCF lock.
  2) While locked: use the tracker to follow the person frame by frame (much
     faster than re-running face recognition every frame).  Every
     FACE_REVERIFY_EVERY_N_FRAMES frames we re-verify identity to catch drift.
  3) If tracking is lost for more than MAX_LOST_FRAMES: release the lock and
     start searching again.

Performance note:
  The MobileNet-SSD forward pass is performed ONCE in main.py and the result is
  passed in as `precomputed_persons`.  This eliminates the duplicate DNN call
  that would otherwise happen (once here, once for obstacle detection).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from config import settings
from target_track_robot.object_detector import Detection, ObjectDetector
from target_track_robot.utils.logger import get_logger

logger = get_logger("face_tracker")

BoxT = tuple[int, int, int, int]
TARGET_LABEL = 1


@dataclass
class TrackResult:
    box: Optional[BoxT]
    locked: bool
    seconds_since_lost: float = 0.0


def _create_tracker():
    """Create a KCF or CSRT tracker, compatible with different OpenCV versions."""
    tracker_type = getattr(settings, "TRACKER_TYPE", "KCF").upper()

    # OpenCV ≥ 4.5 moved trackers to cv2.legacy
    legacy = getattr(cv2, "legacy", cv2)

    if tracker_type == "KCF":
        for factory_name in ("TrackerKCF_create",):
            factory = getattr(legacy, factory_name, None) or getattr(cv2, factory_name, None)
            if factory:
                return factory()

    # Fall back to CSRT if KCF is unavailable or TRACKER_TYPE == "CSRT"
    for factory_name in ("TrackerCSRT_create",):
        factory = getattr(legacy, factory_name, None) or getattr(cv2, factory_name, None)
        if factory:
            return factory()

    raise RuntimeError(
        "No supported tracker found in cv2. "
        "Make sure opencv-contrib-python is installed (not just opencv-python)."
    )


class FaceTracker:
    def __init__(
        self,
        object_detector: ObjectDetector,
        face_net=None,
        recognizer=None,
        tracker_factory=_create_tracker,
    ) -> None:
        self.object_detector = object_detector
        self.face_net = face_net or self._load_face_net()
        self.recognizer, self.label_map = recognizer or self._load_recognizer()
        self._tracker_factory = tracker_factory

        self._locked = False
        self._tracker = None
        self._lost_frames = 0
        self._first_lost_time: Optional[float] = None
        self._match_streak = 0
        self._mismatch_streak = 0
        self._frame_counter = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _load_face_net():
        if not settings.FACE_DETECTOR_PROTO.exists() or not settings.FACE_DETECTOR_WEIGHTS.exists():
            raise RuntimeError(
                "Face detection model files are missing. "
                "Run: uv run python scripts/download_models.py"
            )
        return cv2.dnn.readNetFromCaffe(
            str(settings.FACE_DETECTOR_PROTO), str(settings.FACE_DETECTOR_WEIGHTS)
        )

    @staticmethod
    def _load_recognizer():
        if not settings.FACE_MODEL_PATH.exists():
            raise RuntimeError(
                "Face recognition model is not trained yet. "
                f"Place the target person's photo in {settings.REFERENCE_IMAGE_DIR} "
                "and run: uv run python -m target_track_robot.face_enroll"
            )
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=settings.LBPH_RADIUS,
            neighbors=settings.LBPH_NEIGHBORS,
            grid_x=settings.LBPH_GRID_X,
            grid_y=settings.LBPH_GRID_Y,
        )
        recognizer.read(str(settings.FACE_MODEL_PATH))

        label_map: dict[int, str] = {}
        if settings.FACE_LABELS_PATH.exists():
            with open(settings.FACE_LABELS_PATH, "r", encoding="utf-8") as f:
                label_map = {int(k): v for k, v in json.load(f).items()}
        return recognizer, label_map

    # ------------------------------------------------------------------
    def _detect_face_in_region(self, region: np.ndarray) -> Optional[BoxT]:
        if region.size == 0:
            return None
        h, w = region.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(region, settings.FACE_DETECTOR_INPUT_SIZE),
            1.0,
            settings.FACE_DETECTOR_INPUT_SIZE,
            (104.0, 177.0, 123.0),
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        best_box, best_conf = None, 0.0
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < settings.FACE_DETECTION_CONFIDENCE or confidence < best_conf:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            best_box = (x1, y1, x2 - x1, y2 - y1)
            best_conf = confidence
        return best_box

    def _is_target_face(self, face_gray: np.ndarray) -> bool:
        face_gray = cv2.resize(face_gray, settings.FACE_TRAIN_IMAGE_SIZE)
        face_gray = cv2.equalizeHist(face_gray)
        label, confidence = self.recognizer.predict(face_gray)
        # LBPH: lower confidence value = higher similarity
        return label == TARGET_LABEL and confidence <= settings.LBPH_CONFIDENCE_THRESHOLD

    def _head_region(self, frame: np.ndarray, person_box: BoxT) -> tuple[np.ndarray, int, int]:
        x, y, w, h = person_box
        head_h = max(1, int(h * settings.HEAD_REGION_HEIGHT_RATIO))
        region = frame[y : y + head_h, x : x + w]
        return region, x, y

    # ------------------------------------------------------------------
    def _try_acquire_lock(
        self,
        frame: np.ndarray,
        precomputed_persons: Optional[List[Detection]] = None,
    ) -> Optional[BoxT]:
        """Try to identify and lock onto the target person.

        Parameters
        ----------
        frame:
            Current video frame (BGR).
        precomputed_persons:
            Person detections already produced by the main loop's single
            MobileNet-SSD call.  If None, person detection is run here
            (fallback for unit-test scenarios that bypass the main loop).
        """
        # Use pre-computed persons if available (avoids duplicate MobileNet-SSD call)
        if precomputed_persons is not None:
            persons = precomputed_persons
        else:
            persons = self.object_detector.detect_persons(frame)

        for person in persons:
            region, ox, oy = self._head_region(frame, person.box)
            face_box = self._detect_face_in_region(region)
            if face_box is None:
                continue
            fx, fy, fw, fh = face_box
            gray = cv2.cvtColor(region[fy : fy + fh, fx : fx + fw], cv2.COLOR_BGR2GRAY)
            if gray.size == 0:
                continue
            if self._is_target_face(gray):
                self._match_streak += 1
                if self._match_streak >= settings.MIN_LOCK_CONFIDENCE_FRAMES:
                    self._match_streak = 0
                    return person.box
                return None  # still confirming — need another matching frame
        self._match_streak = 0
        return None

    def _verify_locked_identity(self, frame: np.ndarray, box: BoxT) -> bool:
        region, _, _ = self._head_region(frame, box)
        face_box = self._detect_face_in_region(region)
        if face_box is None:
            # Person may be looking away — not enough evidence of wrong identity
            return True
        fx, fy, fw, fh = face_box
        gray = cv2.cvtColor(region[fy : fy + fh, fx : fx + fw], cv2.COLOR_BGR2GRAY)
        if gray.size == 0:
            return True
        return self._is_target_face(gray)

    # ------------------------------------------------------------------
    def process(
        self,
        frame: np.ndarray,
        precomputed_persons: Optional[List[Detection]] = None,
    ) -> TrackResult:
        """Process one video frame.

        Parameters
        ----------
        frame:
            Current BGR frame from the camera.
        precomputed_persons:
            Person detections from the main loop's shared MobileNet-SSD call.
            Pass None only in tests or when the caller has not run detection
            this frame (the tracker will skip acquisition silently in that case
            to avoid an unexpected DNN call).
        """
        self._frame_counter += 1

        if not self._locked:
            # Only try to acquire lock when persons have been freshly detected.
            # If precomputed_persons is None it means the main loop skipped
            # detection this frame → nothing to search → return unchanged state.
            if precomputed_persons is None:
                return TrackResult(
                    box=None, locked=False,
                    seconds_since_lost=self._seconds_since_lost()
                )
            box = self._try_acquire_lock(frame, precomputed_persons)
            if box is not None:
                self._start_lock(frame, box)
                return TrackResult(box=box, locked=True, seconds_since_lost=0.0)
            return TrackResult(
                box=None, locked=False,
                seconds_since_lost=self._seconds_since_lost()
            )

        # ── Locked: update tracker ────────────────────────────────────────
        ok, tracked_box = self._update_tracker(frame)
        if not ok:
            return self._handle_lost_frame()

        self._lost_frames = 0
        self._first_lost_time = None

        # Periodic identity re-verification (face DNN only, no MobileNet-SSD)
        if self._frame_counter % settings.FACE_REVERIFY_EVERY_N_FRAMES == 0:
            still_target = self._verify_locked_identity(frame, tracked_box)
            if still_target:
                self._mismatch_streak = 0
            else:
                self._mismatch_streak += 1
                if self._mismatch_streak >= settings.MIN_LOCK_CONFIDENCE_FRAMES:
                    logger.info("Lost identity of target person during tracking — releasing lock")
                    self._release_lock()
                    return TrackResult(box=None, locked=False, seconds_since_lost=0.0)

        return TrackResult(box=tracked_box, locked=True, seconds_since_lost=0.0)

    # ------------------------------------------------------------------
    def _start_lock(self, frame: np.ndarray, box: BoxT) -> None:
        self._tracker = self._tracker_factory()
        self._tracker.init(frame, tuple(int(v) for v in box))
        self._locked = True
        self._lost_frames = 0
        self._first_lost_time = None
        self._mismatch_streak = 0
        logger.info("Locked onto target person at box %s", box)

    def _update_tracker(self, frame: np.ndarray):
        if self._tracker is None:
            return False, None
        ok, box = self._tracker.update(frame)
        if not ok:
            return False, None
        x, y, w, h = box
        return True, (int(x), int(y), int(w), int(h))

    def _handle_lost_frame(self) -> TrackResult:
        if self._first_lost_time is None:
            self._first_lost_time = time.monotonic()
        self._lost_frames += 1
        if self._lost_frames > settings.MAX_LOST_FRAMES:
            logger.info("Target tracking lost for too long — releasing lock and searching again")
            self._release_lock()
            return TrackResult(box=None, locked=False, seconds_since_lost=0.0)
        return TrackResult(
            box=None, locked=True,
            seconds_since_lost=self._seconds_since_lost()
        )

    def _release_lock(self) -> None:
        self._locked = False
        self._tracker = None
        self._lost_frames = 0
        self._first_lost_time = None
        self._mismatch_streak = 0
        self._match_streak = 0

    def _seconds_since_lost(self) -> float:
        if self._first_lost_time is None:
            return 0.0
        return time.monotonic() - self._first_lost_time

    @property
    def is_locked(self) -> bool:
        return self._locked
