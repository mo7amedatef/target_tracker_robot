import shutil
from pathlib import Path

import cv2
import pytest

from config import settings
from target_track_robot import face_enroll
from target_track_robot.face_tracker import FaceTracker
from target_track_robot.object_detector import Detection

FIXTURES_DIR = Path(__file__).parent / "fixtures"

MODELS_MISSING = not (
    settings.FACE_DETECTOR_PROTO.exists()
    and settings.FACE_DETECTOR_WEIGHTS.exists()
    and settings.OBJECT_DETECTOR_PROTO.exists()
    and settings.OBJECT_DETECTOR_WEIGHTS.exists()
)

pytestmark = pytest.mark.skipif(
    MODELS_MISSING, reason="Model files are missing — run scripts/download_models.py"
)


class FakeObjectDetector:
    """Always returns a single person box covering the entire image — isolates the lock logic under test."""

    def __init__(self, image_shape):
        h, w = image_shape[:2]
        self._box = (0, 0, w, h)
        self._det = [Detection(label="person", confidence=0.99, box=self._box)]

    def detect_persons(self, frame, min_confidence=0.5):
        return self._det

    def detect_obstacles(self, frame):
        return []

    # Convenience: return detections as a pre-computed persons list
    @property
    def persons(self):
        return self._det


class FakeTracker:
    """Fake tracker that always returns the initial box (simulates perfect tracking)."""

    def __init__(self):
        self._box = None

    def init(self, frame, box):
        self._box = box
        return True

    def update(self, frame):
        return True, self._box


class AlwaysFailTracker:
    def init(self, frame, box):
        return True

    def update(self, frame):
        return False, None


@pytest.fixture
def trained_model_dir(tmp_path, monkeypatch):
    ref_dir = tmp_path / "reference_image"
    ref_dir.mkdir()
    shutil.copy(FIXTURES_DIR / "sample_face.jpg", ref_dir / "sample_face.jpg")

    models_dir = tmp_path / "models"
    monkeypatch.setattr(settings, "MODELS_DIR",        models_dir)
    monkeypatch.setattr(settings, "FACE_MODEL_PATH",   models_dir / "face_lbph_model.yml")
    monkeypatch.setattr(settings, "FACE_LABELS_PATH",  models_dir / "face_labels.json")

    face_enroll.enroll_from_reference_images(reference_dir=ref_dir)
    return models_dir


@pytest.fixture
def sample_image():
    return cv2.imread(str(FIXTURES_DIR / "sample_face.jpg"))


def test_lock_acquired_after_enough_matching_frames(trained_model_dir, sample_image, monkeypatch):
    monkeypatch.setattr(settings, "MIN_LOCK_CONFIDENCE_FRAMES", 2)
    # The test fixture is a headshot covering the entire box, unlike a full-body scenario.
    # Expand the head search region so the test reflects reality.
    monkeypatch.setattr(settings, "HEAD_REGION_HEIGHT_RATIO", 1.0)

    fake_od = FakeObjectDetector(sample_image.shape)
    tracker  = FaceTracker(object_detector=fake_od, tracker_factory=FakeTracker)

    # Pass the pre-computed persons list directly (mirrors what main.py does)
    result1 = tracker.process(sample_image, precomputed_persons=fake_od.persons)
    assert result1.locked is False  # first frame — still awaiting confirmation

    result2 = tracker.process(sample_image, precomputed_persons=fake_od.persons)
    assert result2.locked is True
    assert result2.box is not None
    assert tracker.is_locked is True


def test_no_lock_on_unrelated_blank_frame(trained_model_dir, monkeypatch):
    import numpy as np

    monkeypatch.setattr(settings, "MIN_LOCK_CONFIDENCE_FRAMES", 1)
    blank    = np.zeros((480, 640, 3), dtype="uint8")
    fake_od  = FakeObjectDetector(blank.shape)
    tracker  = FaceTracker(object_detector=fake_od, tracker_factory=FakeTracker)

    # Blank image — face DNN will find no face → lock must not be acquired
    result = tracker.process(blank, precomputed_persons=fake_od.persons)
    assert result.locked is False
    assert result.box is None


def test_no_detection_frame_does_not_lock(trained_model_dir, sample_image, monkeypatch):
    """Passing precomputed_persons=None simulates a non-detection frame — must not lock."""
    monkeypatch.setattr(settings, "MIN_LOCK_CONFIDENCE_FRAMES", 1)
    monkeypatch.setattr(settings, "HEAD_REGION_HEIGHT_RATIO", 1.0)

    fake_od = FakeObjectDetector(sample_image.shape)
    tracker  = FaceTracker(object_detector=fake_od, tracker_factory=FakeTracker)

    # Non-detection frame: persons=None → should stay unlocked
    result = tracker.process(sample_image, precomputed_persons=None)
    assert result.locked is False


def test_lock_released_after_max_lost_frames(trained_model_dir, sample_image, monkeypatch):
    monkeypatch.setattr(settings, "MIN_LOCK_CONFIDENCE_FRAMES", 1)
    monkeypatch.setattr(settings, "MAX_LOST_FRAMES", 3)
    monkeypatch.setattr(settings, "HEAD_REGION_HEIGHT_RATIO", 1.0)

    fake_od = FakeObjectDetector(sample_image.shape)
    tracker  = FaceTracker(
        object_detector=fake_od,
        tracker_factory=AlwaysFailTracker,
    )

    # First frame: not yet locked — lock happens on the next detection frame
    tracker.process(sample_image, precomputed_persons=fake_od.persons)
    result = tracker.process(sample_image, precomputed_persons=fake_od.persons)
    assert result.locked is True  # locked, but tracker fails immediately on update

    # Repeat until we exceed MAX_LOST_FRAMES — at least one unlock must occur
    # (it may immediately re-lock if a new detection frame arrives, which is correct)
    unlock_happened = False
    for i in range(6):
        # Alternate: detection frame / non-detection frame to exercise both paths
        persons = fake_od.persons if i % 2 == 0 else None
        result = tracker.process(sample_image, precomputed_persons=persons)
        if not result.locked:
            unlock_happened = True

    assert unlock_happened is True
