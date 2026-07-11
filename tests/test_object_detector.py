from pathlib import Path

import cv2
import numpy as np
import pytest

from config import settings
from target_track_robot.object_detector import ObjectDetector, ObjectDetectorError

FIXTURES_DIR = Path(__file__).parent / "fixtures"

MODELS_MISSING = not (
    settings.OBJECT_DETECTOR_PROTO.exists() and settings.OBJECT_DETECTOR_WEIGHTS.exists()
)

pytestmark = pytest.mark.skipif(
    MODELS_MISSING,
    reason="Object detection model files are missing — run scripts/download_models.py first",
)


@pytest.fixture
def detector():
    return ObjectDetector()


def test_detector_raises_clear_error_when_models_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "OBJECT_DETECTOR_PROTO", tmp_path / "missing.prototxt")
    monkeypatch.setattr(settings, "OBJECT_DETECTOR_WEIGHTS", tmp_path / "missing.caffemodel")
    with pytest.raises(ObjectDetectorError):
        ObjectDetector()


def test_detect_returns_list_of_detections(detector):
    image = cv2.imread(str(FIXTURES_DIR / "sample_face.jpg"))
    detections = detector.detect(image)
    assert isinstance(detections, list)
    for det in detections:
        assert det.label in detector.detect.__globals__["VOC_CLASSES"]
        assert 0.0 <= det.confidence <= 1.0
        x, y, w, h = det.box
        assert w > 0 and h > 0


def test_detect_on_blank_frame_returns_no_high_confidence_junk(detector):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(blank)
    # A completely blank frame should produce no high-confidence detections
    assert all(d.confidence < 0.9 for d in detections)


def test_detect_persons_filters_by_label_and_confidence(detector):
    image = cv2.imread(str(FIXTURES_DIR / "sample_face.jpg"))
    persons = detector.detect_persons(image, min_confidence=0.3)
    assert all(p.label == "person" for p in persons)
    assert all(p.confidence >= 0.3 for p in persons)


def test_detect_obstacles_only_returns_configured_classes(detector):
    image = cv2.imread(str(FIXTURES_DIR / "sample_face.jpg"))
    obstacles = detector.detect_obstacles(image)
    assert all(o.label in settings.OBSTACLE_CLASSES for o in obstacles)
