import shutil
from pathlib import Path

import cv2
import pytest

from config import settings
from target_track_robot import face_enroll

FIXTURES_DIR = Path(__file__).parent / "fixtures"

MODELS_MISSING = not (
    settings.FACE_DETECTOR_PROTO.exists() and settings.FACE_DETECTOR_WEIGHTS.exists()
)

pytestmark = pytest.mark.skipif(
    MODELS_MISSING,
    reason="Face detection model files are missing — run scripts/download_models.py first",
)


@pytest.fixture
def face_net():
    return face_enroll._load_face_detector()


def test_detect_largest_face_finds_a_face(face_net):
    image = cv2.imread(str(FIXTURES_DIR / "sample_face.jpg"))
    assert image is not None
    box = face_enroll.detect_largest_face(image, face_net)
    assert box is not None
    x, y, w, h = box
    assert w > 0 and h > 0


def test_detect_largest_face_returns_none_on_blank_image(face_net):
    import numpy as np

    blank = np.zeros((300, 300, 3), dtype="uint8")
    box = face_enroll.detect_largest_face(blank, face_net)
    assert box is None


def test_enroll_from_reference_images(tmp_path, monkeypatch):
    ref_dir = tmp_path / "reference_image"
    ref_dir.mkdir()
    shutil.copy(FIXTURES_DIR / "sample_face.jpg", ref_dir / "sample_face.jpg")

    models_dir = tmp_path / "models"
    monkeypatch.setattr(settings, "MODELS_DIR", models_dir)
    monkeypatch.setattr(settings, "FACE_MODEL_PATH", models_dir / "face_lbph_model.yml")
    monkeypatch.setattr(settings, "FACE_LABELS_PATH", models_dir / "face_labels.json")

    face_enroll.enroll_from_reference_images(reference_dir=ref_dir)

    assert settings.FACE_MODEL_PATH.exists()
    assert settings.FACE_LABELS_PATH.exists()

    # The trained model must be able to recognise the same face it was trained on with high confidence
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=settings.LBPH_RADIUS,
        neighbors=settings.LBPH_NEIGHBORS,
        grid_x=settings.LBPH_GRID_X,
        grid_y=settings.LBPH_GRID_Y,
    )
    recognizer.read(str(settings.FACE_MODEL_PATH))

    image = cv2.imread(str(FIXTURES_DIR / "sample_face.jpg"))
    net = face_enroll._load_face_detector()
    box = face_enroll.detect_largest_face(image, net)
    x, y, w, h = box
    gray = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, settings.FACE_TRAIN_IMAGE_SIZE)
    gray = cv2.equalizeHist(gray)

    label, confidence = recognizer.predict(gray)
    assert label == face_enroll.TARGET_LABEL
    assert confidence <= settings.LBPH_CONFIDENCE_THRESHOLD


def test_enroll_raises_on_empty_reference_dir(tmp_path):
    empty_dir = tmp_path / "empty_ref"
    empty_dir.mkdir()
    with pytest.raises(face_enroll.FaceEnrollmentError):
        face_enroll.enroll_from_reference_images(reference_dir=empty_dir)


def test_enroll_raises_on_missing_dir(tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(face_enroll.FaceEnrollmentError):
        face_enroll.enroll_from_reference_images(reference_dir=missing_dir)
