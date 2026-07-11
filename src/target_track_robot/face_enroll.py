"""
face_enroll: Trains the face model on the target person using one or more
images found in data/reference_image.

Since there is typically only a single reference photo, Data Augmentation
(rotation + flip + brightness variation) is applied to generate multiple
variants so the LBPH model learns more robustly.

Model used: OpenCV LBPH (Local Binary Patterns Histograms) — extremely
lightweight and well-suited for the Pi's CPU. Ideal for "lock onto one
specific person" (unlike deep embedding models which are too heavy for a
CPU-only device).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import settings
from target_track_robot.utils.logger import get_logger

logger = get_logger("face_enroll")

TARGET_LABEL = 1  # we have only one class: "the target person"
TARGET_NAME = "target_person"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class FaceEnrollmentError(RuntimeError):
    pass


def _load_face_detector() -> cv2.dnn_Net:
    if not settings.FACE_DETECTOR_PROTO.exists() or not settings.FACE_DETECTOR_WEIGHTS.exists():
        raise FaceEnrollmentError(
            "Face detection model files are missing. Run: "
            "uv run python scripts/download_models.py"
        )
    return cv2.dnn.readNetFromCaffe(
        str(settings.FACE_DETECTOR_PROTO), str(settings.FACE_DETECTOR_WEIGHTS)
    )


def detect_largest_face(
    image: np.ndarray, net: cv2.dnn_Net, confidence_threshold: float = None
) -> Optional[tuple[int, int, int, int]]:
    """Return the largest detected face in the image as (x, y, w, h), or None if none found."""
    confidence_threshold = confidence_threshold or settings.FACE_DETECTION_CONFIDENCE
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, settings.FACE_DETECTOR_INPUT_SIZE),
        1.0,
        settings.FACE_DETECTOR_INPUT_SIZE,
        (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

    best_box = None
    best_area = 0
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        area = bw * bh
        if area > best_area:
            best_area = area
            best_box = (x1, y1, bw, bh)
    return best_box


def _augment_face(face_gray: np.ndarray) -> list[np.ndarray]:
    """Generate multiple variants of a face image (rotation/flip/brightness) to strengthen training."""
    variants: list[np.ndarray] = []
    h, w = face_gray.shape[:2]
    center = (w / 2.0, h / 2.0)

    base_images = [face_gray]
    if settings.AUGMENTATION_FLIP:
        base_images.append(cv2.flip(face_gray, 1))

    for base in base_images:
        for angle in settings.AUGMENTATION_ROTATIONS:
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                base, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE
            )
            for factor in settings.AUGMENTATION_BRIGHTNESS_FACTORS:
                bright = np.clip(rotated.astype(np.float32) * factor, 0, 255).astype(
                    np.uint8
                )
                variants.append(bright)
    return variants


def _iter_reference_images(reference_dir: Path):
    for path in sorted(reference_dir.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def enroll_from_reference_images(
    reference_dir: Path = settings.REFERENCE_IMAGE_DIR,
) -> None:
    """Build an LBPH model from all images in the reference directory and save it."""
    if not reference_dir.exists():
        raise FaceEnrollmentError(f"Directory not found: {reference_dir}")

    image_paths = list(_iter_reference_images(reference_dir))
    if not image_paths:
        raise FaceEnrollmentError(
            f"No images found in {reference_dir}. "
            "Place a clear face photo of the target person there."
        )

    face_net = _load_face_detector()

    training_faces: list[np.ndarray] = []
    training_labels: list[int] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Could not read image: %s", image_path)
            continue

        face_box = detect_largest_face(image, face_net)
        if face_box is None:
            logger.warning("No clear face found in image: %s (skipping)", image_path)
            continue

        x, y, w, h = face_box
        face_crop = image[y : y + h, x : x + w]
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, settings.FACE_TRAIN_IMAGE_SIZE)
        gray = cv2.equalizeHist(gray)

        for variant in _augment_face(gray):
            training_faces.append(variant)
            training_labels.append(TARGET_LABEL)

        logger.info("Face extracted from %s", image_path.name)

    if not training_faces:
        raise FaceEnrollmentError(
            "Could not extract any face from the reference images. "
            "Make sure the photo contains a clear, unobstructed face."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=settings.LBPH_RADIUS,
        neighbors=settings.LBPH_NEIGHBORS,
        grid_x=settings.LBPH_GRID_X,
        grid_y=settings.LBPH_GRID_Y,
    )
    recognizer.train(training_faces, np.array(training_labels))

    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(settings.FACE_MODEL_PATH))

    with open(settings.FACE_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({str(TARGET_LABEL): TARGET_NAME}, f, ensure_ascii=False, indent=2)

    logger.info(
        "Model trained successfully on %d images (after augmentation) and saved to %s",
        len(training_faces),
        settings.FACE_MODEL_PATH,
    )


if __name__ == "__main__":
    enroll_from_reference_images()
