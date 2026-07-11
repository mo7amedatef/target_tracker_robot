"""
ObjectDetector: Wraps a MobileNet-SSD model (Caffe, 20 VOC0712 classes) and
uses it for two purposes in a single forward pass (maximising CPU efficiency
on the Pi):

  1) Detect "person" bounding boxes — used as the full-body tracking box.
  2) Detect "obstacle" classes (chair/table/sofa/plant) — used to avoid collisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import settings
from target_track_robot.utils.logger import get_logger

logger = get_logger("object_detector")

# Standard class order for MobileNet-SSD (VOC0712) — must remain in this exact order
VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x, y, w, h)


class ObjectDetectorError(RuntimeError):
    pass


class ObjectDetector:
    def __init__(self) -> None:
        if (
            not settings.OBJECT_DETECTOR_PROTO.exists()
            or not settings.OBJECT_DETECTOR_WEIGHTS.exists()
        ):
            raise ObjectDetectorError(
                "Object detection model files are missing. Run: "
                "uv run python scripts/download_models.py"
            )
        self.net = cv2.dnn.readNetFromCaffe(
            str(settings.OBJECT_DETECTOR_PROTO), str(settings.OBJECT_DETECTOR_WEIGHTS)
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        raw = self.net.forward()

        results: list[Detection] = []
        for i in range(raw.shape[2]):
            confidence = float(raw[0, 0, i, 2])
            class_id = int(raw[0, 0, i, 1])
            if class_id < 0 or class_id >= len(VOC_CLASSES):
                continue
            label = VOC_CLASSES[class_id]
            if label == "background":
                continue

            box = raw[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            results.append(Detection(label=label, confidence=confidence, box=(x1, y1, bw, bh)))
        return results

    def detect_persons(self, frame: np.ndarray, min_confidence: float = 0.5) -> list[Detection]:
        return [
            d for d in self.detect(frame) if d.label == "person" and d.confidence >= min_confidence
        ]

    def detect_obstacles(self, frame: np.ndarray) -> list[Detection]:
        return [
            d
            for d in self.detect(frame)
            if d.label in settings.OBSTACLE_CLASSES
            and d.confidence >= settings.OBSTACLE_DETECTION_CONFIDENCE
        ]
