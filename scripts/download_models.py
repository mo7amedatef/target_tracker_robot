"""
download_models.py
Downloads the pretrained model files required by the project and places them
in data/models/. Run this script once after the initial installation (the Pi
must be connected to the internet at the time).

Models:
  1) Face Detector (OpenCV DNN - SSD Res10, Caffe) - for face detection.
  2) MobileNet-SSD (Caffe, VOC0712) - for person and obstacle detection.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

FILES = [
    (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        settings.FACE_DETECTOR_PROTO,
    ),
    (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel",
        settings.FACE_DETECTOR_WEIGHTS,
    ),
    (
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
        settings.OBJECT_DETECTOR_PROTO,
    ),
    (
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel",
        settings.OBJECT_DETECTOR_WEIGHTS,
    ),
]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[SKIP] {dest.name} already exists")
        return
    print(f"[DOWNLOAD] {dest.name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[OK] {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {dest.name}: {exc}")
        print(f"   Download it manually from: {url}")
        print(f"   And place it at: {dest}")


def main() -> None:
    print("Downloading required model files...\n")
    for url, dest in FILES:
        download(url, dest)
    print("\nDone. Verify that all files exist in:", settings.MODELS_DIR)


if __name__ == "__main__":
    main()
