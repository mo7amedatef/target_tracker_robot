import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pytest


@pytest.fixture
def frame_size():
    return (640, 480)  # width, height


@pytest.fixture
def blank_frame(frame_size):
    w, h = frame_size
    return np.zeros((h, w, 3), dtype=np.uint8)
