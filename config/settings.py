"""
==============================================================================
 target_track_robot - Central Configuration
==============================================================================
All project constants live here in one place.
To change any robot behaviour (speed, distance, sensitivity, port names, etc.)
edit this file only — no "magic numbers" scattered across other modules.
==============================================================================
"""

from pathlib import Path

# ------------------------------------------------------------------------
# 1) Paths
# ------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_IMAGE_DIR = DATA_DIR / "reference_image"   # put target person's photo(s) here
MODELS_DIR = DATA_DIR / "models"                      # trained models (LBPH + DNN)
LOGS_DIR = PROJECT_ROOT / "logs"

FACE_MODEL_PATH = MODELS_DIR / "face_lbph_model.yml"
FACE_LABELS_PATH = MODELS_DIR / "face_labels.json"

FACE_DETECTOR_PROTO   = MODELS_DIR / "deploy.prototxt"
FACE_DETECTOR_WEIGHTS = MODELS_DIR / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

OBJECT_DETECTOR_PROTO   = MODELS_DIR / "MobileNetSSD_deploy.prototxt"
OBJECT_DETECTOR_WEIGHTS = MODELS_DIR / "MobileNetSSD_deploy.caffemodel"

# ------------------------------------------------------------------------
# 2) Camera
# ------------------------------------------------------------------------
CAMERA_INDEX = 0
# ⚡ 320×240 feeds the DNN (which internally resizes to 300×300) at ¼ the
#    pixel cost of 640×480, with no loss in detection quality.
CAMERA_WIDTH  = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 15           # match the loop target — no point capturing faster
CAMERA_BUFFER_SIZE = 1    # always 1: avoids stale-frame latency

# ------------------------------------------------------------------------
# 3) Detection scheduling
# ------------------------------------------------------------------------
# ── Searching phase (person not yet locked) ─────────────────────────────
# One MobileNet-SSD forward pass per DETECTION_EVERY_N_FRAMES delivers
# BOTH person boxes (for face matching) AND obstacle boxes in a single call.
# This eliminates the duplicate DNN call that existed before.
# At 10 fps → detection every 200 ms → person is recognised within ≤400 ms
# of appearing (2 × 200 ms for MIN_LOCK_CONFIDENCE_FRAMES = 2).
DETECTION_EVERY_N_FRAMES = 2

# ── Locked phase (CSRT/KCF tracker running) ──────────────────────────────
# We no longer need per-frame person detection; only obstacle updates matter.
# Every 6 frames = ~1.7 times/second — plenty of time to react to a chair.
OBSTACLE_DETECT_EVERY_N_FRAMES = 6

# Re-verify the locked person's identity periodically so we don't chase
# the wrong person if the tracker drifts.
FACE_REVERIFY_EVERY_N_FRAMES = 20

# Minimum confidence for person detections from MobileNet-SSD
PERSON_DETECTION_MIN_CONFIDENCE = 0.5

# ------------------------------------------------------------------------
# 4) Face Detection / Recognition
# ------------------------------------------------------------------------
FACE_DETECTOR_INPUT_SIZE = (300, 300)
FACE_DETECTION_CONFIDENCE = 0.6      # minimum confidence to accept a face

# LBPH Face Recognizer
LBPH_CONFIDENCE_THRESHOLD = 70.0    # lower = stricter match required
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
FACE_TRAIN_IMAGE_SIZE = (200, 200)

# Data augmentation (enroll-time only, not at runtime)
AUGMENTATION_ROTATIONS = [-15, -10, -5, 0, 5, 10, 15]
AUGMENTATION_FLIP = True
AUGMENTATION_BRIGHTNESS_FACTORS = [0.8, 1.0, 1.2]

# ------------------------------------------------------------------------
# 5) Person Tracking Lock
# ------------------------------------------------------------------------
# ⚡ KCF runs ~3–4× faster than CSRT on ARM with minimal accuracy trade-off.
#    Switch to "CSRT" if you find KCF drifts too often in your environment.
TRACKER_TYPE = "KCF"
MAX_LOST_FRAMES = 20
MIN_LOCK_CONFIDENCE_FRAMES = 2  # consecutive detection matches before locking
HEAD_REGION_HEIGHT_RATIO = 0.45 # top fraction of person box searched for face

# ------------------------------------------------------------------------
# 6) Bounding Box Geometry
# ------------------------------------------------------------------------
CENTER_DEADZONE_RATIO      = 0.12   # ±12% around centre = "aligned", no L/R turn
TARGET_BOX_HEIGHT_RATIO_MIN = 0.35  # box < this  → person FAR  → Forward
TARGET_BOX_HEIGHT_RATIO_MAX = 0.55  # box > this  → person CLOSE → Backward

# ------------------------------------------------------------------------
# 7) Obstacle Avoidance
# ------------------------------------------------------------------------
OBSTACLE_CLASSES = {"chair", "diningtable", "sofa", "pottedplant"}
OBSTACLE_DETECTION_CONFIDENCE      = 0.5
OBSTACLE_PATH_ZONE_Y_RATIO         = 0.55
OBSTACLE_PATH_ZONE_X_MARGIN_RATIO  = 0.30
OBSTACLE_BOX_HEIGHT_RATIO_TRIGGER  = 0.30
OBSTACLE_AVOID_STEER_FRAMES        = 10

# ------------------------------------------------------------------------
# 8) Serial Protocol
# ------------------------------------------------------------------------
SERIAL_PORT              = "/dev/ttyUSB0"
SERIAL_BAUDRATE          = 9600
SERIAL_TIMEOUT           = 1.0
SERIAL_WRITE_MIN_INTERVAL = 0.08
SERIAL_RECONNECT_INTERVAL = 3.0

CMD_FORWARD = "F"
CMD_BACKWARD = "B"
CMD_RIGHT   = "R"
CMD_LEFT    = "L"
CMD_STOP    = "S"
VALID_COMMANDS = {CMD_FORWARD, CMD_BACKWARD, CMD_RIGHT, CMD_LEFT, CMD_STOP}

# ------------------------------------------------------------------------
# 9) Main Loop
# ------------------------------------------------------------------------
MAIN_LOOP_TARGET_FPS = 10   # 100 ms/frame budget — realistic for two DNNs on Pi 5 CPU
SHOW_DEBUG_WINDOW    = False # True to see live video with bounding boxes
LOG_LEVEL            = "INFO"
PERF_LOG_INTERVAL_SECONDS = 10  # print FPS to console every N seconds (0 = off)

# ------------------------------------------------------------------------
# 10) Search Mode
# ------------------------------------------------------------------------
ENABLE_SEARCH_MODE             = True
SEARCH_MODE_AFTER_LOST_SECONDS = 2.0
SEARCH_ROTATE_COMMAND          = CMD_RIGHT
