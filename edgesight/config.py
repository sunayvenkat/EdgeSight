

from pathlib import Path



# Camera configuration

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

FPS = 30

# Model name
# Old model; identifies people and common objects
# MODEL_PATH = "yolo26n.pt"

# New model: identifies rocks, cones, and debris
MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models" / "selected.pt")

# Minimum confidence to make a detection
DETECTION_CONFIDENCE = 0.40

# Image size
INFERENCE_IMAGE_SIZE = 640

TRACKER_CONFIG = "bytetrack.yaml"

# Normalized monitoring-zone boundaries
ZONE_LEFT = 0.20
ZONE_TOP = 0.35
ZONE_RIGHT = 0.80
ZONE_BOTTOM = 0.95
