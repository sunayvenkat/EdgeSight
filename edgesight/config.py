

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

# Inference backend for the live app: "onnx" or "pytorch"
INFERENCE_BACKEND = "onnx"

# Exported model and its tuned ONNX Runtime settings
ONNX_MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models" / "selected.onnx")
RUNTIME_CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "models" / "runtime_config.json")

# Override the tuned intra-op thread count (None keeps runtime_config.json).
# In the live app, 4 threads gave a steadier p95 (21.9 vs 26.7 ms) at about
# a quarter of the CPU of the all-core preset, and matches the Pi 5's 4 cores.
ONNX_INTRA_OP_THREADS = 4

# Minimum confidence to make a detection
DETECTION_CONFIDENCE = 0.40

# IoU threshold for non-maximum suppression (matches Ultralytics' default)
NMS_IOU_THRESHOLD = 0.7

# Image size
INFERENCE_IMAGE_SIZE = 640

TRACKER_CONFIG = "bytetrack.yaml"

# Normalized monitoring-zone boundaries
ZONE_LEFT = 0.20
ZONE_TOP = 0.35
ZONE_RIGHT = 0.80
ZONE_BOTTOM = 0.95
