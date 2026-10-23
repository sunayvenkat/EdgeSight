

"""
Object detection backends.

ObjectDetector runs the PyTorch checkpoint through Ultralytics. OnnxDetector
runs the exported model with a tuned ONNX Runtime session and does its own
letterboxing, output decoding, and non-maximum suppression. Both return
Detection objects.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from edgesight.config import (DETECTION_CONFIDENCE, INFERENCE_IMAGE_SIZE, MODEL_PATH, NMS_IOU_THRESHOLD, ONNX_INTRA_OP_THREADS, ONNX_MODEL_PATH, RUNTIME_CONFIG_PATH)

@dataclass(frozen=True)
class Detection:
    # A single object detected in an image

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]

class ObjectDetector:
    # Runs YOLO inference on video frames

    def __init__(self, model = None, model_path = MODEL_PATH, confidence = DETECTION_CONFIDENCE):
        self.model = (model if model is not None else YOLO(model_path))

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        self.confidence = confidence

    def detect(self, frame):
        # Return structured detections for a frame

        results = self.model.predict(source=frame, conf=self.confidence, imgsz=INFERENCE_IMAGE_SIZE, device="cpu", verbose=False)

        result = results[0]
        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            score = float(box.conf[0].item())

            if score < self.confidence:
                continue

            class_id = int(box.cls[0].item())
            label = result.names[class_id]

            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())

            detections.append(Detection(label=label, confidence=score, bbox=(x1, y1, x2, y2)))

        return detections


# ONNX Runtime backend

PADDING_VALUE = 114
MAX_DETECTIONS = 300

EXECUTION_MODES = {"sequential": ort.ExecutionMode.ORT_SEQUENTIAL, "parallel": ort.ExecutionMode.ORT_PARALLEL}

GRAPH_LEVELS = {"all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC}


def sha256_file(path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def create_session(model_path=ONNX_MODEL_PATH, runtime_config_path=RUNTIME_CONFIG_PATH, intra_op_threads=ONNX_INTRA_OP_THREADS, verify_hash=True) -> ort.InferenceSession:
    """
    Create a CPU session using the tuned settings in runtime_config.json.

    The released model must match the hash the settings were tuned for.
    Candidate models (retraining, INT8 variants) pass verify_hash=False to
    reuse the same thread and graph settings without that check.
    """
    runtime = json.loads(Path(runtime_config_path).read_text(encoding="utf-8"))

    if verify_hash and runtime["onnx_sha256"] != sha256_file(model_path):
        raise ValueError("runtime_config.json was tuned for a different ONNX file. Re-run scripts/tune_onnx.py, or use --model to run a candidate model.")

    settings = runtime["configuration"]
    threads = settings["intra_op_threads"] if intra_op_threads is None else intra_op_threads

    if threads < 0:
        raise ValueError("intra_op_threads cannot be negative")

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = settings["inter_op_threads"]
    options.execution_mode = EXECUTION_MODES[settings["execution_mode"]]
    options.graph_optimization_level = GRAPH_LEVELS[settings["graph_optimization"]]
    options.add_session_config_entry("session.intra_op.allow_spinning", "1" if settings["spinning"] else "0")

    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def letterbox(frame: np.ndarray, image_size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize and pad a frame to a square, matching Ultralytics' LetterBox."""
    height, width = frame.shape[:2]
    gain = min(image_size / height, image_size / width)

    new_width, new_height = round(width * gain), round(height * gain)

    if (new_width, new_height) != (width, height):
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    pad_x = (image_size - new_width) / 2
    pad_y = (image_size - new_height) / 2

    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)

    padded = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PADDING_VALUE,) * 3)

    return padded, gain, (left, top)


def preprocess(frame: np.ndarray, image_size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Convert a BGR frame to a normalized FP32 NCHW RGB tensor."""
    padded, gain, padding = letterbox(frame, image_size)

    tensor = cv2.dnn.blobFromImage(padded, scalefactor=1.0 / 255.0, swapRB=True)

    return tensor, gain, padding


def decode_output(output: np.ndarray, confidence: float, iou_threshold: float, max_detections: int = MAX_DETECTIONS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn raw [1, 4 + classes, anchors] output into boxes, scores, and class IDs.

    Boxes are xyxy in letterboxed input pixels. Suppression is per class.
    """
    predictions = output[0].T

    class_scores = predictions[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]

    keep = scores > confidence

    centers = predictions[keep, :2]
    sizes = predictions[keep, 2:4]
    scores = scores[keep]
    class_ids = class_ids[keep]

    if len(scores) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)

    corners = centers - sizes / 2

    indices = cv2.dnn.NMSBoxesBatched(np.hstack([corners, sizes]).tolist(), scores.tolist(), class_ids.tolist(), confidence, iou_threshold)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)[:max_detections]

    boxes = np.hstack([corners, corners + sizes])[indices]

    return boxes.astype(np.float32), scores[indices].astype(np.float32), class_ids[indices]


def scale_boxes(boxes: np.ndarray, gain: float, padding: tuple[int, int], frame_shape) -> np.ndarray:
    """Map letterboxed xyxy boxes back to original frame pixels."""
    height, width = frame_shape[:2]
    left, top = padding

    scaled = boxes.copy()
    scaled[:, [0, 2]] = (scaled[:, [0, 2]] - left) / gain
    scaled[:, [1, 3]] = (scaled[:, [1, 3]] - top) / gain

    scaled[:, [0, 2]] = scaled[:, [0, 2]].clip(0, width)
    scaled[:, [1, 3]] = scaled[:, [1, 3]].clip(0, height)

    return scaled


class OnnxDetector:
    """Runs the exported EdgeSight model with ONNX Runtime."""

    def __init__(self, session=None, confidence: float = DETECTION_CONFIDENCE, iou_threshold: float = NMS_IOU_THRESHOLD) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("IoU threshold must be between 0.0 and 1.0")

        self.session = session if session is not None else create_session()
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        model_input = self.session.get_inputs()[0]
        shape = model_input.shape

        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3 or shape[2] != shape[3] or not isinstance(shape[2], int):
            raise ValueError(f"Expected a fixed [1, 3, N, N] input, got {shape}")

        self.input_name = model_input.name
        self.image_size = shape[2]

        metadata = self.session.get_modelmeta().custom_metadata_map
        self.names = {int(key): value for key, value in ast.literal_eval(metadata["names"]).items()}

    def infer(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return xyxy boxes in frame pixels, scores, and class IDs."""
        tensor, gain, padding = preprocess(frame, self.image_size)

        output = self.session.run(None, {self.input_name: tensor})[0]

        boxes, scores, class_ids = decode_output(output, self.confidence, self.iou_threshold)

        return scale_boxes(boxes, gain, padding, frame.shape), scores, class_ids

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return structured detections for a frame."""
        boxes, scores, class_ids = self.infer(frame)

        return [
            Detection(label=self.names[int(class_id)], confidence=float(score), bbox=tuple(int(value) for value in box))
            for box, score, class_id in zip(boxes, scores, class_ids)
        ]
