


# EdgeSight — Model Card

## Model Overview

EdgeSight ML v1 is a custom YOLO26n object
detector fine-tuned to identify ground-level
obstacles for a robotic vision application.

## Architecture

- Model family: YOLO26n
- Framework: PyTorch / Ultralytics
- Task: Object detection
- Input size: 512 during training/evaluation
- Deployment inference size: 640
  (INFERENCE_IMAGE_SIZE in edgesight/config.py;
  the ONNX export has a fixed 1x3x640x640
  input per models/onnx_manifest.json)
- Classes: rock, cone, debris
- Checkpoint: models/selected.pt

## Training Data

- Dataset: EdgeSight Rover Obstacles
- Total images: 75
- Training images: 51
- Validation images: 21
- Held-out test images: 3
- Annotation format: YOLO Detection

## Selected Experiment

- Experiment: pilot_v2_data
- Selection metric: Validation mAP50-95
- Validation mAP50: 0.6261
- Validation mAP50-95: 0.3755
- Validation precision: 0.6144
- Validation recall: 0.5822

## Held-Out Test Results

- Test images: 3
- Test mAP50: 0.8148
- Test mAP50-95: 0.7604
- Test precision: 0.8487
- Test recall: 0.7778

## Reproducibility

- Model checkpoint SHA-256:
  8ef59097dee4889acc8fd327b48fb64547b7f4ba372740fddee6161099a23b9f
- Python version: 3.14.7
- PyTorch version: 2.14.1+cpu
- Ultralytics version: 8.4.175
- Training configuration:
  runs/train/pilot_v2_data/args.yaml
- Dataset split fingerprint:
  fd3234a51aafdf6611e982714c8b69f70946fa46c173dc42ee8696a34ab49ffc
  (SHA-256 of docs/split_lock.json)

## Limitations

The dataset is a small experimental collection.

The three-image held-out test split is not
large enough for statistically reliable
generalization claims.

Model performance may degrade in unseen
lighting, terrain, or camera conditions.

The monitoring zone provides image-space
warnings, not verified distance estimates.

## Intended Use

Prototype object detection and tracking
for research and engineering demonstrations.

The model is not validated for safety-critical
autonomous navigation.

## Next Steps

Completed: CPU inference benchmarks
(docs/benchmarks.md), ONNX export and
runtime tuning (docs/onnx_results.md),
inference-resolution comparison
(docs/inference_optimization.md), and native
C++ inference (cpp/, results in
runs/cpp/native_run1-3).

Remaining:

- Deploy and benchmark on Raspberry Pi 5
  (docs/raspberry_pi.md)
- Run an INT8 quantization experiment and
  compare accuracy and latency against FP32
- Model improvements deferred in
  docs/roadmap.md, including more debris and
  cone examples, retraining at image size 640,
  multi-seed runs, and a larger test split
