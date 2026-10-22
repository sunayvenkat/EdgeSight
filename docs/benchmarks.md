


# EdgeSight — Inference Benchmarks

## Objective

Establish a reproducible CPU inference baseline
for the selected PyTorch YOLO26n model before
ONNX and C++ optimizations.

## Hardware

- CPU: Intel(R) Core(TM) Ultra 9 285H
- Operating system: Windows 11 (10.0.26200)
- RAM: 32 GB
- Python version: 3.14.7
- PyTorch version: 2.14.1+cpu
- Ultralytics version: 8.4.175

## Model

- Checkpoint: models/selected.pt
- Model SHA-256:
  8ef59097dee4889acc8fd327b48fb64547b7f4ba372740fddee6161099a23b9f
- Backend: PyTorch
- Execution device: CPU
- Image size: 640
- Batch size: 1
- Confidence threshold: 0.40
- PyTorch threads: 4

## Methodology

- Dataset: 21 validation images
- Inputs: Preloaded OpenCV images
- Warm-up predictions: 5
- Timed predictions per run: 60
- Independent benchmark runs: 3
- Webcam capture: Excluded
- Rendering and tracking: Excluded

Timing includes model.predict() and its
associated preprocessing/postprocessing.

## Results

| Run | Median (ms) | P95 (ms) | FPS | RSS (MB) |
|-----|-------------|----------|-----|----------|
| R 1 |       40.58 |   45.135 |  25 |   452.58 |
| R 2 |      42.086 |   52.873 |  23 |   453.36 |
| R 3 |      40.209 |   46.356 |  25 |   453.59 |

## Observations

Measurements were stable between runs.
Median latency ranged from 40.21 to 42.09 ms
(average 40.96 ms), and throughput from
22.7 to 24.6 FPS (average 24.0 FPS).

Run 2 had the widest tail (P95 52.87 ms versus
45.14 and 46.36 ms) and the lowest throughput.
Its cause was not identified.

Model inference dominated each prediction
(36.3–39.4 ms mean per run). Preprocessing
took about 2.1 ms and postprocessing under
0.8 ms.

Peak sampled RSS was about 453 MB in every
run, with no sign of memory growth.

Source: runs/benchmarks/pytorch_cpu_run1-3.json

## Limitations

- CPU measurements depend on host power state,
  system load, and processor characteristics.
- Throughput excludes camera capture and display.
- Peak sampled RSS is not model-only memory.
- Model accuracy is evaluated separately.

## Next Steps

- Export selected model to ONNX
- Benchmark ONNX Runtime under matching settings
- Measure latency and memory differences
- Verify detection output consistency
