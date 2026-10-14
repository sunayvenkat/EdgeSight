

# EdgeSight — ONNX Runtime Optimization

## Objective

Determine whether ONNX Runtime improves CPU inference performance relative to PyTorch while preserving the selected model's detection behavior.

## Model Information

- Architecture: YOLO26n
- Classes: Rock, Cone, Debris
- Original checkpoint: models/selected.pt
- Exported checkpoint: models/selected.onnx
- ONNX format: FP32
- Input dimensions: 1 × 3 × 640 × 640
- Fixed input shape: Yes
- Export simplification: Enabled

## Benchmark Environment

- CPU: Intel Core Ultra 9
- RAM: 32 GB
- Operating system:
- Python version: 3.14.7
- PyTorch version:
- ONNX Runtime version: 2.14
- Ultralytics version:

## Benchmark Configuration

- Input images: 21 validation images
- Warm-up predictions: 5
- Timed predictions: 60 per run
- Number of runs: 3 per backend
- Batch size: 1
- Input size: 640
- Confidence threshold: 0.40
- Device: CPU
- PyTorch threads: 4
- ONNX Runtime internal threads: Runtime defaults

## Performance Results

| Metric | PyTorch | ONNX Runtime |
|---|---|---|
| Average run median latency | TBD | TBD |
| Average run p95 latency | TBD | TBD |
| Average throughput (FPS) | TBD | TBD |
| Peak sampled process RSS | TBD | TBD |
| Model file size | TBD | TBD |

## Prediction Consistency

- Images compared:
- Matching detections:
- Unmatched PyTorch detections:
- Unmatched ONNX detections:
- Mean matched bounding-box IoU:

## Interpretation

Describe whether ONNX Runtime improved or reduced inference performance.

Identify any differences in prediction quality or computational resources.

## Limitations

- ONNX Runtime's internal thread configuration may differ from PyTorch.
- Measurements exclude camera capture, tracking, and display.
- Process memory includes runtime overhead, not only model weights.
- Validation images represent a small dataset.
- Performance improvements on the desktop CPU may not transfer directly to Raspberry Pi hardware.

## Conclusion

Summarize whether ONNX Runtime is a promising inference backend for EdgeSight's planned embedded deployment.

## Next Steps

Investigate thread tuning, graph optimizations, and C++ integration while continuing to verify prediction consistency.