

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
- Operating system: Windows 11 (10.0.26200)
- Python version: 3.14.7
- PyTorch version: 2.14.1+cpu
- ONNX Runtime version: 1.31.0
- Ultralytics version: 8.4.175

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
| Average run median latency | 40.96 ms | 74.13 ms |
| Average run p95 latency | 48.12 ms | 192.94 ms |
| Average throughput (FPS) | 23.97 | 15.60 |
| Peak sampled process RSS | 453.59 MB | 499.95 MB |
| Model file size | 5.43 MB | 9.85 MB |

Per-run results:

| Run | PyTorch median / p95 / FPS | ONNX Runtime median / p95 / FPS |
|---|---|---|
| 1 | 40.58 ms / 45.14 ms / 24.60 | 61.13 ms / 194.77 ms / 11.40 |
| 2 | 42.09 ms / 52.87 ms / 22.74 | 135.76 ms / 320.15 ms / 6.34 |
| 3 | 40.21 ms / 46.36 ms / 24.56 | 25.51 ms / 63.89 ms / 29.07 |

Sources: `runs/benchmarks/pytorch_cpu_run1-3.json`, `runs/benchmarks/onnx_cpu_run1-3.json`, `models/onnx_manifest.json`.

## Prediction Consistency

- Images compared: 21
- Matching detections: 66 of 66
- Unmatched PyTorch detections: 0
- Unmatched ONNX detections: 0
- Mean matched bounding-box IoU: 1.000

Source: `runs/onnx/prediction_consistency.json`

## Interpretation

With runtime-default threading inside the Ultralytics `predict()` pipeline, ONNX Runtime was slower and much less stable than PyTorch on this CPU. Its average median latency was 81% higher, its average p95 latency four times higher, and its throughput 35% lower.

ONNX Runtime results varied widely between runs (median 25.5–135.8 ms). Run 3 was faster than every PyTorch run, but runs 1 and 2 were not.

The per-stage timings point to CPU contention rather than slower model execution alone. ONNX preprocessing averaged 4.2–78.7 ms per run, compared with about 2.1 ms for PyTorch, even though both use the same Ultralytics preprocessing. ONNX Runtime's default thread pool drove process CPU use to about 1,300%, compared with about 775% for PyTorch with 4 threads, which likely starved the surrounding Python work.

Prediction quality was unchanged: every detection matched with IoU 1.0. ONNX Runtime used about 46 MB more peak RSS, and the exported file is 81% larger than the PyTorch checkpoint.

## Limitations

- ONNX Runtime's internal thread configuration may differ from PyTorch.
- Measurements exclude camera capture, tracking, and display.
- Process memory includes runtime overhead, not only model weights.
- Validation images represent a small dataset.
- Performance improvements on the desktop CPU may not transfer directly to Raspberry Pi hardware.

## Conclusion

ONNX Runtime reproduces the selected model's detections exactly. With default threading, it did not improve end-to-end `predict()` latency on the development CPU.

It remains a promising backend for embedded deployment because the export is portable and can be run without PyTorch. Later direct-session measurements, which time the ONNX session alone and are not directly comparable to the `predict()` figures above, reached an 11.2 ms median with the tuned "default" preset (`models/runtime_config.json`). Native C++ runs reached 13.8–26.6 ms medians with outputs identical to the Python reference (`runs/cpp/native_run1-3`).

Explicit thread configuration should be validated on the Raspberry Pi 5 rather than relying on runtime defaults.

## Next Steps

Investigate thread tuning, graph optimizations, and C++ integration while continuing to verify prediction consistency.