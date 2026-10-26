# EdgeSight — INT8 Quantization Experiment

## Objective

Determine whether a statically quantized INT8 ONNX model keeps the selected detector's accuracy while reducing CPU latency, to decide whether INT8 should be carried into Raspberry Pi 5 benchmarking.

## Current: INT8 for edgesight-ml-v2.1 (2026-10-26)

The INT8 models were rebuilt from the v2.1 release (`models/selected.onnx`, SHA-256 `bb701896…d239`). The method is the v1 experiment's (described in the sections after this one), with three changes:
- **Calibration:** the 262 v2.1 training images (`datasets/rover_obstacles_v2_1b/images/train`), which include real-world and background frames, instead of v1's 51 stock photos. Validation and test images were never used, and the script refuses `val` and `test` folders.
- **Variants:** only the four working variants were built. The `full` variants detect nothing with this output layout (see "Full quantization breaks the model" below), and that does not depend on the model's weights.
- **Thresholds:** consistency and test-clip scoring use the release's per-class cutoffs (rock 0.50, cone 0.23, debris 0.23). Each INT8 model's hash is recorded in `models/class_confidence.json` under `derived_sha256`, so the app applies the same cutoffs when it runs one with `--model`.

```
python -m scripts.quantize_onnx --variant {no_decode,no_head} --activation-type {quint8,qint8} --calibration-dir datasets/rover_obstacles_v2_1b/images/train
python -m scripts.evaluate_quantization --stage {accuracy,consistency,latency} --model {fp32,u8_no_head,...} --tag v2_1
python -m scripts.analyze_video clips/<clip>.mp4 --expect <class> --model models/selected_int8_u8_no_head.onnx
```

### Validation accuracy (21 stock photos, 640, batch 1)

| Model | Precision | Recall | mAP50 | mAP50-95 | Δ mAP50-95 vs FP32 | Size |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0.758 | 0.583 | 0.689 | 0.4815 | — | 9.85 MB |
| u8_no_decode | 0.688 | 0.519 | 0.657 | 0.4513 | −0.030 (−6.3%) | 3.12 MB |
| u8_no_head | 0.814 | 0.497 | 0.667 | 0.4662 | −0.015 (−3.2%) | 3.44 MB |
| s8_no_decode | 0.711 | 0.555 | 0.665 | 0.4562 | −0.025 (−5.3%) | 3.13 MB |
| s8_no_head | 0.820 | 0.482 | 0.671 | 0.4673 | −0.014 (−2.9%) | 3.44 MB |

Sources: `runs/quantization/accuracy_v2_1_<model>.json`; sizes from `models/quantization_manifest.json`.

### Agreement with FP32 at the released cutoffs

FP32 made 69 detections on the 21 validation images.

| INT8 model | INT8 detections | Matched (IoU ≥ 0.90) | Matched (IoU ≥ 0.50) | Mean matched IoU |
|---|---:|---:|---:|---:|
| u8_no_decode | 64 | 54 | 60 | 0.959 |
| u8_no_head | 63 | 54 | 60 | 0.975 |
| s8_no_decode | 66 | 53 | 60 | 0.961 |
| s8_no_head | 64 | 54 | 60 | 0.971 |

Sources: `runs/quantization/consistency_v2_1_<model>.json`.

### Latency (direct `session.run`, 4 threads, desktop)

| Model | Median | p95 | FPS | Peak RSS | Median vs FP32 |
|---|---:|---:|---:|---:|---:|
| FP32 | 15.77 ms | 17.33 ms | 62.5 | 494 MB | — |
| u8_no_decode | 10.92 ms | 11.95 ms | 90.4 | 434 MB | −31% |
| u8_no_head | 11.53 ms | 12.28 ms | 85.8 | 437 MB | −27% |
| s8_no_decode | 41.45 ms | 42.97 ms | 24.0 | 501 MB | +163% |
| s8_no_head | 37.69 ms | 39.12 ms | 26.4 | 500 MB | +139% |

Medians, p95, and FPS are means of three interleaved runs; RSS is the largest. Per-run medians varied by at most 0.45 ms. Sources: `runs/quantization/latency_v2_1_<model>_run1-3.json`.

### Pi candidates and test-clip check

The candidates were chosen on the results above, before the test clips were scored:
- **`u8_no_head` (primary):** the best accuracy of the fast variants, 27% faster, with the tightest boxes against FP32.
- **`s8_no_head`:** the best accuracy overall. It is slow on this x86 CPU, but ARM's INT8 kernels may favor signed activations, so it is worth measuring on the Pi.

Both were then scored once on the four test clips, through `--model` with the cutoffs applied automatically:

| Measure | Target | FP32 v2.1 | u8_no_head | s8_no_head |
|---|---|---:|---:|---:|
| Cones detected | ≥ 70% | 81.0% | 80.6% | 80.4% |
| Rocks detected | ≥ 85% | 88.3% | 88.5% | 88.1% |
| Debris detected | ≥ 70% | 92.4% | 85.8% | 86.1% |
| False CAUTION, empty clip | ≤ 5% | 0.0% | 0.0% | 0.0% |

Sources: `runs/video_analysis/d_*.json` (FP32) and `q_<model>_*.json`.

### What this shows

- The v1 findings hold for v2.1:
  - keeping the head in FP32 costs about 3% mAP50-95;
  - `u8` activations are 27–31% faster on this CPU;
  - `s8` activations are 2.4–2.6× slower here.
- Both candidates meet every target on the test clips. Debris detection drops the most, by 6–7 points, but stays well above 70%.
- **Recommendation (advisory; the decision is the user's):** benchmark FP32, `u8_no_head`, and `s8_no_head` on the Pi in 8.2. Keep FP32 as the default unless an INT8 model's measured gain on the Pi is worth its accuracy loss.

## v1 Experiment (2026-10-22)

The rest of this document is the original experiment on edgesight-ml-v1. Its INT8 files are archived in `models/archive/edgesight-ml-v1/`.

## Model Information

- Architecture: YOLO26n
- Classes: Rock, Cone, Debris
- Source model: `models/selected.onnx` (FP32, SHA-256 `1a7754f3…b552`, matches `models/onnx_manifest.json`)
- Input: `images`, 1 × 3 × 640 × 640, FP32 (unchanged in every INT8 model)
- Output: `output0`, 1 × 7 × 8400 (decoded boxes in input pixels and class scores); NMS runs after the model
- Quantized models: `models/selected_int8_<s8|u8>_<variant>.onnx`
- Metadata: every source `metadata_props` entry (`names`, `stride`, `imgsz`, `task`, …) is copied into each INT8 model. ONNX Runtime's tooling adds `onnx.infer` and `onnx.quant.pre_process`.

Six INT8 models were built: two activation types × three variants.

| Variant | What stays FP32 | Nodes excluded |
|---|---|---|
| `full` | Nothing | 0 |
| `no_decode` | Head output decode: the reshape, concat, slice, box-decode arithmetic, class sigmoid, and final concat in `/model.23/` | 19 |
| `no_head` | The whole detection head, `/model.23/` | 79 |

| Activation tag | Activations | Weights |
|---|---|---|
| `u8` | QUInt8 (asymmetric) | QInt8, per channel |
| `s8` | QInt8 | QInt8, per channel |

Source: `models/quantization_manifest.json` (hashes, sizes, settings, excluded node names, calibration images).

## Environment

- CPU: Intel Core Ultra 9 285H (16 logical CPUs), which has x86 INT8 dot-product instructions (AVX-VNNI)
- RAM: 32 GB
- Operating system: Windows 11 (10.0.26200)
- Python: 3.14.7
- ONNX: 1.23.2
- ONNX Runtime: 1.31.0 (CPUExecutionProvider)
- Ultralytics: 8.4.175

## Configuration

Quantization (`scripts/quantize_onnx.py`):

- `onnxruntime.quantization.shape_inference.quant_pre_process` with defaults, then `quantize_static`
- QDQ format, per-channel weights, MinMax calibration
- Calibration: the 51 training-split images in `datasets/rover_obstacles/images/train`. Validation and test images were never used.
- Calibration preprocessing: `edgesight.detector.preprocess(frame, 640)` (in `edgesight/onnx_detector.py` when these models were built), the same letterbox and normalization the app uses
- Output is deterministic: re-running all six builds reproduced byte-identical files.

Accuracy (`scripts/evaluate_quantization.py --stage accuracy`):

- `YOLO(path, task="detect").val(data=runs/configs/rover_training.yaml, split="val", imgsz=640, batch=1, conf=0.001, iou=0.7, device="cpu", plots=False)`, plus `workers=0` (as in `scripts/evaluate_model.py`) and a per-model output folder
- 21 validation images, 69 labelled objects

Deployment-threshold consistency (`--stage consistency`):

- `OnnxDetector` at confidence 0.40 and NMS IoU 0.7 (app settings), FP32 vs INT8, on the 21 validation images
- Sessions built directly with the app's settings, because `create_session()` only accepts the hash in `models/runtime_config.json`
- Same-class greedy matching from `scripts/compare_predictions.py`, IoU ≥ 0.90 (the threshold used in `docs/onnx_results.md`); counts at IoU ≥ 0.50 are also given

Latency (`--stage latency`):

- `benchmark_session` from `scripts/tune_onnx.py`: direct `session.run` timing only (no preprocessing, decode, or NMS)
- Session options: 4 intra-op threads, 1 inter-op thread, sequential, all graph optimizations, spinning on (the `intra4` preset, which matches the app)
- Inputs: the 21 validation images preprocessed with `edgesight.detector.preprocess`
- 10 warm-up and 100 timed calls per run, 3 runs per model, one process per run, run sequentially and interleaved across models

Reproduce:

```
python -m scripts.quantize_onnx --variant {full,no_decode,no_head} --activation-type {quint8,qint8}
python -m scripts.evaluate_quantization --stage accuracy --model {fp32,u8_no_head,...}
python -m scripts.evaluate_quantization --stage consistency --model u8_no_head
python -m scripts.evaluate_quantization --stage latency --model u8_no_head --run 1
```

## Results

### Validation accuracy

| Model | Precision | Recall | mAP50 | mAP50-95 | Δ mAP50-95 vs FP32 |
|---|---|---|---|---|---|
| FP32 | 0.5652 | 0.6492 | 0.5947 | 0.3936 | — |
| u8_full | 0.0000 | 0.0000 | 0.0000 | 0.0000 | −0.3936 |
| u8_no_decode | 0.5954 | 0.5912 | 0.5710 | 0.3768 | −0.0168 (−4.3%) |
| u8_no_head | 0.5722 | 0.6045 | 0.5826 | 0.3788 | −0.0148 (−3.8%) |
| s8_full | 0.0000 | 0.0000 | 0.0000 | 0.0000 | −0.3936 |
| s8_no_decode | 0.5785 | 0.6066 | 0.5836 | 0.3920 | −0.0016 (−0.4%) |
| s8_no_head | 0.5341 | 0.6391 | 0.5814 | 0.3822 | −0.0114 (−2.9%) |

Per class (precision / recall / mAP50 / mAP50-95):

| Model | Rock (28) | Cone (24) | Debris (17) |
|---|---|---|---|
| FP32 | 0.480 / 0.750 / 0.747 / 0.543 | 0.863 / 0.786 / 0.813 / 0.561 | 0.352 / 0.412 / 0.224 / 0.077 |
| u8_full | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| u8_no_decode | 0.542 / 0.714 / 0.735 / 0.525 | 0.894 / 0.706 / 0.768 / 0.527 | 0.350 / 0.353 / 0.210 / 0.078 |
| u8_no_head | 0.524 / 0.750 / 0.764 / 0.530 | 0.867 / 0.750 / 0.776 / 0.530 | 0.326 / 0.314 / 0.207 / 0.076 |
| s8_full | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| s8_no_decode | 0.547 / 0.776 / 0.757 / 0.546 | 0.897 / 0.750 / 0.790 / 0.552 | 0.292 / 0.294 / 0.204 / 0.078 |
| s8_no_head | 0.466 / 0.750 / 0.751 / 0.536 | 0.819 / 0.756 / 0.776 / 0.531 | 0.317 / 0.412 / 0.217 / 0.080 |

The FP32 figures differ from `models/selection.json` (mAP50-95 0.3755) because that release evaluation used different settings (`scripts/evaluate_model.py`: imgsz 512, batch 4). All rows above use identical settings.

Sources: `runs/quantization/accuracy_<model>.json`.

### Prediction consistency at confidence 0.40

FP32 produced 66 detections on the 21 validation images.

| INT8 model | INT8 detections | Matched (IoU ≥ 0.90) | FP32 unmatched | INT8 unmatched | Mean matched IoU | Matched (IoU ≥ 0.50) |
|---|---|---|---|---|---|---|
| u8_full | 0 | 0 | 66 | 0 | — | 0 |
| u8_no_decode | 64 | 52 | 14 | 12 | 0.962 | 59 |
| u8_no_head | 64 | 55 | 11 | 9 | 0.974 | 61 |
| s8_full | 0 | 0 | 66 | 0 | — | 0 |
| s8_no_decode | 64 | 57 | 9 | 7 | 0.956 | 63 |
| s8_no_head | 64 | 56 | 10 | 8 | 0.971 | 62 |

For comparison, the FP32 ONNX export matched PyTorch on 66 of 66 detections (`docs/onnx_results.md`).

Sources: `runs/quantization/consistency_<model>.json`.

### Latency (direct `session.run`, 4 threads)

| Model | Median | p95 | FPS | Process CPU | Peak RSS | File size |
|---|---|---|---|---|---|---|
| FP32 | 15.83 ms | 17.06 ms | 62.5 | 399% | 490 MB | 9.85 MB |
| u8_full | 10.75 ms | 11.52 ms | 92.3 | 394% | 430 MB | 3.06 MB |
| u8_no_decode | 10.74 ms | 11.57 ms | 92.0 | 398% | 431 MB | 3.12 MB |
| u8_no_head | 11.54 ms | 12.24 ms | 85.8 | 396% | 433 MB | 3.44 MB |
| s8_full | 42.17 ms | 43.86 ms | 23.6 | 395% | 498 MB | 3.06 MB |
| s8_no_decode | 42.36 ms | 44.14 ms | 23.5 | 395% | 498 MB | 3.13 MB |
| s8_no_head | 38.80 ms | 41.42 ms | 25.6 | 395% | 497 MB | 3.44 MB |

Median, p95, FPS, and CPU are means of three runs; RSS is the largest of three. Per-run medians:

| Model | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| FP32 | 16.03 ms | 15.74 ms | 15.72 ms |
| u8_full | 10.77 ms | 10.81 ms | 10.67 ms |
| u8_no_decode | 10.73 ms | 10.86 ms | 10.61 ms |
| u8_no_head | 11.46 ms | 11.63 ms | 11.54 ms |
| s8_full | 42.30 ms | 42.01 ms | 42.20 ms |
| s8_no_decode | 42.59 ms | 42.49 ms | 42.01 ms |
| s8_no_head | 38.33 ms | 38.88 ms | 39.18 ms |

The FP32 result agrees with the earlier `intra4` tuning median of 16.75 ms (`models/runtime_config.json`).

Sources: `runs/quantization/latency_<model>_run1-3.json`, `models/quantization_manifest.json`.

## Interpretation

**Full quantization breaks the model.** `output0` concatenates box coordinates (about 0–680 input pixels) with class scores (0–1), and the quantizer gives that tensor one scale: 2.73 per integer step. Every score rounds to zero, so no detection passes any threshold. This happens with both activation types and is a property of this output layout, not of calibration quality. Keeping at least the decode nodes in FP32 is required.

**With the decode kept in FP32, accuracy loss is small but measurable.** The four working variants lose 0.002–0.017 mAP50-95 and 0.011–0.024 mAP50. On a 69-object validation set, a few detections changing is enough to move these numbers, so the variants cannot be confidently ranked against each other on accuracy alone. Debris stays the weakest class in every model.

**At the app's 0.40 threshold, INT8 changes some detections.** The working variants reproduce 52–57 of FP32's 66 detections closely (IoU ≥ 0.90) and 59–63 loosely (IoU ≥ 0.50). The rest appear or disappear at the threshold, or shift. That is a real behaviour change compared with the exact FP32 export, though matched boxes stay tight (mean IoU 0.956–0.974).

**Activation type decides the speed on this CPU.** With QUInt8 activations, ONNX Runtime fused all 102 convolutions into `QLinearConv`. Median latency fell by 27% (`u8_no_head`) to 32% (`u8_no_decode`), and p95 fell by a similar amount. With QInt8 activations, only 12 of 102 convolutions were fused. The rest ran in FP32 between extra quantize and dequantize steps, making those models 2.5–2.7× slower than FP32. This was checked by saving the session-optimized graphs. It is consistent with ONNX Runtime's x86 CPU kernels preferring uint8 activations. ARM kernels may behave differently, so the `s8` result should not be assumed to carry over to the Pi in either direction.

**Memory and size improve.** The `u8` models use about 57–60 MB less peak RSS and are 65–69% smaller on disk than FP32. Process CPU stays near 400% for every model because the thread count is fixed at 4.

**Best desktop trade-off:** `u8_no_head`: 27% lower median latency, 3.8% lower mAP50-95, and the closest box agreement of the fast variants (55/66 at IoU ≥ 0.90, mean IoU 0.974). `u8_no_decode` is about 0.8 ms faster but lost slightly more accuracy and consistency. `s8_no_decode` came closest to FP32 accuracy but is unusable on this CPU.

## Limitations

- This desktop CPU has x86 INT8 acceleration (AVX-VNNI). The Raspberry Pi 5's Cortex-A76 uses different INT8 instructions and different ONNX Runtime kernels, so these speedups, and the U8/S8 difference, may not hold there. No Pi numbers are estimated here; they must be measured.
- The validation set is small (21 images, 69 objects), so mAP differences of about 0.01–0.02 are within run-to-run sensitivity to a handful of detections.
- Calibration used 51 training images with MinMax ranges only. Entropy or percentile calibration and larger calibration sets were not tried.
- Latency times only `session.run`. Preprocessing, decode, NMS, tracking, capture, and display are excluded, so end-to-end frame-time savings will be smaller.
- Quantized arithmetic can differ slightly between CPU architectures, so accuracy and consistency should be re-checked on the Pi rather than assumed.
- The app could not load these models at the time: `create_session()` accepted only the FP32 hash in `models/runtime_config.json`. It can now, with `--model` (see the v2.1 section above).

## Conclusion and Recommendation

INT8 can cut CPU latency substantially for this model, but only if the output-decode nodes stay in FP32 and the activation type suits the CPU's kernels. Full quantization is not viable.

Advisory recommendation (the decision is the user's): carry INT8 into Raspberry Pi 5 benchmarking as a candidate alongside FP32, not as a replacement:

- Primary candidate: `models/selected_int8_u8_no_head.onnx`
- Also benchmark `models/selected_int8_s8_no_decode.onnx`. It had the best accuracy, and its slowness here may be specific to x86.
- Keep FP32 as the app's default until Pi measurements show INT8's latency gain is worth the roughly 4% mAP50-95 drop and the changed detections at the 0.40 threshold.

## Next Steps

- On the Pi 5, run `scripts/evaluate_quantization.py` latency and consistency stages for FP32, `u8_no_head`, `s8_no_head`, and `s8_no_decode` (the `intra4` preset already matches the Pi's 4 cores).
- If INT8 is adopted, add an INT8 path to `models/runtime_config.json` (with its hash) so `create_session()` accepts it, and re-run the live-app benchmark.
- Optionally try entropy or percentile calibration, or keep only the final convolution of each head branch in FP32, to recover accuracy while keeping most of the speedup.
