


# EdgeSight — Model Card

## Model Overview

**edgesight-ml-v2.1** (released 2026-10-26) is a custom YOLO26n object detector fine-tuned to identify ground-level obstacles for a robotic vision application.
- It replaces edgesight-ml-v2. It adds cone footage from five new locations and trains with stronger augmentation for a fixed 40 epochs.
- Its settings and per-class confidence cutoffs were chosen by cross-validation on real-world locations.
- It is the first release to meet every Success Criterion on the held-out test clips.

The retraining process is documented in `docs/retraining.md`.

## Architecture

- Model family: YOLO26n
- Framework: PyTorch / Ultralytics
- Task: Object detection
- Training, evaluation, and deployment input size: 640. The ONNX export has a fixed 1×3×640×640 input (`models/onnx_manifest.json`).
- Classes: rock, cone, debris
- Checkpoints: `models/selected.pt`, `models/selected.onnx`
- Per-class confidence cutoffs: rock 0.50, cone 0.23, debris 0.23 (`models/class_confidence.json`). They apply only to these exact model files; v1 and v2 used one cutoff of 0.40.

## Class Definitions

- **rock:** rocks and stones of any shape
- **cone:** traffic cones and tall orange delineator posts (posts added for v2)
- **debris:** sticks, twigs, distinct leaf piles, and single leaves lying alone on a clear surface (single leaves added for v2.1); not leaves within leaf-covered terrain

Full labeling rules are in `docs/retraining.md`.

## Training Data

Dataset: `datasets/rover_obstacles_v2_1b` (plan: `datasets/rover_obstacles_v2_1b_plan.json`).

| Split | Images | Contents |
|---|---:|---|
| Train | 262 | 51 v1 stock photos; 167 labeled real-world frames (`cones_2`, `debris_2`, `mixed_2` up to frame 560, and `more_cones` without the `track_fence` scene); 44 background frames |
| Validation | 21 | v1's stock photos |
| Held-out test | 3 | v1's stock photos (superseded by the labeled test frames below) |

- Training boxes: 262 debris, 192 cone, 169 rock.
- 29 boxes on single leaves inside leaf-covered ground were removed, because that ground counts as terrain.
- Real-world frames were extracted, pre-labeled with an earlier model, and corrected in CVAT.
- Annotation format: YOLO detection.

## Selected Experiment

- **Experiment:** `pilot_v5_final`, the final checkpoint (`last.pt`): image size 640, seed 42, 40 fixed epochs with no early stopping.
- **Settings:** `long_aug` (mosaic 1.0, scale 0.7, hsv_v 0.5, translate 0.15).
- **Why these settings:** two-fold cross-validation by location, with two seeds, favored them (`runs/cv/summary.md`). Stock-photo validation had chosen poor checkpoints, so the epoch count was fixed in advance instead.
- **Candidate declared before scoring:** seed 42. A second seed (7) was trained only to measure variance.

| Validation (21 stock photos, 640) | v2 | v2.1 |
|---|---:|---:|
| Precision | 0.774 | 0.768 |
| Recall | 0.614 | 0.596 |
| mAP50 | 0.687 | 0.709 |
| mAP50-95 | 0.441 | 0.487 |

Sources: `runs/evaluation/v2_on_v21bval/summary.json` and `v21_stockval_640/summary.json`.

## Real-World Test Clips

These are the main evaluation: four 1080p clips that were never trained on, scored with `scripts/analyze_video.py` against Success Criteria set before training.

| Measure | Target | v1 | v2 | v2.1 | Met |
|---|---|---:|---:|---:|---|
| False CAUTION, obstacle-free clip | ≤ 5% | 46.4% | 1.4% | 0.0% | Yes |
| Debris detected | ≥ 70% | 47.9% | 76.6% | 92.4% | Yes |
| Rocks detected | ≥ 85% | 84.5% | 92.2% | 88.3% | Yes |
| Stock validation mAP50-95 (640) | ≥ 0.35 | 0.391 | 0.441 | 0.487 | Yes |
| Cones detected (delineator posts) | ≥ 70% | 18.6% | 45.9% | 81.0% | Yes |

Sources: `runs/video_analysis/{cones,rocks,debris,empty}.json` (v1), `b_*.json` (v2), and `d_*.json` (v2.1). The released ONNX file produces outputs identical to the evaluated candidate (maximum absolute difference 0.0 on 21 images).

## Labeled Test Frames

- **`test_clips`:** 130 frames sampled from the four test clips and labeled by hand (127 rock, 35 cone, and 131 debris boxes; 46 backgrounds).
- **`track_fence`:** 17 frames of traffic cones behind a fence, held out of training as a benchmark for distant, partly hidden cones.

Precision and recall are given at each model's cutoffs.

| Measure | v2 | v2.1 |
|---|---:|---:|
| `test_clips` mAP50 | 0.778 | 0.788 |
| `test_clips` mAP50-95 | 0.617 | 0.541 |
| Rock precision / recall | 0.96 / 0.58 | 0.98 / 0.47 |
| Cone precision / recall | 0.85 / 0.33 | 0.98 / 0.97 |
| Debris precision / recall | 0.76 / 0.36 | 0.60 / 0.51 |
| `track_fence` mAP50-95 | 0.271 | 0.550 |
| `track_fence` cone recall | 0.00 | 0.60 |

Sources: `runs/evaluation/{v2,v21}_test_{test_clips,track_fence}/summary.json`.

**The mAP50-95 comparison favors v2.**
- 58% of the test boxes are v2's own pre-label boxes, kept unchanged, and v2 reproduces them almost exactly.
- The release rule required v2.1 to match v2 on this measure. That condition was waived as a recorded deviation; see `docs/retraining.md`, "Final v2.1: results".

## Run-to-Run Variance

The same recipe with seed 7 detected cones in 84.5% of frames and debris in 98.8%. It also detected rocks in only 72.0% and gave false CAUTION in 6.4% of empty-scene frames, missing both targets (`runs/video_analysis/d7_*.json`).
- **Reliable:** the cone and debris gains hold across seeds.
- **Not reliable:** the rock and empty-scene results depend on the seed. This release's margins on those two are one draw from a wide range.

## Reproducibility

- Checkpoint SHA-256: `04d2d6988d7b6f28d90f9a3a278c63c83c8053a3e9e0c482fe1dfd27cdd79be0`
- ONNX SHA-256: `bb7018966e61caec79b69ca05fce146fd76fae05aedadfb00a3b5826c81da239`
- Training configuration: `runs/train/pilot_v5_final/args.yaml` (`scripts/train_model.py --config long_aug --fixed-epochs --epochs 40 --seed 42`)
- Dataset split fingerprint: `fd3234a51aafdf6611e982714c8b69f70946fa46c173dc42ee8696a34ab49ffc` (`docs/split_lock_v2_1b.json`). The validation and test images are unchanged since v1.
- Confidence cutoffs: `runs/cv/confidence_tuning_long_aug.json`, released as `models/class_confidence.json`
- Runtime settings: `models/runtime_config.json`, re-tuned for v2.1 (`scripts/tune_onnx.py --tag v2_1`)
- INT8 candidates for the Pi: `models/quantization_manifest.json`. `u8_no_head` and `s8_no_head` were quantized from this release and meet every test-clip target (`docs/quantization.md`).
- Python 3.14.7, PyTorch 2.14.1+cpu, Ultralytics 8.4.175

## Limitations

- **Small dataset:** few locations per class. Results vary noticeably between training seeds (see Run-to-Run Variance).
- **Rocks:** recall on the labeled frames fell compared with v2 (0.47 vs 0.58). The rock cutoff of 0.50 rests on only 14 cross-validation boxes.
- **Debris false positives:** precision on the labeled frames fell (0.60 vs 0.76). The cone and rock clips show more debris detections than with v2; some may be single leaves, which now count as debris.
- **Validation still uses stock photos;** checkpoints were fixed by epoch count rather than selected.
- **Rock test caveat:** the early `mixed_2` frames used for training may come from near the rock test location.
- **Zone warnings are image-space,** not verified distance estimates.

## Intended Use

Prototype object detection and tracking for research and engineering demonstrations. The model is not validated for safety-critical autonomous navigation.

## Next Steps

- Raspberry Pi 5 deployment and benchmarking (`docs/raspberry_pi.md`)
- More rock and background footage to stabilize rock recall and false alarms across seeds
- A traffic-cone test clip and a rock test clip from a new location
- Test sets labeled without pre-labels from a model under comparison

## Previous Release: edgesight-ml-v2

- **Experiment:** `pilot_v3_real` (image size 640, seed 42; best epoch 32 of 42, early stopping), trained on `datasets/rover_obstacles_v2`.
- **Test clips:** cones 45.9% (target missed), debris 76.6%, rocks 92.2%, false CAUTION 1.4% (`runs/video_analysis/b_*.json`).
- **Validation (21 stock photos, 640):** mAP50 0.687, mAP50-95 0.441.
- **Checkpoint SHA-256:** `b4a4b6dedfe855ce252c8f9f2499996f5ca4b7d808f4fedbcea636740afedcb4`
- **ONNX SHA-256:** `4df05391b598e7094c62dc0e92b440ceb1bace503ed997ae96ad521b9c94e5dc`
- **Release files:** archived in `models/archive/edgesight-ml-v2/`. It used a single cutoff of 0.40.

## Earlier Release: edgesight-ml-v1

- **Experiment:** `pilot_v2_data`, trained at image size 512 on 51 stock photos
- **Validation (21 images, 512):** mAP50 0.6261, mAP50-95 0.3755, precision 0.6144, recall 0.5822
- **Held-out test (3 images, 512):** mAP50 0.8148, mAP50-95 0.7604, precision 0.8487, recall 0.7778
- **Checkpoint SHA-256:** `8ef59097dee4889acc8fd327b48fb64547b7f4ba372740fddee6161099a23b9f`
- **Release files:** archived in `models/archive/edgesight-ml-v1/`, along with its INT8 variants (`docs/quantization.md`)
- **Training data and history:** `docs/training.md` and `docs/experiments.md`
