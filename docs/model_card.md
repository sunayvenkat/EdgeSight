


# EdgeSight — Model Card

## Model Overview

**edgesight-ml-v2** (released 2026-10-24) is a custom YOLO26n object detector fine-tuned to identify ground-level obstacles for a robotic vision application. It replaces edgesight-ml-v1, which was trained on stock photos only, with a model also trained on labeled frames and obstacle-free background frames from real footage. The retraining process is documented in `docs/retraining.md`.

## Architecture

- Model family: YOLO26n
- Framework: PyTorch / Ultralytics
- Task: Object detection
- Training, evaluation, and deployment input size: 640. The ONNX export has a fixed 1×3×640×640 input (`models/onnx_manifest.json`).
- Classes: rock, cone, debris
- Checkpoints: `models/selected.pt`, `models/selected.onnx`

## Class Definitions

- **rock:** rocks and stones of any shape
- **cone:** traffic cones and tall orange delineator posts (posts added for v2)
- **debris:** sticks, twigs, and distinct leaf piles; not isolated single leaves or leaf-covered terrain

Full labeling rules are in `docs/retraining.md`.

## Training Data

Dataset: `datasets/rover_obstacles_v2` (plan: `datasets/rover_obstacles_v2_plan.json`).

| Split | Images | Contents |
|---|---:|---|
| Train | 221 | 51 v1 stock photos; 126 labeled real-world frames (`cones_2`, `debris_2`, and `mixed_2` up to frame 560); 44 background frames (`empty_2` plus 5 reviewed frames with nothing to label) |
| Validation | 21 | v1's stock photos |
| Held-out test | 3 | v1's stock photos |

- Training boxes: 291 debris, 135 cone, 169 rock.
- Real-world frames were extracted, pre-labeled with v1, and corrected in CVAT.
- Annotation format: YOLO detection.

## Selected Experiment

- **Experiment:** `pilot_v3_real` (image size 640, seed 42; best epoch 32 of 42, early stopping)
- **Selection metric:** validation mAP50-95 at 640, compared against v1 (0.3914)

| Validation (21 stock photos, 640) | v1 | v2 |
|---|---:|---:|
| Precision | 0.596 | 0.774 |
| Recall | 0.612 | 0.614 |
| mAP50 | 0.612 | 0.687 |
| mAP50-95 | 0.391 | 0.441 |

Sources: `runs/evaluation/v1_v2val_640/summary.json` and `pilot_v3_real_v2val_640/summary.json`.

## Real-World Test Clips

These are the main evaluation: four 1080p clips that were never trained on, scored with `scripts/analyze_video.py` against Success Criteria set before training.

| Measure | Target | v1 | v2 | Met |
|---|---|---:|---:|---|
| False CAUTION, obstacle-free clip | ≤ 5% | 46.4% | 1.4% | Yes |
| Debris detected | ≥ 70% | 47.9% | 76.6% | Yes |
| Rocks detected | ≥ 85% | 84.5% | 92.2% | Yes |
| Stock validation mAP50-95 (640) | ≥ 0.35 | 0.391 | 0.441 | Yes |
| Cones detected (delineator posts) | ≥ 70% | 18.6% | 45.9% | **No** |

Sources: `runs/video_analysis/{cones,rocks,debris,empty}.json` (v1) and `runs/video_analysis/b_*.json` (v2). The released ONNX file produces outputs identical to the evaluated candidate (maximum absolute difference 0.0 on 21 images).

**The cone target was missed.** The only cone training footage is one daylight parking-lot clip, while the test clip shows posts behind a fence in the evening. More cone footage is the main goal of v2.1.

## Held-Out Test Results

| Test (3 stock photos, 640) | v1 | v2 |
|---|---:|---:|
| Precision | 0.707 | 0.642 |
| Recall | 0.814 | 0.861 |
| mAP50 | 0.851 | 0.795 |
| mAP50-95 | 0.723 | 0.608 |
| Debris mAP50 | 0.564 | 0.394 |

Sources: `runs/evaluation/v1_test_640/summary.json` and `v2_test_640/summary.json`.

v1 scores higher on these three stock photos, driven by debris, which has only three boxes in the set. Three images are too few for reliable conclusions. The real-world test clips, thousands of frames, show the opposite for debris: 47.9% for v1 and 76.6% for v2.

## Reproducibility

- Checkpoint SHA-256: `b4a4b6dedfe855ce252c8f9f2499996f5ca4b7d808f4fedbcea636740afedcb4`
- ONNX SHA-256: `4df05391b598e7094c62dc0e92b440ceb1bace503ed997ae96ad521b9c94e5dc`
- Training configuration: `runs/train/pilot_v3_real/args.yaml`
- Dataset split fingerprint: `fd3234a51aafdf6611e982714c8b69f70946fa46c173dc42ee8696a34ab49ffc` (`docs/split_lock_v2.json`). It is identical to v1's lock, because the validation and test images are unchanged.
- Runtime settings: `models/runtime_config.json`, re-tuned for v2 (`scripts/tune_onnx.py --tag v2`)
- Python 3.14.7, PyTorch 2.14.1+cpu, Ultralytics 8.4.175

## Limitations

- **Small dataset:** one training clip per class, filmed in a few locations, and a three-image held-out test.
- **Validation cannot measure false alarms.** Validation still uses stock photos; held-out real-world validation clips are needed.
- **Cones:** detected in 45.9% of cone test frames, below target.
- **Remaining false positives:** isolated single leaves and a drainage grate labeled debris.
- **Rock test caveat:** the early `mixed_2` frames used for training may come from near the rock test location.
- **Zone warnings are image-space,** not verified distance estimates.

## Intended Use

Prototype object detection and tracking for research and engineering demonstrations. The model is not validated for safety-critical autonomous navigation.

## Next Steps (v2.1)

- Cone footage across backgrounds, distances, and lighting, for both traffic cones and delineator posts
- Background footage with drainage grates and scattered single leaves
- Held-out real-world validation clips, plus traffic-cone and new-location rock test clips
- A check of the debris labels for boxed single leaves
- Raspberry Pi 5 deployment and benchmarking (`docs/raspberry_pi.md`)

## Previous Release: edgesight-ml-v1

- **Experiment:** `pilot_v2_data`, trained at image size 512 on 51 stock photos
- **Validation (21 images, 512):** mAP50 0.6261, mAP50-95 0.3755, precision 0.6144, recall 0.5822
- **Held-out test (3 images, 512):** mAP50 0.8148, mAP50-95 0.7604, precision 0.8487, recall 0.7778
- **Checkpoint SHA-256:** `8ef59097dee4889acc8fd327b48fb64547b7f4ba372740fddee6161099a23b9f`
- **Release files:** archived in `models/archive/edgesight-ml-v1/`, along with its INT8 variants (`docs/quantization.md`)
- **Training data and history:** `docs/training.md` and `docs/experiments.md`
