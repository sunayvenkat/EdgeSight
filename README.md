# EdgeSight
Real-time edge AI vision system for object detection, tracking, and embedded inference optimization

## Overview

EdgeSight reads frames from a camera or a recorded video, detects rover obstacles (rock, cone, debris) with a fine-tuned YOLO26n model, tracks them across frames with ByteTrack, and raises a CAUTION/CLEAR warning when a tracked object enters a monitoring zone in the lower part of the image. It runs entirely on the CPU and is being prepared for a Raspberry Pi 5.

- **Detection:** YOLO26n fine-tuned on a small custom dataset, run through a tuned ONNX Runtime session by default, or through Ultralytics PyTorch.
- **Tracking:** ByteTrack (`bytetrack.yaml`) assigns stable track IDs.
- **Monitoring zone:** a configurable normalized rectangle (`ZONE_*` in `edgesight/config.py`). Objects whose bottom-center point is inside the zone produce ENTER/EXIT events and a CAUTION status. The zone is an image-space warning, not a distance estimate.
- **Input:** live cameras are read on a background thread that keeps only the newest frame, so a slow frame never builds a backlog. Video files are processed frame by frame, in order, with no frames dropped.

The project goals and success criteria are in [docs/mission.md](docs/mission.md).

## Requirements

- **Desktop:** tested with Python 3.14 on Windows 11, CPU only. All dependencies are pinned in `requirements.txt`.
- **Raspberry Pi 5:** Raspberry Pi OS (64-bit) "Trixie" with Python 3.13. "Bookworm" (Python 3.11) cannot install the pinned `numpy`. See [docs/raspberry_pi.md](docs/raspberry_pi.md) for the Pi setup, which has not yet been run on hardware.
- A webcam that OpenCV can open, or a video file. Neither is needed to run the tests.
- The model files described under [Model Files](#model-files), which are not in git.

## Setup

Windows (PowerShell):

```powershell
git clone https://github.com/sunayvenkat/EdgeSight.git
cd EdgeSight

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux or macOS:

```bash
git clone https://github.com/sunayvenkat/EdgeSight.git
cd EdgeSight

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then copy in the model files below.

## Model Files

`models/*.onnx` and `models/*.pt` are gitignored, so a fresh clone does not include the model. Copy them from the development machine into `models/`:

| File | Needed for |
|---|---|
| `models/selected.onnx` | Running the app with the default ONNX backend |
| `models/selected.pt` | `--backend pytorch`, ONNX re-export and tuning, and the benchmark scripts |

Their expected SHA-256 hashes are committed alongside them:

- `models/onnx_manifest.json`: `onnx_sha256` for `selected.onnx` and `source_sha256` for `selected.pt`
- `models/selection.json`: `checkpoint_sha256` for `selected.pt`, plus the released model's validation metrics

Check a copied file with `sha256sum models/selected.onnx` (Linux) or `Get-FileHash models\selected.onnx` (PowerShell).

The ONNX backend also checks `models/selected.onnx` against the hash in `models/runtime_config.json`, which holds the tuned ONNX Runtime settings, and refuses to start if they do not match. After re-exporting the model, re-run `scripts/tune_onnx.py` so the runtime config matches the new file.

The dataset images under `datasets/` and the saved results under `runs/` are also gitignored. They are only needed for training, evaluation, and the benchmark scripts, not for the app.

## Running

```bash
python main.py [--backend onnx|pytorch] [--model PATH] [--camera N | --video PATH] [--threads N] [--headless] [--duration SECONDS] [--record PATH]
```

| Option | Default | Description |
|---|---|---|
| `--backend onnx\|pytorch` | `onnx` | Inference backend. `onnx` uses `models/selected.onnx` with the tuned ONNX Runtime session; `pytorch` uses `models/selected.pt` through Ultralytics. |
| `--model PATH` | released model | Run a candidate model instead: a `.onnx` file for `onnx` or a `.pt` file for `pytorch`. Candidates reuse the tuned runtime settings without the release hash check. |
| `--camera N` | `0` | OpenCV camera index. Cannot be combined with `--video`. |
| `--video PATH` | none | Process a recorded video file instead of a camera. Cannot be combined with `--camera`. |
| `--threads N` | `4` | ONNX Runtime intra-op threads (`ONNX_INTRA_OP_THREADS` in `edgesight/config.py`). `0` lets ONNX Runtime use all cores. |
| `--headless` | off | Run without a video window and print zone events and a summary only. |
| `--duration SECONDS` | none | Stop after this many seconds. |
| `--record PATH` | none | Save the annotated video (boxes, track IDs, zone, metrics) to a new `.mp4` or `.avi` file. Works with or without a window. Refuses to overwrite an existing file. |

Examples:

```bash
# Default webcam with a video window (press 'q' to quit)
python main.py

# A different camera, headless for 60 seconds
python main.py --camera 1 --headless --duration 60

# A recorded video, shown in a window at the file's frame rate
python main.py --video clips/rover_run.mp4

# A recorded video processed as fast as possible, without a window
python main.py --video clips/rover_run.mp4 --headless

# Compare backends and thread counts
python main.py --backend pytorch --headless --duration 20
python main.py --threads 0 --headless --duration 20

# Record a 60-second annotated demo without a window
python main.py --headless --duration 60 --record demo.mp4
```

**Recordings.** From a camera, the recording plays back in real time: if processing runs slower than the camera, the file's frame rate is corrected to the measured processing rate when the run ends. From a video file, each output frame matches one source frame and keeps the file's frame rate. `.mp4` files use the MPEG-4 Part 2 codec (`mp4v`), which some browsers and sites do not play directly; re-encode to H.264 (for example with `ffmpeg -i demo.mp4 -c:v libx264 demo_h264.mp4`) before sharing online.

**Camera vs video input.** A live camera keeps only its newest frame: if processing falls behind, older frames are dropped so the display stays current, and the summary reports how many were dropped. A video file is read in order with every frame processed, so ByteTrack sees unbroken frame-to-frame motion and no frames are ever dropped. With a window, video playback is paced to never run faster than the file's frame rate (or `FPS` in `edgesight/config.py` if the file does not report one). With `--headless`, frames are processed as fast as possible. The run ends at the end of the file.

On Linux, `main.py` switches to headless mode automatically when no X11 or Wayland display is available, for example over SSH.

The window shows tracked boxes, the monitoring zone, processed FPS, inference time, object count, CAUTION/CLEAR status, and dropped frames. Zone ENTER/EXIT events are printed as they happen. Each run ends with a summary of processed frames and FPS, median and p95 inference-plus-tracking time, and captured and dropped frames.

Camera resolution, inference size, confidence and NMS thresholds, the tracker config, and the monitoring zone are set in `edgesight/config.py`.

## Tests

```bash
python -m pytest -q
```

The tests use synthetic frames, synthetic videos, and fake trackers, so they need no camera. The test that runs the real ONNX model on the validation images is skipped when `models/selected.onnx` or the images are missing.

## Project Layout

```text
main.py                  App entry point: arguments, frame loop, overlay, and run summary
edgesight/                (modules in pipeline order)
  config.py              Camera, model, ONNX Runtime, detection, tracker, and zone settings
  sources.py             Frame sources: live camera (background thread, newest frame only) and video files (every frame, in order)
  detector.py            Detection backends: Ultralytics PyTorch, and ONNX Runtime with its own preprocessing and decoding
  tracker.py             ByteTrack tracking for both backends, and box drawing
  events.py              Monitoring zone and ENTER/EXIT events
  recorder.py            Saving annotated video (--record)
scripts/                 Dataset checks, training, evaluation, model selection, ONNX export and tuning, quantization, benchmarks, profiling
tests/                   pytest suite
cpp/main.cpp             Native C++ ONNX Runtime inference (built with CMakeLists.txt; Windows x64 only for now)
models/                  Model metadata (*.json); model files are copied in, not committed
datasets/rover_obstacles/  Dataset config; images and labels are not committed
docs/                    Project documentation and results
```

## Documentation

| Document | Contents |
|---|---|
| [mission.md](docs/mission.md) | Project goals, MVP, engineering requirements, and success criteria |
| [training.md](docs/training.md) | Training runs pilot_v1 and pilot_v2_data: settings and training curves |
| [evaluation.md](docs/evaluation.md) | Error analysis of the first model (pilot_v1) and the resulting improvement plan |
| [experiments.md](docs/experiments.md) | Data-expansion experiment comparing pilot_v1 and pilot_v2_data on a common validation set |
| [model_card.md](docs/model_card.md) | The released model (edgesight-ml-v1): data, metrics, reproducibility, and limitations |
| [benchmarks.md](docs/benchmarks.md) | PyTorch CPU inference baseline |
| [onnx_results.md](docs/onnx_results.md) | ONNX export, prediction consistency, runtime tuning, and native C++ results |
| [inference_optimization.md](docs/inference_optimization.md) | Inference size 416 vs 512 vs 640: accuracy and camera-pipeline performance |
| [quantization.md](docs/quantization.md) | INT8 quantization variants: accuracy, detection consistency, and latency against FP32 |
| [video_evaluation.md](docs/video_evaluation.md) | Detection, tracking, and false alarms on real recorded video (`scripts/analyze_video.py`) |
| [retraining.md](docs/retraining.md) | Retraining on real footage: class definitions, labeling rules, v2 success criteria, and the step-by-step workflow |
| [raspberry_pi.md](docs/raspberry_pi.md) | Raspberry Pi 5 setup, desktop reference numbers, and the Pi measurement plan |

`docs/split_lock.json` records the locked train/validation/test split.

## Key Results

All figures are from the docs linked above; see them for methodology and limitations. The dataset is small (75 images), so accuracy figures are preliminary.

- **Model accuracy** ([model_card.md](docs/model_card.md)): the released model reaches validation mAP50 0.6261 and mAP50-95 0.3755 on 21 images (precision 0.6144, recall 0.5822). The held-out test split has only 3 images, too few for reliable claims.
- **More data helped** ([experiments.md](docs/experiments.md)): growing the training set from 21 to 51 images raised validation mAP50 from 0.4667 to 0.6261 and recall from 0.4342 to 0.5822. Debris remains the weakest class (mAP50 0.274).
- **ONNX matches PyTorch exactly** ([onnx_results.md](docs/onnx_results.md)): all 66 of 66 detections on the validation images matched, with mean box IoU 1.000.
- **Tuned inference** ([onnx_results.md](docs/onnx_results.md)): the tuned ONNX Runtime session reached an 11.2 ms median in direct-session timing, and native C++ inference reached 13.8–26.6 ms medians with outputs identical to the Python reference.
- **Live app on the desktop** ([raspberry_pi.md](docs/raspberry_pi.md)): with a 640 × 480 webcam at 30 FPS, the default ONNX backend with 4 threads processed 30.0 FPS (camera-limited) at a 20.4 ms median and 21.9 ms p95, using 358% process CPU versus 1,283% for the all-core preset. The mission target is at least 10 FPS on the development computer.
- **Raspberry Pi 5:** not yet measured. The target is at least 3 FPS ([mission.md](docs/mission.md)).
