

# EdgeSight — Raspberry Pi 5 Deployment

## Status

Preparation only. Nothing in this document has been run on a Raspberry Pi yet. The hardware, camera, and display setup are still undecided (see Open Decisions).

## Target Platform

- Board: Raspberry Pi 5 (4-core Cortex-A76)
- OS: Raspberry Pi OS (64-bit), Debian 13 "Trixie" release
- Python: 3.13 (the OS default on Trixie)

Raspberry Pi OS "Bookworm" (Python 3.11) is not supported by the current pins: `numpy==2.5.3` has no Python 3.11 wheel for 64-bit ARM.

On 2026-10-22, `pip install --dry-run` against `requirements.txt` for Linux aarch64 / CPython 3.13 resolved all 52 pinned packages from prebuilt wheels, with no additional dependencies. The same `requirements.txt` is used on the desktop and the Pi.

## Files Not in Git

`*.onnx` and `*.pt` are gitignored, so a fresh clone is missing the model files. Copy them from the development machine:

| File | Needed for | SHA-256 |
|---|---|---|
| `models/selected.onnx` | Running the app (default ONNX backend) | `1a7754f3746511326330c81604ca8c8c9f556fd9d7afc87d6f21c7603114b552` |
| `models/selected.pt` | `--backend pytorch`, `scripts/tune_onnx.py`, and benchmark scripts | `8ef59097dee4889acc8fd327b48fb64547b7f4ba372740fddee6161099a23b9f` |

The ONNX backend refuses to start if `models/selected.onnx` does not match the hash recorded in `models/runtime_config.json`.

Validation images (`datasets/rover_obstacles/images/val`) are also gitignored. They are only needed for the benchmark and tuning scripts, not for the live app.

## Setup

```bash
sudo apt update
sudo apt install -y git python3-venv libgl1 libglib2.0-0

git clone https://github.com/sunayvenkat/EdgeSight.git
cd EdgeSight

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`libgl1` and `libglib2.0-0` are required by `opencv-python` and may be missing on Raspberry Pi OS Lite.

From the development machine, copy the model files (replace the user and host name):

```bash
scp models/selected.onnx models/selected.pt pi@edgesight.local:~/EdgeSight/models/
```

Then verify the install on the Pi:

```bash
sha256sum models/selected.onnx models/selected.pt
python -m pytest -q
```

## Running

```bash
# With a monitor attached
python main.py

# Over SSH or without a display: no video window, summary at the end
python main.py --headless --duration 60
```

`main.py` switches to headless mode automatically on Linux when no X11 or Wayland display is available. Other options:

- `--threads N`: ONNX Runtime intra-op threads (default 4, from `edgesight/config.py`)
- `--backend pytorch`: use the original Ultralytics PyTorch path
- `--camera N`: OpenCV camera index

Each run prints processed FPS, median and p95 inference-plus-tracking time, and captured and dropped camera frames.

## Desktop Reference

Measured on the development machine (Intel Core Ultra 9 285H, Windows 11) with a 640 × 480 webcam at 30 FPS. All settings were limited by the camera's 30 FPS, so processing time and CPU are the meaningful comparisons.

| Setting | Processed FPS | Median (ms) | P95 (ms) | Process CPU |
|---|---:|---:|---:|---:|
| ONNX, 4 threads (default) | 30.0 | 20.4 | 21.9 | 358% |
| ONNX, all cores (`runtime_config.json` preset) | 29.8 | 16.1 | 26.7 | 1,283% |
| PyTorch | 30.0 | 21.6 | 25.5 | 748% |

A 20-second `python main.py --headless --duration 20` run with the default settings processed 29.4 FPS with a median of 21.0 ms, p95 of 33.8 ms, and 4 of 592 frames dropped.

## Pi Measurement Plan

The mission target is at least 3 FPS on the Raspberry Pi.

1. Confirm the app runs: `python main.py --headless --duration 60`.
2. Compare `--threads 3` and `--threads 4`. The camera capture thread competes with ONNX Runtime for the Pi's four cores.
3. Compare against `--backend pytorch` under the same conditions.
4. Re-run `scripts/tune_onnx.py` presets on the Pi; the current `runtime_config.json` was tuned on the desktop CPU. This needs `models/selected.pt` and the validation images.
5. Record temperature and throttling during sustained runs with `vcgencmd measure_temp` and `vcgencmd get_throttled`. The Pi 5 throttles under sustained load without cooling, so use the Active Cooler or equivalent for benchmarks.

## Open Decisions

- **Camera.** A USB webcam works with the current OpenCV capture code. A Raspberry Pi Camera Module uses libcamera, which `cv2.VideoCapture` cannot read directly; it would need a Picamera2 capture path in `edgesight/async_camera.py`.
- **Display.** Both a video window and headless mode are supported.
- **Native C++ build.** `CMakeLists.txt` currently targets Windows x64 only. Building `cpp/main.cpp` on the Pi needs a Linux branch that links `libonnxruntime.so` from the ONNX Runtime aarch64 release.
