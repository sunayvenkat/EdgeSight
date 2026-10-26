

# EdgeSight — Raspberry Pi 5 Deployment

## Status

Preparation only. Nothing in this document has been run on a Raspberry Pi yet. The hardware has been chosen (see Hardware) but not yet purchased.

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
| `models/selected.onnx` | Running the app (default ONNX backend) | `bb7018966e61caec79b69ca05fce146fd76fae05aedadfb00a3b5826c81da239` |
| `models/selected.pt` | `--backend pytorch`, `scripts/tune_onnx.py`, and benchmark scripts | `04d2d6988d7b6f28d90f9a3a278c63c83c8053a3e9e0c482fe1dfd27cdd79be0` |

These are the edgesight-ml-v2.1 release files (2026-10-26). Earlier releases are kept in `models/archive/`.

The ONNX backend refuses to start if `models/selected.onnx` does not match the hash recorded in `models/runtime_config.json`. Both backends also refuse to start if a model file does not match the hashes in `models/class_confidence.json` (the per-class cutoffs, which are tracked in git).

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

# Record a demo while headless, then copy it to another machine to watch
python main.py --headless --duration 60 --record demo.mp4
```

`main.py` switches to headless mode automatically on Linux when no X11 or Wayland display is available. Other options:

- `--threads N`: ONNX Runtime intra-op threads (default 4, from `edgesight/config.py`)
- `--backend pytorch`: use the original Ultralytics PyTorch path
- `--camera N`: OpenCV camera index
- `--video PATH`: process a recorded video file instead of a camera
- `--record PATH`: save the annotated video to a new `.mp4` or `.avi` file. If the Pi processes slower than the camera, the file's frame rate is corrected to the measured processing rate so playback is real time.

Each run prints processed FPS, median and p95 inference-plus-tracking time, and captured and dropped camera frames.

## Desktop Reference

Measured on the development machine (Intel Core Ultra 9 285H, Windows 11) with a 640 × 480 webcam at 30 FPS, using the v1 model. v2 has the same architecture and input size, so its speed is expected to match. All settings were limited by the camera's 30 FPS, so processing time and CPU are the meaningful comparisons.

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
5. Benchmark INT8 against FP32 using the v2.1 candidates from `docs/quantization.md`: `u8_no_head` (primary) and `s8_no_head`.
   - Copy `models/selected_int8_u8_no_head.onnx` (SHA-256 `5b3c19e4476ed4774e68e32488f1c8a8ca18380bc639f0fea269efba8273b449`) and `models/selected_int8_s8_no_head.onnx` (`e5618c06fb08466509e79103cfd77e0f5c69c794fbcd90739488c7bef90f781e`). Both are gitignored and were built from the v2.1 release.
   - Run them with `--model`, e.g. `python main.py --headless --duration 60 --model models/selected_int8_u8_no_head.onnx`. Their hashes are listed in `models/class_confidence.json`, so they get the same per-class cutoffs as FP32.
   - For session-level latency, use `python -m scripts.evaluate_quantization --stage latency --model u8_no_head --run 1 --tag pi`.
   - On the desktop, the s8 variants were 2.4–2.6× slower than FP32 because of x86 kernel coverage; ARM may differ.
6. Record temperature and throttling during sustained runs with `vcgencmd measure_temp` and `vcgencmd get_throttled`. The Pi 5 throttles under sustained load without cooling, so use the Active Cooler or equivalent for benchmarks.

## Hardware

Chosen 2026-10-22 for a tabletop deployment. The choices also allow the same Pi to be mounted on a small rover later.

| Item | Choice | Reason |
|---|---|---|
| Board | Raspberry Pi 5, 8 GB recommended (4 GB sufficient) | The app's peak RSS on the desktop was about 450–530 MB; 8 GB leaves headroom for builds and future work |
| Cooling | Official Active Cooler | Continuous inference loads all four cores; an uncooled Pi 5 throttles and would distort benchmarks |
| Power | Official 27 W USB-C supply | With lower-rated supplies, the Pi 5 limits total USB current to 600 mA, which can make a USB webcam unreliable |
| Storage | 64 GB or larger A2-class microSD card | Sufficient for this project; NVMe is not needed |
| Camera | Existing USB webcam | Works with the current OpenCV capture code, so no changes are needed |
| OS and access | Raspberry Pi OS Lite (64-bit) "Trixie", headless over SSH | Lighter than the desktop image; `main.py` runs headless, and `--record` captures demo video |
| Accelerator | None | The mission targets CPU-only deployment |

### Rover Stage (planned, after v1)

- **Camera.** A Raspberry Pi Camera Module 3 is planned for the rover stage. It is lighter, mounts more easily, and connects by ribbon cable, so it is not subject to the USB current limit on battery power. It uses libcamera, which `cv2.VideoCapture` cannot read directly, so it needs a Picamera2 capture source in `edgesight/sources.py`. Picamera2 is installed with apt, so the venv must be created with `--system-site-packages`. The Pi 5 also needs its own smaller camera cable for this module.
- **Power.** The battery will be chosen together with the chassis and motor driver.

## Open Items

- **Native C++ build.** `CMakeLists.txt` currently targets Windows x64 only. Building `cpp/main.cpp` on the Pi needs a Linux branch that links `libonnxruntime.so` from the ONNX Runtime aarch64 release.
