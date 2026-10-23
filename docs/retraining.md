

# EdgeSight — Real-World Retraining (edgesight-ml-v2)

## Objective

Retrain the detector on real footage, so it works on video like `docs/video_evaluation.md` measured, not only on stock photos. The release target is **edgesight-ml-v2**. This is roadmap step 6.7.

## Class Definitions

The rover only has to recognize an obstacle and move around it, so the classes stay broad. The class IDs are unchanged from v1.

| ID | Class | Label | Do not label |
|---:|---|---|---|
| 0 | rock | Rocks and stones of any shape: boulders, landscaping rocks, loose stones | Gravel and pebbles below the minimum size; curbs; concrete blocks |
| 1 | cone | Traffic cones | Delineator posts, bollards, poles, and other orange objects |
| 2 | debris | Ground litter: sticks, twigs, and leaves. Label each stick or twig, and each pile or cluster of leaves, as one box. | Isolated single leaves; drainage grates; paper or plastic trash (not in scope for now) |

## Labeling Rules

- **Minimum size.** Skip objects smaller than about 10 pixels on their short side in a 1080p frame.
- **Occlusion.** Label partly hidden objects if at least about a third is visible and you can tell what it is.
- **Tight boxes.** Draw boxes around the visible part only.
- **Groups.** A leaf pile or a tangle of sticks gets one box. Sticks that are clearly separate get separate boxes.
- **When unsure, leave it unlabeled** rather than guess. A wrong label teaches the model a wrong answer.
- **Frames with nothing to label** stay in the dataset with an empty label file. They are the background examples the v1 data lacked.

## Success Criteria

These were set before training (2026-10-22) and are measured with `scripts/analyze_video.py` on test clips the model never trained on.

| Measure | v1 baseline (`docs/video_evaluation.md`) | v2 target |
|---|---|---|
| False CAUTION on the empty-scene clip | 46.4% of frames | **5% or less** |
| Debris detected on the debris clip | 47.9% of frames | **70% or more** |
| Cones detected on a re-shot traffic-cone clip | No valid baseline (the v1 clip showed posts) | **70% or more** |
| Rocks detected on the rock clip | 84.5% of frames | **85% or more** |
| Stock-photo validation mAP50-95 (640) | 0.3936 (`docs/quantization.md`, FP32) | **0.35 or more** |

The last row guards against the new data making the model worse on the original stock photos.

## Capture Checklist

Record in daylight, landscape, 1080p at 30 FPS. Keep fingers clear of the lens edge.

**Training footage and test footage must come from separate sessions,** ideally in different spots. Frames from the same clip look nearly identical, so splitting one clip between training and testing would overstate accuracy.

**Training and validation footage** (frames are extracted and labeled):
- 3–5 short clips per class (15–30 s), across at least three locations and surfaces: sidewalk, grass, gravel, dirt, asphalt.
- 4–6 empty-scene clips with no obstacles on the same kinds of surfaces. These become background examples and need no labeling.
- Rover height: hold the camera about 15–30 cm off the ground for most shots, and higher for a few.
- Approach objects from far to near, and pass them so they leave the frame.
- Mix cones, rocks, and debris in some clips.
- Name clips by class, place, and number, for example `cone_parkinglot_1.mp4` or `empty_sidewalk_2.mp4`.

**Test footage** (never labeled or trained on):
- Re-shoot the cone test with traffic cones.
- Optionally, add one mixed-obstacle clip.
- The four clips from `docs/video_evaluation.md` stay test-only.

## Workflow

Run every command from the repository root with the project virtual environment active.

### 1. Extract frames

Put the new training and validation clips in `clips/` (gitignored), then extract them into a new raw session:

```bash
python -m scripts.extract_frames clips/cone_parkinglot_1.mp4 clips/empty_sidewalk_1.mp4 ... --session session_004
```

- **Sampling:** every 15th frame by default (2 per second at 30 FPS). Blurry frames (sharpness below half the clip's median) and near-duplicates are skipped. Frames are resized to 1280 px on the long side.
- **Output:** frames go to `datasets/rover_obstacles/raw/session_004/` (gitignored), named `<clip>_f<frame>.jpg`. `frames.json` records each frame's clip and frame number, plus each clip's hash.
- **Fewer frames to label:** use `--every 30`.

### 2. Pre-label

```bash
python -m scripts.prelabel --session session_004
```

The current model suggests boxes at confidence 0.25. It writes two files into the session folder:

- `cvat_images.zip`: create a CVAT task with labels **Rock, Cone, Debris**, in that order, and upload this zip.
- `cvat_prelabels.zip`: in the task, use **Upload annotations**, format **Ultralytics YOLO Detection 1.0**.

Frames from clips named `empty*` are left out (`--exclude` changes the patterns). They become background examples automatically.

### 3. Correct in CVAT

Fix the suggested boxes against the Class Definitions and Labeling Rules above. Expect to delete many large, wrong boxes; that is the v1 failure being corrected. Add any missed objects. When done, export the task as **Ultralytics YOLO Detection 1.0** and save the zip, for example as `datasets/rover_obstacles/raw/session_004/cvat_export.zip`.

### 4. Build the dataset

Write a plan that assigns **whole clips** to train or val, and save it as `datasets/rover_obstacles_v2_plan.json`:

```json
{
  "base": {"dataset": "datasets/rover_obstacles", "splits": {"train": "train", "test": "test"}},
  "sessions": [
    {
      "session": "session_004",
      "labels": "datasets/rover_obstacles/raw/session_004/cvat_export.zip",
      "clips": {
        "cone_parkinglot_1": "train",
        "cone_park_2": "val",
        "empty_sidewalk_1": "train",
        "empty_grass_2": "val"
      }
    }
  ],
  "background_clips": ["empty*"]
}
```

- **Validation clips:** put at least one clip of each class, plus one empty-scene clip, in `val`. It should be a different location from the training clips of the same class where possible.
- **v1's 21 stock validation photos** are not copied. They stay in `datasets/rover_obstacles` as the stock-photo check.

```bash
python -m scripts.build_dataset --plan datasets/rover_obstacles_v2_plan.json --output datasets/rover_obstacles_v2 --lock docs/split_lock_v2.json
```

The build fails if:
- a clip is used by `docs/video_evaluation.md` (matched by hash);
- an object frame has no label;
- CVAT's class order differs from rock, cone, debris;
- images are duplicated across splits.

### 5. Train candidates

Train both candidates at image size 640, with the same seed:

```bash
# A: v1 data plus background frames only, to measure what negatives alone fix (needs a plan with only the empty clips)
python -m scripts.train_model --dataset datasets/rover_obstacles_v2a --imgsz 640 --epochs 50 --name pilot_v3_negatives

# B: the full v2 data
python -m scripts.train_model --dataset datasets/rover_obstacles_v2 --imgsz 640 --epochs 50 --name pilot_v3_real
```

### 6. Evaluate

Evaluate on the real-world validation split, which is what selection uses, and on the stock-photo check, which is guarded at 0.35:

```bash
python -m scripts.evaluate_model --weights runs/train/pilot_v3_real/weights/best.pt --dataset datasets/rover_obstacles_v2 --imgsz 640 --name pilot_v3_real_val
python -m scripts.evaluate_model --weights runs/train/pilot_v3_real/weights/best.pt --imgsz 640 --name pilot_v3_real_stockval
```

To include v1 as a baseline in the same comparison, evaluate `models/selected.pt` on the v2 validation split the same way.

### 7. Test on video

Export each candidate to ONNX next to its checkpoint, then score the test clips against the Success Criteria:

```bash
python -m scripts.export_onnx --weights runs/train/pilot_v3_real/weights/best.pt
python -m scripts.analyze_video clips/empty.mp4 --expect none --model runs/train/pilot_v3_real/weights/best.onnx --name pilot_v3_real_empty
python main.py --video clips/rocks.mp4 --model runs/train/pilot_v3_real/weights/best.onnx --record clips/rocks_v3.mp4
```

### 8. Release

```bash
python -m scripts.select_model --release edgesight-ml-v2 --dataset datasets/rover_obstacles_v2 --lock docs/split_lock_v2.json --archive \
  --candidate v1 models/selected.pt runs/evaluation/v1_on_v2val/summary.json \
  --candidate pilot_v3_real runs/train/pilot_v3_real/weights/best.pt runs/evaluation/pilot_v3_real_val/summary.json
python -m scripts.export_onnx
python -m scripts.tune_onnx --preset default --run 1 --tag v2    # repeat for each preset and runs 1-3
python -m scripts.summarize_tuning --tag v2
```

- **Archiving:** `--archive` moves the v1 release files to `models/archive/edgesight-ml-v1/` before installing the winner.
- **v1 as a candidate:** it can be listed as `models/selected.pt`. If it wins, it is installed from the archive.
- **Until tuning finishes,** the app has no `runtime_config.json` and will not start without `--model`.
