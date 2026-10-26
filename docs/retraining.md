

# EdgeSight — Real-World Retraining (edgesight-ml-v2)

## Objective

Retrain the detector on real footage, so it works on video like `docs/video_evaluation.md` measured, not only on stock photos. The release target is **edgesight-ml-v2**. This is roadmap step 6.7.

## Class Definitions

The rover only has to recognize an obstacle and move around it, so the classes stay broad. The class IDs are unchanged from v1.

| ID | Class | Label | Do not label |
|---:|---|---|---|
| 0 | rock | Rocks and stones of any shape: boulders, landscaping rocks, loose stones | Gravel and pebbles below the minimum size; curbs; concrete blocks |
| 1 | cone | Traffic cones, and tall orange delineator posts (with or without a base) | Bollards, poles, and other orange objects that are not cones or delineator posts |
| 2 | debris | Ground litter: sticks, twigs, and leaves. Label each stick or twig, and each distinct pile or cluster of leaves, as one box. | Isolated single leaves; ground fully covered in leaves (that is terrain); drainage grates; paper or plastic trash (not in scope for now) |

Delineator posts were added to `cone` on 2026-10-24. They are bright orange, mark hazards the same way, and the rover must avoid them the same way; only the shape differs. The v1 training data contains traffic cones only.

## Labeling Rules

- **Minimum size.** Skip objects smaller than about 10 pixels on their short side in a 1080p frame.
- **Occlusion.** Label partly hidden objects if at least about a third is visible and you can tell what it is.
- **Tight boxes.** Draw boxes around the visible part only.
- **Groups.** A leaf pile or a tangle of sticks gets one box. Sticks that are clearly separate get separate boxes.
- **Leaf-covered ground is terrain.** When leaves carpet the whole ground, do not box the carpet. Label the sticks and rocks on top of it, and leaf piles only where they form a distinct mound with clear edges. Boxing large stretches of ground would teach the model the same oversized boxes that caused v1's false alarms.
- **When unsure, leave it unlabeled** rather than guess. A wrong label teaches the model a wrong answer.
- **Frames with nothing to label** stay in the dataset with an empty label file. They are the background examples the v1 data lacked.

## Success Criteria

These were set before training (2026-10-22) and are measured with `scripts/analyze_video.py` on test clips the model never trained on.

| Measure | v1 baseline (`docs/video_evaluation.md`) | v2 target |
|---|---|---|
| False CAUTION on the empty-scene clip | 46.4% of frames | **5% or less** |
| Debris detected on the debris clip | 47.9% of frames | **70% or more** |
| Cones detected on the cone clip (delineator posts; valid since the 2026-10-24 definition change) | 18.6% of frames | **70% or more** |
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
- The four clips from `docs/video_evaluation.md` stay test-only. Under the updated cone definition, `cones.mp4` is a valid cone test.
- Add a traffic-cone test clip, so both cone shapes are tested.
- Add a rock test clip from a different location: `rocks_2.mp4` was filmed in the same rock garden as `rocks.mp4`.
- Optionally, add one mixed-obstacle clip.

## Collected Data

### Session 004 (recorded and extracted 2026-10-24)

One clip per category was available, so all usable clips went to training. The four clips from `docs/video_evaluation.md` remain the test set.

| Clip | Use | Frames | Notes |
|---|---|---:|---|
| `cones_2` | Train | 47 | Delineator posts on bases in a parking lot, daylight |
| `debris_2` | Train | 46 | Sticks on grass and dirt |
| `mixed_2` | Train, frames 0–559 only | 38 | Leaf-covered ground with rocks and sticks. From about frame 570, the camera shows the rock garden from the `rocks.mp4` test clip, so those frames are excluded (`--end mixed_2:560`). The early close-ups may come from the same area, a small caveat on the rock test result. |
| `empty_2` | Train (background) | 39 | Parking lot with parked cars; no labels needed |
| `rocks_2` | Held back | — | Filmed in the `rocks.mp4` test rock garden. It can be used for training once a rock test clip from a different location exists. |

Extraction used the defaults (every 15th frame; blurry and near-duplicate frames skipped). Pre-labeling suggested 109 cone, 104 debris, and 97 rock boxes across the 131 non-background frames (`datasets/rover_obstacles/raw/session_004/`).

**Labels.** The corrected CVAT export (`cvat_export.zip`) changed every pre-labeled frame:

| | Pre-labels (v1 model) | Corrected |
|---|---:|---:|
| Debris boxes | 104 | 237 |
| Cone boxes | 109 | 106 |
| Rock boxes | 97 (mostly pale leaves) | 14 |
| Median debris box, % of frame | 8.9 | 1.6 |

Five frames were reviewed and left with no boxes, which CVAT exports as no file. They are listed as `empty_frames` in the plan: `cones_2` frames 660–705 (parking lot, with only a post sliver at the edge in 660) and `mixed_2` frame 555 (leaf-covered terrain). Seven boxes snapped to the image edge overshot it by 0.0000005 because of CVAT's 6-decimal rounding, and the builder clipped them.

**Dataset.** `datasets/rover_obstacles_v2` (split lock: `docs/split_lock_v2.json`):
- **train:** 221 images, 44 of them background (20%); 291 debris, 135 cone, and 169 rock boxes
- **val:** v1's 21 stock photos
- **test:** v1's 3 test photos

**Validation split.** With no held-out real-world clips yet, validation uses v1's 21 stock photos (`datasets/rover_obstacles_v2_plan.json`). Stock photos cannot measure false alarms, so the test clips decide whether v2 meets the Success Criteria. Candidate B (`pilot_v3_real`) is the planned v2. Candidate A (`pilot_v3_negatives`, background frames only) is an ablation that shows what negatives alone change.

**v1 on the new footage.** These are diagnostics on training footage, not test results (`--role diagnostic`; `runs/video_analysis/v1_*_2.json`):

| Clip | v1 result |
|---|---|
| `cones_2` | Cones in 60.3% of frames |
| `rocks_2` | Rocks in 98.9% of frames |
| `debris_2` | Debris in 41.8% of frames |
| `mixed_2` | 867 rock detections in 513 frames, many on pale leaves; median box 0.6% of the frame |
| `empty_2` | False detections in 13.5% of frames; CAUTION in 46 of 576 frames |

### Session 005 (recorded 2026-10-25, for v2.1)

| Clip | Use | Frames | Notes |
|---|---|---:|---|
| `more_cones` | Train | 58 | 1080p at 60 FPS, extracted every 30th frame (`--every 30`); 23 blurry pan frames skipped. Scenes: delineator posts in a parking lot; traffic cones at roadsides and crosswalks; traffic cones behind a running-track chain-link fence, one tipped over; a close-up traffic cone. Overcast lighting. The locations differ from the `cones.mp4` test clip. |

**v2 on this clip** (diagnostic, `runs/video_analysis/v2_more_cones.json`):
- cones detected in 18.2% of frames overall;
- 10.4% for the parking-lot posts, 18.4% at the road and crosswalk, and 5.1% behind the track fence;
- the close-up cone was detected at confidence 0.80 or higher.

Distant cones are the gap.

Pre-labeling with v2 suggested only 18 cone boxes across the 58 frames, so most cones will be drawn by hand. The v2.1 plan is `datasets/rover_obstacles_v2_1_plan.json`: the v2 data plus this session.

**Labels:** `more_cones_export.zip`, exported as CVAT for images 1.1.
- 77 cone boxes across all 58 frames.
- The median cone box is 0.69% of the frame, and the smallest 0.13%.
- All 8 rock and debris pre-label suggestions were removed as false.

**Dataset:** `datasets/rover_obstacles_v2_1` (split lock: `docs/split_lock_v2_1.json`).
- train: 279 images, 44 of them background; 212 cone, 291 debris, and 169 rock boxes
- validation and test: unchanged from v2

## Results

### Candidate A: background frames only (ablation, 2026-10-24)

The training data was v1's 51 stock photos plus 39 background frames from `empty_2`, at image size 640 with seed 42. Training stopped early at epoch 33, and the best epoch was 23 (`runs/train/pilot_v3_negatives/results.csv`).

| Measure | v1 | Candidate A |
|---|---:|---:|
| False CAUTION, empty clip | 46.4% | **0.0%** (0 false detections) |
| Debris detected, debris clip | 47.9% | **0.0%** |
| Cones detected, cone clip | 18.6% | 4.8% |
| Rocks detected, rock clip | 84.5% | 80.3% |
| Stock-photo validation mAP50-95 (640) | 0.3914 | 0.3847 |
| Stock-photo validation debris mAP50 (640) | 0.2608 | 0.4136 |

Sources: `runs/video_analysis/{cones,rocks,debris,empty}.json` (v1) and `a_*.json` (A); `runs/evaluation/v1_stockval_640/summary.json` and `pilot_v3_negatives_stockval_640/summary.json`.

**Findings:**

- **Background frames alone eliminated the false alarms but also eliminated debris detection.**
- **This is learned, not a threshold effect.** On 20 frames sampled from the debris clip, v1's highest debris score per frame had a median of 0.41. A's had a median of 0.00, and only one frame reached 0.20.
- **Two likely causes.** The background frames are all a cracked parking lot, so A never saw a labeled stick on pavement to contrast with pavement cracks. And backgrounds made up 43% of A's training images, well above the roughly 0–10% commonly recommended for YOLO training.
- **The stock-photo validation did not detect the regression.** Its overall score barely moved, and its debris score improved. Real-world validation clips are needed to select models.

### Candidate B: full v2 data (2026-10-24)

The training data was `datasets/rover_obstacles_v2`: 221 images, 44 of them background, at image size 640 with seed 42. Training stopped early at epoch 42, and the best epoch was 32 (`runs/train/pilot_v3_real/results.csv`, 1,737 s).

| Success criterion | Target | v1 | A | **B** | Met |
|---|---|---:|---:|---:|---|
| False CAUTION, empty clip | ≤ 5% | 46.4% | 0.0% | **1.4%** | Yes |
| Debris detected, debris clip | ≥ 70% | 47.9% | 0.0% | **76.6%** | Yes |
| Rocks detected, rock clip | ≥ 85% | 84.5% | 80.3% | **92.2%** | Yes* |
| Stock-photo validation mAP50-95 (640) | ≥ 0.35 | 0.3914 | 0.3847 | **0.4411** | Yes |
| Cones detected, cone clip | ≥ 70% | 18.6% | 4.8% | **45.9%** | **No** |

\*Small caveat: the early `mixed_2` frames used for training may come from near the rock test location.

Sources: `runs/video_analysis/b_*.json`; `runs/evaluation/pilot_v3_real_v2val_640/summary.json` and `v1_v2val_640/summary.json`.

**Findings:**
- **B fixes the false alarms without losing debris.** Empty-scene false-positive frames fell from 64.2% to 1.6%, and debris detection rose from 47.9% to 76.6%. The labeled real frames supplied the contrast candidate A lacked.
- **Stock-photo performance improved rather than regressed.** mAP50-95 rose from 0.391 to 0.441; cone mAP50 rose from 0.80 to 0.92 and debris mAP50 from 0.26 to 0.32.
- **Cones improved 2.5× but missed the target.** B now detects distant posts behind the fence (confidence 0.42–0.70), which v1 mostly missed. The only cone training footage is one daylight parking-lot clip, while the test clip is posts behind a fence in the evening.
- **New false positives, mostly in the debris class:**
  - *Rock clip:* 114 debris detections, mostly isolated single leaves (confidence 0.47–0.68) and a dark garden stake. Single leaves are on the "do not label" list; check the debris labels for boxed single leaves.
  - *Cone clip:* 76 debris and 34 rock detections, including a drainage grate labeled debris (0.71).
- **CAUTION rises on clips with real obstacles,** for example 73.0% to 84.9% of frames on the rock clip. That is expected when more real obstacles are found, and the leaf false positives add to it.

**Next data, to close the gaps:**
- cone footage across backgrounds, distances, and lighting, covering both traffic cones and posts;
- negatives containing drainage grates and scattered single leaves;
- held-out real-world validation clips.

### Candidate v2.1: v2 data plus `more_cones` (2026-10-25, not released)

The training data was `datasets/rover_obstacles_v2_1`: 279 images, 212 cone boxes, at image size 640 with seed 42. Training stopped early at epoch 33, and the best epoch on stock validation was 23 (`runs/train/pilot_v4_cones/results.csv`, 1,770 s).

| Measure | Target | v2 (released) | v2.1 (best.pt, epoch 23) |
|---|---|---:|---:|
| Cones detected | ≥ 70% | 45.9% | 63.2% |
| Debris detected | ≥ 70% | 76.6% | 49.1% |
| Rocks detected | ≥ 85% | 92.2% | 86.5% |
| False CAUTION, empty clip | ≤ 5% | 1.4% | 0.7% |
| Stock-photo validation mAP50-95 (640) | ≥ 0.35 | 0.441 | 0.458 |

Sources: `runs/video_analysis/c_*.json`; `runs/evaluation/pilot_v4_cones_v2_1val_640/summary.json`.

**Diagnostic: the final-epoch checkpoint.** Epoch 33 (`last.pt`, analyzed with `--role diagnostic`) detected cones in 79.5% of cone-clip frames and debris in 57.4% of debris-clip frames, yet scored lower on stock validation (mAP50-95 0.424 vs 0.452).

**Findings:**
- **More cone data helped cones,** from 45.9% to 63.2%, and to 79.5% at the final epoch.
- **Debris regressed to near the v1 level,** even though v2.1 has every v2 debris label.
- **Stock-photo validation selects checkpoints that are worse on real footage.** Within one run, cone detection swung from 63% to 80% across ten epochs while the stock score moved the other way. Together with the debris swing, this shows large epoch-to-epoch and run-to-run variance at this data size.
- **The final epoch was not adopted.** Choosing it because it scored well on the test clips would turn the test set into a selection set and inflate the results.

**Decision:** v2 stays the release. Before another candidate is selected, v2.1 needs:
- **held-out real-world validation clips** for each class and an empty scene, from locations used in neither training nor testing, so that checkpoints and models are selected on real footage;
- **training that selects checkpoints on that validation set,** possibly with more patience before early stopping;
- **more than one seed,** to measure variance before attributing changes to data.

### Overnight preparation for v2.1 (2026-10-25)

Candidate v2.1 showed that stock-photo validation chooses the wrong checkpoints. These steps replace it with real-world evaluation, using only existing footage.

**More_cones scenes.** Session 005 frames are grouped by location:
- `posts_lot`: frames 0–530
- `crosswalk_curve`: frames 530–600 and 1280–1430, the same crosswalk filmed twice
- `roadside`: frames 800–1130
- `track_fence`: frames 1430–1960, traffic cones behind the running-track fence with one tipped over; 17 labeled frames and 20 cones
- `crosswalk_tree`: frames 1960–2442, including the close-up

`track_fence` is reserved as a possible test scene and excluded from cross-validation until the user decides.

**Debris label audit** (`datasets/rover_obstacles/raw/session_004/debris_review/`):
- 36 small, square-ish debris boxes were flagged.
- **#8–#36 (29 boxes, `mixed_2`) are isolated single leaves,** which break the labeling rules. They are removed through `remove_boxes` in `datasets/rover_obstacles_v2_1_plan.json` and `datasets/cv_plan.json`; the CVAT export itself is unchanged.
- #1–#7 (`debris_2`, a curled piece of bark or stick) are kept.
- Pending user confirmation.

**Test-set labeling package** (`datasets/rover_obstacles/raw/testset_001/`):
- 130 frames sampled every 15th frame from the four test clips.
- 91 need labels; the 39 `empty.mp4` frames become backgrounds.
- Labels used only for scoring never enter training. They replace the "class appears in X% of frames" proxy with real precision, recall, and mAP on real video.

**Cross-validation by location** (`scripts/cross_validate.py`, `datasets/cv_plan.json`):
- **Folds:** fold 1 holds out `cones_2`, `debris_2`, and the first half of `empty_2`. Fold 2 holds out the four non-reserved `more_cones` scenes, `mixed_2`, and the second half of `empty_2`.
- **Settings compared:**
  - `baseline`: 50 epochs, patience 10, mosaic 0.5
  - `long`: 100 epochs, patience 30, mosaic 1.0
  - `long_aug`: `long` plus scale 0.7, hsv_v 0.5, translate 0.15
  - `long_nostock`: `long` without v1's stock photos
- **Scoring:** each run is scored on its fold's held-out real frames (mAP, per-class mAP50, and the background false-positive rate at confidence 0.40).
- **Results:** `runs/cv/summary.md` and `runs/cv/summary.json`.

**Cross-validation results (seed 42, completed overnight).** Mean over two folds, each scored on held-out real-world locations (`runs/cv/summary.md`, `runs/cv/*/fold*/cv_result.json`):

| Setting | mAP50-95 | mAP50 | Recall | Rock mAP50 | Cone mAP50 | Debris mAP50 | Background FP | Best epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.331 | 0.593 | 0.527 | 0.591 | 0.795 | 0.415 | 0.0% | 11, 22 (of 21, 32) |
| long | 0.425 | 0.693 | 0.587 | 0.806 | 0.855 | 0.468 | 2.1% | 40, 64 (of 70, 94) |
| long_aug | 0.462 | 0.743 | 0.663 | 0.734 | 0.885 | 0.578 | 8.3% | 35, 25 (of 65, 55) |
| long_nostock | 0.145 | 0.339 | 0.339 | 0.0 | 0.612 | 0.226 | 0.0% | 18, 25 (of 48, 55) |

- **The baseline stops too early.** More epochs and patience raised mAP50-95 from 0.331 to 0.425, and the best epochs moved from 11–22 to 40–64.
- **The stock photos are needed.** Without them mAP50-95 fell to 0.145, and rock mAP50 to 0, because nearly all rock examples are stock photos.
- **Augmentation gave the highest score, with caveats.** It was best overall and on debris (+0.11 debris mAP50 over `long`), but flagged more background frames (8.3% vs 2.1%). Its gain came from fold 1 (0.489 vs 0.407), while fold 2 was flat (0.435 vs 0.443).
- **Rock is scored only in fold 2,** on 14 boxes, so rock comparisons are noisy.
**Second seed (seed 7) for `long` and `long_aug`.** Over 4 runs each (2 folds × 2 seeds):

| Setting | mAP50-95, mean ± sd | Debris mAP50 | Background FP | Best epochs |
|---|---:|---:|---:|---|
| long | 0.433 ± 0.020 | 0.458 | 4.6% | 40, 50, 61, 64 |
| long_aug | 0.470 ± 0.044 | 0.533 | 9.8% | 22, 25, 35, 82 |

- **Augmentation's gain is consistent but scene-dependent.** Paired by fold and seed, it scored +0.082 and +0.070 on fold 1 and −0.008 and +0.002 on fold 2. It helps on scenes like fold 1's and is neutral elsewhere.
- **It roughly doubles the background false-positive rate,** about 1 more of each fold's ~22 background frames flagged at confidence 0.40.
- **Best epochs vary widely,** especially with augmentation (22–82).

**Recommendation:** train the final v2.1 with the `long_aug` settings and keep the stock photos.
- **Epochs:** after the data split is final, use a fixed count near the cross-validation best epochs (about 40–45, without early stopping on stock photos), because stock validation chooses checkpoints poorly.
- **False positives:** counter the higher background rate with the planned CAUTION persistence and per-class confidence tuning.

**Morning checklist (user):**
1. **Label the test frames.** Upload `testset_001/cvat_images.zip` to a new CVAT task (labels Rock, Cone, Debris) and label all 91 frames.
   - Pre-labels (`cvat_prelabels.zip`) speed this up but come from v2, the model being evaluated. Check every box, so its mistakes don't become "ground truth".
   - Export in either format into `testset_001/`.
2. **Confirm the debris removals** on `debris_review/flagged_debris_page1.jpg`: #8–#36 removed, #1–#7 kept.
3. **Decide the `track_fence` scene.** Move it to the test set, which adds traffic cones and the fence condition to testing, or keep it for training.

**After that (Claude):**
- Build the labeled test set.
- Choose settings from cross-validation, and train the final v2.1 on all training data, with 2–3 seeds if the folds show high variance.
- Tune CAUTION persistence and per-class confidence cutoffs on the cross-validation predictions.
- Score once on the labeled test set and the test clips, then decide on release.

### Release (2026-10-24)

Candidate B was released as **edgesight-ml-v2**, with the missed cone target recorded.

1. **Select and archive:** `scripts/select_model.py --archive` selected `pilot_v3_real` over v1 on stock validation mAP50-95 (0.4411 vs 0.3914) and moved v1's release files to `models/archive/edgesight-ml-v1/`. v1's INT8 files were archived alongside it.
2. **Export:** `scripts/export_onnx.py` exported `models/selected.onnx`. Its outputs are identical to the evaluated candidate (maximum absolute difference 0.0).
3. **Re-tune:** `scripts/tune_onnx.py --tag v2` and `scripts/summarize_tuning.py --tag v2` rewrote `models/runtime_config.json`. The presets show the same pattern as v1: the all-core default had a 15.4 ms median, and 4 threads without spinning had the steadiest p95, 21.8 ms (`runs/benchmarks/tuning_v2_*`).
4. **Verify:** the app runs v2 without `--model`.

Details and held-out test results are in `docs/model_card.md`.

**v2.1 plan:**
- cone footage across backgrounds, distances, and lighting, for traffic cones and posts;
- background footage with drainage grates and scattered single leaves;
- held-out real-world validation clips;
- a traffic-cone test clip and a rock test clip from a new location;
- a check of the debris labels for boxed single leaves.

**Implication for candidate B (written after candidate A):** the labeled real frames (sticks on grass and dirt, leaf litter, delineator posts) supply the contrast A lacked, and they cut backgrounds to about 18% of training images. If B still under-detects debris, the next variant should use fewer background frames.

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

Fix the suggested boxes against the Class Definitions and Labeling Rules above. Expect to delete many large, wrong boxes; that is the v1 failure being corrected. Add any missed objects. When done, export the task as **Ultralytics YOLO Detection 1.0** or **CVAT for images 1.1** and save the zip in the session folder, for example as `datasets/rover_obstacles/raw/session_004/cvat_export.zip`. The builder reads either format; for the XML format, it matches classes by label name and gives frames with no boxes empty labels.

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
- a clip was analyzed as a test clip (matched by hash; `--role diagnostic` runs don't count);
- an object frame has no label (list frames you deliberately left without boxes under `"empty_frames"` in the session);
- CVAT's class order differs from rock, cone, debris;
- a box extends past the image by more than 0.0001 (smaller overshoots from CVAT's rounding are clipped);
- images are duplicated across splits.

The dataset is built in a temporary folder and moved into place only after every check passes, so a failed build leaves nothing behind.

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
