

# EdgeSight — Real-Video Evaluation

## Objective

Check how the released model (edgesight-ml-v1) and the tracking pipeline perform on real, temporally ordered video, rather than on the still images used for training and validation.

## Clips

Recorded 2026-10-22 on a handheld phone camera moving slowly through each scene, often at low height. The clips are kept locally in `clips/` (gitignored); their SHA-256 hashes are recorded in each result file.

| Clip | Contents | Frames | Resolution | FPS |
|---|---|---:|---|---:|
| `cones.mp4` | Orange delineator posts behind a chain-link fence | 484 | 1920 × 1080 | 30 |
| `rocks.mp4` | Landscaping boulders and smaller rocks in leaf litter | 503 | 1920 × 1080 | 30 |
| `debris.mp4` | Fallen sticks and leaves on a concrete sidewalk | 591 | 1920 × 1080 | 30 |
| `empty.mp4` | A sidewalk with grass on both sides and no obstacles | 575 | 1920 × 1080 | 30 |

## Method

- Command: `python -m scripts.analyze_video clips/<clip>.mp4 --expect <class|none>`
- Pipeline: the app's ONNX backend (`models/selected.onnx`, 4 threads), confidence 0.40, NMS IoU 0.7, ByteTrack (`bytetrack.yaml`), and the default monitoring zone. Every frame is processed in order.
- "Expected class detected" counts frames where the clip's class appears at least once. The object is not necessarily visible in every frame, because the camera moves.
- A "fragment" is a track observed in fewer than 5 frames.

Sources: `runs/video_analysis/cones.json`, `rocks.json`, `debris.json`, `empty.json`.

## Results

| Clip | Expected class detected | Other-class detections | Track IDs | Fragments | CAUTION frames |
|---|---:|---|---:|---:|---:|
| cones | 90 / 484 (18.6%) | 39 (debris 38, rock 1) | 13 | 7 | 68 |
| rocks | 425 / 503 (84.5%) | 30 (cone 28, debris 2) | 32 | 17 | 367 |
| debris | 283 / 591 (47.9%) | 33 (rock 33) | 18 | 8 | 183 |
| empty | — | 578 false detections in 369 frames (64.2%) | 15 | 4 | 267 (46.4%) |

Processing (inference plus tracking) had a median of 28.8–32.0 ms per 1080p frame. That is higher than the 20–21 ms measured with the 640 × 480 webcam, because larger frames take longer to resize.

## Observations

### Rocks: works well

Large boulders were detected and tracked with stable IDs for long stretches: the longest tracks lasted 243, 218, 184, and 141 frames. Several low, flat rocks inside the monitoring zone were missed. The 32 track IDs mostly reflect many different rocks entering and leaving the view as the camera panned; 17 of them were short fragments.

### Debris: detected about half the time

Thin sticks on concrete were detected in some frames and missed in others, including sticks inside the monitoring zone. A pile of leaves at the sidewalk edge was also labeled debris, which is arguably correct. 33 detections were labeled rock.

### Cones: not a valid test of the cone class

The clip shows tall, thin orange delineator posts, mostly far away and behind a fence. The training class `cone` consists of traffic cones: all 17 training images are stock or product photos of traffic cones, usually large in the frame, with several showing a row or group of cones. One post was detected and tracked for 47 frames when the camera came close. A metal drainage grate was labeled debris. **This clip needs to be re-shot with traffic cones.**

### Empty scene: frequent false alarms

The obstacle-free sidewalk produced false detections in 64.2% of frames and a CAUTION warning in 46.4% of frames. The 578 false detections break down as follows:

| Source | Detections | Frames |
|---|---:|---:|
| Large boxes covering 20% or more of the frame, mostly labeled debris or cone, over grass and sidewalk | 291 | 269 |
| Other false detections, for example a lamp-post base labeled cone | 235 | 198 |
| The camera operator's fingertip at the top-right lens edge, labeled cone | 52 | 52 |

The longest false track was labeled debris and lasted 226 frames.

## Interpretation

The pipeline itself works as designed: every frame was processed in order, tracks persisted across frames, and the zone produced ENTER and EXIT events. The gaps are in the model, and they trace back to the training data:

- The training images are mostly internet stock photos with large, prominent objects, some on clean product-photo backgrounds.
- Real footage has small and distant objects, clutter, motion blur, and a low viewpoint.
- The model has learned to propose large boxes on textured scenes, which is the main source of false alarms.
- The training set contains no background-only images, so it has never been shown what "no obstacle" looks like.

On a rover, the empty-scene result would mean stopping for no reason in almost half of all frames.

## Limitations

- One clip per class, handheld, in one location and one lighting condition (evening).
- The cone clip does not contain the trained object.
- There is no frame-by-frame ground truth; results count detections, not verified true positives. Missed objects were identified by viewing the annotated recordings, not measured.
- The fingertip false positives are a filming artifact and would not occur on a mounted camera.

## Recommendations

These are advisory; the decision is the user's.

1. **Background negatives.** Add frames from obstacle-free footage to training with empty label files. This needs no labeling and directly targets the largest false-alarm source.
2. **Real-world positives.** Label frames from your own footage of rocks, debris, and traffic cones, especially small, distant, and low-viewpoint examples.
3. **Keep these four clips as a test set only.** Record separate footage for training, so the model is never evaluated on video it was trained on. Re-shoot the cone test with traffic cones, and keep fingers clear of the lens.
4. **Retrain and re-run this evaluation**, comparing against these results with `scripts/analyze_video.py`. Training is quick on this CPU: pilot_v2_data's 30 epochs took about 3 minutes (`runs/train/pilot_v2_data/results.csv`).
