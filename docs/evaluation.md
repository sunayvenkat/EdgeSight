

# EdgeSight Model Evaluation

## Experiment
- Model: YOLO26n (fine-tuned)
- Checkpoint: pilot_v1/weights/best.pt
- Evaluation split: Validation
- Number of validation images: 6
- Inference image size: 512
- Device: CPU

## Overall Metrics
- Precision: 0.8403
- Recall: 0.5007
- mAP50: 0.6357
- mAP50-95: 0.4517

Source: runs/evaluation/pilot_v1_val/summary.json

## Per-Class Performance
| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| Rock  | 0.5208    | 1.0000 | 0.9950| 0.8955   |
| Cone  | 1.0000    | 0.5022 | 0.6917| 0.4016   |
| Debris| 1.0000    | 0.0000 | 0.2204| 0.0580   |

## Error Analysis

The confusion matrix
(runs/evaluation/pilot_v1_val/confusion_matrix.png)
covers 21 labeled instances: 2 rock, 7 cone,
12 debris. It was generated at conf=0.001, so
its background counts include very low-confidence
boxes and overstate false positives at the 0.40
deployment threshold.

### False Positives
Rock precision was 0.52 with recall 1.0: the
model found every rock but also produced extra
rock boxes. In the confusion matrix, background
regions were predicted most often as debris (945),
then rock (445) and cone (389).

### False Negatives
Debris recall was 0.0 and cone recall 0.50 at
the metric's operating point. The matrix's
background row is empty, so every labeled object
had some overlapping prediction at conf=0.001.
Misses came from low scores or wrong classes,
not from the absence of any box.

### Classification Errors
Debris was the most confused class: of 12 debris
instances, 8 were predicted as rock, 2 as cone,
and only 2 as debris. Of 7 cones, 3 were correct,
2 were predicted as rock and 2 as debris. Both
rocks were classified correctly.

### Localization Errors
The drop from mAP50 to mAP50-95 shows how tightly
boxes fit. Rock boxes were tight (0.995 to 0.896).
Cone (0.692 to 0.402) and especially debris
(0.220 to 0.058) lost most of their precision at
stricter IoU thresholds.

## Improvement Plan
1. Add training images with debris, especially
   debris that resembles rock, since 8 of 12
   debris instances were predicted as rock.
2. Enlarge the validation split so per-class
   metrics are not decided by a handful of
   instances (2 rocks here).
3. Review debris and cone box annotations for
   consistent extents, given their large
   mAP50 to mAP50-95 drop.

Items 1 and 2 were tested in pilot_v2_data
(51 training images, 21 validation images);
see docs/experiments.md.

## Limitations
This evaluation uses only six validation images.
Results are preliminary and are not evidence of
reliable generalization to unseen rover environments.

The held-out test split remains unused for
model selection.
