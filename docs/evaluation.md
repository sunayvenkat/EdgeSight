

# EdgeSight Model Evaluation

## Experiment
- Model: YOLO26n (fine-tuned)
- Checkpoint: pilot_v1/weights/best.pt
- Evaluation split: Validation
- Number of validation images: 6
- Inference image size: 512
- Device: CPU

## Overall Metrics
- Precision:
- Recall:
- mAP50:
- mAP50-95:

## Per-Class Performance
| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| Rock  |           |        |       |          |
| Cone  |           |        |       |          |
| Debris|           |        |       |          |

## Error Analysis

### False Positives
Describe objects incorrectly detected.

### False Negatives
Describe objects the model missed.

### Classification Errors
Describe cases where the wrong class was assigned.

### Localization Errors
Describe incorrectly positioned bounding boxes.

## Improvement Plan
List three evidence-based changes to test next.

## Limitations
This evaluation uses only six validation images.
Results are preliminary and are not evidence of
reliable generalization to unseen rover environments.

The held-out test split remains unused for
model selection.
