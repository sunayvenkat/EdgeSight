

# EdgeSight — Model Improvement Experiments

## Research Question

Does expanding the training dataset with diverse, targeted images improve YOLO26n's ability to detect rover obstacles?

## Experiment 1: pilot_v1

- Starting weights: Pretrained YOLO26n
- Training images: 21
- Comparison validation images: 21
- Epoch limit: 30
- Image size: 512
- Batch size: 4
- Optimizer: AdamW
- Learning rate: 0.001
- Random seed: 42

## Experiment 2: pilot_v2_data

- Starting weights: Pretrained YOLO26n
- Training images: 51
- Comparison validation images: 21
- Epoch limit: 30
- Image size: 512
- Batch size: 4
- Optimizer: AdamW
- Learning rate: 0.001
- Random seed: 42

## Results

|  Metric   | pilot_v1 | pilot_v2_data | Change |
|-----------|----------|-----------|------------|
| Precision | TBD | TBD | TBD |
| Recall    | TBD | TBD | TBD |
| mAP50     | TBD | TBD | TBD |
| mAP50–95  | TBD | TBD | TBD |

## Per-Class Analysis

**Rock:** Record changes in detection performance.

**Cone:** Record changes in detection performance.

**Debris:** Record changes in detection performance.

## Interpretation

Describe whether the additional training data improved overall and per-class performance.

Identify likely causes of improvement, overfitting, or unexpected decreases.

## Limitations

- The training and validation datasets remain relatively small.
- The held-out test set contains only three images.
- The original and updated checkpoints were selected under different validation-set conditions.
- Results may vary with training randomness and dataset composition.
- Further independent evaluation is required before claiming reliable generalization.

## Next Experiments

Document future improvements supported by the results.