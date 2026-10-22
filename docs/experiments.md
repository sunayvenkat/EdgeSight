

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
| Precision | 0.6606 | 0.6144 | −0.0462 |
| Recall    | 0.4342 | 0.5822 | +0.1480 |
| mAP50     | 0.4667 | 0.6261 | +0.1594 |
| mAP50–95  | 0.3068 | 0.3755 | +0.0687 |

Both checkpoints were evaluated on the same 21 validation images (69 labeled instances). Sources: `runs/evaluation/pilot_v1_commonval/summary.json` and `runs/evaluation/pilot_v2_commonval/summary.json`.

## Per-Class Analysis

| Class | Metric | pilot_v1 | pilot_v2_data | Change |
|---|---|---:|---:|---:|
| Rock | Precision | 0.6805 | 0.6685 | −0.0120 |
| Rock | Recall | 0.6849 | 0.7857 | +0.1008 |
| Rock | mAP50 | 0.7056 | 0.8247 | +0.1191 |
| Rock | mAP50–95 | 0.5115 | 0.5814 | +0.0699 |
| Cone | Precision | 0.9434 | 0.8483 | −0.0951 |
| Cone | Recall | 0.5000 | 0.6667 | +0.1667 |
| Cone | mAP50 | 0.5593 | 0.7793 | +0.2200 |
| Cone | mAP50–95 | 0.3682 | 0.4581 | +0.0899 |
| Debris | Precision | 0.3579 | 0.3265 | −0.0314 |
| Debris | Recall | 0.1176 | 0.2941 | +0.1765 |
| Debris | mAP50 | 0.1354 | 0.2744 | +0.1390 |
| Debris | mAP50–95 | 0.0407 | 0.0868 | +0.0461 |

Confusion-matrix counts below come from each run's `confusion_matrix.png`, generated at conf=0.001.

**Rock:** Recall, mAP50, and mAP50–95 all improved, and precision was essentially unchanged. Correctly classified rocks rose from 6 to 19 of 28, missed rocks fell from 5 to 0, and rocks predicted as debris fell from 12 to 6.

**Cone:** Cone had the largest mAP50 gain (+0.220), and recall rose from 0.500 to 0.667. Precision fell from 0.943 to 0.848. Correctly classified cones rose from 6 to 10 of 24 and missed cones fell from 5 to 3, but cones predicted as rock stayed frequent (8 to 9).

**Debris:** Debris roughly doubled its recall and mAP50 but remains the weakest class by a wide margin. Correctly classified debris rose from 6 to 11 of 17. Debris predicted as rock fell from 9 to 1, while debris predicted as cone rose from 2 to 4.

## Interpretation

The additional training data improved recall, mAP50, and mAP50–95 for all three classes, with overall mAP50 rising by 0.159 and mAP50–95 by 0.069. Precision fell slightly overall (−0.046) and in every class, so the expanded model finds more objects at the cost of some additional false or misclassified detections.

The most likely cause of the improvement is the larger and more varied training set (21 to 51 images; the current training split has 155 rock, 29 cone, and 54 debris instances). Rock/debris confusion fell in both directions: rocks predicted as debris dropped from 12 to 6, and debris predicted as rock from 9 to 1.

Neither run shows strong overfitting on this evaluation. Both runs completed all 30 epochs with their best validation mAP50–95 at epoch 21. pilot_v2_data's validation box loss stayed nearly flat after epoch 7 (1.347 to 1.388 at epoch 30).

The two runs used the same seed and hyperparameters, so the comparison isolates the dataset change. However, each was trained once, so run-to-run variance has not been measured.

## Limitations

- The training and validation datasets remain relatively small.
- The held-out test set contains only three images.
- The original and updated checkpoints were selected under different validation-set conditions.
- Results may vary with training randomness and dataset composition.
- Further independent evaluation is required before claiming reliable generalization.

## Next Experiments

- Collect more debris examples, including debris that resembles cones or rocks. Debris has the lowest AP of any class (mAP50 0.274).
- Target cone/rock confusion: 9 of 24 validation cones were still predicted as rock.
- Repeat both experiments with additional seeds to measure run-to-run variance before attributing gains to the data alone.
- Train and evaluate at image size 640, which the runtime now uses (see `docs/inference_optimization.md`), instead of 512.
- Report a confusion matrix at the 0.40 deployment confidence so false-positive counts reflect runtime behavior.