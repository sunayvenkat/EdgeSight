
# EdgeSight Training Experiments

## Experiment: pilot_v1

- Model: YOLO26n
- Initial weights: pretrained
- Dataset: rover_obstacles (30 images)
- Train/Val/Test: 21/6/3
- Epochs requested: 30
- Image size: 512
- Batch size: 4
- Optimizer: AdamW
- Learning rate: 0.001
- Device: CPU
- Seed: 42

## Results

- Epochs completed: 30 of 30 (early stopping,
  patience 10, was not triggered)
- Best validation mAP50: 0.6520 (epoch 21)
- Best validation mAP50-95: 0.4686 (epoch 21)
- Precision: 0.7851 (epoch 21)
- Recall: 0.5714 (epoch 21)
- Training observations: Validation mAP50-95
  rose quickly to 0.414 by epoch 10, peaked at
  epoch 21, and finished at 0.453 at epoch 30.
  Validation box loss reached its minimum
  (1.434) at epoch 9 and drifted up to 1.600 by
  epoch 30 while training box loss kept falling,
  a mild sign of overfitting. Validation
  classification loss kept declining
  (9.69 to 4.45). With only 6 validation images,
  metrics moved in large steps between epochs.

Source: runs/train/pilot_v1/results.csv

## Limitations

Small experimental dataset; metrics are not
sufficient to establish generalization.