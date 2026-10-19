

# Inference Resolution Optimization

## Objective

Compare YOLO inference image sizes of 416, 512, and 640 for EdgeSight. Evaluate detection quality and real-time camera-pipeline performance to choose a defensible default for rover-obstacle detection.

## Experimental Setup

- Model: `models/selected.pt`
- Device: CPU
- Camera source: OpenCV source `0`
- Requested camera resolution: 640 × 480
- Camera benchmark duration: approximately 30 seconds per mode
- Warmup: 5 seconds per mode
- Classes: `rock`, `cone`, `debris`
- Validation set: 21 images and 69 labeled instances

The camera benchmark measures software-side frame age beginning when OpenCV returns a frame. It does not measure latency from sensor exposure.

## Validation Results

| Inference Size | Precision | Recall | mAP@50 | mAP@50–95 |
|---:|---:|---:|---:|---:|
| 416 | 0.597 | 0.590 | 0.586 | 0.385 |
| 512 | 0.637 | 0.594 | 0.627 | 0.379 |
| 640 | 0.568 | 0.625 | 0.597 | 0.390 |

### Per-Class Results

| Size | Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---:|---|---:|---:|---:|---:|
| 416 | Rock | 0.624 | 0.750 | 0.798 | 0.566 |
| 416 | Cone | 0.788 | 0.667 | 0.687 | 0.460 |
| 416 | Debris | 0.380 | 0.353 | 0.273 | 0.128 |
| 512 | Rock | 0.690 | 0.821 | 0.825 | 0.582 |
| 512 | Cone | 0.841 | 0.667 | 0.777 | 0.463 |
| 512 | Debris | 0.379 | 0.294 | 0.279 | 0.0919 |
| 640 | Rock | 0.502 | 0.714 | 0.768 | 0.544 |
| 640 | Cone | 0.856 | 0.750 | 0.808 | 0.550 |
| 640 | Debris | 0.345 | 0.412 | 0.215 | 0.0751 |

## Camera Pipeline Results

### Synchronous Mode

| Size | Processed FPS | P95 Processing Time |
|---:|---:|---:|
| 416 | 29.983 | 24.538 ms |
| 512 | 24.149 | 60.433 ms |
| 640 | 27.640 | 38.460 ms |

### Asynchronous Mode

| Size | Processed FPS | P95 Processing Time | P95 Frame Age at Processing End | Dropped/Overwritten Frames |
|---:|---:|---:|---:|---:|
| 416 | 23.217 | 50.270 ms | 78.709 ms | 196 |
| 512 | 28.354 | 36.011 ms | 57.947 ms | 43 |
| 640 | 27.828 | 37.281 ms | 58.723 ms | 57 |

## Interpretation

- Size 512 achieved the highest mAP@50 and the highest measured asynchronous processed FPS.
- Size 640 achieved the highest overall recall and mAP@50–95, as well as the highest debris recall.
- Size 416 achieved the highest measured synchronous FPS, but its asynchronous result was substantially slower.
- The asynchronous 512 benchmark processed 0.526 more frames per second than the asynchronous 640 benchmark in these individual runs. This difference is not yet established as repeatable.
- The synchronous results vary considerably between runs. These measurements should not be interpreted as a controlled proof that reducing image size necessarily improves throughput.
- The validation set is small, so its metrics are preliminary. Debris detection remains a weakness at all three resolutions.

## Decision

Use inference size **640 as the provisional default** for the next integration stage. It provides the strongest measured overall recall, mAP@50–95, and debris recall, while its asynchronous throughput was close to the 512 result.

Retain 512 as an alternative for future evaluation if repeated benchmarks establish a meaningful performance advantage without an unacceptable reduction in obstacle-detection quality.

This is a provisional engineering decision, not a claim of production-ready obstacle detection.

## Limitations and Follow-Up

1. Repeat resolution benchmarks under comparable CPU conditions and compare the results.
2. Evaluate on a larger, representative dataset, especially for debris.
3. Validate detection and tracking on temporally ordered video before drawing conclusions about rover behavior.
4. Record the final inference-size setting and benchmark methodology in the project documentation.
