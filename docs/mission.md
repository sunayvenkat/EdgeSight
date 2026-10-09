# EdgeSight — Project Mission

## Overview

EdgeSight is a real-time edge AI vision platform designed to detect, track, and analyze objects in a camera feed. The project combines deep learning, computer vision, software architecture, and embedded-system optimization.

## Primary Objectives

1. Develop a modular computer vision pipeline supporting live cameras and recorded videos.
2. Implement neural network object detection and multi-object tracking.
3. Develop a lightweight obstacle assessment system using configurable detection zones.
4. Fine-tune and evaluate a compact detection model on rover-relevant objects.
5. Benchmark inference accuracy, speed, and system resource utilization.
6. Optimize inference using techniques such as model export and quantization.
7. Deploy and evaluate the system on a Raspberry Pi 5 without requiring a cloud inference service.

## Initial MVP

A desktop application that:

- Accepts webcam or video-file input.
- Detects objects and displays bounding boxes.
- Tracks objects across video frames.
- Displays FPS and inference latency.
- Outputs a simple obstacle warning.
- Can run a continuous 60-second demonstration.

## Engineering Requirements

- Modular architecture separating video capture, inference, tracking, and decision logic.
- Automated testing for core components.
- Reproducible model and configuration settings.
- Benchmarking of performance across computing environments.
- Documented installation and deployment instructions.

## Success Criteria

The final project should demonstrate a functioning desktop vision application, experimentally evaluated ML performance, measurable inference optimization, and CPU-based Raspberry Pi deployment.

Initial performance targets are at least 10 FPS on the development computer and 3 FPS on the Raspberry Pi, subject to validation against the chosen hardware and model.

## Out of Scope

- Full autonomous robot navigation.
- Training a large neural network from scratch.
- Cloud-dependent inference.
- Building custom battery systems or motor controllers.

Optional future extensions include robotic integration and an AI hardware accelerator.