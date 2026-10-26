"""Fakes and fixtures shared by tests in several phases."""

import time
from types import SimpleNamespace

import cv2
import numpy as np

from edgesight.sources import FramePacket
from edgesight.tracker import TrackedObject


class FakeCamera:
    def __init__(self, frame_count):
        self.frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(frame_count)]
        self.frames_captured = frame_count
        self.dropped_frames = 0
        self.error = None

    @property
    def is_running(self):
        return bool(self.frames)

    def read_latest(self, after_frame_id=-1, timeout=1.0):
        return FramePacket(frame=self.frames.pop(0), frame_id=after_frame_id + 1, captured_at=time.monotonic())


class FakeTracker:
    def __init__(self):
        self.calls = 0

    def track(self, frame):
        self.calls += 1

        # Bottom-center inside the default monitoring zone
        return [TrackedObject(track_id=1, label="rock", bbox=(300, 300, 340, 400), confidence=0.9)]


def write_video(path, frame_count, fps=30.0, size=(64, 48)):
    # Solid frames whose brightness encodes their index, so order survives MJPG compression
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)

    assert writer.isOpened()

    for index in range(frame_count):
        writer.write(np.full((size[1], size[0], 3), index * 20, dtype=np.uint8))

    writer.release()

    return path


def make_box(confidence):
    return SimpleNamespace(
        xyxy=np.array([[10, 20, 100, 200]]),
        conf=np.array([confidence]),
        cls=np.array([0]),
    )


def make_output(rows, classes=3, anchors=8400):
    # rows are (cx, cy, w, h, class_id, score) in letterboxed pixels
    output = np.zeros((1, 4 + classes, anchors), dtype=np.float32)

    for index, (cx, cy, w, h, class_id, score) in enumerate(rows):
        output[0, :4, index] = (cx, cy, w, h)
        output[0, 4 + class_id, index] = score

    return output


class FakeSession:
    def __init__(self, output, shape=(1, 3, 640, 640)):
        self.output = output
        self.shape = list(shape)
        self.feeds = []

    def get_inputs(self):
        return [SimpleNamespace(name="images", shape=self.shape)]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map={"names": "{0: 'rock', 1: 'cone', 2: 'debris'}"})

    def run(self, output_names, feeds):
        self.feeds.append(feeds)
        return [self.output]
