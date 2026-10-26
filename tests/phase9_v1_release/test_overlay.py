"""Phase 9: overlay readability for the demo recording (9.3)."""

import numpy as np

import main
from edgesight.events import ZoneMonitor
from edgesight.tracker import TrackedObject, darken, draw_tracks, text_scale


def test_text_grows_with_frame_width_only_above_camera_size():
    assert text_scale(np.zeros((480, 640, 3), dtype=np.uint8)) == 1.0
    assert text_scale(np.zeros((240, 320, 3), dtype=np.uint8)) == 1.0
    assert text_scale(np.zeros((1080, 1920, 3), dtype=np.uint8)) == 3.0


def test_darken_clips_to_the_image():
    image = np.full((20, 30, 3), 200, dtype=np.uint8)

    darken(image, -5, -5, 10, 10, keep=0.5)

    assert np.all(image[:10, :10] == 100)
    assert np.all(image[10:, :] == 200) and np.all(image[:, 10:] == 200)


def test_metrics_and_labels_sit_on_dark_backdrops():
    # A bright scene, like sunlit concrete, where white and green text used to wash out
    frame = np.full((480, 640, 3), 230, dtype=np.uint8)
    rock = TrackedObject(track_id=1, label="rock", bbox=(300, 300, 340, 400), confidence=0.9)

    annotated = main.draw_overlay(frame, [rock], ZoneMonitor(), "CAUTION", ["FPS: 30.0", "Inference: 20 ms"])

    assert annotated[8:70, 8:20].mean() < 120
    assert draw_tracks(frame, [rock])[275:285, 300:310].mean() < 120
    assert np.all(frame == 230)
