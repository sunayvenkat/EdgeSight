


from unittest.mock import Mock

import numpy as np
import pytest

import edgesight.camera as camera_module
from edgesight.camera import CameraStream, CameraError


def test_camera_reads_frame(monkeypatch):
    # A working camera should return a frame
    fake_camera = Mock()
    fake_camera.isOpened.return_value = True

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_camera.read.return_value = (True, frame)

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda index: fake_camera,)

    camera = CameraStream()
    camera.open()

    result = camera.read()

    assert result is frame

    camera.close()
    fake_camera.release.assert_called_once()


def test_camera_open_failure(monkeypatch):
    # Unavailable cameras should raise CameraError.
    fake_camera = Mock()
    fake_camera.isOpened.return_value = False

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda index: fake_camera,)

    camera = CameraStream()

    with pytest.raises(CameraError):
        camera.open()

    fake_camera.release.assert_called_once()


def test_camera_read_failure(monkeypatch):
    # Failed frame capture should raise CameraError.
    fake_camera = Mock()
    fake_camera.isOpened.return_value = True
    fake_camera.read.return_value = (False, None)

    monkeypatch.setattr(camera_module.cv2,"VideoCapture",lambda index: fake_camera,)

    camera = CameraStream()
    camera.open()

    with pytest.raises(CameraError):
        camera.read()

    camera.close()
