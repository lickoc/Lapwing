"""Camera capture module using OpenCV."""

import cv2
import numpy as np

import config


class CameraCapture:
    def __init__(self, camera_id: int = config.CAMERA_ID):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

    def read(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if not ret:
            return None
        if config.CAMERA_MIRROR:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    @property
    def is_opened(self) -> bool:
        return self.cap.isOpened()
