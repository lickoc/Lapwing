"""MediaPipe detection wrapper using the Tasks API."""

import os
import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


class FaceDetector:
    def __init__(self):
        model_path = os.path.join(MODELS_DIR, "face_landmarker.task")
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=config.MAX_NUM_FACES,
            min_face_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_rgb: np.ndarray) -> list[dict] | None:
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        faces = []
        for face_landmarks in result.face_landmarks:
            landmarks = []
            for lm in face_landmarks:
                landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
            faces.append(landmarks)
        return faces

    def close(self):
        self.detector.close()


class PoseDetector:
    def __init__(self):
        model_path = os.path.join(MODELS_DIR, "pose_landmarker.task")
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
            print("[Warning] pose_landmarker.task not found, pose detection disabled")
            self.detector = None
            return
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            min_pose_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_rgb: np.ndarray) -> list[dict] | None:
        if self.detector is None:
            return None
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)
        if not result.pose_landmarks:
            return None
        landmarks = []
        for lm in result.pose_landmarks[0]:
            landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
        return landmarks

    def close(self):
        if self.detector:
            self.detector.close()


class HandDetector:
    def __init__(self):
        model_path = os.path.join(MODELS_DIR, "hand_landmarker.task")
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_rgb: np.ndarray) -> list[list[dict]] | None:
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)
        if not result.hand_landmarks:
            return None
        hands = []
        for hand_landmarks in result.hand_landmarks:
            landmarks = []
            for lm in hand_landmarks:
                landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
            hands.append(landmarks)
        return hands

    def close(self):
        self.detector.close()
