"""MediaPipe detection wrapper for face, pose, and hands."""

import cv2
import mediapipe as mp
import numpy as np

import config


class FaceDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=config.MAX_NUM_FACES,
            refine_landmarks=True,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def detect(self, frame_rgb: np.ndarray) -> list[dict] | None:
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None
        faces = []
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
            faces.append(landmarks)
        return faces

    def close(self):
        self.face_mesh.close()


class PoseDetector:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            model_complexity=config.MODEL_COMPLEXITY,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def detect(self, frame_rgb: np.ndarray) -> list[dict] | None:
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
        return landmarks

    def close(self):
        self.pose.close()


class HandDetector:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def detect(self, frame_rgb: np.ndarray) -> list[list[dict]] | None:
        results = self.hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return None
        hands = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
            hands.append(landmarks)
        return hands

    def close(self):
        self.hands.close()
