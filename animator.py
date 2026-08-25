"""Parameter calculation and smoothing from detection results."""

from dataclasses import dataclass, field
import math
import cv2
import numpy as np

import config


@dataclass
class AnimationParams:
    # Eyes (0.0=closed, 1.0=open)
    left_eye_open: float = 0.5
    right_eye_open: float = 0.5
    eye_x: float = 0.0  # -1 left, +1 right
    eye_y: float = 0.0  # -1 up, +1 down

    # Mouth (0.0=closed, 1.0=wide open)
    mouth_open: float = 0.0
    mouth_smile: float = 0.0  # -1=frown, 0=neutral, 1=smile

    # Eyebrows (0=neutral, positive=raised, negative=lowered)
    left_eyebrow: float = 0.0
    right_eyebrow: float = 0.0

    # Head rotation (degrees)
    head_pitch: float = 0.0  # up/down
    head_yaw: float = 0.0  # left/right
    head_roll: float = 0.0  # tilt

    # Body (normalized 0-1)
    body_x: float = 0.5
    body_y: float = 0.5
    shoulder_width: float = 0.5

    # Arms (endpoints relative to body, 0-1)
    left_arm_x: float = 0.0
    left_arm_y: float = 0.0
    right_arm_x: float = 0.0
    right_arm_y: float = 0.0

    def reset(self):
        self.__init__()


@dataclass
class SmoothState:
    left_eye_open: float = 0.5
    right_eye_open: float = 0.5
    eye_x: float = 0.0
    eye_y: float = 0.0
    mouth_open: float = 0.0
    mouth_smile: float = 0.0
    left_eyebrow: float = 0.0
    right_eyebrow: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    body_x: float = 0.5
    body_y: float = 0.5
    shoulder_width: float = 0.5
    left_arm_x: float = 0.0
    left_arm_y: float = 0.0
    right_arm_x: float = 0.0
    right_arm_y: float = 0.0


# Predefined expressions
EXPRESSIONS = {
    1: {"name": "Neutral", "left_eye_open": 0.5, "right_eye_open": 0.5, "mouth_open": 0.0, "mouth_smile": 0.0,
        "left_eyebrow": 0.0, "right_eyebrow": 0.0},
    2: {"name": "Smile", "left_eye_open": 0.4, "right_eye_open": 0.4, "mouth_open": 0.2, "mouth_smile": 1.0,
        "left_eyebrow": 0.3, "right_eyebrow": 0.3},
    3: {"name": "Surprise", "left_eye_open": 1.0, "right_eye_open": 1.0, "mouth_open": 0.8, "mouth_smile": 0.0,
        "left_eyebrow": 0.8, "right_eyebrow": 0.8},
    4: {"name": "Angry", "left_eye_open": 0.3, "right_eye_open": 0.3, "mouth_open": 0.1, "mouth_smile": -0.8,
        "left_eyebrow": -0.7, "right_eyebrow": -0.7},
    5: {"name": "Sad", "left_eye_open": 0.3, "right_eye_open": 0.3, "mouth_open": 0.0, "mouth_smile": -0.5,
        "left_eyebrow": 0.5, "right_eyebrow": -0.3},
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _smooth(current: float, target: float, alpha: float) -> float:
    return _lerp(current, target, 1.0 - alpha)


def _compute_head_pose(landmarks: list[dict]) -> tuple[float, float, float]:
    """Compute head pitch, yaw, roll from face landmarks using solvePnP."""
    h, w = 480, 640  # frame dimensions

    # 3D model points (generic face model)
    model_points = np.array([
        [0.0, 0.0, 0.0],       # Nose tip (index 1)
        [0.0, -63.6, -12.5],    # Chin (index 152)
        [-43.3, 32.7, -26.0],   # Left eye left corner (index 33)
        [43.3, 32.7, -26.0],    # Right eye right corner (index 263)
        [-28.9, -28.9, -24.1],  # Left mouth corner (index 61)
        [28.9, -28.9, -24.1],   # Right mouth corner (index 291)
    ], dtype=np.float64)

    # 2D image points from landmarks
    image_points = np.array([
        [landmarks[1]["x"] * w, landmarks[1]["y"] * h],
        [landmarks[152]["x"] * w, landmarks[152]["y"] * h],
        [landmarks[33]["x"] * w, landmarks[33]["y"] * h],
        [landmarks[263]["x"] * w, landmarks[263]["y"] * h],
        [landmarks[61]["x"] * w, landmarks[61]["y"] * h],
        [landmarks[291]["x"] * w, landmarks[291]["y"] * h],
    ], dtype=np.float64)

    # Camera internals (approximate)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rotation_vec)

    # Extract Euler angles
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(-rmat[2, 0], sy)
        yaw = math.atan2(rmat[1, 0], rmat[0, 0])
        roll = math.atan2(rmat[2, 1], rmat[2, 2])
    else:
        pitch = math.atan2(-rmat[2, 0], sy)
        yaw = math.atan2(-rmat[1, 2], rmat[1, 1])
        roll = 0.0

    # Convert to degrees
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)
    roll = math.degrees(roll)

    return pitch, yaw, roll


class ParameterAnimator:
    def __init__(self):
        self.params = AnimationParams()
        self.smooth = SmoothState()
        self.preset: dict | None = None

    def set_preset(self, preset_id: int):
        self.preset = EXPRESSIONS.get(preset_id)

    def update(self, face_landmarks: list[dict] | None,
               pose_landmarks: list[dict] | None,
               hand_landmarks: list[list[dict]] | None) -> AnimationParams:
        p = AnimationParams()

        if face_landmarks:
            landmarks = face_landmarks[0]

            # Eyes: use iris landmarks (refined face mesh)
            # Left eye: landmarks 468-472 (iris center ~468), open: vertical distance between lids
            left_upper = landmarks[159]["y"]
            left_lower = landmarks[145]["y"]
            left_eye_height = abs(left_upper - left_lower)
            p.left_eye_open = _clamp(left_eye_height * 8.0, 0.0, 1.0)

            right_upper = landmarks[386]["y"]
            right_lower = landmarks[374]["y"]
            right_eye_height = abs(right_upper - right_lower)
            p.right_eye_open = _clamp(right_eye_height * 8.0, 0.0, 1.0)

            # Eye gaze direction (from iris landmarks)
            if len(landmarks) > 473:
                left_iris_x = landmarks[468]["x"]
                left_eye_center_x = (landmarks[33]["x"] + landmarks[133]["x"]) / 2
                p.eye_x = _clamp((left_iris_x - left_eye_center_x) * 10.0)

                left_iris_y = landmarks[468]["y"]
                left_eye_center_y = (landmarks[159]["y"] + landmarks[145]["y"]) / 2
                p.eye_y = _clamp((left_iris_y - left_eye_center_y) * 10.0)

            # Mouth open: vertical distance between lips
            upper_lip = landmarks[13]["y"]
            lower_lip = landmarks[14]["y"]
            mouth_height = abs(upper_lip - lower_lip)
            p.mouth_open = _clamp(mouth_height * 8.0, 0.0, 1.0)

            # Mouth smile: horizontal mouth corners vs center
            left_corner = landmarks[61]["x"]
            right_corner = landmarks[291]["x"]
            mouth_center_x = (left_corner + right_corner) / 2
            mouth_width = abs(right_corner - left_corner)
            left_corner_y = landmarks[61]["y"]
            right_corner_y = landmarks[291]["y"]
            corner_avg_y = (left_corner_y + right_corner_y) / 2
            lip_center_y = landmarks[13]["y"]
            p.mouth_smile = _clamp((lip_center_y - corner_avg_y) / (mouth_width + 1e-6) * 5.0)

            # Eyebrows
            left_brow_y = landmarks[70]["y"]
            left_eye_top = landmarks[159]["y"]
            brow_eye_gap = (left_eye_top - left_brow_y) * 5.0
            p.left_eyebrow = _clamp(brow_eye_gap - 0.5)

            right_brow_y = landmarks[300]["y"]
            right_eye_top = landmarks[386]["y"]
            brow_eye_gap_r = (right_eye_top - right_brow_y) * 5.0
            p.right_eyebrow = _clamp(brow_eye_gap_r - 0.5)

            # Head pose
            pitch, yaw, roll = _compute_head_pose(landmarks)
            p.head_pitch = _clamp(pitch / 30.0) * config.HEAD_ROTATION_SCALE
            p.head_yaw = _clamp(yaw / 30.0) * config.HEAD_ROTATION_SCALE
            p.head_roll = _clamp(roll / 30.0) * config.HEAD_ROTATION_SCALE

        if pose_landmarks:
            # Shoulders (landmarks 11=left, 12=right)
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            p.body_x = (left_shoulder["x"] + right_shoulder["x"]) / 2
            p.body_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
            p.shoulder_width = abs(right_shoulder["x"] - left_shoulder["x"])

            # Arms: elbow position relative to shoulder
            if len(pose_landmarks) > 16:
                left_elbow = pose_landmarks[13]
                left_shoulder_pos = pose_landmarks[11]
                p.left_arm_x = _clamp((left_elbow["x"] - left_shoulder_pos["x"]) * 2.0)
                p.left_arm_y = _clamp((left_elbow["y"] - left_shoulder_pos["y"]) * 2.0)

                right_elbow = pose_landmarks[14]
                right_shoulder_pos = pose_landmarks[12]
                p.right_arm_x = _clamp((right_elbow["x"] - right_shoulder_pos["x"]) * 2.0)
                p.right_arm_y = _clamp((right_elbow["y"] - right_shoulder_pos["y"]) * 2.0)

        # Apply preset overlay
        if self.preset:
            p.left_eye_open = p.left_eye_open * 0.3 + self.preset.get("left_eye_open", 0.5) * 0.7
            p.right_eye_open = p.right_eye_open * 0.3 + self.preset.get("right_eye_open", 0.5) * 0.7
            p.mouth_open = p.mouth_open * 0.3 + self.preset.get("mouth_open", 0.0) * 0.7
            p.mouth_smile = p.mouth_smile * 0.3 + self.preset.get("mouth_smile", 0.0) * 0.7
            p.left_eyebrow = p.left_eyebrow * 0.3 + self.preset.get("left_eyebrow", 0.0) * 0.7
            p.right_eyebrow = p.right_eyebrow * 0.3 + self.preset.get("right_eyebrow", 0.0) * 0.7

        # Smooth
        s = self.smooth
        alpha = config.SMOOTH_ALPHA
        head_alpha = config.HEAD_SMOOTH_ALPHA
        body_alpha = config.BODY_SMOOTH_ALPHA

        s.left_eye_open = _smooth(s.left_eye_open, p.left_eye_open, alpha)
        s.right_eye_open = _smooth(s.right_eye_open, p.right_eye_open, alpha)
        s.eye_x = _smooth(s.eye_x, p.eye_x, alpha)
        s.eye_y = _smooth(s.eye_y, p.eye_y, alpha)
        s.mouth_open = _smooth(s.mouth_open, p.mouth_open, alpha)
        s.mouth_smile = _smooth(s.mouth_smile, p.mouth_smile, alpha)
        s.left_eyebrow = _smooth(s.left_eyebrow, p.left_eyebrow, alpha)
        s.right_eyebrow = _smooth(s.right_eyebrow, p.right_eyebrow, alpha)
        s.head_pitch = _smooth(s.head_pitch, p.head_pitch, head_alpha)
        s.head_yaw = _smooth(s.head_yaw, p.head_yaw, head_alpha)
        s.head_roll = _smooth(s.head_roll, p.head_roll, head_alpha)
        s.body_x = _smooth(s.body_x, p.body_x, body_alpha)
        s.body_y = _smooth(s.body_y, p.body_y, body_alpha)
        s.shoulder_width = _smooth(s.shoulder_width, p.shoulder_width, body_alpha)
        s.left_arm_x = _smooth(s.left_arm_x, p.left_arm_x, body_alpha)
        s.left_arm_y = _smooth(s.left_arm_y, p.left_arm_y, body_alpha)
        s.right_arm_x = _smooth(s.right_arm_x, p.right_arm_x, body_alpha)
        s.right_arm_y = _smooth(s.right_arm_y, p.right_arm_y, body_alpha)

        return AnimationParams(
            left_eye_open=s.left_eye_open,
            right_eye_open=s.right_eye_open,
            eye_x=s.eye_x,
            eye_y=s.eye_y,
            mouth_open=s.mouth_open,
            mouth_smile=s.mouth_smile,
            left_eyebrow=s.left_eyebrow,
            right_eyebrow=s.right_eyebrow,
            head_pitch=s.head_pitch,
            head_yaw=s.head_yaw,
            head_roll=s.head_roll,
            body_x=s.body_x,
            body_y=s.body_y,
            shoulder_width=s.shoulder_width,
            left_arm_x=s.left_arm_x,
            left_arm_y=s.left_arm_y,
            right_arm_x=s.right_arm_x,
            right_arm_y=s.right_arm_y,
        )
