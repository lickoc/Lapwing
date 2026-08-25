"""Parameter calculation and smoothing from detection results."""

from dataclasses import dataclass
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

    # Head rotation (normalized -1 to 1)
    head_pitch: float = 0.0  # up/down
    head_yaw: float = 0.0  # left/right
    head_roll: float = 0.0  # tilt

    # Body (normalized 0-1)
    body_x: float = 0.5
    body_y: float = 0.5
    shoulder_width: float = 0.5

    # Arms (endpoints relative to body)
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


EXPRESSIONS = {
    1: {"name": "Neutral", "left_eye_open": 0.5, "right_eye_open": 0.5,
        "mouth_open": 0.0, "mouth_smile": 0.0,
        "left_eyebrow": 0.0, "right_eyebrow": 0.0},
    2: {"name": "Smile", "left_eye_open": 0.35, "right_eye_open": 0.35,
        "mouth_open": 0.15, "mouth_smile": 1.0,
        "left_eyebrow": 0.3, "right_eyebrow": 0.3},
    3: {"name": "Surprise", "left_eye_open": 1.0, "right_eye_open": 1.0,
        "mouth_open": 0.9, "mouth_smile": 0.0,
        "left_eyebrow": 0.9, "right_eyebrow": 0.9},
    4: {"name": "Angry", "left_eye_open": 0.25, "right_eye_open": 0.25,
        "mouth_open": 0.1, "mouth_smile": -0.9,
        "left_eyebrow": -0.8, "right_eyebrow": -0.8},
    5: {"name": "Sad", "left_eye_open": 0.25, "right_eye_open": 0.25,
        "mouth_open": 0.0, "mouth_smile": -0.6,
        "left_eyebrow": 0.6, "right_eyebrow": -0.4},
}


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _deadzone(v: float, dz: float = config.DEADZONE) -> float:
    """Apply deadzone: return 0 if magnitude is below threshold."""
    if abs(v) < dz:
        return 0.0
    return v


def _smooth(current: float, target: float, alpha: float) -> float:
    return current + (target - current) * (1.0 - alpha)


def _compute_head_pose(landmarks: list[dict]) -> tuple[float, float, float]:
    """Compute head pitch, yaw, roll from face landmarks using solvePnP."""
    h, w = 480, 640

    model_points = np.array([
        [0.0, 0.0, 0.0],       # Nose tip (1)
        [0.0, -63.6, -12.5],    # Chin (152)
        [-43.3, 32.7, -26.0],   # Left eye corner (33)
        [43.3, 32.7, -26.0],    # Right eye corner (263)
        [-28.9, -28.9, -24.1],  # Left mouth (61)
        [28.9, -28.9, -24.1],   # Right mouth (291)
    ], dtype=np.float64)

    image_points = np.array([
        [landmarks[1]["x"] * w, landmarks[1]["y"] * h],
        [landmarks[152]["x"] * w, landmarks[152]["y"] * h],
        [landmarks[33]["x"] * w, landmarks[33]["y"] * h],
        [landmarks[263]["x"] * w, landmarks[263]["y"] * h],
        [landmarks[61]["x"] * w, landmarks[61]["y"] * h],
        [landmarks[291]["x"] * w, landmarks[291]["y"] * h],
    ], dtype=np.float64)

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

    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


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
            lm = face_landmarks[0]

            # --- Eyes ---
            # Eye opening: vertical distance between upper and lower eyelids
            # Left eye: upper lid 159, lower lid 145
            left_eye_h = abs(lm[159]["y"] - lm[145]["y"])
            p.left_eye_open = _clamp(left_eye_h * 10.0, 0.0, 1.0)

            # Right eye: upper lid 386, lower lid 374
            right_eye_h = abs(lm[386]["y"] - lm[374]["y"])
            p.right_eye_open = _clamp(right_eye_h * 10.0, 0.0, 1.0)

            # Eye gaze (iris position relative to eye center)
            if len(lm) > 473:
                # Left eye
                iris_x = lm[468]["x"]
                eye_center_x = (lm[33]["x"] + lm[133]["x"]) / 2
                iris_y = lm[468]["y"]
                eye_center_y = (lm[159]["y"] + lm[145]["y"]) / 2
                p.eye_x = _deadzone(_clamp((iris_x - eye_center_x) * 15.0))
                p.eye_y = _deadzone(_clamp((iris_y - eye_center_y) * 15.0))

            # --- Mouth ---
            # Mouth open: vertical distance between upper and lower lip
            mouth_h = abs(lm[13]["y"] - lm[14]["y"])
            p.mouth_open = _clamp(mouth_h * 10.0, 0.0, 1.0)

            # Mouth smile: compare lip center Y to mouth corner Y
            # When smiling, corners go UP (lower y), lip center stays or goes down
            left_corner_y = lm[61]["y"]
            right_corner_y = lm[291]["y"]
            corner_avg_y = (left_corner_y + right_corner_y) / 2
            lip_center_y = lm[13]["y"]  # upper lip center
            mouth_width = abs(lm[291]["x"] - lm[61]["x"])

            # Positive = corners below center (frown), Negative = corners above (smile)
            # We invert so positive = smile
            if mouth_width > 0.001:
                smile_raw = -(corner_avg_y - lip_center_y) / mouth_width
                p.mouth_smile = _deadzone(_clamp(smile_raw * 4.0))
            else:
                p.mouth_smile = 0.0

            # --- Eyebrows ---
            # Distance from eyebrow to eye top — larger gap = raised brow
            left_brow_y = lm[70]["y"]
            left_eye_top_y = lm[159]["y"]
            left_gap = left_eye_top_y - left_brow_y  # positive = brow above eye
            p.left_eyebrow = _deadzone(_clamp((left_gap - 0.04) * 15.0))

            right_brow_y = lm[300]["y"]
            right_eye_top_y = lm[386]["y"]
            right_gap = right_eye_top_y - right_brow_y
            p.right_eyebrow = _deadzone(_clamp((right_gap - 0.04) * 15.0))

            # --- Head pose ---
            pitch, yaw, roll = _compute_head_pose(lm)
            # Wider range: divide by 15 instead of 30
            p.head_pitch = _deadzone(_clamp(pitch / 15.0))
            p.head_yaw = _deadzone(_clamp(yaw / 15.0))
            p.head_roll = _deadzone(_clamp(roll / 20.0))

        if pose_landmarks and len(pose_landmarks) > 14:
            # Shoulders
            ls = pose_landmarks[11]
            rs = pose_landmarks[12]
            p.body_x = (ls["x"] + rs["x"]) / 2
            p.body_y = (ls["y"] + rs["y"]) / 2
            p.shoulder_width = abs(rs["x"] - ls["x"])

            # Arms
            le = pose_landmarks[13]
            p.left_arm_x = _deadzone(_clamp((le["x"] - ls["x"]) * 2.5))
            p.left_arm_y = _deadzone(_clamp((le["y"] - ls["y"]) * 2.5))

            re = pose_landmarks[14]
            p.right_arm_x = _deadzone(_clamp((re["x"] - rs["x"]) * 2.5))
            p.right_arm_y = _deadzone(_clamp((re["y"] - rs["y"]) * 2.5))

        # Apply preset overlay
        if self.preset:
            blend = 0.7
            inv = 1.0 - blend
            p.left_eye_open = p.left_eye_open * inv + self.preset.get("left_eye_open", 0.5) * blend
            p.right_eye_open = p.right_eye_open * inv + self.preset.get("right_eye_open", 0.5) * blend
            p.mouth_open = p.mouth_open * inv + self.preset.get("mouth_open", 0.0) * blend
            p.mouth_smile = p.mouth_smile * inv + self.preset.get("mouth_smile", 0.0) * blend
            p.left_eyebrow = p.left_eyebrow * inv + self.preset.get("left_eyebrow", 0.0) * blend
            p.right_eyebrow = p.right_eyebrow * inv + self.preset.get("right_eyebrow", 0.0) * blend

        # Smooth all values
        s = self.smooth
        a = config.SMOOTH_ALPHA
        ha = config.HEAD_SMOOTH_ALPHA
        ba = config.BODY_SMOOTH_ALPHA

        s.left_eye_open = _smooth(s.left_eye_open, p.left_eye_open, a)
        s.right_eye_open = _smooth(s.right_eye_open, p.right_eye_open, a)
        s.eye_x = _smooth(s.eye_x, p.eye_x, a)
        s.eye_y = _smooth(s.eye_y, p.eye_y, a)
        s.mouth_open = _smooth(s.mouth_open, p.mouth_open, a)
        s.mouth_smile = _smooth(s.mouth_smile, p.mouth_smile, a)
        s.left_eyebrow = _smooth(s.left_eyebrow, p.left_eyebrow, a)
        s.right_eyebrow = _smooth(s.right_eyebrow, p.right_eyebrow, a)
        s.head_pitch = _smooth(s.head_pitch, p.head_pitch, ha)
        s.head_yaw = _smooth(s.head_yaw, p.head_yaw, ha)
        s.head_roll = _smooth(s.head_roll, p.head_roll, ha)
        s.body_x = _smooth(s.body_x, p.body_x, ba)
        s.body_y = _smooth(s.body_y, p.body_y, ba)
        s.shoulder_width = _smooth(s.shoulder_width, p.shoulder_width, ba)
        s.left_arm_x = _smooth(s.left_arm_x, p.left_arm_x, ba)
        s.left_arm_y = _smooth(s.left_arm_y, p.left_arm_y, ba)
        s.right_arm_x = _smooth(s.right_arm_x, p.right_arm_x, ba)
        s.right_arm_y = _smooth(s.right_arm_y, p.right_arm_y, ba)

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
