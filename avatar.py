"""Cartoon avatar skeleton definition and layer management."""

import math
import pygame

import config
from animator import AnimationParams


class Avatar:
    """Default cartoon character drawn with Pygame primitives."""

    def __init__(self, center_x: int, center_y: int):
        self.cx = center_x
        self.cy = center_y

    def draw(self, surface: pygame.Surface, params: AnimationParams):
        """Draw the avatar based on animation parameters."""
        cx, cy = self.cx, self.cy

        # Head rotation offset (simulates 3D)
        head_offset_x = int(params.head_yaw * 20)
        head_offset_y = int(params.head_pitch * 10)
        head_tilt = -params.head_roll * 5  # degrees

        # Body offset from pose
        body_offset_x = int((params.body_x - 0.5) * 80)
        body_offset_y = int((params.body_y - 0.5) * 40)

        self._draw_body(surface, cx + body_offset_x, cy + 120 + body_offset_y,
                         params.left_arm_x, params.left_arm_y,
                         params.right_arm_x, params.right_arm_y)
        self._draw_head(surface, cx + head_offset_x, cy + head_offset_y,
                        head_tilt, params)
        self._draw_eyebrows(surface, cx + head_offset_x, cy + head_offset_y,
                            params)
        self._draw_eyes(surface, cx + head_offset_x, cy + head_offset_y,
                        params)
        self._draw_mouth(surface, cx + head_offset_x, cy + head_offset_y,
                         params)

    def _draw_head(self, surface, x, y, tilt_deg, params):
        """Draw head as a circle with optional tilt."""
        # For tilt, we draw on a temp surface and rotate
        r = config.AVATAR_HEAD_RADIUS
        if abs(tilt_deg) > 0.5:
            temp = pygame.Surface((r * 2 + 20, r * 2 + 20), pygame.SRCALPHA)
            pygame.draw.circle(temp, config.COLOR_SKIN, (r + 10, r + 10), r)
            rotated = pygame.transform.rotate(temp, tilt_deg)
            rect = rotated.get_rect(center=(x, y))
            surface.blit(rotated, rect)
        else:
            pygame.draw.circle(surface, config.COLOR_SKIN, (x, y), r)

        # Hair (top arc)
        hair_rect = pygame.Rect(x - r - 5, y - r - 15, r * 2 + 10, r + 20)
        pygame.draw.ellipse(surface, config.COLOR_HAIR, hair_rect)
        # Redraw face over hair lower half
        pygame.draw.circle(surface, config.COLOR_SKIN, (x, y), r)

    def _draw_eyebrows(self, surface, hx, hy, params):
        """Draw eyebrows as thick arcs."""
        brow_y = hy - 35
        brow_raise_l = int(params.left_eyebrow * 12)
        brow_raise_r = int(params.right_eyebrow * 12)
        brow_tilt_l = params.left_eyebrow * 5
        brow_tilt_r = params.right_eyebrow * 5

        # Left eyebrow
        lx = hx - 35
        ly = brow_y - brow_raise_l
        self._draw_arc_brow(surface, lx, ly, 22, brow_tilt_l, flip=False)

        # Right eyebrow
        rx = hx + 35
        ry = brow_y - brow_raise_r
        self._draw_arc_brow(surface, rx, ry, 22, brow_tilt_r, flip=True)

    def _draw_arc_brow(self, surface, x, y, width, tilt, flip=False):
        """Draw a single eyebrow arc."""
        color = config.COLOR_EYEBROW
        rect = pygame.Rect(x - width, y - 6, width * 2, 12)
        start_angle = 200 if not flip else 340
        end_angle = 340 if not flip else 200
        # Adjust for tilt
        start_angle += int(tilt * 10)
        end_angle += int(tilt * 10)
        pygame.draw.arc(surface, color, rect,
                        math.radians(start_angle), math.radians(end_angle), 3)

    def _draw_eyes(self, surface, hx, hy, params):
        """Draw eyes: white ellipses with pupil."""
        eye_y = hy - 10
        left_eye_x = hx - 30
        right_eye_x = hx + 30

        for (ex, ey, open_amount) in [
            (left_eye_x, eye_y, params.left_eye_open),
            (right_eye_x, eye_y, params.right_eye_open),
        ]:
            # Eye white
            rh = max(2, int(config.AVATAR_EYE_RADIUS_Y * open_amount))
            rx = config.AVATAR_EYE_RADIUS_X
            eye_rect = pygame.Rect(ex - rx, ey - rh, rx * 2, rh * 2)
            pygame.draw.ellipse(surface, config.COLOR_EYE_WHITE, eye_rect)
            pygame.draw.ellipse(surface, (180, 180, 180), eye_rect, 1)

            # Pupil (only if eye is open enough)
            if open_amount > 0.15:
                pupil_r = config.AVATAR_PUPIL_RADIUS
                pupil_x = ex + int(params.eye_x * 5)
                pupil_y = ey + int(params.eye_y * 3)
                # Clip pupil within eye
                pupil_y = max(ey - rh + pupil_r, min(ey + rh - pupil_r, pupil_y))
                pygame.draw.circle(surface, config.COLOR_PUPIL,
                                   (pupil_x, pupil_y), pupil_r)
                # Highlight
                pygame.draw.circle(surface, (255, 255, 255),
                                   (pupil_x - 2, pupil_y - 2), 2)

    def _draw_mouth(self, surface, hx, hy, params):
        """Draw mouth as arc or ellipse based on smile/openness."""
        mouth_y = hy + 35
        mouth_x = hx
        width = config.AVATAR_MOUTH_WIDTH

        if params.mouth_open > 0.15:
            # Open mouth (ellipse)
            open_h = int(5 + params.mouth_open * 20)
            rect = pygame.Rect(mouth_x - width // 2, mouth_y - open_h // 2,
                               width, open_h)
            pygame.draw.ellipse(surface, config.COLOR_MOUTH, rect)
            pygame.draw.ellipse(surface, (150, 50, 50), rect, 2)
        else:
            # Closed mouth (arc for smile/frown)
            rect = pygame.Rect(mouth_x - width // 2, mouth_y - 10,
                               width, 20)
            if params.mouth_smile > 0.1:
                start_a = 200
                end_a = 340
            elif params.mouth_smile < -0.1:
                start_a = 20
                end_a = 160
            else:
                # Neutral: straight line
                pygame.draw.line(surface, config.COLOR_MOUTH,
                                 (mouth_x - width // 2, mouth_y),
                                 (mouth_x + width // 2, mouth_y), 2)
                return
            pygame.draw.arc(surface, config.COLOR_MOUTH, rect,
                            math.radians(start_a), math.radians(end_a), 3)

    def _draw_body(self, surface, x, y, left_arm_x, left_arm_y,
                   right_arm_x, right_arm_y):
        """Draw body as rounded rectangle with arms."""
        bw = config.AVATAR_BODY_WIDTH
        bh = config.AVATAR_BODY_HEIGHT
        body_rect = pygame.Rect(x - bw // 2, y, bw, bh)
        pygame.draw.rect(surface, config.COLOR_BODY, body_rect,
                         border_radius=15)
        # Collar detail
        collar_rect = pygame.Rect(x - 20, y, 40, 15)
        pygame.draw.rect(surface, (255, 255, 255), collar_rect,
                         border_radius=5)

        # Arms
        shoulder_y = y + 10
        left_shoulder_x = x - bw // 2
        right_shoulder_x = x + bw // 2

        arm_len = 80
        left_elbow_x = left_shoulder_x + int(left_arm_x * arm_len)
        left_elbow_y = shoulder_y + int(left_arm_y * arm_len) + 20
        right_elbow_x = right_shoulder_x + int(right_arm_x * arm_len)
        right_elbow_y = shoulder_y + int(right_arm_y * arm_len) + 20

        # Upper arm
        pygame.draw.line(surface, config.COLOR_ARM,
                         (left_shoulder_x, shoulder_y),
                         (left_elbow_x, left_elbow_y), 8)
        pygame.draw.line(surface, config.COLOR_ARM,
                         (right_shoulder_x, shoulder_y),
                         (right_elbow_x, right_elbow_y), 8)

        # Forearm (extends downward from elbow)
        left_hand_x = left_elbow_x + int(left_arm_x * 30)
        left_hand_y = left_elbow_y + 40 + int(abs(left_arm_y) * 20)
        right_hand_x = right_elbow_x + int(right_arm_x * 30)
        right_hand_y = right_elbow_y + 40 + int(abs(right_arm_y) * 20)

        pygame.draw.line(surface, config.COLOR_ARM,
                         (left_elbow_x, left_elbow_y),
                         (left_hand_x, left_hand_y), 7)
        pygame.draw.line(surface, config.COLOR_ARM,
                         (right_elbow_x, right_elbow_y),
                         (right_hand_x, right_hand_y), 7)

        # Hands (small circles)
        pygame.draw.circle(surface, config.COLOR_SKIN,
                           (left_hand_x, left_hand_y), 8)
        pygame.draw.circle(surface, config.COLOR_SKIN,
                           (right_hand_x, right_hand_y), 8)
