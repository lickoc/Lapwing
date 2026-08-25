"""Cartoon avatar skeleton definition and layer management."""

import math
import pygame

import config
from animator import AnimationParams


class Avatar:
    """Cartoon character drawn with Pygame primitives."""

    def __init__(self, center_x: int, center_y: int):
        self.cx = center_x
        self.cy = center_y

    def draw(self, surface: pygame.Surface, params: AnimationParams):
        cx, cy = self.cx, self.cy

        # Body offset from pose
        body_offset_x = int((params.body_x - 0.5) * 100)
        body_offset_y = int((params.body_y - 0.5) * 50)

        # Draw body first (behind head)
        body_x = cx + body_offset_x
        body_y = cy + 140 + body_offset_y
        self._draw_body(surface, body_x, body_y, params)

        # Head offset (simulates 3D rotation)
        head_offset_x = int(params.head_yaw * 40)
        head_offset_y = int(params.head_pitch * 20)

        # Draw head with all features on a rotated temp surface
        self._draw_head_composite(surface, cx + head_offset_x, cy + head_offset_y,
                                  params)

    def _draw_head_composite(self, surface, x, y, params):
        """Draw entire head (face + features) on a temp surface, then rotate."""
        r = config.AVATAR_HEAD_RADIUS
        pad = 60  # extra space for features that extend beyond the circle
        size = (r + pad) * 2
        center = r + pad

        temp = pygame.Surface((size, size), pygame.SRCALPHA)

        # Head circle
        pygame.draw.circle(temp, config.COLOR_SKIN, (center, center), r)
        # Subtle shadow on lower half
        shadow_rect = pygame.Rect(center - r, center, r * 2, r)
        pygame.draw.ellipse(temp, config.COLOR_SKIN_SHADOW, shadow_rect)
        pygame.draw.circle(temp, config.COLOR_SKIN, (center, center), r - 2)

        # Hair
        self._draw_hair(temp, center, center, r)

        # Face features (all relative to head center)
        self._draw_eyebrows(temp, center, center, params)
        self._draw_eyes(temp, center, center, r, params)
        self._draw_mouth(temp, center, center, r, params)
        self._draw_nose(temp, center, center)

        # Rotate entire head
        tilt = -params.head_roll * 8  # degrees
        if abs(tilt) > 0.5:
            rotated = pygame.transform.rotate(temp, tilt)
            rect = rotated.get_rect(center=(x, y))
            surface.blit(rotated, rect)
        else:
            rect = temp.get_rect(center=(x, y))
            surface.blit(temp, rect)

    def _draw_hair(self, temp, cx, cy, r):
        """Draw hair on top of head."""
        # Main hair mass (ellipse covering top of head)
        hair_rect = pygame.Rect(cx - r - 8, cy - r - 20, r * 2 + 16, r + 30)
        pygame.draw.ellipse(temp, config.COLOR_HAIR, hair_rect)
        # Redraw face over lower hair
        pygame.draw.circle(temp, config.COLOR_SKIN, (cx, cy), r)

        # Side hair tufts
        pygame.draw.ellipse(temp, config.COLOR_HAIR,
                            pygame.Rect(cx - r - 10, cy - 20, 25, 50))
        pygame.draw.ellipse(temp, config.COLOR_HAIR,
                            pygame.Rect(cx + r - 15, cy - 20, 25, 50))

    def _draw_eyebrows(self, temp, cx, cy, params):
        """Draw thick expressive eyebrows."""
        brow_base_y = cy - 40
        brow_left_x = cx - 35
        brow_right_x = cx + 35
        brow_width = 28
        brow_height = 8

        for (bx, by_raw, raise_amt, flip) in [
            (brow_left_x, brow_base_y, params.left_eyebrow, False),
            (brow_right_x, brow_base_y, params.right_eyebrow, True),
        ]:
            by = by_raw - int(raise_amt * 18)
            # Tilt: inner end goes up when raised, down when furrowed
            tilt_angle = raise_amt * 12

            brow_surf = pygame.Surface((brow_width * 2, brow_height * 3), pygame.SRCALPHA)
            brow_center = (brow_width, brow_height * 1.5)
            # Draw rounded rectangle eyebrow
            brow_rect = pygame.Rect(0, brow_height, brow_width * 2, brow_height)
            pygame.draw.rect(brow_surf, config.COLOR_EYEBROW, brow_rect,
                             border_radius=4)

            if flip:
                brow_surf = pygame.transform.flip(brow_surf, True, False)

            rotated = pygame.transform.rotate(brow_surf, tilt_angle)
            rrect = rotated.get_rect(center=(bx, by))
            temp.blit(rotated, rrect)

    def _draw_eyes(self, temp, cx, cy, head_r, params):
        """Draw expressive eyes with moving pupils."""
        eye_y = cy - 12
        left_eye_x = cx - 35
        right_eye_x = cx + 35
        eye_rx = config.AVATAR_EYE_RADIUS_X
        eye_ry = config.AVATAR_EYE_RADIUS_Y

        for (ex, ey, open_amt) in [
            (left_eye_x, eye_y, params.left_eye_open),
            (right_eye_x, eye_y, params.right_eye_open),
        ]:
            # Eye opening height
            rh = max(3, int(eye_ry * open_amt))

            # Eye white (ellipse)
            white_rect = pygame.Rect(ex - eye_rx, ey - rh, eye_rx * 2, rh * 2)
            pygame.draw.ellipse(temp, config.COLOR_EYE_WHITE, white_rect)
            # Outline
            pygame.draw.ellipse(temp, (180, 180, 180), white_rect, 2)

            if open_amt > 0.12:
                # Iris (larger, colored)
                iris_r = min(rh - 2, 14)
                iris_x = ex + int(params.eye_x * 12)
                iris_y = ey + int(params.eye_y * 8)
                # Clip to eye bounds
                iris_x = max(ex - eye_rx + iris_r + 2, min(ex + eye_rx - iris_r - 2, iris_x))
                iris_y = max(ey - rh + iris_r + 2, min(ey + rh - iris_r - 2, iris_y))
                pygame.draw.circle(temp, (100, 160, 220), (iris_x, iris_y), iris_r)

                # Pupil
                pupil_r = min(iris_r - 2, config.AVATAR_PUPIL_RADIUS)
                if pupil_r > 1:
                    pygame.draw.circle(temp, config.COLOR_PUPIL,
                                       (iris_x, iris_y), pupil_r)
                    # Highlight
                    hl_x = iris_x - max(2, pupil_r // 3)
                    hl_y = iris_y - max(2, pupil_r // 3)
                    pygame.draw.circle(temp, (255, 255, 255),
                                       (hl_x, hl_y), max(2, pupil_r // 3))

    def _draw_nose(self, temp, cx, cy):
        """Draw a simple nose."""
        # Small triangle/dot
        nose_y = cy + 12
        pygame.draw.polygon(temp, config.COLOR_SKIN_SHADOW, [
            (cx, nose_y - 4),
            (cx - 5, nose_y + 6),
            (cx + 5, nose_y + 6),
        ])

    def _draw_mouth(self, temp, cx, cy, head_r, params):
        """Draw expressive mouth with curves."""
        mouth_y = cy + 38
        mouth_x = cx
        w = config.AVATAR_MOUTH_WIDTH

        if params.mouth_open > 0.1:
            # Open mouth: ellipse with teeth/tongue hint
            open_h = int(4 + params.mouth_open * 30)
            mouth_rect = pygame.Rect(mouth_x - w // 2, mouth_y - open_h // 2,
                                     w, open_h)
            pygame.draw.ellipse(temp, config.COLOR_MOUTH, mouth_rect)
            pygame.draw.ellipse(temp, (150, 50, 50), mouth_rect, 2)

            # Teeth hint at top
            if open_h > 10:
                teeth_rect = pygame.Rect(mouth_x - w // 2 + 4, mouth_y - open_h // 2,
                                         w - 8, min(6, open_h // 3))
                pygame.draw.rect(temp, (255, 255, 255), teeth_rect,
                                 border_radius=2)

        else:
            # Closed mouth: curve for smile/frown
            half_w = w // 2
            if params.mouth_smile > 0.05:
                # Smile: draw a curved line using polygon
                smile_depth = int(params.mouth_smile * 15)
                points = []
                for i in range(21):
                    t = i / 20.0
                    px = mouth_x - half_w + int(w * t)
                    # Parabola opening downward for smile
                    py = mouth_y - int(smile_depth * (1 - (2 * t - 1) ** 2))
                    points.append((px, py))
                if len(points) >= 2:
                    pygame.draw.lines(temp, config.COLOR_MOUTH, False, points, 3)
                # Fill below the curve for a "cheeky" smile
                if smile_depth > 5:
                    fill_points = points + [
                        (mouth_x + half_w, mouth_y),
                        (mouth_x - half_w, mouth_y),
                    ]
                    pygame.draw.polygon(temp, config.COLOR_MOUTH, fill_points)

            elif params.mouth_smile < -0.05:
                # Frown: curve opening upward
                frown_depth = int(-params.mouth_smile * 12)
                points = []
                for i in range(21):
                    t = i / 20.0
                    px = mouth_x - half_w + int(w * t)
                    py = mouth_y + int(frown_depth * (1 - (2 * t - 1) ** 2))
                    points.append((px, py))
                if len(points) >= 2:
                    pygame.draw.lines(temp, config.COLOR_MOUTH, False, points, 3)
            else:
                # Neutral: slight line
                pygame.draw.line(temp, config.COLOR_MOUTH,
                                 (mouth_x - half_w + 5, mouth_y),
                                 (mouth_x + half_w - 5, mouth_y), 3)

    def _draw_body(self, surface, x, y, params):
        """Draw body with torso and arms."""
        bw = config.AVATAR_BODY_WIDTH
        bh = config.AVATAR_BODY_HEIGHT

        # Neck
        neck_w = 30
        neck_h = config.AVATAR_NECK_HEIGHT
        neck_rect = pygame.Rect(x - neck_w // 2, y - neck_h, neck_w, neck_h)
        pygame.draw.rect(surface, config.COLOR_SKIN, neck_rect, border_radius=5)

        # Torso
        body_rect = pygame.Rect(x - bw // 2, y, bw, bh)
        pygame.draw.rect(surface, config.COLOR_BODY, body_rect, border_radius=20)

        # Shirt collar V-shape
        collar_points = [
            (x - 15, y + 5),
            (x, y + 30),
            (x + 15, y + 5),
        ]
        pygame.draw.lines(surface, config.COLOR_COLLAR, False, collar_points, 3)

        # Shirt pocket detail
        pocket_rect = pygame.Rect(x + 15, y + 40, 20, 20)
        pygame.draw.rect(surface, config.COLOR_BODY_DARK, pocket_rect, 2,
                         border_radius=3)

        # Arms
        shoulder_y = y + 15
        left_shoulder_x = x - bw // 2
        right_shoulder_x = x + bw // 2

        arm_len = 70
        # Left arm
        le_x = left_shoulder_x + int(params.left_arm_x * arm_len)
        le_y = shoulder_y + int(params.left_arm_y * arm_len) + 30
        lh_x = le_x + int(params.left_arm_x * 35)
        lh_y = le_y + 45 + int(abs(params.left_arm_y) * 15)

        # Draw arm segments with thickness
        pygame.draw.line(surface, config.COLOR_ARM,
                         (left_shoulder_x, shoulder_y), (le_x, le_y), 12)
        pygame.draw.line(surface, config.COLOR_ARM,
                         (le_x, le_y), (lh_x, lh_y), 10)
        # Joint circles
        pygame.draw.circle(surface, config.COLOR_BODY_DARK,
                           (left_shoulder_x, shoulder_y), 7)
        pygame.draw.circle(surface, config.COLOR_BODY_DARK, (le_x, le_y), 6)
        # Hand
        pygame.draw.ellipse(surface, config.COLOR_HAND,
                            pygame.Rect(lh_x - 8, lh_y - 6, 16, 14))

        # Right arm
        re_x = right_shoulder_x + int(params.right_arm_x * arm_len)
        re_y = shoulder_y + int(params.right_arm_y * arm_len) + 30
        rh_x = re_x + int(params.right_arm_x * 35)
        rh_y = re_y + 45 + int(abs(params.right_arm_y) * 15)

        pygame.draw.line(surface, config.COLOR_ARM,
                         (right_shoulder_x, shoulder_y), (re_x, re_y), 12)
        pygame.draw.line(surface, config.COLOR_ARM,
                         (re_x, re_y), (rh_x, rh_y), 10)
        pygame.draw.circle(surface, config.COLOR_BODY_DARK,
                           (right_shoulder_x, shoulder_y), 7)
        pygame.draw.circle(surface, config.COLOR_BODY_DARK, (re_x, re_y), 6)
        pygame.draw.ellipse(surface, config.COLOR_HAND,
                            pygame.Rect(rh_x - 8, rh_y - 6, 16, 14))
