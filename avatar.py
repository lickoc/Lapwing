"""Anime-style cartoon avatar with expressive features."""

import math
import pygame

import config
from animator import AnimationParams


class Avatar:
    """Anime-style character drawn with Pygame primitives."""

    def __init__(self, center_x: int, center_y: int):
        self.cx = center_x
        self.cy = center_y

    def draw(self, surface: pygame.Surface, params: AnimationParams):
        cx, cy = self.cx, self.cy

        # Body offset from pose tracking
        body_ox = int((params.body_x - 0.5) * 100)
        body_oy = int((params.body_y - 0.5) * 50)

        # Draw body first (behind head)
        self._draw_body(surface, cx + body_ox, cy + 150 + body_oy, params)

        # Head offset from head tracking
        head_ox = int(params.head_yaw * 45)
        head_oy = int(params.head_pitch * 25)

        # Draw head with features on rotated temp surface
        self._draw_head(surface, cx + head_ox, cy + head_oy, params)

    def _draw_head(self, surface, x, y, params):
        """Draw entire head on temp surface, then rotate for tilt."""
        r = config.AVATAR_HEAD_RADIUS
        pad = 80  # space for hair and features
        size = (r + pad) * 2
        ctr = r + pad

        temp = pygame.Surface((size, size), pygame.SRCALPHA)

        # --- Face shape ---
        # Slightly oval face (wider at cheeks)
        face_rect = pygame.Rect(ctr - r, ctr - r + 5, r * 2, int(r * 2.1))
        pygame.draw.ellipse(temp, config.COLOR_SKIN, face_rect)
        # Jaw/chin refinement - slightly narrower at bottom
        chin_rect = pygame.Rect(ctr - r + 15, ctr + r - 10, r * 2 - 30, 30)
        pygame.draw.ellipse(temp, config.COLOR_SKIN, chin_rect)

        # --- Hair ---
        self._draw_hair(temp, ctr, r)

        # --- Face features ---
        self._draw_eyebrows(temp, ctr, params)
        self._draw_eyes(temp, ctr, r, params)
        self._draw_nose(temp, ctr)
        self._draw_mouth(temp, ctr, r, params)
        self._draw_blush(temp, ctr, params)

        # Rotate entire head for tilt
        tilt = -params.head_roll * 10
        if abs(tilt) > 0.5:
            rotated = pygame.transform.rotate(temp, tilt)
            rect = rotated.get_rect(center=(x, y))
            surface.blit(rotated, rect)
        else:
            rect = temp.get_rect(center=(x, y))
            surface.blit(temp, rect)

    def _draw_hair(self, temp, ctr, r):
        """Draw anime-style hair with bangs and side strands."""
        hair_color = config.COLOR_HAIR

        # Back hair (behind head)
        back_hair = pygame.Rect(ctr - r - 12, ctr - r - 25, r * 2 + 24, r + 35)
        pygame.draw.ellipse(temp, hair_color, back_hair)

        # Redraw face over back hair
        face_rect = pygame.Rect(ctr - r, ctr - r + 5, r * 2, int(r * 2.1))
        pygame.draw.ellipse(temp, config.COLOR_SKIN, face_rect)

        # Top hair (poofy)
        top_hair = pygame.Rect(ctr - r - 8, ctr - r - 30, r * 2 + 16, r + 25)
        pygame.draw.ellipse(temp, hair_color, top_hair)

        # Bangs (fringe) - three triangular sections
        bang_y = ctr - r + 15
        # Left bang
        pygame.draw.polygon(temp, hair_color, [
            (ctr - r + 5, bang_y - 10),
            (ctr - 25, bang_y + 35),
            (ctr - r + 20, bang_y + 40),
        ])
        # Center bang
        pygame.draw.polygon(temp, hair_color, [
            (ctr - 20, bang_y - 15),
            (ctr + 5, bang_y + 30),
            (ctr - 30, bang_y + 25),
        ])
        # Right bang
        pygame.draw.polygon(temp, hair_color, [
            (ctr + 20, bang_y - 15),
            (ctr + r - 5, bang_y - 10),
            (ctr + 25, bang_y + 35),
        ])

        # Side hair strands
        # Left strand
        left_strand = [
            (ctr - r - 5, ctr - 20),
            (ctr - r - 15, ctr + 30),
            (ctr - r - 8, ctr + 50),
            (ctr - r + 10, ctr + 20),
        ]
        pygame.draw.polygon(temp, hair_color, left_strand)
        # Right strand
        right_strand = [
            (ctr + r + 5, ctr - 20),
            (ctr + r + 15, ctr + 30),
            (ctr + r + 8, ctr + 50),
            (ctr + r - 10, ctr + 20),
        ]
        pygame.draw.polygon(temp, hair_color, right_strand)

        # Hair highlights (lighter streaks)
        highlight = (min(255, config.COLOR_HAIR[0] + 60),
                     min(255, config.COLOR_HAIR[1] + 50),
                     min(255, config.COLOR_HAIR[2] + 40))
        # Highlight streak on top
        pygame.draw.arc(temp, highlight,
                        pygame.Rect(ctr - 30, ctr - r - 25, 60, 40),
                        math.radians(200), math.radians(340), 3)

    def _draw_eyebrows(self, temp, ctr, params):
        """Draw anime-style eyebrows - thick with tapered ends."""
        brow_y = ctr - 45
        brow_left_x = ctr - 38
        brow_right_x = ctr + 38

        for (bx, raise_amt, flip) in [
            (brow_left_x, params.left_eyebrow, False),
            (brow_right_x, params.right_eyebrow, True),
        ]:
            by = brow_y - int(raise_amt * 20)
            brow_w = 25
            brow_h = 7

            # Create eyebrow surface
            brow_surf = pygame.Surface((brow_w * 2, brow_h * 4), pygame.SRCALPHA)
            brow_center = (brow_w, brow_h * 2)

            # Draw thick rounded eyebrow
            points = [
                (brow_center[0] - brow_w + 3, brow_center[1] - brow_h),
                (brow_center[0] + brow_w - 3, brow_center[1] - brow_h + 2),
                (brow_center[0] + brow_w - 5, brow_center[1] + brow_h - 2),
                (brow_center[0] - brow_w + 5, brow_center[1] + brow_h),
            ]
            pygame.draw.polygon(brow_surf, config.COLOR_EYEBROW, points)

            if flip:
                brow_surf = pygame.transform.flip(brow_surf, True, False)

            tilt = raise_amt * 15
            rotated = pygame.transform.rotate(brow_surf, tilt)
            rrect = rotated.get_rect(center=(bx, by))
            temp.blit(rotated, rrect)

    def _draw_eyes(self, temp, ctr, head_r, params):
        """Draw large anime-style eyes with detailed iris."""
        eye_y = ctr - 10
        left_eye_x = ctr - 38
        right_eye_x = ctr + 38
        eye_rx = config.AVATAR_EYE_RADIUS_X
        eye_ry = config.AVATAR_EYE_RADIUS_Y

        for (ex, ey, open_amt) in [
            (left_eye_x, eye_y, params.left_eye_open),
            (right_eye_x, eye_y, params.right_eye_open),
        ]:
            rh = max(4, int(eye_ry * open_amt))

            # Eye outline (slightly larger, dark)
            outline_rect = pygame.Rect(ex - eye_rx - 2, ey - rh - 2,
                                       (eye_rx + 2) * 2, (rh + 2) * 2)
            pygame.draw.ellipse(temp, (60, 60, 60), outline_rect)

            # Eye white
            white_rect = pygame.Rect(ex - eye_rx, ey - rh, eye_rx * 2, rh * 2)
            pygame.draw.ellipse(temp, config.COLOR_EYE_WHITE, white_rect)

            if open_amt > 0.1:
                # Iris (large, colorful)
                iris_r = min(rh - 3, 16)
                iris_x = ex + int(params.eye_x * 14)
                iris_y = ey + int(params.eye_y * 10)
                # Clip to eye
                iris_x = max(ex - eye_rx + iris_r + 3,
                             min(ex + eye_rx - iris_r - 3, iris_x))
                iris_y = max(ey - rh + iris_r + 3,
                             min(ey + rh - iris_r - 3, iris_y))

                # Iris gradient (lighter at bottom)
                iris_color_top = (60, 120, 180)
                iris_color_bot = (100, 180, 240)
                # Top half
                iris_rect = pygame.Rect(iris_x - iris_r, iris_y - iris_r,
                                        iris_r * 2, iris_r * 2)
                pygame.draw.circle(temp, iris_color_top, (iris_x, iris_y), iris_r)
                # Bottom highlight arc
                if iris_r > 4:
                    highlight_rect = pygame.Rect(iris_x - iris_r + 2,
                                                 iris_y,
                                                 iris_r * 2 - 4, iris_r - 2)
                    pygame.draw.ellipse(temp, iris_color_bot, highlight_rect)
                    # Re-circle to clip
                    pygame.draw.circle(temp, iris_color_top,
                                       (iris_x, iris_y), iris_r, 1)

                # Pupil
                pupil_r = max(2, min(iris_r - 3, config.AVATAR_PUPIL_RADIUS))
                pygame.draw.circle(temp, config.COLOR_PUPIL,
                                   (iris_x, iris_y), pupil_r)

                # Main highlight (large, top-right)
                hl_r = max(3, pupil_r // 2 + 2)
                hl_x = iris_x - iris_r // 3
                hl_y = iris_y - iris_r // 3
                pygame.draw.circle(temp, (255, 255, 255),
                                   (hl_x, hl_y), hl_r)
                # Small highlight (bottom-left)
                hl2_r = max(2, hl_r // 2)
                hl2_x = iris_x + iris_r // 3
                hl2_y = iris_y + iris_r // 3
                pygame.draw.circle(temp, (255, 255, 255),
                                   (hl2_x, hl2_y), hl2_r)

            # Upper eyelid line (thick, curved)
            if open_amt > 0.05:
                lid_points = []
                for i in range(21):
                    t = i / 20.0
                    px = ex - eye_rx + int(eye_rx * 2 * t)
                    # Arc shape
                    curve = -math.sin(t * math.pi) * (rh + 3)
                    py = ey + int(curve)
                    lid_points.append((px, py))
                if len(lid_points) >= 2:
                    pygame.draw.lines(temp, (40, 40, 40), False, lid_points, 3)

            # Lower eyelid (subtle)
            if open_amt > 0.2:
                lower_pts = []
                for i in range(11):
                    t = i / 10.0
                    px = ex - eye_rx + 5 + int((eye_rx * 2 - 10) * t)
                    curve = math.sin(t * math.pi) * 3
                    py = ey + rh - 2 + int(curve)
                    lower_pts.append((px, py))
                if len(lower_pts) >= 2:
                    pygame.draw.lines(temp, (180, 160, 150), False, lower_pts, 1)

    def _draw_nose(self, temp, ctr):
        """Draw small anime nose."""
        nose_y = ctr + 15
        # Small shadow triangle
        pygame.draw.polygon(temp, config.COLOR_SKIN_SHADOW, [
            (ctr - 2, nose_y - 2),
            (ctr - 6, nose_y + 8),
            (ctr + 2, nose_y + 8),
        ])
        # Subtle highlight
        pygame.draw.line(temp, (255, 230, 200),
                         (ctr, nose_y - 1), (ctr + 3, nose_y + 5), 1)

    def _draw_mouth(self, temp, ctr, head_r, params):
        """Draw expressive anime mouth."""
        mouth_y = ctr + 42
        mouth_x = ctr
        w = config.AVATAR_MOUTH_WIDTH

        if params.mouth_open > 0.1:
            # Open mouth
            open_h = int(5 + params.mouth_open * 35)
            # Outer mouth
            mouth_rect = pygame.Rect(mouth_x - w // 2, mouth_y - open_h // 2,
                                     w, open_h)
            pygame.draw.ellipse(temp, config.COLOR_MOUTH, mouth_rect)
            pygame.draw.ellipse(temp, (140, 40, 40), mouth_rect, 2)

            # Inner mouth (darker)
            if open_h > 12:
                inner_rect = pygame.Rect(mouth_x - w // 2 + 5,
                                         mouth_y - open_h // 2 + 3,
                                         w - 10, open_h - 8)
                pygame.draw.ellipse(temp, config.COLOR_MOUTH_INNER, inner_rect)

            # Teeth
            if open_h > 15:
                teeth_rect = pygame.Rect(mouth_x - w // 2 + 4,
                                         mouth_y - open_h // 2,
                                         w - 8, min(8, open_h // 3))
                pygame.draw.rect(temp, (255, 255, 255), teeth_rect,
                                 border_radius=3)

            # Tongue hint
            if open_h > 20:
                tongue_y = mouth_y + open_h // 4
                pygame.draw.ellipse(temp, (220, 100, 100),
                                    pygame.Rect(mouth_x - 10, tongue_y,
                                                20, open_h // 4))

        else:
            # Closed mouth
            half_w = w // 2
            if params.mouth_smile > 0.05:
                # Smile curve
                depth = int(params.mouth_smile * 18)
                points = []
                for i in range(21):
                    t = i / 20.0
                    px = mouth_x - half_w + int(w * t)
                    py = mouth_y - int(depth * (1 - (2 * t - 1) ** 2))
                    points.append((px, py))
                if len(points) >= 2:
                    pygame.draw.lines(temp, config.COLOR_MOUTH, False, points, 3)
                # Fill
                if depth > 4:
                    fill = points + [(mouth_x + half_w, mouth_y + 2),
                                     (mouth_x - half_w, mouth_y + 2)]
                    pygame.draw.polygon(temp, config.COLOR_MOUTH, fill)

            elif params.mouth_smile < -0.05:
                # Frown
                depth = int(-params.mouth_smile * 14)
                points = []
                for i in range(21):
                    t = i / 20.0
                    px = mouth_x - half_w + int(w * t)
                    py = mouth_y + int(depth * (1 - (2 * t - 1) ** 2))
                    points.append((px, py))
                if len(points) >= 2:
                    pygame.draw.lines(temp, config.COLOR_MOUTH, False, points, 3)
            else:
                # Neutral: small line
                pygame.draw.line(temp, config.COLOR_MOUTH,
                                 (mouth_x - 8, mouth_y),
                                 (mouth_x + 8, mouth_y), 2)

    def _draw_blush(self, temp, ctr, params):
        """Draw subtle blush marks on cheeks when smiling."""
        if params.mouth_smile > 0.3:
            blush_alpha = min(80, int(params.mouth_smile * 80))
            blush_color = (255, 150, 150, blush_alpha)
            blush_surf = pygame.Surface((30, 16), pygame.SRCALPHA)
            pygame.draw.ellipse(blush_surf, blush_color,
                                pygame.Rect(0, 0, 30, 16))
            # Left cheek
            temp.blit(blush_surf, (ctr - 65, ctr + 15))
            # Right cheek
            temp.blit(blush_surf, (ctr + 35, ctr + 15))

    def _draw_body(self, surface, x, y, params):
        """Draw anime-style body with clothing."""
        bw = config.AVATAR_BODY_WIDTH
        bh = config.AVATAR_BODY_HEIGHT

        # Neck
        neck_w = 28
        neck_h = config.AVATAR_NECK_HEIGHT
        pygame.draw.rect(surface, config.COLOR_SKIN,
                         pygame.Rect(x - neck_w // 2, y - neck_h, neck_w, neck_h),
                         border_radius=8)

        # Shoulders (wider at top, tapered)
        shoulder_pts = [
            (x - bw // 2 - 10, y + 5),   # Left shoulder
            (x - bw // 2 + 10, y),        # Left neck
            (x + bw // 2 - 10, y),        # Right neck
            (x + bw // 2 + 10, y + 5),    # Right shoulder
        ]
        pygame.draw.polygon(surface, config.COLOR_BODY, shoulder_pts)

        # Torso (rounded rectangle)
        body_rect = pygame.Rect(x - bw // 2, y + 5, bw, bh - 10)
        pygame.draw.rect(surface, config.COLOR_BODY, body_rect, border_radius=15)

        # V-neck collar
        collar_pts = [
            (x - 12, y + 3),
            (x, y + 28),
            (x + 12, y + 3),
        ]
        pygame.draw.lines(surface, (220, 220, 240), False, collar_pts, 2)
        # Fill collar area
        collar_fill = [
            (x - 12, y + 3),
            (x, y + 28),
            (x + 12, y + 3),
            (x + 8, y + 5),
            (x, y + 25),
            (x - 8, y + 5),
        ]
        pygame.draw.polygon(surface, (255, 255, 255), collar_fill)

        # Button line
        for i in range(3):
            btn_y = y + 35 + i * 18
            pygame.draw.circle(surface, (200, 200, 220), (x, btn_y), 3)

        # Arms
        shoulder_y = y + 12
        ls_x = x - bw // 2 - 5
        rs_x = x + bw // 2 + 5

        arm_len = 65
        # Left arm
        le_x = ls_x + int(params.left_arm_x * arm_len)
        le_y = shoulder_y + int(params.left_arm_y * arm_len) + 25
        lh_x = le_x + int(params.left_arm_x * 30)
        lh_y = le_y + 40 + int(abs(params.left_arm_y) * 15)

        pygame.draw.line(surface, config.COLOR_ARM,
                         (ls_x, shoulder_y), (le_x, le_y), 14)
        pygame.draw.line(surface, config.COLOR_ARM,
                         (le_x, le_y), (lh_x, lh_y), 12)
        # Joint
        pygame.draw.circle(surface, config.COLOR_BODY_DARK, (le_x, le_y), 7)
        # Hand (oval)
        hand_surf = pygame.Surface((18, 22), pygame.SRCALPHA)
        pygame.draw.ellipse(hand_surf, config.COLOR_HAND,
                            pygame.Rect(0, 0, 18, 22))
        hand_rect = hand_surf.get_rect(center=(lh_x, lh_y))
        surface.blit(hand_surf, hand_rect)

        # Right arm
        re_x = rs_x + int(params.right_arm_x * arm_len)
        re_y = shoulder_y + int(params.right_arm_y * arm_len) + 25
        rh_x = re_x + int(params.right_arm_x * 30)
        rh_y = re_y + 40 + int(abs(params.right_arm_y) * 15)

        pygame.draw.line(surface, config.COLOR_ARM,
                         (rs_x, shoulder_y), (re_x, re_y), 14)
        pygame.draw.line(surface, config.COLOR_ARM,
                         (re_x, re_y), (rh_x, rh_y), 12)
        pygame.draw.circle(surface, config.COLOR_BODY_DARK, (re_x, re_y), 7)
        hand_surf2 = pygame.Surface((18, 22), pygame.SRCALPHA)
        pygame.draw.ellipse(hand_surf2, config.COLOR_HAND,
                            pygame.Rect(0, 0, 18, 22))
        hand_rect2 = hand_surf2.get_rect(center=(rh_x, rh_y))
        surface.blit(hand_surf2, hand_rect2)
