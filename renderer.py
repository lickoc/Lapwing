"""Pygame rendering engine with camera background and character overlay."""

import os
import time

import cv2
import numpy as np
import pygame

import config
from animator import AnimationParams
from avatar import Avatar


class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 12)

        self.avatar = Avatar(config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2)

        # Camera state
        self.show_camera_bg = True  # camera as background by default
        self.camera_frame: np.ndarray | None = None
        self._cam_bg_surf: pygame.Surface | None = None
        self._cam_bg_key: tuple[int, int] | None = None

        # Character image overlay
        self.character_image: pygame.Surface | None = None
        self.character_rect: pygame.Rect | None = None

        # HUD
        self.fps_display = 0.0
        self.preset_name = "None"
        self._frame_count = 0
        self._fps_time = time.time()

    def load_character_image(self, path: str) -> bool:
        """Load a PNG character image with transparency."""
        if not os.path.exists(path):
            print(f"[Warning] Character image not found: {path}")
            return False
        try:
            self.character_image = pygame.image.load(path).convert_alpha()
            # Scale to fit screen height (about 80% of window height)
            img_w, img_h = self.character_image.get_size()
            target_h = int(config.WINDOW_HEIGHT * 0.85)
            scale = target_h / img_h
            new_w = int(img_w * scale)
            self.character_image = pygame.transform.smoothscale(
                self.character_image, (new_w, target_h))
            self.character_rect = self.character_image.get_rect(
                center=(config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2))
            print(f"Loaded character: {new_w}x{target_h}")
            return True
        except Exception as e:
            print(f"[Warning] Failed to load character image: {e}")
            return False

    def set_camera_frame(self, frame: np.ndarray):
        self.camera_frame = frame

    def draw(self, params: AnimationParams):
        # Layer 1: Camera background or solid background
        if self.show_camera_bg and self.camera_frame is not None:
            self._draw_camera_background()
        else:
            self.screen.fill(config.COLOR_BG)

        # Layer 2: Character (image or programmatic)
        if self.character_image is not None:
            self._draw_character_image(params)
        else:
            self.avatar.draw(self.screen, params)

        # Layer 3: HUD
        self._draw_hud()

        pygame.display.flip()
        self.clock.tick(config.FPS_TARGET)
        self._update_fps()

    def _draw_camera_background(self):
        """Draw camera feed as fullscreen background."""
        frame = self.camera_frame
        if frame is None:
            self.screen.fill(config.COLOR_BG)
            return

        h, w = frame.shape[:2]
        key = (w, h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._cam_bg_key != key:
            # First time or resolution changed: create surface
            resized = cv2.resize(rgb, (config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
            self._cam_bg_surf = pygame.surfarray.make_surface(
                resized.swapaxes(0, 1))
            self._cam_bg_key = key
        else:
            # Reuse surface, just update pixels
            resized = cv2.resize(rgb, (config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
            pygame.surfarray.blit_array(
                self._cam_bg_surf, resized.swapaxes(0, 1))

        self.screen.blit(self._cam_bg_surf, (0, 0))

    def _draw_character_image(self, params: AnimationParams):
        """Draw loaded character image with position/rotation based on tracking."""
        if self.character_image is None or self.character_rect is None:
            return

        # Offset based on head tracking
        offset_x = int(params.head_yaw * 50)
        offset_y = int(params.head_pitch * 25)

        # Rotation from head roll
        angle = -params.head_roll * 10

        center_x = self.character_rect.centerx + offset_x
        center_y = self.character_rect.centery + offset_y

        if abs(angle) > 0.5:
            rotated = pygame.transform.rotate(self.character_image, angle)
            rect = rotated.get_rect(center=(center_x, center_y))
            self.screen.blit(rotated, rect)
        else:
            rect = self.character_rect.copy()
            rect.center = (center_x, center_y)
            self.screen.blit(self.character_image, rect)

    def _draw_hud(self):
        y = 10
        fps_text = self.font.render(f"FPS: {self.fps_display:.0f}", True,
                                    (200, 200, 200))
        self.screen.blit(fps_text, (10, y))
        y += 22

        # Show background mode
        bg_mode = "Camera BG" if self.show_camera_bg else "Solid BG"
        char_mode = "Image" if self.character_image else "Programmatic"
        status = f"{bg_mode} | {char_mode}"
        status_text = self.small_font.render(status, True, (150, 150, 150))
        self.screen.blit(status_text, (10, y))
        y += 18

        if self.preset_name != "None":
            preset_text = self.font.render(
                f"Preset: {self.preset_name}", True, (100, 200, 255))
            self.screen.blit(preset_text, (10, y))
            y += 22

        hint = "Q:Quit  B:BG  1-5:Expr  R:Reset  S:Shot"
        hint_text = self.small_font.render(hint, True, (120, 120, 120))
        self.screen.blit(hint_text, (10, config.WINDOW_HEIGHT - 20))

    def _update_fps(self):
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 0.5:
            self.fps_display = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_time = now

    def save_screenshot(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        pygame.image.save(self.screen, filename)
        return filename

    def quit(self):
        pygame.quit()
