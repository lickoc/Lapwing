"""Pygame rendering engine."""

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
        self.avatar = Avatar(config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2 - 40)
        self.show_camera = False
        self.camera_frame: np.ndarray | None = None
        self.fps_display = 0.0
        self.preset_name = "None"
        self._frame_count = 0
        self._fps_time = time.time()

    def set_camera_frame(self, frame: np.ndarray):
        """Store camera frame for overlay display."""
        self.camera_frame = frame

    def draw(self, params: AnimationParams):
        """Render one frame."""
        self.screen.fill(config.COLOR_BG)

        # Draw avatar
        self.avatar.draw(self.screen, params)

        # Camera overlay
        if self.show_camera and self.camera_frame is not None:
            self._draw_camera_overlay()

        # HUD
        self._draw_hud()

        pygame.display.flip()
        self.clock.tick(config.FPS_TARGET)
        self._update_fps()

    def _draw_camera_overlay(self):
        """Draw small camera feed in bottom-right corner."""
        frame = self.camera_frame
        if frame is None:
            return
        # BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        ow, oh = config.CAMERA_OVERLAY_SIZE
        scale = min(ow / w, oh / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (nw, nh))
        surf = pygame.surfarray.make_surface(resized.swapaxes(0, 1))

        margin = config.CAMERA_OVERLAY_MARGIN
        x = config.WINDOW_WIDTH - nw - margin
        y = config.WINDOW_HEIGHT - nh - margin
        self.screen.blit(surf, (x, y))
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (x - 1, y - 1, nw + 2, nh + 2), 1)

    def _draw_hud(self):
        """Draw status information."""
        y = 10
        # FPS
        fps_text = self.font.render(f"FPS: {self.fps_display:.0f}", True,
                                    (200, 200, 200))
        self.screen.blit(fps_text, (10, y))
        y += 22

        # Preset
        if self.preset_name != "None":
            preset_text = self.font.render(
                f"Preset: {self.preset_name}", True, (100, 200, 255))
            self.screen.blit(preset_text, (10, y))
            y += 22

        # Controls hint
        hint = "Q:Quit  1-5:Expr  R:Reset  S:Shot  C:Camera"
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
        """Save current frame as screenshot."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        pygame.image.save(self.screen, filename)
        return filename

    def quit(self):
        pygame.quit()
