"""Main entry point for Lapwing avatar system."""

import sys
import os

import pygame

import config
from capture import CameraCapture
from detector import FaceDetector, PoseDetector, HandDetector
from animator import ParameterAnimator, EXPRESSIONS
from renderer import Renderer


def main():
    print("Starting Lapwing...")

    # Initialize modules
    camera = CameraCapture(config.CAMERA_ID)
    face_detector = FaceDetector()
    pose_detector = PoseDetector()
    hand_detector = HandDetector()
    animator = ParameterAnimator()
    renderer = Renderer()

    # Try to load character image from common locations
    for path in ["character.png", "avatar.png", os.path.join("assets", "character.png")]:
        if os.path.exists(path):
            renderer.load_character_image(path)
            break

    # Check for command-line argument
    if len(sys.argv) > 1:
        renderer.load_character_image(sys.argv[1])

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = _handle_key(event.key, animator, renderer)

            frame = camera.read()
            if frame is None:
                continue

            renderer.set_camera_frame(frame)

            import cv2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_landmarks = face_detector.detect(frame_rgb)
            pose_landmarks = pose_detector.detect(frame_rgb)
            hand_landmarks = hand_detector.detect(frame_rgb)

            params = animator.update(face_landmarks, pose_landmarks, hand_landmarks)
            renderer.draw(params)

    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        face_detector.close()
        pose_detector.close()
        hand_detector.close()
        renderer.quit()
        print("Lapwing stopped.")


def _handle_key(key, animator, renderer) -> bool:
    """Handle key press. Returns False to quit."""
    if key == pygame.K_q:
        return False
    elif key == pygame.K_b:
        renderer.show_camera_bg = not renderer.show_camera_bg
    elif key == pygame.K_r:
        animator.params.reset()
        animator.smooth.__init__()
        animator.preset = None
        renderer.preset_name = "None"
    elif key == pygame.K_s:
        filename = renderer.save_screenshot()
        print(f"Screenshot saved: {filename}")
    elif pygame.K_1 <= key <= pygame.K_5:
        preset_id = key - pygame.K_0
        if animator.preset and preset_id in EXPRESSIONS:
            animator.preset = None
            renderer.preset_name = "None"
        else:
            animator.set_preset(preset_id)
            renderer.preset_name = EXPRESSIONS[preset_id]["name"]
    return True


if __name__ == "__main__":
    main()
