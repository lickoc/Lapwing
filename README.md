# Lapwing

Real-time webcam-driven cartoon avatar system. Captures your facial expressions and body movements via camera, then drives a 2D cartoon character to mirror them in real time.

## Requirements

- Python 3.10+
- Webcam
- uv (package manager)

## Installation

```bash
cd /home/liucong/project/Lapwing
uv sync
```

## Usage

```bash
uv run python main.py
```

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `1` | Toggle Neutral expression |
| `2` | Toggle Smile expression |
| `3` | Toggle Surprise expression |
| `4` | Toggle Angry expression |
| `5` | Toggle Sad expression |
| `R` | Reset pose and expression |
| `S` | Save screenshot |
| `C` | Toggle camera overlay |

## Architecture

```
Camera (OpenCV)
    |
    v
MediaPipe Detection
  |-- Face Mesh (468 landmarks)
  |-- Pose (33 landmarks)
  |-- Hands (21 landmarks x 2)
    |
    v
Parameter Calculation & Smoothing
    |
    v
Avatar Rendering (Pygame)
```

## Files

- `config.py` - Global settings (window, camera, smoothing, colors)
- `capture.py` - Camera capture via OpenCV
- `detector.py` - MediaPipe face/pose/hand detection
- `animator.py` - Expression parameter extraction and smoothing
- `avatar.py` - Cartoon character drawing
- `renderer.py` - Pygame rendering engine with HUD
- `main.py` - Main loop entry point

## Customization

Edit `config.py` to adjust:
- Window size and FPS target
- Smoothing coefficients (lower = more responsive, higher = smoother)
- Avatar colors and sizes
- Camera settings
