"""Step 2 (intelligent) — detect the body's 3D facial landmarks.

Render the head frontally (software rasteriser), run MediaPipe FaceLandmarker on
that image, then map each 2D landmark back to a 3D point on the body surface using
the render's position buffer. Output: 478 body landmarks in canonical MediaPipe
index order — directly comparable to the scan's 478 canonical vertices.

MediaPipe is an *optional* dependency: if it (or the model) is unavailable, the
caller falls back to the geometric aligner.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .head_locator import HeadFrame
from . import render_head

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


class LandmarksUnavailable(RuntimeError):
    """MediaPipe missing, model missing, or no face detected on the render."""


def mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401
        return MODEL_PATH.exists()
    except Exception:
        return False


def _detect_2d(image: np.ndarray):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.2,
    )
    landmarker = vision.FaceLandmarker.create_from_options(opts)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
    if not result.face_landmarks:
        raise LandmarksUnavailable("no face detected on the head render")
    return result.face_landmarks[0]


def _nearest_valid(valid: np.ndarray, x: int, y: int, max_r: int = 6):
    h, w = valid.shape
    for r in range(max_r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and valid[yy, xx]:
                    return yy, xx
    return None


def detect_body_landmarks(body: trimesh.Trimesh, hf: HeadFrame, size: int = 600):
    """Return (landmarks_xyz (478,3) world, valid_mask (478,)).

    Raises LandmarksUnavailable if MediaPipe/model missing or no face is found.
    """
    if not mediapipe_available():
        raise LandmarksUnavailable("mediapipe or model file unavailable")

    hr = render_head.render_frontal(body, hf, size=size)
    lm = _detect_2d(hr.image)

    out = np.full((478, 3), np.nan)
    h, w = hr.size, hr.size
    for i, p in enumerate(lm):
        if i >= 478:
            break
        x, y = int(round(p.x * w)), int(round(p.y * h))
        hit = _nearest_valid(hr.valid, x, y)
        if hit is not None:
            out[i] = hr.position[hit[0], hit[1]]

    valid = ~np.isnan(out).any(axis=1)
    if valid.sum() < 100:
        raise LandmarksUnavailable(f"too few landmarks resolved ({int(valid.sum())})")
    return out, valid
