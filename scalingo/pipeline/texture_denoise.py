"""Edge-preserving denoise of the body atlas.

Meshy bodies ship an albedo peppered with grey micro-speckle (high-frequency
salt-and-pepper grain) that makes the skin look granular. We remove that grain
WITHOUT a plain blur: a Non-Local Means filter averages similar patches, so flat
skin is smoothed while real edges (hair/skin/fabric boundaries, woven logos) stay
sharp. This touches only the body's baseColorTexture image — never geometry, UVs
or the face mesh.

OpenCV is optional across the pipeline; if it is absent the body is returned
unchanged (same policy as the landmark/tone-match steps).
"""
from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image

try:
    import cv2  # noqa: F401
    _CV2 = True
except Exception:  # noqa: BLE001
    _CV2 = False


def denoise_image(img: Image.Image, h: int = 7, template: int = 7,
                  search: int = 21) -> Image.Image:
    """Non-Local Means denoise of a PIL image, preserving any alpha channel.

    `h` is the filter strength: higher removes more grain (and eventually fine
    detail). 7 clears Meshy's speckle while keeping edges crisp.
    """
    import cv2
    alpha = img.getchannel("A") if img.mode == "RGBA" else None
    rgb = np.asarray(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    den = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, template, search)
    out = Image.fromarray(cv2.cvtColor(den, cv2.COLOR_BGR2RGB))
    if alpha is not None:
        out.putalpha(alpha)
    return out


def denoise_body_texture(body: trimesh.Trimesh, h: int = 7) -> trimesh.Trimesh:
    """In-place: replace the body's albedo with a grain-free version.

    No-op (returns body unchanged) when OpenCV is unavailable or the body has no
    baseColorTexture.
    """
    if not _CV2:
        return body
    mat = getattr(body.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None)
    if img is None:
        return body
    mat.baseColorTexture = denoise_image(img, h=h)
    return body
