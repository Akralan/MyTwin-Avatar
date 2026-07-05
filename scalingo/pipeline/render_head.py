"""Software (pure-numpy) rasteriser for the body head.

This is what makes the doc's "render the head in 2D, run MediaPipe" step practical
without any offscreen-GL stack: we orthographically rasterise the head in the
detected head frame and, in the same pass, output a **position buffer** (the 3D
world coordinate visible at each pixel). That buffer is the ray-cast: a 2D
landmark pixel maps straight back to a 3D point on the body surface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .head_locator import HeadFrame


@dataclass
class HeadRender:
    image: np.ndarray        # (S, S, 3) uint8 — for the landmark detector
    position: np.ndarray     # (S, S, 3) float — world XYZ visible at each pixel
    valid: np.ndarray        # (S, S) bool — pixel covered by geometry
    size: int


def sample_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """Per-vertex RGB from the base-colour texture via UV (uint8)."""
    try:
        uv = np.asarray(mesh.visual.uv)
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))
        h, w = tex.shape[:2]
        u = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
        v = np.clip(((1 - uv[:, 1]) % 1.0) * (h - 1), 0, h - 1).astype(int)
        return tex[v, u]
    except Exception:
        return np.full((len(mesh.vertices), 3), 180, np.uint8)


def render_frontal(body: trimesh.Trimesh, hf: HeadFrame, size: int = 600,
                   head_fraction: float = 0.34, bg: int = 255) -> HeadRender:
    """Frontal orthographic render of the head in the head frame.

    `head_fraction` is the share of head height (below the crown) kept in frame.
    """
    Rcols = np.stack([hf.right, hf.up, hf.forward], axis=1)
    Vloc = (body.vertices - hf.center) @ Rcols     # columns: right, up, forward
    world = np.asarray(body.vertices)
    colors = sample_vertex_colors(body)

    u = Vloc[:, 1]
    framelo = u.max() - head_fraction * (u.max() - u.min())
    keepv = u >= framelo
    faces = np.asarray(body.faces)[keepv[body.faces].all(1)]

    fr = Vloc[keepv][:, :2]
    mn, mx = fr.min(0), fr.max(0)
    span = (mx - mn).max() * 1.10
    c = (mn + mx) / 2

    sx = ((Vloc[:, 0] - c[0]) / span + 0.5) * (size - 1)
    sy = (0.5 - (Vloc[:, 1] - c[1]) / span) * (size - 1)
    depth = Vloc[:, 2]

    img = np.full((size, size, 3), bg, np.uint8)
    pos = np.zeros((size, size, 3))
    valid = np.zeros((size, size), bool)
    zbuf = np.full((size, size), -np.inf)

    for tri in faces:
        ia, ib, ic = tri
        xs = np.array([sx[ia], sx[ib], sx[ic]])
        ys = np.array([sy[ia], sy[ib], sy[ic]])
        x0, x1 = int(np.floor(xs.min())), int(np.ceil(xs.max()))
        y0, y1 = int(np.floor(ys.min())), int(np.ceil(ys.max()))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, size - 1), min(y1, size - 1)
        if x1 < x0 or y1 < y0:
            continue
        det = (ys[1]-ys[2])*(xs[0]-xs[2]) + (xs[2]-xs[1])*(ys[0]-ys[2])
        if abs(det) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1+1), np.arange(y0, y1+1))
        a = ((ys[1]-ys[2])*(gx-xs[2]) + (xs[2]-xs[1])*(gy-ys[2])) / det
        b = ((ys[2]-ys[0])*(gx-xs[2]) + (xs[0]-xs[2])*(gy-ys[2])) / det
        cc = 1 - a - b
        inside = (a >= -0.01) & (b >= -0.01) & (cc >= -0.01)
        if not inside.any():
            continue
        dd = a*depth[ia] + b*depth[ib] + cc*depth[ic]
        yy, xx = gy[inside], gx[inside]
        ai, bi, ci, di = a[inside], b[inside], cc[inside], dd[inside]
        m = di >= zbuf[yy, xx]
        yy, xx, ai, bi, ci = yy[m], xx[m], ai[m], bi[m], ci[m]
        zbuf[yy, xx] = di[m]
        col = colors[[ia, ib, ic]].mean(0).astype(np.uint8)
        img[yy, xx] = col
        pos[yy, xx] = (ai[:, None]*world[ia] + bi[:, None]*world[ib] + ci[:, None]*world[ic])
        valid[yy, xx] = True

    return HeadRender(image=np.ascontiguousarray(img), position=pos, valid=valid, size=size)
