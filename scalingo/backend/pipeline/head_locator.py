"""Step 2 (part A) — locate the head and its facing direction, geometrically.

No rendering, no MediaPipe: works for an API on arbitrary upright Meshy bodies.

Facing detection idea
---------------------
The nose is the single sharpest *protrusion* of the head. We fit a sphere to the
head vertices and look, within the eye/mouth height band, for the vertex with the
largest outward residual (how far it sticks out beyond the smooth head surface).
The back of the skull matches the sphere (≈0 residual); the nose does not. Its
horizontal direction gives the facing axis *and* sign automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import trimesh


HEAD_FRACTION = 0.15   # head occupies roughly the top 15 % of body height


@dataclass
class HeadFrame:
    center: np.ndarray      # head centre (3,)
    up: np.ndarray          # unit, body vertical
    forward: np.ndarray     # unit, gaze direction (horizontal)
    right: np.ndarray       # unit, ear-to-ear = up x forward
    width: float            # extent along right
    height: float           # extent along up (head only)
    depth: float            # extent along forward
    face_anchor: np.ndarray  # point on the front surface at mid height
    head_mask: np.ndarray   # bool mask over body vertices
    confidence: float       # 0..1 facing-detection confidence

    def to_local(self, pts: np.ndarray) -> np.ndarray:
        d = np.asarray(pts) - self.center
        return np.stack([d @ self.right, d @ self.up, d @ self.forward], axis=-1)


def _fit_sphere(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares sphere fit. Returns (center, radius)."""
    A = np.hstack([2 * pts, np.ones((len(pts), 1))])
    b = (pts ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = float(np.sqrt(max(sol[3] + c @ c, 1e-9)))
    return c, r


def _detect_neck_top(vertices: np.ndarray, up_axis: int, n_slices: int = 80) -> float:
    """Find the `up` coordinate that separates the head from the neck.

    Profile of horizontal width from the top down: the crown is narrow, widens to
    the cranium/cheeks (head max), narrows to the *neck* (local minimum), then
    widens again at the shoulders. We return the neck minimum, clamped to an
    anthropometric range (head ≈ 16–32 % of total height) as a safety net.
    """
    u = vertices[:, up_axis]
    lo, hi = u.min(), u.max()
    span = hi - lo
    horiz = [a for a in range(3) if a != up_axis]
    edges = np.linspace(lo, hi, n_slices + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.full(n_slices, np.nan)
    for i in range(n_slices):
        m = (u >= edges[i]) & (u < edges[i + 1])
        if m.sum() >= 5:
            h = vertices[m][:, horiz]
            widths[i] = np.linalg.norm(np.percentile(h, 95, 0) - np.percentile(h, 5, 0))

    # anthropometric window for the neck cut
    cut_hi = hi - 0.16 * span     # head no thinner than 16 % of height
    cut_lo = hi - 0.32 * span     # head no thicker than 32 % of height

    # head's widest slab within the top 20 %
    headtop = centers > hi - 0.20 * span
    if not np.any(headtop & ~np.isnan(widths)):
        return float(hi - 0.24 * span)
    i_head = np.nanargmax(np.where(headtop, widths, np.nan))

    # scan downward from the head max for the first local minimum (the neck)
    neck_y = hi - 0.24 * span
    for i in range(i_head - 1, 0, -1):
        if np.isnan(widths[i]):
            continue
        left = widths[i - 1] if not np.isnan(widths[i - 1]) else widths[i]
        right = widths[i + 1] if not np.isnan(widths[i + 1]) else widths[i]
        if widths[i] <= left and widths[i] <= right:
            neck_y = centers[i]
            break
        if centers[i] < cut_lo:        # reached the clamp without a clear valley
            neck_y = centers[i]
            break
    return float(np.clip(neck_y, cut_lo, cut_hi))


def locate_head(body: trimesh.Trimesh) -> HeadFrame:
    V = np.asarray(body.vertices, float)

    # --- vertical axis = tallest bbox dimension (Meshy avatars stand on Y) ---
    up_axis = int(np.argmax(V.max(0) - V.min(0)))
    up = np.zeros(3); up[up_axis] = 1.0

    # --- head band ---
    # A fixed anthropometric fraction of the height (head ≈ top ~15 %) is far more
    # robust on stylised/flat Meshy bodies than width-valley neck detection, which
    # easily latches onto the crown or the shoulders. The neck detector is kept
    # available (refinement / debugging) but not used for the primary band.
    u = V[:, up_axis]
    span = u.max() - u.min()
    head_mask = u >= u.max() - HEAD_FRACTION * span
    H = V[head_mask]
    center = H.mean(0)

    # --- facing direction via nose protrusion (sphere residual) ---
    s_center, s_radius = _fit_sphere(H)
    radial = np.linalg.norm(H - s_center, axis=1)
    residual = radial - s_radius            # nose sticks out -> large positive
    out_dir = (H - s_center) / radial[:, None]

    # restrict to the eye/mouth band: mid-height of the head
    hu = H[:, up_axis]
    lo, hi = hu.min(), hu.max()
    band = (hu >= lo + 0.30 * (hi - lo)) & (hu <= lo + 0.68 * (hi - lo))

    res_band = residual.copy()
    res_band[~band] = -np.inf
    k = max(5, int(0.01 * len(H)))
    top = np.argpartition(res_band, -k)[-k:]
    w = np.clip(residual[top], 0, None) + 1e-6
    fwd = (out_dir[top] * w[:, None]).sum(0)
    fwd[up_axis] = 0.0                       # keep horizontal
    fwd /= np.linalg.norm(fwd) + 1e-12

    # confidence: how peaked the nose residual is vs the head's typical bumpiness
    confidence = float(np.clip(
        (residual[top].mean()) / (np.abs(residual[band]).mean() + 1e-6) / 5.0, 0, 1))

    # --- orthonormal head frame ---
    right = np.cross(up, fwd); right /= np.linalg.norm(right) + 1e-12
    forward = np.cross(right, up); forward /= np.linalg.norm(forward) + 1e-12

    # --- head metrics in local frame ---
    loc = (H - center) @ np.stack([right, up, forward], axis=1)
    width = float(np.ptp(loc[:, 0]))
    height = float(np.ptp(loc[:, 1]))
    depth = float(np.ptp(loc[:, 2]))

    # --- face anchor: front surface at mid height, near the symmetry plane ---
    central = (np.abs(loc[:, 0]) < 0.20 * width) & \
              (np.abs(loc[:, 1] - loc[:, 1].mean()) < 0.20 * height)
    if central.sum() < 5:
        central = np.abs(loc[:, 0]) < 0.30 * width
    front_depth = np.percentile(loc[central, 2], 98)
    mid_up = center @ up
    face_anchor = center + forward * front_depth
    face_anchor = face_anchor - up * (face_anchor @ up) + up * mid_up

    return HeadFrame(center, up, forward, right, width, height, depth,
                     face_anchor, head_mask, confidence)


def reframe(body: trimesh.Trimesh, hf: HeadFrame, new_forward: np.ndarray) -> HeadFrame:
    """Rebuild the head frame around a corrected `forward` (kept horizontal), reusing
    the same head band / centre / up and recomputing the local metrics and face
    anchor. Used to close the loop with the MediaPipe eye axis when the geometric
    facing estimate is wrong (e.g. voluminous/asymmetric hair biases the nose
    protrusion). `confidence` is left as-is."""
    V = np.asarray(body.vertices, float)
    up = hf.up / (np.linalg.norm(hf.up) + 1e-12)
    f = np.asarray(new_forward, float)
    f = f - (f @ up) * up
    n = np.linalg.norm(f)
    if n < 1e-9:
        return hf
    forward = f / n
    right = np.cross(up, forward); right /= np.linalg.norm(right) + 1e-12
    forward = np.cross(right, up); forward /= np.linalg.norm(forward) + 1e-12

    center = hf.center
    H = V[hf.head_mask]
    loc = (H - center) @ np.stack([right, up, forward], axis=1)
    width = float(np.ptp(loc[:, 0]))
    height = float(np.ptp(loc[:, 1]))
    depth = float(np.ptp(loc[:, 2]))

    central = (np.abs(loc[:, 0]) < 0.20 * width) & \
              (np.abs(loc[:, 1] - loc[:, 1].mean()) < 0.20 * height)
    if central.sum() < 5:
        central = np.abs(loc[:, 0]) < 0.30 * width
    front_depth = np.percentile(loc[central, 2], 98)
    mid_up = center @ up
    face_anchor = center + forward * front_depth
    face_anchor = face_anchor - up * (face_anchor @ up) + up * mid_up

    return replace(hf, forward=forward, right=right, width=width, height=height,
                   depth=depth, face_anchor=face_anchor)
