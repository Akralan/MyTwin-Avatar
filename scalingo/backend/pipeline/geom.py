"""Low-level geometry helpers: boundaries, angular rings, resampling.

Pure numpy / trimesh — no heavy deps.
"""
from __future__ import annotations

import numpy as np
import trimesh


def boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """(E, 2) vertex-index pairs on an open boundary (edges used by one face)."""
    edges = np.sort(mesh.edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return uniq[counts == 1]


def boundary_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """Unique vertex indices that touch an open boundary edge."""
    return np.unique(boundary_edges(mesh).reshape(-1))


def angular_ring(points2d: np.ndarray, indices: np.ndarray, n_bins: int = 96,
                 target_rho=None) -> np.ndarray:
    """Order a noisy set of boundary points into a single clean ring by angle.

    Robust to messy topology (many tiny loops): instead of walking edges we sort
    candidate boundary vertices by their polar angle around a centre and keep, per
    angular bin, the one closest to the desired radius.

    points2d : (K, 2) candidate positions, centred on the ring centre.
    indices  : (K,) original vertex indices matching points2d.
    target_rho : desired radius (scalar or per-candidate array); defaults to 1.0.
    Returns ordered vertex indices forming the ring.
    """
    ang = np.arctan2(points2d[:, 1], points2d[:, 0])
    rho = np.linalg.norm(points2d, axis=1)
    bins = ((ang + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    tgt = 1.0 if target_rho is None else target_rho
    tgt_arr = np.full(len(rho), tgt) if np.ndim(tgt) == 0 else np.asarray(tgt)
    chosen = []
    for b in range(n_bins):
        cand = np.where(bins == b)[0]
        if cand.size == 0:
            continue
        best = cand[np.argmin(np.abs(rho[cand] - tgt_arr[cand]))]
        chosen.append((ang[best], indices[best]))
    chosen.sort(key=lambda t: t[0])
    return np.array([i for _, i in chosen], dtype=np.int64)


def point_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorised crossing-number test. points (N,2), polygon (M,2) closed implicitly.
    Returns bool (N,)."""
    pts = np.asarray(points, float)
    poly = np.asarray(polygon, float)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


def polygon_inflate(polygon: np.ndarray, factor: float) -> np.ndarray:
    """Scale a polygon about its centroid (factor>1 grows, <1 shrinks)."""
    poly = np.asarray(polygon, float)
    c = poly.mean(0)
    return c + (poly - c) * factor


def resample_closed(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polyline to `n` points by arc length (linear)."""
    pts = np.asarray(points, float)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    targets = np.linspace(0, total, n, endpoint=False)
    out = np.empty((n, pts.shape[1]))
    for i, t in enumerate(targets):
        k = min(max(np.searchsorted(cum, t, side="right") - 1, 0), len(seg) - 1)
        f = (t - cum[k]) / seg[k] if seg[k] > 1e-12 else 0.0
        out[i] = closed[k] * (1 - f) + closed[k + 1] * f
    return out
