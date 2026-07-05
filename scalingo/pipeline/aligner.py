"""Step 4 — align the floating face mesh into the body's hole.

The face scan is in MediaPipe canonical space (+Z forward, +Y up, normalised).
We do NOT solve a free 3D rotation (an irregular Meshy hole rim makes a free
Umeyama fit tilt the face). Instead the orientation is *fixed* to the detected
head frame (right / up / forward) and we only solve a uniform scale and a
translation. The translation is anchored on meaningful landmarks:

  * horizontally : the face's symmetry centre -> the hole centre,
  * vertically   : the face's chin -> the hole's bottom (the body jaw), which
                   removes the "two chins" artefact,
  * in depth     : the silhouette plane -> the hole rim depth, so the face sits
                   proud of the skull (the nose naturally protrudes).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .head_locator import HeadFrame


@dataclass
class AlignResult:
    face: trimesh.Trimesh        # transformed face mesh (visuals preserved)
    face_ring: np.ndarray        # ordered face boundary vertex indices
    transform: np.ndarray        # 4x4 applied to the face


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity transform mapping src -> dst (Umeyama 1991).

    Returns (s, R, t) with dst ≈ s * R @ src + t.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    cov = (Dc.T @ Sc) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (Sc ** 2).sum() / len(src)
    s = (D * np.diag(S)).sum() / var_s if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


# MediaPipe canonical indices that are stable and well-localised across faces
# (eyes, brows, nose, lips, face oval) — used to weight the landmark fit. The iris
# points (468-477) are excluded as they are noisy on a rendered avatar.
_STABLE_MAX = 468


def align_by_landmarks(face: trimesh.Trimesh, body_landmarks: np.ndarray,
                       valid: np.ndarray) -> AlignResult:
    """Intelligent placement: feature-to-feature.

    Both the scan and the detected body landmarks are in MediaPipe's 478-vertex
    canonical order, so vertex i of the scan corresponds to body landmark i. We fit
    a similarity transform over the stable (non-iris) valid landmarks.
    """
    idx = np.array([i for i in range(min(_STABLE_MAX, len(face.vertices)))
                    if valid[i]])
    if len(idx) < 50:
        raise ValueError(f"not enough corresponding landmarks ({len(idx)})")

    s, R, t = umeyama(face.vertices[idx], body_landmarks[idx], with_scale=True)
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t

    face = face.copy()
    face.apply_transform(T)
    f_ring = face_silhouette_ring(face)
    return AlignResult(face=face, face_ring=f_ring, transform=T)


def face_silhouette_ring(face: trimesh.Trimesh, n_bins: int = 96) -> np.ndarray:
    """Ordered boundary vertices of the face mesh (its outer oval), by angle in
    the face's own (x, y) plane."""
    from . import geom

    bverts = geom.boundary_vertices(face)
    P = face.vertices[bverts]
    c = P.mean(0)
    xy = P[:, :2] - c[:2]
    rho = np.linalg.norm(xy, axis=1)
    rho /= (np.median(rho) + 1e-9)
    return geom.angular_ring(xy, bverts, n_bins=n_bins, target_rho=rho.max())


def align_face(face: trimesh.Trimesh, hf: HeadFrame, hole_ring_pts: np.ndarray,
               proud_margin: float = 0.05) -> AlignResult:
    """Fixed-orientation similarity fit (no free rotation -> no tilt).

    `proud_margin` pushes the face forward by that fraction of its scaled depth so
    it never sinks into the skull even if the rim is shallow.
    """
    face = face.copy()
    f_ring = face_silhouette_ring(face)

    # Fixed head orientation: face axes (x,y,z) map to (right, up, forward).
    Rcols = np.stack([hf.right, hf.up, hf.forward], axis=1)

    sil = face.vertices[f_ring]                 # silhouette in face-space == local
    hole = hf.to_local(hole_ring_pts)           # hole rim in head-local (r,u,f)

    face_w, face_h = np.ptp(sil[:, 0]), np.ptp(sil[:, 1])
    hole_w, hole_h = np.ptp(hole[:, 0]), np.ptp(hole[:, 1])
    s = 0.5 * (hole_w / face_w + hole_h / face_h)   # uniform scale, no distortion

    # translation in the head-local frame
    tx = hole[:, 0].mean() - s * sil[:, 0].mean()          # centre horizontally
    ty = hole[:, 1].min() - s * sil[:, 1].min()            # chin -> jaw bottom
    tz = hole[:, 2].mean() - s * sil[:, 2].mean()          # silhouette -> rim depth
    tz += proud_margin * s * np.ptp(face.vertices[:, 2])   # push proud of the skull
    t_local = np.array([tx, ty, tz])

    # world = center + Rcols @ (s * p_face + t_local)
    T = np.eye(4)
    T[:3, :3] = s * Rcols
    T[:3, 3] = hf.center + Rcols @ t_local
    face.apply_transform(T)
    return AlignResult(face=face, face_ring=f_ring, transform=T)
