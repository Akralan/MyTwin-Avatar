"""Step 5 — stitch the aligned face onto the body hole.

Because we keep two materials (face selfie texture vs body atlas), we do NOT build
shared cross-texture triangles. Instead we make the two surfaces meet: the face's
outer silhouette ring is snapped onto the body's hole rim, and that boundary
displacement is diffused smoothly into the face interior with a harmonic
(Laplacian) deformation so no crease appears. The result is a flush, gap-free
junction with each side keeping its own texture.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def place_overlap(face: trimesh.Trimesh, hf, eps_frac: float = 0.015) -> trimesh.Trimesh:
    """Intelligent-path placement: no edge stitching.

    The landmark alignment already sits the scan on the body's face surface and the
    hole is cut smaller than the scan, so the scan overlaps the body all around. We
    only nudge it slightly *proud* (along the head's forward axis) so it stays in
    front of the body in the overlap band — avoiding z-fighting without any boundary
    deformation (which is what produced the dark sliver triangles).
    """
    face = face.copy()
    face.apply_translation(hf.forward * eps_frac * hf.depth)
    return face


def _nearest_on_polyline(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Closest point to p on a closed polyline `poly` (M,3)."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.einsum("ij,ij->i", p - a, ab) / (np.einsum("ij,ij->i", ab, ab) + 1e-12)
    t = np.clip(t, 0, 1)
    proj = a + t[:, None] * ab
    d = np.linalg.norm(proj - p, axis=1)
    return proj[np.argmin(d)]


def _uniform_laplacian(n: int, edges: np.ndarray) -> csr_matrix:
    L = lil_matrix((n, n))
    deg = np.zeros(n)
    for a, b in edges:
        L[a, b] -= 1.0
        L[b, a] -= 1.0
        deg[a] += 1.0
        deg[b] += 1.0
    L.setdiag(deg)
    return L.tocsr()


def stitch(face: trimesh.Trimesh, face_ring: np.ndarray,
           hole_ring_pts: np.ndarray, relax_rings: int = 2) -> trimesh.Trimesh:
    """Return a deformed copy of `face` whose outer ring sits on the hole rim.

    `relax_rings` is kept for API symmetry; the harmonic solve already diffuses
    the displacement across the whole mesh.
    """
    face = face.copy()
    V = np.asarray(face.vertices, float).copy()
    n = len(V)

    # 1. snap outer ring vertices onto the body hole rim
    targets = {}
    for vi in face_ring:
        targets[int(vi)] = _nearest_on_polyline(V[vi], hole_ring_pts)

    # 2. harmonic diffusion of the boundary displacement
    edges = face.edges_unique
    L = _uniform_laplacian(n, edges)

    boundary_idx = np.array(sorted(targets.keys()))
    disp_b = np.array([targets[i] - V[i] for i in boundary_idx])

    is_b = np.zeros(n, bool)
    is_b[boundary_idx] = True
    interior_idx = np.where(~is_b)[0]

    # solve L_II u_I = -L_IB d_B  (per coordinate)
    # +eps*I regularises interior components disconnected from the boundary
    # (e.g. the iris islands of a MediaPipe face mesh), pinning them near zero
    # displacement instead of leaving the system singular.
    from scipy.sparse import identity
    L_II = L[interior_idx][:, interior_idx]
    L_II = L_II + 1e-6 * identity(L_II.shape[0])
    L_IB = L[interior_idx][:, boundary_idx]
    rhs = -L_IB @ disp_b
    u_I = spsolve(L_II.tocsc(), rhs)
    if u_I.ndim == 1:
        u_I = u_I.reshape(-1, 1)

    U = np.zeros((n, 3))
    U[boundary_idx] = disp_b
    U[interior_idx] = u_I
    face.vertices = V + U
    return face
