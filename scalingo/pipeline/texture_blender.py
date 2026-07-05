"""Step 6 — match the scanned face's skin tone to the body (keeps 2 materials).

The selfie texture is usually cooler/darker than the Meshy body skin, so the
inserted face reads as a different colour. We apply a Reinhard transfer (match mean
and standard deviation in CIE-Lab) computed on *skin pixels only* of both sides, so
the face's overall tone and contrast match the body while its identity detail
(features, moustache, shading variation) is preserved.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .render_head import sample_vertex_colors


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Heuristic skin filter on (N,3) uint8 colours: warm, mid-luminance.
    Excludes hair/eyes/brows/lips/moustache (dark or strongly saturated)."""
    r, g, b = rgb[:, 0].astype(int), rgb[:, 1].astype(int), rgb[:, 2].astype(int)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (r >= g) & (g >= b) & (r - b > 8) & (lum > 60) & (lum < 238)


def _lab_stats(rgb: np.ndarray):
    """Mean/std per Lab channel for (N,3) uint8 RGB samples."""
    import cv2
    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab = lab.reshape(-1, 3).astype(np.float32)
    return lab.mean(0), lab.std(0) + 1e-6


def _body_skin_colors(body: trimesh.Trimesh, hf) -> np.ndarray:
    """Body per-vertex colours over the front of the head, skin-filtered."""
    cols = sample_vertex_colors(body)
    loc = hf.to_local(body.vertices)
    head = (loc[:, 1] > -0.1 * hf.height) & (loc[:, 2] > 0.0)  # upper-front of head
    sel = cols[head]
    m = _skin_mask(sel)
    return sel[m] if m.sum() > 50 else sel


def _face_skin_colors(face: trimesh.Trimesh) -> np.ndarray:
    cols = sample_vertex_colors(face)
    m = _skin_mask(cols)
    return cols[m] if m.sum() > 50 else cols


def match_skin_tone(face: trimesh.Trimesh, body: trimesh.Trimesh, hf,
                    std_clip=(0.6, 1.6), strength: float = 1.0) -> trimesh.Trimesh:
    """Return a copy of `face` whose base-colour texture is tone-matched to `body`.

    `strength` in [0,1] blends between the original and the fully matched tone.
    """
    import cv2

    mat = getattr(face.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) if mat is not None else None
    if img is None:
        return face

    body_mean, body_std = _lab_stats(_body_skin_colors(body, hf))
    face_mean, face_std = _lab_stats(_face_skin_colors(face))

    ratio = np.clip(body_std / face_std, *std_clip)
    gain = strength * ratio + (1 - strength)
    shift = strength * body_mean + (1 - strength) * face_mean

    rgba = np.array(img.convert("RGBA"))   # np.array -> writable copy
    rgb = rgba[..., :3]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = (lab - face_mean) * gain + shift
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    rgba[..., :3] = out
    from PIL import Image
    new_img = Image.fromarray(rgba, mode="RGBA")

    face = face.copy()
    face.visual.material.baseColorTexture = new_img
    return face


def _border_weights(face: trimesh.Trimesh, n_rings: int = 5) -> np.ndarray:
    """Per-vertex feather weight: 1 on the silhouette boundary, fading to 0 over
    `n_rings` edge hops inward (BFS)."""
    from collections import defaultdict, deque
    from . import geom

    n = len(face.vertices)
    nb = defaultdict(set)
    for a, b in face.edges_unique:
        nb[int(a)].add(int(b))
        nb[int(b)].add(int(a))

    dist = np.full(n, np.inf)
    dq = deque()
    for v in geom.boundary_vertices(face):
        dist[v] = 0.0
        dq.append(int(v))
    while dq:
        u = dq.popleft()
        for w in nb[u]:
            if dist[w] > dist[u] + 1:
                dist[w] = dist[u] + 1
                dq.append(w)
    return np.clip(1.0 - dist / n_rings, 0.0, 1.0)


def _rasterize_uv(uv: np.ndarray, faces: np.ndarray, vert_rgb: np.ndarray,
                  vert_w: np.ndarray, W: int, H: int):
    """Rasterise per-vertex RGB target + weight into the UV layout.

    Returns (color (H,W,3) float, weight (H,W) float). Where triangles overlap a
    pixel, the highest-weight sample wins.
    """
    px = np.clip(uv[:, 0], 0, 1) * (W - 1)
    py = np.clip(1 - uv[:, 1], 0, 1) * (H - 1)
    color = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)
    for tri in faces:
        ia, ib, ic = tri
        xs = np.array([px[ia], px[ib], px[ic]])
        ys = np.array([py[ia], py[ib], py[ic]])
        x0, x1 = int(np.floor(xs.min())), int(np.ceil(xs.max()))
        y0, y1 = int(np.floor(ys.min())), int(np.ceil(ys.max()))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, W - 1), min(y1, H - 1)
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
        wv = a*vert_w[ia] + b*vert_w[ib] + cc*vert_w[ic]
        cv_ = (a[..., None]*vert_rgb[ia] + b[..., None]*vert_rgb[ib]
               + cc[..., None]*vert_rgb[ic])
        yy, xx = gy[inside], gx[inside]
        wv, cv_ = wv[inside], cv_[inside]
        better = wv > alpha[yy, xx]
        yy, xx, wv, cv_ = yy[better], xx[better], wv[better], cv_[better]
        alpha[yy, xx] = wv
        color[yy, xx] = cv_
    return color, alpha


def feather_border(face: trimesh.Trimesh, body: trimesh.Trimesh, hf,
                   n_rings: int = 5, strength: float = 1.0) -> trimesh.Trimesh:
    """Fade the scan texture toward the *local* body skin colour along the silhouette.

    Rather than a single average tone (which leaves a visible halo where the body
    skin varies), each scan vertex targets the nearest body skin vertex's colour, so
    the feather matches the body exactly at every point of the boundary.
    """
    from scipy.spatial import cKDTree
    from PIL import Image

    mat = getattr(face.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) if mat is not None else None
    if img is None or getattr(face.visual, "uv", None) is None:
        return face

    # nearest body *skin* colour for every scan vertex (placed in world space)
    bcols = sample_vertex_colors(body)
    skin = _skin_mask(bcols)
    tree = cKDTree(body.vertices[skin])
    _, idx = tree.query(face.vertices)
    target = bcols[skin][idx].astype(np.float32)

    weight = _border_weights(face, n_rings=n_rings) * strength
    rgba = np.array(img.convert("RGBA"))
    H, W = rgba.shape[:2]
    tcol, alpha = _rasterize_uv(np.asarray(face.visual.uv), np.asarray(face.faces),
                                target, weight, W, H)

    a3 = alpha[..., None]
    rgba[..., :3] = (rgba[..., :3] * (1 - a3) + tcol * a3).astype(np.uint8)

    face = face.copy()
    face.visual.material.baseColorTexture = Image.fromarray(rgba, mode="RGBA")
    return face


def harmonic_tone_match(face: trimesh.Trimesh, body: trimesh.Trimesh,
                        outer_ring: np.ndarray, strength: float = 1.0,
                        clip_k: float = 3.0) -> trimesh.Trimesh:
    """Spatially-varying tone correction (gradient-domain / Poisson membrane).

    `match_skin_tone` applies ONE global Lab shift, so it can only match the *average*
    body skin tone. Where the body skin varies spatially — the forehead is lighter,
    one cheek is lit more than the other — a single shift leaves a local tint step
    just inside the silhouette (the classic forehead / one-cheek mismatch).

    Here we correct that with a smooth, boundary-driven offset field: along the OUTER
    silhouette we measure the residual Lab offset needed to hit the *local* body skin
    tone (nearest body-skin vertex, as in `feather_border`), then diffuse it across the
    mask by solving a Laplace (membrane) problem — offset pinned on the ring, harmonic
    inside. The result matches the body locally all around the perimeter (both cheeks,
    forehead, chin) and blends smoothly to the interior. Because only a LOW-FREQUENCY
    offset is added, the face's detail and the added grain are preserved.

    `outer_ring` must be the OUTER oval only (aligner.face_silhouette_ring); the eye and
    mouth loops stay free interior vertices so they are never pinned to skin colour.
    Runs after `match_skin_tone`, before `feather_border`. No-op without OpenCV.
    """
    from PIL import Image
    from scipy.sparse import identity
    from scipy.sparse.linalg import spsolve
    from scipy.spatial import cKDTree
    from .junction import _cotangent_laplacian

    try:
        import cv2
    except Exception:  # noqa: BLE001
        return face

    mat = getattr(face.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) if mat is not None else None
    if img is None or getattr(face.visual, "uv", None) is None:
        return face

    V = np.asarray(face.vertices, float)
    F = np.asarray(face.faces)
    n = len(V)
    ring = np.unique(np.asarray(outer_ring, dtype=int))
    ring = ring[(ring >= 0) & (ring < n)]
    if len(ring) < 8 or len(ring) >= n:
        return face

    # current per-vertex face tone and nearest body-skin tone
    bcols = sample_vertex_colors(body)
    skin = _skin_mask(bcols)
    if skin.sum() < 50:
        return face
    tree = cKDTree(body.vertices[skin])
    _, idx = tree.query(V)

    def to_lab(rgb):
        lab = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)
        return lab.reshape(-1, 3).astype(np.float32)

    face_lab = to_lab(sample_vertex_colors(face))
    body_lab = to_lab(bcols[skin][idx])

    # residual offset to the LOCAL body tone, measured on the outer ring only
    off_ring = body_lab[ring] - face_lab[ring]
    # robustify: clamp per-channel outliers (dark background/hair bleed on edge texels)
    med = np.median(off_ring, axis=0)
    mad = np.median(np.abs(off_ring - med), axis=0) + 1e-6
    lo, hi = med - clip_k * 1.4826 * mad, med + clip_k * 1.4826 * mad
    off_ring = np.clip(off_ring, lo, hi)

    # membrane: L x = 0 on the interior, x = off_ring on the ring (per Lab channel)
    L = _cotangent_laplacian(V, F).tocsr()
    interior = np.ones(n, bool)
    interior[ring] = False
    Lii = (L[interior][:, interior] + 1e-8 * identity(int(interior.sum()))).tocsc()
    Lib = L[interior][:, ring]
    xi = spsolve(Lii, -(Lib @ off_ring))
    if xi.ndim == 1:
        xi = xi.reshape(-1, 1)

    offset = np.zeros((n, 3), np.float32)
    offset[ring] = off_ring
    offset[interior] = xi
    offset *= strength

    # rasterise the smooth offset into the atlas and add it in Lab
    rgba = np.array(img.convert("RGBA"))
    H, W = rgba.shape[:2]
    ocol, alpha = _rasterize_uv(np.asarray(face.visual.uv), F, offset,
                                np.ones(n, np.float32), W, H)
    lab = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    m = alpha > 0
    lab[m] += ocol[m]
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    rgba[..., :3] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    face = face.copy()
    face.visual.material.baseColorTexture = Image.fromarray(rgba, mode="RGBA")
    return face
