"""Prototype — texture-only face swap (no geometry surgery).

Instead of cutting a hole in the body and inserting the scan mesh, we keep the
body mesh *intact* and only repaint its base-colour atlas over the facial region
with the scanned face's real texture.

Pipeline
--------
1. Align the scan to the body by MediaPipe landmarks (same fit as the landmark
   placement path) -> a 4x4 transform `T` and the scan placed in body space.
2. Select the body atlas region to repaint = body faces whose centroid falls
   inside the aligned scan silhouette, on the front hemisphere.
3. Rasterise those faces into the body's UV layout; for every covered texel get
   its interpolated 3D body point P.
4. For each P, find the closest point on the aligned scan surface, read its UV,
   and sample the scan texture -> the colour to paint at that texel.
5. Composite into the body atlas with a feathered alpha that is 1 in the interior
   (the scan's real tone is preserved verbatim — NO tone matching) and fades to 0
   over a thin boundary band so the seam is invisible.

Output: the body Trimesh with a new base-colour texture. Single mesh, single
material, geometry untouched -> no 3D junction, the old C1-normal defect is gone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from PIL import Image

from .head_locator import HeadFrame
from . import landmark_detector, aligner, geom, texture_blender


@dataclass
class SwapReport:
    landmark_residual: float
    region_faces: int
    painted_texels: int
    median_surface_dist: float
    atlas_size: tuple


def _trim_boundary(mesh: trimesh.Trimesh, n_rings: int = 1) -> trimesh.Trimesh:
    """Remove `n_rings` of boundary faces from a mesh (visuals/UVs preserved).

    The scan's outermost ring is where the canonical UVs sit on the face/background
    edge of the raw selfie, so it carries dark background/hair/shadow bleed. Cutting
    that thin rim removes the dark side tints without touching interior features
    (eyebrows, eyes, moustache sit many rings inward)."""
    m = mesh.copy()
    for _ in range(n_rings):
        bv = set(int(i) for i in geom.boundary_vertices(m))
        if not bv:
            break
        faces = np.asarray(m.faces)
        touches = np.array([(a in bv) or (b in bv) or (c in bv)
                            for a, b, c in faces])
        if not touches.any():
            break
        m.update_faces(~touches)
        m.remove_unreferenced_vertices()
    return m


def _silhouette_region(body: trimesh.Trimesh, hf: HeadFrame,
                       sil_world: np.ndarray, erode: float = 1.0) -> np.ndarray:
    """Boolean mask over body faces inside the aligned scan silhouette (front)."""
    poly = hf.to_local(sil_world)[:, :2]
    c = poly.mean(0)
    poly = c + (poly - c) * erode
    tri_c = body.vertices[body.faces].mean(axis=1)
    loc = hf.to_local(tri_c)
    inside = geom.point_in_polygon(loc[:, :2], poly)
    in_front = loc[:, 2] > 0.0
    return inside & in_front


def _rasterize_positions(uv: np.ndarray, faces: np.ndarray, vert_pos: np.ndarray,
                         W: int, H: int):
    """Rasterise the given faces into the UV atlas, outputting the interpolated 3D
    position at each covered texel. Returns (pos (H,W,3) float, mask (H,W) bool)."""
    px = np.clip(uv[:, 0] % 1.0, 0, 1) * (W - 1)
    py = np.clip(1 - (uv[:, 1] % 1.0), 0, 1) * (H - 1)
    pos = np.zeros((H, W, 3), np.float32)
    mask = np.zeros((H, W), bool)
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
        yy, xx = gy[inside], gx[inside]
        ai, bi, ci = a[inside], b[inside], cc[inside]
        pos[yy, xx] = (ai[:, None]*vert_pos[ia] + bi[:, None]*vert_pos[ib]
                       + ci[:, None]*vert_pos[ic])
        mask[yy, xx] = True
    return pos, mask


def _dist_to_polyline(pts2d: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Min Euclidean distance from each (N,2) point to a closed polygon's edges."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a                                   # (E,2)
    ab2 = (ab ** 2).sum(1) + 1e-12
    ap = pts2d[:, None, :] - a[None, :, :]       # (N,E,2)
    t = np.clip((ap * ab[None]).sum(2) / ab2[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]     # (N,E,2)
    d = np.linalg.norm(pts2d[:, None, :] - proj, axis=2)
    return d.min(1)


def _sample_texture(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest-texel sample of an (H,W,3) image at (N,2) UVs (v flipped)."""
    H, W = img.shape[:2]
    u = np.clip(uv[:, 0] % 1.0, 0, 1) * (W - 1)
    v = np.clip(1 - (uv[:, 1] % 1.0), 0, 1) * (H - 1)
    return img[np.round(v).astype(int), np.round(u).astype(int)]


def _base_color_image(mesh: trimesh.Trimesh) -> np.ndarray:
    mat = mesh.visual.material
    return np.array(mat.baseColorTexture.convert("RGB"))


def swap_face_texture(body: trimesh.Trimesh, face: trimesh.Trimesh, hf: HeadFrame,
                      feather_frac: float = 0.0, dist_cutoff_frac: float = 0.10,
                      match_tone: bool = True, trim_rings: int = 0,
                      dark_thr: float = 70.0, dark_edge_frac: float = 0.0):
    """Repaint the body atlas with the scan's face texture. Returns (body2, report).

    `trim_rings`        number of boundary face-rings to cut off the scan before
                        baking, to drop the dark background/shadow bleed that sits
                        on the silhouette UVs of the raw selfie. Interior features
                        are untouched. 0 disables.
    `match_tone`        tone-match the scan texture to the body skin (Reinhard in
                        Lab on skin pixels) BEFORE baking, so the painted region's
                        overall tone/lighting matches the surrounding body and the
                        silhouette edge stops reading as a colour jump. Detail and
                        identity are preserved; only mean/std shift.

    `feather_frac`      width of the alpha fade band at the silhouette, as a
                        fraction of the head width. **0 = hard replace** (default):
                        every covered texel becomes 100% scan texture, NO blending
                        with the body skin -> the scan keeps its own grain, the body
                        grain never shows through. >0 softens only the silhouette
                        edge (computed in 3D so the fragmented Meshy UV islands don't
                        wash out the interior).
    `dist_cutoff_frac`  texels whose 3D point is farther than this fraction of the
                        head depth from the scan surface keep the body texture
                        (guards against painting onto ears / far geometry).
    `dark_thr`          luminance (0-255) below which a sampled scan texel counts as
                        "dark" (background / hair / shadow bleed from the selfie).
    `dark_edge_frac`    a dark texel is rejected only if it lies within this fraction
                        of the head width from the silhouette, so interior dark
                        features (eyebrows, eyes, moustache) are preserved.
    """
    # --- 1. align scan to body (landmark fit) ---
    body3d, valid = landmark_detector.detect_body_landmarks(body, hf)
    al = aligner.align_by_landmarks(face, body3d, valid)
    face_a = al.face                                   # scan placed in body space

    # tone-match the scan texture to the body skin BEFORE sampling it into the atlas
    if match_tone:
        try:
            face_a = texture_blender.match_skin_tone(face_a, body, hf)
        except Exception:
            pass
    idx = [i for i in range(468) if valid[i]]
    pred = (al.transform[:3, :3] @ face.vertices[idx].T).T + al.transform[:3, 3]
    residual = float(np.linalg.norm(pred - body3d[idx], axis=1).mean())

    # cut the contaminated outer ring(s) of the scan, then recompute its silhouette
    if trim_rings > 0:
        face_a = _trim_boundary(face_a, trim_rings)

    # --- 2. body atlas region to repaint ---
    ring = aligner.face_silhouette_ring(face_a)
    sil = face_a.vertices[ring]
    region = _silhouette_region(body, hf, sil)

    body = body.copy()
    b_img = _base_color_image(body)
    H, W = b_img.shape[:2]
    b_uv = np.asarray(body.visual.uv)
    region_faces = np.asarray(body.faces)[region]

    # --- 3. rasterise region into the atlas, get per-texel 3D body point ---
    pos, mask = _rasterize_positions(b_uv, region_faces, np.asarray(body.vertices),
                                     W, H)
    ys, xs = np.where(mask)
    pts = pos[ys, xs]

    # --- 4. nearest scan-surface point -> scan UV -> scan texture colour ---
    # closest_point_naive (brute force) avoids the rtree dependency; the scan is
    # tiny (~900 faces), so chunking points keeps the N x F memory bounded.
    closest = np.empty_like(pts)
    dist = np.empty(len(pts))
    tri_id = np.empty(len(pts), dtype=np.int64)
    for s in range(0, len(pts), 4000):
        e = s + 4000
        c, d, t = trimesh.proximity.closest_point_naive(face_a, pts[s:e])
        closest[s:e], dist[s:e], tri_id[s:e] = c, d, t
    bary = trimesh.triangles.points_to_barycentric(
        face_a.triangles[tri_id], closest)
    f_faces = np.asarray(face_a.faces)
    f_uv = np.asarray(face_a.visual.uv)
    uv_hit = (bary[:, :, None] * f_uv[f_faces[tri_id]]).sum(axis=1)
    f_img = _base_color_image(face_a)
    colors = _sample_texture(f_img, uv_hit)

    # distance to the silhouette in the head plane, for every painted texel (drives
    # both the dark-edge rejection and the feather; computed once).
    poly2d = hf.to_local(sil)[:, :2]
    edge_dist = _dist_to_polyline(hf.to_local(pts)[:, :2], poly2d)

    # distance cutoff: drop texels too far from the scan surface
    keep = dist <= dist_cutoff_frac * hf.depth

    # --- dark-edge rejection ---
    # The scan texture is the raw selfie: its facial silhouette bleeds into the dark
    # room background / hair / temple shadow, leaving black tints on the mask sides.
    # Drop texels that are BOTH dark AND near the silhouette; interior dark features
    # (eyebrows, eyes, nostrils, moustache) sit far from the edge and are kept.
    if dark_edge_frac > 0:
        lum = colors.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        near_edge = edge_dist < dark_edge_frac * hf.width
        keep &= ~((lum < dark_thr) & near_edge)

    # --- 5. feathered alpha in 3D (interior = 1, fade to 0 at the silhouette) ---
    # Atlas-space feathering fails on the fragmented Meshy UV: every micro-island
    # edge would pull alpha down. Instead the feather depth is the texel's distance
    # to the silhouette measured in the head plane -> independent of UV islands.
    if feather_frac <= 0:
        # hard replace: 100% scan texture on every covered texel, no body bleed-through
        alpha = np.ones(len(pts), np.float32)
    else:
        alpha = np.clip(edge_dist / (feather_frac * hf.width), 0.0, 1.0)

    out = b_img.astype(np.float32).copy()
    ys_k, xs_k = ys[keep], xs[keep]
    ak = alpha[keep][:, None]
    out[ys_k, xs_k] = (b_img[ys_k, xs_k].astype(np.float32) * (1 - ak)
                       + colors[keep].astype(np.float32) * ak)
    out = np.clip(out, 0, 255).astype(np.uint8)

    body.visual.material.baseColorTexture = Image.fromarray(out, mode="RGB")

    report = SwapReport(
        landmark_residual=round(residual, 5),
        region_faces=int(region.sum()),
        painted_texels=int(keep.sum()),
        median_surface_dist=round(float(np.median(dist)), 5),
        atlas_size=(W, H),
    )
    return body, report
