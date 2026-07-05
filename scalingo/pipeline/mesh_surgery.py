"""Step 3 — cut the facial region out of the body, leaving a clean hole.

The region is an oriented ellipse in the head frame's (right, up) plane,
restricted to the front hemisphere (so we never remove the back of the skull).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .head_locator import HeadFrame


@dataclass
class CutResult:
    body: trimesh.Trimesh        # body with the hole (visuals preserved)
    hole_loop: np.ndarray        # ordered vertex indices of the hole boundary
    removed_faces: int


def select_face_region(body: trimesh.Trimesh, hf: HeadFrame,
                       width_scale: float = 0.34,
                       height_scale: float = 0.44,
                       front_offset: float = 0.30,
                       v_center_shift: float = -0.07) -> np.ndarray:
    """Boolean mask over faces that belong to the facial region.

    A face is selected when its centroid, expressed in the head frame, lies inside
    the facial ellipse *and* on the front hemisphere.

    `width_scale`/`height_scale` are the ellipse semi-axes as a fraction of the
    head's full width/height — keep them well under 0.5 so the cut stays a frontal
    face opening and does not wrap around the temples (which oversizes the face and
    sinks it into the skull). `v_center_shift` (fraction of head height) lowers the
    ellipse centre so the cut covers eyes/nose/mouth/chin instead of the hairline
    (a too-high cut leaves a gap in the hair).
    """
    tri_centroids = body.vertices[body.faces].mean(axis=1)
    loc = hf.to_local(tri_centroids)           # (F, 3): right, up, forward
    rx, ry, rz = loc[:, 0], loc[:, 1], loc[:, 2]

    cy = (hf.face_anchor - hf.center) @ hf.up + v_center_shift * hf.height
    Rw = width_scale * hf.width
    Rh = height_scale * hf.height

    in_ellipse = (rx / Rw) ** 2 + ((ry - cy) / Rh) ** 2 <= 1.0
    in_front = rz > front_offset * hf.depth
    return in_ellipse & in_front


def extract_hole_ring(holed: trimesh.Trimesh, hf: HeadFrame,
                      width_scale: float, height_scale: float,
                      v_center_shift: float = -0.07,
                      n_bins: int = 96) -> np.ndarray:
    """Ordered vertex indices of the hole rim, robust to the body's messy topology.

    Meshy bodies carry hundreds of pre-existing tiny boundary loops, so edge
    walking is unreliable. Instead we take boundary vertices that sit on the front
    of the head near the cut ellipse's perimeter and order them by angle.
    """
    from . import geom

    bverts = geom.boundary_vertices(holed)
    loc = hf.to_local(holed.vertices[bverts])
    rx, ry, rz = loc[:, 0], loc[:, 1], loc[:, 2]
    cy = (hf.face_anchor - hf.center) @ hf.up + v_center_shift * hf.height
    Rw = width_scale * hf.width
    Rh = height_scale * hf.height

    nx = rx / Rw
    ny = (ry - cy) / Rh
    rho = np.sqrt(nx ** 2 + ny ** 2)
    # rim = boundary vertices in an annulus around the ellipse perimeter, on the front
    sel = (rho > 0.75) & (rho < 1.30) & (rz > 0.0)
    if sel.sum() < 12:
        sel = (rho > 0.6) & (rho < 1.5) & (rz > -0.1 * hf.depth)
    pts2d = np.stack([nx[sel], ny[sel]], axis=1)
    ring = geom.angular_ring(pts2d, bverts[sel], n_bins=n_bins)
    return ring


def cut_to_silhouette(body: trimesh.Trimesh, hf: HeadFrame,
                      silhouette_world: np.ndarray, inflate: float = 0.86,
                      inflate_side: float | None = None,
                      inflate_bottom: float | None = None) -> CutResult:
    """Cut the body where the aligned scan sits (intelligent path).

    The hole is the scan's own silhouette projected into the head's (right, up)
    plane, eroded by `inflate < 1` so the hole is clearly *smaller* than the scan.
    The scan then overlaps the surrounding body skin all around (no edge-to-edge
    stitching, no gap that would reveal the dark interior), and the texture feather
    blends the overlap. This is what removes the dark seam triangles.

    `inflate_side` (defaults to `inflate`) controls the erosion along the head's
    lateral (left-right) axis independently: a value closer to 1 widens the hole on
    the sides (less lateral overlap toward the ears), while the vertical extent at
    the forehead/chin keeps `inflate`.

    `inflate_bottom` (defaults to `inflate`) controls the vertical erosion for the
    lower half only (the chin): a value closer to 1 enlarges the hole at the chin
    (the body edge recedes downward, less overlap onto the mask chin), while the
    forehead keeps `inflate`.
    """
    from . import geom

    fx = inflate if inflate_side is None else inflate_side
    fb = inflate if inflate_bottom is None else inflate_bottom
    poly = hf.to_local(silhouette_world)[:, :2]
    c = poly.mean(0)
    poly = poly.copy()
    vy = poly[:, 1] - c[1]                               # >0 forehead, <0 chin
    v_scale = np.where(vy < 0, fb, inflate)             # chin uses its own erosion
    poly[:, 0] = c[0] + (poly[:, 0] - c[0]) * fx        # lateral (widen the sides)
    poly[:, 1] = c[1] + vy * v_scale                    # vertical (forehead / chin)

    tri_c = body.vertices[body.faces].mean(axis=1)
    loc = hf.to_local(tri_c)
    inside = geom.point_in_polygon(loc[:, :2], poly)
    in_front = loc[:, 2] > 0.0
    region = inside & in_front

    holed = body.copy()
    holed.update_faces(~region)
    holed.remove_unreferenced_vertices()

    # hole rim, ordered by angle about the silhouette centre (robust to messy mesh)
    centre2d = poly.mean(0)
    bverts = geom.boundary_vertices(holed)
    bl = hf.to_local(holed.vertices[bverts])[:, :2] - centre2d
    rho = np.linalg.norm(bl, axis=1)
    scale = np.median(np.linalg.norm(poly - centre2d, axis=1))
    sel = (rho < 1.6 * scale) & (hf.to_local(holed.vertices[bverts])[:, 2] > -0.05 * hf.depth)
    ring = geom.angular_ring(bl[sel] / scale, bverts[sel], n_bins=96)

    return CutResult(body=holed, hole_loop=ring, removed_faces=int(region.sum()))


def cut_face(body: trimesh.Trimesh, hf: HeadFrame,
             width_scale: float = 0.34, height_scale: float = 0.44,
             front_offset: float = 0.30, v_center_shift: float = -0.07) -> CutResult:
    region = select_face_region(body, hf, width_scale=width_scale,
                                height_scale=height_scale, front_offset=front_offset,
                                v_center_shift=v_center_shift)
    keep = ~region

    holed = body.copy()
    holed.update_faces(keep)
    holed.remove_unreferenced_vertices()

    hole_ring = extract_hole_ring(holed, hf, width_scale, height_scale, v_center_shift)

    return CutResult(body=holed, hole_loop=hole_ring, removed_faces=int(region.sum()))
