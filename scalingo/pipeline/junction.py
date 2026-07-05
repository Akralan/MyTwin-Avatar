"""Junction reconstruction by Laplacian / gradient-field mesh EDITING.

Goal: a C1 (normal-continuous) transition between the scanned face and the body at
the silhouette, WITHOUT remeshing/rebaking either surface (keeps the two native
textures and two materials). This is the gradient-field mesh-editing family
(Yu 2004 "Mesh Editing with Poisson-Based Gradient Field Manipulation";
Sorkine "Laplacian Surface Editing") rather than the screened-Poisson *surface
reconstruction* (which remeshes everything and was rejected for texture loss).

Why the overlap path shows a seam under variable light
------------------------------------------------------
The MediaPipe scan (478 verts, ~36-vertex OUTER silhouette ring) folds back ~90deg
at its outline. Laid proud over the body, that curled rim is a little cliff: position
is continuous (C0) but normals jump ~10-20deg -> a crease only shading reveals.

CRUCIAL TOPOLOGY NOTE
---------------------
The canonical MediaPipe mesh is NOT a disk: it has open boundary loops at the EYES
and the MOUTH besides the outer oval. So `boundary_vertices` returns all of them and
must never be used to drive the junction. Everything here is keyed off the *outer*
silhouette ring passed in explicitly (aligner.face_silhouette_ring), and graph
distance is measured from that ring only -> eyes/mouth are deep interior and frozen.

Method (deterministic, scipy only)
----------------------------------
Roll only the outer band onto the body, keep the interior rigid:
  * ring 0..`body_pin_rings`  -> pinned onto the body surface (nearest body vertex),
  * inner band               -> free, solved by a Laplacian system that preserves
                                each vertex's differential coordinate (detail),
  * deep interior            -> pinned to its original position (face identity kept).
The free band bends smoothly from the body surface to the untouched face -> the curl
flattens and normals vary smoothly across the join.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
# graph distance from a given set of seed vertices (NOT from all boundaries)
# --------------------------------------------------------------------------- #
def _rings_from(mesh: trimesh.Trimesh, seeds: np.ndarray, max_rings: int) -> np.ndarray:
    from collections import defaultdict, deque
    n = len(mesh.vertices)
    nb = defaultdict(set)
    for a, b in mesh.edges_unique:
        nb[int(a)].add(int(b)); nb[int(b)].add(int(a))
    dist = np.full(n, np.inf)
    dq = deque()
    for v in np.unique(seeds):
        dist[int(v)] = 0.0; dq.append(int(v))
    while dq:
        u = dq.popleft()
        if dist[u] >= max_rings:
            continue
        for w in nb[u]:
            if dist[w] > dist[u] + 1:
                dist[w] = dist[u] + 1; dq.append(w)
    return dist


def _largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Largest vertex-connected component (networkx-free, via scipy csgraph)."""
    from scipy.sparse.csgraph import connected_components
    n = len(mesh.vertices)
    e = mesh.edges_unique
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    ncomp, labels = connected_components(g, directed=False)
    if ncomp <= 1:
        return mesh
    keep_v = labels == np.bincount(labels).argmax()
    m = mesh.copy()
    m.update_faces(keep_v[np.asarray(mesh.faces)].all(axis=1))
    m.remove_unreferenced_vertices()
    return m


def trim_outer_rim(face: trimesh.Trimesh, outer_ring: np.ndarray,
                   n_rings: int = 1) -> trimesh.Trimesh:
    """Peel the outermost `n_rings` of faces off the OUTER silhouette only.

    The MediaPipe shell folds back ~90deg at its outline and that curled band is
    what drapes over the body toward the ear / under the chin. Removing a small
    margin there gives a clean forward-facing boundary that meets the body flush,
    instead of overhanging it. Keyed off `outer_ring` so the eye/mouth loops (deep
    interior) are never touched.
    """
    f = face.copy()
    dist = _rings_from(f, np.asarray(outer_ring), n_rings + 1)
    drop = (dist[np.asarray(f.faces)] < n_rings).any(axis=1)
    f.update_faces(~drop)
    f.remove_unreferenced_vertices()
    return _largest_component(f)


def trim_sides(face: trimesh.Trimesh, outer_ring: np.ndarray, hf,
               cone_deg: float = 45.0, n_rings: int = 1) -> trimesh.Trimesh:
    """Cut a small margin off only the LEFT/RIGHT sides of the mask.

    Unlike `trim_outer_rim` (which peels the whole perimeter and bites into the
    forehead/chin), this removes the outermost `n_rings` of faces only where the
    silhouette runs along the lateral axis (within `cone_deg` of the head's
    left-right direction) — i.e. the part that reaches toward the ears. The
    forehead, chin and the eye/mouth loops are untouched.
    """
    f = face.copy()
    loc = hf.to_local(f.vertices)[:, :2]            # (right, up)
    ring = np.asarray(outer_ring)
    c = loc[ring].mean(0)
    d = loc[ring] - c
    horiz = np.abs(d[:, 0]) / (np.linalg.norm(d, axis=1) + 1e-12)
    side_seeds = ring[horiz >= np.cos(np.radians(cone_deg))]
    if len(side_seeds) == 0:
        return f
    dist = _rings_from(f, side_seeds, n_rings + 1)
    drop = (dist[np.asarray(f.faces)] < n_rings).any(axis=1)
    f.update_faces(~drop)
    f.remove_unreferenced_vertices()
    return _largest_component(f)


def _project_to_surface(body: trimesh.Trimesh, pts: np.ndarray,
                        k_verts: int = 3) -> np.ndarray:
    """Closest point on the body *surface* (triangles), not just nearest vertex.

    rtree/embree are absent, so we localise: gather the faces incident to the few
    nearest body vertices of each query and take the closest point over them.
    """
    from trimesh.triangles import closest_point as tri_closest
    if len(pts) == 0:
        return pts.reshape(0, 3)
    tree = cKDTree(body.vertices)
    vf = body.vertices[body.faces]            # (F,3,3)
    vfaces = body.vertex_faces                # (V,Kmax) padded with -1
    _, vj = tree.query(pts, k=k_verts)
    vj = np.atleast_2d(vj)
    out = np.empty((len(pts), 3))
    for i, p in enumerate(pts):
        fids = np.unique(vfaces[vj[i]].ravel())
        fids = fids[fids >= 0]
        if len(fids) == 0:
            out[i] = body.vertices[vj[i, 0]]
            continue
        tris = vf[fids]
        cp = tri_closest(tris, np.repeat(p[None], len(fids), axis=0))
        out[i] = cp[np.argmin(np.linalg.norm(cp - p, axis=1))]
    return out


# --------------------------------------------------------------------------- #
# cotangent Laplacian (clamped to stay positive -> stable on irregular meshes)
# --------------------------------------------------------------------------- #
def _cotangent_laplacian(V: np.ndarray, F: np.ndarray, clamp: bool = True) -> coo_matrix:
    n = len(V)
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    e0 = V[i2] - V[i1]; e1 = V[i0] - V[i2]; e2 = V[i1] - V[i0]

    def cot(u, w):
        cross = np.linalg.norm(np.cross(u, w), axis=1) + 1e-12
        return np.einsum("ij,ij->i", u, w) / cross

    c0, c1, c2 = cot(-e1, e2), cot(-e2, e0), cot(-e0, e1)
    if clamp:                                   # avoid negative weights (obtuse tris)
        c0 = np.clip(c0, 0, None); c1 = np.clip(c1, 0, None); c2 = np.clip(c2, 0, None)

    I = np.concatenate([i1, i2, i2, i0, i0, i1])
    J = np.concatenate([i2, i1, i0, i2, i1, i0])
    Wd = 0.5 * np.concatenate([c0, c0, c1, c1, c2, c2])
    W = coo_matrix((Wd, (I, J)), shape=(n, n)).tocsr()
    W = W.maximum(W.T)
    d = np.asarray(W.sum(axis=1)).ravel()
    L = (coo_matrix((d, (np.arange(n), np.arange(n))), shape=(n, n)) - W).tocsr()
    return L


# --------------------------------------------------------------------------- #
# core: roll the outer band onto the body, keep interior rigid
# --------------------------------------------------------------------------- #
def roll_band(face: trimesh.Trimesh, body: trimesh.Trimesh, outer_ring: np.ndarray,
              forward: np.ndarray, body_pin_rings: int = 1, band_rings: int = 6,
              proud: float = 0.0) -> trimesh.Trimesh:
    f = face.copy()
    V = np.asarray(f.vertices, float)
    F = np.asarray(f.faces)
    n = len(V)

    dist = _rings_from(f, np.asarray(outer_ring), band_rings + 1)

    L = _cotangent_laplacian(V, F)
    delta = L @ V

    tree = cKDTree(body.vertices)

    targets: dict[int, np.ndarray] = {}
    for vi in range(n):
        d = dist[vi]
        if d <= body_pin_rings:                 # roll onto the body surface
            _, j = tree.query(V[vi])
            targets[vi] = body.vertices[j] + forward * proud
        elif d > band_rings:                    # freeze the interior (identity)
            targets[vi] = V[vi]
        # else: free vertex in the band -> solved by the Laplacian

    pin = np.array(sorted(targets.keys()))
    w = 1.0
    C = coo_matrix((np.full(len(pin), w), (np.arange(len(pin)), pin)),
                   shape=(len(pin), n)).tocsr()
    tgt = np.array([targets[i] for i in pin])

    A = (L.T @ L) + (C.T @ C) + 1e-8 * identity(n)
    rhs = (L.T @ delta) + (C.T @ tgt)
    X = spsolve(A.tocsc(), rhs)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    f.vertices = np.asarray(X)
    return f


# --------------------------------------------------------------------------- #
# gradient-field manipulation (Yu 2004): rotate the band's gradient field so its
# normals interpolate toward the body's, then Poisson-solve for new positions.
# This is what `roll_band` is missing: roll_band keeps delta = L x (the original
# normals) and only moves positions -> C0. Here we build a NEW guidance gradient
# field per face and the normals actually change -> C1.
# --------------------------------------------------------------------------- #
def _grad_operator(V: np.ndarray, F: np.ndarray):
    """Discrete per-face gradient G (3F x N) and per-row area weights M (3F,).

    For a scalar f, (G f)[3t:3t+3] = grad of f over triangle t, using
    grad f = (1/2A) * sum_i f_i (n x e_i), e_i = edge opposite vertex i.
    The cotan Laplacian is exactly G^T diag(M) G.
    """
    m, n = len(F), len(V)
    i, j, k = F[:, 0], F[:, 1], F[:, 2]
    pi, pj, pk = V[i], V[j], V[k]
    cr = np.cross(pj - pi, pk - pi)
    A2 = np.linalg.norm(cr, axis=1) + 1e-12          # = 2*area
    nf = cr / A2[:, None]
    gi = np.cross(nf, pk - pj) / A2[:, None]         # vertex i opp edge (j,k)
    gj = np.cross(nf, pi - pk) / A2[:, None]
    gk = np.cross(nf, pj - pi) / A2[:, None]

    base = (3 * np.arange(m)[:, None] + np.array([0, 1, 2])[None, :]).ravel()
    rows = np.concatenate([base, base, base])
    cols = np.concatenate([np.repeat(i, 3), np.repeat(j, 3), np.repeat(k, 3)])
    data = np.concatenate([gi.ravel(), gj.ravel(), gk.ravel()])
    G = coo_matrix((data, (rows, cols)), shape=(3 * m, n)).tocsr()
    M = np.repeat(0.5 * A2, 3)
    return G, M, nf


def _rodrigues(axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """Vectorised axis-angle -> (m,3,3) rotation matrices."""
    m = len(ang)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    K = np.zeros((m, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -z, y
    K[:, 1, 0], K[:, 1, 2] = z, -x
    K[:, 2, 0], K[:, 2, 1] = -y, x
    s = np.sin(ang)[:, None, None]
    c = (1 - np.cos(ang))[:, None, None]
    return np.eye(3)[None] + s * K + c * (K @ K)


def gradient_field_merge(face: trimesh.Trimesh, body: trimesh.Trimesh,
                         outer_ring: np.ndarray, forward: np.ndarray,
                         band_rings: int = 6, body_pin_rings: int = 1,
                         proud: float = 0.0, max_rot_deg: float = 60.0,
                         anchor_weight: float = 1000.0,
                         up: np.ndarray | None = None,
                         right: np.ndarray | None = None) -> trimesh.Trimesh:
    """Poisson gradient-field edit: rotate band faces toward the body normal."""
    f = face.copy()
    V = np.asarray(f.vertices, float)
    F = np.asarray(f.faces)
    n, m = len(V), len(F)

    G, M, nf = _grad_operator(V, F)
    Msp = coo_matrix((M, (np.arange(3 * m), np.arange(3 * m))), shape=(3 * m, 3 * m)).tocsr()
    L = (G.T @ Msp @ G).tocsr()

    # per-face attenuation weight from graph distance of its vertices to the ring
    vdist = _rings_from(f, np.asarray(outer_ring), band_rings + 1)
    fdist = vdist[F].min(axis=1)
    wface = np.clip(1.0 - fdist / max(band_rings, 1), 0.0, 1.0)

    # target normal per face = nearest body vertex normal (sign-aligned to face)
    tree = cKDTree(body.vertices)
    cen = V[F].mean(axis=1)
    _, jb = tree.query(cen)
    nb = np.asarray(body.vertex_normals)[jb]
    sgn = np.sign(np.einsum("ij,ij->i", nf, nb))
    sgn[sgn == 0] = 1.0
    nb = nb * sgn[:, None]

    axis = np.cross(nf, nb)
    an = np.linalg.norm(axis, axis=1)
    axis_u = axis / (an[:, None] + 1e-12)
    ang_full = np.arctan2(an, np.einsum("ij,ij->i", nf, nb))
    ang = np.clip(wface * ang_full, 0.0, np.radians(max_rot_deg))
    R = _rodrigues(axis_u, ang)                       # (m,3,3); identity where w=0

    # rotate the original per-face gradient field: J' = J @ R^T
    Jf = (G @ V).reshape(m, 3, 3)                      # columns = grad x,y,z
    Jp = np.einsum("frd,fcd->frc", Jf, R)
    g_guided = Jp.reshape(3 * m, 3)

    rhs = G.T @ (M[:, None] * g_guided)               # divergence of guided field

    # position anchors: outer rings HARD onto the body, deep interior frozen.
    # A hard (high) weight is required so the rim actually reaches the body on the
    # side that must travel farthest — a soft weight under-corrects there and leaves
    # a one-sided gap.
    pin_body = [vi for vi in range(n) if vdist[vi] <= body_pin_rings]
    pin_arr = np.asarray(pin_body)
    bproj = _project_to_surface(body, V[pin_arr])

    # Anisotropic proud: keep the SAME direction (`forward`) for every rim vertex — so the
    # boundary keeps its smooth depth profile — but taper the MAGNITUDE. Full proud at the
    # lateral sides (body faces forward -> mask lifts cleanly, no z-fighting); tapering to 0
    # at the forehead/chin, where `forward` is ~tangent and a full push would slide the rim
    # off the skin the body folds over (the residual forehead gap).
    if up is not None and right is not None and len(pin_arr):
        cen = V[pin_arr].mean(0)
        rr = (V[pin_arr] - cen) @ np.asarray(right, float)
        uu = (V[pin_arr] - cen) @ np.asarray(up, float)
        proud_scale = np.abs(rr) / (np.hypot(rr, uu) + 1e-12)   # 1 at sides, 0 at forehead/chin
    else:
        proud_scale = np.ones(len(pin_arr))

    targets: dict[int, np.ndarray] = {}
    for k, vi in enumerate(pin_body):
        targets[vi] = bproj[k] + forward * (proud * proud_scale[k])
    for vi in range(n):
        if vdist[vi] > band_rings:
            targets[vi] = V[vi]

    pin = np.array(sorted(targets.keys()))
    aw = anchor_weight
    C = coo_matrix((np.full(len(pin), aw), (np.arange(len(pin)), pin)),
                   shape=(len(pin), n)).tocsr()
    tgt = np.array([targets[i] for i in pin])

    A = (L + C.T @ C + 1e-8 * identity(n)).tocsc()
    X = spsolve(A, rhs + C.T @ (aw * tgt))
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    f.vertices = np.asarray(X)
    return f


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _interp_ring_by_angle(src_ang: np.ndarray, src_val: np.ndarray,
                          q_ang: np.ndarray) -> np.ndarray:
    """Sample a closed ring `src_val` (M,D) parameterised by `src_ang` at the query
    angles `q_ang` (periodic linear interp) — makes the two boundaries correspond
    point-for-point at equal angle around the seam centre."""
    a = np.mod(np.asarray(src_ang, float), 2 * np.pi)
    order = np.argsort(a)
    a = a[order]
    v = np.asarray(src_val, float)[order]
    q = np.mod(np.asarray(q_ang, float), 2 * np.pi)
    out = np.empty((len(q), v.shape[1]))
    for d in range(v.shape[1]):
        out[:, d] = np.interp(q, a, v[:, d], period=2 * np.pi)
    return out


def _bfs_falloff(mesh: trimesh.Trimesh, seed_idx: np.ndarray, n_rings: int):
    """Multi-source BFS from `seed_idx`; returns (graph distance, index of the seed
    that reached each vertex) up to `n_rings`."""
    from collections import deque
    nbrs = mesh.vertex_neighbors
    n = len(mesh.vertices)
    dist = np.full(n, np.inf)
    src = np.full(n, -1, int)
    dq: deque = deque()
    for k, vi in enumerate(seed_idx):
        vi = int(vi)
        dist[vi] = 0.0
        src[vi] = k
        dq.append(vi)
    while dq:
        u = dq.popleft()
        if dist[u] >= n_rings:
            continue
        for w in nbrs[u]:
            if dist[w] > dist[u] + 1:
                dist[w] = dist[u] + 1
                src[w] = src[u]
                dq.append(int(w))
    return dist, src


def _weld_seam_normals(mesh: trimesh.Trimesh, ring_idx: np.ndarray,
                       seam_normals: np.ndarray, blend_rings: int) -> np.ndarray:
    """Override the ring's vertex normals with `seam_normals` (per ring vertex) and
    blend back to the mesh's own normals over `blend_rings`, for a soft transition."""
    base = np.asarray(mesh.vertex_normals, float).copy()
    dist, src = _bfs_falloff(mesh, ring_idx, blend_rings)
    within = (dist <= blend_rings) & (src >= 0)
    w = np.clip(1.0 - dist / (blend_rings + 1), 0.0, 1.0)
    idxs = np.where(within)[0]
    tgt = seam_normals[src[idxs]]
    base[idxs] = _unit(base[idxs] * (1 - w[idxs])[:, None] + tgt * w[idxs][:, None])
    return base


def _with_normals(mesh: trimesh.Trimesh, normals: np.ndarray) -> trimesh.Trimesh:
    """Rebuild `mesh` (same geometry, same visual/UV/material) carrying explicit
    vertex normals passed to the constructor — so the GLB export ships NORMAL. (A
    post-hoc `.vertex_normals = ...` is not reliably included by the glTF exporter,
    which is why the face previously exported without normals.)"""
    m = trimesh.Trimesh(vertices=np.asarray(mesh.vertices, float),
                        faces=np.asarray(mesh.faces),
                        vertex_normals=np.asarray(normals, float),
                        process=False)
    m.visual = mesh.visual
    return m


def weld_normals(body: trimesh.Trimesh, face: trimesh.Trimesh,
                 hole_loop: np.ndarray, face_ring: np.ndarray, hf,
                 blend_rings: int = 2):
    """Share averaged normals along the seam so shading is continuous across the
    body↔mask junction, WITHOUT moving any vertex (geometry untouched, glue_rim's
    work preserved). Both surfaces get the SAME normal at equal angle on the seam,
    blended inward over `blend_rings`, and both are re-exported with explicit normals.
    Returns (body, face) rebuilt copies."""
    rim = np.asarray(hole_loop, int)
    mring = np.asarray(face_ring, int)

    Pb = hf.to_local(body.vertices[rim])
    Pm = hf.to_local(face.vertices[mring])
    c = Pm[:, :2].mean(0)
    ang_b = np.arctan2(Pb[:, 1] - c[1], Pb[:, 0] - c[0])
    ang_m = np.arctan2(Pm[:, 1] - c[1], Pm[:, 0] - c[0])

    bn = np.asarray(body.vertex_normals, float)
    fn = np.asarray(face.vertex_normals, float)
    nb_at_m = _interp_ring_by_angle(ang_b, bn[rim], ang_m)     # body normal @ mask angles
    nm_at_b = _interp_ring_by_angle(ang_m, fn[mring], ang_b)   # mask normal @ body angles
    seam_m = _unit(fn[mring] + nb_at_m)                        # equal to seam_b at equal angle
    seam_b = _unit(bn[rim] + nm_at_b)

    bnorm = _weld_seam_normals(body, rim, seam_b, blend_rings)
    fnorm = _weld_seam_normals(face, mring, seam_m, blend_rings)
    return _with_normals(body, bnorm), _with_normals(face, fnorm)


def densify(face: trimesh.Trimesh, iterations: int = 1) -> trimesh.Trimesh:
    """Uniform 1->4 subdivision of the mask, UVs interpolated, MUST run post-align.

    More angular resolution around the perimeter so the outer ring the junction pins
    onto the body is a finer polygon -> the boundary hugs the body curvature and the
    festons (the oscillating micro-steps that catch light) shrink. Uniform (all faces)
    keeps the mesh conforming — no T-junctions/cracks — and the subdivision only
    interpolates on the existing surface, so the scan's shape/identity is unchanged;
    the visual gain comes from the denser pins being re-projected onto the body in
    `gradient_field_merge`. Run AFTER `align_by_landmarks` (which needs the canonical
    478-vertex index order); the silhouette ring is geometric so it recomputes fine.

    NOTE each iteration halves the graph-ring spacing, so the junction's `band_rings`
    / `body_pin_rings` must be scaled by 2**iterations to keep the same physical band.
    """
    if iterations <= 0:
        return face
    from trimesh.remesh import subdivide

    V = np.asarray(face.vertices, float)
    F = np.asarray(face.faces)
    uv = None
    mat = getattr(face.visual, "material", None)
    if getattr(face.visual, "uv", None) is not None:
        uv = np.asarray(face.visual.uv, float)

    for _ in range(iterations):
        if uv is not None:
            V, F, attrs = subdivide(V, F, vertex_attributes={"uv": uv})
            uv = attrs["uv"]
        else:
            V, F = subdivide(V, F)

    new = trimesh.Trimesh(vertices=V, faces=F, process=False)
    if uv is not None and mat is not None:
        new.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    return new


def glue_rim(body: trimesh.Trimesh, face: trimesh.Trimesh, hole_loop: np.ndarray,
             hf, n_rings: int = 2, recess_frac: float = 0.0015) -> trimesh.Trimesh:
    """Press the body's hole rim back onto the mask so it stops overhanging it.

    The scan sits proud through the hole; wherever the body's cut edge stands in
    front of the mask it makes a lit lip ("le trou dépasse sur le masque"). We push
    only the OVERHANGING rim vertices back along `forward` to just behind the mask
    surface (nearest point on the mask triangles), keeping their lateral (right/up)
    position so the hole outline is unchanged, and taper the push over `n_rings` of
    body neighbours so the body gets no crease. Rims already behind the mask are
    left untouched.
    """
    b = body.copy()
    V = np.asarray(b.vertices, float)
    fwd = np.asarray(hf.forward, float)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    recess = recess_frac * hf.depth

    rim = np.asarray(hole_loop, int)
    proj = _project_to_surface(face, V[rim])          # nearest point on the mask
    f_rim = V[rim] @ fwd
    f_mask = proj @ fwd
    # push back only where the rim stands in front of the mask; land it just behind.
    delta = np.minimum((f_mask - recess) - f_rim, 0.0)   # <=0 => move back along fwd
    if not (np.abs(delta) > 1e-9).any():
        return b

    # multi-source BFS from the rim, carrying each seed's delta, tapering by ring
    from collections import deque
    nbrs = b.vertex_neighbors
    dist = np.full(len(V), np.inf)
    seed_delta = np.zeros(len(V))
    dq: deque = deque()
    for k, vi in enumerate(rim):
        vi = int(vi)
        dist[vi] = 0.0
        seed_delta[vi] = delta[k]
        dq.append(vi)
    while dq:
        u = dq.popleft()
        if dist[u] >= n_rings:
            continue
        for w in nbrs[u]:
            if dist[w] > dist[u] + 1:
                dist[w] = dist[u] + 1
                seed_delta[w] = seed_delta[u]
                dq.append(int(w))

    within = dist <= n_rings
    taper = np.clip(1.0 - dist / (n_rings + 1), 0.0, 1.0)
    disp = np.zeros(len(V))
    disp[within] = seed_delta[within] * taper[within]
    b.vertices = V + disp[:, None] * fwd[None, :]
    return b


def tangent_merge(face: trimesh.Trimesh, body: trimesh.Trimesh, hf,
                  outer_ring: np.ndarray, band_rings: int = 6,
                  body_pin_rings: int = 1, proud_frac: float = 0.004,
                  mode: str = "gradient") -> trimesh.Trimesh:
    """Junction edit entry point. `outer_ring` = aligner outer silhouette indices.

    mode='gradient' -> gradient-field manipulation (C1, rotates normals);
    mode='roll'     -> position-only Laplacian roll (C0, keeps original normals).
    """
    fwd = np.asarray(hf.forward, float)
    proud = proud_frac * hf.depth
    if mode == "roll":
        return roll_band(face, body, outer_ring, fwd, body_pin_rings=body_pin_rings,
                         band_rings=band_rings, proud=proud)
    return gradient_field_merge(face, body, outer_ring, fwd, band_rings=band_rings,
                                body_pin_rings=body_pin_rings, proud=proud,
                                up=np.asarray(hf.up, float), right=np.asarray(hf.right, float))
