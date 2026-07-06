"""Step 7 — give the inserted face the same skin *grain* as the body.

The Meshy body albedo carries a fine, near-uniform micro-grain (per-channel std
~6.5, mostly achromatic luminance speckle with a little chroma). The scanned face
is smooth by comparison, so at the junction the eye reads two different materials —
granular body vs plastic face — even when the *colour* already matches (that part is
handled upstream by `texture_blender.match_skin_tone` + `feather_border`).

Two facts drive the method (both measured, see `measure_body_grain` /
`face_cell_px`):
  * the body's grain is high frequency — its wavelength is ~0.4 of ONE face-atlas
    texel, i.e. finer than the face atlas can represent. So we first **upsample the
    face atlas** to give the grain room, then synthesize at the right cell size.
  * the grain is ~2/3 luminance, ~1/3 chroma (channel correlation ~0.6).

We synthesize grain matched to those statistics and add it **uniformly across the
face, at full amplitude up to the silhouette**, so it is continuous with the
surrounding body grain and the seam stops reading. A luminance gate keeps very dark
features (brows/eyes/nostrils) from picking up speckle.

Only the face's baseColorTexture image changes (it also grows by `upsample`).
Geometry, UVs, materials and the body are untouched. OpenCV is optional; without it
the grain amplitude falls back to a fixed default (no body measurement).
"""
from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image

from .render_head import sample_vertex_colors

try:
    import cv2  # noqa: F401
    _CV2 = True
except Exception:  # noqa: BLE001
    _CV2 = False


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (r >= g) & (g >= b) & (lum > 60) & (lum < 238)


def measure_body_grain(body: trimesh.Trimesh, hf, default=(5.5, 3.5)) -> tuple:
    """Return (sigma_lum, sigma_chroma) of the body albedo grain over head skin.

    Grain = body albedo minus its Non-Local-Means denoise (the same residual the
    body-denoise step would remove). We split it into an achromatic luminance
    component (shared across channels) and a per-channel chroma remainder, matching
    how the body speckle is distributed. Falls back to `default` without OpenCV or a
    body texture.
    """
    mat = getattr(body.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) if mat is not None else None
    if not _CV2 or img is None:
        return default
    import cv2
    rgb = np.asarray(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    den = cv2.cvtColor(cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21),
                       cv2.COLOR_BGR2RGB)
    resid = rgb.astype(np.float32) - den.astype(np.float32)
    m = _skin_mask(rgb)
    if m.sum() < 500:
        return default
    rs = resid[m]                                   # (N,3)
    lum_axis = np.array([1, 1, 1], np.float32) / np.sqrt(3.0)
    lum = rs @ lum_axis                             # achromatic component
    chroma = rs - lum[:, None] * lum_axis[None]     # per-channel remainder
    s_lum = float(lum.std() / np.sqrt(3.0))         # per-channel contribution
    s_chroma = float(np.linalg.norm(chroma, axis=1).std() / np.sqrt(3.0))
    return s_lum, s_chroma


def _skin_sigma(rgb: np.ndarray) -> tuple:
    """(s_lum, s_chroma) of the NlMeans high-freq residual over the skin of an RGB image.

    Same luminance/chroma split as `measure_body_grain`, factored out so we can measure
    the FACE's own existing grain too. Expects a uint8 RGB array; returns (0,0) without
    OpenCV or enough skin."""
    if not _CV2:
        return (0.0, 0.0)
    import cv2
    m = _skin_mask(rgb)
    if m.sum() < 500:
        return (0.0, 0.0)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    den = cv2.cvtColor(cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21),
                       cv2.COLOR_BGR2RGB)
    resid = rgb.astype(np.float32) - den.astype(np.float32)
    rs = resid[m]
    lax = np.array([1, 1, 1], np.float32) / np.sqrt(3.0)
    lum = rs @ lax
    chroma = rs - lum[:, None] * lax[None]
    return (float(lum.std() / np.sqrt(3.0)),
            float(np.linalg.norm(chroma, axis=1).std() / np.sqrt(3.0)))


def measure_face_grain(arr: np.ndarray, max_side: int = 768) -> tuple:
    """(s_lum, s_chroma) of the grain the FACE atlas *already* carries, on a bounded skin
    crop (so NlMeans stays cheap even on a 4K/upsampled atlas).

    The selfie is not smooth: it comes with its own speckle. Adding the full body grain on
    top would STACK two fields (combined std = sqrt(a^2+b^2)) -> visibly over-grained. We
    measure this so `add_matched_grain` can add only the shortfall to reach the body level.
    `arr` is the (possibly upsampled, float) atlas the grain will be added to."""
    if not _CV2:
        return (0.0, 0.0)
    rgb = np.clip(arr, 0, 255).astype(np.uint8)
    m = _skin_mask(rgb)
    if m.sum() < 500:
        return (0.0, 0.0)
    ys, xs = np.where(m)
    cy, cx = int(ys.mean()), int(xs.mean())
    h = max_side // 2
    y0, y1 = max(cy - h, 0), min(cy + h, rgb.shape[0])
    x0, x1 = max(cx - h, 0), min(cx + h, rgb.shape[1])
    return _skin_sigma(rgb[y0:y1, x0:x1])


def measure_body_grain_wavelength(body: trimesh.Trimesh, hf, default: float = 0.7) -> float:
    """Autocorrelation 1/e wavelength of the body grain, in BODY TEXELS (== atlas px).

    `face_cell_px` sizes the synthesized grain to ONE body texel, but the real body
    speckle is sub-texel (~0.69 measured on Meshy albedos): it is near-white per-texel
    noise, not a 1-texel blob. Sizing to 1 texel therefore makes the grain ~1/0.69 = 1.45x
    too coarse. Multiply `cell_px` by this measured wavelength to match reality. Falls back
    to `default` without OpenCV or enough skin. Clamped to a sane [0.4, 3] texel range."""
    mat = getattr(body.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) if mat is not None else None
    if not _CV2 or img is None:
        return default
    import cv2
    rgb = np.asarray(img.convert("RGB"))
    m = _skin_mask(rgb)
    if m.sum() < 2000:
        return default
    ys, xs = np.where(m)
    patch = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    den = cv2.cvtColor(cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21),
                       cv2.COLOR_BGR2RGB)
    resid = (patch.astype(np.float32) - den.astype(np.float32))[..., 1]  # green channel
    p = resid - resid.mean()
    F = np.fft.rfft2(p)
    ac = np.fft.irfft2(F * np.conj(F), s=p.shape)
    if ac[0, 0] <= 0:
        return default
    prof = (ac[0, :] / ac[0, 0])[:20]
    thr = 1.0 / np.e
    k = int(np.argmax(prof < thr))
    if k == 0:
        return default
    y0v, y1v = prof[k - 1], prof[k]
    wl = (k - 1) + (y0v - thr) / (y0v - y1v + 1e-9)   # sub-texel 1/e crossing
    return float(np.clip(wl, 0.4, 3.0))


def face_cell_px(face: trimesh.Trimesh, body: trimesh.Trimesh, hf) -> float:
    """Body 1-texel grain wavelength expressed in ORIGINAL face-atlas pixels.

    = sqrt(face_texel_density / body_texel_density) over the head, where density is
    UV-area(px^2)/surface-area. Typically < 1 (the face atlas is coarser), which is
    why we upsample before synthesizing grain.
    """
    def density(mesh, res, region=None):
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.faces)
        uv = np.asarray(mesh.visual.uv)
        t3 = v[f]
        a3 = 0.5 * np.linalg.norm(np.cross(t3[:, 1] - t3[:, 0], t3[:, 2] - t3[:, 0]), axis=1)
        tuv = uv[f]
        auv = 0.5 * np.abs((tuv[:, 1, 0] - tuv[:, 0, 0]) * (tuv[:, 2, 1] - tuv[:, 0, 1])
                           - (tuv[:, 2, 0] - tuv[:, 0, 0]) * (tuv[:, 1, 1] - tuv[:, 0, 1]))
        apx = auv * res[0] * res[1]
        if region is not None:
            a3, apx = a3[region], apx[region]
        return apx.sum() / (a3.sum() + 1e-12)

    fimg = np.asarray(face.visual.material.baseColorTexture)
    fh, fw = fimg.shape[:2]
    bimg = np.asarray(body.visual.material.baseColorTexture)
    bh, bw = bimg.shape[:2]
    loc = hf.to_local(body.vertices)
    fc = np.asarray(body.faces)
    headf = (loc[fc, 1].mean(1) > -0.1 * hf.height) & (loc[fc, 2].mean(1) > 0.0)
    db = density(body, (bw, bh), headf)
    dfc = density(face, (fw, fh))
    return float(np.sqrt(dfc / max(db, 1e-9)))


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def stretch_mask_vertex(face: trimesh.Trimesh, ref_pct: float = 5.0,
                        cap: float = 8.0) -> np.ndarray:
    """Per-vertex capture-stretch factor of the frontal-projected face texture.

    The MediaPipe face GLB carries UVs that are an *exact frontal orthographic
    projection* of the vertices (u,v = affine(x,y), z ignored — verified on the real
    asset). So the surface-area / UV-area ratio of a triangle equals 1/cos(theta),
    theta = tilt of the surface away from the capture camera. Frontal patches (nose,
    forehead) sit at the ratio floor; the lateral cheeks/jaw that turn toward the ears
    have a large ratio == exactly the region MediaPipe smears.

    Returns (N,) with ~1.0 where frontal and growing (1/cos theta) where stretched.
    Frame-invariant: the area ratio does not change under the rigid body alignment, so
    this is valid on the already-placed `stitched` mesh. `ref_pct` picks the frontal
    reference (a low percentile is robust to a few near-frontal outliers)."""
    V = np.asarray(face.vertices, float)
    F = np.asarray(face.faces)
    uv = np.asarray(face.visual.uv, float)
    t3 = V[F]
    a3 = 0.5 * np.linalg.norm(
        np.cross(t3[:, 1] - t3[:, 0], t3[:, 2] - t3[:, 0]), axis=1)
    tuv = uv[F]
    auv = 0.5 * np.abs((tuv[:, 1, 0] - tuv[:, 0, 0]) * (tuv[:, 2, 1] - tuv[:, 0, 1])
                       - (tuv[:, 2, 0] - tuv[:, 0, 0]) * (tuv[:, 1, 1] - tuv[:, 0, 1]))
    # Guard degenerate UV slivers (auv -> 0): they blow r up to spurious 80x spikes.
    # Reference on non-tiny triangles only, then cap the factor (beyond `cap` the
    # smoothstep is saturated anyway, so a hard cap costs nothing and kills outliers).
    auv_floor = 0.05 * np.median(auv[auv > 0]) if np.any(auv > 0) else 0.0
    r = np.nan_to_num(a3 / (auv + 1e-12), nan=0.0, posinf=0.0)
    solid = r[(auv > auv_floor) & np.isfinite(r) & (r > 0)]
    r0 = np.percentile(solid, ref_pct) if solid.size else 1.0
    sf_face = np.clip(r / (r0 + 1e-12), 0.0, cap)     # ~1 frontal, >1 oblique

    n = len(V)
    vsum = np.zeros(n, np.float64)
    vcnt = np.zeros(n, np.float64)
    np.add.at(vsum, F.ravel(), np.repeat(sf_face, 3))
    np.add.at(vcnt, F.ravel(), 1.0)
    return (vsum / np.maximum(vcnt, 1.0)).astype(np.float32)


def _grain_field(h, w, cell, s_lum, s_chroma, rng):
    """Synthesize (h,w,3) grain: shared luminance speckle + per-channel chroma, with
    correlation length `cell` px, normalized to the target per-channel std."""
    def band(nch):
        n = rng.standard_normal((h, w, nch)).astype(np.float32)
        if cell > 0.6 and _CV2:
            import cv2
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=cell / 2.0, sigmaY=cell / 2.0)
        n = n.reshape(h, w, nch)                        # GaussianBlur squeezes nch=1
        # renormalize to unit std per channel after blurring reduced the variance
        n /= (n.reshape(-1, nch).std(0) + 1e-6)
        return n

    lum = band(1)[..., 0]                              # (h,w)
    chroma = band(3)                                   # (h,w,3)
    g = s_lum * lum[..., None] + s_chroma * chroma     # broadcast lum to 3 ch
    return g


def add_matched_grain(face: trimesh.Trimesh, body: trimesh.Trimesh, hf,
                      upsample: int = 3, amount: float = 0.5,
                      sigma: tuple | None = None, cell_px: float | None = None,
                      wavelength_scale: float = 0.6,
                      dark_gate: tuple = (18.0, 60.0), seed: int = 7,
                      stretch_fade: bool = True, stretch_strength: float = 1.0,
                      stretch_band: tuple = (1.8, 3.2),
                      detail_sigma: float | None = None) -> trimesh.Trimesh:
    """Return a copy of `face` whose albedo carries body-matched skin grain.

    upsample   integer atlas upscale (bicubic) so fine grain is representable.
    amount     global multiplier on the grain amplitude (tuning knob).
    sigma      (sigma_lum, sigma_chroma) per-channel target; measured from the body
               when None.
    cell_px    grain correlation length in UPSAMPLED atlas px; derived from the
               surface texel-density ratio when None.
    wavelength_scale  aesthetic knob on grain FINENESS. 1.0 = match the body grain
               wavelength exactly (measured). <1 makes the grain finer than the body
               (e.g. 0.7 == 30% finer); >1 coarser. Multiplies the derived cell_px only.
    dark_gate  (lo, hi) luminance band over which grain fades in, so near-black
               features (brows/eyes/nostrils) stay clean.

    De-streak (cheeks). MediaPipe's frontal selfie smears the lateral cheeks/jaw into
    directional streaks, because those surfaces turn away from the camera so few source
    pixels cover a large area (there is *no real detail* there to keep). Where the
    capture-stretch is high (`stretch_mask_vertex`), we crossfade the selfie's own
    high-frequency band OUT and let the isotropic body-matched grain stand in — so the
    streaks go and the region reads as the same skin material as the body. The
    low-frequency albedo (colour/tone, identity of the cheek) is always preserved; only
    unreliable detail is replaced. No-op without OpenCV (needs the band split).

    stretch_fade      enable the de-streak crossfade.
    stretch_strength  fraction of the selfie HF removed where fully stretched (1 = all).
    stretch_band      (lo, hi) foreshortening factors (1/cos theta) mapped by smoothstep
                      to 0..1 fade; ~1.8 == ~55deg tilt starts, ~3.2 == ~72deg full.
    detail_sigma      lowpass sigma (UPSAMPLED px) splitting kept albedo from replaced
                      detail; derived from the grain cell when None.
    """
    mat = getattr(face.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) if mat is not None else None
    if img is None:
        return face

    if sigma is None:
        sigma = measure_body_grain(body, hf)
    s_lum, s_chroma = sigma
    if cell_px is None:
        try:
            # Size the grain to the body's REAL grain wavelength, not to "1 body texel".
            # The Meshy speckle is sub-texel (~0.69), so face_cell_px alone (== 1 texel)
            # makes the grain ~1.45x too coarse; the measured wavelength corrects it.
            cell_px = (face_cell_px(face, body, hf) * upsample
                       * measure_body_grain_wavelength(body, hf) * wavelength_scale)
        except Exception:  # noqa: BLE001
            cell_px = 0.42 * upsample * wavelength_scale

    base = img.convert("RGB")
    if upsample and upsample > 1:
        base = base.resize((base.width * upsample, base.height * upsample),
                           Image.BICUBIC)
    arr = np.asarray(base).astype(np.float32)
    h, w = arr.shape[:2]

    # Don't stack grains: the selfie already carries speckle. Adding the full body grain on
    # top combines to sqrt(existing^2 + added^2) -> over-grained. Measure the face's own
    # grain and add only the shortfall to reach the body level, in quadrature (per channel).
    fs_lum, fs_chroma = measure_face_grain(arr)
    s_lum = float(np.sqrt(max(s_lum ** 2 - fs_lum ** 2, 0.0)))
    s_chroma = float(np.sqrt(max(s_chroma ** 2 - fs_chroma ** 2, 0.0)))

    rng = np.random.default_rng(seed)
    grain = _grain_field(h, w, cell_px, s_lum, s_chroma, rng) * amount

    # luminance gate: keep very dark features clean (no speckle on brows/eyes)
    lum = arr @ np.array([0.299, 0.587, 0.114], np.float32)
    lo, hi = dark_gate
    gate = np.clip((lum - lo) / max(hi - lo, 1e-6), 0.0, 1.0)[..., None]

    # de-streak: suppress the selfie's unreliable HF detail in the stretched cheeks and
    # let the isotropic grain replace it. supp in [0,1] per texel; where 0 the output is
    # identical to the plain "arr + grain" path (exact backward compatibility).
    supp = None
    if stretch_fade and _CV2 and stretch_strength > 0 and \
            getattr(face.visual, "uv", None) is not None:
        try:
            from .texture_blender import _rasterize_uv
            sf = stretch_mask_vertex(face)                       # (N,) ~1..>3
            m = _smoothstep(sf, stretch_band[0], stretch_band[1])   # (N,) 0..1
            mcol, cov = _rasterize_uv(np.asarray(face.visual.uv),
                                      np.asarray(face.faces),
                                      np.repeat(m[:, None], 3, axis=1),
                                      np.ones(len(m), np.float32), w, h)
            supp = (mcol[..., 0] * (cov > 0) * stretch_strength)[..., None]
        except Exception:  # noqa: BLE001
            supp = None

    if supp is not None:
        import cv2
        ds = detail_sigma if detail_sigma is not None else max(2.0, 2.0 * cell_px)
        low = cv2.GaussianBlur(arr, (0, 0), sigmaX=ds, sigmaY=ds)
        detail = arr - low
        # gate the suppression by luminance too: never soften the dark identity features
        # (brows/eyes/nostrils, deep moustache) even where they fall in an oblique zone.
        out = low + (1.0 - supp * gate) * detail + grain * gate
    else:
        out = arr + grain * gate

    out = np.clip(out, 0, 255).astype(np.uint8)

    face = face.copy()
    face.visual.material.baseColorTexture = Image.fromarray(out, mode="RGB")
    return face


def clean_edge_fringe(face: trimesh.Trimesh, band_px: int = 8,
                      cell_px: float = 1.3, seed: int = 11) -> trimesh.Trimesh:
    """Replace the mask's contaminated outer texel band with clean skin + fresh grain.

    The raw selfie's UVs at the mask silhouette sit on the face/hair/background edge of
    the photo, so the OUTERMOST ring of the atlas carries a dusky/dark fringe (measured:
    boundary ring median luminance ~135, dipping to 53, vs ~175 just inside). It reads
    as a black liseré along the whole contour, is baked at a fixed UV, and therefore
    survives camera rotation (it is NOT the mesh seam). `feather_border` targets it but
    its triangle rasterisation misses these exact edge texels, so a dark rim survives.

    Two-layer fill so the band reads as real skin, not a stretched smear:
      * LOW frequency (colour): fill each band texel from its nearest *deep interior*
        texel taken from the **blurred** atlas -> a smooth, tone-matched colour with no
        nearest-neighbour streaks (copying the raw pixels would duplicate detail into
        visible smears, which is the stretch that survived the plain fill).
      * HIGH frequency (grain): re-inject **fresh isotropic** speckle whose per-channel
        luminance/chroma amplitude is measured from the adjacent interior skin, so it
        matches the body-matched grain already on the rest of the face seamlessly.

    Runs after `add_matched_grain` on the final upsampled atlas. Falls back to a plain
    nearest-skin fill without OpenCV; no-op without a texture/UV or scipy.
    """
    mat = getattr(face.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) if mat is not None else None
    if img is None or getattr(face.visual, "uv", None) is None:
        return face
    try:
        from scipy.ndimage import distance_transform_edt
        from .texture_blender import _rasterize_uv
    except Exception:  # noqa: BLE001
        return face

    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    H, W = rgb.shape[:2]
    _, cov = _rasterize_uv(np.asarray(face.visual.uv), np.asarray(face.faces),
                           np.zeros((len(face.visual.uv), 3), np.float32),
                           np.ones(len(face.visual.uv), np.float32), W, H)
    covered = cov > 0
    if not covered.any():
        return face

    # depth of each covered texel inside the chart (distance to the nearest edge)
    depth = distance_transform_edt(covered)
    interior = covered & (depth > band_px)
    band = covered & (depth > 0) & (depth <= band_px)
    if not interior.any() or not band.any():
        return face

    idx = distance_transform_edt(~interior, return_distances=False, return_indices=True)

    out = rgb.copy()
    if _CV2:
        import cv2
        lp = max(2.0, 2.0 * cell_px)
        low = cv2.GaussianBlur(rgb, (0, 0), sigmaX=lp, sigmaY=lp)
        lowfill = low[tuple(idx)]                      # smooth colour, no streaks
        # match the interior grain amplitude, then synthesize fresh isotropic grain
        resid = rgb - low
        ri = resid[interior]
        lax = np.array([1, 1, 1], np.float32) / np.sqrt(3.0)
        lum = ri @ lax
        chroma = ri - lum[:, None] * lax[None]
        s_lum = float(lum.std() / np.sqrt(3.0))
        s_chroma = float(np.linalg.norm(chroma, axis=1).std() / np.sqrt(3.0))
        grain = _grain_field(H, W, cell_px, s_lum, s_chroma,
                             np.random.default_rng(seed))
        out[band] = np.clip(lowfill[band] + grain[band], 0, 255)
    else:
        out[band] = rgb[tuple(idx)][band]              # plain nearest-skin fallback

    face = face.copy()
    face.visual.material.baseColorTexture = Image.fromarray(
        np.clip(out, 0, 255).astype(np.uint8), mode="RGB")
    return face


def dilate_uv_gutter(face: trimesh.Trimesh, margin_px: int | None = None) -> trimesh.Trimesh:
    """Bleed the mask's boundary colour outward into the un-covered atlas texels.

    The face atlas is the raw selfie: the texels *just outside* the used UV chart hold
    the dark room background / hair / shadow the selfie captured there. The chart edge
    itself is already body-matched (`feather_border`), but at render time bilinear
    sampling + mipmaps near the boundary reach into those neighbouring dark texels and
    paint a thin black liseré along the seam (measured: covered edge has 0% black, the
    just-outside band drops to luminance 17). This is the classic missing UV gutter.

    We fill every un-covered texel with its nearest covered texel's colour (exact
    nearest-edge fill via the Euclidean distance transform). The chart is unchanged; the
    surroundings — never mapped onto the model — become a skin-coloured gutter, so any
    sampling that spills past the edge blends skin<->skin instead of skin<->black. Run
    LAST, after `add_matched_grain` (so the gutter carries the final, grained edge tone).

    `margin_px` limits the fill to a band that wide around the chart (enough to cover the
    sampler/mip reach); None fills the whole exterior (simplest, and the far texels are
    just never sampled). No-op without a texture/UV or OpenCV/scipy.
    """
    mat = getattr(face.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) if mat is not None else None
    if img is None or getattr(face.visual, "uv", None) is None:
        return face
    try:
        from scipy.ndimage import distance_transform_edt
        from .texture_blender import _rasterize_uv
    except Exception:  # noqa: BLE001
        return face

    rgb = np.asarray(img.convert("RGB"))
    H, W = rgb.shape[:2]
    _, cov = _rasterize_uv(np.asarray(face.visual.uv), np.asarray(face.faces),
                           np.zeros((len(face.visual.uv), 3), np.float32),
                           np.ones(len(face.visual.uv), np.float32), W, H)
    covered = cov > 0
    if not covered.any() or covered.all():
        return face

    # nearest covered texel index for every texel (covered ones map to themselves)
    idx = distance_transform_edt(~covered, return_distances=False, return_indices=True)
    filled = rgb[tuple(idx)]

    fill_where = ~covered
    if margin_px is not None:
        import cv2
        band = cv2.dilate(covered.astype(np.uint8),
                          np.ones((3, 3), np.uint8), iterations=int(margin_px)) > 0
        fill_where &= band

    out = rgb.copy()
    out[fill_where] = filled[fill_where]

    face = face.copy()
    face.visual.material.baseColorTexture = Image.fromarray(out, mode="RGB")
    return face


def match_material_response(face: trimesh.Trimesh, body: trimesh.Trimesh, hf,
                            res: int = 1024, cell_px: float = 1.6,
                            relief_strength: float = 1.0,
                            seed: int = 13) -> trimesh.Trimesh:
    """Give the mask the body skin's PBR light response (roughness + micro-relief).

    relief_strength  amplitude of the synthesized micro-relief NORMAL MAP, as a fraction
        of the body's measured normal-map amplitude. This relief is *lighting* grain over
        the WHOLE face (bumps that break up the highlight) and is independent of the albedo
        grain / `amount`. 1.0 = match the body; 0.0 = flat normal map (no relief grain) while
        STILL transplanting the body roughness (so the face is not dead-matte again).

    The body skin is glossy (roughness ~0.78) and carries a normal map, so it catches and
    breaks up a specular highlight; the mask is roughness 1.0 with NO normal map -> dead
    matte, it only ever shows flat albedo. Under grazing light the body forehead blows to
    a white highlight while the mask keeps its colour, and the mismatch vanishes head-on
    (view-dependent == a material problem, not tone). The old step only copied the scalar
    `roughnessFactor` (1.0 -> a no-op: the body's real roughness lives in its *texture*).

    We transplant both, baked into the mask's own UV:
      * roughness: sampled from the body's metallicRoughnessTexture at the nearest
        body-skin surface point per mask vertex -> a metallicRoughnessTexture (G channel).
      * micro-relief: a synthesized tangent-space normal map whose per-channel amplitude
        matches the body normal map's (measured over head skin), so the highlight glints
        and breaks up like the body's. Skin stays dielectric (metallic 0).

    No-op without OpenCV/scipy, a face UV, or body PBR textures (falls back to leaving the
    material untouched / uniform roughness).
    """
    if not _CV2:
        return face
    fmat0 = getattr(face.visual, "material", None)
    bmat = getattr(body.visual, "material", None)
    if fmat0 is None or bmat is None or getattr(face.visual, "uv", None) is None:
        return face
    try:
        import cv2
        from scipy.spatial import cKDTree
        from .texture_blender import _rasterize_uv
    except Exception:  # noqa: BLE001
        return face

    face = face.copy()
    fmat = face.visual.material
    Fuv = np.asarray(face.visual.uv)
    Ff = np.asarray(face.faces)
    nV = len(face.vertices)
    buv = getattr(body.visual, "uv", None)

    # ---- (1) roughness transplant: nearest body vertex -> body MR.G (0..255) ----
    b_mr = getattr(bmat, "metallicRoughnessTexture", None)
    if b_mr is not None and buv is not None:
        mr = np.asarray(b_mr.convert("RGB"))
        buv = np.asarray(buv)
        _, idx = cKDTree(body.vertices).query(np.asarray(face.vertices))
        bu = np.clip(buv[idx, 0] % 1, 0, 1) * (mr.shape[1] - 1)
        bv = np.clip(1 - (buv[idx, 1] % 1), 0, 1) * (mr.shape[0] - 1)
        rough = mr[np.round(bv).astype(int), np.round(bu).astype(int), 1].astype(np.float32)
    else:
        rough = np.full(nV, 200.0, np.float32)          # ~0.78 fallback
    gcol, cov = _rasterize_uv(Fuv, Ff, np.repeat(rough[:, None], 3, 1),
                              np.ones(nV, np.float32), res, res)
    covered = cov > 0
    mr_img = np.zeros((res, res, 3), np.uint8)
    mr_img[..., 0] = 255                                 # R (unused by metallic-roughness)
    mr_img[..., 1] = np.where(covered, np.clip(gcol[..., 0], 0, 255),
                              float(np.median(rough))).astype(np.uint8)  # G = roughness
    mr_img[..., 2] = 0                                   # B = metallic ~ 0

    # ---- (2) synthesized micro-relief normal map at body amplitude ----
    s_amp = 8.7
    b_n = getattr(bmat, "normalTexture", None)
    if b_n is not None and buv is not None:
        nb = np.asarray(b_n.convert("RGB")).astype(np.float32)
        loc = hf.to_local(body.vertices)
        fr = np.where((loc[:, 2] > 0) & (loc[:, 1] > -0.1 * hf.height))[0]
        Hn, Wn = nb.shape[:2]
        u = np.clip(buv[fr, 0] % 1, 0, 1) * (Wn - 1)
        v = np.clip(1 - (buv[fr, 1] % 1), 0, 1) * (Hn - 1)
        ss = nb[np.round(v).astype(int), np.round(u).astype(int)]
        s_amp = float(0.5 * (ss[:, 0].std() + ss[:, 1].std())) or 8.7
    rng = np.random.default_rng(seed)

    def _bump():
        n = rng.standard_normal((res, res)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigmaX=cell_px / 2.0, sigmaY=cell_px / 2.0)
        return (n / (n.std() + 1e-6)) * s_amp * relief_strength

    nrm_img = np.empty((res, res, 3), np.uint8)
    nrm_img[..., 0] = np.clip(128 + _bump(), 0, 255).astype(np.uint8)
    nrm_img[..., 1] = np.clip(128 + _bump(), 0, 255).astype(np.uint8)
    nrm_img[..., 2] = 255
    nrm_img[~covered] = (128, 128, 255)                  # flat outside the chart

    fmat.metallicRoughnessTexture = Image.fromarray(mr_img)
    fmat.normalTexture = Image.fromarray(nrm_img)
    fmat.roughnessFactor = 1.0                           # G holds the real value
    fmat.metallicFactor = 0.0                            # dielectric skin
    return face
