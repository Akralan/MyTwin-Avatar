"""End-to-end pipeline: body GLB + face GLB -> final avatar GLB.

Two placement strategies:
  * "landmark" (intelligent, default): MediaPipe detects the body's facial
    landmarks; the scan (canonical 478-vertex topology) is aligned feature-to-
    feature, and the hole is cut to the scan's own silhouette.
  * "geometric" (fallback): pure-geometry contour fit, used when MediaPipe or the
    model file is unavailable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import (loader, head_locator, mesh_surgery, aligner, stitcher, exporter,
               landmark_detector, texture_blender, junction, texture_grain)


@dataclass
class PipelineReport:
    method: str
    forward: tuple
    facing_confidence: float
    removed_faces: int
    hole_ring_points: int
    align_scale: float
    landmark_residual: float | None
    processing_time_ms: int
    warnings: list = field(default_factory=list)


# Uniform subdivisions applied to the mask AFTER alignment: more angular resolution
# around the perimeter so the pinned outer ring is a finer polygon and the junction
# festons shrink. The junction ring counts are scaled by 2**this to keep the same
# physical band (see junction.densify / the tangent_merge call below).
_SUBDIV_ITERS = 1


# MediaPipe canonical OUTER eye-corner indices (subject's right / left).
_EYE_R, _EYE_L = 33, 263


def _eye_forward(body3d, valid, up, ref_forward):
    """Facing direction from the landmark eye-corner axis (robust to hair, unlike the
    geometric nose protrusion). Returns None if either eye corner is missing."""
    if not (valid[_EYE_R] and valid[_EYE_L]):
        return None
    right = body3d[_EYE_L] - body3d[_EYE_R]
    right = right - (right @ up) * up
    nr = np.linalg.norm(right)
    if nr < 1e-9:
        return None
    right = right / nr
    f = np.cross(right, up)
    f = f / (np.linalg.norm(f) + 1e-12)
    if f @ ref_forward < 0:                 # keep the same front/back sign
        f = -f
    return f


def _yaw_deg(a, b, up):
    a = a - (a @ up) * up; a = a / (np.linalg.norm(a) + 1e-12)
    b = b - (b @ up) * up; b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arctan2(np.cross(a, b) @ up, a @ b)))


def _refine_head_frame(body, hf, warnings, max_iter: int = 2, tol_deg: float = 3.0):
    """Close the loop head_locator <-> MediaPipe: bootstrap with the geometric facing,
    then re-derive `forward` from the eye axis and re-render until it stops moving.
    Fixes the facing error the geometric estimator makes on voluminous/asymmetric hair
    (which biases the nose-protrusion cue). Returns (hf, body3d, valid) — the frame and
    the landmarks detected in that final frame."""
    up = hf.up / (np.linalg.norm(hf.up) + 1e-12)
    body3d, valid = landmark_detector.detect_body_landmarks(body, hf)
    iters = 0
    for _ in range(max_iter):
        ef = _eye_forward(body3d, valid, up, hf.forward)
        if ef is None or abs(_yaw_deg(hf.forward, ef, up)) < tol_deg:
            break
        hf_new = head_locator.reframe(body, hf, ef)
        try:
            body3d, valid = landmark_detector.detect_body_landmarks(body, hf_new)
        except landmark_detector.LandmarksUnavailable:
            break                            # keep the last consistent frame/landmarks
        hf = hf_new
        iters += 1
    if iters:
        warnings.append(f"head facing refined via landmarks ({iters} iter)")
    return hf, body3d, valid


def _landmark_path(body, face, hf, warnings, subdiv: int = _SUBDIV_ITERS,
                   side_trim_after: int = 0):
    """Intelligent: align by facial landmarks, then cut to the scan silhouette."""
    hf, body3d, valid = _refine_head_frame(body, hf, warnings)
    al = aligner.align_by_landmarks(face, body3d, valid)

    # Trim 1 ring of the mask's lateral sides (the part curling toward the ears) so
    # it does not overhang the body there -> avoids the side overlap artefacts.
    trimmed = junction.trim_sides(al.face, al.face_ring, hf, n_rings=1)
    # Densify the (canonical-topology) trimmed mask now that alignment is done, then
    # recompute the geometric silhouette ring on the finer mesh.
    trimmed = junction.densify(trimmed, iterations=subdiv)
    # Optional extra lateral trim AFTER subdivision: one *fine* ring off the sides
    # (post-densify, so a much thinner physical band than the pre-densify trim above).
    # Pulls the seam in off the stretched lateral cheeks -> the body skin covers the
    # smeared MediaPipe cheek edge. Experimental knob; 0 keeps the previous behaviour.
    if side_trim_after > 0:
        ring_fine = aligner.face_silhouette_ring(trimmed)
        trimmed = junction.trim_sides(trimmed, ring_fine, hf, n_rings=side_trim_after)
    al = aligner.AlignResult(face=trimmed,
                             face_ring=aligner.face_silhouette_ring(trimmed),
                             transform=al.transform)

    idx = [i for i in range(468) if valid[i]]
    pred = (al.transform[:3, :3] @ face.vertices[idx].T).T + al.transform[:3, 3]
    residual = float(np.linalg.norm(pred - body3d[idx], axis=1).mean())

    sil = al.face.vertices[al.face_ring]
    # widen the hole on the lateral sides (less overlap toward the ears), keep the
    # forehead erosion at the default, and enlarge the hole slightly at the chin
    cut = mesh_surgery.cut_to_silhouette(body, hf, sil, inflate_side=0.96,
                                         inflate_bottom=0.92)
    return al, cut, residual, hf


def _geometric_path(body, face, hf, warnings, **cut_kwargs):
    """Fallback: cut a heuristic ellipse, then contour-fit the scan."""
    cut = mesh_surgery.cut_face(body, hf, **cut_kwargs)
    hole_pts = cut.body.vertices[cut.hole_loop]
    al = aligner.align_face(face, hf, hole_pts)
    return al, cut, None


def replace_face(body_glb, face_glb, out_path: str, force_geometric: bool = False,
                 match_tone: bool = True, match_grain: bool = True,
                 width_scale: float = 0.34, height_scale: float = 0.44,
                 front_offset: float = 0.30, v_center_shift: float = -0.07,
                 side_trim_after: int = 0):
    """Run the full replacement and write `out_path`. Returns (path, report)."""
    t0 = time.time()
    warnings: list[str] = []

    body = loader.load_body(body_glb)
    face = loader.load_face(face_glb)

    hf = head_locator.locate_head(body)

    method = "geometric"
    residual = None
    if not force_geometric and landmark_detector.mediapipe_available():
        try:
            al, cut, residual, hf = _landmark_path(body, face, hf, warnings,
                                                   side_trim_after=side_trim_after)
            method = "landmark"
        except Exception as e:  # noqa: BLE001
            warnings.append(f"landmark path failed ({e}); fell back to geometric")
            al, cut, residual = _geometric_path(
                body, face, hf, warnings, width_scale=width_scale,
                height_scale=height_scale, front_offset=front_offset,
                v_center_shift=v_center_shift)
    else:
        if not force_geometric:
            warnings.append("mediapipe/model unavailable; using geometric placement")
        al, cut, residual = _geometric_path(
            body, face, hf, warnings, width_scale=width_scale,
            height_scale=height_scale, front_offset=front_offset,
            v_center_shift=v_center_shift)

    hole_pts = cut.body.vertices[cut.hole_loop]
    if len(cut.hole_loop) < 12:
        warnings.append("hole rim has very few points; stitching may be poor")

    if method == "landmark":
        # Junction reconstruction: roll the (side-trimmed) outer band onto the body
        # and rotate its gradient field toward the body normals -> C1 join, so the
        # chin/forehead crease disappears under variable light. Falls back to the
        # proud overlap if the solve fails.
        try:
            f = 2 ** _SUBDIV_ITERS      # keep the same physical band on the finer mask
            stitched = junction.tangent_merge(al.face, cut.body, hf, al.face_ring,
                                              band_rings=6 * f, body_pin_rings=1 * f,
                                              mode="gradient")
        except Exception as e:  # noqa: BLE001
            warnings.append(f"junction merge failed ({e}); used proud overlap")
            stitched = stitcher.place_overlap(al.face, hf)
        # Glue the lip: press the body's hole rim back onto the mask surface wherever
        # it overhangs, so the cut edge stops standing proud of the scan (the lit lip).
        try:
            cut.body = junction.glue_rim(cut.body, stitched, cut.hole_loop, hf)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"rim glue skipped ({e})")
    else:
        stitched = stitcher.stitch(al.face, al.face_ring, hole_pts)

    if match_tone:
        try:
            stitched = texture_blender.match_skin_tone(stitched, body, hf)
            # Fix the residual *spatial* tint step (forehead / one cheek) that a single
            # global shift cannot: diffuse a contour-matched Lab offset across the mask.
            if method == "landmark":
                stitched = texture_blender.harmonic_tone_match(
                    stitched, body, al.face_ring)
            stitched = texture_blender.feather_border(stitched, body, hf)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"tone match skipped ({e})")

    if match_grain:
        # Give the smooth scan the body's skin micro-grain so the inserted face and
        # the granular body read as one material at the junction (amplitude/frequency
        # matched to the body albedo; see pipeline.texture_grain). Texture-only.
        try:
            # wavelength_scale<1 makes the grain finer than the body (aesthetic choice):
            # 1.0 matches the body exactly, but that reads a touch coarse on the face, so
            # we bias ~30% finer. Tune here if the grain still looks too big/small.
            stitched = texture_grain.add_matched_grain(stitched, body, hf,
                                                       wavelength_scale=1.0)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"grain match skipped ({e})")

    # Kill the dark liseré: the mask's outermost atlas ring carries the selfie's
    # face/hair/background edge (dusky, baked at a fixed UV -> survives camera rotation).
    # Overwrite that contaminated band with the adjacent clean interior skin.
    try:
        stitched = texture_grain.clean_edge_fringe(stitched)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"edge fringe clean skipped ({e})")

    # UV gutter: bleed the (now-clean) chart edge outward over the un-covered selfie
    # background texels, so bilinear/mip sampling at the seam can never reach the dark
    # background either. Runs last so it carries the final edge tone.
    try:
        stitched = texture_grain.dilate_uv_gutter(stitched)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"uv gutter skipped ({e})")

    if method == "landmark":
        # Method 2 — match the face material's light response to the body's, so identical
        # geometry catches light identically at the seam. The body skin is glossy (rough
        # ~0.78) with a normal map; the raw mask is matte roughness 1.0 with none, so it
        # reads flat under grazing light while the body highlights. Transplant the body's
        # roughness + a matched micro-relief normal map into the mask's UV.
        try:
            # relief_strength = amplitude of the micro-relief normal map = the LIGHTING
            # grain spread over the whole face (independent of the albedo grain / `amount`).
            # It was the dominant visible grain; 0.0 = flat normal map (no relief grain,
            # roughness transplant kept), 1.0 = full body match. Dial up (~0.3-0.5) if you
            # want some highlight break-up back.
            stitched = texture_grain.match_material_response(stitched, body, hf,
                                                             relief_strength=0.65)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"material match skipped ({e})")

        # Method 1 — share averaged normals along the seam (no vertex moved) and export
        # explicit normals on BOTH meshes so shading is continuous across the junction.
        try:
            cut.body, stitched = junction.weld_normals(
                cut.body, stitched, cut.hole_loop, al.face_ring, hf)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"normal weld skipped ({e})")

    exporter.export_glb(cut.body, stitched, out_path)

    scale = float((al.transform[:3, 0] ** 2).sum() ** 0.5)
    report = PipelineReport(
        method=method,
        forward=tuple(round(float(x), 3) for x in hf.forward),
        facing_confidence=round(hf.confidence, 3),
        removed_faces=cut.removed_faces,
        hole_ring_points=len(cut.hole_loop),
        align_scale=round(scale, 4),
        landmark_residual=round(residual, 5) if residual is not None else None,
        processing_time_ms=int((time.time() - t0) * 1000),
        warnings=warnings,
    )
    return out_path, report
