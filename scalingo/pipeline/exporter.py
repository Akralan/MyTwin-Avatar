"""Step 6/7 — assemble the final GLB.

Two-material strategy: the body keeps its Meshy atlas, the face keeps its selfie
texture. We export a glTF scene holding both textured meshes (no texture baking,
no blending) — the simplest path with zero seam artefacts in the textures.
"""
from __future__ import annotations

import trimesh


def build_scene(body_holed: trimesh.Trimesh, face_stitched: trimesh.Trimesh) -> trimesh.Scene:
    scene = trimesh.Scene()
    scene.add_geometry(body_holed, geom_name="body")
    scene.add_geometry(face_stitched, geom_name="face")
    return scene


def export_glb(body_holed: trimesh.Trimesh, face_stitched: trimesh.Trimesh,
               out_path: str) -> str:
    scene = build_scene(body_holed, face_stitched)
    scene.export(out_path)
    return out_path
