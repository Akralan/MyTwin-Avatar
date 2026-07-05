"""Step 1 — load and sanitise GLB meshes.

Keeps texture/UV visuals intact (needed for the 2-material export).
"""
from __future__ import annotations

import numpy as np
import trimesh


def load_mesh(path_or_obj) -> trimesh.Trimesh:
    """Load a GLB (or accept an already-loaded Trimesh/Scene) and return a single
    Trimesh with visuals preserved.

    Meshy bodies often come as a one-geometry Scene; a face scan may be several
    sub-meshes. `force="mesh"` concatenates them while keeping TextureVisuals
    when a single material is involved.
    """
    if isinstance(path_or_obj, trimesh.Trimesh):
        mesh = path_or_obj
    else:
        loaded = trimesh.load(path_or_obj, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            geoms = list(loaded.geometry.values())
            mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
        else:
            mesh = loaded
    return mesh


def sanitise(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Light repair that does NOT touch UVs/topology destructively.

    We avoid merge_vertices here because it can drop UV seams; Meshy/face scans
    are already indexed cleanly. We only drop degenerate/duplicate faces and fix
    winding/normals.
    """
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_infinite_values()
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def load_body(path) -> trimesh.Trimesh:
    return sanitise(load_mesh(path))


def load_face(path) -> trimesh.Trimesh:
    return sanitise(load_mesh(path))


def has_texture(mesh: trimesh.Trimesh) -> bool:
    return (
        hasattr(mesh.visual, "uv")
        and mesh.visual.uv is not None
        and getattr(mesh.visual, "material", None) is not None
    )
