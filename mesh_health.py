"""Read-only mesh diagnostics and conservative, explicitly accepted repair copies.

Counts describe the supplied mesh. Edge topology excludes invalid/zero-area
faces; those defects still make the overall watertight result false. Neither
inspection nor repair attempts a self-intersection or physical-scale analysis.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
import logging
from os import PathLike
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


@dataclass(frozen=True)
class MeshHealth:
    vertex_count: int
    face_count: int
    finite_vertex_count: int
    nonfinite_vertex_count: int
    valid_face_count: int
    invalid_face_count: int
    degenerate_face_count: int
    duplicate_face_count: int
    boundary_edge_count: int
    nonmanifold_edge_count: int
    is_watertight: bool
    is_winding_consistent: bool
    messages: list[str]

    def to_dict(self):
        return asdict(self)


def _load(mesh):
    if isinstance(mesh, (str, PathLike)):
        path = Path(mesh)
        if not path.is_file():
            raise ValueError(f"Mesh file does not exist: {path}")
        mesh = trimesh.load(path, force='mesh', process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError(f"No triangle mesh could be loaded from {path}")
    if not isinstance(mesh, (trimesh.Trimesh, o3d.geometry.TriangleMesh)):
        raise TypeError("Expected an Open3D TriangleMesh, Trimesh, or local mesh path.")
    return mesh


def _arrays(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces if isinstance(mesh, trimesh.Trimesh)
                       else mesh.triangles, dtype=np.int64)
    return vertices.reshape(-1, 3), faces.reshape(-1, 3)


def _face_masks(vertices, faces):
    finite = np.isfinite(vertices).all(axis=1)
    valid = ((faces >= 0) & (faces < len(vertices))).all(axis=1)
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices] &= finite[faces[valid_indices]].all(axis=1)
    degenerate = np.zeros(len(faces), dtype=bool)
    valid_indices = np.flatnonzero(valid)
    triangles = vertices[faces[valid_indices]]
    if len(triangles):
        cross = np.cross(triangles[:, 1] - triangles[:, 0],
                         triangles[:, 2] - triangles[:, 0])
        degenerate[valid_indices] = ~np.any(cross != 0, axis=1)
    return finite, valid, degenerate


def inspect_mesh(mesh) -> MeshHealth:
    """Inspect geometry without changing it; local paths return the same report."""
    try:
        mesh = _load(mesh)
        vertices, faces = _arrays(mesh)
        finite, valid, degenerate = _face_masks(vertices, faces)
        usable = faces[valid & ~degenerate]
        duplicate_count = len(usable) - len(np.unique(np.sort(usable, axis=1), axis=0))
        boundary_count = nonmanifold_count = 0
        winding = bool(len(usable))
        if len(usable):
            directed = np.concatenate([usable[:, [0, 1]], usable[:, [1, 2]], usable[:, [2, 0]]])
            _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                            return_inverse=True, return_counts=True)
            boundary_count = int(np.count_nonzero(counts == 1))
            nonmanifold_count = int(np.count_nonzero(counts > 2))
            directions = np.where(directed[:, 0] < directed[:, 1], 1, -1)
            balance = np.bincount(inverse, weights=directions)
            winding = not nonmanifold_count and bool(np.all(balance[counts == 2] == 0))
        invalid_count = int(np.count_nonzero(~valid))
        nonfinite_count = int(np.count_nonzero(~finite))
        degenerate_count = int(np.count_nonzero(degenerate))
        defects = invalid_count + nonfinite_count + degenerate_count + duplicate_count
        watertight = bool(len(usable) and not defects and not boundary_count and not nonmanifold_count)
        messages = []
        if not len(faces):
            messages.append("No triangle faces are available; generate or load a mesh first.")
        if nonfinite_count or invalid_count:
            messages.append(f"{nonfinite_count} vertices and {invalid_count} faces contain invalid geometry; preview cleanup before export.")
        if degenerate_count or duplicate_count:
            messages.append(f"{degenerate_count} zero-area and {duplicate_count} duplicate faces can be removed in a repair preview.")
        if boundary_count:
            messages.append(f"{boundary_count} boundary edges remain open. Repair preview keeps these openings; inspect intended caps or holes.")
        if nonmanifold_count:
            messages.append(f"{nonmanifold_count} edges belong to more than two faces; inspect overlapping surfaces before printing.")
        if len(usable) and not winding:
            messages.append("Face winding is inconsistent; inspect surface orientation in a mesh editor.")
        if watertight and winding:
            messages.append("The mesh is closed with consistent face winding.")
        messages.append("Self-intersections, real-world dimensions and printability are not checked.")
        return MeshHealth(len(vertices), len(faces), int(finite.sum()), nonfinite_count,
                          int(valid.sum()), invalid_count, degenerate_count, duplicate_count,
                          boundary_count, nonmanifold_count, watertight, winding, messages)
    except Exception:
        logging.getLogger(__name__).exception("Mesh inspection failed")
        raise


def repair_preview(mesh):
    """Return a same-type cleanup copy; paths return Trimesh.

Remove invalid, zero-area and duplicate faces, then unused vertices. Preserve
remaining colors/UVs. Do not fill holes, weld vertices, flip faces or save files.
"""
    try:
        mesh = _load(mesh)
        vertices, faces = _arrays(mesh)
        _, valid, degenerate = _face_masks(vertices, faces)
        keep = valid & ~degenerate
        candidates = np.flatnonzero(keep)
        _, first = np.unique(np.sort(faces[candidates], axis=1), axis=0, return_index=True)
        keep[:] = False
        keep[candidates[first]] = True
        if isinstance(mesh, trimesh.Trimesh):
            # Trimesh.copy() can index invalid faces while copying derived color
            # caches. Build only valid geometry before attaching its appearance.
            result = trimesh.Trimesh(vertices=vertices.copy(), faces=faces[keep].copy(),
                                     metadata=deepcopy(mesh.metadata), process=False)
            if mesh.visual.kind == 'texture':
                result.visual = mesh.visual.copy()
                if mesh.visual.face_materials is not None:
                    result.visual.face_materials = np.asarray(mesh.visual.face_materials)[keep].copy()
            elif mesh.visual.kind == 'face':
                result.visual.face_colors = mesh.visual.face_colors[keep].copy()
            elif mesh.visual.kind == 'vertex':
                result.visual.vertex_colors = mesh.visual.vertex_colors.copy()
            result.vertex_attributes = deepcopy(mesh.vertex_attributes)
            result.face_attributes = {key: np.asarray(value)[keep].copy()
                                      for key, value in mesh.face_attributes.items()}
            result.remove_unreferenced_vertices()
        else:
            result = o3d.geometry.TriangleMesh(mesh)
            result.remove_triangles_by_mask((~keep).tolist())
            result.remove_unreferenced_vertices()
            if len(result.triangles):
                result.compute_vertex_normals()
        return result
    except Exception:
        logging.getLogger(__name__).exception("Mesh repair preview failed")
        raise
