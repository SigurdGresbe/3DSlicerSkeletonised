# slicer_core/mesh.py

import logging
from pathlib import Path
from typing import Any, List, Dict
import numpy as np
import trimesh
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

def get_boundary_data(mesh: trimesh.Trimesh) -> List[Dict[str, Any]]:
    openings_path = mesh.outline(raise_error=False)
    
    if openings_path is None or not hasattr(openings_path, 'entities') or len(openings_path.entities) == 0:
        logger.info("No mesh openings (boundaries) found.")
        return []

    openings = openings_path.entities
    logger.info(f"Found {len(openings)} mesh openings (boundaries).")
    
    opening_data = []
    for i, boundary_entity in enumerate(openings):
        entity_vertex_indices = boundary_entity.points
        entity_vertices = openings_path.vertices[entity_vertex_indices]
        centroid = entity_vertices.mean(axis=0)

        local_indices = np.arange(len(entity_vertices))
        if boundary_entity.closed:
            local_indices = np.append(local_indices, local_indices[0])
        
        new_entity = trimesh.path.entities.Line(local_indices)
        boundary_path = trimesh.path.Path3D(entities=[new_entity], vertices=entity_vertices)
        
        is_planar = False
        origin, normal = None, None

        try:
            path_2d, to_2d_transform = boundary_path.to_2D()
            is_planar = True
            from_2d_transform = np.linalg.inv(to_2d_transform)
            normal = from_2d_transform[2, :3]
            origin = boundary_path.vertices[0]
            logger.info(f"Opening {i} at [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}] (Planar)")
        except ValueError:
            is_planar = False
            logger.info(f"Opening {i} at [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}] (Non-Planar)")
            
        opening_data.append({
            "centroid": centroid, "is_planar": is_planar, "origin": origin, "normal": normal, "path": boundary_path
        })
    return opening_data

def load_and_prepare_mesh(stl_file: Path, mesh_alpha: float, default_mesh_color: NDArray) -> trimesh.Trimesh:
    if not stl_file.is_file():
        logger.error(f"STL file not found: '{stl_file}'")
        raise FileNotFoundError(f"'{stl_file}' not found.")
        
    logger.info(f"Loading STL: {stl_file}")
    mesh = trimesh.load(stl_file)
    logger.info(f"Mesh Bounds (min/max):\n{mesh.bounds}")
    logger.info(f"Mesh Extents (size): {mesh.extents}")
    logger.info(f"Is watertight: {mesh.is_watertight}")
    
    if not mesh.is_watertight:
        logger.warning("Mesh is not watertight. Slicing may produce errors.")
    
    mesh.opening_data = get_boundary_data(mesh)

    if hasattr(mesh.visual, 'material'):
        mesh.visual.material.baseColorFactor = (
            *mesh.visual.material.baseColorFactor[:3], mesh_alpha
        )
    else:
        color = np.append(default_mesh_color, int(255 * mesh_alpha))
        mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=color)
    
    return mesh

def check_2d_transform(mesh: trimesh.Trimesh) -> bool:
    _test_sec = mesh.section(plane_origin=mesh.centroid, plane_normal=[0,0,1])
    return (_test_sec is not None and hasattr(_test_sec, 'transform'))