# slicer_core/centerline.py

import logging
from typing import Any, List, Dict, Optional, Tuple

import numpy as np
import trimesh
import networkx as nx
from numpy.typing import NDArray
from scipy.interpolate import splev, splprep

from .settings_manager import SettingsManager

logger = logging.getLogger(__name__)

def estimate_centerline(mesh: trimesh.Trimesh) -> NDArray:
    """
    Estimate coarse centerline by finding the shortest path on the mesh
    surface graph between the two largest openings.
    """
    if not mesh.opening_data or len(mesh.opening_data) < 2:
        raise RuntimeError(
            "Centerline estimation requires a mesh with at least 2 openings, "
            f"but {len(mesh.opening_data)} were found."
        )

    openings = sorted(mesh.opening_data, key=lambda op: op['path'].length, reverse=True)
    start_op = openings[0]
    end_op = openings[1]
    logger.info("Using 2 largest openings for surface pathfinding.")

    try:
        start_point_3d = start_op['path'].vertices[start_op['path'].entities[0].points[0]]
        end_point_3d = end_op['path'].vertices[end_op['path'].entities[0].points[0]]
    except (IndexError, AttributeError):
        raise RuntimeError("Invalid opening path data. Cannot find start/end point.")

    all_vertices = mesh.vertices
    start_dist = np.linalg.norm(all_vertices - start_point_3d, axis=1)
    start_node = np.argmin(start_dist)
    
    end_dist = np.linalg.norm(all_vertices - end_point_3d, axis=1)
    end_node = np.argmin(end_dist)

    if start_node == end_node:
        try:
            mid_idx = len(end_op['path'].entities[0].points) // 2
            end_point_3d = end_op['path'].vertices[end_op['path'].entities[0].points[mid_idx]]
            end_dist = np.linalg.norm(all_vertices - end_point_3d, axis=1)
            end_node = np.argmin(end_dist)
        except Exception:
            pass 
        
        if start_node == end_node:
             raise RuntimeError("Could not find two distinct start/end nodes on mesh openings.")

    logger.info(f"Start Node (Mesh Vertex): {start_node}")
    logger.info(f"End Node (Mesh Vertex):   {end_node}")

    logger.info("Building mesh surface graph (vertex_adjacency_graph)...")
    g = mesh.vertex_adjacency_graph
    logger.info(f"Graph has {len(g.nodes)} nodes, {len(g.edges)} edges.")

    try:
        path_nodes = nx.shortest_path(g, source=start_node, target=end_node, weight='weight')
        logger.info(f"Found surface path with {len(path_nodes)} nodes.")
    except nx.NetworkXNoPath:
        raise RuntimeError(f"No surface path found between mesh nodes {start_node} and {end_node}.")
    except Exception as e:
        logger.error(f"Shortest path search failed: {e}")
        raise

    points = mesh.vertices[path_nodes]

    if len(points) < 3:
        raise RuntimeError(f"Need >=3 coarse points, got {len(points)}")

    return points

def _extend_points_for_constraints(
    points: NDArray,
    start_normal: Optional[NDArray],
    end_normal: Optional[NDArray]
) -> NDArray:
    """Add ghost points to enforce tangent constraints at endpoints."""
    final_points = points.copy()

    if start_normal is not None:
        dist = np.linalg.norm(final_points[1] - final_points[0]) * 0.1
        ghost = final_points[0] - start_normal * dist
        final_points = np.vstack([ghost, final_points])

    if end_normal is not None:
        dist = np.linalg.norm(final_points[-1] - final_points[-2]) * 0.1
        ghost = final_points[-1] + end_normal * dist
        final_points = np.vstack([final_points, ghost])

    return final_points

def fit_spline_safe(
    points: NDArray,
    settings: SettingsManager,
    start_normal: Optional[NDArray] = None,
    end_normal: Optional[NDArray] = None,
    force_interpolate: bool = False
) -> Tuple[Any, NDArray]:
    """Fit a B-spline with optional endpoint tangent constraints and jitter."""
    num_orig = len(points)
    is_constrained = start_normal is not None and end_normal is not None
    
    if force_interpolate:
        smoothing = 0
        logger.debug(f"Using forced interpolating spline (s=0) with {num_orig} points.")
    else:
        smoothing = 0 if is_constrained else num_orig * settings.get("spline_smooth_factor", 0.5)

        if is_constrained:
            logger.debug(f"Using interpolating spline (s=0) with {num_orig} points and endpoint constraints.")
        else:
            logger.debug(f"Using smoothing spline (s={smoothing:.2f}) with {num_orig} points.")

    extended_points = _extend_points_for_constraints(points, start_normal, end_normal)
    
    jitter = settings.get("jitter_amount", 1e-9) * (np.random.rand(*extended_points.shape) - 0.5)

    try:
        tck, u = splprep(
            (extended_points + jitter).T, 
            s=smoothing, 
            k=settings.get("spline_degree", 3)
        )
        logger.debug("Spline fit successful.")
        return tck, u
    except Exception as e:
        logger.error(f"Spline fitting failed: {e}", exc_info=True)
        raise RuntimeError("Spline fit failed. Check mesh topology and openings.")

def compute_centerline_spline(
    mesh: trimesh.Trimesh,
    settings: SettingsManager
) -> Tuple[Any, NDArray]:
    """
    High-level function to compute smooth centerline spline based on
    a shortest-path surface trace.
    """
    logger.info("Estimating centerline from mesh surface path...")
    coarse_points = estimate_centerline(mesh)
    logger.info(f"{len(coarse_points)} coarse surface points -> fitting spline")

    tck, _ = fit_spline_safe(coarse_points, settings, start_normal=None, end_normal=None, force_interpolate=True)
    
    u_dense = np.linspace(0, 1, settings.get("centerline_dense_points", 10000))
    return tck, u_dense