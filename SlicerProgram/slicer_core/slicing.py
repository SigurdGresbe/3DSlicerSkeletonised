# slicer_core/slicing.py

import logging
from typing import Any, List, Dict, Optional, Callable
import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.interpolate import interp1d, splev

from .settings_manager import SettingsManager
from .spline_utils import compute_curvature

logger = logging.getLogger(__name__)

def generate_adaptive_planes(tck: Any, u_start: float, u_end: float, settings: SettingsManager, log_details: bool = True) -> tuple[NDArray, NDArray, NDArray, float]:
    if log_details:
        logger.info("Computing curvature-adaptive layers...")
    
    geom_epsilon = settings.get("geom_epsilon", 1e-6)
    
    u_dense_for_interp = np.linspace(u_start, u_end, settings.get("centerline_dense_points"))
    pos = np.array(splev(u_dense_for_interp, tck)).T
    seg_lengths = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    arc_dense = np.insert(np.cumsum(seg_lengths), 0, 0.0)
    total_length = arc_dense[-1]

    if total_length < geom_epsilon:
        logger.warning("Total spline length is near zero. Cannot generate adaptive planes.")
        return np.empty((0,3)), np.empty((0,3)), np.empty((0)), 0.0

    u_interp = interp1d(arc_dense, u_dense_for_interp, kind='linear')
    
    kappa_values = compute_curvature(tck, u_dense_for_interp, geom_epsilon)
    curv_interp = interp1d(u_dense_for_interp, kappa_values, kind='linear', fill_value="extrapolate")

    target_arc_steps, current_arc = [0.0], 0.0
    
    base_lh = settings.get("base_layer_height")
    min_lh = settings.get("min_layer_height")
    max_lh = settings.get("max_layer_height")
    curv_factor = settings.get("curvature_factor")
    
    while current_arc < total_length:
        u_current = u_interp(current_arc)
        local_curv = curv_interp(u_current)
        
        step = np.clip(base_lh / (1.0 + curv_factor * local_curv),
                       min_lh, max_lh)
        
        current_arc += step
        if current_arc < total_length:
            target_arc_steps.append(current_arc)

    if not target_arc_steps or total_length - target_arc_steps[-1] > geom_epsilon:
         target_arc_steps.append(total_length)
    else:
        target_arc_steps[-1] = total_length
        
    u_fine = u_interp(np.array(target_arc_steps))
    origins = np.array(splev(u_fine, tck, der=0)).T
    tangents = np.array(splev(u_fine, tck, der=1)).T
    
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < geom_epsilon] = 1.0 
    normals = tangents / norms

    for i in range(1, len(normals)):
        if np.dot(normals[i-1], normals[i]) < 0:
            normals[i] = -normals[i]
            
    if log_details and len(normals) > 1:
        dots = np.einsum('ij,ij->i', normals[:-1], normals[1:])
        min_dot = np.min(dots)
        logger.debug(f"Min dot product between consecutive normals: {min_dot:.4f}")
        if min_dot < -0.5:
            logger.warning(f"Significant normal flip detected (min dot: {min_dot:.4f}); check spline.")

    if log_details:
        logger.info(f"Centerline length ≈ {total_length:.2f} mm")
        logger.info(f"Generated {len(origins)} adaptive layer planes.")
        steps = np.linalg.norm(np.diff(origins, axis=0), axis=1)
        if len(steps) > 0:
            logger.info(f"Layer height stats (mm): Min={np.min(steps):.3f}, Max={np.max(steps):.3f}, Avg={np.mean(steps):.3f}")
        
    return origins, normals, u_fine, total_length

def _filter_best_slice_entity(sec, origin_3d, use_2d_transform: bool):
    """Find the slice entity closest to the given 3D origin point."""
    min_dist, best_entity = np.inf, None
    for entity in sec.entities:
        if len(entity.points) < 3: continue
        verts_array = np.array([tuple(v) for v in sec.vertices[entity.points]])
        
        try:
            if use_2d_transform and hasattr(sec, 'transform'):
                poly = trimesh.path.polygons.Polygon(verts_array)
                cent_2d = poly.centroid
                cent_3d = trimesh.transformations.transform_points(
                    [[cent_2d[0], cent_2d[1], 0]], sec.transform
                )[0]
            else:
                cent_3d = np.mean(verts_array, axis=0)
        except Exception:
            cent_3d = np.mean(verts_array, axis=0)
            
        dist = np.linalg.norm(cent_3d - origin_3d)
        
        if dist < min_dist:
            min_dist, best_entity = dist, entity
            
    return best_entity

def perform_slicing(
    mesh: trimesh.Trimesh, 
    origins: NDArray, 
    normals: NDArray, 
    use_2d_transform: bool,
    start_opening: Optional[Dict] = None, 
    end_opening: Optional[Dict] = None,
    progress_callback: Optional[Callable[[int], None]] = None
) -> List[trimesh.path.path.Path]:

    logger.info(f"Slicing mesh into {len(origins)} layers...")
    slices = []
    
    if progress_callback:
        progress_callback(0)
        
    total = len(origins)
    
    for i, (o, n) in enumerate(zip(origins, normals)):
        
        if i == 0 and start_opening:
            logger.debug("Using pre-computed mesh opening for first slice.")
            sec = start_opening['path'].copy()
            sec.metadata = {'layer': i}
            slices.append(sec)
            continue
            
        if i == total - 1 and end_opening:
            logger.debug("Using pre-computed mesh opening for last slice.")
            sec = end_opening['path'].copy()
            sec.metadata = {'layer': i}
            slices.append(sec)
            continue

        try:
            sec = mesh.section(plane_origin=o.ravel(), plane_normal=n.ravel())
        except Exception as e:
            logger.warning(f"trimesh.section() failed at layer {i}: {e}")
            continue 
        
        if sec is None or not len(sec.vertices) or not len(sec.entities):
            continue

        if len(sec.entities) > 1:
            best_entity = _filter_best_slice_entity(sec, o, use_2d_transform)
            
            if best_entity is not None:
                sec_class = trimesh.path.Path2D if hasattr(sec, 'transform') else trimesh.path.Path3D
                sec = sec_class(entities=[best_entity], vertices=sec.vertices, metadata={'layer': i})
            else:
                logger.warning(f"Layer {i} had multiple entities but none were valid.")
                continue
        else:
             sec.metadata['layer'] = i
        
        slices.append(sec)
            
        if (i + 1) % 100 == 0 or i == total - 1:
            logger.info(f"...slicing progress: {i+1}/{total}")
            if progress_callback:
                progress = int(((i + 1) / total) * 100)
                progress_callback(progress)

    logger.info(f"Generated {len(slices)} valid non-planar contours.")
    return slices

def extract_centroids_full(slices: list, use_2d_transform: bool) -> NDArray:
    """Extracts the 3D centroid from every slice in the list."""
    logger.info("Extracting ALL centroids from slices for optimization.")
    new_center_points = []
    if not slices:
        raise RuntimeError("No slices provided to extract_centroids_full.")

    for sec in slices:
        if not sec.entities:
            logger.warning("Skipping slice with no entities.")
            continue
            
        entity = sec.entities[0]
        if len(entity.points) < 1:
            logger.warning("Skipping slice entity with no points.")
            continue
            
        verts_array = np.array([tuple(v) for v in sec.vertices[entity.points]])
        
        try:
            if use_2d_transform and hasattr(sec, 'transform'):
                cent_2d = trimesh.path.polygons.Polygon(verts_array).centroid
                cent_3d = trimesh.transformations.transform_points([[cent_2d[0], cent_2d[1], 0]], sec.transform)[0]
            else:
                cent_3d = np.mean(verts_array, axis=0)
            new_center_points.append(cent_3d)
        except Exception as e:
            logger.warning(f"Failed to calculate centroid for a slice: {e}")
            pass 
    
    if len(new_center_points) < 3:
        raise RuntimeError(f"Too few valid centroids ({len(new_center_points)}) found from slices.")
        
    logger.info(f"Extracted {len(new_center_points)} full centroid points.")
    return np.array(new_center_points)