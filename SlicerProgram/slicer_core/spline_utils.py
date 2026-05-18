# slicer_core/spline_utils.py

import logging
from typing import Any, List, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splev
from scipy.spatial import KDTree 

from .settings_manager import SettingsManager
from .centerline import fit_spline_safe

logger = logging.getLogger(__name__)

def compute_curvature(tck: Any, u_dense: NDArray, geom_epsilon: float) -> NDArray:
    """Compute curvature kappa(s) along spline."""
    vel = np.array(splev(u_dense, tck, der=1)).T
    acc = np.array(splev(u_dense, tck, der=2)).T
    speed = np.linalg.norm(vel, axis=1)
    speed[speed < geom_epsilon] = 1.0
    return np.linalg.norm(np.cross(vel, acc), axis=1) / (speed ** 3)

def _score_geometric_error(
    tck: Any, 
    u_dense: NDArray, 
    original_points: NDArray
) -> float:
    """
    Scores the spline by calculating the mean geometric distance from the
    original guide points to the fitted spline.
    """
    spline_points = np.array(splev(u_dense, tck)).T
    tree = KDTree(spline_points)
    distances, _ = tree.query(original_points, k=1)
    return np.mean(distances)

def smooth_and_prune_path(
    points: NDArray, 
    window_size: int = 5,
    min_fwd_dot: float = -0.5
) -> NDArray:
    """
    Applies a moving average filter to smooth a path and prunes any
    points that represent a self-intersection or "loop".
    """
    if len(points) < window_size:
        return points
        
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_x = np.convolve(points[:, 0], weights, 'same')
    smoothed_y = np.convolve(points[:, 1], weights, 'same')
    smoothed_z = np.convolve(points[:, 2], weights, 'same')
    
    half_win = window_size // 2
    smoothed_x[:half_win] = points[:half_win, 0]
    smoothed_y[:half_win] = points[:half_win, 1]
    smoothed_z[:half_win] = points[:half_win, 2]
    smoothed_x[-half_win:] = points[-half_win:, 0]
    smoothed_y[-half_win:] = points[-half_win:, 1]
    smoothed_z[-half_win:] = points[-half_win:, 2]
    
    smoothed_points = np.vstack([smoothed_x, smoothed_y, smoothed_z]).T

    dirs = np.diff(smoothed_points, axis=0)
    dirs_norm = np.linalg.norm(dirs, axis=1)
    
    pruned_points = [smoothed_points[0]]
    last_good_dir = None
    
    for i in range(len(dirs)):
        current_dir_norm = dirs_norm[i]
        
        if current_dir_norm < 1e-6:
            continue
            
        current_dir = dirs[i] / current_dir_norm
        
        if last_good_dir is None:
            last_good_dir = current_dir
            pruned_points.append(smoothed_points[i+1])
            continue
            
        dot = np.dot(current_dir, last_good_dir)
        
        if dot > min_fwd_dot: 
            pruned_points.append(smoothed_points[i+1])
            last_good_dir = current_dir
        else:
            logger.debug(f"Pruning point {i+1} due to path reversal (dot: {dot:.2f})")
            pass

    final_points = np.array(pruned_points)

    if len(final_points) < 3:
        logger.warning("Path pruning removed too many points, returning original smoothed path.")
        return smoothed_points

    logger.debug(f"Path smoothed and pruned from {len(points)} to {len(final_points)} points.")
    return final_points

def thin_points_for_spline(
    full_points: NDArray,
    num_guides: int,
    start_op: Optional[Dict],
    end_op: Optional[Dict]
) -> NDArray:
    """Select subset of interior points + fixed endpoints."""
    if start_op and end_op:
        n = len(full_points)
        start_idx = n // 10
        end_idx = n - (n // 10)
        
        interior = full_points[start_idx:end_idx]
        if len(interior) == 0:
            logger.debug("No interior points available to thin, using endpoints only.")
            return np.vstack([start_op['centroid'], end_op['centroid']])
            
        num_guides = np.clip(num_guides, 0, len(interior))
        
        if num_guides == 0:
            logger.debug("Requested 0 interior points, using endpoints only.")
            return np.vstack([start_op['centroid'], end_op['centroid']])

        guides = interior[np.linspace(0, len(interior) - 1, num_guides).astype(int)]
        result = np.vstack([start_op['centroid'], guides, end_op['centroid']])
        return result
    
    logger.warning("Thinning points without start/end openings, using full set.")
    return full_points

def optimize_guide_points(
    full_points: NDArray,
    start_opening: Optional[Dict],
    end_opening: Optional[Dict],
    settings: SettingsManager
) -> int:
    """
    Optimize number of interior guide points by minimizing a cost function
    that balances geometric error and the number of points.
    Cost = Geometric_Error + (Point_Penalty * Num_Points)
    """
    if not (start_opening and end_opening):
        logger.warning("Guide optimization skipped: missing opening constraints.")
        return settings.get("num_interior_guide_points", 15)

    logger.info("--- Guide Point Optimization Sweep (Scoring for Cost) ---")
    penalty = settings.get("spline_point_penalty", 0.05)
    logger.info(f"Spline Point Penalty: {penalty:.3f}")
    
    candidates = [0, 1, 2, 3, 5, 8, 12, 15, 20] 
    
    best_cost, best_count = np.inf, settings.get("num_interior_guide_points", 15)
    best_error = np.inf 

    start_n, end_n = start_opening['normal'], end_opening['normal']

    for count in candidates:
        try:
            pts = thin_points_for_spline(full_points, count, start_opening, end_opening)
            
            tck, u_all = fit_spline_safe(pts, settings, start_n, end_n, force_interpolate=True)
            
            u_start_idx = 1 if start_n is not None else 0
            u_end_idx = -2 if end_n is not None else -1
            
            if u_all[u_end_idx] <= u_all[u_start_idx]:
                logger.debug(f"[{count:2d} pts] Invalid spline parameter range, skipping.")
                continue
                
            u_dense = np.linspace(u_all[u_start_idx], u_all[u_end_idx], settings.get("centerline_dense_points"))

            geometric_error = _score_geometric_error(tck, u_dense, full_points)
            
            cost = geometric_error + (penalty * count)
            
            logger.debug(f"[{count:2d} pts] Avg Error: {geometric_error:.4e} mm | Cost: {cost:.4e}")
            
            if cost < best_cost:
                best_cost, best_count, best_error = cost, count, geometric_error
            elif np.isclose(cost, best_cost) and count < best_count:
                best_cost, best_count, best_error = cost, count, geometric_error
                
        except Exception as e:
            logger.debug(f"[{count:2D} pts] Failed: {e}")

    logger.info(f"Optimal guide points: {best_count} (Final Cost: {best_cost:.4e}, Avg Error: {best_error:.4e} mm)")
    return best_count