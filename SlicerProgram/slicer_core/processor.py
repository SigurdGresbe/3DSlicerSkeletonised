# slicer_core/processor.py

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Dict, Optional, Callable, Tuple
from datetime import datetime
import sys
import json
import os

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.interpolate import splev

from .config import SlicerConfig
from .settings_manager import SettingsManager

@dataclass
class PassResult:
    pass_index: int
    tck: Any
    u_all: NDArray
    origins: NDArray
    normals: NDArray
    u_values: NDArray
    slices: List[trimesh.path.path.Path]
    guide_points: NDArray
    pos_dense_centerline: NDArray

from . import mesh as mesh_utils
from . import centerline as cl_utils
from . import slicing as slice_utils
from . import plotting as plot_utils
from . import spline_utils as spline_utils

logger = logging.getLogger(__name__)

class WatertightMeshNeedsPickingError(Exception):
    """Custom exception raised when a watertight mesh requires manual picking."""
    def __init__(self, message="Watertight mesh requires manual picking."):
        self.message = message
        super().__init__(self.message)

class NonPlanarSlicer:
    def __init__(self, settings: SettingsManager, run_time_config: SlicerConfig):
        self.settings = settings
        self.run_config = run_time_config
        self.mesh: trimesh.Trimesh = None
        self.use_2d_transform: bool = False
        
        self.pass_results: List[PassResult] = []

        self.start_opening: Optional[Dict] = None
        self.end_opening: Optional[Dict] = None
        self.plot_file_path: Path = None
        
        try:
            self.run_config.log_dir.mkdir(parents=True, exist_ok=True)
            self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            self.plot_file_path = self.run_config.log_dir / f"{self.run_config.plot_filename_prefix}_{self.run_timestamp}.png"
            
        except Exception as e:
            raise RuntimeError(f"FATAL: Failed to initialize paths: {e}")

    def _validate_settings(self):
        """
        Validates layer heights from settings and saves them back.
        This logic was moved from SlicerConfig.__post_init__.
        """
        nozzle_dia = self.settings.get("nozzle_diameter")
        min_lh_nom = self.settings.get("min_layer_height_nominal")
        max_lh_nom = self.settings.get("max_layer_height_nominal")
        base_lh    = self.settings.get("base_layer_height")

        safe_min = 0.4 * nozzle_dia
        safe_max = 1.0 * nozzle_dia
        
        min_lh = max(min_lh_nom, safe_min)
        max_lh = min(max_lh_nom, safe_max)
        
        if min_lh > max_lh:
            raise ValueError(f"Min layer height ({min_lh:.3f}) > Max layer height ({max_lh:.3f}).")

        self.settings.set('min_layer_height', min_lh)
        self.settings.set('max_layer_height', max_lh)
        self.settings.set('base_layer_height', np.clip(base_lh, min_lh, max_lh))
        
        logger.info(f"Layer Height (Base): {self.settings.get('base_layer_height'):.3f} mm")
        logger.info(f"Layer Height (Min) : {min_lh:.3f} mm (Validated)")
        logger.info(f"Layer Height (Max) : {max_lh:.3f} mm (Validated)")

    def _prepare_boundary_conditions(self, coarse_points: NDArray):
        """
        Identifies start/end openings and orients them using a deterministic
        lexicographical sort (X, then Y, then Z) of the opening centroids.
        """
        openings = self.mesh.opening_data
        if not (openings and len(openings) == 2
                and openings[0]['is_planar'] and openings[1]['is_planar']):
            logger.info("No 2 planar openings found. Spline will not be constrained.")
            self.start_opening, self.end_opening = None, None
            return

        op_a, op_b = openings[0], openings[1]
        
        coord_a = tuple(op_a['centroid'])
        coord_b = tuple(op_b['centroid'])
        
        if coord_a < coord_b:
            start_op, end_op = op_a, op_b
        else:
            start_op, end_op = op_b, op_a
            
        logger.info("Assigning start/end openings based on lexicographical sort (X, Y, Z) of centroids.")
        logger.debug(f"Coord A: {coord_a}")
        logger.debug(f"Coord B: {coord_b}")
        logger.debug(f"Selected Start: {start_op['centroid']}")

        logger.info("Constraining spline to start and end at opening centroids and normals.")

        start_n = start_op['normal'].copy()
        end_n   = end_op['normal'].copy()
        
        geom_epsilon = self.settings.get("geom_epsilon", 1e-6)
        start_n /= np.linalg.norm(start_n) + geom_epsilon
        end_n   /= np.linalg.norm(end_n)   + geom_epsilon

        self.start_opening, self.end_opening = start_op.copy(), end_op.copy()
        self.start_opening['normal'] = start_n
        self.end_opening['normal']   = end_n
    
    def _anchor_planes(self, origins: NDArray, normals: NDArray) -> (NDArray, NDArray):
        """Forces the first and last slice planes to match the mesh openings."""
        if self.start_opening:
            logger.info("Anchoring first slice plane to mesh opening.")
            origins[0], normals[0] = self.start_opening['centroid'], self.start_opening['normal']
            
        if self.end_opening:
            logger.info("Anchoring last slice plane to mesh opening.")
            origins[-1], normals[-1] = self.end_opening['centroid'], self.end_opening['normal']
        return origins, normals

    def _log_config(self):
        logger.info("--- Non-Planar Slicer Config ---")
        logger.info(f"  STL File: {self.run_config.stl_file}")
        logger.info(f"  Log Dir: {self.run_config.log_dir}")
        logger.info(f"  Nozzle: {self.settings.get('nozzle_diameter')} mm")
        logger.info(f"  Curvature Factor: {self.settings.get('curvature_factor')}")
        logger.info(f"  Spline Penalty: {self.settings.get('spline_point_penalty')}")
        logger.info(f"  Spline Degree: {self.settings.get('spline_degree')}")
        logger.info("--------------------------------")

    def run_initial_pass(self, progress_callback: Optional[Callable[[int], None]] = None):
        logger.info("--- Starting Pass 1: Estimating initial centerline ---")
        
        try:
            tck, u = cl_utils.compute_centerline_spline(
                self.mesh, self.settings
            )
        except RuntimeError as e:
            logger.critical(f"Failed to estimate initial centerline: {e}")
            raise e
        
        origins, normals, u_values, _ = slice_utils.generate_adaptive_planes(
            tck, 0.0, 1.0, self.settings, log_details=False
        )
        
        slices = slice_utils.perform_slicing(
            self.mesh, origins, normals, self.use_2d_transform,
            progress_callback=progress_callback
        )
        
        if not slices:
            raise RuntimeError("Pass 1 generated no valid slices.")

        guide_points = slice_utils.extract_centroids_full(slices, self.use_2d_transform)
        pos_dense = np.array(splev(np.linspace(0, 1, self.settings.get("centerline_dense_points")), tck)).T

        self.pass_results.append(PassResult(
            pass_index=1,
            tck=tck,
            u_all=u,
            origins=origins,
            normals=normals,
            u_values=u_values,
            slices=slices,
            guide_points=guide_points,
            pos_dense_centerline=pos_dense
        ))
        logger.info("--- Pass 1 Finished ---")

    def _prepare_guide_points(self) -> NDArray:
        """
        Extracts, clips, and orients the full set of centroids from the
        previous pass to be used as guide points for the refinement pass.
        """
        try:
            full_coarse_points = slice_utils.extract_centroids_full(
                self.pass_results[-1].slices, self.use_2d_transform
            ) 
        except RuntimeError as e:
            logger.critical(f"Fatal error during centroid extraction: {e}. Exiting.") 
            raise e

        logger.debug(f"Pruning and smoothing {len(full_coarse_points)} guide points.")
        full_coarse_points = spline_utils.smooth_and_prune_path(full_coarse_points)

        if not self.start_opening or not self.end_opening:
             raise RuntimeError("_prepare_guide_points called without openings set.")

        dists_start = np.linalg.norm(full_coarse_points - self.start_opening['centroid'], axis=1) 
        dists_end = np.linalg.norm(full_coarse_points - self.end_opening['centroid'], axis=1) 
        start_idx = np.argmin(dists_start) 
        end_idx = np.argmin(dists_end) 
        
        min_idx, max_idx = min(start_idx, end_idx), max(start_idx, end_idx) 
        if max_idx - min_idx >= 2:
            full_coarse_points = full_coarse_points[min_idx : max_idx + 1] 
            logger.debug(f"Clipped full coarse points to indices [{min_idx}, {max_idx}] near openings.") 
        else:
            logger.warning("Clipped points too few; using original full coarse points.") 

        dist_to_start = np.linalg.norm(full_coarse_points[0] - self.start_opening["centroid"]) 
        dist_to_end   = np.linalg.norm(full_coarse_points[0] - self.end_opening["centroid"]) 
        if dist_to_start > dist_to_end:
            logger.debug("Reversing guide_points to match start→end direction.") 
            full_coarse_points = full_coarse_points[::-1] 
        else:
            logger.debug("Guide_points already aligned start→end.")
            
        return full_coarse_points

    def _orient_boundary_normals(self, guide_points: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Checks and flips boundary normals to ensure they point "along" the
        guide point path, returning the corrected normals.
        """
        start_n = self.start_opening['normal']
        end_n = self.end_opening['normal']
        
        if len(guide_points) < 2:
            logger.warning("Too few guide points to orient normals, using raw normals.")
            return start_n, end_n

        geom_epsilon = self.settings.get("geom_epsilon", 1e-6)

        guide_path_dir = guide_points[1] - guide_points[0]
        guide_path_dir /= (np.linalg.norm(guide_path_dir) + geom_epsilon)
        dot_prod = np.dot(start_n, guide_path_dir)
        logger.debug(f"Orienting Start Normal: Dot(mesh_normal, guide_path) = {dot_prod:.4f}")
        if dot_prod < 0:
            logger.warning("Flipping start normal; was misaligned with guide path.")
            self.start_opening['normal'] = -start_n
            start_n = -start_n
            
        guide_path_dir = guide_points[-1] - guide_points[-2]
        guide_path_dir /= (np.linalg.norm(guide_path_dir) + geom_epsilon)
        dot_prod = np.dot(end_n, guide_path_dir)
        logger.debug(f"Orienting End Normal: Dot(mesh_normal, guide_path) = {dot_prod:.4f}")
        if dot_prod < 0:
            logger.warning("Flipping end normal; was misaligned with guide path.")
            self.end_opening['normal'] = -end_n
            end_n = -end_n
            
        return start_n, end_n
        
    def _fit_refined_spline(self, guide_points: NDArray, start_n: NDArray, end_n: NDArray) -> Tuple[Any, NDArray, float, float]:
        """
        Fits the refined spline, checks its direction, and returns
        the spline (tck), parameters (u_all), and start/end (u_start, u_end).
        """
        logger.debug(f"Guide_points[0]: {guide_points[0]}  Guide_points[-1]: {guide_points[-1]}") 
        logger.debug(f"FINAL Start Normal for Spline: {start_n}")
        logger.debug(f"FINAL End Normal for Spline: {end_n}")

        tck, u_all = cl_utils.fit_spline_safe(
            guide_points, self.settings, start_n, end_n
        ) 

        geom_epsilon = self.settings.get("geom_epsilon", 1e-6)
        try:
            t_start = np.array(splev(u_all[0], tck, der=1)) 
            t_start /= (np.linalg.norm(t_start) + geom_epsilon) 
        except Exception as e:
            logger.debug(f"Could not evaluate endpoint tangents: {e}") 
            t_start = None

        if t_start is not None and start_n is not None and np.dot(t_start, start_n) < 0: 
            logger.warning("Spline tangent opposite to start normal; reversing spline direction.")
            tck = (tck[0], [c[::-1] for c in tck[1]], tck[2])
            u_all = 1.0 - u_all[::-1]

        u_start = u_all[1] if start_n is not None else u_all[0] 
        u_end = u_all[-2] if end_n is not None else u_all[-1] 
        logger.debug(f"Clipping spline to parameters u=[{u_start:.4f}, {u_end:.4f}]")
        
        return tck, u_all, u_start, u_end

    def _align_slice_normals(self, normals: NDArray) -> NDArray:
        """Performs a global check to ensure all normals align with the openings."""
        try:
            if self.start_opening and normals.shape[0] > 0:
                dot0 = np.dot(normals[0], self.start_opening['normal']) 
                logger.debug(f"Dot(first normal, start_opening.normal) = {dot0:.6f}") 
                if dot0 < 0:
                    logger.debug("Global flip: first normal disagrees with start opening normal -> flipping all normals.") 
                    return -normals 
                else:
                    logger.debug("Normals agree with start opening normal (no global flip).") 
            
            elif self.end_opening and normals.shape[0] > 0:
                dot_last = np.dot(normals[-1], self.end_opening['normal']) 
                logger.debug(f"Dot(last normal, end_opening.normal) = {dot_last:.6f}") 
                if dot_last < 0:
                    logger.debug("Global flip: last normal disagrees with end opening normal -> flipping all normals.") 
                    return -normals
                    
        except Exception as e:
            logger.debug(f"Failed global-normal alignment check: {e}")
            
        return normals

    def run_refinement_pass(self, progress_callback: Optional[Callable[[int], None]] = None):
        new_pass_index = len(self.pass_results) + 1
        logger.info(f"--- Starting Pass {new_pass_index}: Refining centerline ---")

        if not self.start_opening or not self.end_opening:
            raise RuntimeError("Cannot run refinement pass: Start or end boundaries are not set.")

        full_coarse_points = self._prepare_guide_points()

        optimized_guide_count = spline_utils.optimize_guide_points(
            full_coarse_points, self.start_opening, self.end_opening, self.settings
        ) 
        guide_points = spline_utils.thin_points_for_spline(
            full_coarse_points, optimized_guide_count, self.start_opening, self.end_opening
        ) 
        
        start_n, end_n = self._orient_boundary_normals(guide_points)

        tck, u_all, u_start, u_end = self._fit_refined_spline(guide_points, start_n, end_n)
        
        pos_dense_centerline = np.array(splev(np.linspace(u_start, u_end, self.settings.get("centerline_dense_points")), tck)).T 

        logger.info(f"--- Starting Final Slicing (Pass {new_pass_index}) ---") 
        origins, normals, u_values, _ = slice_utils.generate_adaptive_planes(
            tck, u_start, u_end, self.settings
        ) 

        origins, normals = self._anchor_planes(origins, normals) 
        normals = self._align_slice_normals(normals)

        slices = slice_utils.perform_slicing(
            self.mesh,
            origins,
            normals,
            self.use_2d_transform,
            self.start_opening,
            self.end_opening,
            progress_callback=progress_callback
        ) 

        if not slices:
            raise RuntimeError(f"Pass {new_pass_index} generated no valid slices.") 

        self.pass_results.append(PassResult(
            pass_index=new_pass_index,
            tck=tck,
            u_all=u_all,
            origins=origins,
            normals=normals,
            u_values=u_values,
            slices=slices,
            guide_points=guide_points,
            pos_dense_centerline=pos_dense_centerline
        )) 

        logger.info(f"--- Pass {new_pass_index} Finished Successfully ---") 

    def run(self, progress_callback: Optional[Callable[[int], None]] = None):
        """
        Runs the full 2-pass slicing process.
        - If mesh is not watertight, runs both passes automatically.
        - If mesh is watertight, runs Pass 1 only and throws
          WatertightMeshNeedsPickingError to signal the GUI to start picking.
        """
        try:
            if progress_callback: progress_callback(0)
            
            self._validate_settings()
            self._log_config()
            
            mesh_alpha = self.settings.get("mesh_alpha", 0.25)

            default_mesh_color = np.array(self.settings.get_as_qcolor("mesh_color").getRgbF()[:3]) * 255
            
            self.mesh = mesh_utils.load_and_prepare_mesh(
                self.run_config.stl_file, 
                mesh_alpha,
                default_mesh_color
            )
            self.use_2d_transform = mesh_utils.check_2d_transform(self.mesh)

            pass1_callback = (lambda p: progress_callback(int(p * 0.5))) if progress_callback else None
            pass2_callback = (lambda p: progress_callback(50 + int(p * 0.5))) if progress_callback else None

            self.run_initial_pass(progress_callback=pass1_callback)
            
            if progress_callback: progress_callback(50)

            if self.mesh.is_watertight:
                logger.info("Mesh is watertight. Stopping thread to request manual picking.")
                raise WatertightMeshNeedsPickingError()
            
            logger.info("Mesh is not watertight. Using automatic boundary detection...")
            coarse_points = self.pass_results[0].guide_points
            self._prepare_boundary_conditions(coarse_points)
            
            if progress_callback: progress_callback(60)

            self.run_refinement_pass(progress_callback=pass2_callback)
            
            if progress_callback:
                progress_callback(100)
            
        except WatertightMeshNeedsPickingError:
            raise
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            logger.critical(f"A fatal error occurred: {e}", exc_info=True)
            logger.critical("Slicer failed. Check log for detailed error trace.")
            raise e 
        except Exception as e:
            logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
            logger.critical("Slicer failed. Check log for detailed error trace.")
            raise e 

    def generate_plot(self):
        if not self.pass_results:
            logger.warning("Cannot generate plot, no pass results found.")
            return
            
        try:
            plot_utils.generate_ortho_plot(self, self.plot_file_path)
        except Exception as e:
            logger.critical(f"2D plot generation failed: {e}")