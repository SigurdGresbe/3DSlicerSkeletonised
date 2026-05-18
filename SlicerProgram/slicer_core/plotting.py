# slicer_core/plotting.py

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, coolwarm

from .spline_utils import compute_curvature

logger = logging.getLogger(__name__)

def generate_ortho_plot(slicer, plot_file_path: Path):
    logger.info("Generating orthographic side views (XZ, YZ)...")

    if not slicer.pass_results:
        logger.error("Cannot generate plot: No pass results found.")
        return
    latest_pass = slicer.pass_results[-1]
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True, gridspec_kw={'width_ratios': [1, 3]})
    bounds = slicer.mesh.bounds
    
    geom_epsilon = slicer.settings.get("geom_epsilon", 1e-6)
    curv_fine = compute_curvature(latest_pass.tck, latest_pass.u_values, geom_epsilon)
    
    curv_min = slicer.settings.get("curv_plot_min", 0.0)
    curv_max = slicer.settings.get("curv_plot_max", 0.04)
    actual_curv_max = np.max(curv_fine)
    if actual_curv_max > curv_max:
        curv_max = actual_curv_max
    
    norm = Normalize(vmin=curv_min, vmax=curv_max)
    if not isinstance(axs, (np.ndarray, list)): axs = [axs]

    slice_extents = {}
    for s in latest_pass.slices:
        layer_index = s.metadata.get('layer')
        if layer_index is None or len(s.vertices) < 2: continue
        
        if hasattr(s, 'extents') and s.extents is not None:
            max_extent = np.max(s.extents)
        else:
            verts = s.vertices
            is_open_contour = len(s.entities) == 1 and not s.entities[0].closed
            
            if is_open_contour or not slicer.mesh.is_watertight:
                max_extent = np.max(slicer.mesh.extents) / 2.0 
            else:
                max_extent = np.max(np.ptp(verts, axis=0))

        slice_extents[layer_index] = max_extent * 1.1
    
    for ax, view in zip(axs, slicer.run_config.views):
        ax.plot(latest_pass.pos_dense_centerline[:, view.u_ax], latest_pass.pos_dense_centerline[:, view.v_ax], 'r-', linewidth=1.5, label='Centerline', alpha=0.7)
        
        is_xz_view = (view.name == 'XZ') 
        
        for i in range(len(latest_pass.origins)):
            current_extent, min_plot_length = slice_extents.get(i, 0.0), 5.0 
            DYNAMIC_LINE_LENGTH = np.maximum(current_extent, min_plot_length)
            HALF_LENGTH = DYNAMIC_LINE_LENGTH / 2.0
            
            origin, normal = latest_pass.origins[i], latest_pass.normals[i]
            
            n_u, n_v = normal[view.u_ax], normal[view.v_ax]
            
            if is_xz_view:
                n_u = 0.0
                
            perp_u, perp_v = -n_v, n_u
            
            mag = np.linalg.norm([perp_u, perp_v])
            if mag < geom_epsilon: perp_u, perp_v = 1.0, 0.0
            else: perp_u /= mag; perp_v /= mag
            
            center_u, center_v = origin[view.u_ax], origin[view.v_ax]
            u1, v1 = center_u - HALF_LENGTH * perp_u, center_v - HALF_LENGTH * perp_v
            u2, v2 = center_u + HALF_LENGTH * perp_u, center_v + HALF_LENGTH * perp_v

            color = coolwarm(norm(curv_fine[i]))
            ax.plot([u1, u2], [v1, v2], color=color, linewidth=0.8, alpha=0.8)

        ax.set_xlabel(f"{view.name[0]} (mm)"); ax.set_ylabel(f"{view.name[1]} (mm)")
        ax.set_title(f"{view.name} View"); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        
        MARGIN_FACTOR = 0.05
        u_min, u_max, v_min, v_max = bounds[:, view.u_ax][0], bounds[:, view.u_ax][1], bounds[:, view.v_ax][0], bounds[:, view.v_ax][1]
        u_range, v_range = u_max - u_min, v_max - v_min
        
        ax.set_xlim(u_min - u_range * MARGIN_FACTOR, u_max + u_range * MARGIN_FACTOR)
        ax.set_ylim(v_min - v_range * MARGIN_FACTOR, v_max + v_range * MARGIN_FACTOR)

    sm = ScalarMappable(cmap=coolwarm, norm=norm)
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.025, pad=0.05)
    cbar.set_label("Curvature (1/mm)")

    model_name = slicer.run_config.stl_file.name
    plt.suptitle(f"Orthographic Projections – Non-Planar Slice Planes | Model: {model_name}")
    
    logger.info(f"Saving orthographic plot to: {plot_file_path}")
    fig.savefig(plot_file_path, dpi=450) 
    plt.close(fig)