# slicer_gui_visualization.py

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import pyvista as pv
import numpy as np
import trimesh
from PyQt5.QtGui import QColor

from slicer_core.processor import NonPlanarSlicer, PassResult
from slicer_core.spline_utils import compute_curvature
from slicer_core.settings_manager import SettingsManager 

logger = logging.getLogger(__name__)

@dataclass
class ViewOptions:
    pass_to_show: int
    show_mesh: bool
    show_slices: bool
    show_centerline: bool
    show_guide_points: bool
    show_layer_normal: bool
    show_cross_section: bool
    vis_skip_layers: int
    slice_alpha: float
    current_layer_index: int

class PyVistaManager:    
    def __init__(self, plotter: pv.Plotter):
        self.plotter = plotter
        self.slicer: Optional[NonPlanarSlicer] = None
        self.settings: Optional[SettingsManager] = None
        self.last_view_options: Optional[ViewOptions] = None
        
        self.slice_actors: Dict[int, List[Tuple[int, pv.Actor]]] = {}
        self.mesh_actors: Dict[int, List[Tuple[int, pv.Actor]]] = {}
        self.centerline_actors: Dict[int, List[Tuple[int, pv.Actor]]] = {}
        self.guide_point_actors: Dict[int, List[Tuple[int, pv.Actor]]] = {}
        
        self.global_stats_actor: Optional[pv.Actor] = None
        self.layer_stats_actor: Optional[pv.Actor] = None
        self.normal_vector_actor: Optional[pv.Actor] = None
        self.vertex_actor: Optional[pv.Actor] = None
        self.clipped_mesh_actor: Optional[pv.Actor] = None
        self.clipped_centerline_actor: Optional[pv.Actor] = None
        
        self.pass_stats_actor: Optional[pv.Actor] = None
        self.complexity_stats_actor: Optional[pv.Actor] = None

    def _clear_all_actors(self):
        self.plotter.clear()
        self.slice_actors = {}
        self.mesh_actors = {}
        self.centerline_actors = {}
        self.guide_point_actors = {}
        
        if self.layer_stats_actor: self.plotter.remove_actor(self.layer_stats_actor)
        if self.normal_vector_actor: self.plotter.remove_actor(self.normal_vector_actor)
        if self.vertex_actor: self.plotter.remove_actor(self.vertex_actor)
        if self.clipped_mesh_actor: self.plotter.remove_actor(self.clipped_mesh_actor)
        if self.clipped_centerline_actor: self.plotter.remove_actor(self.clipped_centerline_actor)
        
        if self.pass_stats_actor: self.plotter.remove_actor(self.pass_stats_actor)
        self.pass_stats_actor = None
        
        self.layer_stats_actor = None
        self.normal_vector_actor = None
        self.vertex_actor = None
        self.clipped_mesh_actor = None
        self.clipped_centerline_actor = None

    def _clear_all_slicer_actors(self):
        self._clear_all_actors()
        if self.global_stats_actor: self.plotter.remove_actor(self.global_stats_actor)
        self.global_stats_actor = None
        
        if self.complexity_stats_actor: self.plotter.remove_actor(self.complexity_stats_actor)
        self.complexity_stats_actor = None

    def preview_mesh(self, mesh: trimesh.Trimesh, alpha: float, color: QColor):
        self._clear_all_slicer_actors() 
        self.slicer = None 

        if self.settings:
            self.plotter.background_color = self.settings.get_as_qcolor("bg_color").getRgbF()[:3]
        
        pv_mesh = pv.wrap(mesh)
        mesh_rgb = color.getRgbF()[:3]
        mesh_actor = self.plotter.add_mesh(
            pv_mesh, color=mesh_rgb, opacity=alpha,
            show_edges=True, name='preview_mesh'
        )
        
        self.plotter.show_axes()
        self.plotter.reset_camera()

    def set_slicer_object(self, slicer: NonPlanarSlicer):
        self.slicer = slicer
        
        self.plotter.suppress_rendering = True
        
        self._clear_all_slicer_actors() 
        
        if not self.slicer or not self.settings:
            self.plotter.suppress_rendering = False
            return
            
        self.plotter.background_color = self.settings.get_as_qcolor("bg_color").getRgbF()[:3]
        font_size = self.settings.get("overlay_font_size", 10)
        
        try:
            latest_pass = slicer.pass_results[-1]
            
            total_slices = len(latest_pass.slices)
            centerline_length = 0.0
            if hasattr(latest_pass, 'pos_dense_centerline') and latest_pass.pos_dense_centerline is not None and latest_pass.pos_dense_centerline.shape[0] > 1:
                centerline_length = np.sum(np.linalg.norm(np.diff(latest_pass.pos_dense_centerline, axis=0), axis=1))

            guide_point_count = len(latest_pass.guide_points)
            ext = slicer.mesh.extents
            stats_str = (
                f"Global Stats (Pass {latest_pass.pass_index})\n"
                f"-----------------\n"
                f"Total Slices: {total_slices}\n"
                f"Guide Points: {guide_point_count}\n"
                f"Centerline: {centerline_length:.2f} mm\n"
                f"Extents (mm): {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f}"
            )
            
            text_color = self.settings.get_as_qcolor("overlay_font_color").getRgbF()[:3]
            
            self.global_stats_actor = self.plotter.add_text(
                stats_str, 
                position='upper_right',
                font_size=font_size, 
                color=text_color, 
                name='global_stats'
            )
            
            v_count = slicer.mesh.vertices.shape[0]
            f_count = slicer.mesh.faces.shape[0]
            complexity_str = (
                f"Mesh Complexity\n"
                f"-----------------\n"
                f"Vertices: {v_count:,}\n"
                f"Faces: {f_count:,}"
            )
            self.complexity_stats_actor = self.plotter.add_text(
                complexity_str, 
                position='lower_right', 
                font_size=font_size, 
                color=text_color, 
                name='complexity_stats'
            )
            
        except Exception as e:
            logging.warning(f"Could not generate global stats: {e}")
        
        for pass_res in self.slicer.pass_results:
            self.build_pass_visualization(pass_res.pass_index)
        
        self.plotter.show_axes()
        
        self.plotter.suppress_rendering = False

    def add_new_pass_visualization(self, pass_index: int):
        if not self.slicer or not self.settings:
            return
        logger.info(f"Building visualization for new Pass {pass_index}.")
        
        self.plotter.suppress_rendering = True
        self.build_pass_visualization(pass_index)
        self.plotter.suppress_rendering = False

    def update_appearance(self, settings: SettingsManager):
        self.settings = settings
        self.plotter.background_color = self.settings.get_as_qcolor("bg_color").getRgbF()[:3]
        if self.slicer:
            self.set_slicer_object(self.slicer)
            if self.last_view_options:
                self.refresh_view(self.last_view_options)

    def build_pass_visualization(self, pass_index: int):
        if not self.slicer or not self.settings:
            return
        
        try:
            pass_data = self.slicer.pass_results[pass_index - 1]
            if pass_data.pass_index != pass_index:
                logger.error(f"Pass index mismatch! Expected {pass_index}, got {pass_data.pass_index}")
                return
        except IndexError:
            logger.error(f"Could not find pass data for index {pass_index}")
            return

        slicer = self.slicer
        
        pv_mesh = pv.wrap(slicer.mesh)
        mesh_rgb = self.settings.get_as_qcolor("mesh_color").getRgbF()[:3]
        
        mesh_alpha = self.settings.get("mesh_alpha", 0.25)
        
        mesh_actor = self.plotter.add_mesh(
            pv_mesh, color=mesh_rgb, opacity=mesh_alpha,
            show_edges=True, name=f'mesh_pass_{pass_index}'
        )
        
        mesh_actor.SetVisibility(False) 
        
        self.mesh_actors[pass_index] = mesh_actor

        if pass_data.pos_dense_centerline is not None:
            cl = pass_data.pos_dense_centerline
            if len(cl) > 1:
                cl_path = trimesh.load_path(cl)
                entity = cl_path.entities[0]
                pts = cl_path.vertices[entity.points]
                if entity.closed and len(pts) > 0 and np.allclose(pts[0], pts[-1]):
                    pts = pts[:-1]
                poly = pv.PolyData(pts)
                n = len(pts)
                poly.lines = np.hstack([n + 1, np.arange(n), 0]) if entity.closed else np.hstack([n, np.arange(n)])
                
                centerline_rgb = self.settings.get_as_qcolor("centerline_color").getRgbF()[:3]
                
                centerline_actor = self.plotter.add_mesh(
                    poly, color=centerline_rgb, 
                    line_width=self.settings.get("centerline_thickness", 4),
                    render_lines_as_tubes=True,
                    style='wireframe', name=f'centerline_pass_{pass_index}'
                )
                centerline_actor.SetVisibility(False) 
                
                self.centerline_actors[pass_index] = centerline_actor

        if pass_data.guide_points is not None and len(pass_data.guide_points) > 0:
            points_poly = pv.PolyData(pass_data.guide_points)
            point_color = self.settings.get_as_qcolor("guide_color").getRgbF()[:3]
            
            guide_actor = self.plotter.add_points(
                points_poly, color=point_color,
                point_size=10, name=f'guide_points_pass_{pass_index}'
            )
            guide_actor.SetVisibility(False)
            self.guide_point_actors[pass_index] = guide_actor

        skip = self.settings.get("vis_skip_layers", 10)
        
        slice_rgb = self.settings.get_as_qcolor("slice_color_normal").getRgbF()[:3]
        pass_slice_actors: List[Tuple[int, pv.Actor]] = [] 
        
        all_slices = pass_data.slices
        if not all_slices:
            self.slice_actors[pass_index] = []
            return 

        indices_to_show = set(range(0, len(all_slices), skip))
        indices_to_show.add(len(all_slices) - 1) 
        
        for global_index in sorted(list(indices_to_show)):
            path = all_slices[global_index]
            
            if not (hasattr(path, 'vertices') and path.vertices.shape[0]):
                continue
            entity = path.entities[0]
            pts_idx = entity.points
            closed = entity.closed
            if closed and pts_idx[0] == pts_idx[-1]:
                pts_idx = pts_idx[:-1]
            pts = path.vertices[pts_idx]
            if len(pts) < 2:
                continue

            poly = pv.PolyData(pts)
            n = len(pts)
            poly.lines = np.hstack([n + 1, np.arange(n), 0]) if closed else np.hstack([n, np.arange(n)])

            actor = self.plotter.add_mesh(
                poly, color=slice_rgb, opacity=1.0,
                line_width=self.settings.get("slice_thickness_normal", 1),
                style='wireframe', name=f'slice_pass_{pass_index}_{global_index}'
            )
            actor.SetVisibility(False) 
            
            pass_slice_actors.append((global_index, actor)) 
            
        self.slice_actors[pass_index] = pass_slice_actors

    def refresh_view(self, options: ViewOptions):
        if not self.slicer or not self.settings:
            return
            
        self.last_view_options = options 

        pass_index = options.pass_to_show

        for p_idx, actor in self.mesh_actors.items():
            if p_idx != pass_index: actor.SetVisibility(False)
        for p_idx, actor in self.centerline_actors.items():
            if p_idx != pass_index: actor.SetVisibility(False)
        for p_idx, actor_list in self.slice_actors.items():
            if p_idx != pass_index:
                for _, actor in actor_list: actor.SetVisibility(False) 
        for p_idx, actor in self.guide_point_actors.items():
            if p_idx != pass_index: actor.SetVisibility(False)

        if pass_index not in self.mesh_actors:
            logger.warning(f"Pass {pass_index} not visualized, building now.")
            self.build_pass_visualization(pass_index)
        
        current_mesh_actor = self.mesh_actors.get(pass_index)
        current_centerline_actor = self.centerline_actors.get(pass_index)
        current_guide_point_actor = self.guide_point_actors.get(pass_index)
        current_slice_actors = self.slice_actors.get(pass_index, []) 
        
        try:
            pass_data = self.slicer.pass_results[pass_index - 1]
        except (IndexError, TypeError):
            logger.error(f"Cannot refresh: No data for pass {pass_index}")
            return

        total = len(pass_data.slices)
        
        vis_skip_layers = options.vis_skip_layers
        
        current_skip_in_actors = 1
        if len(current_slice_actors) > 2:
            idx1 = current_slice_actors[0][0]
            idx2 = current_slice_actors[1][0]
            current_skip_in_actors = idx2 - idx1
        
        if vis_skip_layers != current_skip_in_actors:
            logging.info(f"Rebuilding pass {pass_index} (skip changed to {vis_skip_layers}).")
            self.settings.set("vis_skip_layers", vis_skip_layers)
            
            self.plotter.suppress_rendering = True
            if current_mesh_actor: self.plotter.remove_actor(current_mesh_actor)
            if current_centerline_actor: self.plotter.remove_actor(current_centerline_actor)
            if current_guide_point_actor: self.plotter.remove_actor(current_guide_point_actor)
            for _, actor in current_slice_actors: self.plotter.remove_actor(actor) 
            
            self.build_pass_visualization(pass_index)
            self.plotter.suppress_rendering = False
            
            current_mesh_actor = self.mesh_actors.get(pass_index)
            current_centerline_actor = self.centerline_actors.get(pass_index)
            current_guide_point_actor = self.guide_point_actors.get(pass_index)
            current_slice_actors = self.slice_actors.get(pass_index, [])
            
        if current_mesh_actor:
            current_mesh_actor.SetVisibility(options.show_mesh and not options.show_cross_section)
        if current_centerline_actor:
            current_centerline_actor.SetVisibility(options.show_centerline and not options.show_cross_section)
        if current_guide_point_actor:
            current_guide_point_actor.SetVisibility(options.show_guide_points and not options.show_cross_section)

        selected_rgb = self.settings.get_as_qcolor("slice_color_selected").getRgbF()[:3]
        normal_rgb = self.settings.get_as_qcolor("slice_color_normal").getRgbF()[:3]
        
        snapped_index = options.current_layer_index
        snapped_index = min(snapped_index, total - 1) if total > 0 else 0

        clip_n, clip_c = None, None
        if options.show_cross_section and total > 0:
            try:
                clip_n = pass_data.normals[snapped_index]
                clip_c = pass_data.origins[snapped_index]
            except Exception: pass 
        
        for global_idx, actor in current_slice_actors:
            is_selected = (global_idx == snapped_index)
            
            is_visible = options.show_slices
            
            if options.show_cross_section and clip_n is not None:
                if not is_selected: 
                    try:
                        slice_origin = pass_data.origins[global_idx] 
                        dot = np.dot(slice_origin - clip_c, clip_n)
                        if dot > 1e-6: 
                            is_visible = False
                    except Exception: 
                        is_visible = False
            
            actor.SetVisibility(is_visible)
            if not is_visible: continue

            if is_selected:
                actor.GetProperty().SetColor(*selected_rgb)
                actor.GetProperty().SetLineWidth(self.settings.get("slice_thickness_selected", 3))
                actor.GetProperty().SetOpacity(1.0)
            else:
                actor.GetProperty().SetColor(*normal_rgb)
                actor.GetProperty().SetLineWidth(self.settings.get("slice_thickness_normal", 1))
                actor.GetProperty().SetOpacity(options.slice_alpha)

        if self.layer_stats_actor: self.plotter.remove_actor(self.layer_stats_actor, render=False)
        if self.pass_stats_actor: self.plotter.remove_actor(self.pass_stats_actor, render=False) 
        if self.normal_vector_actor: self.plotter.remove_actor(self.normal_vector_actor, render=False)
        if self.vertex_actor: self.plotter.remove_actor(self.vertex_actor, render=False)
        if self.clipped_mesh_actor: self.plotter.remove_actor(self.clipped_mesh_actor, render=False)
        if self.clipped_centerline_actor: self.plotter.remove_actor(self.clipped_centerline_actor, render=False)

        try:
            idx = snapped_index
            if total > 0 and idx < total:
                current_slice = pass_data.slices[idx]
                n = pass_data.normals[idx] 
                c = pass_data.origins[idx]
                
                text_color = self.settings.get_as_qcolor("overlay_font_color").getRgbF()[:3]
                font_size = self.settings.get("overlay_font_size", 10)
                geom_epsilon = self.settings.get("geom_epsilon", 1e-6)
                
                local_curv = 0.0
                if idx < len(pass_data.u_values):
                    u_current = pass_data.u_values[idx]
                    local_curv = compute_curvature(
                        pass_data.tck, 
                        np.array([u_current]), 
                        geom_epsilon
                    )[0]

                pass_stats_str = f"Showing Pass {pass_index}"
                self.pass_stats_actor = self.plotter.add_text(
                    pass_stats_str, 
                    position='upper_edge', 
                    font_size=font_size, 
                    color=text_color, 
                    name='pass_stats'
                )
                
                layer_stats_str = (
                    f"Layer Stats (Idx: {idx} / {total-1})\n"
                    f"-----------------\n"
                    f"Curvature: {local_curv:.4f} (1/mm)\n"
                    f"Layer Length: {current_slice.length:.2f} mm\n"
                    f"Centroid (mm): {c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}\n"
                    f"Normal: {n[0]:.2f}, {n[1]:.2f}, {n[2]:.2f}"
                )
                
                self.layer_stats_actor = self.plotter.add_text(
                    layer_stats_str, position='upper_left',
                    font_size=font_size, 
                    color=text_color, name='layer_stats'
                )
                
                if options.show_layer_normal:
                    scale = np.mean(self.slicer.mesh.extents) / 10.0
                    self.normal_vector_actor = self.plotter.add_arrows(
                        cent=c, direction=n, mag=scale, 
                        color=text_color, name='current_normal'
                    )
                
                if current_slice.entities:
                    entity = current_slice.entities[0]
                    pts_idx = entity.points
                    if entity.closed and pts_idx[0] == pts_idx[-1]: pts_idx = pts_idx[:-1]
                    pts = current_slice.vertices[pts_idx]
                    
                    if len(pts) > 0:
                        points_poly = pv.PolyData(pts)
                        if options.show_cross_section and clip_n is not None and np.dot(c - clip_c, clip_n) > 1e-6:
                            points_poly = None
                        
                        if points_poly:
                            self.vertex_actor = self.plotter.add_points(
                                points_poly, color=text_color, point_size=5, name='layer_vertices'
                            )
                
                mesh_alpha = self.settings.get("mesh_alpha", 0.25)
                if options.show_cross_section and current_mesh_actor and hasattr(current_mesh_actor, 'mapper'):
                    clipped_data = current_mesh_actor.mapper.dataset.clip(normal=n, origin=c, invert=True) 
                    self.clipped_mesh_actor = self.plotter.add_mesh(
                        clipped_data, color=self.settings.get_as_qcolor("mesh_color").getRgbF()[:3],
                        opacity=mesh_alpha, show_edges=True, render=False
                    )
                    self.clipped_mesh_actor.SetVisibility(options.show_mesh)

                if options.show_cross_section and current_centerline_actor and hasattr(current_centerline_actor, 'mapper'):
                    clipped_data = current_centerline_actor.mapper.dataset.clip(normal=n, origin=c, invert=True)
                    self.clipped_centerline_actor = self.plotter.add_mesh(
                        clipped_data, color=self.settings.get_as_qcolor("centerline_color").getRgbF()[:3],
                        line_width=self.settings.get("centerline_thickness", 4),
                        render_lines_as_tubes=True, style='wireframe', render=False
                    )
                    self.clipped_centerline_actor.SetVisibility(options.show_centerline)

        except Exception as e:
            logging.warning(f"Could not generate layer stats or normal: {e}")

        self.plotter.update()

    def set_camera_parallel(self, is_parallel: bool):
        if self.plotter.renderer:
            self.plotter.camera.parallel_projection = is_parallel
            self.plotter.update()

    def reset_view(self):
        self.plotter.reset_camera()

    def view_x(self):
        self.plotter.view_yz()
        self.plotter.reset_camera()

    def view_y(self):
        self.plotter.view_xz()
        self.plotter.reset_camera()

    def view_z(self):
        self.plotter.view_xy()
        self.plotter.reset_camera()