import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv
import trimesh
from PyQt5.QtGui import QColor

from slicer_core.settings_manager import SettingsManager
from slicer_core.shell_processor import CentrelineShellSlicer, ShellPassResult

logger = logging.getLogger(__name__)


@dataclass
class ViewOptions:
    show_mesh: bool
    show_slices: bool
    show_centerline: bool
    show_spheres: bool
    show_surface_curves: bool
    show_bbox_limit: bool
    show_plane_normal: bool
    vis_skip_layers: int
    slice_alpha: float
    current_layer_index: int


def _polyline_from_points(points: np.ndarray, closed: bool = False) -> pv.PolyData:
    poly = pv.PolyData(points)
    count = len(points)
    if closed:
        poly.lines = np.hstack(([count + 1], np.arange(count), 0))
    else:
        poly.lines = np.hstack(([count], np.arange(count)))
    return poly


def _frame_axes(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=float)
    normal = normal / max(np.linalg.norm(normal), 1e-12)

    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(reference, normal)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)

    axis_u = np.cross(normal, reference)
    axis_u = axis_u / max(np.linalg.norm(axis_u), 1e-12)
    axis_v = np.cross(normal, axis_u)
    axis_v = axis_v / max(np.linalg.norm(axis_v), 1e-12)
    return axis_u, axis_v


def _order_plane_points(
    points: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points

    axis_u, axis_v = _frame_axes(normal)
    rel = points - np.asarray(origin, dtype=float)
    angles = np.arctan2(rel @ axis_v, rel @ axis_u)
    return points[np.argsort(angles)]


def _polyline_spacing(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 1e-6
    step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    step_lengths = step_lengths[step_lengths > 1e-9]
    if len(step_lengths) == 0:
        return 1e-6
    return float(np.median(step_lengths))


def _is_closed_polyline(points: np.ndarray) -> bool:
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return False
    tolerance = max(_polyline_spacing(points) * 0.5, 1e-6)
    return np.linalg.norm(points[0] - points[-1]) <= tolerance


def _plane_fragments(
    points: np.ndarray,
    fragment_index: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return []
    if fragment_index is None or len(fragment_index) != len(points):
        return [points]

    fragment_index = np.asarray(fragment_index, dtype=int)
    fragments = []
    for fragment_id in np.unique(fragment_index):
        fragment = points[fragment_index == fragment_id]
        if len(fragment) >= 2:
            fragments.append(fragment)
    return fragments or [points]


def _point_sphere_glyphs(points: np.ndarray, radius: float) -> pv.PolyData:
    points = np.asarray(points, dtype=float)
    radius = float(max(radius, 1e-6))
    cloud = pv.PolyData(points)
    sphere = pv.Sphere(radius=radius, theta_resolution=16, phi_resolution=16)
    return cloud.glyph(geom=sphere, scale=False)


def _tube_from_polyline(points: np.ndarray, radius: float, closed: bool = False) -> pv.PolyData:
    radius = float(max(radius, 1e-6))
    poly = _polyline_from_points(points, closed=closed)
    return poly.tube(radius=radius, n_sides=16, capping=True)


class PyVistaManager:
    def __init__(self, plotter: pv.Plotter):
        self.plotter = plotter
        self.slicer: Optional[CentrelineShellSlicer] = None
        self.settings: Optional[SettingsManager] = None
        self.last_view_options: Optional[ViewOptions] = None
        self.mesh_actor = None
        self.shell_points_actor = None
        self.sphere_actors: List = []
        self.sphere_center_actor = None
        self.centerline_actors: List = []
        self.surface_curve_actors: List = []
        self.slice_actors: List[Tuple[int, any]] = []
        self.slice_selected_actors: List[Tuple[int, any]] = []
        self.plane_point_actors: List[Tuple[int, any]] = []
        self.layer_stats_actor = None
        self.global_stats_actor = None
        self.legend_actor = None
        self.normal_actor = None
        self.clipped_mesh_actor = None

    def _remove_optional(self, actor):
        if actor:
            self.plotter.remove_actor(actor, render=False)

    def _clear(self):
        self.plotter.clear()
        self.mesh_actor = None
        self.shell_points_actor = None
        self.sphere_actors = []
        self.sphere_center_actor = None
        self.centerline_actors = []
        self.surface_curve_actors = []
        self.slice_actors = []
        self.slice_selected_actors = []
        self.plane_point_actors = []
        self.layer_stats_actor = None
        self.global_stats_actor = None
        self.legend_actor = None
        self.normal_actor = None
        self.clipped_mesh_actor = None

    def _setting_color(self, key: str):
        return self.settings.get_as_qcolor(key).getRgbF()[:3]

    def _line_radius(self, setting_key: str, default: float = 1.0):
        value = float(self.settings.get(setting_key, default))
        if self.slicer and self.slicer.mesh is not None:
            scale = float(np.mean(np.asarray(self.slicer.mesh.extents, dtype=float)))
        else:
            scale = 100.0
        return max(value, 0.1) * max(scale, 1e-6) * 0.001

    def _remove_legend(self):
        self._remove_optional(self.legend_actor)
        self.legend_actor = None

    def _update_legend(self, options: Optional[ViewOptions] = None):
        if not self.settings:
            return

        self._remove_legend()

        labels = []
        if options is None or options.show_mesh:
            labels.append(["STL mesh", self._setting_color("mesh_color")])
        if options is None or options.show_spheres:
            labels.append(["Hitbox spheres", self._setting_color("sphere_color")])
        if options is None or options.show_centerline:
            labels.append(["Centreline", self._setting_color("centerline_color")])
        if options is None or options.show_surface_curves:
            labels.append(["Surface curves", self._setting_color("guide_color")])
        if options is None or options.show_slices:
            labels.append(["Slicing planes", self._setting_color("slice_color_normal")])
            labels.append(["Selected plane", self._setting_color("slice_color_selected")])
        if options is None or (
            options.show_bbox_limit and self.settings.get("limit_curve_bbox", True)
        ):
            labels.append(["BBox limit", self._setting_color("guide_color")])
        if options is None or options.show_plane_normal:
            labels.append(["Plane normal", self._setting_color("overlay_font_color")])

        if not labels:
            return

        try:
            self.legend_actor = self.plotter.add_legend(
                labels=labels,
                bcolor=(1.0, 1.0, 1.0),
                border=True,
                size=(0.22, 0.18),
                name="legend",
            )
        except Exception as exc:
            logger.debug("Could not add plot legend: %s", exc)

    def preview_mesh(self, mesh: trimesh.Trimesh, alpha: float, color: QColor):
        self._clear()
        self.slicer = None
        self.plotter.background_color = (
            self.settings.get_as_qcolor("bg_color").getRgbF()[:3]
            if self.settings
            else (0.17, 0.17, 0.17)
        )
        self.mesh_actor = self.plotter.add_mesh(
            pv.wrap(mesh),
            color=color.getRgbF()[:3],
            opacity=alpha,
            show_edges=True,
            name="preview_mesh",
        )
        self.plotter.show_axes()
        self._update_legend()
        self.plotter.reset_camera()

    def update_appearance(self, settings: SettingsManager):
        self.settings = settings
        self.plotter.background_color = settings.get_as_qcolor("bg_color").getRgbF()[:3]
        if self.slicer:
            self.set_slicer_object(self.slicer)
            if self.last_view_options:
                self.refresh_view(self.last_view_options)

    def set_slicer_object(self, slicer: CentrelineShellSlicer):
        self.slicer = slicer
        self._clear()
        if not self.settings or not self.slicer or not self.slicer.pass_results:
            return

        self.plotter.background_color = self.settings.get_as_qcolor("bg_color").getRgbF()[:3]
        result = self.slicer.pass_results[-1]

        self.mesh_actor = self.plotter.add_mesh(
            pv.wrap(self.slicer.mesh),
            color=self.settings.get_as_qcolor("mesh_color").getRgbF()[:3],
            opacity=self.settings.get("mesh_alpha", 0.1),
            show_edges=True,
            name="mesh",
        )

        sphere_color = self._setting_color("sphere_color")
        sphere_centres = []
        for idx, (center, radius) in enumerate(result.spheres):
            try:
                sphere_centres.append(np.asarray(center, dtype=float))
                sphere = pv.Sphere(
                    radius=float(radius),
                    center=np.asarray(center, dtype=float),
                    theta_resolution=18,
                    phi_resolution=18,
                )
                actor = self.plotter.add_mesh(
                    sphere,
                    color=sphere_color,
                    opacity=self.settings.get("sphere_alpha", 1.0),
                    style="wireframe",
                    line_width=self.settings.get("sphere_thickness", 1),
                    name=f"sphere_{idx}",
                )
                self.sphere_actors.append(actor)
            except Exception as exc:
                logger.debug("Skipping sphere %s visualization: %s", idx, exc)

        if sphere_centres:
            self.sphere_center_actor = self.plotter.add_points(
                pv.PolyData(np.asarray(sphere_centres, dtype=float)),
                color=sphere_color,
                point_size=self.settings.get("sphere_center_point_size", 12),
                opacity=self.settings.get("sphere_center_alpha", 1.0),
                render_points_as_spheres=True,
                name="sphere_centres",
            )

        for idx, (_, centreline) in enumerate(result.centrelines):
            if len(centreline) < 2:
                continue
            poly = _polyline_from_points(centreline, closed=False)
            actor = self.plotter.add_mesh(
                poly,
                color=self._setting_color("centerline_color"),
                line_width=max(6, self.settings.get("centerline_thickness", 4)),
                render_lines_as_tubes=True,
                name=f"centerline_{idx}",
            )
            self.centerline_actors.append(actor)

        for idx, surface in enumerate(result.surface_curves):
            curve = np.asarray(surface.get("curve"), dtype=float)
            if len(curve) < 2:
                continue
            actor = self.plotter.add_mesh(
                _polyline_from_points(curve, closed=False),
                color=self.settings.get_as_qcolor("guide_color").getRgbF()[:3],
                line_width=max(4, self.settings.get("centerline_thickness", 4)),
                render_lines_as_tubes=True,
                opacity=0.9,
                name=f"surface_curve_{idx}",
            )
            self.surface_curve_actors.append(actor)

        skip = max(1, self.settings.get("vis_skip_layers", 10))
        indices = set(range(0, len(result.origins), skip))
        if len(result.origins):
            indices.add(len(result.origins) - 1)
        for idx in sorted(indices):
            plane_mask = result.plane_index == idx
            points = result.shell_points[plane_mask]
            if len(points) < 2:
                continue
            fragment_index = getattr(result, "plane_fragment_index", None)
            if fragment_index is not None:
                fragments = _plane_fragments(points, fragment_index[plane_mask])
            else:
                fragments = [
                    _order_plane_points(
                        points,
                        result.origins[idx],
                        result.normals[idx],
                    )
                ]

            for fragment_pos, fragment_points in enumerate(fragments):
                if len(fragment_points) < 2:
                    continue
                closed = (
                    _is_closed_polyline(fragment_points)
                    if fragment_index is not None
                    else len(fragment_points) >= 3
                )
                actor = self.plotter.add_mesh(
                    _tube_from_polyline(
                        fragment_points,
                        self._line_radius("slice_thickness_normal", 1),
                        closed=closed,
                    ),
                    color=self.settings.get_as_qcolor("slice_color_normal").getRgbF()[:3],
                    opacity=self.settings.get("slice_alpha", 1.0),
                    name=f"slice_{idx}_{fragment_pos}",
                )
                self.slice_actors.append((idx, actor))
                selected_actor = self.plotter.add_mesh(
                    _tube_from_polyline(
                        fragment_points,
                        self._line_radius("slice_thickness_selected", 3),
                        closed=closed,
                    ),
                    color=self.settings.get_as_qcolor("slice_color_selected").getRgbF()[:3],
                    opacity=self.settings.get("selected_slice_alpha", 1.0),
                    name=f"slice_selected_{idx}_{fragment_pos}",
                )
                selected_actor.SetVisibility(False)
                self.slice_selected_actors.append((idx, selected_actor))
                point_actor = self.plotter.add_mesh(
                    _point_sphere_glyphs(
                        fragment_points,
                        self.settings.get("plane_point_size", 0.5),
                    ),
                    color=self.settings.get_as_qcolor("slice_color_normal").getRgbF()[:3],
                    opacity=self.settings.get("slice_alpha", 1.0),
                    name=f"slice_points_{idx}_{fragment_pos}",
                )
                self.plane_point_actors.append((idx, point_actor))

        if self.slicer.mesh is not None:
            bounds = np.asarray(self.slicer.mesh.bounds, dtype=float)
            padding_ratio = max(
                float(self.settings.get("curve_bbox_padding_ratio", 0.1)),
                0.0,
            )
            extents = np.maximum(bounds[1] - bounds[0], 1e-9)
            padding = extents * padding_ratio
            lower = bounds[0] - padding
            upper = bounds[1] + padding
            bbox = pv.Box(
                bounds=(
                    lower[0],
                    upper[0],
                    lower[1],
                    upper[1],
                    lower[2],
                    upper[2],
                )
            )
            self.clipped_mesh_actor = self.plotter.add_mesh(
                bbox,
                color=self.settings.get_as_qcolor("guide_color").getRgbF()[:3],
                style="wireframe",
                line_width=2,
                opacity=0.8,
                name="curve_bbox_limit",
            )

        summary = result.summary
        spacing_line = ""
        if "plane_spacing_min" in summary and "plane_spacing_max" in summary:
            spacing_line = (
                f"\nSpacing: {summary['plane_spacing_min']:.2f}"
                f"-{summary['plane_spacing_max']:.2f} mm"
            )
        stats_str = (
            f"Branches: {summary['branch_count']}\n"
            f"Planes: {summary['plane_count']}\n"
            f"Shell points: {summary['shell_point_count']}\n"
            f"Surface curves: {summary['surface_curve_count']}"
            f"{spacing_line}"
        )
        self.global_stats_actor = self.plotter.add_text(
            stats_str,
            position="upper_right",
            font_size=self.settings.get("overlay_font_size", 10),
            color=self.settings.get_as_qcolor("overlay_font_color").getRgbF()[:3],
            name="global_stats",
        )
        self.plotter.show_axes()
        self._update_legend()
        self.plotter.reset_camera()

    def refresh_view(self, options: ViewOptions):
        if not self.slicer or not self.settings or not self.slicer.pass_results:
            return
        self.last_view_options = options

        result: ShellPassResult = self.slicer.pass_results[-1]
        selected = min(options.current_layer_index, max(0, len(result.origins) - 1))

        if self.mesh_actor:
            self.mesh_actor.SetVisibility(options.show_mesh)
        if self.shell_points_actor:
            self.shell_points_actor.GetProperty().SetPointSize(
                self.settings.get("shell_point_size", 0.5)
            )
            self.shell_points_actor.SetVisibility(False)
        sphere_visible = (
            options.show_spheres
            and self.settings.get("sphere_alpha", 1.0) > 0.0
        )
        for actor in self.sphere_actors:
            actor.SetVisibility(sphere_visible)
        if self.sphere_center_actor:
            self.sphere_center_actor.SetVisibility(
                options.show_spheres
                and self.settings.get("sphere_center_alpha", 1.0) > 0.0
            )
        for actor in self.centerline_actors:
            actor.SetVisibility(options.show_centerline)
        for actor in self.surface_curve_actors:
            actor.SetVisibility(options.show_surface_curves)

        normal_color = self.settings.get_as_qcolor("slice_color_normal").getRgbF()[:3]
        selected_color = self.settings.get_as_qcolor("slice_color_selected").getRgbF()[:3]

        for plane_id, actor in self.slice_actors:
            visible = options.show_slices and plane_id != selected
            actor.SetVisibility(visible)
            if not visible:
                continue
            prop = actor.GetProperty()
            prop.SetColor(*normal_color)
            prop.SetOpacity(options.slice_alpha)

        for plane_id, actor in self.slice_selected_actors:
            visible = options.show_slices and plane_id == selected
            actor.SetVisibility(visible)
            if not visible:
                continue
            prop = actor.GetProperty()
            prop.SetColor(*selected_color)
            prop.SetOpacity(self.settings.get("selected_slice_alpha", 1.0))

        for plane_id, actor in self.plane_point_actors:
            visible = options.show_slices
            actor.SetVisibility(visible)
            if not visible:
                continue
            prop = actor.GetProperty()
            if plane_id == selected:
                prop.SetColor(*selected_color)
                prop.SetOpacity(self.settings.get("selected_slice_alpha", 1.0))
            else:
                prop.SetColor(*normal_color)
                prop.SetOpacity(options.slice_alpha)

        self._remove_optional(self.layer_stats_actor)
        self._remove_optional(self.normal_actor)

        if len(result.origins):
            origin = result.origins[selected]
            normal = result.normals[selected]
            plane_point_count = int(np.count_nonzero(result.plane_index == selected))
            plane_distance = float(result.plane_distances[selected])
            self.layer_stats_actor = self.plotter.add_text(
                (
                    f"Layer {selected + 1} / {len(result.origins)}\n"
                    f"Distance: {plane_distance:.2f} mm\n"
                    f"Points: {plane_point_count}\n"
                    f"Normal: {normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}"
                ),
                position="upper_left",
                font_size=self.settings.get("overlay_font_size", 10),
                color=self.settings.get_as_qcolor("overlay_font_color").getRgbF()[:3],
                name="layer_stats",
            )
            if options.show_plane_normal:
                scale = np.mean(self.slicer.mesh.extents) / 10.0
                self.normal_actor = self.plotter.add_arrows(
                    cent=origin,
                    direction=normal,
                    mag=scale,
                    color=self.settings.get_as_qcolor("overlay_font_color").getRgbF()[:3],
                    name="plane_normal",
                )
        if self.clipped_mesh_actor:
            visible = options.show_bbox_limit and bool(
                self.settings.get("limit_curve_bbox", True)
            )
            self.clipped_mesh_actor.SetVisibility(visible)
        self._update_legend(options)

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

    def save_x_y_views(self, output_dir: Path, run_timestamp: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = {
            "x": output_dir / f"view_x_{run_timestamp}.png",
            "y": output_dir / f"view_y_{run_timestamp}.png",
        }

        original_camera_position = self.plotter.camera_position

        try:
            self.view_x()
            self.plotter.render()
            self.plotter.screenshot(str(image_paths["x"]), return_img=False)

            self.view_y()
            self.plotter.render()
            self.plotter.screenshot(str(image_paths["y"]), return_img=False)
        finally:
            self.plotter.camera_position = original_camera_position
            self.plotter.render()

        return image_paths

    def save_current_view(self, output_dir: Path, run_timestamp: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_path = output_dir / f"current_plot_{run_timestamp}.png"
        self.plotter.render()
        self.plotter.screenshot(str(image_path), return_img=False)
        return image_path

    def save_hitbox_geometry_views(self, output_dir: Path, run_timestamp: str):
        """
        Save clean X/Y views containing only the STL mesh, hitbox spheres, and centrelines.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = {
            "x": output_dir / f"hitbox_geometry_x_{run_timestamp}.png",
            "y": output_dir / f"hitbox_geometry_y_{run_timestamp}.png",
        }

        actors = []

        def remember(actor):
            if actor is not None:
                actors.append((actor, actor.GetVisibility()))

        remember(self.mesh_actor)
        remember(self.shell_points_actor)
        remember(self.sphere_center_actor)
        for actor in self.sphere_actors:
            remember(actor)
        for actor in self.centerline_actors:
            remember(actor)
        for _, actor in self.slice_actors:
            remember(actor)
        for _, actor in self.slice_selected_actors:
            remember(actor)
        for _, actor in self.plane_point_actors:
            remember(actor)
        remember(self.layer_stats_actor)
        remember(self.global_stats_actor)
        remember(self.legend_actor)
        remember(self.normal_actor)
        remember(self.clipped_mesh_actor)

        original_camera_position = self.plotter.camera_position

        try:
            if self.mesh_actor:
                self.mesh_actor.SetVisibility(True)
            if self.shell_points_actor:
                self.shell_points_actor.SetVisibility(False)
            if self.sphere_center_actor:
                self.sphere_center_actor.SetVisibility(True)
            for actor in self.sphere_actors:
                actor.SetVisibility(True)
            for actor in self.centerline_actors:
                actor.SetVisibility(True)
            for _, actor in self.slice_actors:
                actor.SetVisibility(False)
            for _, actor in self.slice_selected_actors:
                actor.SetVisibility(False)
            for _, actor in self.plane_point_actors:
                actor.SetVisibility(False)
            for actor in [
                self.layer_stats_actor,
                self.global_stats_actor,
                self.legend_actor,
                self.normal_actor,
                self.clipped_mesh_actor,
            ]:
                if actor:
                    actor.SetVisibility(False)

            self.view_x()
            self.plotter.render()
            self.plotter.screenshot(str(image_paths["x"]), return_img=False)

            self.view_y()
            self.plotter.render()
            self.plotter.screenshot(str(image_paths["y"]), return_img=False)
        finally:
            for actor, visibility in actors:
                actor.SetVisibility(visibility)
            self.plotter.camera_position = original_camera_position
            self.plotter.render()

        return image_paths
