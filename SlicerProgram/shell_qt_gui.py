import csv
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pyvista as pv
import trimesh
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import BackgroundPlotter

from slicer_core.config import SlicerConfig
from slicer_core.batch_plan import BatchPlan, BatchPlanError, BatchPlanRow, load_batch_plan_csv
from slicer_core.settings_manager import SettingsManager
from slicer_core.shell_processor import (
    CentrelineShellSlicer,
    build_output_dir,
    normalize_path,
)

from shell_gui_threading import LogEmitter, QtLogHandler, SlicerWorker
from shell_gui_visualization import PyVistaManager, ViewOptions
from shell_gui_widgets import ColorSelectWidget, SettingsDialog


SPHERE_GENERATION_OPTIONS = [
    ("Auto (path then component)", "auto"),
    ("Skeleton path sampling", "skeleton_paths"),
    ("Component centroids", "component_centroid"),
]


VIEWER_FALLBACK_SUMMARY = (
    "Viewer fallback active. Live 3D interaction is unavailable on this machine, "
    "but slicing and image export can still run."
)


class NullVisualizationManager:
    def __init__(self, reason: str):
        self.reason = reason

    def update_appearance(self, settings: SettingsManager):
        return None

    def preview_mesh(self, mesh: trimesh.Trimesh, alpha: float, color):
        return None

    def set_slicer_object(self, slicer: CentrelineShellSlicer):
        return None

    def refresh_view(self, options: ViewOptions):
        return None

    def set_camera_parallel(self, is_parallel: bool):
        return None

    def reset_view(self):
        return None

    def view_x(self):
        return None

    def view_y(self):
        return None

    def view_z(self):
        return None

    def save_x_y_views(self, output_dir: Path, run_timestamp: str):
        raise RuntimeError(self.reason)

    def save_current_view(self, output_dir: Path, run_timestamp: str):
        raise RuntimeError(self.reason)

    def save_hitbox_geometry_views(self, output_dir: Path, run_timestamp: str):
        raise RuntimeError(self.reason)


def _probe_interactive_opengl_support() -> tuple[bool, str]:
    try:
        from vtkmodules.vtkCommonCore import vtkObject

        vtkObject.GlobalWarningDisplayOff()
        try:
            if sys.platform.startswith("win"):
                from vtkmodules.vtkRenderingOpenGL2 import vtkWin32OpenGLRenderWindow

                render_window = vtkWin32OpenGLRenderWindow()
            else:
                from vtkmodules.vtkRenderingCore import vtkRenderWindow
                import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

                render_window = vtkRenderWindow()
            supported = bool(render_window.SupportsOpenGL())
            message = render_window.GetOpenGLSupportMessage() or ""
            return supported, message
        finally:
            vtkObject.GlobalWarningDisplayOn()
    except Exception as exc:
        return False, f"OpenGL probe failed: {exc}"


class SlicerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SlicerProgram - Centreline Shell Viewer")
        self.setGeometry(100, 100, 1600, 900)

        self.slicer_object: Optional[CentrelineShellSlicer] = None
        self.settings_manager = SettingsManager()
        self.settings_dialog: Optional[SettingsDialog] = None
        self.vis_manager: Optional[PyVistaManager] = None
        self.worker_thread = None
        self.log_emitter = LogEmitter()
        self._last_run_stl_path: Optional[str] = None
        self._rerun_smoothing_passes: int = 0
        self._loading_algorithm_ui: bool = False
        self._batch_plan: Optional[BatchPlan] = None
        self._batch_active: bool = False
        self._batch_cancel_requested: bool = False
        self._batch_current_index: int = 0
        self._batch_base_settings: Optional[dict] = None
        self._batch_summary_csv_path: Optional[Path] = None
        self._batch_output_root: Optional[Path] = None
        self._active_batch_row: Optional[BatchPlanRow] = None
        self._active_batch_run_settings: Optional[SettingsManager] = None
        self.plotter = None
        self._viewer_backend_mode = "interactive"
        self._viewer_status_message = ""

        self._setup_ui()
        self._connect_signals()
        self._load_config_into_ui()
        self._apply_appearance_settings()
        if self._viewer_backend_mode != "interactive":
            self.result_summary.setText(VIEWER_FALLBACK_SUMMARY)
            logger = logging.getLogger("slicer_core")
            logger.warning(VIEWER_FALLBACK_SUMMARY)
            if self._viewer_status_message:
                logger.info(self._viewer_status_message)

    def _format_viewer_status_message(self, support_message: str) -> str:
        details = (support_message or "").strip() or (
            "VTK could not initialize an OpenGL 3.2+ render window."
        )
        return (
            "Interactive 3D viewer unavailable.\n\n"
            "VTK/PyVista could not initialize an OpenGL 3.2+ window on this "
            "machine. The app will continue with off-screen rendering when "
            "possible so slicing and screenshot export still work.\n\n"
            "Common fixes:\n"
            "- update the GPU driver\n"
            "- avoid Microsoft Remote Desktop for 3D viewing\n"
            "- install Mesa/OSMesa and ensure osmesa.dll is on PATH\n\n"
            f"VTK reported: {details}"
        )

    def _build_viewer_placeholder(self, message: str) -> QWidget:
        placeholder = QWidget()
        layout = QVBoxLayout(placeholder)
        layout.setContentsMargins(12, 12, 12, 12)
        notice = QLabel(message)
        notice.setWordWrap(True)
        notice.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        notice.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(notice)
        layout.addStretch(1)
        return placeholder

    def _create_viewer_backend(self):
        interactive_supported, support_message = _probe_interactive_opengl_support()

        if interactive_supported:
            try:
                plotter = BackgroundPlotter(show=False, multi_samples=0)
                return plotter, PyVistaManager(plotter), "interactive", ""
            except Exception as exc:
                support_message = (
                    "OpenGL support probe passed, but the interactive PyVista "
                    f"viewer still failed to initialize: {exc}"
                )

        status_message = self._format_viewer_status_message(support_message)
        try:
            from vtkmodules.vtkCommonCore import vtkObject

            vtkObject.GlobalWarningDisplayOff()
        except Exception:
            pass

        try:
            plotter = BackgroundPlotter(show=False, off_screen=True, multi_samples=0)
            return plotter, PyVistaManager(plotter), "offscreen", status_message
        except Exception as exc:
            unavailable_message = (
                f"{status_message}\n\n"
                f"Off-screen rendering fallback also failed: {exc}"
            )
            return (
                None,
                NullVisualizationManager(unavailable_message),
                "unavailable",
                unavailable_message,
            )

    def _setup_ui(self):
        self._setup_menu_bar()

        central = QWidget()
        main_layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left.setMinimumWidth(380)

        self.file_path_edit = QLineEdit()
        browse = QPushButton("Browse STL")
        browse.clicked.connect(self._browse_stl)
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("STL File:"))
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(browse)
        left_layout.addLayout(file_layout)

        batch_group = QGroupBox("Planned Parameter Batch")
        batch_layout = QVBoxLayout(batch_group)
        batch_file_layout = QHBoxLayout()
        self.batch_plan_path_edit = QLineEdit()
        self.batch_plan_path_edit.setReadOnly(True)
        self.load_batch_csv_btn = QPushButton("Load Batch CSV")
        batch_file_layout.addWidget(QLabel("Plan File:"))
        batch_file_layout.addWidget(self.batch_plan_path_edit)
        batch_file_layout.addWidget(self.load_batch_csv_btn)
        batch_layout.addLayout(batch_file_layout)

        self.batch_plan_summary = QLabel(
            "No batch plan loaded. CSV headers like k, of, and samples are supported."
        )
        self.batch_plan_summary.setWordWrap(True)
        batch_layout.addWidget(self.batch_plan_summary)

        self.run_batch_button = QPushButton("Run Planned Parameter Batch")
        self.run_batch_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.run_batch_button.setEnabled(False)
        batch_layout.addWidget(self.run_batch_button)
        left_layout.addWidget(batch_group)

        view_group = QGroupBox("Viewer Options")
        view_grid = QGridLayout(view_group)
        self.show_mesh_check = QCheckBox("Show Mesh")
        self.show_slices_check = QCheckBox("Show Slices")
        self.show_centerline_check = QCheckBox("Show Centerline")
        self.show_spheres_check = QCheckBox("Show Spheres")
        self.show_surface_curves_check = QCheckBox("Show Surfaces")
        self.show_bbox_limit_check = QCheckBox("Show BBox")
        self.show_normal_check = QCheckBox("Show Current Plane Normal")
        self.perspective_check = QCheckBox("Perspective View")

        for row, widget in enumerate([
            self.show_mesh_check,
            self.show_slices_check,
            self.show_centerline_check,
            self.show_spheres_check,
            self.show_surface_curves_check,
            self.show_bbox_limit_check,
            self.show_normal_check,
            self.perspective_check,
        ]):
            view_grid.addWidget(widget, row, 0, 1, 2)

        view_buttons = QHBoxLayout()
        self.reset_view_btn = QPushButton("Reset View")
        self.view_x_btn = QPushButton("View X")
        self.view_y_btn = QPushButton("View Y")
        self.view_z_btn = QPushButton("View Z")
        self.print_plot_btn = QPushButton("Print Plot")
        for button in [self.reset_view_btn, self.view_x_btn, self.view_y_btn, self.view_z_btn, self.print_plot_btn]:
            view_buttons.addWidget(button)
        view_grid.addLayout(view_buttons, 8, 0, 1, 2)
        left_layout.addWidget(view_group)

        algorithm_group = QGroupBox("Algorithm Parameters")
        algorithm_layout = QVBoxLayout(algorithm_group)

        major_method_layout = QGridLayout()
        self.sphere_generation_combo = QComboBox()
        for label, value in SPHERE_GENERATION_OPTIONS:
            self.sphere_generation_combo.addItem(label, value)
        major_method_layout.addWidget(QLabel("Sphere Algorithm:"), 0, 0)
        major_method_layout.addWidget(self.sphere_generation_combo, 0, 1)
        algorithm_layout.addLayout(major_method_layout)

        algorithm_tabs = QTabWidget()
        general_tab = QWidget()
        centreline_tab = QWidget()
        surface_tab = QWidget()
        filtering_tab = QWidget()
        colour_tab = QWidget()
        size_tab = QWidget()
        opacity_tab = QWidget()
        general_layout = QGridLayout(general_tab)
        centreline_layout = QGridLayout(centreline_tab)
        surface_layout = QGridLayout(surface_tab)
        filtering_layout = QGridLayout(filtering_tab)
        colour_layout = QVBoxLayout(colour_tab)
        size_layout = QGridLayout(size_tab)
        opacity_layout = QGridLayout(opacity_tab)
        algorithm_tabs.addTab(general_tab, "General")
        algorithm_tabs.addTab(centreline_tab, "Centreline")
        algorithm_tabs.addTab(surface_tab, "Surfaces")
        algorithm_tabs.addTab(filtering_tab, "Filtering")
        algorithm_tabs.addTab(colour_tab, "Colours")
        algorithm_tabs.addTab(size_tab, "Sizes")
        algorithm_tabs.addTab(opacity_tab, "Opacity")
        algorithm_layout.addWidget(algorithm_tabs)

        self.bg_picker = ColorSelectWidget("Background:", self.settings_manager.get_as_qcolor("bg_color"))
        self.mesh_picker = ColorSelectWidget("Mesh:", self.settings_manager.get_as_qcolor("mesh_color"))
        self.centerline_picker = ColorSelectWidget("Centreline:", self.settings_manager.get_as_qcolor("centerline_color"))
        self.sphere_picker = ColorSelectWidget("Spheres:", self.settings_manager.get_as_qcolor("sphere_color"))
        self.slice_picker = ColorSelectWidget("Planes:", self.settings_manager.get_as_qcolor("slice_color_normal"))
        self.selected_slice_picker = ColorSelectWidget("Selected Plane:", self.settings_manager.get_as_qcolor("slice_color_selected"))
        self.overlay_picker = ColorSelectWidget("Text/Normal:", self.settings_manager.get_as_qcolor("overlay_font_color"))
        for picker in [
            self.bg_picker,
            self.mesh_picker,
            self.centerline_picker,
            self.sphere_picker,
            self.slice_picker,
            self.selected_slice_picker,
            self.overlay_picker,
        ]:
            colour_layout.addWidget(picker)
        colour_layout.addStretch()

        self.centerline_thickness_spin = QSpinBox()
        self.centerline_thickness_spin.setRange(1, 50)
        self.sphere_thickness_spin = QSpinBox()
        self.sphere_thickness_spin.setRange(1, 50)
        self.plane_thickness_spin = QSpinBox()
        self.plane_thickness_spin.setRange(1, 50)
        self.selected_plane_thickness_spin = QSpinBox()
        self.selected_plane_thickness_spin.setRange(1, 50)
        self.sphere_center_point_size_spin = QSpinBox()
        self.sphere_center_point_size_spin.setRange(1, 100)
        self.shell_point_size_spin = QDoubleSpinBox()
        self.shell_point_size_spin.setRange(0.1, 100.0)
        self.shell_point_size_spin.setDecimals(2)
        self.shell_point_size_spin.setSingleStep(0.1)
        self.plane_point_size_spin = QDoubleSpinBox()
        self.plane_point_size_spin.setRange(0.1, 100.0)
        self.plane_point_size_spin.setDecimals(2)
        self.plane_point_size_spin.setSingleStep(0.1)
        size_layout.addWidget(QLabel("Centreline Thickness:"), 0, 0)
        size_layout.addWidget(self.centerline_thickness_spin, 0, 1)
        size_layout.addWidget(QLabel("Sphere Thickness:"), 1, 0)
        size_layout.addWidget(self.sphere_thickness_spin, 1, 1)
        size_layout.addWidget(QLabel("Plane Thickness:"), 2, 0)
        size_layout.addWidget(self.plane_thickness_spin, 2, 1)
        size_layout.addWidget(QLabel("Selected Plane Thickness:"), 3, 0)
        size_layout.addWidget(self.selected_plane_thickness_spin, 3, 1)
        size_layout.addWidget(QLabel("Sphere Centre Point Size:"), 4, 0)
        size_layout.addWidget(self.sphere_center_point_size_spin, 4, 1)
        size_layout.addWidget(QLabel("Shell Point Size:"), 5, 0)
        size_layout.addWidget(self.shell_point_size_spin, 5, 1)
        size_layout.addWidget(QLabel("Plane Point Size:"), 6, 0)
        size_layout.addWidget(self.plane_point_size_spin, 6, 1)
        size_layout.setRowStretch(7, 1)

        self.mesh_alpha_spin = QDoubleSpinBox()
        self.mesh_alpha_spin.setRange(0.0, 1.0)
        self.mesh_alpha_spin.setDecimals(2)
        self.mesh_alpha_spin.setSingleStep(0.05)
        self.sphere_alpha_spin = QDoubleSpinBox()
        self.sphere_alpha_spin.setRange(0.0, 1.0)
        self.sphere_alpha_spin.setDecimals(2)
        self.sphere_alpha_spin.setSingleStep(0.05)
        self.sphere_center_alpha_spin = QDoubleSpinBox()
        self.sphere_center_alpha_spin.setRange(0.0, 1.0)
        self.sphere_center_alpha_spin.setDecimals(2)
        self.sphere_center_alpha_spin.setSingleStep(0.05)
        self.slice_alpha_spin = QDoubleSpinBox()
        self.slice_alpha_spin.setRange(0.0, 1.0)
        self.slice_alpha_spin.setDecimals(2)
        self.slice_alpha_spin.setSingleStep(0.05)
        self.selected_slice_alpha_spin = QDoubleSpinBox()
        self.selected_slice_alpha_spin.setRange(0.0, 1.0)
        self.selected_slice_alpha_spin.setDecimals(2)
        self.selected_slice_alpha_spin.setSingleStep(0.05)
        opacity_layout.addWidget(QLabel("Mesh Opacity:"), 0, 0)
        opacity_layout.addWidget(self.mesh_alpha_spin, 0, 1)
        opacity_layout.addWidget(QLabel("Sphere Opacity:"), 1, 0)
        opacity_layout.addWidget(self.sphere_alpha_spin, 1, 1)
        opacity_layout.addWidget(QLabel("Sphere Centre Opacity:"), 2, 0)
        opacity_layout.addWidget(self.sphere_center_alpha_spin, 2, 1)
        opacity_layout.addWidget(QLabel("Plane Opacity:"), 3, 0)
        opacity_layout.addWidget(self.slice_alpha_spin, 3, 1)
        opacity_layout.addWidget(QLabel("Selected Plane Opacity:"), 4, 0)
        opacity_layout.addWidget(self.selected_slice_alpha_spin, 4, 1)
        opacity_layout.setRowStretch(5, 1)

        self.line_method_combo = QComboBox()
        self.line_method_combo.addItems(["auto", "single", "tree"])
        general_layout.addWidget(QLabel("Line Method:"), 0, 0)
        general_layout.addWidget(self.line_method_combo, 0, 1)

        self.slice_interval_spin = QDoubleSpinBox()
        self.slice_interval_spin.setRange(0.1, 100.0)
        self.slice_interval_spin.setDecimals(3)
        self.slice_interval_spin.setSingleStep(0.1)
        general_layout.addWidget(QLabel("Plane Spacing (mm):"), 1, 0)
        general_layout.addWidget(self.slice_interval_spin, 1, 1)

        self.shell_point_spacing_spin = QDoubleSpinBox()
        self.shell_point_spacing_spin.setRange(0.1, 100.0)
        self.shell_point_spacing_spin.setDecimals(3)
        self.shell_point_spacing_spin.setSingleStep(0.1)
        general_layout.addWidget(QLabel("Shell Point Spacing (mm):"), 2, 0)
        general_layout.addWidget(self.shell_point_spacing_spin, 2, 1)

        self.start_z_spin = QDoubleSpinBox()
        self.start_z_spin.setRange(-1000.0, 1000.0)
        self.start_z_spin.setDecimals(3)
        self.start_z_spin.setSingleStep(0.1)
        general_layout.addWidget(QLabel("Start Z (mm):"), 3, 0)
        general_layout.addWidget(self.start_z_spin, 3, 1)

        self.start_z_tolerance_spin = QDoubleSpinBox()
        self.start_z_tolerance_spin.setRange(0.0, 1000.0)
        self.start_z_tolerance_spin.setDecimals(3)
        self.start_z_tolerance_spin.setSingleStep(0.1)
        general_layout.addWidget(QLabel("Start Z Tolerance:"), 4, 0)
        general_layout.addWidget(self.start_z_tolerance_spin, 4, 1)

        self.knn_k_spin = QSpinBox()
        self.knn_k_spin.setRange(1, 100)
        centreline_layout.addWidget(QLabel("k-Value:"), 0, 0)
        centreline_layout.addWidget(self.knn_k_spin, 0, 1)

        self.sphere_min_diameter_spin = QDoubleSpinBox()
        self.sphere_min_diameter_spin.setRange(0.0, 10000.0)
        self.sphere_min_diameter_spin.setDecimals(3)
        self.sphere_min_diameter_spin.setSingleStep(0.1)
        centreline_layout.addWidget(QLabel("Min Sphere Diameter (mm):"), 1, 0)
        centreline_layout.addWidget(self.sphere_min_diameter_spin, 1, 1)

        self.sphere_max_diameter_spin = QDoubleSpinBox()
        self.sphere_max_diameter_spin.setRange(0.0, 10000.0)
        self.sphere_max_diameter_spin.setDecimals(3)
        self.sphere_max_diameter_spin.setSingleStep(0.1)
        centreline_layout.addWidget(QLabel("Max Sphere Diameter (mm):"), 2, 0)
        centreline_layout.addWidget(self.sphere_max_diameter_spin, 2, 1)

        self.overlap_factor_spin = QDoubleSpinBox()
        self.overlap_factor_spin.setRange(0.1, 10.0)
        self.overlap_factor_spin.setDecimals(3)
        self.overlap_factor_spin.setSingleStep(0.1)
        centreline_layout.addWidget(QLabel("Overlap Factor:"), 3, 0)
        centreline_layout.addWidget(self.overlap_factor_spin, 3, 1)

        self.spline_s_spin = QDoubleSpinBox()
        self.spline_s_spin.setRange(0.0, 1000.0)
        self.spline_s_spin.setDecimals(3)
        self.spline_s_spin.setSingleStep(0.1)
        centreline_layout.addWidget(QLabel("Spline Smoothing:"), 4, 0)
        centreline_layout.addWidget(self.spline_s_spin, 4, 1)

        self.centreline_samples_spin = QSpinBox()
        self.centreline_samples_spin.setRange(10, 100000)
        centreline_layout.addWidget(QLabel("Centreline Samples:"), 5, 0)
        centreline_layout.addWidget(self.centreline_samples_spin, 5, 1)

        self.centreline_extension_spin = QDoubleSpinBox()
        self.centreline_extension_spin.setRange(0.0, 10000.0)
        self.centreline_extension_spin.setDecimals(3)
        self.centreline_extension_spin.setSingleStep(0.1)
        centreline_layout.addWidget(QLabel("Extension Length (mm):"), 6, 0)
        centreline_layout.addWidget(self.centreline_extension_spin, 6, 1)

        self.tree_sphere_graph_k_spin = QSpinBox()
        self.tree_sphere_graph_k_spin.setRange(1, 100)
        centreline_layout.addWidget(QLabel("Tree Sphere Graph K:"), 7, 0)
        centreline_layout.addWidget(self.tree_sphere_graph_k_spin, 7, 1)

        self.graph_strategy_combo = QComboBox()
        self.graph_strategy_combo.addItems(["mst", "knn", "complete_mst"])
        centreline_layout.addWidget(QLabel("Graph Strategy:"), 8, 0)
        centreline_layout.addWidget(self.graph_strategy_combo, 8, 1)

        self.nurbs_surface_count_spin = QSpinBox()
        self.nurbs_surface_count_spin.setRange(1, 1000)
        self.enable_nurbs_check = QCheckBox("Enable NURBS Surfaces")
        surface_layout.addWidget(self.enable_nurbs_check, 0, 0, 1, 2)
        surface_layout.addWidget(QLabel("NURBS Surface Count:"), 1, 0)
        surface_layout.addWidget(self.nurbs_surface_count_spin, 1, 1)

        self.nurbs_angle_tolerance_spin = QDoubleSpinBox()
        self.nurbs_angle_tolerance_spin.setRange(0.1, 180.0)
        self.nurbs_angle_tolerance_spin.setDecimals(2)
        self.nurbs_angle_tolerance_spin.setSingleStep(0.1)
        surface_layout.addWidget(QLabel("Angle Tolerance (deg):"), 2, 0)
        surface_layout.addWidget(self.nurbs_angle_tolerance_spin, 2, 1)

        self.nurbs_min_points_spin = QSpinBox()
        self.nurbs_min_points_spin.setRange(1, 1000)
        surface_layout.addWidget(QLabel("Min Points per Surface:"), 3, 0)
        surface_layout.addWidget(self.nurbs_min_points_spin, 3, 1)

        self.adaptive_spacing_check = QCheckBox("Adaptive Outer-Corner Spacing")
        filtering_layout.addWidget(self.adaptive_spacing_check, 0, 0, 1, 2)

        self.adaptive_min_factor_spin = QDoubleSpinBox()
        self.adaptive_min_factor_spin.setRange(0.05, 1.0)
        self.adaptive_min_factor_spin.setDecimals(2)
        self.adaptive_min_factor_spin.setSingleStep(0.1)
        filtering_layout.addWidget(QLabel("Min Adaptive Factor:"), 1, 0)
        filtering_layout.addWidget(self.adaptive_min_factor_spin, 1, 1)

        self.outer_corner_safety_spin = QDoubleSpinBox()
        self.outer_corner_safety_spin.setRange(0.0, 10.0)
        self.outer_corner_safety_spin.setDecimals(2)
        self.outer_corner_safety_spin.setSingleStep(0.1)
        filtering_layout.addWidget(QLabel("Outer Corner Safety:"), 2, 0)
        filtering_layout.addWidget(self.outer_corner_safety_spin, 2, 1)

        self.export_csv_check = QCheckBox("Export CSV")
        filtering_layout.addWidget(self.export_csv_check, 3, 0, 1, 2)

        self.export_json_check = QCheckBox("Export Layer JSON")
        filtering_layout.addWidget(self.export_json_check, 4, 0, 1, 2)

        self.max_point_distance_spin = QDoubleSpinBox()
        self.max_point_distance_spin.setRange(0.0, 10000.0)
        self.max_point_distance_spin.setDecimals(3)
        self.max_point_distance_spin.setSingleStep(1.0)
        filtering_layout.addWidget(QLabel("Max Point Distance (mm):"), 5, 0)
        filtering_layout.addWidget(self.max_point_distance_spin, 5, 1)

        self.limit_curve_bbox_check = QCheckBox("Limit Curves To Bounding Box")
        filtering_layout.addWidget(self.limit_curve_bbox_check, 6, 0, 1, 2)

        self.curve_bbox_padding_spin = QDoubleSpinBox()
        self.curve_bbox_padding_spin.setRange(0.0, 1.0)
        self.curve_bbox_padding_spin.setDecimals(2)
        self.curve_bbox_padding_spin.setSingleStep(0.05)
        filtering_layout.addWidget(QLabel("BBox Padding Ratio:"), 7, 0)
        filtering_layout.addWidget(self.curve_bbox_padding_spin, 7, 1)

        defaults_buttons = QHBoxLayout()
        self.restore_defaults_btn = QPushButton("Restore Working Defaults")
        self.save_defaults_btn = QPushButton("Save as Default")
        defaults_buttons.addWidget(self.restore_defaults_btn)
        defaults_buttons.addWidget(self.save_defaults_btn)
        algorithm_layout.addLayout(defaults_buttons)

        left_layout.addWidget(algorithm_group)

        self.run_button = QPushButton("Run Centreline Shell Extraction")
        self.run_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.run_button.clicked.connect(self._start_slicing)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_slicing)
        run_buttons = QHBoxLayout()
        run_buttons.addWidget(self.run_button)
        run_buttons.addWidget(self.stop_button)
        left_layout.addLayout(run_buttons)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        left_layout.addWidget(self.progress_bar)

        self.result_summary = QLabel("No results yet.")
        self.result_summary.setWordWrap(True)
        left_layout.addWidget(self.result_summary)

        log_group = QGroupBox("Run Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        left_layout.addWidget(log_group)
        left_layout.setStretchFactor(log_group, 1)

        self.splitter.addWidget(left)

        self.plotter, self.vis_manager, self._viewer_backend_mode, self._viewer_status_message = (
            self._create_viewer_backend()
        )

        self.layer_slider = QSlider(Qt.Vertical)
        self.layer_slider.setRange(0, 0)
        self.layer_spinbox = QSpinBox()
        self.layer_spinbox.setRange(0, 0)
        self.layer_distance_label = QLabel("Distance: -")
        self.layer_distance_label.setAlignment(Qt.AlignCenter)
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(self.layer_slider)
        slider_layout.addWidget(self.layer_spinbox)
        slider_layout.addWidget(self.layer_distance_label)

        viewer_widget = QWidget()
        viewer_layout = QHBoxLayout(viewer_widget)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.addLayout(slider_layout)
        render_panel = QWidget()
        render_panel_layout = QVBoxLayout(render_panel)
        render_panel_layout.setContentsMargins(0, 0, 0, 0)
        if self._viewer_status_message:
            viewer_notice = QLabel(VIEWER_FALLBACK_SUMMARY)
            viewer_notice.setWordWrap(True)
            viewer_notice.setToolTip(self._viewer_status_message)
            viewer_notice.setStyleSheet(
                "padding:8px; border:1px solid #7c6d2e; background:#3f3a1f; color:#f1e2a8;"
            )
            render_panel_layout.addWidget(viewer_notice)
        render_surface = (
            self.plotter
            if self.plotter is not None
            else self._build_viewer_placeholder(self._viewer_status_message)
        )
        render_panel_layout.addWidget(render_surface, 1)
        viewer_layout.addWidget(render_panel)
        viewer_layout.setStretch(1, 1)
        self.splitter.addWidget(viewer_widget)

    def _setup_menu_bar(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)
        settings_menu = menu_bar.addMenu("Settings")
        settings_action = QAction("Application Settings...", self)
        settings_action.triggered.connect(self._open_settings_dialog)
        settings_menu.addAction(settings_action)

    def _apply_appearance_settings(self):
        font_size = self.settings_manager.get("global_font_size", 10)
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{background:#2b2b2b; color:#eee; font-size:{font_size}pt;}}
            QGroupBox {{font-weight:bold; border:1px solid #555; border-radius:5px; margin:5px; padding-top:10px;}}
            QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{background:#3c3c3c; border:1px solid #555;}}
            QTabWidget::pane {{border:1px solid #555; background:#2f2f2f;}}
            QTabBar::tab {{background:#444; color:#eee; border:1px solid #555; padding:6px 10px;}}
            QTabBar::tab:selected {{background:#666; color:#fff; border-bottom-color:#666;}}
            QTabBar::tab:!selected {{background:#333; color:#cfcfcf;}}
            QTabBar::tab:hover {{background:#555; color:#fff;}}
            QPushButton {{background:#4a4a4a; border:1px solid #666; border-radius:4px; padding:6px;}}
            QPushButton:hover {{background:#5a5a5a;}}
            """
        )
        log_font = QFont()
        log_font.setFamily("Consolas")
        log_font.setPointSize(self.settings_manager.get("log_font_size", 9))
        self.log_text.setFont(log_font)
        self.splitter.setSizes(self.settings_manager.get("splitter_sizes", [500, 1100]))
        self.vis_manager.update_appearance(self.settings_manager)

    def _selected_sphere_generation_method(self) -> str:
        return self.sphere_generation_combo.currentData() or "auto"

    def _set_sphere_generation_method(self, method: str):
        index = self.sphere_generation_combo.findData(method or "auto")
        if index < 0:
            index = self.sphere_generation_combo.findData("auto")
        self.sphere_generation_combo.setCurrentIndex(max(index, 0))

    def _connect_signals(self):
        slicer_logger = logging.getLogger("slicer_core")
        slicer_logger.setLevel(logging.DEBUG)
        self.log_emitter.log_signal.connect(self.log_text.append)
        gui_handler = QtLogHandler(self.log_emitter)
        gui_handler.setLevel(logging.DEBUG)
        slicer_logger.addHandler(gui_handler)

        for widget in [
            self.show_mesh_check,
            self.show_slices_check,
            self.show_centerline_check,
            self.show_spheres_check,
            self.show_surface_curves_check,
            self.show_bbox_limit_check,
            self.show_normal_check,
        ]:
            widget.stateChanged.connect(self._on_view_setting_changed)
        self.perspective_check.toggled.connect(self._on_view_setting_changed)
        for picker in [
            self.bg_picker,
            self.mesh_picker,
            self.centerline_picker,
            self.sphere_picker,
            self.slice_picker,
            self.selected_slice_picker,
            self.overlay_picker,
        ]:
            picker.color_changed.connect(self._on_colour_setting_changed)
        for widget in [
            self.centerline_thickness_spin,
            self.sphere_thickness_spin,
            self.plane_thickness_spin,
            self.selected_plane_thickness_spin,
            self.sphere_center_point_size_spin,
            self.shell_point_size_spin,
            self.plane_point_size_spin,
            self.mesh_alpha_spin,
            self.sphere_alpha_spin,
            self.sphere_center_alpha_spin,
            self.slice_alpha_spin,
            self.selected_slice_alpha_spin,
        ]:
            widget.valueChanged.connect(self._on_colour_setting_changed)
        self.layer_slider.valueChanged.connect(self._on_layer_slider_changed)
        self.layer_spinbox.valueChanged.connect(self._on_layer_spinbox_changed)
        self.slice_interval_spin.valueChanged.connect(self._on_slice_interval_changed)
        for widget in [
            self.sphere_generation_combo,
            self.line_method_combo,
            self.shell_point_spacing_spin,
            self.start_z_spin,
            self.start_z_tolerance_spin,
            self.knn_k_spin,
            self.sphere_min_diameter_spin,
            self.sphere_max_diameter_spin,
            self.overlap_factor_spin,
            self.spline_s_spin,
            self.centreline_samples_spin,
            self.centreline_extension_spin,
            self.tree_sphere_graph_k_spin,
            self.graph_strategy_combo,
            self.enable_nurbs_check,
            self.nurbs_surface_count_spin,
            self.nurbs_angle_tolerance_spin,
            self.nurbs_min_points_spin,
            self.adaptive_spacing_check,
            self.adaptive_min_factor_spin,
            self.outer_corner_safety_spin,
            self.export_csv_check,
            self.export_json_check,
            self.max_point_distance_spin,
            self.limit_curve_bbox_check,
            self.curve_bbox_padding_spin,
        ]:
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._apply_algorithm_controls_to_settings)
            if hasattr(widget, "currentTextChanged"):
                widget.currentTextChanged.connect(self._apply_algorithm_controls_to_settings)
            if hasattr(widget, "toggled"):
                widget.toggled.connect(self._apply_algorithm_controls_to_settings)
        self.adaptive_spacing_check.toggled.connect(self._update_algorithm_control_states)
        self.limit_curve_bbox_check.toggled.connect(self._update_algorithm_control_states)
        self.enable_nurbs_check.toggled.connect(self._update_algorithm_control_states)
        self.reset_view_btn.clicked.connect(self.vis_manager.reset_view)
        self.view_x_btn.clicked.connect(self.vis_manager.view_x)
        self.view_y_btn.clicked.connect(self.vis_manager.view_y)
        self.view_z_btn.clicked.connect(self.vis_manager.view_z)
        self.print_plot_btn.clicked.connect(self._print_current_plot)
        self.load_batch_csv_btn.clicked.connect(self._browse_batch_plan_csv)
        self.run_batch_button.clicked.connect(self._start_planned_batch)
        self.restore_defaults_btn.clicked.connect(self._restore_working_defaults)
        self.save_defaults_btn.clicked.connect(self._save_current_as_defaults)
        self.splitter.splitterMoved.connect(
            lambda: self.settings_manager.set("splitter_sizes", self.splitter.sizes())
        )

    def _load_config_into_ui(self):
        self._loading_algorithm_ui = True
        self.file_path_edit.setText(
            self.settings_manager.get("stl_file", str(SlicerConfig().stl_file))
        )
        self.show_mesh_check.setChecked(self.settings_manager.get("show_mesh", True))
        self.show_slices_check.setChecked(self.settings_manager.get("show_slices", True))
        self.show_centerline_check.setChecked(self.settings_manager.get("show_centerline", True))
        self.show_spheres_check.setChecked(self.settings_manager.get("show_spheres", False))
        self.show_surface_curves_check.setChecked(self.settings_manager.get("show_surface_curves", True))
        self.show_bbox_limit_check.setChecked(self.settings_manager.get("show_bbox_limit", True))
        self.show_normal_check.setChecked(self.settings_manager.get("show_layer_normal", True))
        self.perspective_check.setChecked(self.settings_manager.get("perspective_view", False))
        self.bg_picker.set_color(self.settings_manager.get_as_qcolor("bg_color"))
        self.mesh_picker.set_color(self.settings_manager.get_as_qcolor("mesh_color"))
        self.centerline_picker.set_color(self.settings_manager.get_as_qcolor("centerline_color"))
        self.sphere_picker.set_color(self.settings_manager.get_as_qcolor("sphere_color"))
        self.slice_picker.set_color(self.settings_manager.get_as_qcolor("slice_color_normal"))
        self.selected_slice_picker.set_color(self.settings_manager.get_as_qcolor("slice_color_selected"))
        self.overlay_picker.set_color(self.settings_manager.get_as_qcolor("overlay_font_color"))
        self.centerline_thickness_spin.setValue(self.settings_manager.get("centerline_thickness", 4))
        self.sphere_thickness_spin.setValue(self.settings_manager.get("sphere_thickness", 1))
        self.plane_thickness_spin.setValue(self.settings_manager.get("slice_thickness_normal", 1))
        self.selected_plane_thickness_spin.setValue(self.settings_manager.get("slice_thickness_selected", 4))
        self.sphere_center_point_size_spin.setValue(self.settings_manager.get("sphere_center_point_size", 12))
        self.shell_point_size_spin.setValue(self.settings_manager.get("shell_point_size", 0.5))
        self.plane_point_size_spin.setValue(self.settings_manager.get("plane_point_size", 0.5))
        self.mesh_alpha_spin.setValue(self.settings_manager.get("mesh_alpha", 0.1))
        self.sphere_alpha_spin.setValue(self.settings_manager.get("sphere_alpha", 1.0))
        self.sphere_center_alpha_spin.setValue(self.settings_manager.get("sphere_center_alpha", 1.0))
        self.slice_alpha_spin.setValue(self.settings_manager.get("slice_alpha", 1.0))
        self.selected_slice_alpha_spin.setValue(self.settings_manager.get("selected_slice_alpha", 1.0))
        self._set_sphere_generation_method(self.settings_manager.get("sphere_generation_method", "auto"))
        self.line_method_combo.setCurrentText(self.settings_manager.get("line_method", "auto"))
        self.slice_interval_spin.setValue(self.settings_manager.get("plane_spacing", 2.0))
        self.shell_point_spacing_spin.setValue(self.settings_manager.get("shell_point_spacing", 1.0))
        self.start_z_spin.setValue(self.settings_manager.get("start_z", 0.0))
        self.start_z_tolerance_spin.setValue(self.settings_manager.get("start_z_tolerance", 5.0))
        self.knn_k_spin.setValue(self.settings_manager.get("knn_k", 4))
        self.sphere_min_diameter_spin.setValue(self.settings_manager.get("sphere_min_diameter", 0.0))
        self.sphere_max_diameter_spin.setValue(self.settings_manager.get("sphere_max_diameter", 0.0))
        self.overlap_factor_spin.setValue(self.settings_manager.get("overlap_factor", 1.0))
        self.spline_s_spin.setValue(self.settings_manager.get("spline_s", 2.0))
        self.centreline_samples_spin.setValue(self.settings_manager.get("centreline_samples", 200))
        self.centreline_extension_spin.setValue(self.settings_manager.get("centreline_extension_length", 0.0))
        self.tree_sphere_graph_k_spin.setValue(self.settings_manager.get("tree_sphere_graph_k", 4))
        self.graph_strategy_combo.setCurrentText(self.settings_manager.get("graph_strategy", "mst"))
        self.enable_nurbs_check.setChecked(self.settings_manager.get("enable_nurbs", True))
        self.nurbs_surface_count_spin.setValue(self.settings_manager.get("nurbs_surface_count", 16))
        self.nurbs_angle_tolerance_spin.setValue(self.settings_manager.get("nurbs_angle_tolerance_deg", 20.0))
        self.nurbs_min_points_spin.setValue(self.settings_manager.get("nurbs_min_points", 3))
        self.adaptive_spacing_check.setChecked(self.settings_manager.get("adaptive_plane_spacing", True))
        self.adaptive_min_factor_spin.setValue(self.settings_manager.get("adaptive_spacing_min_factor", 0.6))
        self.outer_corner_safety_spin.setValue(self.settings_manager.get("outer_corner_spacing_safety", 0.35))
        self.export_csv_check.setChecked(self.settings_manager.get("export_csv", True))
        self.export_json_check.setChecked(self.settings_manager.get("export_layers_json", True))
        self.max_point_distance_spin.setValue(self.settings_manager.get("max_point_distance_from_centreline", 0.0))
        self.limit_curve_bbox_check.setChecked(self.settings_manager.get("limit_curve_bbox", True))
        self.curve_bbox_padding_spin.setValue(self.settings_manager.get("curve_bbox_padding_ratio", 0.1))
        self._update_algorithm_control_states()
        self._loading_algorithm_ui = False

    def _update_algorithm_control_states(self):
        adaptive_enabled = self.adaptive_spacing_check.isChecked()
        bbox_enabled = self.limit_curve_bbox_check.isChecked()
        nurbs_enabled = self.enable_nurbs_check.isChecked()
        self.adaptive_min_factor_spin.setEnabled(adaptive_enabled)
        self.outer_corner_safety_spin.setEnabled(adaptive_enabled)
        self.curve_bbox_padding_spin.setEnabled(bbox_enabled)
        self.nurbs_surface_count_spin.setEnabled(nurbs_enabled)
        self.nurbs_angle_tolerance_spin.setEnabled(nurbs_enabled)
        self.nurbs_min_points_spin.setEnabled(nurbs_enabled)

    def _browse_stl(self):
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Select STL File",
            str(SlicerConfig().stl_file.parent),
            "STL Files (*.stl)",
        )
        if fn:
            self.file_path_edit.setText(fn)
            self.settings_manager.set("stl_file", fn)
            self._last_run_stl_path = None
            self._rerun_smoothing_passes = 0
            self.settings_manager.set("rerun_smoothing_passes", 0)
            try:
                mesh = trimesh.load(normalize_path(fn))
                self.slicer_object = None
                self.layer_slider.setRange(0, 0)
                self.layer_spinbox.setRange(0, 0)
                self.result_summary.setText("Mesh preview loaded. Run the extractor to compute centrelines and shell points.")
                self.layer_distance_label.setText("Distance: -")
                self.vis_manager.preview_mesh(
                    mesh,
                    self.settings_manager.get("mesh_alpha", 0.1),
                    self.settings_manager.get_as_qcolor("mesh_color"),
                )
            except Exception as exc:
                logging.getLogger("slicer_core").error("Failed to preview mesh: %s", exc)

    def _browse_batch_plan_csv(self):
        template_dir = Path(__file__).resolve().parent / "batch_templates"
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Select Planned Parameter Batch CSV",
            str(template_dir if template_dir.exists() else SlicerConfig().output_dir),
            "CSV Files (*.csv)",
        )
        if not fn:
            return

        try:
            self._batch_plan = load_batch_plan_csv(Path(fn))
        except BatchPlanError as exc:
            QMessageBox.critical(self, "Batch Plan Error", str(exc))
            return

        self.batch_plan_path_edit.setText(str(self._batch_plan.path))
        summary = f"Loaded {len(self._batch_plan.rows)} planned runs."
        if self._batch_plan.ignored_headers:
            summary += (
                " Ignored headers: "
                + ", ".join(self._batch_plan.ignored_headers)
            )
        self.batch_plan_summary.setText(summary)
        self.run_batch_button.setEnabled(self.worker_thread is None)

    def _build_settings_manager_from_dict(self, settings_dict: dict) -> SettingsManager:
        settings = SettingsManager()
        settings.settings = dict(settings_dict)
        return settings

    def _create_batch_output_root(self) -> Path:
        plan_name = (
            self._batch_plan.path.stem
            if self._batch_plan is not None
            else "planned_batch"
        )
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_root = SlicerConfig().output_dir / "planned_batches" / f"{plan_name}_{timestamp}"
        output_root.mkdir(parents=True, exist_ok=True)
        if self._batch_plan is not None:
            try:
                shutil.copy2(self._batch_plan.path, output_root / self._batch_plan.path.name)
            except Exception as exc:
                logging.getLogger("slicer_core").warning(
                    "Could not copy batch plan CSV into %s: %s",
                    output_root,
                    exc,
                )
        return output_root

    def _initialize_batch_summary_csv(self):
        if self._batch_output_root is None:
            return

        self._batch_summary_csv_path = self._batch_output_root / "batch_summary.csv"
        with self._batch_summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "trial",
                    "label",
                    "source_row",
                    "stl_file",
                    "knn_k",
                    "overlap_factor",
                    "centreline_samples",
                    "line_method",
                    "sphere_generation_method",
                    "runtime_seconds",
                    "branch_count",
                    "plane_count",
                    "shell_point_count",
                    "output_directory",
                    "run_parameters_csv",
                    "view_y_image",
                    "status",
                    "message",
                ]
            )

    def _append_batch_summary_row(
        self,
        *,
        status: str,
        message: str = "",
        slicer: Optional[CentrelineShellSlicer] = None,
    ):
        if self._batch_summary_csv_path is None or self._active_batch_row is None:
            return

        summary = slicer.pass_results[-1].summary if slicer and slicer.pass_results else {}
        settings_source = self._active_batch_run_settings.settings if hasattr(self, "_active_batch_run_settings") else {}

        with self._batch_summary_csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    self._active_batch_row.index,
                    self._active_batch_row.label,
                    self._active_batch_row.source_row_number,
                    settings_source.get("stl_file", ""),
                    settings_source.get("knn_k", ""),
                    settings_source.get("overlap_factor", ""),
                    settings_source.get("centreline_samples", ""),
                    summary.get("line_method", settings_source.get("line_method", "")),
                    summary.get(
                        "sphere_generation_method",
                        settings_source.get("sphere_generation_method", ""),
                    ),
                    summary.get("runtime_seconds", ""),
                    summary.get("branch_count", ""),
                    summary.get("plane_count", ""),
                    summary.get("shell_point_count", ""),
                    summary.get("output_directory", ""),
                    summary.get("run_parameters_csv", ""),
                    summary.get("view_y_image", ""),
                    status,
                    message,
                ]
            )

    def _start_worker(
        self,
        settings: SettingsManager,
        run_config: SlicerConfig,
        *,
        clear_log: bool,
        summary_text: str,
        primary_button_text: str,
        batch_button_text: Optional[str] = None,
    ):
        if clear_log:
            self.log_text.clear()
        self.result_summary.setText(summary_text)
        self._set_buttons_enabled(False)
        self.run_button.setText(primary_button_text)
        if batch_button_text is not None:
            self.run_batch_button.setText(batch_button_text)
        self.worker_thread = SlicerWorker(settings, run_config)
        self.worker_thread.slicer_finished.connect(self._finish_slicing_success)
        self.worker_thread.slicer_cancelled.connect(self._finish_slicing_cancelled)
        self.worker_thread.slicer_failed.connect(self._finish_slicing_failure)
        self.worker_thread.progress_updated.connect(self._handle_worker_progress)
        self.worker_thread.start()

    def _start_planned_batch(self):
        if self.worker_thread is not None:
            return
        if self._batch_plan is None or not self._batch_plan.rows:
            QMessageBox.information(
                self,
                "No Batch Plan Loaded",
                "Load a batch CSV first. Supported columns include k, of, and samples.",
            )
            return

        self.settings_manager.set("stl_file", self.file_path_edit.text())
        self._apply_algorithm_controls_to_settings()
        self._batch_base_settings = self._collect_current_gui_settings()
        self._batch_base_settings["rerun_smoothing_passes"] = 0
        self._batch_base_settings["export_csv"] = True
        self._batch_active = True
        self._batch_cancel_requested = False
        self._batch_current_index = 0
        self._batch_output_root = self._create_batch_output_root()
        self._initialize_batch_summary_csv()
        self._launch_next_batch_run(clear_log=True)

    def _launch_next_batch_run(self, *, clear_log: bool = False):
        if self._batch_plan is None or self._batch_base_settings is None:
            return

        if self._batch_cancel_requested or self._batch_current_index >= len(self._batch_plan.rows):
            self._finish_planned_batch()
            return

        self._active_batch_row = self._batch_plan.rows[self._batch_current_index]
        run_settings = dict(self._batch_base_settings)
        run_settings.update(self._active_batch_row.overrides)
        run_settings["export_csv"] = True
        run_settings["rerun_smoothing_passes"] = 0

        if not str(run_settings.get("stl_file", "")).strip():
            run_settings["stl_file"] = self.file_path_edit.text()
        if not str(run_settings.get("stl_file", "")).strip():
            self._append_batch_summary_row(
                status="failed",
                message="No STL file was provided in the GUI or batch CSV.",
            )
            QMessageBox.critical(
                self,
                "Batch Run Error",
                "The planned batch could not start because no STL file is selected.",
            )
            self._batch_cancel_requested = True
            self._finish_planned_batch()
            return

        self._active_batch_run_settings = self._build_settings_manager_from_dict(run_settings)
        run_index = self._batch_current_index + 1
        total_runs = len(self._batch_plan.rows)
        mesh_path = normalize_path(str(run_settings["stl_file"]))
        run_config = SlicerConfig(
            stl_file=mesh_path,
            output_dir=self._batch_output_root or SlicerConfig().output_dir,
        )
        self.progress_bar.setValue(int(round((self._batch_current_index / max(total_runs, 1)) * 100)))
        self._start_worker(
            self._active_batch_run_settings,
            run_config,
            clear_log=clear_log,
            summary_text=(
                f"Running planned batch {run_index}/{total_runs}: "
                f"{self._active_batch_row.label}"
            ),
            primary_button_text="Run Centreline Shell Extraction",
            batch_button_text=f"Running Batch {run_index}/{total_runs}...",
        )

    def _handle_worker_progress(self, progress: int):
        if self._batch_active and self._batch_plan is not None and self._batch_plan.rows:
            total_runs = len(self._batch_plan.rows)
            overall_progress = (
                (self._batch_current_index + (progress / 100.0)) / total_runs
            ) * 100.0
            self.progress_bar.setValue(int(round(overall_progress)))
            return
        self.progress_bar.setValue(progress)

    def _finish_planned_batch(self):
        completed_runs = self._batch_current_index
        total_runs = len(self._batch_plan.rows) if self._batch_plan is not None else 0
        summary_path = str(self._batch_summary_csv_path) if self._batch_summary_csv_path else "not saved"

        if self._batch_cancel_requested:
            self.result_summary.setText(
                f"Planned batch cancelled after {completed_runs} of {total_runs} runs. "
                f"Summary: {summary_path}"
            )
        else:
            self.progress_bar.setValue(100)
            self.result_summary.setText(
                f"Planned batch complete. Executed {completed_runs} of {total_runs} runs. "
                f"Summary: {summary_path}"
            )

        self._batch_active = False
        self._batch_cancel_requested = False
        self._active_batch_row = None
        self._active_batch_run_settings = None
        self._batch_base_settings = None
        self.worker_thread = None
        self.run_button.setText("Run Centreline Shell Extraction")
        self.run_batch_button.setText("Run Planned Parameter Batch")
        self._set_buttons_enabled(True)

    def _open_settings_dialog(self):
        if self.settings_dialog is None:
            self.settings_dialog = SettingsDialog(self.settings_manager)
            self.settings_dialog.settings_updated.connect(self._on_settings_updated)
            self.settings_dialog.finished.connect(lambda: setattr(self, "settings_dialog", None))
        self.settings_dialog.show()

    def _on_settings_updated(self):
        self._apply_appearance_settings()
        self._load_config_into_ui()
        if self.slicer_object:
            self.vis_manager.set_slicer_object(self.slicer_object)
            self._refresh_viewer()

    def _set_buttons_enabled(self, enabled: bool):
        self.run_button.setEnabled(enabled)
        self.load_batch_csv_btn.setEnabled(enabled)
        self.batch_plan_path_edit.setEnabled(enabled)
        self.run_batch_button.setEnabled(enabled and self._batch_plan is not None)
        self.stop_button.setEnabled(not enabled)

    def _start_slicing(self):
        self.progress_bar.setValue(0)
        self.settings_manager.set("stl_file", self.file_path_edit.text())
        self._apply_algorithm_controls_to_settings()
        self._rerun_smoothing_passes = 0
        self._last_run_stl_path = str(normalize_path(self.file_path_edit.text()))
        self.settings_manager.set("rerun_smoothing_passes", 0)
        self._active_batch_run_settings = self.settings_manager
        self._start_worker(
            self.settings_manager,
            SlicerConfig(stl_file=normalize_path(self.file_path_edit.text())),
            clear_log=True,
            summary_text="Running centreline shell extraction...",
            primary_button_text="Running...",
            batch_button_text="Run Planned Parameter Batch",
        )

    def _stop_slicing(self):
        if self.worker_thread is None:
            return
        if self._batch_active:
            self._batch_cancel_requested = True
        self.result_summary.setText("Stopping generation...")
        self.stop_button.setEnabled(False)
        logging.getLogger("slicer_core").info("Stop requested by user.")
        self.worker_thread.cancel()

    def _finish_slicing_success(self, slicer: CentrelineShellSlicer):
        self.slicer_object = slicer
        result = slicer.pass_results[-1]
        plane_count = len(result.origins)
        max_index = plane_count - 1 if plane_count else 0
        self.layer_slider.setRange(0, max_index)
        self.layer_spinbox.setRange(0, max_index)
        self.layer_slider.setValue(max_index)
        self.layer_spinbox.setValue(max_index)
        self.result_summary.setText(
            f"Run complete. Method: {result.summary['line_method']}. "
            f"Spheres: {result.summary.get('sphere_generation_method', 'auto')}. "
            f"Branches: {result.summary['branch_count']}, "
            f"Planes: {result.summary['plane_count']}, "
            f"Shell points: {result.summary['shell_point_count']}. "
            f"Surface curves: {result.summary['surface_curve_count']}. "
            f"Runtime: {result.summary.get('runtime_seconds', 0.0):.2f}s."
            + (
                f" Plane spacing: {result.summary['plane_spacing_min']:.2f}"
                f"-{result.summary['plane_spacing_max']:.2f} mm."
                if "plane_spacing_min" in result.summary
                and "plane_spacing_max" in result.summary
                else ""
            )
        )
        # Keep the post-run view focused on the extracted result geometry
        # without forcing optional debug overlays like hitbox spheres.
        self.show_mesh_check.setChecked(False)
        self.show_slices_check.setChecked(True)
        self.show_centerline_check.setChecked(True)
        if result.surface_curves:
            self.show_surface_curves_check.setChecked(True)
        self.vis_manager.set_slicer_object(slicer)
        self._update_layer_distance_label(max_index)
        self._refresh_viewer()
        if result.summary.get("run_parameters_csv"):
            try:
                image_paths = self.vis_manager.save_x_y_views(
                    slicer.run_config.output_dir,
                    slicer.run_timestamp,
                )
                hitbox_image_paths = self.vis_manager.save_hitbox_geometry_views(
                    slicer.run_config.output_dir,
                    slicer.run_timestamp,
                )
                result.summary["view_x_image"] = str(image_paths["x"])
                result.summary["view_y_image"] = str(image_paths["y"])
                result.summary["hitbox_geometry_x_image"] = str(hitbox_image_paths["x"])
                result.summary["hitbox_geometry_y_image"] = str(hitbox_image_paths["y"])
                parameter_csv_path = result.summary.get("run_parameters_csv")
                if parameter_csv_path:
                    with Path(parameter_csv_path).open("a", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(["summary.view_x_image", image_paths["x"]])
                        writer.writerow(["summary.view_y_image", image_paths["y"]])
                        writer.writerow(["summary.hitbox_geometry_x_image", hitbox_image_paths["x"]])
                        writer.writerow(["summary.hitbox_geometry_y_image", hitbox_image_paths["y"]])
                logging.getLogger("slicer_core").info(
                    "Saved X/Y view images to %s and %s",
                    image_paths["x"],
                    image_paths["y"],
                )
                logging.getLogger("slicer_core").info(
                    "Saved hitbox geometry images to %s and %s",
                    hitbox_image_paths["x"],
                    hitbox_image_paths["y"],
                )
            except Exception as exc:
                logging.getLogger("slicer_core").error(
                    "Failed to save X/Y view images: %s",
                    exc,
                )
        if self._batch_active:
            self._append_batch_summary_row(status="success", slicer=slicer)
            self._batch_current_index += 1
            self.worker_thread = None
            self._active_batch_run_settings = None
            if self._batch_cancel_requested or (
                self._batch_plan is not None
                and self._batch_current_index >= len(self._batch_plan.rows)
            ):
                self._finish_planned_batch()
            else:
                self._launch_next_batch_run(clear_log=False)
            return
        self._cleanup_thread()

    def _finish_slicing_failure(self, msg: str):
        if self._batch_active:
            self._append_batch_summary_row(status="failed", message=msg)
            QMessageBox.critical(
                self,
                "Batch Run Failed",
                f"Planned batch failed on {self._active_batch_row.label if self._active_batch_row else 'the current run'}.\n\n{msg}",
            )
            self.result_summary.setText("Planned batch failed. Check the log and batch summary CSV.")
            self._batch_cancel_requested = True
            self._finish_planned_batch()
            return
        QMessageBox.critical(self, "Slicer Error", f"Slicing failed:\n\n{msg}")
        self.result_summary.setText("Run failed. Check the log for details.")
        self._cleanup_thread()

    def _finish_slicing_cancelled(self, msg: str):
        if self._batch_active:
            self._append_batch_summary_row(status="cancelled", message=msg or "Generation interrupted by user.")
            self._finish_planned_batch()
            return
        self.result_summary.setText(msg or "Generation stopped.")
        self._cleanup_thread()

    def _cleanup_thread(self):
        self._set_buttons_enabled(True)
        self.run_button.setText("Run Centreline Shell Extraction")
        self.run_batch_button.setText("Run Planned Parameter Batch")
        self._active_batch_run_settings = None
        self.worker_thread = None

    def _on_layer_slider_changed(self, value: int):
        self.layer_spinbox.blockSignals(True)
        self.layer_spinbox.setValue(value)
        self.layer_spinbox.blockSignals(False)
        self._update_layer_distance_label(value)
        self._refresh_viewer()

    def _on_layer_spinbox_changed(self, value: int):
        self.layer_slider.blockSignals(True)
        self.layer_slider.setValue(value)
        self.layer_slider.blockSignals(False)
        self._update_layer_distance_label(value)
        self._refresh_viewer()

    def _update_layer_distance_label(self, index: int):
        if not self.slicer_object or not self.slicer_object.pass_results:
            self.layer_distance_label.setText("Distance: -")
            return

        result = self.slicer_object.pass_results[-1]
        if len(result.plane_distances) == 0:
            self.layer_distance_label.setText("Distance: -")
            return

        index = max(0, min(index, len(result.plane_distances) - 1))
        distance = float(result.plane_distances[index])
        self.layer_distance_label.setText(f"Distance: {distance:.2f} mm")

    def _on_slice_interval_changed(self, value: float):
        self.settings_manager.set("plane_spacing", value)

    def _apply_algorithm_controls_to_settings(self, *args):
        if self._loading_algorithm_ui:
            return
        self.settings_manager.set("line_method", self.line_method_combo.currentText())
        self.settings_manager.set("sphere_generation_method", self._selected_sphere_generation_method())
        self.settings_manager.set("plane_spacing", self.slice_interval_spin.value())
        self.settings_manager.set("shell_point_spacing", self.shell_point_spacing_spin.value())
        self.settings_manager.set("start_z", self.start_z_spin.value())
        self.settings_manager.set("start_z_tolerance", self.start_z_tolerance_spin.value())
        self.settings_manager.set("knn_k", self.knn_k_spin.value())
        self.settings_manager.set("sphere_min_diameter", self.sphere_min_diameter_spin.value())
        self.settings_manager.set("sphere_max_diameter", self.sphere_max_diameter_spin.value())
        self.settings_manager.set("overlap_factor", self.overlap_factor_spin.value())
        self.settings_manager.set("spline_s", self.spline_s_spin.value())
        self.settings_manager.set("centreline_samples", self.centreline_samples_spin.value())
        self.settings_manager.set("centreline_extension_length", self.centreline_extension_spin.value())
        self.settings_manager.set("tree_sphere_graph_k", self.tree_sphere_graph_k_spin.value())
        self.settings_manager.set("graph_strategy", self.graph_strategy_combo.currentText())
        self.settings_manager.set("enable_nurbs", self.enable_nurbs_check.isChecked())
        self.settings_manager.set("nurbs_surface_count", self.nurbs_surface_count_spin.value())
        self.settings_manager.set("nurbs_angle_tolerance_deg", self.nurbs_angle_tolerance_spin.value())
        self.settings_manager.set("nurbs_min_points", self.nurbs_min_points_spin.value())
        self.settings_manager.set("adaptive_plane_spacing", self.adaptive_spacing_check.isChecked())
        self.settings_manager.set("adaptive_spacing_min_factor", self.adaptive_min_factor_spin.value())
        self.settings_manager.set("outer_corner_spacing_safety", self.outer_corner_safety_spin.value())
        self.settings_manager.set("export_csv", self.export_csv_check.isChecked())
        self.settings_manager.set("export_layers_json", self.export_json_check.isChecked())
        self.settings_manager.set("max_point_distance_from_centreline", self.max_point_distance_spin.value())
        self.settings_manager.set("limit_curve_bbox", self.limit_curve_bbox_check.isChecked())
        self.settings_manager.set("curve_bbox_padding_ratio", self.curve_bbox_padding_spin.value())

    def _restore_working_defaults(self):
        response = QMessageBox.question(
            self,
            "Restore Defaults?",
            (
                "Are you sure you want to restore the working defaults?\n\n"
                "This will clear the current user settings."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if response != QMessageBox.Yes:
            return

        self.settings_manager.reset_to_defaults()
        self._last_run_stl_path = None
        self._rerun_smoothing_passes = 0
        self.settings_manager.set("rerun_smoothing_passes", 0)
        self._load_config_into_ui()
        self._apply_appearance_settings()
        self.result_summary.setText("Working defaults restored.")

    def _collect_current_gui_settings(self):
        settings = dict(self.settings_manager.settings)

        settings.update(
            {
                "stl_file": self.file_path_edit.text(),
                "splitter_sizes": self.splitter.sizes(),
                "line_method": self.line_method_combo.currentText(),
                "sphere_generation_method": self._selected_sphere_generation_method(),
                "plane_spacing": self.slice_interval_spin.value(),
                "shell_point_spacing": self.shell_point_spacing_spin.value(),
                "start_z": self.start_z_spin.value(),
                "start_z_tolerance": self.start_z_tolerance_spin.value(),
                "knn_k": self.knn_k_spin.value(),
                "sphere_min_diameter": self.sphere_min_diameter_spin.value(),
                "sphere_max_diameter": self.sphere_max_diameter_spin.value(),
                "overlap_factor": self.overlap_factor_spin.value(),
                "spline_s": self.spline_s_spin.value(),
                "centreline_samples": self.centreline_samples_spin.value(),
                "centreline_extension_length": self.centreline_extension_spin.value(),
                "tree_sphere_graph_k": self.tree_sphere_graph_k_spin.value(),
                "graph_strategy": self.graph_strategy_combo.currentText(),
                "enable_nurbs": self.enable_nurbs_check.isChecked(),
                "nurbs_surface_count": self.nurbs_surface_count_spin.value(),
                "nurbs_angle_tolerance_deg": self.nurbs_angle_tolerance_spin.value(),
                "nurbs_min_points": self.nurbs_min_points_spin.value(),
                "adaptive_plane_spacing": self.adaptive_spacing_check.isChecked(),
                "adaptive_spacing_min_factor": self.adaptive_min_factor_spin.value(),
                "outer_corner_spacing_safety": self.outer_corner_safety_spin.value(),
                "export_csv": self.export_csv_check.isChecked(),
                "export_layers_json": self.export_json_check.isChecked(),
                "max_point_distance_from_centreline": self.max_point_distance_spin.value(),
                "limit_curve_bbox": self.limit_curve_bbox_check.isChecked(),
                "curve_bbox_padding_ratio": self.curve_bbox_padding_spin.value(),
                "show_mesh": self.show_mesh_check.isChecked(),
                "show_slices": self.show_slices_check.isChecked(),
                "show_centerline": self.show_centerline_check.isChecked(),
                "show_spheres": self.show_spheres_check.isChecked(),
                "show_surface_curves": self.show_surface_curves_check.isChecked(),
                "show_bbox_limit": self.show_bbox_limit_check.isChecked(),
                "show_layer_normal": self.show_normal_check.isChecked(),
                "perspective_view": self.perspective_check.isChecked(),
                "bg_color": self.bg_picker.get_color().name(),
                "mesh_color": self.mesh_picker.get_color().name(),
                "centerline_color": self.centerline_picker.get_color().name(),
                "sphere_color": self.sphere_picker.get_color().name(),
                "slice_color_normal": self.slice_picker.get_color().name(),
                "slice_color_selected": self.selected_slice_picker.get_color().name(),
                "overlay_font_color": self.overlay_picker.get_color().name(),
                "centerline_thickness": self.centerline_thickness_spin.value(),
                "sphere_thickness": self.sphere_thickness_spin.value(),
                "slice_thickness_normal": self.plane_thickness_spin.value(),
                "slice_thickness_selected": self.selected_plane_thickness_spin.value(),
                "sphere_center_point_size": self.sphere_center_point_size_spin.value(),
                "shell_point_size": self.shell_point_size_spin.value(),
                "plane_point_size": self.plane_point_size_spin.value(),
                "mesh_alpha": self.mesh_alpha_spin.value(),
                "sphere_alpha": self.sphere_alpha_spin.value(),
                "sphere_center_alpha": self.sphere_center_alpha_spin.value(),
                "slice_alpha": self.slice_alpha_spin.value(),
                "selected_slice_alpha": self.selected_slice_alpha_spin.value(),
            }
        )
        return settings

    def _save_current_as_defaults(self):
        current_settings = self._collect_current_gui_settings()
        response = QMessageBox.question(
            self,
            "Save Defaults?",
            (
                "Are you sure you want to save the current GUI settings as the "
                "new working defaults?\n\nThis will overwrite config_defaults.json "
                "and clear user_settings.json."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if response != QMessageBox.Yes:
            return

        saved = self.settings_manager.save_as_defaults(
            settings=current_settings,
            exclude_keys={"rerun_smoothing_passes"}
        )
        if saved:
            self._last_run_stl_path = None
            self._rerun_smoothing_passes = 0
            self.settings_manager.set("rerun_smoothing_passes", 0)
            self.result_summary.setText("Current settings saved as working defaults.")
            QMessageBox.information(
                self,
                "Defaults Saved",
                "Current GUI settings were saved as the new working defaults.",
            )
        else:
            QMessageBox.critical(
                self,
                "Save Failed",
                "Could not save the current settings as defaults. Check the run log.",
            )

    def _on_view_setting_changed(self):
        self.settings_manager.set("show_mesh", self.show_mesh_check.isChecked())
        self.settings_manager.set("show_slices", self.show_slices_check.isChecked())
        self.settings_manager.set("show_centerline", self.show_centerline_check.isChecked())
        self.settings_manager.set("show_spheres", self.show_spheres_check.isChecked())
        self.settings_manager.set("show_surface_curves", self.show_surface_curves_check.isChecked())
        self.settings_manager.set("show_bbox_limit", self.show_bbox_limit_check.isChecked())
        self.settings_manager.set("show_layer_normal", self.show_normal_check.isChecked())
        self.settings_manager.set("perspective_view", self.perspective_check.isChecked())
        self.vis_manager.set_camera_parallel(not self.perspective_check.isChecked())
        self._refresh_viewer()

    def _on_colour_setting_changed(self, *args):
        self.settings_manager.set_from_qcolor("bg_color", self.bg_picker.get_color())
        self.settings_manager.set_from_qcolor("mesh_color", self.mesh_picker.get_color())
        self.settings_manager.set_from_qcolor("centerline_color", self.centerline_picker.get_color())
        self.settings_manager.set_from_qcolor("sphere_color", self.sphere_picker.get_color())
        self.settings_manager.set_from_qcolor("slice_color_normal", self.slice_picker.get_color())
        self.settings_manager.set_from_qcolor("slice_color_selected", self.selected_slice_picker.get_color())
        self.settings_manager.set_from_qcolor("overlay_font_color", self.overlay_picker.get_color())
        self.settings_manager.set("centerline_thickness", self.centerline_thickness_spin.value())
        self.settings_manager.set("sphere_thickness", self.sphere_thickness_spin.value())
        self.settings_manager.set("slice_thickness_normal", self.plane_thickness_spin.value())
        self.settings_manager.set("slice_thickness_selected", self.selected_plane_thickness_spin.value())
        self.settings_manager.set("sphere_center_point_size", self.sphere_center_point_size_spin.value())
        self.settings_manager.set("shell_point_size", self.shell_point_size_spin.value())
        self.settings_manager.set("plane_point_size", self.plane_point_size_spin.value())
        self.settings_manager.set("mesh_alpha", self.mesh_alpha_spin.value())
        self.settings_manager.set("sphere_alpha", self.sphere_alpha_spin.value())
        self.settings_manager.set("sphere_center_alpha", self.sphere_center_alpha_spin.value())
        self.settings_manager.set("slice_alpha", self.slice_alpha_spin.value())
        self.settings_manager.set("selected_slice_alpha", self.selected_slice_alpha_spin.value())
        self._apply_appearance_settings()
        self._refresh_viewer()

    def _current_plot_output_dir(self) -> Path:
        if self.slicer_object is not None:
            return self.slicer_object.run_config.output_dir

        current_settings = self._collect_current_gui_settings()
        stl_path_text = str(current_settings.get("stl_file", "")).strip()
        stl_path = normalize_path(stl_path_text) if stl_path_text else SlicerConfig().stl_file
        return build_output_dir(
            base_output_dir=SlicerConfig().output_dir,
            stl_file=stl_path,
            settings_source=current_settings,
            line_method=current_settings.get("line_method", "auto"),
            sphere_generation_method=current_settings.get(
                "sphere_generation_method",
                "auto",
            ),
        )

    def _print_current_plot(self):
        if not self.vis_manager:
            return

        timestamp = (
            self.slicer_object.run_timestamp
            if self.slicer_object is not None
            else "preview"
        )
        try:
            image_path = self.vis_manager.save_current_view(
                self._current_plot_output_dir(),
                timestamp,
            )
            self.result_summary.setText(f"Current plot saved to {image_path}")
            logging.getLogger("slicer_core").info("Saved current plot to %s", image_path)
        except Exception as exc:
            QMessageBox.critical(self, "Plot Export Error", f"Could not save plot:\n\n{exc}")

    def _refresh_viewer(self):
        if not self.slicer_object:
            return
        options = ViewOptions(
            show_mesh=self.show_mesh_check.isChecked(),
            show_slices=self.show_slices_check.isChecked(),
            show_centerline=self.show_centerline_check.isChecked(),
            show_spheres=self.show_spheres_check.isChecked(),
            show_surface_curves=self.show_surface_curves_check.isChecked(),
            show_bbox_limit=self.show_bbox_limit_check.isChecked(),
            show_plane_normal=self.show_normal_check.isChecked(),
            vis_skip_layers=self.settings_manager.get("vis_skip_layers", 10),
            slice_alpha=self.slice_alpha_spin.value(),
            current_layer_index=self.layer_slider.value(),
        )
        self.vis_manager.refresh_view(options)

    def closeEvent(self, event):
        self.settings_manager.set("splitter_sizes", self.splitter.sizes())
        self.settings_manager.save()
        if self.plotter is not None:
            self.plotter.close()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    pv.set_plot_theme("document")
    win = SlicerGUI()
    win.show()
    sys.exit(app.exec_())
