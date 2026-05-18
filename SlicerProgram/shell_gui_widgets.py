from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from slicer_core.settings_manager import SettingsManager


SPHERE_GENERATION_OPTIONS = [
    ("Auto (path then component)", "auto"),
    ("Skeleton path sampling", "skeleton_paths"),
    ("Component centroids", "component_centroid"),
]


def _set_combo_data(combo: QComboBox, value: str, default: str = "auto"):
    index = combo.findData(value or default)
    if index < 0:
        index = combo.findData(default)
    combo.setCurrentIndex(max(index, 0))


class ColorSelectWidget(QWidget):
    color_changed = pyqtSignal(object)

    def __init__(self, label: str, initial_color):
        super().__init__()
        self._color = initial_color
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(label)
        self.button = QPushButton("")
        self.button.setFixedWidth(50)
        self.button.clicked.connect(self._show_color_dialog)
        layout.addWidget(self.label)
        layout.addStretch()
        layout.addWidget(self.button)
        self._update_button_color()

    def _update_button_color(self):
        self.button.setStyleSheet(
            f"background-color: {self._color.name()}; border: 1px solid #555;"
        )

    def _show_color_dialog(self):
        color = QColorDialog.getColor(self._color, self, "Select Color")
        if color.isValid():
            self._color = color
            self._update_button_color()
            self.color_changed.emit(color)

    def get_color(self):
        return self._color

    def set_color(self, color):
        self._color = color
        self._update_button_color()


def _make_float(label: str, min_val: float, max_val: float, value: float, decimals: int = 3):
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    spin = QDoubleSpinBox()
    spin.setRange(min_val, max_val)
    spin.setDecimals(decimals)
    spin.setValue(value)
    layout.addWidget(QLabel(label))
    layout.addStretch()
    layout.addWidget(spin)
    return widget, spin


def _make_int(label: str, min_val: int, max_val: int, value: int):
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    spin = QSpinBox()
    spin.setRange(min_val, max_val)
    spin.setValue(value)
    layout.addWidget(QLabel(label))
    layout.addStretch()
    layout.addWidget(spin)
    return widget, spin


class AlgorithmSettingsTab(QWidget):
    def __init__(self, settings: SettingsManager):
        super().__init__()
        self.settings = settings
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        sphere_row = QWidget()
        sphere_layout = QHBoxLayout(sphere_row)
        sphere_layout.setContentsMargins(0, 0, 0, 0)
        sphere_layout.addWidget(QLabel("Sphere Algorithm:"))
        sphere_layout.addStretch()
        self.sphere_generation_combo = QComboBox()
        for label, value in SPHERE_GENERATION_OPTIONS:
            self.sphere_generation_combo.addItem(label, value)
        _set_combo_data(
            self.sphere_generation_combo,
            self.settings.get("sphere_generation_method", "auto"),
        )
        sphere_layout.addWidget(self.sphere_generation_combo)
        layout.addWidget(sphere_row)

        line_row = QWidget()
        line_layout = QHBoxLayout(line_row)
        line_layout.setContentsMargins(0, 0, 0, 0)
        line_layout.addWidget(QLabel("Line Method:"))
        line_layout.addStretch()
        self.line_method_combo = QComboBox()
        self.line_method_combo.addItems(["auto", "single", "tree"])
        self.line_method_combo.setCurrentText(self.settings.get("line_method", "auto"))
        line_layout.addWidget(self.line_method_combo)
        layout.addWidget(line_row)

        self.enable_nurbs_check = QCheckBox("Enable NURBS surfaces")
        self.enable_nurbs_check.setChecked(self.settings.get("enable_nurbs", True))
        self.enable_nurbs_check.toggled.connect(self._update_surface_controls)
        layout.addWidget(self.enable_nurbs_check)

        for label, attr, min_val, max_val, key, decimals in [
            ("Plane Spacing (mm):", "plane_spacing_spin", 0.1, 100.0, "plane_spacing", 3),
            ("Shell Point Spacing (mm):", "shell_spacing_spin", 0.1, 100.0, "shell_point_spacing", 3),
            ("Start Z (mm):", "start_z_spin", -1000.0, 1000.0, "start_z", 3),
            ("Start Z Tolerance (mm):", "start_z_tol_spin", 0.0, 100.0, "start_z_tolerance", 3),
            ("Min Adaptive Factor:", "adaptive_min_spin", 0.05, 1.0, "adaptive_spacing_min_factor", 2),
            ("Min Sphere Diameter (mm):", "sphere_min_diameter_spin", 0.0, 10000.0, "sphere_min_diameter", 3),
            ("Max Sphere Diameter (mm):", "sphere_max_diameter_spin", 0.0, 10000.0, "sphere_max_diameter", 3),
            ("Overlap Factor:", "overlap_spin", 0.1, 5.0, "overlap_factor", 3),
            ("Spline Smoothing:", "spline_spin", 0.0, 100.0, "spline_s", 3),
            ("Centreline Extension Length (mm):", "centreline_extension_spin", 0.0, 10000.0, "centreline_extension_length", 3),
            ("Outer Corner Safety:", "corner_safety_spin", 0.0, 5.0, "outer_corner_spacing_safety", 2),
            ("Angle Tolerance (deg):", "angle_tol_spin", 1.0, 90.0, "nurbs_angle_tolerance_deg", 1),
            ("Max Point Distance from Centreline (mm):", "max_point_distance_spin", 0.0, 10000.0, "max_point_distance_from_centreline", 3),
        ]:
            row, spin = _make_float(label, min_val, max_val, self.settings.get(key), decimals)
            setattr(self, attr, spin)
            layout.addWidget(row)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        layout.addWidget(separator)

        for label, attr, min_val, max_val, key in [
            ("KNN K:", "knn_spin", 1, 30, "knn_k"),
            ("Centreline Samples:", "centreline_samples_spin", 10, 5000, "centreline_samples"),
            ("Tree Sphere Graph K:", "tree_graph_spin", 1, 20, "tree_sphere_graph_k"),
            ("NURBS Surface Count:", "surface_count_spin", 1, 128, "nurbs_surface_count"),
            ("Min Points per Surface:", "surface_min_spin", 2, 100, "nurbs_min_points"),
        ]:
            row, spin = _make_int(label, min_val, max_val, self.settings.get(key))
            setattr(self, attr, spin)
            layout.addWidget(row)

        self.export_csv_check = QCheckBox("Export CSV after each run")
        self.export_csv_check.setChecked(self.settings.get("export_csv", True))
        layout.addWidget(self.export_csv_check)
        self.export_json_check = QCheckBox("Export layer JSON after each run")
        self.export_json_check.setChecked(self.settings.get("export_layers_json", True))
        layout.addWidget(self.export_json_check)
        self.adaptive_spacing_check = QCheckBox("Adaptive spacing for outer corners")
        self.adaptive_spacing_check.setChecked(self.settings.get("adaptive_plane_spacing", True))
        layout.addWidget(self.adaptive_spacing_check)
        self._update_surface_controls()
        layout.addStretch()

    def _update_surface_controls(self):
        enabled = self.enable_nurbs_check.isChecked()
        self.surface_count_spin.setEnabled(enabled)
        self.surface_min_spin.setEnabled(enabled)
        self.angle_tol_spin.setEnabled(enabled)

    def apply_to_config(self, settings: SettingsManager):
        settings.set("sphere_generation_method", self.sphere_generation_combo.currentData() or "auto")
        settings.set("line_method", self.line_method_combo.currentText())
        settings.set("plane_spacing", self.plane_spacing_spin.value())
        settings.set("shell_point_spacing", self.shell_spacing_spin.value())
        settings.set("start_z", self.start_z_spin.value())
        settings.set("start_z_tolerance", self.start_z_tol_spin.value())
        settings.set("adaptive_spacing_min_factor", self.adaptive_min_spin.value())
        settings.set("sphere_min_diameter", self.sphere_min_diameter_spin.value())
        settings.set("sphere_max_diameter", self.sphere_max_diameter_spin.value())
        settings.set("overlap_factor", self.overlap_spin.value())
        settings.set("spline_s", self.spline_spin.value())
        settings.set("centreline_extension_length", self.centreline_extension_spin.value())
        settings.set("outer_corner_spacing_safety", self.corner_safety_spin.value())
        settings.set("max_point_distance_from_centreline", self.max_point_distance_spin.value())
        settings.set("knn_k", self.knn_spin.value())
        settings.set("centreline_samples", self.centreline_samples_spin.value())
        settings.set("tree_sphere_graph_k", self.tree_graph_spin.value())
        settings.set("enable_nurbs", self.enable_nurbs_check.isChecked())
        settings.set("nurbs_surface_count", self.surface_count_spin.value())
        settings.set("nurbs_min_points", self.surface_min_spin.value())
        settings.set("nurbs_angle_tolerance_deg", self.angle_tol_spin.value())
        settings.set("export_csv", self.export_csv_check.isChecked())
        settings.set("export_layers_json", self.export_json_check.isChecked())
        settings.set("adaptive_plane_spacing", self.adaptive_spacing_check.isChecked())

    def reset_widgets(self):
        _set_combo_data(
            self.sphere_generation_combo,
            self.settings.get("sphere_generation_method", "auto"),
        )
        self.line_method_combo.setCurrentText(self.settings.get("line_method", "auto"))
        self.plane_spacing_spin.setValue(self.settings.get("plane_spacing"))
        self.shell_spacing_spin.setValue(self.settings.get("shell_point_spacing"))
        self.start_z_spin.setValue(self.settings.get("start_z"))
        self.start_z_tol_spin.setValue(self.settings.get("start_z_tolerance"))
        self.adaptive_min_spin.setValue(self.settings.get("adaptive_spacing_min_factor", 0.6))
        self.sphere_min_diameter_spin.setValue(self.settings.get("sphere_min_diameter", 0.0))
        self.sphere_max_diameter_spin.setValue(self.settings.get("sphere_max_diameter", 0.0))
        self.overlap_spin.setValue(self.settings.get("overlap_factor"))
        self.spline_spin.setValue(self.settings.get("spline_s"))
        self.centreline_extension_spin.setValue(self.settings.get("centreline_extension_length", 0.0))
        self.corner_safety_spin.setValue(self.settings.get("outer_corner_spacing_safety", 0.35))
        self.max_point_distance_spin.setValue(self.settings.get("max_point_distance_from_centreline", 0.0))
        self.knn_spin.setValue(self.settings.get("knn_k"))
        self.centreline_samples_spin.setValue(self.settings.get("centreline_samples"))
        self.tree_graph_spin.setValue(self.settings.get("tree_sphere_graph_k"))
        self.enable_nurbs_check.setChecked(self.settings.get("enable_nurbs", True))
        self.surface_count_spin.setValue(self.settings.get("nurbs_surface_count"))
        self.surface_min_spin.setValue(self.settings.get("nurbs_min_points"))
        self.angle_tol_spin.setValue(self.settings.get("nurbs_angle_tolerance_deg"))
        self.export_csv_check.setChecked(self.settings.get("export_csv", True))
        self.export_json_check.setChecked(self.settings.get("export_layers_json", True))
        self.adaptive_spacing_check.setChecked(self.settings.get("adaptive_plane_spacing", True))
        self._update_surface_controls()


class ColorSettingsTab(QWidget):
    def __init__(self, settings: SettingsManager):
        super().__init__()
        self.settings = settings
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.bg_picker = ColorSelectWidget("Background:", self.settings.get_as_qcolor("bg_color"))
        self.mesh_picker = ColorSelectWidget("Mesh:", self.settings.get_as_qcolor("mesh_color"))
        self.centerline_picker = ColorSelectWidget("Centerline:", self.settings.get_as_qcolor("centerline_color"))
        self.sphere_picker = ColorSelectWidget("Spheres:", self.settings.get_as_qcolor("sphere_color"))
        self.slice_norm_picker = ColorSelectWidget("Plane Slice:", self.settings.get_as_qcolor("slice_color_normal"))
        self.slice_sel_picker = ColorSelectWidget("Selected Slice:", self.settings.get_as_qcolor("slice_color_selected"))
        self.overlay_font_picker = ColorSelectWidget("Overlay Text:", self.settings.get_as_qcolor("overlay_font_color"))
        for widget in [
            self.bg_picker,
            self.mesh_picker,
            self.centerline_picker,
            self.sphere_picker,
            self.slice_norm_picker,
            self.slice_sel_picker,
            self.overlay_font_picker,
        ]:
            layout.addWidget(widget)
        layout.addStretch()

    def apply_to_config(self, settings: SettingsManager):
        settings.set_from_qcolor("bg_color", self.bg_picker.get_color())
        settings.set_from_qcolor("mesh_color", self.mesh_picker.get_color())
        settings.set_from_qcolor("centerline_color", self.centerline_picker.get_color())
        settings.set_from_qcolor("sphere_color", self.sphere_picker.get_color())
        settings.set_from_qcolor("slice_color_normal", self.slice_norm_picker.get_color())
        settings.set_from_qcolor("slice_color_selected", self.slice_sel_picker.get_color())
        settings.set_from_qcolor("overlay_font_color", self.overlay_font_picker.get_color())

    def reset_widgets(self):
        self.bg_picker.set_color(self.settings.get_as_qcolor("bg_color"))
        self.mesh_picker.set_color(self.settings.get_as_qcolor("mesh_color"))
        self.centerline_picker.set_color(self.settings.get_as_qcolor("centerline_color"))
        self.sphere_picker.set_color(self.settings.get_as_qcolor("sphere_color"))
        self.slice_norm_picker.set_color(self.settings.get_as_qcolor("slice_color_normal"))
        self.slice_sel_picker.set_color(self.settings.get_as_qcolor("slice_color_selected"))
        self.overlay_font_picker.set_color(self.settings.get_as_qcolor("overlay_font_color"))


class AppearanceSettingsTab(QWidget):
    def __init__(self, settings: SettingsManager):
        super().__init__()
        self.settings = settings
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        for label, attr, min_val, max_val, key in [
            ("UI Font Size:", "global_font_spin", 8, 24, "global_font_size"),
            ("Log Font Size:", "log_font_spin", 8, 24, "log_font_size"),
            ("Overlay Font Size:", "overlay_font_spin", 8, 24, "overlay_font_size"),
            ("Centerline Thickness:", "centerline_spin", 1, 20, "centerline_thickness"),
            ("Sphere Thickness:", "sphere_spin", 1, 20, "sphere_thickness"),
            ("Slice Thickness:", "slice_norm_spin", 1, 20, "slice_thickness_normal"),
            ("Selected Slice Thickness:", "slice_sel_spin", 1, 20, "slice_thickness_selected"),
            ("Sphere Centre Point Size:", "sphere_center_point_spin", 1, 100, "sphere_center_point_size"),
            ("Viewer Layer Skip:", "vis_skip_spin", 1, 100, "vis_skip_layers"),
        ]:
            row, spin = _make_int(label, min_val, max_val, self.settings.get(key))
            setattr(self, attr, spin)
            layout.addWidget(row)

        row, self.shell_point_spin = _make_float(
            "Shell Point Size:",
            0.1,
            100.0,
            self.settings.get("shell_point_size", 0.5),
            2,
        )
        layout.addWidget(row)
        row, self.plane_point_spin = _make_float(
            "Plane Point Size:",
            0.1,
            100.0,
            self.settings.get("plane_point_size", 0.5),
            2,
        )
        layout.addWidget(row)

        for label, attr, min_val, max_val, key in [
            ("Mesh Alpha:", "mesh_alpha_spin", 0.0, 1.0, "mesh_alpha"),
            ("Sphere Alpha:", "sphere_alpha_spin", 0.0, 1.0, "sphere_alpha"),
            ("Sphere Centre Alpha:", "sphere_center_alpha_spin", 0.0, 1.0, "sphere_center_alpha"),
            ("Slice Alpha:", "slice_alpha_spin", 0.0, 1.0, "slice_alpha"),
            ("Selected Slice Alpha:", "selected_slice_alpha_spin", 0.0, 1.0, "selected_slice_alpha"),
        ]:
            row, spin = _make_float(label, min_val, max_val, self.settings.get(key), 2)
            setattr(self, attr, spin)
            layout.addWidget(row)

        self.perspective_check = QCheckBox("Use Perspective View")
        self.perspective_check.setChecked(self.settings.get("perspective_view", False))
        layout.addWidget(self.perspective_check)
        layout.addStretch()

    def apply_to_config(self, settings: SettingsManager):
        settings.set("global_font_size", self.global_font_spin.value())
        settings.set("log_font_size", self.log_font_spin.value())
        settings.set("overlay_font_size", self.overlay_font_spin.value())
        settings.set("centerline_thickness", self.centerline_spin.value())
        settings.set("sphere_thickness", self.sphere_spin.value())
        settings.set("slice_thickness_normal", self.slice_norm_spin.value())
        settings.set("slice_thickness_selected", self.slice_sel_spin.value())
        settings.set("sphere_center_point_size", self.sphere_center_point_spin.value())
        settings.set("shell_point_size", self.shell_point_spin.value())
        settings.set("plane_point_size", self.plane_point_spin.value())
        settings.set("vis_skip_layers", self.vis_skip_spin.value())
        settings.set("mesh_alpha", self.mesh_alpha_spin.value())
        settings.set("sphere_alpha", self.sphere_alpha_spin.value())
        settings.set("sphere_center_alpha", self.sphere_center_alpha_spin.value())
        settings.set("slice_alpha", self.slice_alpha_spin.value())
        settings.set("selected_slice_alpha", self.selected_slice_alpha_spin.value())
        settings.set("perspective_view", self.perspective_check.isChecked())

    def reset_widgets(self):
        self.global_font_spin.setValue(self.settings.get("global_font_size"))
        self.log_font_spin.setValue(self.settings.get("log_font_size"))
        self.overlay_font_spin.setValue(self.settings.get("overlay_font_size"))
        self.centerline_spin.setValue(self.settings.get("centerline_thickness"))
        self.sphere_spin.setValue(self.settings.get("sphere_thickness", 1))
        self.slice_norm_spin.setValue(self.settings.get("slice_thickness_normal"))
        self.slice_sel_spin.setValue(self.settings.get("slice_thickness_selected"))
        self.sphere_center_point_spin.setValue(self.settings.get("sphere_center_point_size", 12))
        self.shell_point_spin.setValue(self.settings.get("shell_point_size", 0.5))
        self.plane_point_spin.setValue(self.settings.get("plane_point_size", 0.5))
        self.vis_skip_spin.setValue(self.settings.get("vis_skip_layers"))
        self.mesh_alpha_spin.setValue(self.settings.get("mesh_alpha"))
        self.sphere_alpha_spin.setValue(self.settings.get("sphere_alpha", 1.0))
        self.sphere_center_alpha_spin.setValue(self.settings.get("sphere_center_alpha", 1.0))
        self.slice_alpha_spin.setValue(self.settings.get("slice_alpha"))
        self.selected_slice_alpha_spin.setValue(self.settings.get("selected_slice_alpha", 1.0))
        self.perspective_check.setChecked(self.settings.get("perspective_view", False))


class SettingsDialog(QDialog):
    settings_updated = pyqtSignal()

    def __init__(self, settings_manager: SettingsManager):
        super().__init__()
        self.settings_manager = settings_manager
        self._original_settings_snapshot = dict(self.settings_manager.settings)
        self.setWindowTitle("Application Settings")
        self.setGeometry(200, 200, 420, 560)
        self._setup_ui()
        self.setStyleSheet(
            "QDialog, QWidget {background:#2b2b2b; color:#eee;} "
            "QSpinBox, QDoubleSpinBox, QComboBox {background:#3c3c3c; border:1px solid #555;}"
        )

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        self.algorithm_tab = AlgorithmSettingsTab(self.settings_manager)
        self.color_tab = ColorSettingsTab(self.settings_manager)
        self.appearance_tab = AppearanceSettingsTab(self.settings_manager)
        tabs.addTab(self.algorithm_tab, "Algorithm")
        tabs.addTab(self.color_tab, "Viewer")
        tabs.addTab(self.appearance_tab, "Appearance")
        layout.addWidget(tabs)

        for tab in [self.algorithm_tab, self.color_tab, self.appearance_tab]:
            self._connect_live_signals(tab)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        self.reset_button = buttons.addButton("Reset Defaults", QDialogButtonBox.ResetRole)
        buttons.button(QDialogButtonBox.Ok).clicked.connect(self._on_ok)
        buttons.button(QDialogButtonBox.Cancel).clicked.connect(self._on_cancel)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        self.reset_button.clicked.connect(self._on_reset_defaults)
        layout.addWidget(buttons)

    def _connect_live_signals(self, tab: QWidget):
        for widget in tab.findChildren(QWidget):
            if hasattr(widget, "color_changed"):
                widget.color_changed.connect(self._on_live_update)
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._on_live_update)
            if hasattr(widget, "toggled"):
                widget.toggled.connect(self._on_live_update)
            if hasattr(widget, "currentTextChanged"):
                widget.currentTextChanged.connect(self._on_live_update)

    def _on_live_update(self):
        self.algorithm_tab.apply_to_config(self.settings_manager)
        self.color_tab.apply_to_config(self.settings_manager)
        self.appearance_tab.apply_to_config(self.settings_manager)
        self.settings_updated.emit()

    def _on_apply(self):
        self._on_live_update()
        self.settings_manager.save()
        self._original_settings_snapshot = dict(self.settings_manager.settings)

    def _on_ok(self):
        self._on_apply()
        self.accept()

    def _on_cancel(self):
        self.settings_manager.settings = self._original_settings_snapshot
        self.algorithm_tab.reset_widgets()
        self.color_tab.reset_widgets()
        self.appearance_tab.reset_widgets()
        self.settings_updated.emit()
        self.reject()

    def _on_reset_defaults(self):
        self.settings_manager.reset_to_defaults()
        self.algorithm_tab.reset_widgets()
        self.color_tab.reset_widgets()
        self.appearance_tab.reset_widgets()
        self.settings_updated.emit()
