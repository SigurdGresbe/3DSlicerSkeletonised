# slicer_gui_widgets.py

from dataclasses import dataclass, field
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QColorDialog,
    QDialog, QVBoxLayout, QFrame, QSpinBox, QTabWidget,
    QGridLayout, QFontComboBox, QDialogButtonBox, QDoubleSpinBox,
    QCheckBox
)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor, QFont

from slicer_core.settings_manager import SettingsManager

class ColorSelectWidget(QWidget):
    color_changed = pyqtSignal(QColor)

    def __init__(self, label: str, initial_color: QColor):
        super().__init__()
        self._color = initial_color
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(label)
        self.button = QPushButton("")
        self.button.setFixedWidth(50)
        
        self._update_button_color()
        self.button.clicked.connect(self._show_color_dialog)
        
        layout.addWidget(self.label)
        layout.addStretch()
        layout.addWidget(self.button)

    def _update_button_color(self):
        style = f"background-color: {self._color.name()}; border: 1px solid #555; border-radius: 3px;"
        self.button.setStyleSheet(style)

    def _show_color_dialog(self):
        color = QColorDialog.getColor(self._color, self, "Select Color")
        if color.isValid():
            self._color = color
            self._update_button_color()
            self.color_changed.emit(self._color)

    def get_color(self) -> QColor:
        return self._color

    def set_color(self, color: QColor):
        self._color = color
        self._update_button_color()

def _create_spin_widget(label: str, min_val: int, max_val: int, default_val: int) -> (QWidget, QSpinBox):
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    spin = QSpinBox()
    spin.setRange(min_val, max_val)
    spin.setValue(default_val)
    spin.setFixedWidth(60)
    layout.addWidget(QLabel(label))
    layout.addStretch()
    layout.addWidget(spin)
    return widget, spin

def _create_float_spin_widget(label: str, min_val: float, max_val: float, default_val: float, decimals: int = 2) -> (QWidget, QDoubleSpinBox):
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    spin = QDoubleSpinBox()
    spin.setRange(min_val, max_val)
    spin.setValue(default_val)
    spin.setDecimals(decimals)
    spin.setFixedWidth(60)
    layout.addWidget(QLabel(label))
    layout.addStretch()
    layout.addWidget(spin)
    return widget, spin

class SlicingSettingsTab(QWidget):
    """A widget containing all core slicing parameters."""
    def __init__(self, settings: SettingsManager):
        super().__init__()
        self.settings = settings
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        spin_widget, self.nozzle_spin = _create_float_spin_widget("Nozzle Dia (mm):", 0.1, 5.0, self.settings.get("nozzle_diameter"), 2)
        main_layout.addWidget(spin_widget)
        spin_widget, self.base_lh_spin = _create_float_spin_widget("Base Layer Ht (mm):", 0.01, 5.0, self.settings.get("base_layer_height"), 3)
        main_layout.addWidget(spin_widget)
        spin_widget, self.min_lh_spin = _create_float_spin_widget("Min Layer Ht (mm):", 0.01, 5.0, self.settings.get("min_layer_height_nominal"), 3)
        main_layout.addWidget(spin_widget)
        spin_widget, self.max_lh_spin = _create_float_spin_widget("Max Layer Ht (mm):", 0.01, 5.0, self.settings.get("max_layer_height_nominal"), 3)
        main_layout.addWidget(spin_widget)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("border: 1px solid #444;")
        main_layout.addWidget(separator)
        
        spin_widget, self.curv_spin = _create_float_spin_widget("Curvature Factor:", 0.0, 10.0, self.settings.get("curvature_factor"), 2)
        main_layout.addWidget(spin_widget)
        spin_widget, self.penalty_spin = _create_float_spin_widget("Spline Penalty:", 0.0, 1.0, self.settings.get("spline_point_penalty"), 3)
        main_layout.addWidget(spin_widget)
        spin_widget, self.degree_spin = _create_spin_widget("Spline Degree (k):", 1, 5, self.settings.get("spline_degree"))
        main_layout.addWidget(spin_widget)
        spin_widget, self.smooth_spin = _create_float_spin_widget("Spline Smooth Factor:", 0.0, 100.0, self.settings.get("spline_smooth_factor"), 2)
        main_layout.addWidget(spin_widget)
        
        main_layout.addStretch()

    def apply_to_config(self, settings: SettingsManager):
        """Applies the settings from this tab to the provided settings manager."""
        settings.set("nozzle_diameter", self.nozzle_spin.value())
        settings.set("base_layer_height", self.base_lh_spin.value())
        settings.set("min_layer_height_nominal", self.min_lh_spin.value())
        settings.set("max_layer_height_nominal", self.max_lh_spin.value())
        settings.set("curvature_factor", self.curv_spin.value())
        settings.set("spline_point_penalty", self.penalty_spin.value())
        settings.set("spline_degree", self.degree_spin.value())
        settings.set("spline_smooth_factor", self.smooth_spin.value())
        
    def reset_widgets(self):
        """Resets widgets to the values currently in the settings manager."""
        self.nozzle_spin.setValue(self.settings.get("nozzle_diameter"))
        self.base_lh_spin.setValue(self.settings.get("base_layer_height"))
        self.min_lh_spin.setValue(self.settings.get("min_layer_height_nominal"))
        self.max_lh_spin.setValue(self.settings.get("max_layer_height_nominal"))
        self.curv_spin.setValue(self.settings.get("curvature_factor"))
        self.penalty_spin.setValue(self.settings.get("spline_point_penalty"))
        self.degree_spin.setValue(self.settings.get("spline_degree"))
        self.smooth_spin.setValue(self.settings.get("spline_smooth_factor"))

class ColorSettingsTab(QWidget):
    """A widget containing all color and thickness settings."""
    def __init__(self, settings: SettingsManager):
        super().__init__()
        self.settings = settings
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        self.bg_picker = ColorSelectWidget("Background Color:", self.settings.get_as_qcolor("bg_color"))
        self.mesh_picker = ColorSelectWidget("Mesh Color:", self.settings.get_as_qcolor("mesh_color"))
        self.centerline_picker = ColorSelectWidget("Centerline Color:", self.settings.get_as_qcolor("centerline_color"))
        self.slice_norm_picker = ColorSelectWidget("Slice (Normal) Color:", self.settings.get_as_qcolor("slice_color_normal"))
        self.slice_sel_picker = ColorSelectWidget("Slice (Selected) Color:", self.settings.get_as_qcolor("slice_color_selected"))
        self.guide_picker = ColorSelectWidget("Guide Anchor Color:", self.settings.get_as_qcolor("guide_color"))
        self.overlay_font_picker = ColorSelectWidget("Overlay Font Color:", self.settings.get_as_qcolor("overlay_font_color"))

        main_layout.addWidget(self.bg_picker)
        main_layout.addWidget(self.mesh_picker)
        main_layout.addWidget(self.centerline_picker)
        main_layout.addWidget(self.slice_norm_picker)
        main_layout.addWidget(self.slice_sel_picker)
        main_layout.addWidget(self.guide_picker)
        main_layout.addWidget(self.overlay_font_picker)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("border: 1px solid #444;")
        main_layout.addWidget(separator)
        
        spin_widget, self.centerline_spin = _create_spin_widget("Centerline Thickness:", 1, 20, self.settings.get("centerline_thickness"))
        main_layout.addWidget(spin_widget)
        
        spin_widget, self.slice_norm_spin = _create_spin_widget("Slice Thickness (Normal):", 1, 20, self.settings.get("slice_thickness_normal"))
        main_layout.addWidget(spin_widget)

        spin_widget, self.slice_sel_spin = _create_spin_widget("Slice Thickness (Selected):", 1, 20, self.settings.get("slice_thickness_selected"))
        main_layout.addWidget(spin_widget)
        
        main_layout.addStretch()

    def apply_to_config(self, settings: SettingsManager):
        """Applies the settings from this tab to the provided settings manager."""
        settings.set_from_qcolor("bg_color", self.bg_picker.get_color())
        settings.set_from_qcolor("mesh_color", self.mesh_picker.get_color())
        settings.set_from_qcolor("centerline_color", self.centerline_picker.get_color())
        settings.set_from_qcolor("slice_color_normal", self.slice_norm_picker.get_color())
        settings.set_from_qcolor("slice_color_selected", self.slice_sel_picker.get_color())
        settings.set_from_qcolor("guide_color", self.guide_picker.get_color())
        settings.set_from_qcolor("overlay_font_color", self.overlay_font_picker.get_color())
        
        settings.set("centerline_thickness", self.centerline_spin.value())
        settings.set("slice_thickness_normal", self.slice_norm_spin.value())
        settings.set("slice_thickness_selected", self.slice_sel_spin.value())

    def reset_widgets(self):
        """Resets widgets to the values currently in the settings manager."""
        self.bg_picker.set_color(self.settings.get_as_qcolor("bg_color"))
        self.mesh_picker.set_color(self.settings.get_as_qcolor("mesh_color"))
        self.centerline_picker.set_color(self.settings.get_as_qcolor("centerline_color"))
        self.slice_norm_picker.set_color(self.settings.get_as_qcolor("slice_color_normal"))
        self.slice_sel_picker.set_color(self.settings.get_as_qcolor("slice_color_selected"))
        self.guide_picker.set_color(self.settings.get_as_qcolor("guide_color"))
        self.overlay_font_picker.set_color(self.settings.get_as_qcolor("overlay_font_color"))
        
        self.centerline_spin.setValue(self.settings.get("centerline_thickness"))
        self.slice_norm_spin.setValue(self.settings.get("slice_thickness_normal"))
        self.slice_sel_spin.setValue(self.settings.get("slice_thickness_selected"))

class AppearanceSettingsTab(QWidget):
    """A widget containing all font and UI appearance settings."""
    def __init__(self, settings: SettingsManager):
        super().__init__()
        self.settings = settings
        self._setup_ui()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        font_group = QFrame()
        font_layout = QVBoxLayout(font_group)
        font_layout.setContentsMargins(0, 0, 0, 0)
        spin_widget, self.global_font_spin = _create_spin_widget("General UI Font Size:", 8, 20, self.settings.get("global_font_size"))
        font_layout.addWidget(spin_widget)
        spin_widget, self.log_font_spin = _create_spin_widget("Log Window Font Size:", 8, 20, self.settings.get("log_font_size"))
        font_layout.addWidget(spin_widget)
        spin_widget, self.overlay_font_spin = _create_spin_widget("Viewer Overlay Font Size:", 8, 20, self.settings.get("overlay_font_size"))
        font_layout.addWidget(spin_widget)
        main_layout.addWidget(font_group)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("border: 1px solid #444;")
        main_layout.addWidget(separator)

        viewer_group = QFrame()
        viewer_layout = QVBoxLayout(viewer_group)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        
        self.perspective_check = QCheckBox("Use Perspective View")
        self.perspective_check.setChecked(self.settings.get("perspective_view", True))
        viewer_layout.addWidget(self.perspective_check)

        spin_widget, self.mesh_alpha_spin = _create_float_spin_widget("Mesh Opacity (Alpha):", 0.0, 1.0, self.settings.get("mesh_alpha", 0.25))
        viewer_layout.addWidget(spin_widget)

        spin_widget, self.vis_skip_spin = _create_spin_widget("Viewer Layer Skip:", 1, 100, self.settings.get("vis_skip_layers", 10))
        viewer_layout.addWidget(spin_widget)
        
        spin_widget, self.slice_alpha_spin = _create_float_spin_widget("Slice Opacity (Alpha):", 0.0, 1.0, self.settings.get("slice_alpha", 0.8))
        viewer_layout.addWidget(spin_widget)

        main_layout.addWidget(viewer_group)
        main_layout.addStretch()

    def apply_to_config(self, settings: SettingsManager):
        """Applies the settings from this tab to the provided settings manager."""
        settings.set("global_font_size", self.global_font_spin.value())
        settings.set("log_font_size", self.log_font_spin.value())
        settings.set("overlay_font_size", self.overlay_font_spin.value())
        
        settings.set("perspective_view", self.perspective_check.isChecked())
        settings.set("mesh_alpha", self.mesh_alpha_spin.value())
        settings.set("vis_skip_layers", self.vis_skip_spin.value())
        settings.set("slice_alpha", self.slice_alpha_spin.value())

    def reset_widgets(self):
        """Resets widgets to the values currently in the settings manager."""
        self.global_font_spin.setValue(self.settings.get("global_font_size"))
        self.log_font_spin.setValue(self.settings.get("log_font_size"))
        self.overlay_font_spin.setValue(self.settings.get("overlay_font_size"))
        
        self.perspective_check.setChecked(self.settings.get("perspective_view"))
        self.mesh_alpha_spin.setValue(self.settings.get("mesh_alpha"))
        self.vis_skip_spin.setValue(self.settings.get("vis_skip_layers"))
        self.slice_alpha_spin.setValue(self.settings.get("slice_alpha"))


class SettingsDialog(QDialog):
    settings_updated = pyqtSignal()

    def __init__(self, settings_manager: SettingsManager):
        super().__init__()
        self.settings_manager = settings_manager
        
        self._original_settings_snapshot = dict(self.settings_manager.settings)

        self.setWindowTitle("Application Settings")
        self.setGeometry(200, 200, 400, 520)
        
        self._setup_ui()
        self._apply_dark_mode()

    def _apply_dark_mode(self):
        dark = """
        QDialog, QWidget {background:#2b2b2b; color:#eee;}
        QTabWidget::pane { border: 1px solid #444; }
        QTabBar::tab { 
            background: #3c3c3c; 
            border: 1px solid #444; 
            border-bottom: none; 
            padding: 8px 20px;
        }
        QTabBar::tab:selected { background: #4a4a4a; }
        QTabBar::tab:!selected { background: #2b2b2b; }
        QLabel {color:#ddd;}
        QSpinBox, QDoubleSpinBox {background:#3c3c3c; border:1px solid #555; border-radius:3px; padding:2px;}
        QPushButton {background:#4a4a4a; border:1px solid #666; border-radius:4px; padding:6px;}
        QPushButton:hover {background:#5a5a5a;}
        QPushButton:pressed {background:#3a3a3a;}
        """
        self.setStyleSheet(dark)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)

        self.slicing_tab = SlicingSettingsTab(self.settings_manager)
        tab_widget.addTab(self.slicing_tab, "Slicing")
        
        self.color_tab = ColorSettingsTab(self.settings_manager)
        tab_widget.addTab(self.color_tab, "Viewer Colors")
        
        self.appearance_tab = AppearanceSettingsTab(self.settings_manager)
        tab_widget.addTab(self.appearance_tab, "Appearance")

        self._connect_live_signals(self.slicing_tab)
        self._connect_live_signals(self.color_tab)
        self._connect_live_signals(self.appearance_tab)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        self.reset_button = button_box.addButton("Reset Defaults", QDialogButtonBox.ResetRole)
        
        button_box.button(QDialogButtonBox.Ok).clicked.connect(self._on_ok)
        button_box.button(QDialogButtonBox.Cancel).clicked.connect(self._on_cancel)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        self.reset_button.clicked.connect(self._on_reset_defaults)
        
        main_layout.addWidget(button_box)
        
    def _connect_live_signals(self, tab: QWidget):
        """Helper to connect all widgets in a tab to the update signal."""
        for widget in tab.findChildren(QWidget):
            if hasattr(widget, 'color_changed'):
                widget.color_changed.connect(self._on_live_update)
            if hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self._on_live_update)
            if hasattr(widget, 'toggled'):
                widget.toggled.connect(self._on_live_update)
                
    def _on_live_update(self):
        """Applies settings to the manager *without* saving, then emits."""
        self.slicing_tab.apply_to_config(self.settings_manager)
        self.color_tab.apply_to_config(self.settings_manager)
        self.appearance_tab.apply_to_config(self.settings_manager)
        self.settings_updated.emit()

    def _on_apply(self):
        """Apply and save the pending changes without closing."""
        self._on_live_update()
        self.settings_manager.save()
        self._original_settings_snapshot = dict(self.settings_manager.settings)

    def _on_ok(self):
        """Apply, save, and close the dialog."""
        self._on_apply()
        self.accept()

    def _on_cancel(self):
        """Revert any un-saved changes and close the dialog."""
        self.settings_manager.settings = self._original_settings_snapshot
        
        self.slicing_tab.reset_widgets()
        self.color_tab.reset_widgets()
        self.appearance_tab.reset_widgets()
        
        self.settings_updated.emit()
        self.reject()
        
    def _on_reset_defaults(self):
        """Resets all settings to factory defaults."""
        self.settings_manager.reset_to_defaults()
        
        self.slicing_tab.reset_widgets()
        self.color_tab.reset_widgets()
        self.appearance_tab.reset_widgets()
        
        self.settings_updated.emit()