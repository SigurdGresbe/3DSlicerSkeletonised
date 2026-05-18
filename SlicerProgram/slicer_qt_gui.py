# slicer_qt_gui.py

import sys
import logging
from pathlib import Path
from typing import Optional

import pyvista as pv
import trimesh 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QPushButton, QLineEdit, QFileDialog,
    QTextEdit, QLabel, QDoubleSpinBox, QCheckBox, QSplitter,
    QSpinBox, QSlider, QProgressBar, QMenuBar, QAction,
    QRadioButton, QButtonGroup,
    QMessageBox
)
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QFont
from pyvistaqt import BackgroundPlotter

from slicer_core.config import SlicerConfig
from slicer_core.processor import NonPlanarSlicer
from slicer_core.settings_manager import SettingsManager 

from slicer_gui_widgets import SettingsDialog 

from slicer_gui_threading import QtLogHandler, SlicerWorker, LogEmitter, RefinementWorker
from slicer_gui_visualization import PyVistaManager, ViewOptions

class SlicerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Non-Planar Slicer – PyQt / PyVista")
        self.setGeometry(100, 100, 1600, 900)

        self.slicer_object: Optional[NonPlanarSlicer] = None
        self.settings_manager = SettingsManager()
        self.settings_dialog: Optional[SettingsDialog] = None 
        self.vis_manager: Optional[PyVistaManager] = None

        self.worker_thread: Optional[QThread] = None
        self.log_emitter = LogEmitter()

        self._setup_ui()
        self._connect_signals()
        self._load_config_into_ui()
        self._apply_appearance_settings()

    def _apply_appearance_settings(self):
        """Applies all dynamic style and font settings from the manager."""
        
        config = self.settings_manager
        font_size = config.get("global_font_size", 10)
        dark_style = f"""
        QMainWindow, QWidget {{
            background:#2b2b2b; 
            color:#eee;
            font-size: {font_size}pt;
        }}
        QMenuBar {{background:#2b2b2b; color:#eee;}}
        QMenuBar::item {{background:transparent; color:#eee;}}
        QMenuBar::item:selected {{background:#4a4a4a;}}
        QMenu {{background:#3c3c3c; color:#eee; border:1px solid #555;}}
        QMenu::item:selected {{background:#5a5a5a;}}
        QGroupBox {{
            font-weight:bold; 
            border:1px solid #555; 
            border-radius:5px; 
            margin:5px; 
            padding-top:10px;
        }}
        QGroupBox::title {{subcontrol-origin:margin; left:10px; padding:0 5px;}}
        QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox {{
            background:#3c3c3c; 
            border:1px solid #555; 
            border-radius:3px; 
            padding:2px;
        }}
        QPushButton {{background:#4a4a4a; border:1px solid #666; border-radius:4px; padding:6px;}}
        QPushButton:hover {{background:#5a5a5a;}}
        QPushButton:pressed {{background:#3a3a3a;}}
        QPushButton:disabled {{background:#3c3c3c; color:#777; border-color:#555;}}
        QCheckBox, QLabel, QRadioButton {{color:#ddd;}}
        QSlider::groove:horizontal {{background:#444; height:8px; border-radius:4px;}}
        QSlider::handle:horizontal {{background:#888; width:16px; border-radius:8px; margin:-4px 0;}}
        QSlider::groove:vertical {{background:#444; width:8px; border-radius:4px;}}
        QSlider::handle:vertical {{background:#888; height:16px; border-radius:8px; margin:0 -4px;}}
        QProgressBar {{border:1px solid #555; border-radius:5px; text-align:center; background:#3c3c3c;}}
        QProgressBar::chunk {{background:#00aa00;}}
        """
        self.setStyleSheet(dark_style)
        
        log_font = QFont()
        log_font.setFamily("Consolas, 'Courier New', monospace")
        log_font.setPointSize(config.get("log_font_size", 10))
        self.log_text.setFont(log_font)
        
        self.splitter.setSizes(self.settings_manager.get("splitter_sizes", [500, 1100]))
        self.vis_manager.update_appearance(self.settings_manager)

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
        hl = QHBoxLayout()
        hl.addWidget(QLabel("STL File:"))
        hl.addWidget(self.file_path_edit)
        hl.addWidget(browse)
        left_layout.addLayout(hl)

        view_g = QGroupBox("Viewer Options")
        vg = QGridLayout(view_g)
        
        self.show_mesh_check = QCheckBox("Show Mesh")
        self.show_mesh_check.setChecked(self.settings_manager.get("show_mesh", True))
        vg.addWidget(self.show_mesh_check, 0, 0, 1, 2)

        self.show_slices_check = QCheckBox("Show Slices")
        self.show_slices_check.setChecked(self.settings_manager.get("show_slices", True))
        vg.addWidget(self.show_slices_check, 1, 0, 1, 2)

        self.show_centerline_check = QCheckBox("Show Centerline")
        self.show_centerline_check.setChecked(self.settings_manager.get("show_centerline", True))
        vg.addWidget(self.show_centerline_check, 2, 0, 1, 2)

        self.show_guide_points_check = QCheckBox("Show Guide Points")
        self.show_guide_points_check.setChecked(self.settings_manager.get("show_guide_points", False))
        vg.addWidget(self.show_guide_points_check, 3, 0, 1, 2)
   
        self.show_normal_check = QCheckBox("Show Current Layer Normal")
        self.show_normal_check.setChecked(self.settings_manager.get("show_layer_normal", True))
        vg.addWidget(self.show_normal_check, 4, 0, 1, 2)
        
        self.show_clip_check = QCheckBox("Show Live Cross-Section")
        self.show_clip_check.setChecked(self.settings_manager.get("show_clip_section", False))
        vg.addWidget(self.show_clip_check, 5, 0, 1, 2)
        
        view_layout = QHBoxLayout()
        self.perspective_check = QCheckBox("Perspective View")
        self.perspective_check.setChecked(self.settings_manager.get("perspective_view", True))
        view_layout.addWidget(self.perspective_check)
        view_layout.addStretch()
        
        self.reset_view_btn = QPushButton("Reset View")
        self.view_x_btn = QPushButton("View X")
        self.view_y_btn = QPushButton("View Y")
        self.view_z_btn = QPushButton("View Z")
        view_layout.addWidget(self.reset_view_btn)
        view_layout.addWidget(self.view_x_btn)
        view_layout.addWidget(self.view_y_btn)
        view_layout.addWidget(self.view_z_btn)
        vg.addLayout(view_layout, 6, 0, 1, 2)

        left_layout.addWidget(view_g)

        self.pass_group = QGroupBox("Pass Selection")
        self.pass_layout = QHBoxLayout(self.pass_group)
        self.pass_button_group = QButtonGroup(self)
        self.pass_button_group.buttonClicked.connect(self._refresh_viewer)
        self.pass_layout.addWidget(QLabel("No passes generated."))
        left_layout.addWidget(self.pass_group)
        
        run_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Slicer (2-Pass)")
        self.run_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.run_button.clicked.connect(self._start_slicing)
        
        self.refine_button = QPushButton("Run Refinement Pass")
        self.refine_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.refine_button.clicked.connect(self._start_refinement)
        self.refine_button.setEnabled(False) 
        
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.refine_button)
        left_layout.addLayout(run_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        log_g = QGroupBox("Slicer Output Log")
        lg = QVBoxLayout(log_g)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        lg.addWidget(self.log_text)
        left_layout.addWidget(log_g)
        left_layout.setStretchFactor(log_g, 1)

        self.splitter.addWidget(left)

        self.plotter = BackgroundPlotter(show=False)
        self.vis_manager = PyVistaManager(self.plotter)

        self.layer_slider = QSlider(Qt.Vertical)
        self.layer_slider.setRange(0, 0)
        self.layer_slider.setTickPosition(QSlider.TicksLeft)
        
        self.layer_spinbox = QSpinBox()
        self.layer_spinbox.setRange(0, 0)
        self.layer_spinbox.setButtonSymbols(QSpinBox.PlusMinus)
        
        slider_layout = QVBoxLayout()
        slider_layout.setContentsMargins(5, 5, 0, 5) 
        slider_layout.addWidget(self.layer_slider)
        slider_layout.addWidget(self.layer_spinbox)

        viewer_widget = QWidget()
        viewer_layout = QHBoxLayout(viewer_widget)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.addLayout(slider_layout) 
        viewer_layout.addWidget(self.plotter)  
        viewer_layout.setStretch(1, 1) 

        self.splitter.addWidget(viewer_widget) 

    def _setup_menu_bar(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)
        
        settings_menu = menu_bar.addMenu("Settings")
        settings_action = QAction("Application Settings...", self)
        settings_action.triggered.connect(self._open_settings_dialog)
        settings_menu.addAction(settings_action)

    def _toggle_projection(self):
        self.vis_manager.set_camera_parallel(not self.perspective_check.isChecked())

    def _reset_view(self):
        self.vis_manager.reset_view()

    def _view_x(self):
        self.vis_manager.view_x()

    def _view_y(self):
        self.vis_manager.view_y()

    def _view_z(self):
        self.vis_manager.view_z()
        
    def _open_settings_dialog(self):
        if self.settings_dialog is None:
            self.settings_dialog = SettingsDialog(self.settings_manager)
            self.settings_dialog.settings_updated.connect(self._on_settings_updated)
            self.settings_dialog.finished.connect(self._settings_dialog_finished)
        self.settings_dialog.show()

    def _on_settings_updated(self):
        """Called when settings are changed (Apply, OK, Cancel)."""
        logging.info("Applying new appearance configuration.")
        
        self._apply_appearance_settings()
        self._load_config_into_ui()
        
        if not self.slicer_object:
            try:
                mesh = trimesh.load(self.settings_manager.get("stl_file"))
                self.vis_manager.preview_mesh(
                    mesh, 
                    self.settings_manager.get("mesh_alpha", 0.25), 
                    self.settings_manager.get_as_qcolor("mesh_color")
                )
            except Exception as e:
                logging.warning(f"Could not refresh mesh preview on settings change: {e}")
        else:
            self._refresh_viewer()

    def _settings_dialog_finished(self):
        self.settings_dialog = None

    def _add_spin(self, layout: QGridLayout, txt: str, row: int, col: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.01, 5.0)
        spin.setDecimals(3)
        layout.addWidget(QLabel(txt), row, col)
        layout.addWidget(spin, row, col + 1)
        return spin

    def _connect_signals(self):
        slicer_logger = logging.getLogger('slicer_core')
        slicer_logger.setLevel(logging.DEBUG)
        self.log_emitter.log_signal.connect(self.log_text.append)
        gui_handler = QtLogHandler(self.log_emitter)
        gui_handler.setLevel(logging.DEBUG) 
        slicer_logger.addHandler(gui_handler)
  
        self.show_mesh_check.stateChanged.connect(self._on_view_setting_changed)
        self.show_slices_check.stateChanged.connect(self._on_view_setting_changed)
        self.show_centerline_check.stateChanged.connect(self._on_view_setting_changed)
        self.show_guide_points_check.stateChanged.connect(self._on_view_setting_changed)
        self.show_normal_check.stateChanged.connect(self._on_view_setting_changed)
        self.show_clip_check.stateChanged.connect(self._on_view_setting_changed)
        self.perspective_check.toggled.connect(self._on_view_setting_changed)
        
        self.layer_slider.valueChanged.connect(self._on_layer_slider_changed)
        self.layer_spinbox.valueChanged.connect(self._on_layer_spinbox_changed)
  
        self.reset_view_btn.clicked.connect(self._reset_view)
        self.view_x_btn.clicked.connect(self._view_x)
        self.view_y_btn.clicked.connect(self._view_y)
        self.view_z_btn.clicked.connect(self._view_z)
 
        self.splitter.splitterMoved.connect(lambda: self.settings_manager.set("splitter_sizes", self.splitter.sizes()))

    def _on_layer_slider_changed(self, value):
        self.layer_spinbox.blockSignals(True)
        self.layer_spinbox.setValue(value)
        self.layer_spinbox.blockSignals(False)
        self._refresh_viewer()
        
    def _on_layer_spinbox_changed(self, value):
        self.layer_slider.blockSignals(True)
        self.layer_slider.setValue(value)
        self.layer_slider.blockSignals(False)
        self._refresh_viewer()
        
    def _on_view_setting_changed(self):
        """A single handler for all viewer checkboxes."""
        self.settings_manager.set("show_mesh", self.show_mesh_check.isChecked())
        self.settings_manager.set("show_slices", self.show_slices_check.isChecked())
        self.settings_manager.set("show_centerline", self.show_centerline_check.isChecked())
        self.settings_manager.set("show_guide_points", self.show_guide_points_check.isChecked())
        self.settings_manager.set("show_layer_normal", self.show_normal_check.isChecked())
        self.settings_manager.set("show_clip_section", self.show_clip_check.isChecked())
        self.settings_manager.set("perspective_view", self.perspective_check.isChecked())
        
        self.vis_manager.set_camera_parallel(not self.perspective_check.isChecked())
        self._refresh_viewer()

    def _load_config_into_ui(self):
        """Loads settings from the manager into the main UI widgets."""
        self.file_path_edit.setText(self.settings_manager.get("stl_file", "mesh/default.stl"))

        self.show_mesh_check.setChecked(self.settings_manager.get("show_mesh", True))
        self.show_slices_check.setChecked(self.settings_manager.get("show_slices", True))
        self.show_centerline_check.setChecked(self.settings_manager.get("show_centerline", True))
        self.show_guide_points_check.setChecked(self.settings_manager.get("show_guide_points", False))
        self.show_normal_check.setChecked(self.settings_manager.get("show_layer_normal", True))
        self.show_clip_check.setChecked(self.settings_manager.get("show_clip_section", False))
        self.perspective_check.setChecked(self.settings_manager.get("perspective_view", True))

    def _browse_stl(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select STL File", str(Path.cwd()), "STL Files (*.stl)")
        if fn:
            self.file_path_edit.setText(fn)
            self.settings_manager.set("stl_file", fn)
            
            try:
                mesh = trimesh.load(fn)
                self.slicer_object = None
                self._update_pass_selection_ui() 
                self.layer_slider.setRange(0, 0)
                self.layer_spinbox.setRange(0, 0)
                
                self.vis_manager.preview_mesh(
                    mesh, 
                    self.settings_manager.get("mesh_alpha", 0.25), 
                    self.settings_manager.get_as_qcolor("mesh_color")
                )
            except Exception as e:
                logging.error(f"Failed to load and preview mesh {fn}: {e}")

    def _set_buttons_enabled(self, enabled: bool):
        self.run_button.setEnabled(enabled)
        self.refine_button.setEnabled(enabled and self.slicer_object is not None)

    def _start_slicing(self):
        self.log_text.clear()
        self._set_buttons_enabled(False)
        self.run_button.setText("Slicing…")
        self.progress_bar.setValue(0)
        
        self.slicer_object = None
        self._update_pass_selection_ui()
        
        self.layer_slider.setRange(0, 0)
        self.layer_spinbox.setRange(0, 0)

        self.settings_manager.set("stl_file", self.file_path_edit.text())

        try:
            run_config = SlicerConfig(
                stl_file=Path(self.settings_manager.get("stl_file"))
            )
        except Exception as e:
            logging.error(f"Invalid file path: {e}")
            self._set_buttons_enabled(True)
            self.run_button.setText("Run Slicer (2-Pass)")
            return

        self.worker_thread = SlicerWorker(self.settings_manager, run_config) 
        self.worker_thread.slicer_finished.connect(self._finish_slicing_success)
        self.worker_thread.slicer_failed.connect(self._finish_slicing_failure)
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.start()

    def _start_refinement(self):
        if not self.slicer_object:
            logging.error("Cannot refine: Slicer object is missing.")
            return

        self.log_text.append("\n--- Starting new refinement pass ---")
        self._set_buttons_enabled(False)
        self.refine_button.setText("Refining…")
        self.progress_bar.setValue(0)

        self.slicer_object.settings = self.settings_manager
        
        self.worker_thread = RefinementWorker(self.slicer_object)
        self.worker_thread.slicer_finished.connect(self._finish_slicing_success)
        self.worker_thread.slicer_failed.connect(self._finish_slicing_failure)
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.start()

    def _update_pass_selection_ui(self):
        self.pass_button_group.blockSignals(True)
        
        for button in self.pass_button_group.buttons():
            self.pass_button_group.removeButton(button)
            button.deleteLater()
            
        while self.pass_layout.count():
            child = self.pass_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not self.slicer_object or not self.slicer_object.pass_results:
            self.pass_layout.addWidget(QLabel("No passes generated."))
            self.pass_button_group.blockSignals(False) 
            return

        for i, pass_res in enumerate(self.slicer_object.pass_results):
            pass_index = pass_res.pass_index
            radio_btn = QRadioButton(f"Pass {pass_index}")
            self.pass_layout.addWidget(radio_btn)
            self.pass_button_group.addButton(radio_btn, pass_index)
            
            if i == len(self.slicer_object.pass_results) - 1:
                radio_btn.setChecked(True)
        
        self.pass_layout.addStretch()
        self.pass_button_group.blockSignals(False) 

    def _finish_slicing_success(self, slicer: NonPlanarSlicer):
        self.slicer_object = slicer
        slicer.generate_plot()
        
        try:
            latest_pass = slicer.pass_results[-1]
            total_layers = len(latest_pass.slices)
            new_max = total_layers - 1 if total_layers > 0 else 0
            
            self.layer_slider.setRange(0, new_max)
            self.layer_spinbox.setRange(0, new_max)
            self.layer_slider.setValue(new_max)
            self.layer_spinbox.setValue(new_max)
            
        except Exception as e:
            logging.error(f"Failed to update layer controls: {e}")
            self.layer_slider.setRange(0, 0)
            self.layer_spinbox.setRange(0, 0)
        
        self._update_pass_selection_ui()

        if self.settings_manager.get("show_3d_viewer", True):
            self.vis_manager.set_slicer_object(slicer)
            self._refresh_viewer() 

        logging.info("\n--- SLICER PASS FINISHED SUCCESSFULLY! ---")
        self._cleanup_thread()

    def _finish_slicing_failure(self, msg: str):
        logging.error(f"Slicing failed with message: {msg}")
        QMessageBox.critical(self, "Slicer Error", f"Slicing failed:\n\n{msg}\n\nCheck the log for details.")
        self._cleanup_thread()

    def _cleanup_thread(self):
        self._set_buttons_enabled(True)
        self.run_button.setText("Run Slicer (2-Pass)")
        self.refine_button.setText("Run Refinement Pass")
        self.worker_thread = None

    def _refresh_viewer(self):
        if not self.vis_manager or not self.slicer_object:
            return
        
        pass_to_show = self.pass_button_group.checkedId()
        if pass_to_show == -1: 
            return 
            
        try:
            total = len(self.vis_manager.slicer.pass_results[pass_to_show - 1].slices)
        except (IndexError, TypeError):
            total = 0

        vis_skip_layers = self.settings_manager.get("vis_skip_layers", 10)
        slice_alpha = self.settings_manager.get("slice_alpha", 0.8)
        current_layer_index = self.layer_slider.value()

        options = ViewOptions(
            pass_to_show=pass_to_show, 
            show_mesh=self.show_mesh_check.isChecked(),
            show_slices=self.show_slices_check.isChecked(),
            show_centerline=self.show_centerline_check.isChecked(),
            show_guide_points=self.show_guide_points_check.isChecked(),
            show_layer_normal=self.show_normal_check.isChecked(),
            show_cross_section=self.show_clip_check.isChecked(),
            vis_skip_layers=vis_skip_layers,
            slice_alpha=slice_alpha,
            current_layer_index=current_layer_index
        )
        
        self.vis_manager.refresh_view(options)

    def closeEvent(self, event):
        logging.info("Saving UI state to settings...")
        self.settings_manager.set("splitter_sizes", self.splitter.sizes())
        self.settings_manager.set("show_mesh", self.show_mesh_check.isChecked())
        self.settings_manager.set("show_slices", self.show_slices_check.isChecked())
        self.settings_manager.set("show_centerline", self.show_centerline_check.isChecked())
        self.settings_manager.set("show_guide_points", self.show_guide_points_check.isChecked())
        self.settings_manager.set("show_layer_normal", self.show_normal_check.isChecked())
        self.settings_manager.set("show_clip_section", self.show_clip_check.isChecked())
        self.settings_manager.set("perspective_view", self.perspective_check.isChecked())

        self.settings_manager.save()
        
        self.plotter.close()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication.instance() or QApplication(sys.argv)
    pv.set_plot_theme("document")
    win = SlicerGUI()
    win.show()
    sys.exit(app.exec_())