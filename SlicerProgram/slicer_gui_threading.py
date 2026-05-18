# slicer_gui_threading.py

import logging
from PyQt5.QtCore import QThread, pyqtSignal, QObject

from slicer_core.config import SlicerConfig
from slicer_core.processor import NonPlanarSlicer
from slicer_core.utils import setup_logging
from slicer_core.settings_manager import SettingsManager

class LogEmitter(QObject):
    log_signal = pyqtSignal(str)

class QtLogHandler(logging.Handler):
    def __init__(self, emitter: LogEmitter):
        super().__init__()
        self.emitter = emitter
        self.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)-8s - %(name)-25s - %(message)s'
            )
        )

    def emit(self, record):
        self.emitter.log_signal.emit(self.format(record))

class SlicerWorker(QThread):
    slicer_finished = pyqtSignal(object)
    slicer_failed = pyqtSignal(str)
    progress_updated = pyqtSignal(int)

    def __init__(self, settings: SettingsManager, run_config: SlicerConfig):
        super().__init__()
        self.settings = settings
        self.run_config = run_config

    def run(self):
        try:
            slicer = NonPlanarSlicer(self.settings, self.run_config)
            
            log_file = self.run_config.log_dir / f"slicer_run_{slicer.run_timestamp}.log"
            setup_logging(log_file)
            
            slicer.run(progress_callback=self.progress_updated.emit)

            self.slicer_finished.emit(slicer)
        except Exception as e:
            logging.critical("--- SLICER RUN FAILED! ---")
            logging.critical(f"Error: {e}", exc_info=True)
            self.slicer_failed.emit(str(e))

class RefinementWorker(QThread):
    slicer_finished = pyqtSignal(object)
    slicer_failed = pyqtSignal(str)
    progress_updated = pyqtSignal(int)

    def __init__(self, slicer: NonPlanarSlicer):
        super().__init__()
        self.slicer = slicer

    def run(self):
        try:

            self.slicer._validate_settings()
            self.slicer.run_refinement_pass(progress_callback=self.progress_updated.emit) 
            self.slicer_finished.emit(self.slicer)
        except Exception as e:
            logging.critical("--- REFINEMENT PASS FAILED! ---")
            logging.critical(f"Error: {e}", exc_info=True)
            self.slicer_failed.emit(str(e))