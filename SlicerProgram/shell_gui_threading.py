import logging

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from slicer_core.config import SlicerConfig
from slicer_core.settings_manager import SettingsManager
from slicer_core.shell_processor import CentrelineShellSlicer, SlicerRunCancelled
from slicer_core.utils import setup_logging


class LogEmitter(QObject):
    log_signal = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, emitter: LogEmitter):
        super().__init__()
        self.emitter = emitter
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)-8s - %(name)-25s - %(message)s"
            )
        )

    def emit(self, record):
        self.emitter.log_signal.emit(self.format(record))


class SlicerWorker(QThread):
    slicer_finished = pyqtSignal(object)
    slicer_cancelled = pyqtSignal(str)
    slicer_failed = pyqtSignal(str)
    progress_updated = pyqtSignal(int)

    def __init__(self, settings: SettingsManager, run_config: SlicerConfig):
        super().__init__()
        self.settings = settings
        self.run_config = run_config

    def run(self):
        try:
            slicer = CentrelineShellSlicer(self.settings, self.run_config)
            log_file = self.run_config.log_dir / f"slicer_run_{slicer.run_timestamp}.log"
            setup_logging(log_file)
            slicer.run(
                progress_callback=self.progress_updated.emit,
                cancel_callback=self._raise_if_cancelled,
            )
            self.slicer_finished.emit(slicer)
        except SlicerRunCancelled as exc:
            logging.getLogger("slicer_core").info("Slicer run cancelled: %s", exc)
            self.slicer_cancelled.emit(str(exc))
        except Exception as exc:
            logging.getLogger("slicer_core").exception("Slicer run failed.")
            self.slicer_failed.emit(str(exc))

    def cancel(self):
        self.requestInterruption()

    def _raise_if_cancelled(self):
        if self.isInterruptionRequested():
            raise SlicerRunCancelled("Generation interrupted by user.")
