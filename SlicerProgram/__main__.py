import sys
from pathlib import Path


def main() -> int:
    package_root = Path(__file__).resolve().parent

    # Preserve the existing local-import style used by the GUI modules.
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    from shell_qt_gui import QApplication, SlicerGUI, pv

    app = QApplication.instance() or QApplication(sys.argv)
    pv.set_plot_theme("document")
    win = SlicerGUI()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
