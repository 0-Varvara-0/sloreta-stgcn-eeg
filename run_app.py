from __future__ import annotations

import os
import sys

# Переменная QT_API должна задаваться строго до любых импортов PySide6 / PyVistaQt / PyQtGraph
os.environ["QT_API"] = "pyside6"

from PySide6.QtWidgets import QApplication

from src.viz.main_window import BrainViewerApp
from src.viz.styles import DARK_STYLE, configure_pyqtgraph


def main() -> int:
    configure_pyqtgraph()
    app = QApplication(sys.argv)
    app.setApplicationName("NeuroViz Desktop")
    app.setStyleSheet(DARK_STYLE)

    window = BrainViewerApp()
    app.aboutToQuit.connect(window.brain_view.cleanup)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())