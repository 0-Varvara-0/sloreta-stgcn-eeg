from __future__ import annotations

PLOT_BG_COLOR = "#111722"
PLOT_FG_COLOR = "#cbd5e1"
PLOT_GRID_ALPHA = 0.18
SPIKE_REGION_COLOR = (239, 68, 68, 55)
CURSOR_COLOR = "#38bdf8"

TRACE_COLORS: list[str] = [
    "#38bdf8",
    "#34d399",
    "#fbbf24",
    "#f472b6",
    "#a78bfa",
    "#fb923c",
    "#2dd4bf",
    "#f87171",
    "#60a5fa",
    "#c084fc",
]

DARK_STYLE: str = """
QMainWindow, QWidget {
    background-color: #0f141c;
    color: #e2e8f0;
    font-family: "Segoe UI", "Inter", "Ubuntu", sans-serif;
    font-size: 12px;
}

QGroupBox {
    background-color: #151c27;
    border: 1px solid #253042;
    border-radius: 6px;
    margin-top: 10px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
    color: #93c5fd;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
}

QPushButton {
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 5px;
    padding: 6px 10px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #283548;
    border-color: #38bdf8;
}

QPushButton:pressed {
    background-color: #172030;
}

QPushButton:disabled {
    background-color: #151b26;
    color: #64748b;
    border-color: #1e293b;
}

QPushButton[accent="true"] {
    background-color: #0284c7;
    border-color: #38bdf8;
    color: #ffffff;
    font-weight: 600;
}

QPushButton[accent="true"]:hover {
    background-color: #0369a1;
}

QPushButton[accent="true"]:disabled {
    background-color: #1e3a5f;
    color: #94a3b8;
    border-color: #1e3a5f;
}

QComboBox, QDoubleSpinBox, QSpinBox {
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 4px 6px;
}

QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {
    border-color: #38bdf8;
}

QComboBox QAbstractItemView {
    background-color: #161e2b;
    color: #e2e8f0;
    selection-background-color: #0284c7;
    border: 1px solid #334155;
}

QCheckBox {
    spacing: 6px;
    color: #cbd5e1;
}

QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 3px;
    border: 1px solid #475569;
    background-color: #1e293b;
}

QCheckBox::indicator:checked {
    background-color: #0284c7;
    border-color: #38bdf8;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #0284c7;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #38bdf8;
    border: 1px solid #0c4a6e;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}

QListWidget, QTextEdit, QPlainTextEdit {
    background-color: #111722;
    color: #cbd5e1;
    border: 1px solid #253042;
    border-radius: 4px;
    padding: 4px;
}

QListWidget::item:selected {
    background-color: #0c4a6e;
    color: #f8fafc;
}

QProgressBar {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    text-align: center;
    color: #f8fafc;
    height: 16px;
}

QProgressBar::chunk {
    background-color: #0284c7;
    border-radius: 3px;
}

QSplitter::handle {
    background-color: #253042;
}

QSplitter::handle:hover {
    background-color: #38bdf8;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background: #0f141c;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #334155;
    min-height: 24px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QStatusBar {
    background-color: #0b0f17;
    color: #94a3b8;
    border-top: 1px solid #1e293b;
}
"""


def configure_pyqtgraph() -> None:
    """Настраивает глобальные параметры рендеринга PyQtGraph под тёмную тему."""
    import pyqtgraph as pg

    pg.setConfigOptions(
        background=PLOT_BG_COLOR,
        foreground=PLOT_FG_COLOR,
        antialias=True,
    )