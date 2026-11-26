from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from src.viz.config import translate_roi_name

PLOT_BG_COLOR = "#0f141c"
PLOT_AXIS_COLOR = "#94a3b8"
CURSOR_COLOR = "#38bdf8"
SELECTED_ROI_COLOR = "#38bdf8"

CHANNEL_PALETTE = [
    "#60a5fa",
    "#34d399",
    "#a78bfa",
    "#f472b6",
    "#fbbf24",
    "#2dd4bf",
    "#fb7185",
    "#818cf8",
    "#4ade80",
    "#f97316",
]


def _shorten_label(text: str, max_len: int = 24) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _decimate_for_plot(
    times: np.ndarray,
    y: np.ndarray,
    max_points: int = 6000,
) -> tuple[np.ndarray, np.ndarray]:
    """Сохраняет положительные и отрицательные пики сигнала при отображении длинных рядов."""
    n = len(times)
    if n <= max_points or max_points < 4:
        return times, y

    n_bins = max_points // 2
    usable = (n // n_bins) * n_bins
    t_reshaped = times[:usable].reshape(n_bins, -1)
    y_reshaped = y[:usable].reshape(n_bins, -1)

    t_out = np.empty(n_bins * 2, dtype=times.dtype)
    y_out = np.empty(n_bins * 2, dtype=y.dtype)
    t_out[0::2] = t_reshaped[:, 0]
    t_out[1::2] = t_reshaped[:, t_reshaped.shape[1] // 2]
    y_out[0::2] = np.min(y_reshaped, axis=1)
    y_out[1::2] = np.max(y_reshaped, axis=1)
    return t_out, y_out


class EEGPlotWidget(QWidget):
    """Интерактивный 2D-виджет ЭЭГ и временных рядов ROI с переключением временных окон."""

    time_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Исходные данные
        self._eeg_signals: np.ndarray | None = None
        self._eeg_ch_names: list[str] = []
        self._eeg_times: np.ndarray | None = None

        self._roi_data: np.ndarray | None = None
        self._roi_names: list[str] = []
        self._roi_times: np.ndarray | None = None
        self._selected_roi_idx: int | None = None

        # Состояние отображения и навигации по времени
        self._mode: str = "eeg"  # "eeg" | "roi" | "gfp"
        self._window_sec: float | None = None  # None -> весь сигнал
        self._t_min: float = 0.0
        self._t_max: float = 10.0
        self._current_time: float = 0.0
        self._updating_cursor: bool = False

        self._curve_items: list[pg.PlotDataItem] = []

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # Панель выбора графика и переключения временных окон
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(2, 0, 2, 0)
        toolbar.setSpacing(8)

        lbl_mode = QLabel("График:")
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("ЭЭГ-каналы", "eeg")
        self.combo_mode.addItem("Топ ROI (sLORETA)", "roi")
        self.combo_mode.addItem("Огибающая активности", "gfp")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)

        lbl_win = QLabel("Окно:")
        self.combo_window = QComboBox()
        self.combo_window.addItem("Весь сигнал", None)
        self.combo_window.addItem("2 с", 2.0)
        self.combo_window.addItem("5 с", 5.0)
        self.combo_window.addItem("10 с", 10.0)
        self.combo_window.addItem("30 с", 30.0)
        self.combo_window.currentIndexChanged.connect(self._on_window_changed)

        self.btn_prev_win = QPushButton("◀ Окно")
        self.btn_prev_win.setEnabled(False)
        self.btn_prev_win.setToolTip("Перейти на одно выбранное окно времени назад")
        self.btn_prev_win.clicked.connect(lambda: self.step_time_window(-1))

        self.btn_next_win = QPushButton("Окно ▶")
        self.btn_next_win.setEnabled(False)
        self.btn_next_win.setToolTip("Перейти на одно выбранное окно времени вперёд")
        self.btn_next_win.clicked.connect(lambda: self.step_time_window(1))

        self.lbl_window_info = QLabel("0.0–0.0 с")
        self.lbl_window_info.setStyleSheet("color: #38bdf8; font-weight: 600;")

        toolbar.addWidget(lbl_mode)
        toolbar.addWidget(self.combo_mode)
        toolbar.addWidget(lbl_win)
        toolbar.addWidget(self.combo_window)
        toolbar.addStretch(1)
        toolbar.addWidget(self.btn_prev_win)
        toolbar.addWidget(self.btn_next_win)
        toolbar.addWidget(self.lbl_window_info)

        root.addLayout(toolbar)

        # Статичный по масштабу холст PyQtGraph
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(PLOT_BG_COLOR)
        self.plot_widget.showGrid(x=True, y=False, alpha=0.18)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMenuEnabled(False)

        plot_item = self.plot_widget.getPlotItem()
        plot_item.setLabel("bottom", "Время", units="с", color=PLOT_AXIS_COLOR)
        for axis_name in ("bottom", "left"):
            axis = plot_item.getAxis(axis_name)
            axis.setPen(pg.mkPen("#334155"))
            axis.setTextPen(pg.mkPen(PLOT_AXIS_COLOR))

        # Вертикальный синхронный курсор
        self.cursor_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=pg.mkPen(CURSOR_COLOR, width=2),
            hoverPen=pg.mkPen("#7dd3fc", width=3),
        )
        self.cursor_line.setZValue(100)
        self.cursor_line.sigPositionChanged.connect(self._on_cursor_dragged)
        self.plot_widget.addItem(self.cursor_line)

        # Клик по области графика перемещает курсор
        self.plot_widget.scene().sigMouseClicked.connect(self._on_plot_clicked)

        root.addWidget(self.plot_widget, stretch=1)

    # ------------------------------------------------------------------
    # Загрузка и обновление данных
    # ------------------------------------------------------------------

    def set_eeg_data(
        self,
        signals: np.ndarray,
        ch_names: list[str],
        times: np.ndarray,
    ) -> None:
        """Загружает многоканальные сигналы ЭЭГ для 2D-визуализации."""
        sig = np.asarray(signals, dtype=np.float64)
        t = np.asarray(times, dtype=np.float64).ravel()
        if sig.ndim == 1:
            sig = sig[np.newaxis, :]
        if sig.shape[1] != t.shape[0]:
            min_len = min(sig.shape[1], t.shape[0])
            sig = sig[:, :min_len]
            t = t[:min_len]

        self._eeg_signals = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        self._eeg_ch_names = list(ch_names[: self._eeg_signals.shape[0]])
        self._eeg_times = t

        self._update_time_bounds()
        self._redraw_curves()

    def set_roi_data(
        self,
        roi_data: np.ndarray,
        roi_names: list[str],
        times: np.ndarray,
    ) -> None:
        """Загружает матрицу активности корковых зон (68, T) и временную ось."""
        data = np.asarray(roi_data, dtype=np.float64)
        t = np.asarray(times, dtype=np.float64).ravel()
        if data.ndim != 2:
            raise ValueError(f"Ожидалась 2D-матрица ROI, получено shape={data.shape}.")
        if data.shape[1] != t.shape[0]:
            min_len = min(data.shape[1], t.shape[0])
            data = data[:, :min_len]
            t = t[:min_len]

        self._roi_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        self._roi_names = list(roi_names)
        self._roi_times = t

        # Если ЭЭГ не загружена, автоматически переключаемся в режим ROI
        if self._eeg_signals is None and self._mode == "eeg":
            self._set_mode_programmatically("roi")

        self._update_time_bounds()
        self._redraw_curves()

    def set_selected_roi(self, roi_idx: int | None) -> None:
        """Выделяет выбранную в 3D корковую зону на графике (в режиме ROI)."""
        self._selected_roi_idx = roi_idx
        if self._mode == "roi":
            self._redraw_curves()

    def clear_data(self) -> None:
        self._eeg_signals = None
        self._eeg_ch_names = []
        self._eeg_times = None
        self._roi_data = None
        self._roi_names = []
        self._roi_times = None
        self._selected_roi_idx = None
        self._t_min = 0.0
        self._t_max = 10.0
        self._current_time = 0.0

        self._updating_cursor = True
        try:
            self.cursor_line.setBounds((0.0, 10.0))
            self.cursor_line.setValue(0.0)
        finally:
            self._updating_cursor = False

        self._clear_plot_items()
        self.plot_widget.getPlotItem().getAxis("left").setTicks([[]])
        self._update_window_buttons_state()
        self.lbl_window_info.setText("0.0–0.0 с")

    def _set_mode_programmatically(self, mode: str) -> None:
        idx = self.combo_mode.findData(mode)
        if idx >= 0:
            self.combo_mode.blockSignals(True)
            self.combo_mode.setCurrentIndex(idx)
            self.combo_mode.blockSignals(False)
            self._mode = mode

    def _update_time_bounds(self) -> None:
        mins: list[float] = []
        maxs: list[float] = []
        if self._eeg_times is not None and len(self._eeg_times) > 1:
            mins.append(float(self._eeg_times[0]))
            maxs.append(float(self._eeg_times[-1]))
        if self._roi_times is not None and len(self._roi_times) > 1:
            mins.append(float(self._roi_times[0]))
            maxs.append(float(self._roi_times[-1]))

        if mins and maxs:
            self._t_min = min(mins)
            self._t_max = max(maxs)
        else:
            self._t_min, self._t_max = 0.0, 10.0

        self.cursor_line.setBounds((self._t_min, self._t_max))
        self._current_time = float(np.clip(self._current_time, self._t_min, self._t_max))
        self._update_window_buttons_state()
        self._apply_x_range(center_on_cursor=False)

    def _update_window_buttons_state(self) -> None:
        total_dur = self._t_max - self._t_min
        can_step = self._window_sec is not None and total_dur > self._window_sec
        self.btn_prev_win.setEnabled(can_step)
        self.btn_next_win.setEnabled(can_step)

    def _update_window_info_label(self, start_sec: float, end_sec: float) -> None:
        self.lbl_window_info.setText(f"{start_sec:.1f}–{end_sec:.1f} с")

    # ------------------------------------------------------------------
    # Отрисовка кривых (фиксированный амплитудный масштаб)
    # ------------------------------------------------------------------

    def _clear_plot_items(self) -> None:
        for item in self._curve_items:
            self.plot_widget.removeItem(item)
        self._curve_items.clear()

    def _redraw_curves(self) -> None:
        self._clear_plot_items()
        left_axis = self.plot_widget.getPlotItem().getAxis("left")

        if self._mode == "eeg":
            if self._eeg_signals is not None and self._eeg_times is not None:
                self._draw_multichannel_eeg()
                return
            if self._roi_data is not None and self._roi_times is not None:
                self._draw_top_rois()
                return
        elif self._mode == "roi":
            if self._roi_data is not None and self._roi_times is not None:
                self._draw_top_rois()
                return
            if self._eeg_signals is not None and self._eeg_times is not None:
                self._draw_multichannel_eeg()
                return
        elif self._mode == "gfp":
            self._draw_gfp_envelope()
            return

        left_axis.setTicks([[]])

    def _draw_multichannel_eeg(self) -> None:
        assert self._eeg_signals is not None and self._eeg_times is not None
        signals = self._eeg_signals
        times = self._eeg_times
        n_ch = signals.shape[0]
        spacing = 1.0

        ticks: list[tuple[float, str]] = []
        for idx in range(n_ch):
            y = signals[idx]
            y_centered = y - np.median(y)
            robust_scale = float(np.percentile(np.abs(y_centered), 98))
            if robust_scale < 1e-12:
                robust_scale = 1.0

            offset = (n_ch - 1 - idx) * spacing
            y_plot = (y_centered / (robust_scale * 2.4)) + offset
            t_vis, y_vis = _decimate_for_plot(times, y_plot)

            color = CHANNEL_PALETTE[idx % len(CHANNEL_PALETTE)]
            curve = self.plot_widget.plot(
                t_vis,
                y_vis,
                pen=pg.mkPen(color, width=1.15),
            )
            self._curve_items.append(curve)

            name = self._eeg_ch_names[idx] if idx < len(self._eeg_ch_names) else f"Ch {idx + 1}"
            ticks.append((offset, name))

        self.plot_widget.getPlotItem().getAxis("left").setTicks([ticks])
        self.plot_widget.setYRange(-0.75, max(0.75, (n_ch - 1) * spacing + 0.75), padding=0.02)

    def _draw_top_rois(self) -> None:
        assert self._roi_data is not None and self._roi_times is not None
        roi_data = self._roi_data
        times = self._roi_times

        # Отбираем до 8 самых активных корковых зон по 95-му перцентилю
        scores = np.percentile(roi_data, 95, axis=1)
        top_indices = list(np.argsort(scores)[::-1][:8])

        # Гарантируем присутствие выбранной в 3D зоны на первом месте
        if (
            self._selected_roi_idx is not None
            and 0 <= self._selected_roi_idx < roi_data.shape[0]
        ):
            if self._selected_roi_idx in top_indices:
                top_indices.remove(self._selected_roi_idx)
            top_indices.insert(0, self._selected_roi_idx)
            top_indices = top_indices[:8]

        n_show = len(top_indices)
        spacing = 1.0
        global_peak = float(np.percentile(roi_data[top_indices], 99))
        if global_peak < 1e-12:
            global_peak = 1.0

        ticks: list[tuple[float, str]] = []
        for row, roi_idx in enumerate(top_indices):
            y = roi_data[roi_idx]
            offset = (n_show - 1 - row) * spacing
            y_plot = (y / (global_peak * 1.15)) * 0.85 + offset
            t_vis, y_vis = _decimate_for_plot(times, y_plot)

            is_selected = roi_idx == self._selected_roi_idx
            color = SELECTED_ROI_COLOR if is_selected else CHANNEL_PALETTE[row % len(CHANNEL_PALETTE)]
            width = 2.0 if is_selected else 1.25

            curve = self.plot_widget.plot(
                t_vis,
                y_vis,
                pen=pg.mkPen(color, width=width),
            )
            self._curve_items.append(curve)

            raw_name = (
                self._roi_names[roi_idx]
                if roi_idx < len(self._roi_names)
                else f"ROI {roi_idx + 1}"
            )
            ru_name = _shorten_label(translate_roi_name(raw_name), max_len=24)
            if is_selected:
                ru_name = f"★ {ru_name}"
            ticks.append((offset, ru_name))

        self.plot_widget.getPlotItem().getAxis("left").setTicks([ticks])
        self.plot_widget.setYRange(-0.35, max(1.0, (n_show - 1) * spacing + 1.05), padding=0.02)

    def _draw_gfp_envelope(self) -> None:
        env, times, label = self._compute_envelope_series()
        if env is None or times is None:
            self.plot_widget.getPlotItem().getAxis("left").setTicks([[]])
            return

        peak = float(np.max(env))
        if peak < 1e-12:
            peak = 1.0
        y_plot = env / peak
        t_vis, y_vis = _decimate_for_plot(times, y_plot)

        curve = self.plot_widget.plot(
            t_vis,
            y_vis,
            pen=pg.mkPen("#38bdf8", width=1.6),
            fillLevel=0.0,
            brush=(56, 189, 248, 42),
        )
        self._curve_items.append(curve)

        self.plot_widget.getPlotItem().getAxis("left").setTicks([[(0.0, "0"), (0.5, label), (1.0, "Макс")]])
        self.plot_widget.setYRange(-0.05, 1.1, padding=0.02)

    def _compute_envelope_series(self) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        """Возвращает 1D-огибающую активности (приоритетно из sLORETA ROI, иначе из ЭЭГ GFP)."""
        if self._roi_data is not None and self._roi_times is not None and len(self._roi_times) > 2:
            env = np.percentile(self._roi_data, 95, axis=0)
            return env, self._roi_times, "sLORETA (95%)"

        if self._eeg_signals is not None and self._eeg_times is not None and len(self._eeg_times) > 2:
            centered = self._eeg_signals - np.mean(self._eeg_signals, axis=1, keepdims=True)
            gfp = np.std(centered, axis=0)
            return gfp, self._eeg_times, "ЭЭГ GFP"

        return None, None, ""

    # ------------------------------------------------------------------
    # Пошаговое переключение по временным окнам и управление курсором
    # ------------------------------------------------------------------

    def step_time_window(self, direction: int) -> None:
        """Сдвигает временное окно и курсор на +/- выбранную длительность окна (например, ±10 с)."""
        if self._window_sec is None:
            return

        total_dur = self._t_max - self._t_min
        if total_dur <= self._window_sec:
            return

        x_min, _ = self.plot_widget.getViewBox().viewRange()[0]
        step = float(direction) * self._window_sec
        max_start = self._t_max - self._window_sec

        new_start = float(np.clip(x_min + step, self._t_min, max_start))
        new_end = min(self._t_max, new_start + self._window_sec)

        # Сохраняем относительное положение курсора внутри окна при переходе
        rel_offset = float(np.clip(self._current_time - x_min, 0.0, self._window_sec))
        new_time = float(np.clip(new_start + rel_offset, new_start, new_end))

        self.plot_widget.setXRange(new_start, new_end, padding=0.0)
        self._update_window_info_label(new_start, new_end)

        self._current_time = new_time
        self._updating_cursor = True
        try:
            self.cursor_line.setValue(new_time)
        finally:
            self._updating_cursor = False

        self.time_changed.emit(new_time)

    def set_current_time(self, time_sec: float) -> None:
        """Программно обновляет положение вертикального курсора без обратного сигнала."""
        t_clamped = float(np.clip(time_sec, self._t_min, self._t_max))
        self._current_time = t_clamped

        self._updating_cursor = True
        try:
            self.cursor_line.setValue(t_clamped)
        finally:
            self._updating_cursor = False

        # Перелистывание окна при выходе курсора за его границы во время воспроизведения
        if self._window_sec is not None:
            x_min, x_max = self.plot_widget.getViewBox().viewRange()[0]
            if t_clamped < x_min or t_clamped > x_max:
                max_start = max(self._t_min, self._t_max - self._window_sec)
                start = float(np.clip(t_clamped, self._t_min, max_start))
                end = min(self._t_max, start + self._window_sec)
                self.plot_widget.setXRange(start, end, padding=0.0)
                self._update_window_info_label(start, end)

    def _on_cursor_dragged(self) -> None:
        if self._updating_cursor:
            return
        pos = float(np.clip(self.cursor_line.value(), self._t_min, self._t_max))
        self._current_time = pos
        self.time_changed.emit(pos)

    def _on_plot_clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        vb = self.plot_widget.getViewBox()
        scene_pos = event.scenePos()
        if not vb.sceneBoundingRect().contains(scene_pos):
            return

        view_pt = vb.mapSceneToView(scene_pos)
        t_clicked = float(np.clip(view_pt.x(), self._t_min, self._t_max))
        self.set_current_time(t_clicked)
        self.time_changed.emit(t_clicked)

    def _on_mode_changed(self) -> None:
        self._mode = str(self.combo_mode.currentData())
        self._redraw_curves()

    def _on_window_changed(self) -> None:
        self._window_sec = self.combo_window.currentData()
        self._update_window_buttons_state()
        self._apply_x_range(center_on_cursor=True)

    def _apply_x_range(self, center_on_cursor: bool = False) -> None:
        total_dur = self._t_max - self._t_min
        if self._window_sec is None or total_dur <= self._window_sec:
            self.plot_widget.setXRange(self._t_min, self._t_max, padding=0.01)
            self._update_window_info_label(self._t_min, self._t_max)
            return

        if center_on_cursor:
            start = self._current_time - self._window_sec * 0.5
        else:
            start = self._t_min
        start = max(self._t_min, min(start, self._t_max - self._window_sec))
        end = min(self._t_max, start + self._window_sec)
        self.plot_widget.setXRange(start, end, padding=0.0)
        self._update_window_info_label(start, end)