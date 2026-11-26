from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.preprocessing.signal_utils import clean_roi_matrix
from src.viz.brain_view_3d import BrainView3DWidget
from src.viz.config import (
    AVAILABLE_COLORMAPS,
    DEFAULT_ELECTRODES,
    DEFAULT_SFREQ,
    get_ordered_roi_names,
    translate_roi_name,
)
from src.viz.eeg_plot_widget import EEGPlotWidget
from src.viz.workers import (
    LoadEEGWorker,
    LoadROIWorker,
    PreprocessWorker,
    SloretaWorker,
)

STATIC_SCALP_OPACITY = 0.15


class BrainViewerApp(QMainWindow):
    """Главное окно десктопного приложения для анализа ЭЭГ и 3D-реконструкция sLORETA."""

    def __init__(self, subjects_dir: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("NeuroViz Desktop — Анализ ЭЭГ и 3D-реконструкция sLORETA")
        self.resize(1560, 900)

        self._subjects_dir = Path(subjects_dir) if subjects_dir is not None else None

        # Состояние загруженных и расчётных данных
        self._raw_eeg: mne.io.BaseRaw | mne.BaseEpochs | None = None
        self._clean_eeg: mne.io.BaseRaw | mne.BaseEpochs | None = None
        self._eeg_file_path: Path | None = None

        self._eeg_signals: np.ndarray | None = None
        self._eeg_ch_names: list[str] = []
        self._eeg_times: np.ndarray | None = None

        self._roi_raw_data: np.ndarray | None = None  # исходная матрица (68, T)
        self._roi_data: np.ndarray | None = None      # активная матрица для рендера (68, T)
        self._roi_names: list[str] = get_ordered_roi_names()
        self._roi_times: np.ndarray | None = None
        self._roi_sfreq: float = DEFAULT_SFREQ
        self._roi_file_path: Path | None = None

        # Параметры отображения и воспроизведения
        self._current_frame: int = 0
        self._current_time: float = 0.0
        self._vmax_dynamic: float = 1.0
        self._is_average_mode: bool = False
        self._use_russian_names: bool = True
        self._selected_roi_idx: int | None = None
        self._last_displayed_vec: np.ndarray | None = None

        # Активный фоновый поток
        self._active_worker: Any = None

        # Таймер строго покадрового воспроизведения (без пропуска кадров)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(50)
        self._play_timer.timeout.connect(self._on_play_tick)

        # Объединение частых событий мыши при перетаскивании ползунка/курсора
        self._pending_render: tuple[int, bool] | None = None
        self._drag_render_timer = QTimer(self)
        self._drag_render_timer.setSingleShot(True)
        self._drag_render_timer.setInterval(16)
        self._drag_render_timer.timeout.connect(self._flush_pending_render)

        self._build_ui()
        self._connect_signals()
        self._setup_shortcuts()
        self._init_empty_state()

    # ------------------------------------------------------------------
    # Построение интерфейса
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # ================= ЛЕВАЯ ПАНЕЛЬ УПРАВЛЕНИЯ (С ПРОКРУТКОЙ) =================
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(345)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(2, 0, 6, 4)
        left_layout.setSpacing(6)

        # --- Блок 1: Файлы и расчёт sLORETA ---
        gb_files = QGroupBox("1. Данные и реконструкция sLORETA")
        f_lay = QVBoxLayout(gb_files)
        f_lay.setSpacing(6)

        self.lbl_eeg_status = QLabel("ЭЭГ: не загружена")
        self.lbl_eeg_status.setWordWrap(True)
        self.lbl_eeg_status.setStyleSheet("color: #94a3b8; font-weight: normal;")

        self.lbl_roi_status = QLabel("ROI (68 зон): не рассчитано")
        self.lbl_roi_status.setWordWrap(True)
        self.lbl_roi_status.setStyleSheet("color: #94a3b8; font-weight: normal;")

        btn_row_load = QHBoxLayout()
        self.btn_load_eeg = QPushButton("Загрузить ЭЭГ")
        self.btn_load_eeg.setProperty("accent", "true")
        self.btn_load_eeg.setToolTip("Открыть запись ЭЭГ (.set, .edf, .bdf, .fif, .vhdr)")

        self.btn_load_roi = QPushButton("Загрузить .npy")
        self.btn_load_roi.setToolTip("Открыть готовую матрицу активности 68 зон (.npy)")

        btn_row_load.addWidget(self.btn_load_eeg)
        btn_row_load.addWidget(self.btn_load_roi)

        btn_row_sloreta = QHBoxLayout()
        self.btn_run_sloreta = QPushButton("Рассчитать sLORETA (68 зон)")
        self.btn_run_sloreta.setProperty("accent", "true")
        self.btn_run_sloreta.setEnabled(False)
        self.btn_run_sloreta.setToolTip(
            "Выполнить реконструкцию источников sLORETA по загруженной ЭЭГ"
        )

        self.btn_save_roi = QPushButton("Сохранить .npy")
        self.btn_save_roi.setToolTip("Сохранить текущую матрицу активности (68, T) в файл .npy")
        btn_row_sloreta.addWidget(self.btn_run_sloreta, stretch=2)
        btn_row_sloreta.addWidget(self.btn_save_roi, stretch=1)

        self.btn_clear_all = QPushButton("Очистить файлы и сбросить")
        self.btn_clear_all.setStyleSheet(
            "background-color: #451a1a; border-color: #991b1b; color: #fecaca;"
        )

        f_lay.addWidget(self.lbl_eeg_status)
        f_lay.addWidget(self.lbl_roi_status)
        f_lay.addLayout(btn_row_load)
        f_lay.addLayout(btn_row_sloreta)
        f_lay.addWidget(self.btn_clear_all)
        left_layout.addWidget(gb_files)

        # --- Блок 2: Предобработка ЭЭГ и нормализация ROI ---
        gb_prep = QGroupBox("2. Очистка ЭЭГ и нормализация")
        p_lay = QVBoxLayout(gb_prep)
        p_lay.setSpacing(5)

        freq_form = QFormLayout()
        freq_form.setSpacing(4)

        self.spin_lfreq = QDoubleSpinBox()
        self.spin_lfreq.setRange(0.1, 40.0)
        self.spin_lfreq.setSingleStep(0.5)
        self.spin_lfreq.setValue(1.0)
        self.spin_lfreq.setSuffix(" Гц")

        self.spin_hfreq = QDoubleSpinBox()
        self.spin_hfreq.setRange(5.0, 120.0)
        self.spin_hfreq.setSingleStep(1.0)
        self.spin_hfreq.setValue(40.0)
        self.spin_hfreq.setSuffix(" Гц")

        freq_form.addRow("ФВЧ (нижняя):", self.spin_lfreq)
        freq_form.addRow("ФНЧ (верхняя):", self.spin_hfreq)

        self.chk_notch = QCheckBox("Notch 50 Гц (сетевая наводка)")
        self.chk_notch.setChecked(True)

        self.chk_car = QCheckBox("Средний референс CAR")
        self.chk_car.setChecked(True)

        self.chk_ica = QCheckBox("ICA + ICLabel (удаление артефактов)")
        self.chk_ica.setChecked(False)
        self.chk_ica.setToolTip(
            "Нейросетевая очистка морганий, миограммы и кардиоартефактов (занимает 5–20 с)"
        )

        self.chk_zscore = QCheckBox("Сглаживание и Z-score карты коры")
        self.chk_zscore.setChecked(True)

        self.btn_clean = QPushButton("Выполнить очистку")
        self.btn_clean.setProperty("accent", "true")

        p_lay.addLayout(freq_form)
        p_lay.addWidget(self.chk_notch)
        p_lay.addWidget(self.chk_car)
        p_lay.addWidget(self.chk_ica)
        p_lay.addWidget(self.chk_zscore)
        p_lay.addWidget(self.btn_clean)
        left_layout.addWidget(gb_prep)

        # --- Блок 3: 3D-модель головы, слои и ракурсы ---
        gb_3d = QGroupBox("3. 3D-модель головы и ракурсы")
        v3_lay = QVBoxLayout(gb_3d)
        v3_lay.setSpacing(5)

        layers_row1 = QHBoxLayout()
        self.chk_scalp = QCheckBox("3D-голова (скальп)")
        self.chk_scalp.setChecked(True)

        self.chk_electrodes = QCheckBox("Электроды 10–20")
        self.chk_electrodes.setChecked(True)
        layers_row1.addWidget(self.chk_scalp)
        layers_row1.addWidget(self.chk_electrodes)

        layers_row2 = QHBoxLayout()
        self.chk_elec_labels = QCheckBox("Подписи (P7, T7...)")
        self.chk_elec_labels.setChecked(True)

        self.chk_lh = QCheckBox("Лев.")
        self.chk_lh.setChecked(True)
        self.chk_rh = QCheckBox("Прав.")
        self.chk_rh.setChecked(True)
        layers_row2.addWidget(self.chk_elec_labels)
        layers_row2.addStretch(1)
        layers_row2.addWidget(self.chk_lh)
        layers_row2.addWidget(self.chk_rh)

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Палитра:"))
        self.combo_cmap = QComboBox()
        for cmap_name in AVAILABLE_COLORMAPS:
            self.combo_cmap.addItem(cmap_name, cmap_name)
        cmap_row.addWidget(self.combo_cmap, stretch=1)

        self.lbl_thresh = QLabel("Порог очагов активности: 20%")
        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(0, 90)
        self.slider_thresh.setValue(20)

        # 5 анатомических ракурсов камеры
        cam_row1 = QHBoxLayout()
        self._cam_buttons: dict[str, QPushButton] = {}
        for title, preset in [("Спереди", "front"), ("Сзади", "back"), ("Сверху", "top")]:
            btn = QPushButton(title)
            self._cam_buttons[preset] = btn
            cam_row1.addWidget(btn)

        cam_row2 = QHBoxLayout()
        for title, preset in [("Слева", "left"), ("Справа", "right")]:
            btn = QPushButton(title)
            self._cam_buttons[preset] = btn
            cam_row2.addWidget(btn)

        v3_lay.addLayout(layers_row1)
        v3_lay.addLayout(layers_row2)
        v3_lay.addLayout(cmap_row)
        v3_lay.addWidget(self.lbl_thresh)
        v3_lay.addWidget(self.slider_thresh)
        v3_lay.addLayout(cam_row1)
        v3_lay.addLayout(cam_row2)
        left_layout.addWidget(gb_3d)

        # --- Блок 4: Топ-5 активных корковых зон ---
        gb_top_rois = QGroupBox("4. Топ-5 активных корковых зон")
        tr_lay = QVBoxLayout(gb_top_rois)
        tr_lay.setSpacing(4)

        self.btn_lang = QPushButton("Язык зон: Русский (RU / EN)")
        self.btn_lang.setCheckable(True)
        self.btn_lang.setChecked(True)

        self.list_top_rois = QListWidget()
        self.list_top_rois.setMaximumHeight(118)
        self.list_top_rois.setToolTip("Нажмите на зону, чтобы выделить её на 3D-модели")

        tr_lay.addWidget(self.btn_lang)
        tr_lay.addWidget(self.list_top_rois)
        left_layout.addWidget(gb_top_rois)

        # --- Блок 5: Топ-5 активных электродов ЭЭГ ---
        gb_top_elecs = QGroupBox("5. Топ-5 активных электродов ЭЭГ")
        te_lay = QVBoxLayout(gb_top_elecs)
        te_lay.setSpacing(4)

        self.list_top_elecs = QListWidget()
        self.list_top_elecs.setMaximumHeight(112)
        te_lay.addWidget(self.list_top_elecs)
        left_layout.addWidget(gb_top_elecs)

        # --- Журнал операций ---
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(80)
        self.log_box.setMaximumHeight(110)
        left_layout.addWidget(self.log_box)

        scroll.setWidget(left_container)
        root_layout.addWidget(scroll)

        # ================= ЦЕНТР + ПРАВАЯ ПАНЕЛЬ =================
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Центр: 3D-голова + панель времени и покадровой навигации ---
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)

        self.brain_view = BrainView3DWidget(subjects_dir=self._subjects_dir)
        self.brain_view.set_scalp_opacity(STATIC_SCALP_OPACITY)
        self.brain_view.set_electrode_labels_visible(True)
        self._roi_names = self.brain_view.roi_names
        center_layout.addWidget(self.brain_view, stretch=1)

        # Нижняя панель навигации по кадрам и времени
        time_bar = QHBoxLayout()
        time_bar.setSpacing(6)

        self.btn_play = QPushButton("▶ Старт")
        self.btn_play.setFixedWidth(86)

        self.btn_prev_frame = QPushButton("◀ Кадр")
        self.btn_prev_frame.setFixedWidth(74)
        self.btn_prev_frame.setToolTip("Предыдущий кадр (стрелка влево)")

        self.btn_next_frame = QPushButton("Кадр ▶")
        self.btn_next_frame.setFixedWidth(74)
        self.btn_next_frame.setToolTip("Следующий кадр (стрелка вправо)")

        self.combo_speed = QComboBox()
        self.combo_speed.addItem("5 к/с", 200)
        self.combo_speed.addItem("10 к/с", 100)
        self.combo_speed.addItem("20 к/с", 50)
        self.combo_speed.addItem("40 к/с", 25)
        self.combo_speed.setCurrentIndex(2)  # 20 кадров/с по умолчанию
        self.combo_speed.setFixedWidth(74)
        self.combo_speed.setToolTip("Скорость покадрового воспроизведения (без пропуска кадров)")
        self.combo_speed.currentIndexChanged.connect(
            lambda: self._play_timer.setInterval(int(self.combo_speed.currentData() or 50))
        )

        self.btn_average = QPushButton("Среднее")
        self.btn_average.setCheckable(True)
        self.btn_average.setFixedWidth(80)
        self.btn_average.setToolTip("Усреднённая RMS-активность за всю запись")

        self.slider_time = QSlider(Qt.Orientation.Horizontal)
        self.slider_time.setRange(0, 0)

        self.lbl_time = QLabel("0.00 с | Кадр 0/0")
        self.lbl_time.setFixedWidth(155)
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        time_bar.addWidget(self.btn_play)
        time_bar.addWidget(self.btn_prev_frame)
        time_bar.addWidget(self.btn_next_frame)
        time_bar.addWidget(self.combo_speed)
        time_bar.addWidget(self.btn_average)
        time_bar.addWidget(self.slider_time, stretch=1)
        time_bar.addWidget(self.lbl_time)
        center_layout.addLayout(time_bar)

        splitter.addWidget(center_panel)

        # --- Правая панель: сигналы ЭЭГ / разряды ---
        self.eeg_plot = EEGPlotWidget()
        splitter.addWidget(self.eeg_plot)

        splitter.setSizes([720, 480])
        root_layout.addWidget(splitter, stretch=1)

        # --- Строка состояния и индикатор прогресса ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(220)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.statusBar().showMessage("Готово к работе")

    def _connect_signals(self) -> None:
        # Файлы и расчёты
        self.btn_load_eeg.clicked.connect(self._on_load_eeg_clicked)
        self.btn_load_roi.clicked.connect(self._on_load_roi_clicked)
        self.btn_run_sloreta.clicked.connect(self._on_run_sloreta_clicked)
        self.btn_save_roi.clicked.connect(self._on_save_roi_clicked)
        self.btn_clear_all.clicked.connect(self._on_clear_all_clicked)

        # Очистка и нормализация
        self.btn_clean.clicked.connect(self._on_clean_clicked)
        self.chk_zscore.toggled.connect(self._on_zscore_toggled)

        # Слои 3D-модели и ракурсы
        self.chk_scalp.toggled.connect(self.brain_view.set_scalp_visible)
        self.chk_electrodes.toggled.connect(self.brain_view.set_electrodes_visible)
        self.chk_elec_labels.toggled.connect(self.brain_view.set_electrode_labels_visible)
        self.chk_lh.toggled.connect(self._on_hemispheres_toggled)
        self.chk_rh.toggled.connect(self._on_hemispheres_toggled)
        self.combo_cmap.currentIndexChanged.connect(self._on_cmap_changed)
        self.slider_thresh.valueChanged.connect(self._on_threshold_changed)

        for preset, btn in self._cam_buttons.items():
            btn.clicked.connect(lambda _, p=preset: self.brain_view.set_camera_preset(p))

        # Списки топ-зон и электродов
        self.btn_lang.toggled.connect(self._on_language_toggled)
        self.list_top_rois.itemClicked.connect(self._on_top_roi_item_clicked)

        # Навигация по кадрам и синхронизация 2D <-> 3D
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_prev_frame.clicked.connect(lambda: self._step_single_frame(-1))
        self.btn_next_frame.clicked.connect(lambda: self._step_single_frame(1))
        self.btn_average.toggled.connect(self._on_average_toggled)
        self.slider_time.valueChanged.connect(self._on_slider_frame_changed)

        self.eeg_plot.time_changed.connect(self._on_plot_time_changed)
        self.brain_view.roi_clicked.connect(self._on_brain_roi_clicked)

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self._step_single_frame(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._step_single_frame(1))
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle_play)

    def log(self, message: str) -> None:
        self.log_box.append(f"• {message}")
        self.statusBar().showMessage(message, 6000)

    def _set_busy(self, busy: bool, status_text: str = "") -> None:
        self.progress_bar.setVisible(busy)
        self.btn_load_eeg.setEnabled(not busy)
        self.btn_load_roi.setEnabled(not busy)
        self.btn_clean.setEnabled(not busy)
        self.btn_clear_all.setEnabled(not busy)
        self.btn_run_sloreta.setEnabled(
            not busy and (self._raw_eeg is not None or self._clean_eeg is not None)
        )
        if status_text:
            self.statusBar().showMessage(status_text)

    # ------------------------------------------------------------------
    # Инициализация и сброс пустого состояния (без демо-данных)
    # ------------------------------------------------------------------

    def _init_empty_state(self) -> None:
        self._reset_scene_to_empty(
            "Приложение готово. Загрузите запись ЭЭГ или готовую матрицу ROI (.npy)."
        )

    def _clear_roi_state(self) -> None:
        """Сбрасывает расчётную активность коры в нейтральное серое состояние."""
        self._roi_raw_data = None
        self._roi_data = None
        self._roi_times = None
        self._roi_file_path = None
        self._selected_roi_idx = None
        self._last_displayed_vec = None
        self.lbl_roi_status.setText("ROI (68 зон): не рассчитано")
        self.list_top_rois.clear()
        self.brain_view.highlight_roi(None)
        self.brain_view.update_activity(
            np.zeros(len(self._roi_names), dtype=np.float32), vmax=1.0
        )

    def _reset_scene_to_empty(self, log_message: str) -> None:
        self._play_timer.stop()
        self.btn_play.setText("▶ Старт")

        self._raw_eeg = None
        self._clean_eeg = None
        self._eeg_file_path = None
        self._eeg_signals = None
        self._eeg_ch_names = []
        self._eeg_times = None

        self._current_frame = 0
        self._current_time = 0.0

        if self._is_average_mode:
            self.btn_average.blockSignals(True)
            self.btn_average.setChecked(False)
            self.btn_average.blockSignals(False)
            self._is_average_mode = False

        self.btn_run_sloreta.setEnabled(False)
        self.lbl_eeg_status.setText("ЭЭГ: не загружена")
        self.list_top_elecs.clear()

        self.slider_time.blockSignals(True)
        self.slider_time.setRange(0, 0)
        self.slider_time.setValue(0)
        self.slider_time.blockSignals(False)

        self.eeg_plot.clear_data()
        self._clear_roi_state()
        self.brain_view.update_electrodes(DEFAULT_ELECTRODES)
        self.lbl_time.setText("0.00 с | Кадр 0/0")
        self.log(log_message)

    # ------------------------------------------------------------------
    # Управление данными ROI / ЭЭГ и синхронизация кадров
    # ------------------------------------------------------------------

    def _get_active_timeline(self) -> tuple[str, np.ndarray | None]:
        """Возвращает ('roi', roi_times) если рассчитан sLORETA, иначе ('eeg', eeg_times)."""
        if self._roi_data is not None and self._roi_times is not None and len(self._roi_times) > 0:
            return "roi", self._roi_times
        if self._eeg_signals is not None and self._eeg_times is not None and len(self._eeg_times) > 0:
            return "eeg", self._eeg_times
        return "none", None

    def _apply_roi_dataset(
        self,
        roi_raw: np.ndarray,
        roi_names: list[str],
        times: np.ndarray,
        sfreq: float,
        source_label: str,
    ) -> None:
        self._roi_raw_data = np.asarray(roi_raw, dtype=np.float32)
        self._roi_names = list(roi_names)
        self._roi_times = np.asarray(times, dtype=np.float64)
        self._roi_sfreq = float(sfreq) if sfreq > 0 else DEFAULT_SFREQ

        self._rebuild_processed_roi()
        self.lbl_roi_status.setText(
            f"ROI: {source_label} ({self._roi_data.shape[0]}×{self._roi_data.shape[1]})"
        )

    def _rebuild_processed_roi(self) -> None:
        if self._roi_raw_data is None:
            return

        if self.chk_zscore.isChecked():
            processed = clean_roi_matrix(
                self._roi_raw_data, smooth_window=5, apply_zscore=True
            )
        else:
            processed = clean_roi_matrix(
                np.abs(self._roi_raw_data), smooth_window=5, apply_zscore=False
            )

        self._roi_data = processed
        self._vmax_dynamic = float(np.percentile(self._roi_data, 99.2))
        if self._vmax_dynamic <= 1e-8:
            peak = float(np.max(self._roi_data))
            self._vmax_dynamic = peak if peak > 1e-8 else 1.0

        n_frames = self._roi_data.shape[1]
        if self._roi_times is None or len(self._roi_times) != n_frames:
            self._roi_times = np.arange(n_frames, dtype=np.float64) / self._roi_sfreq

        self.slider_time.blockSignals(True)
        self.slider_time.setRange(0, max(0, n_frames - 1))
        self._current_frame = int(np.argmin(np.abs(self._roi_times - self._current_time)))
        self.slider_time.setValue(self._current_frame)
        self.slider_time.blockSignals(False)

        self.eeg_plot.set_roi_data(self._roi_data, self._roi_names, self._roi_times)

        if self._is_average_mode:
            self._render_average_map()
        else:
            self._render_frame(self._current_frame, sync_plot=True)

    def _render_frame(self, frame_idx: int, sync_plot: bool = True) -> None:
        source, times = self._get_active_timeline()
        if source == "none" or times is None:
            return

        n_frames = len(times)
        idx = int(np.clip(frame_idx, 0, max(0, n_frames - 1)))
        self._current_frame = idx
        self._current_time = float(times[idx])

        if source == "roi" and self._roi_data is not None:
            vec = self._roi_data[:, idx]
            self._last_displayed_vec = vec
            self.brain_view.update_activity(vec, vmax=self._vmax_dynamic)
            self._refresh_top_rois_list(vec)

        self.slider_time.blockSignals(True)
        self.slider_time.setValue(idx)
        self.slider_time.blockSignals(False)

        self.lbl_time.setText(f"{self._current_time:.2f} с | Кадр {idx + 1}/{n_frames}")

        if sync_plot:
            self.eeg_plot.set_current_time(self._current_time)

        self._refresh_top_electrodes_list(center_time=self._current_time, full_record=False)

    def _render_average_map(self) -> None:
        self.lbl_time.setText("Среднее (RMS за запись)")
        self._refresh_top_electrodes_list(center_time=None, full_record=True)

        if self._roi_data is None:
            return

        avg_vec = np.sqrt(np.mean(np.square(self._roi_data), axis=1)).astype(np.float32)
        avg_vmax = float(np.percentile(avg_vec, 98.0))
        if avg_vmax <= 1e-8:
            peak = float(np.max(avg_vec))
            avg_vmax = peak if peak > 1e-8 else 1.0

        self._last_displayed_vec = avg_vec
        self.brain_view.update_activity(avg_vec, vmax=avg_vmax)
        self._refresh_top_rois_list(avg_vec)

    # ------------------------------------------------------------------
    # Обновление рейтингов топ-5 корковых зон и топ-5 электродов
    # ------------------------------------------------------------------

    def _refresh_top_rois_list(self, values_68: np.ndarray) -> None:
        self.list_top_rois.blockSignals(True)
        top_indices = np.argsort(values_68)[::-1][:5]

        while self.list_top_rois.count() > len(top_indices):
            self.list_top_rois.takeItem(self.list_top_rois.count() - 1)

        for rank, idx in enumerate(top_indices):
            raw_name = self._roi_names[idx] if idx < len(self._roi_names) else f"ROI {idx + 1}"
            title = (
                translate_roi_name(raw_name, include_raw=False)
                if self._use_russian_names
                else raw_name
            )
            text = f"{rank + 1}. {title}  [{values_68[idx]:.3f}]"
            if rank < self.list_top_rois.count():
                item = self.list_top_rois.item(rank)
                item.setText(text)
            else:
                item = QListWidgetItem(text)
                self.list_top_rois.addItem(item)
            item.setData(Qt.ItemDataRole.UserRole, int(idx))
            item.setSelected(self._selected_roi_idx == int(idx))

        self.list_top_rois.blockSignals(False)

    def _refresh_top_electrodes_list(
        self,
        center_time: float | None,
        full_record: bool = False,
    ) -> None:
        if (
            self._eeg_signals is None
            or self._eeg_times is None
            or len(self._eeg_ch_names) == 0
        ):
            self.list_top_elecs.clear()
            self.list_top_elecs.addItem("Нет загруженных каналов ЭЭГ")
            return

        if full_record or center_time is None or len(self._eeg_times) < 2:
            rms_uv = np.sqrt(np.mean(np.square(self._eeg_signals), axis=1)) * 1e6
        else:
            half_win = 0.10
            mask = np.abs(self._eeg_times - center_time) <= half_win
            if not np.any(mask):
                nearest = int(np.argmin(np.abs(self._eeg_times - center_time)))
                segment = self._eeg_signals[:, nearest : nearest + 1]
            else:
                segment = self._eeg_signals[:, mask]
            rms_uv = np.sqrt(np.mean(np.square(segment), axis=1)) * 1e6

        top_k = min(5, len(self._eeg_ch_names))
        top_idx = np.argsort(rms_uv)[::-1][:top_k]

        while self.list_top_elecs.count() > top_k:
            self.list_top_elecs.takeItem(self.list_top_elecs.count() - 1)

        for rank, ch_idx in enumerate(top_idx):
            ch_name = self._eeg_ch_names[ch_idx]
            text = f"{rank + 1}. {ch_name:<8}  {rms_uv[ch_idx]:.1f} мкВ"
            if rank < self.list_top_elecs.count():
                self.list_top_elecs.item(rank).setText(text)
            else:
                self.list_top_elecs.addItem(text)

    # ------------------------------------------------------------------
    # Навигация по кадрам и воспроизведение
    # ------------------------------------------------------------------

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setText("▶ Старт")
            return

        source, times = self._get_active_timeline()
        if source == "none" or times is None or len(times) <= 1:
            self.log("Сначала загрузите файл ЭЭГ или матрицу ROI (.npy).")
            return

        self._flush_pending_render()

        if self._is_average_mode:
            self.btn_average.blockSignals(True)
            self.btn_average.setChecked(False)
            self.btn_average.blockSignals(False)
            self._is_average_mode = False

        n_frames = len(times)
        self._current_frame = int(np.argmin(np.abs(times - self._current_time)))
        if self._current_frame >= n_frames - 1:
            self._current_frame = 0

        self.btn_play.setText("⏸ Пауза")
        self._play_timer.start()

    def _on_play_tick(self) -> None:
        source, times = self._get_active_timeline()
        if source == "none" or times is None:
            self._play_timer.stop()
            self.btn_play.setText("▶ Старт")
            return

        n_frames = len(times)
        if n_frames <= 1:
            return

        # Строго +1 кадр на каждый тик — без пропуска ни одного временного отсчёта
        next_frame = (self._current_frame + 1) % n_frames
        self._render_frame(next_frame, sync_plot=True)

    def _schedule_drag_render(self, frame_idx: int, sync_plot: bool) -> None:
        self._pending_render = (frame_idx, sync_plot)
        if not self._drag_render_timer.isActive():
            self._drag_render_timer.start()

    def _flush_pending_render(self) -> None:
        if self._pending_render is not None:
            frame_idx, sync_plot = self._pending_render
            self._pending_render = None
            self._render_frame(frame_idx, sync_plot=sync_plot)

    def _step_single_frame(self, delta: int) -> None:
        source, times = self._get_active_timeline()
        if source == "none" or times is None or len(times) == 0:
            return

        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setText("▶ Старт")
        if self._is_average_mode:
            self.btn_average.setChecked(False)

        n_frames = len(times)
        target = int(np.clip(self._current_frame + delta, 0, n_frames - 1))
        self._render_frame(target, sync_plot=True)

    def _on_slider_frame_changed(self, frame_idx: int) -> None:
        if self._is_average_mode:
            self.btn_average.blockSignals(True)
            self.btn_average.setChecked(False)
            self.btn_average.blockSignals(False)
            self._is_average_mode = False

        self._current_frame = int(frame_idx)
        self._schedule_drag_render(frame_idx, sync_plot=True)

    def _on_plot_time_changed(self, time_sec: float) -> None:
        if self._is_average_mode:
            self.btn_average.blockSignals(True)
            self.btn_average.setChecked(False)
            self.btn_average.blockSignals(False)
            self._is_average_mode = False

        self._current_time = float(time_sec)
        source, times = self._get_active_timeline()
        if source != "none" and times is not None and len(times) > 0:
            frame_idx = int(np.argmin(np.abs(times - time_sec)))
            self._current_frame = frame_idx

            self.slider_time.blockSignals(True)
            self.slider_time.setValue(frame_idx)
            self.slider_time.blockSignals(False)

            self._schedule_drag_render(frame_idx, sync_plot=False)

    def _on_average_toggled(self, checked: bool) -> None:
        self._is_average_mode = bool(checked)
        if checked:
            if self._play_timer.isActive():
                self._play_timer.stop()
                self.btn_play.setText("▶ Старт")
            self._render_average_map()
            self.log("Включён режим усреднённой активности (RMS за всю запись).")
        else:
            source, _ = self._get_active_timeline()
            if source != "none":
                self._render_frame(self._current_frame, sync_plot=True)
            else:
                self.lbl_time.setText("0.00 с | Кадр 0/0")

    # ------------------------------------------------------------------
    # Загрузка файлов, очистка, расчёт sLORETA и сохранение
    # ------------------------------------------------------------------

    def _start_worker(self, worker, on_success, status_text: str) -> None:
        if self._active_worker is not None and self._active_worker.isRunning():
            QMessageBox.information(self, "Выполняется задача", "Дождитесь завершения текущей операции.")
            return

        self._active_worker = worker
        self._set_busy(True, status_text)
        worker.progress.connect(self.log)
        worker.finished_ok.connect(on_success)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(self._on_worker_finished)
        worker.start()

    def _on_worker_error(self, err_msg: str) -> None:
        self.log(f"Ошибка: {err_msg}")
        QMessageBox.critical(self, "Ошибка обработки", err_msg)

    def _on_worker_finished(self) -> None:
        self._set_busy(False, "Готово")
        self._active_worker = None

    def _on_load_eeg_clicked(self) -> None:
        start_dir = "data/raw" if Path("data/raw").exists() else ""
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл записи ЭЭГ",
            start_dir,
            "EEG файлы (*.set *.edf *.bdf *.fif *.vhdr);;Все файлы (*.*)",
        )
        if not path_str:
            return

        worker = LoadEEGWorker(path_str, max_preview_channels=10, parent=self)
        self._start_worker(worker, self._on_eeg_loaded, "Загрузка файла ЭЭГ...")

    def _on_eeg_loaded(self, payload: dict) -> None:
        path: Path = payload["path"]
        eeg_data = payload["eeg_data"]

        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setText("▶ Старт")

        self._raw_eeg = eeg_data
        self._clean_eeg = None
        self._eeg_file_path = path
        self._eeg_signals = payload["signals"]
        self._eeg_ch_names = payload["ch_names"]
        self._eeg_times = payload["times"]
        self._current_frame = 0
        self._current_time = float(self._eeg_times[0]) if len(self._eeg_times) > 0 else 0.0

        # Сбрасываем старую активность коры, чтобы при загрузке .set зоны не светились без sLORETA
        self._clear_roi_state()
        self.eeg_plot.clear_data()

        self.lbl_eeg_status.setText(
            f"ЭЭГ: {path.name} ({payload['data_type']}, {payload['sfreq']:.0f} Гц, {payload['duration_sec']:.1f} с)"
        )
        self.btn_run_sloreta.setEnabled(True)

        # Обновляем 3D-электроды и подписи под реальный монтаж файла
        self.brain_view.update_electrodes(
            ch_names=list(eeg_data.ch_names),
            montage=eeg_data.get_montage(),
        )
        self.eeg_plot.set_eeg_data(self._eeg_signals, self._eeg_ch_names, self._eeg_times)

        n_samples = len(self._eeg_times)
        self.slider_time.blockSignals(True)
        self.slider_time.setRange(0, max(0, n_samples - 1))
        self.slider_time.setValue(0)
        self.slider_time.blockSignals(False)
        self.lbl_time.setText(f"{self._current_time:.2f} с | Кадр 1/{n_samples}")

        self._refresh_top_electrodes_list(
            center_time=self._current_time, full_record=self._is_average_mode
        )

        self.log(
            f"Загружена ЭЭГ '{path.name}' ({len(eeg_data.ch_names)} кан., {payload['duration_sec']:.1f} с). "
            "Для расчёта корковых очагов нажмите «Рассчитать sLORETA (68 зон)»."
        )

    def _on_load_roi_clicked(self) -> None:
        start_dir = "data/processed" if Path("data/processed").exists() else ""
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл активности sLORETA (.npy)",
            start_dir,
            "NumPy матрицы (*.npy)",
        )
        if not path_str:
            return

        worker = LoadROIWorker(path_str, parent=self)
        self._start_worker(worker, self._on_roi_loaded, "Загрузка матрицы ROI...")

    def _on_roi_loaded(self, payload: dict) -> None:
        path: Path = payload["path"]
        roi_arr: np.ndarray = payload["roi_data"]
        self._roi_file_path = path

        n_frames = roi_arr.shape[1]
        if (
            self._eeg_times is not None
            and len(self._eeg_times) > 1
            and self._eeg_file_path is not None
            and self._eeg_file_path.stem in path.stem
        ):
            t_start = float(self._eeg_times[0])
            t_end = float(self._eeg_times[-1])
            times = np.linspace(t_start, t_end, n_frames, dtype=np.float64)
            sfreq = (n_frames - 1) / max(1e-6, t_end - t_start)
        else:
            sfreq = self._roi_sfreq if self._roi_sfreq > 0 else DEFAULT_SFREQ
            times = np.arange(n_frames, dtype=np.float64) / sfreq

        self._apply_roi_dataset(
            roi_raw=roi_arr,
            roi_names=self.brain_view.roi_names,
            times=times,
            sfreq=sfreq,
            source_label=path.name,
        )
        self.log(f"Загружена матрица ROI '{path.name}' формы {roi_arr.shape}.")

    def _on_clean_clicked(self) -> None:
        target_eeg = self._raw_eeg
        if target_eeg is not None:
            l_freq = float(self.spin_lfreq.value())
            h_freq = float(self.spin_hfreq.value())
            notch = 50.0 if self.chk_notch.isChecked() else None
            use_car = bool(self.chk_car.isChecked())
            use_ica = bool(self.chk_ica.isChecked())

            worker = PreprocessWorker(
                eeg_data=target_eeg,
                l_freq=l_freq,
                h_freq=h_freq,
                notch_freq=notch,
                use_car=use_car,
                use_ica=use_ica,
                parent=self,
            )
            self._start_worker(worker, self._on_eeg_cleaned, "Очистка сигналов ЭЭГ...")
            return

        if self._roi_raw_data is not None:
            self._rebuild_processed_roi()
            self.log("Выполнены сглаживание и нормализация матрицы ROI.")

    def _on_eeg_cleaned(self, payload: dict) -> None:
        self._clean_eeg = payload["eeg_data"]
        self._eeg_signals = payload["signals"]
        self._eeg_ch_names = payload["ch_names"]
        self._eeg_times = payload["times"]
        report: dict = payload["report"]

        self.eeg_plot.set_eeg_data(self._eeg_signals, self._eeg_ch_names, self._eeg_times)
        self._refresh_top_electrodes_list(
            center_time=self._current_time, full_record=self._is_average_mode
        )

        excluded = report.get("excluded_components", [])
        ica_note = (
            f", ICA удалила комп.: {len(excluded)}" if report.get("used_ica") else ""
        )
        self.log(
            f"ЭЭГ очищена (RMS: {report['rms_before']:.2e} → {report['rms_after']:.2e}{ica_note})."
        )

        if self._roi_raw_data is not None:
            self._rebuild_processed_roi()

    def _on_run_sloreta_clicked(self) -> None:
        source_eeg = self._clean_eeg if self._clean_eeg is not None else self._raw_eeg
        if source_eeg is None:
            QMessageBox.warning(self, "Нет данных ЭЭГ", "Сначала загрузите файл ЭЭГ.")
            return

        worker = SloretaWorker(
            eeg_data=source_eeg,
            snr=3.0,
            target_sfreq=DEFAULT_SFREQ,
            chunk_sec=2.0,
            subjects_dir=self._subjects_dir,
            parent=self,
        )
        self._start_worker(worker, self._on_sloreta_finished, "Расчёт sLORETA (68 зон)...")

    def _on_sloreta_finished(self, result: dict) -> None:
        roi_data: np.ndarray = result["roi_data"]
        roi_names: list[str] = result["roi_names"]
        sfreq: float = float(result["sfreq"])
        times = np.arange(roi_data.shape[1], dtype=np.float64) / sfreq

        stem = self._eeg_file_path.stem if self._eeg_file_path is not None else "eeg_recording"
        self._apply_roi_dataset(
            roi_raw=roi_data,
            roi_names=roi_names,
            times=times,
            sfreq=sfreq,
            source_label=f"sLORETA ({stem})",
        )
        self.log(f"Реконструкция sLORETA завершена: матрица {roi_data.shape} ({sfreq:.0f} Гц).")

    def _on_save_roi_clicked(self) -> None:
        if self._roi_data is None:
            QMessageBox.information(self, "Нечего сохранять", "Нет активной матрицы ROI.")
            return

        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = self._eeg_file_path.stem if self._eeg_file_path is not None else "sloreta_68roi"
        default_path = str(out_dir / f"{stem}_sloreta_68roi.npy")

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить матрицу активности ROI (.npy)",
            default_path,
            "NumPy матрицы (*.npy)",
        )
        if not save_path:
            return

        np.save(save_path, self._roi_data)
        self.log(f"Матрица ROI сохранена: {Path(save_path).name}")

    def _on_clear_all_clicked(self) -> None:
        self._reset_scene_to_empty("Все файлы, графики и 3D-сцена очищены.")

    # ------------------------------------------------------------------
    # Обработчики элементов 3D-сцены и списков
    # ------------------------------------------------------------------

    def _on_zscore_toggled(self, _: bool) -> None:
        if self._roi_raw_data is not None:
            self._rebuild_processed_roi()

    def _on_hemispheres_toggled(self, _: bool) -> None:
        self.brain_view.set_hemispheres_visible(
            left=self.chk_lh.isChecked(),
            right=self.chk_rh.isChecked(),
        )

    def _on_cmap_changed(self) -> None:
        cmap_name = str(self.combo_cmap.currentData() or "inferno")
        self.brain_view.set_colormap(cmap_name)

    def _on_threshold_changed(self, value: int) -> None:
        self.lbl_thresh.setText(f"Порог очагов активности: {value}%")
        self.brain_view.set_threshold(value / 100.0)

    def _on_language_toggled(self, checked: bool) -> None:
        self._use_russian_names = bool(checked)
        self.btn_lang.setText(
            "Язык зон: Русский (RU / EN)" if checked else "Язык зон: English (EN / RU)"
        )
        if self._last_displayed_vec is not None:
            self._refresh_top_rois_list(self._last_displayed_vec)

    def _on_top_roi_item_clicked(self, item: QListWidgetItem) -> None:
        roi_idx = item.data(Qt.ItemDataRole.UserRole)
        if roi_idx is None:
            return
        self._selected_roi_idx = int(roi_idx)
        self.brain_view.highlight_roi(self._selected_roi_idx)
        self.eeg_plot.set_selected_roi(self._selected_roi_idx)

    def _on_brain_roi_clicked(self, roi_idx: int, raw_name: str) -> None:
        self._selected_roi_idx = int(roi_idx)
        self.eeg_plot.set_selected_roi(self._selected_roi_idx)
        if self._last_displayed_vec is not None:
            self._refresh_top_rois_list(self._last_displayed_vec)
        ru_name = translate_roi_name(raw_name, include_raw=True)
        self.log(f"Выбрана зона: {ru_name}")

    def closeEvent(self, event) -> None:
        self._play_timer.stop()
        if self._active_worker is not None and self._active_worker.isRunning():
            self._active_worker.requestInterruption()
            self._active_worker.wait(1500)
        self.brain_view.cleanup()
        super().closeEvent(event)