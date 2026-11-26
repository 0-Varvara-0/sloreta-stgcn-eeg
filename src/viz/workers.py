from __future__ import annotations

from pathlib import Path
import traceback

import mne
from PySide6.QtCore import QThread, Signal

from src.io.loader import load_eeg_file, load_roi_npy
from src.preprocessing.pipeline import preprocess_eeg
from src.preprocessing.signal_utils import extract_eeg_preview
from src.source_imaging.sloreta import compute_sloreta_rois
from src.viz.config import EXPECTED_ROI_COUNT


class _BaseWorker(QThread):
    """Базовый фоновый поток с единым набором сигналов прогресса, результата и ошибки."""

    progress = Signal(str)
    finished_ok = Signal(object)
    error = Signal(str)

    def _run_task(self) -> object:
        raise NotImplementedError

    def run(self) -> None:
        try:
            result = self._run_task()
            if not self.isInterruptionRequested():
                self.finished_ok.emit(result)
        except Exception as exc:
            details = traceback.format_exc()
            print(details)
            if not self.isInterruptionRequested():
                self.error.emit(f"{type(exc).__name__}: {exc}")


class LoadEEGWorker(_BaseWorker):
    """Фоновая загрузка ЭЭГ-файла и извлечение каналов для 2D-графика."""

    def __init__(
        self,
        file_path: str | Path,
        max_preview_channels: int = 10,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.file_path = Path(file_path)
        self.max_preview_channels = max_preview_channels

    def _run_task(self) -> dict:
        self.progress.emit(f"Загрузка ЭЭГ: {self.file_path.name}...")
        eeg_data = load_eeg_file(self.file_path)

        self.progress.emit("Подготовка каналов для 2D-графика...")
        signals, ch_names, times = extract_eeg_preview(
            eeg_data, max_channels=self.max_preview_channels
        )
        is_raw = isinstance(eeg_data, mne.io.BaseRaw)
        duration_sec = float(times[-1] - times[0]) if len(times) > 1 else 0.0

        return {
            "path": self.file_path,
            "eeg_data": eeg_data,
            "signals": signals,
            "ch_names": ch_names,
            "times": times,
            "sfreq": float(eeg_data.info["sfreq"]),
            "duration_sec": duration_sec,
            "data_type": "Continuous (Raw)" if is_raw else f"Epochs ({len(eeg_data)} шт.)",
        }


class LoadROIWorker(_BaseWorker):
    """Фоновая загрузка матрицы корковых источников ROI (.npy)."""

    def __init__(
        self,
        file_path: str | Path,
        expected_rois: int = EXPECTED_ROI_COUNT,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.file_path = Path(file_path)
        self.expected_rois = expected_rois

    def _run_task(self) -> dict:
        self.progress.emit(f"Загрузка карты активности: {self.file_path.name}...")
        roi_data = load_roi_npy(self.file_path, expected_rois=self.expected_rois)
        return {
            "path": self.file_path,
            "roi_data": roi_data,
        }


class PreprocessWorker(_BaseWorker):
    """Фоновая предобработка ЭЭГ: фильтры, CAR и опционально ICA + ICLabel."""

    def __init__(
        self,
        eeg_data: mne.io.BaseRaw | mne.BaseEpochs,
        l_freq: float | None = 1.0,
        h_freq: float | None = 40.0,
        notch_freq: float | None = 50.0,
        use_car: bool = True,
        use_ica: bool = True,
        artifact_threshold: float = 0.80,
        max_preview_channels: int = 10,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.eeg_data = eeg_data
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.use_car = use_car
        self.use_ica = use_ica
        self.artifact_threshold = artifact_threshold
        self.max_preview_channels = max_preview_channels

    def _run_task(self) -> dict:
        mode_desc = "фильтры + CAR + ICA/ICLabel" if self.use_ica else "быстрая фильтрация"
        self.progress.emit(f"Предобработка ЭЭГ ({mode_desc})...")

        clean_data, report = preprocess_eeg(
            self.eeg_data,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            notch_freq=self.notch_freq,
            use_car=self.use_car,
            use_ica=self.use_ica,
            artifact_threshold=self.artifact_threshold,
        )

        self.progress.emit("Обновление каналов для отображения...")
        signals, ch_names, times = extract_eeg_preview(
            clean_data, max_channels=self.max_preview_channels
        )

        return {
            "eeg_data": clean_data,
            "report": report,
            "signals": signals,
            "ch_names": ch_names,
            "times": times,
        }


class SloretaWorker(_BaseWorker):
    """Фоновая реконструкция корковых источников sLORETA (68 зон fsaverage)."""

    def __init__(
        self,
        eeg_data: mne.io.BaseRaw | mne.BaseEpochs,
        snr: float = 3.0,
        target_sfreq: float = 128.0,
        chunk_sec: float = 2.0,
        subjects_dir: Path | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.eeg_data = eeg_data
        self.snr = snr
        self.target_sfreq = target_sfreq
        self.chunk_sec = chunk_sec
        self.subjects_dir = subjects_dir

    def _run_task(self) -> dict:
        self.progress.emit(
            f"Расчёт sLORETA (SNR={self.snr:.1f}, {self.target_sfreq:.0f} Гц)... "
            "Первый запуск может занять 20–60 сек."
        )
        result = compute_sloreta_rois(
            self.eeg_data,
            subjects_dir=self.subjects_dir,
            snr=self.snr,
            target_sfreq=self.target_sfreq,
            chunk_sec=self.chunk_sec,
        )
        return result