import mne
import numpy as np
from scipy.ndimage import uniform_filter1d


def clean_roi_matrix(
    roi_data: np.ndarray,
    smooth_window: int = 5,
    apply_zscore: bool = True,
) -> np.ndarray:
    """Сглаживает матрицу активности ROI (68, T) и опционально применяет Z-нормализацию."""
    cleaned = roi_data.astype(np.float32, copy=True)
    if smooth_window > 1 and cleaned.shape[1] > 1:
        cleaned = uniform_filter1d(cleaned, size=smooth_window, axis=1)

    if apply_zscore and cleaned.shape[1] > 1:
        mean = cleaned.mean(axis=1, keepdims=True)
        std = cleaned.std(axis=1, keepdims=True) + 1e-8
        cleaned = np.clip((cleaned - mean) / std, 0.0, None)

    return cleaned.astype(np.float32)


def extract_eeg_preview(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    max_channels: int = 10,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Извлекает до max_channels ЭЭГ-каналов для 2D-графика.
    Для эпох (BaseEpochs) склеивает их последовательно по оси времени,
    синхронно с форматом матрицы ROI из sLORETA.
    """
    picks = mne.pick_types(data.info, eeg=True)
    if len(picks) == 0:
        picks = np.arange(len(data.ch_names))
    picks = picks[:max_channels]
    ch_names = [data.ch_names[i] for i in picks]

    if isinstance(data, mne.io.BaseRaw):
        signals = data.get_data(picks=picks).astype(np.float32)
        times = data.times.astype(np.float64)
    else:
        # shape: (n_epochs, n_ch, n_times) -> (n_ch, n_epochs * n_times)
        ep_data = data.get_data(picks=picks)
        n_ep, n_ch, n_t = ep_data.shape
        signals = np.transpose(ep_data, (1, 0, 2)).reshape(n_ch, n_ep * n_t).astype(np.float32)
        sfreq = float(data.info["sfreq"]) or 1.0
        times = np.arange(n_ep * n_t, dtype=np.float64) / sfreq

    return signals, ch_names, times


def select_top_roi_traces(
    roi_data: np.ndarray,
    top_k: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Выбирает top_k самых динамичных корковых зон по стандартному отклонению."""
    stds = np.std(roi_data, axis=1)
    top_idx = np.argsort(stds)[::-1][:top_k]
    signals = roi_data[top_idx].astype(np.float32)
    times = np.arange(roi_data.shape[1], dtype=np.float64)
    return top_idx, signals, times


def prepare_traces_and_detect_spikes(
    signals: np.ndarray,
    times: np.ndarray,
    n_roi_frames: int | None = None,
    max_plot_points: int = 5000,
    z_threshold: float = 3.2,
) -> dict:
    """
    Даунсэмплит сигналы для быстрой отрисовки, нормализует каналы по z-score
    и находит интервалы резких всплесков / эпилептиформных разрядов.
    """
    n_t = signals.shape[1]
    duration = float(times[-1] - times[0]) if len(times) > 1 else 1.0

    step = max(1, n_t // max_plot_points)
    t_vis = times[::step]
    sig_vis = signals[:, ::step]

    med = np.median(sig_vis, axis=1, keepdims=True)
    std = np.std(sig_vis, axis=1, keepdims=True) + 1e-8
    z_vis = (sig_vis - med) / std

    max_abs_z = np.max(np.abs(z_vis), axis=0)
    above = np.where(max_abs_z > z_threshold)[0]

    spike_regions: list[tuple[float, float]] = []
    spike_frames: list[int] = []

    if len(above) > 0:
        min_gap = max(3, len(t_vis) // 120)
        splits = np.where(np.diff(above) > min_gap)[0] + 1
        groups = np.split(above, splits)
        target_frames = n_roi_frames if (n_roi_frames and n_roi_frames > 0) else len(t_vis)

        for grp in groups:
            t_start = float(t_vis[grp[0]])
            t_end = float(t_vis[grp[-1]])
            min_width = duration * 0.008
            if t_end - t_start < min_width:
                t_end = t_start + min_width
            spike_regions.append((t_start, t_end))

            peak_local = grp[int(np.argmax(max_abs_z[grp]))]
            peak_t = float(t_vis[peak_local])
            frac = (peak_t - float(times[0])) / max(duration, 1e-6)
            frame_idx = int(
                np.clip(round(frac * (target_frames - 1)), 0, max(0, target_frames - 1))
            )
            spike_frames.append(frame_idx)

    return {
        "times_vis": t_vis,
        "z_vis": z_vis,
        "duration": duration,
        "spike_regions": spike_regions,
        "spike_frames": spike_frames,
    }