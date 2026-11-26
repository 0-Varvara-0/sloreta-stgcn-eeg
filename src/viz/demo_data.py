from __future__ import annotations

import numpy as np

from src.viz.config import DEFAULT_SFREQ, get_ordered_roi_names


def generate_demo_dataset(
    duration_sec: float = 10.0,
    sfreq: float = DEFAULT_SFREQ,
    seed: int = 42,
) -> dict:
    """
    Генерирует синхронизированный демо-набор:
    - roi_data: (68, T) с фокальными эпилептиформными разрядами в левой височной доле
      и вторичным распространением на соседние корковые зоны;
    - eeg_signals: (10, T) с фоновым альфа/тета-ритмом и синхронными пик-волновыми всплесками.
    """
    rng = np.random.default_rng(seed)
    roi_names = get_ordered_roi_names()
    n_rois = len(roi_names)
    n_times = max(128, int(round(duration_sec * sfreq)))
    times = np.arange(n_times, dtype=np.float64) / sfreq

    # 1. Базовая фоновая активность коры (слабый шум + медленная модуляция)
    roi_data = rng.normal(loc=0.18, scale=0.06, size=(n_rois, n_times)).astype(np.float32)
    alpha_mod = 0.08 * (1.0 + np.sin(2.0 * np.pi * 1.2 * times))[None, :]
    roi_data = np.clip(roi_data + alpha_mod.astype(np.float32), 0.0, None)

    # Очаг в левой височной доле + зоны вторичного распространения
    primary_focus = {
        "superiortemporal-lh": 1.00,
        "middletemporal-lh": 0.92,
        "temporalpole-lh": 0.85,
        "inferiortemporal-lh": 0.78,
        "insula-lh": 0.72,
    }
    spread_zones = {
        "supramarginal-lh": 0.58,
        "parsopercularis-lh": 0.52,
        "parahippocampal-lh": 0.55,
        "entorhinal-lh": 0.60,
        "bankssts-lh": 0.65,
    }

    name_to_idx = {name: idx for idx, name in enumerate(roi_names)}

    # Временные центры трёх разрядов (в секундах)
    burst_centers = [duration_sec * 0.24, duration_sec * 0.56, duration_sec * 0.82]
    burst_envelope = np.zeros(n_times, dtype=np.float64)
    spread_envelope = np.zeros(n_times, dtype=np.float64)

    for center in burst_centers:
        # Основной резкий всплеск в очаге
        burst_envelope += np.exp(-0.5 * ((times - center) / 0.14) ** 2)
        # Небольшая задержка распространения на соседние зоны (~90 мс)
        spread_envelope += np.exp(-0.5 * ((times - (center + 0.09)) / 0.18) ** 2)

    for zone, weight in primary_focus.items():
        idx = name_to_idx[zone]
        roi_data[idx] += (weight * 3.4 * burst_envelope).astype(np.float32)

    for zone, weight in spread_zones.items():
        idx = name_to_idx[zone]
        roi_data[idx] += (weight * 2.4 * spread_envelope).astype(np.float32)

    # 2. Синхронизированные 10 каналов ЭЭГ
    eeg_ch_names = ["T7", "F7", "P7", "C3", "F3", "Fp1", "O1", "T8", "C4", "Fz"]
    ch_spike_weights = {
        "T7": 1.00,
        "F7": 0.82,
        "P7": 0.70,
        "C3": 0.48,
        "F3": 0.40,
        "Fp1": 0.25,
        "O1": 0.18,
        "T8": 0.14,
        "C4": 0.12,
        "Fz": 0.20,
    }

    # Морфология пик-волны для ЭЭГ
    eeg_spike_wave = np.zeros(n_times, dtype=np.float64)
    for center in burst_centers:
        sharp_spike = np.exp(-0.5 * ((times - center) / 0.028) ** 2)
        slow_wave = -0.55 * np.exp(-0.5 * ((times - (center + 0.11)) / 0.085) ** 2)
        eeg_spike_wave += sharp_spike + slow_wave

    eeg_signals = np.zeros((len(eeg_ch_names), n_times), dtype=np.float32)
    for i, ch in enumerate(eeg_ch_names):
        phase_a = rng.uniform(0, 2 * np.pi)
        phase_t = rng.uniform(0, 2 * np.pi)
        alpha = 12e-6 * np.sin(2.0 * np.pi * (9.5 + 0.15 * i) * times + phase_a)
        theta = 6e-6 * np.sin(2.0 * np.pi * (5.2 + 0.1 * i) * times + phase_t)
        noise = rng.normal(0.0, 4.5e-6, size=n_times)
        spike = ch_spike_weights[ch] * 110e-6 * eeg_spike_wave
        eeg_signals[i] = (alpha + theta + noise + spike).astype(np.float32)

    return {
        "roi_data": roi_data,
        "roi_names": roi_names,
        "eeg_signals": eeg_signals,
        "eeg_ch_names": eeg_ch_names,
        "times": times,
        "sfreq": float(sfreq),
        "description": "Демо-запись: фокальные разряды в левой височной доле (68 ROI, 10 каналов ЭЭГ)",
    }