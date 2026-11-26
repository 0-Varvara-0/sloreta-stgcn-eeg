import mne
import numpy as np


def preprocess_eeg(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    l_freq: float | None = 1.0,
    h_freq: float | None = 40.0,
    notch_freq: float | None = 50.0,
    use_car: bool = True,
    use_ica: bool = True,
    artifact_threshold: float = 0.80,
) -> tuple[mne.io.BaseRaw | mne.BaseEpochs, dict]:
    """
    Полный или облегчённый цикл предобработки ЭЭГ:
    1. Режекторная (Notch 50 Гц) и полосовая (1–40 Гц) фильтрация.
    2. Общий средний референс (CAR) — обязателен для ICLabel и sLORETA.
    3. Опционально: декомпозиция ICA (Extended Infomax) + удаление артефактов через ICLabel.
    """
    if l_freq is not None and h_freq is not None and l_freq >= h_freq:
        raise ValueError(
            f"Нижняя частота фильтра ({l_freq} Гц) должна быть меньше верхней ({h_freq} Гц)."
        )

    clean_data = data.copy()
    rms_before = float(np.sqrt(np.mean(clean_data.get_data() ** 2)))

    # 1. Фильтрация
    nyquist = clean_data.info["sfreq"] / 2.0
    if notch_freq and 0 < notch_freq < nyquist:
        if isinstance(clean_data, mne.io.BaseRaw):
            clean_data.notch_filter(freqs=notch_freq, verbose=False)

    if l_freq is not None or h_freq is not None:
        clean_data.filter(
            l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False
        )

    # 2. Общий средний референс (CAR обязателен, если включён ICA + ICLabel)
    effective_car = bool(use_car or use_ica)
    if effective_car:
        clean_data.set_eeg_reference("average", projection=False, verbose=False)

    # 3. Опциональная очистка артефактов через ICA (Extended Infomax) + ICLabel
    excluded_details: list[dict] = []
    if use_ica:
        n_channels = len(clean_data.ch_names)
        if n_channels < 3:
            raise ValueError(
                f"Для запуска ICA нужно минимум 3 ЭЭГ-канала (сейчас: {n_channels})."
            )

        if clean_data.get_montage() is None:
            try:
                montage = mne.channels.make_standard_montage("colin27_1005")
            except ValueError:
                montage = mne.channels.make_standard_montage("standard_1005")
            clean_data.set_montage(montage, on_missing="ignore", verbose=False)

        # Ленивый импорт, чтобы не тормозить запуск интерфейса при быстрой фильтрации
        from mne.preprocessing import ICA
        from mne_icalabel import label_components

        n_components = min(20, n_channels - 1)
        ica = ICA(
            n_components=n_components,
            method="infomax",
            fit_params=dict(extended=True),
            random_state=42,
            max_iter="auto",
            verbose=False,
        )
        ica.fit(clean_data, verbose=False)

        ic_labels = label_components(clean_data, ica, method="iclabel")
        labels = ic_labels["labels"]
        probs = ic_labels["y_pred_proba"]

        artifact_types = ("eye blink", "muscle artifact", "heart beat", "line noise")
        exclude_idx: list[int] = []
        for idx, (label, prob) in enumerate(zip(labels, probs)):
            if label in artifact_types and prob >= artifact_threshold:
                exclude_idx.append(idx)
                excluded_details.append(
                    {
                        "component": idx,
                        "type": label,
                        "confidence": round(float(prob), 3),
                    }
                )

        ica.exclude = exclude_idx
        ica.apply(clean_data, verbose=False)

    rms_after = float(np.sqrt(np.mean(clean_data.get_data() ** 2)))
    is_raw = isinstance(clean_data, mne.io.BaseRaw)
    duration_sec = (
        round(float(clean_data.times[-1]), 2)
        if is_raw
        else round(
            float(len(clean_data) * (clean_data.times[-1] - clean_data.times[0])), 2
        )
    )

    report = {
        "data_type": "Continuous (Raw)" if is_raw else f"Epochs ({len(clean_data)} шт.)",
        "n_channels": len(clean_data.ch_names),
        "channels": list(clean_data.ch_names),
        "sfreq": float(clean_data.info["sfreq"]),
        "duration_sec": duration_sec,
        "used_car": effective_car,
        "used_ica": bool(use_ica),
        "rms_before": rms_before,
        "rms_after": rms_after,
        "excluded_components": excluded_details,
    }
    return clean_data, report