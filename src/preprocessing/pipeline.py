import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

# Полный цикл автоматической предобработки ЭЭГ:
#1. Полосовая (1-40 Гц) и режекторная (50 Гц) фильтрация.
#2. Переход к общему среднему референсу (CAR) - требование ICLabel и sLORETA.
#3. Декомпозиция ICA (Extended Infomax) и удаление артефактов через ICLabel.
def preprocess_eeg(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch_freq: float = 50.0,
    artifact_threshold: float = 0.80,
) -> tuple[mne.io.BaseRaw | mne.BaseEpochs, dict]:
    
    clean_data = data.copy()

    # 1. Фильтрация
    if notch_freq and notch_freq < clean_data.info["sfreq"] / 2:
        if isinstance(clean_data, mne.io.BaseRaw):
            clean_data.notch_filter(freqs=notch_freq, verbose=False)
    clean_data.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)

    # 2. Общий средний референс (CAR)
    clean_data.set_eeg_reference("average", projection=False, verbose=False)

    # 3. ICA (Extended Infomax обязателен для корректной работы ICLabel)
    n_components = min(20, len(clean_data.ch_names) - 1)
    ica = ICA(
        n_components=n_components,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=42,
        max_iter="auto",
        verbose=False,
    )
    ica.fit(clean_data, verbose=False)

    # 4. Нейросетевая классификация компонент (ICLabel)
    ic_labels = label_components(clean_data, ica, method="iclabel")
    labels = ic_labels["labels"]
    probs = ic_labels["y_pred_proba"]

    exclude_idx = []
    excluded_details = []
    for idx, (label, prob) in enumerate(zip(labels, probs)):
        if label in ("eye blink", "muscle artifact", "heart beat", "line noise") and prob >= artifact_threshold:
            exclude_idx.append(idx)
            excluded_details.append({
                "component": idx,
                "type": label,
                "confidence": round(float(prob), 3),
            })

    ica.exclude = exclude_idx
    ica.apply(clean_data, verbose=False)

    is_raw = isinstance(clean_data, mne.io.BaseRaw)
    report = {
        "data_type": "Continuous (Raw)" if is_raw else f"Epochs ({len(clean_data)} шт.)",
        "n_channels": len(clean_data.ch_names),
        "channels": clean_data.ch_names,
        "sfreq": clean_data.info["sfreq"],
        "duration_sec": round(clean_data.times[-1], 2) if is_raw else round(len(clean_data) * (clean_data.times[-1] - clean_data.times[0]), 2),
        "excluded_components": excluded_details,
    }
    return clean_data, report