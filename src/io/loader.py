from pathlib import Path
import mne

# Загружает файл EEGLAB (.set) — автоматически определяет, непрерывная это запись
#или нарезанные эпохи, чистит имена каналов и привязывает стандартный 3D-монтаж электродов для sLORETA и ICLabel.
def load_eeg_set(file_path: str | Path) -> mne.io.BaseRaw | mne.BaseEpochs:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    try:
        data = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    except TypeError:
        # Если в .set файле лежат уже нарезанные эпохи
        data = mne.io.read_epochs_eeglab(path, verbose=False)

    # Очищаем префиксы вроде 'EEG ' или '-Ref', если они остались из EEGLAB
    rename_map = {}
    for ch in data.ch_names:
        clean_name = ch.replace("EEG ", "").replace("-Ref", "").replace("-REF", "").strip()
        if clean_name != ch:
            rename_map[ch] = clean_name
    if rename_map:
        data.rename_channels(rename_map)

    # Привязываем 3D-координаты электродов
    montage = mne.channels.make_standard_montage("standard_1005")
    data.set_montage(montage, on_missing="ignore", verbose=False)

    # Оставляем только ЭЭГ-каналы с известными координатами
    data.pick("eeg")
    return data