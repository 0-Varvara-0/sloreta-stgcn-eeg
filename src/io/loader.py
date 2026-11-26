from pathlib import Path
import re

import h5py
import mne
import numpy as np


_CH_PREFIX_RE = re.compile(r"^(EEG|POL|CHAN|CH)\s*[-_:]?\s*", re.IGNORECASE)
_CH_SUFFIX_RE = re.compile(r"\s*[-_:]?\s*(REF|AVG|LE|AR|M1|M2|A1|A2)$", re.IGNORECASE)


def _get_standard_montage() -> mne.channels.DigMontage:
    try:
        return mne.channels.make_standard_montage("colin27_1005")
    except ValueError:
        return mne.channels.make_standard_montage("standard_1005")


def _clean_channel_name(raw_name: str) -> str:
    name = raw_name.strip()
    name = _CH_PREFIX_RE.sub("", name)
    name = _CH_SUFFIX_RE.sub("", name)
    return name.strip()


def _apply_montage_safely(inst: mne.io.BaseRaw | mne.BaseEpochs) -> None:
    montage = _get_standard_montage()
    lookup = {ch.lower(): ch for ch in montage.ch_names}

    rename_map: dict[str, str] = {}
    for orig in inst.ch_names:
        cleaned = _clean_channel_name(orig)
        canonical = lookup.get(cleaned.lower(), cleaned)
        if canonical != orig and canonical not in inst.ch_names and canonical not in rename_map.values():
            rename_map[orig] = canonical

    if rename_map:
        inst.rename_channels(rename_map)

    ch_types: dict[str, str] = {}
    for ch in inst.ch_names:
        if ch.lower() in lookup:
            ch_types[ch] = "eeg"
        else:
            upper = ch.upper()
            if "EOG" in upper:
                ch_types[ch] = "eog"
            elif "ECG" in upper or "EKG" in upper:
                ch_types[ch] = "ecg"
            elif "EMG" in upper:
                ch_types[ch] = "emg"
            else:
                ch_types[ch] = "misc"

    inst.set_channel_types(ch_types, verbose=False)
    inst.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)


def _decode_h5_string(obj, h5file: h5py.File) -> str:
    if isinstance(obj, h5py.Reference):
        if not obj:
            return ""
        obj = h5file[obj]
    if isinstance(obj, h5py.Dataset):
        val = obj[()]
    else:
        val = obj

    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore").strip("\x00 ").strip()
    if isinstance(val, str):
        return val.strip("\x00 ").strip()

    arr = np.asarray(val)
    if arr.size == 0:
        return ""
    if arr.dtype.kind in ("U", "S"):
        return "".join(str(x) for x in arr.ravel()).strip("\x00 ").strip()
    if np.issubdtype(arr.dtype, np.integer):
        chars = [chr(int(c)) for c in arr.ravel() if int(c) > 0]
        return "".join(chars).strip()
    return str(val).strip()


def _load_eeglab_v73_hdf5(path: Path) -> mne.io.BaseRaw | mne.BaseEpochs:
    """Резервное чтение EEGLAB .set в формате MATLAB v7.3 (HDF5), минуя сбойные служебные поля."""
    with h5py.File(path, "r") as f:
        eeg_grp = f["EEG"] if "EEG" in f else f

        srate = float(np.asarray(eeg_grp["srate"][()]).ravel()[0])
        if srate <= 0:
            srate = 250.0

        nbchan = int(np.asarray(eeg_grp["nbchan"][()]).ravel()[0]) if "nbchan" in eeg_grp else 0
        pnts = int(np.asarray(eeg_grp["pnts"][()]).ravel()[0]) if "pnts" in eeg_grp else 0
        trials = int(np.asarray(eeg_grp["trials"][()]).ravel()[0]) if "trials" in eeg_grp else 1

        ch_names: list[str] = []
        if "chanlocs" in eeg_grp and isinstance(eeg_grp["chanlocs"], h5py.Group):
            chanlocs = eeg_grp["chanlocs"]
            if "labels" in chanlocs:
                labels_ds = chanlocs["labels"]
                if isinstance(labels_ds, h5py.Dataset):
                    raw_refs = np.asarray(labels_ds[()]).ravel()
                    for ref in raw_refs:
                        lbl = _decode_h5_string(ref, f)
                        if lbl:
                            ch_names.append(lbl)

        data_ds = eeg_grp["data"]
        if data_ds.dtype.kind in ("U", "S") or (
            np.issubdtype(data_ds.dtype, np.integer) and data_ds.ndim <= 2 and min(data_ds.shape) == 1
        ):
            fdt_name = _decode_h5_string(data_ds, f)
            fdt_path = path.parent / Path(fdt_name).name
            if not fdt_path.exists():
                fdt_path = path.with_suffix(".fdt")
            if not fdt_path.exists():
                raise FileNotFoundError(f"Не найден парный файл данных ЭЭГ: {fdt_path}")
            raw_flat = np.fromfile(fdt_path, dtype="<f4").astype(np.float64)
            if trials > 1 and nbchan > 0 and pnts > 0:
                arr = raw_flat.reshape((trials, pnts, nbchan)).transpose(0, 2, 1)
            elif nbchan > 0:
                arr = raw_flat.reshape((-1, nbchan)).T
            else:
                raise ValueError("Не удалось определить число каналов для .fdt файла.")
        else:
            arr = np.asarray(data_ds[()], dtype=np.float64)
            if arr.ndim == 2:
                if nbchan > 0 and arr.shape[1] == nbchan and arr.shape[0] != nbchan:
                    arr = arr.T
                elif arr.shape[0] > arr.shape[1]:
                    arr = arr.T
            elif arr.ndim == 3:
                if nbchan > 0 and arr.shape[2] == nbchan:
                    arr = arr.transpose(0, 2, 1)

        n_ch = arr.shape[-2]
        if len(ch_names) != n_ch:
            ch_names = [f"Ch{i + 1}" for i in range(n_ch)]

        # EEGLAB хранит амплитуды в мкВ, MNE — в Вольтах
        if float(np.nanpercentile(np.abs(arr), 99)) > 1e-2:
            arr = arr * 1e-6

        info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types="eeg")
        if arr.ndim == 3:
            inst: mne.io.BaseRaw | mne.BaseEpochs = mne.EpochsArray(arr, info, verbose=False)
        else:
            inst = mne.io.RawArray(arr, info, verbose=False)

        _apply_montage_safely(inst)
        return inst


def load_eeg_set(path: Path) -> mne.io.BaseRaw | mne.BaseEpochs:
    try:
        data: mne.io.BaseRaw | mne.BaseEpochs = mne.io.read_raw_eeglab(
            path, preload=True, verbose=False
        )
    except TypeError:
        data = mne.io.read_epochs_eeglab(path, verbose=False)
    except (KeyError, OSError, NotImplementedError, ValueError):
        return _load_eeglab_v73_hdf5(path)

    _apply_montage_safely(data)
    return data


def load_eeg_file(file_path: str | Path) -> mne.io.BaseRaw | mne.BaseEpochs:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    suffix = path.suffix.lower()
    if suffix == ".set":
        return load_eeg_set(path)
    if suffix == ".edf":
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        _apply_montage_safely(raw)
        return raw
    if suffix == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)
        _apply_montage_safely(raw)
        return raw
    if suffix == ".fif":
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
        _apply_montage_safely(raw)
        return raw
    if suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
        _apply_montage_safely(raw)
        return raw

    raw = mne.io.read_raw(path, preload=True, verbose=False)
    _apply_montage_safely(raw)
    return raw


def load_roi_npy(file_path: str | Path, expected_rois: int = 68) -> np.ndarray:
    """Загружает и проверяет 2D-матрицу активности корковых зон формы (68, T)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл ROI не найден: {path}")

    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Ожидалась 2D-матрица (68, T), получена форма {arr.shape}.")

    if arr.shape[0] != expected_rois and arr.shape[1] == expected_rois:
        arr = arr.T

    if arr.shape[0] != expected_rois:
        raise ValueError(
            f"Неверное число корковых зон: ожидалось {expected_rois}, получено {arr.shape[0]}."
        )

    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)