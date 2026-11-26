from pathlib import Path
import gc
import numpy as np
import mne
from mne.minimum_norm import (
    make_inverse_operator,
    prepare_inverse_operator,
    apply_inverse_raw,
    apply_inverse_epochs,
)


def compute_sloreta_rois(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    subjects_dir: Path | None = None,
    snr: float = 3.0,
    lambda2: float | None = None,
    target_sfreq: float = 128.0,
    chunk_sec: float = 2.0,
) -> dict:
    
    # Экономная по памяти реконструкция корковых источников методом sLORETA (fsaverage)
    # с агрегацией по 68 зонам коры

    if subjects_dir is None:
        fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
        subjects_dir = fs_dir.parent
    else:
        subjects_dir = Path(subjects_dir)
        fs_dir = subjects_dir / "fsaverage"

    subject = "fsaverage"
    inst = data.copy()

    # 1. Понижаем частоту до 128 Гц (после фильтра 1-40 Гц информация сохраняется на 100%)
    if inst.info["sfreq"] > target_sfreq:
        inst.resample(target_sfreq, verbose=False)

    if inst.get_montage() is None:
        inst.set_montage(
            mne.channels.make_standard_montage("standard_1005"),
            on_missing="ignore",
            verbose=False,
        )
    inst.set_eeg_reference("average", projection=True, verbose=False)
    inst.apply_proj(verbose=False)

    # 2. Шаблонная модель головы fsaverage
    src = mne.read_source_spaces(fs_dir / "bem" / "fsaverage-ico-5-src.fif", verbose=False)
    bem = mne.read_bem_solution(fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif", verbose=False)

    # 3. Прямая задача (Forward solution)
    fwd = mne.make_forward_solution(
        inst.info,
        trans="fsaverage",
        src=src,
        bem=bem,
        eeg=True,
        mindist=5.0,
        n_jobs=1,
        verbose=False,
    )

    # 4. Ковариация шума и обратный оператор
    is_raw = isinstance(inst, mne.io.BaseRaw)
    noise_cov = (
        mne.compute_raw_covariance(inst, verbose=False)
        if is_raw
        else mne.compute_covariance(inst, verbose=False)
    )
    inverse_operator = make_inverse_operator(
        inst.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=False
    )

    if lambda2 is None:
        lambda2 = 1.0 / (snr ** 2)

    # 5. 68 корковых областей атласа Desikan-Killiany (aparc)
    labels = mne.read_labels_from_annot(
        subject=subject, parc="aparc", subjects_dir=subjects_dir, verbose=False
    )
    cortical_labels = sorted(
        [lbl for lbl in labels if not lbl.name.startswith(("unknown", "corpuscallosum"))],
        key=lambda x: x.name,
    )

    # 6. Поблочный расчёт sLORETA, чтобы не переполнять RAM
    if is_raw:
        inv_prepared = prepare_inverse_operator(
            inverse_operator, nave=1, lambda2=lambda2, method="sLORETA", verbose=False
        )
        chunk_samples = max(1, int(chunk_sec * inst.info["sfreq"]))
        roi_chunks = []

        for start in range(0, inst.n_times, chunk_samples):
            stop = min(start + chunk_samples, inst.n_times)
            stc_chunk = apply_inverse_raw(
                inst,
                inv_prepared,
                lambda2=lambda2,
                method="sLORETA",
                start=start,
                stop=stop,
                pick_ori="normal",
                prepared=True,
                verbose=False,
            )
            roi_chunk = mne.extract_label_time_course(
                stc_chunk, cortical_labels, fwd["src"], mode="mean_flip", verbose=False
            )
            roi_chunks.append(roi_chunk.astype(np.float32))
            del stc_chunk
            gc.collect()

        roi_data = np.concatenate(roi_chunks, axis=1)
    else:
        # Для эпох используем генератор — обрабатывает по одной эпохе за раз
        stcs = apply_inverse_epochs(
            inst,
            inverse_operator,
            lambda2=lambda2,
            method="sLORETA",
            pick_ori="normal",
            return_generator=True,
            verbose=False,
        )
        roi_list = mne.extract_label_time_course(
            stcs, cortical_labels, fwd["src"], mode="mean_flip", verbose=False
        )
        roi_data = np.concatenate([r.astype(np.float32) for r in roi_list], axis=1)

    return {
        "roi_data": roi_data,
        "roi_names": [lbl.name for lbl in cortical_labels],
        "sfreq": float(inst.info["sfreq"]),
    }