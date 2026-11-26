from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne
from src.source_imaging.sloreta import compute_sloreta_rois


def main():
    proc_dir = Path("data/processed")
    clean_files = sorted(list(proc_dir.glob("*_clean_raw.fif")) + list(proc_dir.glob("*_clean-epo.fif")))
    if not clean_files:
        raise FileNotFoundError("В data/processed/ нет очищенных .fif файлов. Сначала запустите run_preprocessing_demo.py.")

    target_fif = clean_files[0]
    print(f"1. Загрузка очищенного файла: {target_fif.name}")
    if target_fif.name.endswith("_clean_raw.fif"):
        data = mne.io.read_raw_fif(target_fif, preload=True, verbose=False)
        # Для быстрого прототипа возьмём первые 30 секунд (потом уберём ограничение)
        if data.times[-1] > 30.0:
            data.crop(tmin=0.0, tmax=30.0)
    else:
        data = mne.read_epochs(target_fif, preload=True, verbose=False)

    print("2. Расчёт источников методом sLORETA и агрегация по 68 зонам коры (fsaverage)...")
    res = compute_sloreta_rois(data)
    roi_data = res["roi_data"]
    roi_names = res["roi_names"]

    print(f"   Получена матрица активности ROI: {roi_data.shape} (зон × временных отсчётов)")
    print(f"   Число зон коры: {len(roi_names)} (пример: {roi_names[:4]})")

    # Сохраняем матрицу активности 68 зон для следующего шага (динамическая связность)
    out_npy = proc_dir / f"{target_fif.stem}_sloreta_68roi.npy"
    np.save(out_npy, roi_data)

    # Строим топ-15 наиболее активных зон по средней амплитуде и тепловую карту
    mean_per_roi = np.mean(np.abs(roi_data), axis=1)
    top_idx = np.argsort(mean_per_roi)[::-1][:15]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Реконструкция источников sLORETA (68 зон коры): {target_fif.stem}", fontsize=13)

    top_names = [roi_names[i] for i in top_idx][::-1]
    top_vals = mean_per_roi[top_idx][::-1]
    ax1.barh(top_names, top_vals, color="#2c7bb6")
    ax1.set_xlabel("Средняя амплитуда активности источника (усл. ед.)")
    ax1.set_title("Топ-15 зон коры по средней активности")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.4)

    # Показываем динамику активности топ-10 зон во времени
    n_show = min(10, len(top_idx))
    show_idx = top_idx[:n_show]
    im = ax2.imshow(np.abs(roi_data[show_idx, :]), aspect="auto", cmap="viridis")
    ax2.set_yticks(range(n_show))
    ax2.set_yticklabels([roi_names[i] for i in show_idx], fontsize=9)
    ax2.set_xlabel("Временные отсчёты")
    ax2.set_title("Временная динамика топ-10 зон")
    plt.colorbar(im, ax=ax2, label="Активность")

    plt.tight_layout()
    plot_path = proc_dir / f"{target_fif.stem}_sloreta_rois.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\n3. Готово! Сохранены:\n   - Матрица ROI: {out_npy}\n   - График зон:  {plot_path}")


if __name__ == "__main__":
    main()
