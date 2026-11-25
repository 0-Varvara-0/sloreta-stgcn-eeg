from pathlib import Path
import mne
import matplotlib.pyplot as plt
from src.io.loader import load_eeg_set
from src.preprocessing.pipeline import preprocess_eeg


def main():
    raw_dir = Path("data/raw")
    set_files = sorted(raw_dir.glob("*.set"))
    if not set_files:
        raise FileNotFoundError("В папке data/raw/ нет ни одного .set файла! Скопируйте файл туда.")

    target_file = set_files[0]
    print(f"1. Загрузка файла: {target_file.name}")
    data = load_eeg_set(target_file)
    print(f"   Тип данных: {'Непрерывная запись' if isinstance(data, mne.io.BaseRaw) else f'Эпохи ({len(data)} шт.)'}")
    print(f"   Каналов ({len(data.ch_names)}): {data.ch_names}")
    print(f"   Частота дискретизации: {data.info['sfreq']} Гц")

    print("\n2. Запуск фильтрации и автоочистки (ICA + ICLabel)...")
    clean_data, report = preprocess_eeg(data)
    print(f"   Удалено артефактных компонент: {len(report['excluded_components'])}")
    for comp in report["excluded_components"]:
        print(f"   - IC #{comp['component']}: {comp['type']} (уверенность {comp['confidence']*100:.1f}%)")

    # Сохраняем очищенный файл для Этапа 3 (sLORETA)
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(clean_data, mne.io.BaseRaw):
        out_file = out_dir / f"{target_file.stem}_clean_raw.fif"
    else:
        out_file = out_dir / f"{target_file.stem}_clean-epo.fif"
    clean_data.save(out_file, overwrite=True, verbose=False)

    # Строим сравнительный график До / После
    preferred = ["Fp1", "Fp2", "F3", "F4", "Cz", "Pz", "O1", "O2"]
    channels_to_plot = [ch for ch in preferred if ch in data.ch_names][:4]
    if not channels_to_plot:
        channels_to_plot = data.ch_names[:4]

    fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(12, 7), sharex=True)
    if len(channels_to_plot) == 1:
        axes = [axes]
    fig.suptitle(f"Очистка ЭЭГ (ICA + ICLabel): {target_file.name}", fontsize=13)

    if isinstance(data, mne.io.BaseRaw):
        sfreq = int(data.info["sfreq"])
        s_idx = min(int(5 * sfreq), max(0, data.n_times - int(10 * sfreq)))
        e_idx = min(s_idx + int(10 * sfreq), data.n_times)
        times = data.times[s_idx:e_idx]
        for ax, ch in zip(axes, channels_to_plot):
            orig = data.get_data(picks=[ch])[0, s_idx:e_idx] * 1e6
            cln = clean_data.get_data(picks=[ch])[0, s_idx:e_idx] * 1e6
            ax.plot(times, orig, color="#d9534f", alpha=0.65, linewidth=1.0, label="Исходный сигнал")
            ax.plot(times, cln, color="#0275d8", linewidth=1.2, label="После очистки (CAR + ICA)")
            ax.set_ylabel(f"{ch}\n(мкВ)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper right", fontsize=8)
    else:
        # Если в .set лежат эпохи, берём первую эпоху
        times = data.times
        for ax, ch in zip(axes, channels_to_plot):
            orig = data.get_data(picks=[ch])[0, 0, :] * 1e6
            cln = clean_data.get_data(picks=[ch])[0, 0, :] * 1e6
            ax.plot(times, orig, color="#d9534f", alpha=0.65, linewidth=1.0, label="Исходный сигнал (Эпоха 1)")
            ax.plot(times, cln, color="#0275d8", linewidth=1.2, label="После очистки (CAR + ICA)")
            ax.set_ylabel(f"{ch}\n(мкВ)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Время (с)")
    plt.tight_layout()
    plot_path = out_dir / f"{target_file.stem}_preprocessing.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\n3. Готово! Сохранены:\n   - Очищенный сигнал: {out_file}\n   - График сравнения: {plot_path}")


if __name__ == "__main__":
    main()