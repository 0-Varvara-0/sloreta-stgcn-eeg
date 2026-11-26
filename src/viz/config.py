from __future__ import annotations

# Все 34 корковые области атласа Desikan-Killiany (aparc) × 2 полушария = 68 зон
DK_RU: dict[str, str] = {
    "bankssts": "Борозда верхней височной извилины",
    "caudalanteriorcingulate": "Каудальная передняя поясная кора",
    "caudalmiddlefrontal": "Каудальная средняя лобная извилина",
    "cuneus": "Клин (затылочная доля)",
    "entorhinal": "Энторинальная кора",
    "frontalpole": "Лобный полюс",
    "fusiform": "Веретенообразная извилина",
    "inferiorparietal": "Нижняя теменная долька",
    "inferiortemporal": "Нижняя височная извилина",
    "insula": "Островковая кора",
    "isthmuscingulate": "Перешеек поясной извилины",
    "lateraloccipital": "Латеральная затылочная кора",
    "lateralorbitofrontal": "Латеральная орбитофронтальная кора",
    "lingual": "Язычная извилина",
    "medialorbitofrontal": "Медиальная орбитофронтальная кора",
    "middletemporal": "Средняя височная извилина",
    "paracentral": "Парацентральная долька",
    "parahippocampal": "Парагиппокампальная извилина",
    "parsopercularis": "Оперкулярная часть нижней лобной извилины",
    "parsorbitalis": "Орбитальная часть нижней лобной извилины",
    "parstriangularis": "Треугольная часть нижней лобной извилины",
    "pericalcarine": "Околошпорная кора",
    "postcentral": "Постцентральная извилина",
    "posteriorcingulate": "Задняя поясная кора",
    "precentral": "Прецентральная извилина",
    "precuneus": "Предклинье",
    "rostralanteriorcingulate": "Ростральная передняя поясная кора",
    "rostralmiddlefrontal": "Ростральная средняя лобная извилина",
    "superiorfrontal": "Верхняя лобная извилина",
    "superiorparietal": "Верхняя теменная долька",
    "superiortemporal": "Верхняя височная извилина",
    "supramarginal": "Надкраевая извилина",
    "temporalpole": "Височный полюс",
    "transversetemporal": "Поперечная височная извилина (Гешля)",
}

# Стандартная раскладка 19 электродов системы 10-20 (совместима со standard_1005)
DEFAULT_ELECTRODES: list[str] = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]

# Доступные цветовые палитры для карты активности коры
AVAILABLE_COLORMAPS: list[str] = [
    "inferno",
    "hot",
    "plasma",
    "magma",
    "turbo",
    "viridis",
]

# Имена пресетов камеры для 3D-окна
CAMERA_PRESETS: dict[str, str] = {
    "left": "Слева (Л)",
    "right": "Справа (П)",
    "top": "Сверху",
    "front": "Спереди",
    "back": "Сзади",
    "iso": "Изометрия",
}

DEFAULT_SFREQ: float = 128.0
EXPECTED_ROI_COUNT: int = 68


def get_ordered_roi_names() -> list[str]:
    """
    Возвращает 68 имён корковых зон в точном алфавитном порядке,
    совпадающем с сортировкой в src/source_imaging/sloreta.py.
    """
    names = [f"{region}-{hemi}" for region in DK_RU for hemi in ("lh", "rh")]
    return sorted(names)


def translate_roi_name(label_name: str, include_raw: bool = False) -> str:
    """Переводит метку FreeSurfer (например, 'superiortemporal-lh') на русский язык."""
    hemi_tag = ""
    base = label_name

    if label_name.endswith("-lh"):
        base = label_name[:-3]
        hemi_tag = " (Л)"
    elif label_name.endswith("-rh"):
        base = label_name[:-3]
        hemi_tag = " (П)"

    ru_name = DK_RU.get(base, base)
    formatted = f"{ru_name}{hemi_tag}"
    if include_raw:
        return f"{formatted} [{label_name}]"
    return formatted