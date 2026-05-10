# utils/paths.py

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций
from pathlib import Path
from typing import Union

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
PathLike = Union[str, Path]


# ──────────────────────────────────────────────────────────────────────
# КОНСТАНТЫ ПУТЕЙ
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
"""Корневая директория проекта (на два уровня выше этого файла)."""

# 🔹 Пути к данным и моделям
MODELS_DIR: Path = PROJECT_ROOT / "models"
"""Директория для сохранения/загрузки обученных моделей."""

DATA_DIR: Path = PROJECT_ROOT / "data"
"""Базовая директория для всех данных (датасеты, результаты, артефакты)."""

ADE20K_DIR: Path = DATA_DIR / "ade20k_test_trained"
"""Директория с тестовыми данными ADE20K (изображения, маски, отчёты)."""

VALIDATION_DIR: Path = DATA_DIR / "validation_web"
"""Директория для результатов веб-валидации."""

# 🔹 Дефолтные файлы для демо/тестов
DEFAULT_IMAGE: Path = ADE20K_DIR / "original_image_0.jpg"
"""Путь к изображению по умолчанию для демонстраций."""

DEFAULT_GT: Path = ADE20K_DIR / "original_image_mask_0.png"
"""Путь к ground truth маске по умолчанию."""


# ──────────────────────────────────────────────────────────────────────
# УТИЛИТЫ РАБОТЫ С ПУТЯМИ
# ──────────────────────────────────────────────────────────────────────
def ensure_dirs(*dirs: PathLike) -> None:
    """
    Создаёт указанные директории, если они не существуют.

    Args:
        *dirs: Переменное число путей (строк или Path) для создания.

    Example:
        ```python
        ensure_dirs(MODELS_DIR, DATA_DIR / "results")
        ```
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
def get_model_path(filename: str) -> Path:
    """
    Возвращает абсолютный путь к файлу модели в директории `MODELS_DIR`.

    Args:
        filename: Имя файла модели (без пути).

    Returns:
        Path: Полный путь `MODELS_DIR / filename`.

    Example:
        ```python
        get_model_path("unet_best.pth")  # Path('/project/models/unet_best.pth')
        ```
    """
    return MODELS_DIR / filename


# ──────────────────────────────────────────────────────────────────────
def get_data_path(subpath: str) -> Path:
    """
    Возвращает абсолютный путь к файлу в директории `DATA_DIR`.

    Args:
        subpath: Относительный путь внутри `DATA_DIR` (например, `"ade20k/img.jpg"`).

    Returns:
        Path: Полный путь `DATA_DIR / subpath`.

    Example:
        ```python
        get_data_path("ade20k/original_image_0.jpg")
        ```
    """
    return DATA_DIR / subpath


# ──────────────────────────────────────────────────────────────────────
def resolve_relative_path(base: PathLike, relative: str) -> Path:
    """
    Безопасно разрешает относительный путь относительно базовой директории.

    Защищает от выхода за пределы `base` через `..` (path traversal).

    Args:
        base: Базовая директория.
        relative: Относительный путь (может содержать `../`).

    Returns:
        Path: Абсолютный нормализованный путь.

    Raises:
        ValueError: Если разрешённый путь выходит за пределы `base`.
    """
    base_path: Path = Path(base).resolve()
    target_path: Path = (base_path / relative).resolve()

    # Проверка на выход за пределы base (защита от path traversal)
    try:
        target_path.relative_to(base_path)
    except ValueError:
        raise ValueError(
            f"Path traversal detected: {relative} resolves outside {base_path}"
        )

    return target_path


# ──────────────────────────────────────────────────────────────────────
def get_file_extension(path: PathLike) -> str:
    """
    Возвращает расширение файла в нижнем регистре (без точки).

    Args:
        path: Путь к файлу.

    Returns:
        str: Расширение (например, `"jpg"`, `"png"`, `"pth"`).

    Example:
        ```python
        get_file_extension("model.pth")  # "pth"
        ```
    """
    return Path(path).suffix.lstrip(".").lower()


# ──────────────────────────────────────────────────────────────────────
def is_image_file(path: PathLike) -> bool:
    """
    Проверяет, является ли файл изображением по расширению.

    Args:
        path: Путь к файлу.

    Returns:
        bool: `True` если расширение в `[".jpg", ".jpeg", ".png", ".bmp", ".tiff"]`.
    """
    return get_file_extension(path) in {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}


# ──────────────────────────────────────────────────────────────────────
def is_model_file(path: PathLike) -> bool:
    """
    Проверяет, является ли файл моделью PyTorch по расширению.

    Args:
        path: Путь к файлу.

    Returns:
        bool: `True` если расширение в `[".pth", ".pt", ".safetensors"]`.
    """
    return get_file_extension(path) in {"pth", "pt", "safetensors", "bin"}
