# utils/paths.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 🔹 Пути к данным и моделям
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
ADE20K_DIR = DATA_DIR / "ade20k_test_trained"
VALIDATION_DIR = DATA_DIR / "validation_web"

# 🔹 Дефолтные файлы
DEFAULT_IMAGE = ADE20K_DIR / "original_image_0.jpg"
DEFAULT_GT = ADE20K_DIR / "original_image_mask_0.png"


def ensure_dirs(*dirs):
    """Создаёт директории если их нет"""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_model_path(filename: str) -> Path:
    """Возвращает полный путь к модели"""
    return MODELS_DIR / filename


def get_data_path(subpath: str) -> Path:
    """Возвращает полный путь к файлу в data/"""
    return DATA_DIR / subpath
