# utils/settings.py

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций
import os
import sys
from pathlib import Path
from typing import Union

import logging

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Добавляем корень проекта в sys.path для импортов
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic_settings import BaseSettings, SettingsConfigDict
from utils.paths import MODELS_DIR, ADE20K_DIR  # PROJECT_ROOT

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
PathLike = Union[str, Path]


# ──────────────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    """
    Конфигурация приложения через Pydantic Settings.

    Загружает переменные окружения из `.env` файла и предоставляет
    типизированный доступ к путям и параметрам моделей.

    Attributes:
        model_config (SettingsConfigDict): Конфигурация Pydantic (env_file).
        MODEL_DIR (Path): Базовая директория для сохранения/загрузки моделей.
        DEFAULT_IMAGE (Path): Путь к изображению по умолчанию для тестов.
        DEFAULT_GT (Path): Путь к ground truth маске по умолчанию.
        SEGFORMER_PATH (str): Имя/путь модели SegFormer.
        UNET_CHECKPOINT (str): Имя чекпоинта U-Net.
        DEEPLAB_CHECKPOINT (str): Имя чекпоинта DeepLabV3+.
        FPN_MIT_CHECKPOINT (str): Имя чекпоинта FPN + MiT-B5.
        PSP_MIT_CHECKPOINT (str): Имя чекпоинта PSPNet + MiT-B5.
        FCN_RESNET50_CHECKPOINT (str): Имя чекпоинта FCN ResNet50.
        SEGNET_RESNET34_CHECKPOINT (str): Имя чекпоинта SegNet ResNet34.

    Example:
        ```python
        from utils.settings import settings
        print(settings.MODEL_DIR)  # Path('/path/to/models')
        ```
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    MODEL_DIR: Path = MODELS_DIR
    DEFAULT_IMAGE: Path = ADE20K_DIR / "original_image_0.jpg"
    DEFAULT_GT: Path = ADE20K_DIR / "original_image_mask_0.png"

    # 🔹 Имена файлов чекпоинтов
    SEGFORMER_PATH: str = "segformer-b5-ready"
    UNET_CHECKPOINT: str = "unet_ade20k_best_200_epochs.pth"
    DEEPLAB_CHECKPOINT: str = "deeplab_ade20k_best_200_epochs.pth"
    FPN_MIT_CHECKPOINT: str = "fpn_mit_b5_ade20k_best_200_epochs.pth"
    PSP_MIT_CHECKPOINT: str = "psp_mit_b5_ade20k_best_200_epochs.pth"
    FCN_RESNET50_CHECKPOINT: str = "fcn_resnet50_ade20k_best_200_epochs.pth"
    SEGNET_RESNET34_CHECKPOINT: str = "segnet_ade20k_best_200_epochs.pth"

    # ──────────────────────────────────────────────────────────────
    # HELPER METHODS
    # ──────────────────────────────────────────────────────────────
    def get_model_full_path(self, checkpoint_name: str) -> Path:
        """
        Возвращает полный путь к файлу модели.

        Args:
            checkpoint_name: Имя файла чекпоинта (без директории).

        Returns:
            Path: Полный путь `MODEL_DIR / checkpoint_name`.
        """
        return self.MODEL_DIR / checkpoint_name

    # ──────────────────────────────────────────────────────────────────────
    def ensure_model_dir_exists(self) -> None:
        """Создаёт директорию моделей, если она не существует."""
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Глобальный экземпляр настроек
settings: Settings = Settings()
