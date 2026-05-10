# datasets/load_datasets.py

"""
Менеджер загрузки и подготовки датасетов для сегментации.

Поддержка:
- ADE20K, Cityscapes, COCO, ISIC, CheXpert
- Медицинские форматы: DICOM, NIfTI, NPZ, HDF5
- Источники: HuggingFace Hub, ZIP/TAR-архивы, прямые ссылки

Workflow:
1. Регистрация конфигураций датасетов (через код или YAML).
2. Скачивание с валидацией контрольных сумм.
3. Распаковка и реорганизация структуры.
4. Пост-обработка (Parquet → файлы, медицинская нормализация).
5. Создание индекса для быстрого доступа.

Example:
    ```python
    manager = DatasetManager(base_dir="./data")
    path = manager.download("ade20k", force=False)
    img, mask = manager.load_sample("ade20k", split="val", idx=0)
    dataset = manager.create_pytorch_dataset("ade20k", split="train")
    ```
"""


# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций

import os
import sys
import json
import hashlib
import zipfile
import tarfile
import shutil
import requests
from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Literal,
    Callable,
    Any,
    Union,
    TypeAlias,
)
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum, auto
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import yaml
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
    from datasets import load_dataset  # type: ignore[attr-defined]

    HF_AVAILABLE: bool = True
except ImportError:
    HF_AVAILABLE = False
    load_dataset = None  # type: ignore
    yaml = None  # type: ignore

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

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
PathLike: TypeAlias = Union[str, Path]
SplitDict: TypeAlias = Dict[str, str]
PreprocessingDict: TypeAlias = Dict[str, Any]
MetadataDict: TypeAlias = Dict[str, Any]
ExpectedStructureDict: TypeAlias = Dict[str, List[str]]
ImageDict: TypeAlias = Dict[str, Any]  # Для декодирования изображений из HF
ParquetRow: TypeAlias = Dict[str, Any]
DownloadProgressCallback: TypeAlias = Callable[[int, int], None]

# ============================================================================
# КОНФИГУРАЦИЯ И ТИПЫ ДАННЫХ
# ============================================================================

# class DatasetType(Enum):
#     """Типы датасетов для сегментации"""
#     SEMANTIC = auto()      # ADE20K, Cityscapes
#     INSTANCE = auto()      # COCO
#     MEDICAL_BINARY = auto() # ISIC, CheXpert (binary masks)
#     MEDICAL_MULTICLASS = auto() # Multi-organ segmentation
#     CUSTOM = auto()


# ──────────────────────────────────────────────────────────────────────
# ENUMS
# ──────────────────────────────────────────────────────────────────────
class DatasetType(Enum):
    """
    Типы датасетов для сегментации.

    Attributes:
        SEMANTIC: Семантическая сегментация (один класс на пиксель).
        INSTANCE: Instance segmentation (отдельные объекты).
        PANOPTIC: Паноптическая сегментация (семантика + инстансы).
        MEDICAL: Общий медицинский датасет.
        MEDICAL_BINARY: Бинарная медицинская сегментация (фон/объект).
        MEDICAL_MULTICLASS: Многоклассовая медицинская сегментация.
        BINARY: Бинарная сегментация (общий случай).
        CUSTOM: Пользовательский датасет.
    """

    SEMANTIC = "semantic"  # ADE20K, Cityscapes
    INSTANCE = "instance"  # COCO
    PANOPTIC = "panoptic"
    MEDICAL = "medical"
    MEDICAL_BINARY = "medical_binary"  # ISIC, CheXpert (binary masks)
    MEDICAL_MULTICLASS = "medical_multiclass"  # Multi-organ segmentation
    BINARY = "binary"
    CUSTOM = auto()


class SourceType(Enum):
    """
    Источники данных для загрузки.

    Attributes:
        HF: HuggingFace Hub.
        ZIP: ZIP-архив.
        TAR: TAR/GZ архив.
        DIRECT: Прямая ссылка на файл/папку.
    """

    HF = "hf"  # HuggingFace Hub
    ZIP = "zip"  # ZIP-архив
    TAR = "tar"  # TAR/GZ архив
    DIRECT = "direct"  # Прямая ссылка на файл/папку


class DataFormat(Enum):
    """
    Поддерживаемые форматы данных.

    Attributes:
        JPG: JPEG изображения.
        PNG: PNG изображения.
        NIFTI: NIfTI для медицинских 3D-данных.
        DICOM: DICOM серии.
        NPZ: NumPy архивы.
        HDF5: HDF5 для больших данных.
    """

    JPG = "jpg"
    PNG = "png"
    NIFTI = "nii.gz"  # Медицинские 3D
    DICOM = "dcm"  # DICOM серии
    NPZ = "npz"  # NumPy архивы
    HDF5 = "h5"  # HDF5 для больших данных


# ──────────────────────────────────────────────────────────────────────
# DATACLASSES
# ──────────────────────────────────────────────────────────────────────
@dataclass
class DatasetConfig:
    """
    Конфигурация одного датасета.

    Attributes:
        name: Уникальное имя датасета.
        dataset_type: Тип сегментации (из DatasetType).
        source_type: Источник данных (из SourceType).
        source_url: URL или ID репозитория для загрузки.
        root_dir: Базовая директория для сохранения.
        num_classes: Количество классов сегментации.
        ignore_index: Индекс пикселей для игнорирования в лоссе.
        image_ext: Расширение файлов изображений.
        mask_ext: Расширение файлов масок.
        description: Человекочитаемое описание.
        checksum: SHA256 хеш для валидации загрузки (опционально).
        splits: Словарь `{split_name: directory_name}`.
        preprocessing: Параметры предобработки (опционально).
        metadata: Дополнительные метаданные (лицензия, ссылки, ...).
        expected_structure: Ожидаемая структура файлов для валидации.
        postprocess_script: Путь к скрипту пост-обработки (опционально).
    """

    name: str
    dataset_type: DatasetType
    source_type: SourceType
    source_url: str
    root_dir: str
    num_classes: int = 150
    ignore_index: int = 255
    image_ext: str = ".jpg"
    mask_ext: str = ".png"
    description: str = ""
    checksum: Optional[str] = None  # SHA256 для валидации
    splits: SplitDict = field(
        default_factory=lambda: {
            "train": "training",
            "val": "validation",
            "test": "testing",
        }
    )
    preprocessing: PreprocessingDict = field(default_factory=dict)
    metadata: MetadataDict = field(default_factory=dict)
    expected_structure: ExpectedStructureDict = field(default_factory=dict)
    postprocess_script: Optional[str] = None

    # ──────────────────────────────────────────────────────────────────────
    @property
    def full_path(self) -> Path:
        """
        Возвращает полный путь к директории датасета.

        Returns:
            Path: `Path(root_dir) / name`.
        """
        return Path(self.root_dir) / self.name


# ──────────────────────────────────────────────────────────────────────
@dataclass
class MedicalConfig(DatasetConfig):
    """
    Расширенная конфигурация для медицинских датасетов.

    Наследует все поля `DatasetConfig` + добавляет медицинские специфичные.

    Attributes:
        modality: Тип медицинской модальности ("X-Ray", "CT", "MRI", ...).
        pixel_spacing: Физический размер пикселя в мм (опционально).
        intensity_normalization: Метод нормализации интенсивности.
        anatomical_regions: Список анатомических областей в датасете.
        privacy_compliant: Соответствует ли датасет требованиям приватности.
        anatomy: Основная анатомическая область (опционально).
        task_type: Тип задачи ("segmentation", "classification", "detection").
    """

    modality: Literal["X-Ray", "CT", "MRI", "Ultrasound", "Dermoscopy"] = "X-Ray"
    pixel_spacing: Optional[Tuple[float, float]] = None  # мм/пиксель
    intensity_normalization: Literal["minmax", "zscore", "histogram"] = "zscore"
    anatomical_regions: List[str] = field(default_factory=list)
    privacy_compliant: bool = True
    anatomy: Optional[str] = None
    task_type: str = "segmentation"  # segmentation, classification, detection


# ============================================================================
# БАЗОВЫЙ МЕНЕДЖЕР ДАТАСЕТОВ (MAIN CLASS: DatasetManager)
# ============================================================================
class DatasetManager:
    """
    Универсальный менеджер загрузки и валидации датасетов для сегментации.

    Поддерживает:
    - Загрузку из HuggingFace Hub, ZIP/TAR-архивов, прямых ссылок.
    - Валидацию контрольных сумм (SHA256).
    - Автоматическую реорганизацию структуры к стандартному формату.
    - Пост-обработку: Parquet → файлы, медицинская нормализация.
    - Создание индексных файлов для быстрого доступа.
    - Загрузку примеров и создание PyTorch Dataset.

    Attributes:
        base_dir (Path): Базовая директория для всех датасетов.
        verbose (bool): Если `True`, выводит подробные логи.
        _registry (Dict[str, DatasetConfig]): Реестр зарегистрированных конфигураций.

    Example:
        ```python
        manager = DatasetManager(base_dir="./data", verbose=True)
        path = manager.download("ade20k", force=False)
        img, mask = manager.load_sample("ade20k", split="val", idx=0)
        dataset = manager.create_pytorch_dataset("ade20k", split="train")
        ```
    """

    # Глобальный реестр конфигураций
    _registry: Dict[str, DatasetConfig] = {}

    def __init__(
        self,
        base_dir: PathLike = "./data",
        verbose: bool = True,
    ) -> None:
        """
        Инициализация менеджера датасетов.

        Args:
            base_dir: Базовая директория для сохранения датасетов.
            verbose: Если `True`, выводит подробные логи с цветным форматированием.
        """
        self.base_dir: Path = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.verbose: bool = verbose
        self._register_default_datasets()
        self._load_configs()

    # ──────────────────────────────────────────────────────────────────────
    def _load_configs(self) -> None:
        """
        Загружает конфигурации датасетов из внешнего YAML-файла.

        Если файл `configs/datasets.yaml` существует, парсит его и регистрирует
        конфигурации. Медицинские датасеты автоматически используют `MedicalConfig`.
        """
        config_path: Path = Path("configs/datasets.yaml")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                configs_data: Dict[str, Dict[str, Any]] = yaml.safe_load(f)
                for name, cfg in configs_data.items():
                    if cfg.get("modality"):  # Медицинский датасет
                        self._registry[name] = MedicalConfig(name=name, **cfg)
                    else:
                        self._registry[name] = DatasetConfig(name=name, **cfg)

    # ──────────────────────────────────────────────────────────────────────
    def get_config(self, dataset_name: str) -> DatasetConfig:
        """
        Возвращает конфигурацию указанного датасета.

        Args:
            dataset_name: Имя датасета (ключ в `_registry`).

        Returns:
            DatasetConfig: Конфигурация датасета.

        Raises:
            ValueError: Если датасет не найден в реестре.
        """
        if dataset_name not in self._registry:
            available: List[str] = list(self._registry.keys())
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
        return self._registry[dataset_name]

    # ──────────────────────────────────────────────────────────────────────
    def _log(
        self,
        message: str,
        level: Literal["info", "success", "warning", "error"] = "info",
    ) -> None:
        """
        Логирует сообщение с цветным форматированием и таймстампом.

        Args:
            message: Текст сообщения.
            level: Уровень лога ("info", "success", "warning", "error").
        """
        if not self.verbose:
            return
        colors: Dict[str, str] = {
            "info": "\033[94m",
            "success": "\033[92m",
            "warning": "\033[93m",
            "error": "\033[91m",
            "reset": "\033[0m",
        }
        timestamp: str = datetime.now().strftime("%H:%M:%S")
        prefix: Dict[str, str] = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
        }

        logger.info(
            f"[{timestamp}] {colors.get(level, '')}{prefix.get(level, '•')} {message}{colors['reset']}"
        )

    # ──────────────────────────────────────────────────────────────────────
    def _check_disk_space(self, path: Path, required_gb: float) -> bool:
        """
        Проверяет наличие достаточного свободного места на диске.

        Args:
            path: Путь для проверки.
            required_gb: Требуемое пространство в ГБ.

        Returns:
            bool: `True` если места достаточно, иначе `False`.
        """

        total, used, free = shutil.disk_usage(path)
        free_gb: float = free / (1024**3)

        if free_gb < required_gb:
            self._log(
                f"❌ Недостаточно места: {free_gb:.1f}GB свободно, требуется {required_gb}GB",
                "error",
            )
            return False
        return True

    # ──────────────────────────────────────────────────────────────────────
    def _compute_sha256(self, filepath: Path) -> str:
        """
        Вычисляет SHA256 хеш файла для валидации целостности.

        Args:
            filepath: Путь к файлу.

        Returns:
            str: SHA256 хеш в шестнадцатеричном формате.
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    # ──────────────────────────────────────────────────────────────────────
    def _register_default_datasets(self) -> None:
        """
        Регистрирует стандартные датасеты в `_registry`.

        Включает:
        - ADE20K (семантическая, 150 классов)
        - Cityscapes (городские сцены, 19 классов)
        - COCO (instance segmentation, 80 классов)
        - ISIC 2018 (медицинская, бинарная, дерматоскопия)
        - CheXpert (медицинская, бинарная, рентген грудной клетки)
        """

        # ============================================================================
        # 1. ADE20K (Большой датасет)
        # ============================================================================

        # === ADE20K ===
        self._registry["ade20k"] = DatasetConfig(
            name="ade20k",
            dataset_type=DatasetType.SEMANTIC,
            source_type=SourceType.ZIP,
            source_url="http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
            root_dir=str(self.base_dir),
            num_classes=150,
            checksum="7ff1be44964418441f542a7cc1e1a650e7dc0fc275f5d23252bc9bbdbc977b29",
            metadata={
                "license": "MIT",
                "citation": "Zhou et al., 2017",
                "homepage": "http://sceneparsing.csail.mit.edu/",
            },
        )

        # ============================================================================
        # 2. CITYSCAPES (городские сцены)
        # ============================================================================

        # === Cityscapes ===
        self._registry["cityscapes"] = DatasetConfig(
            name="cityscapes",
            dataset_type=DatasetType.SEMANTIC,
            source_type=SourceType.HF,
            source_url="Chris1/cityscapes",
            root_dir=str(self.base_dir),
            num_classes=19,
            metadata={
                "license": "CC-BY-NC-SA 3.0",
                "requires_registration": True,
                "homepage": "https://www.cityscapes-dataset.com/",
            },
        )

        # ============================================================================
        # 3. COCO (общие объекты)
        # ============================================================================

        # === COCO ===
        self._registry["coco"] = DatasetConfig(
            name="coco",
            dataset_type=DatasetType.INSTANCE,
            source_type=SourceType.HF,
            source_url="detection-datasets/coco",
            root_dir=str(self.base_dir),
            num_classes=80,
            metadata={"license": "CC BY 4.0", "homepage": "https://cocodataset.org/"},
        )

        # ============================================================================
        # 4. МЕДИЦИНСКИЙ ДАТАСЕТ (ISIC - skin lesion segmentation)
        # ============================================================================

        # === ISIC 2018 (Medical - Dermoscopy) ===
        self._registry["isic2018"] = MedicalConfig(
            name="isic2018",
            dataset_type=DatasetType.MEDICAL_BINARY,
            source_type=SourceType.HF,
            source_url="researchjyotsna/isic2018_10",
            root_dir=str(self.base_dir),
            num_classes=2,  # background / lesion
            modality="Dermoscopy",
            pixel_spacing=(0.025, 0.025),  # ~25 мкм/пиксель
            intensity_normalization="minmax",
            anatomical_regions=["skin"],
            metadata={
                "license": "CC BY-NC 4.0",
                "task": "Skin lesion segmentation",
                "classes": {"0": "background", "1": "lesion"},
                "homepage": "https://challenge.isic-archive.com/",
            },
        )

        # ============================================================================
        # 5. Chest X-Ray
        # ============================================================================

        # === CheXpert (Medical - Chest X-Ray) ===
        self._registry["chexpert"] = MedicalConfig(
            name="chexpert",
            dataset_type=DatasetType.MEDICAL_BINARY,
            source_type=SourceType.HF,
            source_url="stanfordmlgroup/chexpert",
            root_dir=str(self.base_dir),
            num_classes=2,  # background / lung
            modality="X-Ray",
            pixel_spacing=(0.143, 0.143),  # ~143 мкм/пиксель
            intensity_normalization="histogram",
            anatomical_regions=["lung", "heart", "mediastinum"],
            privacy_compliant=True,
            metadata={
                "license": "MIT",
                "task": "Lung field segmentation",
                "classes": {"0": "background", "1": "lung"},
                "homepage": "https://stanfordmlgroup.github.io/competitions/chexpert/",
            },
        )
        self._log("✅ Готово! Датасеты зарегистрированы.", "success")

    # ──────────────────────────────────────────────────────────────────────
    def download(
        self,
        dataset_name: str,
        force: bool = False,
    ) -> Path:
        """
        Скачивает и валидирует датасет.

        Логика:
        1. Проверяет наличие и валидность существующей копии.
        2. В зависимости от `source_type` вызывает соответствующий метод загрузки.
        3. Выполняет пост-обработку (Parquet → файлы, медицинская нормализация).
        4. Валидирует структуру и создаёт индекс.

        Args:
            dataset_name: Имя датасета (ключ в `_registry`).
            force: Если `True`, перезагрузить даже при наличии валидной копии.

        Returns:
            Path: Путь к директории датасета.

        Raises:
            ValueError: Если датасет не найден или источник не поддерживается.
            Exception: При ошибке загрузки или валидации.
        """
        config: DatasetConfig = self.get_config(dataset_name)

        logger.info(f"\n{'=' * 70}")
        logger.info(
            f"📦 ЗАГРУЗКА ДАТАСЕТА: {config.name.upper()} ({config.dataset_type.name})..."
        )
        logger.info(f"{'=' * 70}")
        logger.info(f"Тип: {config.dataset_type.name}")
        logger.info(f"Источник: {config.source_type.name.upper()}")
        logger.info(f"Классы: {config.num_classes}")
        if isinstance(config, MedicalConfig):
            logger.info(f"Модальность: {config.modality}")
        logger.info(f"Целевая директория: {config.full_path}")
        logger.info(f"{'=' * 70}\n")

        if config.full_path.exists() and not force:
            if self._validate_dataset(config):
                self._log(
                    f"✅ Датасет уже существует и валиден: {config.full_path}",
                    "success",
                )
                self._print_dataset_summary(config)
                return config.full_path
            else:
                self._log(
                    "⚠️ Существующий датасет не прошёл валидацию, перезагрузка...",
                    "warning",
                )

        try:
            if config.source_type == SourceType.HF:
                self._download_huggingface(config)
            elif config.source_type == SourceType.ZIP:
                self._download_zip(config)
            elif config.source_type == SourceType.TAR:
                self._download_tar(config)
            elif config.source_type == SourceType.DIRECT:
                self._download_direct(config)
            else:
                raise ValueError(f"Unsupported source type: {config.source_type}")

            # Пост-обработка
            self._postprocess_dataset(config)

            # Валидация
            if self._validate_dataset(config):
                self._log(f"\n✅ {config.name.upper()} загружен успешно!", "success")
                self._print_dataset_summary(config)
                return config.full_path
            else:
                self._log("\n⚠️ Датасет загружен, но валидация не пройдена", "warning")
                return config.full_path

        except Exception as e:
            self._log(f"❌ Ошибка загрузки: {e}", "error")
            raise

    # ──────────────────────────────────────────────────────────────────────
    def _print_dataset_summary(self, config: DatasetConfig) -> None:
        """
        Печатает сводную информацию о датасете.

        Args:
            config: Конфигурация датасета.
        """
        logger.info(f"\n📊 СВОДКА ПО ДАТАСЕТУ: {config.name.upper()}")
        logger.info(f"{'=' * 50}")
        logger.info(f"Тип: {config.dataset_type.value}")
        logger.info(f"Классы: {config.num_classes}")
        if isinstance(config, MedicalConfig):
            logger.info(f"Модальность: {config.modality}")
        logger.info(f"Путь: {config.full_path}")

        for split_name, split_dir in config.splits.items():
            img_dir: Path = config.full_path / "images" / split_dir
            ann_dir: Path = config.full_path / "annotations" / split_dir

            if img_dir.exists():
                n_images: int = len(list(img_dir.glob(f"*{config.image_ext}")))
                n_masks: int = (
                    len(list(ann_dir.glob(f"*{config.mask_ext}")))
                    if ann_dir.exists()
                    else 0
                )
                logger.info(
                    f"   {split_name:12s}: {n_images:5d} images, {n_masks:5d} masks"
                )

        logger.info(f"{'-' * 50}")

    # ──────────────────────────────────────────────────────────────────────
    def _download_huggingface(self, config: DatasetConfig) -> None:
        """
        Загружает датасет из HuggingFace Hub.

        Args:
            config: Конфигурация датасета.

        Raises:
            ValueError: Для датасетов с ограниченным доступом (например, CheXpert).
        """
        self._log(f"📥 Загрузка из HuggingFace Hub: {config.source_url}")
        local_dir: Path = config.full_path

        try:
            if config.name in ["coco", "cityscapes"]:
                if not self._check_disk_space(self.base_dir, required_gb=20.0):
                    logger.error("❌ Недостаточно места на диске для датасета")
                    return

                self._log("📊 Загрузка через datasets library...")
                hf_dataset = load_dataset(
                    path=config.source_url, cache_dir=str(self.base_dir / ".cache")
                )
                self._log("✅ Загружено через datasets library")
                self._create_index_from_hf_dataset(config, hf_dataset)
                return

            logger.info("📦 Скачивание файлов датасета...")
            snapshot_download(  # type: ignore
                repo_id=config.source_url,
                repo_type="dataset",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                cache_dir=None,
                force_download=False,
            )
            logger.info("✅ Скачивание завершено")

        except Exception as e:
            error_msg: str = str(e)
            if config.name == "chexpert" and (
                "401" in error_msg or "Repository Not Found" in error_msg
            ):
                self._log("❌ CheXpert requires manual download:", "error")
                self._log(
                    "   1. Register at https://stanfordmlgroup.github.io/competitions/chexpert/",
                    "error",
                )
                self._log("   2. Download CheXpert-v1.0.zip manually", "error")
                self._log(
                    "   3. Extract to data/chexpert/ and run validation again", "error"
                )
                raise ValueError(
                    "CheXpert requires manual download due to access restrictions"
                )

            self._log(f"⚠️ HF download failed, trying fallback: {error_msg}", "warning")
            self._hf_fallback_download(config)

    # ──────────────────────────────────────────────────────────────────────
    def _hf_fallback_download(
        self, config: DatasetConfig, use_api: bool = False
    ) -> None:
        """
        Fallback-метод загрузки из HF: пофайловая загрузка.

        Args:
            config: Конфигурация датасета.
            use_api: Если `True`, использовать экспериментальный API-метод.
        """
        if use_api:
            return self._download_via_api(config)
        local_dir: Path = config.full_path
        local_dir.mkdir(parents=True, exist_ok=True)

        repo_id: str = config.source_url
        files: List[str] = list_repo_files(repo_id, repo_type="dataset")

        # Фильтруем только нужные файлы
        image_files: List[str] = [
            f for f in files if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        mask_files: List[str] = [f for f in files if f.endswith((".png", ".nii.gz"))]

        for file_list, desc in [
            (image_files, "🖼️  Изображения"),
            (mask_files, "🎭 Маски"),
        ]:
            if not file_list:
                continue
            logger.info(f"{desc}: {len(file_list)} файлов")
            for filename in tqdm(
                file_list, desc=f"{config.name}/{desc.split()[1]}", unit="files"
            ):
                local_path: Path = local_dir / filename
                local_path.parent.mkdir(parents=True, exist_ok=True)
                if not local_path.exists():
                    hf_hub_download(  # type: ignore
                        repo_id=repo_id,
                        filename=filename,
                        repo_type="dataset",
                        local_dir=str(local_dir),
                    )

    # ──────────────────────────────────────────────────────────────────────
    def _download_via_api(self, config: DatasetConfig) -> None:
        """
        Экспериментальный метод загрузки через HF API.

        Args:
            config: Конфигурация датасета.
        """
        repo_id: str = config.source_url
        local_dir: Path = config.full_path

        try:
            api_url: str = f"https://huggingface.co/api/datasets/{repo_id}"
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()

            files_info: List[Dict[str, Any]] = response.json().get("siblings", [])
            target_files: List[str] = [
                f["rfilename"]
                for f in files_info
                if f["rfilename"].endswith((".jpg", ".png"))
            ]

            for filename in tqdm(target_files, desc=f"📥 {config.name}", unit="files"):
                hf_hub_download(  # type: ignore
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(local_dir),
                )
        except Exception as e:
            logger.warn(
                f"⚠️ API-скачивание не удалось: {e}. Переключаюсь на стандартный метод..."
            )
            # Fallback на стандартный метод
            return self._hf_fallback_download(config, use_api=False)

    # ──────────────────────────────────────────────────────────────────────
    def _download_zip(self, config: DatasetConfig) -> None:
        """
        Скачивает и распаковывает ZIP-архив с прогрессом и валидацией.

        Args:
            config: Конфигурация датасета.

        Raises:
            ValueError: При обнаружении небезопасных путей в архиве.
        """
        zip_path: Path = config.full_path.with_suffix(".zip")
        logger.info(f"📦 ЗАГРУЗКА АРХИВА: {config.name}")
        logger.info(f"{'-' * 50}")

        if not zip_path.exists():
            self._log("📥 Скачивание архива...")
            self._log(f"   URL: {config.source_url}")
            if config.checksum:
                logger.info(f"   SHA256: {config.checksum[:16]}...")
            try:
                head = requests.head(config.source_url, allow_redirects=True)
                size_bytes: int = int(head.headers.get("content-length", 0))
                logger.info(f"   Размер: ~{size_bytes / (1024 * 1024 * 1024):.1f} GB")
            except Exception:
                logger.error("   Размер: неизвестен")
            self._streaming_download(config.source_url, zip_path, config.checksum)
            zip_size: float = os.path.getsize(zip_path) / (1024 * 1024 * 1024)
            logger.info(f"✅ Скачивание завершено! Размер: {zip_size:.2f} GB")
        else:
            zip_size = os.path.getsize(zip_path) / (1024 * 1024 * 1024)
            logger.warn(f"✅ Архив уже существует: {zip_path} ({zip_size:.2f} GB)")

        extract_dir: Path = config.full_path / "temp_extract"
        if not (config.full_path / "images").exists():
            self._log("\n📦 Распаковка архива...")
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                members: List[str] = zip_ref.namelist()
                logger.info(f"   Всего файлов в архиве: {len(members)}")
                for member in tqdm(members, desc="Распаковка", unit="files"):
                    # Безопасная распаковка (защита от path traversal)
                    target_path: Path = extract_dir / member
                    if not str(target_path.resolve()).startswith(
                        str(extract_dir.resolve())
                    ):
                        raise ValueError(f"Unsafe path in archive: {member}")
                    zip_ref.extract(member, extract_dir)
            logger.info("✅ Распаковка завершена!")
            logger.info("\n🔍 Анализ структуры распакованных файлов...")
            self._reorganize_ade_structure(extract_dir, config.full_path)
            logger.warn("\n🧹 Очистка временных файлов...")
            shutil.rmtree(extract_dir, ignore_errors=True)
            if config.checksum:
                zip_path.unlink()
                logger.info("✅ Архив удалён")
        else:
            logger.info(f"✅ Датасет уже распакован в {config.full_path}")

    # ──────────────────────────────────────────────────────────────────────
    def _download_tar(self, config: DatasetConfig) -> None:
        """
        Скачивает и распаковывает TAR/GZ архив.

        Args:
            config: Конфигурация датасета.

        Raises:
            ValueError: При обнаружении небезопасных путей в архиве.
        """
        tar_path: Path = config.full_path.with_suffix(".tar.gz")
        logger.info(f"📦 ЗАГРУЗКА TAR-АРХИВА: {config.name}")

        if not tar_path.exists():
            self._log("📥 Скачивание архива...")
            self._streaming_download(config.source_url, tar_path, config.checksum)
            logger.info("✅ Скачивание завершено")

        # Распаковка
        extract_dir: Path = config.full_path / "temp_extract"
        if not (config.full_path / "images").exists():
            self._log("\n📦 Распаковка TAR-архива...")
            extract_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(tar_path, "r:gz") as tar_ref:
                members = tar_ref.getmembers()
                logger.info(f"   Всего файлов: {len(members)}")
                for member in tqdm(members, desc="Распаковка", unit="files"):
                    target_path = extract_dir / member.name
                    if not str(target_path.resolve()).startswith(
                        str(extract_dir.resolve())
                    ):
                        raise ValueError(f"Unsafe path in archive: {member.name}")
                    tar_ref.extract(member, extract_dir)

            self._reorganize_ade_structure(extract_dir, config.full_path)
            shutil.rmtree(extract_dir, ignore_errors=True)
            if config.checksum:
                tar_path.unlink()
        else:
            logger.info("✅ Датасет уже распакован")

    # ──────────────────────────────────────────────────────────────────────
    def _download_direct(self, config: DatasetConfig) -> None:
        """
        Прямая загрузка файлов по ссылке.

        Args:
            config: Конфигурация датасета.
        """
        self._log(f"📥 Прямая загрузка: {config.source_url}")
        target_dir: Path = config.full_path
        target_dir.mkdir(parents=True, exist_ok=True)

        # Если ссылка на файл
        if config.source_url.endswith((".jpg", ".png", ".npy", ".zip")):
            filename: str = config.source_url.split("/")[-1]
            filepath: Path = target_dir / filename
            self._streaming_download(config.source_url, filepath, config.checksum)
        else:
            # Если ссылка на директорию (рекурсивное скачивание)
            self._log(
                "⚠️ Прямая загрузка директорий требует дополнительной реализации",
                "warning",
            )

    # ──────────────────────────────────────────────────────────────────────
    def _streaming_download(
        self,
        url: str,
        destination: Path,
        expected_checksum: Optional[str] = None,
    ) -> None:
        """
        Потоковая загрузка с прогрессом и проверкой контрольной суммы.

        Args:
            url: URL для загрузки.
            destination: Путь для сохранения файла.
            expected_checksum: Ожидаемый SHA256 хеш (опционально).

        Raises:
            ValueError: Если контрольная сумма не совпадает.
        """
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size: int = int(response.headers.get("content-length", 0))
        destination.parent.mkdir(parents=True, exist_ok=True)
        # sha256_hash = hashlib.sha256() if expected_checksum else None

        with open(destination, "wb") as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Валидация контрольной суммы
        if expected_checksum:
            actual_checksum: str = self._compute_sha256(destination)
            if actual_checksum != expected_checksum:
                destination.unlink()
                raise ValueError(
                    f"❌ Checksum mismatch!\n"
                    f"Expected: {expected_checksum}\n"
                    f"Actual:   {actual_checksum}"
                )
        self._log("✅ Контрольная сумма совпадает", "success")

    # ──────────────────────────────────────────────────────────────────────
    def _reorganize_ade_structure(
        self,
        source: Path,
        target: Path,
        use_symlinks: bool = False,
    ) -> None:
        """
        Приводит структуру распакованного датасета к стандартному формату.

        Ожидаемая структура:
        ```
        target/
        ├── images/
        │   ├── training/
        │   └── validation/
        └── annotations/
            ├── training/
            └── validation/
        ```

        Args:
            source: Исходная директория с распакованными файлами.
            target: Целевая директория для реорганизации.
            use_symlinks: Если `True`, использовать символические ссылки (только Unix).

        Raises:
            ValueError: Если не найдена ожидаемая структура `images/annotations`.
        """
        found_structure: bool = False
        for root, dirs, _ in os.walk(source):
            if "images" in dirs and "annotations" in dirs:
                src_images: Path = Path(root) / "images"
                src_annotations: Path = Path(root) / "annotations"
                logger.info(f"✅ Найдена структура в: {root}")
                found_structure = True

                for split in ["training", "validation"]:
                    src_img_split: Path = src_images / split
                    src_ann_split: Path = src_annotations / split

                    if src_img_split.exists():
                        tgt_img_split: Path = target / "images" / split
                        tgt_ann_split: Path = target / "annotations" / split
                        tgt_img_split.mkdir(parents=True, exist_ok=True)
                        tgt_ann_split.mkdir(parents=True, exist_ok=True)

                        # Копирование с прогрессом
                        img_files: List[Path] = list(src_img_split.glob("*.jpg"))
                        ann_files: List[Path] = list(src_ann_split.glob("*.png"))

                        logger.info(f"\n   📋 {split}:")
                        logger.info(f"      Изображения: {len(img_files)} файлов")
                        logger.info(f"      Маски: {len(ann_files)} файлов")

                        for img_file in tqdm(
                            img_files,
                            desc=f"   Копирование {split}/images",
                            unit="files",
                        ):
                            shutil.copy2(img_file, tgt_img_split / img_file.name)

                        for ann_file in tqdm(
                            ann_files,
                            desc=f"   Копирование {split}/annotations",
                            unit="files",
                        ):
                            shutil.copy2(ann_file, tgt_ann_split / ann_file.name)
                logger.info(f"\n✅ Файлы организованы в: {target}")
                return
        if not found_structure:
            logger.error("❌ Не удалось найти папки images и annotations")
            logger.info("\n📂 Содержимое распакованной папки:")
            for root, dirs, files in os.walk(source):
                level: int = root.replace(str(source), "").count(os.sep)
                indent: str = " " * 2 * level
                logger.info(f"{indent}{Path(root).name}/")
                subindent: str = " " * 2 * (level + 1)
                for d in dirs[:5]:
                    logger.info(f"{subindent}📁 {d}/")
                for f in files[:5]:
                    logger.info(f"{subindent}📄 {f}")
            raise ValueError("Could not find images/annotations structure in archive")

        if (
            use_symlinks and sys.platform != "win32"
        ):  # Windows требует прав администратора
            try:
                for split in ["training", "validation"]:
                    src: Path = target / split
                    tgt: Path = target / "images" / split
                    if src.exists() and not tgt.exists():
                        tgt.parent.mkdir(parents=True, exist_ok=True)
                        os.symlink(src.relative_to(tgt.parent), tgt)
                        logger.info(f"   🔗 Создана ссылка: {tgt} -> {src}")
            except (OSError, PermissionError) as e:
                logger.error(f"⚠️ Не удалось создать символические ссылки: {e}")

    # ──────────────────────────────────────────────────────────────────────
    def _postprocess_dataset(self, config: DatasetConfig) -> None:
        """
        Выполняет пост-обработку датасета.

        Поддерживает:
        - Конвертацию Parquet → файловая структура.
        - Медицинскую пост-обработку (CLAHE, нормализация масок).
        - Создание индексного файла.

        Args:
            config: Конфигурация датасета.
        """
        self._log("\n🔧 Пост-обработка датасета...")

        parquet_files: List[Path] = list(config.full_path.rglob("*.parquet"))
        if parquet_files:
            logger.info(f"Обнаружено Parquet-файлов: {len(parquet_files)}")
            self._convert_parquet_to_files(config)

        if isinstance(config, MedicalConfig):
            logger.info("🏥 Применение медицинской пост-обработки...")
            self._medical_postprocess(config)

        logger.info("Создание индексного файла...")
        self._create_index(config)
        logger.info("✅ Пост-обработка завершена")

    # ──────────────────────────────────────────────────────────────────────
    def _decode_image_from_hf(
        self, data: Any, convert_to_rgb: bool = True
    ) -> Optional[Image.Image]:
        """
        Универсальный декодер для данных из HuggingFace datasets.

        Поддерживает форматы:
        - `datasets.Image` объект
        - `bytes` объект
        - `dict` с ключами `"bytes"` или `"path"`
        - base64 строка
        - `PIL.Image`

        Args:
             Данные изображения в любом поддерживаемом формате.
            convert_to_rgb: Если `True`, конвертировать в режим "RGB".

        Returns:
            Optional[PIL.Image]: Декодированное изображение или `None`.
        """
        if data is None:
            return None

        # 🔹 datasets.Image объект (имеет метод convert)
        if hasattr(data, "convert") and hasattr(data, "mode"):
            return (
                data.convert("RGB") if convert_to_rgb and data.mode != "RGB" else data
            )

        # 🔹 bytes объект
        if isinstance(data, bytes) and len(data) > 0:
            try:
                from PIL import Image
                from io import BytesIO

                return (
                    Image.open(BytesIO(data)).convert("RGB")
                    if convert_to_rgb
                    else Image.open(BytesIO(data))
                )
            except ImportError:
                return None

        # 🔹 dict с 'bytes' или 'path' (формат HF datasets)
        if isinstance(data, dict):
            if "bytes" in data and data["bytes"] and isinstance(data["bytes"], bytes):
                try:
                    from PIL import Image
                    from io import BytesIO

                    return (
                        Image.open(BytesIO(data["bytes"])).convert("RGB")
                        if convert_to_rgb
                        else Image.open(BytesIO(data["bytes"]))
                    )
                except ImportError:
                    return None
            if "path" in data and data["path"] and os.path.exists(data["path"]):
                try:
                    from PIL import Image

                    return (
                        Image.open(data["path"]).convert("RGB")
                        if convert_to_rgb
                        else Image.open(data["path"])
                    )
                except ImportError:
                    return None

        # 🔹 base64 строка
        if isinstance(data, str) and data.startswith("data:image"):
            try:
                import base64
                from PIL import Image
                from io import BytesIO

                _, encoded = data.split(",", 1)
                return (
                    Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
                    if convert_to_rgb
                    else Image.open(BytesIO(base64.b64decode(encoded)))
                )
            except ImportError:
                return None

        # 🔹 PIL.Image (уже готов)
        if hasattr(data, "save") and hasattr(data, "mode"):
            return (
                data.convert("RGB") if convert_to_rgb and data.mode != "RGB" else data
            )

        return None

    # ──────────────────────────────────────────────────────────────────────
    def _convert_parquet_to_files(self, config: DatasetConfig) -> None:
        """
        Конвертирует Parquet-файлы в файловую структуру изображений/масок.

        Поддерживает различные форматы данных из HF datasets:
        - `datasets.Image` объекты
        - `bytes`, `dict{"bytes": ...}`, base64 строки
        - Пути к файлам

        Args:
            config: Конфигурация датасета.
        """
        try:
            import pandas as pd
        except ImportError as e:
            self._log(f"⚠️ Missing dependencies for Parquet conversion: {e}", "warning")
            return

        self._log(f"\n🔄 Конвертация Parquet → файловая структура для {config.name}...")

        parquet_files: List[Path] = list(config.full_path.rglob("*.parquet"))
        if not parquet_files:
            self._log("⚠️ Parquet-файлы не найдены", "warning")
            return
        logger.info(f"Найдено Parquet-файлов: {len(parquet_files)}")

        # Создаём целевые директории
        for split in ["training", "validation", "train", "val", "test"]:
            (config.full_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (config.full_path / "annotations" / split).mkdir(
                parents=True, exist_ok=True
            )

        image_keys: List[str] = [
            "image",
            "images",
            "img",
            "input",
            "pixel_values",
            "source",
        ]
        mask_keys: List[str] = [
            "mask",
            "masks",
            "annotation",
            "segmentation_mask",
            "label",
            "ground_truth",
            "segmentation",
        ]
        split_keys: List[str] = ["split", "phase", "mode", "subset"]

        converted_count: int = 0
        error_count: int = 0
        total_rows: int = 0

        for pq_file in parquet_files:
            try:
                df: pd.DataFrame = pd.read_parquet(pq_file)
                total_rows += len(df)
                self._log(f"\n📊 Обработка {pq_file.name}: {len(df)} строк")
                img_key: Optional[str] = next(
                    (k for k in image_keys if k in df.columns), None
                )
                mask_key: Optional[str] = next(
                    (k for k in mask_keys if k in df.columns), None
                )
                split_key: Optional[str] = next(
                    (k for k in split_keys if k in df.columns), None
                )

                if not img_key:
                    self._log(
                        f"⚠️ Нет колонки с изображениями в {pq_file.name}. Доступные: {list(df.columns)}",
                        "warning",
                    )
                    continue

                logger.info(f"   🖼️  Ключ изображения: '{img_key}'")
                if mask_key:
                    logger.info(f"   🎭 Ключ маски: '{mask_key}'")
                if split_key:
                    logger.info(f"   📑 Ключ split: '{split_key}'")

                for idx, row in tqdm(
                    df.iterrows(), total=len(df), desc="   Конвертация", unit="rows"
                ):
                    if converted_count >= 1000 and not self.verbose:
                        break

                    try:
                        if split_key and split_key in row:
                            split_val: str = str(row[split_key]).lower()
                            if split_val in ["train", "training"]:
                                split = "training"
                            elif split_val in ["val", "validation", "valid"]:
                                split = "validation"
                            elif split_val in ["test", "testing"]:
                                split = "testing"
                            else:
                                split = "training"
                        else:
                            fname: str = pq_file.name.lower()
                            if "train" in fname:
                                split = "training"
                            elif "val" in fname or "valid" in fname:
                                split = "validation"
                            elif "test" in fname:
                                split = "testing"
                            else:
                                split = "training"

                        # === Декодирование изображения ===
                        img: Optional[Image.Image] = self._decode_image_from_hf(
                            row[img_key], convert_to_rgb=True
                        )
                        if img is None:
                            error_count += 1
                            continue

                        # === Декодирование маски (если есть) ===
                        mask: Optional[Image.Image] = None
                        if mask_key and mask_key in row and row[mask_key] is not None:
                            mask = self._decode_image_from_hf(
                                row[mask_key], convert_to_rgb=False
                            )
                            if mask and mask.mode != "L":
                                mask = mask.convert("L")

                        # === Сохранение ===
                        filename: str = f"{config.name}_{split}_{idx:06d}"
                        img_path: Path = (
                            config.full_path / "images" / split / f"{filename}.jpg"
                        )
                        img.save(img_path, quality=95)

                        if mask:
                            mask_path = (
                                config.full_path
                                / "annotations"
                                / split
                                / f"{filename}.png"
                            )
                            mask.save(mask_path)

                        converted_count += 1

                    except Exception as e:
                        error_count += 1
                        if self.verbose:
                            self._log(
                                f"   ⚠️ Ошибка в строке {idx}: {type(e).__name__}",
                                "warning",
                            )
                        continue

            except Exception as e:
                self._log(f"⚠️ Ошибка обработки {pq_file.name}: {e}", "warning")
                continue

        self._log(
            f"✅ Converted {converted_count} samples, {error_count} errors", "success"
        )

        logger.info(f"\n{'=' * 50}")
        logger.info("📊 СТАТИСТИКА КОНВЕРТАЦИИ")
        logger.info(f"{'-' * 50}")
        logger.info(f"   Всего обработано строк: {total_rows}")
        logger.info(f"   ✅ Успешно конвертировано: {converted_count}")
        logger.info(f"   ❌ Ошибок: {error_count}")
        if total_rows > 0:
            success_rate: float = (converted_count / total_rows) * 100
            logger.info(f"   📈 Успешность: {success_rate:.1f}%")
        logger.info(f"{'-' * 50}")
        self._create_index(config)
        logger.info("✅ Конвертация завершена")

    # ──────────────────────────────────────────────────────────────────────
    def _medical_postprocess(self, config: MedicalConfig) -> None:
        """
        Специальная обработка медицинских датасетов.

        Применяет:
        - CLAHE для улучшения контраста рентгеновских снимков.
        - Нормализацию масок к единому формату (0/1 для бинарных).

        Args:
            config: Конфигурация медицинского датасета.
        """
        self._log(
            f"🏥 Применение медицинской пост-обработки: {config.intensity_normalization}"
        )

        # Пример: нормализация интенсивности для рентгена
        if config.modality == "X-Ray" and config.intensity_normalization == "histogram":
            self._apply_clahe(config.full_path / "images")

        # Конвертация масок к единому формату (0/1 или 0/255)
        self._normalize_masks(config.full_path / "annotations", config.num_classes)

    # ──────────────────────────────────────────────────────────────────────
    def _apply_clahe(self, images_dir: Path, clip_limit: float = 2.0) -> None:
        """
        Применяет CLAHE для улучшения контраста рентгеновских снимков.

        Args:
            images_dir: Директория с изображениями.
            clip_limit: Лимит клиппинга для CLAHE.
        """
        try:
            import cv2
        except ImportError:
            self._log("⚠️ OpenCV не установлен, пропускаем CLAHE", "warning")
            return

        logger.info("🔧 Применение CLAHE к изображениям...")
        for img_path in tqdm(
            list(images_dir.rglob("*.jpg")), desc="CLAHE", unit="imgs"
        ):
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                enhanced = clahe.apply(img)
                cv2.imwrite(str(img_path), enhanced)
            except Exception as e:
                self._log(f"⚠️ Ошибка обработки {img_path.name}: {e}", "warning")

    # ──────────────────────────────────────────────────────────────────────
    def _normalize_masks(self, masks_dir: Path, num_classes: int) -> None:
        """
        Приводит маски к единому формату.

        Для бинарных медицинских масок: всё кроме 0 → 1.

        Args:
            masks_dir: Директория с масками.
            num_classes: Количество классов (2 для бинарных).
        """
        logger.info("🎭 Нормализация масок...")
        for mask_path in tqdm(list(masks_dir.rglob("*.png")), desc="Normalizing masks"):
            try:
                mask: np.ndarray = np.array(Image.open(mask_path))

                # Для бинарных медицинских масок: всё кроме 0 -> 1
                if num_classes == 2:
                    mask = (mask > 0).astype(np.uint8)

                # Сохранение с явным указанием типа
                Image.fromarray(mask).save(mask_path)
            except Exception as e:
                self._log(f"⚠️ Failed to normalize {mask_path.name}: {e}", "warning")

    # ──────────────────────────────────────────────────────────────────────
    def _create_index(self, config: DatasetConfig) -> None:
        """
        Создаёт индексный файл для быстрого доступа к датасету.

        Формат индекса:
        ```json
        {
            "dataset": "ade20k",
            "type": "SEMANTIC",
            "created": "2024-01-01T12:00:00",
            "splits": {
                "train": {"count": 20000, "images": [...]},
                "val": {"count": 2000, "images": [...]}
            }
        }
        ```

        Args:
            config: Конфигурация датасета.
        """
        index: Dict[str, Any] = {
            "dataset": config.name,
            "type": config.dataset_type.name,
            "created": datetime.now().isoformat(),
            "splits": {},
        }

        for split_name, split_dir in config.splits.items():
            img_dir: Path = config.full_path / "images" / split_dir
            if img_dir.exists():
                images: List[str] = sorted(
                    [f.name for f in img_dir.glob(f"*{config.image_ext}")]
                )
                masks: List[str] = sorted(
                    [
                        f.name
                        for f in (config.full_path / "annotations" / split_dir).glob(
                            f"*{config.mask_ext}"
                        )
                    ]
                )

                # Валидация соответствия
                if images and masks and set(images) != set(masks):
                    self._log(
                        f"⚠️ Несоответствие в {split_name}: {len(set(images) ^ set(masks))} файлов",
                        "warning",
                    )

                index["splits"][split_name] = {  # type: ignore[index]
                    "count": len(images),
                    "images": images[:10] + ["..."] if len(images) > 10 else images,
                }

        index_path: Path = config.full_path / "index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        self._log(f"📄 Создан индекс: {index_path}")

    # ──────────────────────────────────────────────────────────────────────
    def _validate_dataset(self, config: DatasetConfig) -> bool:
        """
        Гибкая валидация структуры датасета.

        Поддерживает:
        - Проверку наличия обязательных директорий.
        - Соответствие имён изображений и масок.
        - Валидацию для медицинских датасетов.
        - Проверку индексного файла от HF datasets.

        Args:
            config: Конфигурация датасета.

        Returns:
            bool: `True` если валидация пройдена, иначе `False`.
        """
        self._log(f"\n🔍 Валидация датасета {config.name}...")
        index_path: Path = config.full_path / "index.json"
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    index = json.load(f)
                if index.get("source") == "huggingface_datasets":
                    self._log("✅ Валидировано как HF datasets формат", "success")
                    return True
            except Exception:
                pass
        required_dirs: List[str] = [
            "images/training",
            "images/validation",
            "annotations/training",
            "annotations/validation",
        ]
        if config.dataset_type == DatasetType.MEDICAL_BINARY:
            has_images: bool = any(
                (config.full_path / "images").rglob(f"*{config.image_ext}")
            )
            has_masks: bool = any(
                (config.full_path / "annotations").rglob(f"*{config.mask_ext}")
            )
            if has_images and has_masks:
                self._log("✅ Структура медицинского датасета валидирована", "success")
                return True

        missing: List[str] = [
            d for d in required_dirs if not (config.full_path / d).exists()
        ]
        if missing:
            self._log(f"❌ Отсутствуют директории: {missing}", "error")
            return False
        for split in ["training", "validation"]:
            img_dir: Path = config.full_path / "images" / split
            ann_dir: Path = config.full_path / "annotations" / split
            if not img_dir.exists() or not ann_dir.exists():
                continue
            img_files: set[str] = {f.stem for f in img_dir.glob(f"*{config.image_ext}")}
            ann_files: set[str] = {f.stem for f in ann_dir.glob(f"*{config.mask_ext}")}

            if img_files != ann_files:
                missing_in_ann: set[str] = img_files - ann_files
                missing_in_img: set[str] = ann_files - img_files
                if missing_in_ann:
                    self._log(
                        f"⚠️ {len(missing_in_ann)} изображений без масок в {split}",
                        "warning",
                    )
                if missing_in_img:
                    self._log(
                        f"⚠️ {len(missing_in_img)} масок без изображений в {split}",
                        "warning",
                    )
        if config.expected_structure:
            for key, expected_files in config.expected_structure.items():
                actual_files: List[Path] = (
                    list((config.full_path / key).glob("*"))
                    if (config.full_path / key).exists()
                    else []
                )
                if (
                    len(actual_files) < len(expected_files) * 0.9
                ):  # Допускаем 10% потерь
                    self._log(
                        f"⚠️ Несоответствие файлов в {key}: ожидалось ~{len(expected_files)}, найдено {len(actual_files)}",
                        "warning",
                    )
        self._log("✅ Валидация пройдена", "success")
        return True

    # ──────────────────────────────────────────────────────────────────────
    def load_sample(
        self,
        dataset_name: str,
        split: str = "val",
        idx: int = 0,
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        """
        Загружает один пример (изображение + маска) из датасета.

        Args:
            dataset_name: Имя датасета.
            split: Название сплита ("train", "val", "test").
            idx: Индекс примера в сплите.

        Returns:
            Tuple[PIL.Image, Optional[PIL.Image]]: (изображение, маска или None).

        Raises:
            ValueError: Если директория сплита не найдена.
            IndexError: Если индекс выходит за границы.
        """
        config: DatasetConfig = self.get_config(dataset_name)

        split_dir: str = config.splits.get(split, split)
        img_dir: Path = config.full_path / "images" / split_dir
        ann_dir: Path = config.full_path / "annotations" / split_dir

        if not img_dir.exists():
            raise ValueError(f"Split directory not found: {img_dir}")

        images: List[Path] = sorted(list(img_dir.glob(f"*{config.image_ext}")))
        if not images or idx >= len(images):
            raise IndexError(f"Index {idx} out of range [0, {len(images)})")

        img: Image.Image = Image.open(images[idx]).convert("RGB")

        # Попытка загрузить соответствующую маску
        mask: Optional[Image.Image] = None
        mask_path: Path = ann_dir / images[idx].stem / config.mask_ext
        if mask_path.exists():
            mask = Image.open(mask_path)
        else:
            alt_mask: Path = ann_dir / (images[idx].stem + config.mask_ext)
            if alt_mask.exists():
                mask = Image.open(alt_mask)
        return img, mask

    # ──────────────────────────────────────────────────────────────────────
    def _create_index_from_hf_dataset(
        self, config: DatasetConfig, hf_dataset: Any
    ) -> None:
        """
        Создаёт индексный файл для датасета, загруженного через `datasets` library.

        Args:
            config: Конфигурация датасета.
            hf_dataset: Объект датасета из `datasets.load_dataset`.
        """
        index_path: Path = config.full_path / "index.json"
        index: Dict[str, Any] = {
            "dataset": config.name,
            "type": config.dataset_type.name,
            "num_classes": config.num_classes,
            "source": "huggingface_datasets",
            "created": datetime.now().isoformat(),
            "splits": {},
        }

        for split_name in hf_dataset.keys():
            split_data = hf_dataset[split_name]
            index["splits"][split_name] = {  # type: ignore[index]
                "count": len(split_data),
                "columns": (
                    list(split_data.features.keys())
                    if hasattr(split_data, "features")
                    else []
                ),
            }

        config.full_path.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        self._log(f"📄 Создан HF индекс: {index_path}")

    # ──────────────────────────────────────────────────────────────────────
    def load_test_image_from_hf(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        split: str = "validation",
    ) -> Optional[Image.Image]:
        """
        Универсальная загрузка тестового изображения из HuggingFace.

        Args:
            repo_id: ID репозитория (например, "cityscapes").
            filename: Конкретный файл (если `None`, берётся первый из сплита).
            split: Название сплита датасета.

        Returns:
            Optional[PIL.Image]: Изображение в режиме "RGB" или `None` при ошибке.
        """
        try:
            if filename:
                path: str = hf_hub_download(repo_id, filename, repo_type="dataset")
                return Image.open(path).convert("RGB")
            else:
                dataset = load_dataset(repo_id, split=split)
                if "image" in dataset.features:
                    return dataset[0]["image"].convert("RGB")
                else:
                    logger.warn(f"   ⚠️  Нет признака 'image' в {repo_id}")
                    return None
        except Exception as e:
            logger.error(f"   ❌ Ошибка загрузки {repo_id}: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────
    def create_pytorch_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> "torch.utils.data.Dataset":
        """
        Создаёт PyTorch Dataset для обучения на указанном датасете.

        Args:
            dataset_name: Имя датасета.
            split: Название сплита ("train", "val", "test").
            transform: Опциональная функция аугментаций (albumentations-style).

        Returns:
            torch.utils.data.Dataset: Dataset, возвращающий `(image_tensor, mask_tensor)`.
        """
        from torch.utils.data import Dataset

        config: DatasetConfig = self.get_config(dataset_name)

        class SegmentationDataset(Dataset):
            def __init__(
                self,
                config: DatasetConfig,
                split: str,
                transform: Optional[Callable] = None,
            ) -> None:
                self.config: DatasetConfig = config
                self.split_dir: str = config.splits.get(split, split)
                self.img_dir: Path = config.full_path / "images" / self.split_dir
                self.ann_dir: Path = config.full_path / "annotations" / self.split_dir
                self.transform: Optional[Callable] = transform

                self.images: List[Path] = sorted(
                    list(self.img_dir.glob(f"*{config.image_ext}"))
                )
                self._log(f"Загружено {len(self.images)} образцов из {split}")

            # ──────────────────────────────────────────────────────────────────────
            def _log(self, msg: str) -> None:
                if getattr(self, "verbose", True):
                    logger.info(f"[{self.config.name}/{self.split_dir}] {msg}")

            # ──────────────────────────────────────────────────────────────────────
            def __len__(self) -> int:
                return len(self.images)

            # ──────────────────────────────────────────────────────────────────────
            def __getitem__(self, idx: int) -> Dict[str, Any]:
                img_path: Path = self.images[idx]
                img_pil: Image.Image = Image.open(img_path).convert("RGB")

                mask_path: Path = self.ann_dir / (img_path.stem + self.config.mask_ext)
                if not mask_path.exists():
                    mask_path = self.ann_dir / img_path.stem / self.config.mask_ext

                mask_np: np.ndarray
                if mask_path.exists():
                    mask_pil = Image.open(mask_path)
                    if self.config.num_classes == 2:
                        mask_np = (np.array(mask_pil) > 0).astype(np.uint8)
                    else:
                        mask_np = np.array(mask_pil, dtype=np.uint8)
                else:
                    mask_np = np.zeros(
                        (img_pil.size[1], img_pil.size[0]), dtype=np.uint8
                    )

                if self.transform:
                    augmented = self.transform(image=np.array(img_pil), mask=mask_np)
                    img_np: np.ndarray = augmented["image"]
                    mask_np = augmented["mask"]  # type: ignore[assignment]
                    img_pil = Image.fromarray(img_np)
                else:
                    img_np = np.array(img_pil)

                from torchvision import transforms as T

                img_tensor: torch.Tensor = T.ToTensor()(img_pil)
                mask_tensor: torch.Tensor = torch.from_numpy(mask_np).long()
                return {
                    "image": img_tensor,
                    "mask": mask_tensor,
                    "image_id": img_path.name,
                }

        return SegmentationDataset(config, split, transform)


# ============================================================================
# УТИЛИТЫ ДЛЯ МЕДИЦИНСКИХ ДАТАСЕТОВ
# ============================================================================


class MedicalDatasetUtils:
    """Утилиты для работы с медицинскими изображениями"""

    @staticmethod
    def load_dicom_series(series_dir: Path) -> np.ndarray:
        """Загрузка DICOM серии в 3D numpy массив"""
        try:
            import pydicom
        except ImportError:
            raise ImportError("Install pydicom: pip install pydicom")

        dicom_files: List[Path] = sorted(series_dir.glob("*.dcm"))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {series_dir}")

        slices: List = []
        for dcm_path in dicom_files:
            ds = pydicom.dcmread(dcm_path)
            # Конвертация к Hounsfield units для CT
            if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                pixel_array = ds.pixel_array.astype(np.float32)
                pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                slices.append(pixel_array)
            else:
                slices.append(ds.pixel_array)
        return np.stack(slices, axis=0)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def load_nifti(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Загрузка NIfTI файла с метаданными"""
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("Install nibabel: pip install nibabel")

        nii = nib.load(str(path))
        data = nii.get_fdata()  # type: ignore[attr-defined]
        metadata: Dict[str, Any] = {
            "shape": data.shape,
            "affine": nii.affine.tolist(),  # type: ignore[attr-defined]
            "header": dict(nii.header),  # type: ignore[call-overload]
        }
        return data, metadata

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def window_ct(
        image: np.ndarray, window_center: float, window_width: float
    ) -> np.ndarray:
        """Применение CT windowing для визуализации"""
        min_val: float = window_center - window_width / 2
        max_val: float = window_center + window_width / 2
        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val) * 255
        return image.astype(np.uint8)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def save_for_training(
        image: np.ndarray,
        mask: np.ndarray,
        output_dir: Path,
        prefix: str,
        config: MedicalConfig,
    ) -> None:
        """Сохранение в формате, готовом для обучения"""
        output_dir.mkdir(parents=True, exist_ok=True)
        if config.modality == "X-Ray":
            if image.dtype != np.uint8:
                image = (
                    (image - image.min()) / (image.max() - image.min()) * 255
                ).astype(np.uint8)
            Image.fromarray(image).save(output_dir / f"{prefix}_img.png")
        else:
            # Для 3D: сохраняем как npz
            np.savez_compressed(output_dir / f"{prefix}.npz", image=image, mask=mask)

        # Сохранение маски
        if mask.ndim == 2:
            Image.fromarray(mask.astype(np.uint8)).save(
                output_dir / f"{prefix}_mask.png"
            )


# ============================================================================
# CLI ИНТЕРФЕЙС
# ============================================================================


def main():
    """CLI для загрузки датасетов"""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Manager for Segmentation")
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=list(DatasetManager._registry.keys()),
        help="Datasets to download",
    )
    parser.add_argument(
        "--all", action="store_true", help="Download all registered datasets"
    )
    parser.add_argument(
        "--medical-only", action="store_true", help="Download only medical datasets"
    )
    parser.add_argument(
        "--base-dir", default="./data", help="Base directory for datasets"
    )
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument(
        "--sample", type=str, help="Load and display sample: dataset_name:split:idx"
    )

    args = parser.parse_args()

    manager = DatasetManager(base_dir=args.base_dir)

    if args.list:
        logger.info("\n📋 Available datasets:")
        for name, config in manager._registry.items():
            med_tag = " [MEDICAL]" if isinstance(config, MedicalConfig) else ""
            logger.info(
                f"  • {name:15s} {med_tag:10s} {config.dataset_type.name:12s} [{config.source_type}]"
            )
            if isinstance(config, MedicalConfig):
                logger.info(
                    f"    └─ Modality: {config.modality}, Classes: {config.num_classes}"
                )
        return

    # Определение списка датасетов
    if args.all:
        datasets = list(manager._registry.keys())
    elif args.medical_only:
        datasets = [
            n for n, c in manager._registry.items() if isinstance(c, MedicalConfig)
        ]
    elif args.datasets:
        datasets = args.datasets
    else:
        parser.print_help()
        return

    # Загрузка
    for dataset_name in datasets:
        try:
            path = manager.download(dataset_name, force=args.force)
            logger.info(f"\n📁 {dataset_name} available at: {path}")
        except Exception as e:
            logger.error(f"\n❌ Failed to download {dataset_name}: {e}")
            continue

    # Загрузка примера
    if args.sample:
        try:
            name, split, idx = args.sample.split(":")
            img, mask = manager.load_sample(name, split, int(idx))
            logger.info(f"\n🖼️  Loaded sample from {name}/{split}#{idx}")
            logger.info(f"   Image: {img.size}, Mode: {img.mode}")
            if mask:
                logger.info(f"   Mask:  {mask.size}, Mode: {mask.mode}")
                output = Path(args.base_dir) / "samples"
                output.mkdir(exist_ok=True)
                img.save(output / f"{name}_{split}_{idx}_img.jpg")
                mask.save(output / f"{name}_{split}_{idx}_mask.png")
                logger.info(f"   Saved to {output}/")
        except Exception as e:
            logger.error(f"❌ Failed to load sample: {e}")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Manager for Segmentation")
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=list(DatasetManager._registry.keys()),
        help="Datasets to download",
    )
    parser.add_argument(
        "--all", action="store_true", help="Download all registered datasets"
    )
    parser.add_argument(
        "--medical-only", action="store_true", help="Download only medical datasets"
    )
    parser.add_argument(
        "--base-dir", default="./data", help="Base directory for datasets"
    )
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument(
        "--sample", type=str, help="Load and display sample: dataset_name:split:idx"
    )
    args = parser.parse_args()
    manager = DatasetManager(base_dir=args.base_dir)
    logger.info("\n Cityscapes Dataset...")
    cityscapes_img = manager.load_test_image_from_hf("Chris1/cityscapes", split="train")
    if cityscapes_img is not None:
        cityscapes_img.save("./../data/cityscapes_img.jpg")
    else:
        logger.warn("⚠️ Не удалось загрузить изображение cityscapes")
    logger.info("\n COCO Dataset...")
    coco_img = manager.load_test_image_from_hf("detection-datasets/coco", split="train")
    if coco_img is not None:
        coco_img.save("./../data/coco_img.jpg")
    logger.info("\n Medical Dataset (ISIC - Skin Lesion)...")
    isic_img = manager.load_test_image_from_hf(
        "researchjyotsna/isic2018_10", split="train"
    )
    if isic_img is not None:
        isic_img.save("./../data/isic_img.jpg")
    logger.info("\n Chest X-Ray Segmentation...")
    chest_x_ray = manager.load_test_image_from_hf("danjacobellis/chexpert")
    if chest_x_ray is not None:
        chest_x_ray.save("./../data/chest_x_ray.jpg")

    logger.info("\n" + "=" * 70)
    logger.info("✅ DATASET DOWNLOAD COMPLETE")
    logger.info("=" * 70)
    logger.info("\n📁 Available test images:")
    logger.info("   • ADE20K:     ADE20K_img.jpg")
    logger.info("   • Cityscapes: cityscapes_img.jpg")
    logger.info("   • COCO:       coco_img.jpg")
    logger.info("   • Medical:    isic_img.jpg")
    logger.info("   • Medical:    chest_x_ray.jpg")
