# datasets/load_datasets.py

"""
Менеджер загрузки и подготовки датасетов для сегментации
Поддержка: ADE20K, Cityscapes, COCO, ISIC, CheXpert + медицинские форматы
"""

# Импорт основных библиотек
from typing import (
    List, Union, Tuple, Dict, Optional, Literal, Callable,
    Any, TypeVar, Protocol, runtime_checkable
)
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum, auto
import os
import sys
import json
import hashlib
import zipfile
import tarfile
import shutil
import requests
from io import BytesIO
from datetime import datetime
import warnings

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

try:
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
    from datasets import load_dataset, Dataset as HFDataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

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

# Типы датасетов
class DatasetType(Enum):
    """Типы датасетов для сегментации"""
    SEMANTIC = "semantic" # ADE20K, Cityscapes
    INSTANCE = "instance" # COCO
    PANOPTIC = "panoptic"
    MEDICAL = "medical"
    MEDICAL_BINARY = "medical_binary" # ISIC, CheXpert (binary masks)
    MEDICAL_MULTICLASS = "medical_multiclass" # Multi-organ segmentation
    BINARY = "binary"
    CUSTOM = auto()

# Источник данных
class SourceType(Enum):
    HF = "hf"           # HuggingFace Hub
    ZIP = "zip"         # ZIP-архив
    TAR = "tar"         # TAR/GZ архив
    DIRECT = "direct"   # Прямая ссылка на файл/папку

class DataFormat(Enum):
    """Поддерживаемые форматы данных"""
    JPG = "jpg"
    PNG = "png"
    NIFTI = "nii.gz"  # Медицинские 3D
    DICOM = "dcm"     # DICOM серии
    NPZ = "npz"       # NumPy архивы
    HDF5 = "h5"       # HDF5 для больших данных


@dataclass
class DatasetConfig:
    """Конфигурация одного датасета"""
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
    splits: Dict[str, str] = field(default_factory=lambda: {
        "train": "training", "val": "validation", "test": "testing"
    })
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expected_structure: Dict[str, List[str]] = field(default_factory=dict)
    postprocess_script: Optional[str] = None
    
    @property
    def full_path(self) -> Path:
        return Path(self.root_dir) / self.name


@dataclass
class MedicalConfig(DatasetConfig):
    """Расширенная конфигурация для медицинских датасетов"""
    modality: Literal["X-Ray", "CT", "MRI", "Ultrasound", "Dermoscopy"] = "X-Ray"
    pixel_spacing: Optional[Tuple[float, float]] = None  # мм/пиксель
    intensity_normalization: Literal["minmax", "zscore", "histogram"] = "zscore"
    anatomical_regions: List[str] = field(default_factory=list)
    privacy_compliant: bool = True
    anatomy: Optional[str] = None
    task_type: str = "segmentation"  # segmentation, classification, detection

# ============================================================================
# БАЗОВЫЙ МЕНЕДЖЕР ДАТАСЕТОВ
# ============================================================================

class DatasetManager:
    """Универсальный менеджер загрузки и валидации датасетов"""
    
    # Глобальный реестр конфигураций
    _registry: Dict[str, DatasetConfig] = {}
    
    def __init__(self, base_dir: str = "./data", verbose: bool = True):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._register_default_datasets()
        self._load_configs()

    def _load_configs(self):
        """Загрузка конфигураций из YAML"""
        config_path = Path("configs/datasets.yaml")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                configs_data = yaml.safe_load(f)
                for name, cfg in configs_data.items():
                    if cfg.get('modality'):  # Медицинский датасет
                        self._registry[name] = MedicalConfig(name=name, **cfg)
                    else:
                        self._registry[name] = DatasetConfig(name=name, **cfg)
    
    def get_config(self, dataset_name: str) -> DatasetConfig:
        """Получение конфигурации датасета"""
        if dataset_name not in self._registry:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[dataset_name]
    
    def _log(self, message: str, level: str = "info"):
        """Логирование с цветным выводом"""
        colors = {
            "info": "\033[94m",
            "success": "\033[92m",
            "warning": "\033[93m",
            "error": "\033[91m",
            "reset": "\033[0m"
        }
        if not self.verbose:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(level, "•")
        print(f"[{timestamp}] {colors.get(level, '')}{prefix} {message}{colors['reset']}")
    
    def _check_disk_space(self, path: Path, required_gb: float) -> bool:
        """Проверка доступного места на диске"""
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)
        
        if free_gb < required_gb:
            self._log(f"❌ Недостаточно места: {free_gb:.1f}GB свободно, требуется {required_gb}GB", "error")
            return False
        return True
    
    def _compute_sha256(self, filepath: Path) -> str:
        """Вычисление SHA256 хеша файла"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _register_default_datasets(self):
        """Регистрация стандартных датасетов"""

        # ============================================================================
        # 1. ADE20K (Большой датасет)
        # ============================================================================
        
        # === ADE20K ===
        self._registry["ade20k"] = DatasetConfig(
            name="ade20k",
            dataset_type=DatasetType.SEMANTIC,
            source_type="zip",
            source_url="http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
            root_dir=str(self.base_dir),
            num_classes=150,
            checksum="7ff1be44964418441f542a7cc1e1a650e7dc0fc275f5d23252bc9bbdbc977b29",
            metadata={
                "license": "MIT",
                "citation": "Zhou et al., 2017",
                "homepage": "http://sceneparsing.csail.mit.edu/"
            }
        )

        # ============================================================================
        # 2. CITYSCAPES (городские сцены)
        # ============================================================================
        
        # === Cityscapes ===
        self._registry["cityscapes"] = DatasetConfig(
            name="cityscapes",
            dataset_type=DatasetType.SEMANTIC,
            source_type="hf",
            source_url="Chris1/cityscapes",
            root_dir=str(self.base_dir),
            num_classes=19,
            metadata={
                "license": "CC-BY-NC-SA 3.0",
                "requires_registration": True,
                "homepage": "https://www.cityscapes-dataset.com/"
            }
        )

        # ============================================================================
        # 3. COCO (общие объекты)
        # ============================================================================
        
        # === COCO ===
        self._registry["coco"] = DatasetConfig(
            name="coco",
            dataset_type=DatasetType.INSTANCE,
            source_type="hf",
            source_url="detection-datasets/coco",
            root_dir=str(self.base_dir),
            num_classes=80,
            metadata={
                "license": "CC BY 4.0",
                "homepage": "https://cocodataset.org/"
            }
        )

        # ============================================================================
        # 4. МЕДИЦИНСКИЙ ДАТАСЕТ (ISIC - skin lesion segmentation)
        # ============================================================================
        
        # === ISIC 2018 (Medical - Dermoscopy) ===
        self._registry["isic2018"] = MedicalConfig(
            name="isic2018",
            dataset_type=DatasetType.MEDICAL_BINARY,
            source_type="hf",
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
                "homepage": "https://challenge.isic-archive.com/"
            }
        )

        # ============================================================================
        # 5. Chest X-Ray
        # ============================================================================
        
        # === CheXpert (Medical - Chest X-Ray) ===
        self._registry["chexpert"] = MedicalConfig(
            name="chexpert",
            dataset_type=DatasetType.MEDICAL_BINARY,
            source_type="hf",
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
                "homepage": "https://stanfordmlgroup.github.io/competitions/chexpert/"
            }
        )
        self._log("✅ Готово! Датасеты зарегистрированы.", "success")
    
    def download(self, dataset_name: str, force: bool = False) -> Path:
        """Скачивание датасета с валидацией"""
        config = self.get_config(dataset_name)

        print(f"\n{'='*70}")
        print(f"📦 ЗАГРУЗКА ДАТАСЕТА: {config.name.upper()} ({config.dataset_type.name})...")
        print(f"{'='*70}")
        print(f"Тип: {config.dataset_type.name}")
        print(f"Источник: {config.source_type.upper()}")
        print(f"Классы: {config.num_classes}")
        if isinstance(config, MedicalConfig):
            print(f"Модальность: {config.modality}")
        print(f"Целевая директория: {config.full_path}")
        print(f"{'='*70}\n")
        
        if config.full_path.exists() and not force:
            if self._validate_dataset(config):
                self._log(f"✅ Датасет уже существует и валиден: {config.full_path}", "success")
                self._print_dataset_summary(config)
                return config.full_path
            else:
                self._log("⚠️ Существующий датасет не прошёл валидацию, перезагрузка...", "warning")
        
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
                self._log(f"\n⚠️ Датасет загружен, но валидация не пройдена", "warning")
                return config.full_path
                
        except Exception as e:
            self._log(f"❌ Ошибка загрузки: {e}", "error")
            raise

    def _print_dataset_summary(self, config: DatasetConfig):
        """Печать сводной информации о датасете"""
        print(f"\n📊 СВОДКА ПО ДАТАСЕТУ: {config.name.upper()}")
        print(f"{'='*50}")
        print(f"Тип: {config.dataset_type.value}")
        print(f"Классы: {config.num_classes}")
        if isinstance(config, MedicalConfig):
            print(f"Модальность: {config.modality}")
        print(f"Путь: {config.full_path}")
        
        for split_name, split_dir in config.splits.items():
            img_dir = config.full_path / "images" / split_dir
            ann_dir = config.full_path / "annotations" / split_dir
            
            if img_dir.exists():
                n_images = len(list(img_dir.glob(f"*{config.image_ext}")))
                n_masks = len(list(ann_dir.glob(f"*{config.mask_ext}"))) if ann_dir.exists() else 0
                print(f"   {split_name:12s}: {n_images:5d} images, {n_masks:5d} masks")
        
        print(f"{'-'*50}")
    
    def _download_huggingface(self, config: DatasetConfig):
        """Загрузка из HuggingFace Hub"""
        self._log(f"📥 Загрузка из HuggingFace Hub: {config.source_url}")
        local_dir = config.full_path
        
        try:
            if config.name in ["coco", "cityscapes"]:
                if not self._check_disk_space(self.base_dir, required_gb=20.0):
                    print("❌ Недостаточно места на диске для датасета")
                    return

                self._log(f"📊 Загрузка через datasets library...")
                hf_dataset = load_dataset(config.source_url, cache_dir=str(self.base_dir / ".cache"))
                self._log(f"✅ Загружено через datasets library")
                self._create_index_from_hf_dataset(config, hf_dataset)
                return
            
            print(f"📦 Скачивание файлов датасета...")
            snapshot_download(
                repo_id=config.source_url,
                repo_type="dataset",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"✅ Скачивание завершено")
            
        except Exception as e:
            error_msg = str(e)
            if config.name == "chexpert" and ("401" in error_msg or "Repository Not Found" in error_msg):
                self._log("❌ CheXpert requires manual download:", "error")
                self._log("   1. Register at https://stanfordmlgroup.github.io/competitions/chexpert/", "error")
                self._log("   2. Download CheXpert-v1.0.zip manually", "error")
                self._log("   3. Extract to data/chexpert/ and run validation again", "error")
                raise ValueError("CheXpert requires manual download due to access restrictions")
            
            self._log(f"⚠️ HF download failed, trying fallback: {error_msg}", "warning")
            self._hf_fallback_download(config)
    
    def _hf_fallback_download(self, config: DatasetConfig, use_api: bool = False):
        """Fallback для HF: пофайловая загрузка"""
        if use_api:
            return self._download_via_api(config)
        local_dir = config.full_path
        local_dir.mkdir(parents=True, exist_ok=True)

        repo_id = config.source_url
        files = list_repo_files(repo_id, repo_type="dataset")
        
        # Фильтруем только нужные файлы
        image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
        mask_files = [f for f in files if f.endswith(('.png', '.nii.gz'))]
        
        for file_list, desc in [(image_files, "🖼️  Изображения"), (mask_files, "🎭 Маски")]:
            if not file_list:
                continue
            print(f"{desc}: {len(file_list)} файлов")
            for filename in tqdm(file_list, desc=f"{config.name}/{desc.split()[1]}", unit="files"):
                local_path = local_dir / filename
                local_path.parent.mkdir(parents=True, exist_ok=True)
                if not local_path.exists():
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type="dataset",
                        local_dir=str(local_dir)
                    )

    def _download_via_api(self, config: DatasetConfig):
        """Экспериментальный метод через HF API"""
        import requests
        from huggingface_hub import hf_hub_download
        
        repo_id = config.source_url
        local_dir = config.full_path
        
        try:
            # Получаем метаданные через API
            api_url = f"https://huggingface.co/api/datasets/{repo_id}"
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            files_info = response.json().get('siblings', [])
            target_files = [f['rfilename'] for f in files_info 
                        if f['rfilename'].endswith(('.jpg', '.png'))]
            
            for filename in tqdm(target_files, desc=f"📥 {config.name}", unit="files"):
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(local_dir)
                )
        except Exception as e:
            print(f"⚠️ API-скачивание не удалось: {e}. Переключаюсь на стандартный метод...")
            # Fallback на стандартный метод
            return self._hf_fallback_download(config, use_api=False)
    
    def _download_zip(self, config: DatasetConfig):
        """Скачивание ZIP-архива с прогрессом и валидацией"""
        zip_path = config.full_path.with_suffix(".zip")
        print(f"📦 ЗАГРУЗКА АРХИВА: {config.name}")
        print(f"{'-'*50}")
        
        if not zip_path.exists():
            self._log(f"📥 Скачивание архива...")
            self._log(f"   URL: {config.source_url}")
            if config.checksum:
                print(f"   SHA256: {config.checksum[:16]}...")
            try:
                head = requests.head(config.source_url, allow_redirects=True)
                size_bytes = int(head.headers.get('content-length', 0))
                print(f"   Размер: ~{size_bytes / (1024*1024*1024):.1f} GB")
            except:
                print("   Размер: неизвестен")
            self._streaming_download(config.source_url, zip_path, config.checksum)
            zip_size = os.path.getsize(zip_path) / (1024*1024*1024)
            print(f"✅ Скачивание завершено! Размер: {zip_size:.2f} GB")
        else:
            zip_size = os.path.getsize(zip_path) / (1024*1024*1024)
            print(f"✅ Архив уже существует: {zip_path} ({zip_size:.2f} GB)")
        
        extract_dir = config.full_path / "temp_extract"
        if not (config.full_path / "images").exists():
            self._log(f"\n📦 Распаковка архива...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                print(f"   Всего файлов в архиве: {len(members)}")
                for member in tqdm(members, desc="Распаковка", unit="files"):
                    # Безопасная распаковка (защита от path traversal)
                    target_path = extract_dir / member
                    if not str(target_path.resolve()).startswith(str(extract_dir.resolve())):
                        raise ValueError(f"Unsafe path in archive: {member}")
                    zip_ref.extract(member, extract_dir)
            print(f"✅ Распаковка завершена!")
            print(f"\n🔍 Анализ структуры распакованных файлов...")
            self._reorganize_ade_structure(extract_dir, config.full_path)
            print(f"\n🧹 Очистка временных файлов...")
            shutil.rmtree(extract_dir, ignore_errors=True)
            if config.checksum:
                zip_path.unlink()
                print(f"✅ Архив удалён")
        else:
            print(f"✅ Датасет уже распакован в {config.full_path}")

    def _download_tar(self, config: DatasetConfig):
        """Скачивание TAR/GZ архива"""
        tar_path = config.full_path.with_suffix(".tar.gz")
        print(f"📦 ЗАГРУЗКА TAR-АРХИВА: {config.name}")
        
        if not tar_path.exists():
            self._log(f"📥 Скачивание архива...")
            self._streaming_download(config.source_url, tar_path, config.checksum)
            print(f"✅ Скачивание завершено")
        
        # Распаковка
        extract_dir = config.full_path / "temp_extract"
        if not (config.full_path / "images").exists():
            self._log(f"\n📦 Распаковка TAR-архива...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(tar_path, 'r:gz') as tar_ref:
                members = tar_ref.getmembers()
                print(f"   Всего файлов: {len(members)}")
                for member in tqdm(members, desc="Распаковка", unit="files"):
                    target_path = extract_dir / member.name
                    if not str(target_path.resolve()).startswith(str(extract_dir.resolve())):
                        raise ValueError(f"Unsafe path in archive: {member.name}")
                    tar_ref.extract(member, extract_dir)
            
            self._reorganize_ade_structure(extract_dir, config.full_path)
            shutil.rmtree(extract_dir, ignore_errors=True)
            if config.checksum:
                tar_path.unlink()
        else:
            print(f"✅ Датасет уже распакован")

    def _download_direct(self, config: DatasetConfig):
        """Прямая загрузка файлов"""
        self._log(f"📥 Прямая загрузка: {config.source_url}")
        target_dir = config.full_path
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Если ссылка на файл
        if config.source_url.endswith(('.jpg', '.png', '.npy', '.zip')):
            filename = config.source_url.split('/')[-1]
            filepath = target_dir / filename
            self._streaming_download(config.source_url, filepath, config.checksum)
        else:
            # Если ссылка на директорию (рекурсивное скачивание)
            self._log("⚠️ Прямая загрузка директорий требует дополнительной реализации", "warning")
    
    def _streaming_download(self, url: str, destination: Path, expected_checksum: Optional[str] = None):
        """Потоковая загрузка с прогрессом и проверкой контрольной суммы"""
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        destination.parent.mkdir(parents=True, exist_ok=True)
        # sha256_hash = hashlib.sha256() if expected_checksum else None
        
        with open(destination, 'wb') as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Валидация контрольной суммы
        if expected_checksum:
            actual_checksum = self._compute_sha256(destination)
            if actual_checksum != expected_checksum:
                destination.unlink()
                raise ValueError(
                    f"❌ Checksum mismatch!\n"
                    f"Expected: {expected_checksum}\n"
                    f"Actual:   {actual_checksum}"
                )
        self._log("✅ Контрольная сумма совпадает", "success")
    
    def _reorganize_ade_structure(self, source: Path, target: Path, use_symlinks: bool = Fals):
        """Приведение структуры к стандартному формату"""
        found_structure = False
        for root, dirs, _ in os.walk(source):
            if 'images' in dirs and 'annotations' in dirs:
                src_images = Path(root) / 'images'
                src_annotations = Path(root) / 'annotations'
                print(f"✅ Найдена структура в: {root}")
                found_structure = True
                
                for split in ['training', 'validation']:
                    src_img_split = src_images / split
                    src_ann_split = src_annotations / split
                    
                    if src_img_split.exists():
                        tgt_img_split = target / 'images' / split
                        tgt_ann_split = target / 'annotations' / split
                        tgt_img_split.mkdir(parents=True, exist_ok=True)
                        tgt_ann_split.mkdir(parents=True, exist_ok=True)
                        
                        # Копирование с прогрессом
                        img_files = list(src_img_split.glob('*.jpg'))
                        ann_files = list(src_ann_split.glob('*.png'))
                        
                        print(f"\n   📋 {split}:")
                        print(f"      Изображения: {len(img_files)} файлов")
                        print(f"      Маски: {len(ann_files)} файлов")
                        
                        for img_file in tqdm(img_files, desc=f"   Копирование {split}/images", unit="files"):
                            shutil.copy2(img_file, tgt_img_split / img_file.name)
                        
                        for ann_file in tqdm(ann_files, desc=f"   Копирование {split}/annotations", unit="files"):
                            shutil.copy2(ann_file, tgt_ann_split / ann_file.name)
                print(f"\n✅ Файлы организованы в: {target}")
                return
        if not found_structure:
            print(f"❌ Не удалось найти папки images и annotations")
            print(f"\n📂 Содержимое распакованной папки:")
            for root, dirs, files in os.walk(source):
                level = root.replace(str(source), '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{Path(root).name}/')
                subindent = ' ' * 2 * (level + 1)
                for d in dirs[:5]:
                    print(f'{subindent}📁 {d}/')
                for f in files[:5]:
                    print(f'{subindent}📄 {f}')
            raise ValueError("Could not find images/annotations structure in archive")
        
        if use_symlinks and sys.platform != "win32":  # Windows требует прав администратора
            try:
                for split in ['training', 'validation']:
                    src = target / split
                    tgt = target / 'images' / split
                    if src.exists() and not tgt.exists():
                        tgt.parent.mkdir(parents=True, exist_ok=True)
                        os.symlink(src.relative_to(tgt.parent), tgt)
                        print(f"   🔗 Создана ссылка: {tgt} -> {src}")
            except (OSError, PermissionError) as e:
                print(f"⚠️ Не удалось создать символические ссылки: {e}")
    
    def _postprocess_dataset(self, config: DatasetConfig):
        """Пост-обработка с поддержкой Parquet"""
        self._log(f"\n🔧 Пост-обработка датасета...")
        
        parquet_files = list(config.full_path.rglob("*.parquet"))
        if parquet_files:
            print(f"📊 Обнаружено Parquet-файлов: {len(parquet_files)}")
            self._convert_parquet_to_files(config)
        
        if isinstance(config, MedicalConfig):
            print(f"🏥 Применение медицинской пост-обработки...")
            self._medical_postprocess(config)
        
        print(f"📄 Создание индексного файла...")
        self._create_index(config)
        print(f"✅ Пост-обработка завершена")

    def _decode_image_from_hf(self, data, convert_to_rgb: bool = True):
        """
        Универсальный декодер для данных из HuggingFace datasets.
        Поддерживает: datasets.Image, bytes, dict{'bytes'}, base64, PIL.Image
        """
        if data is None:
            return None
        
        # 🔹 datasets.Image объект (имеет метод convert)
        if hasattr(data, 'convert') and hasattr(data, 'mode'):
            return data.convert('RGB') if convert_to_rgb and data.mode != 'RGB' else data
        
        # 🔹 bytes объект
        if isinstance(data, bytes) and len(data) > 0:
            try:
                from PIL import Image
                from io import BytesIO
                return Image.open(BytesIO(data)).convert('RGB') if convert_to_rgb else Image.open(BytesIO(data))
            except:
                return None
        
        # 🔹 dict с 'bytes' или 'path' (формат HF datasets)
        if isinstance(data, dict):
            if 'bytes' in data and data['bytes'] and isinstance(data['bytes'], bytes):
                try:
                    from PIL import Image
                    from io import BytesIO
                    return Image.open(BytesIO(data['bytes'])).convert('RGB') if convert_to_rgb else Image.open(BytesIO(data['bytes']))
                except:
                    return None
            if 'path' in data and data['path'] and os.path.exists(data['path']):
                try:
                    from PIL import Image
                    return Image.open(data['path']).convert('RGB') if convert_to_rgb else Image.open(data['path'])
                except:
                    return None
        
        # 🔹 base64 строка
        if isinstance(data, str) and data.startswith('data:image'):
            try:
                import base64
                from PIL import Image
                from io import BytesIO
                header, encoded = data.split(',', 1)
                return Image.open(BytesIO(base64.b64decode(encoded))).convert('RGB') if convert_to_rgb else Image.open(BytesIO(base64.b64decode(encoded)))
            except:
                return None
        
        # 🔹 PIL.Image (уже готов)
        if hasattr(data, 'save') and hasattr(data, 'mode'):
            return data.convert('RGB') if convert_to_rgb and data.mode != 'RGB' else data
        
        return None

    def _convert_parquet_to_files(self, config: DatasetConfig):
        """Конвертация Parquet → файловая структура с полной поддержкой HF форматов"""
        try:
            import pandas as pd
            from PIL import Image
            from io import BytesIO
            import base64
        except ImportError as e:
            self._log(f"⚠️ Missing dependencies for Parquet conversion: {e}", "warning")
            return
        
        self._log(f"\n🔄 Конвертация Parquet → файловая структура для {config.name}...")
        
        parquet_files = list(config.full_path.rglob("*.parquet"))
        if not parquet_files:
            self._log(f"⚠️ Parquet-файлы не найдены", "warning")
            return
        print(f"📁 Найдено Parquet-файлов: {len(parquet_files)}")
        
        # Создаём целевые директории
        for split in ["training", "validation", "train", "val", "test"]:
            (config.full_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (config.full_path / "annotations" / split).mkdir(parents=True, exist_ok=True)
        
        image_keys = ["image", "images", "img", "input", "pixel_values", "source"]
        mask_keys = ["mask", "masks", "annotation", "segmentation_mask", "label", "ground_truth", "segmentation"]
        split_keys = ["split", "phase", "mode", "subset"]
        
        converted_count = 0
        error_count = 0
        total_rows = 0
        
        for pq_file in parquet_files:
            try:
                df = pd.read_parquet(pq_file)
                total_rows += len(df)
                self._log(f"\n📊 Обработка {pq_file.name}: {len(df)} строк")
                img_key = next((k for k in image_keys if k in df.columns), None)
                mask_key = next((k for k in mask_keys if k in df.columns), None)
                split_key = next((k for k in split_keys if k in df.columns), None)
                
                if not img_key:
                    self._log(f"⚠️ Нет колонки с изображениями в {pq_file.name}. Доступные: {list(df.columns)}", "warning")
                    continue

                print(f"   🖼️  Ключ изображения: '{img_key}'")
                if mask_key:
                    print(f"   🎭 Ключ маски: '{mask_key}'")
                if split_key:
                    print(f"   📑 Ключ split: '{split_key}'")
                
                for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Конвертация", unit="rows"):
                    if converted_count >= 1000 and not self.verbose:
                        break
                        
                    try:
                        if split_key and split_key in row:
                            split_val = str(row[split_key]).lower()
                            if split_val in ["train", "training"]: split = "training"
                            elif split_val in ["val", "validation", "valid"]: split = "validation"
                            elif split_val in ["test", "testing"]: split = "testing"
                            else: split = "training"
                        else:
                            fname = pq_file.name.lower()
                            if "train" in fname: split = "training"
                            elif "val" in fname or "valid" in fname: split = "validation"
                            elif "test" in fname: split = "testing"
                            else: split = "training"
                        
                        # === Декодирование изображения ===
                        img = self._decode_image_from_hf(row[img_key], convert_to_rgb=True)
                        if img is None:
                            error_count += 1
                            continue
                        
                        # === Декодирование маски (если есть) ===
                        mask = None
                        if mask_key and mask_key in row and row[mask_key] is not None:
                            mask = self._decode_image_from_hf(row[mask_key], convert_to_rgb=False)
                            if mask and mask.mode != 'L':
                                mask = mask.convert('L')
                        
                        # === Сохранение ===
                        filename = f"{config.name}_{split}_{idx:06d}"
                        img_path = config.full_path / "images" / split / f"{filename}.jpg"
                        img.save(img_path, quality=95)
                        
                        if mask:
                            mask_path = config.full_path / "annotations" / split / f"{filename}.png"
                            mask.save(mask_path)
                        
                        converted_count += 1
                        
                    except Exception as e:
                        error_count += 1
                        if self.verbose:
                            self._log(f"   ⚠️ Ошибка в строке {idx}: {type(e).__name__}", "warning")
                        continue
                            
            except Exception as e:
                self._log(f"⚠️ Ошибка обработки {pq_file.name}: {e}", "warning")
                continue
        
        self._log(f"✅ Converted {converted_count} samples, {error_count} errors", "success")
        
        print(f"\n{'='*50}")
        print(f"📊 СТАТИСТИКА КОНВЕРТАЦИИ")
        print(f"{'-'*50}")
        print(f"   Всего обработано строк: {total_rows}")
        print(f"   ✅ Успешно конвертировано: {converted_count}")
        print(f"   ❌ Ошибок: {error_count}")
        if total_rows > 0:
            success_rate = (converted_count / total_rows) * 100
            print(f"   📈 Успешность: {success_rate:.1f}%")
        print(f"{'-'*50}")
        self._create_index(config)
        print(f"✅ Конвертация завершена")
        
    def _medical_postprocess(self, config: MedicalConfig):
        """Специальная обработка медицинских датасетов"""
        self._log(f"🏥 Применение медицинской пост-обработки: {config.intensity_normalization}")
        
        # Пример: нормализация интенсивности для рентгена
        if config.modality == "X-Ray" and config.intensity_normalization == "histogram":
            self._apply_clahe(config.full_path / "images")
        
        # Конвертация масок к единому формату (0/1 или 0/255)
        self._normalize_masks(config.full_path / "annotations", config.num_classes)
    
    def _apply_clahe(self, images_dir: Path, clip_limit: float = 2.0):
        """Применение CLAHE для улучшения контраста рентгеновских снимков"""
        try:
            import cv2
        except ImportError:
            self._log("⚠️ OpenCV не установлен, пропускаем CLAHE", "warning")
            return
        
        print(f"🔧 Применение CLAHE к изображениям...")
        for img_path in tqdm(list(images_dir.rglob("*.jpg")), desc="CLAHE", unit="imgs"):
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
                enhanced = clahe.apply(img)
                cv2.imwrite(str(img_path), enhanced)
            except Exception as e:
                self._log(f"⚠️ Ошибка обработки {img_path.name}: {e}", "warning")
    
    def _normalize_masks(self, masks_dir: Path, num_classes: int):
        """Приведение масок к единому формату"""
        print(f"🎭 Нормализация масок...")
        for mask_path in tqdm(list(masks_dir.rglob("*.png")), desc="Normalizing masks"):
            try:
                mask = np.array(Image.open(mask_path))
                
                # Для бинарных медицинских масок: всё кроме 0 -> 1
                if num_classes == 2:
                    mask = (mask > 0).astype(np.uint8)
                
                # Сохранение с явным указанием типа
                Image.fromarray(mask).save(mask_path)
            except Exception as e:
                self._log(f"⚠️ Failed to normalize {mask_path.name}: {e}", "warning")
    
    def _create_index(self, config: DatasetConfig):
        """Создание индексного файла для быстрого доступа"""
        index = {
            "dataset": config.name,
            "type": config.dataset_type.name,
            "created": datetime.now().isoformat(),
            "splits": {}
        }
        
        for split_name, split_dir in config.splits.items():
            img_dir = config.full_path / "images" / split_dir
            if img_dir.exists():
                images = sorted([f.name for f in img_dir.glob(f"*{config.image_ext}")])
                masks = sorted([f.name for f in (config.full_path / "annotations" / split_dir).glob(f"*{config.mask_ext}")])
                
                # Валидация соответствия
                if images and masks and set(images) != set(masks):
                    self._log(f"⚠️ Несоответствие в {split_name}: {len(set(images) ^ set(masks))} файлов", "warning")
                
                index["splits"][split_name] = {
                    "count": len(images),
                    "images": images[:10] + ["..."] if len(images) > 10 else images
                }
        
        index_path = config.full_path / "index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        self._log(f"📄 Создан индекс: {index_path}")
    
    def _validate_dataset(self, config: DatasetConfig) -> bool:
        """Гибкая валидация с поддержкой разных форматов"""
        self._log(f"\n🔍 Валидация датасета {config.name}...")
        index_path = config.full_path / "index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    index = json.load(f)
                if index.get("source") == "huggingface_datasets":
                    self._log("✅ Валидировано как HF datasets формат", "success")
                    return True
            except:
                pass
        required_dirs = ["images/training", "images/validation", "annotations/training", "annotations/validation"]
        if config.dataset_type == DatasetType.MEDICAL_BINARY:
            has_images = any((config.full_path / "images").rglob(f"*{config.image_ext}"))
            has_masks = any((config.full_path / "annotations").rglob(f"*{config.mask_ext}"))
            if has_images and has_masks:
                self._log(f"✅ Структура медицинского датасета валидирована", "success")
                return True
        
        missing = [d for d in required_dirs if not (config.full_path / d).exists()]
        if missing:
            self._log(f"❌ Отсутствуют директории: {missing}", "error")
            return False
        for split in ["training", "validation"]:
            img_dir = config.full_path / "images" / split
            ann_dir = config.full_path / "annotations" / split
            if not img_dir.exists() or not ann_dir.exists():
                continue  
            img_files = {f.stem for f in img_dir.glob(f"*{config.image_ext}")}
            ann_files = {f.stem for f in ann_dir.glob(f"*{config.mask_ext}")}
            
            if img_files != ann_files:
                missing_in_ann = img_files - ann_files
                missing_in_img = ann_files - img_files
                if missing_in_ann:
                    self._log(f"⚠️ {len(missing_in_ann)} изображений без масок в {split}", "warning")
                if missing_in_img:
                    self._log(f"⚠️ {len(missing_in_img)} масок без изображений в {split}", "warning")
        if config.expected_structure:
            for key, expected_files in config.expected_structure.items():
                actual_files = list((config.full_path / key).glob("*")) if (config.full_path / key).exists() else []
                if len(actual_files) < len(expected_files) * 0.9:  # Допускаем 10% потерь
                    self._log(f"⚠️ Несоответствие файлов в {key}: ожидалось ~{len(expected_files)}, найдено {len(actual_files)}", "warning")
        self._log(f"✅ Валидация пройдена", "success")
        return True
    
    def load_sample(self, dataset_name: str, split: str = "val", idx: int = 0) -> Tuple[Image.Image, Optional[Image.Image]]:
        """Загрузка одного примера из датасета"""
        config = self.get_config(dataset_name)
        
        split_dir = config.splits.get(split, split)
        img_dir = config.full_path / "images" / split_dir
        ann_dir = config.full_path / "annotations" / split_dir
        
        if not img_dir.exists():
            raise ValueError(f"Split directory not found: {img_dir}")
        
        images = sorted(list(img_dir.glob(f"*{config.image_ext}")))
        if not images or idx >= len(images):
            raise IndexError(f"Index {idx} out of range [0, {len(images)})")
        
        img = Image.open(images[idx]).convert("RGB")
        
        # Попытка загрузить соответствующую маску
        mask = None
        mask_path = ann_dir / images[idx].stem / config.mask_ext
        if mask_path.exists():
            mask = Image.open(mask_path)
        else:
            alt_mask = ann_dir / (images[idx].stem + config.mask_ext)
            if alt_mask.exists():
                mask = Image.open(alt_mask)   
        return img, mask
    
    def _create_index_from_hf_dataset(self, config: DatasetConfig, hf_dataset):
        """Создание index.json для датасета из datasets library"""
        index_path = config.full_path / "index.json"
        index = {
            "dataset": config.name,
            "type": config.dataset_type.name,
            "num_classes": config.num_classes,
            "source": "huggingface_datasets",
            "created": datetime.now().isoformat(),
            "splits": {}
        }
        
        for split_name in hf_dataset.keys():
            split_data = hf_dataset[split_name]
            index["splits"][split_name] = {
                "count": len(split_data),
                "columns": list(split_data.features.keys()) if hasattr(split_data, 'features') else []
            }
        
        config.full_path.mkdir(parents=True, exist_ok=True)
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        self._log(f"📄 Создан HF индекс: {index_path}")
    
    def load_test_image_from_hf(
        self,
        repo_id: str, 
        filename: str = None, 
        split: str = "validation"
    ) -> Optional[Image.Image | None]:
        """
        Универсальная загрузка тестового изображения из HuggingFace.
        
        Args:
            repo_id: ID репозитория (например, "cityscapes")
            filename: конкретный файл (если None, берем первый из split)
            split: split датасета
        
        Returns:
            PIL.Image или None
        """
        try:
            if filename:
                path = hf_hub_download(repo_id, filename, repo_type="dataset")
                return Image.open(path).convert("RGB")
            else:
                dataset = load_dataset(repo_id, split=split)
                if 'image' in dataset.features:
                    return dataset[0]['image'].convert("RGB")
                else:
                    print(f"   ⚠️  Нет признака 'image' в {repo_id}")
                    return None
        except Exception as e:
            print(f"   ❌ Ошибка загрузки {repo_id}: {e}")
            return None
    
    def create_pytorch_dataset(self, dataset_name: str, split: str = "train", 
                              transform: Optional[Callable] = None):
        """Создание PyTorch Dataset для обучения"""
        from torch.utils.data import Dataset
        
        config = self.get_config(dataset_name)
        
        class SegmentationDataset(Dataset):
            def __init__(self, config: DatasetConfig, split: str, transform=None):
                self.config = config
                self.split_dir = config.splits.get(split, split)
                self.img_dir = config.full_path / "images" / self.split_dir
                self.ann_dir = config.full_path / "annotations" / self.split_dir
                self.transform = transform
                
                self.images = sorted(list(self.img_dir.glob(f"*{config.image_ext}")))
                self._log(f"Загружено {len(self.images)} образцов из {split}")
            
            def _log(self, msg):
                if getattr(self, 'verbose', True):
                    print(f"[{self.config.name}/{self.split_dir}] {msg}")
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                img_path = self.images[idx]
                img = Image.open(img_path).convert("RGB")
                
                mask_path = self.ann_dir / (img_path.stem + self.config.mask_ext)
                if not mask_path.exists():
                    mask_path = self.ann_dir / img_path.stem / self.config.mask_ext
                
                if mask_path.exists():
                    mask = Image.open(mask_path)
                    if self.config.num_classes == 2:
                        mask = np.array(mask) > 0
                    else:
                        mask = np.array(mask)
                else:
                    mask = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
                
                if self.transform:
                    augmented = self.transform(image=np.array(img), mask=mask)
                    img = Image.fromarray(augmented['image'])
                    mask = augmented['mask']
                
                # Конвертация к тензорам
                from torchvision import transforms as T
                img_tensor = T.ToTensor()(img)
                mask_tensor = torch.from_numpy(mask).long() if isinstance(mask, np.ndarray) else mask
                return img_tensor, mask_tensor
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
        
        dicom_files = sorted(series_dir.glob("*.dcm"))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {series_dir}")
        
        slices = []
        for dcm_path in dicom_files:
            ds = pydicom.dcmread(dcm_path)
            # Конвертация к Hounsfield units для CT
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_array = ds.pixel_array.astype(np.float32)
                pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                slices.append(pixel_array)
            else:
                slices.append(ds.pixel_array)
        return np.stack(slices, axis=0)
    
    @staticmethod
    def load_nifti(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Загрузка NIfTI файла с метаданными"""
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("Install nibabel: pip install nibabel")
        
        nii = nib.load(str(path))
        data = nii.get_fdata()
        metadata = {
            "shape": data.shape,
            "affine": nii.affine.tolist(),
            "header": dict(nii.header)
        }
        return data, metadata
    
    @staticmethod
    def window_ct(image: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
        """Применение CT windowing для визуализации"""
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val) * 255
        return image.astype(np.uint8)
    
    @staticmethod
    def save_for_training(image: np.ndarray, mask: np.ndarray, 
                         output_dir: Path, prefix: str, 
                         config: MedicalConfig):
        """Сохранение в формате, готовом для обучения"""
        output_dir.mkdir(parents=True, exist_ok=True)
        if config.modality == "X-Ray":
            if image.dtype != np.uint8:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            Image.fromarray(image).save(output_dir / f"{prefix}_img.png")
        else:
            # Для 3D: сохраняем как npz
            np.savez_compressed(output_dir / f"{prefix}.npz", image=image, mask=mask)
        
        # Сохранение маски
        if mask.ndim == 2:
            Image.fromarray(mask.astype(np.uint8)).save(output_dir / f"{prefix}_mask.png")


# ============================================================================
# CLI ИНТЕРФЕЙС
# ============================================================================

def main():
    """CLI для загрузки датасетов"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Manager for Segmentation")
    parser.add_argument("datasets", nargs="*", 
                       choices=list(DatasetManager._registry.keys()),
                       help="Datasets to download")
    parser.add_argument("--all", action="store_true", help="Download all registered datasets")
    parser.add_argument("--medical-only", action="store_true", help="Download only medical datasets")
    parser.add_argument("--base-dir", default="./data", help="Base directory for datasets")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--sample", type=str, help="Load and display sample: dataset_name:split:idx")
    
    args = parser.parse_args()
    
    manager = DatasetManager(base_dir=args.base_dir)
    
    if args.list:
        print("\n📋 Available datasets:")
        for name, config in manager._registry.items():
            med_tag = " [MEDICAL]" if isinstance(config, MedicalConfig) else ""
            print(f"  • {name:15s} {med_tag:10s} {config.dataset_type.name:12s} [{config.source_type}]")
            if isinstance(config, MedicalConfig):
                print(f"    └─ Modality: {config.modality}, Classes: {config.num_classes}")
        return
    
    # Определение списка датасетов
    if args.all:
        datasets = list(manager._registry.keys())
    elif args.medical_only:
        datasets = [n for n, c in manager._registry.items() if isinstance(c, MedicalConfig)]
    elif args.datasets:
        datasets = args.datasets
    else:
        parser.print_help()
        return
    
    # Загрузка
    for dataset_name in datasets:
        try:
            path = manager.download(dataset_name, force=args.force)
            print(f"\n📁 {dataset_name} available at: {path}")
        except Exception as e:
            print(f"\n❌ Failed to download {dataset_name}: {e}")
            continue
    
    # Загрузка примера
    if args.sample:
        try:
            name, split, idx = args.sample.split(":")
            img, mask = manager.load_sample(name, split, int(idx))
            print(f"\n🖼️  Loaded sample from {name}/{split}#{idx}")
            print(f"   Image: {img.size}, Mode: {img.mode}")
            if mask:
                print(f"   Mask:  {mask.size}, Mode: {mask.mode}")
                output = Path(args.base_dir) / "samples"
                output.mkdir(exist_ok=True)
                img.save(output / f"{name}_{split}_{idx}_img.jpg")
                mask.save(output / f"{name}_{split}_{idx}_mask.png")
                print(f"   Saved to {output}/")
        except Exception as e:
            print(f"❌ Failed to load sample: {e}")


if __name__ == "__main__":
    main()
    import argparse
    parser = argparse.ArgumentParser(description="Dataset Manager for Segmentation")
    parser.add_argument("datasets", nargs="*", 
                       choices=list(DatasetManager._registry.keys()),
                       help="Datasets to download")
    parser.add_argument("--all", action="store_true", help="Download all registered datasets")
    parser.add_argument("--medical-only", action="store_true", help="Download only medical datasets")
    parser.add_argument("--base-dir", default="./data", help="Base directory for datasets")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--sample", type=str, help="Load and display sample: dataset_name:split:idx")
    args = parser.parse_args()
    manager = DatasetManager(base_dir=args.base_dir)
    print("\n Cityscapes Dataset...")
    cityscapes_img = manager.load_test_image_from_hf("Chris1/cityscapes", split="train")
    cityscapes_img.save("./../data/cityscapes_img.jpg")
    print("\n COCO Dataset...")
    coco_img = manager.load_test_image_from_hf("detection-datasets/coco", split="train")
    coco_img.save("./../data/coco_img.jpg")
    print("\n Medical Dataset (ISIC - Skin Lesion)...")
    isic_img = manager.load_test_image_from_hf("researchjyotsna/isic2018_10", split="train")
    isic_img.save("./../data/isic_img.jpg")
    print("\n Chest X-Ray Segmentation...")
    chest_x_ray = manager.load_test_image_from_hf("danjacobellis/chexpert")
    chest_x_ray.save("./../data/chest_x_ray.jpg")

    print("\n" + "=" * 70)
    print("✅ DATASET DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"\n📁 Available test images:")
    print(f"   • ADE20K:     ADE20K_img.jpg")
    print(f"   • Cityscapes: cityscapes_img.jpg")
    print(f"   • COCO:       coco_img.jpg")
    print(f"   • Medical:    isic_img.jpg")
    print(f"   • Medical:    chest_x_ray.jpg")