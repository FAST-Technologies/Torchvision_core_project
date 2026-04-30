# segmenters/ModelTrainer.py

# Импорт основных библиотек
import os
import sys
import time
import gc
import glob
from pathlib import Path
from datetime import datetime
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    Literal,
    TypedDict,
    NotRequired,
    Union,
    cast,
)
from dataclasses import dataclass
from matplotlib.colors import Colormap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torchvision.models.segmentation as tv_seg
from sklearn.metrics import jaccard_score

# Локальные импорты
from dataseters.ADE20KDataset import ADE20KDataset
from .NeuralTrainer import NeuralTrainer
from utils.strategies import SegNet

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & CONSTANTS
# ──────────────────────────────────────────────────────────────────────
DEFAULT_ROOT_DIR: str = "./data/ade20k"
DEFAULT_CHECKPOINT_DIR: str = "./models"
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (512, 512)
NUM_CLASSES: int = 150

ModelType = Literal["unet_smp", "fpn_smp", "psp_smp", "deeplab_tv", "fcn_tv", "segnet"]
EncoderName = Literal["resnet34", "resnet50", "resnet101", "mit_b5", "efficientnet-b0"]
AugmentationLevel = Literal["none", "basic", "medium", "aggressive"]
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────────────
# ПРАВИЛЬНЫЕ ignore_index ПО ТИПУ МОДЕЛИ
# ──────────────────────────────────────────────────────────────────────
# DeepLab/FCN (torchvision) обучались с ignore_index=0, потому что:
#   - ADE20K: класс 0 = "wall" доминирует (>30% пикселей во многих изображениях)
#   - Включение его в лосс перегружает модель предсказывать только класс 0
# SMP-модели (U-Net, FPN, PSPNet): ignore_index=255 — стандарт ADE20K
IGNORE_INDEX_BY_MODEL: Dict[str, int] = {
    "deeplab_tv": 255,
    "fcn_tv": 255,
    "unet_smp": 255,
    "fpn_smp": 255,
    "psp_smp": 255,
    "segnet": 255,
}


class ModelConfig(TypedDict):
    """
    Конфигурация модели для обучения.

    Attributes:
        model_type: Тип архитектуры сегментации.
        encoder_name: Название encoder'а (для SMP-моделей).
        variant: Вариант модели (например, "b5" для MiT, "fcn_resnet50" для FCN).
        lr: Learning rate (опционально, переопределяет значение по умолчанию).
    """

    model_type: ModelType
    encoder_name: EncoderName
    variant: str
    lr: NotRequired[float]


class TrainingResult(TypedDict):
    """
    Результаты обучения одного эксперимента.

    Attributes:
        experiment_name: Уникальное имя эксперимента.
        augmentation_level: Уровень аугментаций данных.
        model_type: Тип обученной модели.
        ignore_index: Индекс игнорируемых пикселей в лоссе.
        epochs_trained: Фактическое количество проведённых эпох.
        best_miou: Лучший достигнутый mIoU на валидации.
        final_train_loss: Значение тренировочного лосса на последней эпохе.
        final_val_loss: Значение валидационного лосса на последней эпохе.
        checkpoint_path: Путь к сохранённому чекпоинту.
        history: Словарь с историей метрик по эпохам.
        config: Исходная конфигурация эксперимента.
    """

    experiment_name: str
    augmentation_level: str
    model_type: str
    ignore_index: int
    epochs_trained: int
    best_miou: float
    final_train_loss: Optional[float]
    final_val_loss: Optional[float]
    checkpoint_path: str
    history: Dict[str, List[float]]
    config: "TrainingConfig"


# @dataclass
# class ModelConfig:
#     model_type: str
#     encoder_name: str
#     variant: str
#     lr: Optional[float] = None

# # Преобразование из вашего списка:
# model_configs: list[ModelConfig] = [ModelConfig(**cfg) for cfg in raw_configs]

# from pydantic import BaseModel
# from typing import Literal, Optional

# class ModelConfig(BaseModel):
#     model_type: Literal["unet_smp", "fpn_smp", "psp_smp", "deeplab_tv", "fcn_tv", "segnet"]
#     encoder_name: str
#     variant: str
#     lr: Optional[float] = None

# # Валидация и приведение типов "на лету":
# model_configs = [ModelConfig(**cfg) for cfg in raw_configs]

# # Пример: если передать "lr": "1e-5" (строку), Pydantic сам превратит её в float.
# # Если передать неверный model_type → выбросит ValidationError.


class TrainingConfig:
    """
    Конфигурация для эксперимента обучения моделей сегментации.

    Инкапсулирует все гиперпараметры и мета-информацию для воспроизводимости.
    Автоматически генерирует уникальное имя чекпоинта на основе таймстампа.

    Attributes:
        experiment_name: Человекочитаемое имя эксперимента.
        model_type: Тип архитектуры (из ModelType).
        augmentation_level: Уровень аугментаций данных ("none", "basic", "medium", "aggressive").
        epochs: Максимальное количество эпох обучения.
        batch_size: Размер батча для DataLoader.
        lr: Начальный learning rate.
        encoder_name: Название encoder'а (для SMP-моделей).
        variant: Вариант модели (например, "b5" для MiT).
        subset_fraction: Доля данных для использования (для быстрых тестов).
        early_stop_patience: Количество эпох без улучшения для early stopping.
        use_class_weights: Использовать ли взвешенный CrossEntropyLoss.
        checkpoint_name: Имя файла чекпоинта (генерируется автоматически, если не задано).

    Example:
        ```python
        config = TrainingConfig(
            experiment_name="unet_baseline",
            model_type="unet_smp",
            augmentation_level="medium",
            epochs=100,
            lr=1e-4,
        )
        print(config.checkpoint_name)  # "unet_smp_medium_20240101_120000.pth"
        ```
    """

    def __init__(
        self,
        experiment_name: str,
        model_type: str,
        augmentation_level: AugmentationLevel = "none",
        epochs: int = 20,
        batch_size: int = 4,
        lr: float = 1e-4,
        encoder_name: EncoderName = "resnet34",
        variant: str = "b5",
        subset_fraction: float = 0.05,
        early_stop_patience: int = 200,
        checkpoint_name: Optional[str] = None,
        use_class_weights: bool = False,
    ) -> None:
        """
        Инициализация конфигурации обучения.

        Args:
            experiment_name: Уникальное имя эксперимента для логирования.
            model_type: Тип модели (должен быть в ModelType).
            augmentation_level: Уровень аугментаций данных.
            epochs: Количество эпох обучения.
            batch_size: Размер батча для DataLoader.
            lr: Начальный learning rate.
            encoder_name: Название encoder'а (для SMP-моделей).
            variant: Вариант модели (например, "b5" для MiT, "fcn_resnet50" для FCN).
            subset_fraction: Доля данных для использования (1.0 = все данные).
            early_stop_patience: Эпох без улучшения mIoU для early stopping.
            checkpoint_name: Имя файла чекпоинта (опционально).
            use_class_weights: Использовать ли взвешенный лосс для дисбаланса классов.
        """
        self.experiment_name: str = experiment_name
        self.model_type: str = model_type
        self.augmentation_level: AugmentationLevel = augmentation_level
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.lr: float = lr
        self.encoder_name: EncoderName = encoder_name
        self.variant: str = variant
        self.subset_fraction: float = subset_fraction
        self.early_stop_patience: int = early_stop_patience
        self.use_class_weights: bool = use_class_weights

        # Генерируем уникальное имя чекпоинта
        if checkpoint_name is None:
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.checkpoint_name: str = (
                f"{model_type}_{augmentation_level}_{timestamp}.pth"
            )
        else:
            self.checkpoint_name = checkpoint_name

    def __repr__(self) -> str:
        return (
            f"TrainingConfig({self.experiment_name}, "
            f"model={self.model_type}, "
            f"aug={self.augmentation_level}, "
            f"epochs={self.epochs}, lr={self.lr})"
        )


class ModelTrainer:
    """
    Универсальный трейнер для обучения моделей семантической сегментации.

    Поддерживает:
    - Множество архитектур: U-Net, FPN, PSPNet, DeepLabV3+, FCN, SegNet (через SMP/torchvision).
    - Гибкую настройку аугментаций данных (4 уровня: none/basic/medium/aggressive).
    - Обучение с правильным ignore_index для каждого типа модели.
    - Взвешенный CrossEntropyLoss для борьбы с дисбалансом классов.
    - Early stopping по валидационному mIoU.
    - Сравнение обученных моделей на валидационном наборе.
    - Визуализацию learning curves и сводных результатов.

    Workflow:
    1. Создать экземпляр с путями к данным и чекпоинтам.
    2. Создать TrainingConfig с гиперпараметрами.
    3. Вызвать train_experiment(config) для обучения.
    4. (Опционально) Сравнить модели через compare_trained_models().
    5. (Опционально) Визуализировать результаты через plot_experiment_comparison().

    Attributes:
        checkpoint_dir (Path): Директория для сохранения чекпоинтов.
        root_dir (Path): Корневая директория датасета (ADE20K).
        device (torch.device): Устройство для вычислений (cuda/cpu).
        experiment_results (List[TrainingResult]): История результатов всех экспериментов.
    """

    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        root_dir: str = DEFAULT_ROOT_DIR,
        device: str = "cuda",
    ) -> None:
        """
        Инициализация трейнера.

        Args:
            checkpoint_dir: Директория для сохранения чекпоинтов моделей.
            root_dir: Корневая директория датасета (поддиректории: images/, annotations/).
            device: Предпочтительное устройство ("cuda" или "cpu").
        """
        self.checkpoint_dir: Path = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.root_dir: Path = Path(root_dir)
        self.device: torch.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.experiment_results: List[TrainingResult] = []

    def create_model(
        self,
        model_type: ModelType,
        encoder_name: EncoderName = "resnet34",
        variant: str = "b5",
        for_training: bool = True,
    ) -> torch.nn.Module:
        """
        Создаёт экземпляр модели сегментации по указанному типу.

        Поддерживаемые архитектуры:
        - SMP: U-Net, FPN, PSPNet (с encoder'ами ResNet, MiT, EfficientNet).
        - Torchvision: DeepLabV3+, FCN (с предобученными бэкбонами).
        - Custom: SegNet (через U-Net proxy или кастомную реализацию).

        Для torchvision-моделей:
        - Заменяет последний классификатор на NUM_CLASSES выходных каналов.
        - Инициализирует веса нового слоя (normal для weight, zero для bias).
        - Опционально замораживает бэкбон для начального обучения (for_training=True).

        Args:
            model_type: Тип архитектуры из ModelType.
            encoder_name: Название encoder'а (для SMP-моделей).
            variant: Вариант модели (например, "b5" для MiT, "fcn_resnet50" для FCN).
            for_training: Если `True`, замораживает бэкбон для torchvision-моделей.

        Returns:
            nn.Module: Инициализированная модель в режиме eval() (для инференса) или train() (для обучения).

        Raises:
            ValueError: Если model_type не поддерживается.

        Example:
            ```python
            trainer = ModelTrainer()
            model = trainer.create_model("unet_smp", encoder_name="resnet34")
            print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
            ```
        """
        # ──────────────────────────────────────────────────────────────
        # SMP МОДЕЛИ
        # ──────────────────────────────────────────────────────────────
        if model_type == "unet_smp":
            print("🔹 Creating U-Net (SMP)...")
            return smp.Unet(
                encoder_name=encoder_name,  # Можно заменить на "resnet50", "efficientnet-b0"
                encoder_weights="imagenet",
                in_channels=3,
                classes=NUM_CLASSES,
                activation=None,
            )
        elif model_type == "fpn_smp":
            print(f"🔹 Creating FPN + MiT-{variant}...")
            encoder = f"mit_{variant}"
            return smp.FPN(
                encoder_name=encoder,
                encoder_weights="imagenet",
                in_channels=3,
                classes=NUM_CLASSES,
                activation=None,
            )
        elif model_type == "psp_smp":
            print(f"🔹 Creating PSPNet + MiT-{variant}...")
            encoder = f"mit_{variant}"
            psp_size = 2048 if "mit" in encoder else 512
            return smp.PSPNet(
                encoder_name=encoder,
                encoder_weights="imagenet",
                in_channels=3,
                classes=NUM_CLASSES,
                activation=None,
                psp_size=psp_size,
            )

        # ──────────────────────────────────────────────────────────────
        # TORCHVISION МОДЕЛИ
        # ──────────────────────────────────────────────────────────────
        elif model_type == "deeplab_tv":
            print("🔹 Creating DeepLabV3+ (Torchvision)...")
            model = tv_seg.deeplabv3_resnet101(weights="COCO_WITH_VOC_LABELS_V1")
            in_channels = model.classifier[4].in_channels
            # model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
            model.classifier[4] = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1)
            nn.init.normal_(model.classifier[4].weight, 0, 0.01)
            nn.init.constant_(model.classifier[4].bias, 0)
            # if for_training:
            #     if isinstance(model.backbone, nn.Module):
            #         for param in model.backbone.parameters():
            #             param.requires_grad = False
            #     else:
            #         print(f"⚠️  Backbone не является nn.Module: {type(model.backbone)}")
            if for_training:
                for param in model.backbone.parameters():
                    param.requires_grad = False
                print("   🔒 Backbone frozen for initial training")
            # if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
            #     aux_in_ch = model.aux_classifier[4].in_channels
            #     model.aux_classifier[4] = nn.Conv2d(
            #         aux_in_ch, NUM_CLASSES, kernel_size=1
            #     )
            #     nn.init.normal_(model.aux_classifier[4].weight, 0, 0.01)
            #     nn.init.constant_(model.aux_classifier[4].bias, 0)

            return model
        elif model_type == "fcn_tv":
            print(f"🔹 Creating FCN ({variant})...")
            variants = {
                "fcn_resnet50": tv_seg.fcn_resnet50,
                "fcn_resnet101": tv_seg.fcn_resnet101,
            }
            if variant not in variants:
                raise ValueError(
                    f"Unknown FCN variant: {variant}. Available: {list(variants.keys())}"
                )

            model = variants[variant](weights="DEFAULT")
            in_channels = model.classifier[4].in_channels
            old_out_ch = model.classifier[4].out_channels
            model.classifier[4] = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1)
            print(f"   Classifier: {old_out_ch} → {NUM_CLASSES} classes")
            # model.classifier[4] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
            nn.init.normal_(model.classifier[4].weight, 0, 0.01)
            nn.init.constant_(model.classifier[4].bias, 0)
            if for_training:
                for param in model.backbone.parameters():
                    param.requires_grad = False
                print("   🔒 Backbone frozen for initial training")
            # if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
            #     aux_in_ch = model.aux_classifier[4].in_channels
            #     model.aux_classifier[4] = nn.Conv2d(
            #         aux_in_ch, NUM_CLASSES, kernel_size=1
            #     )
            #     nn.init.normal_(model.aux_classifier[4].weight, 0, 0.01)
            #     nn.init.constant_(model.aux_classifier[4].bias, 0)
            return model
        elif model_type == "segnet":
            print("🔹 Creating SegNet (U-Net proxy)...")
            try:
                # Используем U-Net как SegNet proxy (рекомендуется)
                model = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=NUM_CLASSES,
                    activation=None,
                )
                print("   Using SMP U-Net as SegNet proxy")
            except Exception as e:
                # Fallback к кастомной реализации
                print(f"   ⚠️  SMP U-Net failed: {e}")
                print("   Using custom SegNet implementation")
                model = SegNet(num_classes=NUM_CLASSES)

                # Инициализация весов
                def _init_weights(m):
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

                model.apply(_init_weights)

            return model
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. Available: {list(ModelType.__args__)}"  # type: ignore[attr-defined]
            )

    # ──────────────────────────────────────────────────────────────────────
    # DataLoaders
    # ──────────────────────────────────────────────────────────────────────
    def create_dataloaders(
        self,
        augmentation_level: AugmentationLevel,
        batch_size: int = 4,
        subset_fraction: float = 0.05,
        ignore_index: int = 255,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Создаёт DataLoader для тренировочного и валидационного наборов.

        Аугментации применяются только к тренировочному набору и зависят от уровня:
        - "none": только ресайз и нормализация.
        - "basic": горизонтальный флип.
        - "medium": + ротация, color jitter, небольшое масштабирование.
        - "aggressive": + вертикальный флип, более агрессивные трансформации.

        Args:
            augmentation_level: Уровень аугментаций для тренировочного набора.
            batch_size: Размер батча для обоих DataLoader.
            subset_fraction: Доля данных для использования (для быстрых тестов).
            ignore_index: Индекс пикселей для игнорирования в лоссе.

        Returns:
            Tuple[DataLoader, DataLoader]: (train_loader, val_loader).
        """

        print(f"   Augmentation level: {augmentation_level}")
        # Train с аугментациями
        is_augmented: bool = augmentation_level != "none"
        train_dataset = ADE20KDataset(
            root_dir=self.root_dir,
            split="training",
            image_size=DEFAULT_IMAGE_SIZE,
            augment=is_augmented,
            augmentation_level=augmentation_level,
            hflip_prob=0.5,
            # vflip_prob=0.1 if augmentation_level == "aggressive" else 0.0,
            rotation_prob=(
                0.3 if augmentation_level in ["medium", "aggressive"] else 0.0
            ),
            color_jitter_prob=(
                0.3 if augmentation_level in ["medium", "aggressive"] else 0.0
            ),
            scale_range=(
                (0.9, 1.1)
                if augmentation_level in ["medium", "aggressive"]
                else (1.0, 1.0)
            ),
            ignore_index=ignore_index,
            subset_fraction=subset_fraction,
        )

        val_dataset = ADE20KDataset(
            root_dir=self.root_dir,
            split="validation",
            image_size=DEFAULT_IMAGE_SIZE,
            augment=False,
            augmentation_level="none",
            ignore_index=ignore_index,
            subset_fraction=subset_fraction,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        return train_loader, val_loader

    # ──────────────────────────────────────────────────────────────────────
    # Веса классов (опционально)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_class_weights(
        train_loader: DataLoader,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = 255,
        max_batches: int = 100,
    ) -> torch.Tensor:
        """
        Считает инвертированные частоты классов для борьбы с дисбалансом.

        Использует median frequency balancing:
        ```
        weight[c] = median_freq / freq[c]
        ```
        где `freq[c]` — количество пикселей класса `c` в тренировочном наборе.

        Результат нормируется так, чтобы сумма весов равнялась `num_classes`.

        Args:
            train_loader: DataLoader с тренировочными данными.
            num_classes: Общее количество классов.
            ignore_index: Индекс игнорируемых пикселей.
            max_batches: Максимальное количество батчей для подсчёта (для скорости).

        Returns:
            torch.Tensor: Вектор весов размера `[num_classes]`, dtype float32.
        """
        class_counts: torch.Tensor = torch.zeros(num_classes, dtype=torch.float64)
        for idx, batch in enumerate(train_loader):
            if idx >= max_batches:
                break
            masks = batch["mask"]
            for c in range(num_classes):
                class_counts[c] += (masks == c).sum().item()
        class_counts = class_counts.clamp(min=1.0)
        # median frequency balancing
        median_freq: torch.Tensor = class_counts.median()
        weights: torch.Tensor = median_freq / class_counts
        weights = weights / weights.sum() * num_classes
        print(
            f"   Class weights computed (top-5 highest): "
            f"{sorted(weights.tolist(), reverse=True)[:5]}"
        )
        return weights.float()

    # ──────────────────────────────────────────────────────────────────────
    # Основной метод обучения
    # ──────────────────────────────────────────────────────────────────────
    def train_experiment(self, config: TrainingConfig) -> TrainingResult:
        """
        Обучает один эксперимент с заданной конфигурацией.

        Ключевые особенности:
        - Автоматический выбор ignore_index из `IGNORE_INDEX_BY_MODEL`.
        - Правильная инициализация критерия с учётом class weights.
        - Разморозка бэкбона для torchvision-моделей на эпохе 5.
        - Early stopping по валидационному mIoU.
        - Сохранение лучшего чекпоинта по mIoU.

        Args:
            config: Конфигурация эксперимента.

        Returns:
            TrainingResult: Словарь с результатами обучения (см. TypedDict выше).

        Note:
            - Для torchvision-моделей бэкбон замораживается на первые 5 эпох,
              затем размораживается с уменьшенным LR (lr / 10).
            - Scheduler CosineAnnealingLR создаётся на всё обучение и не сбрасывается
              при разморозке (чтобы избежать скачков LR).
        """
        print(f"\n{'=' * 70}")
        print(f"ЭКСПЕРИМЕНТ: {config.experiment_name}")
        print(f"Модель: {config.model_type} | Аугментации: {config.augmentation_level}")
        print(f"{'=' * 70}")

        # ── FIX 1: правильный ignore_index для данного типа модели ──
        ignore_index: int = IGNORE_INDEX_BY_MODEL.get(config.model_type, 255)
        print(f"   ignore_index: {ignore_index} (для {config.model_type})")

        model_type: ModelType = cast(ModelType, config.model_type)
        model = self.create_model(
            model_type, config.encoder_name, config.variant, for_training=True
        ).to(self.device)
        if config.model_type in ["deeplab_tv", "fcn_tv", "segnet"]:
            model.train()
            print("Strating train mode")
        print(f"🔍 Model training mode: {model.training}")

        print(f"✅ Pre-training checks:")
        print(f"   Model training mode: {model.training}")
        backbone: nn.Module = cast(nn.Module, model.backbone)
        print(
            f"   Backbone frozen: {all(not p.requires_grad for p in backbone.parameters())}"
        )

        # ── FIX 2: mask_fill_value=255 при аугментациях ──
        train_loader, val_loader = self.create_dataloaders(
            config.augmentation_level,
            config.batch_size,
            config.subset_fraction,
            ignore_index=ignore_index,
        )

        # ── Валидация первого батча ──
        print("🔍 Validating first batch...")
        sample_batch = next(iter(train_loader))
        masks = sample_batch["mask"]
        valid_masks = masks[masks != ignore_index]
        print(
            f"   Mask range (excl. ignore): [{valid_masks.min() if len(valid_masks) else 'N/A'}, "
            f"{valid_masks.max() if len(valid_masks) else 'N/A'}]"
        )
        print(f"   Unique values: {torch.unique(masks)[:20].tolist()}")
        print(f"Count of 255: {(masks == 255).sum().item()}")
        if len(valid_masks) > 0:
            assert (
                valid_masks.min() >= 0 and valid_masks.max() <= NUM_CLASSES - 1
            ), f"Mask values out of range! [{valid_masks.min()}, {valid_masks.max()}]"

        if ignore_index == 0:
            print("⚠️  WARNING: ignore_index=0, но класс 0='wall' доминирует в ADE20K!")
            print(
                "   Эти пиксели будут пропущены в loss — модель не научится предсказывать стены!"
            )

        print(f"✅ ignore_index: {ignore_index}")
        print(f"   Unique mask values (sample): {torch.unique(masks)[:20]}")
        print(f"   Mask range: [{masks.min()}, {masks.max()}]")

        # Проверка что класс 0 присутствует
        if (masks == 0).sum() > 0:
            print(f"   ✅ Class 0 (wall) found: {(masks == 0).sum()} pixels")
        else:
            print(f"   ❌ Class 0 NOT FOUND - check your ignore_index!")

        # ── Веса классов (опционально) ──
        class_weights = None
        if config.use_class_weights:
            print("   Computing class weights...")
            class_weights = self.compute_class_weights(
                train_loader, NUM_CLASSES, ignore_index
            ).to(self.device)

        # ── FIX 1: criterion с правильным ignore_index ──
        criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            weight=class_weights,
        )

        # ── Optimizer: только unfrozen параметры ──
        is_tv_model: bool = config.model_type in ["deeplab_tv", "fcn_tv"]
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=1e-4)
        if is_tv_model:
            print(
                f"   📊 Optimizer: {len(trainable_params)} trainable tensors (frozen backbone)"
            )
        else:
            print(f"   📊 Optimizer: all {len(trainable_params)} parameters")

        # ── FIX 3: единый CosineAnnealingLR на всё обучение ──
        # Больше НЕ создаём новый scheduler при разморозке.
        # T_max = полное число шагов батчей.
        total_steps: int = len(train_loader) * config.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=config.lr * 0.01
        )

        trainer = NeuralTrainer.__new__(NeuralTrainer)
        trainer.model = model
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader
        trainer.device = (
            str(self.device) if not isinstance(self.device, str) else self.device
        )
        trainer.num_classes = NUM_CLASSES
        trainer.ignore_index = ignore_index
        trainer.criterion = criterion
        trainer.optimizer = optimizer
        trainer.scheduler = scheduler
        trainer.best_miou = 0.0
        trainer.aux_loss_weight = 0.0
        trainer.verbose_first_batch = False
        trainer.history = {"train_loss": [], "val_loss": [], "val_miou": []}
        print(f"Criterion ignore_index: {trainer.criterion.ignore_index}")
        print(
            f"✅ Trainer attributes: ignore_index={trainer.ignore_index}, "
            f"aux_loss_w={trainer.aux_loss_weight}, "
            f"criterion.ignore_index={trainer.criterion.ignore_index},"
            f"device={trainer.device}"
        )
        print(f"   Criterion ignore_index: {trainer.criterion.ignore_index}")
        print(f"   Optimizer param groups: {len(trainer.optimizer.param_groups)}")

        checkpoint_path = os.path.join(self.checkpoint_dir, config.checkpoint_name)
        print(f"🔍 DEBUG FCN training setup:")
        print(f"   ignore_index: {ignore_index}")
        print(f"   aux_loss_weight: {trainer.aux_loss_weight}")
        print(f"   model.training: {model.training}")
        backbone = cast(nn.Module, model.backbone)
        print(
            f"   Backbone frozen: {all(not p.requires_grad for p in backbone.parameters())}"
        )
        print(f"   has aux_classifier: {hasattr(model, 'aux_classifier')}")
        if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
            aux_cls = cast(nn.Sequential, model.aux_classifier)
            print(f"   aux_classifier[4].out_channels: {aux_cls[4].out_channels}")  # type: ignore[index, union-attr]

        cls_layer = cast(nn.Sequential, model.classifier)
        print(f"   classifier[4].out_channels: {cls_layer[4].out_channels}")  # type: ignore[index, union-attr]
        print("🎯 Starting training...")

        # ──────────────────────────────────────────────────────────────
        # ЦИКЛ ОБУЧЕНИЯ
        # ──────────────────────────────────────────────────────────────
        for epoch in range(config.epochs):
            # ── FIX 3: разморозка backbone без пересоздания scheduler ──
            if is_tv_model and epoch == 5:
                backbone_opt: Optional[nn.Module] = getattr(model, "backbone", None)
                if isinstance(backbone_opt, nn.Module):
                    for param in backbone_opt.parameters():
                        param.requires_grad = True
                    frozen_count = sum(
                        1 for p in backbone_opt.parameters() if not p.requires_grad
                    )
                    print(f"   🔓 Unfroze backbone: {frozen_count} params still frozen")
                    # Добавляем backbone-параметры в существующий optimizer
                    # trainer.optimizer.add_param_group({
                    #     "params": [p for p in backbone.parameters()],
                    #     "lr": config.lr / 10,  # в 10 раз меньше, как было раньше
                    #     "weight_decay": 1e-4,
                    # })
                    # print(f"   🔓 Unfroze backbone (LR={config.lr / 10:.1e}), "
                    #       f"scheduler continues without reset")
                    trainer.optimizer = torch.optim.AdamW(
                        model.parameters(), lr=config.lr / 10, weight_decay=1e-4
                    )
                    remaining_epochs = config.epochs - epoch
                    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        trainer.optimizer,
                        T_max=len(train_loader) * remaining_epochs,
                        eta_min=config.lr / 10 * 0.01,
                    )
                    print(
                        f"   🔓 Unfroze backbone, new optimizer + scheduler (LR={config.lr/10:.1e})"
                    )
                else:
                    print(f"   ⚠️  Cannot unfreeze backbone: {type(backbone)}")

            start_time: float = time.time()
            train_loss = trainer.train_epoch()
            val_loss, val_miou = trainer.validate()
            epoch_time: float = time.time() - start_time

            trainer.history["train_loss"].append(train_loss)
            trainer.history["val_loss"].append(val_loss)
            trainer.history["val_miou"].append(val_miou)

            print(
                f"\n📊 Epoch {epoch + 1}/{config.epochs} | Time: {epoch_time:.1f}s "
                f"| LR: {trainer.optimizer.param_groups[0]['lr']:.2e}"
            )
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   Val mIoU:   {val_miou:.4f}")

            if val_miou > trainer.best_miou:
                trainer.best_miou = val_miou
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        "miou": val_miou,
                        "config": {
                            "model_type": config.model_type,
                            "ignore_index": ignore_index,
                            "augmentation_level": config.augmentation_level,
                        },
                    },
                    checkpoint_path,
                )
                print(f"   💾 Saved best model (mIoU: {val_miou:.4f})")

            # Early stopping
            if epoch >= config.early_stop_patience:
                recent_miou = trainer.history["val_miou"][-config.early_stop_patience :]
                if max(recent_miou) <= trainer.best_miou * 0.999:
                    print(f"   ⏹️  Early stopping at epoch {epoch + 1}")
                    break

            torch.cuda.empty_cache()
            gc.collect()

        # ──────────────────────────────────────────────────────────────
        # ФОРМИРОВАНИЕ РЕЗУЛЬТАТА
        # ──────────────────────────────────────────────────────────────
        result: TrainingResult = {
            "experiment_name": config.experiment_name,
            "augmentation_level": config.augmentation_level,
            "model_type": config.model_type,
            "ignore_index": ignore_index,
            "epochs_trained": len(trainer.history["train_loss"]),
            "best_miou": trainer.best_miou,
            "final_train_loss": (
                trainer.history["train_loss"][-1]
                if trainer.history["train_loss"]
                else None
            ),
            "final_val_loss": (
                trainer.history["val_loss"][-1] if trainer.history["val_loss"] else None
            ),
            "checkpoint_path": checkpoint_path,
            "history": trainer.history,
            "config": config,
        }
        self.experiment_results.append(result)

        print(f"\n✅ Experiment done! Best mIoU: {trainer.best_miou * 100:.2f}%")
        print(f"   Checkpoint: {checkpoint_path}")
        return result

    def compare_trained_models(
        self,
        augmentation_level: str = "medium",
        checkpoint_paths: Optional[Dict[str, str]] = None,
        model_types: Optional[List[ModelType]] = None,
    ) -> Dict[str, float]:
        """
        Сравнивает обученные модели на валидационном наборе ADE20K.

        Автоматически ищет чекпоинты по шаблону `{model_type}_{augmentation_level}_*.pth`,
        если `checkpoint_paths` не задан явно.

        Для каждой модели:
        1. Загружает веса из чекпоинта.
        2. Выполняет инференс на валидационном наборе (без аугментаций).
        3. Рассчитывает weighted mIoU через `sklearn.metrics.jaccard_score`.

        Args:
            augmentation_level: Уровень аугментаций для поиска чекпоинтов.
            checkpoint_paths: Словарь `{имя_модели: путь_к_чекпоинту}` (опционально).
            model_types: Список типов моделей для оценки (по умолчанию все поддерживаемые).

        Returns:
            Dict[str, float]: Словарь `{имя_модели: mIoU}`.
        """
        print("\n" + "=" * 60)
        print("📊 COMPARING TRAINED MODELS ON ADE20K VALIDATION")
        print("=" * 60)

        if model_types is None:
            model_types = [
                "unet_smp",
                "deeplab_tv",
                "fpn_smp",
                "psp_smp",
                "fcn_tv",
                "segnet",
            ]

        if checkpoint_paths is None:
            checkpoint_paths = {}
            for model_typer in model_types:
                if model_typer == "unet_smp":
                    name = "U-Net"
                elif model_typer == "deeplab_tv":
                    name = "DeepLabV3+"
                elif model_typer == "fpn_smp":
                    name = "FPN+MiT-B5"
                elif model_typer == "psp_smp":
                    name = "PSPNet+MiT-B5"
                elif model_typer == "fcn_tv":
                    name = "FCN ResNet-50"
                elif model_typer == "segnet":
                    name = "SegNet"
                else:
                    name = model_typer

                pattern = os.path.join(
                    self.checkpoint_dir, f"{model_typer}_{augmentation_level}_*.pth"
                )
                files = glob.glob(pattern)
                if files:
                    checkpoint_path = max(files, key=os.path.getctime)
                    checkpoint_paths[name] = checkpoint_path
                    print(f"   ✅ {name}: {checkpoint_path}")
                else:
                    print(f"   ⚠️  {name}: чекпоинт не найден ({pattern})")

        # Загружаем модели
        models: Dict[str, nn.Module] = {}
        if checkpoint_paths is not None:
            for name, path in checkpoint_paths.items():
                if not os.path.exists(path):
                    print(f"⚠️  Чекпоинт не найден: {path}")
                    continue

                # Определяем тип модели по имени
                model_type: Optional[ModelType] = None
                for mt in model_types:
                    if mt in path.lower() or mt in name.lower():
                        model_type = mt
                        break

                if model_type is None:
                    print(f"⚠️  Не удалось определить тип модели для {name}")
                    continue

                # Создаём модель
                if model_type == "fcn_tv":
                    model = self.create_model(model_type, variant="fcn_resnet50")
                else:
                    model = self.create_model(model_type)

                # Загружаем веса
                checkpoint = torch.load(path, map_location=self.device)

                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint

                if model_type == "deeplab_tv":
                    model_keys = {
                        k: v
                        for k, v in state_dict.items()
                        if not k.startswith("aux_classifier")
                    }
                    model.load_state_dict(model_keys, strict=False)
                    print("   🔍 Filtered aux_classifier keys for DeepLab")
                else:
                    model.load_state_dict(state_dict)

                if model_type in ["deeplab_tv", "fcn_tv", "segnet"]:
                    models[name] = model.to(self.device).train()
                else:
                    models[name] = model.to(self.device).eval()
                model = model.to(self.device).eval()

                print(f"✅ Loaded {name}")

        if not models:
            print("⚠️  No trained models found. Run training first.")
            return {}
        # Валидационный датасет
        val_dataset = ADE20KDataset(
            root_dir=self.root_dir,
            split="validation",
            image_size=DEFAULT_IMAGE_SIZE,
            augment=False,
            augmentation_level="none",
            subset_fraction=0.05,
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

        # Оценка каждой модели
        results: Dict[str, float] = {}
        for name, model in models.items():
            print(f"\n🔹 Evaluating {name}...")
            all_preds: List[int] = []
            all_targets: List[int] = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    images = batch["image"].to(self.device)
                    masks_gt = batch["mask"].to(self.device)

                    outputs = model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs["out"]

                    preds = outputs.argmax(1)
                    all_preds.extend(preds.cpu().flatten().tolist())
                    all_targets.extend(masks_gt.cpu().flatten().tolist())

                    if (batch_idx + 1) % 10 == 0:
                        print(f"   Processed {batch_idx + 1}/{len(val_loader)} batches")

            # Вычисление mIoU
            miou = jaccard_score(
                all_targets,
                all_preds,
                average="weighted",
                labels=range(NUM_CLASSES),
                zero_division=0,
            )

            results[name] = miou
            print(f"   ✅ mIoU: {miou * 100:.2f}%")

        # Таблица результатов
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        for name, miou in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:20s} : {miou * 100:6.2f}% mIoU")

        return results

    def evaluate_trained_models_on_val(
        self,
        checkpoints: Dict[str, str],
        val_fraction: float = 0.05,
        model_types: Optional[List[ModelType]] = None,
    ) -> Dict[str, float]:
        """
        Оценивает обученные модели на валидационном наборе.

        Аналогично `compare_trained_models()`, но принимает явный словарь чекпоинтов.

        Args:
            checkpoints: Dict `{model_name: checkpoint_path}`.
            val_fraction: Доля валидационного набора для оценки.
            model_types: Список типов моделей для оценки.

        Returns:
            Dict[str, float]: Словарь `{имя_модели: mIoU}`.
        """
        # Валидационный датасет
        if model_types is None:
            model_types = list(ModelType.__args__)  # type: ignore[attr-defined]

        val_dataset = ADE20KDataset(
            root_dir=self.root_dir,
            split="validation",
            image_size=DEFAULT_IMAGE_SIZE,
            augment=False,
            augmentation_level="none",
            subset_fraction=val_fraction,
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

        results: Dict[str, float] = {}

        for model_name, checkpoint_path in checkpoints.items():
            print(f"\n🔹 Evaluating {model_name}...")
            if not os.path.exists(checkpoint_path):
                print(f"   ⚠️  Checkpoint not found: {checkpoint_path}")
                continue
            # Создаём модель
            model_type: Optional[ModelType] = None
            for mt in model_types:
                if mt in checkpoint_path.lower() or mt in model_name.lower():
                    model_type = mt
                    break

            if model_type is None:
                # Попытка определить по ключевым словам
                keyword_map: Dict[str, ModelType] = {
                    "unet": "unet_smp",
                    "deeplab": "deeplab_tv",
                    "fpn": "fpn_smp",
                    "psp": "psp_smp",
                    "fcn": "fcn_tv",
                    "segnet": "segnet",
                }
                for kw, mt in keyword_map.items():
                    if kw in model_name.lower():
                        model_type = mt  # type: ignore[assignment]
                        break

            if model_type is None:
                print(f"   ⚠️  Unknown model type: {model_name}. Skipping...")
                continue

            model = self.create_model(model_type)

            # Загружаем веса
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            if model_type == "deeplab_tv":
                model_keys = {
                    k: v
                    for k, v in state_dict.items()
                    if not k.startswith("aux_classifier")
                }
                model.load_state_dict(model_keys, strict=False)
                print("   🔍 Filtered aux_classifier keys for DeepLab")
            else:
                model.load_state_dict(state_dict)

            print(f"   ✅ Loaded from {checkpoint_path}")

            if model_type in ["deeplab_tv", "fcn_tv", "segnet"]:
                model = model.to(self.device).train()
            else:
                model = model.to(self.device).eval()
            all_preds: List[int] = []
            all_targets: List[int] = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    images = batch["image"].to(self.device)
                    masks_gt = batch["mask"].to(self.device)

                    outputs = model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs["out"]

                    preds = outputs.argmax(1)
                    all_preds.extend(preds.cpu().flatten().tolist())
                    all_targets.extend(masks_gt.cpu().flatten().tolist())

                    if (batch_idx + 1) % 20 == 0:
                        print(f"   Processed {batch_idx + 1}/{len(val_loader)} batches")

            # Метрики
            miou = jaccard_score(
                all_targets,
                all_preds,
                average="weighted",
                labels=range(NUM_CLASSES),
                zero_division=0,
            )

            results[model_name] = miou
            print(f"   ✅ mIoU: {miou * 100:.2f}%")

            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Таблица
        print("\n" + "=" * 60)
        print("TRAINED MODELS COMPARISON (on validation set)")
        print("=" * 60)
        for name, miou in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:20s} : {miou * 100:6.2f}% mIoU")

        return results

    # ──────────────────────────────────────────────────────────────────────
    # Сравнение аугментаций
    # ──────────────────────────────────────────────────────────────────────
    def compare_augmentations(
        self,
        model_type: ModelType = "unet_smp",
        augmentation_levels: Optional[List[AugmentationLevel]] = None,
        base_config: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Сравнивает обучение с разными уровнями аугментаций данных.

        Для каждого уровня:
        1. Создаёт TrainingConfig с одинаковыми гиперпараметрами.
        2. Запускает train_experiment().
        3. Агрегирует результаты в сравнительную таблицу.

        Args:
            model_type: Тип модели для сравнения.
            augmentation_levels: Список уровней аугментаций (по умолчанию все 4).
            base_config: Базовые гиперпараметры (переопределяются для каждого эксперимента).

        Returns:
            pd.DataFrame: Таблица с колонками:
            - Augmentation, Model Type, Best mIoU (%), Epochs, Final Train/Val Loss, Checkpoint.
        """
        if augmentation_levels is None:
            augmentation_levels = ["none", "basic", "medium", "aggressive"]

        if base_config is None:
            base_config = {
                "epochs": 20,
                "batch_size": 4,
                "lr": 1e-4,
                "encoder_name": "resnet34",
                "variant": "b5",
                "subset_fraction": 0.05,
            }

        print(f"\n{'=' * 70}")
        print("СРАВНЕНИЕ УРОВНЕЙ АУГМЕНТАЦИЙ")
        print(f"Модель: {model_type}")
        print(f"{'=' * 70}")

        results: List[TrainingResult] = []

        for aug_level in augmentation_levels:
            config = TrainingConfig(
                experiment_name=f"aug_comparison_{model_type}",
                model_type=str(model_type),
                augmentation_level=aug_level,
                **(base_config or {}),
            )
            result = self.train_experiment(config)
            results.append(result)

        # Создание сравнительной таблицы
        comparison_df: pd.DataFrame = pd.DataFrame(
            [
                {
                    "Augmentation": r["augmentation_level"],
                    "Model Type": r["model_type"],
                    "Best mIoU (%)": r["best_miou"] * 100,
                    "Epochs": r["epochs_trained"],
                    "Final Train Loss": r["final_train_loss"],
                    "Final Val Loss": r["final_val_loss"],
                    "Checkpoint": os.path.basename(r["checkpoint_path"]),
                }
                for r in results
            ]
        )

        # Сортировка по mIoU
        comparison_df = comparison_df.sort_values("Best mIoU (%)", ascending=False)

        # Сохранение
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path: str = os.path.join(
            self.checkpoint_dir, f"augmentation_comparison_{model_type}_{timestamp}.csv"
        )
        comparison_df.to_csv(comparison_path, index=False)

        print(f"\n{'=' * 70}")
        print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
        print(f"{'=' * 70}")
        print(comparison_df.to_string(index=False))
        print(f"\n📊 Таблица сохранена: {comparison_path}")

        return comparison_df

    # ──────────────────────────────────────────────────────────────────────
    # Визуализация
    # ──────────────────────────────────────────────────────────────────────
    def plot_experiment_comparison(self, output_path: Optional[str] = None) -> None:
        """
        Визуализирует сравнение проведённых экспериментов.

        Строит 2×2 grid:
        1. Bar-чарт Best mIoU по уровням аугментаций.
        2. Learning curves (Val mIoU по эпохам).
        3. Train vs Val Loss для первых двух экспериментов.
        4. Сводная таблица результатов.

        Args:
            output_path: Путь для сохранения графика (опционально).
        """
        if len(self.experiment_results) < 2:
            print("⚠️ Нужно минимум 2 эксперимента для сравнения")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. mIoU по уровням аугментаций
        ax1 = axes[0, 0]
        miou_values = [r["best_miou"] * 100 for r in self.experiment_results]
        aug_labels = [r["augmentation_level"] for r in self.experiment_results]

        cmap: Colormap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 1, len(miou_values)))
        ax1.bar(aug_labels, miou_values, color=colors, edgecolor="black")
        ax1.set_xlabel("Уровень аугментаций")
        ax1.set_ylabel("Best mIoU (%)")
        ax1.set_title("Влияние аугментаций на Best mIoU")
        ax1.grid(axis="y", alpha=0.3)

        for i, v in enumerate(miou_values):
            ax1.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

        # 2. Learning curves
        ax2 = axes[0, 1]
        for result in self.experiment_results:
            epochs = range(1, len(result["history"]["val_miou"]) + 1)
            ax2.plot(
                epochs,
                np.array(result["history"]["val_miou"]) * 100,
                label=f"{result['augmentation_level']} (mIoU={result['best_miou'] * 100:.2f}%)",
                linewidth=2,
            )

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation mIoU (%)")
        ax2.set_title("Динамика обучения (Val mIoU)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Train vs Val Loss
        ax3 = axes[1, 0]
        for result in self.experiment_results[:2]:  # Показываем первые 2
            epochs = range(1, len(result["history"]["train_loss"]) + 1)
            ax3.plot(
                epochs,
                result["history"]["train_loss"],
                label=f"{result['augmentation_level']} (Train)",
                linestyle="-",
                linewidth=2,
            )
            ax3.plot(
                epochs,
                result["history"]["val_loss"],
                label=f"{result['augmentation_level']} (Val)",
                linestyle="--",
                linewidth=2,
            )

        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.set_title("Train vs Validation Loss")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Сводная таблица
        ax4 = axes[1, 1]
        ax4.axis("off")

        table_data = []
        for result in self.experiment_results:
            table_data.append(
                [
                    result["augmentation_level"],
                    result["model_type"],
                    f"{result['best_miou'] * 100:.2f}%",
                    result["ignore_index"],
                    f"{result['epochs_trained']}",
                    (
                        f"{result['final_train_loss']:.4f}"
                        if result["final_train_loss"]
                        else "N/A"
                    ),
                ]
            )

        table = ax4.table(
            cellText=table_data,
            colLabels=[
                "Augmentation",
                "Model",
                "Best mIoU",
                "ignore_idx",
                "Epochs",
                "Final Train Loss",
            ],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title("Сводка экспериментов", fontsize=12, fontweight="bold")

        plt.tight_layout()

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.checkpoint_dir, f"experiment_comparison_{timestamp}.png"
            )

        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\n📊 Визуализация сохранена: {output_path}")
        plt.show()

    # ──────────────────────────────────────────────────────────────────────
    # Обучение всех моделей
    # ──────────────────────────────────────────────────────────────────────
    def train_all_models(
        self,
        augmentation_level: AugmentationLevel = "medium",
        epochs: int = 20,
        batch_size: int = 4,
        lr: float = 1e-4,
        subset_fraction: float = 0.05,
    ) -> Dict[str, TrainingResult]:
        """
        Обучает все поддерживаемые модели с одинаковыми гиперпараметрами.

        Удобно для быстрого бенчмарка архитектур перед углублённой настройкой.

        Args:
            augmentation_level: Уровень аугментаций для всех экспериментов.
            epochs: Количество эпох обучения.
            batch_size: Размер батча.
            lr: Learning rate.
            subset_fraction: Доля данных для использования.

        Returns:
            Dict[str, TrainingResult]: Результаты по каждой модели `{model_type: result}`.
        """
        model_configs: List[ModelConfig] = [
            {"model_type": "unet_smp", "encoder_name": "resnet34", "variant": "b5"},
            {"model_type": "fpn_smp", "encoder_name": "resnet34", "variant": "b5"},
            {"model_type": "psp_smp", "encoder_name": "resnet34", "variant": "b5"},
            {
                "model_type": "deeplab_tv",
                "encoder_name": "resnet34",
                "variant": "b5",
                "lr": 1e-5,
            },
            {
                "model_type": "fcn_tv",
                "encoder_name": "resnet34",
                "variant": "fcn_resnet50",
                "lr": 1e-5,
            },
            {
                "model_type": "segnet",
                "encoder_name": "resnet34",
                "variant": "b5",
                "lr": 1e-4,
            },
        ]

        all_results: Dict[str, TrainingResult] = {}

        for config in model_configs:
            experiment_config = TrainingConfig(
                experiment_name=f"all_models_{config['model_type']}",
                model_type=config["model_type"],
                augmentation_level=augmentation_level,
                epochs=epochs,
                batch_size=batch_size,
                lr=config.get("lr", lr),
                encoder_name=cast(EncoderName, config.get("encoder_name", "resnet34")),
                variant=str(config.get("variant", "b5")),
                subset_fraction=subset_fraction,
            )
            result: TrainingResult = self.train_experiment(experiment_config)
            all_results[str(config["model_type"])] = result

        # Сводная таблица
        summary_df: pd.DataFrame = pd.DataFrame(
            [
                {
                    "Model": r["model_type"],
                    "ignore_index": r["ignore_index"],
                    "Best mIoU (%)": r["best_miou"] * 100,
                    "Epochs": r["epochs_trained"],
                    "Checkpoint": os.path.basename(r["checkpoint_path"]),
                }
                for r in all_results.values()
            ]
        )

        summary_df = summary_df.sort_values("Best mIoU (%)", ascending=False)

        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path: str = os.path.join(
            self.checkpoint_dir, f"all_models_summary_{timestamp}.csv"
        )
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'=' * 70}")
        print("СВОДНАЯ ТАБЛИЦА ВСЕХ МОДЕЛЕЙ")
        print(f"{'=' * 70}")
        print(summary_df.to_string(index=False))
        print(f"\n📊 Таблица сохранена: {summary_path}")

        return all_results

    def evaluate_checkpoints(
        self,
        checkpoint_paths: List[str],
        model_type: ModelType,
        encoder_name: EncoderName = "resnet34",
        variant: str = "b5",
    ) -> pd.DataFrame:
        """
        Оценивает список чекпоинтов на валидационном наборе.

        Полезно для сравнения разных эпох или конфигураций одной архитектуры.

        Args:
            checkpoint_paths: Список путей к чекпоинтам.
            model_type: Тип модели для всех чекпоинтов.
            encoder_name: Название encoder'а (для SMP-моделей).
            variant: Вариант модели.

        Returns:
            pd.DataFrame: Таблица с колонками: Checkpoint, mIoU (%), Path.
        """
        print(f"\n{'=' * 70}")
        print("ОЦЕНКА ЧЕКПОИНТОВ")
        print(f"{'=' * 70}")

        # Валидационный датасет (без аугментаций!)
        val_dataset = ADE20KDataset(
            root_dir=self.root_dir,
            split="validation",
            image_size=DEFAULT_IMAGE_SIZE,
            augment=False,
            augmentation_level="none",
            subset_fraction=0.1,  # 10% для быстрой оценки
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

        results: List[Dict[str, Any]] = []
        for checkpoint_path in checkpoint_paths:
            if not os.path.exists(checkpoint_path):
                print(f"⚠️ Чекпоинт не найден: {checkpoint_path}")
                continue

            # Извлекаем имя эксперимента из пути
            checkpoint_name = os.path.basename(checkpoint_path)

            # Создание модели
            model = self.create_model(
                model_type, encoder_name, variant, for_training=False
            ).to(self.device)

            # Загрузка весов
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Фильтрация aux_classifier для DeepLab (как в старом варианте)
            if model_type == "deeplab_tv":
                model_keys = {
                    k: v
                    for k, v in state_dict.items()
                    if not k.startswith("aux_classifier")
                }
                model.load_state_dict(model_keys, strict=False)
                print("   🔍 Filtered aux_classifier keys for DeepLab")
            else:
                model.load_state_dict(state_dict)

            model.eval()

            # Оценка
            all_preds: List[int] = []
            all_targets: List[int] = []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device)

                    outputs = model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs["out"]
                    preds = outputs.argmax(1)
                    all_preds.extend(preds.cpu().flatten().tolist())
                    all_targets.extend(masks.cpu().flatten().tolist())

            # Вычисление mIoU
            miou = jaccard_score(
                all_targets,
                all_preds,
                average="weighted",
                labels=range(NUM_CLASSES),
                zero_division=0,
            )

            result: Dict[str, Any] = {
                "Checkpoint": checkpoint_name,
                "mIoU (%)": miou * 100,
                "Path": checkpoint_path,
            }
            results.append(result)

            print(f"✅ {checkpoint_name}: mIoU = {miou * 100:.2f}%")

            # Очистка памяти
            del model
            torch.cuda.empty_cache()

        # Создание таблицы
        results_df: pd.DataFrame = pd.DataFrame(results)
        results_df = results_df.sort_values("mIoU (%)", ascending=False)

        # Сохранение
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_path: str = os.path.join(
            self.checkpoint_dir, f"checkpoint_evaluation_{timestamp}.csv"
        )
        results_df.to_csv(eval_path, index=False)

        print(f"\n{'=' * 70}")
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
        print(f"{'=' * 70}")
        print(results_df.to_string(index=False))
        print(f"\n📊 Таблица сохранена: {eval_path}")

        return results_df


# unet_model, unet_history = train_unet_ade20k(epochs=200, subset_fraction=0.05)

# # Пример теста: U-Net: 5 эпох, 5% данных, batch=2
# # unet_model, _ = train_unet_ade20k(epochs=5, batch_size=2, subset_fraction=0.05, lr=1e-4)
# # train_unet_ade20k(epochs=50, batch_size=8, subset_fraction=1.0, lr=1e-4)

# # Запуск (для теста)
# deeplab_model, deeplab_history = train_deeplab_ade20k(epochs=200, subset_fraction=0.05)

# # Пример теста: DeepLab: 5 эпох, 5% данных
# # deeplab_model, _ = train_deeplab_ade20k(epochs=5, batch_size=2, subset_fraction=0.05, lr=1e-5)
# # train_deeplab_ade20k(epochs=30, batch_size=4, subset_fraction=1.0, lr=1e-5)

# # Запуск (для теста)
# fpn_model, fpn_history = train_fpn_mit_ade20k(epochs=200, subset_fraction=0.05, variant="b5")

# # Запуск (для теста)
# psp_model, psp_history = train_psp_mit_ade20k(epochs=200, subset_fraction=0.05, variant="b5")

# # Запуск (для теста)
# fcn_model, fcn_history = train_fcn_resnet50_ade20k(epochs=20, subset_fraction=0.05)

# # Запуск (для теста)
# segnet_model, segnet_history = train_segnet_ade20k(epochs=200, subset_fraction=0.05)

# results = compare_trained_models()
# print(f"Results: {results}")

# checkpoints: Dict[str, str] = {
#     "U-Net (trained)": "./../models/unet_ade20k_best.pth",
#     "DeepLabV3+ (trained)": "./../models/deeplab_ade20k_best.pth",
#     "FPN+MiT-B5 (trained)": "./../models/fpn_mit_b5_ade20k_best.pth",
#     "PSPNet+MiT-B5 (trained)": "./../models/psp_mit_b5_ade20k_best.pth",
#     "FCN ResNet-50 (trained)": "./../models/fcn_resnet50_ade20k_best.pth",
#     "SegNet (trained)": "./../models/segnet_ade20k_best.pth"
# }
# trained_results = evaluate_trained_models_on_val(checkpoints, val_fraction=0.05)
