# segmenters/ModelTrainer.py

# Импорт основных библиотек
import os
import sys
import time
from datetime import datetime
import gc
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import torchvision.models.segmentation as tv_seg

from sklearn.metrics import jaccard_score

from datasets.ADE20KDataset import ADE20KDataset
from .NeuralTrainer import NeuralTrainer
from utils.strategies import SegNet

from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Константы
DEFAULT_ROOT_DIR: str = "./data/ade20k"
DEFAULT_CHECKPOINT_DIR: str = "./models"
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (512, 512)
NUM_CLASSES: int = 150
device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TrainingConfig:
    """Конфигурация для эксперимента обучения"""

    def __init__(
        self,
        experiment_name: str,
        model_type: str,
        augmentation_level: str = "none",
        epochs: int = 20,
        batch_size: int = 4,
        lr: float = 1e-4,
        encoder_name: str = "resnet34",
        variant: str = "b5",  # Для MiT encoder
        subset_fraction: float = 0.05,
        early_stop_patience: int = 5,
        checkpoint_name: Optional[str] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.augmentation_level = augmentation_level
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.encoder_name = encoder_name
        self.variant = variant
        self.subset_fraction = subset_fraction
        self.early_stop_patience = early_stop_patience

        # Генерируем уникальное имя чекпоинта
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.checkpoint_name = f"{model_type}_{augmentation_level}_{timestamp}.pth"
        else:
            self.checkpoint_name = checkpoint_name

    def __repr__(self):
        return (
            f"TrainingConfig({self.experiment_name}, "
            f"model={self.model_type}"
            f"aug={self.augmentation_level}, "
            f"epochs={self.epochs}, lr={self.lr})"
        )


class ModelTrainer:
    """
    Универсальный трейнер для обучения моделей сегментации
    с поддержкой сравнения аугментаций
    """

    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        root_dir: str = DEFAULT_ROOT_DIR,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.root_dir = root_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.experiment_results: List = []
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def create_model(
        self,
        model_type: str,
        encoder_name: str = "resnet34",
        variant: str = "b5",
        for_training: bool = True,
    ) -> torch.nn.Module:
        """Создание модели по типу"""

        # ========== SMP МОДЕЛИ ==========
        if model_type == "unet_smp":
            print("🔹 Training U-Net (SMP) on ADE20K...")
            return smp.Unet(
                encoder_name=encoder_name,  # Можно заменить на "resnet50", "efficientnet-b0"
                encoder_weights="imagenet",
                in_channels=3,
                classes=NUM_CLASSES,
                activation=None,
            )
        elif model_type == "fpn_smp":
            print(f"🔹 Training FPN + MiT-{variant} on ADE20K...")
            encoder = f"mit_{variant}"
            return smp.FPN(
                encoder_name=encoder,
                encoder_weights="imagenet",
                in_channels=3,
                classes=NUM_CLASSES,
                activation=None,
            )
        elif model_type == "psp_smp":
            print(f"🔹 Training PSPNet + MiT-{variant} on ADE20K...")
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

        # ========== TORCHVISION МОДЕЛИ ==========
        elif model_type == "deeplab_tv":
            print("🔹 Training DeepLabV3+ (Torchvision) on ADE20K...")
            model = tv_seg.deeplabv3_resnet101(weights="COCO_WITH_VOC_LABELS_V1")
            in_channels = model.classifier[4].in_channels
            # model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
            model.classifier[4] = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1)
            nn.init.normal_(model.classifier[4].weight, 0, 0.01)
            nn.init.constant_(model.classifier[4].bias, 0)
            if for_training:
                if isinstance(model.backbone, nn.Module):
                    for param in model.backbone.parameters():
                        param.requires_grad = False
                else:
                    print(f"⚠️  Backbone не является nn.Module: {type(model.backbone)}")
            print("   🔒 Backbone frozen for initial training")

            return model
        elif model_type == "fcn_tv":
            print(f"🔹 Training FCN ({variant}) on ADE20K...")
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
            model.classifier[4] = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1)
            # model.classifier[4] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
            nn.init.normal_(model.classifier[4].weight, 0, 0.01)
            nn.init.constant_(model.classifier[4].bias, 0)
            if for_training:
                for param in model.backbone.parameters():
                    param.requires_grad = False
                print("   🔒 Backbone frozen for initial training")
            return model
        elif model_type == "segnet":
            print("🔹 Training SegNet on ADE20K...")
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
                f"Unknown model_type: {model_type}. Available: unet_smp, fpn_smp, psp_smp, deeplab_tv, fcn_tv, segnet"
            )

    def create_dataloaders(
        self,
        augmentation_level: str,
        batch_size: int = 4,
        subset_fraction: float = 0.05,
    ) -> Tuple[DataLoader, DataLoader]:
        """Создание DataLoader с нужным уровнем аугментаций"""

        print(f"   Augmentation level: {augmentation_level}")
        # Train с аугментациями
        train_dataset = ADE20KDataset(
            root_dir=self.root_dir,
            split="training",
            image_size=DEFAULT_IMAGE_SIZE,
            augment=True if augmentation_level != "none" else False,
            augmentation_level=augmentation_level,
            hflip_prob=0.5,
            vflip_prob=0.1 if augmentation_level == "aggressive" else 0.0,
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
            subset_fraction=subset_fraction,
        )

        val_dataset = ADE20KDataset(
            root_dir=self.root_dir,
            split="validation",
            image_size=DEFAULT_IMAGE_SIZE,
            augment=False,
            augmentation_level="none",
            subset_fraction=subset_fraction,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        return train_loader, val_loader

    def train_experiment(self, config: TrainingConfig) -> Dict[str, Any]:
        """
        Обучение одного эксперимента
        Returns:
            Dict с результатами обучения
        """
        print(f"\n{'='*70}")
        print(f"ЭКСПЕРИМЕНТ: {config.experiment_name}")
        print(f"Аугментации: {config.augmentation_level}")
        print(f"{'='*70}")

        # Создание модели
        model = self.create_model(
            config.model_type, config.encoder_name, config.variant, for_training=True
        ).to(self.device)
        # Создание DataLoader
        train_loader, val_loader = self.create_dataloaders(
            config.augmentation_level, config.batch_size, config.subset_fraction
        )

        # 🔥 Настройка оптимизатора в зависимости от типа модели
        is_torchvision_model = config.model_type in ["deeplab_tv", "fcn_tv"]

        if is_torchvision_model:
            # 🔥 Только classifier на первых эпохах (как в старом варианте)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=config.lr,
                weight_decay=1e-4,
            )
            print("   📊 Optimizer: classifier only (frozen backbone)")
        else:
            # SMP модели — все параметры
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.lr, weight_decay=1e-4
            )
            print("   📊 Optimizer: all parameters")

        # Трейнер
        trainer = NeuralTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=NUM_CLASSES,
            lr=config.lr,
            device=str(self.device),
            ignore_index=255,
        )

        # Переопределяем оптимизатор и scheduler
        if config.model_type in ["deeplab_tv"]:
            trainer.criterion = nn.CrossEntropyLoss(ignore_index=0)
        elif config.model_type in ["fcn_tv"]:
            trainer.criterion = nn.CrossEntropyLoss(ignore_index=255)
        trainer.optimizer = optimizer
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader) * config.epochs
        )

        # Валидация первого батча
        print("🔍 Validating first batch...")
        sample_batch = next(iter(train_loader))
        masks = sample_batch["mask"]
        print(f"   Mask range: [{masks.min()}, {masks.max()}]")
        print(f"   Unique values: {torch.unique(masks)[:20].tolist()}")
        assert masks.min() >= 0 and masks.max() <= 149, "Mask values out of range!"

        # Обучение
        checkpoint_path = os.path.join(self.checkpoint_dir, config.checkpoint_name)

        print("🎯 Starting training...")

        for epoch in range(config.epochs):
            # 🔥 Разморозка backbone после 5 эпох для Torchvision моделей
            if is_torchvision_model and epoch == 5:
                for param in model.backbone.parameters():
                    param.requires_grad = True

                # 🔥 Новый оптимизатор для всех параметров (как в старом варианте)
                trainer.optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config.lr / 10,  # 🔥 Уменьшаем LR
                    weight_decay=1e-4,
                )
                trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    trainer.optimizer, T_max=len(train_loader) * (config.epochs - 5)
                )
                print("   🔓 Unfroze backbone, reduced LR")

            start_time = time.time()
            train_loss = trainer.train_epoch()
            val_loss, val_miou = trainer.validate()
            epoch_time = time.time() - start_time

            trainer.history["train_loss"].append(train_loss)
            trainer.history["val_loss"].append(val_loss)
            trainer.history["val_miou"].append(val_miou)

            print(f"\n📊 Epoch {epoch+1}/{config.epochs} | Time: {epoch_time:.1f}s")
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
                    },
                    checkpoint_path,
                )
                print(f"   💾 Saved best model (mIoU: {val_miou:.4f})")

            torch.cuda.empty_cache()
            gc.collect()

        # Результаты
        result = {
            "experiment_name": config.experiment_name,
            "augmentation_level": config.augmentation_level,
            "model_type": config.model_type,
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

        print("\n✅ Эксперимент завершён!")
        print(f"   Best mIoU: {trainer.best_miou*100:.4f}%")
        print(f"   Чекпоинт: {checkpoint_path}")

        return result

    def compare_trained_models(
        self,
        augmentation_level: str = "medium",
        checkpoint_paths: Optional[Dict[str, str]] = None,
        model_types: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Сравнение обученных моделей на валидационном наборе

        Args:
            augmentation_level: Уровень аугментаций для поиска чекпоинтов
            checkpoint_paths: Список путей к чекпоинтам (если None, ищет по умолчанию)
            model_types: Список типов моделей для оценки

        Returns:
            Dict с mIoU для каждой модели
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

        # Если пути не указаны, используем стандартные имена
        if checkpoint_paths is None:
            checkpoint_paths = {}
            for model_type in model_types:
                # 🔥 Имя модели для отображения
                if model_type == "unet_smp":
                    name = "U-Net"
                elif model_type == "deeplab_tv":
                    name = "DeepLabV3+"
                elif model_type == "fpn_smp":
                    name = "FPN+MiT-B5"
                elif model_type == "psp_smp":
                    name = "PSPNet+MiT-B5"
                elif model_type == "fcn_tv":
                    name = "FCN ResNet-50"
                elif model_type == "segnet":
                    name = "SegNet"
                else:
                    name = model_type

                # 🔥 Ищем чекпоинт с правильным именем
                pattern = os.path.join(
                    self.checkpoint_dir, f"{model_type}_{augmentation_level}_*.pth"
                )
                files = glob.glob(pattern)
                if files:
                    checkpoint_path = max(files, key=os.path.getctime)
                    checkpoint_paths[name] = checkpoint_path
                    print(f"   ✅ {name}: {checkpoint_path}")
                else:
                    print(f"   ⚠️  {name}: чекпоинт не найден ({pattern})")

        # Загружаем модели
        models = {}
        if checkpoint_paths is not None:
            for name, path in checkpoint_paths.items():
                if not os.path.exists(path):
                    print(f"⚠️  Чекпоинт не найден: {path}")
                    continue

                # Определяем тип модели по имени
                model_type = None
                for mt in model_types:
                    if mt in path.lower() or mt in name.lower():
                        model_type = mt
                        break

                if model_type is None:
                    print(f"⚠️  Не удалось определить тип модели для {name}")
                    continue

                # Создаём модель
                model = self.create_model(model_type)

                # Загружаем веса
                checkpoint = torch.load(path, map_location=self.device)

                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint

                # 🔥 Фильтрация aux_classifier для DeepLab (как в старом варианте)
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
        results = {}
        for name, model in models.items():
            print(f"\n🔹 Evaluating {name}...")
            all_preds = []
            all_targets = []

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
                        print(f"   Processed {batch_idx+1}/{len(val_loader)} batches")

            # Вычисление mIoU
            miou = jaccard_score(
                all_targets,
                all_preds,
                average="weighted",
                labels=range(NUM_CLASSES),
                zero_division=0,
            )

            results[name] = miou
            print(f"   ✅ mIoU: {miou*100:.2f}%")

        # Таблица результатов
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        for name, miou in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:20s} : {miou*100:6.2f}% mIoU")

        return results

    def evaluate_trained_models_on_val(
        self,
        checkpoints: Dict[str, str],
        val_fraction: float = 0.05,
        model_types: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Оценка обученных моделей на валидационном наборе

        Args:
            checkpoints: Dict {model_name: checkpoint_path}
            val_fraction: Доля валидационного набора для оценки
            model_types: Список типов моделей для оценки

        Returns:
            Dict с mIoU для каждой модели
        """
        # Валидационный датасет
        if model_types is None:
            model_types = [
                "unet_smp",
                "deeplab_tv",
                "fpn_smp",
                "psp_smp",
                "fcn_tv",
                "segnet",
            ]

        val_dataset = ADE20KDataset(
            root_dir=self.root_dir,
            split="validation",
            image_size=DEFAULT_IMAGE_SIZE,
            augment=False,
            augmentation_level="none",
            subset_fraction=val_fraction,
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

        results = {}

        for model_name, checkpoint_path in checkpoints.items():
            print(f"\n🔹 Evaluating {model_name}...")
            if not os.path.exists(checkpoint_path):
                print(f"   ⚠️  Checkpoint not found: {checkpoint_path}")
                continue
            # Создаём модель
            model_type = None
            for mt in model_types:
                if mt in checkpoint_path.lower() or mt in model_name.lower():
                    model_type = mt
                    break

            if model_type is None:
                # Пытаемся определить по ключевым словам
                if "unet" in model_name.lower():
                    model_type = "unet_smp"
                elif "deeplab" in model_name.lower():
                    model_type = "deeplab_tv"
                elif "fpn" in model_name.lower():
                    model_type = "fpn_smp"
                elif "psp" in model_name.lower():
                    model_type = "psp_smp"
                elif "fcn" in model_name.lower():
                    model_type = "fcn_tv"
                elif "segnet" in model_name.lower():
                    model_type = "segnet"

            if model_type is None:
                print(f"   ⚠️  Unknown model type: {model_name}. Skipping...")
                continue

            model = self.create_model(model_type)

            # Загружаем веса
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # 🔥 Фильтрация aux_classifier для DeepLab (как в старом варианте)
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
            all_preds = []
            all_targets = []
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
                        print(f"   Processed {batch_idx+1}/{len(val_loader)} batches")

            # Метрики
            miou = jaccard_score(
                all_targets,
                all_preds,
                average="weighted",
                labels=range(NUM_CLASSES),
                zero_division=0,
            )

            results[model_name] = miou
            print(f"   ✅ mIoU: {miou*100:.2f}%")

            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Таблица
        print("\n" + "=" * 60)
        print("TRAINED MODELS COMPARISON (on validation set)")
        print("=" * 60)
        for name, miou in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:20s} : {miou*100:6.2f}% mIoU")

        return results

    def compare_augmentations(
        self,
        model_type: str = "unet_smp",
        augmentation_levels: Optional[List[str]] = None,
        base_config: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Сравнение обучения с разными уровнями аугментаций
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

        print(f"\n{'='*70}")
        print("СРАВНЕНИЕ УРОВНЕЙ АУГМЕНТАЦИЙ")
        print(f"Модель: {model_type}")
        print(f"{'='*70}")

        results = []

        for aug_level in augmentation_levels:
            config = TrainingConfig(
                experiment_name=f"aug_comparison_{model_type}",
                model_type=model_type,
                augmentation_level=aug_level,
                **base_config,
            )
            result = self.train_experiment(config)
            results.append(result)

        # Создание сравнительной таблицы
        comparison_df = pd.DataFrame(
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = os.path.join(
            self.checkpoint_dir, f"augmentation_comparison_{model_type}_{timestamp}.csv"
        )
        comparison_df.to_csv(comparison_path, index=False)

        print(f"\n{'='*70}")
        print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
        print(f"{'='*70}")
        print(comparison_df.to_string(index=False))
        print(f"\n📊 Таблица сохранена: {comparison_path}")

        return comparison_df

    def plot_experiment_comparison(self, output_path: Optional[str] = None):
        """Визуализация сравнения экспериментов"""
        if len(self.experiment_results) < 2:
            print("⚠️ Нужно минимум 2 эксперимента для сравнения")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. mIoU по уровням аугментаций
        ax1 = axes[0, 0]
        miou_values = []
        aug_labels = []

        for result in self.experiment_results:
            aug_labels.append(result["augmentation_level"])
            miou_values.append(result["best_miou"] * 100)
        from matplotlib.colors import Colormap

        cmap: Colormap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 1, len(miou_values)))
        ax1.bar(aug_labels, miou_values, color=colors, edgecolor="black")
        ax1.set_xlabel("Уровень аугментаций")
        ax1.set_ylabel("Best mIoU (%)")
        ax1.set_title("Влияние аугментаций на качество")
        ax1.grid(axis="y", alpha=0.3)

        # Добавляем значения на столбцы
        for i, v in enumerate(miou_values):
            ax1.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

        # 2. Learning curves
        ax2 = axes[0, 1]
        for result in self.experiment_results:
            epochs = range(1, len(result["history"]["val_miou"]) + 1)
            ax2.plot(
                epochs,
                np.array(result["history"]["val_miou"]) * 100,
                label=f"{result['augmentation_level']} (mIoU={result['best_miou']*100:.2f}%)",
                linewidth=2,
            )

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation mIoU (%)")
        ax2.set_title("Динамика обучения")
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
                    f"{result['best_miou']*100:.2f}%",
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
            colLabels=["Augmentation", "Best mIoU", "Epochs", "Final Train Loss"],
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

    def train_all_models(
        self,
        augmentation_level: str = "medium",
        epochs: int = 20,
        batch_size: int = 4,
        lr: float = 1e-4,
        subset_fraction: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Обучение всех поддерживаемых моделей
        Returns:
            Dict с результатами по каждой модели
        """
        model_configs = [
            {"model_type": "unet_smp", "encoder_name": "resnet34", "variant": "b5"},
            {"model_type": "fpn_smp", "encoder_name": "resnet34", "variant": "b5"},
            {"model_type": "psp_smp", "encoder_name": "resnet34", "variant": "b5"},
            {"model_type": "deeplab_tv", "encoder_name": "resnet34", "variant": "b5"},
            {
                "model_type": "fcn_tv",
                "encoder_name": "resnet34",
                "variant": "fcn_resnet50",
            },
            {"model_type": "segnet", "encoder_name": "resnet34", "variant": "b5"},
        ]

        all_results = {}

        for config in model_configs:
            experiment_config = TrainingConfig(
                experiment_name=f"all_models_{config['model_type']}",
                model_type=config["model_type"],
                augmentation_level=augmentation_level,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                encoder_name=config["encoder_name"],
                variant=config["variant"],
                subset_fraction=subset_fraction,
            )
            result = self.train_experiment(experiment_config)
            all_results[config["model_type"]] = result

        # Сводная таблица
        summary_df = pd.DataFrame(
            [
                {
                    "Model": r["model_type"],
                    "Best mIoU (%)": r["best_miou"] * 100,
                    "Epochs": r["epochs_trained"],
                    "Checkpoint": os.path.basename(r["checkpoint_path"]),
                }
                for r in all_results.values()
            ]
        )

        summary_df = summary_df.sort_values("Best mIoU (%)", ascending=False)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(
            self.checkpoint_dir, f"all_models_summary_{timestamp}.csv"
        )
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'='*70}")
        print("СВОДНАЯ ТАБЛИЦА ВСЕХ МОДЕЛЕЙ")
        print(f"{'='*70}")
        print(summary_df.to_string(index=False))
        print(f"\n📊 Таблица сохранена: {summary_path}")

        return all_results

    def evaluate_checkpoints(
        self,
        checkpoint_paths: List[str],
        model_type: str,
        encoder_name: str = "resnet34",
        variant: str = "b5",
    ) -> pd.DataFrame:
        """
        Оценка обученных чекпоинтов на валидационном наборе
        """
        from sklearn.metrics import jaccard_score

        print(f"\n{'='*70}")
        print("ОЦЕНКА ЧЕКПОИНТОВ")
        print(f"{'='*70}")

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

        results = []
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
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # 🔥 Фильтрация aux_classifier для DeepLab (как в старом варианте)
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
            all_preds = []
            all_targets = []

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

            result = {
                "Checkpoint": checkpoint_name,
                "mIoU (%)": miou * 100,
                "Path": checkpoint_path,
            }
            results.append(result)

            print(f"✅ {checkpoint_name}: mIoU = {miou*100:.2f}%")

            # Очистка памяти
            del model
            torch.cuda.empty_cache()

        # Создание таблицы
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("mIoU (%)", ascending=False)

        # Сохранение
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_evaluation_{timestamp}.csv"
        )
        results_df.to_csv(eval_path, index=False)

        print(f"\n{'='*70}")
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
        print(f"{'='*70}")
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
