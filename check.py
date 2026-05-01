# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
import sys
import os

# Добавляем корень проекта в PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Теперь импорт сработает
from dataseters.ADE20KDataset import ADE20KDataset
from segmenters.NeuralTrainer import NeuralTrainer
from utils.strategies import SegNet

import os
from typing import List, Dict, Tuple, Any
from typing import Literal
import time
import gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.models.segmentation as tv_seg
import torchvision.models.detection as tv_det

from sklearn.metrics import jaccard_score


import segmentation_models_pytorch as smp

device = "cuda"


# ──────────────────────────────────────────────────────────────────────
def train_unet_ade20k(
    epochs=20, batch_size=4, subset_fraction=0.05, lr=1e-4, device="cuda"
):
    """Обучение U-Net на ADE20K"""
    print("🔹 Training U-Net (SMP) on ADE20K...")

    # DataLoader'ы
    train_dataset = ADE20KDataset(
        root_dir="./data/ade20k",
        split="training",
        image_size=(512, 512),
        augment=True,  # Аугментации для train
        subset_fraction=subset_fraction,
    )
    val_dataset = ADE20KDataset(
        root_dir="./data/ade20k",
        split="validation",
        image_size=(512, 512),
        augment=False,  # Аугментации для val
        subset_fraction=subset_fraction,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Модель
    model = smp.Unet(
        encoder_name="resnet34",  # Можно заменить на "resnet50", "efficientnet-b0"
        encoder_weights="imagenet",  # Предобученный encoder
        in_channels=3,
        classes=150,  # ADE20K имеет 150 классов
        activation=None,  # Без активации (CrossEntropy применяет softmax)
    )

    # Трейнер
    trainer = NeuralTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=150,
        lr=lr,
        device=device,
        ignore_index=255,
    )

    print("🔍 Validating first batch...")
    sample_batch = next(iter(train_loader))
    masks = sample_batch["mask"]
    print(f"   Mask range: [{masks.min()}, {masks.max()}]")
    print(f"   Unique values: {torch.unique(masks)[:20].tolist()}")
    assert masks.min() >= 0 and masks.max() <= 149, "Mask values out of range!"

    # Обучение
    history = trainer.fit(
        epochs=epochs, checkpoint_path="./models/unet_ade20k_best.pth"
    )

    return model, history


# ──────────────────────────────────────────────────────────────────────
def train_deeplab_ade20k(
    epochs: int = 20,
    batch_size: int = 4,
    subset_fraction: float = 0.05,
    lr: float = 1e-5,
) -> Tuple[Any, Dict[str, List]]:
    """Fine-tuning DeepLabV3+ на ADE20K"""
    print("🔹 Training DeepLabV3+ (Torchvision) on ADE20K...")

    # DataLoader'ы (те же)
    augmentation_level: Literal["none", "basic", "medium", "aggressive"] = "none"
    is_augmented = augmentation_level != "none"
    train_dataset = ADE20KDataset(
        root_dir="./data/ade20k",
        split="training",
        image_size=(512, 512),
        augment=is_augmented,
        augmentation_level=augmentation_level,
        hflip_prob=0.5,
        # vflip_prob=0.1 if augmentation_level == "aggressive" else 0.0,
        rotation_prob=(0.3 if augmentation_level in ["medium", "aggressive"] else 0.0),
        color_jitter_prob=(
            0.3 if augmentation_level in ["medium", "aggressive"] else 0.0
        ),
        scale_range=(
            (0.9, 1.1) if augmentation_level in ["medium", "aggressive"] else (1.0, 1.0)
        ),
        subset_fraction=subset_fraction,
    )
    val_dataset = ADE20KDataset(
        root_dir="./data/ade20k",
        split="validation",
        image_size=(512, 512),
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

    # Загрузка предобученной модели
    model = tv_seg.deeplabv3_resnet101(weights="COCO_WITH_VOC_LABELS_V1")

    # 🔥 Адаптация head под 150 классов
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, 150, kernel_size=1)

    # Инициализация нового слоя
    nn.init.normal_(model.classifier[4].weight, 0, 0.01)
    nn.init.constant_(model.classifier[4].bias, 0)

    # 🔥 Заморозка backbone на первые эпохи
    for param in model.backbone.parameters():
        param.requires_grad = False

    model = model.to(device).train()

    # Трейнер с отдельным оптимизатором для frozen backbone
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )

    trainer = NeuralTrainer.__new__(NeuralTrainer)
    trainer.model = model
    trainer.train_loader = train_loader
    trainer.val_loader = val_loader
    trainer.num_classes = 150
    trainer.device = device
    trainer.ignore_index = 255
    trainer.aux_loss_weight = 0.0
    trainer.criterion = nn.CrossEntropyLoss(ignore_index=255)
    trainer.optimizer = optimizer
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * epochs
    )
    trainer.best_miou = 0
    trainer.history = {"train_loss": [], "val_loss": [], "val_miou": []}

    # Обучение с разморозкой после 5 эпох
    print("🎯 Starting training (backbone frozen for first 5 epochs)...")

    for epoch in range(epochs):
        # 🔥 Разморозка backbone после 5 эпох
        if epoch == 5:
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Новый оптимизатор для всех параметров
            trainer.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr / 10, weight_decay=1e-4
            )
            trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer, T_max=len(train_loader) * (epochs - 5)
            )
            print("   🔓 Unfroze backbone, reduced LR")

        start_time = time.time()
        train_loss = trainer.train_epoch()
        val_loss, val_miou = trainer.validate()
        epoch_time = time.time() - start_time

        trainer.history["train_loss"].append(train_loss)
        trainer.history["val_loss"].append(val_loss)
        trainer.history["val_miou"].append(val_miou)

        print(f"\n📊 Epoch {epoch + 1}/{epochs} | Time: {epoch_time:.1f}s")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   Val mIoU:   {val_miou:.4f}")

        if val_miou > trainer.best_miou:
            trainer.best_miou = val_miou
            torch.save(
                model.state_dict(), "./models/deeplab_ade20k_best_201_epochs.pth"
            )
            print(f"   💾 Saved best model (mIoU: {val_miou:.4f})")

        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n✅ Training complete! Best mIoU: {trainer.best_miou:.4f}")
    return model, trainer.history


# ──────────────────────────────────────────────────────────────────────
def train_fpn_mit_ade20k(
    epochs=20, batch_size=4, subset_fraction=0.05, lr=5e-5, device="cuda", variant="b5"
):
    """Обучение FPN + Mix Transformer на ADE20K"""
    print(f"🔹 Training FPN + MiT-{variant} on ADE20K...")

    # DataLoader'ы
    train_dataset = ADE20KDataset(
        root_dir="./data/ade20k",
        split="training",
        image_size=(512, 512),
        augment=True,
        subset_fraction=subset_fraction,
    )
    val_dataset = ADE20KDataset(
        root_dir="./data/ade20k",
        split="validation",
        image_size=(512, 512),
        augment=False,
        subset_fraction=subset_fraction,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Модель
    encoder = f"mit_{variant}"
    model = smp.FPN(
        encoder_name=encoder,
        encoder_weights="imagenet",  # Предобученный encoder
        in_channels=3,
        classes=150,
        activation=None,
    )

    # Трейнер
    trainer = NeuralTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=150,
        lr=lr,
        device=device,
        ignore_index=255,
    )

    # Валидация первого батча
    print("🔍 Validating first batch...")
    sample_batch = next(iter(train_loader))
    masks = sample_batch["mask"]
    print(f"   Mask range: [{masks.min()}, {masks.max()}]")
    assert masks.min() >= 0 and masks.max() <= 149, "Mask values out of range!"

    # Обучение
    history = trainer.fit(
        epochs=epochs, checkpoint_path=f"./models/fpn_mit_{variant}_ade20k_best.pth"
    )

    return model, history


# ──────────────────────────────────────────────────────────────────────
def train_psp_mit_ade20k(
    epochs=20, batch_size=4, subset_fraction=0.05, lr=5e-5, device="cuda", variant="b5"
):
    """Обучение PSPNet + Mix Transformer на ADE20K"""
    print(f"🔹 Training PSPNet + MiT-{variant} on ADE20K...")

    # DataLoader'ы
    train_dataset = ADE20KDataset(
        root_dir="./data/ade20k",
        split="training",
        image_size=(512, 512),
        augment=True,
        subset_fraction=subset_fraction,
    )
    val_dataset = ADE20KDataset(
        root_dir="./data/ade20k",
        split="validation",
        image_size=(512, 512),
        augment=False,
        subset_fraction=subset_fraction,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Модель
    encoder = f"mit_{variant}"
    psp_size = 2048 if "mit" in encoder else 512

    model = smp.PSPNet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=150,
        activation=None,
        psp_size=psp_size,
    )

    # Трейнер
    trainer = NeuralTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=150,
        lr=lr,
        device=device,
        ignore_index=255,
    )

    # Валидация первого батча
    print("🔍 Validating first batch...")
    sample_batch = next(iter(train_loader))
    masks = sample_batch["mask"]
    print(f"   Mask range: [{masks.min()}, {masks.max()}]")
    assert masks.min() >= 0 and masks.max() <= 149, "Mask values out of range!"

    # Обучение
    history = trainer.fit(
        epochs=epochs, checkpoint_path=f"./models/psp_mit_{variant}_ade20k_best.pth"
    )

    return model, history


# ──────────────────────────────────────────────────────────────────────
def train_fcn_resnet50_ade20k(
    epochs=20,
    batch_size=4,
    subset_fraction=0.05,
    lr=1e-5,
    device="cuda",
    variant="fcn_resnet50",
):
    """Обучение FCN ResNet-50 на ADE20K (fine-tuning)"""
    print(f"🔹 Training {variant} on ADE20K...")

    # DataLoader'ы
    augmentation_level = "medium"
    is_augmented = augmentation_level != "none"
    train_dataset = ADE20KDataset(
        root_dir="./data/ade20k",
        split="training",
        image_size=(512, 512),
        augment=is_augmented,
        augmentation_level=augmentation_level,
        hflip_prob=0.5,
        # vflip_prob=0.1 if augmentation_level == "aggressive" else 0.0,
        rotation_prob=(0.3 if augmentation_level in ["medium", "aggressive"] else 0.0),
        color_jitter_prob=(
            0.3 if augmentation_level in ["medium", "aggressive"] else 0.0
        ),
        scale_range=(
            (0.9, 1.1) if augmentation_level in ["medium", "aggressive"] else (1.0, 1.0)
        ),
        subset_fraction=subset_fraction,
    )
    val_dataset = ADE20KDataset(
        root_dir="./data/ade20k",
        split="validation",
        image_size=(512, 512),
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

    variants = {
        "fcn_resnet50": tv_seg.fcn_resnet50,
        "fcn_resnet101": tv_seg.fcn_resnet101,
    }

    if variant not in variants:
        raise ValueError(f"Unknown FCN variant: {variant}")

    # Загрузка с предобученным backbone (ImageNet)
    model = variants[variant](weights="DEFAULT")

    # Адаптация head под 150 классов
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, 150, kernel_size=1)

    # Инициализация нового слоя
    nn.init.normal_(model.classifier[4].weight, 0, 0.01)
    nn.init.constant_(model.classifier[4].bias, 0)

    # Заморозка backbone на первые эпохи
    for param in model.backbone.parameters():
        param.requires_grad = False

    model = model.to(device).train()

    # Оптимизатор только для classifier
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )

    # Трейнер с кастомным оптимизатором
    trainer = NeuralTrainer.__new__(NeuralTrainer)
    trainer.model = model
    trainer.train_loader = train_loader
    trainer.val_loader = val_loader
    trainer.num_classes = 150
    trainer.device = device
    trainer.ignore_index = 255
    trainer.aux_loss_weight = 0.0
    trainer.criterion = nn.CrossEntropyLoss(ignore_index=255)
    trainer.optimizer = optimizer
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * epochs
    )
    trainer.best_miou = 0
    trainer.history = {"train_loss": [], "val_loss": [], "val_miou": []}

    # Обучение с разморозкой после 5 эпох
    print("🎯 Starting training (backbone frozen for first 5 epochs)...")

    for epoch in range(epochs):
        # Разморозка backbone после 5 эпох
        if epoch == 5:
            for param in model.backbone.parameters():
                param.requires_grad = True
            trainer.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr / 10, weight_decay=1e-4
            )
            trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer, T_max=len(train_loader) * (epochs - 5)
            )
            print("   🔓 Unfroze backbone, reduced LR")

        start_time = time.time()
        train_loss = trainer.train_epoch()
        val_loss, val_miou = trainer.validate()
        epoch_time = time.time() - start_time

        trainer.history["train_loss"].append(train_loss)
        trainer.history["val_loss"].append(val_loss)
        trainer.history["val_miou"].append(val_miou)

        print(f"\n📊 Epoch {epoch + 1}/{epochs} | Time: {epoch_time:.1f}s")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   Val mIoU:   {val_miou:.4f}")

        if val_miou > trainer.best_miou:
            trainer.best_miou = val_miou
            torch.save(
                model.state_dict(),
                f"./models/fcn_{variant.replace('fcn_', '')}_ade20k_best_201_epochs.pth",
            )
            print(f"   💾 Saved best model (mIoU: {val_miou:.4f})")

        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n✅ Training complete! Best mIoU: {trainer.best_miou:.4f}")
    return model, trainer.history


# ──────────────────────────────────────────────────────────────────────
def train_segnet_ade20k(
    epochs=20, batch_size=4, subset_fraction=0.05, lr=1e-4, device="cuda"
):
    """Обучение SegNet на ADE20K"""
    print("🔹 Training SegNet on ADE20K...")

    # DataLoader'ы
    train_dataset = ADE20KDataset(
        root_dir="./data/ade20k",
        split="training",
        image_size=(512, 512),
        augment=True,
        subset_fraction=subset_fraction,
    )
    val_dataset = ADE20KDataset(
        root_dir="./data/ade20k",
        split="validation",
        image_size=(512, 512),
        augment=False,
        subset_fraction=subset_fraction,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Модель (кастомная SegNet или U-Net как proxy)
    try:
        # Попытка использовать SMP U-Net как SegNet proxy
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=150,
            activation=None,
        )
        print("   Using SMP U-Net as SegNet proxy")
    except Exception:
        # Fallback к кастомной реализации
        model = SegNet(num_classes=150)

        def _init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        model.apply(_init_weights)
        print("   Using custom SegNet implementation")

    model = model.to(device).train()

    # Трейнер
    trainer = NeuralTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=150,
        lr=lr,
        device=device,
        ignore_index=255,
    )

    # Валидация первого батча
    print("🔍 Validating first batch...")
    sample_batch = next(iter(train_loader))
    masks = sample_batch["mask"]
    print(f"   Mask range: [{masks.min()}, {masks.max()}]")
    assert masks.min() >= 0 and masks.max() <= 149, "Mask values out of range!"

    # Обучение
    history = trainer.fit(
        epochs=epochs, checkpoint_path="./models/segnet_ade20k_best.pth"
    )

    return model, history


# ──────────────────────────────────────────────────────────────────────
def compare_trained_models():
    """Сравнение обученных моделей на валидационном наборе"""
    print("\n" + "=" * 60)
    print("📊 COMPARING TRAINED MODELS ON ADE20K VALIDATION")
    print("=" * 60)

    # Загружаем чекпоинты (если есть)
    models = {}

    # === U-Net ===
    if os.path.exists("./models/unet_ade20k_best.pth"):
        unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=150,
            activation=None,
        )
        checkpoint = torch.load("./models/unet_ade20k_best.pth", map_location=device)
        unet.load_state_dict(
            checkpoint["model_state_dict"]
        )  # 🔥 Ключ 'model_state_dict'
        models["U-Net"] = unet.to(device).eval()
        print("✅ Loaded U-Net")

    # === DeepLabV3+ ===
    if os.path.exists("./models/deeplab_ade20k_best.pth"):
        deeplab = tv_seg.deeplabv3_resnet101(weights=None)
        deeplab.classifier[4] = nn.Conv2d(256, 150, kernel_size=1)
        checkpoint = torch.load("./models/deeplab_ade20k_best.pth", map_location=device)
        model_keys = {
            k: v for k, v in checkpoint.items() if not k.startswith("aux_classifier")
        }
        deeplab.load_state_dict(model_keys, strict=False)
        # deeplab.load_state_dict(checkpoint, strict=False)  # 🔥 strict=False!

        models["DeepLabV3+"] = deeplab.to(device).eval()
        print("✅ Loaded DeepLabV3+")

    # === Mask R-CNN (опционально) ===
    if os.path.exists("./models/maskrcnn_ade20k_semantic_best.pth"):
        mrcnn = tv_det.maskrcnn_resnet50_fpn(weights=None, pretrained=False)
        mrcnn.roi_heads.box_predictor = tv_det.faster_rcnn.FastRCNNPredictor(
            mrcnn.roi_heads.box_predictor.cls_score.in_features, 151
        )
        checkpoint = torch.load(
            "./models/maskrcnn_ade20k_semantic_best.pth", map_location=device
        )
        mrcnn.load_state_dict(checkpoint, strict=False)
        models["Mask R-CNN"] = mrcnn.to(device).eval()
        print("✅ Loaded Mask R-CNN")

    # === FPN + MiT ===
    if os.path.exists("./models/fpn_mit_b5_ade20k_best.pth"):
        fpn = smp.FPN(
            encoder_name="mit_b5",
            encoder_weights="imagenet",
            in_channels=3,
            classes=150,
            activation=None,
        )
        checkpoint = torch.load(
            "./models/fpn_mit_b5_ade20k_best.pth", map_location=device
        )
        if "model_state_dict" in checkpoint:
            fpn.load_state_dict(checkpoint["model_state_dict"])
        else:
            fpn.load_state_dict(checkpoint)
        models["FPN+MiT-B5"] = fpn.to(device).eval()
        print("✅ Loaded FPN + MiT-B5")

    # === PSPNet + MiT ===
    if os.path.exists("./models/psp_mit_b5_ade20k_best.pth"):
        psp = smp.PSPNet(
            encoder_name="mit_b5",
            encoder_weights="imagenet",
            in_channels=3,
            classes=150,
            activation=None,
            psp_size=2048,
        )
        checkpoint = torch.load(
            "./models/psp_mit_b5_ade20k_best.pth", map_location=device
        )
        if "model_state_dict" in checkpoint:
            psp.load_state_dict(checkpoint["model_state_dict"])
        else:
            psp.load_state_dict(checkpoint)
        models["PSP+MiT-B5"] = psp.to(device).eval()
        print("✅ Loaded PSPNet + MiT-B5")

    # === FCN ResNet-50 ===
    if os.path.exists("./models/fcn_resnet50_ade20k_best.pth"):
        fcn = tv_seg.fcn_resnet50(weights=None)
        fcn.classifier[4] = nn.Conv2d(512, 150, kernel_size=1)
        checkpoint = torch.load(
            "./models/fcn_resnet50_ade20k_best.pth", map_location=device
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            fcn.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            fcn.load_state_dict(checkpoint, strict=False)
        models["FCN ResNet-50"] = fcn.to(device).eval()
        print("✅ Loaded FCN ResNet-50")

    # === SegNet ===
    if os.path.exists("./models/segnet_ade20k_best.pth"):
        try:
            segnet = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=150,
                activation=None,
            )
        except Exception:
            segnet = SegNet(num_classes=150)

        checkpoint = torch.load("./models/segnet_ade20k_best.pth", map_location=device)
        if "model_state_dict" in checkpoint:
            segnet.load_state_dict(checkpoint["model_state_dict"])
        else:
            segnet.load_state_dict(checkpoint)
        models["SegNet"] = segnet.to(device).eval()
        print("✅ Loaded SegNet")

    if not models:
        print("⚠️  No trained models found. Run training first.")
        return None

    # Валидационный датасет
    val_dataset = ADE20KDataset(
        root_dir="./data/ade20k",
        split="validation",
        image_size=(512, 512),
        augment=False,
        subset_fraction=0.05,  # 5% для быстрого сравнения
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
                images = batch["image"].to(device)
                masks_gt = batch["mask"].to(device)

                if name == "Mask R-CNN":
                    # Особый инференс для Mask R-CNN
                    images_list = [images[i] for i in range(images.shape[0])]
                    outputs = model(images_list)
                    # for out, gt in zip(outputs, masks_gt):
                    #     pred = convert_maskrcnn_to_semantic([out], gt.shape)
                    #     all_preds.extend(pred.cpu().flatten().tolist())
                    #     all_targets.extend(gt.cpu().flatten().tolist())
                else:
                    # Стандартный инференс
                    outputs = model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs["out"]
                    preds = outputs.argmax(1)
                    all_preds.extend(preds.cpu().flatten().tolist())
                    all_targets.extend(masks_gt.cpu().flatten().tolist())

                # Прогресс
                if (batch_idx + 1) % 10 == 0:
                    print(f"   Processed {batch_idx + 1}/{len(val_loader)} batches")

        # Вычисление mIoU
        miou = jaccard_score(
            all_targets,
            all_preds,
            average="weighted",
            labels=range(150),
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


# ──────────────────────────────────────────────────────────────────────
def evaluate_trained_models_on_val(checkpoints, val_fraction=0.05, device="cuda"):
    """Оценка обученных моделей на валидационном наборе"""

    # Валидационный датасет (тот же, что использовали при обучении)
    val_dataset = ADE20KDataset(
        root_dir="./data/ade20k",
        split="validation",
        image_size=(512, 512),
        augment=False,
        subset_fraction=val_fraction,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    results = {}

    for model_name, checkpoint_path in checkpoints.items():
        print(f"\n🔹 Evaluating {model_name}...")
        if not os.path.exists(checkpoint_path):
            print(f"   ⚠️  Checkpoint not found: {checkpoint_path}")
            continue

        # 🔥 Инициализируем model = None перед условиями
        model = None

        # Загрузка модели
        if "unet" in model_name.lower() or "u-net" in model_name.lower():
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=150,
                activation=None,
            )
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"   ✅ Loaded U-Net from {checkpoint_path}")

        elif "deeplab" in model_name.lower():
            model = tv_seg.deeplabv3_resnet101(weights=None)
            model.classifier[4] = nn.Conv2d(256, 150, kernel_size=1)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Фильтруем aux_classifier если есть
            model_keys = {
                k: v
                for k, v in checkpoint.items()
                if not k.startswith("aux_classifier")
            }
            model.load_state_dict(model_keys, strict=False)
            print(f"   ✅ Loaded DeepLabV3+ from {checkpoint_path}")

        # 🔥 FPN + MiT
        elif "fpn" in model_name.lower() or "fpn_mit" in model_name.lower():
            model = smp.FPN(
                encoder_name="mit_b5",
                encoder_weights="imagenet",
                in_channels=3,
                classes=150,
                activation=None,
            )
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"   ✅ Loaded FPN+MiT from {checkpoint_path}")

        # 🔥 PSPNet + MiT
        elif "psp" in model_name.lower() or "psp_mit" in model_name.lower():
            model = smp.PSPNet(
                encoder_name="mit_b5",
                encoder_weights="imagenet",
                in_channels=3,
                classes=150,
                activation=None,
                psp_size=2048,
            )
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"   ✅ Loaded PSPNet+MiT from {checkpoint_path}")

        # 🔥 FCN
        elif "fcn" in model_name.lower():
            model = tv_seg.fcn_resnet50(weights=None)
            model.classifier[4] = nn.Conv2d(512, 150, kernel_size=1)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"   ✅ Loaded FCN from {checkpoint_path}")

        # 🔥 SegNet
        elif "segnet" in model_name.lower():
            try:
                model = smp.Unet(
                    encoder_name="resnet34",
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=150,
                    activation=None,
                )
            except Exception:
                model = SegNet(num_classes=150)

            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"   ✅ Loaded SegNet from {checkpoint_path}")

        else:
            print(f"   ⚠️  Unknown model type: {model_name}. Skipping...")
            continue  # 🔥 Пропускаем неизвестные модели

        # 🔥 Теперь model гарантированно определён
        model = model.to(device).eval()

        # Инференс на всём валидационном наборе
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                masks_gt = batch["mask"].to(device)

                outputs = model(images)
                if isinstance(outputs, dict):
                    outputs = outputs["out"]
                preds = outputs.argmax(1)

                all_preds.extend(preds.cpu().flatten().tolist())
                all_targets.extend(masks_gt.cpu().flatten().tolist())

                # Прогресс
                if (batch_idx + 1) % 20 == 0:
                    print(f"   Processed {batch_idx + 1}/{len(val_loader)} batches")

        # Метрики
        miou = jaccard_score(
            all_targets,
            all_preds,
            average="weighted",
            labels=range(150),
            zero_division=0,
        )
        results[model_name] = miou
        print(f"   ✅ mIoU: {miou * 100:.2f}%")

        # Очистка памяти
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


# Запуск обучения (для теста: 20 эпох, 5% данных)
# unet_model, unet_history = train_unet_ade20k(epochs=20, subset_fraction=0.05)

# Пример теста: U-Net: 5 эпох, 5% данных, batch=2
# unet_model, _ = train_unet_ade20k(epochs=5, batch_size=2, subset_fraction=0.05, lr=1e-4)
# train_unet_ade20k(epochs=50, batch_size=8, subset_fraction=1.0, lr=1e-4)

# Запуск (для теста)
deeplab_model, deeplab_history = train_deeplab_ade20k(epochs=200, subset_fraction=0.05)

# Пример теста: DeepLab: 5 эпох, 5% данных
# deeplab_model, _ = train_deeplab_ade20k(epochs=5, batch_size=2, subset_fraction=0.05, lr=1e-5)
# train_deeplab_ade20k(epochs=30, batch_size=4, subset_fraction=1.0, lr=1e-5)

# Запуск (для теста)
# fpn_model, fpn_history = train_fpn_mit_ade20k(epochs=20, subset_fraction=0.05, variant="b5")

# Запуск (для теста)
# psp_model, psp_history = train_psp_mit_ade20k(epochs=20, subset_fraction=0.05, variant="b5")

# Запуск (для теста)
# fcn_model, fcn_history = train_fcn_resnet50_ade20k(epochs=200, subset_fraction=0.05)

# Запуск (для теста)
# segnet_model, segnet_history = train_segnet_ade20k(epochs=20, subset_fraction=0.05)

results = compare_trained_models()
print(f"Results: {results}")

checkpoints: Dict[str, str] = {
    "U-Net (trained)": "./models/unet_ade20k_best_200_epochs.pth",
    "DeepLabV3+ (trained)": "./models/deeplab_ade20k_best_200_epochs.pth",
    "FPN+MiT-B5 (trained)": "./models/fpn_mit_b5_ade20k_best_200_epochs.pth",
    "PSPNet+MiT-B5 (trained)": "./models/psp_mit_b5_ade20k_best_200_epochs.pth",
    "FCN ResNet-50 (trained)": "./models/fcn_resnet50_ade20k_best_201_epochs.pth",
    "SegNet (trained)": "./models/segnet_ade20k_best_200_epochs.pth",
}
trained_results = evaluate_trained_models_on_val(checkpoints, val_fraction=0.05)
