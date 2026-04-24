# datasets/ADE20KDataset.py

# Импорт основных библиотек
import os
from typing import (
    Optional,
    List,
    Tuple,
    Dict,
    Any,
)
import random
import traceback
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision import transforms


class ADE20KDataset(Dataset):
    """
    Загрузчик датасета ADE20K с расширенными аугментациями.
    Все геометрические трансформации применяются одинаково к изображению и маске.
    """

    def __init__(
        self,
        root_dir: str = "./data/ade20k",
        split: str = "training",
        image_size: tuple = (512, 512),
        augment: bool = False,
        subset_fraction: Optional[float] = None,
        augmentation_level: str = "basic",  # 'none', 'basic', 'medium', 'aggressive'
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.0,
        rotation_prob: float = 0.0,
        color_jitter_prob: float = 0.0,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        ignore_index: int = 255,
    ) -> None:
        self.image_size: tuple = image_size
        self.augment: bool = augment
        self.augmentation_level: str = augmentation_level
        self.ignore_index: int = ignore_index

        # Настройка уровня аугментаций
        self._configure_augmentations(
            hflip_prob=hflip_prob,
            vflip_prob=vflip_prob,
            rotation_prob=rotation_prob,
            color_jitter_prob=color_jitter_prob,
            scale_range=scale_range,
        )

        base_dir: str = os.path.join(root_dir, "ADEChallengeData2016")
        self.images_dir: str = os.path.join(base_dir, "images", split)
        self.masks_dir: str = os.path.join(base_dir, "annotations", split)

        print(f"📂 Загрузка {split} датасета...")

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Masks dir not found: {self.masks_dir}")

        self.image_files: List[str] = sorted(
            [f for f in os.listdir(self.images_dir) if f.endswith(".jpg")]
        )
        print(f"   Найдено {len(self.image_files)} изображений")

        self.valid_indices: list = []
        for i, img_file in enumerate(self.image_files):
            mask_file = img_file.replace(".jpg", ".png")
            if os.path.exists(os.path.join(self.masks_dir, mask_file)):
                self.valid_indices.append(i)
        print(f"   Валидных пар: {len(self.valid_indices)}")

        if subset_fraction is not None and subset_fraction < 1.0:
            n = int(len(self.valid_indices) * subset_fraction)
            self.valid_indices = self.valid_indices[:n]
            print(f"   Используем {n} образцов ({subset_fraction * 100:.0f}%)")

        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # ──────────────────────────────────────────────────────────────────────
    def _configure_augmentations(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.0,
        rotation_prob: float = 0.0,
        color_jitter_prob: float = 0.0,
        scale_range: Tuple[float, float] = (0.8, 1.2),
    ) -> None:
        """Настройка параметров аугментаций в зависимости от уровня"""

        if self.augmentation_level == "none":
            self.hflip_prob = 0.0
            self.vflip_prob = 0.0
            self.rotation_prob = 0.0
            self.color_jitter_prob = 0.0
            self.scale_range = (1.0, 1.0)

        elif self.augmentation_level == "basic":
            # Базовые аугментации (рекомендуется для начала)
            self.hflip_prob = hflip_prob  # 0.5
            self.vflip_prob = 0.0
            self.rotation_prob = 0.0
            self.color_jitter_prob = 0.0
            self.scale_range = (1.0, 1.0)

        elif self.augmentation_level == "medium":
            # Средние аугментации (хороший баланс)
            self.hflip_prob = hflip_prob  # 0.5
            self.vflip_prob = vflip_prob  # 0.1
            self.rotation_prob = rotation_prob  # 0.3
            self.color_jitter_prob = color_jitter_prob  # 0.3
            self.scale_range = scale_range  # (0.9, 1.1)

        elif self.augmentation_level == "aggressive":
            # Агрессивные аугментации (для больших датасетов)
            self.hflip_prob = 0.5
            self.vflip_prob = 0.1
            self.rotation_prob = 0.5
            self.color_jitter_prob = 0.5
            self.scale_range = (0.8, 1.2)
        else:
            raise ValueError(f"Unknown augmentation_level: {self.augmentation_level}")

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.valid_indices)

    # ──────────────────────────────────────────────────────────────────────
    def _apply_geometric_augmentations(
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Геометрические аугментации применяются ОДИНАКОВО к image и mask.
        Для маски: fill = self.ignore_index (не 0, не 255 по умолчанию —
        именно то значение, которое проигнорирует CrossEntropyLoss).
        """
        transforms_applied = []

        # 1. Random Horizontal Flip
        if self.augment and random.random() < self.hflip_prob:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
            transforms_applied.append("hflip")

        # 2. Random Vertical Flip
        if self.augment and random.random() < self.vflip_prob:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
            transforms_applied.append("vflip")

        # 3. Random Rotation
        if self.augment and random.random() < self.rotation_prob:
            angle = random.uniform(-30, 30)  # ±30 градусов
            img = TF.rotate(img, angle, fill=(0, 0, 0))  # fill=0 для RGB
            mask = TF.rotate(mask, angle, fill=self.ignore_index)  # fill=255 для маски
            transforms_applied.append(f"rotate_{angle:.1f}°")

        # 4. Random Scale + Crop (если scale_range != (1.0, 1.0))
        if self.augment and self.scale_range != (1.0, 1.0):
            scale = random.uniform(*self.scale_range)
            orig_w, orig_h = img.size
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))

            # Ресайз
            img = TF.resize(
                img, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR
            )
            mask = TF.resize(
                mask, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST
            )

            # Crop/Pad к исходному размеру
            if new_w >= orig_w and new_h >= orig_h:
                # Crop
                left = random.randint(0, new_w - orig_w)
                top = random.randint(0, new_h - orig_h)
                img = TF.crop(img, top, left, orig_h, orig_w)
                mask = TF.crop(mask, top, left, orig_h, orig_w)
            else:
                # Pad
                pad_w = max(0, orig_w - new_w)
                pad_h = max(0, orig_h - new_h)
                padding = [
                    pad_w // 2,
                    pad_h // 2,
                    pad_w - pad_w // 2,
                    pad_h - pad_h // 2,
                ]
                img = TF.pad(img, padding, fill=(0, 0, 0))
                mask = TF.pad(mask, padding, fill=self.ignore_index)
            transforms_applied.append(f"scale_{scale:.2f}")

        # 5. Random Affine (опционально, для aggressive уровня)
        if (
            self.augment
            and self.augmentation_level == "aggressive"
            and random.random() < 0.3
        ):
            angle = random.uniform(-15, 15)
            translate = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-10, 10)

            img = TF.affine(
                img,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                fill=self.ignore_index,
            )
            transforms_applied.append(
                f"random_affine_{angle:.2f}°_translate={translate}_scale={scale}_shear={shear}"
            )
        if transforms_applied and self.augment:
            print(f"🔧 Applied: {', '.join(transforms_applied)}")

        return img, mask

    # ──────────────────────────────────────────────────────────────────────
    def _apply_photometric_augmentations(self, img: Image.Image) -> Image.Image:
        """
        Применяет фотометрические аугментации только к изображению.
        Маска НЕ трансформируется!
        """

        if not self.augment:
            return img

        # 1. Color Jitter
        if random.random() < self.color_jitter_prob:
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
            img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
            # Hue не рекомендуется для сегментации (может изменить классы)

        # 2. Random Grayscale
        if self.augmentation_level == "aggressive" and random.random() < 0.1:
            img = TF.to_grayscale(
                img, num_output_channels=3
            )  # Конвертируем обратно в RGB

        # 3. Random Gamma
        if self.augmentation_level == "aggressive" and random.random() < 0.2:
            gamma = random.uniform(0.7, 1.5)
            img = TF.adjust_gamma(img, gamma)

        return img

    # ──────────────────────────────────────────────────────────────────────
    def __getitem__(self, idx) -> Dict[str, Any]:
        real_idx = self.valid_indices[idx]
        img_file = self.image_files[real_idx]
        img = Image.open(os.path.join(self.images_dir, img_file)).convert("RGB")
        mask_pil = Image.open(
            os.path.join(self.masks_dir, img_file.replace(".jpg", ".png"))
        ).convert("L")
        # 1. Применяем геометрические аугментации
        img, mask_pil = self._apply_geometric_augmentations(img, mask_pil)

        # 2. Фотометрические аугментации (только к изображению)
        img = self._apply_photometric_augmentations(img)

        # 3. Resize к целевому размеру
        if (img.width, img.height) != (self.image_size[1], self.image_size[0]):
            img = TF.resize(
                img,
                self.image_size,
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask_pil = TF.resize(
                mask_pil,
                self.image_size,
                interpolation=TF.InterpolationMode.NEAREST,
            )

        # 4. Маска → numpy int64
        mask_np: np.ndarray = np.array(mask_pil, dtype=np.int64)

        # FIX: clip только НЕ-ignore пикселей
        # Было: np.clip(mask_np, 0, 149) — уничтожало ignore_index=255
        # Теперь: ignore-пиксели остаются нетронутыми, остальные → [0, 149]
        valid_mask = mask_np != self.ignore_index
        mask_np[valid_mask] = np.clip(mask_np[valid_mask], 0, 149)

        # 5. Нормализация изображения
        img_tensor: torch.Tensor = self.img_transform(img)

        # 6. Маска → tensor
        mask_tensor: torch.Tensor = torch.from_numpy(mask_np).long()

        return {"image": img_tensor, "mask": mask_tensor, "image_id": img_file}

    # ──────────────────────────────────────────────────────────────────────
    def test_augmentation_sync(self) -> bool:
        """Простой unit-тест: image и mask одного размера после аугментаций."""
        sample = self[0]
        h, w = sample["image"].shape[1], sample["image"].shape[2]
        mh, mw = sample["mask"].shape
        assert (h, w) == (mh, mw), f"Shape mismatch: image=({h},{w}) mask=({mh},{mw})"
        # ignore-пиксели не должны менять диапазон
        m = sample["mask"]
        valid = m[m != self.ignore_index]
        if len(valid) > 0:
            assert (
                valid.min() >= 0 and valid.max() <= 149
            ), f"Mask values out of range: [{valid.min()}, {valid.max()}]"
        return True


class ADE20KDatasetWithTransforms(Dataset):
    """
    Версия с использованием Compose трансформаций.
    Более чистая архитектура, но требует careful handling масок.
    """

    def __init__(
        self,
        root_dir: str = "./data/ade20k",
        split: str = "training",
        image_size: tuple = (512, 512),
        augment: bool = False,
        subset_fraction: Optional[float] = None,
        ignore_index: int = 255,
    ) -> None:
        self.image_size = image_size
        self.augment = augment
        self.ignore_index = ignore_index

        base_dir = os.path.join(root_dir, "ADEChallengeData2016")
        self.images_dir = os.path.join(base_dir, "images", split)
        self.masks_dir = os.path.join(base_dir, "annotations", split)

        # Раздельные трансформации для train/val
        if augment and split == "training":
            self.transform = self._get_train_transforms()
        else:
            self.transform = self._get_val_transforms()

        self.image_files = sorted(
            [f for f in os.listdir(self.images_dir) if f.endswith(".jpg")]
        )
        self.valid_indices = []
        for i, img_file in enumerate(self.image_files):
            mask_file = img_file.replace(".jpg", ".png")
            if os.path.exists(os.path.join(self.masks_dir, mask_file)):
                self.valid_indices.append(i)

        if subset_fraction is not None and subset_fraction < 1.0:
            n = int(len(self.valid_indices) * subset_fraction)
            self.valid_indices = self.valid_indices[:n]

    def _get_train_transforms(
        self,
    ) -> Dict[str, transforms.Compose]:  # ✅ Добавлен метод
        """Трансформации для обучения с аугментациями"""
        return {
            "image": transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "mask": transforms.Compose(
                [
                    # Только ToTensor для маски, без нормализации
                    transforms.ToTensor(),
                ]
            ),
        }

    def _get_val_transforms(self) -> Dict[str, transforms.Compose]:
        """Трансформации для валидации"""
        return {
            "image": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "mask": transforms.Compose(
                [
                    # Без аугментаций
                ]
            ),
        }

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx) -> Dict[str, Any]:
        real_idx = self.valid_indices[idx]
        img_file = self.image_files[real_idx]

        # Загрузка
        img = Image.open(os.path.join(self.images_dir, img_file)).convert("RGB")
        mask = Image.open(
            os.path.join(self.masks_dir, img_file.replace(".jpg", ".png"))
        ).convert("L")

        # Применяем геометрические аугментации
        if self.augment:
            # RandomHorizontalFlip
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # RandomRotation
            if random.random() > 0.7:
                angle = random.uniform(-15, 15)
                img = TF.rotate(img, angle, fill=(0, 0, 0))
                mask = TF.rotate(mask, angle, fill=self.ignore_index)

        # Resize
        img = TF.resize(
            img, self.image_size, interpolation=TF.InterpolationMode.BILINEAR
        )
        mask = TF.resize(
            mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST
        )

        # Конвертация в tensor
        # img_tensor: torch.Tensor = self.transform["image"](img)
        # mask_tensor: torch.Tensor = self.transform["mask"](mask)

        # # Для маски используем long для совместимости с loss функциями
        # mask_tensor = mask_tensor.squeeze(0).long()  # (1, H, W) -> (H, W)
        img_tensor: torch.Tensor = self.transform["image"](img)
        mask_np = np.array(mask, dtype=np.int64)
        valid = mask_np != self.ignore_index
        mask_np[valid] = np.clip(mask_np[valid], 0, 149)
        mask_tensor = torch.from_numpy(mask_np).long()

        return {"image": img_tensor, "mask": mask_tensor, "image_id": img_file}


# ──────────────────────────────────────────────────────────────────────────────
def test_dataloader() -> bool:
    print("\n" + "=" * 50)
    print("Тестирование загрузчика ADE20K")
    print("=" * 50)
    try:
        train_dataset = ADE20KDataset(
            root_dir="./data/ade20k",
            split="training",
            image_size=(512, 512),
            augment=True,
            augmentation_level="medium",
            hflip_prob=0.5,
            rotation_prob=0.3,
            color_jitter_prob=0.3,
            subset_fraction=0.01,
            ignore_index=0,
        )

        #  Начинаем с num_workers=0 для отладки
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        print(f"✅ DataLoader ready: {len(train_loader)} batches")
        print("\n📊 Проверка загрузки данных:")
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"]
            masks = batch["mask"]
            print(f"\nBatch {batch_idx + 1}:")
            print(
                f"   Images: {images.shape}, dtype={images.dtype}, range=[{images.min():.3f}, {images.max():.3f}]"
            )
            print(
                f"   Masks: {masks.shape}, dtype={masks.dtype}, unique={torch.unique(masks)[:15].tolist()}"
            )
            assert not torch.isnan(images).any(), "NaN in images!"
            assert (
                masks.min() >= 0 and masks.max() <= 150
            ), f"Mask out of range: [{masks.min()}, {masks.max()}]"
            non_ignore = masks[masks != 0]
            if len(non_ignore) > 0:
                assert (
                    non_ignore.min() >= 1 and non_ignore.max() <= 149
                ), f"Non-ignore mask values out of range: [{non_ignore.min()}, {non_ignore.max()}]"
            print("   ✅ Batch valid")
            if batch_idx >= 2:
                break
        print("\n🔍 Визуализация аугментаций...")
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        for i in range(4):
            batch = next(iter(train_loader))
            img = batch["image"][0].permute(1, 2, 0).numpy()
            mask = batch["mask"][0].numpy()

            # Denormalize image
            img = img * np.array([0.229, 0.224, 0.225]) + np.array(
                [0.485, 0.456, 0.406]
            )
            img = np.clip(img, 0, 1)

            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Image {i + 1}")
            axes[0, i].axis("off")

            axes[1, i].imshow(mask, cmap="tab20")
            axes[1, i].set_title(f"Mask {i + 1}")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig("./data/ade20k_augmentations_preview.png", dpi=150)
        print("   ✅ Preview saved to ./data/ade20k_augmentations_preview.png")
        print("\n✅ Все тесты пройдены!")
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()
        return False
