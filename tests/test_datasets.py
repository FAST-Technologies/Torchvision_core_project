# tests/test_datasets.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import os
import torch


def import_ade20k():
    from datasets.ADE20KDataset import ADE20KDataset

    return ADE20KDataset


class TestADE20KDataset:
    @pytest.fixture
    def temp_dataset_dir(self, tmp_path):
        """Создаёт временную структуру ADE20K"""
        base_dir = tmp_path / "ADEChallengeData2016"
        images_dir = base_dir / "images" / "training"
        masks_dir = base_dir / "annotations" / "training"

        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Создаём тестовые изображения
        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            img.save(images_dir / f"test_{i}.jpg")

            mask = Image.fromarray(
                np.random.randint(0, 150, (256, 256), dtype=np.uint8)
            )
            mask.save(masks_dir / f"test_{i}.png")

        return str(tmp_path)

    def test_import(self):
        ADE20KDataset = import_ade20k()
        assert ADE20KDataset is not None

    def test_dataset_initialization(self, temp_dataset_dir):
        ADE20KDataset = import_ade20k()
        dataset = ADE20KDataset(
            root_dir=temp_dataset_dir,
            split="training",
            image_size=(128, 128),
            augment=False,
        )
        assert len(dataset) == 3
        assert dataset.image_size == (128, 128)

    def test_dataset_getitem(self, temp_dataset_dir):
        ADE20KDataset = import_ade20k()
        dataset = ADE20KDataset(
            root_dir=temp_dataset_dir,
            split="training",
            image_size=(128, 128),
            augment=False,
        )

        item = dataset[0]
        assert "image" in item
        assert "mask" in item
        assert "image_id" in item

        # Проверка размеров
        assert item["image"].shape == (3, 128, 128)  # (C, H, W)
        assert item["mask"].shape == (128, 128)

        # Проверка типов
        assert item["image"].dtype == torch.float32
        assert item["mask"].dtype == torch.int64

    def test_dataset_with_augmentation(self, temp_dataset_dir):
        ADE20KDataset = import_ade20k()
        dataset = ADE20KDataset(
            root_dir=temp_dataset_dir,
            split="training",
            image_size=(128, 128),
            augment=True,
            augmentation_level="basic",
        )

        item1 = dataset[0]
        item2 = dataset[0]
        assert item1["mask"].shape == item2["mask"].shape

    def test_subset_fraction(self, temp_dataset_dir):
        ADE20KDataset = import_ade20k()
        dataset = ADE20KDataset(
            root_dir=temp_dataset_dir,
            split="training",
            subset_fraction=0.5,  # 50% данных
        )
        assert len(dataset) <= 2

    def test_ignore_index_in_mask(self, temp_dataset_dir):
        ADE20KDataset = import_ade20k()
        dataset = ADE20KDataset(root_dir=temp_dataset_dir, ignore_index=255)

        item = dataset[0]
        mask = item["mask"]
        valid_values = mask[mask != 255]
        assert valid_values.min() >= 0
        assert valid_values.max() <= 149

    def test_validation_split(self, temp_dataset_dir):
        ADE20KDataset = import_ade20k()
        # Создаём валидационную директорию
        base_dir = Path(temp_dataset_dir) / "ADEChallengeData2016"
        val_images = base_dir / "images" / "validation"
        val_masks = base_dir / "annotations" / "validation"
        val_images.mkdir(parents=True)
        val_masks.mkdir(parents=True)

        # Создаём тестовые файлы для валидации
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        img.save(val_images / "val_0.jpg")
        mask = Image.fromarray(np.random.randint(0, 150, (256, 256), dtype=np.uint8))
        mask.save(val_masks / "val_0.png")

        dataset = ADE20KDataset(root_dir=temp_dataset_dir, split="validation")
        assert len(dataset) == 1
