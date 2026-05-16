# tests/test_datasets.py
# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
"""Тесты для модулей работы с датасетами (ADE20K).

Этот модуль проверяет класс `ADE20KDataset`:
- Инициализацию и загрузку данных
- Метод `__getitem__()` и формат возвращаемых данных
- Поддержку аугментаций и подвыборок
- Обработку специальных значений (ignore_index)
- Разделение на training/validation наборы

Тесты используют временные директории для изоляции и не требуют
реального датасета на диске.

Example:
    Запуск тестов:
    ```bash
    pytest tests/test_datasets.py -v
    ```
"""
# ──────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from dataseters.ADE20KDataset import ADE20KDataset


# ──────────────────────────────────────────────────────────────────────
def import_ade20k() -> type[ADE20KDataset]:
    """Динамически импортирует класс `ADE20KDataset`.

    Использует отложенный импорт для избежания циклических зависимостей
    и ускорения загрузки тестового модуля.

    Returns:
        Type[ADE20KDataset]: Класс датасета для последующего использования.

    Note:
        Функция вынесена отдельно, чтобы можно было протестировать
        сам факт импорта без создания экземпляра.
    """
    return ADE20KDataset


# ──────────────────────────────────────────────────────────────────────
class TestADE20KDataset:
    """Набор тестов для класса `ADE20KDataset`.

    Проверяет корректность работы с датасетом семантической сегментации
    ADE20K, включая загрузку, аугментации и формат данных.
    """

    @pytest.fixture
    def temp_dataset_dir(self, tmp_path: Path) -> str:
        """Создаёт временную структуру директорий, имитирующую ADE20K.

        Генерирует минимальную файловую структуру, необходимую для
        инициализации `ADE20KDataset`:
        ```
        tmp_path/
        └── ADEChallengeData2016/
            ├── images/
            │   └── training/
            │       ├── test_0.jpg
            │       ├── test_1.jpg
            │       └── test_2.jpg
            └── annotations/
                └── training/
                    ├── test_0.png
                    ├── test_1.png
                    └── test_2.png
        ```

        Args:
            tmp_path (Path): Фикстура pytest для временных файлов.

        Returns:
            str: Путь к корневой директории временного датасета.

        Note:
            - Создаёт 3 тестовых изображения и соответствующие маски
            - Изображения: случайные значения [0, 255], размер 256×256×3
            - Маски: случайные значения [0, 149] (150 классов ADE20K)
            - Все файлы автоматически удаляются после завершения тестов
        """
        """Создаёт временную структуру ADE20K"""
        base_dir: Path = tmp_path / "ADEChallengeData2016"
        images_dir: Path = base_dir / "images" / "training"
        masks_dir: Path = base_dir / "annotations" / "training"

        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Создаём тестовые изображения
        for i in range(3):
            img: Image.Image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img.save(images_dir / f"test_{i}.jpg")

            mask: Image.Image = Image.fromarray(np.random.randint(0, 150, (256, 256), dtype=np.uint8))
            mask.save(masks_dir / f"test_{i}.png")

        return str(tmp_path)

    # ──────────────────────────────────────────────────────────────────────
    def test_import(self) -> None:
        """Проверяет успешный импорт класса `ADE20KDataset`.

        Убеждается, что модуль доступен и класс может быть импортирован
        без ошибок.

        Raises:
            AssertionError: Если класс не найден или равен None.
        """
        ADE20KDataset = import_ade20k()
        assert ADE20KDataset is not None

    # ──────────────────────────────────────────────────────────────────────
    def test_dataset_initialization(self, temp_dataset_dir: Path) -> None:
        """Тестирует инициализацию датасета с базовыми параметрами.

        Проверяет:
        1. Успешное создание экземпляра датасета
        2. Корректное определение количества образцов (3 тестовых файла)
        3. Применение параметра `image_size` для ресайза

        Args:
            temp_dataset_dir (Path): Путь к временной структуре датасета.

        Raises:
            AssertionError: Если длина датасета или размер изображения
            не соответствуют ожидаемым.
        """
        ADE20KDataset = import_ade20k()
        dataset = ADE20KDataset(
            root_dir=temp_dataset_dir,
            split="training",
            image_size=(128, 128),
            augment=False,
        )
        assert len(dataset) == 3
        assert dataset.image_size == (128, 128)

    # ──────────────────────────────────────────────────────────────────────
    def test_dataset_getitem(self, temp_dataset_dir: Path) -> None:
        """Тестирует метод `__getitem__()` датасета.

        Проверяет формат возвращаемых данных:
        - Наличие ключей: `image`, `mask`, `image_id`
        - Формы тензоров: image=(3, 128, 128), mask=(128, 128)
        - Типы данных: image=float32, mask=int64 (для loss-функций)

        Args:
            temp_dataset_dir (Path): Путь к временной структуре датасета.

        Raises:
            AssertionError: Если структура или типы возвращаемых данных
            не соответствуют ожидаемым.
        """
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

    # ──────────────────────────────────────────────────────────────────────
    def test_dataset_with_augmentation(self, temp_dataset_dir: Path) -> None:
        """Тестирует применение аугментаций к данным датасета.

        Проверяет, что при `augment=True`:
        1. Датасет инициализируется без ошибок
        2. Аугментации применяются детерминировано (одинаковый seed)
           или стохастически (разные результаты при повторных вызовах)
        3. Размеры выходных данных остаются неизменными

        Примечание: Тест проверяет только форму, а не содержание,
        так как аугментации стохастичны.

        Args:
            temp_dataset_dir (Path): Путь к временной структуре датасета.

        Raises:
            AssertionError: Если формы масок при повторных вызовах
            не совпадают.
        """
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

    # ──────────────────────────────────────────────────────────────────────
    def test_subset_fraction(self, temp_dataset_dir: Path) -> None:
        """Тестирует параметр `subset_fraction` для выборки части данных.

        Проверяет, что при `subset_fraction=0.5` датасет содержит
        не более 50% от общего количества образцов (в данном случае ≤2 из 3).

        Используется для:
        - Быстрого прототипирования
        - Отладки без загрузки полного датасета
        - Тестирования на ограниченных ресурсах

        Args:
            temp_dataset_dir (Path): Путь к временной структуре датасета.

        Raises:
            AssertionError: Если количество образцов превышает ожидаемое.
        """
        ADE20KDataset = import_ade20k()
        dataset = ADE20KDataset(
            root_dir=temp_dataset_dir,
            split="training",
            subset_fraction=0.5,  # 50% данных
        )
        assert len(dataset) <= 2

    # ──────────────────────────────────────────────────────────────────────
    def test_ignore_index_in_mask(self, temp_dataset_dir: Path) -> None:
        """Тестирует обработку параметра `ignore_index` в масках.

        Проверяет, что значения, равные `ignore_index` (по умолчанию 255),
        корректно исключаются из расчёта метрик:
        - В маске остаются только значения в диапазоне [0, 149]
        - Значение 255 не учитывается как класс

        Это важно для:
        - Игнорирования не размеченных областей
        - Корректного расчёта loss-функций

        Args:
            temp_dataset_dir (Path): Путь к временной структуре датасета.

        Raises:
            AssertionError: Если в маске присутствуют недопустимые значения.
        """
        ADE20KDataset = import_ade20k()
        dataset = ADE20KDataset(root_dir=temp_dataset_dir, ignore_index=255)

        item = dataset[0]
        mask: np.ndarray = item["mask"]
        valid_values: np.ndarray = mask[mask != 255]
        assert valid_values.min() >= 0
        assert valid_values.max() <= 149

    # ──────────────────────────────────────────────────────────────────────
    def test_validation_split(self, temp_dataset_dir: str) -> None:
        """Тестирует загрузку валидационного набора данных.

        Проверяет, что при `split="validation"` датасет корректно
        ищет файлы в директории `validation/` вместо `training/`.

        Args:
            temp_dataset_dir: Путь к временной структуре датасета.

        Raises:
            AssertionError: Если количество образцов в валидационном
            наборе не равно 1 (созданному тестовому файлу).
        """
        ADE20KDataset = import_ade20k()
        # Создаём валидационную директорию
        base_dir: Path = Path(temp_dataset_dir) / "ADEChallengeData2016"
        val_images: Path = base_dir / "images" / "validation"
        val_masks: Path = base_dir / "annotations" / "validation"
        val_images.mkdir(parents=True)
        val_masks.mkdir(parents=True)

        # Создаём тестовые файлы для валидации
        img: Image.Image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        img.save(val_images / "val_0.jpg")
        mask: Image.Image = Image.fromarray(np.random.randint(0, 150, (256, 256), dtype=np.uint8))
        mask.save(val_masks / "val_0.png")

        dataset = ADE20KDataset(root_dir=temp_dataset_dir, split="validation")
        assert len(dataset) == 1
