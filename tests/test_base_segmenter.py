# tests/test_base_segmenter.py
# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
"""Тесты для базового класса сегментации `BaseSegmenter`.

Этот модуль проверяет:
1. Корректность импорта и инициализации базового класса
2. Работу метода `preprocess_image()` с разными типами входа
   (путь к файлу, PIL.Image, numpy.ndarray)
3. Поведение метода `segment_with_mask()` и формат возвращаемых данных

Тесты используют фикстуры из `conftest.py` для создания тестовых данных.

Example:
    Запуск тестов:
    ```bash
    pytest tests/test_base_segmenter.py -v
    ```
"""
# ──────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import Tuple, Any
from PIL import Image
from segmenters.BaseSegmenter import (
    BaseSegmenter,
    NumpyImage,
    ImageInput,
)


# ──────────────────────────────────────────────────────────────────────
class DummySegmenter(BaseSegmenter):
    """Реализация-заглушка `BaseSegmenter` для тестирования.

    Предоставляет минимальную реализацию абстрактных методов:
    - `segment()`: Возвращает нулевую маску того же размера, что и вход
    - `segment_with_mask()`: Возвращает кортеж (визуализация, маска)

    Используется для тестирования общей логики базового класса
    без зависимости от конкретных алгоритмов сегментации.

    Attributes:
        None

    Note:
        Этот класс не предназначен для реального использования —
        только для изолированного тестирования инфраструктуры.
    """

    def segment(self, image: ImageInput, **kwargs: Any) -> np.ndarray:
        """Возвращает пустую бинарную маску того же размера, что и вход.

        Заглушка метода сегментации для тестов. Не выполняет
        никакой реальной обработки — просто создаёт нулевой массив.

        Args:
            image (ImageInput): Входное изображение (путь, PIL, или numpy).
            **kwargs: Дополнительные параметры (игнорируются).

        Returns:
            np.ndarray: Бинарная маска формы (H, W), dtype=uint8,
            заполненная нулями.
        """
        # Возвращаем пустую маску того же размера что и вход
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = 256, 256
        return np.zeros((h, w), dtype=np.uint8)

    # ──────────────────────────────────────────────────────────────────────
    def segment_with_mask(self, image: ImageInput, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает кортеж (визуализация, маска) для тестов.

        Заглушка метода с визуализацией. Возвращает:
        1. Копию входного изображения (или нулевое, если вход не массив)
        2. Пустую маску от метода `segment()`

        Args:
            image (ImageInput): Входное изображение.
            **kwargs: Дополнительные параметры (игнорируются).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - result: Визуализация формы (H, W, 3), dtype=uint8
                - mask: Бинарная маска формы (H, W), dtype=uint8
        """
        # Возвращаем кортеж (результат, маска)
        mask: np.ndarray = self.segment(image, **kwargs)
        # Для визуализации просто копируем изображение
        if isinstance(image, np.ndarray):
            result: np.ndarray = image.copy() if len(image.shape) == 3 else np.stack([image] * 3, axis=-1)
        else:
            result = np.zeros((256, 256, 3), dtype=np.uint8)
        return result, mask


# ──────────────────────────────────────────────────────────────────────
class TestBaseSegmenter:
    """Набор тестов для проверки базовой функциональности `BaseSegmenter`.

    Проверяет:
    - Импорт и доступность класса
    - Обработку различных форматов входных данных
    - Корректность возвращаемых типов и размеров
    """

    def test_import(self) -> None:
        """Проверяет успешный импорт класса `BaseSegmenter`.

        Убеждается, что модуль `segmenters.BaseSegmenter` доступен
        и класс `BaseSegmenter` может быть импортирован без ошибок.

        Raises:
            AssertionError: Если класс не найден или равен None.
        """
        from segmenters.BaseSegmenter import BaseSegmenter

        assert BaseSegmenter is not None

    # def test_abstract_methods(self) -> None:
    #     """Базовый класс не должен инстанцироваться напрямую"""
    #     with pytest.raises(TypeError):
    #         BaseSegmenter()

    # ──────────────────────────────────────────────────────────────────────
    def test_preprocess_image_from_path(self, temp_image_file: np.ndarray) -> None:
        """Тестирует предобработку изображения из пути к файлу.

        Проверяет, что метод `preprocess_image()` корректно:
        1. Загружает изображение по указанному пути
        2. Конвертирует в формат, пригодный для обработки
        3. Возвращает непустой результат

        Args:
            temp_image_file (str): Путь к временному файлу изображения
            (предоставляется фикстурой).

        Raises:
            AssertionError: Если результат предобработки равен None.
        """
        """Тест предобработки из файла"""
        seg = DummySegmenter()
        result: NumpyImage = seg.preprocess_image(temp_image_file)
        assert result is not None

    # ──────────────────────────────────────────────────────────────────────
    def test_preprocess_image_from_pil(self) -> None:
        """Тестирует предобработку изображения из объекта PIL.Image.

        Проверяет, что метод `preprocess_image()` принимает объект
        `PIL.Image` и корректно преобразует его во внутренний формат.

        Raises:
            AssertionError: Если результат предобработки равен None.
        """
        """Тест предобработки из PIL Image"""
        img: Image.Image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        seg = DummySegmenter()
        result: NumpyImage = seg.preprocess_image(img)
        assert result is not None

    # ──────────────────────────────────────────────────────────────────────
    def test_preprocess_image_from_numpy(self) -> None:
        """Тестирует предобработку изображения из numpy массива.

        Проверяет, что метод `preprocess_image()` принимает `np.ndarray`
        и возвращает результат без ошибок.

        Raises:
            AssertionError: Если результат предобработки равен None.
        """
        """Тест предобработки из numpy array"""
        img: np.ndarray = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        seg = DummySegmenter()
        result: NumpyImage = seg.preprocess_image(img)
        assert result is not None

    # ──────────────────────────────────────────────────────────────────────
    def test_segment_with_mask_base(self) -> None:
        """Тестирует базовое поведение метода `segment_with_mask()`.

        Проверяет, что метод возвращает кортеж из двух массивов:
        1. Визуализация: форма совпадает с входным изображением, dtype=uint8
        2. Маска: форма (H, W), dtype=uint8

        Также проверяет типы данных возвращаемых значений.

        Raises:
            AssertionError: Если формы или типы данных не соответствуют
            ожидаемым.
        """
        """Тест базового segment_with_mask"""
        seg = DummySegmenter()
        img: np.ndarray = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result: np.ndarray
        mask: np.ndarray
        result, mask = seg.segment_with_mask(img)
        assert result.shape == img.shape
        assert mask.shape == img.shape[:2]
        assert result.dtype == np.uint8
        assert mask.dtype == np.uint8
