"""Модульные тесты для сегментера на базе scikit-learn/scikit-image.

Тестирует:
- Корректность инициализации и передачи параметров.
- Обработку RGB и нормализованных grayscale-изображений.
- Параметризованные запуски базовых методов (thresholding, edge detection).
- Конвертацию в оттенки серого через `preprocess_image(as_gray=True)`.
- Поведение при попытке использовать неизвестный метод.

Все тесты проверяют целостность выходной маски: форма `(H, W)`,
тип `uint8`, отсутствие `None` или артефактов.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from segmenters.SklearnSegmenter import SklearnSegmenter


# ──────────────────────────────────────────────────────────────────────
class TestSklearnSegmenter:
    """Класс тестов для SklearnSegmenter."""

    @pytest.fixture
    def segmenter(self) -> SklearnSegmenter:
        """Создаёт базовый экземпляр сегментера для изолированных тестов.

        Returns:
            SklearnSegmenter: Сегментер с методом `global_thresholding`
            и порогом 0.5.
        """
        return SklearnSegmenter("global_thresholding", threshold=0.5)

    def test_import(self) -> None:
        """Проверяет успешный импорт класса SklearnSegmenter."""
        from segmenters.SklearnSegmenter import SklearnSegmenter

        assert SklearnSegmenter is not None

    def test_initialization(self) -> None:
        """Валидирует корректное сохранение имени метода и параметров при инициализации."""
        seg = SklearnSegmenter("adaptive_thresholding", block_size=11)
        assert seg.method == "adaptive_thresholding"
        assert seg.params["block_size"] == 11

    def test_segment_rgb(self, rgb_image: np.ndarray) -> None:
        """Проверяет сегментацию стандартного RGB-изображения."""
        seg = SklearnSegmenter("global_thresholding", threshold=0.5)
        mask: np.ndarray = seg.segment(rgb_image)
        assert mask.shape == rgb_image.shape[:2]
        assert mask.dtype == np.uint8

    def test_segment_grayscale_normalized(self) -> None:
        """Тестирует обработку нормализованного grayscale-входа [0, 1].

        Убеждается, что метод корректно работает с float32-входом
        без предварительного масштабирования к [0, 255].
        """
        gray: np.ndarray = np.random.rand(128, 128).astype(np.float32)
        seg = SklearnSegmenter("global_thresholding", threshold=0.5)
        mask: np.ndarray = seg.segment(gray)
        assert mask.shape == gray.shape
        assert mask.dtype == np.uint8

    @pytest.mark.parametrize(
        "method,params",
        [
            ("global_thresholding", {"threshold": 0.5}),
            ("otsu_thresholding", {}),
            ("adaptive_thresholding", {"block_size": 11, "C": 2}),
            ("threshold_niblack", {"window_size": 15, "k": -0.2}),
            ("threshold_sauvola", {"window_size": 15, "k": 0.5, "r": 128}),
            ("sobel_edge", {"threshold": 0.1}),
            ("canny_edge", {"low": 0.1, "high": 0.3, "sigma": 1.0}),
        ],
    )
    def test_methods_basic(self, rgb_image: np.ndarray, method: str, params: Dict[str, Any]) -> None:
        """Параметризованный тест базовых методов сегментации.

        Проверяет, что каждый метод из списка успешно инициализируется,
        принимает указанные параметры и возвращает валидную бинарную маску
        правильной формы и типа данных.
        """
        seg = SklearnSegmenter(method, **params)
        mask: np.ndarray = seg.segment(rgb_image)
        assert mask is not None
        assert mask.shape == rgb_image.shape[:2]
        assert mask.dtype == np.uint8

    def test_sauvola_with_r_parameter(self, gray_image: np.ndarray) -> None:
        """Проверяет корректную работу Sauvola с параметром динамического диапазона `r`."""
        seg = SklearnSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128)
        mask: np.ndarray = seg.segment(gray_image)
        assert mask.shape == gray_image.shape
        assert mask.dtype == np.uint8

    def test_canny_with_quantiles(self, rgb_image: np.ndarray) -> None:
        """Валидирует работу Canny с отключённым режимом квантилей (`use_quantiles=False`).

        Ожидается корректная интерпретация порогов `low`/`high` как
        абсолютных значений вместо процентилей.
        """
        seg = SklearnSegmenter("canny_edge", low=0.1, high=0.3, use_quantiles=False)
        mask: np.ndarray = seg.segment(rgb_image)
        assert mask.shape == rgb_image.shape[:2]

    def test_invalid_method_raises(self) -> None:
        """Проверяет выброс ValueError при инициализации несуществующего метода."""
        with pytest.raises(ValueError):
            seg = SklearnSegmenter("unknown_method")
            seg.segment(np.zeros((100, 100)))

    def test_preprocess_image_as_gray(self, rgb_image: np.ndarray) -> None:
        """Проверяет корректность конвертации RGB → Grayscale через preprocess_image."""
        seg = SklearnSegmenter("global_thresholding")
        gray: np.ndarray = seg.preprocess_image(rgb_image, as_gray=True)
        assert gray.ndim == 2 or gray.shape[1] == 1
