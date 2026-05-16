# tests/test_torch_segmenter.py

"""Модульные тесты для базового PyTorch-сегментера (TorchSegmenter v1).

Тестирует:
- Корректность импорта, инициализации и сохранения параметров.
- Сегментацию RGB и Grayscale изображений.
- Обработку неизвестных методов и валидацию входных данных.
- Параметризованные запуски основных методов.
- Доступность CUDA и переключение устройств.
- Гарантированный возврат numpy-массива при любом типе входа (включая путь к файлу).

Фокус на стабильности API и обратной совместимости с первой версией
реализации без оптимизаций (torch.compile, mixed precision, кэширование).
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
from segmenters.TorchSegmenter import TorchSegmenter
import torch


# ──────────────────────────────────────────────────────────────────────
class TestTorchSegmenter:
    """Класс тестов для базовой реализации TorchSegmenter."""

    @pytest.fixture
    def segmenter(self) -> TorchSegmenter:
        """Создаёт экземпляр сегментера по умолчанию для тестов."""
        return TorchSegmenter("global_thresholding", threshold=0.5)

    @pytest.fixture
    def test_image(self) -> np.ndarray:
        """Генерирует стандартное тестовое изображение 256×256×3."""
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    @pytest.fixture
    def test_gray_image(self) -> np.ndarray:
        """Генерирует одноканальное тестовое изображение 256×256."""
        return np.random.randint(0, 255, (256, 256), dtype=np.uint8)

    def test_import(self) -> None:
        """Проверяет успешный импорт класса TorchSegmenter."""
        from segmenters.TorchSegmenter import TorchSegmenter

        assert TorchSegmenter is not None

    def test_initialization(self) -> None:
        """Валидирует сохранение имени метода и параметров после инициализации."""
        segmenter = TorchSegmenter("global_thresholding", threshold=0.5)
        assert segmenter.method == "global_thresholding"
        assert segmenter.params["threshold"] == 0.5

    def test_segment_rgb(self, test_image: np.ndarray) -> None:
        """Проверяет сегментацию RGB-изображения: валидность формы, типа и диапазона."""
        segmenter = TorchSegmenter("global_thresholding", threshold=0.5)
        mask: np.ndarray = segmenter.segment(test_image)

        assert mask is not None
        assert mask.shape == test_image.shape[:2]
        assert mask.dtype == np.uint8
        assert mask.min() >= 0
        assert mask.max() <= 255

    def test_segment_grayscale(self, test_gray_image: np.ndarray) -> None:
        """Проверяет корректную обработку одноканального входа."""
        segmenter = TorchSegmenter("otsu_thresholding")
        mask: np.ndarray = segmenter.segment(test_gray_image)

        assert mask is not None
        assert mask.shape == test_gray_image.shape
        assert mask.dtype == np.uint8

    def test_unknown_method(self, test_image: np.ndarray) -> None:
        """Проверяет выброс ValueError при попытке использовать неизвестный метод."""
        with pytest.raises(ValueError):
            segmenter = TorchSegmenter("unknown_method")
            segmenter.segment(test_image)

    @pytest.mark.parametrize(
        "method,params",
        [
            ("global_thresholding", {"threshold": 0.5}),
            ("otsu_thresholding", {}),
            ("adaptive_thresholding", {"block_size": 11, "C": 2}),
            ("sobel_edge", {"threshold": 0.1}),
            ("canny_edge", {"low": 0.1, "high": 0.3}),
        ],
    )
    def test_methods(self, test_image: np.ndarray, method: str, params: Dict[str, Any]) -> None:
        """Параметризованный тест совместимости базовых методов.

        Проверяет, что каждый метод успешно запускается с заданными
        параметрами и возвращает маску ожидаемой формы и типа.
        """
        segmenter = TorchSegmenter(method, **params)
        mask: np.ndarray = segmenter.segment(test_image)

        assert mask is not None
        assert mask.shape == test_image.shape[:2]
        assert mask.dtype == np.uint8

    @pytest.mark.gpu
    def test_cuda_availability(self) -> None:
        """Проверяет корректное переключение на CUDA-устройство.

        Skip-маркер позволяет пропускать тест на CPU-машинах.
        Ожидается, что `str(segmenter.device) == "cuda"` при наличии GPU.
        """
        if torch.cuda.is_available():
            segmenter = TorchSegmenter("global_thresholding", device="cuda")
            assert str(segmenter.device) == "cuda"
        else:
            pytest.skip("CUDA not available")

    def test_segment_returns_numpy(self, segmenter: TorchSegmenter) -> None:
        """Проверяет, что метод всегда возвращает numpy-массив даже при входе-строке.

        Тестирует поддержку путей к файлам как входных данных.
        Ожидается корректная загрузка, обработка и возврат `np.ndarray`
        с типом `uint8`.
        """
        result: np.ndarray = segmenter.segment("test_images/animals.jpg")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    def test_invalid_method_raises(self) -> None:
        """Проверяет валидацию имени метода на этапе инициализации."""
        with pytest.raises(ValueError):
            TorchSegmenter("invalid_method")

    # def test_threshold_range(self):
    #     with pytest.raises(ValueError):
    #         TorchSegmenter("global_thresholding", threshold=1.5)
