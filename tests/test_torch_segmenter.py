# tests/test_torch_segmenter.py
import pytest
import numpy as np
import sys
from pathlib import Path

from segmenters.TorchSegmenter import TorchSegmenter

sys.path.insert(0, str(Path(__file__).parent.parent))

class TestTorchSegmenter:
    @pytest.fixture
    def segmenter(self):
        return TorchSegmenter("global_thresholding", threshold=0.5)

    @pytest.fixture
    def test_image(self):
        """Тестовое изображение"""
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    @pytest.fixture
    def test_gray_image(self):
        """Тестовое grayscale изображение"""
        return np.random.randint(0, 255, (256, 256), dtype=np.uint8)

    def test_import(self):
        """Проверка импорта"""
        from segmenters.TorchSegmenter import TorchSegmenter

        assert TorchSegmenter is not None

    def test_initialization(self):
        """Проверка инициализации"""
        segmenter = TorchSegmenter("global_thresholding", threshold=0.5)
        assert segmenter.method == "global_thresholding"
        assert segmenter.params["threshold"] == 0.5

    def test_segment_rgb(self, test_image):
        """Сегментация RGB изображения"""
        segmenter = TorchSegmenter("global_thresholding", threshold=0.5)
        mask = segmenter.segment(test_image)

        assert mask is not None
        assert mask.shape == test_image.shape[:2]
        assert mask.dtype == np.uint8
        assert mask.min() >= 0
        assert mask.max() <= 255

    def test_segment_grayscale(self, test_gray_image):
        """Сегментация grayscale изображения"""
        segmenter = TorchSegmenter("otsu_thresholding")
        mask = segmenter.segment(test_gray_image)

        assert mask is not None
        assert mask.shape == test_gray_image.shape
        assert mask.dtype == np.uint8

    def test_unknown_method(self, test_image):
        """Проверка обработки неизвестного метода"""
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
    def test_methods(self, test_image, method, params):
        """Параметризованный тест методов"""
        segmenter = TorchSegmenter(method, **params)
        mask = segmenter.segment(test_image)

        assert mask is not None
        assert mask.shape == test_image.shape[:2]
        assert mask.dtype == np.uint8

    @pytest.mark.gpu
    def test_cuda_availability(self):
        """Проверка доступности CUDA (только для GPU тестов)"""
        import torch

        if torch.cuda.is_available():
            segmenter = TorchSegmenter("global_thresholding", device="cuda")
            assert str(segmenter.device) == "cuda"
        else:
            pytest.skip("CUDA not available")

    def test_segment_returns_numpy(self, segmenter):
        result = segmenter.segment("test_images/animals.jpg")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            TorchSegmenter("invalid_method")

    # def test_threshold_range(self):
    #     with pytest.raises(ValueError):
    #         TorchSegmenter("global_thresholding", threshold=1.5)
