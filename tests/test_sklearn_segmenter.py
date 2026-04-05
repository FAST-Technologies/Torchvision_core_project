# tests/test_sklearn_segmenter.py

# Импорт основных библиотек
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from segmenters.SklearnSegmenter import SklearnSegmenter


class TestSklearnSegmenter:
    @pytest.fixture
    def segmenter(self):
        return SklearnSegmenter("global_thresholding", threshold=0.5)

    def test_import(self) -> None:
        from segmenters.SklearnSegmenter import SklearnSegmenter

        assert SklearnSegmenter is not None

    def test_initialization(self) -> None:
        seg = SklearnSegmenter("adaptive_thresholding", block_size=11)
        assert seg.method == "adaptive_thresholding"
        assert seg.params["block_size"] == 11

    def test_segment_rgb(self, rgb_image) -> None:
        seg = SklearnSegmenter("global_thresholding", threshold=0.5)
        mask = seg.segment(rgb_image)
        assert mask.shape == rgb_image.shape[:2]
        assert mask.dtype == np.uint8

    def test_segment_grayscale_normalized(self) -> None:
        """Тест с нормализованным [0,1] изображением"""
        gray = np.random.rand(128, 128).astype(np.float32)
        seg = SklearnSegmenter("global_thresholding", threshold=0.5)
        mask = seg.segment(gray)
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
    def test_methods_basic(self, rgb_image, method, params) -> None:
        seg = SklearnSegmenter(method, **params)
        mask = seg.segment(rgb_image)
        assert mask is not None
        assert mask.shape == rgb_image.shape[:2]
        assert mask.dtype == np.uint8

    def test_sauvola_with_r_parameter(self, gray_image) -> None:
        """Sauvola с параметром r"""
        seg = SklearnSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128)
        mask = seg.segment(gray_image)
        assert mask.shape == gray_image.shape
        assert mask.dtype == np.uint8

    def test_canny_with_quantiles(self, rgb_image) -> None:
        """Canny с use_quantiles=False"""
        seg = SklearnSegmenter("canny_edge", low=0.1, high=0.3, use_quantiles=False)
        mask = seg.segment(rgb_image)
        assert mask.shape == rgb_image.shape[:2]

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError):
            seg = SklearnSegmenter("unknown_method")
            seg.segment(np.zeros((100, 100)))

    def test_preprocess_image_as_gray(self, rgb_image) -> None:
        """Проверка конвертации в grayscale"""
        seg = SklearnSegmenter("global_thresholding")
        gray = seg.preprocess_image(rgb_image, as_gray=True)
        assert gray.ndim == 2 or gray.shape[1] == 1
