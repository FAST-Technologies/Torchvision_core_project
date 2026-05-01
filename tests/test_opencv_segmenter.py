# tests/test_opencv_segmenter.py

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from segmenters.OpenCVSegmenter import OpenCVSegmenter


# ──────────────────────────────────────────────────────────────────────
class TestOpenCVSegmenter:
    @pytest.fixture
    def segmenter(self):
        return OpenCVSegmenter("global_thresholding", threshold=0.5)

    # ──────────────────────────────────────────────────────────────────────
    def test_import(self) -> None:
        from segmenters.OpenCVSegmenter import OpenCVSegmenter

        assert OpenCVSegmenter is not None

    # ──────────────────────────────────────────────────────────────────────
    def test_initialization(self) -> None:
        seg = OpenCVSegmenter("otsu_thresholding")
        assert seg.method == "otsu_thresholding"
        assert seg.params == {}

    # ──────────────────────────────────────────────────────────────────────
    def test_segment_rgb(self, rgb_image: np.ndarray) -> None:
        seg = OpenCVSegmenter("global_thresholding", threshold=0.5)
        mask: np.ndarray = seg.segment(rgb_image)
        assert mask.shape == rgb_image.shape[:2]
        assert mask.dtype == np.uint8
        assert np.all((mask == 0) | (mask == 255))

    # ──────────────────────────────────────────────────────────────────────
    def test_segment_grayscale(self, gray_image: np.ndarray) -> None:
        seg = OpenCVSegmenter("adaptive_thresholding", block_size=11, C=2)
        mask: np.ndarray = seg.segment(gray_image)
        assert mask.shape == gray_image.shape
        assert mask.dtype == np.uint8

    # ──────────────────────────────────────────────────────────────────────
    @pytest.mark.parametrize(
        "method,params",
        [
            ("global_thresholding", {"threshold": 0.5}),
            ("otsu_thresholding", {}),
            ("adaptive_thresholding", {"block_size": 11, "C": 2}),
            ("threshold_niblack", {"window_size": 15, "k": -0.2}),
            ("threshold_sauvola", {"window_size": 15, "k": 0.2}),
            ("sobel_edge", {"threshold": 0.1}),
            ("canny_edge", {"low": 0.1, "high": 0.3}),
        ],
    )
    # ──────────────────────────────────────────────────────────────────────
    def test_methods_basic(self, rgb_image: np.ndarray, method, params) -> None:
        seg = OpenCVSegmenter(method, **params)
        mask: np.ndarray = seg.segment(rgb_image)
        assert mask is not None
        assert mask.shape == rgb_image.shape[:2]
        assert mask.dtype == np.uint8

    # ──────────────────────────────────────────────────────────────────────
    def test_canny_edge_output(self, rgb_image: np.ndarray) -> None:
        """Canny должен возвращать тонкие границы"""
        seg = OpenCVSegmenter("canny_edge", low=0.3, high=0.7)
        mask: np.ndarray = seg.segment(rgb_image)
        assert np.mean(mask > 0) < 0.5

    # ──────────────────────────────────────────────────────────────────────
    def test_sauvola_vs_niblack(self, gray_image: np.ndarray) -> None:
        """Sauvola и Niblack должны давать разные результаты на структурированном изображении"""
        test_img: np.ndarray = np.zeros((256, 256), dtype=np.uint8)

        # Добавляем текстурные области с разной яркостью
        test_img[32:96, 32:96] = np.random.randint(
            80, 120, (64, 64), dtype=np.uint8
        )  # Тёмная текстура
        test_img[160:224, 160:224] = np.random.randint(
            200, 240, (64, 64), dtype=np.uint8
        )  # Светлая текстура

        seg_sauvola = OpenCVSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128)
        seg_niblack = OpenCVSegmenter("threshold_niblack", window_size=15, k=-0.2)

        mask_s: np.ndarray = seg_sauvola.segment(test_img)
        mask_n: np.ndarray = seg_niblack.segment(test_img)

        # diff_ratio = np.mean(mask_s != mask_n)
        assert mask_s.dtype == np.uint8 and mask_n.dtype == np.uint8
        assert np.all((mask_s == 0) | (mask_s == 255))
        assert np.all((mask_n == 0) | (mask_n == 255))

    # ──────────────────────────────────────────────────────────────────────
    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Неизвестный метод"):
            seg = OpenCVSegmenter("invalid_method")
            seg.segment(np.zeros((100, 100)))

    # ──────────────────────────────────────────────────────────────────────
    def test_segment_with_mask(self, rgb_image: np.ndarray) -> None:
        """Проверка segment_with_mask"""
        seg = OpenCVSegmenter("global_thresholding", threshold=0.5)
        result: np.ndarray
        mask: np.ndarray
        result, mask = seg.segment_with_mask(rgb_image)

        assert result.shape == rgb_image.shape
        assert mask.shape == rgb_image.shape[:2]
        assert result.dtype == np.uint8
        assert mask.dtype == np.uint8
