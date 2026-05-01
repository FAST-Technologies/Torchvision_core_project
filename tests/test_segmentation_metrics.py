# tests/test_segmentation_metrics.py

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
from typing import Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from metrics.SegmentationMetrics import SegmentationMetrics, MetricsDict


class TestSegmentationMetrics:
    @pytest.fixture
    def perfect_prediction(self) -> Tuple[np.ndarray, np.ndarray]:
        """Идеальное предсказание"""
        gt: np.ndarray = np.zeros((100, 100), dtype=np.uint8)
        gt[25:75, 25:75] = 255
        return gt, gt.copy()

    @pytest.fixture
    def random_prediction(self) -> Tuple[np.ndarray, np.ndarray]:
        """Случайное предсказание"""
        gt: np.ndarray = np.zeros((100, 100), dtype=np.uint8)
        gt[25:75, 25:75] = 255
        pred: np.ndarray = np.random.randint(0, 2, (100, 100)) * 255
        return gt, pred.astype(np.uint8)

    @pytest.fixture
    def empty_prediction(self) -> Tuple[np.ndarray, np.ndarray]:
        """Пустое предсказание"""
        gt: np.ndarray = np.zeros((100, 100), dtype=np.uint8)
        gt[25:75, 25:75] = 255
        pred: np.ndarray = np.zeros((100, 100), dtype=np.uint8)
        return gt, pred

    def test_import(self) -> None:
        from metrics.SegmentationMetrics import SegmentationMetrics

        assert SegmentationMetrics is not None

    def test_calculate_all_metrics_perfect(self, perfect_prediction) -> None:
        gt, pred = perfect_prediction
        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt)

        assert metrics["iou"] == pytest.approx(1.0, rel=1e-5)
        assert metrics["dice"] == pytest.approx(1.0, rel=1e-5)
        assert metrics["precision"] == pytest.approx(1.0, rel=1e-5)
        assert metrics["recall"] == pytest.approx(1.0, rel=1e-5)
        assert metrics["f1_score"] == pytest.approx(1.0, rel=1e-5)
        assert metrics["pixel_accuracy"] == pytest.approx(1.0, rel=1e-5)

    def test_calculate_all_metrics_random(self, random_prediction) -> None:
        gt, pred = random_prediction
        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt)

        assert 0 <= metrics["iou"] <= 1
        assert 0 <= metrics["dice"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert 0 <= metrics["pixel_accuracy"] <= 1

    def test_empty_prediction_metrics(self, empty_prediction) -> None:
        gt, pred = empty_prediction
        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt)

        assert metrics["iou"] == pytest.approx(0, abs=1e-6)
        assert metrics["dice"] == pytest.approx(0, abs=1e-6)
        assert metrics["precision"] == pytest.approx(0, abs=1e-6)
        assert metrics["recall"] == pytest.approx(0, abs=1e-6)

    def test_metrics_with_threshold(self) -> None:
        """Тест с порогом для бинаризации"""
        gt: np.ndarray = np.array([[0, 1], [1, 0]], dtype=np.float32)
        pred_prob: np.ndarray = np.array([[0.3, 0.6], [0.7, 0.4]], dtype=np.float32)

        # С порогом 0.4
        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(
            pred_prob, gt, threshold=0.4
        )
        assert "iou" in metrics

        # С порогом 0.6
        metrics_high: MetricsDict = SegmentationMetrics.calculate_all_metrics(
            pred_prob, gt, threshold=0.6
        )
        assert metrics["iou"] != metrics_high["iou"]

    def test_hausdorff_distance_included(self) -> None:
        """Проверка включения Hausdorff distance"""
        gt: np.ndarray = np.zeros((50, 50), dtype=np.uint8)
        gt[10:40, 10:40] = 255
        pred: np.ndarray = gt.copy()
        pred[15:35, 15:35] = 0  # Небольшое смещение

        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(
            pred, gt, include_hausdorff=True
        )
        assert "hausdorff_distance" in metrics
        assert metrics["hausdorff_distance"] >= 0

        # Без включения
        metrics_no_h: MetricsDict = SegmentationMetrics.calculate_all_metrics(
            pred, gt, include_hausdorff=False
        )
        assert "hausdorff_distance" not in metrics_no_h

    def test_area_metrics(self) -> None:
        """Проверка метрик площади"""
        gt: np.ndarray = np.zeros((100, 100), dtype=np.uint8)
        gt[20:80, 20:80] = 255  # 60x60 = 3600 пикселей
        pred: np.ndarray = np.zeros((100, 100), dtype=np.uint8)
        pred[10:90, 10:90] = 255  # 80x80 = 6400 пикселей

        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt)

        assert metrics["predicted_area"] == 6400
        assert metrics["ground_truth_area"] == 3600
        assert metrics["area_difference"] == 2800
        # min/max (коэффициент перекрытия)
        assert metrics["area_ratio"] == pytest.approx(3600 / 6400, rel=1e-3)  # 0.5625

    def test_mae_metric(self) -> None:
        """Проверка MAE (Mean Absolute Error)"""
        # Используем нормализованные данные [0, 1]
        gt: np.ndarray = np.array([[0, 1], [1, 0]], dtype=np.float32)
        pred: np.ndarray = np.array(
            [[0, 0.8], [1, 0.2]], dtype=np.float32
        )  # [0, 200/255, 1, 50/255]

        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt)
        # MAE = (|0-0| + |0.8-1| + |1-1| + |0.2-0|) / 4 = (0+0.2+0+0.2)/4 = 0.1
        assert metrics["mae"] == pytest.approx(0.1, rel=1e-2)

    def test_invalid_input_shapes(self) -> None:
        """Обработка несовпадающих размеров"""
        gt: np.ndarray = np.zeros((50, 50), dtype=np.uint8)
        pred: np.ndarray = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises((ValueError, AssertionError)):
            SegmentationMetrics.calculate_all_metrics(pred, gt)

    def test_binary_input_handling(self) -> None:
        """Обработка бинарных входов [0,1]"""
        gt: np.ndarray = np.array([[0, 1], [1, 0]], dtype=np.float32)
        pred: np.ndarray = np.array([[0, 1], [0, 0]], dtype=np.float32)

        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt)
        assert "iou" in metrics
