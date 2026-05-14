"""Модульные тесты для модуля расчёта метрик качества сегментации.

Тестирует корректность вычисления:
- IoU, Dice, Precision, Recall, F1-Score, Pixel Accuracy
- MAE (Mean Absolute Error) и Hausdorff Distance
- Метрик площади (predicted_area, ground_truth_area, area_ratio)
- Поведения при различных сценариях: идеальное совпадение, случайный шум,
  пустое предсказание, бинарные входы [0,1], несовпадающие размеры.

Все метрики сравниваются с аналитически рассчитанными значениями или
проверяются на вхождение в допустимые диапазоны [0, 1].
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from metrics.SegmentationMetrics import SegmentationMetrics, MetricsDict


# ──────────────────────────────────────────────────────────────────────
class TestSegmentationMetrics:
    """Класс тестов валидации математической корректности метрик."""

    @pytest.fixture
    def perfect_prediction(self) -> Tuple[np.ndarray, np.ndarray]:
        """Фикстура: идеальное предсказание (GT == Prediction).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (gt, pred) формы (100, 100),
            dtype=uint8. Квадрат 50×50 пикселей со значением 255 на фоне 0.
            Используется для проверки метрик на значениях, близких к 1.0.
        """
        gt: np.ndarray = np.zeros((100, 100), dtype=np.uint8)
        gt[25:75, 25:75] = 255
        return gt, gt.copy()

    @pytest.fixture
    def random_prediction(self) -> Tuple[np.ndarray, np.ndarray]:
        """Фикстура: случайное бинарное предсказание.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (gt, pred). GT содержит квадрат,
            pred заполнен случайными 0/255. Используется для проверки
            корректности работы с низким overlap и диапазона [0, 1].
        """
        gt: np.ndarray = np.zeros((100, 100), dtype=np.uint8)
        gt[25:75, 25:75] = 255
        pred: np.ndarray = np.random.randint(0, 2, (100, 100)) * 255
        return gt, pred.astype(np.uint8)

    @pytest.fixture
    def empty_prediction(self) -> Tuple[np.ndarray, np.ndarray]:
        """Фикстура: пустое предсказание (все пиксели фона).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (gt, pred). pred состоит только из 0.
            Проверяет корректную обработку деления на ноль и граничные условия
            (IoU/Dice ≈ 0, Precision/Recall ≈ 0).
        """
        gt: np.ndarray = np.zeros((100, 100), dtype=np.uint8)
        gt[25:75, 25:75] = 255
        pred: np.ndarray = np.zeros((100, 100), dtype=np.uint8)
        return gt, pred

    def test_import(self) -> None:
        """Проверяет успешный импорт класса SegmentationMetrics."""
        from metrics.SegmentationMetrics import SegmentationMetrics

        assert SegmentationMetrics is not None

    def test_calculate_all_metrics_perfect(self, perfect_prediction) -> None:
        """Валидирует метрики при идеальном совпадении GT и Prediction.

        Ожидается, что IoU, Dice, Precision, Recall, F1 и Pixel Accuracy
        будут равны 1.0 с точностью до 1e-5.
        """
        gt, pred = perfect_prediction
        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt)

        assert metrics["iou"] == pytest.approx(1.0, rel=1e-5)
        assert metrics["dice"] == pytest.approx(1.0, rel=1e-5)
        assert metrics["precision"] == pytest.approx(1.0, rel=1e-5)
        assert metrics["recall"] == pytest.approx(1.0, rel=1e-5)
        assert metrics["f1_score"] == pytest.approx(1.0, rel=1e-5)
        assert metrics["pixel_accuracy"] == pytest.approx(1.0, rel=1e-5)

    def test_calculate_all_metrics_random(self, random_prediction) -> None:
        """Проверяет корректность метрик при случайном предсказании.

        Убеждается, что все метрики лежат в допустимом диапазоне [0, 1],
        даже при отсутствии осмысленного overlap.
        """
        gt, pred = random_prediction
        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt)

        assert 0 <= metrics["iou"] <= 1
        assert 0 <= metrics["dice"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert 0 <= metrics["pixel_accuracy"] <= 1

    def test_empty_prediction_metrics(self, empty_prediction) -> None:
        """Валидирует поведение метрик при полностью пустом предсказании.

        Ожидается, что IoU, Dice, Precision и Recall будут равны 0
        (с погрешностью 1e-6 из-за float-арифметики).
        """
        gt, pred = empty_prediction
        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt)

        assert metrics["iou"] == pytest.approx(0, abs=1e-6)
        assert metrics["dice"] == pytest.approx(0, abs=1e-6)
        assert metrics["precision"] == pytest.approx(0, abs=1e-6)
        assert metrics["recall"] == pytest.approx(0, abs=1e-6)

    def test_metrics_with_threshold(self) -> None:
        """Проверяет влияние параметра `threshold` на бинаризацию вероятностных масок.

        Сравнивает метрики при двух разных порогах (0.4 и 0.6).
        Ожидается, что изменение порога изменит результат бинаризации и,
        как следствие, значение IoU.
        """
        gt: np.ndarray = np.array([[0, 1], [1, 0]], dtype=np.float32)
        pred_prob: np.ndarray = np.array([[0.3, 0.6], [0.7, 0.4]], dtype=np.float32)

        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred_prob, gt, threshold=0.4)
        assert "iou" in metrics

        metrics_high: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred_prob, gt, threshold=0.6)
        assert metrics["iou"] != metrics_high["iou"]

    def test_hausdorff_distance_included(self) -> None:
        """Проверяет опциональное выключение/включение расчёта Hausdorff Distance.

        Hausdorff Distance вычислительно затратен, поэтому включается
        только при `include_hausdorff=True`. Тест проверяет наличие ключа
        в словаре метрик и корректность неотрицательного значения.
        """
        gt: np.ndarray = np.zeros((50, 50), dtype=np.uint8)
        gt[10:40, 10:40] = 255
        pred: np.ndarray = gt.copy()
        pred[15:35, 15:35] = 0

        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt, include_hausdorff=True)
        assert "hausdorff_distance" in metrics
        assert metrics["hausdorff_distance"] >= 0

        metrics_no_h: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt, include_hausdorff=False)
        assert "hausdorff_distance" not in metrics_no_h

    def test_area_metrics(self) -> None:
        """Валидирует расчёт метрик, связанных с площадью сегментированных областей.

        Проверяет:
        - `predicted_area`: количество пикселей в предсказании.
        - `ground_truth_area`: количество пикселей в GT.
        - `area_difference`: абсолютная разница площадей.
        - `area_ratio`: коэффициент перекрытия (min/max).
        """
        gt: np.ndarray = np.zeros((100, 100), dtype=np.uint8)
        gt[20:80, 20:80] = 255  # 60x60 = 3600 пикселей
        pred: np.ndarray = np.zeros((100, 100), dtype=np.uint8)
        pred[10:90, 10:90] = 255  # 80x80 = 6400 пикселей

        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt)

        assert metrics["predicted_area"] == 6400
        assert metrics["ground_truth_area"] == 3600
        assert metrics["area_difference"] == 2800
        assert metrics["area_ratio"] == pytest.approx(3600 / 6400, rel=1e-3)

    def test_mae_metric(self) -> None:
        """Проверяет корректность расчёта MAE (Mean Absolute Error).

        Использует нормализованные входы [0, 1] для точного ручного расчёта:
        MAE = (|0-0| + |0.8-1| + |1-1| + |0.2-0|) / 4 = 0.1.
        """
        gt: np.ndarray = np.array([[0, 1], [1, 0]], dtype=np.float32)
        pred: np.ndarray = np.array([[0, 0.8], [1, 0.2]], dtype=np.float32)

        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt)
        assert metrics["mae"] == pytest.approx(0.1, rel=1e-2)

    def test_invalid_input_shapes(self) -> None:
        """Проверяет обработку исключений при несовпадающих размерах входов.

        Ожидается выброс ValueError или AssertionError, если `pred.shape != gt.shape`.
        """
        gt: np.ndarray = np.zeros((50, 50), dtype=np.uint8)
        pred: np.ndarray = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises((ValueError, AssertionError)):
            SegmentationMetrics.calculate_all_metrics(pred, gt)

    def test_binary_input_handling(self) -> None:
        """Проверяет корректную работу с бинарными float-входами [0, 1].

        Убеждается, что метрики рассчитываются без предварительного
        масштабирования к [0, 255] и возвращают валидные значения.
        """
        gt: np.ndarray = np.array([[0, 1], [1, 0]], dtype=np.float32)
        pred: np.ndarray = np.array([[0, 1], [0, 0]], dtype=np.float32)

        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt)
        assert "iou" in metrics
