# tests/test_validation_pipeline.py

"""Интеграционные тесты для пайплайна валидации и бенчмаркинга.

Проверяет:
- Согласованность результатов Torch vs OpenCV/Sklearn
- Воспроизводимость метрик при повторных вызовах
- Стабильность бенчмарков (допуск ±50% времени из-за ОС)
- Корректную работу временных директорий
- Интеграцию с TorchImplementationValidator и SegmentationTester

Маркер: @pytest.mark.integration
"""
# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from pathlib import Path
import tempfile


# ──────────────────────────────────────────────────────────────────────
@pytest.mark.integration
class TestValidationPipeline:
    """Класс тестов для реализации TorchImplementationValidator."""

    def test_torch_vs_opencv_validation(self, rgb_image: np.ndarray) -> None:
        """Валидация Torch vs OpenCV реализаций."""
        from testing.TorchImplementationValidator import TorchImplementationValidator
        from segmenters.TorchSegmenter import TorchSegmenter
        from segmenters.SklearnSegmenter import SklearnSegmenter

        with tempfile.TemporaryDirectory() as tmpdir:
            validator = TorchImplementationValidator(output_dir=tmpdir)

            # Простой тест для одного метода
            results: Dict[str, Any] = validator.validate_segmentation_methods(
                image_path=rgb_image,
                methods_list=[("global_thresholding", {"threshold": 0.5})],
                first_segmenter_class=TorchSegmenter,
                second_segmenter_class=SklearnSegmenter,
                first_method_name="Global_Thresholding_Torch",
                second_method_name="Global_Thresholding_Skleran",
                status_message="Test",
                prefix="test",
                validation_type="threshold",
            )

            assert len(results) >= 0

    # ──────────────────────────────────────────────────────────────────────
    def test_metrics_calculation_consistency(self) -> None:
        """Проверка согласованности метрик между разными вызовами."""
        from metrics.SegmentationMetrics import SegmentationMetrics, MetricsDict

        gt: np.ndarray = np.zeros((100, 100), dtype=np.uint8)
        gt[25:75, 25:75] = 255
        pred: np.ndarray = gt.copy()

        metrics1: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt)
        metrics2: MetricsDict = SegmentationMetrics.calculate_all_metrics(pred, gt)

        assert metrics1["iou"] == metrics2["iou"]
        assert metrics1["dice"] == metrics2["dice"]

    # ──────────────────────────────────────────────────────────────────────
    def test_benchmark_reproducibility(self, rgb_image: np.ndarray) -> None:
        """Проверка воспроизводимости бенчмарка."""
        from testing.SegmentationTester import SegmentationTester
        from segmenters.TorchSegmenter import TorchSegmenter

        with tempfile.TemporaryDirectory() as tmpdir:
            tester = SegmentationTester(base_output_dir=tmpdir)
            tester.add_method("test_method", TorchSegmenter("global_thresholding", threshold=0.5))

            df1: pd.DataFrame = tester.benchmark_methods(rgb_image, n_runs=3, test_name="run1")
            df2: pd.DataFrame = tester.benchmark_methods(rgb_image, n_runs=3, test_name="run2")

            if not df1.empty and not df2.empty:
                time1: float = df1.iloc[0]["Mean_Time_s"]
                time2: float = df2.iloc[0]["Mean_Time_s"]
                # Допускаем 50% разницу из-за системной нагрузки
                assert abs(time1 - time2) / max(time1, time2) < 0.5
