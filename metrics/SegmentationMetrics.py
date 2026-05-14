# metrics/SegmentationMetrics.py

"""Модуль для расчёта метрик качества семантической и бинарной сегментации.

Предназначен для универсальной оценки соответствия предсказанных масок 
эталонным (Ground Truth) с поддержкой:
- Бинарной сегментации: объект/фон (0/1 или 0/255).
- Многоклассовой сегментации: пер-классовые метрики, матрица ошибок.
- Кластеризации: внутренние метрики компактности (Silhouette, CH, DB).

Поддерживаемые метрики:
┌─────────────────────────────────────────────────────┐
│ Категория          │ Метрики                         │
├─────────────────────────────────────────────────────┤
│ Пересечение/Объединение │ IoU, Jaccard, Dice/F1      │
│ Классификация        │ Precision, Recall, Accuracy   │
│ Ошибки              │ MAE, Pixel Accuracy           │
│ Геометрия контуров   │ Hausdorff Distance            │
│ Статистика областей  │ Area, Ratio, Difference       │
│ Конфузионная матрица │ TP, FP, FN, TN                │
│ Кластеризация*       │ Silhouette, CH, Davies-Bouldin│
└─────────────────────────────────────────────────────┘
* — только для масок с >2 уникальных классов

Особенности реализации:
- 🔄 Все методы статические: класс используется как пространство имён.
- 🎚️ Автоматическая бинаризация с адаптивным порогом для [0,1] и [0,255].
- 🛡️ Защита от деления на ноль через `smooth=1e-6` и `1e-8` в знаменателях.
- 🔍 Опциональное сравнение кастомных и sklearn-реализаций (через `verbose`).
- ⚡ Hausdorff Distance вычисляется только по координатам контуров, не по всей маске.
- 📦 Групповой расчёт: `calculate_all_metrics()` для получения полного набора за один вызов.
- 📊 Пакетная оценка: `evaluate_multiple_masks()` с агрегацией средних/стандартных значений.

Workflow:
1. Подготовить предсказанную и эталонную маски (формат: `np.ndarray`, формы `(H, W)`).
2. Вызвать `SegmentationMetrics.calculate_all_metrics(pred_mask, gt_mask, threshold=0.5)`.
3. Получить словарь с метриками для логирования, визуализации или агрегации.

Примечание:
- Для бинарных масок значения должны быть в {0, 1} или {0, 255}; модуль автоматически
  приводит к единому формату через `_normalize_masks()`.
- Hausdorff Distance может возвращать `inf`, если один из контуров пуст — это ожидаемое
  поведение, обрабатывайте такие случаи в вызывающем коде.
- Метрики кластеризации возвращают `np.nan` для бинарных масок, так как неинформативны
  при двух классах (фон/объект).
- Для массового тестирования на датасетах используйте обёртки в `BatchClassicTester2`,
  `SegmentationBenchmark` или `SegmentationTester`.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций

import warnings
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    confusion_matrix,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
)

import logging

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
MaskArray = np.ndarray  # Binary or multi-class mask: H×W, dtype uint8/int/float
MetricValue = Union[Any]
MetricsDict = Dict[str, MetricValue]
ClusteringMetricsDict = Dict[str, Optional[float]]


class SegmentationMetrics:
    """Класс для расчёта метрик качества сегментации.

    Все методы статические — класс используется как пространство имён.
    Поддерживает как бинарную, так и многоклассовую сегментацию.

    Особенности:
    - Автоматическая бинаризация масок с порогом.
    - Защита от деления на ноль через `smooth` и `1e-8`.
    - Сравнение кастомных и sklearn-реализаций (опционально, через `verbose`).
    - Вычисление расстояния Хаусдорфа только для контуров (не для всей маски).

    Example:
        ```python
        metrics = SegmentationMetrics.calculate_all_metrics(
            pred_mask=pred,
            gt_mask=gt,
            threshold=0.5,
            include_hausdorff=True,
        )
        logging.info(f"IoU: {metrics['iou']:.3f}, Dice: {metrics['dice']:.3f}")
        ```
    """

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize_masks(
        pred_mask: MaskArray,
        gt_mask: MaskArray,
        threshold: float = 0.5,
    ) -> Tuple[MaskArray, MaskArray]:
        """Приводит маски к бинарному виду (0 и 1) и возвращает плоские массивы.

        Логика:
        - Если макс. значение > 1, предполагается диапазон [0, 255] → порог умножается на 255.
        - Иначе предполагается диапазон [0, 1] → порог применяется напрямую.
        - Результат: `uint8` массивы формы `(H*W,)` для удобного сравнения.

        Args:
            pred_mask: Предсказанная маска (бинарная или вероятностная).
            gt_mask: Ground truth маска.
            threshold: Порог бинаризации (0.0–1.0).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            - Бинаризованное предсказание, форма `(N,)`, dtype `uint8`.
            - Бинаризованный ground truth, форма `(N,)`, dtype `uint8`.
        """
        # Бинаризация предсказания
        pred_binary: np.ndarray
        gt_binary: np.ndarray
        if pred_mask.max() > 1:
            pred_binary = (pred_mask > threshold * 255).astype(np.uint8)
        else:
            pred_binary = (pred_mask > threshold).astype(np.uint8)

        # Бинаризация Ground Truth
        if gt_mask.max() > 1:
            gt_binary = (gt_mask > threshold * 255).astype(np.uint8)
        else:
            gt_binary = (gt_mask > threshold).astype(np.uint8)
        return pred_binary, gt_binary

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_iou(
        pred_mask: MaskArray,
        gt_mask: MaskArray,
        threshold: float = 0.5,
    ) -> MetricValue:
        """Intersection over Union (IoU) / Jaccard Index.

        Формула:
        ```
        IoU = |A ∩ B| / |A ∪ B|
        ```

        Args:
            pred_mask: Предсказанная маска (0–1 или 0–255).
            gt_mask: Ground truth маска.
            threshold: Порог для бинаризации.

        Returns:
            float: Значение IoU в диапазоне [0, 1].
        """
        # Нормализуем маски к бинарному формату 0/1
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )

        # Вычисляем пересечение и объединение
        intersection: int = int(np.logical_and(pred_binary, gt_binary).sum())
        union: int = int(np.logical_or(pred_binary, gt_binary).sum())
        if union == 0:
            return 0.0
        return float(intersection / (union + 1e-8))

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_accuracy_sklearn(
        pred_mask: MaskArray,
        gt_mask: MaskArray,
        threshold: float = 0.5,
    ) -> MetricValue:
        """Accuracy Score через `sklearn.metrics.accuracy_score`.

        Args:
            pred_mask: Предсказанная маска.
            gt_mask: Ground truth маска.
            threshold: Порог для бинаризации.

        Returns:
            float: Значение Accuracy в диапазоне [0, 1].
        """
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )

        try:
            score: MetricValue = accuracy_score(gt_binary.ravel(), pred_binary.ravel())
            return float(score)
        except ValueError as e:
            warnings.warn(f"Ошибка вычисления accuracy_score: {e}. Возвращаем 0.0")
            return 0.0

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_jaccard_sklearn(
        pred_mask: MaskArray,
        gt_mask: MaskArray,
        threshold: float = 0.5,
    ) -> MetricValue:
        """Jaccard Score через `sklearn.metrics.jaccard_score` (эквивалент IoU).

        Args:
            pred_mask: Предсказанная маска.
            gt_mask: Ground truth маска.
            threshold: Порог для бинаризации.

        Returns:
            float: Значение Jaccard score в диапазоне [0, 1].
        """
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )

        try:
            score: MetricValue = jaccard_score(
                gt_binary.ravel(), pred_binary.ravel(), zero_division=0.0
            )
            return float(score)
        except ValueError as e:
            warnings.warn(f"Ошибка вычисления jaccard_score: {e}. Возвращаем 0.0")
            return 0.0

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_dice_coefficient(
        pred_mask: MaskArray,
        gt_mask: MaskArray,
        threshold: float = 0.5,
        smooth: float = 1e-6,
    ) -> MetricValue:
        """Dice Coefficient / F1 Score для бинарной сегментации.

        Формула:
        ```
        Dice = (2 * |A ∩ B| + smooth) / (|A| + |B| + smooth)
        ```

        Args:
            pred_mask: Предсказанная маска.
            gt_mask: Ground truth маска.
            threshold: Порог для бинаризации.
            smooth: Малое значение для избежания деления на ноль.

        Returns:
            float: Значение Dice coefficient в диапазоне [0, 1].
        """
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )
        intersection: int = int(np.logical_and(pred_binary, gt_binary).sum())
        dice: MetricValue = (2.0 * intersection + smooth) / (
            int(pred_binary.sum()) + int(gt_binary.sum()) + smooth
        )
        return float(dice)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_precision_recall(
        pred_mask: MaskArray,
        gt_mask: MaskArray,
        threshold: float = 0.5,
        verbose: bool = False,
    ) -> Tuple[MetricValue, MetricValue]:
        """Precision и Recall для бинарной сегментации.

        Формулы:
        ```
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        ```

        Args:
            pred_mask: Предсказанная маска.
            gt_mask: Ground truth маска.
            threshold: Порог для бинаризации.
            verbose: Если `True`, выводит сравнение sklearn vs custom.

        Returns:
            Tuple[float, float]: (precision, recall) в диапазоне [0, 1].
        """
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )

        # Sklearn (Эталон)
        p_sklearn: MetricValue = precision_score(
            gt_binary.ravel(), pred_binary.ravel(), zero_division=0
        )
        r_sklearn: MetricValue = recall_score(
            gt_binary.ravel(), pred_binary.ravel(), zero_division=0
        )

        # Confusion matrix также требует 1D
        tn, fp, fn, tp = confusion_matrix(
            gt_binary.ravel(), pred_binary.ravel(), labels=[0, 1]
        ).ravel()

        p_custom: MetricValue = tp / (tp + fp + 1e-8)
        r_custom: MetricValue = tp / (tp + fn + 1e-8)

        if verbose:
            logging.info("--- Сравнение Precision/Recall ---")
            logging.info(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
            logging.info(
                f"Precision: Sklearn={p_sklearn:.6f} | Custom={p_custom:.6f} | Diff={abs(p_sklearn - p_custom):.2e}"
            )
            logging.info(
                f"Recall:    Sklearn={r_sklearn:.6f} | Custom={r_custom:.6f} | Diff={abs(r_sklearn - r_custom):.2e}"
            )

        return float(p_custom), float(r_custom)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_f1_score(
        pred_mask: MaskArray,
        gt_mask: MaskArray,
        threshold: float = 0.5,
        verbose: bool = False,
    ) -> MetricValue:
        """F1 Score (среднее гармоническое precision и recall).

        Формула:
        ```
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        ```

        Args:
            pred_mask: Предсказанная маска.
            gt_mask: Ground truth маска.
            threshold: Порог для бинаризации.
            verbose: Если `True`, выводит сравнение sklearn vs custom.

        Returns:
            float: Значение F1 Score в диапазоне [0, 1].
        """
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )
        precision, recall = SegmentationMetrics.calculate_precision_recall(
            pred_mask, gt_mask, threshold
        )
        if precision + recall == 0:
            f1_custom: MetricValue = 0.0
        else:
            f1_custom = 2 * (precision * recall) / (precision + recall + 1e-8)

        f1_sklearn: MetricValue = f1_score(gt_binary.ravel(), pred_binary.ravel())
        if verbose:
            logging.info(
                f"F1-Score: Sklearn={f1_sklearn:.6f} | Custom={f1_custom:.6f} | Diff={abs(f1_sklearn - f1_custom):.2e}"
            )
        return float(f1_custom)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_mae(
        pred_mask: MaskArray,
        gt_mask: MaskArray,
        normalize: bool = True,
        verbose: bool = False,
    ) -> MetricValue:
        """Mean Absolute Error (Средняя абсолютная погрешность).

        Args:
            pred_mask: Предсказанная маска.
            gt_mask: Ground truth маска.
            normalize: Если `True`, нормализует маски к [0, 1].
            verbose: Если `True`, выводит сравнение sklearn vs custom.

        Returns:
            float: Значение MAE.
        """
        # Нормализация к [0, 1] если нужно
        pred_norm: np.ndarray
        gt_norm: np.ndarray
        if normalize:
            pred_norm = (
                pred_mask.astype(np.float32) / 255.0
                if pred_mask.max() > 1
                else pred_mask.astype(np.float32)
            )
            gt_norm = (
                gt_mask.astype(np.float32) / 255.0
                if gt_mask.max() > 1
                else gt_mask.astype(np.float32)
            )
        else:
            pred_norm = pred_mask.astype(np.float32)
            gt_norm = gt_mask.astype(np.float32)

        # Кастомный
        mae_custom: MetricValue = np.abs(pred_norm - gt_norm).mean()

        # Sklearn
        if verbose:
            mae_sklearn: MetricValue = mean_absolute_error(gt_norm, pred_norm)
            logging.info(
                f"MAE: Sklearn={mae_sklearn:.6f} | Custom={mae_custom:.6f} | Diff={abs(mae_sklearn - mae_custom):.2e}"
            )
        return float(mae_custom)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_hausdorff_distance(
        pred_mask: MaskArray,
        gt_mask: MaskArray,
        threshold: float = 0.5,
    ) -> MetricValue:
        """Hausdorff Distance (Расстояние Хаусдорфа) между контурами масок.

        Вычисляется как максимум из двух направленных расстояний:
        ```
        H(A, B) = max( h(A→B), h(B→A) )
        ```

        Args:
            pred_mask: Предсказанная маска.
            gt_mask: Ground truth маска.
            threshold: Порог для бинаризации.

        Returns:
            float: Значение расстояния Хаусдорфа (в пикселях) или `inf` при ошибке.
        """
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )

        # Получаем координаты точек контуров
        pred_coords: np.ndarray = np.column_stack(np.where(pred_binary))
        gt_coords: np.ndarray = np.column_stack(np.where(gt_binary))

        # Если один из контуров пустой, возвращаем бесконечность
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return float("inf")

        try:
            # Вычисляем двунаправленное расстояние Хаусдорфа
            h1: MetricValue = directed_hausdorff(pred_coords, gt_coords)[0]
            h2: MetricValue = directed_hausdorff(gt_coords, pred_coords)[0]
            hausdorff_dist = max(h1, h2)
        except ValueError:
            hausdorff_dist = float("inf")
        return float(hausdorff_dist)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_clustering_metrics(
        pred_mask: MaskArray,
        n_samples: int = 1000,
    ) -> ClusteringMetricsDict:
        """Оценивает внутреннюю компактность сегментов (только если классов > 2).

        Поддерживаемые метрики:
        - Silhouette Score: [-1, 1], чем больше — тем лучше.
        - Calinski-Harabasz: [0, ∞), чем больше — тем лучше.
        - Davies-Bouldin: [0, ∞), чем меньше — тем лучше.

        ⚠️ Для бинарной маски (фон/объект) эти метрики не информативны.

        Args:
            pred_mask: Многоклассовая маска с целочисленными метками.
            n_samples: Количество случайных пикселей для сэмплирования (для скорости).

        Returns:
            Dict[str, Optional[float]]:
            ```python
            {
                "silhouette_score": float | None,
                "calinski_harabasz_score": float | None,
                "davies_bouldin_score": float | None,
            }
            ```
        """
        unique_labels: np.ndarray = np.unique(pred_mask)

        if len(unique_labels) <= 2:
            return {
                "silhouette_score": np.nan,
                "calinski_harabasz_score": np.nan,
                "davies_bouldin_score": np.nan,
            }

        h, w = pred_mask.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # Сэмплируем 1000 случайных пикселей для скорости
        n_pixels: int = int(h * w)
        indices: np.ndarray = np.random.choice(
            n_pixels, min(n_samples, n_pixels), replace=False
        )

        X: np.ndarray = np.column_stack(
            [y_coords.ravel()[indices], x_coords.ravel()[indices]]
        )
        labels: np.ndarray = pred_mask.ravel()[indices]

        metrics: ClusteringMetricsDict = {}
        try:
            if len(np.unique(labels)) > 1:
                metrics["silhouette_score"] = silhouette_score(X, labels)
                metrics["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)
                metrics["davies_bouldin_score"] = davies_bouldin_score(X, labels)
            else:
                metrics["silhouette_score"] = None
                metrics["calinski_harabasz_score"] = None
                metrics["davies_bouldin_score"] = None
        except Exception:
            # При любой ошибке возвращаем None
            metrics["silhouette_score"] = None
            metrics["calinski_harabasz_score"] = None
            metrics["davies_bouldin_score"] = None
        return metrics

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_pixel_accuracy(
        pred_mask: MaskArray,
        gt_mask: MaskArray,
        threshold: float = 0.5,
    ) -> MetricValue:
        """Pixel Accuracy (Пиксельная точность).

        Формула:
        ```
        Accuracy = |{i : pred[i] == gt[i]}| / N
        ```

        Args:
            pred_mask: Предсказанная маска.
            gt_mask: Ground truth маска.
            threshold: Порог для бинаризации.

        Returns:
            float: Значение точности в диапазоне [0, 1].
        """
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )

        correct_pixels: int = int((pred_binary == gt_binary).sum())
        total_pixels: int = int(pred_binary.size)

        if total_pixels == 0:
            return 0.0
        return float(correct_pixels / total_pixels)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_all_metrics(
        pred_mask: MaskArray,
        gt_mask: MaskArray,
        threshold: float = 0.5,
        include_hausdorff: bool = True,
        verbose_comparison: bool = False,
        metrics_list: Optional[List[str]] = None,
    ) -> MetricsDict:
        """Вычисляет все метрики качества сегментации в одном вызове.

        Возвращаемые метрики:
        - `accuracy`, `iou`, `jaccard_score`, `dice`
        - `precision`, `recall`, `f1_score`, `pixel_accuracy`, `mae`
        - `hausdorff_distance` (если `include_hausdorff=True`)
        - `predicted_area`, `ground_truth_area`, `area_difference`, `area_ratio`
        - `true_positive`, `false_positive`, `false_negative`, `true_negative`

        Args:
            pred_mask: Предсказанная маска.
            gt_mask: Ground truth маска.
            threshold: Порог для бинаризации.
            include_hausdorff: Включать ли вычисление расстояния Хаусдорфа (медленно!).
            verbose_comparison: Если `True`, выводит сравнение кастомных и sklearn-реализаций.
            metrics_list: Список метрик для вычисления. Если None — вычисляются все.

        Returns:
            Dict[str, float]: Словарь со всеми рассчитанными метриками.
        """
        metrics: MetricsDict = {}

        if metrics_list is None:
            metrics_list = [
                "accuracy",
                "iou",
                "jaccard_score",
                "dice",
                "precision",
                "recall",
                "f1_score",
                "pixel_accuracy",
                "mae",
                "hausdorff_distance",
                "predicted_area",
                "ground_truth_area",
                "area_difference",
                "area_ratio",
                "true_positive",
                "false_positive",
                "false_negative",
                "true_negative",
            ]

        def _need(metric_name: str) -> bool:
            return metric_name in metrics_list

        # 1. Точность (Accuracy)
        if _need("accuracy"):
            metrics["accuracy"] = SegmentationMetrics.calculate_accuracy_sklearn(
                pred_mask, gt_mask, threshold
            )

        # 2. IoU / Jaccard
        if _need("iou") or _need("jaccard_score"):
            iou_custom: MetricValue = SegmentationMetrics.calculate_iou(
                pred_mask, gt_mask, threshold
            )
            if _need("iou"):
                metrics["iou"] = iou_custom
            if _need("jaccard_score"):
                iou_sklearn: MetricValue = (
                    SegmentationMetrics.calculate_jaccard_sklearn(
                        pred_mask, gt_mask, threshold
                    )
                )
                metrics["jaccard_score"] = iou_sklearn

            if verbose_comparison and _need("iou") and _need("jaccard_score"):
                logging.info(
                    f"IoU Check: Custom={iou_custom:.6f} | Sklearn={iou_sklearn:.6f} | Diff={abs(iou_custom - iou_sklearn):.2e}"
                )

        # 3. Dice
        if _need("dice"):
            metrics["dice"] = SegmentationMetrics.calculate_dice_coefficient(
                pred_mask, gt_mask, threshold
            )

        # 4. Precision & Recall (с внутренней проверкой)
        if _need("precision") or _need("recall") or _need("f1_score"):
            precision, recall = SegmentationMetrics.calculate_precision_recall(
                pred_mask, gt_mask, threshold, verbose=verbose_comparison
            )
            if _need("precision"):
                metrics["precision"] = precision
            if _need("recall"):
                metrics["recall"] = recall

        # 5. F1 Score (с внутренней проверкой)
        if _need("f1_score"):
            metrics["f1_score"] = SegmentationMetrics.calculate_f1_score(
                pred_mask, gt_mask, threshold, verbose=verbose_comparison
            )

        # 6. Pixel Accuracy
        if _need("pixel_accuracy"):
            metrics["pixel_accuracy"] = SegmentationMetrics.calculate_pixel_accuracy(
                pred_mask, gt_mask, threshold
            )

        # 7. MAE (с внутренней проверкой)
        if _need("mae"):
            metrics["mae"] = SegmentationMetrics.calculate_mae(
                pred_mask, gt_mask, verbose=verbose_comparison
            )

        # 8. Hausdorff
        if _need("hausdorff_distance") and include_hausdorff:
            metrics["hausdorff_distance"] = (
                SegmentationMetrics.calculate_hausdorff_distance(
                    pred_mask, gt_mask, threshold
                )
            )

        # 9. Статистика областей и Confusion Matrix
        area_or_cm_metrics = [
            "predicted_area",
            "ground_truth_area",
            "area_difference",
            "area_ratio",
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        ]
        if any(_need(m) for m in area_or_cm_metrics):
            pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
                pred_mask, gt_mask, threshold
            )

            # Площади
            if _need("predicted_area"):
                metrics["predicted_area"] = float(np.sum(pred_binary))
            if _need("ground_truth_area"):
                metrics["ground_truth_area"] = float(np.sum(gt_binary))
            if _need("area_difference"):
                pred_area = int(np.sum(pred_binary))
                gt_area = int(np.sum(gt_binary))
                metrics["area_difference"] = float(abs(pred_area - gt_area))
            if _need("area_ratio"):
                pred_area = int(np.sum(pred_binary))
                gt_area = int(np.sum(gt_binary))
                if max(pred_area, gt_area) > 0:
                    metrics["area_ratio"] = min(pred_area, gt_area) / max(
                        pred_area, gt_area
                    )
                else:
                    metrics["area_ratio"] = 0.0

            # Confusion Matrix Elements
            cm_metrics = [
                "true_positive",
                "false_positive",
                "false_negative",
                "true_negative",
            ]
            if any(_need(m) for m in cm_metrics):
                try:
                    tn, fp, fn, tp = confusion_matrix(
                        gt_binary.ravel(), pred_binary.ravel(), labels=[0, 1]
                    ).ravel()
                    if _need("true_negative"):
                        metrics["true_negative"] = int(tn)
                    if _need("false_positive"):
                        metrics["false_positive"] = int(fp)
                    if _need("false_negative"):
                        metrics["false_negative"] = int(fn)
                    if _need("true_positive"):
                        metrics["true_positive"] = int(tp)
                except ValueError as e:
                    warnings.warn(f"Не удалось вычислить матрицу ошибок: {e}")
                    for m in cm_metrics:
                        if _need(m):
                            metrics[m] = 0

        return metrics

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def evaluate_multiple_masks(
        pred_masks: List[MaskArray],
        gt_masks: List[MaskArray],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Оценка нескольких масок с вычислением средних метрик.

        Args:
            pred_masks: Список предсказанных масок.
            gt_masks: Список ground truth масок.
            threshold: Порог для бинаризации.

        Returns:
            Dict[str, Any]:
            ```python
            {
                "individual_results": Dict[str, MetricsDict],  # по индексам
                "average_metrics": Dict[str, float],           # средние значения
                "total_masks": int,
            }
            ```

        Raises:
            ValueError: Если длины списков не совпадают.
        """
        if len(pred_masks) != len(gt_masks):
            raise ValueError(
                "Количество предсказанных и ground truth масок должно совпадать"
            )

        all_metrics: List[MetricsDict] = []
        individual_results: Dict[str, MetricsDict] = {}

        for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
            metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(
                pred_mask, gt_mask, threshold, include_hausdorff=False
            )
            all_metrics.append(metrics)
            individual_results[f"mask_{i}"] = metrics

        # Вычисляем средние метрики
        avg_metrics: Dict[str, MetricValue] = {}
        if not all_metrics:
            return {"average_metrics": {}, "individual_results": {}}

        for key in all_metrics[0].keys():
            values: List[MetricValue] = [
                m[key] for m in all_metrics if key in m and not np.isnan(m[key])
            ]
            if values:
                avg_metrics[f"avg_{key}"] = float(np.mean(values))
                avg_metrics[f"std_{key}"] = float(np.std(values))
                avg_metrics[f"min_{key}"] = float(np.min(values))
                avg_metrics[f"max_{key}"] = float(np.max(values))
        return {
            "individual_results": individual_results,
            "average_metrics": avg_metrics,
            "total_masks": len(pred_masks),
        }
