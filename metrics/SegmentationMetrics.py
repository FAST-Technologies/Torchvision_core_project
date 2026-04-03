# metrics/SegmentationMetrics.py

"""
Модуль для расчёта метрик качества сегментации
"""

# Импорт основных библиотек
from typing import (
    List,
    Union,
    Tuple,
    Dict,
    Set,
    Any,
    TypeVar,
    Optional,
    Literal,
    Protocol,
    runtime_checkable,
    overload,
    TYPE_CHECKING,
)
import warnings
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


class SegmentationMetrics:
    """
    Класс для расчёта метрик качества сегментации
    """

    @staticmethod
    def _normalize_masks(
        pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Приводит маски к бинарному виду (0 и 1).
        Возвращает плоские (flattened) массивы для удобного сравнения.
        """
        # Бинаризация предсказания
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

    @staticmethod
    def calculate_iou(
        pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5
    ) -> float:
        """
        Intersection over Union (IoU) / Jaccard Index

        Args:
            pred_mask: Предсказанная маска (0-1 или 0-255)
            gt_mask: Ground truth маска (0-1 или 0-255)
            threshold: Порог для бинаризации (если маска не бинарная)

        Returns:
            Значение IoU от 0 до 1
        """
        # Нормализуем маски к бинарному формату 0/1
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )

        # Вычисляем пересечение и объединение
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        if union == 0:
            return 0.0
        return float(intersection / (union + 1e-8))

    @staticmethod
    def calculate_accuracy_sklearn(
        pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5
    ) -> float:
        """
        Accuracy Score через sklearn.metrics.accuracy_score.
        Accuracy с использованием оптимизированной функции sklearn.

        Args:
            pred_mask: Предсказанная маска
            gt_mask: Ground truth маска
            threshold: Порог для бинаризации

        Returns:
            Значение Accuracy score от 0 до 1
        """
        # Нормализуем маски
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )

        # Вычисляем Accuracy score
        try:
            score = accuracy_score(gt_binary.ravel(), pred_binary.ravel())
            return float(score)
        except ValueError as e:
            warnings.warn(f"Ошибка вычисления accuracy_score: {e}. Возвращаем 0.0")
            return 0.0

    @staticmethod
    def calculate_jaccard_sklearn(
        pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5
    ) -> float:
        """
        Jaccard Score через sklearn.metrics.jaccard_score.
        Эквивалент IoU, но с использованием оптимизированной функции sklearn.

        Args:
            pred_mask: Предсказанная маска
            gt_mask: Ground truth маска
            threshold: Порог для бинаризации

        Returns:
            Значение Jaccard score от 0 до 1
        """
        # Нормализуем маски
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )

        # Вычисляем Jaccard score
        try:
            score = jaccard_score(
                gt_binary.ravel(), pred_binary.ravel(), zero_division=0.0
            )
            return float(score)
        except ValueError as e:
            warnings.warn(f"Ошибка вычисления jaccard_score: {e}. Возвращаем 0.0")
            return 0.0

    @staticmethod
    def calculate_dice_coefficient(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        threshold: float = 0.5,
        smooth: float = 1e-6,
    ) -> float:
        """
        Dice Coefficient / F1 Score для сегментации

        Args:
            pred_mask: Предсказанная маска
            gt_mask: Ground truth маска
            threshold: Порог для бинаризации
            smooth: Малое значение для избежания деления на ноль

        Returns:
            Значение Dice coefficient от 0 до 1
        """
        # Нормализуем маски
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        dice = (2.0 * intersection + smooth) / (
            pred_binary.sum() + gt_binary.sum() + smooth
        )
        return float(dice)

    @staticmethod
    def calculate_precision_recall(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        threshold: float = 0.5,
        verbose: bool = False,
    ) -> Tuple[float, float]:
        """
        Precision и Recall для бинарной сегментации

        Args:
            pred_mask: Предсказанная маска
            gt_mask: Ground truth маска
            threshold: Порог для бинаризации

        Returns:
            (precision, recall) значения от 0 до 1
        """
        # Нормализуем маски
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )

        # Sklearn (Эталон)
        p_sklearn = precision_score(
            gt_binary.ravel(), pred_binary.ravel(), zero_division=0
        )
        r_sklearn = recall_score(
            gt_binary.ravel(), pred_binary.ravel(), zero_division=0
        )

        # Вычисляем True Positives, False Positives, False Negatives
        # Кастомный (Через матрицу ошибок)
        tn, fp, fn, tp = confusion_matrix(
            gt_binary.ravel(), pred_binary.ravel(), labels=[0, 1]
        ).ravel()

        p_custom = tp / (tp + fp + 1e-8)
        r_custom = tp / (tp + fn + 1e-8)

        if verbose:
            print(f"--- Сравнение Precision/Recall ---")
            print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
            print(
                f"Precision: Sklearn={p_sklearn:.6f} | Custom={p_custom:.6f} | Diff={abs(p_sklearn-p_custom):.2e}"
            )
            print(
                f"Recall:    Sklearn={r_sklearn:.6f} | Custom={r_custom:.6f} | Diff={abs(r_sklearn-r_custom):.2e}"
            )

        # print(f"Check difference precision: sklearn {p_custom} && custom {p_sklearn}")
        # print(f"Check difference recall: sklearn {r_custom} && custom {r_sklearn}")

        return float(p_custom), float(r_custom)

    @staticmethod
    def calculate_f1_score(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        threshold: float = 0.5,
        verbose: bool = False,
    ) -> float:
        """
        F1 Score (среднее гармоническое precision и recall)

        Args:
            pred_mask: Предсказанная маска
            gt_mask: Ground truth маска
            threshold: Порог для бинаризации

        Returns:
            Значение F1 Score от 0 до 1
        """
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )
        precision, recall = SegmentationMetrics.calculate_precision_recall(
            pred_mask, gt_mask, threshold
        )
        if precision + recall == 0:
            f1_custom = 0.0
        else:
            f1_custom = 2 * (precision * recall) / (precision + recall + 1e-8)

        f1_sklearn = f1_score(gt_binary.ravel(), pred_binary.ravel())
        if verbose:
            print(
                f"F1-Score: Sklearn={f1_sklearn:.6f} | Custom={f1_custom:.6f} | Diff={abs(f1_sklearn-f1_custom):.2e}"
            )
        return float(f1_custom)

    @staticmethod
    def calculate_mae(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        normalize: bool = True,
        verbose: bool = False,
    ) -> float:
        """
        Mean Absolute Error (Средняя абсолютная погрешность)

        Args:
            pred_mask: Предсказанная маска
            gt_mask: Ground truth маска
            normalize: Нормализовать ли маски к [0, 1]

        Returns:
            Значение MAE
        """
        # Нормализуем маски к [0, 1] если нужно
        if normalize:
            if pred_mask.max() > 1:
                pred_norm = pred_mask.astype(np.float32) / 255.0
            else:
                pred_norm = pred_mask.astype(np.float32)

            if gt_mask.max() > 1:
                gt_norm = gt_mask.astype(np.float32) / 255.0
            else:
                gt_norm = gt_mask.astype(np.float32)
        else:
            pred_norm = pred_mask.astype(np.float32)
            gt_norm = gt_mask.astype(np.float32)

        # Кастомный
        mae_custom = np.abs(pred_norm - gt_norm).mean()

        # Sklearn
        mae_sklearn = mean_absolute_error(gt_norm, pred_norm)

        if verbose:
            print(
                f"MAE: Sklearn={mae_sklearn:.6f} | Custom={mae_custom:.6f} | Diff={abs(mae_sklearn-mae_custom):.2e}"
            )
        return float(mae_custom)

    @staticmethod
    def calculate_hausdorff_distance(
        pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5
    ) -> float:
        """
        Hausdorff Distance (Расстояние Хаусдорфа)

        Args:
            pred_mask: Предсказанная маска
            gt_mask: Ground truth маска
            threshold: Порог для бинаризации

        Returns:
            Значение расстояния Хаусдорфа
        """
        # Нормализуем маски
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )

        # Получаем координаты точек контуров
        pred_coords = np.column_stack(np.where(pred_binary))
        gt_coords = np.column_stack(np.where(gt_binary))

        # Если один из контуров пустой, возвращаем бесконечность
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return float("inf")

        try:
            # Вычисляем двунаправленное расстояние Хаусдорфа
            h1 = directed_hausdorff(pred_coords, gt_coords)[0]
            h2 = directed_hausdorff(gt_coords, pred_coords)[0]
            hausdorff_dist = max(h1, h2)
        except:
            hausdorff_dist = float("inf")
        return float(hausdorff_dist)

    @staticmethod
    def calculate_clustering_metrics(pred_mask: np.ndarray) -> Dict[str, float]:
        """
        Оценивает внутреннюю компактность сегментов (только если классов > 2).
        Для бинарной маски (фон/объект) эти метрики не информативны.
        """
        unique_labels = np.unique(pred_mask)
        # Исключаем фон (0), если он есть
        if len(unique_labels) <= 2:
            return {
                "silhouette_score": np.nan,
                "calinski_harabasz_score": np.nan,
                "davies_bouldin_score": np.nan,
            }

        # Формируем признаки: координаты пикселей (y, x)
        # Для больших изображений лучше делать сэмплирование!
        h, w = pred_mask.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # Сэмплируем 1000 случайных пикселей для скорости
        n_samples = min(1000, h * w)
        indices = np.random.choice(h * w, n_samples, replace=False)

        X = np.column_stack([y_coords.ravel()[indices], x_coords.ravel()[indices]])
        labels = pred_mask.ravel()[indices]

        metrics = {}
        try:
            if len(np.unique(labels)) > 1:
                metrics["silhouette_score"] = silhouette_score(X, labels)
                metrics["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)
                metrics["davies_bouldin_score"] = davies_bouldin_score(X, labels)
            else:
                metrics["silhouette_score"] = np.nan
                metrics["calinski_harabasz_score"] = np.nan
                metrics["davies_bouldin_score"] = np.nan
        except Exception:
            metrics["silhouette_score"] = np.nan
            metrics["calinski_harabasz_score"] = np.nan
            metrics["davies_bouldin_score"] = np.nan
        return metrics

    @staticmethod
    def calculate_pixel_accuracy(
        pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5
    ) -> float:
        """
        Pixel Accuracy (Пиксельная точность)

        Args:
            pred_mask: Предсказанная маска
            gt_mask: Ground truth маска
            threshold: Порог для бинаризации

        Returns:
            Значение точности от 0 до 1
        """
        # Нормализуем маски
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )

        # Вычисляем точность
        correct_pixels = (pred_binary == gt_binary).sum()
        total_pixels = pred_binary.size
        accuracy = correct_pixels / total_pixels
        return float(accuracy)

    @staticmethod
    def calculate_all_metrics(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        threshold: float = 0.5,
        include_hausdorff: bool = True,
        verbose_comparison: bool = False,
    ) -> Dict[str, float]:
        """
        Вычисляет все метрики качества сегментации

        Args:
            pred_mask: Предсказанная маска
            gt_mask: Ground truth маска
            threshold: Порог для бинаризации
            include_hausdorff: Включать ли вычисление расстояния Хаусдорфа

        Returns:
            Словарь со всеми метриками
        """
        metrics = {}
        # 1. Точность (Accuracy)
        metrics["accuracy"] = SegmentationMetrics.calculate_accuracy_sklearn(
            pred_mask, gt_mask, threshold
        )

        # 2. IoU / Jaccard
        iou_custom = SegmentationMetrics.calculate_iou(pred_mask, gt_mask, threshold)
        iou_sklearn = SegmentationMetrics.calculate_jaccard_sklearn(
            pred_mask, gt_mask, threshold
        )
        metrics["iou"] = iou_custom
        metrics["jaccard_score"] = iou_sklearn

        if verbose_comparison:
            print(
                f"IoU Check: Custom={iou_custom:.6f} | Sklearn={iou_sklearn:.6f} | Diff={abs(iou_custom-iou_sklearn):.2e}"
            )

        # 3. Dice
        metrics["dice"] = SegmentationMetrics.calculate_dice_coefficient(
            pred_mask, gt_mask, threshold
        )

        # 4. Precision & Recall (с внутренней проверкой)
        metrics["precision"], metrics["recall"] = (
            SegmentationMetrics.calculate_precision_recall(
                pred_mask, gt_mask, threshold, verbose=verbose_comparison
            )
        )

        # 5. F1 Score (с внутренней проверкой)
        metrics["f1_score"] = SegmentationMetrics.calculate_f1_score(
            pred_mask, gt_mask, threshold, verbose=verbose_comparison
        )

        # 6. Pixel Accuracy
        metrics["pixel_accuracy"] = SegmentationMetrics.calculate_pixel_accuracy(
            pred_mask, gt_mask, threshold
        )

        # 7. MAE (с внутренней проверкой)
        metrics["mae"] = SegmentationMetrics.calculate_mae(
            pred_mask, gt_mask, verbose=verbose_comparison
        )

        # 8. Hausdorff
        if include_hausdorff:
            metrics["hausdorff_distance"] = (
                SegmentationMetrics.calculate_hausdorff_distance(
                    pred_mask, gt_mask, threshold
                )
            )

        # 9. Статистика областей и Confusion Matrix
        pred_binary, gt_binary = SegmentationMetrics._normalize_masks(
            pred_mask, gt_mask, threshold
        )

        # Площади (сумма единичек)
        pred_area = int(np.sum(pred_binary))
        gt_area = int(np.sum(gt_binary))

        metrics["predicted_area"] = float(pred_area)
        metrics["ground_truth_area"] = float(gt_area)
        metrics["area_difference"] = float(abs(pred_area - gt_area))

        if max(pred_area, gt_area) > 0:
            metrics["area_ratio"] = min(pred_area, gt_area) / max(pred_area, gt_area)
        else:
            metrics["area_ratio"] = 0.0

        # Confusion Matrix Elements
        try:
            tn, fp, fn, tp = confusion_matrix(
                gt_binary.ravel(), pred_binary.ravel(), labels=[0, 1]
            ).ravel()
            metrics["true_negative"] = int(tn)
            metrics["false_positive"] = int(fp)
            metrics["false_negative"] = int(fn)
            metrics["true_positive"] = int(tp)

            if verbose_comparison:
                print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        except ValueError as e:
            warnings.warn(f"Не удалось вычислить матрицу ошибок: {e}")
            metrics["true_negative"] = 0
            metrics["false_positive"] = 0
            metrics["false_negative"] = 0
            metrics["true_positive"] = 0

        return metrics

    @staticmethod
    def evaluate_multiple_masks(
        pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], threshold: float = 0.5
    ) -> Dict[str, Dict[str, float]]:
        """
        Оценка нескольких масок с вычислением средних метрик

        Args:
            pred_masks: Список предсказанных масок
            gt_masks: Список ground truth масок
            threshold: Порог для бинаризации

        Returns:
            Словарь с метриками для каждой маски и средними значениями
        """
        if len(pred_masks) != len(gt_masks):
            raise ValueError(
                "Количество предсказанных и ground truth масок должно совпадать"
            )

        all_metrics = []
        individual_results = {}

        for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
            metrics = SegmentationMetrics.calculate_all_metrics(
                pred_mask, gt_mask, threshold, include_hausdorff=False
            )
            all_metrics.append(metrics)
            individual_results[f"mask_{i}"] = metrics

        # Вычисляем средние метрики
        avg_metrics = {}
        if not all_metrics:
            return {"average_metrics": {}, "individual_results": {}}

        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
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
