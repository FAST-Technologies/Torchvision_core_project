"""
Модуль для расчёта метрик качества сегментации
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union
import warnings
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
    """
    Класс для расчёта метрик качества сегментации
    """
    
    @staticmethod
    def calculate_iou(pred_mask: np.ndarray, 
                      gt_mask: np.ndarray, 
                      threshold: float = 0.5) -> float:
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
        if pred_mask.max() > 1:
            pred_binary = (pred_mask > threshold * 255).astype(np.uint8)
        else:
            pred_binary = (pred_mask > threshold).astype(np.uint8)
            
        if gt_mask.max() > 1:
            gt_binary = (gt_mask > threshold * 255).astype(np.uint8)
        else:
            gt_binary = (gt_mask > threshold).astype(np.uint8)
        
        # Вычисляем пересечение и объединение
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        # Избегаем деления на ноль
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return float(iou)
    
    @staticmethod
    def calculate_dice_coefficient(pred_mask: np.ndarray, 
                                   gt_mask: np.ndarray, 
                                   threshold: float = 0.5,
                                   smooth: float = 1e-6) -> float:
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
        if pred_mask.max() > 1:
            pred_binary = (pred_mask > threshold * 255).astype(np.uint8)
        else:
            pred_binary = (pred_mask > threshold).astype(np.uint8)
            
        if gt_mask.max() > 1:
            gt_binary = (gt_mask > threshold * 255).astype(np.uint8)
        else:
            gt_binary = (gt_mask > threshold).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        dice = (2. * intersection + smooth) / (pred_binary.sum() + gt_binary.sum() + smooth)
        
        return float(dice)
    
    @staticmethod
    def calculate_precision_recall(pred_mask: np.ndarray, 
                                   gt_mask: np.ndarray, 
                                   threshold: float = 0.5) -> Tuple[float, float]:
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
        if pred_mask.max() > 1:
            pred_binary = (pred_mask > threshold * 255).astype(np.uint8)
            gt_binary = (gt_mask > threshold * 255).astype(np.uint8)
        else:
            pred_binary = (pred_mask > threshold).astype(np.uint8)
            gt_binary = (gt_mask > threshold).astype(np.uint8)
        
        # Вычисляем True Positives, False Positives, False Negatives
        tp = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
        fp = np.logical_and(pred_binary == 1, gt_binary == 0).sum()
        fn = np.logical_and(pred_binary == 0, gt_binary == 1).sum()
        
        # Вычисляем precision и recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        return float(precision), float(recall)
    
    @staticmethod
    def calculate_f1_score(pred_mask: np.ndarray, 
                           gt_mask: np.ndarray, 
                           threshold: float = 0.5) -> float:
        """
        F1 Score (среднее гармоническое precision и recall)
        
        Args:
            pred_mask: Предсказанная маска
            gt_mask: Ground truth маска
            threshold: Порог для бинаризации
            
        Returns:
            Значение F1 Score от 0 до 1
        """
        precision, recall = SegmentationMetrics.calculate_precision_recall(
            pred_mask, gt_mask, threshold
        )
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return float(f1)
    
    @staticmethod
    def calculate_mae(pred_mask: np.ndarray, 
                      gt_mask: np.ndarray, 
                      normalize: bool = True) -> float:
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
        
        # Вычисляем MAE
        mae = np.abs(pred_norm - gt_norm).mean()
        return float(mae)
    
    @staticmethod
    def calculate_hausdorff_distance(pred_mask: np.ndarray, 
                                     gt_mask: np.ndarray, 
                                     threshold: float = 0.5) -> float:
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
        if pred_mask.max() > 1:
            pred_binary = (pred_mask > threshold * 255)
            gt_binary = (gt_mask > threshold * 255)
        else:
            pred_binary = (pred_mask > threshold)
            gt_binary = (gt_mask > threshold)
        
        # Получаем координаты точек контуров
        pred_coords = np.column_stack(np.where(pred_binary))
        gt_coords = np.column_stack(np.where(gt_binary))
        
        # Если один из контуров пустой, возвращаем бесконечность
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return float('inf')
        
        try:
            # Вычисляем двунаправленное расстояние Хаусдорфа
            h1 = directed_hausdorff(pred_coords, gt_coords)[0]
            h2 = directed_hausdorff(gt_coords, pred_coords)[0]
            hausdorff_dist = max(h1, h2)
        except:
            # Fallback если scipy недоступен
            hausdorff_dist = float('inf')
        
        return float(hausdorff_dist)
    
    @staticmethod
    def calculate_pixel_accuracy(pred_mask: np.ndarray, 
                                 gt_mask: np.ndarray, 
                                 threshold: float = 0.5) -> float:
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
        if pred_mask.max() > 1:
            pred_binary = (pred_mask > threshold * 255).astype(np.uint8)
            gt_binary = (gt_mask > threshold * 255).astype(np.uint8)
        else:
            pred_binary = (pred_mask > threshold).astype(np.uint8)
            gt_binary = (gt_mask > threshold).astype(np.uint8)
        
        # Вычисляем точность
        correct_pixels = (pred_binary == gt_binary).sum()
        total_pixels = pred_binary.size
        
        accuracy = correct_pixels / total_pixels
        return float(accuracy)
    
    @staticmethod
    def calculate_all_metrics(pred_mask: np.ndarray, 
                              gt_mask: np.ndarray, 
                              threshold: float = 0.5,
                              include_hausdorff: bool = True) -> Dict[str, float]:
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
        
        # Основные метрики
        metrics['iou'] = SegmentationMetrics.calculate_iou(pred_mask, gt_mask, threshold)
        metrics['dice'] = SegmentationMetrics.calculate_dice_coefficient(pred_mask, gt_mask, threshold)
        metrics['precision'], metrics['recall'] = SegmentationMetrics.calculate_precision_recall(
            pred_mask, gt_mask, threshold
        )
        metrics['f1_score'] = SegmentationMetrics.calculate_f1_score(pred_mask, gt_mask, threshold)
        metrics['pixel_accuracy'] = SegmentationMetrics.calculate_pixel_accuracy(
            pred_mask, gt_mask, threshold
        )
        metrics['mae'] = SegmentationMetrics.calculate_mae(pred_mask, gt_mask)
        
        # Расстояние Хаусдорфа (может быть вычислительно затратным)
        if include_hausdorff:
            metrics['hausdorff_distance'] = SegmentationMetrics.calculate_hausdorff_distance(
                pred_mask, gt_mask, threshold
            )
        
        # Дополнительные статистики
        if pred_mask.max() > 1:
            pred_area = (pred_mask > threshold * 255).sum()
            gt_area = (gt_mask > threshold * 255).sum()
        else:
            pred_area = (pred_mask > threshold).sum()
            gt_area = (gt_mask > threshold).sum()
        
        metrics['predicted_area'] = float(pred_area)
        metrics['ground_truth_area'] = float(gt_area)
        metrics['area_difference'] = float(abs(pred_area - gt_area))
        
        return metrics
    
    @staticmethod
    def evaluate_multiple_masks(pred_masks: List[np.ndarray],
                                gt_masks: List[np.ndarray],
                                threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
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
            raise ValueError("Количество предсказанных и ground truth масок должно совпадать")
        
        all_metrics = []
        individual_results = {}
        
        for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
            metrics = SegmentationMetrics.calculate_all_metrics(
                pred_mask, gt_mask, threshold, include_hausdorff=False
            )
            all_metrics.append(metrics)
            individual_results[f'mask_{i}'] = metrics
        
        # Вычисляем средние метрики
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                avg_metrics[f'avg_{key}'] = float(np.mean(values))
                avg_metrics[f'std_{key}'] = float(np.std(values))
                avg_metrics[f'min_{key}'] = float(np.min(values))
                avg_metrics[f'max_{key}'] = float(np.max(values))
        
        return {
            'individual_results': individual_results,
            'average_metrics': avg_metrics,
            'total_masks': len(pred_masks)
        }