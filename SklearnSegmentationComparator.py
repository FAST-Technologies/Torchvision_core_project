# SklearnSegmentationComparator.py

# Импорт основных библиотек

from SklearnSegmenter import SklearnSegmenter
from OpenCVSegmenter import OpenCVSegmenter

import os
import time
import warnings

import requests
from io import BytesIO
from PIL import Image
from typing import (
    List, Union, Tuple, Dict, Any, TypeVar, Optional, 
    Literal, Protocol, runtime_checkable, overload, TYPE_CHECKING
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    jaccard_score, confusion_matrix
)

class SklearnSegmentationComparator:
    """
    Класс для сравнения SklearnSegmenter с другими реализациями.
    """
    
    def __init__(self):
        self.results = {}
        self.metrics_history = {}
        self.sklearn_segmenter = None
    
    def compare_with_opencv(self, 
                          image: np.ndarray,
                          sklearn_method: str,
                          opencv_method: str,
                          sklearn_params: Dict = None,
                          opencv_params: Dict = None,
                          save_comparison: bool = True,
                          output_path: str = None) -> Dict[str, Any]:
        """
        Сравнение SklearnSegmenter с OpenCVSegmenter.
        
        Args:
            image: Входное изображение
            sklearn_method: Метод для SklearnSegmenter
            opencv_method: Метод для OpenCVSegmenter
            sklearn_params: Параметры SklearnSegmenter
            opencv_params: Параметры OpenCVSegmenter
            save_comparison: Сохранять визуализацию
            output_path: Путь для сохранения
        
        Returns:
            Результаты сравнения
        """
        sklearn_params = sklearn_params or {}
        opencv_params = opencv_params or {}
        
        # Sklearn сегментация
        sklearn_segmenter = SklearnSegmenter(method=sklearn_method, **sklearn_params)
        start_time = time.time()
        sklearn_mask = sklearn_segmenter.segment(image)
        sklearn_time = time.time() - start_time
        
        # OpenCV сегментация
        # opencv_segmenter = OpenCVSegmenter(method=opencv_method, **opencv_params)
        # start_time = time.time()
        # opencv_mask = opencv_segmenter.segment(image)
        # opencv_time = time.time() - start_time
        
        # Временная заглушка - используем KMeans из OpenCV для сравнения
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        start_time = time.time()
        _, opencv_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        opencv_time = time.time() - start_time
        
        # Вычисляем метрики
        metrics = self._compute_metrics(sklearn_mask, opencv_mask)
        
        # Сохраняем результаты
        result = {
            'sklearn_mask': sklearn_mask,
            'opencv_mask': opencv_mask,
            'sklearn_time': sklearn_time,
            'opencv_time': opencv_time,
            'metrics': metrics,
            'sklearn_method': sklearn_method,
            'opencv_method': opencv_method,
            'sklearn_params': sklearn_params,
            'opencv_params': opencv_params
        }
        
        key = f"sklearn_{sklearn_method}_vs_opencv_{opencv_method}"
        self.results[key] = result
        
        # Визуализация
        if save_comparison:
            self._visualize_comparison(image, result, output_path)
        
        return result
    
    def compare_sklearn_methods(self,
                              image: np.ndarray,
                              methods: List[Tuple[str, Dict]] = None,
                              reference_method: str = "kmeans",
                              save_results: bool = True,
                              output_dir: str = "./data/sklearn_comparison") -> pd.DataFrame:
        """
        Сравнение нескольких методов SklearnSegmenter между собой.
        
        Args:
            image: Входное изображение
            methods: Список методов и параметров
            reference_method: Референсный метод
            save_results: Сохранять результаты
            output_dir: Директория для сохранения
        
        Returns:
            DataFrame с результатами сравнения
        """
        if methods is None:
            methods = [
                ("kmeans", {"n_clusters": 3}),
                ("dbscan", {"eps": "auto", "min_samples": "auto"}),
                ("meanshift", {"bandwidth": None}),
                ("gmm", {"n_components": 3}),
                ("random_forest", {"n_estimators": 50}),
                ("svm", {"C": 1.0}),
                ("isolation_forest", {"n_estimators": 100}),
            ]
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Получаем референсную маску
        ref_segmenter = SklearnSegmenter(method=reference_method)
        ref_mask = ref_segmenter.segment(image)
        
        results = []
        
        for method_name, params in methods:
            try:
                print(f"Тестирование метода: {method_name}")
                
                # Создаем и запускаем сегментатор
                segmenter = SklearnSegmenter(method=method_name, **params)
                start_time = time.time()
                mask = segmenter.segment(image)
                exec_time = time.time() - start_time
                
                # Вычисляем метрики относительно референса
                metrics = self._compute_metrics(ref_mask, mask)
                
                # Дополнительные метрики качества
                unique_labels = np.unique(mask)
                n_regions = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
                
                result = {
                    'method': method_name,
                    'parameters': str(params),
                    'execution_time': exec_time,
                    'n_regions': n_regions,
                    'mask_mean': float(mask.mean()),
                    'mask_std': float(mask.std()),
                    **metrics
                }
                
                results.append(result)
                
                # Сохраняем маску
                if save_results:
                    mask_path = os.path.join(output_dir, f"{method_name}_mask.jpg")
                    cv2.imwrite(mask_path, mask)
                    
                    # Визуализация
                    self._save_method_visualization(image, mask, method_name, 
                                                   metrics, exec_time, output_dir)
                
                print(f"  ✓ Время: {exec_time:.3f}s, F1: {metrics['f1_score']:.3f}")
                
            except Exception as e:
                print(f"  ✗ Ошибка: {e}")
                continue
        
        # Создаем DataFrame
        df = pd.DataFrame(results)
        
        if save_results and not df.empty:
            # Сохраняем CSV
            csv_path = os.path.join(output_dir, "sklearn_methods_comparison.csv")
            df.to_csv(csv_path, index=False)
            
            # Создаем сводный график
            self._create_summary_plot(df, output_dir)
        
        return df
    
    def _compute_metrics(self, mask1: np.ndarray, mask2: np.ndarray) -> Dict[str, float]:
        """Вычисление метрик сравнения двух масок."""
        # Бинаризуем маски
        mask1_bin = (mask1 > 127).astype(np.uint8).flatten()
        mask2_bin = (mask2 > 127).astype(np.uint8).flatten()
        
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(mask1_bin, mask2_bin)
            metrics['precision'] = precision_score(mask1_bin, mask2_bin, zero_division=0)
            metrics['recall'] = recall_score(mask1_bin, mask2_bin, zero_division=0)
            metrics['f1_score'] = f1_score(mask1_bin, mask2_bin, zero_division=0)
            metrics['jaccard'] = jaccard_score(mask1_bin, mask2_bin, zero_division=0)
        except Exception as e:
            warnings.warn(f"Error computing metrics: {e}")
            metrics.update({
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'jaccard': 0.0
            })
        
        # Дополнительные метрики
        intersection = np.sum(mask1_bin & mask2_bin)
        union = np.sum(mask1_bin | mask2_bin)
        
        metrics['dice'] = (2 * intersection) / (np.sum(mask1_bin) + np.sum(mask2_bin) + 1e-8)
        metrics['iou'] = intersection / (union + 1e-8)
        
        return metrics
    
    def _visualize_comparison(self, 
                            image: np.ndarray,
                            result: Dict[str, Any],
                            output_path: str = None):
        """Визуализация сравнения методов."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Оригинальное изображение
        if len(image.shape) == 2:
            axes[0, 0].imshow(image, cmap='gray')
        else:
            axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Sklearn маска
        axes[0, 1].imshow(result['sklearn_mask'], cmap='gray')
        axes[0, 1].set_title(f"Sklearn: {result['sklearn_method']}\n"
                           f"Time: {result['sklearn_time']:.3f}s")
        axes[0, 1].axis('off')
        
        # OpenCV маска
        axes[0, 2].imshow(result['opencv_mask'], cmap='gray')
        axes[0, 2].set_title(f"OpenCV: {result['opencv_method']}\n"
                           f"Time: {result['opencv_time']:.3f}s")
        axes[0, 2].axis('off')
        
        # Разность масок
        diff = np.abs(result['sklearn_mask'].astype(float) - 
                     result['opencv_mask'].astype(float))
        axes[1, 0].imshow(diff, cmap='hot')
        axes[1, 0].set_title("Absolute Difference")
        axes[1, 0].axis('off')
        
        # Наложение Sklearn
        if len(image.shape) == 2:
            overlay_sklearn = np.stack([image] * 3, axis=-1)
        else:
            overlay_sklearn = image.copy()
        
        overlay_sklearn[result['sklearn_mask'] > 127] = [255, 0, 0]
        axes[1, 1].imshow(overlay_sklearn)
        axes[1, 1].set_title("Sklearn Overlay")
        axes[1, 1].axis('off')
        
        # Метрики
        axes[1, 2].axis('off')
        metrics = result['metrics']
        metrics_text = (f"Comparison Metrics:\n"
                       f"F1-Score: {metrics['f1_score']:.3f}\n"
                       f"Accuracy: {metrics['accuracy']:.3f}\n"
                       f"Precision: {metrics['precision']:.3f}\n"
                       f"Recall: {metrics['recall']:.3f}\n"
                       f"Jaccard: {metrics['jaccard']:.3f}\n"
                       f"Dice: {metrics['dice']:.3f}\n"
                       f"IoU: {metrics['iou']:.3f}\n"
                       f"Time Ratio: {result['sklearn_time']/result['opencv_time']:.2f}x")
        
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=10,
                       verticalalignment='center',
                       transform=axes[1, 2].transAxes)
        
        plt.suptitle(f"Sklearn vs OpenCV Segmentation Comparison", fontsize=14)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Визуализация сохранена: {output_path}")
        
        plt.show()
    
    def _save_method_visualization(self,
                                 image: np.ndarray,
                                 mask: np.ndarray,
                                 method_name: str,
                                 metrics: Dict[str, float],
                                 exec_time: float,
                                 output_dir: str):
        """Сохранение визуализации для отдельного метода."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Оригинальное изображение
        if len(image.shape) == 2:
            axes[0].imshow(image, cmap='gray')
        else:
            axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Маска
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f"{method_name}\nTime: {exec_time:.3f}s")
        axes[1].axis('off')
        
        # Наложение
        if len(image.shape) == 2:
            overlay = np.stack([image] * 3, axis=-1)
        else:
            overlay = image.copy()
        
        overlay[mask > 127] = [255, 0, 0]
        axes[2].imshow(overlay)
        axes[2].set_title("Segmentation Overlay")
        axes[2].axis('off')
        
        plt.suptitle(f"SklearnSegmenter: {method_name}\n"
                    f"F1: {metrics['f1_score']:.3f}, Accuracy: {metrics['accuracy']:.3f}",
                    fontsize=12)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"{method_name}_visualization.jpg")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_summary_plot(self, df: pd.DataFrame, output_dir: str):
        """Создание сводного графика сравнения методов."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. F1-Score
        axes[0, 0].bar(df['method'], df['f1_score'])
        axes[0, 0].set_ylabel('F1-Score')
        axes[0, 0].set_title('F1-Score Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Время выполнения
        axes[0, 1].bar(df['method'], df['execution_time'])
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].set_title('Execution Time')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Accuracy
        axes[1, 0].bar(df['method'], df['accuracy'])
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Accuracy Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Количество регионов
        axes[1, 1].bar(df['method'], df['n_regions'])
        axes[1, 1].set_ylabel('Number of Regions')
        axes[1, 1].set_title('Detected Regions')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('SklearnSegmenter Methods Comparison Summary', fontsize=14)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, "summary_plot.jpg")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# Пример использования
def demo_comparator():
    """Демонстрация работы компаратора."""
    
    # Загрузка тестового изображения
    url = "https://upload.wikimedia.org/wikipedia/commons/7/7d/Colorful_spring_garden.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_np = np.array(img)
    
    print("Демонстрация SklearnSegmentationComparator")
    print("=" * 60)
    
    # Создаем компаратор
    comparator = SklearnSegmentationComparator()
    
    # Сравнение Sklearn методов между собой
    print("\n1. Сравнение методов SklearnSegmenter:")
    df_results = comparator.compare_sklearn_methods(
        img_np,
        output_dir="sklearn_comparison_results"
    )
    
    if not df_results.empty:
        print("\nРезультаты сравнения:")
        print(df_results[['method', 'f1_score', 'execution_time', 'n_regions']])
        
        # Выводим лучшие методы
        print("\nТоп-3 метода по F1-Score:")
        print(df_results.nlargest(3, 'f1_score')[['method', 'f1_score', 'execution_time']])
        
        print("\nТоп-3 метода по скорости:")
        print(df_results.nsmallest(3, 'execution_time')[['method', 'execution_time', 'f1_score']])
    
    # Сравнение с OpenCV (если доступно)
    print("\n2. Сравнение Sklearn KMeans с OpenCV Otsu:")
    try:
        comparison_result = comparator.compare_with_opencv(
            img_np,
            sklearn_method="kmeans",
            opencv_method="otsu",
            sklearn_params={"n_clusters": 3},
            opencv_params={},
            output_path="./data/sklearn_vs_opencv.jpg"
        )
        
        print(f"  F1-Score: {comparison_result['metrics']['f1_score']:.3f}")
        print(f"  Sklearn time: {comparison_result['sklearn_time']:.3f}s")
        print(f"  OpenCV time: {comparison_result['opencv_time']:.3f}s")
        print(f"  Time ratio: {comparison_result['sklearn_time']/comparison_result['opencv_time']:.2f}x")
        
    except Exception as e:
        print(f"  Ошибка при сравнении: {e}")
    
    return comparator, df_results


if __name__ == "__main__":
    comparator, results = demo_comparator()
    print("\n✅ Демонстрация компаратора завершена!")