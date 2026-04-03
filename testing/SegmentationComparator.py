# testing/SegmentationComparator.py

# Импорт основных библиотек
import os
import time
import warnings
from PIL import Image
import requests
from io import BytesIO
from typing import (
    List, Union, Tuple, Dict, Set, Any, TypeVar, Optional, 
    Literal, Protocol, runtime_checkable, overload, TYPE_CHECKING
)
from datetime import datetime
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from scipy import ndimage as ndi
import traceback
from metrics.SegmentationMetrics import SegmentationMetrics

class SegmentationComparator:
    """
    Класс для сравнительного тестирования сегментационных методов.
    Использует готовые реализации из реализованных сегментаторов (OpenCVSegmenter, TorchSegmenter, SklearnSegmenter)
    для валидации кастомных реализаций.
    """
    def __init__(self) -> None:
        self.results = {}
    
    # ============ МЕТРИКИ КАЧЕСТВА ============
    
    def compute_metrics(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray,
        method1_name: str = "Method1",
        method2_name: str = "Method2"
    ) -> Dict[str, float]:
        """
        Вычисляет метрики сходства между двумя масками, используя общий модуль.
        
        Args:
            mask1: Первая маска (предсказание или референс 1)
            mask2: Вторая маска (референс или предсказание 2)
            method1_name: Имя первого метода (для логов, если нужно)
            method2_name: Имя второго метода
            
        Returns:
            Dict[str, float]: Словарь с метриками. 
            Ключи адаптированы для удобного чтения в отчетах компаратора.
        """
        # Бинаризируем маски
        raw_metrics: Dict[str, float] = SegmentationMetrics.calculate_all_metrics(
           pred_mask=mask1,
           gt_mask=mask2,
           threshold=0.5,
           include_hausdorff=True 
        )
        
        metrics: Dict[str, float] = {
            'accuracy': raw_metrics.get('pixel_accuracy', 0.0),
            'precision': raw_metrics.get('precision', 0.0),
            'recall': raw_metrics.get('recall', 0.0),
            'f1_score': raw_metrics.get('f1_score', 0.0),
            'jaccard': raw_metrics.get('iou', 0.0),
            'dice_coefficient': raw_metrics.get('dice', 0.0),
            'intersection_over_union': raw_metrics.get('iou', 0.0),
            'pixel_agreement': raw_metrics.get('pixel_accuracy', 0.0),
            
            # Площади
            'area1': raw_metrics.get('predicted_area', 0.0),
            'area2': raw_metrics.get('ground_truth_area', 0.0),
            'area_difference': raw_metrics.get('area_difference', 0.0),
            'area_ratio': raw_metrics.get('area_ratio', 0.0),
            
            # Хаусдорф
            'hausdorff_distance': raw_metrics.get('hausdorff_distance', 0.0),
            
            # Матрица ошибок
            'true_positive': raw_metrics.get('true_positive', 0),
            'false_positive': raw_metrics.get('false_positive', 0),
            'false_negative': raw_metrics.get('false_negative', 0),
            'true_negative': raw_metrics.get('true_negative', 0),
        }
        return metrics
    
    def compare_methods(
        self,
        image: np.ndarray,
        segmenter1: Any,
        segmenter2: Any,
        name1: Optional[str] = None,
        name2: Optional[str] = None,
        save_comparison: bool = True,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Сравнивает две реализации методов сегментации, используя переданные объекты.
        
        Args:
            image: Входное изображение (numpy array)
            segmenter1: Экземпляр первого сегментера
            segmenter2: Экземпляр второго сегментера
            name1: Имя первого метода (для отчета). Если None, берется из объекта.
            name2: Имя второго метода.
            save_comparison: Сохранять ли визуализацию
            output_path: Путь для сохранения
        
        Returns:
            Dict[str, Any]: Результаты сравнения
        """
        m1_name = name1 or getattr(segmenter1, 'method', 'Method1')
        m2_name = name2 or getattr(segmenter2, 'method', 'Method2')
        
        start_time_1 = time.time()
        mask1 = segmenter1.segment(image)
        time_1 = time.time() - start_time_1
        info1 = {'execution_time': time_1, 'method': m1_name}
        
        start_time_2 = time.time()
        mask2 = segmenter2.segment(image)
        time_2 = time.time() - start_time_2
        info2 = {'execution_time': time_2, 'method': m2_name}
        
        metrics: Dict[str, float] = self.compute_metrics(mask1, mask2, m1_name, m2_name)
        
        result_key = f"{m1_name}_vs_{m2_name}"
        self.results[result_key] = {
            'mask1': mask1,
            'mask2': mask2,
            'info1': info1,
            'info2': info2,
            'metrics': metrics
        }
        
        if save_comparison:
            self.visualize_comparison(
                image, mask1, mask2,
                info1, info2, metrics,
                method1_name=m1_name,
                method2_name=m2_name,
                output_path=output_path
            )
        return self.results[result_key]
    
    def visualize_comparison(
        self,
        image: np.ndarray,
        mask1: np.ndarray,
        mask2: np.ndarray,
        info1: Dict[str, Any],
        info2: Dict[str, Any],
        metrics: Dict[str, float],
        method1_name: str = "Method 1",
        method2_name: str = "Method 2",
        output_path: str = None
    ) -> None:
        """
        Визуализирует сравнение двух методов.
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        if len(image.shape) == 2:
            axes[0, 0].imshow(image, cmap='gray')
        else:
            axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask1, cmap='gray')
        time1 = info1.get('execution_time', 0)
        axes[0, 1].set_title(f"{method1_name}\nTime: {time1:.3f}s")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(mask2, cmap='gray')
        time2 = info2.get('execution_time', 0)
        axes[0, 2].set_title(f"{method2_name}\nTime: {time2:.3f}s")
        axes[0, 2].axis('off')
        
        # Разность масок
        diff = np.abs(mask1.astype(float) - mask2.astype(float))
        axes[0, 3].imshow(diff, cmap='hot')
        axes[0, 3].set_title("Difference")
        axes[0, 3].axis('off')
        
        # Наложение масок на изображение
        if len(image.shape) == 2:
            overlay1 = np.stack([image] * 3, axis=-1)
            overlay2 = np.stack([image] * 3, axis=-1)
        else:
            overlay1 = image.copy()
            overlay2 = image.copy()
        
        overlay1[mask1 > 127] = [255, 0, 0]  # Красный
        overlay2[mask2 > 127] = [0, 255, 0]  # Зеленый
        
        axes[1, 0].imshow(overlay1)
        axes[1, 0].set_title(f"{method1_name} Overlay")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(overlay2)
        axes[1, 1].set_title(f"{method2_name} Overlay")
        axes[1, 1].axis('off')
        
        # Комбинированное наложение
        combined = image.copy() if len(image.shape) == 3 else np.stack([image] * 3, axis=-1)
        combined[mask1 > 127] = [255, 0, 0]
        combined[mask2 > 127] = [0, 255, 0]
        
        # Желтый для пересечения
        intersection = (mask1 > 127) & (mask2 > 127)
        combined[intersection] = [255, 255, 0]
        
        axes[1, 2].imshow(combined)
        axes[1, 2].set_title("Combined Overlay\n(Red: Method1, Green: Method2, Yellow: Both)")
        axes[1, 2].axis('off')
        
        # Текстовые метрики
        axes[1, 3].axis('off')
        text_str = (
            f"Metrics Comparison:\n"
            f"IoU (Jaccard): {metrics.get('jaccard', 0):.3f}\n"
            f"Dice Coeff:    {metrics.get('dice_coefficient', 0):.3f}\n"
            f"F1-Score:      {metrics.get('f1_score', 0):.3f}\n"
            f"Accuracy:      {metrics.get('accuracy', 0):.3f}\n"
            f"Precision:     {metrics.get('precision', 0):.3f}\n"
            f"Recall:        {metrics.get('recall', 0):.3f}\n"
            f"Hausdorff:     {metrics.get('hausdorff_distance', 0):.2f}\n"
            f"Area Diff:     {metrics.get('area_difference', 0):.0f}"
        )
        axes[1, 3].text(0.1, 0.6, text_str, fontsize=10,
                       verticalalignment='center',
                       transform=axes[1, 3].transAxes)
        
        plt.suptitle(f"Segmentation Methods Comparison: {method1_name} vs {method2_name}", fontsize=14)
        plt.tight_layout()
        if output_path:
            plt.savefig(
                output_path, 
                dpi=150, 
                bbox_inches='tight')
            print(f"✅ Визуализация сохранена: {output_path}")
        plt.show()
    
    def batch_comparison(
        self,
        image: np.ndarray,
        methods_config: List[Dict[str, Any]],
        reference_segmenter: Any,
        reference_name: Optional[str] = None,
        save_results: bool = True,
        output_dir: str = "./data/comparison_results"
    ) -> pd.DataFrame:
        """
        Пакетное сравнение нескольких методов с референсным.
        
        Args:
            image: Входное изображение
            methods_config: Список словарей вида {"name": "...", "segmenter": obj}
            reference_segmenter: Объект референсного сегментера
            reference_name: Имя референсного метода (опционально)
            save_results: Сохранять ли результаты
            output_dir: Директория для сохранения
        
        Returns:
            pd.DataFrame: DataFrame с результатами сравнения
        """
        
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        ref_name = reference_name or getattr(reference_segmenter, 'method', 'Reference')
        start_time_ref = time.time()
        ref_mask = reference_segmenter.segment(image)
        ref_time = time.time() - start_time_ref
        ref_info = {'execution_time': ref_time, 'method': ref_name}
        
        comparison_results = []
        
        for config in methods_config:
            segmenter = config.get('segmenter')
            method_name = config.get('name') or getattr(segmenter, 'method', 'Unknown')
            if segmenter is None:
                print(f"⚠️ Пропущен конфиг без сегментера: {config}")
                continue
            try:
                start_time_test = time.time()
                test_mask = segmenter.segment(image)
                test_time = time.time() - start_time_test
                test_info = {'execution_time': test_time, 'method': method_name}
                
                # Вычисляем метрики
                metrics = self.compute_metrics(ref_mask, test_mask, 
                                             f"Ref_{ref_name}",
                                             f"Test_{method_name}")
                
                # Сохраняем результаты
                result = {
                    'method': method_name,
                    **metrics,
                    'test_time': test_info.get('execution_time', 0),
                    'ref_time': ref_info.get('execution_time', 0),
                    'parameters': str(getattr(segmenter, 'params', {}))
                }
                
                comparison_results.append(result)
                
                # Сохраняем визуализацию
                if save_results:
                    output_path = os.path.join(output_dir, f"comparison_{method_name}.jpg")
                    self.visualize_comparison(
                        image, ref_mask, test_mask,
                        ref_info, test_info, metrics,
                        method1_name=f"Reference: {ref_name}",
                        method2_name=f"Test: {method_name}",
                        output_path=output_path
                    )
                
                print(f"✅ Сравнение {method_name}: F1={metrics.get('f1_score', 0):.3f}, IoU={metrics.get('jaccard', 0):.3f}")
                
            except Exception as e:
                print(f"❌ Ошибка при тестировании {method_name}: {e}")
                traceback.print_exc()
                continue
    
        df = pd.DataFrame(comparison_results)
        if save_results and not df.empty:
            csv_path = os.path.join(output_dir, "comparison_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"📊 Результаты сохранены в CSV: {csv_path}")
            self._create_summary_visualization(df, output_dir)
        return df
    
    def _create_summary_visualization(
        self,
        df: pd.DataFrame,
        output_dir: str
    ) -> None:
        """Создает сводную визуализацию результатов сравнения."""
        if df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # График 1: Метрики качества
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'jaccard']
        available_metrics = [m for m in metrics_to_plot if m in df.columns]
        
        if available_metrics:
            metrics_data = df[metrics_to_plot].mean()
            axes[0, 0].bar(range(len(metrics_data)), metrics_data.values)
            axes[0, 0].set_xticks(range(len(metrics_data)))
            axes[0, 0].set_xticklabels(metrics_data.index, rotation=45)
            axes[0, 0].set_title('Average Metrics')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_ylim(0, 1)
        
        # График 2: Время выполнения
        if 'test_time' in df.columns and 'ref_time' in df.columns:
            methods = df['method'].tolist()
            test_times = df['test_time'].tolist()
            ref_time = df['ref_time'].iloc[0] if len(df) > 0 else 0
            
            x = np.arange(len(methods))
            width = 0.35
            
            axes[0, 1].bar(x - width/2, test_times, width, label='Test Methods')
            axes[0, 1].bar(x[-1] + width/2, ref_time, width, label='Reference', alpha=0.7)
            axes[0, 1].set_xlabel('Methods')
            axes[0, 1].set_ylabel('Execution Time (s)')
            axes[0, 1].set_title('Execution Time Comparison')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(methods, rotation=45)
            axes[0, 1].legend()
        
        # График 3: Площадь масок
        area_cols = [col for col in df.columns if 'area' in col.lower() and 'difference' not in col.lower()]
        if len(area_cols) >= 2:
            area_data = df[area_cols].mean()
            axes[1, 0].bar(range(len(area_data)), area_data.values)
            axes[1, 0].set_xticks(range(len(area_data)))
            axes[1, 0].set_xticklabels(area_data.index, rotation=45)
            axes[1, 0].set_title('Average Mask Areas')
            axes[1, 0].set_ylabel('Pixels')
        
        # График 4: Корреляционная матрица метрик
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 1].set_title('Correlation Matrix')
            axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
            axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
            axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=90, fontsize=8)
            axes[1, 1].set_yticklabels(corr_matrix.columns, fontsize=8)
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.suptitle('Segmentation Methods Comparison Summary', fontsize=16)
        plt.tight_layout()
        summary_path = os.path.join(output_dir, "comparison_summary.jpg")
        plt.savefig(
            summary_path, 
            dpi=150, 
            bbox_inches='tight')
        plt.close()
        print(f"📈 Сводная визуализация сохранена: {summary_path}")

    def matrix_comparison(
        self,
        image: np.ndarray,
        methods_config: List[Dict[str, Any]], # Список словарей вида {"name": "...", "segmenter": obj}
        reference_method: Optional[str] = None,
        comparison_type: str = "all_vs_all",
        save_results: bool = True,
        output_dir: str = "./data/matrix_comparison_results"
    ) -> Dict[str, Any]:
        """
        Матричное сравнение всех методов между собой.
        """
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir: str = os.path.join(output_dir, f"comparison_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
        
        method_names = []
        segmenters_map = {} # Маппинг: Имя -> Объект сегментер
        for config in methods_config:
            name = config.get('name')
            segmenter = config.get('segmenter')
            if name is None or segmenter is None:
                print(f"⚠️ Пропущен конфиг без имени или сегментера: {config}")
                continue 
            method_names.append(name)
            segmenters_map[name] = segmenter
        
        print(f"Выполняем сегментацию {len(method_names)} методами...")
        masks = {}
        execution_times = {}
        method_infos = {}
        
        for name in method_names:
            segmenter = segmenters_map[name]
            try:
                start_time = time.time()
                mask = segmenter.segment(image)
                exec_time = time.time() - start_time
                
                masks[name] = mask
                execution_times[name] = exec_time
                method_infos[name] = {'execution_time': exec_time}
                
                print(f"  ✅ {name}: {exec_time:.3f}s")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
                import traceback
                traceback.print_exc()
                h, w = image.shape[:2] if len(image.shape) >= 2 else (256, 256)
                masks[name] = np.zeros((h, w), dtype=np.uint8)
                execution_times[name] = 0
                method_infos[name] = {'error': str(e)}
        
        if comparison_type == "all_vs_ref" and reference_method:
            comparison_pairs = [(reference_method, other) for other in method_names 
                            if other != reference_method]
            ref_name = reference_method
        elif comparison_type == "pairwise":
            comparison_pairs = list(itertools.combinations(method_names, 2))
            ref_name = None
        else:  # "all_vs_all"
            comparison_pairs = [(m1, m2) for m1 in method_names 
                            for m2 in method_names]
            ref_name = None
        
        print(f"\nВыполняем сравнение {len(comparison_pairs)} пар...")
        comparison_results = []
        for i, (method1, method2) in enumerate(comparison_pairs):
            if method1 not in masks or method2 not in masks:
                continue
            mask1 = masks[method1]
            mask2 = masks[method2]
            try:
                metrics = self.compute_metrics(mask1, mask2, method1, method2)
                result = {
                    'method1': method1,
                    'method2': method2,
                    **metrics,
                    'time1': execution_times.get(method1, 0),
                    'time2': execution_times.get(method2, 0),
                    'time_diff': abs(execution_times.get(method1, 0) - 
                                execution_times.get(method2, 0))
                }
                comparison_results.append(result)
                if (i + 1) % 10 == 0:
                    print(f"  Обработано {i + 1}/{len(comparison_pairs)} пар...")      
            except Exception as e:
                print(f"  Ошибка сравнения {method1} vs {method2}: {e}")
        
        df_comparisons = pd.DataFrame(comparison_results)
        if save_results:
            masks_dir = os.path.join(output_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            for name, mask in masks.items():
                mask_path = os.path.join(masks_dir, f"{name}_mask.png")
                plt.imsave(mask_path, mask, cmap='gray')
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            if len(image.shape) == 2:
                plt.imsave(os.path.join(images_dir, "original.png"), image, cmap='gray')
            else:
                plt.imsave(os.path.join(images_dir, "original.png"), image)
            for name, mask in masks.items():
                if len(image.shape) == 2:
                    overlay = np.stack([image] * 3, axis=-1)
                else:
                    overlay = image.copy()
                overlay[mask > 127] = [255, 0, 0]
                overlay_path = os.path.join(images_dir, f"{name}_overlay.png")
                plt.imsave(overlay_path, overlay)
            
            self._save_matrix_results(df_comparisons, masks, method_infos, 
                                    output_dir, comparison_type, ref_name)
        return {
            'df_comparisons': df_comparisons,
            'masks': masks,
            'execution_times': execution_times,
            'method_infos': method_infos
        }
        
    def _save_matrix_results(
        self,
        df_comparisons: pd.DataFrame,
        masks: Dict[str, np.ndarray],
        method_infos: Dict[str, Any],
        output_dir: str,
        comparison_type: str,
        reference_method: Optional[str] = None
    ):
        """Сохраняет результаты матричного сравнения."""
        csv_path = os.path.join(output_dir, "comparisons.csv")
        df_comparisons.to_csv(csv_path, index=False)
        print(f"📊 CSV с результатами: {csv_path}")
        
        # Сводная таблица метрик
        summary_metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score', 'jaccard', 'dice_coefficient', 
                           'intersection_over_union', 'pixel_agreement', 'area_difference']
        
        if comparison_type == "all_vs_ref" and reference_method:
            # Средние метрики по сравнению с референсом
            ref_comparisons = df_comparisons[df_comparisons['method1'] == reference_method]
            if not ref_comparisons.empty:
                summary_df = ref_comparisons[['method2'] + summary_metrics].copy()
                summary_df = summary_df.rename(columns={'method2': 'method'})
                summary_df = summary_df.sort_values('f1_score', ascending=False)
                
                summary_path = os.path.join(output_dir, "summary_vs_ref.csv")
                summary_df.to_csv(summary_path, index=False)
                
                print(f"📋 Сводная таблица (vs {reference_method}): {summary_path}")
        
        # Матрицы сравнения
        methods = sorted(list(masks.keys()))
        n_methods = len(methods)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'jaccard', 'dice_coefficient', 
                           'intersection_over_union', 'pixel_agreement', 'area_difference']:
            if metric not in df_comparisons.columns:
                continue
            
            # Создаем матрицу N x N
            matrix = np.zeros((n_methods, n_methods))
            
            for i, m1 in enumerate(methods):
                for j, m2 in enumerate(methods):
                    if i == j:
                        matrix[i, j] = 1.0
                    else:
                        mask = ((df_comparisons['method1'] == m1) & 
                               (df_comparisons['method2'] == m2)) | \
                               ((df_comparisons['method1'] == m2) & 
                               (df_comparisons['method2'] == m1))
                        
                        if mask.any():
                            matrix[i, j] = df_comparisons.loc[mask, metric].values[0]
                        else:
                            matrix[i, j] = np.nan
            
            fig, ax = plt.subplots(figsize=(12, 10))
            short_names = [name[:15] + "..." if len(name) > 15 else name 
                          for name in methods]
            
            im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_xticks(np.arange(n_methods))
            ax.set_yticks(np.arange(n_methods))
            ax.set_xticklabels(short_names, rotation=45, ha='right')
            ax.set_yticklabels(short_names)
            for i in range(n_methods):
                for j in range(n_methods):
                    if not np.isnan(matrix[i, j]):
                        text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                                     ha="center", va="center", 
                                     color="black" if matrix[i, j] < 0.7 else "white",
                                     fontsize=8)
            
            ax.set_title(f"Матрица сравнения: {metric.upper()}", fontsize=14)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            
            matrix_path = os.path.join(output_dir, f"{metric}_matrix.png")
            plt.savefig(
                matrix_path, 
                dpi=150, 
                bbox_inches='tight')
            plt.close()
            
            print(f"📈 Матрица {metric}: {matrix_path}")
        
        # Визуализация всех масок
        self._visualize_all_masks(masks, output_dir)
        
        # Создаем HTML отчет
        self._create_html_report(df_comparisons, masks, method_infos, 
                                output_dir, comparison_type, reference_method)
    
    def _visualize_all_masks(
        self,
        masks: Dict[str, np.ndarray],
        output_dir: str
    ) -> None:
        """Визуализирует все маски в одной фигуре."""
        methods = list(masks.keys())
        n_methods = len(methods)
        n_cols = 4
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
        axes = axes.flatten()
        
        for i, (name, mask) in enumerate(masks.items()):
            ax = axes[i]
            ax.imshow(mask, cmap='gray')
            ax.set_title(f"{name}", fontsize=10)
            ax.axis('off')
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle("Все маски сегментации", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        all_masks_path = os.path.join(output_dir, "all_masks.png")
        plt.savefig(
            all_masks_path, 
            dpi=150, 
            bbox_inches='tight'
        )
        plt.close()
        
        print(f"🖼️ Все маски: {all_masks_path}")
    
    def _create_html_report(
        self,
        df_comparisons: pd.DataFrame,
        masks: Dict[str, np.ndarray],
        method_infos: Dict[str, Any],
        output_dir: str,
        comparison_type: str,
        reference_method: Optional[str] = None
    ) -> None:
        """Создает HTML отчет с результатами."""
        html_path = os.path.join(output_dir, "report.html")
        methods_stats = []
        for name, mask in masks.items():
            mask_binary = mask > 127
            area = np.sum(mask_binary)
            total_pixels = mask.size
            coverage = area / total_pixels * 100
            
            methods_stats.append({
                'method': name,
                'area': area,
                'coverage': f"{coverage:.3f}%",
                'pixels': f"{area:,}",
                'time': method_infos.get(name, {}).get('execution_time', 0)
            })
        
        # Топ методов по F1
        top_methods_html = "<p>Нет данных</p>"
        if reference_method and 'f1_score' in df_comparisons.columns:
            ref_df = df_comparisons[df_comparisons['method1'] == reference_method]
            if not ref_df.empty:
                top_methods = ref_df.nlargest(5, 'f1_score')[['method2', 'f1_score']]
                top_methods_html = top_methods.to_html(index=False, 
                                                      float_format=lambda x: f"{x:.3f}")
        else:
            top_methods_html = "<p>Сравнение всех со всеми</p>"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Отчет сравнения методов сегментации</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                    .metric-card {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                    .highlight {{ background-color: #e6f7ff; }}
                </style>
            </head>
            <body>
                <h1>📊 Отчет сравнения методов сегментации</h1>
                
                <div class="summary">
                    <h2>Общая информация</h2>
                    <p><strong>Дата:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Всего методов:</strong> {len(masks)}</p>
                    <p><strong>Тип сравнения:</strong> {comparison_type}</p>
                    <p><strong>Референсный метод:</strong> {reference_method if reference_method else 'Нет (все со всеми)'}</p>
                </div>
                
                <h2>📈 Матрицы сравнения</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>F1-Score матрица</h3>
                        <img src="f1_score_matrix.png" alt="F1 Matrix">
                    </div>
                    <div class="metric-card">
                        <h3>Accuracy матрица</h3>
                        <img src="accuracy_matrix.png" alt="Accuracy Matrix">
                    </div>
                    <div class="metric-card">
                        <h3>Все маски</h3>
                        <img src="all_masks.png" alt="All Masks">
                    </div>
                </div>
                
                <h2>🏆 Топ методов</h2>
                {top_methods_html}
                
                <h2>📋 Статистика методов</h2>
                <table>
                    <tr>
                        <th>Метод</th>
                        <th>Площадь маски</th>
                        <th>Покрытие</th>
                        <th>Время (с)</th>
                    </tr>
            """)
            
            for stat in sorted(methods_stats, key=lambda x: x['area'], reverse=True):
                f.write(f"""
                    <tr>
                        <td>{stat['method']}</td>
                        <td>{stat['pixels']}</td>
                        <td>{stat['coverage']}</td>
                        <td>{stat['time']:.3f}</td>
                    </tr>
                """)
            
            f.write("""
                </table>
                
                <h2>🔗 Быстрые ссылки</h2>
                <ul>
                    <li><a href="comparisons.csv">CSV с результатами сравнения</a></li>
                    <li><a href="masks/">Папка с масками</a></li>
                    <li><a href="images/">Папка с изображениями</a></li>
                </ul>
                
                <footer>
                    <p>Сгенерировано автоматически с помощью SegmentationComparator</p>
                </footer>
            </body>
            </html>
            """)
        
        print(f"📄 HTML отчет: {html_path}")