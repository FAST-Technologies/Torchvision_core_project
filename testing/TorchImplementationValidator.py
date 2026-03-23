# testing/TorchImplementationValidator.py

# Импорт основных библиотек
from segmenters.TorchSegmenter import TorchSegmenter
from segmenters.SklearnSegmenter import SklearnSegmenter
from segmenters.OpenCVSegmenter import OpenCVSegmenter
from metrics.SegmentationMetrics import SegmentationMetrics
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback
import time
from datetime import datetime
from typing import (
    List, Union, Tuple, Dict, Any, TypeVar, Optional, 
    Literal, Protocol, runtime_checkable, overload, TYPE_CHECKING
)
from PIL import Image

class TorchImplementationValidator:
    """
    Класс для валидации кастомных PyTorch реализаций
    против оригинальных реализаций из библиотек
    """
    def __init__(
        self, 
        output_dir: str = "./data/validation_results"
    ) -> None:
        self.output_dir: str = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.validation_results: Dict[str, Any] = {}
        self.threshold_methods: List[Tuple[str, Dict[str, Any]]] = [
            ("global_thresholding", {"threshold": 0.5}),
            ("otsu_thresholding", {}),
            ("adaptive_thresholding", {"block_size": 11, "C": 2}),
            ("threshold_niblack", {"window_size": 15, "k": -0.2}),
            ("threshold_sauvola", {"window_size": 15, "k": 0.5, "r": 128}),
        ]
        
        self.edge_methods: List[Tuple[str, Dict[str, Any]]] = [
            ("sobel_edge", {"threshold": 0.1}),
            ("canny_edge", {"low": 0.1, "high": 0.3, "sigma": 1.0}),
        ]

        self.region_methods: List[Tuple[str, Dict[str, Any]]] = [
            ("region_growing", {'tolerance': 0.1}),
            ("split_and_merge", {'min_size': 50, 'threshold': 20}),
            ("floodfill", {'tolerance': 0.15}),
        ]

        self.clastering_methods: List[Tuple[str, Dict[str, Any]]] = [
            ("kmeans_segmentation", {'k': 3}),
            ("dbscan_segmentation", {'eps': 0.1, 'min_samples': 10}),
            ("meanshift", {'bandwidth': 0.5, 'spatial_radius': 35, 'color_radius': 60, 'max_level': 1}),
        ]

        self.active_contour_methods: List[Tuple[str, Dict[str, Any]]] = [
            ("active_contour", {'alpha': 0.015, 'beta': 10, 'gamma': 0.001, 'max_iterations': 2000, 'w_edge': 1, 'w_line': 0}),
            ("gvf_contour", {'mu': 0.1, 'iterations': 50}),
            ("morphological_snakes", {'iterations': 100, 'smoothing': 1, 'threshold': 0.5}),
            ("chan_vese", {'mu': 0.25, 'lambda1': 1.0, 'lambda2': 1.0, 'tol': 1e-3, 'max_iter': 100, 'dt': 0.5, 'eps': 1.0, 'init_level_set': 'checkerboard'}),
        ]

        self.watershed_methods: List[Tuple[str, Dict[str, Any]]] = [
            ("watershed", {}),
            ("random_walker", {'beta': 130, 'tol': 1e-3, 'max_iter': 300, 'target_label': 2}),
        ]

        self.super_pixel_methods: List[Tuple[str, Dict[str, Any]]] = [
            # ("quickshift", {'kernel_size': 5, 'max_dist': 10, 'ratio': 1.0, 'sigma': 0.0, 'convert2lab': True}),
            ("slic", {'n_segments': 100, 'compactness': 10.0, 'max_iter': 10, 'sigma': 0.0, 'enforce_connectivity': True, 'min_size_factor': 0.5, 'max_size_factor': 3.0, 'ruler': 10.0, 'region_size': 20}),
            ("felzenszwalb", {'scale': 100, 'sigma': 0.5, 'min_size': 50}),
        ]

        self.interactive_methods: List[Tuple[str, Dict[str, Any]]] = [
            ("grabcut", {'num_iterations': 5}),
        ]
        
        # Пороги успешности валидации
        self.success_thresholds: Dict[str, float] = {
            'iou': 0.85,             # IoU > 0.85
            'dice': 0.90,            # Dice > 0.90
            'pixel_accuracy': 0.95,  # Pixel Accuracy > 0.95
            'precision': 0.9,        # Precision > 0.9
            'recall': 0.9,           # Recall > 0.9
            'f1_score': 0.9,         # F1 Score > 0.9
            'mae': 0.1               # MAE < 0.1
        }
    
    def _load_image(
        self, 
        image_path: str
    ) -> np.ndarray:
        """
        Универсальная загрузка изображения для всех сегментаторов.
        Возвращает numpy array в формате RGB.
        """
        if isinstance(image_path, str) and os.path.exists(image_path):
            img = Image.open(image_path).convert('RGB')
            return np.array(img)
        elif isinstance(image_path, np.ndarray):
            return image_path
        elif isinstance(image_path, Image.Image):
            return np.array(image_path.convert('RGB'))
        else:
            raise ValueError(f"Неподдерживаемый тип изображения: {type(image_path)}")
        
    def validate_segmentation_methods(
        self,
        image_path: str,
        methods_list: List[str],
        torch_segmenter_class: type = TorchSegmenter,
        reference_segmenter_class: type = SklearnSegmenter,
        reference: str = "sklearn",
        status_message: str = "ВАЛИДАЦИЯ ПОРОГОВЫХ МЕТОДОВ",
        prefix: str = "threshold_validation",
        validation_type: str = "threshold",
        additional_method: str = "Torch"
    ) -> Dict[str, Any]:
        """
        Универсальная функция валидации методов сегментации
        
        Args:
            image_path: Путь к изображению или numpy array
            methods_list: Список методов для валидации (например, self.threshold_methods)
            torch_segmenter_class: Класс сегментера для Torch реализации
            reference_segmenter_class: Класс сегментера для референсной реализации
            reference: Название референсной библиотеки (для отчётов)
            status_message: Заголовок для вывода в консоль
            prefix: Префикс для сохранения результатов
            validation_type: Тип валидации для визуализации
        
        Returns:
            Dict с результатами валидации по каждому методу
        """
        print(f"\n{'='*60}")
        print(f"{status_message}")
        print(f"Референс: {reference.upper()}")
        print(f"{'='*60}")
        results = {}
        img_array: np.ndarray = self._load_image(image_path)
        
        for (method_name, params) in methods_list:
            print(f"\n📊 Метод: {method_name}")
            try:
                # Torch реализация
                start_method_1_time = time.time()
                torch_segmenter = torch_segmenter_class(method=method_name, **params)
                torch_mask = torch_segmenter.segment(img_array, **params)
                execution_method_1_time = time.time() - start_method_1_time
                
                # Референсная реализация
                start_method_2_time = time.time()
                ref_params = params.copy()
                ref_params['postprocess'] = False
                ref_segmenter = reference_segmenter_class(method=method_name, **ref_params)
                ref_mask = ref_segmenter.segment(img_array, **ref_params)
                execution_method_2_time = time.time() - start_method_2_time

                difference_methods_time = abs(execution_method_2_time - execution_method_1_time)
                
                # Вычисляем метрики соответствия
                metrics = SegmentationMetrics.calculate_all_metrics(
                    torch_mask, ref_mask, threshold=0.5, include_hausdorff=True
                )

                metrics["first_method_time"] = execution_method_1_time
                metrics["second_method_time"] = execution_method_2_time
                metrics["methods_time_difference"] = difference_methods_time
                
                # Определяем статус валидации
                validation_status = self._check_validation_status(metrics)
                
                results[method_name] = {
                    'torch_mask': torch_mask,
                    'reference_mask': ref_mask,
                    'metrics': metrics,
                    'parameters': params,
                    'validation_status': validation_status,
                    'success': True,
                    'reference_library': reference,
                    'first_method_time': execution_method_1_time,
                    'second_method_time': execution_method_2_time,
                    'methods_time_difference': difference_methods_time,
                }
                
                # Вывод результатов
                status_icon = "✅" if validation_status == "PASS" else "⚠️"
                print(f"   {status_icon} IoU: {metrics['iou']:.4f}")
                print(f"   {status_icon} Dice: {metrics['dice']:.4f}")
                print(f"   {status_icon} Precision: {metrics['precision']:.4f}")
                print(f"   {status_icon} Recall: {metrics['recall']:.4f}")
                print(f"   {status_icon} F1-Score: {metrics['f1_score']:.4f}")
                print(f"   {status_icon} MAE: {metrics['mae']:.4f}")
                print(f"   {status_icon} Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
                print(f"   Hausdorf distance: {metrics['hausdorff_distance']:.4f}")

                print(f"   Predicted Area: {metrics['predicted_area']:.4f}")
                print(f"   Ground Truth Area: {metrics['ground_truth_area']:.4f}")
                print(f"   Area Difference: {metrics['area_difference']:.4f}")
                print(f"   Статус: {validation_status}")

                print(f"    ✅ Время первого метода {execution_method_1_time:.3f}s")
                print(f"    ✅ Время второго метода {execution_method_2_time:.3f}s")
                print(f"    ✅ Разница по времени {difference_methods_time:.3f}s")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                traceback.print_exc()
                results[method_name] = {
                    'success': False,
                    'error': str(e),
                    'reference_library': reference
                }
        if additional_method=="Torch":
            self._save_validation_results(results, prefix, reference, flag_torch=True)
        else:
            self._save_validation_results(results, prefix, reference, flag_torch=False)
        self._visualize_validation(results, img_array, validation_type, reference, additional_method)
        return results
    
    def _check_validation_status(
        self, 
        metrics: Dict[str, float]
    ) -> str:
        """Определяет статус валидации на основе метрик"""
        passed = 0
        total = 7
        
        if metrics['iou'] >= self.success_thresholds['iou']:
            passed += 1
        if metrics['dice'] >= self.success_thresholds['dice']:
            passed += 1
        if metrics['pixel_accuracy'] >= self.success_thresholds['pixel_accuracy']:
            passed += 1
        if metrics['precision'] >= self.success_thresholds['precision']:
            passed += 1
        if metrics['recall'] >= self.success_thresholds['recall']:
            passed += 1
        if metrics['f1_score'] >= self.success_thresholds['f1_score']:
            passed += 1
        if metrics['mae'] <= self.success_thresholds['mae']:
            passed += 1
        
        if passed == total:
            return "PASS"
        elif passed >= total // 2:
            return "WARNING"
        else:
            return "FAIL"
    
    def _save_validation_results(
        self, 
        results: Dict[str, Any], 
        prefix: str, 
        reference: str, 
        flag_torch: bool = True
    ) -> None:
        """Сохранение результатов валидации"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.output_dir, f"{prefix}_{reference}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Сохраняем маски и метрики
        for method, data in results.items():
            if data.get('success'):
                method_dir = os.path.join(results_dir, method)
                os.makedirs(method_dir, exist_ok=True)
                
                # Torch маска
                if flag_torch == True:
                    torch_mask = data['torch_mask']
                    np.save(os.path.join(method_dir, "torch_mask.npy"), torch_mask)
                else:
                    opencv_mask = data['torch_mask']
                    np.save(os.path.join(method_dir, "opencv_mask.npy"), opencv_mask)
                
                # Reference маска
                ref_mask = data['reference_mask']
                np.save(os.path.join(method_dir, "reference_mask.npy"), ref_mask)
                
                # Метрики
                metrics = data['metrics']
                metrics_path = os.path.join(method_dir, "metrics.txt")
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    f.write(f"Результаты валидации: {method}\n")
                    f.write(f"Референсная библиотека: {reference}\n")
                    f.write(f"Статус: {data['validation_status']}\n")
                    f.write("="*50 + "\n")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.6f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
        
        print(f"\n💾 Результаты сохранены: {results_dir}")
    
    def _visualize_validation(
        self,
        results: Dict[str, Any],
        image_array: np.ndarray,
        validation_type: str,
        reference: str,
        additional_method: str = "Torch"
    ) -> None:
        """Визуализация результатов валидации"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        original = image_array
        
        n_methods = len([r for r in results.values() if r.get('success')])
        if n_methods == 0:
            print("⚠️ Нет успешных результатов для визуализации")
            return
        
        fig, axes = plt.subplots(n_methods, 4, figsize=(20, 5*n_methods))
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        
        row = 0
        for method, data in results.items():
            if not data.get('success'):
                continue
            
            torch_mask = data['torch_mask']
            ref_mask = data['reference_mask']
            if isinstance(torch_mask, torch.Tensor):
                torch_mask_np = torch_mask.cpu().numpy()
            else:
                torch_mask_np = torch_mask

            # Удаляем все оси размером 1, чтобы получить (H, W)
            torch_mask_np = np.squeeze(torch_mask_np)
            
            if isinstance(ref_mask, torch.Tensor):
                ref_mask_np = ref_mask.cpu().numpy()
            else:
                ref_mask_np = ref_mask
            ref_mask_np = np.squeeze(ref_mask_np)

            metrics = data['metrics']
            status = data['validation_status']
            
            # Оригинальное изображение
            axes[row, 0].imshow(original)
            axes[row, 0].set_title(f"Original Image")
            axes[row, 0].axis('off')
            
            # Torch маска
            axes[row, 1].imshow(torch_mask_np, cmap='gray')
            if additional_method == "Torch":
                axes[row, 1].set_title(f"Torch {method}\nIoU: {metrics['iou']:.3f}")
            else:
                axes[row, 1].set_title(f"OpenCV {method}\nIoU: {metrics['iou']:.3f}")
            axes[row, 1].axis('off')
            
            # Reference маска
            axes[row, 2].imshow(ref_mask_np, cmap='gray')
            axes[row, 2].set_title(f"{reference.upper()} {method}")
            axes[row, 2].axis('off')
            
            # Разность
            diff = np.abs(torch_mask_np.astype(float) - ref_mask_np.astype(float))
            im = axes[row, 3].imshow(diff, cmap='hot')
            status_color = 'green' if status == 'PASS' else 'orange' if status == 'WARNING' else 'red'
            axes[row, 3].set_title(f"Difference\nStatus: {status}", color=status_color)
            axes[row, 3].axis('off')
            plt.colorbar(im, ax=axes[row, 3], fraction=0.046)
            row += 1
        
        plt.suptitle(f"{validation_type.title()} Validation ({additional_method} vs {reference.upper()})", fontsize=16)
        plt.tight_layout()
        
        viz_path = os.path.join(
            self.output_dir,
            f"{validation_type}_validation_{reference}_{timestamp}.jpg"
        )
        plt.savefig(
            viz_path, 
            dpi=150, 
            bbox_inches='tight')
        plt.close()
        
        print(f"📊 Визуализация: {viz_path}")
    
    def generate_validation_report(
        self, 
        all_results: Dict[str, Any]
    ) -> str:
        """Генерация сводного отчёта по валидации"""
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("ОТЧЁТ ПО ВАЛИДАЦИИ TORCH РЕАЛИЗАЦИЙ")
        report_lines.append("="*60)
        report_lines.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        total_methods = 0
        passed_methods = 0
        warning_methods = 0
        failed_methods = 0
        
        for validation_type, results in all_results.items():
            report_lines.append(f"\n{validation_type.upper()}")
            report_lines.append("-"*40)
            
            for method, data in results.items():
                if not data.get('success'):
                    continue
                
                total_methods += 1
                status = data['validation_status']
                
                if status == "PASS":
                    passed_methods += 1
                    icon = "✅"
                elif status == "WARNING":
                    warning_methods += 1
                    icon = "⚠️"
                else:
                    failed_methods += 1
                    icon = "❌"
                
                metrics = data['metrics']
                report_lines.append(
                    f"{icon} {method}: "
                    f"Accuracy={metrics['accuracy']:.3f}, "
                    f"IoU={metrics['iou']:.3f}, "
                    f"Dice={metrics['dice']:.3f}, "
                    f"Precision={metrics['precision']:.3f}, "
                    f"Recall={metrics['recall']:.3f}, "
                    f"F1_Score={metrics['f1_score']:.3f}, "
                    f"Pixel_accuracy={metrics['pixel_accuracy']:.3f} "
                    f"MAE={metrics['mae']:.3f} "
                    f"Hausdorf_distance={metrics['mae']:.3f} "
                    f"Area_ratio={metrics['area_ratio']:.3f} "
                    f"Area_difference={metrics['area_difference']:.3f} "
                    f"Segmenter_1_time={metrics['first_method_time']:.3f} "
                    f"Segmenter_2_time={metrics['second_method_time']:.3f} "
                    f"Segmenter_difference_time={metrics['methods_time_difference']:.3f} "
                    f"[{status}]"
                )

        report_lines.append("")
        report_lines.append("="*60)
        report_lines.append("СВОДНАЯ СТАТИСТИКА")
        report_lines.append("="*60)
        report_lines.append(f"Всего методов: {total_methods}")
        
        if total_methods > 0:
            report_lines.append(f"✅ PASS: {passed_methods} ({passed_methods/total_methods*100:.2f}%)")
            report_lines.append(f"⚠️ WARNING: {warning_methods} ({warning_methods/total_methods*100:.2f}%)")
            report_lines.append(f"❌ FAIL: {failed_methods} ({failed_methods/total_methods*100:.2f}%)")
        else:
            report_lines.append("⚠️ Нет данных для статистики (все методы не прошли)")
        report_lines.append("="*60)
        report = "\n".join(report_lines)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"validation_report_{timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📄 Отчёт сохранён: {report_path}")
        print("\n" + report)
        return report
    
    def validate_all_methods(self, image_path: str) -> Dict:
        """Валидация всех методов одним вызовом"""
        all_results = {}
        validation_configs = [
            ('threshold_sklearn', self.threshold_methods, TorchSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ ПОРОГОВЫХ МЕТОДОВ (Torch + Sklearn)", 'sklearn', 'threshold', 'Torch'),
            ('threshold_opencv', self.threshold_methods, TorchSegmenter, OpenCVSegmenter, "ВАЛИДАЦИЯ ПОРОГОВЫХ МЕТОДОВ (Torch + OpenCV)", 'opencv', 'threshold', 'Torch'),
            ('threshold_custom', self.threshold_methods, OpenCVSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ ПОРОГОВЫХ МЕТОДОВ (Sklearn + OpenCV)", 'sklearn', 'threshold', 'OpenCV'),
            ('edge_sklearn', self.edge_methods, TorchSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ ОПЕРАТОРОВ ГРАНИЦ (Torch + OpenCV)", 'sklearn', 'edge', 'Torch'),
            ('edge_opencv', self.edge_methods, TorchSegmenter, OpenCVSegmenter, "ВАЛИДАЦИЯ ОПЕРАТОРОВ ГРАНИЦ (Torch + Sklearn)", 'opencv', 'edge', 'Torch'),
            ('edge_custom', self.edge_methods, OpenCVSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ ОПЕРАТОРОВ ГРАНИЦ (Sklearn + OpenCV)", 'sklearn', 'edge', 'OpenCV'),
            # ('region_sklearn', self.region_methods, TorchSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ РЕГИОНАЛЬНЫХ МЕТОДОВ (Torch + Sklearn)", 'sklearn', 'region', 'Torch'),
            # ('region_opencv', self.region_methods, TorchSegmenter, OpenCVSegmenter, "ВАЛИДАЦИЯ РЕГИОНАЛЬНЫХ МЕТОДОВ (Torch + OpenCV)", 'opencv', 'region', 'Torch'),
            # ('region_custom', self.region_methods, OpenCVSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ РЕГИОНАЛЬНЫХ МЕТОДОВ (Sklearn + OpenCV)", 'sklearn', 'region', 'OpenCV'),
            # ('clastering_sklearn', self.clastering_methods, TorchSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ КЛАСТЕРИЗАЦИИ (Torch + Sklearn)", 'sklearn', 'claster', 'Torch'),
            # ('clastering_opencv', self.clastering_methods, TorchSegmenter, OpenCVSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ КЛАСТЕРИЗАЦИИ (Torch + OpenCV)", 'opencv', 'claster', 'Torch'),
            # ('clastering_custom', self.clastering_methods, OpenCVSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ КЛАСТЕРИЗАЦИИ (Sklearn + OpenCV)", 'sklearn', 'claster', 'OpenCV'),
            # ('active_contour_sklearn', self.active_contour_methods, TorchSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ АКТИВНЫХ КОНТУРОВ (Torch + Sklearn)", 'sklearn', 'active_contour', 'Torch'),
            # ('active_contour_opencv', self.active_contour_methods, TorchSegmenter, OpenCVSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ АКТИВНЫХ КОНТУРОВ (Torch + OpenCV)", 'opencv', 'active_contour', 'Torch'),
            # ('active_contour_custom', self.active_contour_methods, OpenCVSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ АКТИВНЫХ КОНТУРОВ (Sklearn + OpenCV)", 'sklearn', 'active_contour', 'OpenCV'),
            # ('watershed_sklearn', self.watershed_methods, TorchSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ ВОДОРАЗДЕЛА (Torch + Sklearn)", 'sklearn', 'watershed', 'Torch'),
            # ('watershed_opencv', self.watershed_methods, TorchSegmenter, OpenCVSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ ВОДОРАЗДЕЛА (Torch + OpenCV)", 'opencv', 'watershed', 'Torch'),
            # ('watershed_custom', self.watershed_methods, OpenCVSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ ВОДОРАЗДЕЛА (Sklearn + OpenCV)", 'sklearn', 'watershed', 'OpenCV'),
            # ('super_pixel_sklearn', self.super_pixel_methods, TorchSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ СУПЕРПИКСЕЛЬНЫХ МЕТОДОВ (Torch + Sklearn)", 'sklearn', 'super_pixel', 'Torch'),
            # ('super_pixel_opencv', self.super_pixel_methods, TorchSegmenter, OpenCVSegmenter, "ВАЛИДАЦИЯ СУПЕРПИКСЕЛЬНЫХ МЕТОДОВ (Torch + OpenCV)", 'opencv', 'super_pixel', 'Torch'),
            # ('super_pixel_custom', self.super_pixel_methods, OpenCVSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ СУПЕРПИКСЕЛЬНЫХ МЕТОДОВ (Sklearn + OpenCV)", 'sklearn', 'super_pixel', 'OpenCV'),
            # ('interactive_sklearn', self.interactive_methods, TorchSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ ИНТЕРАКТИВНЫХ МЕТОДОВ (Torch + Sklearn)", 'sklearn', 'interactive', 'Torch'),
            # ('interactive_opencv', self.interactive_methods, TorchSegmenter, OpenCVSegmenter, "ВАЛИДАЦИЯ ИНТЕРАКТИВНЫХ МЕТОДОВ (Torch + OpenCV)", 'opencv', 'interactive', 'Torch'),
            # ('interactive_custom', self.interactive_methods, OpenCVSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ ИНТЕРАКТИВНЫХ МЕТОДОВ (Sklearn + OpenCV)", 'sklearn', 'interactive', 'OpenCV'),
        ]
        
        for key, methods, base_class, ref_class, message, reference, v_type, additional_method in validation_configs:
            all_results[key] = self.validate_segmentation_methods(
                image_path=image_path,
                methods_list=methods,
                torch_segmenter_class=base_class,
                reference_segmenter_class=ref_class,
                reference=reference,
                status_message=message,
                prefix=f"{v_type}_validation",
                validation_type=v_type,
                additional_method=additional_method
            )
        return all_results