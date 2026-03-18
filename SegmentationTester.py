# SegmentationTester.py

# Импорт основных библиотек

from BaseSegmenter import BaseSegmenter
from SegmentationMetrics import SegmentationMetrics

import os
import time
import json
from datetime import datetime
from PIL import Image
from typing import (
    List, Union, Tuple, Dict, Any, TypeVar, Optional, 
    Literal, Protocol, runtime_checkable, overload, TYPE_CHECKING
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from sklearn.metrics import confusion_matrix

class SegmentationTester:
    """Класс для тестирования и сравнения методов сегментации"""
    def __init__(
        self,
        base_output_dir: str = "./data/segmentation_results",
        ground_truth_path: Optional[str] = None
    ) -> None:
        self.methods = {}
        self.results = {}
        self.base_output_dir: str = base_output_dir
        self.current_test_id: str = None
        self.ground_truth_path: str = ground_truth_path
        self.ground_truth_mask: Optional[np.ndarray] = None
        
        if ground_truth_path:
            self.load_ground_truth(ground_truth_path)

    def load_ground_truth(
        self, 
        gt_path: str
    ) -> None:
        """Загрузка ground truth маски"""
        try:
            if gt_path.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.ground_truth_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                print(f"✅ Ground truth загружен: {gt_path}")
            elif gt_path.endswith('.npy'):
                self.ground_truth_mask = np.load(gt_path)
                print(f"✅ Ground truth загружен: {gt_path}")
            else:
                raise ValueError(f"Неизвестный формат ground truth: {gt_path}")
        except Exception as e:
            print(f"❌ Ошибка загрузки ground truth: {e}")
            self.ground_truth_mask = None

    def _create_test_directory(
        self, 
        test_name: str = None
    ) -> str:
        """Создает уникальную директорию для теста"""
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir: str
        if test_name:
            test_dir = f"{test_name}_{timestamp}"
        else:
            test_dir = f"test_{timestamp}"
        
        full_path: str = os.path.join(self.base_output_dir, test_dir)
        os.makedirs(full_path, exist_ok=True)
        
        # Создаем поддиректории
        os.makedirs(os.path.join(full_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "masks"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "comparisons"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "statistics"), exist_ok=True)
        
        self.current_test_id: str = test_dir
        print(f"📁 Создана директория для теста: {full_path}")
        return full_path
    
    def add_method(
        self, 
        name: str, 
        segmenter: BaseSegmenter
    ) -> None:
        """Добавление метода сегментации"""
        self.methods[name] = segmenter
    
    def test_single_method(
        self, 
        image: Union[str, np.ndarray, Image.Image], 
        method_name: str, 
        save_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Тестирование одного метода с сохранением результатов"""
        if method_name not in self.methods:
            raise ValueError(f"Метод {method_name} не найден")
        
        segmenter = self.methods[method_name]
        
        # Измеряем время выполнения
        start_time: float = time.time()
        result: np.ndarray
        mask: np.ndarray
        result, mask = segmenter.segment_with_mask(image)
        execution_time: float = time.time() - start_time

        # Получаем оригинальное изображение для сохранения
        original_img: Image.Image
        img_array: np.ndarray
        if isinstance(image, str):
            original_img = Image.open(image).convert('RGB')
            img_array = np.array(original_img)
        elif isinstance(image, Image.Image):
            original_img = image
            img_array = np.array(image)
        else:
            img_array = image
            original_img = Image.fromarray(image.astype(np.uint8))
        
        # Статистика
        mask_area = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        result_data: Dict[str, Any] = {
            'method': method_name,
            'result': result,
            'mask': mask,
            'time': execution_time,
            'mask_area': mask_area,
            'mask_percentage': (mask_area / total_pixels) * 100,
            'image_shape': result.shape,
            'timestamp': datetime.now().isoformat()
        }
        
        # Сохранение результатов
        if output_dir:
            # Сохраняем оригинальное изображение
            orig_path: str = os.path.join(output_dir, "images", "original.jpg")
            original_img.save(orig_path)
            
            # Сохраняем результат сегментации
            result_path: str = os.path.join(output_dir, "images", f"{method_name}_result.jpg")
            result_pil: Image.Image = Image.fromarray(result.astype(np.uint8))
            result_pil.save(result_path)
            
            # Сохраняем маску
            mask_path: str = os.path.join(output_dir, "masks", f"{method_name}_mask.png")
            mask_pil: Image.Image = Image.fromarray(mask.astype(np.uint8))
            mask_pil.save(mask_path)
            
            # Сохраняем наложение (overlay)
            try:
                # Создаем overlay (50% оригинал + 50% результат)
                # Создаем BRIGHT overlay (30% оригинал + 70% результат) - БОЛЬШЕ контраста!
                overlay_alpha = 0.7  # Яркость наложения
                original_alpha = 0.3  # Прозрачность оригинала
                
                # Если результат уже цветной (скорее всего, так и есть)
                overlay = (img_array * original_alpha + result * overlay_alpha).astype(np.uint8)
                overlay = overlay.astype(np.uint8)
                overlay_path: str = os.path.join(output_dir, "images", f"{method_name}_overlay.jpg")
                overlay_pil: Image.Image = Image.fromarray(overlay)
                overlay_pil.save(overlay_path)
                
                result_data['overlay_path'] = overlay_path
                bright_overlay = cv2.addWeighted(img_array, 0.1, result, 0.9, 0)
                bright_overlay_path: str = os.path.join(output_dir, "images", f"{method_name}_bright_overlay.jpg")
                Image.fromarray(bright_overlay.astype(np.uint8)).save(bright_overlay_path)
            except Exception as e:
                print(f"⚠️ Ошибка создания overlay для {method_name}: {e}")
                pass
            
            result_data['result_path'] = result_path
            result_data['mask_path'] = mask_path
            result_data['original_path'] = orig_path
            
            print(f"✅ {method_name}: сохранено в {output_dir}")
        
        elif save_path:
            result_pil = Image.fromarray(result.astype(np.uint8))
            result_pil.save(save_path)
            print(f"✅ Результат сохранен: {save_path}")
        
        print(f"   ⏱️ Время: {execution_time:.2f}s, 📏 Площадь: {result_data['mask_percentage']:.1f}%")
        
        return result_data
    
    def _save_overlay_image(
        self, 
        result_data: Dict[str, Any], 
        method_dir: str, 
        method_name: str
    ) -> None:
        """
        Сохраняет наложение маски на оригинальное изображение.
        """
        try:
            # Пытаемся создать overlay
            mask = result_data.get('mask')
            result_img = result_data.get('result')
            
            if mask is None or result_img is None:
                return
            
            # Конвертируем result_img в numpy если нужно
            if isinstance(result_img, Image.Image):
                result_np = np.array(result_img)
            else:
                result_np = result_img
            
            # Конвертируем маску в правильный формат
            if isinstance(mask, np.ndarray):
                mask_np = mask.copy()
                if mask_np.dtype != np.uint8:
                    if mask_np.max() <= 1.0:
                        mask_np = (mask_np * 255).astype(np.uint8)
                    else:
                        mask_np = mask_np.astype(np.uint8)
            else:
                return
            
            # Создаем overlay
            if len(result_np.shape) == 2:
                # Grayscale оригинал
                overlay = np.stack([result_np] * 3, axis=-1)
            else:
                # RGB оригинал
                overlay = result_np.copy()
            
            # Накладываем красную маску
            if mask_np.ndim == 2:
                mask_bool = mask_np > 127
                overlay[mask_bool] = [255, 0, 0]  # Красный
            
            # Сохраняем overlay
            overlay_path = os.path.join(method_dir, "overlay.jpg")
            Image.fromarray(overlay.astype(np.uint8)).save(overlay_path)
            
            # Также сохраняем прозрачное наложение
            alpha = 0.5
            if len(result_np.shape) == 2:
                result_colored = np.stack([result_np] * 3, axis=-1)
            else:
                result_colored = result_np
            
            transparent_overlay = result_colored.copy()
            if mask_np.ndim == 2:
                mask_bool = mask_np > 127
                transparent_overlay[mask_bool] = [255, 0, 0]  # Красный
                # Смешиваем с оригиналом
                blended = cv2.addWeighted(result_colored, 1 - alpha, 
                                        transparent_overlay, alpha, 0)
                
                blended_path = os.path.join(method_dir, "blended_overlay.jpg")
                Image.fromarray(blended.astype(np.uint8)).save(blended_path)
                
        except Exception as e:
            print(f"    ⚠️ Ошибка создания overlay для {method_name}: {e}")
    
    def _save_metrics_file(
        self, 
        result_data: Dict[str, Any], 
        method_dir: str, 
        method_name: str
    ) -> None:
        """
        Сохраняет метрики в JSON и текстовый файл.
        """
        metrics = result_data.get('metrics', {})
        
        if not metrics:
            return
        
        # JSON файл
        json_path = os.path.join(method_dir, "metrics.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        
        # Текстовый файл
        txt_path = os.path.join(method_dir, "metrics.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write(f"МЕТРИКИ СЕГМЕНТАЦИИ: {method_name}\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Время выполнения: {result_data.get('time', 0):.3f} секунд\n\n")
            
            if 'has_ground_truth' in result_data and result_data['has_ground_truth']:
                f.write("Метрики качества (с Ground Truth):\n")
                f.write("-"*50 + "\n")
                
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if 0 <= value <= 1:
                            f.write(f"{key:<20}: {value:.4f}\n")
                        else:
                            f.write(f"{key:<20}: {value}\n")
                    else:
                        f.write(f"{key:<20}: {value}\n")
            else:
                f.write("Метрики (без Ground Truth):\n")
                f.write("-"*50 + "\n")
                f.write(f"Площадь маски: {result_data.get('mask_area', 0):,} пикселей\n")
                f.write(f"Процент покрытия: {result_data.get('mask_percentage', 0):.1f}%\n")
        
        print(f"    📊 Метрики сохранены для {method_name}")
    
    def _save_method_info(
        self, 
        result_data: Dict[str, Any], 
        method_dir: str, 
        method_name: str
    ) -> None:
        """
        Сохраняет информацию о методе и параметрах.
        """
        try:
            segmenter = self.methods.get(method_name)
            
            if segmenter is None:
                return
            
            info_path = os.path.join(method_dir, "method_info.txt")
            
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write("="*50 + "\n")
                f.write(f"ИНФОРМАЦИЯ О МЕТОДЕ: {method_name}\n")
                f.write("="*50 + "\n\n")
                
                # Основная информация
                f.write(f"Имя метода: {method_name}\n")
                f.write(f"Тип сегментатора: {type(segmenter).__name__}\n")
                f.write(f"Время выполнения: {result_data.get('time', 0):.3f} секунд\n\n")
                
                # Параметры метода
                if hasattr(segmenter, 'method'):
                    f.write(f"Алгоритм: {segmenter.method}\n")
                
                if hasattr(segmenter, 'params') and segmenter.params:
                    f.write("\nПараметры метода:\n")
                    f.write("-"*30 + "\n")
                    for key, value in segmenter.params.items():
                        f.write(f"{key}: {value}\n")
                
                # Информация о маске
                mask = result_data.get('mask')
                if mask is not None:
                    f.write(f"\nИнформация о маске:\n")
                    f.write("-"*30 + "\n")
                    f.write(f"Размер: {mask.shape}\n")
                    f.write(f"Тип данных: {mask.dtype}\n")
                    f.write(f"Min значение: {mask.min()}\n")
                    f.write(f"Max значение: {mask.max()}\n")
                    
                    if hasattr(mask, 'size'):
                        mask_binary = mask > 127 if mask.max() > 1 else mask > 0.5
                        f.write(f"Площадь: {np.sum(mask_binary):,} пикселей\n")
                        f.write(f"Покрытие: {np.sum(mask_binary) / mask.size * 100:.1f}%\n")
                
                # Информация о ground truth
                if result_data.get('has_ground_truth', False):
                    f.write("\nGround Truth:\n")
                    f.write("-"*30 + "\n")
                    f.write("Метрики качества доступны в metrics.txt\n")
                else:
                    f.write("\nGround Truth:\n")
                    f.write("-"*30 + "\n")
                    f.write("Отсутствует\n")
                
        except Exception as e:
            print(f"    ⚠️ Ошибка сохранения информации о методе {method_name}: {e}")
    
    def _save_method_results(
        self, 
        result_data: Dict[str, Any], 
        output_dir: str, 
        method_name: str
    ) -> None:
        """
        Сохраняет результаты одного метода в указанную директорию.
        
        Args:
            result_data: Данные результатов
            output_dir: Базовая директория для сохранения
            method_name: Имя метода (для имен файлов)
        """
        # Создаем поддиректории
        method_dir = os.path.join(output_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)
        
        # Сохраняем изображение результата
        result_img = result_data.get('result')
        if result_img is not None:
            if isinstance(result_img, np.ndarray):
                result_path = os.path.join(method_dir, "result.jpg")
                if len(result_img.shape) == 2:
                    # Grayscale
                    Image.fromarray(result_img).save(result_path)
                else:
                    # RGB
                    Image.fromarray(result_img.astype(np.uint8)).save(result_path)
            elif isinstance(result_img, Image.Image):
                result_path = os.path.join(method_dir, "result.jpg")
                result_img.save(result_path)
        
        # Сохраняем маску
        mask = result_data.get('mask')
        if mask is not None and isinstance(mask, np.ndarray):
            mask_path = os.path.join(method_dir, "mask.png")
            
            # Нормализуем маску если нужно
            if mask.dtype != np.uint8:
                if mask.max() <= 1.0:
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)
            
            Image.fromarray(mask).save(mask_path)
        
        # Сохраняем overlay (наложение маски на оригинал)
        self._save_overlay_image(result_data, method_dir, method_name)
        
        # Сохраняем метрики
        if result_data.get('has_ground_truth', False):
            self._save_metrics_file(result_data, method_dir, method_name)
        
        # Сохраняем информацию о методе
        self._save_method_info(result_data, method_dir, method_name)
    
    def test_single_method_with_metrics(
        self, 
        image: Union[str, np.ndarray, Image.Image],
        method_name: str,
        ground_truth: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Тестирование метода с расчётом метрик
        
        Args:
            image: Входное изображение
            method_name: Название метода
            ground_truth: Ground truth маска (если None, используется загруженная)
            threshold: Порог для метрик
            output_dir: Директория для сохранения
            
        Returns:
            Результаты с метриками
        """
        if method_name not in self.methods:
            raise ValueError(f"Метод {method_name} не найден")
        
        # Используем ground truth если передан, иначе из класса
        gt_mask = ground_truth if ground_truth is not None else self.ground_truth_mask
        
        segmenter = self.methods[method_name]
        
        # Измеряем время выполнения
        start_time = time.time()
        
        if gt_mask is not None:
            # Если есть ground truth, вычисляем метрики
            metrics, pred_mask = segmenter.segment_and_evaluate(image, gt_mask, threshold)
            result_img, _ = segmenter.segment_with_mask(image)
            execution_time = time.time() - start_time
            
            result_data = {
                'method': method_name,
                'result': result_img,
                'mask': pred_mask,
                'time': execution_time,
                'metrics': metrics,
                'has_ground_truth': True
            }
        else:
            # Если нет ground truth, просто сегментируем
            result_img, pred_mask = segmenter.segment_with_mask(image)
            execution_time = time.time() - start_time
            
            # Базовые метрики без ground truth
            mask_area = np.sum(pred_mask > 0)
            total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
            
            result_data = {
                'method': method_name,
                'result': result_img,
                'mask': pred_mask,
                'time': execution_time,
                'mask_area': mask_area,
                'mask_percentage': (mask_area / total_pixels) * 100,
                'has_ground_truth': False
            }
        
        # Сохранение результатов
        if output_dir:
            self._save_method_results(result_data, output_dir, method_name)
        
        return result_data
    
    def _calculate_segmentation_metrics(
        self, 
        pred_mask: np.ndarray, 
        gt_mask: np.ndarray
    ) -> Dict[str, float]:
        """Вычисляет метрики качества сегментации"""
        # Бинаризация
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        try:
            tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat, labels=[0, 1]).ravel()
        except ValueError:
            # Если только один класс присутствует
            if np.all(pred_flat == 0) and np.all(gt_flat == 0):
                tn, fp, fn, tp = len(pred_flat), 0, 0, 0
            elif np.all(pred_flat == 1) and np.all(gt_flat == 1):
                tn, fp, fn, tp = 0, 0, 0, len(pred_flat)
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
        
        metrics = {}
        
        # Basic metrics
        total = tp + tn + fp + fn
        metrics['accuracy'] = (tp + tn) / (total + 1e-8)
        metrics['precision'] = tp / (tp + fp + 1e-8)
        metrics['recall'] = tp / (tp + fn + 1e-8)
        
        # F1 Score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                (metrics['precision'] + metrics['recall'] + 1e-8)
        else:
            metrics['f1_score'] = 0.0
        
        # IoU
        intersection = np.sum(pred_flat & gt_flat)
        union = np.sum(pred_flat | gt_flat)
        metrics['iou'] = intersection / (union + 1e-8)
        
        # Dice
        metrics['dice'] = (2 * intersection) / (np.sum(pred_flat) + np.sum(gt_flat) + 1e-8)
        
        # Pixel accuracy
        metrics['pixel_accuracy'] = np.sum(pred_flat == gt_flat) / len(pred_flat)
        
        # Area
        metrics['predicted_area'] = float(np.sum(pred_flat))
        metrics['ground_truth_area'] = float(np.sum(gt_flat))
        metrics['area_difference'] = abs(metrics['predicted_area'] - metrics['ground_truth_area'])
        
        return metrics
    
    def compare_methods(
        self, 
        image: Union[str, np.ndarray, Image.Image], 
        method_names: List[str] = None, 
        figsize: Tuple[int, int] = (20, 15),
        save_comparison: bool = True,
        test_name: str = None,
        show_plots: bool = True
    ) -> Dict[str, Any]:
        """Сравнение нескольких методов"""
        if method_names is None:
            method_names: List[str] = list(self.methods.keys())
        
        test_dir: str = self._create_test_directory(test_name)
        results = {}

        original_img: Image.Image
        image_path: str
        # Оригинальное изображение
        if isinstance(image, str):
            original_img = Image.open(image).convert('RGB')
            image_path = image
        elif isinstance(image, Image.Image):
            original_img = image
            image_path = None
        else:
            original_img = Image.fromarray(image.astype(np.uint8))
            image_path = None

        # Сохраняем оригинальное изображение в директории теста
        orig_save_path: str = os.path.join(test_dir, "images", "original.jpg")
        original_img.save(orig_save_path)
        print(f"📸 Оригинальное изображение сохранено: {orig_save_path}")
        
        # Создаем фигуру для отображения
        n_methods: int = len(method_names)
        n_cols: int = min(4, n_methods + 1)  # +1 для оригинального изображения
        n_rows: int = (n_methods + n_cols) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        all_stats = []
        # Тестируем каждый метод
        for i, method_name in enumerate(method_names, 1):
            if i >= len(axes):
                break
            
            try:
                result_data: Dict[str, Any] = self.test_single_method(
                    image, 
                    method_name, 
                    output_dir=test_dir
                )
                results[method_name] = result_data
                
                # Добавляем статистику
                stats: Dict[str, Any] = {
                    'method': method_name,
                    'time_seconds': result_data['time'],
                    'mask_area_pixels': result_data['mask_area'],
                    'mask_percentage': result_data['mask_percentage'],
                    'image_shape': result_data['image_shape']
                }
                all_stats.append(stats)
                
                # Отображение результата
                axes[i].imshow(result_data['result'])
                title: str = f"{method_name}\n{result_data['time']:.2f}s, {result_data['mask_percentage']:.1f}%"
                axes[i].set_title(title, fontsize=9)
                axes[i].axis('off')
                
                print(f"{method_name}: {result_data['time']:.2f}s, {result_data['mask_percentage']:.1f}% площади")
                
            except Exception as e:
                error_msg: str = str(e)[:50]
                axes[i].text(0.5, 0.5, f"Error:\n{error_msg}", 
                           ha='center', va='center', 
                           transform=axes[i].transAxes,
                           fontsize=8)
                axes[i].set_title(f"{method_name}\n(Error)", fontsize=9)
                axes[i].axis('off')
                
                print(f"❌ Ошибка в методе {method_name}: {e}")
                
                # Добавляем запись об ошибке в статистику
                stats: Dict[str, Any] = {
                    'method': method_name,
                    'error': error_msg,
                    'time_seconds': None,
                    'mask_area_pixels': None,
                    'mask_percentage': None
                }
                all_stats.append(stats)
        
        # Скрываем пустые оси
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle(f"Сравнение методов сегментации\n{self.current_test_id}", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Сохраняем сравнение
        if save_comparison:
            comparison_path: str = os.path.join(test_dir, "comparisons", "methods_comparison.jpg")
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            print(f"📊 Сравнительный график сохранен: {comparison_path}")
            
            # Сохраняем отдельно уменьшенную версию
            comparison_small_path: str = os.path.join(test_dir, "comparisons", "methods_comparison_small.jpg")
            plt.savefig(comparison_small_path, dpi=100, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Сохраняем статистику
        self._save_statistics(all_stats, test_dir)
        
        # Сохраняем результаты
        self._save_results_summary(results, test_dir)
        
        self.results[test_dir] = results
        print(f"✅ Тестирование завершено. Результаты в: {test_dir}")
        print(f"📋 Протестировано методов: {len(results)}/{len(method_names)}")
        
        return results

    
    def compare_methods_with_metrics(
        self,
        image: Union[str, np.ndarray, Image.Image],
        method_names: List[str] = None,
        ground_truth: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        figsize: Tuple[int, int] = (20, 15),
        test_name: str = None,
        show_plots: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Сравнение методов с метриками качества
        
        Args:
            image: Входное изображение
            method_names: Список методов для сравнения
            ground_truth: Ground truth маска
            threshold: Порог для метрик
            figsize: Размер фигуры
            test_name: Имя теста
            show_plots: Показывать графики
            
        Returns:
            Результаты всех методов с метриками
        """
        if method_names is None:
            method_names = list(self.methods.keys())
        
        test_dir = self._create_test_directory(test_name)
        results = {}
        
        gt_mask = ground_truth if ground_truth is not None else self.ground_truth_mask
        has_gt = gt_mask is not None
        
        print(f"Сравнение методов {'с' if has_gt else 'без'} ground truth")
        
        # Создаем фигуру для отображения
        n_methods = len(method_names)
        n_cols = min(4, n_methods + 1)
        n_rows = (n_methods + n_cols) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        # Оригинальное изображение
        original_img = Image.open(image).convert('RGB') if isinstance(image, str) else image
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        if has_gt:
            # Если есть ground truth, показываем его
            axes[1].imshow(gt_mask, cmap='gray')
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')
            start_idx = 2
        else:
            start_idx = 1
        
        all_metrics_data = []
        
        # Тестируем каждый метод
        for i, method_name in enumerate(method_names, start_idx):
            if i >= len(axes):
                break
            
            try:
                result_data = self.test_single_method_with_metrics(
                    image, method_name, gt_mask, threshold, test_dir
                )
                results[method_name] = result_data
                
                # Отображение результата
                axes[i].imshow(result_data['result'])
                
                if has_gt:
                    # Показываем метрики
                    metrics = result_data['metrics']
                    title = (f"{method_name}\n"
                            f"IoU: {metrics['iou']:.3f}, Dice: {metrics['dice']:.3f}\n"
                            f"F1: {metrics['f1_score']:.3f}, Acc: {metrics['pixel_accuracy']:.3f}")
                else:
                    # Показываем базовую информацию
                    title = (f"{method_name}\n"
                            f"Time: {result_data['time']:.2f}s\n"
                            f"Area: {result_data['mask_percentage']:.1f}%")
                
                axes[i].set_title(title, fontsize=9)
                axes[i].axis('off')
                
                # Собираем метрики для сводной таблицы
                if has_gt:
                    metrics = result_data['metrics'].copy()
                    metrics['method'] = method_name
                    metrics['time'] = result_data['time']
                    all_metrics_data.append(metrics)
                
                print(f"{method_name}: {'метрики вычислены' if has_gt else 'без ground truth'}")
                
            except Exception as e:
                error_msg = str(e)[:50]
                axes[i].text(0.5, 0.5, f"Error:\n{error_msg}", 
                           ha='center', va='center', 
                           transform=axes[i].transAxes,
                           fontsize=8)
                axes[i].set_title(f"{method_name}\n(Error)", fontsize=9)
                axes[i].axis('off')
                print(f"❌ Ошибка в методе {method_name}: {e}")
        
        # Скрываем пустые оси
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle(f"Сравнение методов сегментации {'с метриками' if has_gt else ''}\n{self.current_test_id}", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Сохраняем сравнение
        comparison_path = os.path.join(test_dir, "comparisons", "methods_comparison.jpg")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Сохраняем метрики если есть ground truth
        if has_gt and all_metrics_data:
            self._save_metrics_comparison(all_metrics_data, test_dir)
        
        self.results[test_dir] = results
        return results
    
    def _save_statistics(
        self, 
        stats: List[Dict], 
        output_dir: str
    ) -> None:
        """Сохраняет статистику тестирования"""
        
        # Функция для конвертации numpy типов в стандартные Python типы
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Конвертируем все numpy типы перед сохранением в JSON
        serializable_stats = convert_numpy_types(stats)
        
        # Сохраняем как JSON
        stats_json_path: str = os.path.join(output_dir, "statistics", "statistics.json")
        try:
            with open(stats_json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_stats, f, indent=2, ensure_ascii=False, default=str)
            print(f"📊 Статистика сохранена (JSON): {stats_json_path}")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения JSON статистики: {e}")
            # Пробуем сохранить с обработкой всех типов
            try:
                # Пытаемся сериализовать все что можно
                def default_serializer(o):
                    if isinstance(o, np.integer):
                        return int(o)
                    elif isinstance(o, np.floating):
                        return float(o)
                    elif isinstance(o, np.ndarray):
                        return o.tolist()
                    elif hasattr(o, 'tolist'):  # Для других numpy типов
                        return o.tolist()
                    elif hasattr(o, '__dict__'):
                        return str(o)
                    else:
                        return str(o)
                
                with open(stats_json_path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False, default=default_serializer)
            except Exception as e2:
                print(f"❌ Критическая ошибка сохранения JSON: {e2}")
                # Сохраняем как текстовый файл в крайнем случае
                with open(stats_json_path.replace('.json', '_fallback.txt'), 'w', encoding='utf-8') as f:
                    f.write(str(stats))
        
        # Сохраняем как CSV (если pandas доступен)
        try:
            # Создаем DataFrame из сериализуемых данных
            df_stats = []
            for stat in stats:
                # Конвертируем каждый словарь отдельно
                row = {}
                for key, value in stat.items():
                    if isinstance(value, np.integer):
                        row[key] = int(value)
                    elif isinstance(value, np.floating):
                        row[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        row[key] = str(value.tolist())
                    elif isinstance(value, tuple):
                        row[key] = str(value)
                    else:
                        row[key] = value
                df_stats.append(row)
            
            df: pd.DataFrame = pd.DataFrame(df_stats)
            stats_csv_path: str = os.path.join(output_dir, "statistics", "statistics.csv")
            df.to_csv(stats_csv_path, index=False)
            print(f"📈 Статистика сохранена (CSV): {stats_csv_path}")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения CSV: {e}")
        
        # Создаем текстовый отчет
        report_path: str = os.path.join(output_dir, "statistics", "test_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("ОТЧЕТ О ТЕСТИРОВАНИИ МЕТОДОВ СЕГМЕНТАЦИИ\n")
            f.write("="*60 + "\n\n")
            f.write(f"Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"ID теста: {self.current_test_id}\n")
            f.write(f"Всего методов: {len(stats)}\n\n")
            
            # Успешные методы
            successful: List[Dict] = [s for s in stats if 'error' not in s]
            if successful:
                f.write("УСПЕШНЫЕ МЕТОДЫ:\n")
                f.write("-"*40 + "\n")
                for stat in successful:
                    f.write(f"{stat['method']}:\n")
                    f.write(f"  Время: {float(stat.get('time_seconds', 0)):.3f} сек\n")
                    f.write(f"  Площадь маски: {int(stat.get('mask_area_pixels', 0)):,} пикселей\n")
                    f.write(f"  Процент покрытия: {float(stat.get('mask_percentage', 0)):.2f}%\n")
                    if 'image_shape' in stat:
                        shape = stat['image_shape']
                        if isinstance(shape, tuple):
                            f.write(f"  Размер результата: {shape}\n")
                        elif isinstance(shape, np.ndarray):
                            f.write(f"  Размер результата: {tuple(shape)}\n")
                        else:
                            f.write(f"  Размер результата: {shape}\n")
                    f.write("\n")
            
            # Методы с ошибками
            failed: List[Dict] = [s for s in stats if 'error' in s]
            if failed:
                f.write("МЕТОДЫ С ОШИБКАМИ:\n")
                f.write("-"*40 + "\n")
                for stat in failed:
                    f.write(f"{stat['method']}: {stat.get('error', 'Unknown error')}\n")
        
        print(f"📋 Текстовый отчет сохранен: {report_path}")

    def _save_metrics_comparison(
        self, 
        metrics_data: List[Dict[str, float]], 
        output_dir: str
    ) -> None:
        """
        Сохраняет сравнение метрик в различных форматах
        """
        metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Сохраняем как JSON
        json_path = os.path.join(metrics_dir, "metrics_comparison.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        print(f"📊 Метрики сохранены (JSON): {json_path}")
        
        # Сохраняем как CSV если pandas доступен
        try:
            df = pd.DataFrame(metrics_data)
            
            # Сортируем по IoU
            df = df.sort_values('iou', ascending=False)
            
            csv_path = os.path.join(metrics_dir, "metrics_comparison.csv")
            df.to_csv(csv_path, index=False)
            print(f"📊 Метрики сохранены (CSV): {csv_path}")
            
            # Создаем сводную таблицу в виде изображения
            self._create_metrics_table_image(df, metrics_dir)
            
            # Создаем графики сравнения метрик
            # self._create_metrics_plots(df, metrics_dir)
            
        except ImportError:
            print("⚠️ Pandas не установлен. Пропускаем создание CSV и графиков.")
    
    def _create_metrics_table_image(
        self, 
        df, 
        metrics_dir
    ):
        """Создает изображение со сводной таблицей метрик"""
        try:
            # Создаем таблицу в виде изображения
            fig, ax = plt.subplots(figsize=(12, len(df) * 0.4 + 2))
            ax.axis('tight')
            ax.axis('off')
            
            # Выбираем только основные метрики для таблицы
            table_columns = ['method', 'iou', 'dice', 'f1_score', 'precision', 
                           'recall', 'pixel_accuracy', 'mae', 'time']
            
            # Фильтруем доступные колонки
            available_columns = [col for col in table_columns if col in df.columns]
            table_data = df[available_columns].copy()
            
            # Форматируем значения
            for col in table_data.columns:
                if col != 'method':
                    table_data[col] = table_data[col].apply(lambda x: f"{x:.4f}")
            
            # Создаем таблицу
            table = ax.table(cellText=table_data.values,
                           colLabels=table_data.columns,
                           cellLoc='center',
                           loc='center',
                           colColours=['#f0f0f0']*len(table_data.columns))
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            plt.title("Сравнение метрик сегментации", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            table_path = os.path.join(metrics_dir, "metrics_table.jpg")
            plt.savefig(table_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"📊 Таблица метрик сохранена: {table_path}")
            
        except Exception as e:
            print(f"⚠️ Ошибка создания таблицы метрик: {e}")
    
    def _save_results_summary(
        self, 
        results: Dict, 
        output_dir: str
    ) -> None:
        """Сохраняет сводку результатов с конвертацией numpy типов"""
        summary_path: str = os.path.join(output_dir, "statistics", "results_summary.json")
        
        # Функция для рекурсивной конвертации numpy типов
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        # Подготовка данных для сохранения
        summary_data: Dict[str, Any] = {
            'test_id': self.current_test_id,
            'timestamp': datetime.now().isoformat(),
            'total_methods': len(results),
            'methods': {}
        }
        
        for method_name, result in results.items():
            # Конвертируем значения в result
            method_data: Dict[str, Any] = {}
            for key, value in result.items():
                if key == 'result' or key == 'mask':
                    # Пропускаем большие массивы
                    continue
                elif key == 'image_shape':
                    if isinstance(value, np.ndarray):
                        method_data[key] = value.tolist()
                    elif isinstance(value, tuple):
                        method_data[key] = list(value)
                    else:
                        method_data[key] = value
                else:
                    method_data[key] = convert_for_json(value)
            
            # Добавляем пути к файлам, если они есть
            for key in ['result_path', 'mask_path', 'overlay_path', 'original_path']:
                if key in result:
                    method_data[key] = str(result[key])
            
            summary_data['methods'][method_name] = method_data
        
        # Сохраняем с обработкой типов
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"📋 Сводка результатов сохранена: {summary_path}")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения сводки результатов: {e}")
            # Альтернативное сохранение
            with open(summary_path.replace('.json', '_simple.txt'), 'w', encoding='utf-8') as f:
                for method_name, method_data in summary_data['methods'].items():
                    f.write(f"\n{'='*40}\n")
                    f.write(f"Метод: {method_name}\n")
                    f.write(f"{'='*40}\n")
                    for key, value in method_data.items():
                        f.write(f"{key}: {value}\n")
    
    def benchmark_methods(
        self, 
        image: Union[str, np.ndarray, Image.Image], 
        n_runs: int = 3,
        save_benchmark: bool = True,
        test_name: str = None,
        save_results: bool = True
    ) -> pd.DataFrame:
        """Бенчмарк методов (требует pandas) с сохранением результатов"""
        
        benchmark_results = []

        # Создаем директорию для бенчмарка
        bench_dir: str
        if test_name:
            bench_dir = self._create_test_directory(f"benchmark_{test_name}")
        else:
            bench_dir = self._create_test_directory("benchmark")

        original_img: Image.Image
        image_array: np.ndarray
        orig_path: str
        if isinstance(image, str):
            original_img = Image.open(image).convert('RGB')
            image_array = np.array(original_img)
            # Сохраняем оригинальное изображение
            orig_path = os.path.join(bench_dir, "images", "original.jpg")
            original_img.save(orig_path)
            print(f"📸 Оригинальное изображение сохранено: {orig_path}")
        elif isinstance(image, Image.Image):
            original_img = image
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            original_img = Image.fromarray(image.astype(np.uint8))
            image_array = image
            orig_path = os.path.join(bench_dir, "images", "original.jpg")
            original_img.save(orig_path)
        
        print(f"🏃 Запуск бенчмарка ({n_runs} прогонов)...")
        
        for method_name in self.methods.keys():
            print(f"  📊 Тестируем {method_name}...")
            
            times: List[float] = []
            results_list: List[np.ndarray] = []
            masks_list: List[np.ndarray] = []

            for run in range(n_runs):
                start_time: float = time.time()
                result: np.ndarray
                mask: np.ndarray
                result, mask = self.methods[method_name].segment_with_mask(image)
                times.append(time.time() - start_time)
                if run == 0:  # Сохраняем только первую маску для статистики
                    masks_list.append(mask)
                    results_list.append(result)
            
            mask_area: np.bool
            total_pixels: int
            if masks_list and results_list:
                mask = masks_list[0]
                result_img: np.ndarray = results_list[0]
                mask_area = np.sum(mask > 0)
                total_pixels = mask.shape[0] * mask.shape[1]
            else:
                mask_area = 0
                total_pixels = 1
            
            mean_time = np.mean(times)
            std_time = np.std(times)

            if save_results and masks_list and results_list:
                try:
                    # Сохраняем результат сегментации
                    result_path: str = os.path.join(bench_dir, "images", f"{method_name}_result.jpg")
                    result_pil: Image.Image = Image.fromarray(result_img.astype(np.uint8))
                    result_pil.save(result_path)
                    
                    # Сохраняем маску
                    mask_path: str = os.path.join(bench_dir, "masks", f"{method_name}_mask.png")
                    mask_pil: Image.Image = Image.fromarray(mask.astype(np.uint8))
                    mask_pil.save(mask_path)
                    
                    # Сохраняем overlay (30% оригинал + 70% результат)
                    if image_array is not None:
                        overlay: np.ndarray = image_array * 0.3 + result_img * 0.7
                        overlay: np.ndarray = overlay.astype(np.uint8)
                        overlay_path: str = os.path.join(bench_dir, "images", f"{method_name}_overlay.jpg")
                        overlay_pil: Image.Image = Image.fromarray(overlay)
                        overlay_pil.save(overlay_path)
                        
                        print(f"    💾 Результаты сохранены в {bench_dir}")
                except Exception as e:
                    print(f"    ⚠️ Ошибка сохранения результатов: {e}")
            
            benchmark_results.append({
                'Method': method_name,
                'Mean_Time_s': mean_time,
                'Std_Time_s': std_time,
                'Time_String': f"{mean_time:.3f} ± {std_time:.3f}",
                'Mask_Area': mask_area,
                'Mask_Percentage': (mask_area / total_pixels * 100) if total_pixels > 0 else 0,
                'Min_Time_s': min(times) if times else 0,
                'Max_Time_s': max(times) if times else 0,
                'Num_Runs': n_runs
            })
        
        df: pd.DataFrame = pd.DataFrame(benchmark_results)
        if 'Mean_Time_s' in df.columns:
            df = df.sort_values('Mean_Time_s')
        elif 'Mean_Time_s (s)' in df.columns:
            df = df.sort_values('Mean_Time_s (s)')
            
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТЫ БЕНЧМАРКА:")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        # Сохраняем результаты бенчмарка
        if save_benchmark:
            self._save_benchmark_results(df, bench_dir)
        
        return df
    
    def _save_benchmark_results(
        self, 
        df: pd.DataFrame, 
        output_dir: str
    ) -> None:
        """Сохраняет результаты бенчмарка"""
        bench_stats_dir: str = os.path.join(output_dir, "statistics")
        os.makedirs(bench_stats_dir, exist_ok=True)
        
        # Сохраняем как CSV
        csv_path: str = os.path.join(bench_stats_dir, "benchmark_results.csv")
        df.to_csv(csv_path, index=False)
        
        # Сохраняем как Excel (если установлен openpyxl)
        try:
            excel_path: str = os.path.join(bench_stats_dir, "benchmark_results.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"📊 Excel отчет сохранен: {excel_path}")
        except:
            pass
        
        # Создаем текстовый отчет
        report_path: str = os.path.join(bench_stats_dir, "benchmark_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ОТЧЕТ О БЕНЧМАРКЕ МЕТОДОВ СЕГМЕНТАЦИИ\n")
            f.write("="*80 + "\n\n")
            f.write(f"Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Количество прогонов: {df['Num_Runs'].iloc[0] if not df.empty else 0}\n")
            f.write(f"Всего методов: {len(df)}\n\n")
            
            f.write("ТАБЛИЦА РЕЗУЛЬТАТОВ:\n")
            f.write("-"*80 + "\n")
            
            # Заголовок таблицы
            f.write(f"{'Метод':<30} {'Время (с)':<20} {'Площадь маски':<15} {'Процент':<10}\n")
            f.write("-"*80 + "\n")
            
            # Данные таблицы
            for _, row in df.iterrows():
                time_str = row.get('Time_String', f"{row.get('Mean_Time_s', 0):.3f} ± {row.get('Std_Time_s', 0):.3f}")
                mask_area: int = int(row.get('Mask_Area', 0))
                mask_percentage: float = row.get('Mask_Percentage', 0)
                f.write(f"{row['Method']:<30} {time_str:<20} "
                    f"{mask_area:<15,} {mask_percentage:.1f}%\n")
            
            # Сводка
            f.write("\n" + "="*80 + "\n")
            f.write("СВОДКА:\n")
            f.write("-"*80 + "\n")
            if not df.empty:
                fastest = df.iloc[0]
                slowest = df.iloc[-1]
                f.write(f"Самый быстрый метод: {fastest['Method']} ({fastest['Mean_Time_s']:.3f} с)\n")
                f.write(f"Самый медленный метод: {slowest['Method']} ({slowest['Mean_Time_s']:.3f} с)\n")
                f.write(f"Среднее время: {df['Mean_Time_s'].mean():.3f} с\n")
                f.write(f"Стандартное отклонение: {df['Mean_Time_s'].std():.3f} с\n")
        
        print(f"📋 Отчет бенчмарка сохранен: {report_path}")
        print(f"📊 CSV с результатами: {csv_path}")
        
        # Визуализация результатов бенчмарка
        self._plot_benchmark_results(df, output_dir)
    
    def _plot_benchmark_results(
        self, 
        df: pd.DataFrame, 
        output_dir: str
    ) -> None:
        """Создает графики результатов бенчмарка"""
        
        if df.empty:
            return
        
        # Создаем папку для сравнений
        comp_dir: str = os.path.join(output_dir, "comparisons")
        os.makedirs(comp_dir, exist_ok=True)
        
        # График 1: Время выполнения
        plt.figure(figsize=(12, 6))
        bars: plt.BarContainer = plt.barh(df['Method'], df['Mean_Time_s'])
        plt.xlabel('Время выполнения (секунды)')
        plt.title('Бенчмарк методов сегментации: Время выполнения')
        
        # Добавляем ошибки (стандартное отклонение)
        if 'Std_Time_s' in df.columns:
            plt.errorbar(df['Mean_Time_s'], df['Method'], 
                        xerr=df['Std_Time_s'], 
                        fmt='none', ecolor='black', capsize=5)
        
        # Добавляем значения на столбцы
        max_time = df['Mean_Time_s'].max() if not df.empty else 0
        for bar, time_val in zip(bars, df['Mean_Time_s']):
            plt.text(time_val + max_time * 0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{time_val:.3f}s', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        bench_plot_path: str = os.path.join(comp_dir, "benchmark_time.png")
        plt.savefig(bench_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # График 2: Время vs Площадь маски
        plt.figure(figsize=(10, 6))
        if 'Mask_Percentage' in df.columns:
            scatter: plt.PathCollection = plt.scatter(df['Mean_Time_s'], df['Mask_Percentage'], 
                                s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
            
            # Подписи точек
            for i, (x, y, method) in enumerate(zip(df['Mean_Time_s'], df['Mask_Percentage'], df['Method'])):
                plt.annotate(method, (x, y), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=9)
            
            plt.xlabel('Время выполнения (секунды)')
            plt.ylabel('Площадь маски (%)')
            plt.title('Бенчмарк: Время vs Площадь покрытия')
            plt.colorbar(scatter, label='Ранг метода')
            plt.grid(True, alpha=0.3)
            
            bench_scatter_path: str = os.path.join(comp_dir, "benchmark_scatter.png")
            plt.savefig(bench_scatter_path, dpi=150, bbox_inches='tight')
            plt.close()

        # График 3: Сравнительная визуализация результатов (маленькие превью)
        self._create_benchmark_preview(df, output_dir, comp_dir)
        
        print(f"📈 Графики бенчмарка сохранены в {output_dir}/comparisons/")

    def _create_metrics_plots(
        self, 
        df, 
        metrics_dir
    ):
        """Создает графики сравнения метрик"""
        try:
            # График 1: Барчарт основных метрик
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # IoU сравнение
            ax1 = axes[0, 0]
            bars1 = ax1.barh(df['method'], df['iou'])
            ax1.set_xlabel('IoU')
            ax1.set_title('Intersection over Union (IoU) по методам')
            for bar, val in zip(bars1, df['iou']):
                ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center')
            
            # Dice coefficient сравнение
            ax2 = axes[0, 1]
            bars2 = ax2.barh(df['method'], df['dice'])
            ax2.set_xlabel('Dice Coefficient')
            ax2.set_title('Dice Coefficient по методам')
            for bar, val in zip(bars2, df['dice']):
                ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center')
            
            # F1 Score сравнение
            ax3 = axes[1, 0]
            bars3 = ax3.barh(df['method'], df['f1_score'])
            ax3.set_xlabel('F1 Score')
            ax3.set_title('F1 Score по методам')
            for bar, val in zip(bars3, df['f1_score']):
                ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center')
            
            # Время выполнения
            ax4 = axes[1, 1]
            bars4 = ax4.barh(df['method'], df['time'])
            ax4.set_xlabel('Время (секунды)')
            ax4.set_title('Время выполнения по методам')
            for bar, val in zip(bars4, df['time']):
                ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.2f}s', va='center')
            
            plt.tight_layout()
            plots_path = os.path.join(metrics_dir, "metrics_comparison_plots.jpg")
            plt.savefig(plots_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"📊 Графики метрик сохранены: {plots_path}")
            
            # График 2: Scatter plot IoU vs Время
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df['time'], df['iou'], s=100, alpha=0.7)
            
            # Подписи точек
            for i, row in df.iterrows():
                ax.annotate(row['method'], (row['time'], row['iou']),
                          textcoords="offset points", xytext=(0,10),
                          ha='center', fontsize=8)
            
            ax.set_xlabel('Время выполнения (секунды)')
            ax.set_ylabel('IoU')
            ax.set_title('Соотношение точности и скорости')
            ax.grid(True, alpha=0.3)
            
            scatter_path = os.path.join(metrics_dir, "iou_vs_time_scatter.jpg")
            plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ Ошибка создания графиков метрик: {e}")

    def _create_benchmark_preview(
        self, 
        df: pd.DataFrame, 
        output_dir: str, 
        comp_dir: str
    ) -> None:
        """Создает превью результатов всех методов"""
        
        # Получаем все изображения результатов
        images_dir: str = os.path.join(output_dir, "images")
        result_files: List[str] = [f for f in os.listdir(images_dir) if f.endswith('_result.jpg')]
        
        if not result_files:
            return
        
        # Сортируем файлы по времени выполнения (согласно бенчмарку)
        sorted_methods: List[Any] = df.sort_values('Mean_Time_s')['Method'].tolist()
        
        # Загружаем изображения
        images = []
        titles = []
        for method in sorted_methods:
            result_file: str = f"{method}_result.jpg"
            if result_file in result_files:
                img_path: str = os.path.join(images_dir, result_file)
                img: Image.Image = Image.open(img_path)
                images.append(img)
                
                # Формируем заголовок
                method_data = df[df['Method'] == method]
                if not method_data.empty:
                    time_val = method_data.iloc[0]['Mean_Time_s']
                    mask_percent: float = method_data.iloc[0]['Mask_Percentage'] if 'Mask_Percentage' in method_data.columns else 0
                    title: str = f"{method}\n{time_val:.3f}s, {mask_percent:.1f}%"
                else:
                    title = method
                titles.append(title)
        
        # Создаем сетку превью
        n_images: int = len(images)
        n_cols: int = 4
        n_rows: int = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
        axes = axes.flatten()
        
        for i, (img, title) in enumerate(zip(images, titles)):
            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=8)
            axes[i].axis('off')
        
        # Скрываем пустые оси
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle("Превью результатов сегментации (сортировка по скорости)", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        preview_path:str = os.path.join(comp_dir, "methods_preview.jpg")
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_comparison(
        self, 
        results: Dict[str, Dict], 
        show_masks: bool = True,
        save_visualization: bool = True,
        output_dir: str = None,
        show_plots: bool = True
    ) -> None:
        """Визуализация сравнения результатов с сохранением"""
        output_dir: str
        if output_dir is None and self.current_test_id:
            output_dir = os.path.join(self.base_output_dir, self.current_test_id)
        elif output_dir is None:
            output_dir = self._create_test_directory("./data/visualization")
        n_methods: int = len(results)
        
        if show_masks:
            fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 10))
            
            for i, (method_name, result) in enumerate(results.items()):
                # Результат
                axes[0, i].imshow(result['result'])
                title: str = f"{method_name}\n{result['time']:.2f}s"
                axes[0, i].set_title(title, fontsize=10)
                axes[0, i].axis('off')
                
                # Маска
                axes[1, i].imshow(result['mask'], cmap='gray')
                mask_title: str = f"Mask\n{result['mask_percentage']:.1f}%"
                axes[1, i].set_title(mask_title, fontsize=10)
                axes[1, i].axis('off')
        else:
            fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
            
            for i, (method_name, result) in enumerate(results.items()):
                axes[i].imshow(result['result'])
                title: str = f"{method_name}\n{result['time']:.2f}s, {result['mask_percentage']:.1f}%"
                axes[i].set_title(title, fontsize=10)
                axes[i].axis('off')
        
        plt.suptitle("Визуализация результатов сегментации", fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Сохраняем визуализацию
        if save_visualization:
            vis_dir: str = os.path.join(output_dir, "comparisons")
            os.makedirs(vis_dir, exist_ok=True)
            
            vis_path: str = os.path.join(vis_dir, "results_visualization.jpg")
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            print(f"🖼️ Визуализация сохранена: {vis_path}")
            
            # Сохраняем отдельно для каждого метода
            for method_name, result in results.items():
                # Сохраняем увеличенный результат
                result_fig, result_ax = plt.subplots(1, 1, figsize=(8, 6))
                result_ax.imshow(result['result'])
                result_ax.set_title(f"{method_name} - Result", fontsize=12)
                result_ax.axis('off')
                
                result_path: str = os.path.join(output_dir, "images", f"{method_name}_large.jpg")
                result_fig.savefig(result_path, dpi=150, bbox_inches='tight')
                plt.close(result_fig)
                
                # Сохраняем увеличенную маску
                mask_fig, mask_ax = plt.subplots(1, 1, figsize=(8, 6))
                mask_ax.imshow(result['mask'], cmap='gray')
                mask_ax.set_title(f"{method_name} - Mask", fontsize=12)
                mask_ax.axis('off')
                
                mask_path: str = os.path.join(output_dir, "masks", f"{method_name}_large_mask.jpg")
                mask_fig.savefig(mask_path, dpi=150, bbox_inches='tight')
                plt.close(mask_fig)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def save_results(
        self, 
        results: Dict[str, Dict], 
        output_dir: str = "./data/segmentation_results"
    ) -> None:
        """Сохранение результатов всех методов"""
    
        if output_dir is None:
            output_dir = self._create_test_directory("./data/results_save")
        
        print(f"💾 Сохранение результатов в {output_dir}...")
        
        for method_name, result in results.items():
            # Сохраняем результат
            result_path: str = os.path.join(output_dir, f"{method_name}_result.jpg")
            result_img: Image.Image = Image.fromarray(result['result'].astype(np.uint8))
            result_img.save(result_path)
            
            # Сохраняем маску
            mask_path: str = os.path.join(output_dir, f"{method_name}_mask.png")
            mask_img: Image.Image = Image.fromarray(result['mask'].astype(np.uint8))
            mask_img.save(mask_path)
            
            # Сохраняем статистику
            stats_path: str = os.path.join(output_dir, f"{method_name}_stats.txt")
            with open(stats_path, 'w') as f:
                f.write(f"Method: {method_name}\n")
                f.write(f"Execution Time: {result['time']:.3f}s\n")
                f.write(f"Mask Area: {result['mask_area']} pixels\n")
                f.write(f"Mask Percentage: {result['mask_percentage']:.2f}%\n")
                f.write(f"Image Shape: {result['image_shape']}\n")
                if 'timestamp' in result:
                    f.write(f"Timestamp: {result['timestamp']}\n")
        
        print(f"✅ Все результаты сохранены в директории: {output_dir}")