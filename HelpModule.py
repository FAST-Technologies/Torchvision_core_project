# HelpModule.py

# Импорт основных библиотек
import os
import warnings
from BaseSegmenter import BaseSegmenter
from cv2SklearnSegmenter import CV2SklearnSegmenter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from typing import Union, Any, Dict, List, Tuple, Optional, TYPE_CHECKING
import time
import datetime
import cv2
from torchvision import transforms
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

if TYPE_CHECKING:
    from BaseSegmenter import BaseSegmenter

try:
    from BaseSegmenter import BaseSegmenter
    from cv2SklearnSegmenter import CV2SklearnSegmenter
except ImportError as e:
    print(f"⚠️ Warning: Could not import segmentation classes: {e}")
    print("Creating dummy classes for demonstration...")
    CV2SklearnSegmenter = None

try:
    from TorchSegmenter import TorchSegmenter
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import segmentation classes: {e}")
    TORCH_AVAILABLE = False
    TorchSegmenter = None

try:
    from NeuralSegmenter import NeuralSegmenter
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    NeuralSegmenter = None


def create_segmentation_pipeline():
    """Создание пайплайна сегментации с различными методами"""
    
    class SegmentationPipeline:
        """Пайплайн для последовательного применения методов сегментации"""
        
        def __init__(self) -> None:
            self.steps: List[Dict[str, Any]] = []
            
        def add_step(
            self, 
            name: str, 
            segmenter: BaseSegmenter, 
            params: Optional[Dict[str, Any]]  = None
        ) -> None:
            """Добавление шага в пайплайн"""
            self.steps.append({
                'name': name,
                'segmenter': segmenter,
                'params': params or {}
            })
            
        def run(
            self, 
            image: Union[str, np.ndarray, Image.Image], 
            visualize: bool = True
        ) -> Dict[str, Dict[str, Any]]:
            """Запуск пайплайна"""
            results: Dict[str, Dict[str, Any]] = {}
            
            # Загружаем оригинальное изображение и сохраняем его размер
            if isinstance(image, str):
                original_img = Image.open(image).convert('RGB')
                original_np = np.array(original_img)
            elif isinstance(image, Image.Image):
                original_img = image
                original_np = np.array(image)
            elif isinstance(image, np.ndarray):
                original_img = Image.fromarray(image.astype(np.uint8))
                original_np = image.copy()
            else:
                raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")
            
            current_input = original_np  # Всегда работаем с изображением оригинального размера
            
            for step in self.steps:
                print(f"Выполнение шага: {step['name']}")
                
                # Обновляем параметры сегментатора
                if step['params']:
                    for key, value in step['params'].items():
                        setattr(step['segmenter'], key, value)
                
                try:
                    # Выполняем сегментацию на текущем изображении
                    result: np.ndarray
                    mask: np.ndarray
                    result, mask = step['segmenter'].segment_with_mask(current_input)
                    
                    # Сохраняем результаты
                    results[step['name']] = {
                        'result': result,
                        'mask': mask,
                        'segmenter': step['segmenter'],
                        'input_shape': current_input.shape
                    }
                    
                    # Для следующего шага используем ОРИГИНАЛЬНОЕ изображение, но с маской
                    # Это гарантирует, что размеры всегда совпадают
                    if mask.ndim == 2:
                        # Если маска 2D, расширяем до 3D
                        mask_3d = np.stack([mask] * 3, axis=-1)
                        # Создаем новое изображение, где маска влияет на результат
                        alpha = 0.3  # Прозрачность наложения
                        current_input = (original_np * (1 - alpha) + 
                                    np.where(mask_3d > 0, original_np, 0) * alpha).astype(np.uint8)
                    else:
                        # Если маска уже 3D, используем как есть
                        current_input = result
                    
                    print(f"  ✓ Размер результата: {result.shape}, Размер маски: {mask.shape}")
                    
                except Exception as e:
                    print(f"  ✗ Ошибка на шаге {step['name']}: {e}")
                    # Сохраняем ошибку
                    results[step['name']] = {
                        'error': str(e),
                        'segmenter': step['segmenter']
                    }
                    # Продолжаем с предыдущим изображением
                    continue
            
            if visualize:
                self.visualize_results(results, original_np)
            
            return results
        
        def visualize_results(
            self, 
            results: Dict[str, Dict[str, Any]], 
            original_image: np.ndarray
        ) -> None:
            """Визуализация результатов пайплайна"""
            n_steps: int = len(results)
            
            fig_height: int = 10
            fig_width: int = 5 * (n_steps + 1)
            
            fig, axes = plt.subplots(2, n_steps + 1, figsize=(fig_width, fig_height))
            
            # Оригинальное изображение
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title("Original")
            axes[0, 0].axis('off')
            
            axes[1, 0].axis('off')
            
            # Результаты каждого шага
            for i, (step_name, step_result) in enumerate(results.items(), 1):
                if 'error' in step_result:
                    # Показываем ошибку
                    axes[0, i].text(0.5, 0.5, f"Error:\n{step_result['error'][:50]}", 
                                ha='center', va='center', transform=axes[0, i].transAxes)
                    axes[0, i].set_title(f"Step {i}: {step_name}\n(ERROR)")
                    axes[0, i].axis('off')
                    
                    axes[1, i].text(0.5, 0.5, "No mask", 
                                ha='center', va='center', transform=axes[1, i].transAxes)
                    axes[1, i].set_title(f"Mask {i}\n(ERROR)")
                    axes[1, i].axis('off')
                    continue
                
                result_data: np.ndarray = step_result.get('result', np.zeros_like(original_image))
                mask_data: np.ndarray = step_result.get('mask', np.zeros(original_image.shape[:2]))
                
                # Результат сегментации
                axes[0, i].imshow(result_data)
                axes[0, i].set_title(f"Step {i}: {step_name}")
                axes[0, i].axis('off')
                
                # Маска
                if mask_data.ndim == 2:
                    # Grayscale mask
                    axes[1, i].imshow(mask_data, cmap='gray', vmin=0, vmax=255)
                else:
                    # Color mask
                    axes[1, i].imshow(mask_data)
                
                axes[1, i].set_title(f"Mask {i}")
                axes[1, i].axis('off')

            # Скрываем пустые оси, если шагов меньше, чем слотов
            for j in range(i + 1, n_steps + 1):
                axes[0, j].axis('off')
                axes[1, j].axis('off')
            
            plt.tight_layout()
            plt.show()
    
        def get_step_names(self) -> List[str]:
            """Получить имена всех шагов в пайплайне"""
            return [step['name'] for step in self.steps]
        
        def get_step(
            self, 
            name: str
        ) -> Optional[Dict[str, Any]]:
            """Получить информацию о конкретном шаге по имени"""
            for step in self.steps:
                if step['name'] == name:
                    return step
            return None
        
        def remove_step(
            self, 
            name: str
        ) -> bool:
            """Удалить шаг из пайплайна по имени"""
            for i, step in enumerate(self.steps):
                if step['name'] == name:
                    self.steps.pop(i)
                    return True
            return False
        
        def clear_pipeline(self) -> None:
            """Очистить весь пайплайн"""
            self.steps = []
        
        def save_visualization(
            self, 
            results: Dict[str, Dict[str, Any]], 
            original_image: Union[str, np.ndarray, Image.Image],
            save_path: str = "./data/pipeline_results.jpg"
        ) -> None:
            """Сохранить визуализацию результатов пайплайна"""
            n_steps: int = len(results)
            fig, axes = plt.subplots(2, n_steps + 1, figsize=(5 * (n_steps + 1), 10))
            
            # Обрабатываем оригинальное изображение
            if isinstance(original_image, str):
                orig_img = Image.open(original_image).convert('RGB')
            elif isinstance(original_image, Image.Image):
                orig_img = original_image
            else:
                orig_img = Image.fromarray(original_image.astype(np.uint8))
            
            axes[0, 0].imshow(orig_img)
            axes[0, 0].set_title("Original")
            axes[0, 0].axis('off')
            
            axes[1, 0].axis('off')
            
            # Результаты каждого шага
            for i, (step_name, step_result) in enumerate(results.items(), 1):
                axes[0, i].imshow(step_result['result'])
                axes[0, i].set_title(f"Step {i}: {step_name}")
                axes[0, i].axis('off')
                
                axes[1, i].imshow(step_result['mask'], cmap='gray')
                axes[1, i].set_title(f"Mask {i}")
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Визуализация сохранена: {save_path}")
    
    return SegmentationPipeline()

def create_advanced_pipeline_example() -> Any:
    """Пример создания сложного пайплайна с различными методами"""
    
    pipeline = create_segmentation_pipeline()
    
    # Этап 1: Детекция краев
    pipeline.add_step(
        "Sobel_Edge_Detection",
        CV2SklearnSegmenter("sobel_edge", threshold=40)
    )
    
    # Этап 2: Бинаризация
    pipeline.add_step(
        "Otsu_Thresholding",
        CV2SklearnSegmenter("otsu_thresholding")
    )
    
    # Этап 3: Морфологические операции (через адаптивный порог)
    pipeline.add_step(
        "Morphological_Refinement",
        CV2SklearnSegmenter("adaptive_thresholding", block_size=25, C=10)
    )
    
    # Этап 4: Кластеризация для сегментации
    pipeline.add_step(
        "KMeans_Clustering",
        CV2SklearnSegmenter("kmeans_segmentation", k=4)
    )
    
    # Этап 5: Окончательная сегментация
    pipeline.add_step(
        "Final_Segmentation",
        CV2SklearnSegmenter("watershed")
    )
    
    return pipeline


def analyze_pipeline_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Анализ результатов пайплайна"""
    
    analysis: Dict[str, Any] = {
        'total_steps': len(results),
        'step_names': list(results.keys()),
        'mask_statistics': {},
        'execution_info': {},
        'size_mismatch_errors': [],  # Добавим отслеживание ошибок размеров
        'successful_steps': 0,
        'failed_steps': 0
    }
    
    for step_name, step_data in results.items():
        if 'error' in step_data:
            analysis['failed_steps'] += 1
            continue
            
        analysis['successful_steps'] += 1
        mask: np.ndarray = step_data.get('mask', None)
        
        if mask is None:
            analysis['size_mismatch_errors'].append(f"{step_name}: маска отсутствует")
            continue
            
        # Статистика маски
        try:
            mask_stats = {
                'shape': mask.shape,
                'dtype': str(mask.dtype),
                'min_value': float(mask.min()),
                'max_value': float(mask.max()),
                'mean_value': float(mask.mean()),
                'nonzero_pixels': int(np.count_nonzero(mask)),
                'total_pixels': int(mask.size),
                'coverage_percentage': float(np.count_nonzero(mask) / mask.size * 100) if mask.size > 0 else 0
            }
            
            analysis['mask_statistics'][step_name] = mask_stats
            
        except Exception as e:
            analysis['size_mismatch_errors'].append(f"{step_name}: ошибка статистики - {str(e)}")
        
        # Информация о сегментаторе
        segmenter: BaseSegmenter = step_data.get('segmenter', None)
        if segmenter:
            analysis['execution_info'][step_name] = {
                'segmenter_type': type(segmenter).__name__,
                'segmenter_name': segmenter.name if hasattr(segmenter, 'name') else 'Unknown'
            }
    
    return analysis

def print_pipeline_analysis(analysis: Dict[str, Any]) -> None:
    """Печать анализа результатов пайплайна"""
    
    print("\n" + "="*60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ ПАЙПЛАЙНА")
    print("="*60)
    
    print(f"Всего шагов: {analysis['total_steps']}")
    print(f"Успешных: {analysis['successful_steps']}")
    print(f"Неудачных: {analysis['failed_steps']}")
    print(f"Шаги: {', '.join(analysis['step_names'])}")
    
    print("\nСтатистика масок:")
    print("-"*40)
    
    for step_name, stats in analysis['mask_statistics'].items():
        print(f"\n{step_name}:")
        print(f"  Размер: {stats['shape']}")
        print(f"  Тип данных: {stats['dtype']}")
        print(f"  Диапазон значений: [{stats['min_value']:.2f}, {stats['max_value']:.2f}]")
        print(f"  Среднее значение: {stats['mean_value']:.2f}")
        print(f"  Ненулевых пикселей: {stats['nonzero_pixels']:,}")
        print(f"  Покрытие: {stats['coverage_percentage']:.2f}%")
    
    print("\nИнформация о сегментаторах:")
    print("-"*40)
    
    for step_name, info in analysis['execution_info'].items():
        print(f"{step_name}: {info['segmenter_type']} ({info['segmenter_name']})")


def apply_torch_method(
    image_path: str, 
    method: str, 
    method_name: str = None, 
    save_dir: str = "results", 
    **kwargs
) -> Tuple[Dict[str, Any], Image.Image, np.ndarray, float]:
    """
    Применяет метод PyTorch сегментации с сохранением результатов
    
    Args:
        image_path: Путь к изображению
        method: Название метода (например, "watershed", "grabcut")
        method_name: Отображаемое имя метода
        save_dir: Директория для сохранения
        **kwargs: Параметры для метода
    
    Returns:
        Словарь с результатами и путями к файлам
    """
    os.makedirs(save_dir, exist_ok=True)

    if TorchSegmenter is None:
        raise ImportError("TorchSegmenter не доступен")
    
    if method_name is None:
        method_name = method
    
    print(f"  Применяем {method_name} ({method})...")
    start_time = time.time()
    
    try:
        segmenter = create_torch_segmenter(method, **kwargs)
        result_img, mask = segmenter.segment_with_mask(image_path)
        execution_time = time.time() - start_time
        
        # Сохраняем результаты
        result_path = os.path.join(save_dir, f"{method_name.lower()}_result.jpg")
        if isinstance(result_img, np.ndarray):
            result_pil = Image.fromarray(result_img.astype(np.uint8))
        else:
            result_pil = result_img
        result_pil.save(result_path)
        
        mask_path = os.path.join(save_dir, f"{method_name.lower()}_mask.png")
        if mask.ndim == 2:
            mask_img = Image.fromarray(mask.astype(np.uint8))
        else:
            # Если маска цветная, конвертируем в grayscale
            mask_gray = np.mean(mask, axis=2) if mask.ndim == 3 else mask
            mask_img = Image.fromarray(mask_gray.astype(np.uint8))

        mask_new = mask
        if isinstance(mask_new, torch.Tensor):
            if mask_new.dim() == 4:
                mask_new = mask_new.squeeze(0)
            if mask_new.dim() == 3 and mask_new.shape[0] == 1:
                mask_new = mask_new.squeeze(0)
            mask_np = mask_new.cpu().numpy()
        else:
            mask_np = mask_new
        
        mask_img.save(mask_path)
        
        # Создаем визуализацию
        original_img = Image.open(image_path).convert('RGB')
        original_np = np.array(original_img)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(result_img)
        axes[1].set_title(f'{method_name} Result')
        axes[1].axis('off')
        
        if mask.ndim == 2:
            axes[2].imshow(mask, cmap='gray')
        else:
            axes[2].imshow(mask)
        axes[2].set_title(f'{method_name} Mask')
        axes[2].axis('off')
        
        plt.suptitle(f'{method_name} Segmentation', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        viz_path = os.path.join(save_dir, f"{method_name.lower()}_visualization.jpg")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ {method_name} сохранен в {save_dir}")
        print(f"    ✅ {method_name} успешно за {execution_time:.3f}s")
        result_Dict: Dict[str, Any] = {
            'result': result_img,
            'mask': mask,
            'result_path': result_path,
            'mask_path': mask_path,
            'viz_path': viz_path,
            'segmenter': segmenter,
            'success': True
        }
        return result_Dict, result_pil, mask_np, execution_time
        
    except Exception as e:
        print(f"    ❌ Ошибка {method_name}: {e}")
        warnings.warn(f"{method_name} failed: {e}")
        
        # Создаем заглушку
        original_img = Image.open(image_path).convert('RGB')
        original_np = np.array(original_img)
        dummy_mask = np.zeros(original_np.shape[:2], dtype=np.uint8)
        dummy_result = original_np.copy()
        result_Dict: Dict[str, Any] = {
            'result': dummy_result,
            'mask': dummy_mask,
            'success': False,
            'error': str(e)
        }
        return result_Dict, original_img, dummy_mask, execution_time

def create_torch_segmenter(method: str, **kwargs) -> TorchSegmenter:
    """Создает сегментатор PyTorch с указанными параметрами"""
    return TorchSegmenter(method=method, **kwargs)

def apply_neural_segmentation(
    image_path: str, 
    save_dir: str = "./data/results"
) -> Tuple[Dict[str, Any], Image.Image, np.ndarray, float]:
    """
    Применяет нейросетевую сегментацию с замером времени
    Возвращает (result_image, mask, execution_time)
    """
    if NeuralSegmenter is None:
        raise ImportError("NeuralSegmenter не доступен")
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("  Применяем Neural Segmentation...")
    start_time = time.time()
    
    try:
        # Создаем нейросетевой сегментатор
        neural_segmenter = NeuralSegmenter(
            local_path="/home/yamshchikov/models/segformer-b5-ready"
        )
        
        # Сегментируем изображение
        neural_result = neural_segmenter.segment_image(image_path, alpha=0.7)
        execution_time = time.time() - start_time
        
        # Сохраняем результат
        result_path = os.path.join(save_dir, "neural_segmentation.jpg")
        neural_result.save(result_path)
        
        # Также получаем детальную информацию если доступно
        detailed_info = None
        if hasattr(neural_segmenter, 'detailed_segmentation'):
            detailed_info = neural_segmenter.detailed_segmentation(image_path)
            
            # Сохраняем overlay из detailed segmentation
            if 'overlay' in detailed_info:
                overlay = Image.fromarray(detailed_info['overlay'].astype(np.uint8))
                overlay_path = os.path.join(save_dir, "neural_detailed_overlay.jpg")
                overlay.save(overlay_path)
        
        # Получаем маску сегментации
        mask = None
        if hasattr(neural_segmenter, 'segment_with_mask'):
            result_np, mask = neural_segmenter.segment_with_mask(image_path, alpha=0.5)
            
            if mask is not None:
                mask_path = os.path.join(save_dir, "neural_mask.png")
                if mask.ndim == 2:
                    mask_img = Image.fromarray(mask.astype(np.uint8))
                else:
                    mask_img = Image.fromarray(mask.astype(np.uint8))
                mask_img.save(mask_path)

        if mask is None:
            print("Current mask is none")
            original_img = Image.open(image_path).convert('RGB')
            mask = np.zeros(original_img.size[::-1], dtype=np.uint8)
        
        print(f"    ✅ Neural segmentation сохранен в {save_dir}")
        print(f"    ✅ Neural segmentation успешно за {execution_time:.3f}s")
        result_dict: Dict[str, Any] = {
            'result': np.array(neural_result),
            'mask': mask,
            'result_path': result_path,
            'segmenter': neural_segmenter,
            'detailed_info': detailed_info,
            'success': True
        }
        return result_dict, neural_result, mask, execution_time
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"    ❌ Ошибка Neural segmentation: {e}")
        
        # Создаем заглушку
        original_img = Image.open(image_path).convert('RGB')
        original_np = np.array(original_img)
        mask = np.zeros(original_np.shape[:2], dtype=np.uint8)
        result_dict: Dict[str, Any] = {
            'result': original_np,
            'mask': np.zeros(original_np.shape[:2], dtype=np.uint8),
            'success': False,
            'error': str(e)
        }
        return result_dict, original_img, mask, execution_time

def original_compare_segmentation_methods(
    image_path: str, 
    save_dir: str = "./data/legacy_comparison"
) -> Dict[str, Any]:
    """
    ОРИГИНАЛЬНАЯ функция сравнения основных методов сегментации 
    с использованием TorchSegmenter
    
    Args:
        image_path: Путь к тестовому изображению
        save_dir: Базовая директория для сохранения
    
    Returns:
        Словарь с результатами всех методов
    """

    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("COMPARISON OF SEGMENTATION METHODS (USING TorchSegmenter)")
    print("="*60)
    
    # Загружаем оригинальное изображение
    original_img = Image.open(image_path).convert('RGB')
    original_np = np.array(original_img)
    
    # Применяем методы
    methods_results = {}

    print("Applying Watershed...")
    watershed_dir = os.path.join(save_dir, "watershed")
    methods_results['watershed'] = apply_torch_method(
        image_path=image_path,
        method="watershed",
        method_name="Watershed",
        save_dir=watershed_dir
    )[0]  # Берем только словарь (первый элемент кортежа)

    print("Applying MeanShift...")
    meanshift_dir = os.path.join(save_dir, "meanshift")
    methods_results['meanshift'] = apply_torch_method(
        image_path=image_path,
        method="meanshift",
        method_name="MeanShift",
        save_dir=meanshift_dir,
        bandwidth=0.5,
        spatial_radius=35,
        color_radius=60
    )[0]  # Берем только словарь

    print("Applying GrabCut...")
    grabcut_dir = os.path.join(save_dir, "grabcut")
    methods_results['grabcut'] = apply_torch_method(
        image_path=image_path,
        method="grabcut",
        method_name="GrabCut",
        save_dir=grabcut_dir,
        rect=(100, 100, 200, 200),
        num_iterations=5
    )[0]  # Берем только словарь

    print("Applying FloodFill...")
    floodfill_dir = os.path.join(save_dir, "floodfill")
    methods_results['floodfill'] = apply_torch_method(
        image_path=image_path,
        method="floodfill",
        method_name="FloodFill",
        save_dir=floodfill_dir,
        tolerance=0.15,
        points=[(100, 100), (200, 200), (300, 300)]
    )[0]  # Берем только словарь

    print("Applying Neural Segmentation...")
    neural_dir = os.path.join(save_dir, "neural")
    methods_results['neural'] = apply_neural_segmentation(
        image_path=image_path,
        save_dir=neural_dir
    )[0]  # Берем только словарь

    # Создаем сводный график
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))

    # Original
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Watershed
    if methods_results['watershed']['success']:
        axes[0, 1].imshow(methods_results['watershed']['result'], cmap='tab20')
    else:
        axes[0, 1].text(0.5, 0.5, 'Watershed\nFailed', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    axes[0, 1].set_title('Watershed')
    axes[0, 1].axis('off')

    # MeanShift
    if methods_results['meanshift']['success']:
        axes[0, 2].imshow(methods_results['meanshift']['result'], cmap='tab20')
    else:
        axes[0, 2].text(0.5, 0.5, 'MeanShift\nFailed', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
    axes[0, 2].set_title('MeanShift Result')
    axes[0, 2].axis('off')

    # GrabCut
    if methods_results['grabcut']['success']:
        mask = methods_results['grabcut']['result']
        if mask.ndim == 2:
            axes[0, 3].imshow(mask, cmap='gray')
        else:
            axes[0, 3].imshow(mask)
    else:
        axes[0, 3].text(0.5, 0.5, 'GrabCut\nFailed', 
                       ha='center', va='center', transform=axes[0, 3].transAxes)
    axes[0, 3].set_title('GrabCut Result')
    axes[0, 3].axis('off')

    # Watershed Mask
    if methods_results['watershed']['success']:
        axes[1, 0].imshow(methods_results['watershed']['mask'], cmap='tab20')
    else:
        axes[1, 0].text(0.5, 0.5, 'Watershed\nFailed', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Watershed Mask')
    axes[1, 0].axis('off')

    # MeanShift Mask
    if methods_results['meanshift']['success']:
        axes[1, 1].imshow(methods_results['meanshift']['mask'], cmap='tab20')
    else:
        axes[1, 1].text(0.5, 0.5, 'MeanShift\nFailed', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('MeanShift Mask')
    axes[1, 1].axis('off')

    # GrabCut Mask
    if methods_results['grabcut']['success']:
        mask = methods_results['grabcut']['mask']
        if mask.ndim == 2:
            axes[1, 2].imshow(mask, cmap='gray')
        else:
            axes[1, 2].imshow(mask)
    else:
        axes[1, 2].text(0.5, 0.5, 'GrabCut\nFailed', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('GrabCut Mask')
    axes[1, 2].axis('off')

    # Neural
    if methods_results['neural']['success']:
        axes[1, 3].imshow(methods_results['neural']['result'])
    else:
        axes[1, 3].text(0.5, 0.5, 'Neural\nFailed', 
                       ha='center', va='center', transform=axes[1, 3].transAxes)
    axes[1, 3].set_title('Neural Segmentation')
    axes[1, 3].axis('off')

    plt.suptitle("Segmentation Methods Comparison (Using TorchSegmenter)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Сохраняем сводный график
    summary_path = os.path.join(save_dir, "methods_comparison_summary.jpg")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Создаем отчет
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    successful_methods = []
    failed_methods = []
    
    for method_name, result in methods_results.items():
        if result.get('success', False):
            successful_methods.append(method_name)
        else:
            failed_methods.append((method_name, result.get('error', 'Unknown error')))
    
    print(f"✓ Успешные методы ({len(successful_methods)}): {', '.join(successful_methods)}")
    
    if failed_methods:
        print(f"✗ Неуспешные методы ({len(failed_methods)}):")
        for method_name, error in failed_methods:
            print(f"  - {method_name}: {error[:50]}...")
    
    print(f"\n📊 Сводный график: {summary_path}")
    print(f"📁 Все результаты в: {save_dir}")
    print("="*60)
    
    # Печатаем описание методов
    print("\n" + "="*50)
    print("SEGMENTATION METHODS DESCRIPTION")
    print("="*50)
    print("1. Watershed - Сегментация на основе градиентов изображения")
    print("2. MeanShift - Кластеризация в пространственно-цветовом пространстве")
    print("3. GrabCut - Разделение переднего и заднего плана с использованием GMM")
    print("4. FloodFill - Заполнение регионов от начальных точек (Region Growing)")
    print("5. Neural - Семантическая сегментация с использованием нейросетей")
    print("="*50)
    
    return {
        'methods': methods_results,
        'summary_path': summary_path,
        'save_dir': save_dir,
        'successful_methods': successful_methods,
        'failed_methods': failed_methods
    }

def compare_cv2_torch_methods(image_path: str, save_dir: str = "./data/cv2_vs_torch") -> Dict[str, Any]:
    """
    Сравнение методов CV2 и PyTorch реализаций
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("CV2 vs PyTorch METHODS COMPARISON")
    print("="*60)
    
    # Загружаем изображение
    original_img = Image.open(image_path).convert('RGB')
    
    # Создаем список методов для сравнения
    comparison_methods = [
        # (method_name, torch_method, cv2_method, params)
        ("Global Threshold", "global_thresholding", "global_thresholding", {"threshold": 0.5}),
        ("Adaptive Threshold", "adaptive_thresholding", "adaptive_thresholding", {"block_size": 11, "C": 2}),
        ("Otsu", "otsu_thresholding", "otsu_thresholding", {}),
        ("Sobel Edge", "sobel_edge", "sobel_edge", {"threshold": 0.1}),
        ("Canny Edge", "canny_edge", "canny_edge", {"low": 0.1, "high": 0.3}),
        ("KMeans", "kmeans_segmentation", "kmeans_segmentation", {"k": 3}),
    ]
    
    results = {}
    
    for display_name, torch_method, cv2_method, params in comparison_methods:
        print(f"\nСравниваем {display_name}...")
        
        # Создаем директории
        method_dir = os.path.join(save_dir, display_name.lower().replace(" ", "_"))
        torch_dir = os.path.join(method_dir, "torch")
        cv2_dir = os.path.join(method_dir, "cv2")
        
        # Применяем PyTorch метод
        print(f"  PyTorch {torch_method}...")
        torch_result = apply_torch_method(
            image_path=image_path,
            method=torch_method,
            method_name=f"Torch_{display_name}",
            save_dir=torch_dir,
            **params
        )
        
        # Применяем CV2 метод (если доступен)
        try:
            from cv2SklearnSegmenter import CV2SklearnSegmenter
            print(f"  CV2 {cv2_method}...")
            
            cv2_segmenter = CV2SklearnSegmenter(cv2_method, **params)
            cv2_result_img, cv2_mask = cv2_segmenter.segment_with_mask(image_path)
            
            # Сохраняем результаты CV2
            os.makedirs(cv2_dir, exist_ok=True)
            cv2_result_path = os.path.join(cv2_dir, "result.jpg")
            Image.fromarray(cv2_result_img.astype(np.uint8)).save(cv2_result_path)
            
            cv2_mask_path = os.path.join(cv2_dir, "mask.png")
            if cv2_mask.ndim == 2:
                Image.fromarray(cv2_mask.astype(np.uint8)).save(cv2_mask_path)
            
            cv2_success = True
        except Exception as e:
            print(f"    ❌ CV2 {cv2_method} failed: {e}")
            cv2_success = False
            cv2_result_img = np.zeros_like(original_img)
            cv2_mask = np.zeros(original_img.size[::-1], dtype=np.uint8)
        
        results[display_name] = {
            'torch': torch_result,
            'cv2': {
                'result': cv2_result_img,
                'mask': cv2_mask,
                'success': cv2_success
            }
        }
    
    # Создаем сравнение
    n_methods = len(comparison_methods)
    fig, axes = plt.subplots(n_methods, 3, figsize=(15, 5 * n_methods))
    
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (display_name, torch_method, cv2_method, params) in enumerate(comparison_methods):
        # Оригинал (только для первой строки)
        if idx == 0:
            axes[idx, 0].imshow(original_img)
            axes[idx, 0].set_title('Original')
        else:
            axes[idx, 0].axis('off')
        
        # PyTorch результат
        torch_data = results[display_name]['torch']
        if torch_data.get('success', False):
            axes[idx, 1].imshow(torch_data['result'])
            axes[idx, 1].set_title(f'PyTorch\n{display_name}')
        else:
            axes[idx, 1].text(0.5, 0.5, 'Failed', 
                            ha='center', va='center', transform=axes[idx, 1].transAxes)
            axes[idx, 1].set_title(f'PyTorch\n{display_name}')
        axes[idx, 1].axis('off')
        
        # CV2 результат
        cv2_data = results[display_name]['cv2']
        if cv2_data['success']:
            axes[idx, 2].imshow(cv2_data['result'])
            axes[idx, 2].set_title(f'CV2\n{display_name}')
        else:
            axes[idx, 2].text(0.5, 0.5, 'Failed', 
                            ha='center', va='center', transform=axes[idx, 2].transAxes)
            axes[idx, 2].set_title(f'CV2\n{display_name}')
        axes[idx, 2].axis('off')
    
    plt.suptitle("CV2 vs PyTorch Implementations Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    comparison_path = os.path.join(save_dir, "cv2_vs_torch_comparison.jpg")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Сравнение сохранено: {comparison_path}")
    return results

def run_comprehensive_comparison(image_path: str) -> Dict[str, Any]:
    """
    Запуск всестороннего сравнения методов сегментации
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_dir = f"comprehensive_comparison_{timestamp}"
    
    print("="*60)
    print(f"COMPREHENSIVE SEGMENTATION COMPARISON")
    print("="*60)
    print(f"📁 Все результаты будут сохранены в: {main_dir}")
    print(f"📷 Тестовое изображение: {image_path}")
    print("="*60)
    
    # 1. Основное сравнение методов
    print("\n1. Основное сравнение методов сегментации...")
    basic_results = original_compare_segmentation_methods(
        image_path=image_path,
        save_dir=os.path.join(main_dir, "basic_comparison")
    )
    
    # 2. Сравнение CV2 vs PyTorch
    print("\n2. Сравнение CV2 и PyTorch реализаций...")
    try:
        cv2_torch_results = compare_cv2_torch_methods(
            image_path=image_path,
            save_dir=os.path.join(main_dir, "cv2_vs_torch")
        )
    except Exception as e:
        print(f"⚠️ CV2 vs PyTorch comparison failed: {e}")
        cv2_torch_results = None
    
    # 3. Создаем README файл
    readme_path = os.path.join(main_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# Comprehensive Segmentation Methods Comparison\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Test Image:** `{image_path}`\n")
        f.write(f"**Main Directory:** `{main_dir}`\n\n")
        
        f.write("## 📁 Directory Structure\n\n")
        f.write("```\n")
        f.write(f"{main_dir}/\n")
        f.write("├── basic_comparison/     # Основное сравнение методов\n")
        f.write("│   ├── watershed/        # Watershed segmentation\n")
        f.write("│   ├── meanshift/        # MeanShift clustering\n")
        f.write("│   ├── grabcut/          # GrabCut segmentation\n")
        f.write("│   ├── floodfill/        # FloodFill segmentation\n")
        f.write("│   ├── neural/           # Neural segmentation\n")
        f.write("│   └── methods_comparison_summary.jpg\n")
        
        if cv2_torch_results:
            f.write("├── cv2_vs_torch/        # CV2 vs PyTorch сравнение\n")
            f.write("│   ├── global_threshold/\n")
            f.write("│   ├── adaptive_threshold/\n")
            f.write("│   └── cv2_vs_torch_comparison.jpg\n")
        
        f.write("└── README.md            # Этот файл\n")
        f.write("```\n\n")
        
        f.write("## 🎯 Methods Overview\n\n")
        f.write("| Method | Description | Implementation |\n")
        f.write("|--------|-------------|----------------|\n")
        f.write("| Watershed | Edge-based segmentation using gradients | PyTorch |\n")
        f.write("| MeanShift | Clustering in spatial-color feature space | PyTorch |\n")
        f.write("| GrabCut | Interactive foreground/background separation | PyTorch |\n")
        f.write("| FloodFill | Region growing from seed points | PyTorch |\n")
        f.write("| Neural | Deep learning semantic segmentation | SegFormer |\n")
        
        if cv2_torch_results:
            f.write("| Global Threshold | Simple thresholding | PyTorch & CV2 |\n")
            f.write("| Adaptive Threshold | Local thresholding | PyTorch & CV2 |\n")
            f.write("| Otsu | Automatic threshold selection | PyTorch & CV2 |\n")
            f.write("| Sobel | Edge detection | PyTorch & CV2 |\n")
            f.write("| Canny | Edge detection with hysteresis | PyTorch & CV2 |\n")
            f.write("| KMeans | Color clustering | PyTorch & CV2 |\n")
        
        f.write("\n## 📊 Results Summary\n\n")
        f.write(f"**Successful Methods:** {len(basic_results['successful_methods'])}\n")
        f.write(f"**Failed Methods:** {len(basic_results['failed_methods'])}\n\n")
        
        if basic_results['failed_methods']:
            f.write("### Failed Methods\n\n")
            for method_name, error in basic_results['failed_methods']:
                f.write(f"- **{method_name}:** `{error[:100]}`\n")
    
    print(f"\n📄 README файл создан: {readme_path}")
    
    # 4. Сводная информация
    print("\n" + "="*60)
    print("COMPARISON COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📁 Main directory: {main_dir}")
    print(f"📊 Basic comparison: {basic_results['summary_path']}")
    
    if cv2_torch_results:
        print(f"🔬 CV2 vs PyTorch: {os.path.join(main_dir, 'cv2_vs_torch', 'cv2_vs_torch_comparison.jpg')}")
    
    print(f"📋 Summary: {len(basic_results['successful_methods'])} successful, {len(basic_results['failed_methods'])} failed")
    print("="*60)
    
    return {
        'main_dir': main_dir,
        'basic_results': basic_results,
        'cv2_torch_results': cv2_torch_results,
        'readme_path': readme_path
    }

# ============ ОСНОВНАЯ ФУНКЦИЯ ДЛЯ СРАВНЕНИЯ С ТАЙМИНГОМ ============

def compare_segmentation_methods_with_timing(image_path: str) -> Tuple[List[str], List[Image.Image], List[np.ndarray], List[float]]:
    """
    Compare all methods with execution timing - ЧИСТАЯ АДАПТАЦИЯ
    
    Возвращает: (methods, results, masks, times)
    """
    
    print("="*60)
    print("COMPARISON OF SEGMENTATION METHODS WITH TIMING")
    print("="*60)
    
    # Загружаем оригинальное изображение
    original_img = Image.open(image_path).convert('RGB')
    
    methods = []
    results = []    # List[Image.Image]
    masks = []      # List[np.ndarray]
    times = []      # List[float]
    results_dicts = []  # List[Dict] - для сохранения всех результатов

    # 1. Neural Segmentation
    print("\n1. Neural Segmentation...")
    try:
        neural_dict, neural_result, neural_mask, neural_time = apply_neural_segmentation(image_path)
        methods.append("Neural")
        results.append(neural_result)
        masks.append(neural_mask)
        times.append(neural_time)
        results_dicts.append(neural_dict)
        print(f"   ✓ Успешно за {neural_time:.3f}s")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        # Добавляем заглушку
        methods.append("Neural")
        results.append(original_img)
        masks.append(np.zeros(original_img.size[::-1], dtype=np.uint8))
        times.append(0.0)
        results_dicts.append({'success': False, 'error': str(e)})

    # 2. Watershed (PyTorch)
    print("\n2. Watershed (PyTorch)...")
    try:
        watershed_dict, watershed_result, watershed_mask, watershed_time = apply_torch_method(
            image_path=image_path,
            method="watershed",
            method_name="Watershed"
        )
        methods.append("Watershed")
        results.append(watershed_result)
        masks.append(watershed_mask)
        times.append(watershed_time)
        results_dicts.append(watershed_dict)
        print(f"   ✓ Успешно за {watershed_time:.3f}s")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        methods.append("Watershed")
        results.append(original_img)
        masks.append(np.zeros(original_img.size[::-1], dtype=np.uint8))
        times.append(0.0)
        results_dicts.append({'success': False, 'error': str(e)})

    # 3. MeanShift (PyTorch)
    print("\n3. MeanShift (PyTorch)...")
    try:
        meanshift_dict, meanshift_result, meanshift_mask, meanshift_time = apply_torch_method(
            image_path=image_path,
            method="meanshift",
            method_name="MeanShift",
            bandwidth=0.5,
            spatial_radius=35,
            color_radius=60
        )
        methods.append("MeanShift")
        results.append(meanshift_result)
        masks.append(meanshift_mask)
        times.append(meanshift_time)
        results_dicts.append(meanshift_dict)
        print(f"   ✓ Успешно за {meanshift_time:.3f}s")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        methods.append("MeanShift")
        results.append(original_img)
        masks.append(np.zeros(original_img.size[::-1], dtype=np.uint8))
        times.append(0.0)
        results_dicts.append({'success': False, 'error': str(e)})

    # 4. GrabCut (PyTorch)
    print("\n4. GrabCut (PyTorch)...")
    try:
        grabcut_dict, grabcut_result, grabcut_mask, grabcut_time = apply_torch_method(
            image_path=image_path,
            method="grabcut",
            method_name="GrabCut",
            rect=(100, 100, 200, 200),
            num_iterations=5
        )
        methods.append("GrabCut")
        results.append(grabcut_result)
        masks.append(grabcut_mask)
        times.append(grabcut_time)
        results_dicts.append(grabcut_dict)
        print(f"   ✓ Успешно за {grabcut_time:.3f}s")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        methods.append("GrabCut")
        results.append(original_img)
        masks.append(np.zeros(original_img.size[::-1], dtype=np.uint8))
        times.append(0.0)
        results_dicts.append({'success': False, 'error': str(e)})

    # 5. FloodFill (PyTorch)
    print("\n5. FloodFill (PyTorch)...")
    try:
        floodfill_dict, floodfill_result, floodfill_mask, floodfill_time = apply_torch_method(
            image_path=image_path,
            method="floodfill",
            method_name="FloodFill",
            tolerance=0.15,
            points=[(100, 100), (200, 200), (300, 300)]
        )
        methods.append("FloodFill")
        results.append(floodfill_result)
        masks.append(floodfill_mask)
        times.append(floodfill_time)
        results_dicts.append(floodfill_dict)
        print(f"   ✓ Успешно за {floodfill_time:.3f}s")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        methods.append("FloodFill")
        results.append(original_img)
        masks.append(np.zeros(original_img.size[::-1], dtype=np.uint8))
        times.append(0.0)
        results_dicts.append({'success': False, 'error': str(e)})

    # 6. KMeans (PyTorch) - опционально
    print("\n6. KMeans (PyTorch)...")
    try:
        kmeans_dict, kmeans_result, kmeans_mask, kmeans_time = apply_torch_method(
            image_path=image_path,
            method="kmeans_segmentation",
            method_name="KMeans",
            k=3
        )
        methods.append("KMeans")
        results.append(kmeans_result)
        masks.append(kmeans_mask)
        times.append(kmeans_time)
        results_dicts.append(kmeans_dict)
        print(f"   ✓ Успешно за {kmeans_time:.3f}s")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        # Можно пропусить, если не критично

    # Визуализация результатов
    create_timing_comparison_visualization(methods, results, masks, times, original_img, image_path)
    
    # Сохранение результатов
    save_timing_comparison_results(methods, results_dicts, times, image_path)
    
    return methods, results, masks, times

# ============ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ВИЗУАЛИЗАЦИИ И СОХРАНЕНИЯ ============

def create_timing_comparison_visualization(
    methods: List[str], 
    results: List[Image.Image], 
    masks: List[np.ndarray], 
    times: List[float],
    original_img: Image.Image, 
    image_path: str
) -> None:
    """Создает визуализацию сравнения с таймингом"""
    
    n_methods = len(methods)
    fig, axes = plt.subplots(3, n_methods + 1, figsize=(5 * (n_methods + 1), 12))

    # Оригинальное изображение
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[1, 0].axis('off')
    axes[2, 0].axis('off')

    # Результаты методов
    for i, (method, result, mask, exec_time) in enumerate(zip(methods, results, masks, times), 1):
        # Результат сегментации
        axes[0, i].imshow(result)
        axes[0, i].set_title(f'{method}\n({exec_time:.2f}s)')
        axes[0, i].axis('off')
        
        # Маска
        if mask is not None and mask.size > 0:
            if mask.ndim == 2:
                axes[1, i].imshow(mask, cmap='gray')
            elif mask.ndim == 3 and mask.shape[2] == 3:
                axes[1, i].imshow(mask)
            else:
                axes[1, i].text(0.5, 0.5, 'Invalid Mask', 
                              ha='center', va='center', transform=axes[1, i].transAxes)
        else:
            axes[1, i].text(0.5, 0.5, 'No Mask', 
                          ha='center', va='center', transform=axes[1, i].transAxes)
        
        axes[1, i].set_title(f'{method} Mask')
        axes[1, i].axis('off')
        
        # Смешанный результат
        try:
            if isinstance(result, Image.Image):
                result_np = np.array(result)
            else:
                result_np = result
                
            original_np = np.array(original_img)
            if result_np.shape[:2] == original_np.shape[:2]:
                import cv2
                blended = cv2.addWeighted(original_np, 0.5, result_np, 0.5, 0)
                axes[2, i].imshow(blended)
            else:
                axes[2, i].imshow(result_np)
        except Exception as e:
            axes[2, i].text(0.5, 0.5, 'Blend Error', 
                          ha='center', va='center', transform=axes[2, i].transAxes)
        
        axes[2, i].set_title(f'{method} Blended')
        axes[2, i].axis('off')

    plt.suptitle("Segmentation Methods Comparison with Timing", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Сохраняем график
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"./data/methods_comparison_timing_{timestamp}.jpg"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 График сравнения сохранен: {save_path}")
    
    plt.show()

    print("\n" + "="*60)
    print("EXECUTION TIME SUMMARY")
    print("="*60)
    for method, time_taken in zip(methods, times):
        print(f"{method:15}: {time_taken:.3f} seconds")
    print("="*60)

def save_timing_comparison_results(
    methods: List[str], 
    results_dicts: List[Dict], 
    times: List[float], 
    image_path: str
) -> None:
    """Сохраняет результаты сравнения с таймингом"""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"timing_comparison_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"\n💾 Сохранение результатов в {report_dir}/...")
    
    # Сохраняем оригинальное изображение
    original_img = Image.open(image_path).convert('RGB')
    original_img.save(os.path.join(report_dir, "original.jpg"))
    
    # Подготовка данных для сохранения
    summary_data = []
    
    for i, (method, result_dict, exec_time) in enumerate(zip(methods, results_dicts, times)):
        method_dir = os.path.join(report_dir, method.replace(" ", "_").lower())
        os.makedirs(method_dir, exist_ok=True)
        
        # Сохраняем результат если есть
        if result_dict.get('success', False) and 'result' in result_dict:
            result = result_dict['result']
            if isinstance(result, Image.Image):
                result_path = os.path.join(method_dir, "result.jpg")
                result.save(result_path)
            elif isinstance(result, np.ndarray):
                result_path = os.path.join(method_dir, "result.jpg")
                Image.fromarray(result.astype(np.uint8)).save(result_path)
        
        # Сохраняем маску если есть
        if result_dict.get('success', False) and 'mask' in result_dict:
            mask = result_dict['mask']
            if mask is not None and mask.size > 0:
                mask_path = os.path.join(method_dir, "mask.png")
                if mask.ndim == 2:
                    Image.fromarray(mask.astype(np.uint8)).save(mask_path)
                elif mask.ndim == 3 and mask.shape[2] == 3:
                    Image.fromarray(mask.astype(np.uint8)).save(mask_path)
        
        # Собираем статистику с конвертацией типов
        mask_area = 0
        mask_shape = 'N/A'
        if result_dict.get('success', False) and 'mask' in result_dict:
            mask = result_dict['mask']
            if mask is not None and mask.size > 0:
                mask_shape = str(mask.shape)
                if mask.ndim == 2:
                    # Конвертируем np.int64 в int
                    mask_area = int(np.sum(mask > 0))
                elif mask.ndim == 3:
                    # Конвертируем np.int64 в int
                    mask_area = int(np.sum(np.any(mask > 0, axis=2)))
        
        summary_data.append({
            'Method': method,
            'Time (s)': float(exec_time),  # Конвертируем в float
            'Success': bool(result_dict.get('success', False)),  # Конвертируем в bool
            'Mask Area': int(mask_area),  # Конвертируем в int
            'Mask Shape': str(mask_shape),
            'Error': str(result_dict.get('error', ''))  # Конвертируем в str
        })
    
    # Сохраняем JSON с обработкой типов
    import json
    
    # Создаем сериализуемую копию данных
    serializable_data = []
    for data in summary_data:
        serializable_data.append({
            'Method': str(data['Method']),
            'Time (s)': float(data['Time (s)']),
            'Success': bool(data['Success']),
            'Mask Area': int(data['Mask Area']),
            'Mask Shape': str(data['Mask Shape']),
            'Error': str(data['Error'])
        })
    
    json_path = os.path.join(report_dir, "results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 JSON отчет сохранен: {json_path}")
    
    # Сохраняем CSV если pandas доступен
    try:
        import pandas as pd
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Time (s)')
        csv_path = os.path.join(report_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"📄 CSV отчет сохранен: {csv_path}")
        
        # Вывод таблицы в консоль
        print("\n" + "="*70)
        print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
        print("="*70)
        print(df.to_string(index=False))
        
    except ImportError:
        print("Для CSV формата установите pandas: pip install pandas")
        
        # Простой вывод в консоль
        print("\n" + "="*70)
        print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
        print("="*70)
        print(f"{'Метод':<15} {'Время (с)':<10} {'Успех':<8} {'Площадь маски':<15}")
        print("-"*70)
        for data in sorted(summary_data, key=lambda x: x['Time (s)']):
            print(f"{data['Method']:<15} {data['Time (s)']:<10.3f} {str(data['Success']):<8} {data['Mask Area']:<15,}")
    
    # Создаем текстовый отчет
    txt_path = os.path.join(report_dir, "summary.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ОТЧЕТ О СРАВНЕНИИ МЕТОДОВ СЕГМЕНТАЦИИ С ТАЙМИНГОМ\n")
        f.write("="*70 + "\n\n")
        f.write(f"Дата тестирования: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Тестовое изображение: {image_path}\n")
        f.write(f"Всего методов: {len(methods)}\n\n")
        
        # Сортировка по времени выполнения
        sorted_data = sorted(summary_data, key=lambda x: x['Time (s)'])
        
        f.write("РЕЗУЛЬТАТЫ (сортировка по скорости):\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Метод':<15} {'Время (с)':<10} {'Успех':<8} {'Площадь маски':<15}\n")
        f.write("-"*70 + "\n")
        
        for data in sorted_data:
            f.write(f"{data['Method']:<15} {data['Time (s)']:<10.3f} {str(data['Success']):<8} {data['Mask Area']:<15,}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("СТАТИСТИКА:\n")
        f.write("-"*70 + "\n")
        
        # Рассчитываем статистику только для успешных методов
        successful_times = [float(data['Time (s)']) for data in summary_data if data['Success']]
        if successful_times:
            f.write(f"Самый быстрый: {sorted_data[0]['Method']} ({sorted_data[0]['Time (s)']:.3f} с)\n")
            f.write(f"Самый медленный: {sorted_data[-1]['Method']} ({sorted_data[-1]['Time (s)']:.3f} с)\n")
            f.write(f"Среднее время: {np.mean(successful_times):.3f} с\n")
            f.write(f"Стандартное отклонение: {np.std(successful_times):.3f} с\n")
        else:
            f.write("Нет успешных методов для статистики\n")
    
    print(f"\n📝 Текстовый отчет сохранен: {txt_path}")
    print(f"✅ Все результаты сохранены в папке: {report_dir}")
    
    # Вывод итоговой статистики в консоль
    print("\n" + "="*60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("="*60)
    
    successful_methods = [data['Method'] for data in summary_data if data['Success']]
    failed_methods = [data['Method'] for data in summary_data if not data['Success']]
    
    print(f"✓ Успешных методов: {len(successful_methods)}")
    if successful_methods:
        print(f"  {', '.join(successful_methods)}")
    
    print(f"✗ Неуспешных методов: {len(failed_methods)}")
    if failed_methods:
        print(f"  {', '.join(failed_methods)}")
    
    if successful_times:
        print(f"\n📊 Время выполнения:")
        print(f"  Самый быстрый: {min(successful_times):.3f} с")
        print(f"  Самый медленный: {max(successful_times):.3f} с")
        print(f"  Среднее: {np.mean(successful_times):.3f} с")