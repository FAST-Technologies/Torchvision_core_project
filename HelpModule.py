
from BaseSegmenter import BaseSegmenter
from cv2SklearnSegmenter import CV2SklearnSegmenter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Union, Any, Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from BaseSegmenter import BaseSegmenter

def create_segmentation_pipeline():
    """Создание пайплайна сегментации с различными методами"""
    
    class SegmentationPipeline:
        """Пайплайн для последовательного применения методов сегментации"""
        
        def __init__(self) -> None:
            self.steps: List[Dict[str, Any]] = []
            
        def add_step(self, 
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
            
        def run(self, 
                image: Union[str, np.ndarray, Image.Image], 
                visualize: bool = True
        ) -> Dict[str, Dict[str, Any]]:
            """Запуск пайплайна"""
            results: Dict[str, Dict[str, Any]] = {}
            current_image: Union[str, np.ndarray, Image.Image] = image
            
            for step in self.steps:
                print(f"Выполнение шага: {step['name']}")
                
                # Обновляем параметры сегментатора
                if step['params']:
                    for key, value in step['params'].items():
                        setattr(step['segmenter'], key, value)
                
                # Выполняем сегментацию
                result: np.ndarray
                mask: np.ndarray
                result, mask = step['segmenter'].segment_with_mask(current_image)
                results[step['name']] = {
                    'result': result,
                    'mask': mask,
                    'segmenter': step['segmenter']
                }
                
                # Для следующего шага используем маску
                current_image = mask
            
            if visualize:
                self.visualize_results(results, image)
            
            return results
        
        def visualize_results(self, 
                              results: Dict[str, Dict[str, Any]], 
                              original_image: Union[str, np.ndarray, Image.Image]
        ) -> None:
            """Визуализация результатов пайплайна"""
            n_steps: int = len(results)
            
            fig_height: int = 10
            fig_width: int = 5 * (n_steps + 1)
            
            fig, axes = plt.subplots(2, n_steps + 1, figsize=(fig_width, fig_height))
            
            # Оригинальное изображение
            orig_img: Image.Image
            if isinstance(original_image, str):
                orig_img = Image.open(original_image).convert('RGB')
            elif isinstance(original_image, Image.Image):
                orig_img = original_image
            elif isinstance(original_image, np.ndarray):
                orig_img = Image.fromarray(original_image.astype(np.uint8))
            else:
                raise TypeError(f"Неподдерживаемый тип изображения: {type(original_image)}")
            
            axes[0, 0].imshow(orig_img)
            axes[0, 0].set_title("Original")
            axes[0, 0].axis('off')
            
            axes[1, 0].axis('off')
            
            # Результаты каждого шага
            for i, (step_name, step_result) in enumerate(results.items(), 1):
                result_data: np.ndarray = step_result['result']
                mask_data: np.ndarray = step_result['mask']
                
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
        
        def get_step(self, name: str) -> Optional[Dict[str, Any]]:
            """Получить информацию о конкретном шаге по имени"""
            for step in self.steps:
                if step['name'] == name:
                    return step
            return None
        
        def remove_step(self, name: str) -> bool:
            """Удалить шаг из пайплайна по имени"""
            for i, step in enumerate(self.steps):
                if step['name'] == name:
                    self.steps.pop(i)
                    return True
            return False
        
        def clear_pipeline(self) -> None:
            """Очистить весь пайплайн"""
            self.steps = []
        
        def save_visualization(self, 
                               results: Dict[str, Dict[str, Any]], 
                               original_image: Union[str, np.ndarray, Image.Image],
                               save_path: str = "pipeline_results.jpg"
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


# Пример создания пайплайна
def example_pipeline() -> Tuple[Any, Dict[str, Dict[str, Any]]]:
    """Пример пайплайна сегментации"""
    
    pipeline = create_segmentation_pipeline()
    
    # Шаг 1: Предварительная обработка - выделение краев
    pipeline.add_step(
        "Canny_Edge_Detection",
        CV2SklearnSegmenter("canny_edge", low=30, high=100)
    )
    
    # Шаг 2: Уточнение с помощью адаптивного порога
    pipeline.add_step(
        "Adaptive_Threshold_Refinement",
        CV2SklearnSegmenter("adaptive_thresholding", block_size=15, C=5)
    )
    
    # Шаг 3: Region Growing для объединения регионов
    pipeline.add_step(
        "Region_Growing",
        CV2SklearnSegmenter("region_growing", tolerance=25)
    )
    
    # Шаг 4: Постобработка с помощью Watershed
    pipeline.add_step(
        "Watershed_Segmentation",
        CV2SklearnSegmenter("watershed")
    )
    
    # Запуск пайплайна
    image_path = "test_image.jpg"
    results = pipeline.run(image_path)
    
    return pipeline, results

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
        'execution_info': {}
    }
    
    for step_name, step_data in results.items():
        mask: np.ndarray = step_data['mask']
        
        # Статистика маски
        mask_stats = {
            'shape': mask.shape,
            'dtype': str(mask.dtype),
            'min_value': float(mask.min()),
            'max_value': float(mask.max()),
            'mean_value': float(mask.mean()),
            'nonzero_pixels': int(np.count_nonzero(mask)),
            'total_pixels': int(mask.size),
            'coverage_percentage': float(np.count_nonzero(mask) / mask.size * 100)
        }
        
        analysis['mask_statistics'][step_name] = mask_stats
        
        # Информация о сегментаторе
        segmenter: BaseSegmenter = step_data['segmenter']
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