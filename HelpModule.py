
from BaseSegmenter import BaseSegmenter
from cv2_sklearn_segmenter import CV2SklearnSegmenter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Union, Any, Dict

def create_segmentation_pipeline():
    """Создание пайплайна сегментации с различными методами"""
    
    class SegmentationPipeline:
        """Пайплайн для последовательного применения методов сегментации"""
        
        def __init__(self) -> None:
            self.steps = []
            
        def add_step(self, 
                     name: str, 
                     segmenter: BaseSegmenter, 
                     params: Dict = None
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
        ) -> Dict[str, Any]:
            """Запуск пайплайна"""
            results = {}
            current_image = image
            
            for step in self.steps:
                print(f"Выполнение шага: {step['name']}")
                
                # Обновляем параметры сегментатора
                if step['params']:
                    for key, value in step['params'].items():
                        setattr(step['segmenter'], key, value)
                
                # Выполняем сегментацию
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
                              results: Dict[str, Any], 
                              original_image
        ) -> None:
            """Визуализация результатов пайплайна"""
            n_steps = len(results)
            
            fig, axes = plt.subplots(2, n_steps + 1, figsize=(5 * (n_steps + 1), 10))
            
            # Оригинальное изображение
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
            plt.show()
    
    return SegmentationPipeline()


# Пример создания пайплайна
def example_pipeline() -> None:
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