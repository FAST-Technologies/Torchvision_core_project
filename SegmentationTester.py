from base_segmenter import BaseSegmenter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from typing import Union, Tuple, List, Dict, Any
import time

class SegmentationTester:
    """Класс для тестирования и сравнения методов сегментации"""
    
    def __init__(self) -> None:
        self.methods = {}
        self.results = {}
    
    def add_method(self, 
                   name: str, 
                   segmenter: BaseSegmenter
    ) -> None:
        """Добавление метода сегментации"""
        self.methods[name] = segmenter
    
    def test_single_method(self, 
                           image: Union[str, np.ndarray, Image.Image], 
                           method_name: str, 
                           save_path: str = None
    ) -> Dict[str, Any]:
        """Тестирование одного метода"""
        if method_name not in self.methods:
            raise ValueError(f"Метод {method_name} не найден")
        
        segmenter = self.methods[method_name]
        
        # Измеряем время выполнения
        start_time = time.time()
        result, mask = segmenter.segment_with_mask(image)
        execution_time = time.time() - start_time
        
        # Сохранение результатов
        if save_path:
            result_pil = Image.fromarray(result.astype(np.uint8))
            result_pil.save(save_path)
            print(f"Результат сохранен: {save_path}")
        
        # Статистика
        mask_area = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        return {
            'method': method_name,
            'result': result,
            'mask': mask,
            'time': execution_time,
            'mask_area': mask_area,
            'mask_percentage': (mask_area / total_pixels) * 100
        }
    
    def compare_methods(self, 
                        image: Union[str, np.ndarray, Image.Image], 
                        method_names: List[str] = None, 
                        figsize: Tuple[int, int] = (20, 15)
    ) -> Dict[str, Any]:
        """Сравнение нескольких методов"""
        if method_names is None:
            method_names = list(self.methods.keys())
        
        results = {}
        
        # Создаем фигуру для отображения
        n_methods = len(method_names)
        n_cols = min(4, n_methods + 1)  # +1 для оригинального изображения
        n_rows = (n_methods + n_cols) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        # Оригинальное изображение
        if isinstance(image, str):
            original_img = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            original_img = image
        else:
            original_img = Image.fromarray(image.astype(np.uint8))
        
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Тестируем каждый метод
        for i, method_name in enumerate(method_names, 1):
            if i >= len(axes):
                break
            
            try:
                result = self.test_single_method(image, method_name)
                results[method_name] = result
                
                # Отображение результата
                axes[i].imshow(result['result'])
                axes[i].set_title(f"{method_name}\n({result['time']:.2f}s, {result['mask_percentage']:.1f}%)")
                axes[i].axis('off')
                
                print(f"{method_name}: {result['time']:.2f}s, {result['mask_percentage']:.1f}% площади")
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error:\n{str(e)[:50]}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{method_name}\n(Error)")
                axes[i].axis('off')
                print(f"Ошибка в методе {method_name}: {e}")
        
        # Скрываем пустые оси
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def benchmark_methods(self, 
                          image: Union[str, np.ndarray, Image.Image], 
                          n_runs: int = 3
    ) -> pd.DataFrame:
        """Бенчмарк методов (требует pandas)"""
        import pandas as pd
        
        benchmark_results = []
        
        for method_name in self.methods.keys():
            times = []
            for _ in range(n_runs):
                start_time = time.time()
                self.methods[method_name].segment(image)
                times.append(time.time() - start_time)
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            
            # Получаем маску для статистики
            _, mask = self.methods[method_name].segment_with_mask(image)
            mask_area = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            
            benchmark_results.append({
                'Method': method_name,
                'Mean Time (s)': f"{mean_time:.3f} ± {std_time:.3f}",
                'Mask Area': mask_area,
                'Mask %': f"{(mask_area / total_pixels * 100):.1f}%",
                'Min Time (s)': f"{min(times):.3f}",
                'Max Time (s)': f"{max(times):.3f}"
            })
        
        df = pd.DataFrame(benchmark_results)
        df = df.sort_values('Mean Time (s)')
        
        print("Бенчмарк методов сегментации:")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        return df
    
    def visualize_comparison(self, 
                             results: Dict[str, Dict], 
                             show_masks: bool = True
    ) -> None:
        """Визуализация сравнения результатов"""
        n_methods = len(results)
        
        if show_masks:
            fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 10))
            
            for i, (method_name, result) in enumerate(results.items()):
                # Результат
                axes[0, i].imshow(result['result'])
                axes[0, i].set_title(f"{method_name}\n{result['time']:.2f}s")
                axes[0, i].axis('off')
                
                # Маска
                axes[1, i].imshow(result['mask'], cmap='gray')
                axes[1, i].set_title(f"Mask\n{result['mask_percentage']:.1f}%")
                axes[1, i].axis('off')
        else:
            fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
            
            for i, (method_name, result) in enumerate(results.items()):
                axes[i].imshow(result['result'])
                axes[i].set_title(f"{method_name}\n{result['time']:.2f}s")
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, 
                     results: Dict[str, Dict], 
                     output_dir: str = "segmentation_results"
    ) -> None:
        """Сохранение результатов всех методов"""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for method_name, result in results.items():
            # Сохраняем результат
            result_path = os.path.join(output_dir, f"{method_name}_result.jpg")
            result_img = Image.fromarray(result['result'].astype(np.uint8))
            result_img.save(result_path)
            
            # Сохраняем маску
            mask_path = os.path.join(output_dir, f"{method_name}_mask.png")
            mask_img = Image.fromarray(result['mask'].astype(np.uint8))
            mask_img.save(mask_path)
            
            # Сохраняем статистику
            stats_path = os.path.join(output_dir, f"{method_name}_stats.txt")
            with open(stats_path, 'w') as f:
                f.write(f"Method: {method_name}\n")
                f.write(f"Execution Time: {result['time']:.3f}s\n")
                f.write(f"Mask Area: {result['mask_area']} pixels\n")
                f.write(f"Mask Percentage: {result['mask_percentage']:.2f}%\n")
        
        print(f"Результаты сохранены в директории: {output_dir}")