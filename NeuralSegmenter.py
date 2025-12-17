# neural_segmenter.py
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Union, Tuple, List, Dict, Any, Optional
import requests
from io import BytesIO
import cv2
import time

from BaseSegmenter import BaseSegmenter

try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers")


class NeuralSegmenter(BaseSegmenter):
    """Класс для нейросетевой сегментации (расширенный вариант)"""
    
    def __init__(self, 
                 model_name: str = "nvidia/segformer-b5-finetuned-ade-640-640",
                 device: str = None,
                 local_path: str = None):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        self.model_name = model_name
        self.local_path = local_path
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Инициализация модели и процессора
        self._initialize_model()
        
        # Палитта ADE20K
        self.palette = self.ade_palette()
        
        print(f"✅ Нейросетевая модель загружена!")
        print(f"   Источник: {self.model_name if not self.local_path else self.local_path}")
        print(f"   Устройство: {self.device}")
        print(f"   Количество классов: {len(self.model.config.id2label)}")
    
    def _initialize_model(self):
        """Инициализация модели и процессора"""
        start_time = time.time()
        
        try:
            # Сначала создаем процессор
            self.processor = SegformerImageProcessor(do_resize=False)
            
            # Загружаем модель
            if self.local_path:
                print(f"Загрузка модели из локального пути: {self.local_path}")
                self.model = SegformerForSemanticSegmentation.from_pretrained(self.local_path)
            else:
                print(f"Загрузка модели из Hugging Face: {self.model_name}")
                self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
            
            # Переносим модель на устройство
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Модель загружена за {time.time() - start_time:.2f} секунд")
            
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {e}")
    
    @staticmethod
    def ade_palette() -> List[List[int]]:
        """ADE20K palette that maps each class to RGB values."""
        return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                [102, 255, 0], [92, 0, 255]]
    
    def load_image(self, input_image: Union[str, Image.Image]) -> Image.Image:
        """Загрузка изображения из различных источников"""
        if isinstance(input_image, str):
            if input_image.startswith(('http://', 'https://')):
                resp = requests.get(input_image)
                img = Image.open(BytesIO(resp.content)).convert("RGB")
            else:
                img = Image.open(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            img = input_image.convert("RGB")
        else:
            raise ValueError("Unsupported input type. Provide a file path, URL, or PIL.Image.")
        
        return img
    
    def segment(self, 
                image: Union[str, Image.Image], 
                alpha: float = 0.5) -> np.ndarray:
        """Основной метод сегментации (упрощенный вариант)"""
        result_img = self.segment_image(image, alpha)
        return np.array(result_img)
    
    def segment_with_mask(self, 
                          image: Union[str, Image.Image], 
                          alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Сегментация с возвратом маски и обработанного изображения"""
        start_time = time.time()
        
        # Получаем сегментированное изображение
        result_img = self.segment_image(image, alpha)
        result_np = np.array(result_img)
        
        # Получаем карту сегментации
        seg_map = self.predict_segmentation_map(image)
        
        # Создаем бинарную маску (все, что не фон)
        mask = (seg_map > 0).astype(np.uint8) * 255
        
        # Если нужна визуализация с красным оверлеем (как в других методах)
        if len(result_np.shape) == 2:
            result_np = cv2.cvtColor(result_np, cv2.COLOR_GRAY2RGB)
        
        overlay = result_np.copy()
        mask_bool = mask > 0
        overlay[mask_bool] = [255, 0, 0]
        result = cv2.addWeighted(result_np, 0.5, overlay, 0.5, 0)
        
        print(f"Neural segmentation completed in {time.time() - start_time:.2f}s")
        
        return result, mask
    
    def segment_image(self, 
                      input_image: Union[str, Image.Image], 
                      alpha: float = 0.5) -> Image.Image:
        """
        Performs semantic segmentation on an image and returns an overlay mask.

        Args:
            input_image (str or PIL.Image): Path to an image file, URL, or a PIL Image instance.
            alpha (float): Blending factor for overlay. 0 = only original, 1 = only mask.

        Returns:
            PIL.Image: The original image blended with the segmentation mask.
        """
        # Load image
        img = self.load_image(input_image)
        
        # Preprocess
        pixel_values = self.processor(img, return_tensors="pt").pixel_values.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values)
        
        # Post-process to get mask
        seg_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[img.size[::-1]]
        )[0].cpu().numpy()
        
        # Create color mask
        palette_array = np.array(self.palette, dtype=np.uint8)
        color_mask = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette_array):
            color_mask[seg_map == label] = color
        
        # Blend original and mask
        orig_arr = np.array(img)
        overlay = (orig_arr * (1 - alpha) + color_mask * alpha).astype(np.uint8)
        return Image.fromarray(overlay)
    
    def predict_segmentation_map(self, 
                                 input_image: Union[str, Image.Image]) -> np.ndarray:
        """Предсказание карты сегментации"""
        img = self.load_image(input_image)
        
        pixel_values = self.processor(img, return_tensors="pt").pixel_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(pixel_values)
        
        seg_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[img.size[::-1]]
        )[0].cpu().numpy()
        
        return seg_map
    
    def detailed_segmentation(self, 
                              input_image: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Детальная сегментация с возвратом всех промежуточных результатов
        """
        img = self.load_image(input_image)
        
        # Получаем карту сегментации
        seg_map = self.predict_segmentation_map(img)
        
        # Создаем цветную сегментацию
        palette_array = np.array(self.palette, dtype=np.uint8)
        color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette_array):
            color_seg[seg_map == label] = color
        
        # Конвертируем из BGR в RGB
        color_seg = color_seg[..., ::-1]
        
        # Создаем наложение
        orig_arr = np.array(img)
        overlay = orig_arr * 0.5 + color_seg * 0.5
        overlay = overlay.astype(np.uint8)

        # Анализ распределения классов
        unique_classes, counts = np.unique(seg_map, return_counts=True)
        class_distribution = {}
        total_pixels = seg_map.size
        
        for cls, count in zip(unique_classes, counts):
            class_name = self.model.config.id2label.get(cls, f"Class_{cls}")
            percentage = (count / total_pixels) * 100
            class_distribution[class_name] = {
                'class_id': int(cls),
                'pixel_count': int(count),
                'percentage': float(percentage)
            }
        
        return {
            'original': img,
            'segmentation_map': seg_map,
            'color_seg': color_seg,
            'overlay': overlay,
            'class_distribution': class_distribution,
            'total_classes': len(unique_classes)
        }
    
    def get_class_info(self):
        """Получить информацию о классах модели"""
        if hasattr(self, 'model') and hasattr(self.model, 'config'):
            return {
                'num_classes': self.model.config.num_labels,
                'id2label': self.model.config.id2label,
                'label2id': self.model.config.label2id
            }
        return None
    
    def visualize_segmentation(self, 
                               input_image: Union[str, Image.Image],
                               alpha: float = 0.5,
                               figsize: Tuple[int, int] = (15, 5)):
        """Базовая визуализация сегментации"""
        img = self.load_image(input_image)
        result_img = self.segment_image(input_image, alpha)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(result_img)
        axes[1].set_title(f"Neural Segmentation (alpha={alpha})")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return result_img