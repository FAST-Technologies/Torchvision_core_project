# BaseSegmenter.py
import torch
import cv2
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
from typing import Union, Tuple, Dict
from segmentation_metrics import SegmentationMetrics

class BaseSegmenter(ABC):
    """Базовый класс для всех методов сегментации"""
    
    def __init__(self) -> None:
        self.name: str = self.__class__.__name__
        
    @abstractmethod
    def segment(self, 
                image: Union[str, np.ndarray, Image.Image, torch.Tensor]
    ) -> np.ndarray:
        """Основной метод сегментации"""
        pass
    
    @abstractmethod
    def segment_with_mask(self, 
                          image: Union[str, np.ndarray, Image.Image, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Сегментация с возвратом маски"""
        pass
    
    def preprocess_image(self, 
                         image: Union[str, np.ndarray, Image.Image, torch.Tensor],
                         as_gray: bool = False
    ) -> np.ndarray:
        """Предобработка изображения"""
        if isinstance(image, str):
            # Загрузка из файла
            if as_gray:
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Не удалось загрузить изображение: {image}")
                return img  # Уже в GRAY
            else:
                img = cv2.imread(image)
                if img is None:
                    raise ValueError(f"Не удалось загрузить изображение: {image}")
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR→RGB
        elif isinstance(image, Image.Image):
            # PIL Image
            if as_gray:
                return np.array(image.convert('L'))  # 'L' = grayscale
            else:
                return np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            # NumPy array
            if as_gray and len(image.shape) == 3:
                # Конвертируем RGB/BGR в GRAY
                if image.shape[2] == 3:
                    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return image.copy()
        elif isinstance(image, torch.Tensor):
            # PyTorch tensor
            img_np = image.permute(1, 2, 0).cpu().numpy()
            if as_gray and img_np.shape[2] == 3:
                return cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            return img_np
        else:
            raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")
    
    def visualize(self, 
                  image: np.ndarray, 
                  mask: np.ndarray, 
                  alpha: float = 0.5, 
                  overlay_color: Tuple[int, int, int] = (255, 0, 0)
    ) -> Image.Image:
        """Визуализация результата сегментации"""
        overlay: np.ndarray = image.copy()
        overlay[mask > 0] = overlay_color
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        return Image.fromarray(result)
    
    def evaluate_metrics(self, 
                         pred_mask: np.ndarray, 
                         gt_mask: np.ndarray,
                         threshold: float = 0.5) -> Dict[str, float]:
        """
        Оценка качества сегментации с помощью различных метрик
        
        Args:
            pred_mask: Предсказанная маска
            gt_mask: Ground truth маска
            threshold: Порог для бинаризации
            
        Returns:
            Словарь с метриками
        """
        return SegmentationMetrics.calculate_all_metrics(pred_mask, gt_mask, threshold)
    
    def segment_and_evaluate(self,
                            image: Union[str, np.ndarray, Image.Image, torch.Tensor],
                            gt_mask: np.ndarray,
                            threshold: float = 0.5) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Выполняет сегментацию и сразу оценивает результат
        
        Args:
            image: Входное изображение
            gt_mask: Ground truth маска
            threshold: Порог для бинаризации
            
        Returns:
            (метрики, предсказанная маска)
        """
        pred_mask = self.segment(image)
        
        # Конвертируем в uint8 если нужно
        if pred_mask.dtype != np.uint8:
            if pred_mask.max() <= 1.0:
                pred_mask = (pred_mask * 255).astype(np.uint8)
            else:
                pred_mask = pred_mask.astype(np.uint8)
        
        metrics = self.evaluate_metrics(pred_mask, gt_mask, threshold)
        
        return metrics, pred_mask
    
    
    def __call__(self, 
                 image: Union[str, np.ndarray, Image.Image, torch.Tensor], 
                 return_mask: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Вызов метода сегментации"""
        if return_mask:
            return self.segment_with_mask(image)
        return self.segment(image)