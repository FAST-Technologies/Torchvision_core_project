import torch
import cv2
import numpy as np
# from torchvision import transforms
from abc import ABC, abstractmethod
from PIL import Image
from typing import Union, Tuple


class BaseSegmenter(ABC):
    """Базовый класс для всех методов сегментации"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        
    @abstractmethod
    def segment(self, image: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> np.ndarray:
        """Основной метод сегментации"""
        pass
    
    @abstractmethod
    def segment_with_mask(self, image: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """Сегментация с возвратом маски"""
        pass
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> np.ndarray:
        """Предобработка изображения"""
        if isinstance(image, str):
            # Загрузка из файла
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Не удалось загрузить изображение: {image}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            # PIL Image
            return np.array(image)
        elif isinstance(image, np.ndarray):
            # NumPy array
            return image.copy()
        elif isinstance(image, torch.Tensor):
            # PyTorch tensor
            return image.permute(1, 2, 0).cpu().numpy()
        else:
            raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")
    
    def visualize(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, 
                  overlay_color: Tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
        """Визуализация результата сегментации"""
        overlay = image.copy()
        overlay[mask > 0] = overlay_color
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        return Image.fromarray(result)
    
    def __call__(self, image: Union[str, np.ndarray, Image.Image, torch.Tensor], 
                 return_mask: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Вызов метода сегментации"""
        if return_mask:
            return self.segment_with_mask(image)
        return self.segment(image)