# NeuralSegmenter.py

# Импорт основных библиотек

from segmenters.BaseSegmenter import BaseSegmenter
from segmenters.NeuralModelFactory import NeuralModelFactory, ModelType
from inference.strategies import INFERENCE_STRATEGIES
from inference.utils import extract_logits_info, analyze_prediction, generate_class_report, export_class_report
from inference.strategies import segment_image_unified as infer_unified
from inference.palettes import ade_palette, get_ade_class_names, get_coco_class_names, coco_palette, get_cityscapes_extended_class_names, cityscapes_extended_palette, get_cityscapes_class_names, cityscapes_palette

from typing import (
    List, Union, Tuple, Dict, Any, TypeVar, Optional, 
    Literal, Protocol, runtime_checkable, overload, TYPE_CHECKING
)
import time
import requests
from io import BytesIO
from PIL import Image

import numpy as np

import torch
import cv2

try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers")

class NeuralSegmenter(BaseSegmenter):
    """
    Универсальный сегментатор с поддержкой множественных нейронных архитектур
    """
    
    def __init__(
        self, 
        model_type: str = "segformer",
        model_name: str = "nvidia/segformer-b5-finetuned-ade-640-640",
        device: str = None,
        local_path: str = None,
        num_classes: int = 150,
        palette: List[List[int]] = None,
        **kwargs
    ) -> None:
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        self.model_type_str = model_type
        self.model_type = ModelType(model_type)
        self.model_name: str = model_name
        self.local_path: str = local_path
        self.device: str = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.params: Dict[str, Any] = kwargs
        self.num_classes = num_classes

        start_time: float = time.time()
        self.model, self.processor, self.model_type_str = NeuralModelFactory.create_model(
            model_type=self.model_type,
            model_name=model_name,
            local_path=local_path,
            device=str(self.device),
            num_classes=num_classes,
            **kwargs
        )
        print(f"Модель загружена за {time.time() - start_time:.4f} секунд")
        
        # Палитра ADE20K
        self.palette: List[List[int]] = palette if palette else self._get_default_palette()
        
        print(f"✅ Нейросетевая модель загружена!")
        print(f"   Тип: {self.model_type_str}")
        print(f"   Источник: {self.local_path if self.local_path else self.model_name}")
        print(f"   Устройство: {self.device}")
        print(f"   Количество классов: {self.num_classes}")

        if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
            print(f"   Количество классов: {len(self.model.config.id2label)}")
            print("Текущие имена классов:")
            for class_id, class_name in self.model.config.id2label.items():
                print(f"{class_id}: {class_name}")

    def _get_default_palette(self) -> List[List[int]]:
        """Палитра ADE20K по умолчанию"""
        return ade_palette()

    @staticmethod
    def get_ade_class_names() -> Dict[int, str]:
        # ADE20K Class Names (0-indexed, 150 classes)
        # Source: http://sceneparsing.csail.mit.edu/
        return get_ade_class_names()
    
    @staticmethod
    def ade_palette() -> List[List[int]]:
        """ADE20K palette that maps each class to RGB values."""
        return ade_palette()
    
    @staticmethod
    def get_coco_class_names() -> Dict[int, str]:
        # COCO Class Names (0-indexed, 80 classes)
        # Source: https://docs.ultralytics.com/datasets/detect/coco/#dataset-yaml
        return get_coco_class_names()
    
    @staticmethod
    def coco_palette() -> List[List[int]]:
        """ADE20K palette that maps each class to RGB values."""
        return coco_palette()
    
    @staticmethod
    def get_cityscapes_extended_class_names() -> Dict[int, str]:
        # Cityscapes Extended (34 classes - includes "grouped" categories)
        return get_cityscapes_extended_class_names()
    
    @staticmethod
    def cityscapes_extended_palette() -> List[List[int]]:
        """ADE20K palette that maps each class to RGB values."""
        return cityscapes_extended_palette()
    
    @staticmethod
    def get_cityscapes_class_names() -> Dict[int, str]:
        # Cityscapes Class Names (0-indexed, 19 classes for semantic segmentation)
        # Source: https://www.cityscapes-dataset.com/
        return get_cityscapes_class_names()
    
    @staticmethod
    def cityscapes_palette() -> List[List[int]]:
        """ADE20K palette that maps each class to RGB values."""
        return cityscapes_palette
    
    @staticmethod
    def get_chexpert_observation_class_names() -> Dict[int, str]:
        # CheXpert Observation Classes (14 labels for classification)
        # Source: https://stanfordmlgroup.github.io/competitions/chexpert/
        chexpert_observation_names: Dict[int, str] = {
            0: "No Finding", 1: "Enlarged Cardiomediastinum", 2: "Cardiomegaly", 3: "Lung Opacity", 4: "Lung Lesion", 5: "Edema", 6: "Consolidation",
            7: "Pneumonia", 8: "Atelectasis", 9: "Pneumothorax", 10: "Pleural Effusion", 11: "Pleural Other", 12: "Fracture", 13: "Support Devices"
        }

        # Для сегментации лёгких (если есть маски):
        chest_segmentation_class_names: Dict[int, str] = {
            0: "background",  # Non-lung area
            1: "lung"         # Lung field (left + right)
        }

        # Проверка
        print(f"✅ CheXpert observations: {len(chexpert_observation_names)} classes")
        print(f"✅ Chest segmentation: {len(chest_segmentation_class_names)} classes (binary)")
        return chexpert_observation_names
    
    @staticmethod
    def chexpert_observation_palette() -> List[List[int]]:
        """ADE20K palette that maps each class to RGB values."""
        return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                [150, 5, 61], [120, 120, 70]]
    
    @staticmethod
    def get_isic_class_names() -> Dict[int, str]:
        # ISIC 2018 Class Names (Binary: skin lesion segmentation)
        # Source: https://challenge.isic-archive.com/
        isic_class_names: Dict[int, str] = {
            0: "background",  # Healthy skin / non-lesion area
            1: "lesion"       # Skin lesion (melanoma, nevus, etc.)
        }

        # Проверка
        print(f"✅ ISIC classes loaded: {len(isic_class_names)} classes (binary)")
        print(f"   Classes: {list(isic_class_names.values())}")
        return isic_class_names
    
    @staticmethod
    def binary_palette() -> List[List[int]]:
        """ADE20K palette that maps each class to RGB values."""
        return [[120, 120, 120], [180, 120, 120]]
    
    @staticmethod
    def _resize_mask_to_original(mask: np.ndarray, target_size: tuple) -> np.ndarray:
        """Утилита для ресайза маски — используется в стратегиях при необходимости"""
        from scipy.ndimage import zoom
        if mask.shape != target_size:
            sh, sw = target_size[0] / mask.shape[0], target_size[1] / mask.shape[1]
            return zoom(mask, (sh, sw), order=0)
        return mask
    
    def load_image(
        self, 
        input_image: Union[str, Image.Image, np.ndarray]
    ) -> Image.Image:
        """Загрузка изображения из различных источников"""
        img: Image.Image
        if isinstance(input_image, str):
            if input_image.startswith(('http://', 'https://')):
                resp = requests.get(input_image)
                img = Image.open(BytesIO(resp.content)).convert("RGB")
            else:
                img = Image.open(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            img = input_image.convert("RGB")
        elif isinstance(input_image, np.ndarray):
            if len(input_image.shape) == 2:
                img = Image.fromarray(input_image).convert("RGB")
            elif len(input_image.shape) == 3:
                # RGB или BGR
                if input_image.shape[2] == 3:
                    img = Image.fromarray(input_image)
                elif input_image.shape[2] == 4:
                    # RGBA
                    img = Image.fromarray(input_image).convert("RGB")
                else:
                    raise ValueError(f"Неподдерживаемое количество каналов: {input_image.shape[2]}")
        else:
            raise ValueError("Unsupported input type. Provide a file path, URL, or PIL.Image.")
        return img
    
    def predict_segmentation_map(
        self, 
        input_image: Union[str, Image.Image],
        verbose: bool = True,
        class_names: dict = None,
        gt_mask = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Предсказание карты сегментации с опциональной вербозностью и метриками.
        Инференс полностью делегируется стратегиям из inference/strategies.py
        ДЕЛЕГИРУЕТ standalone функции из strategies.py
        
        Returns:
            Tuple[np.ndarray, Dict]: (seg_map, result_dict)
        """
        # Вызываем standalone функцию
        overlay, result_info = infer_unified(
            model=self.model,
            processor=self.processor,
            image_input=input_image,
            model_type=self.model_type_str,
            alpha=0.5,
            palette=self.palette,
            device=str(self.device),
            verbose=verbose,
            num_classes=self.num_classes,
            class_names=class_names,
            gt_mask=gt_mask
        )
        
        # Возвращаем маску + инфо (как было раньше)
        return result_info["mask"], result_info

    def segment_image_unified(
        self,
        input_image: Union[str, Image.Image],
        alpha: float = 0.5,
        verbose: bool = True,
        class_names: dict = None,
        gt_mask = None
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Универсальная функция сегментации для любой архитектуры.
        """

        # Получаем маску и инфо
        return infer_unified(
            model=self.model,
            processor=self.processor,
            image_input=input_image,
            model_type=self.model_type_str,
            alpha=alpha,
            palette=self.palette,
            device=str(self.device),
            verbose=verbose,
            num_classes=self.num_classes,
            class_names=class_names,
            gt_mask=gt_mask
        )
    
    def prepare_mask_for_overlay(self, mask_input) -> np.ndarray:
        """
        Конвертирует маску в 2D numpy array для create_overlay.
        
        Handles:
        - PIL Image (RGB or L)
        - numpy array with extra dimensions
        - RGB label images (converts to single channel)
        """
        
        # Конвертируем PIL → numpy если нужно
        if isinstance(mask_input, Image.Image):
            mask = np.array(mask_input)
        else:
            mask = np.array(mask_input)
        
        if mask.ndim == 3:
            if mask.shape[2] == 1:
                # (H, W, 1) → (H, W)
                mask = mask.squeeze(2)
            elif mask.shape[2] == 3:
                # RGB изображение → нужно конвертировать в классы
                # Для Cityscapes: используем первый канал или конвертируем через палитру
                print(f"⚠️  RGB mask detected, using first channel")
                mask = mask[:, :, 0]  # или используйте proper label conversion
            else:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")
        elif mask.ndim > 3:
            mask = np.squeeze(mask)
        
        # Финальная проверка
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D after processing, got {mask.ndim}D")
        
        return mask
    
    def segment_image(
        self, 
        input_image: Union[str, Image.Image], 
        alpha: float = 0.5
    ) -> Image.Image:
        """
        Performs semantic segmentation on an image and returns an overlay mask.

        Args:
            input_image (str or PIL.Image): Path to an image file, URL, or a PIL Image instance.
            alpha (float): Blending factor for overlay. 0 = only original, 1 = only mask.

        Returns:
            PIL.Image: The original image blended with the segmentation mask.
        """
        img: Image.Image = self.load_image(input_image)
        
        # Получаем карту сегментации
        seg_map, _ = self.predict_segmentation_map(image, verbose=False)
        
        # Create color mask
        palette_array: np.ndarray = np.array(self.palette, dtype=np.uint8)
        h, w = seg_map.shape
        color_mask = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette_array[:seg_map.max()+1]):
            color_mask[seg_map == label] = color
        
        # Blend original and mask
        orig_arr: np.ndarray = np.array(img.convert("RGB"))
        overlay: np.ndarray = (orig_arr * (1 - alpha) + color_mask * alpha).astype(np.uint8)
        return Image.fromarray(overlay)
    
    def segment(
        self, 
        image: Union[str, Image.Image], 
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Основной метод сегментации.
        
        Args:
            image: Входное изображение (RGB, grayscale или любой формат)
        
        Returns:
            np.ndarray: Бинарная маска сегментации (0-255)
        """
        result_img: Image.Image = self.segment_image(image, alpha)
        return np.array(result_img)
    
    def segment_with_mask(
        self, 
        image: Union[str, Image.Image], 
        alpha: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сегментация с возвратом визуализации и маски.
        
        Args:
            image: Входное изображение
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Визуализация: исходное изображение с наложенной маской (0–255, RGB).
                - Маска: бинарная маска (0–255, grayscale).
        """
        start_time: float = time.time()
        
        # Получаем сегментированное изображение
        result_img: Image.Image = self.segment_image(image, alpha)
        result_np: np.ndarray = np.array(result_img)
        
        # Получаем карту сегментации
        seg_map, _ = self.predict_segmentation_map(image, verbose=False)

        unique_classes = np.unique(seg_map)
        print("Предугаданные классы:", unique_classes)

        # Проверяем количество пикселей для каждого класса
        for cls in unique_classes:
            count = (seg_map == cls).sum()
            print(f"Class {cls}: {count} pixels")
        
        # Создаем бинарную маску (все, что не фон)
        mask: np.ndarray = (seg_map > 0).astype(np.uint8) * 255
        
        # if len(result_np.shape) == 2:
        #     result_np = cv2.cvtColor(result_np, cv2.COLOR_GRAY2RGB)
        
        # overlay = result_np.copy()
        # mask_bool: np.ndarray = mask > 0
        # overlay[mask_bool] = [255, 0, 0]
        # result = cv2.addWeighted(result_np, alpha, overlay, 1 - alpha, 0)
        
        print(f"Neural segmentation completed in {time.time() - start_time:.2f}s")
        
        return result_np, mask
    
    def detailed_segmentation(
        self, 
        input_image: Union[str, Image.Image]
    ) -> Dict[str, Any]:
        """
        Детальная сегментация с возвратом всех промежуточных результатов
        """
        img: Image.Image = self.load_image(input_image)
        
        # Получаем карту сегментации
        seg_map, _ = self.predict_segmentation_map(image, verbose=False)
        
        # Создаем цветную сегментацию
        palette_array: np.ndarray = np.array(self.palette, dtype=np.uint8)
        color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette_array):
            color_seg[seg_map == label] = color
        
        # Конвертируем из BGR в RGB
        color_seg: np.ndarray = color_seg[..., ::-1]
        
        # Создаем наложение
        orig_arr: np.ndarray = np.array(img)
        overlay: np.ndarray = orig_arr * 0.2 + color_seg * 0.8
        overlay: np.ndarray = overlay.astype(np.uint8)

        # Анализ распределения классов
        unique_classes: np.ndarray
        counts: np.ndarray
        unique_classes, counts = np.unique(seg_map, return_counts=True)
        class_distribution = {}
        total_pixels: int = seg_map.size
        
        for cls, count in zip(unique_classes, counts):
            class_name: str = self.model.config.id2label.get(cls, f"Class_{cls}")
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
    
    def get_class_info(self) -> Dict[str, Any]:
        """Получить информацию о классах модели"""
        if not hasattr(self, 'model'):
            return {'error': 'Model not initialized'}
        
        # 🔥 HuggingFace модели
        if hasattr(self.model, 'config'):
            config = self.model.config
            if hasattr(config, 'num_labels'):
                return {
                    'num_classes': int(config.num_labels),
                    'id2label': getattr(config, 'id2label', {}),
                    'label2id': getattr(config, 'label2id', {})
                }
        
        # 🔥 SMP / Torchvision модели: ищем последний Conv2d
        for module in reversed(list(self.model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return {
                    'num_classes': int(module.out_channels),
                    'id2label': {},  # Нет mapping для SMP
                    'label2id': {}
                }
        
        # Fallback
        return {
            'num_classes': self.num_classes,
            'id2label': {},
            'label2id': {}
        }