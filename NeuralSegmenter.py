# NeuralSegmenter.py

# Импорт основных библиотек

from BaseSegmenter import BaseSegmenter

from typing import (
    List, Union, Tuple, Dict, Any, TypeVar, Optional, 
    Literal, Protocol, runtime_checkable, overload, TYPE_CHECKING
)
import time
import requests
from io import BytesIO
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
import cv2
from sklearn.metrics import confusion_matrix

try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers")

class NeuralSegmenter(BaseSegmenter):
    """Класс для нейросетевой сегментации"""
    
    def __init__(
        self, 
        model_name: str = "nvidia/segformer-b5-finetuned-ade-640-640",
        device: str = None,
        local_path: str = None
    ) -> None:
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        self.model_name: str = model_name
        self.local_path: str = local_path
        self.device: str
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self._initialize_model()
        
        # Палитта ADE20K
        self.palette: List[List[int]] = self.ade_palette()
        
        print(f"✅ Нейросетевая модель загружена!")
        print(f"   Источник: {self.model_name if not self.local_path else self.local_path}")
        print(f"   Устройство: {self.device}")
        print(f"   Количество классов: {len(self.model.config.id2label)}")
        print("Текущие имена классов:")
        for class_id, class_name in self.model.config.id2label.items():
            print(f"{class_id}: {class_name}")
    
    def _initialize_model(self) -> None:
        """Инициализация модели и процессора"""
        start_time: float = time.time()
        try:
            self.processor: SegformerImageProcessor = SegformerImageProcessor(do_resize=False)
            self.model: SegformerForSemanticSegmentation
            if self.local_path:
                print(f"Загрузка модели из локального пути: {self.local_path}")
                self.model = SegformerForSemanticSegmentation.from_pretrained(self.local_path)
            else:
                print(f"Загрузка модели из Hugging Face: {self.model_name}")
                self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Модель загружена за {time.time() - start_time:.4f} секунд")
            print(f"Модель загружена за {time.time() - start_time:.4f} секунд")
            print(f"Текущая конфигурация модели: {self.model.config}")
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
    
    def load_image(
        self, 
        input_image: Union[str, Image.Image]
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
        else:
            raise ValueError("Unsupported input type. Provide a file path, URL, or PIL.Image.")
        return img
    
    def segment(
        self, 
        image: Union[str, Image.Image], 
        alpha: float = 0.5
    ) -> np.ndarray:
        """Основной метод сегментации"""
        result_img: Image.Image = self.segment_image(image, alpha)
        return np.array(result_img)
    
    def segment_with_mask(
        self, 
        image: Union[str, Image.Image], 
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Сегментация с возвратом маски и обработанного изображения"""
        start_time: float = time.time()
        
        # Получаем сегментированное изображение
        result_img: Image.Image = self.segment_image(image, alpha)
        result_np: np.ndarray = np.array(result_img)
        
        # Получаем карту сегментации
        seg_map: np.ndarray = self.predict_segmentation_map(image)

        unique_classes = np.unique(seg_map)
        print("Предугаданные классы:", unique_classes)

        # Проверяем количество пикселей для каждого класса
        for cls in unique_classes:
            count = (seg_map == cls).sum()
            print(f"Class {cls}: {count} pixels")
        
        # Создаем бинарную маску (все, что не фон)
        mask: np.ndarray = (seg_map > 0).astype(np.uint8) * 255
        
        if len(result_np.shape) == 2:
            result_np = cv2.cvtColor(result_np, cv2.COLOR_GRAY2RGB)
        
        overlay = result_np.copy()
        mask_bool: np.ndarray = mask > 0
        overlay[mask_bool] = [255, 0, 0]
        result = cv2.addWeighted(result_np, 0.2, overlay, 0.8, 0)
        
        print(f"Neural segmentation completed in {time.time() - start_time:.2f}s")
        
        return result, mask

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
        seg_map: np.ndarray = self.predict_segmentation_map(img)
        
        # Create color mask
        palette_array: np.ndarray = np.array(self.palette, dtype=np.uint8)
        color_mask = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette_array):
            color_mask[seg_map == label] = color
        
        # Blend original and mask
        orig_arr = np.array(img)
        overlay = (orig_arr * (1 - alpha) + color_mask * alpha).astype(np.uint8)
        return Image.fromarray(overlay)
    
    def predict_segmentation_map(
        self, 
        input_image: Union[str, Image.Image]
    ) -> np.ndarray:
        """Предсказание карты сегментации"""
        # Load image
        img: Image.Image = self.load_image(input_image)
        
        # Preprocess
        pixel_values = self.processor(img, return_tensors="pt").pixel_values.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values)
            logits = outputs.logits
        
        # Post-process to get mask
        seg_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[img.size[::-1]]
        )[0].cpu().numpy()
        print(seg_map)

        unique_classes = np.unique(seg_map)
        print("Предугаданные классы:", unique_classes)

        # Проверяем количество пикселей для каждого класса
        for cls in unique_classes:
            count = (seg_map == cls).sum()
            print(f"Class {cls}: {count} pixels")
        
        return seg_map
    
    def detailed_segmentation(
        self, 
        input_image: Union[str, Image.Image]
    ) -> Dict[str, Any]:
        """
        Детальная сегментация с возвратом всех промежуточных результатов
        """
        img: Image.Image = self.load_image(input_image)
        
        # Получаем карту сегментации
        seg_map: np.ndarray = self.predict_segmentation_map(img)
        
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
    
    def get_class_info(self) -> None:
        """Получить информацию о классах модели"""
        if hasattr(self, 'model') and hasattr(self.model, 'config'):
            return {
                'num_classes': self.model.config.num_labels,
                'id2label': self.model.config.id2label,
                'label2id': self.model.config.label2id
            }
    
    def visualize_segmentation(
        self, 
        input_image: Union[str, Image.Image],
        alpha: float = 0.5,
        figsize: Tuple[int, int] = (15, 5)
    ) -> Image.Image:
        """Базовая визуализация сегментации"""
        img: Image.Image = self.load_image(input_image)
        result_img: Image.Image = self.segment_image(input_image, alpha)
        
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
    
    def segment_and_evaluate(
        self, 
        image: Union[str, np.ndarray, Image.Image],
        ground_truth: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Сегментирует изображение и вычисляет метрики относительно ground truth.
        
        Args:
            image: Входное изображение
            ground_truth: Ground truth маска
            threshold: Порог для метрик
            
        Returns:
            Tuple[Dict[str, float], np.ndarray]: (метрики, предсказанная маска)
        """
        # Получаем сегментированное изображение и маску
        result_img, pred_mask = self.segment_with_mask(image)
        
        # Для нейросетевого сегментатора: получаем карту сегментации
        seg_map = self.predict_segmentation_map(image)
        
        # Определяем фон как самый частый класс
        unique_classes, counts = np.unique(seg_map, return_counts=True)
        if len(unique_classes) > 0:
            bg_class = unique_classes[np.argmax(counts)]
            # Создаем бинарную маску (все не-фоновые классы = объект)
            pred_mask_binary = (seg_map != bg_class).astype(np.uint8) * 255
        else:
            pred_mask_binary = np.zeros_like(seg_map, dtype=np.uint8)
        
        # Приводим к одинаковому размеру с ground truth
        h, w = min(pred_mask_binary.shape[0], ground_truth.shape[0]), \
            min(pred_mask_binary.shape[1], ground_truth.shape[1])
        
        pred_mask_resized = pred_mask_binary[:h, :w]
        gt_mask_resized = ground_truth[:h, :w]
        
        # Вычисляем метрики
        metrics = self._calculate_segmentation_metrics(pred_mask_resized, gt_mask_resized)
        
        return metrics, pred_mask_binary

    def _calculate_segmentation_metrics(
        self, 
        pred_mask: np.ndarray, 
        gt_mask: np.ndarray
    ) -> Dict[str, float]:
        """Вычисляет метрики качества сегментации"""
        # Бинаризация
        pred_bin = (pred_mask > 127).astype(np.uint8).flatten()
        gt_bin = (gt_mask > 127).astype(np.uint8).flatten()
        
        # Вычисляем базовые метрики
        
        try:
            tn, fp, fn, tp = confusion_matrix(gt_bin, pred_bin, labels=[0, 1]).ravel()
        except ValueError:
            # Если только один класс присутствует
            if np.all(pred_bin == 0) and np.all(gt_bin == 0):
                tn, fp, fn, tp = len(pred_bin), 0, 0, 0
            elif np.all(pred_bin == 1) and np.all(gt_bin == 1):
                tn, fp, fn, tp = 0, 0, 0, len(pred_bin)
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        # Precision
        metrics['precision'] = tp / (tp + fp + 1e-8)
        
        # Recall
        metrics['recall'] = tp / (tp + fn + 1e-8)
        
        # F1 Score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                (metrics['precision'] + metrics['recall'] + 1e-8)
        else:
            metrics['f1_score'] = 0.0
        
        # IoU (Jaccard)
        intersection = np.sum(pred_bin & gt_bin)
        union = np.sum(pred_bin | gt_bin)
        metrics['iou'] = intersection / (union + 1e-8)
        
        # Dice Coefficient
        metrics['dice'] = (2 * intersection) / (np.sum(pred_bin) + np.sum(gt_bin) + 1e-8)
        
        # Pixel Accuracy
        metrics['pixel_accuracy'] = np.sum(pred_bin == gt_bin) / len(pred_bin)
        
        # MAE
        if pred_mask.max() > 1:
            pred_norm = pred_mask.astype(float) / 255.0
        else:
            pred_norm = pred_mask.astype(float)
        
        if gt_mask.max() > 1:
            gt_norm = gt_mask.astype(float) / 255.0
        else:
            gt_norm = gt_mask.astype(float)
        
        metrics['mae'] = np.abs(pred_norm - gt_norm).mean()
        
        # Area metrics
        metrics['predicted_area'] = float(np.sum(pred_bin))
        metrics['ground_truth_area'] = float(np.sum(gt_bin))
        metrics['area_difference'] = abs(metrics['predicted_area'] - metrics['ground_truth_area'])
        
        return metrics