# neural_segmenter.py
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Union, Tuple, List, Dict, Any, Optional
import requests
from io import BytesIO
import cv2

from base_segmenter import BaseSegmenter

try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
except ImportError:
    print("Warning: transformers not installed. Install with: pip install transformers")


class NeuralSegmenter(BaseSegmenter):
    """Класс для нейросетевой сегментации (расширенный вариант)"""
    
    def __init__(self, 
                 model_name: str = "nvidia/segformer-b5-finetuned-ade-640-640",
                 device: str = None):
        super().__init__()
        self.model_name = model_name
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Инициализация модели и процессора
        self.processor = SegformerImageProcessor(do_resize=False)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        print(self.model.config)
        self.model.to(self.device)
        self.model.eval()
        
        # Палитта ADE20K
        self.palette = self.ade_palette()
        
        print(f"✅ Нейросетевая модель загружена: {model_name}")
        print(f"   Устройство: {self.device}")
        print(f"   Количество классов: {len(self.model.config.id2label)}")
    
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
        return self.segment_image(image, alpha)
    
    def segment_with_mask(self, 
                          image: Union[str, Image.Image], 
                          alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Сегментация с возвратом маски и обработанного изображения"""
        result_img = self.segment_image(image, alpha)
        seg_map = self.predict_segmentation_map(image)
        
        # Конвертируем result_img обратно в маску для совместимости
        result_np = np.array(result_img)
        
        # Создаем бинарную маску (где есть любой цвет кроме оригинального)
        orig_img = self.load_image(image)
        orig_np = np.array(orig_img)
        
        # Простая бинаризация - разница между оригиналом и результатом
        mask = np.any(result_np != orig_np, axis=2).astype(np.uint8) * 255
        
        return result_np, mask
    
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
        
        # Preprocess and forward pass
        pixel_values = self.processor(img, return_tensors="pt").pixel_values.to(self.device)
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
        
        Returns:
            Dictionary containing:
            - 'original': оригинальное изображение
            - 'segmentation_map': карта сегментации (2D массив с классами)
            - 'color_seg': цветная сегментация
            - 'overlay': наложение на оригинал
            - 'class_distribution': распределение классов
        """
        img = self.load_image(input_image)
        
        # Получаем карту сегментации
        seg_map = self.predict_segmentation_map(img)
        
        # Создаем цветную сегментацию
        palette_array = np.array(self.palette, dtype=np.uint8)
        color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette_array):
            color_seg[seg_map == label, :] = color
        
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
    
    def compare_with_ground_truth(self, 
                                  image_path: str, 
                                  gt_path: str) -> Dict[str, Any]:
        """
        Сравнение предсказания с ground truth
        
        Args:
            image_path: путь к изображению
            gt_path: путь к ground truth карте сегментации
        """
        # Загрузка изображения и ground truth
        image = Image.open(image_path)
        gt_map = Image.open(gt_path)
        
        # Предсказание
        pred_result = self.detailed_segmentation(image_path)
        pred_map = pred_result['segmentation_map']
        
        # Конвертируем ground truth
        gt_array = np.array(gt_map)
        
        # Для ADE20K: ground truth начинается с 1, а предсказание с 0
        if gt_array.min() == 1:
            gt_array = gt_array - 1
        
        # Вычисляем метрики
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Выравниваем массивы
        h, w = min(pred_map.shape[0], gt_array.shape[0]), min(pred_map.shape[1], gt_array.shape[1])
        pred_flat = pred_map[:h, :w].flatten()
        gt_flat = gt_array[:h, :w].flatten()
        
        # Вычисляем метрики только для общих классов
        common_classes = np.intersect1d(np.unique(pred_flat), np.unique(gt_flat))
        
        if len(common_classes) > 0:
            accuracy = accuracy_score(gt_flat, pred_flat)
            precision = precision_score(gt_flat, pred_flat, average='weighted', zero_division=0)
            recall = recall_score(gt_flat, pred_flat, average='weighted', zero_division=0)
            f1 = f1_score(gt_flat, pred_flat, average='weighted', zero_division=0)
        else:
            accuracy = precision = recall = f1 = 0.0
        
        # Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Оригинальное изображение
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Предсказание
        axes[0, 1].imshow(pred_result['overlay'])
        axes[0, 1].set_title("Neural Segmentation")
        axes[0, 1].axis('off')
        
        # Ground truth
        gt_color_seg = np.zeros((gt_array.shape[0], gt_array.shape[1], 3), dtype=np.uint8)
        palette_array = np.array(self.palette, dtype=np.uint8)
        for label, color in enumerate(palette_array):
            gt_color_seg[gt_array == label, :] = color
        gt_color_seg = gt_color_seg[..., ::-1]
        gt_overlay = np.array(image) * 0.5 + gt_color_seg * 0.5
        gt_overlay = gt_overlay.astype(np.uint8)
        
        axes[1, 0].imshow(gt_overlay)
        axes[1, 0].set_title("Ground Truth")
        axes[1, 0].axis('off')
        
        # Разность
        diff = np.abs(pred_map[:h, :w] - gt_array[:h, :w])
        diff_normalized = diff / diff.max() if diff.max() > 0 else diff
        
        axes[1, 1].imshow(diff_normalized, cmap='hot')
        axes[1, 1].set_title("Difference (Prediction vs Ground Truth)")
        axes[1, 1].axis('off')
        
        plt.suptitle(f"Comparison with Ground Truth\nAccuracy: {accuracy:.3f}, F1: {f1:.3f}", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'common_classes': len(common_classes),
            'prediction': pred_result,
            'ground_truth': {
                'map': gt_array,
                'color_seg': gt_color_seg,
                'overlay': gt_overlay
            }
        }
    
    def visualize_detailed_segmentation(self, 
                                        input_image: Union[str, Image.Image],
                                        figsize: Tuple[int, int] = (15, 10)):
        """Визуализация детальной сегментации"""
        result = self.detailed_segmentation(input_image)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Оригинальное изображение
        axes[0, 0].imshow(result['original'])
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Карта сегментации
        im1 = axes[0, 1].imshow(result['segmentation_map'], cmap='tab20')
        axes[0, 1].set_title("Segmentation Map")
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Цветная сегментация
        axes[0, 2].imshow(result['color_seg'])
        axes[0, 2].set_title("Color Segmentation")
        axes[0, 2].axis('off')
        
        # Наложение
        axes[1, 0].imshow(result['overlay'])
        axes[1, 0].set_title("Overlay (alpha=0.5)")
        axes[1, 0].axis('off')
        
        # Распределение классов (график)
        class_dist = result['class_distribution']
        if class_dist:
            class_names = list(class_dist.keys())[:10]  # Первые 10 классов
            percentages = [class_dist[name]['percentage'] for name in class_names]
            
            axes[1, 1].barh(class_names, percentages)
            axes[1, 1].set_title("Top 10 Classes Distribution")
            axes[1, 1].set_xlabel("Percentage (%)")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "No classes detected", 
                          ha='center', va='center')
            axes[1, 1].set_title("Class Distribution")
            axes[1, 1].axis('off')
        
        # Информация о модели
        info_text = f"Model: {self.model_name}\n"
        info_text += f"Device: {self.device}\n"
        info_text += f"Total classes: {result['total_classes']}\n"
        info_text += f"Image size: {result['original'].size}"
        
        axes[1, 2].text(0.1, 0.5, info_text, fontsize=10, 
                       verticalalignment='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title("Model Information")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Выводим статистику в консоль
        print("=" * 60)
        print("DETAILED SEGMENTATION ANALYSIS")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Total classes detected: {result['total_classes']}")
        print("\nTop 10 classes by area:")
        print("-" * 40)
        
        sorted_classes = sorted(result['class_distribution'].items(), 
                               key=lambda x: x[1]['pixel_count'], 
                               reverse=True)[:10]
        
        for i, (class_name, info) in enumerate(sorted_classes, 1):
            print(f"{i:2}. {class_name:30} {info['percentage']:6.2f}% ({info['pixel_count']} pixels)")
        
        print("=" * 60)
        
        return result