import sys
import os

# Добавляем корень проекта в PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Теперь импорт сработает
from datasets.ADE20KDataset import ADE20KDataset
from segmenters.NeuralTrainer import NeuralTrainer
import inference.utils
from inference.strategies import SegNet

import os
from typing import List, Optional, Set, Dict, List, Union, Tuple, Any
import time
import zipfile
from tqdm import tqdm
import shutil
import requests
import gc

from io import BytesIO
from PIL import Image

from huggingface_hub import hf_hub_download, list_repo_files

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import tabulate

import torch
import torch.nn.functional as F
from torch import nn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import MultivariateNormal

import torchvision
from torchvision import transforms
import torchvision.transforms as T
from torchvision.ops import boxes as box_ops
import torchvision.models.segmentation as tv_seg
import torchvision.models.detection as tv_det

from sklearn.cluster import MeanShift as SkMeanShift
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, jaccard_score

from ultralytics import SAM

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import DPTImageProcessor, DPTForSemanticSegmentation
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

import segmentation_models_pytorch as smp

from segmenters.NeuralModelFactory import NeuralModelFactory, ModelType
from inference.palettes import ade_palette
from inference.utils import compute_metrics, extract_logits_info, analyze_prediction
from inference.strategies import segment_image_unified

class SegmentationBenchmark:
    """Полноценный бенчмарк для сравнения архитектур сегментации"""
    
    def __init__(
        self, 
        device: str = "cuda", 
        num_classes: int = 150, 
        ignore_index: int = 255, 
        class_names: list = None, 
        gt_mask: Optional[Union[np.ndarray, Image.Image]] = None,
        palette: Optional[Union[List[List[int]], callable]] = None
    ) -> None:
        """
        Инициализация бенчмарка для сравнения моделей сегментации.
        
        Args:
            device: Устройство для вычислений ("cuda" или "cpu")
            num_classes: Количество классов сегментации (по умолчанию 150 для ADE20K)
            ignore_index: Индекс игнорируемых пикселей (по умолчанию 255)
            class_names: Список имён классов для визуализации
            gt_mask: Ground truth маска для вычисления метрик
            palette: Цветовая палитра для визуализации масок
        """
        if callable(palette):
            palette = palette()
    
        self.device: str = device
        self.num_classes: int = num_classes
        self.ignore_index: int = ignore_index
        self.models: Dict[str, Dict[str, Any]] = {}
        self.palette: List[List[int]] = palette if palette else ade_palette()
        self.results: Dict[str, Dict[str, Any]] = {}  # {model_name: {metrics, time, overlay, ...}}
        self.class_names: List[str] = class_names or [f"Class {i}" for i in range(num_classes)]
        self.gt_mask: Optional[Union[np.ndarray, Image.Image]] = gt_mask

    # Загрузка моделей для инференса

    def load_trained_model(
        self,
        key: str,
        model_type: ModelType,
        checkpoint_path: str,
        **kwargs
    ) -> "SegmentationBenchmark":
        """Загрузка обученной модели из чекпоинта"""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            device=self.device,
            num_classes=self.num_classes,
            **kwargs
        )
        
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
            "checkpoint": checkpoint_path
        }
        print(f"✅ Loaded {key} from {checkpoint_path}")
        return self
    
    
    def load_all_trained_models(
        self,
        checkpoint_dir: str = "./../models"
    ) -> "SegmentationBenchmark":
        """Загрузка всех обученных моделей для бенчмарка"""
        checkpoints = {
            "unet_trained": (ModelType.UNET_SMP, "./../models/unet_ade20k_best.pth", {"encoder_name": "resnet34"}),
            "deeplab_trained": (ModelType.DEEPLAB_TV, "./../models/deeplab_ade20k_best.pth", {}),
            "fpn_mit_b5_trained": (ModelType.FPN_SMP, "./../models/fpn_mit_b5_ade20k_best.pth", {"encoder_name": "mit_b5"}),
            "psp_mit_b5_trained": (ModelType.PSPNET_SMP, "./../models/psp_mit_b5_ade20k_best.pth", {"encoder_name": "mit_b5"}),
            "fcn_resnet50_trained": (ModelType.FCN_TV, "./../models/fcn_resnet50_ade20k_best.pth", {"variant": "fcn_resnet50"}),
            "segnet_trained": (ModelType.SEGNET, "./../models/segnet_ade20k_best.pth", {"encoder_name": "resnet34"}),
        }
        
        for key, (model_type, checkpoint_file, kwargs) in checkpoints.items():
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                self.load_trained_model(key, model_type, checkpoint_path, **kwargs)
            else:
                print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        
        return self
    
    # ============ ЗАГРУЗКА ПРЕДОБУЧЕННЫХ МОДЕЛЕЙ ============
    def load_segformer(self, path: str) -> "SegmentationBenchmark":
        """Загрузка SegFormer модели"""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.SEGFORMER,
            local_path=path,
            device=self.device,
            num_classes=self.num_classes
        )
        self.models["segformer"] = {
            "model": model,
            "processor": processor,
            "type": model_type_str
        }
        print(f"✅ Loaded SegFormer from {path}")
        return self
    
    def load_segformer_variant(self, variant: str = "b2") -> "SegmentationBenchmark":
        """
        Загрузка разных версий SegFormer для сравнения.
        
        Args:
            variant: Версия модели ("b0", "b1", "b2", "b3", "b4", "b5")
        
        Returns:
            self: Для цепочки вызовов
        
        Raises:
            ValueError: Если указана неизвестная версия
        """
        # 🔥 Используем NeuralModelFactory вместо прямой загрузки!
        model, processor, model_type_str = NeuralModelFactory.load_segformer_variant(
            variant=variant,
            device=self.device
        )
        
        # 🔥 Регистрируем через load_model (единый интерфейс)
        key = f"segformer_{variant}"
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str
        }
        
        print(f"✅ Loaded SegFormer-{variant} from HuggingFace")
        return self
    
    def load_mask2former(self, name: str = "facebook/mask2former-swin-base-ade-semantic") -> "SegmentationBenchmark":
        """Загрузка Mask2Former модели"""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.MASK2FORMER,
            model_name=name,
            device=self.device,
            num_classes=self.num_classes
        )
        self.models["mask2former"] = {
            "model": model,
            "processor": processor,
            "type": model_type_str
        }
        print(f"✅ Loaded Mask2Former from {name}")
        return self
    
    def load_oneformer(self, name: str = "shi-labs/oneformer_ade20k_swin_large") -> "SegmentationBenchmark":
        """Загрузка OneFormer модели"""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.ONEFORMER,
            model_name=name,
            device=self.device,
            num_classes=self.num_classes
        )
        self.models["oneformer"] = {
            "model": model,
            "processor": processor,
            "type": model_type_str
        }
        print(f"✅ Loaded OneFormer from {name}")
        return self
    
    def load_dpt(self, model_name: str = "Intel/dpt-large-ade") -> "SegmentationBenchmark":
        """Загрузка DPT модели"""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.DPT,
            model_name=model_name,
            device=self.device,
            num_classes=self.num_classes
        )
        self.models["dpt"] = {
            "model": model,
            "processor": processor,
            "type": model_type_str
        }
        print(f"✅ Loaded DPT from {model_name}")
        return self
    
    def load_upernet(self, model_name: str = "openmmlab/upernet-convnext-small") -> "SegmentationBenchmark":
        """Загрузка UPerNet модели"""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.UPERNET,
            model_name=model_name,
            device=self.device,
            num_classes=self.num_classes
        )
        self.models["upernet"] = {
            "model": model,
            "processor": processor,
            "type": model_type_str
        }
        print(f"✅ Loaded UPerNet from {model_name}")
        return self
    
    def load_sam(self, model_name: str = "mobile_sam.pt") -> "SegmentationBenchmark":
        """Загрузка SAM-моделей"""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.SAM,
            model_name=model_name,
            device=self.device,
            num_classes=self.num_classes
        )
        model_key = "sam2" if "sam2" in model_name.lower() else "sam"
        self.models[model_key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str
        }
        print(f"✅ Loaded SAM from {model_name}")
        return self
    
    def load_fpn_mit_pretrained(self, variant: str = "b5", checkpoint_path: Optional[str] = None) -> "SegmentationBenchmark":
        """Загрузка FPN + MiT"""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.FPN_SMP,
            device=self.device,
            num_classes=self.num_classes,
            encoder_name=f"mit_{variant}",
            checkpoint_path=checkpoint_path
        )
        key = f"fpn_mit_{variant}_pretrained"
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
            "checkpoint": checkpoint_path
        }
        print(f"✅ Loaded FPN+MiT-{variant}")
        return self
    
    def load_psp_mit_pretrained(self, variant: str = "b5", checkpoint_path: Optional[str] = None) -> "SegmentationBenchmark":
        """Загрузка PSPNet + MiT"""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.PSPNET_SMP,
            device=self.device,
            num_classes=self.num_classes,
            encoder_name=f"mit_{variant}",
            checkpoint_path=checkpoint_path
        )
        key = f"psp_mit_{variant}_pretrained"
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
            "checkpoint": checkpoint_path
        }
        print(f"✅ Loaded PSPNet+MiT-{variant}")
        return self
    
    def load_fcn_resnet50_pretrained(self, variant: str = "fcn_resnet50") -> "SegmentationBenchmark":
        """Загрузка FCN"""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.FCN_TV,
            device=self.device,
            num_classes=self.num_classes,
            variant=variant
        )
        key = f"fcn_{variant.replace('fcn_', '')}_pretrained"
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str
        }
        print(f"✅ Loaded {variant}")
        return self
    
    def load_segnet_pretrained(self, encoder_name: str = "resnet34") -> "SegmentationBenchmark":
        """Загрузка SegNet"""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.SEGNET,
            device=self.device,
            num_classes=self.num_classes,
            encoder_name=encoder_name
        )
        key = f"segnet_{encoder_name.replace('-', '_')}_pretrained"
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str
        }
        print(f"✅ Loaded SegNet-like")
        return self
    
    def load_mask_rcnn_pretrained(self, variant: str = "maskrcnn_resnet50_fpn") -> "SegmentationBenchmark":
        """Загрузка Mask R-CNN"""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.MASKRCNN_TV,
            device=self.device,
            num_classes=self.num_classes,
            variant=variant
        )
        self.models["maskrcnn_pretrained"] = {
            "model": model,
            "processor": processor,
            "type": model_type_str
        }
        print(f"✅ Loaded Mask R-CNN")
        return self
    
    def load_unet_trained(
        self,
        checkpoint_path: str = "unet_ade20k_best.pth",
        encoder_name: str = "resnet34"
    ) -> "SegmentationBenchmark":
        """
        Загрузка ОБУЧЕННОЙ U-Net с чекпоинта.
        
        Args:
            checkpoint_path: Путь к файлу чекпоинта
            encoder_name: Название encoder для архитектуры
        
        Returns:
            self: Для цепочки вызовов
        """
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.UNET_SMP,
            device=self.device,
            num_classes=self.num_classes,
            encoder_name=encoder_name,
            checkpoint_path=checkpoint_path
        )
        key = "unet_pretrained"
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
            "checkpoint": checkpoint_path
        }
        print(f"✅ Loaded Unet pretrained")
        return self
    
    def load_deeplab_trained(
        self,
        checkpoint_path: str = "deeplab_ade20k_best.pth"
    ) -> "SegmentationBenchmark":
        """
        Загрузка ОБУЧЕННОЙ DeepLabV3+ с чекпоинта.
        
        Args:
            checkpoint_path: Путь к файлу чекпоинта
        
        Returns:
            self: Для цепочки вызовов
        """
        
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.DEEPLAB_TV,
            device=self.device,
            num_classes=self.num_classes,
            encoder_name=None,
            checkpoint_path=checkpoint_path
        )
        key = "deeplab_pretrained"
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
            "checkpoint": checkpoint_path
        }
        print(f"✅ Loaded deeplab pretrained")
        return self
    
    def load_all_pretrained_cnn(self, checkpoint_dir: str = "./checkpoints") -> "SegmentationBenchmark":
        """Загрузка всех CNN-моделей"""
        print("\n" + "="*60)
        print("📦 Loading all pre-trained CNN models for benchmark")
        print("="*60)
        
        self.load_mask_rcnn_pretrained(variant="maskrcnn_resnet50_fpn")
        
        fpn_checkpoint = os.path.join(checkpoint_dir, "fpn_mit_b5_best.pth")
        self.load_fpn_mit_pretrained(variant="b5", checkpoint_path=fpn_checkpoint)
        
        psp_checkpoint = os.path.join(checkpoint_dir, "psp_mit_b5_best.pth")
        self.load_psp_mit_pretrained(variant="b5", checkpoint_path=psp_checkpoint)
        
        self.load_fcn_resnet50_pretrained(variant="fcn_resnet50")
        self.load_segnet_pretrained(encoder_name="resnet34")
        
        print("\n✅ All pre-trained CNN models loaded!")
        print(f"   Total models in benchmark: {len(self.models)}")
        return self

    def run_single(
        self,
        image_input: Union[str, Image.Image],
        model_key: str,
        alpha: float = 0.6,
        log_logits: bool = True
    ) -> Dict[str, Any]:
        """
        Запуск одной модели с сбором метрик.
        
        Args:
            image_input: Путь к изображению или PIL.Image объект
            model_key: Ключ загруженной модели
            alpha: Прозрачность наложения маски (0.0–1.0)
            log_logits: Логировать информацию о логитах
        
        Returns:
            result: Словарь с результатами (overlay, mask, metrics, time_ms, etc.)
        
        Raises:
            ValueError: Если модель с указанным ключом не загружена
        """

        # Загрузка изображения
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported input type")
        
        model_cfg = self.models[model_key]
        model = model_cfg["model"]
        processor = model_cfg["processor"]
        model_type = model_cfg["type"]
        n_classes = self.get_model_num_classes(model_key)

        torch.cuda.empty_cache()
        gc.collect()

        # Замер времени + warm-up
        if model_type not in ["maskformer", "mask2former", "oneformer"]:
            for _ in range(2):
                _, _ = segment_image_unified(model, processor, image, model_type, 
                                        alpha=0.0, palette=self.palette, device=self.device, num_classes=n_classes, gt_mask=self.gt_mask)

        torch.cuda.synchronize()
        t0 = time.time()
        
        # Инференс
        overlay, resd = segment_image_unified(model, processor, image, model_type, 
                                       alpha=alpha, palette=self.palette, device=self.device, num_classes=n_classes, 
                                       gt_mask=self.gt_mask)
        
        torch.cuda.synchronize()
        inference_time = time.time() - t0
        mask = resd["mask"]

        # 📊 Метрики (если есть ground truth)
        metrics = {}
        print(f"gt_maske: {self.gt_mask}")
        if self.gt_mask is not None:
            gt_np = np.array(self.gt_mask) if isinstance(self.gt_mask, Image.Image) else self.gt_mask
            metrics = compute_metrics(mask, gt_np, self.num_classes, self.ignore_index)
        
        print(f"Метрики {metrics}")
        result = {
            "model": model_type,
            "overlay": overlay,
            "mask": mask,
            "inference_time_ms": inference_time * 1000,
            "metrics": metrics,
            "image_size": image.size[::-1],
            "output_shape": mask.shape,
            "unique_classes": len(np.unique(mask))
        }
        
        self.results[model_key] = result
        return result
    
    def predict(
        self,
        image_input: Union[str, Image.Image],
        model_key: str,
        alpha: float = 0.5,
        gt_mask: Optional[np.ndarray] = None
    ) -> Image.Image:
        """
        Предсказание сегментации для одного изображения.
        
        Args:
            image_input: Путь к изображению или PIL.Image объект
            model_key: Ключ загруженной модели
            alpha: Прозрачность наложения маски
            gt_mask: Ground truth маска для метрик (опционально)
        
        Returns:
            overlay: Изображение с наложенной маской
        
        Raises:
            ValueError: Если модель с указанным ключом не загружена
        """
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not loaded. Available: {list(self.models.keys())}")
        model, processor = self.models[model_key]
        n_classes = self.get_model_num_classes(model_key)
        return segment_image_unified(model, processor, image_input, model_key, 
                                    alpha=alpha, palette=self.palette, device=self.device, num_classes=n_classes, gt_mask=gt_mask)[0]
    
    def compare(
        self,
        image_input: Union[str, Image.Image],
        alpha: float = 0.6
    ) -> Dict[str, Dict[str, Any]]:
        """
        Запускает все загруженные модели и возвращает dict со сводными результатами.
        
        Args:
            image_input: Путь к изображению или PIL.Image объект
            alpha: Прозрачность наложения маски
        
        Returns:
            summary: Сводная таблица метрик по всем моделям
        """
        print(f"🚀 Starting benchmark on {len(self.models)} models...")
        model_keys = list(self.models.keys())  # Копия ключей
    
        for i, key in enumerate(model_keys):
            print(f"\n🔹 Running {key}...")
            
            self.run_single(image_input, key, alpha=alpha)
            
            # 🔥 ОСВОБОЖДЕНИЕ ПАМЯТИ ПОСЛЕ МОДЕЛИ
            if i < len(model_keys) - 1:  # Не удаляем последнюю (может понадобиться)
                del self.models[key]["model"]
                del self.models[key]["processor"]
                torch.cuda.empty_cache()
                gc.collect()
                print(f"   🗑️  Freed {key} from VRAM")
        
        return self.get_summary()

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает сводную таблицу метрик.
        
        Returns:
            summary: Словарь {model_key: {mIoU, pixel_acc, f1_weighted, time_ms, unique_classes}}
        """
        summary = {}
        for key, res in self.results.items():
            summary[key] = {
                "mIoU": res["metrics"].get("mIoU", np.nan),
                "pixel_acc": res["metrics"].get("pixel_acc", np.nan),
                "f1_weighted": res["metrics"].get("f1_weighted", np.nan),
                "time_ms": res["inference_time_ms"],
                "unique_classes": res["unique_classes"]
            }
        return summary
    
    def get_model_num_classes(self, model_key: str) -> int:
        """
        Определяет число классов для модели по ключу.
        
        Args:
            model_key: Ключ загруженной модели
        
        Returns:
            num_classes: Количество выходных классов модели
        """
        cfg = self.models[model_key]
        model = cfg["model"]
        model_type = cfg["type"]
        
        # HF модели
        if hasattr(model, 'config') and hasattr(model.config, 'id2label'):
            return len(model.config.id2label)
        
        # Torchvision
        if model_type in ["deeplab_tv", "fcn_tv"]:
            if hasattr(model, 'classifier'):
                return model.classifier[-1].out_channels
        
        # SMP / Custom: ищем последний Conv2d
        import torch
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module.out_channels
        
        # Fallback
        return self.num_classes


    # ============ ВИЗУАЛИЗАЦИЯ ============
    def plot_comparison_chart(self, metric_name: str, title: str = None, 
                          figsize=(12, 6), show_values: bool = True, path: str = './data/ade20k_test_trained/plot_comparison_chart.jpg'):
        """
        Строит бар-чарт сравнения одной метрики.
        
        🔧 FIX:
        - Корректное масштабирование для маленьких значений
        - Автоматический выбор формата (проценты или десятичные)
        - Улучшенное размещение подписей
        """
        summary = self.get_summary()
        if not summary:
            print("⚠️ No results to plot. Run compare() first.")
            return
        
        models = list(summary.keys())
        values = [summary[m].get(metric_name, np.nan) for m in models]
        
        # Фильтруем модели без данных
        valid = [(m, v) for m, v in zip(models, values) if not np.isnan(v)]
        
        if not valid:
            print(f"⚠️ No valid data for metric '{metric_name}'")
            return
        
        models, values = zip(*valid)
        
        # Определяем, нужно ли умножать на 100 (для процентов)
        is_percentage = metric_name in ["mIoU", "pixel_acc", "f1_weighted"]
        multiplier = 100 if is_percentage else 1
        
        # Проверяем диапазон значений
        max_val = max(values) * multiplier
        use_percent_format = max_val < 1.0
        
        plt.figure(figsize=figsize)
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        # Создаем бары
        x_pos = range(len(models))
        bars = plt.bar(x_pos, [v * multiplier for v in values], 
                    color=colors, edgecolor='black', linewidth=1.2)
        
        # Подписи значений с авто-форматированием
        if show_values:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                
                # Форматируем значение
                if use_percent_format:
                    label = f"{val * 100:.3f}%"  # 0.137%
                else:
                    label = f"{val:.1f}" if val >= 1 else f"{val:.3f}"
                
                # Размещаем подпись
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        label, ha='center', va='bottom', 
                        fontsize=8, rotation=0)
        
        # Настройка оси Y
        if use_percent_format:
            plt.ylabel(f"{metric_name} (%)", fontsize=10)
            plt.ylim(0, max_val * 1.2)  # 20% запас сверху
        else:
            plt.ylabel(metric_name, fontsize=10)
            if metric_name == "time_ms":
                plt.ylim(0, max(values) * 1.15)
            else:
                plt.ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
        
        # Подписи моделей с переносом
        plt.xticks(x_pos, 
                [m.replace('_', '\n') if len(m) > 10 else m for m in models], 
                rotation=45, ha='right', fontsize=9)
        
        plt.title(title or f"Model Comparison: {metric_name}", 
                fontsize=12, fontweight='bold', pad=20)
        plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Улучшенный tight_layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path, dpi=300,
                bbox_inches='tight',
                facecolor='white',
                format='png')
        plt.show()

    def plot_per_class_iou(
        self, 
        top_k: int = 20, 
        figsize=(14, 8), 
        cmap: str = "RdYlGn",
        show_only_present_classes: bool = True,
        path: str = './data/ade20k_test_trained/plot_per_class_iou.jpg'
    ):
        """
        Строит heatmap per-class IoU только для классов, присутствующих в GT.
        """
        
        data = []
        model_names = []
        
        # Собираем все per_class_iou
        for model_name, res in self.results.items():
            iou_arr = res["metrics"].get("per_class_iou")
            
            if iou_arr is not None and len(iou_arr) > 0:
                iou_arr = np.array(iou_arr[:self.num_classes])
                
                # Фильтруем только валидные (не NaN) значения
                if np.any(np.isfinite(iou_arr)):
                    data.append(iou_arr)
                    model_names.append(model_name)
                else:
                    print(f"⚠️  {model_name}: all NaN IoU")
        
        if not data:
            print("❌ No valid per-class IoU data.")
            return
        
        data = np.array(data)  # [n_models, n_classes]
        
        # Находим классы, которые есть в ground truth хотя бы одной модели
        if show_only_present_classes:
            # Класс считается "present" если хотя бы одна модель дала валидный IoU
            valid_classes = np.any(np.isfinite(data), axis=0)
            class_indices = np.where(valid_classes)[0]
        else:
            class_indices = np.arange(self.num_classes)
        
        if len(class_indices) == 0:
            print("❌ No classes with valid IoU found!")
            return
        
        # Берем top-k среди присутствующих классов
        mean_iou = np.nanmean(data[:, class_indices], axis=0)
        top_indices_in_subset = np.argsort(mean_iou)[::-1][:top_k]
        top_class_indices = class_indices[top_indices_in_subset]
        
        # Подписи классов
        class_labels = [f"Class {c}" for c in top_class_indices]
        
        # Фильтруем данные
        data_filtered = data[:, top_class_indices]
        
        plt.figure(figsize=figsize)
        ax = sns.heatmap(data_filtered, 
                    xticklabels=class_labels, 
                    yticklabels=model_names,
                    cmap=cmap, 
                    center=0.5, 
                    annot=False,
                    cbar_kws={'label': 'IoU'},
                    vmin=0, vmax=1)
        
        plt.title(f"Per-class IoU (top {len(top_class_indices)} present classes)")
        plt.xlabel("Class")
        plt.ylabel("Model")
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(path, dpi=300,
                bbox_inches='tight',
                facecolor='white',
                format='png')
        plt.show()
        
        # Дополнительная статистика
        print(f"\n📊 Per-class IoU Statistics:")
        print(f"  Total classes in dataset: {self.num_classes}")
        print(f"  Classes present in GT: {len(class_indices)}")
        print(f"  Showing top {len(top_class_indices)} classes")
        print(f"  Mean IoU (all classes): {np.nanmean(data):.3f}")

    def plot_confusion_matrix(self, model_key: str, normalize: str = 'true', 
                          figsize=(10, 8), show_values: bool = True,
        path: str = './data/ade20k_test_trained/plot_confusion_matrix.jpg'):
        """
        Строит матрицу ошибок с отображением значений.
        """
        if model_key not in self.results:
            print(f"⚠️  Model '{model_key}' not found.")
            return
        
        cm = self.results[model_key]["metrics"].get("confusion_matrix")
        if cm is None:
            print("⚠️  No confusion matrix available.")
            return
        
        # Нормализация
        if normalize == 'true':
            cm_display = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
            title_suffix = "(recall)"
            fmt = '.2f'
        elif normalize == 'pred':
            cm_display = cm.astype('float') / (cm.sum(axis=0, keepdims=True) + 1e-8)
            title_suffix = "(precision)"
            fmt = '.2f'
        else:
            cm_display = cm
            title_suffix = "(counts)"
            fmt = 'd'
        
        # Показываем только классы, которые есть в ground truth
        gt_classes = np.where(cm.sum(axis=1) > 0)[0][:20]
        if len(gt_classes) == 0:
            print("⚠️  No ground truth classes found!")
            return
        
        class_labels = [f"C{c}" for c in gt_classes]
        cm_subset = cm_display[np.ix_(gt_classes, gt_classes)]
        
        plt.figure(figsize=figsize)
        ax = sns.heatmap(cm_subset, 
                    xticklabels=class_labels, 
                    yticklabels=class_labels,
                    cmap="Blues", 
                    annot=show_values,
                    fmt=fmt,
                    cbar_kws={'label': 'Normalized count' if normalize else 'Count'})
        
        plt.title(f"Confusion Matrix: {model_key} {title_suffix}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=300,
                bbox_inches='tight',
                facecolor='white',
                format='png')
        plt.show()

    def plot_all_metrics(self, figsize=(15, 5), skip_empty: bool = True,
        path: str = './data/ade20k_test_trained/plot_all_metrix.jpg'):
        """
        Строит сводные графики по всем основным метрикам.
        
        🔧 FIX: 
        - Пропускаем пустые метрики до создания subplot
        - Уменьшаем шрифт подписей для длинных названий
        - Добавляем отступ для suptitle
        """
        
        summary = self.get_summary()
        if not summary:
            print("⚠️ No results to plot.")
            return
        
        # Определяем, какие метрики имеют валидные данные
        metrics_to_plot = [
            ("mIoU", "Mean IoU ↑", lambda x: x*100),
            ("pixel_acc", "Pixel Accuracy ↑", lambda x: x*100),
            ("time_ms", "Inference Time ↓ (ms)", lambda x: x)
        ]
        
        # Фильтруем только метрики с валидными данными
        valid_metrics = []
        for metric, label, transform in metrics_to_plot:
            values = [summary[m].get(metric, np.nan) for m in summary]
            if any(not np.isnan(v) for v in values):
                valid_metrics.append((metric, label, transform))
        
        if not valid_metrics:
            print("⚠️ No valid metrics to plot")
            return
        
        # Создаём нужное количество subplot'ов
        n_plots = len(valid_metrics)
        fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0] * n_plots / 3, figsize[1]))
        if n_plots == 1:
            axes = [axes]  # Ensure axes is always a list
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(summary)))
        
        for ax, (metric, label, transform) in zip(axes, valid_metrics):
            models = list(summary.keys())
            values = [transform(summary[m].get(metric, np.nan)) for m in models]
            
            # Фильтруем валидные значения
            valid = [(m, v) for m, v in zip(models, values) if not np.isnan(v)]
            
            if valid:
                plot_models, plot_values = zip(*valid)
                
                # Уменьшаем шрифт если моделей много
                fontsize = 7 if len(plot_models) > 10 else 8
                
                bars = ax.bar(range(len(plot_models)), plot_values, 
                            color=colors[:len(plot_models)], edgecolor='black')
                
                # Подписи с переносом названий моделей
                for bar, val, name in zip(bars, plot_values, plot_models):
                    # Перенос длинных названий
                    display_name = name.replace('_', '_\n') if len(name) > 15 else name
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f"{val:.1f}", ha='center', va='bottom', fontsize=fontsize-1)
                
                ax.set_xticks(range(len(plot_models)))
                ax.set_xticklabels([m.replace('_', '_\n') for m in plot_models], 
                                rotation=45, ha='right', fontsize=fontsize)
                
                ax.set_ylabel(label, fontsize=9)
                ax.set_title(metric, fontsize=10, fontweight='bold')
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Авто-масштабирование оси Y
                if metric == "time_ms":
                    ax.set_ylim(0, max(plot_values) * 1.2)
                else:
                    ax.set_ylim(0, 100)
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12, color='gray')
                ax.set_title(metric, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle("Model Comparison Summary", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(path, dpi=300,
                bbox_inches='tight',
                facecolor='white',
                format='png')
        plt.show()

    def plot_summary(self, metrics: list = ["mIoU", "pixel_acc", "time_ms"],
        path: str = './data/ade20k_test_trained/plot_summary.jpg'):
        """Визуализация сводных результатов"""
        summary = self.get_summary()
        
        for metric in metrics:
            values = [summary[k].get(metric, np.nan) for k in summary]
            if all(np.isnan(v) for v in values):
                continue
                
            plt.figure(figsize=(10, 5))
            plt.bar(summary.keys(), values, color=plt.cm.Set2(np.linspace(0, 1, len(summary))))
            plt.ylabel(metric)
            plt.title(f"Model Comparison: {metric}")
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(path, dpi=300,
                bbox_inches='tight',
                facecolor='white',
                format='png')
            plt.show()

    def save_results(self, output_dir: str = "benchmark_results") -> None:
        """
        Сохранение всех результатов с корректной сериализацией numpy-типов.
        
        Args:
            output_dir: Директория для сохранения результатов
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохранение изображений и масок
        for key, res in self.results.items():
            if res["overlay"] is not None:
                res["overlay"].save(f"{output_dir}/overlay_{key}.jpg")
            if res["mask"] is not None:
                np.save(f"{output_dir}/mask_{key}.npy", res["mask"])
        
        # Сводная таблица
        df = pd.DataFrame(self.get_summary()).T
        df.to_csv(f"{output_dir}/summary.csv")
        
        def convert_numpy_types(obj):
            """Рекурсивно конвертирует numpy-типы в Python-native для JSON"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj
        
        # Детальные метрики
        detailed = {}
        for key, res in self.results.items():
            detailed[key] = {
                "inference_time_ms": float(res["inference_time_ms"]),
                "metrics": convert_numpy_types(res["metrics"]),
                "image_size": [int(x) for x in res["image_size"]],
                "output_shape": [int(x) for x in res["output_shape"]],
                "unique_classes": int(res["unique_classes"])
            }
        
        with open(f"{output_dir}/detailed.json", "w") as f:
            json.dump(detailed, f, indent=2, default=str)
        
        print(f"✅ Results saved to {output_dir}/")

    def export_latex_table(self, caption: str = "Segmentation Benchmark Results") -> str:
        """
        Генерирует LaTeX-код таблицы для публикации.
        
        Args:
            caption: Заголовок таблицы для LaTeX
        
        Returns:
            latex_code: Строка с LaTeX кодом таблицы
        """
        summary = self.get_summary()
        if not summary:
            return ""
        
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{" + caption + r"}",
            r"\label{tab:benchmark}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"\textbf{Model} & \textbf{mIoU (\%)} & \textbf{Acc (\%)} & \textbf{Time (ms)} \\",
            r"\midrule"
        ]
        
        for model, metrics in summary.items():
            mIoU = f"{metrics['mIoU']*100:.1f}" if not np.isnan(metrics['mIoU']) else "-"
            acc = f"{metrics['pixel_acc']*100:.1f}" if not np.isnan(metrics['pixel_acc']) else "-"
            time = f"{metrics['time_ms']:.1f}"
            model_clean = model.replace("_", r"\_").replace("-", r"\-")
            lines.append(f"{model_clean} & {mIoU} & {acc} & {time} \\\\")
        
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)
    

def export_comparison_table(
    bench: SegmentationBenchmark, 
    output_file: str = "model_comparison.md"
) -> pd.DataFrame:
    """Экспорт сравнительной таблицы всех моделей"""
    
    df = pd.DataFrame(bench.get_summary()).T.sort_values("mIoU", ascending=False)
    
    # Категоризация
    categories = {
        "segformer": "Transformer",
        "mask2former": "Universal",
        "oneformer": "Multi-task",
        "dpt": "Hybrid",
        "upernet": "CNN+FPN",
        "deeplab_tv": "CNN+ASPP",
        "fcn_tv": "FCN",
        "maskrcnn_tv": "Instance",
        "segnet": "Encoder-Decoder",
        "unet_smp": "Encoder-Decoder",
        "fpn_mit": "Transformer+FPN",
        "sam": "Promptable",
        "sam2": "Promptable",
    }
    
    df["Category"] = df.index.map(lambda x: categories.get(x.split("_")[0], "Other"))
    
    # Форматирование
    df["mIoU (%)"] = (df["mIoU"] * 100).round(1)
    df["Time (ms)"] = df["time_ms"].round(1)
    df["Params (M)"] = df.get("params", pd.Series([0]*len(df))).round(1)
    
    # Markdown таблица
    md_table = df[["Category", "mIoU (%)", "pixel_acc", "Time (ms)", "unique_classes"]].to_markdown()
    
    with open(output_file, "w") as f:
        f.write("# Segmentation Models Comparison\n\n")
        f.write(md_table)
    
    print(f"✅ Table saved to {output_file}")
    return df