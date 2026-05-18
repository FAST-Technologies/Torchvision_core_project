# testing/SegmentationBenchmark.py

"""Модуль для сравнительного бенчмаркинга нейросетевых архитектур сегментации изображений.

Предназначен для автоматизированного сравнения качества и производительности
различных моделей сегментации (CNN, Transformers, Universal) на едином датасете.
Поддерживает как обученные пользователем модели (.pth чекпоинты), так и
предобученные модели из Hugging Face, TorchVision, Segment Anything и др.

Основные возможности:
- 🔄 Fluent Interface для загрузки моделей: цепочка вызовов `load_*().load_*()...`
- 🧠 Поддержка 15+ архитектур: UNet, DeepLab, FPN, PSPNet, SegFormer, Mask2Former,
  OneFormer, DPT, UPerNet, SAM/SAM2, YOLOv8-seg, Mask R-CNN и др.
- ⚡ Управление памятью: автоматическая очистка VRAM между моделями,
  асинхронный режим с обновлением прогресса
- 📊 Полный набор метрик: mIoU, Pixel Accuracy, F1-weighted, Per-Class IoU,
  Confusion Matrix, уникальные классы
- 🎨 Визуализация: бар-чарты, heatmap per-class IoU, матрицы ошибок,
  наложенные маски с настраиваемой прозрачностью
- 📤 Экспорт результатов: CSV, JSON (с сериализацией numpy-типов),
  Markdown-таблицы, LaTeX-код для публикаций

Примечание:
- Для корректной работы требуется предварительная загрузка моделей через
  методы `load_*()` перед вызовом `compare()` или `run_single()`.
- Ground Truth маска задаётся при инициализации (`gt_mask`) и используется
  для расчёта метрик; если не указана — бенчмарк работает в режиме
  "только инференс + визуализация".
- Метод `compare()` автоматически освобождает VRAM после каждой модели,
  оставляя только метаданные в `self.models` для совместимости с API.
- Для валидации качества классических методов используйте `BatchClassicTester2`;
  для проверки консистентности реализаций — `BatchClassicTester`.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций

import sys
import os
import json
import time
import gc
import asyncio
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Callable,
    TypedDict,
    Union,
    Tuple,
)
from typing import Literal, cast, TypeAlias

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import torch

from segmenters.NeuralModelFactory import NeuralModelFactory, ModelType
from utils.palettes import ade_palette
from utils.utils import compute_metrics
from utils.strategies import segment_image_unified
from transformers import PreTrainedModel

import logging

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler: logging.StreamHandler = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Настройка путей проекта
project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & CONSTANTS
# ──────────────────────────────────────────────────────────────────────

num_classes_custom: int = 150
"""Общее число классов датасета, dtype=int."""

# Alias для удобства
ImageInput: TypeAlias = Union[str, Image.Image]
"""Входное изображение (путь, `np.ndarray` или `PIL.Image`), dtype=Union[str, np.ndarray, Image.Image]."""

PaletteType: TypeAlias = Optional[Union[List[List[int]], Callable[[], List[List[int]]]]]
"""Входной тип палитры, dtype=Optional[Union[List[List[int]], Callable[[], List[List[int]]]]]."""

DeviceType = Literal["cuda", "cpu"]
"""Тип текущего устройства, dtype=DeviceType."""


# 🔹 TypedDict для спецификации метрики
class MetricPlotSpec(TypedDict):
    """Спецификация метрики для построения графиков сравнения.

    Используется в методах визуализации бенчмарка для динамической настройки
    отображения различных метрик качества сегментации.

    Attributes:
        key (str): Ключ метрики в словаре `summary` (например, "mIoU", "pixel_acc").
        label (str): Человекочитаемая подпись для оси Y на графике.
        transform (Callable[[float], float]): Функция преобразования значения метрики
            перед отображением (например, умножение на 100 для перевода в проценты).

    Example:
        >>> spec: MetricPlotSpec = {
        ...     "key": "mIoU",
        ...     "label": "Mean IoU (%)",
        ...     "transform": lambda x: x * 100,
        ... }
        >>> value = 0.75
        >>> print(f"{spec['label']}: {spec['transform'](value):.2f}")
        Mean IoU (%): 75.00
    """

    key: str  # Ключ метрики в summary (например, "mIoU")
    label: str  # Подпись для оси Y
    transform: Callable[[float], float]  # Функция трансформации значения


# 🔹 Type alias для функции трансформации
TransformFunc = Callable[[float], float]
"""Функция преобразования числового значения метрики, dtype=Callable[[float], float].

Принимает сырое значение метрики (обычно в диапазоне [0, 1]) и возвращает
преобразованное значение для отображения (например, в процентах или с округлением).

Args:
    value (float): Исходное значение метрики.

Returns:
    float: Преобразованное значение для визуализации.

Example:
    >>> to_percent: TransformFunc = lambda x: x * 100
    >>> to_percent(0.842)
    84.2
"""


# ──────────────────────────────────────────────────────────────────────
class SegmentationBenchmark:
    """Полноценный бенчмарк для сравнения архитектур сегментации."""

    def __init__(
        self,
        device: str = "cuda",
        num_classes: int = num_classes_custom,
        ignore_index: int = 255,
        class_names: Optional[List] = None,
        gt_mask: Optional[Union[np.ndarray, Image.Image]] = None,
        palette: Optional[Union[List[List[int]], Callable[[], List[List[int]]]]] = None,
    ) -> None:
        """Бенчмарк для сравнительного анализа моделей семантической сегментации.

        Предоставляет унифицированный интерфейс для загрузки, инференса и оценки
        разнородных архитектур: от классических CNN (UNet, DeepLab) до современных
        Transformer-based моделей (SegFormer, Mask2Former) и универсальных сегментаторов
        (SAM, SAM2).

        Key Features:
            🔗 Fluent API: Цепочечная загрузка моделей через `load_*()` методы.
            🧠 Поддержка 15+ архитектур: Включая предобученные веса из HF, TorchVision, SMP.
            ⚡ Управление памятью: Автоматическая очистка VRAM между прогонами моделей.
            📊 Метрики: mIoU, Pixel Accuracy, F1-weighted, Per-Class IoU, Confusion Matrix.
            🎨 Визуализация: Бар-чарты, heatmaps, наложенные маски с alpha-blend.
            📤 Экспорт: CSV, JSON (с numpy-сериализацией), Markdown, LaTeX для публикаций.

        Workflow:
            1. Инициализация с указанием устройства и параметров датасета.
            2. Загрузка моделей через цепочку `load_unet().load_deeplab()...`.
            3. Запуск бенчмарка: `benchmark.compare(image, alpha=0.6)`.
            4. Анализ: `benchmark.get_summary()`, `benchmark.plot_all_metrics()`.
            5. Экспорт: `benchmark.export_latex_table()`, `benchmark.save_results()`.

        Note:
            - Ground Truth маска (`gt_mask`) обязательна для расчёта метрик качества.
            - Метод `compare()` автоматически освобождает VRAM после каждой модели,
            сохраняя только метаданные в `self.models` для совместимости с API.
            - Для бенчмарка классических (не-нейросетевых) методов используйте
            `BatchClassicTester2`; для валидации консистентности — `BatchClassicTester`.

        Example:
            >>> benchmark = SegmentationBenchmark(
            ...     device="cuda",
            ...     num_classes=150,
            ...     gt_mask=gt_array,
            ...     class_names=ADE20K_CLASSES,
            ... )
            >>> (benchmark
            ...     .load_unet_trained("./models/unet_best.pth")
            ...     .load_segformer("nvidia/segformer-b5-finetuned-ade-640-640")
            ...     .load_sam("./models/mobile_sam.pt"))
            >>> benchmark.compare(image="test.jpg", alpha=0.5)
            >>> summary = benchmark.get_summary()
            >>> print(f"Лучшая модель по mIoU: {max(summary, key=lambda x: x['mIoU'])}")
            >>> benchmark.plot_comparison_chart("mIoU", figsize=(10, 6))
            >>> benchmark.export_latex_table(caption="Segmentation Benchmark Results")
        """
        if callable(palette):
            resolved_palette = palette()
        elif palette is not None:
            resolved_palette = palette
        else:
            resolved_palette = ade_palette()

        self.device: str = device
        self.num_classes: int = int(num_classes)
        self.ignore_index: int = ignore_index
        self.models: Dict[str, Dict[str, Any]] = {}
        self.palette: List[List[int]] = resolved_palette
        self.results: Dict[str, Dict[str, Any]] = {}  # {model_name: {metrics, time, overlay, ...}}
        self.class_names: Optional[List] = class_names or [f"Class {i}" for i in range(num_classes)]
        self.gt_mask: Optional[Union[np.ndarray, Image.Image]] = gt_mask

    # ──────────────────────────────────────────────────────────────────────
    # ЗАГРУЗКА МОДЕЛЕЙ (Fluent Interface)
    # ──────────────────────────────────────────────────────────────────────
    def load_trained_model(
        self, key: str, model_type: ModelType, checkpoint_path: str, **kwargs: Any
    ) -> "SegmentationBenchmark":
        """Регистрация загруженной модели из чекпоинта для бенчмарка.

        Args:
            key: Уникальный идентификатор модели для доступа к результатам.
            model_type: Enum-тип модели для выбора правильного инференс-пайплайна.
            checkpoint_path: Путь к файлу `.pth` / `.pt`.
            **kwargs: Дополнительные аргументы для `NeuralModelFactory` (encoder_name, variant и т.д.).

        Returns:
            self: Для цепочки вызовов.
        """
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            device=cast(DeviceType, self.device),
            num_classes=self.num_classes,
            **kwargs,
        )
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
            "checkpoint": checkpoint_path,
        }
        print(f"✅ Loaded {key} from {checkpoint_path}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_all_trained_models(self, checkpoint_dir: str = "./../models") -> "SegmentationBenchmark":
        """Пакетная загрузка всех обученных моделей из директории.

        Args:
            checkpoint_dir: Базовая директория с `.pth` файлами.

        Returns:
            self: Для цепочки вызовов.
        """
        checkpoints: Dict[str, Tuple[ModelType, str, Dict[str, Any]]] = {
            "unet_trained": (
                ModelType.UNET_SMP,
                "unet_ade20k_best.pth",
                {"encoder_name": "resnet34"},
            ),
            "deeplab_trained": (ModelType.DEEPLAB_TV, "deeplab_ade20k_best.pth", {}),
            "fpn_mit_b5_trained": (
                ModelType.FPN_SMP,
                "fpn_mit_b5_ade20k_best.pth",
                {"encoder_name": "mit_b5"},
            ),
            "psp_mit_b5_trained": (
                ModelType.PSPNET_SMP,
                "psp_mit_b5_ade20k_best.pth",
                {"encoder_name": "mit_b5"},
            ),
            "fcn_resnet50_trained": (
                ModelType.FCN_TV,
                "fcn_resnet50_ade20k_best.pth",
                {"variant": "fcn_resnet50"},
            ),
            "segnet_trained": (
                ModelType.SEGNET,
                "segnet_ade20k_best.pth",
                {"encoder_name": "resnet34"},
            ),
        }
        for key, (model_type, checkpoint_file, kwargs) in checkpoints.items():
            checkpoint_path: str = os.path.join(checkpoint_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                self.load_trained_model(key, model_type, checkpoint_path, **kwargs)
            else:
                print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    # ЗАГРУЗКА ПРЕДОБУЧЕННЫХ МОДЕЛЕЙ
    # ──────────────────────────────────────────────────────────────────────
    def load_segformer(self, path: str) -> "SegmentationBenchmark":
        """Загрузка кастомного SegFormer из локального пути."""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.SEGFORMER,
            local_path=path,
            device=cast(DeviceType, self.device),
            num_classes=self.num_classes,
        )
        self.models["segformer"] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
        }
        print(f"✅ Loaded SegFormer from {path}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_segformer_variant(self, variant: str = "b2") -> "SegmentationBenchmark":
        """Загрузка разных версий SegFormer для сравнения.

        Args:
            variant: Архитектура / Версия модели (`"b0"`, `"b1"`, `"b2"`, `"b3"`, `"b4"`, `"b5"`).

        Returns:
            self: Для цепочки вызовов.

        Raises:
            ValueError: Если указана неподдерживаемая версия.
        """
        model, processor, model_type_str = NeuralModelFactory.load_segformer_variant(
            variant=variant,
            device=cast(DeviceType, self.device),
        )
        key: str = f"segformer_{variant}"
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
        }
        print(f"✅ Loaded SegFormer-{variant} from HuggingFace")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_mask2former(self, name: str = "facebook/mask2former-swin-base-ade-semantic") -> "SegmentationBenchmark":
        """Загрузка Mask2Former модели."""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.MASK2FORMER,
            model_name=name,
            device=cast(DeviceType, self.device),
            num_classes=self.num_classes,
        )
        self.models["mask2former"] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
        }
        print(f"✅ Loaded Mask2Former from {name}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_oneformer(self, name: str = "shi-labs/oneformer_ade20k_swin_large") -> "SegmentationBenchmark":
        """Загрузка OneFormer модели."""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.ONEFORMER,
            model_name=name,
            device=cast(DeviceType, self.device),
            num_classes=self.num_classes,
        )
        self.models["oneformer"] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
        }
        print(f"✅ Loaded OneFormer from {name}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_dpt(self, model_name: str = "Intel/dpt-large-ade") -> "SegmentationBenchmark":
        """Загрузка DPT модели."""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.DPT,
            model_name=model_name,
            device=cast(DeviceType, self.device),
            num_classes=self.num_classes,
        )
        self.models["dpt"] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
        }
        print(f"✅ Loaded DPT from {model_name}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_upernet(self, model_name: str = "openmmlab/upernet-convnext-small") -> "SegmentationBenchmark":
        """Загрузка UPerNet модели."""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.UPERNET,
            model_name=model_name,
            device=cast(DeviceType, self.device),
            num_classes=self.num_classes,
        )
        self.models["upernet"] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
        }
        print(f"✅ Loaded UPerNet from {model_name}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_sam(self, model_name: str = "mobile_sam.pt") -> "SegmentationBenchmark":
        """Загрузка SAM / SAM2 (Segment Anything)."""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.SAM,
            model_name=model_name,
            device=cast(DeviceType, self.device),
            num_classes=self.num_classes,
        )
        model_key: str = "sam2" if "sam2" in model_name.lower() else "sam"
        self.models[model_key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
        }
        print(f"✅ Loaded SAM from {model_name}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_fpn_mit_pretrained(
        self, variant: str = "b5", checkpoint_path: Optional[str] = None
    ) -> "SegmentationBenchmark":
        """Загрузка FPN + MiT (pretrained weights или custom checkpoint)."""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.FPN_SMP,
            device=cast(DeviceType, self.device),
            num_classes=self.num_classes,
            encoder_name=f"mit_{variant}",
            checkpoint_path=checkpoint_path,
        )
        key: str = f"fpn_mit_{variant}_pretrained"
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
            "checkpoint": checkpoint_path,
        }
        print(f"✅ Loaded FPN+MiT-{variant}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_psp_mit_pretrained(
        self, variant: str = "b5", checkpoint_path: str = "psp_smp_none"
    ) -> "SegmentationBenchmark":
        """Загрузка PSPNet + MiT."""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.PSPNET_SMP,
            device=cast(DeviceType, self.device),
            num_classes=self.num_classes,
            encoder_name=f"mit_{variant}",
            checkpoint_path=checkpoint_path,
        )
        key: str = f"psp_mit_{variant}_pretrained"
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
            "checkpoint": checkpoint_path,
        }
        print(f"✅ Loaded PSPNet+MiT-{variant}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_fcn_resnet50_pretrained(
        self, variant: str = "fcn_resnet50", checkpoint_path: str = "fcn_resnet50_none"
    ) -> "SegmentationBenchmark":
        """Загрузка FCN."""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.FCN_TV,
            device=cast(DeviceType, self.device),
            num_classes=self.num_classes,
            variant=variant,
            checkpoint_path=checkpoint_path,
        )
        key: str = f"fcn_{variant.replace('fcn_', '')}_pretrained"
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
            "checkpoint": checkpoint_path,
        }
        print(f"✅ Loaded {variant}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_segnet_pretrained(
        self, encoder_name: str = "resnet34", checkpoint_path: str = "segnet_none"
    ) -> "SegmentationBenchmark":
        """Загрузка SegNet."""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.SEGNET,
            device=cast(DeviceType, self.device),
            num_classes=self.num_classes,
            encoder_name=encoder_name,
            checkpoint_path=checkpoint_path,
        )
        key: str = f"segnet_{encoder_name.replace('-', '_')}_pretrained"
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
            "checkpoint": checkpoint_path,
        }
        print("✅ Loaded SegNet-like")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_mask_rcnn_pretrained(self, variant: str = "maskrcnn_resnet50_fpn") -> "SegmentationBenchmark":
        """Загрузка Mask R-CNN (Instance Segmentation)."""
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.MASKRCNN_TV,
            device=cast(DeviceType, self.device),
            num_classes=self.num_classes,
            variant=variant,
        )
        self.models["maskrcnn_pretrained"] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
        }
        print("✅ Loaded Mask R-CNN")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_unet_trained(
        self,
        checkpoint_path: str = "unet_ade20k_best.pth",
        encoder_name: str = "resnet34",
    ) -> "SegmentationBenchmark":
        """Загрузка ОБУЧЕННОЙ U-Net с чекпоинта.

        Args:
            checkpoint_path: Путь к файлу чекпоинта
            encoder_name: Название encoder для архитектуры

        Returns:
            self: Для цепочки вызовов
        """
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.UNET_SMP,
            device=cast(DeviceType, self.device),
            num_classes=self.num_classes,
            encoder_name=encoder_name,
            checkpoint_path=checkpoint_path,
        )
        key: str = "unet_pretrained"
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
            "checkpoint": checkpoint_path,
        }
        print("✅ Loaded Unet pretrained")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_deeplab_trained(self, checkpoint_path: str = "deeplab_ade20k_best.pth") -> "SegmentationBenchmark":
        """Загрузка ОБУЧЕННОЙ DeepLabV3+ с чекпоинта.

        Args:
            checkpoint_path: Путь к файлу чекпоинта

        Returns:
            self: Для цепочки вызовов
        """
        model, processor, model_type_str = NeuralModelFactory.create_model(
            model_type=ModelType.DEEPLAB_TV,
            device=cast(DeviceType, self.device),
            num_classes=self.num_classes,
            encoder_name=None,
            checkpoint_path=checkpoint_path,
        )
        key: str = "deeplab_pretrained"
        self.models[key] = {
            "model": model,
            "processor": processor,
            "type": model_type_str,
            "checkpoint": checkpoint_path,
        }
        print("✅ Loaded deeplab pretrained")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_maskformer(self, name: str = "facebook/maskformer-resnet50-ade20k-full") -> "SegmentationBenchmark":
        """Загрузка MaskFormer модели."""
        from transformers import (
            MaskFormerImageProcessor,
            MaskFormerForInstanceSegmentation,
        )

        processor = MaskFormerImageProcessor.from_pretrained(name)
        model = cast(PreTrainedModel, MaskFormerForInstanceSegmentation.from_pretrained(name))  # type: ignore[arg-type]
        model = model.to(self.device).eval()  # type: ignore[arg-type]

        self.models["maskformer"] = {
            "model": model,
            "processor": processor,
            "type": "maskformer",
        }
        print(f"✅ Loaded MaskFormer from {name}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_yolov8(self, model_name: str = "yolov8n-seg.pt") -> "SegmentationBenchmark":
        """Загрузка YOLOv8 Segment модели."""
        from ultralytics import YOLO

        model = YOLO(model_name)
        key: str = os.path.splitext(os.path.basename(model_name))[0]

        self.models[key] = {
            "model": model,
            "processor": None,
            "type": "yolov8",
        }
        print(f"✅ Loaded YOLOv8 from {model_name}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def load_all_pretrained_cnn(self, checkpoint_dir: str = "./checkpoints") -> "SegmentationBenchmark":
        """Пакетная загрузка всех CNN-бэкендов."""
        print("\n" + "=" * 60)
        print("📦 Loading all pre-trained CNN models for benchmark")
        print("=" * 60)
        self.load_mask_rcnn_pretrained(variant="maskrcnn_resnet50_fpn")
        fpn_checkpoint: str = os.path.join(checkpoint_dir, "fpn_mit_b5_best.pth")
        self.load_fpn_mit_pretrained(variant="b5", checkpoint_path=fpn_checkpoint)
        psp_checkpoint: str = os.path.join(checkpoint_dir, "psp_mit_b5_best.pth")
        self.load_psp_mit_pretrained(variant="b5", checkpoint_path=psp_checkpoint)
        self.load_fcn_resnet50_pretrained(variant="fcn_resnet50")
        self.load_segnet_pretrained(encoder_name="resnet34")
        print("\n✅ All pre-trained CNN models loaded!")
        print(f"   Total models in benchmark: {len(self.models)}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    # ИНФЕРЕНС И СБОР РЕЗУЛЬТАТОВ
    # ──────────────────────────────────────────────────────────────────────
    def run_single(
        self,
        image_input: Union[str, Image.Image],
        model_key: str,
        alpha: float = 0.6,
        log_logits: bool = True,
    ) -> Dict[str, Any]:
        """Запуск одной модели с замером времени и расчётом метрик.

        Args:
            image_input: Путь к изображению или `PIL.Image`.
            model_key: Ключ загруженной модели.
            alpha: Прозрачность наложения маски (0.0–1.0).
            log_logits: Флаг логирования промежуточных тензоров.

        Returns:
            dict: Ключи `overlay`, `mask`, `inference_time_ms`, `metrics`, `image_size`, `output_shape`, `unique_classes`.

        Raises:
            ValueError: Если `model_key` отсутствует в реестре.
        """
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not loaded. Available: {list(self.models.keys())}")

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
        n_classes: int = int(self.get_model_num_classes(model_key))
        torch.cuda.empty_cache()
        gc.collect()

        gt_mask_np: Optional[np.ndarray] = None
        if self.gt_mask is not None:
            gt_mask_np = np.array(self.gt_mask) if isinstance(self.gt_mask, Image.Image) else self.gt_mask

        # Замер времени + warm-up
        if model_type not in ["maskformer", "mask2former", "oneformer"]:
            for _ in range(2):
                _, _ = segment_image_unified(
                    model,
                    processor,
                    image,
                    model_type,
                    alpha=0.0,
                    palette=self.palette,
                    device=self.device,
                    num_classes=n_classes,
                    gt_mask=gt_mask_np,
                )

        torch.cuda.synchronize()
        t0: float = time.time()
        overlay, resd = segment_image_unified(
            model,
            processor,
            image,
            model_type,
            alpha=alpha,
            palette=self.palette,
            device=self.device,
            num_classes=n_classes,
            gt_mask=gt_mask_np,
        )

        torch.cuda.synchronize()
        inference_time: float = time.time() - t0
        mask = resd["mask"]

        # 📊 Метрики (если есть ground truth)
        metrics: Dict[str, Any] = {}
        print(f"gt_maske: {self.gt_mask}")
        if self.gt_mask is not None:
            gt_np: np.ndarray = np.array(self.gt_mask) if isinstance(self.gt_mask, Image.Image) else self.gt_mask
            metrics = compute_metrics(mask, gt_np, self.num_classes, self.ignore_index)

        print(f"Метрики {metrics}")
        result: Dict[str, Any] = {
            "model": model_type,
            "overlay": overlay,
            "mask": mask,
            "inference_time_ms": inference_time * 1000,
            "metrics": metrics,
            "image_size": image.size[::-1],
            "output_shape": mask.shape,
            "unique_classes": len(np.unique(mask)),
        }
        self.results[model_key] = result
        return result

    # ──────────────────────────────────────────────────────────────────────
    def predict(
        self,
        image_input: Union[str, Image.Image],
        model_key: str,
        alpha: float = 0.5,
        gt_mask: Optional[np.ndarray] = None,
    ) -> Image.Image:
        """Предсказание сегментации для одного изображения.

        Args:
            image_input: Путь к изображению или PIL.Image объект.
            model_key: Ключ загруженной модели.
            alpha: Прозрачность наложения маски.
            gt_mask: Ground truth маска для метрик (опционально).

        Returns:
            overlay: Изображение с наложенной маской.

        Raises:
            ValueError: Если модель с указанным ключом не загружена.
        """
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not loaded. Available: {list(self.models.keys())}")
        model_info: Dict[str, Any] = self.models[model_key]
        n_classes: int = self.get_model_num_classes(model_key)
        return segment_image_unified(
            model_info["model"],
            model_info["processor"],
            image_input,
            model_info["type"],
            alpha=alpha,
            palette=self.palette,
            device=self.device,
            num_classes=n_classes,
            gt_mask=gt_mask,
        )[0]

    # ──────────────────────────────────────────────────────────────────────
    def compare(self, image_input: Union[str, Image.Image], alpha: float = 0.6) -> Dict[str, Dict[str, Any]]:
        """Поочерёдный запуск всех загруженных моделей с управлением VRAM.

        После каждой модели удаляет `.model` и `.processor` из памяти,
        оставляя только ключи в `self.models` для совместимости с API.

        Args:
            image_input: Путь к изображению или PIL.Image объект.
            alpha: Прозрачность наложения маски.

        Returns:
            dict: Сводная таблица метрик по всем моделям `{model_key: {mIoU, pixel_acc, ...}}`.
        """
        print(f"🚀 Starting benchmark on {len(self.models)} models...")
        model_keys: List[str] = list(self.models.keys())
        for i, key in enumerate(model_keys):
            print(f"\n🔹 Running {key}...")
            self.run_single(image_input, key, alpha=alpha)
            if i < len(model_keys) - 1:
                del self.models[key]["model"]
                del self.models[key]["processor"]
                torch.cuda.empty_cache()
                gc.collect()
                print(f"   🗑️  Freed {key} from VRAM")
        return self.get_summary()

    # ──────────────────────────────────────────────────────────────────────
    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Извлекает агрегированные метрики из выполненных тестов.

        Returns:
            dict: `{model_key: {"mIoU": float, "pixel_acc": float, "f1_weighted": float, "time_ms": float, "unique_classes": int}}`
        """
        summary: Dict[str, Dict[str, Any]] = {}
        for key, res in self.results.items():
            summary[key] = {
                "mIoU": res["metrics"].get("mIoU", np.nan),
                "pixel_acc": res["metrics"].get("pixel_acc", np.nan),
                "f1_weighted": res["metrics"].get("f1_weighted", np.nan),
                "time_ms": res["inference_time_ms"],
                "unique_classes": res["unique_classes"],
            }
        return summary

    def get_summary_dataframe(self) -> pd.DataFrame:  # вместо Dict[str, Dict[str, Any]]
        """Возвращает сводку как DataFrame для удобной сортировки."""
        summary_list = []
        for key, res in self.results.items():
            summary_list.append(
                {
                    "model": key,
                    "mIoU": res["metrics"].get("mIoU", np.nan),
                    "pixel_acc": res["metrics"].get("pixel_acc", np.nan),
                    "f1_weighted": res["metrics"].get("f1_weighted", np.nan),
                    "time_ms": res["inference_time_ms"],
                    "unique_classes": res["unique_classes"],
                }
            )
        return pd.DataFrame(summary_list).set_index("model")

    # ──────────────────────────────────────────────────────────────────────
    def get_model_num_classes(self, model_key: str) -> int:
        """Эвристическое определение количества выходных каналов модели.

        Проверяет:
        1. HF `config.id2label`
        2. Torchvision `classifier[-1].out_channels`
        3. Последний `torch.nn.Conv2d` в архитектуре
        4. Fallback на `self.num_classes`

        Args:
            model_key: Ключ загруженной модели.

        Returns:
            num_classes (int): Количество выходных классов модели.
        """
        cfg = self.models[model_key]
        model = cfg["model"]
        model_type = cfg["type"]

        # HF модели
        if hasattr(model, "config") and hasattr(model.config, "id2label"):
            return len(model.config.id2label)

        # Torchvision
        if model_type in ["deeplab_tv", "fcn_tv"]:
            if hasattr(model, "classifier"):
                return int(model.classifier[-1].out_channels)

        # SMP / Custom: ищем последний Conv2d
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module.out_channels

        # Fallback
        return self.num_classes

    # ──────────────────────────────────────────────────────────────────────
    # ВИЗУАЛИЗАЦИЯ
    # ──────────────────────────────────────────────────────────────────────
    def plot_comparison_chart(
        self,
        metric_name: str,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        show_values: bool = True,
        path: str = "./data/ade20k_test_trained/plot_comparison_chart.jpg",
    ) -> None:
        """Строит бар-чарт сравнения одной метрики с автоформатированием.

        - Корректное масштабирование для маленьких значений
        - Автоматический выбор формата (проценты или десятичные)
        - Улучшенное размещение подписей
        """
        summary: Dict[str, Dict[str, Any]] = self.get_summary()
        if not summary:
            print("⚠️ No results to plot. Run compare() first.")
            return

        models: List[str] = list(summary.keys())
        values: List[float] = [summary[m].get(metric_name, np.nan) for m in models]

        # Фильтруем модели без данных
        valid: List[Tuple[str, float]] = [(m, v) for m, v in zip(models, values) if not np.isnan(v)]

        if not valid:
            print(f"⚠️ No valid data for metric '{metric_name}'")
            return

        models = [m for m, _ in valid]
        values = [v for _, v in valid]

        is_percentage: bool = metric_name in ["mIoU", "pixel_acc", "f1_weighted"]
        multiplier: int = 100 if is_percentage else 1

        max_val: float = max(values) * multiplier
        use_percent_format: bool = max_val < 1.0

        plt.figure(figsize=figsize)
        cmap = plt.get_cmap("Set2")
        colors: np.ndarray = cmap(np.linspace(0, 1, len(models)))

        # Создаем бары
        x_pos = range(len(models))
        bars = plt.bar(
            x_pos,
            [v * multiplier for v in values],
            color=colors,
            edgecolor="black",
            linewidth=1.2,
        )

        # Подписи значений с авто-форматированием
        if show_values:
            for bar, val in zip(bars, values):
                height = bar.get_height()

                if use_percent_format:
                    label = f"{val * 100:.3f}%"
                else:
                    label = f"{val:.3f}" if val >= 1 else f"{val:.3f}"
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

        # Настройка оси Y
        if use_percent_format:
            plt.ylabel(f"{metric_name} (%)", fontsize=10)
            plt.ylim(0, max_val * 1.2)
        else:
            plt.ylabel(metric_name, fontsize=10)
            if metric_name == "time_ms":
                plt.ylim(0, max(values) * 1.15)
            else:
                plt.ylim(0, max(values) * 1.15 if max(values) > 0 else 1)

        # Подписи моделей с переносом
        plt.xticks(
            x_pos,
            [m.replace("_", "\n") if len(m) > 10 else m for m in models],
            rotation=45,
            ha="right",
            fontsize=9,
        )

        plt.title(
            title or f"Model Comparison: {metric_name}",
            fontsize=12,
            fontweight="bold",
            pad=20,
        )
        plt.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", format="png")
        plt.show()
        plt.close()

    # ──────────────────────────────────────────────────────────────────────
    def plot_per_class_iou(
        self,
        top_k: int = 20,
        figsize: Tuple[int, int] = (14, 8),
        cmap: str = "RdYlGn",
        show_only_present_classes: bool = True,
        path: str = "./data/ade20k_test_trained/plot_per_class_iou.jpg",
    ) -> None:
        """Строит heatmap per-class IoU только для классов, присутствующих в GT."""
        data: List[np.ndarray] = []
        model_names: List[str] = []
        for model_name, res in self.results.items():
            iou_arr = res["metrics"].get("per_class_iou")
            if iou_arr is not None and len(iou_arr) > 0:
                iou_arr = np.array(iou_arr[: self.num_classes])
                if np.any(np.isfinite(iou_arr)):
                    data.append(iou_arr)
                    model_names.append(model_name)
                else:
                    print(f"⚠️  {model_name}: all NaN IoU")
        if not data:
            print("❌ No valid per-class IoU data.")
            return

        data_arr = np.array(data)  # type: np.ndarray
        if show_only_present_classes:
            valid_classes = np.any(np.isfinite(data_arr), axis=0)
            class_indices = np.where(valid_classes)[0]
        else:
            class_indices = np.arange(self.num_classes)

        if len(class_indices) == 0:
            print("❌ No classes with valid IoU found!")
            return

        # Берем top-k среди присутствующих классов
        mean_iou = np.nanmean(data_arr[:, class_indices], axis=0)
        top_indices_in_subset: np.ndarray = np.argsort(mean_iou)[::-1][:top_k]
        top_class_indices: np.ndarray = class_indices[top_indices_in_subset]

        # Подписи классов
        class_labels: List[str] = [f"Class {c}" for c in top_class_indices]
        data_filtered: np.ndarray = data_arr[:, top_class_indices]

        plt.figure(figsize=figsize)
        _ = sns.heatmap(
            data_filtered,
            xticklabels=class_labels,
            yticklabels=model_names,
            cmap=cmap,
            center=0.5,
            annot=False,
            cbar_kws={"label": "IoU"},
            vmin=0,
            vmax=1,
        )

        plt.title(f"Per-class IoU (top {len(top_class_indices)} present classes)")
        plt.xlabel("Class")
        plt.ylabel("Model")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", format="png")
        plt.show()
        plt.close()
        print("\n📊 Per-class IoU Statistics:")
        print(f"  Total classes in dataset: {self.num_classes}")
        print(f"  Classes present in GT: {len(class_indices)}")
        print(f"  Showing top {len(top_class_indices)} classes")
        print(f"  Mean IoU (all classes): {np.nanmean(data):.3f}")

    # ──────────────────────────────────────────────────────────────────────
    def plot_confusion_matrix(
        self,
        model_key: str,
        normalize: str = "true",
        figsize: Tuple[int, int] = (10, 8),
        show_values: bool = True,
        path: str = "./data/ade20k_test_trained/plot_confusion_matrix.jpg",
    ) -> None:
        """Визуализация матрицы ошибок с нормализацией."""
        if model_key not in self.results:
            print(f"⚠️  Model '{model_key}' not found.")
            return

        cm = self.results[model_key]["metrics"].get("confusion_matrix")
        if cm is None:
            print("⚠️  No confusion matrix available.")
            return

        # Нормализация
        if normalize == "true":
            cm_display = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)
            title_suffix = "(recall)"
            fmt = ".2f"
        elif normalize == "pred":
            cm_display = cm.astype("float") / (cm.sum(axis=0, keepdims=True) + 1e-8)
            title_suffix = "(precision)"
            fmt = ".2f"
        else:
            cm_display = cm
            title_suffix = "(counts)"
            fmt = "d"
        gt_classes: np.ndarray = np.where(cm.sum(axis=1) > 0)[0][:20]
        if len(gt_classes) == 0:
            print("⚠️  No ground truth classes found!")
            return

        class_labels: List[str] = [f"C{c}" for c in gt_classes]
        cm_subset = cm_display[np.ix_(gt_classes, gt_classes)]

        plt.figure(figsize=figsize)
        _ = sns.heatmap(
            cm_subset,
            xticklabels=class_labels,
            yticklabels=class_labels,
            cmap="Blues",
            annot=show_values,
            fmt=fmt,
            cbar_kws={"label": "Normalized count" if normalize else "Count"},
        )

        plt.title(f"Confusion Matrix: {model_key} {title_suffix}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", format="png")
        plt.show()
        plt.close()

    # ──────────────────────────────────────────────────────────────────────
    def plot_all_metrics(
        self,
        figsize: Tuple[int, int] = (15, 5),
        path: str = "./data/ade20k_test_trained/plot_all_metrix.jpg",
    ) -> None:
        """Строит сводные графики по всем основным метрикам.

        - Пропускаем пустые метрики до создания subplot
        - Уменьшаем шрифт подписей для длинных названий
        - Добавляем отступ для suptitle
        """
        summary: Dict[str, Dict[str, Any]] = self.get_summary()
        if not summary:
            print("⚠️ No results to plot.")
            return
        metrics_to_plot: List[MetricPlotSpec] = [
            {
                "key": "mIoU",
                "label": "Mean IoU ↑",
                "transform": lambda x: float(x * 100),
            },
            {
                "key": "pixel_acc",
                "label": "Pixel Accuracy ↑",
                "transform": lambda x: float(x * 100),
            },
            {
                "key": "time_ms",
                "label": "Inference Time ↓ (ms)",
                "transform": lambda x: float(x),
            },
        ]
        valid_metrics: List[Tuple[str, str, TransformFunc]] = []
        for spec in metrics_to_plot:
            key: str = spec["key"]
            label: str = spec["label"]
            transform: TransformFunc = spec["transform"]

            values: List[float] = [summary[m].get(key, np.nan) for m in summary]
            if any(not np.isnan(v) for v in values):
                valid_metrics.append((key, label, transform))

        if not valid_metrics:
            print("⚠️ No valid metrics to plot")
            return
        n_plots: int = len(valid_metrics)
        fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0] * n_plots / 3, figsize[1]))
        axes_list: List[Any] = [axes] if n_plots == 1 else axes  # type: ignore[assignment]

        cmap = plt.get_cmap("Set2")
        colors = cmap(np.linspace(0, 1, len(summary)))
        for ax, (metric_key, metric_label, transform_func) in zip(axes_list, valid_metrics):
            models = list(summary.keys())
            values = [transform_func(summary[m].get(metric_key, np.nan)) for m in models]

            # Фильтрация валидных данных
            valid: List[Tuple[str, float]] = [(m, v) for m, v in zip(models, values) if not np.isnan(v)]

            if valid:
                # Разделение кортежей с явными типами
                plot_models: Tuple[str, ...]
                plot_values: Tuple[float, ...]
                plot_models, plot_values = zip(*valid)

                fontsize: int = 7 if len(plot_models) > 10 else 8

                bars = ax.bar(
                    range(len(plot_models)),
                    plot_values,
                    color=colors[: len(plot_models)],
                    edgecolor="black",
                )
                for bar, val, name in zip(bars, plot_values, plot_models):
                    display_name: str = name.replace("_", "_\n") if len(name) > 15 else name
                    print(display_name)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=fontsize - 1,
                    )

                ax.set_xticks(range(len(plot_models)))
                ax.set_xticklabels(
                    [m.replace("_", "_\n") for m in plot_models],
                    rotation=45,
                    ha="right",
                    fontsize=fontsize,
                )

                ax.set_ylabel(metric_label, fontsize=9)
                ax.set_title(metric_key, fontsize=10, fontweight="bold")
                ax.grid(axis="y", alpha=0.3, linestyle="--")

                if metric_key == "time_ms":
                    ax.set_ylim(0, max(plot_values) * 1.2)
                else:
                    ax.set_ylim(0, 100)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="gray",
                )
                ax.set_title(metric_key, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle("Model Comparison Summary", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", format="png")
        plt.show()
        plt.close()

    # ──────────────────────────────────────────────────────────────────────
    def plot_summary(
        self,
        metrics: List[str] = ["mIoU", "pixel_acc", "time_ms"],
        path: str = "./data/ade20k_test_trained/plot_summary.jpg",
    ) -> None:
        """Визуализация сводных результатов."""
        summary: Dict[str, Dict[str, Any]] = self.get_summary()
        for metric in metrics:
            values: List[Any] = [summary[k].get(metric, np.nan) for k in summary]
            if all(np.isnan(v) for v in values):
                continue
            plt.figure(figsize=(10, 5))
            plt.bar(
                list(summary.keys()),
                values,
                color=plt.get_cmap("Set2")(np.linspace(0, 1, len(summary))),
            )
            plt.ylabel(metric)
            plt.title(f"Model Comparison: {metric}")
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            # plt.savefig(path.replace(".jpg", f"_{metric}.jpg"), dpi=300)
            plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", format="png")
            plt.show()
            plt.close()

    # ──────────────────────────────────────────────────────────────────────
    # ЭКСПОРТ РЕЗУЛЬТАТОВ
    # ──────────────────────────────────────────────────────────────────────
    def save_results(self, output_dir: str = "benchmark_results") -> None:
        """Сохранение всех результатов с корректной сериализацией numpy-типов (масок, оверлеев, сводки и детальных метрик).

        Args:
            output_dir: Директория для сохранения результатов
        """
        os.makedirs(output_dir, exist_ok=True)
        for key, res in self.results.items():
            if res["overlay"] is not None:
                res["overlay"].save(f"{output_dir}/overlay_{key}.jpg")
            if res["mask"] is not None:
                np.save(f"{output_dir}/mask_{key}.npy", res["mask"])

        # Сводная таблица
        df: pd.DataFrame = pd.DataFrame(self.get_summary()).T
        df.to_csv(f"{output_dir}/summary.csv")

        def convert_numpy_types(obj: Any) -> Any:
            """Рекурсивно конвертирует numpy-типы в Python-native для JSON."""
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
        detailed: Dict[str, Any] = {}
        for key, res in self.results.items():
            detailed[key] = {
                "inference_time_ms": float(res["inference_time_ms"]),
                "metrics": convert_numpy_types(res["metrics"]),
                "image_size": [int(x) for x in res["image_size"]],
                "output_shape": [int(x) for x in res["output_shape"]],
                "unique_classes": int(res["unique_classes"]),
            }

        with open(f"{output_dir}/detailed.json", "w") as f:
            json.dump(detailed, f, indent=2, default=str)
        print(f"✅ Results saved to {output_dir}/")

    # ──────────────────────────────────────────────────────────────────────
    def export_latex_table(self, caption: str = "Segmentation Benchmark Results") -> str:
        """Генерирует LaTeX-код таблицы для публикации.

        Args:
            caption: Заголовок таблицы для LaTeX.

        Returns:
            latex_code: Строка с LaTeX кодом таблицы.
        """
        summary: Dict[str, Dict[str, Any]] = self.get_summary()
        if not summary:
            return ""

        lines: List[str] = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{" + caption + r"}",
            r"\label{tab:benchmark}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"\textbf{Model} & \textbf{mIoU (\%)} & \textbf{Acc (\%)} & \textbf{Time (ms)} \\",
            r"\midrule",
        ]

        for model, metrics in summary.items():
            mIoU: str = f"{metrics['mIoU'] * 100:.3f}" if not np.isnan(metrics["mIoU"]) else "-"
            acc: str = f"{metrics['pixel_acc'] * 100:.3f}" if not np.isnan(metrics["pixel_acc"]) else "-"
            time: str = f"{metrics['time_ms']:.3f}"
            model_clean: str = model.replace("_", r"\_").replace("-", r"\-")
            lines.append(f"{model_clean} & {mIoU} & {acc} & {time} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    async def compare_step_by_step(
        self,
        image_input: Union[str, Image.Image],
        alpha: float = 0.6,
        task_id: Optional[str] = None,
        benchmark_tasks: Optional[Dict] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Асинхронный запуск бенчмарка с обновлением прогресса в реальном времени.

        Args:
            image_input: Изображение для инференса.
            alpha: Прозрачность наложения.
            task_id: ID задачи для внешнего трекера.
            benchmark_tasks: Словарь состояния задач (мутация in-place).

        Returns:
            dict: Сводная таблица метрик.
        """
        print(f"🚀 Starting step-by-step benchmark on {len(self.models)} models...")
        model_keys: List[str] = list(self.models.keys())

        # Прогресс: 50% -> 99% на этапе инференса (49% диапазона)
        progress_start: int = 50
        progress_range: int = 49

        for i, key in enumerate(model_keys):
            print(f"\n🔹 Running {key} ({i + 1}/{len(model_keys)})...")

            # 🔹 Обновляем прогресс ПЕРЕД запуском модели
            if task_id and benchmark_tasks:
                progress: float = progress_start + (i / len(model_keys)) * progress_range
                benchmark_tasks[task_id]["progress"] = progress
                benchmark_tasks[task_id]["message"] = f"🔍 Инференс {key} ({i + 1}/{len(model_keys)})..."
                await asyncio.sleep(0)

            # Запускаем инференс одной модели (синхронно)
            self.run_single(image_input, key, alpha=alpha)

            # 🔹 Обновляем прогресс ПОСЛЕ завершения модели
            if task_id and benchmark_tasks:
                benchmark_tasks[task_id]["progress"] = progress_start + ((i + 1) / len(model_keys)) * progress_range
                benchmark_tasks[task_id]["message"] = f"✅ {key} завершён"
                await asyncio.sleep(0)

            if i < len(model_keys) - 1:
                if key in self.models:
                    self.models[key].pop("model", None)
                    self.models[key].pop("processor", None)
                torch.cuda.empty_cache()
                gc.collect()
                print(f"   🗑️  Freed {key} from VRAM")

        return self.get_summary()


# ──────────────────────────────────────────────────────────────────────
def export_comparison_table(
    bench: SegmentationBenchmark, output_file: str = "./../reports/model_comparison.md"
) -> pd.DataFrame:
    """Экспорт сравнительной таблицы всех моделей в Markdown.

    Args:
        bench: Инициализированный и запущенный бенчмарк.
        output_file: Путь для сохранения `.md` файла.

    Returns:
        pd.DataFrame: Отформатированная таблица результатов.
    """
    df: pd.DataFrame = pd.DataFrame(bench.get_summary()).T.sort_values("mIoU", ascending=False)

    # Категоризация
    categories: Dict[str, str] = {
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
    df["Params (M)"] = df.get("params", pd.Series([0] * len(df))).round(1)

    # Markdown таблица
    md_table: str = df[["Category", "mIoU (%)", "pixel_acc", "Time (ms)", "unique_classes"]].to_markdown()

    with open(output_file, "w") as f:
        f.write("# Segmentation Models Comparison\n\n")
        f.write(md_table)
    print(f"✅ Table saved to {output_file}")
    return df
