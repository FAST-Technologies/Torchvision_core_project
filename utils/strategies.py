# utils/strategies.py

"""Универсальные стратегии инференса для нейросетевой сегментации.

Поддерживаемые задачи:
1. **Единый интерфейс**: `segment_image_unified()` для 20+ архитектур
2. **Авто-препроцессинг**: выбор нормализации через SMP/HF процессоры
3. **Постобработка**: ресайз к оригиналу, паддинг под output_stride, instance→semantic
4. **Визуализация**: создание overlay с настраиваемой прозрачностью и палитрой
5. **Метрики**: автоматический расчёт IoU, Dice, Accuracy при наличии GT

Ключевые особенности:
- ✅ Паттерн "Стратегия": реестр `INFERENCE_STRATEGIES` для авто-диспетчеризации
- ✅ Универсальный ввод: str/Path/URL, PIL, np.ndarray, torch.Tensor
- ✅ Авто-ресайз: все маски возвращаются в размере исходного изображения
- ✅ Паддинг: поддержка output_stride для FPN/PSPNet/DeepLab
- ✅ Instance→Semantic: конвертация масок SAM/YOLOv8 в семантическую карту
- ✅ Логирование: детальный вывод классов, метрик, времени при `verbose=True`

Типичный workflow:
```python
from utils.strategies import segment_image_unified
from segmenters.NeuralModelFactory import NeuralModelFactory, ModelType

# 1. Загрузка модели
model, processor, _ = NeuralModelFactory.create_model(
    ModelType.SEGFORMER,
    model_name="nvidia/segformer-b5-finetuned-ade-640-640",
    device="cuda"
)

# 2. Инференс с единым интерфейсом
overlay, info = segment_image_unified(
    model=model,
    processor=processor,
    image_input="test.jpg",
    model_type="segformer",
    device="cuda",
    alpha=0.7,
    verbose=True
)

# 3. Доступ к результатам
print(f"Time: {info['inference_time_ms']:.2f}ms, Classes: {info['unique_classes']}")
if info['metrics']:
    print(f"IoU: {info['metrics']['iou']:.3f}")
overlay.save("result.png")
```

Note:
- Все стратегии выполняют инференс в контексте `torch.no_grad()`.
- Для SMP-моделей препроцессинг определяется через `smp.encoders.get_preprocessing_fn()`.
- Ресайз масок использует `order=0` (nearest-neighbor) для сохранения целочисленных меток.
- При `verbose=True` автоматически генерируется Markdown-отчёт в `ADE20K_DIR`.
- Instance-сегментация (SAM/YOLOv8) назначает уникальный ID каждому объекту; для семантической сегментации используйте модели с class logits.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций
import os
import sys
import time

import requests
from io import BytesIO
from pathlib import Path
from typing import List, Union, Tuple, Dict, Any, Optional, Callable, Literal, TypeAlias

import torch
import numpy as np
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from scipy.ndimage import zoom
from PIL import Image

# Локальные импорты
from utils.utils import (
    extract_logits_info,
    compute_metrics,
    analyze_prediction,
    generate_class_report,
    export_class_report,
)
from utils.palettes import ade_palette
from utils.paths import ADE20K_DIR, ensure_dirs

import logging

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler: logging.StreamHandler = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)



project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from metrics.SegmentationMetrics import SegmentationMetrics

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & CONSTANTS
# ──────────────────────────────────────────────────────────────────────
ImageInput: TypeAlias = Union[str, Path, Image.Image, np.ndarray, torch.Tensor]
"""Входное изображение (путь, `np.ndarray` или `PIL.Image`), dtype=Union[str, torch.Tensor, np.ndarray, Image.Image]."""

MaskArray: TypeAlias = np.ndarray  # Binary/semantic mask: H×W, dtype uint8/int
"""Тип для бинарной маски сегментации: (H, W), dtype=uint8, значения {0, 255}."""

ImageArray: TypeAlias = np.ndarray  # RGB image: H×W×3, dtype uint8
"""Тип для входного изображения: (H, W) для grayscale или (H, W, 3) для RGB, dtype=uint8."""

ModelType: TypeAlias = Literal[
    "segformer",
    "maskformer",
    "mask2former",
    "oneformer",
    "dpt",
    "upernet",
    "deeplab_tv",
    "fcn_tv",
    "maskrcnn_tv",
    "unet_smp",
    "mit_smp",
    "fpn_mit",
    "psp_mit",
    "deeplab_smp",
    "segnet",
    "segnet_custom",
    "sam",
    "mobile_sam",
    "sam2",
    "yolov8",
    "yolov8n_seg",
    "yolov8s_seg",
    "yolov8m_seg",
]
"""Используемый тип модели, dtype=Literal."""

InferFunc: TypeAlias = Callable[[Any, Any, Image.Image, str], Tuple[MaskArray, Image.Image]]
"""Общий тип функции для инференса, dtype=Callable[[Any, Any, Image.Image, str], Tuple[MaskArray, Image.Image]]."""

project_root: Path = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

num_classes: int = 150
"""Общее число классов датасета, dtype=int."""


# ──────────────────────────────────────────────────────────────────────
def infer_segformer(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str = "cuda",
) -> Tuple[MaskArray, Image.Image]:
    """Инференс для SegFormer (HuggingFace Transformers).

    Возвращает семантическую маску в размере оригинального изображения.
    Поддерживает все варианты: B0, B1, B2, B3, B4, B5.

    Args:
        model: Загруженная модель `SegformerForSemanticSegmentation`.
        processor: `SegformerImageProcessor` для препроцессинга и постпроцессинга.
        image: Входное изображение `PIL.Image` в RGB.
        device: Устройство для вычислений (`"cuda"` или `"cpu"`).

    Returns:
        Tuple[np.ndarray, PIL.Image]:
        - `seg_map`: Семантическая маска `[H, W]`, dtype `uint8`, значения 0..149.
        - `image`: Оригинальное изображение (без изменений).

    Note:
        - Использует `post_process_semantic_segmentation` с `target_sizes` для ресайза к оригиналу.
        - Все вычисления выполняются в контексте `torch.no_grad()`.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    logits_info: Dict[str, Any] = extract_logits_info(outputs, "segformer")
    print(f"📈 SegFormer logits: {logits_info}")
    print(f"📈 SegFormer custom logits: {logits}")
    seg_map: MaskArray = (
        processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]  # [H, W] = [height, width]
        )[0]
        .cpu()
        .numpy()
    )
    return seg_map, image


# ──────────────────────────────────────────────────────────────────────
def infer_mask2former(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str = "cuda",
) -> Tuple[MaskArray, Image.Image]:
    """Инференс для Mask2Former (HuggingFace Transformers).

    Args:
        model: Загруженная модель `MCXg9A2HnWdvPyVuJosKiPA2iGvNGZYVsV`.
        processor: `Mask2FormerImageProcessor`.
        image: Входное изображение `PIL.Image` в RGB.
        device: Устройство для вычислений.

    Returns:
        Tuple[np.ndarray, PIL.Image]: Семантическая маска и оригинальное изображение.

    Raises:
        ValueError: Если изображение имеет нулевые размеры.
    """
    if image.width == 0 or image.height == 0:
        raise ValueError("Image has zero dimensions")

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_info: Dict[str, Any] = extract_logits_info(outputs, "mask2former")
    print(f"📈 Mask2Former logits: {logits_info}")
    result = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    predicted_mask: MaskArray = result.cpu().numpy()
    return predicted_mask, image


# ──────────────────────────────────────────────────────────────────────
def infer_oneformer(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str = "cuda",
) -> Tuple[MaskArray, Image.Image]:
    """Инференс для OneFormer (HuggingFace Transformers).

    Поддерживает мультитасковое обучение: в вызове указывается `task_inputs=["semantic"]`.

    Args:
        model: Загруженная модель `OneFormerForUniversalSegmentation`.
        processor: `OneFormerImageProcessor`.
        image: Входное изображение `PIL.Image` в RGB.
        device: Устройство для вычислений.

    Returns:
        Tuple[np.ndarray, PIL.Image]: Семантическая маска и оригинальное изображение.
    """
    inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_info: Dict[str, Any] = extract_logits_info(outputs, "oneformer")
    print(f"📈 OneFormer logits: {logits_info}")
    predicted_mask: MaskArray = (
        processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0].cpu().numpy()
    )
    return predicted_mask, image


# ──────────────────────────────────────────────────────────────────────
def infer_deeplab_torchvision(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str = "cuda",
    target_size: Tuple[int, int] = (512, 512),
) -> Tuple[MaskArray, Image.Image]:
    """Инференс для DeepLabV3+ из torchvision.

    Args:
        model: Загруженная модель `DeepLabV3` из `torchvision.models.segmentation`.
        processor: Не используется (для совместимости интерфейса).
        image: Входное изображение `PIL.Image`.
        device: Устройство для вычислений.
        target_size: Размер, к которому ресайзится изображение перед инференсом
            (должен совпадать с размером при обучении модели).

    Returns:
        Tuple[np.ndarray, PIL.Image]: Семантическая маска в размере оригинала и оригинальное изображение.
    """
    # Preprocessing
    preprocess = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Ресайз к target_size
    image_resized: Image.Image = image.resize(target_size, Image.Resampling.BILINEAR)
    input_tensor: torch.Tensor = preprocess(image_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_output = model(input_tensor)

    logits_info: Dict[str, Any] = extract_logits_info(raw_output, "deeplab_tv")
    print(f"📈 DeepLabV3+ logits: {logits_info}")

    # Извлечение логитов
    if isinstance(raw_output, dict):
        logits = raw_output["out"][0]
    else:
        logits = raw_output[0] if hasattr(raw_output, "__getitem__") else raw_output  # [C, H, W]

    predicted_mask: MaskArray = logits.argmax(0).cpu().numpy().astype(np.uint8)  # [H, W]

    # Ресайз к оригиналу
    if predicted_mask.shape != (image.size[1], image.size[0]):
        sh, sw = (
            image.size[1] / predicted_mask.shape[0],
            image.size[0] / predicted_mask.shape[1],
        )
        predicted_mask = zoom(predicted_mask, (sh, sw), order=0)
    return predicted_mask, image


# ──────────────────────────────────────────────────────────────────────
def infer_unet_smp(
    model: Any,
    processor: Any,
    image: Image.Image,
    encoder_name: str = "resnet34",
    device: str = "cuda",
    output_stride: int = 1,
) -> Tuple[MaskArray, Image.Image]:
    """Универсальный инференс для SMP-моделей (U-Net, SegNet) и кастомных архитектур.

    Особенности:
    - Авто-определение препроцессинга через `smp.encoders.get_preprocessing_fn`.
    - Паддинг под `output_stride` для моделей с downsample (FPN, PSPNet).
    - Кроппинг паддинга и ресайз к оригинальному размеру.

    Args:
        model: Загруженная модель из `segmentation_models_pytorch`.
        processor: Не используется (для совместимости интерфейса).
        image: Входное изображение `PIL.Image`.
        encoder_name: Название encoder'а для препроцессинга (по умолчанию `"resnet34"`).
        device: Устройство для вычислений.
        output_stride: Кратность размера (1 для U-Net/SegNet, 8/16/32 для других).

    Returns:
        Tuple[np.ndarray, PIL.Image]: Семантическая маска в размере оригинала и оригинальное изображение.
    """
    # Preprocessing function
    try:
        # Для SMP-моделей с encoder
        if hasattr(model, "encoder") and hasattr(model.encoder, "name"):
            encoder_name = model.encoder.name
            preprocess_fn = smp.encoders.get_preprocessing_fn(encoder_name, "imagenet")
        else:
            # Fallback для SegNet без encoder атрибута
            raise AttributeError("No encoder attribute")
    except Exception:
        # Fallback: стандартный ImageNet preprocessing для SegNet
        mean: np.ndarray = np.array([0.485, 0.456, 0.406])
        std: np.ndarray = np.array([0.229, 0.224, 0.225])

        def preprocess_fn(x: np.ndarray) -> np.ndarray:
            result: np.ndarray = (x.astype(np.float32) / 255.0 - mean) / std
            return result

    image_np: np.ndarray = np.array(image)

    # Preprocessing
    input_tensor = preprocess_fn(image_np)
    input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # Паддинг под output_stride
    pad_h, pad_w = 0, 0
    if output_stride > 1:
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        pad_h = (output_stride - h % output_stride) % output_stride
        pad_w = (output_stride - w % output_stride) % output_stride
        if pad_h > 0 or pad_w > 0:
            input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        outputs = model(input_tensor)  # [B, C, H, W]
    is_segnet: bool = "SegNet" in str(type(model))
    model_type = "segnet" if is_segnet else "unet_smp"
    logits_info: Dict[str, Any] = extract_logits_info(outputs, model_type)
    print(f"📈 {'SegNet' if is_segnet else 'SMP'} logits: {logits_info}")

    # Пост-процессинг: argmax + ресайз к оригиналу
    predicted_mask: MaskArray = outputs.argmax(1).squeeze(0).cpu().numpy()  # [H, W]

    # Кроппинг после паддинга
    if output_stride > 1 and (pad_h > 0 or pad_w > 0):
        predicted_mask = predicted_mask[: image.size[1], : image.size[0]]

    if predicted_mask.shape != image.size[::-1]:
        sh, sw = (
            image.size[1] / predicted_mask.shape[0],
            image.size[0] / predicted_mask.shape[1],
        )
        predicted_mask = zoom(predicted_mask, (sh, sw), order=0)
    return predicted_mask, image


# ──────────────────────────────────────────────────────────────────────
def infer_sam(model: Any, processor: Any, image: Image.Image, device: str = "cuda") -> Tuple[np.ndarray, Image.Image]:
    """Инференс для модели SAM."""
    img_w, img_h = image.size

    # Инференс (без prompts=None!)
    results = model(image)

    print("📈 MobileSAM: instance segmentation (no class logits)")
    if results[0].masks is not None:
        print(
            f"   Masks: {results[0].masks.data.shape}, conf: {results[0].boxes.conf if hasattr(results[0], 'boxes') else 'N/A'}"
        )
    # Создаём семантическую карту из инстанс-масок
    seg_map: MaskArray = np.zeros((img_h, img_w), dtype=np.uint8)

    if results[0].masks is not None:
        masks: np.ndarray = results[0].masks.data.cpu().numpy()
        for i, mask in enumerate(masks, start=1):
            mask_bin = (mask > 0.5).astype(np.uint8)
            mask_pil: Image.Image = Image.fromarray(mask_bin)
            mask_resized: np.ndarray = np.array(mask_pil.resize((img_w, img_h), Image.Resampling.NEAREST))
            empty = seg_map == 0
            # Теперь размеры совпадают: seg_map[H,W] и mask_resized[H,W]
            seg_map[empty & (mask_resized > 0)] = i
    print(seg_map)  # [H, W], dtype=uint8
    return seg_map, image


# ──────────────────────────────────────────────────────────────────────
def infer_dpt(model: Any, processor: Any, image: Image.Image, device: str = "cuda") -> Tuple[np.ndarray, Image.Image]:
    """Инференс для модели DPT."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_info: Dict[str, Any] = extract_logits_info(outputs, "dpt")
    print(f"📈 DPT logits: {logits_info}")

    print(f"📈 DPT logits: {outputs.logits.shape if hasattr(outputs, 'logits') else 'N/A'}")
    seg_map: MaskArray = (
        processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0].cpu().numpy()
    )
    return seg_map, image


# ──────────────────────────────────────────────────────────────────────
def infer_smp_model(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str = "cuda",
    output_stride: int = 32,
    log_logits: bool = True,
) -> Tuple[np.ndarray, Image.Image]:
    """Универсальный инференс для SMP-моделей (U-Net, FPN, PSPNet, DeepLabV3+) с авто-паддингом под output_stride.

    Args:
        model: название модели.
        processor: текущззий процессор модели.
        image: исходное изображение для обработки.
        device: текущее устройство (CPU/CUDA).
        output_stride: кратность размера (32 для FPN/DeepLab, 8 для PSPNet, 1 для U-Net).
        log_logits: флаг для логирования логитов.
    """
    orig_w, orig_h = image.size

    # Preprocessing
    try:
        encoder_name = model.encoder.name
    except Exception:
        encoder_name = "mit_b5"

    preprocess_fn = smp.encoders.get_preprocessing_fn(encoder_name, "imagenet")

    # Конвертация + препроцессинг
    image_np: np.ndarray = np.array(image)
    input_tensor = preprocess_fn(image_np)
    input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).float()

    # ПАДДИНГ под output_stride
    if output_stride > 1:
        h, w = input_tensor.shape[1], input_tensor.shape[2]
        pad_h = (output_stride - h % output_stride) % output_stride
        pad_w = (output_stride - w % output_stride) % output_stride

        if pad_h > 0 or pad_w > 0:
            input_tensor = torch.nn.functional.pad(
                input_tensor,
                (0, pad_w, 0, pad_h),
                mode="reflect",
            )
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Инференс
    with torch.no_grad():
        outputs = model(input_tensor)  # [B, C, H_pad, W_pad]

    if log_logits:
        logits_info: Dict[str, Any] = extract_logits_info(outputs, "smp")
        print(f"📈 SMP logits: {logits_info}")

    # Пост-процессинг: argmax + кроппинг к оригиналу
    pred_mask: np.ndarray = outputs.argmax(1).squeeze(0).cpu().numpy()  # [H_pad, W_pad]

    # КРОППИНГ к оригинальному размеру
    if output_stride > 1 and (pad_h > 0 or pad_w > 0):
        pred_mask = pred_mask[:orig_h, :orig_w]

    return pred_mask, image


# ──────────────────────────────────────────────────────────────────────
def infer_smp_model_fixed(
    model: Any,
    image: Image.Image,
    device: str = "cuda",
    output_stride: int = 32,
    target_size: Tuple[int, int] = (512, 512),
) -> Tuple[np.ndarray, Image.Image]:
    """Инференс для SMP-моделей.

    Args:
        model: название модели.
        image: исходное изображение для обработки.
        device: текущее устройство (CPU/CUDA).
        target_size: Размер для ресайза (должен совпадать с обучением!).
        output_stride: 32 для FPN, 8 для PSPNet, 1 для U-Net.
    """
    orig_w, orig_h = image.size

    # Ресайз к target_size
    image_resized: Image.Image = image.resize(target_size, Image.Resampling.BILINEAR)

    try:
        encoder_name = model.encoder.name
    except Exception:
        encoder_name = "resnet34"

    preprocess_fn = smp.encoders.get_preprocessing_fn(encoder_name, "imagenet")

    # Preprocessing
    image_np: np.ndarray = np.array(image_resized)
    input_tensor = preprocess_fn(image_np)
    input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).float()

    # Паддинг под output_stride
    h, w = input_tensor.shape[1], input_tensor.shape[2]
    pad_h = pad_w = 0
    if output_stride > 1:
        pad_h = (output_stride - h % output_stride) % output_stride
        pad_w = (output_stride - w % output_stride) % output_stride
        if pad_h > 0 or pad_w > 0:
            input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Инференс
    with torch.no_grad():
        outputs = model(input_tensor)

    logits_info: Dict[str, Any] = extract_logits_info(outputs, "smp")
    print(f"📈 SMP logits: {logits_info}")

    # Пост-процессинг
    pred_mask: np.ndarray = outputs.argmax(1).squeeze(0).cpu().numpy()

    # Кроппинг паддинга (target_size=(W,H) как в PIL, поэтому [1]=H, [0]=W)
    if pad_h > 0 or pad_w > 0:
        pred_mask = pred_mask[: target_size[1], : target_size[0]]

    # Ресайз к оригинальному размеру
    if pred_mask.shape != (orig_h, orig_w):
        sh, sw = orig_h / pred_mask.shape[0], orig_w / pred_mask.shape[1]
        pred_mask = zoom(pred_mask, (sh, sw), order=0)

    return pred_mask, image


# ──────────────────────────────────────────────────────────────────────
def infer_fcn_torchvision(
    model: Any, processor: Any, image: Image.Image, device: str = "cuda"
) -> Tuple[np.ndarray, Image.Image]:
    """Инференс для FCN из torchvision."""
    preprocess = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    logits_info: Dict[str, Any] = extract_logits_info(outputs, "fcn_tv")
    print(f"📈 FCN logits: {logits_info}")

    if isinstance(outputs, dict):
        logits = outputs["out"][0]  # [C, H, W]
    elif isinstance(outputs, (tuple, list)):
        logits = outputs[0] if isinstance(outputs[0], torch.Tensor) else outputs[0][0]
    else:
        logits = outputs[0] if hasattr(outputs, "__getitem__") else outputs

    pred_mask: np.ndarray = logits.argmax(0).cpu().numpy()

    # Ресайз к оригиналу
    if pred_mask.shape != image.size[::-1]:
        sh, sw = image.size[1] / pred_mask.shape[0], image.size[0] / pred_mask.shape[1]
        pred_mask = zoom(pred_mask, (sh, sw), order=0)

    return pred_mask, image


# ──────────────────────────────────────────────────────────────────────
def infer_fcn_torchvision_fixed(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str = "cuda",
    target_size: Tuple[int, int] = (512, 512),
) -> Tuple[np.ndarray, Image.Image]:
    """Инференс для модели FCN (fixed)."""
    orig_w, orig_h = image.size

    # Ресайз к target_size
    image_resized: Image.Image = image.resize(target_size, Image.Resampling.BILINEAR)

    # Preprocessing (ImageNet stats)
    preprocess = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor: np.ndarray = preprocess(image_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    if isinstance(outputs, dict):
        logits = outputs["out"][0]
    else:
        logits = outputs[0] if hasattr(outputs, "__getitem__") else outputs

    pred_mask: np.ndarray = logits.argmax(0).cpu().numpy()

    # Ресайз к оригиналу
    if pred_mask.shape != (orig_h, orig_w):
        sh, sw = orig_h / pred_mask.shape[0], orig_w / pred_mask.shape[1]
        pred_mask = zoom(pred_mask, (sh, sw), order=0)

    return pred_mask, image


# ──────────────────────────────────────────────────────────────────────
def infer_mask_rcnn(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str = "cuda",
    log_logits: bool = True,
    score_threshold: float = 0.5,
) -> Tuple[np.ndarray, Image.Image]:
    """Инференс Mask R-CNN с конверсией instance → semantic."""
    preprocess = T.Compose(
        [
            T.ToTensor(),
        ]
    )

    input_tensor: np.ndarray = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    logits_info: Dict[str, Any] = extract_logits_info(outputs, "maskrcnn_tv")
    print(f"📈 Mask R-CNN: {logits_info}")

    result = outputs[0]

    # Извлекаем маски и классы
    masks: np.ndarray = result["masks"].cpu().numpy()  # [N, 1, H, W]
    labels: np.ndarray = result["labels"].cpu().numpy()  # [N]
    scores: np.ndarray = result["scores"].cpu().numpy()  # [N]

    # Фильтруем по confidence
    valid: np.ndarray = scores > score_threshold
    masks = masks[valid]
    labels = labels[valid]

    print(f"📈 Mask R-CNN: {len(masks)} instances detected (score > {score_threshold})")
    print(f"   Detected {len(masks)} instances (score > {score_threshold})")
    if len(scores) > 0:
        print(f"   Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Конвертация instance → semantic
    # Создаём семантическую карту, объединяя все маски
    img_h, img_w = image.size[1], image.size[0]
    semantic_map: np.ndarray = np.zeros((img_h, img_w), dtype=np.uint8)
    for mask, label in zip(masks, labels):
        # mask: [1, H, W] → [H, W]
        mask_bin: np.ndarray = (mask[0] > 0.5).astype(np.uint8)

        # Ресайз
        if mask_bin.shape != (img_h, img_w):
            mask_pil: Image.Image = Image.fromarray(mask_bin)
            mask_bin = np.array(mask_pil.resize((img_w, img_h), Image.Resampling.NEAREST))

        semantic_map[mask_bin > 0] = label

    return semantic_map, image


# ──────────────────────────────────────────────────────────────────────
def infer_yolov8(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str = "cuda",
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
) -> Tuple[np.ndarray, Image.Image]:
    """Инференс для YOLOv8 segmentation.

    Конвертирует instance masks → semantic map.
    """
    img_h, img_w = image.size[1], image.size[0]

    # YOLO принимает numpy array или путь
    results = model.predict(
        source=np.array(image),
        conf=confidence,
        iou=iou_threshold,
        device=device if device == "cuda" else "cpu",
        verbose=False,
    )

    # Создаём семантическую карту из инстанс-масок
    semantic_map: np.ndarray = np.zeros((img_h, img_w), dtype=np.uint8)

    if results[0].masks is not None:
        masks: np.ndarray = results[0].masks.data.cpu().numpy()  # [N, H, W]
        for i, mask in enumerate(masks, start=1):
            mask_bin: np.ndarray = (mask > 0.5).astype(np.uint8)
            # Ресайз если нужно
            if mask_bin.shape != (img_h, img_w):
                mask_pil: Image.Image = Image.fromarray(mask_bin)
                mask_bin = np.array(mask_pil.resize((img_w, img_h), Image.Resampling.NEAREST))
            # Назначаем уникальный ID для каждого инстанса
            semantic_map[mask_bin > 0] = i

    return semantic_map, image


# ──────────────────────────────────────────────────────────────────────
class SegNet(torch.nn.Module):
    """Простая реализация SegNet для бенчмарка.

    Encoder-Decoder с max pooling indices.
    """

    def __init__(self, num_classes: int = num_classes) -> None:
        """Инициализация модели SegNet."""
        super().__init__()

        # Encoder (VGG16-like)
        self.enc1 = self._make_encoder(3, 64)
        self.enc2 = self._make_encoder(64, 128)
        self.enc3 = self._make_encoder(128, 256)
        self.enc4 = self._make_encoder(256, 512)
        self.enc5 = self._make_encoder(512, 512)

        # Decoder
        self.dec5 = self._make_decoder(512, 512)
        self.dec4 = self._make_decoder(512, 256)
        self.dec3 = self._make_decoder(256, 128)
        self.dec2 = self._make_decoder(128, 64)
        self.dec1 = self._make_decoder(64, 64)

        # Classifier
        self.classifier = torch.nn.Conv2d(64, num_classes, kernel_size=1)

    # ──────────────────────────────────────────────────────────────────────
    def _make_encoder(self, in_ch: int, out_ch: int) -> torch.nn.Sequential:
        """Создание энкодера для модели SegNet."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    # ──────────────────────────────────────────────────────────────────────
    def _make_decoder(self, in_ch: int, out_ch: int) -> torch.nn.Sequential:
        """Создание декодера для модели SegNet."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой ход для модели SegNet."""
        # Encoder
        e1, p1 = self._encode(self.enc1, x)
        e2, p2 = self._encode(self.enc2, p1)
        e3, p3 = self._encode(self.enc3, p2)
        e4, p4 = self._encode(self.enc4, p3)
        e5, p5 = self._encode(self.enc5, p4)

        # Decoder
        d5 = self._decode(self.dec5, p5, e5.size())
        d4 = self._decode(self.dec4, d5, e4.size())
        d3 = self._decode(self.dec3, d4, e3.size())
        d2 = self._decode(self.dec2, d3, e2.size())
        d1 = self._decode(self.dec1, d2, e1.size())

        result: torch.Tensor = self.classifier(d1)
        return result

    # ──────────────────────────────────────────────────────────────────────
    def _encode(self, encoder: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder step: conv → batchnorm → relu → maxpool.

        max_pool2d возвращает (output, indices), берём только output.
        """
        x = encoder(x)
        pooled, indices = torch.nn.functional.max_pool2d(x, 2, 2, return_indices=True)
        return x, pooled

    # ──────────────────────────────────────────────────────────────────────
    def _decode(self, decoder: torch.nn.Module, x: torch.Tensor, output_size: torch.Size) -> torch.Tensor:
        """Decoder step: upsample → conv → batchnorm → relu."""
        # Upsampling к размеру encoder features
        x = torch.nn.functional.interpolate(x, size=output_size[2:], mode="bilinear", align_corners=False)
        result: torch.Tensor = decoder(x)
        return result


# ──────────────────────────────────────────────────────────────────────
# УНИВЕРСАЛЬНАЯ ФУНКЦИЯ СЕГМЕНТАЦИИ
# ──────────────────────────────────────────────────────────────────────
def segment_image_unified(
    model: Any,
    processor: Any,
    image_input: ImageInput,
    model_type: ModelType,
    alpha: float = 0.5,
    palette: Optional[Union[List[List[int]], Callable[[], List[List[int]]]]] = None,
    device: str = "cuda",
    verbose: bool = True,
    num_classes: int = num_classes,
    class_names: Optional[Dict[int, str]] = None,
    gt_mask: Optional[MaskArray] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Универсальная функция сегментации для любой архитектуры.

    Автоматически:
    1. Загружает/конвертирует изображение из разных форматов.
    2. Выбирает стратегию инференса по `model_type`.
    3. Выполняет предсказание и постпроцессинг.
    4. (Опционально) Логирует детали и считает метрики при наличии GT.
    5. Создаёт overlay-визуализацию.

    Args:
        model: Загруженная модель (PyTorch module или HF transformers).
        processor: Процессор для препроцессинга (HF) или `None`.
        image_input: Входные данные: путь (строка/Path), URL, `PIL.Image`, `np.ndarray` или `torch.Tensor`.
        model_type: Тип модели из `INFERENCE_STRATEGIES` (см. ниже).
        alpha: Прозрачность наложения маски (0.0 = только фото, 1.0 = только маска).
        palette: Цветовая палитра для визуализации (список `[R, G, B]` на класс).
        device: Устройство для вычислений (`"cuda"` или `"cpu"`).
        verbose: Если `True`, выводит детали инференса в консоль.
        num_classes: Количество классов сегментации (по умолчанию 150 для ADE20K).
        class_names: Словарь `{class_id: class_name}` для логирования.
        gt_mask: Ground truth маска для расчёта метрик (опционально).

    Returns:
        Tuple[PIL.Image, Dict[str, Any]]:
        - `overlay`: Изображение с наложенной цветной маской.
        - `result_ Словарь с метаданными:
            ```python
            {
                "model": str,
                "overlay": PIL.Image,
                "mask": np.ndarray,
                "inference_time_ms": float,
                "metrics": Dict[str, float],  # если есть GT
                "image_size": Tuple[int, int],  # (H, W)
                "output_shape": Tuple[int, int],  # (H, W)
                "unique_classes": int,
                "class_stats": List[Tuple[int, str, int, float]],  # top-10 классов
            }
            ```

    Raises:
        ValueError: Если `model_type` не найден в `INFERENCE_STRATEGIES` или входной формат не поддерживается.

    Example:
        ```python
        overlay, info = segment_image_unified(
            model=segformer_model,
            processor=segformer_processor,
            image_input="test.jpg",
            model_type="segformer",
            gt_mask=gt_array,
        )
        print(f"IoU: {info['metrics']['iou']:.3f}")
        overlay.save("result.png")
        ```
    """
    # ──────────────────────────────────────────────────────────────
    # 1. Загрузка и нормализация изображения
    # ──────────────────────────────────────────────────────────────
    if isinstance(image_input, (str, Path)):
        path_str: str = str(image_input)
        if path_str.startswith(("http://", "https://")):
            resp: requests.Response = requests.get(path_str, timeout=30)
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            image = Image.open(path_str).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        # Конвертация numpy array в PIL.Image
        if len(image_input.shape) == 2:
            # Grayscale → RGB
            image = Image.fromarray(image_input).convert("RGB")
        elif len(image_input.shape) == 3:
            if image_input.shape[2] == 3:
                # RGB
                image = Image.fromarray(image_input)
            elif image_input.shape[2] == 4:
                # RGBA → RGB
                image = Image.fromarray(image_input).convert("RGB")
            else:
                raise ValueError(f"Unsupported number of channels: {image_input.shape[2]}")
        else:
            raise ValueError(f"Unsupported array shape: {image_input.shape}")
    elif isinstance(image_input, torch.Tensor):
        # Конвертация torch.Tensor -> np.ndarray -> PIL.Image
        tensor_np: np.ndarray = image_input.cpu().numpy()
        if tensor_np.ndim == 3 and tensor_np.shape[0] in [1, 3]:
            tensor_np = np.transpose(tensor_np, (1, 2, 0))  # CHW -> HWC
        if tensor_np.max() <= 1.0:
            tensor_np = (tensor_np * 255).astype(np.uint8)
        else:
            tensor_np = tensor_np.astype(np.uint8)
        image = Image.fromarray(tensor_np).convert("RGB")
    else:
        raise ValueError(
            f"Unsupported input type: {type(image_input)}. "
            f"Expected str, Path, PIL.Image, np.ndarray, or torch.Tensor"
        )

    t0: float = time.perf_counter()

    # ──────────────────────────────────────────────────────────────
    # 2. Выбор и выполнение стратегии инференса
    # ──────────────────────────────────────────────────────────────
    if model_type not in INFERENCE_STRATEGIES:
        available: List[str] = list(INFERENCE_STRATEGIES.keys())
        raise ValueError(f"Unknown model_type: {model_type}. Available: {available}")

    infer_func = INFERENCE_STRATEGIES[model_type]

    # Выполнение инференса (стратегии уже содержат torch.no_grad() внутри)
    seg_map: MaskArray
    seg_map, _ = infer_func(model=model, processor=processor, image=image, device=device)

    # ──────────────────────────────────────────────────────────────
    # 3. Логирование и метрики (если verbose)
    # ──────────────────────────────────────────────────────────────
    result_info: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────
    # 4. Создание overlay-визуализации
    # ──────────────────────────────────────────────────────────────
    overlay = _create_overlay_standalone(image, seg_map, alpha=alpha, palette=palette)
    if verbose:
        result_info = _log_inference_details_standalone(
            image=image,
            seg_map=seg_map,
            model_type=model_type,
            model=model,
            class_names=class_names,
            gt_mask=gt_mask,
            num_classes=num_classes,
            initial_time=t0,
            palette=palette,
        )
    else:
        # Минимальный result_info без verbose
        result_info = {
            "model": model_type,
            "overlay": overlay,
            "mask": seg_map,
            "inference_time_ms": (time.perf_counter() - t0) * 1000,
            "metrics": {},
            "image_size": image.size[::-1],
            "output_shape": seg_map.shape,
            "unique_classes": len(np.unique(seg_map)),
        }

    return overlay, result_info


# ──────────────────────────────────────────────────────────────────────
def _log_inference_details_standalone(
    image: Image.Image,
    seg_map: MaskArray,
    model_type: str,
    model: Any,
    class_names: Optional[Dict[int, str]] = None,
    gt_mask: Optional[MaskArray] = None,
    num_classes: int = num_classes,
    initial_time: float = 0.0,
    palette: Optional[Union[List[List[int]], Callable[[], List[List[int]]]]] = None,
) -> Dict[str, Any]:
    """Логирует детали инференса и рассчитывает метрики (если есть GT).

    Args:
        image: Оригинальное изображение.
        seg_map: Предсказанная семантическая маска.
        model_type: Тип модели.
        model: Экземпляр модели.
        class_names: Словарь имён классов.
        gt_mask: Ground truth для метрик.
        num_classes: Количество классов.
        initial_time: Время начала инференса (для замера).
        palette: Палитра для визуализации.

    Returns:
        Dict[str, Any]: Словарь с метаданными (см. `segment_image_unified`).
    """
    if callable(class_names):
        class_names = class_names()

    print(f"\n🔍 Model: {model_type}")
    print(f"   Mask shape: {seg_map.shape}, dtype: {seg_map.dtype}")

    ensure_dirs(ADE20K_DIR)

    unique_classes: np.ndarray = np.unique(seg_map)
    print(
        f"   Predicted classes ({len(unique_classes)}): {unique_classes[:20]}{'...' if len(unique_classes) > 20 else ''}"
    )

    total_pixels: int = seg_map.size
    print("   Top 5 classes by pixel count:")
    class_stats: List[Tuple[int, str, int, float]] = []
    for cls in unique_classes:
        count: int = np.sum(seg_map == cls)
        if count > 0:
            name_idx = cls - 1
            name: str = class_names.get(name_idx, class_names.get(cls, f"Class_{cls}")) if class_names else f"Class_{cls}"
            pct: float = 100 * count / total_pixels
            class_stats.append((cls, name, count, pct))
            print(f"     Class {cls:3d}: {count:6d} px ({pct:5.3f}%)")

    n_classes: Optional[int] = _get_num_classes_standalone(model, model_type, fallback=num_classes)

    class_stats.sort(key=lambda x: x[2], reverse=True)
    for cls, name, count, pct in class_stats[:10]:
        print(f"     {cls:3d}: {name:25s} {count:7,} px ({pct:5.3f}%)")

    # Вывод информации о классах
    if hasattr(model, "config") and hasattr(model.config, "id2label"):
        print(f"\n   Class names (from model.config.id2label, {n_classes} total):")
        print("\n   ✅ HF model: class names from config.id2label")
    elif model_type in ["deeplab_tv", "fcn_tv", "maskrcnn_tv"]:
        print(f"\n   ⚠️  Torchvision models: {n_classes} output channels")
        print("      Class mapping depends on training dataset (COCO/ADE20K)")
    elif model_type in ["unet_smp", "mit_smp", "fpn_mit", "psp_mit", "segnet_custom"] and n_classes is not None:
        print(f"\n   ⚠️  SMP/Custom models: {n_classes} output channels (indices 0..{n_classes - 1})")
        print("      Mapping to ADE20K names requires external label file")
    elif model_type in ["sam", "mobile_sam", "sam2"]:
        print(f"\n   ⚠️  SAM models: instance IDs (1..{len(unique_classes)}), not semantic classes")
        print("      Each detected object gets a unique ID, no class names")

    # Анализ предсказания
    try:
        class_names_fixed: Optional[Dict[Union[int, str], str]] = (
            class_names if class_names is None else {k: v for k, v in class_names.items()}
        )
        analyze_prediction(seg_map, class_names=class_names_fixed)
        report: Dict[str, Any] = generate_class_report(seg_map, class_names=class_names_fixed)
        print(f"\n📊 Coverage: {report['coverage_pct']}% valid pixels")
        print(f"🏆 Top class: {report['top_class']} ({report['top_class_pct']}%)")
        export_class_report(
            report,
            str(ADE20K_DIR / f"{model_type}_prediction_report.md"),
            format="markdown",
        )
    except Exception as e:
        print(f"⚠️  Analysis skipped: {e}")

    # Метрики если есть GT
    metrics: Dict[str, float] = {}
    if gt_mask is not None:
        try:
            gt_np = np.array(gt_mask) if isinstance(gt_mask, Image.Image) else gt_mask
            metrics = SegmentationMetrics.compute_segmentation_metrics(
                pred_mask=seg_map,
                gt_mask=gt_np,
                num_classes=num_classes,
                ignore_index=255,
                include_confusion_matrix=True,
                include_hausdorff=False,
            )
            print(f"   ✅ Metrics computed: mIoU={metrics.get('mIoU', 0):.4f}, binary_IoU={metrics.get('iou', 0):.4f}")
        except Exception as e:
            print(f"⚠️  Metrics computation failed: {e}")
    else:
        print("⚠️  GT mask not provided, metrics skipped")

    # Создание overlay
    alpha: float = 0.5
    overlay: Image.Image = _create_overlay_standalone(image, seg_map, alpha=alpha, palette=palette)
    inference_time: float = time.perf_counter() - initial_time

    return {
        "model": model_type,
        "overlay": overlay,
        "mask": seg_map,
        "inference_time_ms": inference_time * 1000,
        "metrics": metrics,
        "image_size": image.size[::-1],
        "output_shape": seg_map.shape,
        "unique_classes": len(unique_classes),
        "class_stats": class_stats[:10],
    }


# ──────────────────────────────────────────────────────────────────────
def _get_num_classes_standalone(
    model: Any,
    model_type: str,
    fallback: int = num_classes,
) -> Optional[int]:
    """Безопасно получает число выходных классов из модели.

    Проверяет в порядке:
    1. `model.config.id2label` (HF transformers)
    2. `model.classifier[-1].out_channels` (torchvision)
    3. Последний `torch.nn.Conv2d` в архитектуре (SMP/custom)
    4. `fallback` по умолчанию

    Args:
        model: Экземпляр модели.
        model_type: Тип модели для эвристик.
        fallback: Значение по умолчанию, если не удалось определить.

    Returns:
        Optional[int]: Количество классов или `None` для instance-сегментации (SAM).
    """
    try:
        if hasattr(model, "config") and hasattr(model.config, "id2label"):
            return len(model.config.id2label)

        if model_type in ["deeplab_tv", "fcn_tv"]:
            if hasattr(model, "classifier"):
                return int(model.classifier[-1].out_channels)
            elif hasattr(model, "out_channels"):
                return int(model.out_channels)

        if model_type in ["unet_smp", "mit_smp", "fpn_mit", "psp_mit", "segnet_custom"]:
            for module in reversed(list(model.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    return module.out_channels
            return fallback

        if model_type in ["sam", "mobile_sam", "sam2"]:
            return None
        return fallback
    except Exception:
        return fallback


# ──────────────────────────────────────────────────────────────────────
def _create_overlay_standalone(
    image: Image.Image,
    mask: MaskArray,
    alpha: float = 0.5,
    palette: Optional[Union[List[List[int]], Callable[[], List[List[int]]]]] = None,
) -> Image.Image:
    """Создаёт визуализацию: оригинал + цветная маска.

    Алгоритм:
    1. Загружает палитру (по умолчанию ADE20K).
    2. Для каждого класса назначает цвет из палитры.
    3. Блендит оригинал и цветную маску с коэффициентом `alpha`.

    Args:
        image: Оригинальное изображение `PIL.Image` в RGB.
        mask: Семантическая маска `[H, W]` с целочисленными метками классов.
        alpha: Коэффициент блендинга (0.0 = только фото, 1.0 = только маска).
        palette: Палитра цветов `[R, G, B]` для каждого класса.
            Если `None`, используется `ade_palette()`. Если `callable`, вызывается.

    Returns:
        PIL.Image: Изображение с наложенной цветной маской, режим `"RGB"`.
    """
    palette_resolved: Optional[List[List[int]]] = None
    if palette is None:
        palette_resolved = ade_palette()
    elif callable(palette):
        palette_resolved = palette()
    else:
        palette_resolved = palette

    palette_array: np.ndarray = np.array(palette_resolved, dtype=np.uint8)

    h, w = mask.shape
    color_mask: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)

    # Защита от выхода за пределы палитры
    max_label: int = int(mask.max())
    for label in range(max_label + 1):
        palette_idx = label  # ← КЛЮЧЕВОЙ ФИКС!
        if 0 <= palette_idx < len(palette_array):
            color_mask[mask == label] = palette_array[palette_idx]

    img_arr: np.ndarray = np.array(image.convert("RGB"))
    overlay: np.ndarray = (img_arr * (1 - alpha) + color_mask * alpha).astype(np.uint8)

    return Image.fromarray(overlay)


# ──────────────────────────────────────────────────────────────────────
# РЕЕСТР СТРАТЕГИЙ ИНФЕРЕНСА
# ──────────────────────────────────────────────────────────────────────
INFERENCE_STRATEGIES: Dict[str, Any] = {
    # === Transformer-based HuggingFace модели ===
    "segformer": infer_segformer,
    "segformer_b2": infer_segformer,
    "mask2former": infer_mask2former,
    "maskformer": infer_mask2former,
    "oneformer": infer_oneformer,
    "dpt": infer_dpt,
    "upernet": infer_mask2former,
    # Torchvision модели
    "deeplab_tv": infer_deeplab_torchvision,
    "fcn_tv": infer_fcn_torchvision_fixed,
    "maskrcnn_tv": infer_mask_rcnn,
    # SMP модели с разными output_stride
    "unet_smp": lambda model, processor, image, device: infer_smp_model_fixed(model, image, device, output_stride=1),
    "mit_smp": lambda model, processor, image, device: infer_smp_model_fixed(model, image, device, output_stride=1),
    "fpn_mit": lambda model, processor, image, device: infer_smp_model_fixed(model, image, device, output_stride=32),
    "fpn_effnet": lambda model, processor, image, device: infer_smp_model_fixed(model, image, device, output_stride=32),
    "fpn_se_resnext": lambda model, processor, image, device: infer_smp_model_fixed(
        model, image, device, output_stride=32
    ),
    "psp_mit": lambda model, processor, image, device: infer_smp_model_fixed(model, image, device, output_stride=8),
    "psp_effnet": lambda model, processor, image, device: infer_smp_model_fixed(model, image, device, output_stride=8),
    "psp_resnet": lambda model, processor, image, device: infer_smp_model_fixed(model, image, device, output_stride=8),
    "deeplab_smp": lambda model, processor, image, device: infer_smp_model_fixed(
        model, image, device, output_stride=16
    ),
    "segnet": lambda model, processor, image, device: infer_unet_smp(
        model, processor, image, device=device, output_stride=1
    ),
    "segnet_custom": lambda model, processor, image, device: infer_unet_smp(
        model, processor, image, device=device, output_stride=1
    ),
    # === SAM_Models ===
    "sam": infer_sam,
    "mobile_sam": infer_sam,
    "sam2": infer_sam,
    # === YOLOv8 ===
    "yolov8": infer_yolov8,
    "yolov8n_seg": infer_yolov8,
    "yolov8s_seg": infer_yolov8,
    "yolov8m_seg": infer_yolov8,
}
