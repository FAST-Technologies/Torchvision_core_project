# backend/main.py

r"""AutoSegmenter API — улучшенная версия FastAPI бэкенда.

Модуль предоставляет REST API для интеллектуальной сегментации изображений с поддержкой:
- Классических методов (пороговые, градиентные, кластеризация, активные контуры)
- Нейросетевых моделей (SegFormer, Mask2Former, SAM, YOLOv8, SMP, TorchVision)
- Автоматического выбора метода на основе характеристик изображения
- Расчёта метрик качества при наличии Ground Truth
- Генерации рекомендаций и визуализаций

Исправления и улучшения версии 2.0:
  1. Дублирующийся маршрут /api/methods убран.
  2. Поле best_for добавлено в рекомендации.
  3. Кеш нейронных моделей — модель грузится один раз (LRU, max 3).
  4. Bare except → except (json.JSONDecodeError, ValueError).
  5. Две функции b64-кодирования объединены в arr_to_b64.
  6. Добавлены /api/health и /api/cache_info.
  7. HTTP 422 вместо 500 при невалидных входных данных.
  8. elapsed_ms и library возвращаются клиенту.
  9. Пользовательские параметры корректно мёржатся с дефолтами.

Пример использования:
    ```bash
    # Запуск сервера
    python backend/main.py

    # Сегментация классическим методом
    curl -X POST http://localhost:8000/api/segment \\
    -F "file=@image.jpg" \\
    -F "mode=classical" \\
    -F "auto_select=true"

    # Сегментация нейросетью
    curl -X POST http://localhost:8000/api/segment \\
    -F "file=@image.jpg" \\
    -F "mode=neural" \\
    -F "task=semantic" \\
    -F "model=segformer_b2"

Attributes:
    _model_cache (Dict[str, Any]): LRU-кеш загруженных нейронных моделей.
    _CACHE_MAX (int): Максимальное количество моделей в кеше (по умолчанию 3).
    NEURAL_CONFIGS (NeuralConfigDict): Конфигурации предобученных нейросетей.
    auto_seg (AutoSegmenter): Глобальный экземпляр селектора методов.

Note:
    - Все эндпоинты возвращают JSON с base64-кодированными изображениями.
    - Для расчёта метрик необходимо передать GT-маску через параметр gt_mask.
    - Нейронные модели загружаются лениво и кешируются для повторного использования.
"""
# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
import os
import sys
import json
import io
import math
import logging
import time
import base64
from typing import (
    Dict,
    Any,
    Optional,
    List,
    Union,
    Callable,
    Literal,
    TypeAlias,
)
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# from fastapi.responses import JSONResponse

# Локальные импорты
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segmenters.AutoSegmenter import (
    AutoSegmenter,
    SegmentationGoal,
    METHODS_BY_LIBRARY,
    MethodProfile,
)
from metrics.SegmentationMetrics import SegmentationMetrics
from routers import benchmark, comparator, validator
from segmenters.NeuralSegmenter import NeuralSegmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autoseg")

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & CONSTANTS
# ──────────────────────────────────────────────────────────────────────
ImageArray: TypeAlias = np.ndarray
"""Тип для входного изображения: (H, W) для grayscale или (H, W, 3) для RGB, dtype=uint8."""

MaskArray: TypeAlias = np.ndarray
"""Тип для бинарной маски сегментации: (H, W), dtype=uint8, значения {0, 255}."""

MetricsDict: TypeAlias = Dict[str, Any]
"""Словарь метрик качества: {имя_метрики: значение}, например {"iou": 0.85, "dice": 0.91}."""

PathLike: TypeAlias = Union[str, Path]
"""Унифицированный тип для путей к файлам: строка или pathlib.Path."""

DeviceStr: TypeAlias = Literal["cuda", "cpu"]
"""Строковое обозначение устройства для выполнения вычислений."""

ModelConfigDict: TypeAlias = Dict[str, Dict[str, Any]]
"""Конфигурация модели: {имя_модели: {параметры}}."""

NeuralConfigDict: TypeAlias = Dict[str, ModelConfigDict]
"""Словарь конфигураций нейросетей по типам задач: {task: {model_name: config}}."""

RecommendationDict: TypeAlias = Dict[str, Any]
"""Словарь рекомендации метода: {метод: скор, время, IoU, параметры}."""

AnalysisDataDict: TypeAlias = Dict[str, Any]
"""Результаты анализа изображения: гистограмма, плотность границ, края в base64."""

ChartDict: TypeAlias = Dict[str, str]
"""Словарь графиков: {имя_файла: base64-строка изображения}."""

SegmentResponseDict: TypeAlias = Dict[str, Any]
"""Структура ответа эндпоинта /api/segment с маской, оверлеем и метаданными."""

# ──────────────────────────────────────────────────────────────────────
# КЕШ МОДЕЛЕЙ
# ──────────────────────────────────────────────────────────────────────
_model_cache: Dict[str, Any] = {}
"""LRU-кеш загруженных экземпляров NeuralSegmenter.

Ключ: JSON-строка с конфигурацией модели + тип задачи.

Значение: Экземпляр NeuralSegmenter, готовый к инференсу.

Алгоритм вытеснения:
    При достижении _CACHE_MAX удаляется старейшая запись (FIFO).
    Новая модель загружается и добавляется в кеш.
    Повторные запросы с той же конфигурацией используют закэшированный экземпляр.

Note:
    - Кеш очищается при завершении приложения через lifespan-контекст.
    - Для продакшена рекомендуется использовать Redis или внешний кеш-сервер.
"""

_CACHE_MAX: int = 3
"""Максимальное количество моделей, одновременно хранящихся в _model_cache."""

print(f"🔍 CWD: {os.getcwd()}")
print(f"🔍 __file__: {__file__}")


# ──────────────────────────────────────────────────────────────────────
def _get_or_load_neural(config: Dict[str, Any], task: str) -> Any:
    """Загружает нейронный сегментер с LRU-кешем (макс. 3 модели).

    Алгоритм:
    1. Формирует уникальный ключ кеша из конфигурации и типа задачи.
    2. Проверяет наличие модели в _model_cache.
    3. При отсутствии:
    - Вытесняет старейшую запись если кеш полон.
    - Загружает NeuralSegmenter с указанной конфигурацией.
    - Сохраняет экземпляр в кеш.
    4. Возвращает закэшированный или вновь загруженный экземпляр.

    Args:
        config (Dict[str, Any]): Конфигурация модели:
            - model_type: Тип архитектуры ("segformer", "mask2former", "sam", ...).
            - model_name: Имя модели в HuggingFace или путь к локальному чекпоинту.
            - Дополнительные параметры для инициализации NeuralSegmenter.
        task (str): Тип задачи сегментации:
            - "semantic": Семантическая сегментация (класс на пиксель).
            - "instance": Instance segmentation (отдельные объекты).
            - "panoptic": Паноптическая сегментация (вещи + вещи).

    Returns:
        NeuralSegmenter: Готовый к использованию экземпляр сегментера.

    Raises:
        RuntimeError: Если загрузка модели завершилась ошибкой.
        ValueError: Если конфигурация не содержит обязательных полей.

    Example:
        ```python
        config = {"model_type": "segformer", "model_name": "nvidia/segformer-b2-..."}
        model = _get_or_load_neural(config, task="semantic")
        result = model.segment_image(pil_image)
        ```

    Note:
        - Устройство (cuda/cpu) определяется автоматически при загрузке.
        - Повторные вызовы с идентичной конфигурацией возвращают тот же объект.
    """
    cache_key: str = json.dumps({**config, "_task": task}, sort_keys=True)
    if cache_key not in _model_cache:
        if len(_model_cache) >= _CACHE_MAX:
            oldest: str = next(iter(_model_cache))
            del _model_cache[oldest]
            logger.info("Model cache evicted oldest entry")
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {config.get('model_type')} on {device}")
        _model_cache[cache_key] = NeuralSegmenter(**config, device=device)
    return _model_cache[cache_key]


# ──────────────────────────────────────────────────────────────────────
# УТИЛИТЫ
# ──────────────────────────────────────────────────────────────────────
def arr_to_b64(arr: np.ndarray) -> str:
    """Конвертирует numpy-массив в base64 PNG-строку для передачи по HTTP.

    Алгоритм обработки:
    1. Приводит dtype к uint8 если необходимо (масштабирует [0,1] → [0,255]).
    2. Удаляет лишнее измерение для одноканальных изображений (H×W×1 → H×W).
    3. Конвертирует массив в PIL.Image и сохраняет в BytesIO в формате PNG.
    4. Кодирует бинарные данные в base64 и добавляет data: URI префикс.

    Args:
        arr (np.ndarray): Входной массив:
            - Форма: (H, W) для grayscale или (H, W, C) для RGB.
            - dtype: любой числовой тип (автоматически приводится к uint8).
            - Диапазон значений: [0, 1] (float) или [0, 255] (uint8).

    Returns:
        str: Строка формата `data:image/png;base64,...` готовая для вставки в <img src="">.

    Example:
        ```python
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        b64_str = arr_to_b64(mask)
        # b64_str: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
        ```

    Note:
        - Функция не изменяет входной массив (работает с копией при конвертации).
        - Для больших изображений (>4096×4096) рекомендуется предварительный ресайз.
        - PNG выбран как формат с потерями-без-потерь для бинарных масок.
    """
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ──────────────────────────────────────────────────────────────────────
def analyze_image_data(img_array: ImageArray) -> AnalysisDataDict:
    """Извлекает статистические данные изображения для визуализации и анализа.

    Вычисляемые метрики:
    1. **Гистограмма интенсивностей**: 64 бина, диапазон [0, 256].
    2. **Плотность границ**: Доля пикселей с градиентом > 30% от максимума.
    3. **Края Собеля**: Визуализация градиентов в base64 для предпросмотра.

    Алгоритм:
    1. Конвертация RGB → grayscale если необходимо (среднее по каналам).
    2. Вычисление гистограммы через np.histogram с 64 бинами.
    3. Применение оператора Собеля по осям X и Y через scipy.ndimage.
    4. Расчёт магнитуды градиента: |∇I| = √(Gx² + Gy²).
    5. Нормализация и конвертация краёв в base64 через arr_to_b64.

    Args:
        img_array (ImageArray): Входное изображение:
            - Форма: (H, W, 3) для RGB или (H, W) для grayscale.
            - dtype: uint8, диапазон [0, 255].

    Returns:
        AnalysisDataDict: Словарь с результатами анализа:
            - histogram (List[int]): Значения гистограммы (64 элемента).
            - hist_bins (List[float]): Границы бинов гистограммы (65 элементов).
            - edge_density (float): Доля пикселей-границ [0.0, 1.0].
            - edges_b64 (str): Base64-строка с визуализацией краёв Собеля.

    Example:
        ```python
        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        analysis = analyze_image_data(img)
        print(f"Edge density: {analysis['edge_density']:.2%}")
        ```

    Note:
        - Гистограмма вычисляется по всем каналам для RGB (после усреднения).
        - Порог для edge_density (30% от максимума) выбран эмпирически.
        - Визуализация краёв нормализуется к [0, 255] перед кодированием.
    """
    from scipy import ndimage

    # Гистограмма интенсивностей
    hist, bins = np.histogram(img_array.flatten(), bins=64, range=(0, 256))
    gray: np.ndarray = (
        np.mean(img_array, axis=2).astype(np.float32) if img_array.ndim == 3 else img_array.astype(np.float32)
    )
    sobel_x: np.ndarray = ndimage.sobel(gray, axis=0)
    sobel_y: np.ndarray = ndimage.sobel(gray, axis=1)
    edges: np.ndarray = np.hypot(sobel_x, sobel_y)
    edges_norm: np.ndarray = (edges / (edges.max() + 1e-8) * 255).astype(np.uint8)

    return {
        "histogram": hist.tolist(),
        "hist_bins": bins.tolist(),
        "edge_density": float(np.mean(edges > edges.max() * 0.3)),
        "edges_b64": arr_to_b64(edges_norm),
    }


# ──────────────────────────────────────────────────────────────────────
def sanitize_metrics(m: MetricsDict) -> MetricsDict:
    """Заменяет `inf` и `NaN` на `None` для JSON-совместимости.

    Проблема:
        Некоторые метрики (например, при делении на ноль) могут возвращать
        `float('inf')` или `float('nan')`, которые не сериализуются в JSON.

    Решение:
        Рекурсивно обходит словарь и заменяет проблемные значения на None.

    Args:
        m (MetricsDict): Словарь с метриками качества сегментации, например:
            {"iou": 0.85, "dice": 0.91, "precision": float('inf'), ...}

    Returns:
        MetricsDict: Очищенный словарь, безопасный для json.dumps():
            {"iou": 0.85, "dice": 0.91, "precision": None, ...}

    Example:
        ```python
        raw_metrics = {"iou": 0.8, "hausdorff": float('inf')}
        safe_metrics = sanitize_metrics(raw_metrics)
        # safe_metrics: {"iou": 0.8, "hausdorff": None}
        json.dumps(safe_metrics)  # ✅ Успешная сериализация
        ```

    Note:
        - Функция не изменяет исходный словарь (возвращает новый).
        - Замена на None позволяет фронтенду корректно отображать "N/A".
        - Для продакшена рекомендуется логировать случаи замены для отладки.
    """
    return {k: (None if isinstance(v, float) and (math.isinf(v) or math.isnan(v)) else v) for k, v in m.items()}


# ──────────────────────────────────────────────────────────────────────
def build_overlay(img: ImageArray, mask: MaskArray, alpha: float = 0.4) -> ImageArray:
    """Создаёт наложение маски на оригинальное изображение с прозрачностью.

    Алгоритм смешивания (alpha blending):
        ```
        result = img * (1 - alpha) + overlay_color * alpha
        ```
        где overlay_color — красный канал (255, 0, 0) для пикселей маски.

    Args:
        img (ImageArray): Оригинальное изображение:
            - Форма: (H, W) для grayscale или (H, W, 3) для RGB.
            - dtype: uint8, диапазон [0, 255].
        mask (MaskArray): Бинарная маска сегментации:
            - Форма: (H, W), dtype=uint8.
            - Значения: 0 (фон) или 255 (объект).
        alpha (float, optional): Прозрачность наложения [0.0, 1.0].
            - 0.0: только оригинальное изображение.
            - 1.0: только красная маска.
            - По умолчанию: 0.4 (баланс видимости).

    Returns:
        np.ndarray: RGB-изображение формы (H, W, 3), dtype=uint8:
            - Пиксели фона: оригинальные цвета с прозрачностью.
            - Пиксели объекта: смесь оригинала и красного (255, 0, 0).

    Example:
        ```python
        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        overlay = build_overlay(img, mask, alpha=0.5)
        # overlay: RGB изображение с полупрозрачной красной маской
        ```

    Note:
        - Grayscale изображения автоматически дублируются в 3 канала.
        - Маска с значениями != 0 и != 255 интерпретируется как бинарная (>0 → объект).
        - Для производительности используется векторизованная индексация массивов.
    """
    rgb: np.ndarray = np.stack([img] * 3, axis=-1) if img.ndim == 2 else img.copy()
    col: np.ndarray = np.zeros_like(rgb)
    col[mask > 0] = [255, 0, 0]
    return (rgb * (1 - alpha) + col * alpha).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────
def params_to_schema(params: Dict[str, Any]) -> Dict[str, Any]:
    """Генерирует JSON-схему UI-конфигуратора из параметров по умолчанию.

    Алгоритм определения типа и диапазона:
    1. **boolean**: Если значение — bool, схема: {"type": "boolean", "default": v}.
    2. **int**:
    - "Большие" параметры (size/bin/iter/scale/radius): max=500.
    - Остальные: max=100.
    - min=1, step=1.
    3. **float**:
    - "Нормализованные" (|v|≤2.0 или threshold/k/ratio/factor): max=1.0.
    - Остальные: max=100.0.
    - min=0.0, step=0.01.
    4. **string**: Для остальных типов: {"type": "string", "default": str(v)}.

    Args:
        params (Dict[str, Any]): Словарь параметров метода, например:
            {"threshold": 0.5, "window_size": 15, "k": -0.2, "enable": True}

    Returns:
        Dict[str, Any]: JSON-схема для динамической генерации UI-формы:
            {
                "threshold": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.5},
                "window_size": {"type": "int", "min": 1, "max": 500, "step": 1, "default": 15},
                "k": {"type": "float", "min": 0.0, "max": 100.0, "step": 0.01, "default": -0.2},
                "enable": {"type": "boolean", "default": True}
            }

    Example:
        ```python
        params = {"threshold": 0.5, "iterations": 100}
        schema = params_to_schema(params)
        # schema: {
        #   "threshold": {"type": "float", "min": 0.0, "max": 1.0, ...},
        #   "iterations": {"type": "int", "min": 1, "max": 500, ...}
        # }
        ```

    Note:
        - Эвристики для "больших" и "нормализованных" параметров можно расширять.
        - Схема совместима с библиотеками типа react-jsonschema-form.
        - Для сложных типов (списки, словари) требуется ручная доработка.
    """
    schema: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, bool):
            schema[k] = {"type": "boolean", "default": v}
        elif isinstance(v, int):
            big: bool = any(x in k for x in ("size", "bin", "iter", "scale", "radius"))
            schema[k] = {
                "type": "int",
                "min": 1,
                "max": 500 if big else 100,
                "step": 1,
                "default": v,
            }
        elif isinstance(v, float):
            norm: bool = abs(v) <= 2.0 or any(x in k for x in ("threshold", "k", "ratio", "factor"))
            schema[k] = {
                "type": "float",
                "min": 0.0,
                "max": 1.0 if norm else 100.0,
                "step": 0.01,
                "default": v,
            }
        else:
            schema[k] = {"type": "string", "default": str(v)}
    return schema


# ──────────────────────────────────────────────────────────────────────
def _best_for(method_name: str) -> List[str]:
    """Возвращает список типов изображений, для которых метод оптимален.

    Извлекает информацию из профилей бенчмарков AutoSegmenter.

    Args:
        method_name (str): Название метода сегментации, например:
            "otsu_thresholding", "canny_edge", "watershed", ...

    Returns:
        List[str]: Список значений `ImageType.value`, для которых метод
            показал наилучшие результаты в бенчмарках, например:
            ["document", "industrial"] для otsu_thresholding.

    Example:
        ```python
        best = _best_for("threshold_sauvola")
        # best: ["document", "microscopy"]
        ```

    Note:
        - Если метод не найден в бенчмарках, возвращается пустой список.
        - Значения соответствуют перечислению ImageType в AutoSegmenter.
        - Используется для отображения подсказок в пользовательском интерфейсе.
    """
    p: Optional[MethodProfile] = auto_seg.benchmark_data.get(method_name)
    return [t.value for t in p.best_for_type] if p else []


# ──────────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ НЕЙРОСЕТЕЙ
# ──────────────────────────────────────────────────────────────────────
NEURAL_CONFIGS: NeuralConfigDict = {
    """Конфигурации предобученных нейросетевых моделей по типам задач.

    Структура:
    ```
    {
        "semantic": {  # Семантическая сегментация
            "segformer_b0": {"model_type": "segformer", "model_name": "..."},
            "mask2former_swin_base": {"model_type": "mask2former", "model_name": "..."},
            ...
        },
        "instance": {  # Instance segmentation
            "mask2former_coco_instance": {...},
            "yolov8n_seg": {...},
            ...
        },
        "panoptic": {  # Паноптическая сегментация
            "mask2former_ade_panoptic": {...},
            ...
        }
    }
    ```

    Поддерживаемые архитектуры:
        - **SegFormer** (B0–B5): Transformer-based, ADE20K, 512×512 / 640×640.
        - **Mask2Former**: Универсальная, поддержка semantic/instance/panoptic.
        - **OneFormer**: Multi-task, ADE20K / COCO.
        - **DPT / UPerNet**: Dense prediction transformers.
        - **SMP** (Unet/FPN/PSP): С энкодерами ResNet, EfficientNet, MiT.
        - **TorchVision** (FCN/DeepLab/SegNet): Классические архитектуры.
        - **SAM / MobileSAM / SAM2**: Segment Anything Model.
        - **YOLOv8-seg**: Real-time instance segmentation.

    Note:
        - model_name может быть:
        - Идентификатором HuggingFace Hub (загружается автоматически).
        - Путём к локальному файлу .pt / .pth / .bin.
        - Для SMP-моделей дополнительно указывается encoder_name.
        - Конфигурации можно расширять без изменения кода эндпоинтов.
    """
    "semantic": {
        "segformer_b0": {
            "model_type": "segformer",
            "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
        },
        "segformer_b1": {
            "model_type": "segformer",
            "model_name": "nvidia/segformer-b1-finetuned-ade-512-512",
        },
        "segformer_b2": {
            "model_type": "segformer",
            "model_name": "nvidia/segformer-b2-finetuned-ade-512-512",
        },
        "segformer_b3": {
            "model_type": "segformer",
            "model_name": "nvidia/segformer-b3-finetuned-ade-640-640",
        },
        "segformer_b4": {
            "model_type": "segformer",
            "model_name": "nvidia/segformer-b4-finetuned-ade-640-640",
        },
        "segformer_b5": {
            "model_type": "segformer",
            "model_name": "nvidia/segformer-b5-finetuned-ade-640-640",
        },
        "mask2former_swin_base": {
            "model_type": "mask2former",
            "model_name": "facebook/mask2former-swin-base-ade-semantic",
        },
        "mask2former_swin_large": {
            "model_type": "mask2former",
            "model_name": "facebook/mask2former-swin-large-ade-semantic",
        },
        "oneformer_swin_large": {
            "model_type": "oneformer",
            "model_name": "shi-labs/oneformer_ade20k_swin_large",
        },
        "dpt_large": {"model_type": "dpt", "model_name": "Intel/dpt-large-ade"},
        "upernet_convnext_small": {
            "model_type": "upernet",
            "model_name": "openmmlab/upernet-convnext-small",
        },
        "unet_resnet34": {"model_type": "unet_smp", "encoder_name": "resnet34"},
        "unet_resnet50": {"model_type": "unet_smp", "encoder_name": "resnet50"},
        "unet_efficientnet_b0": {
            "model_type": "unet_smp",
            "encoder_name": "efficientnet-b0",
        },
        "unet_mit_b5": {"model_type": "unet_smp", "encoder_name": "mit_b5"},
        "fpn_mit_b5": {"model_type": "fpn_smp", "encoder_name": "mit_b5"},
        "fpn_efficientnet": {
            "model_type": "fpn_smp",
            "encoder_name": "efficientnet-b5",
        },
        "psp_mit_b5": {"model_type": "pspnet_smp", "encoder_name": "mit_b5"},
        "psp_resnet50": {"model_type": "pspnet_smp", "encoder_name": "resnet50"},
        "deeplab_resnet101": {"model_type": "deeplab_tv"},
        "fcn_resnet50": {"model_type": "fcn_tv", "variant": "fcn_resnet50"},
        "fcn_resnet101": {"model_type": "fcn_tv", "variant": "fcn_resnet101"},
        "segnet_resnet34": {"model_type": "segnet", "encoder_name": "resnet34"},
        "mobile_sam": {"model_type": "sam", "model_name": "mobile_sam.pt"},
        "sam2_tiny": {"model_type": "sam", "model_name": "sam2_t.pt"},
    },
    "instance": {
        "mask2former_coco_instance": {
            "model_type": "mask2former",
            "model_name": "facebook/mask2former-swin-base-coco-instance",
        },
        "maskformer_resnet50": {
            "model_type": "maskformer",
            "model_name": "facebook/maskformer-resnet50-ade20k-full",
        },
        "yolov8n_seg": {"model_type": "yolov8", "model_name": "yolov8n-seg.pt"},
        "yolov8s_seg": {"model_type": "yolov8", "model_name": "yolov8s-seg.pt"},
        "yolov8m_seg": {"model_type": "yolov8", "model_name": "yolov8m-seg.pt"},
        "maskrcnn_resnet50": {
            "model_type": "maskrcnn_tv",
            "variant": "maskrcnn_resnet50_fpn",
        },
        "maskrcnn_resnet50_v2": {
            "model_type": "maskrcnn_tv",
            "variant": "maskrcnn_resnet50_fpn_v2",
        },
        "mobile_sam": {"model_type": "sam", "model_name": "mobile_sam.pt"},
        "sam2_tiny": {"model_type": "sam", "model_name": "sam2_t.pt"},
    },
    "panoptic": {
        "mask2former_ade_panoptic": {
            "model_type": "mask2former",
            "model_name": "facebook/mask2former-swin-base-ade-panoptic",
        },
        "mask2former_coco_panoptic": {
            "model_type": "mask2former",
            "model_name": "facebook/mask2former-swin-base-coco-panoptic",
        },
        "oneformer_coco_panoptic": {
            "model_type": "oneformer",
            "model_name": "shi-labs/oneformer_coco_swin_large",
        },
    },
}


# def build_neural_configs() -> Dict[str, Dict[str, Dict]]:
#     """Авто-генерация NEURAL_CONFIGS из конфига фабрики"""
#     config = NeuralModelFactory.load_config()
#     result = {"semantic": {}, "instance": {}, "panoptic": {}}

#     # SegFormer
#     for variant, name in config["models"]["segformer"]["variants"].items():
#         result["semantic"][f"segformer_{variant}"] = {
#             "model_type": "segformer",
#             "model_name": name,
#         }

#     # Mask2Former
#     for variant, name in config["models"]["mask2former"]["variants"].items():
#         result["semantic"][f"mask2former_{variant}"] = {
#             "model_type": "mask2former",
#             "model_name": name,
#         }
#         result["instance"][f"mask2former_{variant}_instance"] = {
#             "model_type": "mask2former",
#             "model_name": name.replace("-semantic", "-coco-instance"),
#         }
#         result["panoptic"][f"mask2former_{variant}_panoptic"] = {
#             "model_type": "mask2former",
#             "model_name": name.replace("-semantic", "-coco-panoptic"),
#         }

#     # SMP модели
#     for encoder in config["models"]["unet"]["encoders"]:
#         result["semantic"][f"unet_{encoder}"] = {
#             "model_type": "unet_smp",
#             "encoder_name": encoder,
#         }

#     return result


# ──────────────────────────────────────────────────────────────────────
# FASTAPI ПРИЛОЖЕНИЕ
# ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Жизненный цикл приложения: инициализация и очистка ресурсов.

    Выполняется:
    1. **При старте** (до yield):
    - Логирование запуска API.
    - (Опционально) Предзагрузка часто используемых моделей.
    2. **При завершении** (после yield):
    - Очистка _model_cache для освобождения VRAM.
    - Логирование завершения работы.

    Args:
        app (FastAPI): Экземпляр приложения (не используется, но требуется сигнатурой).

    Yields:
        None: Передаёт управление основному циклу обработки запросов.

    Note:
        - Контекстный менеджер автоматически вызывается FastAPI.
        - Очистка кеша критична для предотвращения утечек памяти при hot-reload.
        - Для продакшена можно добавить health-checks и graceful shutdown.
    """
    logger.info("AutoSegmenter API starting…")
    yield
    _model_cache.clear()
    logger.info("Model cache cleared on shutdown")


app: FastAPI = FastAPI(title="AutoSegmenter API", version="2.0", lifespan=lifespan)
"""Экземпляр основного FastAPI приложения.

Конфигурация:
    title: Название API для OpenAPI docs (/docs).
    version: Версия для отслеживания изменений.
    lifespan: Контекстный менеджер для управления ресурсами.
    Middleware:
    CORSMiddleware: Разрешает запросы с любых источников (для разработки).

В продакшне следует ограничить allow_origins.

Registered routers:
    benchmark: Эндпоинты для массового тестирования методов.
    comparator: Матричные сравнения и попарный анализ.
    validator: Валидация согласованности реализаций.

Note:
    - Маршруты регистрируются до инициализации auto_seg для корректного импорта.
    - print-отладка зарегистрированных роутов помогает при отладке 404 ошибок.
"""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(benchmark.router)
app.include_router(comparator.router)
app.include_router(validator.router)
print(f"📋 Registered routes: {[r.path for r in app.routes if hasattr(r, 'path')]}")

auto_seg = AutoSegmenter()
"""Глобальный экземпляр AutoSegmenter для выбора и выполнения методов.

Инициализируется:
    При старте приложения (после регистрации роутов).
    С параметрами по умолчанию (goal=BALANCED, встроенные бенчмарки).

Использование:
    /api/segment: Основной эндпоинт сегментации.
    /api/methods_library: Получение списка доступных методов.
    /recommendations/: Генерация рекомендаций для изображения.

Note:
    - Экземпляр является singleton в рамках процесса.
    - Для многопроцессного развёртывания (gunicorn) требуется пересмотр архитектуры.
"""


# ──────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def log_benchmark_requests(request: Request, call_next: Callable[[Request], Any]) -> Any:
    """Мидлвэр логирования времени выполнения бенчмарк-запросов.

    Назначение:
    - Мониторинг производительности эндпоинтов /api/benchmark/*.
    - Выявление "медленных" запросов для оптимизации.
    - Сбор метрик для систем мониторинга (Prometheus, Grafana).

    Алгоритм:
    1. Проверяет, начинается ли путь запроса с "/api/benchmark".
    2. Замеряет время до и после обработки запроса через time.perf_counter().
    3. Логирует длительность выполнения в секундах с двумя знаками после запятой.
    4. Возвращает оригинальный response без изменений.

    Args:
        request (Request): Входящий HTTP-запрос (FastAPI).
        call_next (Callable[[Request], Any]): Следующий обработчик в цепочке.

    Returns:
        Any: Response от следующего обработчика (без модификаций).

    Note:
        - time.perf_counter() выбран за высокую точность (наносекунды).
        - Логирование только для /api/benchmark снижает накладные расходы.
        - Для продакшена рекомендуется отправлять метрики в time-series БД.
    """
    if request.url.path.startswith("/api/benchmark"):
        start: float = time.perf_counter()
        response = await call_next(request)
        duration: float = time.perf_counter() - start
        logger.info(f"Benchmark {request.url.path} took {duration:.2f}s")
        return response
    return await call_next(request)


# ── Routes ─────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health() -> Dict[str, Any]:
    """Возвращает статус системы: доступность CUDA, использование VRAM, активные задачи.

    Предназначение:
    - Health-check для оркестраторов (Kubernetes, Docker Swarm).
    - Мониторинг ресурсов для автоскейлинга.
    - Отладка проблем с памятью и загрузкой моделей.

    Возвращаемые метрики:
    - **cuda_available**: bool — доступен ли GPU.
    - **device_name**: str — название GPU (например, "NVIDIA RTX 4090").
    - **vram_mb**: float — общий объём VRAM в мегабайтах.
    - **vram_allocated_mb**: float — текущее использование VRAM.
    - **vram_free_mb**: float — свободная VRAM (total - allocated).
    - **reserved_vram_mb**: float — зарезервированная, но не использованная память.
    - **active_tasks**: int — количество запущенных бенчмарков.
    - **cached_models**: int — количество моделей в _model_cache.
    - **cache_max**: int — лимит размера кеша.

    Returns:
        Dict[str, Any]: Словарь со статусом и метриками, например:
            {
                "status": "ok",
                "cuda_available": True,
                "device_name": "NVIDIA GeForce RTX 4090",
                "vram_mb": 24576.0,
                "vram_allocated_mb": 1024.5,
                "vram_free_mb": 23551.5,
                "reserved_vram_mb": 2048.0,
                "active_tasks": 2,
                "cached_models": 1,
                "cache_max": 3
            }

    Example:
        ```bash
        curl http://localhost:8000/api/health | jq
        # {
        #   "status": "ok",
        #   "cuda_available": true,
        #   "device_name": "NVIDIA GeForce RTX 4090",
        #   "vram_mb": 24576.0,
        #   ...
        # }
        ```

    Note:
        - При отсутствии CUDA все VRAM-метрики возвращают 0.0.
        - active_tasks считывается из benchmark.benchmark_tasks (внешний модуль).
        - Эндпоинт не требует аутентификации — подходит для public health checks.
    """
    if torch.cuda.is_available():
        total_vram_mb: float = torch.cuda.get_device_properties(0).total_memory / 1024**2
        alloc_vram_mb: float = torch.cuda.memory_allocated(0) / 1024**2
        free_vram_mb: float = total_vram_mb - alloc_vram_mb
        reserved_vram_mb: float = torch.cuda.memory_reserved(0) / 1024**2
        device_name: str = torch.cuda.get_device_name(0)
    else:
        total_vram_mb = alloc_vram_mb = free_vram_mb = reserved_vram_mb = 0.0
        device_name = "cpu"

    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device_name": device_name,
        "vram_mb": total_vram_mb,
        "vram_allocated_mb": alloc_vram_mb,
        "vram_free_mb": free_vram_mb,
        "reserved_vram_mb": reserved_vram_mb,
        "active_tasks": len([t for t in benchmark.benchmark_tasks.values() if t.get("status") == "running"]),
        "cached_models": len(_model_cache),
        "cache_max": _CACHE_MAX,
    }


# ──────────────────────────────────────────────────────────────────────
@app.get("/api/cache_info")
async def cache_info() -> Dict[str, Any]:
    """Возвращает информацию о кеше загруженных нейронных моделей.

    Предназначение:
    - Отладка кеш-поведения при разработке.
    - Мониторинг использования памяти в продакшне.
    - Принятие решений о масштабировании (увеличение _CACHE_MAX).

    Возвращаемые данные:
    - **count**: int — текущее количество закэшированных моделей.
    - **models**: List[str] — список ключей кеша (обрезанных до 80 символов):
        - Ключ: JSON-строка с конфигурацией модели + тип задачи.
        - Пример: '{"model_type": "segformer", "model_name": "nvidia/...", "_task": "semantic"}'

    Returns:
        Dict[str, Any]: Словарь с информацией о кеше, например:
            {
                "count": 2,
                "models": [
                    '{"model_type": "segformer", "model_name": "nvidia/segformer-b2-...',
                    '{"model_type": "mask2former", "model_name": "facebook/mask2fo...'
                ]
            }

    Example:
        ```bash
        curl http://localhost:8000/api/cache_info | jq
        # {
        #   "count": 2,
        #   "models": ["{...}", "{...}"]
        # }
        ```

    Note:
        - Ключи обрезаются для избежания переполнения ответа при длинных путях.
        - Эндпоинт не требует аутентификации — подходит для мониторинга.
        - Для продакшена можно добавить метрики hit/miss ratio.
    """
    return {"count": len(_model_cache), "models": [k[:80] for k in _model_cache]}


# ──────────────────────────────────────────────────────────────────────
@app.get("/api/methods_library")
async def get_methods_by_library(
    library: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Получает методы сегментации из определённой библиотеки.

    Предназначение:
    - Динамическая генерация списка методов в пользовательском интерфейсе.
    - Получение параметров по умолчанию и схем для UI-конфигуратора.
    - Фильтрация методов по библиотеке (opencv/sklearn/torch).

    Алгоритм:
    1. Если указана library и она существует в METHODS_BY_LIBRARY:
    - Использует соответствующий подсловарь.
    2. Иначе:
    - Объединяет методы из всех библиотек в один словарь.
    3. Для каждого метода:
    - Если профиль — экземпляр MethodProfile:
        * Извлекает метрики (avg_iou, avg_time_ms, memory_mb, ...).
        * Преобразует best_for_type в список строк.
        * Генерирует схему через params_to_schema если schema отсутствует.
    - Иначе (fallback):
        * Использует .get() с дефолтными значениями.

    Args:
        library (Optional[str], optional): Название библиотеки для фильтрации:
            - "opencv": Методы на основе OpenCV.
            - "sklearn": Методы на основе scikit-learn.
            - "torch": Методы на основе PyTorch.
            - None: Вернуть все методы из всех библиотек.
            По умолчанию: None.

    Returns:
        Dict[str, Dict[str, Any]]: Словарь методов с метаданными:
            {
                "methods": {
                    "otsu_thresholding": {
                        "name": "otsu_thresholding",
                        "library": "opencv",
                        "avg_iou": 0.75,
                        "avg_time_ms": 15.0,
                        "memory_mb": 50,
                        "robustness": 0.8,
                        "description": "Автоматический порог Оцу...",
                        "best_for": ["document", "natural"],
                        "defaults": {},
                        "schema": {}
                    },
                    ...
                }
            }

    Example:
        ```bash
        # Все методы
        curl http://localhost:8000/api/methods_library | jq '.methods | keys'

        # Только OpenCV
        curl "http://localhost:8000/api/methods_library?library=opencv" | jq
        ```

    Note:
        - Эндпоинт не требует аутентификации — подходит для public API.
        - Схема параметров генерируется "на лету" если не задана в профиле.
        - Для больших списков методов можно добавить пагинацию.
    """
    source_dict: Dict[str, MethodProfile]
    if library and library in METHODS_BY_LIBRARY:
        source_dict = METHODS_BY_LIBRARY.get(library, {})
    else:
        source_dict = {
            name: profile for lib_methods in METHODS_BY_LIBRARY.values() for name, profile in lib_methods.items()
        }

    result: Dict[str, Any] = {}
    for name, profile in source_dict.items():
        if isinstance(profile, MethodProfile):
            result[name] = {
                "name": profile.name,
                "library": profile.library,
                "avg_iou": profile.avg_iou,
                "avg_time_ms": profile.avg_time_ms,
                "memory_mb": profile.memory_mb,
                "robustness": profile.robustness,
                "description": profile.description,
                "best_for": [t.value for t in profile.best_for_type],
                "defaults": profile.params if profile.params else {},
                "schema": (profile.schema if profile.schema else params_to_schema(profile.params)),
            }
        else:
            # Fallback
            result[name] = {
                "name": profile.get("name", name),
                "avg_iou": profile.get("avg_iou", 0.0),
                "description": profile.get("description", ""),
                "defaults": profile.get("params", {}),
                "schema": profile.get("schema", {}),
            }

    return {"methods": result}


# ──────────────────────────────────────────────────────────────────────
@app.get("/api/methods")
async def get_methods(library: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Возвращает доступные методы для указанной библиотеки (алиас `/api/methods_library`).

    Предназначение:
    - Обратная совместимость со старыми клиентами, использующими /api/methods.
    - Делегирование логики к get_methods_by_library для избежания дублирования.

    Алгоритм:
    1. Валидация library: если указана и не найдена в METHODS_BY_LIBRARY → HTTP 422.
    2. Выбор источника:
    - Если library указана: METHODS_BY_LIBRARY[library].
    - Иначе: объединение всех библиотек.
    3. Формирование ответа:
    - Для каждого профиля извлекает метрики, описание, параметры, схему.
    - Обрабатывает как экземпляры MethodProfile, так и "сырые" словари.

    Args:
        library (Optional[str], optional): Название библиотеки для фильтрации.
            См. get_methods_by_library для деталей.

    Returns:
        Dict[str, Dict[str, Any]]: Словарь методов, идентичный /api/methods_library.

    Raises:
        HTTPException (422): Если указана неизвестная библиотека.

    Example:
        ```bash
        # Ошибка: неизвестная библиотека
        curl "http://localhost:8000/api/methods?library=unknown"
        # {"detail": "Unknown library: 'unknown'. Available: ['opencv', 'sklearn', 'torch']"}

        # Успех: методы OpenCV
        curl "http://localhost:8000/api/methods?library=opencv" | jq '.methods | length'
        # 42
        ```

    Note:
        - Эндпоинт является алиасом — предпочтительно использовать /api/methods_library.
        - В будущих версиях может быть удалён для упрощения API.
        - Валидация библиотеки предотвращает ошибки на уровне бизнес-логики.
    """
    if library and library not in METHODS_BY_LIBRARY:
        raise HTTPException(
            422,
            f"Unknown library: {library!r}. Available: {list(METHODS_BY_LIBRARY.keys())}",
        )
    source: Dict[str, MethodProfile] = (
        METHODS_BY_LIBRARY.get(library, {})
        if library
        else {n: p for lib in METHODS_BY_LIBRARY.values() for n, p in lib.items()}
    )
    result: Dict[str, Any] = {}
    for name, profile in source.items():
        if isinstance(profile, MethodProfile):
            result[name] = {
                "name": profile.name,
                "library": profile.library,
                "avg_iou": profile.avg_iou,
                "avg_time_ms": profile.avg_time_ms,
                "memory_mb": profile.memory_mb,
                "robustness": profile.robustness,
                "description": profile.description,
                "best_for": [t.value for t in profile.best_for_type],
                "defaults": profile.params or {},
                "schema": (profile.schema if profile.schema else params_to_schema(profile.params or {})),
            }
        else:
            result[name] = {
                "name": profile.get("name", name),
                "avg_iou": profile.get("avg_iou", 0.0),
                "description": profile.get("description", ""),
                "defaults": profile.get("params", {}),
                "schema": profile.get("schema", {}),
            }
    return {"methods": result}


# ──────────────────────────────────────────────────────────────────────
@app.post("/api/segment")
async def segment(
    file: UploadFile = File(...),
    mode: str = Form("classical"),  # "classical" | "neural"
    task: str = Form("semantic"),
    model: str = Form("segformer_b2"),
    goal: str = Form("balanced"),
    auto_select: bool = Form(True),
    method: Optional[str] = Form(None),
    library: Optional[str] = Form("opencv"),
    custom_params: str = Form("{}"),
    gt_mask: Optional[UploadFile] = File(default=None),
) -> SegmentResponseDict:
    r"""Основной эндпоинт сегментации изображения.

    Поддерживает два режима работы:

    ## Классический режим (mode="classical")
    - Использует AutoSegmenter для выбора или выполнения метода.
    - Поддерживает авто-выбор (auto_select=True) или ручной выбор метода.
    - Библиотеки: OpenCV, scikit-learn, PyTorch.
    - Методы: пороговые, градиентные, кластеризация, активные контуры, watershed.

    ## Нейронный режим (mode="neural")
    - Использует NeuralSegmenter с предобученными моделями.
    - Поддерживает задачи: semantic, instance, panoptic.
    - Модели: SegFormer, Mask2Former, SAM, YOLOv8, SMP, TorchVision.
    - Автоматический выбор палитры и имён классов по типу задачи.

    ## Общие возможности
    - Расчёт метрик качества при наличии Ground Truth (gt_mask).
    - Генерация топ-5 рекомендаций методов для изображения.
    - Анализ изображения: гистограмма, плотность границ, сложность.
    - Возврат маски и оверлея в base64 для отображения в браузере.

    Параметры запроса:
    - **file** (UploadFile, required): Входное изображение (JPEG/PNG).
    - **mode** (str, default="classical"): Режим работы:
        - "classical": Классические алгоритмы.
        - "neural": Нейросетевые модели.
    - **task** (str, default="semantic"): Тип задачи для нейросетей:
        - "semantic": Семантическая сегментация.
        - "instance": Instance segmentation.
        - "panoptic": Паноптическая сегментация.
    - **model** (str, default="segformer_b2"): Имя модели для mode="neural".
    - **goal** (str, default="balanced"): Цель оптимизации для авто-выбора:
        - "speed": Максимальная скорость.
        - "accuracy": Максимальная точность.
        - "balanced": Баланс скорости и точности.
        - "low_memory": Минимальное потребление памяти.
    - **auto_select** (bool, default=True): Автоматически выбрать метод.
    - **method** (Optional[str]): Ручной выбор метода (если auto_select=False).
    - **library** (Optional[str], default="opencv"): Библиотека для классических методов.
    - **custom_params** (str, default="{}"): JSON-строка с пользовательскими параметрами.
    - **gt_mask** (Optional[UploadFile]): Опциональная GT-маска для расчёта метрик.

    Возвращаемые данные:
    - **success** (bool): Флаг успешного выполнения.
    - **method** (str): Имя использованного метода/модели.
    - **confidence** (float): Уверенность выбора [0.0, 1.0] (для классических методов).
    - **elapsed_ms** (float): Время выполнения в миллисекундах.
    - **mask_b64** (str): Base64-строка с бинарной маской.
    - **overlay_b64** (str): Base64-строка с оверлеем (изображение + маска).
    - **chars** (Dict): Характеристики изображения:
        - type: Предсказанный тип (document/natural/medical/...).
        - size: Разрешение (Ш×В).
        - contrast, noise, edge_density, complexity: Числовые метрики.
    - **metrics** (Dict): Метрики качества (если предоставлен gt_mask):
        - iou, dice, precision, recall, f1_score, mae, hausdorff, ...
    - **recommendations** (List[Dict]): Топ-5 рекомендаций методов:
        - method, score, estimated_time_ms, estimated_iou, best_for.
    - **analysis** (Dict): Результаты анализа изображения (гистограмма, края).
    - **examples** (Dict): Примеры методов для разных типов изображений.

    Args:
        file: Входное изображение.
        mode: Режим работы ("classical" | "neural").
        task: Тип задачи ("semantic" | "instance" | "panoptic").
        model: Имя нейронной модели (если mode="neural").
        goal: Цель оптимизации ("speed" | "accuracy" | "balanced" | "low_memory").
        auto_select: Автоматически выбрать метод.
        method: Ручной выбор метода.
        library: Библиотека для классических методов.
        custom_params: JSON-строка с пользовательскими параметрами.
        gt_mask: Опциональная GT-маска для расчёта метрик.

    Returns:
        SegmentResponseDict: Словарь с маской, overlay, метриками, рекомендациями и временем.


    Raises:
        HTTPException (422): При невалидных входных данных:
            - Неизвестная библиотека или метод.
            - Невалидный JSON в custom_params.
            - Отсутствие обязательных параметров.
        HTTPException (500): При внутренних ошибках обработки.

    Example:
        ```bash
        # Классический метод с авто-выбором
        curl -X POST http://localhost:8000/api/segment \\
        -F "file=@document.jpg" \\
        -F "mode=classical" \\
        -F "auto_select=true" \\
        -F "goal=accuracy"

        # Нейронная модель с метриками
        curl -X POST http://localhost:8000/api/segment \\
        -F "file=@medical.png" \\
        -F "mode=neural" \\
        -F "task=semantic" \\
        -F "model=segformer_b2" \\
        -F "gt_mask=@ground_truth.png"
        ```

    Note:
        - Время выполнения включает загрузку модели (если не в кеше).
        - Для больших изображений (>4096×4096) рекомендуется предварительный ресайз.
        - Метрики рассчитываются только при наличии gt_mask и совпадении размеров.
    """
    t0: float = time.perf_counter()
    try:
        img_pil: Image.Image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_array: np.ndarray = np.array(img_pil)
        mask: np.ndarray
        metadata: Dict[str, Any]
        overlay_np: np.ndarray

        # ─── НЕЙРОННЫЙ РЕЖИМ ───────────────────────────────────────────────
        if mode == "neural":
            from segmenters.NeuralSegmenter import NeuralSegmenter
            from utils.strategies import segment_image_unified
            from utils.palettes import ade_palette, coco_palette, cityscapes_palette

            # 🔹 PALETTES: словарь, где значения — функции без аргументов, возвращающие палитру
            PALETTES: Dict[str, Callable[[], List[List[int]]]] = {
                "semantic": ade_palette,  # ADE20K: 150 классов
                "instance": coco_palette,  # COCO: 80 классов
                "panoptic": cityscapes_palette,  # Cityscapes: 19 классов
            }

            # 🔹 CLASS_FN: словарь, где значения — функции без аргументов, возвращающие имена классов
            CLASS_FN: Dict[str, Callable[[], Dict[int, str]]] = {
                "semantic": NeuralSegmenter.get_ade_class_names,
                "instance": NeuralSegmenter.get_coco_class_names,
                "panoptic": NeuralSegmenter.get_cityscapes_class_names,
            }

            cfg: Optional[Dict[str, Any]] = NEURAL_CONFIGS.get(task, {}).get(model)
            if not cfg:
                raise HTTPException(422, f"Unknown neural config: task={task!r} model={model!r}")
            ns = _get_or_load_neural(cfg, task)
            dev: str = "cuda" if torch.cuda.is_available() else "cpu"

            overlay_pil: Image.Image
            result_info: Dict[str, Any]
            overlay_pil, result_info = segment_image_unified(
                model=ns.model,
                processor=ns.processor,
                image_input=img_pil,
                model_type=cfg["model_type"],
                alpha=0.6,  # Прозрачность наложения
                palette=PALETTES[task],
                device=dev,
                verbose=False,
                num_classes=ns.num_classes,
                class_names=CLASS_FN[task](),
                gt_mask=None,
            )

            raw = result_info.get("mask")
            mask = raw if raw is not None else (np.array(overlay_pil)[:, :, 0] > 0).astype(np.uint8) * 255

            overlay_np = np.array(overlay_pil)

            metadata = {
                "method": model,
                "library": "neural",
                "task": task,
                "parameters": cfg,
                "confidence": 1.0,
                "image_characteristics": auto_seg.analyze_image(img_array),
                "inference_time_ms": result_info.get("inference_time_ms", 0),
                "unique_classes": result_info.get("unique_classes", 0),
            }
        else:
            auto_seg.goal = (
                SegmentationGoal(goal)
                if goal in ["speed", "accuracy", "balanced", "low_memory"]
                else SegmentationGoal.BALANCED
            )

            try:
                user_params: Dict[str, Any] = json.loads(custom_params)
            except (json.JSONDecodeError, ValueError):
                user_params = {}

            if auto_select:
                mask, metadata = auto_seg.segment(img_array, auto_select=True, library=library, return_metadata=True)
            else:
                if not method:
                    raise HTTPException(422, "method required when auto_select=False")

                if library not in METHODS_BY_LIBRARY:
                    raise HTTPException(422, f"Unknown library: {library!r}")
                if method not in METHODS_BY_LIBRARY[library]:
                    raise HTTPException(
                        422,
                        f"Method {method!r} not in {library!r}. "
                        f"Available: {list(METHODS_BY_LIBRARY[library].keys())}",
                    )

                profile: MethodProfile = METHODS_BY_LIBRARY[library][method]
                final_params: Dict[str, Any] = {**(profile.params or {}), **user_params}
                logger.info(f"🛠 Using params for {method}/{library} params={final_params}")

                segmenter = auto_seg._get_segmenter_class(method, library)(**final_params)
                result = segmenter.segment_with_mask(img_array)
                if isinstance(result, tuple) and len(result) == 2:
                    _, mask_opt = result
                    mask = mask_opt if mask_opt is not None else np.zeros_like(img_array[:, :, 0], dtype=np.uint8)
                else:
                    # Fallback для методов, возвращающих только одну маску
                    mask = result
                metadata = {
                    "method": method,
                    "library": library,
                    "parameters": final_params,
                    "confidence": 1.0,
                    "image_characteristics": auto_seg.analyze_image(img_array),
                }

        # ─── Метрики ───────────────────────────────────────────────────────
        metrics: MetricsDict = {}
        if gt_mask is not None:
            logger.info(f"✅ GT получен: {gt_mask.filename}")
            gt_array: np.ndarray = np.array(Image.open(io.BytesIO(await gt_mask.read())).convert("L"))
            metrics = sanitize_metrics(SegmentationMetrics.calculate_all_metrics(mask, gt_array, threshold=0.5))
        else:
            logger.warning("⚠️ GT не предоставлен, метрики не рассчитываются")

        # ─── Рекомендации & Анализ ──────────────────────────────────────────────────
        recommendations: List[RecommendationDict] = auto_seg.get_recommendations(img_array, top_k=5)
        analysis_data: AnalysisDataDict = analyze_image_data(img_array)
        chars = metadata["image_characteristics"]

        if mode == "neural":
            pass
        else:
            overlay_np = build_overlay(img_array, mask)

        return {
            "success": True,
            "method": metadata["method"],
            "confidence": float(metadata["confidence"]),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "mask_b64": arr_to_b64(mask),
            "overlay_b64": arr_to_b64(overlay_np),
            "chars": {
                "type": chars.estimated_type.value,
                "size": f"{chars.width}×{chars.height}",
                "contrast": round(float(chars.contrast), 4),
                "noise": round(float(chars.noise_level), 4),
                "channels": chars.channels,
                "mean_intensity": round(float(chars.mean_intensity), 4),
                "edge_density": round(float(chars.edge_density), 4),
                "complexity": round(float(chars.complexity_score), 4),
            },
            "metrics": metrics,
            "recommendations": [
                {
                    "method": r["method"],
                    "score": round(float(r["score"]), 4),
                    "estimated_time_ms": float(r.get("estimated_time_ms", 0)),
                    "estimated_iou": float(r.get("estimated_iou", 0)),
                    "best_for": _best_for(r["method"]),
                }
                for r in recommendations
            ],
            "analysis": analysis_data,
            "examples": {
                "medical": ["otsu", "sauvola", "phansalkar", "adaptive_thresholding"],
                "documents": ["otsu", "adaptive_thresholding", "bernsen", "niblack"],
                "nature": ["canny_edge", "sobel_edge", "watershed", "felzenszwalb"],
                "industrial": [
                    "adaptive_thresholding",
                    "bernsen",
                    "gradient_magnitude_direction",
                    "log_edge",
                ],
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        logger.error(f"❌ /api/segment error: {exc}\n{traceback.format_exc()}")
        raise HTTPException(500, str(exc))


# ──────────────────────────────────────────────────────────────────────
@app.get("/recommendations/")
async def get_recommendations_ep(
    file: UploadFile = File(...),
) -> Dict[str, List[RecommendationDict]]:
    r"""Возвращает топ-5 рекомендаций методов для загруженного изображения.

    Предназначение:
    - Быстрый предпросмотр подходящих методов без выполнения сегментации.
    - Помощь пользователю в выборе метода перед запуском тяжёлых вычислений.
    - Интеграция в UI для отображения подсказок и примеров.

    Алгоритм:
    1. Загружает изображение из UploadFile в numpy-массив.
    2. Вызывает auto_seg.get_recommendations(img, top_k=5).
    3. Возвращает список рекомендаций в формате JSON.

    Args:
        file (UploadFile, required): Входное изображение (JPEG/PNG).

    Returns:
        Dict[str, List[RecommendationDict]]: Словарь с ключом "recommendations":
            [
                {
                    "rank": 1,
                    "method": "threshold_sauvola",
                    "score": 0.92,
                    "estimated_time_ms": 42.0,
                    "estimated_iou": 0.88,
                    "parameters": {"window_size": 15, "k": 0.5, "r": 128},
                    "best_for": ["document", "microscopy"]
                },
                ... ещё 4 элемента
            ]

    Example:
        ```bash
        curl -X GET http://localhost:8000/recommendations/ \\
        -F "file=@document.jpg" | jq '.recommendations[0]'
        # {
        #   "rank": 1,
        #   "method": "threshold_sauvola",
        #   "score": 0.92,
        #   "estimated_time_ms": 42.0,
        #   "estimated_iou": 0.88,
        #   "parameters": {...},
        #   "best_for": ["document", "microscopy"]
        # }
        ```

    Note:
        - Рекомендации основаны на бенчмарках и характеристиках изображения.
        - Скор (score) — интегральная метрика, зависящая от цели (goal).
        - Эндпоинт не выполняет сегментацию — только анализ и ранжирование.
    """
    img: np.ndarray = np.array(Image.open(io.BytesIO(await file.read())))
    return {"recommendations": auto_seg.get_recommendations(img, top_k=5)}


# ──────────────────────────────────────────────────────────────────────
# STATIC FILES & ENTRY POINT
# ──────────────────────────────────────────────────────────────────────
_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
"""Путь к собранным статическим файлам фронтенда (React/Vue/Angular).

Используется для раздачи SPA (Single Page Application):
Если директория существует, монтируется как StaticFiles.

Все запросы, не совпадающие с API-роутами, перенаправляются на index.html.

Note:
    - Для разработки используйте npm run dev на фронтенде с CORS.
    - Для продакшена соберите фронтенд (npm run build) перед запуском.
    - Альтернатива: использовать отдельный nginx для раздачи статики.
"""

if os.path.exists(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
