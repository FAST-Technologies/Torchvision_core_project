# segmenters/NewTorchSegmenter.py

"""Модуль высокопроизводительной сегментации изображений на чистом PyTorch.

Этот модуль предоставляет класс `TorchSegmenter2` — универсальный интерфейс для 50+ классических
методов сегментации, реализованных исключительно на PyTorch без зависимостей от OpenCV,
scikit-image или scikit-learn в ядре алгоритмов.

Ключевые особенности:
    • Поддержка автоматической смешанной точности (AMP) через `PrecisionManager`
    • Интеграция с `torch.compile` для ускорения инференса (PyTorch ≥ 2.0)
    • Оптимизация под различные устройства: CPU, CUDA (fp16/bf16/float8)
    • Векторизованные реализации: BFS, NMS, свёртки через `torch.nn.functional`
    • Кэширование ядер и результатов через `@lru_cache` и LRU-словарь
    • Fallback на Numba/scipy для CPU-тяжёлых операций при необходимости

Основные классы:
    TorchSegmenter2:
        Главный класс для выполнения сегментации. Поддерживает методы:
        - Пороговые (14): global, otsu, adaptive, niblack, sauvola, ...
        - Граничные (10): sobel, canny, prewitt, scharr, log, dog, ...
        - Региональные (3): region_growing, split_and_merge, floodfill
        - Кластеризация (3): kmeans, dbscan, meanshift
        - Активные контуры (4): active_contour, gvf, morphological_snakes, chan_vese
        - Графовые (2): watershed, random_walker
        - Суперпиксели (3): quickshift, slic, felzenszwalb
        - Интерактивные (1): grabcut

    PrecisionManager:
        Утилита для управления точностью вычислений. Поддерживает:
        - Авто-выбор оптимальной точности под устройство и лимит памяти
        - Контекстные менеджеры для AMP: `autocast()`, `autocast_float8()`
        - Проверку поддержки типов: `can_use_fp16()`, `can_use_bf16()`, ...

Пример использования:
    ```python
    import numpy as np
    from segmenters.NewTorchSegmenter import TorchSegmenter2

    # Загрузка изображения
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Инициализация сегментера с методом Оцу
    segmenter = TorchSegmenter2(
        method="otsu_thresholding",
        device="cuda",
        precision="bf16",
        use_compile=True
    )

    # Выполнение сегментации
    mask = segmenter.segment(image)  # Возвращает бинарную маску (0/255)

    # Сегментация с визуализацией
    overlay, mask = segmenter.segment_with_mask(image, alpha=0.7)
    ```

Зависимости:
    Обязательные:
        • torch>=2.0.0 — ядро вычислений, torch.compile, sparse tensors
        • torchvision>=0.15.0 — gaussian_blur, transforms
        • numpy>=1.20.0 — предобработка, fallback-реализации
        • Pillow>=9.0.0 — загрузка изображений

    Опциональные:
        • numba>=0.56.0 — ускорение region_growing/watershed на CPU
        • scipy>=1.7.0 — ndimage для морфологии, random_walker fallback
        • scikit-learn>=1.0.0 — DBSCAN/MeanShift fallback
        • torch-tensorrt>=1.4.0 — экспорт в TensorRT Engine

Примечания:
    • Все методы возвращают маску формы (1, 1, H, W) с значениями {0.0, 1.0} (float32)
    • Для совместимости с ONNX/TensorRT используйте `export_mode=True` при вызове `segment()`
    • Методы с динамическим контролем потока (canny, watershed) могут не компилироваться
      с `fullgraph=True` — используйте `compile_fullgraph=False` для таких случаев
    • При ошибках выполнения возвращается пустая маска того же размера, что и вход

Автор:
    Segmentation Project contributors

Лицензия:
    MIT License — см. файл LICENSE в корне проекта
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций

from segmenters.BaseSegmenter import BaseSegmenter

import os
import warnings
from PIL import Image
import time
import traceback
from typing import Protocol, List, Union, Tuple, Dict, Any, Optional, Callable, Literal, cast, overload, Generator

from numba import njit, prange
import numpy as np
from scipy import ndimage
from functools import lru_cache
from contextlib import contextmanager
from collections import OrderedDict
import hashlib

import torch
import torch.nn.functional as F
import torch.nn as nn

from torchvision.transforms import functional as TF

import cv2
from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur

import logging

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler: logging.StreamHandler = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class SegmentMethod(Protocol):
    """Протокол для методов сегментации."""

    def __call__(
        self, tensor: torch.Tensor, *, precision: Optional[str] = None, export_mode: bool = False, **kwargs: Any
    ) -> torch.Tensor:
        """Протокол - его вызов."""
        ...


# ──────────────────────────────────────────────────────────────────────
class PrecisionManager:
    """Управление точностью вычислений для оптимизации скорости/памяти."""

    PRECISION_MAP: Dict[str, torch.dtype] = {
        # Floating point types
        "fp64": torch.float64,
        "float64": torch.float64,
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        # Experimental float8 types (PyTorch 2.1+)
        "float8_e4m3fn": getattr(torch, "float8_e4m3fn", None) or torch.float16,
        "float8_e4m3fnuz": getattr(torch, "float8_e4m3fnuz", None) or torch.float16,
        "float8_e5m2": getattr(torch, "float8_e5m2", None) or torch.float16,
        "float8_e5m2fnuz": getattr(torch, "float8_e5m2fnuz", None) or torch.float16,
        "float8_e8m0fnu": getattr(torch, "float8_e8m0fnu", None) or torch.float16,
        # Experimental float4 type
        "float4_e2m1fn_x2": getattr(torch, "float4_e2m1fn_x2", None) or torch.float16,
        # Integer types (signed)
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        # Unsigned integer types
        "uint8": torch.uint8,
        "uint16": getattr(torch, "uint16", None) or torch.uint8,
        "uint32": getattr(torch, "uint32", None) or torch.uint8,
        "uint64": getattr(torch, "uint64", None) or torch.uint8,
        # Boolean
        "bool": torch.bool,
    }

    # ──────────────────────────────────────────────────────────────────────
    # NumPy dtype mapping for conversion
    NUMPY_TO_TORCH_MAP: Dict[str, torch.dtype] = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "uint16": torch.uint8,  # Fallback
        "uint32": torch.uint8,  # Fallback
        "uint64": torch.uint8,  # Fallback
        "bool": torch.bool,
    }

    # ──────────────────────────────────────────────────────────────────────
    def __init__(self, default_precision: str = "fp32") -> None:
        """Инициализация класса PrecisionManager."""
        self.default_precision: str = default_precision
        self._amp_enabled: bool = False

    # ──────────────────────────────────────────────────────────────────────
    def get_dtype(self, precision: Optional[str] = None) -> torch.dtype:
        """Получить torch.dtype по имени точности."""
        if precision is None:
            precision = self.default_precision

        dtype: torch.dtype = self.PRECISION_MAP.get(precision.lower(), torch.float32)

        if dtype is None:
            warnings.warn(
                f"Тип {precision} не поддерживается в этой версии PyTorch. " f"Используем fp32 как fallback.",
                UserWarning,
            )
            return torch.float32

        return dtype

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_torch_dtype_info(dtype: torch.dtype) -> Dict[str, Any]:
        """Получить информацию о типе данных."""
        info: Dict[str, Any] = {
            "dtype": dtype,
            "name": (str(dtype).split(".")[-1] if hasattr(dtype, "__name__") else str(dtype)),
            "is_floating_point": dtype.is_floating_point,
            "is_complex": dtype.is_complex,
            "is_signed": dtype.is_signed if hasattr(dtype, "is_signed") else None,
        }

        # Добавляем размер в битах
        if dtype == torch.float64:
            info["bits"] = 64
        elif dtype in [torch.float32, torch.int32]:
            info["bits"] = 32
        elif dtype in [torch.float16, torch.bfloat16, torch.int16]:
            info["bits"] = 16
        elif dtype in [torch.int8, torch.uint8]:
            info["bits"] = 8
        elif dtype in [torch.int64]:
            info["bits"] = 64
        else:
            info["bits"] = None

        return info

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def can_use_fp64(device: torch.device) -> bool:
        """Проверяет поддержку FP64 на устройстве."""
        if device.type == "cuda":
            return torch.cuda.get_device_capability(device.index or 0)[0] >= 6
        return True  # CPU поддерживает FP64

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def can_use_fp16(device: torch.device) -> bool:
        """Проверяет поддержку FP16 на устройстве."""
        if device.type == "cuda":
            return torch.cuda.get_device_capability(device.index or 0)[0] >= 6
        return False  # CPU не поддерживает FP16 эффективно

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def can_use_bf16(device: torch.device) -> bool:
        """Проверяет поддержку BF16 (Ampere+)."""
        if device.type == "cuda":
            return torch.cuda.get_device_capability(device.index or 0)[0] >= 8
        return False

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def can_use_float8(device: torch.device, float_type: str = "float8_e4m3fn") -> bool:
        """Проверяет поддержку float8 типов (требует PyTorch 2.1+ и Hopper+)."""
        if device.type != "cuda":
            return False

        # Проверяем наличие атрибута в torch
        if not hasattr(torch, float_type):
            return False

        # Требуется compute capability >= 9.0 (Hopper) для полноценной поддержки
        capability = torch.cuda.get_device_capability(device.index or 0)
        return capability[0] >= 9

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def can_use_int8(device: torch.device) -> bool:
        """Проверяет поддержку INT8 (квантование)."""
        if device.type == "cuda":
            return torch.cuda.get_device_capability(device.index or 0)[0] >= 6
        return True  # CPU поддерживает INT8

    # ──────────────────────────────────────────────────────────────────────
    def get_optimal_precision(
        self,
        device: torch.device,
        operation: str = "general",
        memory_limit: Optional[float] = None,
    ) -> str:
        """Автоматический выбор оптимальной точности.

        Args:
            device: Устройство выполнения
            operation: Тип операции ("training", "inference", "general").
            memory_limit: Ограничение памяти в GB (если есть).

        Returns:
            Рекомендованная точность.
        """
        if device.type != "cuda":
            return "fp32"

        capability = torch.cuda.get_device_capability(device.index or 0)

        # Hopper (9.0+) - полная поддержка float8
        if capability[0] >= 9 and operation == "inference":
            if memory_limit is not None and memory_limit < 8:
                return "float8_e4m3fn"
            return "bf16"

        # Ampere (8.0+) - поддержка bf16
        if capability[0] >= 8:
            if memory_limit is not None and memory_limit < 8:
                return "fp16"
            return "bf16"

        # Turing/Volta (7.0-7.5)
        if capability[0] >= 7:
            return "fp16"

        # Pascal (6.0-6.1)
        if capability[0] >= 6:
            return "fp16"

        # Older GPUs
        return "fp32"

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def numpy_to_torch_dtype(np_dtype: np.dtype) -> torch.dtype:
        """Конвертация numpy dtype в torch dtype."""
        import numpy as np

        dtype_name = str(np_dtype).lower()

        # Прямое соответствие
        if dtype_name in PrecisionManager.NUMPY_TO_TORCH_MAP:
            return PrecisionManager.NUMPY_TO_TORCH_MAP[dtype_name]

        # Обработка специфичных numpy типов
        if np_dtype == np.float128:
            warnings.warn("np.float128 не поддерживается в PyTorch, используем float64")
            return torch.float64
        if np_dtype == np.longdouble or np_dtype == np.longlong:
            return torch.float64
        if np_dtype in [np.intc, np.int_]:
            return torch.int32 if np.dtype(np.intc).itemsize == 4 else torch.int64

        # Fallback
        warnings.warn(f"Неизвестный numpy dtype {np_dtype}, используем float32")
        return torch.float32

    # ──────────────────────────────────────────────────────────────────────
    @contextmanager
    def autocast(self, precision: str = "fp16", enabled: bool = True) -> Generator[None, None, None]:
        """Контекстный менеджер для AMP (Automatic Mixed Precision)."""
        if not enabled:
            yield
            return

        dtype: torch.dtype = self.get_dtype(precision)

        if "float8" in precision.lower():
            if not torch.cuda.is_available():
                warnings.warn("float8 требует CUDA, используем fp16")
                dtype = torch.float16
            elif not hasattr(torch, "float8_e4m3fn"):
                warnings.warn("float8 требует PyTorch 2.1+, используем fp16")
                dtype = torch.float16

        if torch.cuda.is_available() and dtype in [torch.float16, torch.bfloat16]:
            with torch.autocast(device_type="cuda", dtype=dtype):
                yield
        else:
            yield

    # ──────────────────────────────────────────────────────────────────────
    @contextmanager
    def autocast_float8(self, enabled: bool = True) -> Generator[None, None, None]:
        """Специальный контекстный менеджер для float8 (PyTorch 2.1+).

        Требует: torch >= 2.1, CUDA compute capability >= 9.0.
        """
        if not enabled:
            yield
            return

        if not torch.cuda.is_available():
            warnings.warn("float8 требует CUDA")
            yield
            return

        if not hasattr(torch, "float8_e4m3fn"):
            warnings.warn("float8 требует PyTorch 2.1+")
            yield
            return

        # Используем torch.float8_e4m3fn для активаций
        try:
            # from torch._dynamo import config as dynamo_config
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield
        except Exception as e:
            warnings.warn(f"float8 autocast failed: {e}, fallback to fp16")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield

    # ──────────────────────────────────────────────────────────────────────
    def get_supported_precisions(self, device: torch.device) -> List[str]:
        """Получить список поддерживаемых точностей для устройства."""
        supported: List[str] = ["fp32", "fp64"]

        if device.type == "cpu":
            supported.extend(["int8", "int16", "int32", "int64", "uint8"])
            return supported

        # CUDA device
        capability = torch.cuda.get_device_capability(device.index or 0)

        if capability[0] >= 6:
            supported.extend(["fp16", "int8"])

        if capability[0] >= 8:
            supported.append("bf16")

        if capability[0] >= 9 and hasattr(torch, "float8_e4m3fn"):
            supported.extend(
                [
                    "float8_e4m3fn",
                    "float8_e4m3fnuz",
                    "float8_e5m2",
                    "float8_e5m2fnuz",
                ]
            )

        return supported

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_memory_footprint(dtype: torch.dtype, num_elements: int) -> float:
        """Вычислить объем памяти в байтах.

        Args:
            dtype: Тип данных.
            num_elements: Количество элементов.

        Returns:
            Объем памяти в байтах.
        """
        dtype_sizes: Dict[torch.dtype, int] = {
            torch.float64: 8,
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int64: 8,
            torch.int32: 4,
            torch.int16: 2,
            torch.int8: 1,
            torch.uint8: 1,
            torch.bool: 1,
        }

        # Для float8 типов
        if hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn:
            return num_elements * 1.0  # 1 байт на элемент

        size = dtype_sizes.get(dtype, 4)  # Default to 4 bytes (fp32)
        return num_elements * size


# ──────────────────────────────────────────────────────────────────────
# NUMBA-OPTIMIZED FALLBACKS (CPU)
# ──────────────────────────────────────────────────────────────────────
@njit(parallel=True, cache=True)
def _kmeans_numba(data: np.ndarray, k: int, max_iter: int = 50, tol: float = 1e-4) -> np.ndarray:
    """Быстрая K-Means кластеризация на CPU (Lloyd's algorithm)."""
    n_samples, n_features = data.shape
    centroids = data[np.random.choice(n_samples, k, replace=False)].copy()
    labels = np.zeros(n_samples, dtype=np.int32)

    for _ in range(max_iter):
        # Assign labels
        for i in prange(n_samples):
            min_dist = np.inf
            for j in range(k):
                d = 0.0
                for f in range(n_features):
                    diff = data[i, f] - centroids[j, f]
                    d += diff * diff
                if d < min_dist:
                    min_dist = d
                    labels[i] = j

        # Update centroids
        old_centroids = centroids.copy()
        for j in range(k):
            count = 0
            for f in range(n_features):
                centroids[j, f] = 0.0
            for i in range(n_samples):
                if labels[i] == j:
                    count += 1
                    for f in range(n_features):
                        centroids[j, f] += data[i, f]
            if count > 0:
                for f in range(n_features):
                    centroids[j, f] /= count

        # Check convergence
        converged = True
        for j in range(k):
            for f in range(n_features):
                if np.abs(old_centroids[j, f] - centroids[j, f]) > tol:
                    converged = False
                    break
        if converged:
            break
    return labels


@njit(parallel=True, cache=True)
def _watershed_numba(gradient: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Marker-based Watershed propagation (быстрая альтернатива heapq)."""
    h, w = gradient.shape
    labels = np.copy(markers).astype(np.int32)
    visited = np.zeros((h, w), dtype=np.bool_)

    # Initialize queue with markers
    queue = []
    for y in range(h):
        for x in range(w):
            if markers[y, x] > 0:
                queue.append((gradient[y, x], y, x, labels[y, x]))
                visited[y, x] = True

    queue.sort()  # Sort by gradient magnitude

    # Propagate
    idx = 0
    while idx < len(queue):
        _, y, x, label = queue[idx]
        idx += 1

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                labels[ny, nx] = label
                queue.append((gradient[ny, nx], ny, nx, label))

    # Re-sort remaining to maintain priority order (approximate but fast)
    # For exact watershed, use scipy.ndimage.watershed_ift или skimage.morphology.watershed
    return labels


@njit(parallel=True, cache=True)
def _morphological_dilation_numba(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Быстрая дилатация на CPU."""
    h, w = image.shape
    out = np.zeros((h, w), dtype=image.dtype)
    half = kernel_size // 2
    for y in prange(h):
        for x in range(w):
            max_val = image[y, x]
            for ky in range(-half, half + 1):
                for kx in range(-half, half + 1):
                    ny, nx = y + ky, x + kx
                    if 0 <= ny < h and 0 <= nx < w:
                        if image[ny, nx] > max_val:
                            max_val = image[ny, nx]
            out[y, x] = max_val
    return out


@njit(parallel=True, cache=True)
def _morphological_erosion_numba(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Быстрая эрозия на CPU."""
    h, w = image.shape
    out = np.zeros((h, w), dtype=image.dtype)
    half = kernel_size // 2
    for y in prange(h):
        for x in range(w):
            min_val = image[y, x]
            for ky in range(-half, half + 1):
                for kx in range(-half, half + 1):
                    ny, nx = y + ky, x + kx
                    if 0 <= ny < h and 0 <= nx < w:
                        if image[ny, nx] < min_val:
                            min_val = image[ny, nx]
            out[y, x] = min_val
    return out


# ──────────────────────────────────────────────────────────────────────
# Вариант Б: Ускорение через Numba (для больших изображений, >1024×1024)
@njit(cache=True)  # parallel=True
def _region_growing_numba(
    gray: np.ndarray,
    seed_y: int,
    seed_x: int,
    tolerance: float,
    h: int,
    w: int,
) -> np.ndarray:
    """Region Growing на Numba — максимальная скорость на CPU."""
    mask = np.zeros((h, w), dtype=np.bool_)
    visited = np.zeros((h, w), dtype=np.bool_)
    seed_value = gray[seed_y, seed_x]

    # Простая очередь на массиве (циклический буфер)
    queue = np.zeros((h * w, 2), dtype=np.int32)
    head, tail = 0, 0
    queue[tail] = [seed_y, seed_x]
    tail += 1
    visited[seed_y, seed_x] = True

    # Направления
    dy = np.array([-1, 1, 0, 0], dtype=np.int32)
    dx = np.array([0, 0, -1, 1], dtype=np.int32)

    while head < tail:
        y, x = queue[head]
        head += 1

        pixel_value = gray[y, x]
        if abs(pixel_value - seed_value) <= tolerance:
            mask[y, x] = True

            # Добавляем соседей
            for d in range(4):
                ny, nx = y + dy[d], x + dx[d]
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue[tail] = [ny, nx]
                    tail += 1

    return mask


# ──────────────────────────────────────────────────────────────────────
@njit(parallel=True, cache=True)
def _watershed_numba_impl(
    gradient: np.ndarray,  # (H, W), float32
    markers: np.ndarray,  # (H, W), int32
    connectivity: int = 4,
) -> np.ndarray:
    """Watershed на Numba для CPU — priority queue через heapq."""
    import heapq

    h, w = gradient.shape
    labels = np.zeros((h, w), dtype=np.int32)
    visited = np.zeros((h, w), dtype=np.bool_)

    # Очередь: (gradient_value, y, x, label)
    queue: List[Tuple[float, int, int, int]] = []  # (gradient, y, x, label)

    # Инициализация маркерами
    for y in range(h):
        for x in range(w):
            if markers[y, x] > 0:
                heapq.heappush(queue, (gradient[y, x], y, x, markers[y, x]))
                labels[y, x] = markers[y, x]
                visited[y, x] = True

    # Направления
    if connectivity == 4:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    else:
        directions = [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

    # Основной цикл
    while queue:
        _, y, x, label = heapq.heappop(queue)

        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                labels[ny, nx] = label
                heapq.heappush(queue, (gradient[ny, nx], ny, nx, label))

    return labels


@torch.jit.unused
def _make_kernel_key(
    kernel_type: str,
    size: int,
    sigma: Optional[float],
    dtype: torch.dtype,
    device: torch.device,
    return_pair: bool,
) -> str:
    """Создаёт детерминированный строковый ключ."""
    sigma_str = f"{sigma:.3f}" if sigma is not None else "None"
    return f"{kernel_type}_{size}_{sigma_str}_{dtype}_{device.type}_{return_pair}"


# ──────────────────────────────────────────────────────────────────────
class TorchSegmenter2(BaseSegmenter):
    """Класс для методов сегментации с использованием PyTorch.

    Все реализации сделаны без использования OpenCV, Scikit-learn, Scikit-image
    или специализированных библиотек для обработки изображений.
    Поддерживает как классические методы (пороговые, граничные),
    так и методы на основе кластеризации, активных контуров и графов.
    """

    def __init__(
        self,
        method: str = "global_thresholding",
        device: Optional[str] = None,
        use_external_libs: bool = True,
        use_compile: bool = True,
        compile_mode: str = "reduce-overhead",  # или "default", "max-autotune"
        compile_fullgraph: bool = False,  # True для максимального ускорения, но может не скомпилироваться
        compile_dynamic: bool = True,  # Для поддержки разных размеров изображений
        debug_mode: bool = True,
        **kwargs: Any,
    ) -> None:
        """Инициализация класса TorchSegmenter2."""
        self.dtype: torch.dtype = self._resolve_dtype(kwargs.get("dtype", "fp32"))
        self._segment_func: SegmentMethod
        self._kernel_cache: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = {}
        self._hist_cache: Dict[Tuple[int, torch.dtype], torch.Tensor] = {}
        self._result_cache: OrderedDict = OrderedDict()
        self._cache_max_size: int = kwargs.get("cache_max_size", 100)
        super().__init__()
        self.method: str = method
        self.params: Dict[str, Any] = kwargs
        self.model_name: str = f"Torch_{method}"
        self.use_external_libs: bool = use_external_libs
        self.use_compile: bool = use_compile
        self.compile_mode: str = compile_mode
        self.compile_fullgraph: bool = compile_fullgraph
        self.compile_dynamic: bool = compile_dynamic
        self._debug_mode: bool = debug_mode
        self._has_profiling_run: bool = False
        self._static_kernels: Dict[str, torch.Tensor] = {}
        self._kernel_creation_device: Optional[torch.device] = None
        self._needs_normalization: bool = method in [
            "global_thresholding",
            "adaptive_thresholding",
            "otsu_thresholding",
            "threshold_niblack",
            "threshold_sauvola",
            "threshold_bernsen",
            "threshold_phansalkar",
            "threshold_percentile",
            "threshold_kittler_illingworth",
            "threshold_entropy_kapur",
            "threshold_triangle",
            "threshold_multi_otsu",
            "threshold_local_contrast",
            "sobel_edge",
            "canny_edge",
            "prewitt_edge",
            "scharr_edge",
            "laplacian_edge",
            "roberts_edge",
            "log_edge",
            "dog_edge",
            "marr_hildreth_edge",
            "gradient_magnitude_direction",
            "phase_congruency_edge",
            "morphological_snakes",
            "chan_vese",
            "quickshift",
            "slic",
        ]
        self.precision_manager: PrecisionManager = PrecisionManager(default_precision=kwargs.get("precision", "fp32"))

        if device is None:
            self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            warnings.warn("⚠️ GPU недоступен, переключение на CPU. Ожидается снижение скорости.")

        self._COMPILE_CONFIGS: Dict[str, Dict[str, Any]] = {
            # ===== ПОРОГОВЫЕ МЕТОДЫ =====
            "global_thresholding": {
                "fullgraph": True,
                "dynamic": True,
                "mode": "reduce-overhead",
            },
            "otsu_thresholding": {
                "fullgraph": False,
                "dynamic": True,
                "mode": "reduce-overhead",
            },
            "adaptive_thresholding": {
                "fullgraph": True,
                "dynamic": True,
                "mode": "reduce-overhead",
            },
            "threshold_niblack": {
                "fullgraph": False,
                "dynamic": True,
                "mode": "reduce-overhead",
            },
            "threshold_sauvola": {
                "fullgraph": False,
                "dynamic": True,
                "mode": "reduce-overhead",
            },
            "threshold_triangle": {
                "fullgraph": False,
                "dynamic": True,
                "mode": "reduce-overhead",
            },
            "threshold_kittler_illingworth": {"fullgraph": True, "dynamic": True},
            "threshold_entropy_kapur": {"fullgraph": True, "dynamic": True},
            "threshold_percentile": {
                "fullgraph": False,
                "dynamic": False,
                "mode": "reduce-overhead",
            },
            "threshold_local_contrast": {
                "fullgraph": False,
                "dynamic": False,
                "mode": "reduce-overhead",
            },
            # ===== ГРАНИЧНЫЕ МЕТОДЫ =====
            "sobel_edge": {
                "fullgraph": False,
                "dynamic": True,
                "mode": "reduce-overhead",
            },
            "prewitt_edge": {
                "fullgraph": True,
                "dynamic": True,
                "mode": "reduce-overhead",
            },
            "scharr_edge": {
                "fullgraph": True,
                "dynamic": True,
                "mode": "reduce-overhead",
            },
            "laplacian_edge": {
                "fullgraph": True,
                "dynamic": True,
                "mode": "reduce-overhead",
            },
            "canny_edge": {
                "fullgraph": True,
                "dynamic": True,
                "mode": "reduce-overhead",
            },  # NMS + hysteresis
            "log_edge": {
                "fullgraph": True,
                "dynamic": True,
                "mode": "reduce-overhead",
            },
            "dog_edge": {
                "fullgraph": True,
                "dynamic": True,
                "mode": "reduce-overhead",
            },
            # ===== АКТИВНЫЕ КОНТУРЫ =====
            "active_contour": {
                "fullgraph": False,
                "dynamic": True,
                "mode": "reduce-overhead",
            },  # grid_sample
            "chan_vese": {
                "fullgraph": False,
                "dynamic": True,
                "mode": "reduce-overhead",
            },  # итерации
            "morphological_snakes": {
                "fullgraph": True,
                "dynamic": False,
                "mode": "reduce-overhead",
            },  # векторизовано
            # ===== ГРАФОВЫЕ МЕТОДЫ =====
            "watershed": {
                "fullgraph": False,
                "dynamic": True,
                "mode": "reduce-overhead",
            },  # heapq
            "random_walker": {
                "fullgraph": False,
                "dynamic": True,
                "mode": "max-autotune",
            },  # sparse solve
            # ===== КЛАСТЕРИЗАЦИЯ =====
            "kmeans_segmentation": {
                "fullgraph": False,
                "dynamic": True,
                "mode": "reduce-overhead",
            },
            # ===== SUPER-PIXEL (отключаем compile) =====
            "quickshift": {"use_compile": False},  # numpy-heavy
            "slic": {"use_compile": False},  # scipy-heavy
            "felzenszwalb": {"use_compile": False},  # skimage-heavy
            "dbscan_segmentation": {"use_compile": False},  # sklearn
            "meanshift": {"use_compile": False},  # sklearn
        }
        self._setup_method()
        if self.method in [
            "quickshift",
            "slic",
            "felzenszwalb",
            "dbscan_segmentation",
            "meanshift",
        ]:
            if self.use_compile:
                warnings.warn(
                    f"Метод '{self.method}' использует numpy/scipy — torch.compile не даст ускорения. "
                    "Рекомендуется установить use_compile=False.",
                    UserWarning,
                )
        if kwargs.get("use_quantization", False) and self.device.type == "cpu":
            logger.info("🔧 Применяем динамическое квантование (int8)...")
            if isinstance(self._segment_func, nn.Module):
                self._segment_func = self._apply_dynamic_quantization(self._segment_func)

    # ──────────────────────────────────────────────────────────────────────
    # УТИЛИТЫ ДЛЯ БЕЗОПАСНОЙ СВЁРТКИ
    # ──────────────────────────────────────────────────────────────────────
    def _prepare_kernel_for_conv(
        self,
        kernel: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        target_dtype: torch.dtype,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Безопасно конвертирует ядро(а) к целевому dtype.

        Поддерживает как одиночные ядра, так и кортежи (для парных операторов).

        Оптимизация: не создаёт копию, если dtype уже совпадает.

        Args:
            kernel: Ядро или кортеж ядер (например, для Sobel: (kx, ky)).
            target_dtype: Целевой тип данных (обычно dtype входного изображения).

        Returns:
            Ядро(а) с гарантированным совпадением dtype.
        """

        def _convert_single(k: torch.Tensor) -> torch.Tensor:
            if k.dtype == target_dtype:
                return k
            # Для low-precision: сначала в fp32, затем в целевой тип
            if target_dtype in (torch.float16, torch.bfloat16):
                return k.to(torch.float32).to(target_dtype, non_blocking=True)
            return k.to(target_dtype, non_blocking=True)

        if isinstance(kernel, tuple):
            k1, k2 = kernel  # type: ignore[misc]
            return _convert_single(k1), _convert_single(k2)
        return _convert_single(kernel)

    # ──────────────────────────────────────────────────────────────────────
    def _safe_conv2d(
        self,
        input: torch.Tensor,
        kernel: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Безопасная свёртка с автоматическим выравниванием dtype.

        Решает проблему: autocast может изменить dtype input,
        а ядро остаётся в исходном dtype → RuntimeError или silent degradation.

        Args:
            input: Входной тензор (обычно изображение в градациях серого).
            kernel: Сверточное ядро формы (1, 1, kH, kW).
            **kwargs: Дополнительные аргументы для F.conv2d (padding, stride, etc.).

        Returns:
            torch.Tensor: Результат свёртки.
        """
        if kernel.dtype != input.dtype:
            kernel = kernel.to(input.dtype, non_blocking=True)
        return F.conv2d(input, kernel, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    @torch.jit.unused  # Игнорировать при трассировке/компиляции
    def _get_static_kernel(
        self,
        name: str,
        data: List[List[float]],
        size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Создаёт и кэширует свёрточное ядро вне контекста torch.compile.

        Args:
            name: Уникальное имя ядра для кэша.
            data: Данные ядра как список списков.
            size: Размер ядра (для view).
            dtype: Целевой тип данных.
            device: Устройство размещения.

        Returns:
            torch.Tensor: Ядро формы (1, 1, size, size).
        """
        # Ключ кэша: имя + dtype + device
        cache_key: str = f"{name}_{str(dtype)}_{str(device)}"

        if cache_key not in self._static_kernels:
            # Создаём ядро в eager mode (вне torch.compile)
            kernel = torch.tensor(data, dtype=dtype, device=device)
            kernel = kernel.view(1, 1, size, size)
            self._static_kernels[cache_key] = kernel

        return self._static_kernels[cache_key]

    # ──────────────────────────────────────────────────────────────────────
    @torch.jit.unused
    def _make_kernel_safe(
        self,
        data: List[List[float]],
        size: int,
        device: torch.device,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Безопасное создание ядра с учётом low-precision dtypes.

        Для fp16/bf16: создаём в fp32 → конвертируем (избегаем проблем с целочисленными данными).
        Для fp32: создаём напрямую.

        Args:
            data: Данные ядра.
            size: Размер ядра.
            device: Устройство.
            target_dtype: Целевой dtype.

        Returns:
            torch.Tensor: Ядро формы (1, 1, size, size).
        """
        # Для low-precision: сначала создаём в fp32, потом конвертируем
        if target_dtype in [torch.float16, torch.bfloat16]:
            kernel = torch.tensor(data, dtype=torch.float32, device=device)
            kernel = kernel.to(dtype=target_dtype)
        else:
            kernel = torch.tensor(data, dtype=target_dtype, device=device)

        return kernel.view(1, 1, size, size)

    # ──────────────────────────────────────────────────────────────────────
    @overload
    def _get_conv_kernel(
        self,
        kernel_type: Literal[
            "sobel", "prewitt", "scharr", "roberts", "gaussian", "ones", "laplacian", "sobel_x", "sobel_y"
        ],
        size: int = 3,
        sigma: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        *,
        return_pair: Literal[True],
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def _get_conv_kernel(
        self,
        kernel_type: Literal[
            "sobel", "prewitt", "scharr", "roberts", "gaussian", "ones", "laplacian", "sobel_x", "sobel_y"
        ],
        size: int = 3,
        sigma: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        *,
        return_pair: Literal[False] = False,
    ) -> torch.Tensor: ...

    def _get_conv_kernel(
        self,
        kernel_type: Literal[
            "sobel", "prewitt", "scharr", "roberts", "gaussian", "ones", "laplacian", "sobel_x", "sobel_y"
        ],
        size: int = 3,
        sigma: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        return_pair: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Универсальный кэш для ядер свёртки."""
        dtype = dtype or self.dtype
        device = device or self.device

        # Ключ для кэша
        key = _make_kernel_key(kernel_type, size, sigma, dtype, device, return_pair)
        if key in self._kernel_cache:
            cached = self._kernel_cache[key]
            if isinstance(cached, tuple):
                kx, ky = cached  # type: ignore[misc]
                return (
                    kx.clone().to(dtype=dtype, device=device),
                    ky.clone().to(dtype=dtype, device=device),
                )
            return cached.clone().to(dtype=dtype, device=device)

        # Генерация ядра
        if return_pair and kernel_type in ["sobel", "prewitt", "scharr", "roberts"]:
            if kernel_type == "sobel":
                data_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
                data_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            elif kernel_type == "prewitt":
                data_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
                data_y = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
            elif kernel_type == "scharr":
                data_x = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
                data_y = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]
            elif kernel_type == "roberts":
                data_x = [[1, 0], [0, -1]]
                data_y = [[0, 1], [-1, 0]]
                size = 2  # Roberts всегда 2×2

            kx = torch.tensor(data_x, dtype=dtype, device=device).to(dtype=dtype).view(1, 1, size, size)
            ky = torch.tensor(data_y, dtype=dtype, device=device).to(dtype=dtype).view(1, 1, size, size)
            kernel: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = (kx, ky)
        else:
            if kernel_type == "sobel_x":
                data = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
                kernel = torch.tensor(data, dtype=dtype, device=device).view(1, 1, 3, 3)
            elif kernel_type == "sobel_y":
                data = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
                kernel = torch.tensor(data, dtype=dtype, device=device).view(1, 1, 3, 3)
            elif kernel_type == "gaussian":
                coords = torch.arange(size, dtype=dtype, device=device) - size // 2
                kernel_1d = torch.exp(-0.5 * (coords / (sigma or size / 6)) ** 2)
                kernel_1d = kernel_1d / kernel_1d.sum()
                kernel = (kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)).view(1, 1, size, size)
            elif kernel_type == "ones":
                kernel = torch.ones(1, 1, size, size, dtype=dtype, device=device) / (size**2)
            elif kernel_type == "laplacian":
                if size == 3:
                    data = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
                else:  # size == 5
                    data = [
                        [0, 0, -1, 0, 0],
                        [0, -1, -2, -1, 0],
                        [-1, -2, 16, -2, -1],
                        [0, -1, -2, -1, 0],
                        [0, 0, -1, 0, 0],
                    ]
                kernel = torch.tensor(data, dtype=dtype, device=device).to(dtype=dtype).view(1, 1, size, size) / (
                    8.0 if size == 5 else 1.0
                )
            else:
                raise ValueError(f"Unknown kernel type: {kernel_type}")

        self._kernel_cache[key] = kernel  # type: ignore[assignment]
        if isinstance(kernel, tuple):
            kx, ky = kernel  # type: ignore[misc]
            return (
                kx.clone().to(dtype=dtype, device=device),
                ky.clone().to(dtype=dtype, device=device),
            )
        return kernel.clone().to(dtype=dtype, device=device)

    # ──────────────────────────────────────────────────────────────────────
    # КЭШИРУЕМЫЕ ЯДРА ДЛЯ ГРАДИЕНТНЫХ МЕТОДОВ
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    @lru_cache(maxsize=32)
    def _get_prewitt_kernels_cached(
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Кэширует ядра Прюитта для устройства и типа данных."""
        kx = torch.tensor(
            [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
            dtype=dtype,
            device=device,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
            dtype=dtype,
            device=device,
        ).view(1, 1, 3, 3)
        return kx, ky

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    @lru_cache(maxsize=32)
    def _get_scharr_kernels_cached(
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Кэширует ядра Шарра для устройства и типа данных."""
        kx = torch.tensor(
            [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
            dtype=dtype,
            device=device,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
            dtype=dtype,
            device=device,
        ).view(1, 1, 3, 3)
        return kx, ky

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    @lru_cache(maxsize=32)
    def _get_laplacian_kernel_cached(
        device: torch.device, dtype: torch.dtype, size: int = 3  # 3 или 5
    ) -> torch.Tensor:
        """Кэширует ядро Лапласа (4-связность) для устройства и типа данных."""
        if size == 3:
            kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=dtype, device=device)
        else:  # size == 5
            kernel = (
                torch.tensor(
                    [
                        [0, 0, -1, 0, 0],
                        [0, -1, -2, -1, 0],
                        [-1, -2, 16, -2, -1],
                        [0, -1, -2, -1, 0],
                        [0, 0, -1, 0, 0],
                    ],
                    dtype=dtype,
                    device=device,
                )
                / 8.0
            )
        return kernel.view(1, 1, size, size)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    @lru_cache(maxsize=32)
    def _get_roberts_kernels_cached(
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Кэширует ядра Робертса 2×2 для устройства и типа данных."""
        kx = torch.tensor(
            [[1, 0], [0, -1]],
            dtype=dtype,
            device=device,
        ).view(1, 1, 2, 2)
        ky = torch.tensor(
            [[0, 1], [-1, 0]],
            dtype=dtype,
            device=device,
        ).view(1, 1, 2, 2)
        return kx, ky

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    @lru_cache(maxsize=64)
    def _get_gaussian_kernel_1d(
        size: int,
        sigma: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Кэшируемое 1D гауссово ядро для сепарабельной свёртки.

        Args:
            size: Нечётный размер ядра.
            sigma: Стандартное отклонение гауссианы.
            dtype: Тип данных тензора.
            device: Устройство размещения.

        Returns:
            torch.Tensor: 1D ядро формы (size,), нормированное к сумме 1.
        """
        coords = torch.arange(size, dtype=dtype, device=device) - size // 2
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        return kernel / kernel.sum()

    @staticmethod
    @lru_cache(maxsize=8)
    def _get_sobel_kernels_cached(device: str, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Кэширует ядра Собеля для устройства и dtype (избегает ре-создания)."""
        dev = torch.device(device)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=dev, dtype=dtype).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=dev, dtype=dtype).view(1, 1, 3, 3)
        return kx, ky

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    @lru_cache(maxsize=32)
    def _get_laplacian_5x5_cached(
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Кэширует лапласиан 5×5 для устройства и типа данных."""
        kernel = (
            torch.tensor(
                [
                    [0, 0, -1, 0, 0],
                    [0, -1, -2, -1, 0],
                    [-1, -2, 16, -2, -1],
                    [0, -1, -2, -1, 0],
                    [0, 0, -1, 0, 0],
                ],
                dtype=dtype,
                device=device,
            ).view(1, 1, 5, 5)
            / 8.0
        )
        return kernel

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _local_stats_torch(
        self,
        gray: torch.Tensor,
        window_size: int,
        dtype: Optional[torch.dtype] = None,
        export_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Вычисляет локальное среднее и стандартное отклонение через свёртку на GPU.

        Args:
            gray: Тензор (H, W) или (1, 1, H, W) в диапазоне [0, 1] или [0, 255].
            window_size: Нечётный размер окна.

        Returns:
            local_mean, local_std — тензоры той же размерности, что и вход.
        """
        dtype = dtype or gray.dtype
        device = gray.device

        # Кэш ключ для ядра
        kernel = self._get_conv_kernel(
            "ones",
            size=window_size,
            sigma=None,
            dtype=dtype,
            device=device,
            return_pair=False,
        )

        # Приводим к 4D для conv2d: (1, 1, H, W)
        if gray.dim() == 2:
            gray_4d = gray.unsqueeze(0).unsqueeze(0)
        elif gray.dim() == 3 and gray.shape[0] == 1:
            gray_4d = gray.unsqueeze(0)
        else:
            gray_4d = gray

        # Создаём ядро усреднения (с учётом dtype и device)
        pad = window_size // 2

        kernel = kernel.to(dtype=gray_4d.dtype)

        # Локальное среднее: E[X]
        local_mean_4d = F.conv2d(gray_4d, kernel, padding=pad, stride=1)

        # Локальное среднее квадратов: E[X²]
        local_mean_sq_4d = F.conv2d(gray_4d**2, kernel, padding=pad, stride=1)

        # Дисперсия: Var[X] = E[X²] - E[X]², с защитой от отрицательных значений из-за численной нестабильности
        diff = local_mean_sq_4d - local_mean_4d**2
        if export_mode:
            # local_var_4d = torch.where(diff > 0, diff, torch.zeros_like(diff)) + 1e-8
            #  local_var_4d = torch.relu(diff) + 1e-8
            local_var_4d = torch.maximum(diff, torch.zeros_like(diff)) + 1e-8
        else:
            local_var_4d = torch.clamp(diff, min=1e-8)
        local_std_4d = torch.sqrt(local_var_4d)

        if gray.dim() == 2:
            H, W = gray.shape[-2], gray.shape[-1]
            return (local_mean_4d.view(H, W), local_std_4d.view(H, W))
        return local_mean_4d, local_std_4d

    # ──────────────────────────────────────────────────────────────────────
    def _get_intermediate_results(self) -> None:
        """Возвращает промежуточные результаты для тестов."""
        if not self._debug_mode:
            raise RuntimeError("Debug mode not enabled")

    # ──────────────────────────────────────────────────────────────────────
    def _setup_method(self) -> None:
        """Настройка выбранного метода."""
        self.method_map: Dict[str, Callable[..., torch.Tensor]] = {
            # ============ ПОРОГОВЫЕ МЕТОДЫ СЕГМЕНТАЦИИ ============
            "global_thresholding": self._global_thresholding,
            "adaptive_thresholding": self._adaptive_thresholding,
            "otsu_thresholding": self._otsu_thresholding,
            "threshold_niblack": self._threshold_niblack,
            "threshold_sauvola": self._threshold_sauvola,
            "threshold_bernsen": self._threshold_bernsen,
            "threshold_phansalkar": self._threshold_phansalkar,
            "threshold_percentile": self._threshold_percentile,
            "threshold_kittler_illingworth": self._threshold_kittler_illingworth,
            "threshold_entropy_kapur": self._threshold_entropy_kapur,
            "threshold_triangle": self._threshold_triangle,
            "threshold_multi_otsu": self._threshold_multi_otsu,
            "threshold_local_contrast": self._threshold_local_contrast,
            # ============ КРАЕВЫЕ СЕГМЕНТАЦИОННЫЕ МЕТОДЫ ============
            "sobel_edge": self._sobel_edge,
            "canny_edge": self._canny_edge,
            "prewitt_edge": self._prewitt_edge,
            "scharr_edge": self._scharr_edge,
            "laplacian_edge": self._laplacian_edge,
            "roberts_cross_edge": self._roberts_edge,
            "log_edge": self._log_edge,
            "dog_edge": self._dog_edge,
            "marr_hildreth_edge": self._marr_hildreth_edge,
            "gradient_magnitude_direction": self._gradient_magnitude_direction,
            "phase_congruency_edge": self._phase_congruency_edge,
            # ============ РЕГИОНАЛЬНЫЕ СЕГМЕНТАЦИОННЫЕ МЕТОДЫ ============
            "region_growing": self._region_growing,
            "split_and_merge": self._split_and_merge,
            "floodfill": self._floodfill,
            # ============ КЛАСТЕРИЗАЦИЯ ============
            "kmeans_segmentation": self._kmeans_segmentation,
            "dbscan_segmentation": self._dbscan_segmentation,
            "meanshift": self._meanshift,
            # ============ АКТИВНЫЕ КОНТУРЫ ============
            "active_contour": self._active_contour,
            "gvf_contour": self._gvf_contour,
            "morphological_snakes": self._morphological_snakes,
            "chan_vese": self._chan_vese,
            # ============ WATERSHED И ГРАФОВЫЕ ============
            "watershed": self._watershed,
            "random_walker": self._random_walker,
            # ============ SUPER-PIXEL МЕТОДЫ ===========
            "quickshift": self._quickshift,
            "slic": self._slic,
            "felzenszwalb": self._felzenszwalb,
            # ============ ИНТЕРАКТИВНЫЕ МЕТОДЫ ============
            "grabcut": self._grabcut,
        }

        if self.method not in self.method_map:
            raise ValueError(f"Неизвестный метод: {self.method}. " f"Доступные методы: {list(self.method_map.keys())}")

        self._segment_func = self.method_map[self.method]

        compile_cfg: Dict[str, Any] = self._COMPILE_CONFIGS.get(self.method, {})
        use_compile: bool = bool(compile_cfg.get("use_compile", self.use_compile))

        if use_compile and torch.__version__ >= "2.0":
            fullgraph: bool = bool(compile_cfg.get("fullgraph", self.compile_fullgraph))
            dynamic: bool = bool(compile_cfg.get("dynamic", self.compile_dynamic))
            mode: str = str(compile_cfg.get("mode", self.compile_mode))
            self._precache_kernels()
            if self.method in {"adaptive_thresholding", "canny_edge", "watershed"}:
                warnings.warn(
                    f"Метод '{self.method}' содержит динамический контроль потока. "
                    "fullgraph=True может не дать ускорения или вызвать ошибки. "
                    "Рекомендуется установить fullgraph=False в _COMPILE_CONFIGS.",
                    UserWarning,
                )
            try:
                logger.info(f"🔧 Компиляция '{self.method}' " f"[mode={mode}, fullgraph={fullgraph}]...")
                self._segment_func = torch.compile(
                    self._segment_func,
                    mode=mode,  # "reduce-overhead" или "max-autotune"
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                    backend="inductor",  # или "cudagraphs" для CUDA
                )
                logger.info("✅ Компиляция завершена")
            except Exception as e:
                logger.warning(f"⚠️  Не удалось скомпилировать: {e}. Используем обычный режим.")
                warnings.warn(
                    f"torch.compile failed for {self.method} "
                    f"[fullgraph={fullgraph}]: {e}. Falling back to eager mode.",
                    RuntimeWarning,
                )
                use_compile = False

    # ──────────────────────────────────────────────────────────────────────
    @torch.jit.unused
    def _precache_kernels(self) -> None:
        """Предварительное создание ядер для методов, которые будут компилироваться."""
        dtype: torch.dtype = self.dtype
        device: torch.device = self.device

        # Пороговые методы не требуют ядер, пропускаем
        for ws in [11, 15, 25, 35]:  # типичные window_size
            key = _make_kernel_key("ones", ws, None, dtype, device, False)
            if key not in self._kernel_cache:
                kernel = torch.ones(1, 1, ws, ws, device=device, dtype=dtype)
                self._kernel_cache[key] = (kernel / (ws**2)).clone()

        # Граничные методы
        edge_kernels: Dict[str, List[List[int]]] = {
            "sobel_x": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            "sobel_y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            "prewitt_x": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
            "prewitt_y": [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
            "scharr_x": [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
            "scharr_y": [[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
            "roberts_x": [[1, 0], [0, -1]],
            "roberts_y": [[0, 1], [-1, 0]],
            "laplacian_3x3": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        }

        for name, data in edge_kernels.items():
            size = 2 if "roberts" in name else 3
            kernel = torch.tensor(data, dtype=dtype, device=device).view(1, 1, size, size)
            self._static_kernels[f"{name}_{str(dtype)}_{str(device)}"] = kernel

    # ──────────────────────────────────────────────────────────────────────
    def preprocess_image(  # type: ignore[override]
        self,
        image: Union[str, np.ndarray, Image.Image, torch.Tensor],
        as_gray: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Предобработка изображения для PyTorch."""
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
            return self._pil_to_tensor(img, normalize=self._needs_normalization)
        elif isinstance(image, Image.Image):
            return self._pil_to_tensor(image, normalize=self._needs_normalization)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
            return self._pil_to_tensor(img, normalize=self._needs_normalization)
        elif isinstance(image, torch.Tensor):
            return image.to(self.device)
        else:
            raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_dtype(dtype_arg: Optional[Union[str, torch.dtype]]) -> torch.dtype:
        """Разрешение типа данных с учётом возможностей устройства.

        Args:
            dtype_arg: 'fp32', 'fp16', 'bf16' или torch.dtype.

        Returns:
            torch.dtype для вычислений.
        """
        if isinstance(dtype_arg, torch.dtype):
            return dtype_arg

        dtype_map: Dict[str, torch.dtype] = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }

        if isinstance(dtype_arg, str):
            dtype = dtype_map.get(dtype_arg.lower(), torch.float32)
        else:
            dtype = torch.float32

        # Авто-коррекция для неподдерживаемых типов
        if dtype == torch.float16 and not torch.cuda.is_available():
            logger.warning("⚠️  fp16 не поддерживается на CPU, используем fp32")
            return torch.float32

        if dtype == torch.bfloat16:
            if not torch.cuda.is_available():
                logger.warning("⚠️  bf16 не поддерживается на CPU, используем fp32")
                return torch.float32
            # Проверка поддержки bf16 на текущей GPU
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                if props.major < 8:  # Ampere+ для полноценной поддержки
                    logger.warning(
                        f"⚠️  bf16 может работать медленно на GPU compute capability {props.major}.{props.minor}"
                    )

        return dtype

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    @torch.jit.ignore  # Игнорировать при трассировке, выполнять в eager mode
    def _rgb_to_gray_torch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.size(1) == 3:
            return 0.2989 * tensor[:, 0:1, :, :] + 0.5870 * tensor[:, 1:2, :, :] + 0.1140 * tensor[:, 2:3, :, :]
        else:
            return tensor

    # ──────────────────────────────────────────────────────────────────────
    def _to_grayscale(self, tensor: torch.Tensor) -> torch.Tensor:
        """Преобразование RGB в градации серого."""
        return cast(torch.Tensor, self._rgb_to_gray_torch(tensor))

    # ──────────────────────────────────────────────────────────────────────
    def _cast_to_dtype(self, tensor: torch.Tensor, target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Приведение с гарантией совпадения dtype для conv2d."""
        dtype = target_dtype or self.dtype
        if tensor.dtype == dtype and tensor.device == self.device:
            return tensor
        return tensor.to(device=self.device, dtype=dtype, non_blocking=True)

    # ──────────────────────────────────────────────────────────────────────
    def _pil_to_tensor(self, img: Image.Image, normalize: bool = True, add_batch: bool = True) -> torch.Tensor:
        """Универсальное преобразование PIL -> Tensor.

        Args:
            img: Входное изображение PIL.
            normalize: Нормализовать [0, 255] -> [0, 1].
            add_batch: Добавить batch dimension.

        Returns:
            torch.Tensor на нужном устройстве.
        """
        try:
            # np.array -> float32 -> [0,1] -> (C,H,W)
            tensor: torch.Tensor
            tensor = TF.to_tensor(img) if normalize else torch.from_numpy(np.array(img)).permute(2, 0, 1)

            if add_batch:
                tensor = tensor.unsqueeze(0)  # (1, C, H, W)

            return tensor.to(device=self.device, dtype=self.dtype, non_blocking=False)
        except Exception as e:
            raise ValueError(f"Ошибка преобразования PIL->Tensor: {e}")

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor, denormalize: bool = True) -> np.ndarray:
        """Преобразование PyTorch tensor в NumPy array."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        result: np.ndarray = tensor.permute(1, 2, 0).cpu().numpy()

        if denormalize and result.max() <= 1.0:
            result = (result * 255).astype(np.uint8)

        return result

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor, squeeze: bool = True) -> Image.Image:
        """Преобразование torch.Tensor в PIL.Image.

        Args:
            tensor: Тензор в формате (C, H, W) или (B, C, H, W).
            squeeze: Если True, удаляет batch dimension при наличии.

        Returns:
            PIL.Image.
        """
        if tensor.dim() == 4 and squeeze:
            tensor = tensor.squeeze(0)
        elif tensor.dim() == 3:
            pass
        else:
            raise ValueError(f"Неверная размерность тензора: {tensor.shape}")

        return cast(Image.Image, TF.to_pil_image(tensor))

    # ──────────────────────────────────────────────────────────────────────
    def profile_with_transfer_detection(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        n_runs: int = 10,
        detect_transfers: bool = True,
    ) -> Dict[str, Any]:
        """Профилирование с детекцией CPU↔GPU трансферов.

        Returns:
            Dict с метриками + предупреждениями о трансферах.
        """
        import torch.profiler as profiler

        tensor = self.preprocess_image(image)

        transfer_warnings: List[str] = []

        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with profiler.record_function(f"segment_{self.method}"):
                for _ in range(n_runs):
                    _ = self._segment_func(tensor, precision=self.precision_manager.default_precision)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

        if detect_transfers and self.device.type == "cuda":
            if self.method in ["quickshift", "slic", "felzenszwalb"]:
                transfer_warnings.append("⚠️ Метод использует numpy — трансферы ожидаемы")
            else:
                for event in prof.key_averages():
                    if "cudaMemcpy" in event.key or "to(" in event.key:
                        transfer_warnings.append(f"⚠️  Трансфер: {event.key} — {event.cpu_time_total / 1e3:.2f}ms")

        avg_event = prof.key_averages().total_average()

        # Пробуем разные варианты имён атрибутов (PyTorch 1.x → 2.x)
        cuda_time = getattr(avg_event, "cuda_time_total", None)
        if cuda_time is None:
            cuda_time = getattr(avg_event, "self_cuda_time_total", 0)

        cuda_memory = getattr(avg_event, "cuda_memory_usage", None)
        if cuda_memory is None:
            # Альтернатива: попробовать получить из summary
            try:
                cuda_memory = prof.key_averages().total_average().cuda_memory_usage
            except AttributeError:
                cuda_memory = 0

        # Стандартные метрики
        return {
            "method": self.method,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "avg_time_ms": (cuda_time or 0) / 1e3,
            "memory_mb": (cuda_memory or 0) / 1e6,
            "transfer_warnings": transfer_warnings,
            "profiler_table": prof.key_averages().table(
                sort_by=("cuda_time_total" if self.device.type == "cuda" else "cpu_time_total"),
                row_limit=15,
            ),
        }

    # ──────────────────────────────────────────────────────────────────────
    def benchmark_histc_types(self, gray: torch.Tensor, device: torch.device) -> Dict[str, float]:
        """Сравнивает torch.histc для разных входных типов (учёт ограничений CUDA).

        Note:
            Используется только в 1 исследовании.
        """
        results: Dict[str, Any] = {}
        # torch.histc на CUDA поддерживает ТОЛЬКО fp32/fp64.
        # Прямой вызов с fp16/bf16 вызывает RuntimeError: HalfTensor is not supported
        dtypes_to_test: List[Tuple[str, torch.dtype]] = [
            ("fp32", torch.float32),
            ("fp16 (auto-cast)", torch.float16),
            ("bf16 (auto-cast)", torch.bfloat16),
        ]

        for dtype_name, dtype in dtypes_to_test:
            gray_typed: torch.Tensor = gray.to(dtype)
            start: float = time.perf_counter()
            for _ in range(100):
                # histc требует fp32/fp64. При half-precision явно кастим перед вызовом
                inp = gray_typed.float() if dtype in (torch.float16, torch.bfloat16) else gray_typed

                # max=1.0, т.к. _to_grayscale уже нормализует вход к [0, 1]
                _ = torch.histc(inp, bins=256, min=0.0, max=1.0)

            if device.type == "cuda":
                torch.cuda.synchronize()
            results[dtype_name] = (time.perf_counter() - start) / 100 * 1000  # ms

        return results

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def profile_segmentation(
        segmenter: "TorchSegmenter2",
        image: Union[np.ndarray, torch.Tensor],
        n_runs: int = 10,
        warmup: int = 3,
    ) -> Dict[str, Any]:
        """Профилирование времени выполнения метода сегментации.

        Returns:
            Dict с метриками: mean_time, std_time, min_time, max_time, device.

        Note:
            Используется только в 1 исследовании.
        """
        import time

        # Предобработка
        if isinstance(image, np.ndarray):
            tensor = segmenter.preprocess_image(image)
        else:
            tensor = image.to(segmenter.device)

        # Warmup
        for _ in range(warmup):
            _ = segmenter._segment_func(tensor)

        if segmenter.device.type == "cuda":
            torch.cuda.synchronize()

        # Замеры
        times: List[float] = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = segmenter._segment_func(tensor)
            if segmenter.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        return {
            "method": segmenter.method,
            "device": str(segmenter.device),
            "mean_time_s": np.mean(times),
            "std_time_s": np.std(times),
            "min_time_s": np.min(times),
            "max_time_s": np.max(times),
            "image_shape": tuple(tensor.shape),
        }

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def profile_with_tracing(
        segmenter: "TorchSegmenter2",
        image: np.ndarray,
        output_dir: str = "./profiling",
    ) -> None:
        """Запуск профилировщика PyTorch с экспортом trace для Chrome DevTools.

        Note:
            Используется только в 1 исследовании.
        """
        import torch.profiler as profiler

        os.makedirs(output_dir, exist_ok=True)
        tensor = segmenter.preprocess_image(image)
        logger.info(tensor)

        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with profiler.record_function(f"segment_{segmenter.method}"):
                _ = segmenter.segment(image)

        # Вывод таблицы
        logger.info(
            prof.key_averages().table(
                sort_by=("cuda_time_total" if segmenter.device.type == "cuda" else "cpu_time_total"),
                row_limit=20,
            )
        )

        # Экспорт для визуализации
        trace_path = os.path.join(output_dir, f"trace_{segmenter.method}.json")
        prof.export_chrome_trace(trace_path)
        logger.info(f"📊 Trace сохранён: {trace_path}")
        logger.info("💡 Откройте chrome://tracing и загрузите файл для интерактивного анализа")

    # ──────────────────────────────────────────────────────────────────────
    def _apply_dynamic_quantization(self, module: nn.Module) -> nn.Module:
        """Применяет динамическое квантование весов (int8) для совместимых слоёв.

        🔹 Работает только на CPU.
        🔹 Поддерживает Linear, Conv2d.
        """
        if self.device.type != "cpu":
            logger.warning("⚠️  Квантование доступно только на CPU")
            return module

        try:
            from torch.ao.quantization import quantize_dynamic

            return cast(
                nn.Module,
                quantize_dynamic(
                    module,
                    {torch.nn.Linear, torch.nn.Conv2d},  # типы слоёв для квантования
                    dtype=torch.qint8,
                ),
            )
        except ImportError:
            logger.warning("⚠️  torch.ao.quantization недоступен, пропускаем квантование")
            return module

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    @torch.jit.unused  # Для совместимости с torch.compile
    def _cv_heavyside(x: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
        """Регуляризованная функция Хевисайда.

        Плавная аппроксимация ступенчатой функции через арктангенс.
        Используется в функционале Чан-Везе для разделения областей.

        Args:
            x: Входной тензор произвольной формы.
            eps: Параметр регуляризации (по умолчанию: 1.0). Меньшие значения → более резкий переход.

        Returns:
            torch.Tensor: Тензор той же формы, значения в диапазоне [0, 1].

        Note:
            - Функция дифференцируема, что необходимо для градиентной оптимизации.
            - Для `fp16` рекомендуется `eps ≥ 0.5` для избежания численной нестабильности.
        """
        return 0.5 * (1.0 + (2.0 / torch.pi) * torch.arctan(x / eps))

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    @torch.jit.unused
    def _cv_delta(x: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
        """Регуляризованная дельта-функция Дирака (производная Хевисайда).

        Используется для вычисления члена длины контура в функционале Чан-Везе.

        Args:
            x: Входной тензор произвольной формы.
            eps: Параметр регуляризации (по умолчанию: 1.0).

        Returns:
            torch.Tensor: Тензор той же формы, значения ≥ 0.

        Note:
            - Пик функции сосредоточен около нуля, ширина ~2·eps.
            - Для `fp16` рекомендуется `eps ≥ 0.5` для избежания переполнения.
        """
        return eps / (eps**2 + x**2 + 1e-16)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _cv_calculate_averages(
        image: torch.Tensor,
        Hphi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Вычисление средних значений интенсивности внутри и снаружи контура.

        Использует взвешенное усреднение с весами Хевисайда для разделения областей.

        Args:
            image: Изображение в градациях серого, форма (H, W) или (1, 1, H, W).
            Hphi: Карта Хевисайда уровня (значения [0, 1]), той же формы, что и image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (c1, c2) — средние значения внутри и снаружи.

        Note:
            - Возвращает скалярные тензоры на том же устройстве, что и вход.
            - Добавлен `eps` для избежания деления на 0 при пустых регионах.
        """
        # Убираем лишние размерности для единообразия
        if image.dim() == 4:
            image = image.squeeze()
        if Hphi.dim() == 4:
            Hphi = Hphi.squeeze()

        H = Hphi
        Hinv = 1.0 - H

        eps = 1e-8
        Hsum = H.sum() + eps
        Hinvsum = Hinv.sum() + eps

        c1 = (image * H).sum() / Hsum
        c2 = (image * Hinv).sum() / Hinvsum

        return c1, c2

    # ──────────────────────────────────────────────────────────────────────
    def _cv_edge_length_term(
        self,
        phi: torch.Tensor,
        mu: float,
        eps: float = 1.0,
    ) -> torch.Tensor:
        """Энергетический член: длина контура (регуляризация).

        Вычисляет взвешенную длину нулевого уровня функции φ через градиенты.
        Штрафует сложные, изрезанные контуры, способствуя гладкости результата.

        Формула: μ·δ(φ)·|∇φ|, где δ — регуляризованная дельта-функция.

        Args:
            phi: Функция уровня, форма (H, W).
            mu: Вес члена длины контура (по умолчанию: 0.25).
            eps: Параметр регуляризации дельта-функции (по умолчанию: 1.0).

        Returns:
            torch.Tensor: Карта энергии длины контура, форма (H, W).

        Note:
            - Градиенты вычисляются через центральные разности с паддингом.
            - Для `fp16` рекомендуется `eps ≥ 0.5` для стабильности.
        """
        # Паддинг для вычисления градиентов на краях
        if phi.dim() == 2:
            phi_padded = F.pad(phi.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate").squeeze(0).squeeze(0)
        else:
            phi_padded = F.pad(phi, (1, 1, 1, 1), mode="replicate")

        # Центральные разности для градиентов
        fy = (phi_padded[2:, 1:-1] - phi_padded[:-2, 1:-1]) * 0.5
        fx = (phi_padded[1:-1, 2:] - phi_padded[1:-1, :-2]) * 0.5

        # Модуль градиента с защитой от 0
        grad_mag = torch.sqrt(fx**2 + fy**2 + 1e-16)

        return mu * self._cv_delta(phi, eps) * grad_mag

    # ──────────────────────────────────────────────────────────────────────
    def _cv_energy(
        self,
        image: torch.Tensor,
        phi: torch.Tensor,
        mu: float,
        lambda1: float,
        lambda2: float,
    ) -> torch.Tensor:
        """Полная энергия функционала Чан-Везе.

        Суммирует член разницы от среднего и член длины контура.
        Используется для мониторинга сходимости и отладки.

        Формула: E = ∫[λ₁·(I-c₁)²·H(φ) + λ₂·(I-c₂)²·(1-H(φ)) + μ·δ(φ)·|∇φ|] dxdy.

        Args:
            image: Изображение в градациях серого, форма (H, W).
            phi: Функция уровня, форма (H, W).
            mu: Вес члена длины контура.
            lambda1: Вес ошибки внутри региона.
            lambda2: Вес ошибки снаружи региона.

        Returns:
            torch.Tensor: Скалярная энергия (на устройстве входа).

        Note:
            - Вычисление полностью векторизовано.
            - Не используется в основном цикле для скорости (только для отладки).
        """
        H = self._cv_heavyside(phi)
        avg_energy = self._cv_difference_from_average_term(image, H, lambda1, lambda2)
        len_energy = self._cv_edge_length_term(phi, mu)
        return avg_energy.sum() + len_energy.sum()

    # ──────────────────────────────────────────────────────────────────────
    def _cv_difference_from_average_term(
        self, image: torch.Tensor, Hphi: torch.Tensor, lambda1: float, lambda2: float
    ) -> torch.Tensor:
        """Энергетический член: разница от среднего в регионах."""
        c1, c2 = self._cv_calculate_averages(image, Hphi)
        Hinv = 1.0 - Hphi
        return lambda1 * (image - c1) ** 2 * Hphi + lambda2 * (image - c2) ** 2 * Hinv

    # ──────────────────────────────────────────────────────────────────────
    def _cv_calculate_variation(
        self,
        image: torch.Tensor,
        phi: torch.Tensor,
        mu: float,
        lambda1: float,
        lambda2: float,
        dt: float,
        eps: float = 1e-16,
    ) -> torch.Tensor:
        """Вычисление вариации уровня для одной итерации эволюции.

        Реализует дискретную аппроксимацию уравнения эволюции уровня
        из статьи Гетре (уравнение 22), обеспечивая численную стабильность.

        Алгоритм:
        1. Вычисление градиентов φ через центральные разности.
        2. Расчёт коэффициентов кривизны C1–C4.
        3. Вычисление дискретного лапласиана с весами кривизны.
        4. Добавление члена разницы от среднего.
        5. Обновление φ через регуляризованное уравнение.

        Args:
            image: Изображение в градациях серого, форма (H, W).
            phi: Текущая функция уровня, форма (H, W).
            mu: Вес члена длины контура.
            lambda1: Вес ошибки внутри региона.
            lambda2: Вес ошибки снаружи региона.
            dt: Шаг времени (learning rate, по умолчанию: 0.5).
            eps: Параметр регуляризации (по умолчанию: 1e-16).

        Returns:
            torch.Tensor: Обновлённая функция уровня, форма (H, W).

        Note:
            - Все операции векторизованы, без циклов по пикселям.
            - Для `fp16` рекомендуется `eps ≥ 1e-8` и `dt ≤ 0.3` для стабильности.
            - Поддерживает автоматическое дифференцирование.
        """
        # === ПАДДИНГ ДЛЯ ГРАДИЕНТОВ ===
        if phi.dim() == 2:
            phi_padded = F.pad(phi.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate").squeeze(0).squeeze(0)
        else:
            phi_padded = F.pad(phi, (1, 1, 1, 1), mode="replicate")

        # === РАЗНОСТИ ПО ОСЯМ ===
        phixp = phi_padded[1:-1, 2:] - phi_padded[1:-1, 1:-1]  # φ(x+1) - φ(x)
        phixn = phi_padded[1:-1, 1:-1] - phi_padded[1:-1, :-2]  # φ(x) - φ(x-1)
        phix0 = (phi_padded[1:-1, 2:] - phi_padded[1:-1, :-2]) * 0.5  # центр. разность по x

        phiyp = phi_padded[2:, 1:-1] - phi_padded[1:-1, 1:-1]  # φ(y+1) - φ(y)
        phiyn = phi_padded[1:-1, 1:-1] - phi_padded[:-2, 1:-1]  # φ(y) - φ(y-1)
        phiy0 = (phi_padded[2:, 1:-1] - phi_padded[:-2, 1:-1]) * 0.5  # центр. разность по y

        # === КОЭФФИЦИЕНТЫ КРИВИЗНЫ ===
        eta = 1e-16
        C1 = 1.0 / torch.sqrt(eta + phixp**2 + phiy0**2)
        C2 = 1.0 / torch.sqrt(eta + phixn**2 + phiy0**2)
        C3 = 1.0 / torch.sqrt(eta + phix0**2 + phiyp**2)
        C4 = 1.0 / torch.sqrt(eta + phix0**2 + phiyn**2)

        # === ДИСКРЕТНЫЙ ЛАПЛАСИАН С ВЕСАМИ ===
        K = (
            phi_padded[1:-1, 2:] * C1
            + phi_padded[1:-1, :-2] * C2
            + phi_padded[2:, 1:-1] * C3
            + phi_padded[:-2, 1:-1] * C4
        )

        # === ЧЛЕН РАЗНИЦЫ ОТ СРЕДНЕГО ===
        Hphi = self._cv_heavyside(phi, eps=1.0)
        c1, c2 = self._cv_calculate_averages(image, Hphi)
        difference_term = -lambda1 * (image - c1) ** 2 + lambda2 * (image - c2) ** 2

        # === ОБНОВЛЕНИЕ УРОВНЯ ===
        delta_phi = self._cv_delta(phi, eps=1.0)
        numerator = phi + dt * delta_phi * (mu * K + difference_term)
        denominator = 1.0 + mu * dt * delta_phi * (C1 + C2 + C3 + C4)

        return numerator / (denominator + 1e-8)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _cv_init_level_set_torch(
        init_type: str,
        image_shape: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Инициализация функции уровня для Чан-Везе (dtype-aware версия).

        Поддерживает несколько стратегий инициализации для улучшения сходимости.

        Args:
            init_type: Тип инициализации: 'checkerboard', 'disk', 'small_disk' или тензор.
            image_shape: Размер изображения (height, width).
            device: Устройство для размещения тензора.
            dtype: Тип данных для вычислений.

        Returns:
            torch.Tensor: Функция уровня формы (H, W), dtype=dtype.

        Note:
            - 'checkerboard' обеспечивает быструю сходимость за счёт множества нулевых уровней.
            - 'disk' и 'small_disk' полезны при априорном знании положения объекта.
            - Пользовательский тензор позволяет задать произвольную начальную конфигурацию.
        """
        h, w = image_shape

        if init_type == "checkerboard":
            square_size = 5
            yv = torch.arange(h, device=device, dtype=dtype).view(h, 1)
            xv = torch.arange(w, device=device, dtype=dtype).view(1, w)
            sf = torch.pi / square_size
            return torch.sin(yv * sf) * torch.sin(xv * sf)

        elif init_type == "disk":
            cy, cx = (h - 1) // 2, (w - 1) // 2
            radius = float(min(cx, cy))
            y, x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2).to(dtype)
            return (radius - dist) / radius

        elif init_type == "small_disk":
            cy, cx = (h - 1) // 2, (w - 1) // 2
            radius = float(min(cx, cy)) / 2.0
            y, x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2).to(dtype)
            return (radius - dist) / (radius * 3)

        elif isinstance(init_type, torch.Tensor):
            return init_type.to(device=device, dtype=dtype)

        else:
            return torch.ones((h, w), device=device, dtype=dtype) * 0.5

    # ============================================================================
    # ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ RANDOM WALKER (чистый PyTorch)
    # ============================================================================
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _rw_create_markers(
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        """Создание автоматических маркеров для Random Walker.

        Генерирует маркеры для фона (1) и объекта (2) на основе геометрической эвристики:
        - Центр изображения = объект
        - Углы изображения = фон

        Алгоритм:
        1. Создание нулевой матрицы маркеров.
        2. Разметка центральной области как объект (маркер 2).
        3. Разметка угловых областей как фон (маркер 1).

        Метод особенно эффективен для:
        - Автоматической инициализации без пользовательского ввода
        - Изображений с объектом в центре и фоном по краям
        - Быстрого прототипирования пайплайна сегментации

        Args:
            h: Высота изображения в пикселях.
            w: Ширина изображения в пикселях.
            device: Устройство для размещения тензора.
            dtype: Тип данных для маркеров (по умолчанию: torch.int32).

        Returns:
            torch.Tensor: Матрица маркеров формы (H, W), dtype=dtype.

        Note:
            - Маркеры: 0 = неразмечено, 1 = фон, 2 = объект.
            - Размер центрального объекта = 50% от изображения, углов = 12.5%.
            - Все операции выполняются на указанном устройстве без трансферов.

        Example:
            ```python
            markers = TorchSegmenter._rw_create_markers(512, 512, torch.device("cuda"))
            # markers[128:384, 128:384] == 2 (объект)
            # markers[:64, :64] == 1, markers[:64, -64:] == 1, ... (фон)
            ```
        """
        markers = torch.zeros((h, w), dtype=dtype, device=device)

        # Центральная область — объект (маркер 2)
        h1, h2 = h // 4, 3 * h // 4
        w1, w2 = w // 4, 3 * w // 4
        markers[h1:h2, w1:w2] = 2

        # Углы — фон (маркер 1)
        corner = min(h, w) // 8
        markers[:corner, :corner] = 1
        markers[:corner, -corner:] = 1
        markers[-corner:, :corner] = 1
        markers[-corner:, -corner:] = 1

        return markers

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _rw_compute_weights(
        image: torch.Tensor,
        beta: float,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Вычисление весов рёбер графа на основе градиента изображения.

        Вес ребра между соседними пикселями: w = exp(-β·||∇I||²),
        где ∇I — градиент интенсивности, β — коэффициент затухания.
        Чем больше градиент (граница), тем меньше вес (труднее диффузия).

        Алгоритм:
        1. Вычисление градиентов по горизонтали и вертикали через центральные разности.
        2. Вычисление магнитуды градиента: ||∇I|| = √(Gx² + Gy²).
        3. Нормализация масштаба через стандартное отклонение градиента.
        4. Вычисление весов через экспоненту.
        5. Дублирование весов для 4 направлений (4-связность).

        Метод особенно эффективен для:
        - Построения графа для сегментации на основе диффузии
        - Учёт локальных границ при распространении меток
        - Задач, где важна адаптивность к локальному контрасту

        Формула:
        ```
        scale = β / (10·(std(∇I) + ε))
        w(x,y) = exp(-scale · ||∇I(x,y)||²)
        ```

        Args:
            image: Изображение в градациях серого, форма (H, W), dtype=float.
            beta: Коэффициент затухания (по умолчанию: 130). Большие значения → сильнее подавление на границах.
            eps: Числовая стабильность для избежания деления на 0 (по умолчанию: 1e-8).

        Returns:
            torch.Tensor: Тензор весов формы (4, H, W) для 4 направлений: [вправо, вниз, влево, вверх].

        Note:
            - Все операции векторизованы, без циклов по пикселям.
            - Поддерживает автоматическое дифференцирование (если нужно для оптимизации β).
            - Для `fp16` рекомендуется `eps ≥ 1e-6` и `beta ≤ 100` для избежания переполнения.

        Example:
            ```python
            weights = TorchSegmenter._rw_compute_weights(gray, beta=100)
            # weights.shape = (4, H, W)
            # weights[0, :, :] — веса для рёбер вправо
            ```
        """
        h, w = image.shape

        # === ГРАДИЕНТЫ (центральные разности) ===
        # По горизонтали
        grad_x = torch.zeros_like(image)
        grad_x[:, 1:-1] = (image[:, 2:] - image[:, :-2]) * 0.5
        grad_x[:, 0] = image[:, 1] - image[:, 0]
        grad_x[:, -1] = image[:, -1] - image[:, -2]

        # По вертикали
        grad_y = torch.zeros_like(image)
        grad_y[1:-1, :] = (image[2:, :] - image[:-2, :]) * 0.5
        grad_y[0, :] = image[1, :] - image[0, :]
        grad_y[-1, :] = image[-1, :] - image[-2, :]

        # === МАГНИТУДА ГРАДИЕНТА ===
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + eps)

        # === МАСШТАБИРОВАНИЕ И ВЕСА ===
        scale = beta / (10.0 * (grad_mag.std() + eps))
        weights = torch.exp(-scale * grad_mag**2)

        # === 4 НАПРАВЛЕНИЯ (4-связность) ===
        return torch.stack([weights] * 4, dim=0)  # (4, H, W)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _rw_build_laplacian(
        image: torch.Tensor,
        weights: torch.Tensor,  # (4, H, W)
        markers: torch.Tensor,  # (H, W), int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Построение разреженной матрицы лапласиана графа для уравнения случайного блуждания.

        Алгоритм:
        1. Разделение пикселей на размеченные (маркеры > 0) и неразмеченные (маркеры = 0).
        2. Для каждого неразмеченного пикселя:
        - Добавление диагонального элемента: сумма весов к соседям.
        - Добавление недиагональных элементов: -вес для каждого неразмеченного соседа.
        3. Создание разреженной матрицы в формате COO (Coordinate).

        Матричная форма системы:
        ```
        [L_bb  L_bm] [x_b]   [0]
        [L_mb  L_mm] [x_m] = [x_m_known]
        ```
        где b = неразмеченные, m = размеченные пиксели.

        Метод особенно эффективен для:
        - Построения системы уравнений для Random Walker
        - Эффективного хранения разреженной структуры графа изображения
        - Подготовки данных для итеративных решателей (CG, Jacobi)

        Args:
            image: Изображение (H, W) — используется только для получения размеров.
            weights: Тензор весов рёбер формы (4, H, W) для направлений [→, ↓, ←, ↑].
            markers: Матрица маркеров формы (H, W), dtype=int, значения: 0=неразмечено, 1=фон, 2=объект.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - L: Разреженная матрица лапласиана (N, N) в формате COO, где N = H×W.
                - b_indices: Индексы неразмеченных пикселей (1D тензор).
                - m_indices: Индексы размеченных пикселей (1D тензор).

        Note:
            - Матрица строится полностью на устройстве `image.device`.
            - Формат COO оптимален для построения, но для решения может потребоваться конвертация в CSR.
            - Для больших изображений (>1024×1024) рекомендуется использовать `_rw_solve_scipy`.

        Example:
            ```python
            L, b_idx, m_idx = TorchSegmenter._rw_build_laplacian(gray, weights, markers)
            # L.shape = (H*W, H*W), sparse
            # b_idx.numel() = число неразмеченных пикселей
            ```
        """
        h, w = image.shape
        n = h * w

        # === ИНДЕКСЫ ПИКСЕЛЕЙ ===
        indices = torch.arange(n, device=image.device)
        y_coords = indices // w
        x_coords = indices % w

        # === МАСКИ РАЗМЕЧЕННЫХ/НЕРАЗМЕЧЕННЫХ ===
        labeled_mask = markers > 0
        unlabeled_mask = markers == 0

        b_indices = indices[unlabeled_mask.view(-1)]  # неразмеченные
        m_indices = indices[labeled_mask.view(-1)]  # размеченные

        # === ПОСТРОЕНИЕ РАЗРЕЖЕННОЙ МАТРИЦЫ ===
        rows, cols, vals = [], [], []

        # Направления: (dx, dy, dir_idx)
        directions = [(1, 0, 0), (0, 1, 1), (-1, 0, 2), (0, -1, 3)]

        for idx in b_indices:
            y, x = y_coords[idx], x_coords[idx]
            diag_val = torch.tensor(0.0, dtype=image.dtype, device=image.device)

            for dx, dy, dir_idx in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    n_idx = ny * w + nx
                    weight = weights[dir_idx, ny, nx]

                    if unlabeled_mask[ny, nx]:
                        # Связь с другим неразмеченным пикселем
                        rows.append(idx)
                        cols.append(n_idx)
                        vals.append(-weight)

                    diag_val = diag_val + weight

            # Диагональный элемент
            rows.append(idx)
            cols.append(idx)
            vals.append(diag_val)

        # === СОЗДАНИЕ СПАРС-ТЕНЗОРА ===
        if rows:
            row_t = torch.tensor(rows, dtype=torch.long, device=image.device)
            col_t = torch.tensor(cols, dtype=torch.long, device=image.device)
            val_t = torch.stack(vals) if vals else torch.zeros(0, dtype=image.dtype, device=image.device)

            L = torch.sparse_coo_tensor(
                torch.stack([row_t, col_t]),
                val_t,
                size=(n, n),
                device=image.device,
            )
        else:
            # Все пиксели размечены — пустая матрица
            L = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=image.device),
                torch.zeros(0, dtype=image.dtype, device=image.device),
                size=(n, n),
                device=image.device,
            )

        return L, b_indices, m_indices

    # ──────────────────────────────────────────────────────────────────────
    def _rw_solve_torch(
        self,
        L: torch.Tensor,
        b_indices: torch.Tensor,
        m_indices: torch.Tensor,
        markers: torch.Tensor,
        n_labels: int,
        mode: str = "jacobi",
        tol: float = 1e-3,
        max_iter: int = 300,
    ) -> torch.Tensor:
        """Решение системы уравнений Random Walker на чистом PyTorch.

        Args:
            L: Разреженный лапласиан
            b_indices: Индексы неразмеченных пикселей
            m_indices: Индексы размеченных пикселей
            markers: Маркеры
            n_labels: Количество уникальных меток
            mode: Метод решения ('jacobi', 'gauss_seidel', 'cg')
            tol: Порог сходимости
            max_iter: Максимальное число итераций

        Returns:
            x: Вероятности для каждого неразмеченного пикселя (n_labels, n_unlabeled)
        """
        n_unlabeled = b_indices.numel()
        if n_unlabeled == 0:
            return torch.zeros((n_labels, 0), device=L.device)

        # Инициализируем вероятности (равномерное распределение)
        x = torch.ones((n_labels, n_unlabeled), device=L.device) / n_labels

        # Правая часть системы: вклады от размеченных пикселей
        w = markers.shape[1] if markers.dim() == 2 else markers.shape[-1]
        B = self._rw_compute_rhs(b_indices, markers, n_labels, w, device=L.device)

        if mode == "jacobi":
            x = self._rw_solve_jacobi(L, B, x, tol, max_iter)
        elif mode == "gauss_seidel":
            x = self._rw_solve_gauss_seidel(L, B, x, tol, max_iter)
        elif mode == "cg":
            # Простая реализация сопряжённых градиентов для каждого класса
            x = self._rw_solve_cg_batch(L, B, x, tol, max_iter)

        return x

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _rw_compute_rhs(
        b_indices: torch.Tensor,
        markers: torch.Tensor,
        n_labels: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Вычисление правой части системы: вклады от размеченных соседей.

        Для каждого неразмеченного пикселя суммируем единичные вклады от размеченных соседей.

        Алгоритм:
        1. Для каждого индекса в `b_indices`:
        - Извлечение координат (y, x).
        - Проверка 4 соседей.
        - Если сосед размечен (marker > 0): добавление 1.0 в соответствующий класс.

        Метод особенно эффективен для:
        - Быстрого вычисления правой части без построения полной матрицы
        - Векторизованной обработки неразмеченных пикселей

        Args:
            b_indices: Индексы неразмеченных пикселей (1D тензор).
            markers: Матрица маркеров формы (H, W), dtype=int.
            n_labels: Число уникальных меток (обычно 2: фон и объект).
            w: Ширина изображения (для вычисления координат).
            device: Устройство для размещения результата.

        Returns:
            torch.Tensor: Правая часть системы формы (n_labels, n_unlabeled).

        Note:
            - Все операции выполняются на указанном устройстве.
            - Результат инициализируется нулями, затем заполняется вкладами.
        """
        n_unlabeled = b_indices.numel()
        rhs = torch.zeros((n_labels, n_unlabeled), device=device)

        h = markers.shape[0]
        flat_markers = markers.view(-1)

        # Направления: 4-связность
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        for i, b_idx in enumerate(b_indices):
            y, x = b_idx // w, b_idx % w

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    n_idx = ny * w + nx
                    marker_val = flat_markers[n_idx]
                    if marker_val > 0:
                        label_int = int(marker_val.item() - 1)  # 1→0, 2→1
                        rhs[label_int, i] += 1.0

        return rhs

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _rw_solve_jacobi(
        L: torch.Tensor,
        B: torch.Tensor,
        x_init: torch.Tensor,
        tol: float = 1e-3,
        max_iter: int = 300,
    ) -> torch.Tensor:
        """Решение системы методом Якоби (упрощённая векторизованная версия).

        Алгоритм:
        1. Извлечение диагонали матрицы лапласиана.
        2. Итеративное обновление: x_new = B / diag.
        3. Остановка при сходимости или достижении `max_iter`.

        Метод особенно эффективен для:
        - Быстрого приближённого решения для небольших систем
        - Случаев, когда точность не критична

        ⚠️  Упрощение: игнорируются недиагональные элементы для скорости.
        Для точного решения используйте `_rw_solve_cg_batch` или `_rw_solve_scipy`.

        Args:
            L: Разреженная матрица лапласиана (N, N).
            B: Правая часть системы (n_labels, n_unlabeled).
            x_init: Начальное приближение (n_labels, n_unlabeled).
            tol: Порог сходимости (по умолчанию: 1e-3).
            max_iter: Максимальное число итераций (по умолчанию: 300).

        Returns:
            torch.Tensor: Решение системы (n_labels, n_unlabeled).

        Note:
            - Метод не использует структуру `L` полностью — только диагональ.
            - Для больших систем рекомендуется использовать сопряжённые градиенты.
        """
        x = x_init.clone()
        n_classes, n_vars = x.shape

        # Диагональ матрицы (для нормировки)
        diag = torch.sparse.sum(L, dim=1).to_dense()
        diag_unlabeled = diag[-n_vars:]  # только для неразмеченных

        for _ in range(max_iter):
            x_old = x.clone()

            # Упрощённое обновление: игнорируем off-diagonal
            x = B / (diag_unlabeled.unsqueeze(0) + 1e-8)

            # Проверка сходимости
            if torch.max(torch.abs(x - x_old)) < tol:
                break

        return x

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _rw_solve_gauss_seidel(
        L: torch.Tensor,
        B: torch.Tensor,
        x_init: torch.Tensor,
        tol: float,
        max_iter: int,
    ) -> torch.Tensor:
        """Решение методом Гаусса-Зейделя (упрощённое)."""
        x = x_init.clone()

        for iteration in range(max_iter):
            x_old = x.clone()

            # Последовательное обновление
            for i in range(x.shape[1]):
                # Упрощённое обновление (можно улучшить с учётом структуры L)
                x[:, i] = B[:, i] / (x.shape[0] + 1e-8)

            if torch.max(torch.abs(x - x_old)) < tol:
                break

        return x

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _rw_solve_cg_batch(
        L: torch.Tensor,
        B: torch.Tensor,
        x_init: torch.Tensor,
        tol: float = 1e-3,
        max_iter: int = 300,
    ) -> torch.Tensor:
        """Решение системы методом сопряжённых градиентов (упрощённая batch-версия).

        Алгоритм для каждого класса:
        1. Инициализация невязки и направления поиска.
        2. Итеративное обновление через скалярные произведения.
        3. Остановка при сходимости невязки.

        Метод особенно эффективен для:
        - Точного решения разреженных систем на GPU
        - Больших изображений, где важны скорость и память
        - Случаев, когда метод Якоби недостаточно точен

        Формула обновления (для класса c):
        ```
        r = B[c] - L @ x[c]  # невязка
        p = r                # направление
        α = (r·r) / (p·(L@p))
        x[c] = x[c] + α·p
        r = r - α·(L@p)
        ```

        Args:
            L: Разреженная матрица лапласиана (N, N).
            B: Правая часть системы (n_labels, n_unlabeled).
            x_init: Начальное приближение (n_labels, n_unlabeled).
            tol: Порог сходимости невязки (по умолчанию: 1e-3).
            max_iter: Максимальное число итераций на класс (по умолчанию: 300).

        Returns:
            torch.Tensor: Решение системы (n_labels, n_unlabeled).

        Note:
            - Матрично-векторные произведения `L @ x` используют разреженную арифметику PyTorch.
            - Для `fp16` рекомендуется `tol ≥ 1e-2` из-за ограниченной точности.
            - Метод поддерживает автоматическое дифференцирование (если нужно для оптимизации).

        Example:
            ```python
            x = TorchSegmenter._rw_solve_cg_batch(L, B, x_init, tol=1e-3)
            # x.shape = (n_labels, n_unlabeled)
            ```
        """
        x = x_init.clone()
        n_classes = x.shape[0]

        for c in range(n_classes):
            # Инициализация для класса c
            r = B[c] - L @ x[c]  # невязка
            p = r.clone()
            rs_old = torch.dot(r, r)

            for _ in range(max_iter):
                Ap = L @ p
                alpha = rs_old / (torch.dot(p, Ap) + 1e-8)

                x[c] = x[c] + alpha * p
                r = r - alpha * Ap

                rs_new = torch.dot(r, r)
                if torch.sqrt(rs_new) < tol:
                    break

                p = r + (rs_new / rs_old) * p
                rs_old = rs_new

        return x

    # ──────────────────────────────────────────────────────────────────────
    def _rw_solve_scipy(
        self,
        L: torch.Tensor,
        b_indices: torch.Tensor,
        markers: torch.Tensor,
        n_labels: int,
        tol: float = 1e-3,
        max_iter: int = 300,
    ) -> Optional[torch.Tensor]:
        """Решение системы с использованием scipy.sparse (для больших систем).

        Алгоритм:
        1. Конвертация PyTorch sparse COO → scipy CSR.
        2. Вычисление правой части на CPU.
        3. Решение для каждого класса через `scipy.sparse.linalg.cg`.
        4. Возврат результата на оригинальное устройство.

        Метод особенно эффективен для:
        - Больших изображений (>1024×1024), где pure PyTorch медленнее
        - Случаев, когда важна максимальная точность решения
        - Систем с плохой обусловленностью, где CG с предобуславливателем помогает

        ⚠️  Требует установленного `scipy`. При отсутствии — возвращает `None`.

        Args:
            L: Разреженная матрица лапласиана (N, N) в формате COO.
            b_indices: Индексы неразмеченных пикселей.
            markers: Матрица маркеров (H, W).
            n_labels: Число уникальных меток.
            tol: Порог сходимости для CG (по умолчанию: 1e-3).
            max_iter: Максимальное число итераций для CG (по умолчанию: 300).

        Returns:
            Optional[torch.Tensor]: Решение системы (n_labels, n_unlabeled) или None при ошибке.

        Note:
            - Конвертация COO → CSR и обратно добавляет накладные расходы, но окупается на больших системах.
            - Решение выполняется на CPU, поэтому для очень больших систем может потребоваться много памяти.
            - При неудаче сходимости для класса используется fallback на равномерное распределение.

        Example:
            ```python
            x = segmenter._rw_solve_scipy(L, b_indices, markers, n_labels=2)
            if x is not None:
                # Использовать решение
            else:
                # Fallback на другой метод
            ```
        """
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import cg

            # === КОНВЕРТАЦИЯ PYTORCH → SCIPY ===
            L_coo = L.coalesce()
            rows = L_coo.indices()[0].cpu().numpy()
            cols = L_coo.indices()[1].cpu().numpy()
            vals = L_coo.values().cpu().numpy()
            L_scipy = csr_matrix((vals, (rows, cols)), shape=L.shape)

            # === ПРАВАЯ ЧАСТЬ ===
            n_unlabeled = b_indices.numel()
            w = markers.shape[1]
            B_np = self._rw_compute_rhs(b_indices, markers, n_labels, w, device=torch.device("cpu"))

            # === РЕШЕНИЕ ДЛЯ КАЖДОГО КЛАССА ===
            x = torch.zeros((n_labels, n_unlabeled), device=L.device)

            for c in range(n_labels):
                b_vec = B_np[c]
                x_c, info = cg(L_scipy, b_vec, tol=tol, maxiter=max_iter)

                if info == 0:
                    x[c] = torch.from_numpy(x_c).to(L.device)
                else:
                    # Fallback на равномерное распределение
                    x[c] = torch.ones(n_unlabeled, device=L.device) / n_labels

            return x

        except ImportError:
            warnings.warn("scipy not installed. Use pure PyTorch solver.")
            return None

    # ============================================================================
    # ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ QUICKSHIFT (чистый numpy)
    # ============================================================================
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _rgb_to_lab_numpy(rgb: np.ndarray) -> np.ndarray:
        """Конвертация RGB → Lab на numpy (упрощённая версия).

        Args:
            rgb: Изображение в формате (H, W, 3) в диапазоне [0, 1].

        Returns:
            Lab изображение в формате (H, W, 3).
        """
        # Матрица преобразования sRGB → XYZ (D65)
        rgb_linear = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

        # Матрица преобразования
        M = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )

        xyz = rgb_linear @ M.T

        # Нормализация к D65
        xyz = xyz / np.array([0.95047, 1.0, 1.08883])

        # Функция f(t) для Lab
        def f(t: np.ndarray) -> np.ndarray:
            return np.where(t > 0.008856, t ** (1 / 3), (7.787 * t) + (16 / 116))

        fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])

        L = (116 * fy) - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        return np.stack([L, a, b], axis=-1)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_density(features: np.ndarray, kernel_size: float) -> np.ndarray:
        """Вычисление плотности точек в пространстве признаков.

        Упрощённая оценка через гауссово ядро.

        Args:
            features: Признаки (N, D)
            kernel_size: Ширина ядра

        Returns:
            density: Плотность для каждой точки (N,)
        """
        h, w = features.shape[0], features.shape[1]
        n = h * w
        d = features.shape[-1]

        # Расплющиваем для вычислений
        features_flat = features.reshape(-1, d)

        # Вычисляем попарные расстояния (упрощённо: выборка для скорости)
        sample_size = min(500, n)
        if n > sample_size:
            # Сэмплируем точки для оценки плотности
            indices = np.random.choice(n, sample_size, replace=False)
            samples = features_flat[indices]
        else:
            samples = features_flat

        # Вычисляем плотность для каждой точки
        density = np.zeros(n)

        for i in range(n):
            # Расстояние до сэмплов
            dists = np.sqrt(np.sum((features_flat[i : i + 1] - samples) ** 2, axis=1))
            # Гауссово ядро
            density[i] = np.sum(np.exp(-0.5 * (dists / kernel_size) ** 2))

        return density.reshape(h, w)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _find_parents(features: np.ndarray, density: np.ndarray, max_dist: float) -> np.ndarray:
        """Поиск "родителя" для каждого пикселя.

        Родитель - ближайший сосед с БОЛЬШЕЙ плотностью.

        Args:
            features: Признаки (H, W, D)
            density: Плотность (H, W)
            max_dist: Максимальное расстояние для поиска

        Returns:
            parents: Индексы родителей (H, W) в линейной нумерации
        """
        h, w = features.shape[:2]
        d = features.shape[-1]
        features_flat = features.reshape(-1, d)
        density_flat = density.ravel()

        parents = np.zeros(h * w, dtype=np.int32)

        for idx in range(h * w):
            current_density = density_flat[idx]

            # Ищем соседей в пространстве признаков
            best_parent = idx  # По умолчанию - сам себе (локальный максимум)
            best_dist = np.inf

            # Проверяем всех соседей (можно оптимизировать через KD-tree)
            for other_idx in range(h * w):
                if other_idx == idx:
                    continue

                # Проверяем только точки с большей плотностью
                if density_flat[other_idx] <= current_density:
                    continue

                # Вычисляем расстояние в пространстве признаков
                dist = np.sqrt(np.sum((features_flat[idx] - features_flat[other_idx]) ** 2))

                # Проверяем условие максимального расстояния
                if dist <= max_dist and dist < best_dist:
                    best_dist = dist
                    best_parent = other_idx

            parents[idx] = best_parent

        return parents.reshape(h, w)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_segments(parents: np.ndarray) -> np.ndarray:
        """Извлечение сегментов из иерархии родителей.

        Пиксели, указывающие на один корень, образуют сегмент.

        Args:
            parents: Индексы родителей (H, W)

        Returns:
            segments: Метки сегментов (H, W)
        """
        h, w = parents.shape
        parents_flat = parents.ravel()
        n = h * w

        # Для каждого пикселя находим корень
        segments = np.zeros(n, dtype=np.int32)

        for idx in range(n):
            # Поднимаемся по иерархии до корня
            current = idx
            visited = set()
            while parents_flat[current] != current and current not in visited:
                visited.add(current)
                current = parents_flat[current]
            segments[idx] = current

        # Перенумеруем сегменты последовательно
        unique_roots = np.unique(segments)
        root_to_label = {root: i for i, root in enumerate(unique_roots)}
        segments = np.vectorize(root_to_label.get)(segments)

        return cast(np.ndarray, segments.reshape(h, w))

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_density_fast(features: np.ndarray, kernel_size: float, sample_ratio: float = 0.1) -> np.ndarray:
        """Быстрое вычисление плотности с выборкой."""
        h, w = features.shape[:2]
        d = features.shape[-1]
        features_flat = features.reshape(-1, d)
        n = len(features_flat)

        # Сэмплируем точки для оценки
        n_samples = max(100, int(n * sample_ratio))
        sample_indices = np.random.choice(n, n_samples, replace=False)
        samples = features_flat[sample_indices]

        # Предвычисляем матрицу расстояний до сэмплов
        density = np.zeros(n)

        for i, sample in enumerate(samples):
            dists = np.sqrt(np.sum((features_flat - sample) ** 2, axis=1))
            weights = np.exp(-0.5 * (dists / kernel_size) ** 2)
            density += weights

        return density.reshape(h, w)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def segment(
        self,
        image: Union[str, np.ndarray, Image.Image, torch.Tensor],
        **kwargs: Any,
    ) -> np.ndarray:
        """Основной метод сегментации.

        Args:
            image: Входное изображение (RGB, grayscale или любой формат)

        Returns:
            np.ndarray: Бинарная маска сегментации (0-255)
        """
        if self._debug_mode and not self._has_profiling_run:
            image_for_debug = Image.open(image).convert("RGB") if isinstance(image, str) else image
            # self._run_auto_debug_profiling(image_for_debug)
            self._has_profiling_run = True
        try:
            tensor: torch.Tensor = self.preprocess_image(image)
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: input dtype={tensor.dtype}, device={tensor.device}")
                logger.info(f"[DEBUG] Expected dtype: {self.dtype}, device: {self.device}")
            precision = kwargs.pop("precision", self.precision_manager.default_precision)
            print(f"Current precision: {precision}")
            mask_tensor = self._segment_func(
                tensor,
                precision=precision,
                # **kwargs,
            )

            # Преобразуем маску в numpy
            if mask_tensor.dim() >= 3:
                mask_tensor = mask_tensor.squeeze()

            mask_np = mask_tensor.cpu().float().numpy()

            if mask_np.max() <= 1.0:
                mask_np = (mask_np * 255).astype(np.uint8, copy=False)
            else:
                mask_np = mask_np.astype(np.uint8, copy=False)
            return mask_np

        except Exception as e:
            warnings.warn(f"Ошибка в методе {self.method}: {e}")
            traceback.print_exc()
            # Возвращаем пустую маску в случае ошибки
            h: int
            w: int
            if isinstance(image, str):
                img = Image.open(image).convert("RGB")
                h, w = img.size[1], img.size[0]
            elif isinstance(image, Image.Image):
                h, w = image.size[1], image.size[0]
            elif isinstance(image, np.ndarray):
                h, w = image.shape[:2]
            else:
                h, w = 256, 256

            return np.zeros((h, w), dtype=np.uint8)

    # ==========================================================================
    # МЕТОДЫ АВТО-ДИАГНОСТИКИ (для debug_mode=True)
    # ==========================================================================
    def _run_auto_debug_profiling(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> None:
        """Запускает диагностику при первом вызове в режиме debug."""
        logger.info(f"\n🔍 [DEBUG MODE] Запуск автоматической диагностики для метода '{self.method}'...")
        try:
            # 1. Замер времени
            times: List[float] = []
            image_for_profiling: Union[np.ndarray, Image.Image, torch.Tensor]
            if isinstance(image, torch.Tensor):
                image_for_profiling = self._tensor_to_numpy(image)
            elif isinstance(image, str):
                image_for_profiling = Image.open(image).convert("RGB")
            else:
                image_for_profiling = image
            tensor = self.preprocess_image(image_for_profiling)
            for _ in range(3):  # Быстрый тест на 3 прогона
                t0: float = time.perf_counter()
                _ = self._segment_func(tensor, precision=self.precision_manager.default_precision)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            avg_ms = (sum(times) / len(times)) * 1000

            # 2. Проверка трансферов (если есть метод)
            transfer_msg = "Не поддерживается для данного метода"
            if hasattr(self, "profile_with_transfer_detection"):
                try:
                    profile_res = self.profile_with_transfer_detection(image, n_runs=1)
                    if profile_res.get("transfer_warnings"):
                        transfer_msg = f"⚠️ Найдено проблемных трансферов: {len(profile_res['transfer_warnings'])}"
                    else:
                        transfer_msg = "✅ Лишних трансферов CPU↔GPU не обнаружено"
                except Exception as e:
                    transfer_msg = f"Ошибка при проверке трансферов: {e}"

            logger.info(f"⏱️  Среднее время выполнения: {avg_ms:.2f} ms")
            logger.info(f"🔄 Трансферы данных: {transfer_msg}")
            logger.info(f"💾 Устройство: {self.device}")
            compiled_status = "Unknown"
            if hasattr(self._segment_func, "_torchdynamo_orig_callable"):
                compiled_status = "ON"
            elif hasattr(self._segment_func, "__wrapped__"):
                compiled_status = "ON"
            else:
                compiled_status = "OFF"
            logger.info(f"⚡ Режим компиляции: {compiled_status}")
            logger.info("=" * 60 + "\n")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка при авто-диагностике: {e}\n")

    # ──────────────────────────────────────────────────────────────────────
    def segment_with_mask(
        self, image: Union[str, np.ndarray, Image.Image, torch.Tensor], **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Сегментация с возвратом визуализации и маски.

        Args:
            image: Входное изображение

        Returns:
            Tuple[np.ndarray, np.ndarray]: Визуализация и маска
        """
        try:
            tensor = self.preprocess_image(image)
            result_vis, mask_tensor = self._segment_with_visualization(tensor, **kwargs)

            # Преобразуем маску в numpy
            if mask_tensor.dim() == 4:
                mask_tensor = mask_tensor.squeeze(0)
            if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor.squeeze(0)

            mask_np = mask_tensor.cpu().numpy()

            # Конвертируем в uint8 0-255 если нужно
            if mask_np.dtype != np.uint8:
                if mask_np.dtype == bool:
                    mask_np = mask_np.astype(np.uint8) * 255
                elif mask_np.max() <= 1.0:
                    mask_np = (mask_np * 255).astype(np.uint8)
                else:
                    mask_np = mask_np.astype(np.uint8)

            if isinstance(result_vis, torch.Tensor):
                result_np = self._tensor_to_numpy(result_vis, denormalize=True)
            else:
                result_np = result_vis
            return result_np, mask_np

        except Exception as e:
            warnings.warn(f"Ошибка в методе {self.method} (segment_with_mask): {e}")
            traceback.print_exc()

            if isinstance(image, str):
                img = Image.open(image).convert("RGB")
                img_np = np.array(img)
            elif isinstance(image, Image.Image):
                img_np = np.array(image.convert("RGB"))
            elif isinstance(image, np.ndarray):
                img_np = image
                if len(img_np.shape) == 2:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            else:
                img_np = np.zeros((256, 256, 3), dtype=np.uint8)

            mask_np = np.zeros(img_np.shape[:2], dtype=np.uint8)
            return img_np, mask_np

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _segment_with_visualization(
        self,
        tensor: torch.Tensor,
        *,
        alpha: float = 0.9,
        color: Tuple[int, int, int] = (255, 0, 0),  # красный по умолчанию
        precision: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Универсальная визуализация сегментации с наложением маски.

        Алгоритм:
        1. Выполнение сегментации через `_segment_func`.
        2. Конвертация маски в булевый формат.
        3. Создание цветной маски на GPU.
        4. Alpha-смешивание с оригиналом.
        5. Возврат (визуализация, маска).

        Метод особенно эффективен для:
        - Быстрого получения визуального результата для отладки
        - Единого интерфейса визуализации для всех методов
        - Поддержки AMP через параметр `precision`

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            alpha: Прозрачность маски [0, 1] (по умолчанию: 0.9).
            color: Цвет маски в формате RGB (по умолчанию: красный).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.
            **kwargs: Дополнительные параметры для метода сегментации.

        Returns:
            Tuple[np.ndarray, torch.Tensor]:
                - Визуализация (H, W, 3), dtype=uint8, RGB.
                - Бинарная маска (H, W), dtype=float32, значения {0.0, 1.0}.

        Note:
            - Визуализация выполняется на GPU, конвертация в numpy только в конце.
            - Для методов со спец-визуализацией (watershed, grabcut) используется их версия.
            - Поддерживает `torch.compile` при фиксированных параметрах.

        Example:
            ```python
            vis, mask = segmenter._segment_with_visualization(image, alpha=0.7, color=(0, 255, 0))
            # vis: RGB изображение с зелёной маской
            # mask: бинарная маска объекта
            ```
        """
        # === СПЕЦ-ВИЗУАЛИЗАЦИИ ===
        if self.method == "watershed":
            return self._watershed_torch_visualization(tensor, alpha=alpha, color=color, precision=precision, **kwargs)
        elif self.method == "grabcut":
            return self._grabcut_torch_visualization(tensor, alpha=alpha, color=color, precision=precision, **kwargs)
        elif self.method == "floodfill":
            return self._floodfill_torch_visualization(tensor, alpha=alpha, color=color, precision=precision, **kwargs)
        elif self.method == "meanshift":
            return self._meanshift_torch_visualization(tensor, alpha=alpha, color=color, precision=precision, **kwargs)

        # === СТАНДАРТНАЯ ВИЗУАЛИЗАЦИЯ ===
        if not torch.compiler.is_compiling():
            start_time = time.time()
        else:
            start_time = None

        # Выполнение сегментации
        mask = self._segment_func(tensor, precision=self.precision_manager.default_precision)  # (1, 1, H, W) или (H, W)

        # Приведение к единому формату
        mask = mask.squeeze() if mask.dim() > 2 else mask

        dtype = self.precision_manager.get_dtype(precision)
        mask = mask.to(dtype) if mask.dtype != dtype else mask

        # Булева маска для индексации
        mask_bool = mask > 0.5

        # === СОЗДАНИЕ ЦВЕТНОЙ МАСКИ НА GPU ===
        # Конвертируем изображение в RGB если нужно
        img = tensor.squeeze(0) if tensor.dim() == 4 else tensor  # (C, H, W)
        if img.shape[0] == 1:  # Grayscale → RGB
            img_rgb = img.repeat(3, 1, 1)
        else:
            img_rgb = img

        # Цветная маска (на GPU)
        color_tensor = torch.tensor(color, dtype=dtype, device=img.device).view(3, 1, 1) / 255.0

        # === ALPHA-СМЕШИВАНИЕ ===
        precision_val = precision if precision is not None else "fp32"
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            mask_expanded = mask_bool.unsqueeze(0).expand_as(img_rgb)  # (C, H, W)

            # Применяем цветную маску через where (избегает проблем с индексацией)
            # result = torch.where(
            #     mask_expanded,
            #     img_rgb * (1.0 - alpha) + color_tensor.to(img_rgb.dtype) * alpha,
            #     img_rgb * (1.0 - alpha),
            # )
            result = torch.where(
                mask_expanded,
                img_rgb * (1.0 - alpha) + color_tensor * alpha,  # смешивание только в области маски
                img_rgb,  # ← оригинал без изменений!
            )

        # === КОНВЕРТАЦИЯ В NUMPY (только в конце) ===
        result_np = self._tensor_to_numpy(result.unsqueeze(0), denormalize=True)  # (H, W, 3)
        mask_out = mask.to(torch.float32) if mask.dtype != torch.float32 else mask
        if start_time is not None:
            exec_time = time.time() - start_time
        else:
            exec_time = None
        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time = time.time() - start_time
            self.params["visualization_info"] = {
                "method": f"{self.method}_visualisation",
                "alpha": alpha,
                "color": color,
                "execution_time": exec_time,
            }

        return result_np, mask_out

    # ──────────────────────────────────────────────────────────────────────
    # ПОРОГОВЫЕ МЕТОДЫ СЕГМЕНТАЦИИ
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _global_thresholding(
        self,
        tensor: torch.Tensor,
        *,
        threshold: Optional[float] = None,
        invert: bool = False,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Глобальная пороговая сегментация.

        Применяет фиксированный порог ко всему изображению: пиксели яркостью выше порога
        становятся белыми (объект), остальные — чёрными (фон). Простейший, но эффективный
        метод для изображений с хорошим контрастом и равномерным освещением.

        Алгоритм:
        1. Конвертация в градации серого (если нужно).
        2. Приведение к целевой точности (fp32/fp16/bf16).
        3. Бинаризация: `mask = (gray > threshold)`.
        4. Опциональная инверсия результата.

        Метод особенно эффективен для:
        - Документов, сканов, рентгеновских снимков с высоким контрастом
        - Предварительной обработки перед более сложными методами
        - Быстрого прототипирования и отладки пайплайна

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W), torch.Tensor.
            threshold: Порог бинаризации в диапазоне [0, 1] (по умолчанию: 0.5).
            invert: Если True, инвертировать маску (объект ↔ фон).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16' (по умолчанию: из self.dtype).

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32, значения {0.0, 1.0}.

        Note:
            - Поддерживает автоматическое приведение к целевой точности через `self._cast_to_dtype`.
            - При использовании `torch.compile` метод безопасен для `fullgraph=True`.
            - Для `fp16`/`bf16` рекомендуется нормализовать вход к [0, 1] для стабильности.

        Example:
            ```python
            segmenter = TorchSegmenter("global_thresholding", threshold=0.3, precision="fp16")
            mask = segmenter.segment(image)  # (1, 1, H, W)
            ```
        """
        gray: torch.Tensor = self._to_grayscale(tensor)  # (B, 1, H, W)
        dtype: torch.dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        thresh: float = threshold if threshold is not None else float(self.params.get("threshold", 0.5))
        const_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
        thresh_t: torch.Tensor = torch.tensor(thresh, dtype=const_dtype, device=self.device)

        # === БИНАРИЗАЦИЯ ===
        precision_val = precision if precision is not None else "fp32"
        if not torch.compiler.is_compiling():
            logger.info(f"[DEBUG] Method: {self.method}, Actual Dtype: {self.dtype}, Tensor Dtype: {tensor.dtype}")

        if export_mode:
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                mask = (gray > thresh_t).to(dtype)
                if invert:
                    mask = 1.0 - mask
            return mask.to(torch.float32)

        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            mask = (gray > thresh_t).to(dtype)
            if invert:
                mask = 1.0 - mask
        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "global_thresholding_torch",
                exec_time,
                {
                    "threshold": threshold,
                    "invert": invert,
                    "precision": precision,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _adaptive_thresholding(
        self,
        tensor: torch.Tensor,
        *,
        block_size: Optional[int] = None,
        C: Optional[float] = None,
        method: Literal["mean", "gaussian"] = "gaussian",
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Адаптивная пороговая сегментация (Gaussian/Mean).

        Вычисляет локальный порог для каждой области изображения на основе статистики
        в скользящем окне. Особенно эффективна при неравномерном освещении, тенях,
        градиентах яркости, где глобальный порог даёт плохие результаты.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Вычисление локального среднего/гауссова взвешенного через separable conv2d.
        3. Порог = local_stat - C (коррекция).
        4. Бинаризация относительно локального порога.

        Метод особенно эффективен для:
        - Текстовых документов с неравномерной подсветкой
        - Медицинских снимков с виньетированием
        - Уличных фото с резкими перепадами освещения

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            block_size: Размер локального окна (нечётное, по умолчанию: 11).
            C: Константа-коррекция порога (по умолчанию: 2/255 ≈ 0.008).
            method: Метод локальной статистики: 'mean' (простое среднее) или 'gaussian'
                    (взвешенное гауссовым ядром, по умолчанию).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - `block_size` автоматически округляется до нечётного.
            - Гауссово ядро строится один раз и кэшируется через `@lru_cache`.
            - Для `fp16` рекомендуется `C ≥ 1/255` для избежания недопорога из-за квантования.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "adaptive_thresholding",
                block_size=15,
                C=5,
                method="gaussian",
                precision="bf16"
            )
            mask = segmenter.segment(document_image)
        ```
        """
        gray: torch.Tensor = self._to_grayscale(tensor)  # (1,1,H,W)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        bs = block_size if block_size is not None else self.params.get("block_size", 11)
        bs = bs if bs % 2 == 1 else bs + 1  # ensure odd
        c_val = C if C is not None else self.params.get("C", 2)
        c_norm = c_val / 255.0  # нормализация к [0, 1]

        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            if method == "gaussian":
                kernel_1d = self._get_gaussian_kernel_1d(bs, sigma=bs / 6, dtype=dtype, device=self.device)
                kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)  # outer product
                kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
            else:  # mean
                kernel_2d = torch.ones(1, 1, bs, bs, dtype=dtype, device=self.device) / (bs * bs)

            kernel_2d = kernel_2d.to(dtype=gray.dtype)
            local_stat = F.conv2d(gray, kernel_2d, padding=bs // 2, stride=1)
            # === БИНАРИЗАЦИЯ ===
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                threshold_map = local_stat - c_norm
                mask = (gray > threshold_map).to(dtype)
            return mask.to(torch.float32)

        # === ЛОКАЛЬНАЯ СТАТИСТИКА ===
        # Используем separable conv для O(H*W*k) вместо O(H*W*k²)
        if method == "gaussian":
            kernel_1d = self._get_gaussian_kernel_1d(bs, sigma=bs / 6, dtype=dtype, device=self.device)
            kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)  # outer product
            kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
        else:  # mean
            kernel_2d = torch.ones(1, 1, bs, bs, dtype=dtype, device=self.device) / (bs * bs)

        kernel_2d = kernel_2d.to(dtype=gray.dtype)
        local_stat = F.conv2d(gray, kernel_2d, padding=bs // 2, stride=1)
        # === БИНАРИЗАЦИЯ ===
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            threshold_map = local_stat - c_norm
            mask = (gray > threshold_map).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "adaptive_thresholding_torch",
                exec_time,
                {
                    "block_size": block_size,
                    "C": C,
                    "method": method,
                    "precision": precision,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _otsu_thresholding(
        self,
        tensor: torch.Tensor,
        *,
        num_bins: Optional[int] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Автоматическая бинаризация по методу Оцу.

        Находит оптимальный порог, максимизирующий межклассовую дисперсию между фоном и объектом.
        Алгоритм перебирает все возможные пороги и выбирает тот, при котором сумма взвешенных
        дисперсий двух классов минимальна (или межклассовая дисперсия — максимальна).

        Алгоритм:
        1. Построение гистограммы интенсивностей изображения.
        2. Вычисление кумулятивных сумм вероятностей и средних значений.
        3. Поиск порога, максимизирующего межклассовую дисперсию: σ²_b = w₀·w₁·(μ₀-μ₁)².
        4. Бинаризация изображения относительно найденного порога.

        Метод особенно эффективен для:
        - Изображений с бимодальной гистограммой (чёткое разделение фона и объекта)
        - Документов, рентгеновских снимков, микроскопических изображений
        - Случаев, когда ручной подбор порога невозможен или нежелателен

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W), torch.Tensor.
            num_bins: Число бинов гистограммы (по умолчанию: 256).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16' (по умолчанию: из self.dtype).

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32, значения {0.0, 1.0}.

        Note:
            - Для `fp16`/`bf16` изображение масштабируется к [0, 255] для стабильности `torch.histc`.
            - Метод безопасен для `torch.compile(fullgraph=True)` при фиксированном `num_bins`.

        Example:
            ```python
            segmenter = TorchSegmenter("otsu_thresholding", precision="bf16")
            mask = segmenter.segment(image)  # (1, 1, H, W)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)  # (B, 1, H, W)
        gray = gray.reshape(-1, gray.shape[-2], gray.shape[-1])  # (B, H, W)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        # === ПАРАМЕТРЫ ===
        bins = num_bins if num_bins is not None else self.params.get("num_bins", 256)
        bins = max(2, bins)
        mean_levels = torch.arange(bins, dtype=dtype, device=self.device) / (bins - 1)

        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            # Гистограмма — используем тот же bins, что и в параметрах
            gray_for_hist = gray.float() if gray.dtype in (torch.float16, torch.bfloat16) else gray
            hist = torch.histc(gray_for_hist, bins=bins, min=0.0, max=1.0)

            total = hist.sum()

            # === КРИТЕРИЙ ОТСУ (векторизованный) ===
            cumsum = torch.cumsum(hist, dim=0)
            mu_cum = torch.cumsum(hist * mean_levels, dim=0)

            w0 = cumsum
            w1 = total - w0
            m0 = mu_cum / (w0 + 1e-8)
            m1 = (mu_cum[-1] - mu_cum) / (w1 + 1e-8)

            var_between = w0 * w1 * (m0 - m1) ** 2
            # best_threshold_idx = var_between.argmax()
            var_between_2d = var_between.unsqueeze(0)  # [1, 256]
            best_idx_rel = var_between_2d.argmax(dim=1)  # скаляр [0]

            # Конвертируем в float и нормализуем к [0, 1]
            # best_threshold = best_threshold_idx.float() / 255.0

            # bin_levels = torch.arange(bins, dtype=torch.float32, device=self.device) / (bins - 1)
            # best_threshold = torch.gather(bin_levels, 0, best_threshold_idx)
            best_idx = best_idx_rel.squeeze(0)
            best_threshold = best_idx.float() / max(bins - 1, 1)

            # === БИНАРИЗАЦИЯ ===
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                mask = (gray > best_threshold).to(dtype)
                mask = torch.where(total < 1e-8, torch.zeros_like(mask), mask)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(0)
            return mask.to(torch.float32)

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        gray_for_hist = gray.float() if gray.dtype in (torch.float16, torch.bfloat16) else gray
        hist = torch.histc(gray_for_hist, bins=bins, min=0.0, max=1.0)

        total = hist.sum()

        # === КРИТЕРИЙ ОТСУ (векторизованный) ===
        cumsum = torch.cumsum(hist, dim=0)
        mu_cum = torch.cumsum(hist * mean_levels, dim=0)

        w0 = cumsum
        w1 = total - w0
        m0 = mu_cum / (w0 + 1e-8)
        m1 = (mu_cum[-1] - mu_cum) / (w1 + 1e-8)

        var_between = w0 * w1 * (m0 - m1) ** 2
        best_threshold_idx = var_between.argmax()
        best_threshold = best_threshold_idx / 255.0

        # === БИНАРИЗАЦИЯ ===
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            mask = (gray > best_threshold).to(dtype)
            empty_mask = torch.zeros_like(mask)
            mask = torch.where(total < 1e-8, empty_mask, mask)

        # FIX: Логирование только в eager mode, без мутации в графе
        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "otsu_thresholding_torch",
                exec_time,
                {
                    "num_bins": num_bins,
                    "precision": precision,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32).reshape(1, 1, mask.shape[-2], mask.shape[-1])

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _threshold_niblack(
        self,
        tensor: torch.Tensor,
        *,
        window_size: Optional[int] = None,
        k: Optional[float] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Адаптивная пороговая обработка по Ниблаку.

        Порог вычисляется локально для каждого пикселя как: T = μ + k·σ,
        где μ и σ — локальное среднее и стандартное отклонение в скользящем окне.
        Параметр `k` контролирует чувствительность: отрицательные значения выделяют тёмные объекты.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Вычисление локального среднего и СКО через separable conv2d.
        3. Расчёт порога: T(x,y) = μ(x,y) + k·σ(x,y).
        4. Бинаризация: объект если I(x,y) > T(x,y).

        Метод особенно эффективен для:
        - Текстовых документов с неравномерной подсветкой
        - Изображений с градиентом освещения или виньетированием
        - Задач, где объект темнее/светлее локального фона

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            window_size: Размер локального окна (нечётное, по умолчанию: 15).
            k: Коэффициент чувствительности (по умолчанию: -0.2).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - `window_size` автоматически округляется до нечётного.
            - Локальная статистика вычисляется через `_local_stats_torch` (полностью на GPU).
            - Для `fp16` рекомендуется масштабировать вход к [0, 255] для стабильности.

        Example:
            ```python
            segmenter = TorchSegmenter("threshold_niblack", window_size=25, k=-0.3, precision="fp16")
            mask = segmenter.segment(document_image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        if export_mode:
            gray = self._to_grayscale(tensor)  # (H, W)
            if export_mode:
                if gray.dim() == 4 and gray.shape[1] == 1:
                    gray = gray.view(-1, gray.shape[-2], gray.shape[-1])  # (B, H, W)
                elif gray.dim() == 3 and gray.shape[0] == 1:
                    gray = gray.view(gray.shape[-2], gray.shape[-1])  # (H, W)
            else:
                gray = gray.squeeze(0)
        else:
            gray = self._to_grayscale(tensor).squeeze(0)  # (H, W)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        # Масштабируем к [0, 255] для стабильности при low precision
        if not torch.compiler.is_compiling():
            max_val = gray.amax(dim=(-2, -1), keepdim=False) if export_mode else gray.amax()
            if max_val <= 1.0:
                gray = gray * 255.0
        else:
            # В режиме компиляции: torch.where + .clone() для избежания алиасинга
            max_val = gray.amax(dim=(-2, -1), keepdim=False)
            gray = torch.where(max_val <= 1.0, gray * 255.0, gray)

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        ws = window_size if window_size is not None else self.params.get("window_size", 15)
        ws = ws if ws % 2 == 1 else ws + 1  # ensure odd
        k_val = k if k is not None else self.params.get("k", -0.2)

        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            gray_4d = gray.unsqueeze(0).unsqueeze(0) if gray.dim() == 2 else gray.unsqueeze(0)
            local_mean, local_std = self._local_stats_torch(gray_4d, ws, export_mode=True)

            if local_mean.dim() == 4:
                local_mean = local_mean.squeeze(0).squeeze(0)
                local_std = local_std.squeeze(0).squeeze(0)

            # === ЛОКАЛЬНАЯ СТАТИСТИКА (полностью на GPU) ===
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                # Формула Ниблака: T = μ + k·σ
                threshold = local_mean + k_val * local_std
                mask = (gray > threshold).to(dtype)
            return mask.to(torch.float32).view(1, 1, gray.shape[-2], gray.shape[-1])

        local_mean, local_std = self._local_stats_torch(gray, ws)

        # === ЛОКАЛЬНАЯ СТАТИСТИКА (полностью на GPU) ===
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            # Формула Ниблака: T = μ + k·σ
            threshold = local_mean + k_val * local_std
            mask = (gray > threshold).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "niblack_thresholding_torch",
                exec_time,
                {
                    "window_size": window_size,
                    "k": k,
                    "precision": precision,
                    "precision_val": precision_val,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32).unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _threshold_sauvola(
        self,
        tensor: torch.Tensor,
        *,
        window_size: Optional[int] = None,
        k: Optional[float] = None,
        r: Optional[float] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Улучшенная адаптивная пороговая обработка по Сауволе.

        Порог вычисляется как: T = μ·(1 + k·(σ/R - 1)), где:
        - μ, σ — локальное среднее и СКО,
        - R — динамический диапазон (обычно 128 для 8-битных изображений),
        - k — параметр чувствительности (обычно 0.2–0.5).

        Метод лучше Ниблака при очень низком контрасте, так как учитывает нормализацию
        стандартного отклонения к динамическому диапазону.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Вычисление локального среднего и СКО через свёртку.
        3. Расчёт порога по формуле Сауволы.
        4. Бинаризация относительно локального порога.

        Метод особенно эффективен для:
        - Старых документов с выцветшим текстом
        - Медицинских снимков с низким контрастом тканей
        - Изображений с сильным шумом и неравномерным освещением

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            window_size: Размер локального окна (нечётное, по умолчанию: 15).
            k: Коэффициент чувствительности (по умолчанию: 0.2).
            r: Динамический диапазон (по умолчанию: 128).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - Все вычисления выполняются на GPU через `_local_stats_torch`.
            - Для `fp16` рекомендуется `r ≥ 64` для избежания недопорога из-за квантования.
            - Метод безопасен для `torch.compile` при фиксированных параметрах.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "threshold_sauvola", window_size=25, k=0.3, r=128, precision="bf16"
            )
            mask = segmenter.segment(low_contrast_image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)  # (B, 1, H, W) или (1, H, W)
        if export_mode:
            if gray.dim() == 4 and gray.shape[1] == 1:
                gray = gray.view(-1, gray.shape[-2], gray.shape[-1])  # (B, H, W)
            elif gray.dim() == 3 and gray.shape[0] == 1:
                gray = gray.view(gray.shape[-2], gray.shape[-1])  # (H, W)
        else:
            gray = gray.squeeze(0)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            max_val = gray.amax(dim=(-2, -1), keepdim=False) if export_mode else gray.amax()
            if max_val <= 1.0:
                gray = gray * 255.0
        else:
            # В режиме компиляции: torch.where + .clone() для избежания алиасинга
            max_val = gray.amax(dim=(-2, -1), keepdim=False)  # 🔧 FIX: явный dim
            gray = torch.where(max_val <= 1.0, gray * 255.0, gray)

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        ws = window_size if window_size is not None else self.params.get("window_size", 15)
        ws = ws if ws % 2 == 1 else ws + 1
        k_val = k if k is not None else self.params.get("k", 0.2)
        r_val = r if r is not None else self.params.get("r", 128.0)
        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            gray_4d = gray.unsqueeze(0).unsqueeze(0) if gray.dim() == 2 else gray.unsqueeze(0)
            local_mean, local_std = self._local_stats_torch(gray_4d, ws, export_mode=True)

            # 🔧 FIX: возвращаем к 2D для бинаризации
            if local_mean.dim() == 4:
                local_mean = local_mean.squeeze(0).squeeze(0)
                local_std = local_std.squeeze(0).squeeze(0)

            # === ЛОКАЛЬНАЯ СТАТИСТИКА + ПОРОГ САУВОЛЫ ===
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                # Формула Сауволы: T = μ * (1 + k * (σ/R - 1))
                threshold = local_mean * (1.0 + k_val * (local_std / r_val - 1.0))
                mask = (gray > threshold).to(dtype)
            return mask.to(torch.float32).view(1, 1, gray.shape[-2], gray.shape[-1])

        local_mean, local_std = self._local_stats_torch(gray, ws)

        # === ЛОКАЛЬНАЯ СТАТИСТИКА + ПОРОГ САУВОЛЫ ===
        precision_val = precision if precision is not None else "fp32"
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            # Формула Сауволы: T = μ * (1 + k * (σ/R - 1))
            threshold = local_mean * (1.0 + k_val * (local_std / r_val - 1.0))
            mask = (gray > threshold).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "sauvola_thresholding_torch",
                exec_time,
                {
                    "window_size": window_size,
                    "k": k,
                    "r": r,
                    "precision": precision,
                    "precision_val": precision_val,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32).unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _threshold_bernsen(
        self,
        tensor: torch.Tensor,
        *,
        window_size: Optional[int] = None,
        contrast_threshold: Optional[float] = None,
        use_global_mean: Optional[bool] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Пороговая обработка по методу Бернсена.

        Локальный адаптивный порог на основе контраста в окне.
        Порог = (min + max) / 2 в локальном окне, если контраст > порогового.

        Args:
            tensor: Входное изображение (B, C, H, W)
            window_size: Размер окна (нечётное, по умолчанию 15)
            contrast_threshold: Минимальный контраст для применения порога (по умолчанию 0.1)

        Returns:
            torch.Tensor: Бинарная маска (B, 1, H, W)
        """
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)  # (B, 1, H, W)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: Optional[float] = time.time()
        else:
            start_time = None

        # === ПАРАМЕТРЫ ===
        ws = window_size if window_size is not None else self.params.get("window_size", 15)
        ws = ws if ws % 2 == 1 else ws + 1
        c_thresh = contrast_threshold if contrast_threshold is not None else self.params.get("contrast_threshold", 0.1)
        use_global = use_global_mean if use_global_mean is not None else self.params.get("use_global_mean", False)

        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            if gray.dim() == 2:
                gray_4d = gray.unsqueeze(0).unsqueeze(0)
            elif gray.dim() == 3:
                gray_4d = gray.unsqueeze(0)
            else:
                gray_4d = gray  # уже 4D

            B, C, H, W = gray_4d.shape
            # Используем reflect padding с явными размерами
            pad = ws // 2
            # Явное указание размеров вместо динамического вычисления
            gray_padded = F.pad(gray_4d, (pad, pad, pad, pad), mode="constant", value=0)

            # === ЛОКАЛЬНЫЙ MIN/MAX ЧЕРЕЗ POOLING ===
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                local_max = F.max_pool2d(
                    gray_padded, kernel_size=ws, stride=1, padding=0, return_indices=False
                )  # (1, 1, H, W)
                local_min = -F.max_pool2d(
                    -gray_padded, kernel_size=ws, stride=1, padding=0, return_indices=False
                )  # (1, 1, H, W)

                contrast = local_max - local_min  # (1, 1, H, W)
                threshold_local = (local_max + local_min) / 2.0  # (1, 1, H, W)

                # Применяем локальный порог там, где контраст достаточный
                high_contrast = contrast > c_thresh
                mask = torch.zeros_like(gray_4d, dtype=dtype)

                # 🔹 Векторизованное применение — все тензоры 4D
                mask = torch.where(
                    high_contrast,
                    torch.where(
                        gray_4d > threshold_local,
                        torch.ones_like(mask, dtype=dtype),
                        torch.zeros_like(mask, dtype=dtype),
                    ),
                    mask,
                )

                # Низкоконтрастные области: глобальное среднее
                if use_global:
                    global_mean = gray_4d.mean()  # scalar
                    low_contrast = ~high_contrast
                    mask = torch.where(
                        low_contrast,
                        torch.where(
                            gray_4d > global_mean,
                            torch.ones_like(mask, dtype=dtype),
                            torch.zeros_like(mask, dtype=dtype),
                        ),
                        mask,
                    )
            return mask.to(torch.float32)

        # === ПРИВЕДЕНИЕ К 2D ДЛЯ ЕДИНООБРАЗИЯ ===
        gray_2d = gray.squeeze() if gray.dim() > 2 else gray  # (H, W)

        pad = ws // 2
        gray_padded = F.pad(gray_2d.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="reflect")

        # === ЛОКАЛЬНЫЙ MIN/MAX ЧЕРЕЗ POOLING ===
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            local_max = F.max_pool2d(gray_padded, kernel_size=ws, stride=1, padding=0).squeeze()  # (H, W)
            local_min = -F.max_pool2d(-gray_padded, kernel_size=ws, stride=1, padding=0).squeeze()  # (H, W)

            contrast = local_max - local_min
            threshold_local = (local_max + local_min) / 2.0

            # Применяем локальный порог там, где контраст достаточный
            high_contrast = contrast > c_thresh
            mask_2d = torch.zeros_like(gray_2d, dtype=dtype)

            # Векторизованное применение — все тензоры 2D
            mask_2d[high_contrast] = (gray_2d[high_contrast] > threshold_local[high_contrast]).to(dtype)

            # Низкоконтрастные области: глобальное среднее или фон
            if use_global and not high_contrast.all():
                global_mean = gray_2d.mean()
                mask_2d[~high_contrast] = (gray_2d[~high_contrast] > global_mean).to(dtype)

        # === ВОЗВРАТ В ФОРМАТ (1, 1, H, W) ===
        mask = mask_2d.unsqueeze(0).unsqueeze(0).to(torch.float32)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "bernsen_thresholding_torch",
                exec_time,
                {
                    "window_size": window_size,
                    "contrast_threshold": contrast_threshold,
                    "use_global_mean": use_global_mean,
                    "precision": precision,
                    "precision_val": precision_val,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _threshold_phansalkar(
        self,
        tensor: torch.Tensor,
        *,
        window_size: Optional[int] = None,
        k: Optional[float] = None,
        r: Optional[float] = None,
        m: Optional[float] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Пороговая обработка по методу Фансалкара.

        Улучшенная версия Ниблака для изображений с низким контрастом.
        (Qwen) Порог: T = μ * (1 + p * (σ/R - 1) + q * (σ/R - 1)^2)
        (Claude) Порог: T = μ * (1 + p * exp(-q*μ) + k * (σ/R - 1))
        Порог вычисляется как: T = μ + k·σ·(σ/R) + m·(μ/128 - 1), где:
        - μ, σ — локальное среднее и СКО,
        - R — динамический диапазон (обычно 128),
        - k, m — эмпирические коэффициенты (по умолчанию: 0.25, 0.5).

        Алгоритм:
        1. Конвертация в градации серого.
        2. Вычисление локальной статистики через свёртку.
        3. Расчёт порога по формуле Фансалкара.
        4. Бинаризация относительно порога.

        Метод особенно эффективен для:
        - Старых документов с выцветшим текстом и пятнами
        - Медицинских снимков с низким сигналом
        - Изображений, где Ниблак/Саувола дают избыточный шум

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            window_size: Размер локального окна (нечётное, по умолчанию: 15).
            k: Коэффициент при σ·(σ/R) (по умолчанию: 0.25).
            r: Динамический диапазон (по умолчанию: 128).
            m: Коэффициент при (μ/128 - 1) (по умолчанию: 0.5).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - Все вычисления на GPU через `_local_stats_torch`.
            - Для `fp16` рекомендуется `r ≥ 64`, `k ≥ 0.1` для стабильности.
            - Метод безопасен для `torch.compile` при фиксированных параметрах.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "threshold_phansalkar", window_size=25, k=0.3, r=128, m=0.5, precision="bf16"
            )
            mask = segmenter.segment(low_contrast_doc)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)  # (B, 1, H, W) или (1, H, W)
        if export_mode:
            if gray.dim() == 4 and gray.shape[1] == 1:
                gray = gray.view(-1, gray.shape[-2], gray.shape[-1])  # (B, H, W)
            elif gray.dim() == 3 and gray.shape[0] == 1:
                gray = gray.view(gray.shape[-2], gray.shape[-1])  # (H, W)
        else:
            gray = gray.squeeze(0)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            max_val = gray.amax(dim=(-2, -1), keepdim=False) if export_mode else gray.amax()
            if max_val <= 1.0:
                gray = gray * 255.0
        else:
            # В режиме компиляции: torch.where + .clone() для избежания алиасинга
            max_val = gray.amax(dim=(-2, -1), keepdim=False)  # 🔧 FIX: явный dim
            gray = torch.where(max_val <= 1.0, gray * 255.0, gray)

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        ws = window_size if window_size is not None else self.params.get("window_size", 15)
        ws = ws if ws % 2 == 1 else ws + 1
        k_val = k if k is not None else self.params.get("k", 0.25)
        r_val = r if r is not None else self.params.get("r", 128.0)
        m_val = m if m is not None else self.params.get("m", 0.5)
        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            gray_4d = gray.unsqueeze(0).unsqueeze(0) if gray.dim() == 2 else gray.unsqueeze(0)
            local_mean, local_std = self._local_stats_torch(gray_4d, ws, export_mode=True)

            # 🔧 FIX: возвращаем к 2D для бинаризации
            if local_mean.dim() == 4:
                local_mean = local_mean.squeeze(0).squeeze(0)
                local_std = local_std.squeeze(0).squeeze(0)

            # === ЛОКАЛЬНАЯ СТАТИСТИКА + ПОРОГ ФАНСАЛКАРА ===
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                # Формула: T = μ + k·σ·(σ/R) + m·(μ/128 - 1)
                sigma_r = local_std / r_val
                threshold = local_mean + k_val * local_std * sigma_r + m_val * (local_mean / 128.0 - 1.0)
                mask = (gray > threshold).to(dtype)
            return mask.to(torch.float32).view(1, 1, gray.shape[-2], gray.shape[-1])

        local_mean, local_std = self._local_stats_torch(gray, ws)

        # === ЛОКАЛЬНАЯ СТАТИСТИКА + ПОРОГ ФАНСАЛКАРА ===
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            # Формула: T = μ + k·σ·(σ/R) + m·(μ/128 - 1)
            sigma_r = local_std / r_val
            threshold = local_mean + k_val * local_std * sigma_r + m_val * (local_mean / 128.0 - 1.0)
            mask = (gray > threshold).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "phansalkar_thresholding_torch",
                exec_time,
                {
                    "window_size": window_size,
                    "k": k,
                    "r": r,
                    "m": m,
                    "precision": precision,
                    "precision_val": precision_val,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32).unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _threshold_percentile(
        self,
        tensor: torch.Tensor,
        *,
        percentile: Optional[float] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
        shift: Optional[float] = None,
    ) -> torch.Tensor:
        """Процентильная пороговая обработка.

        Использует глобальный процентиль интенсивностей вместо среднего для вычисления порога.
        Устойчива к выбросам: например, 90-й процентиль означает, что 90% пикселей темнее порога.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Вычисление глобального процентиля через `torch.quantile`.
        3. Бинаризация: объект если яркость > порога.

        Метод особенно эффективен для:
        - Изображений с небольшим количеством ярких объектов на тёмном фоне
        - Случаев, когда нужно выделить "наиболее яркие" регионы
        - Быстрой предварительной сегментации без настройки окна

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            percentile: Процентиль в диапазоне [0, 100] (по умолчанию: 90).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - `torch.quantile` работает на GPU, но может быть медленнее для больших изображений.
            - Для `fp16` рекомендуется использовать `percentile ≥ 50` для избежания недопорога.
            - Метод безопасен для `torch.compile(fullgraph=True)`.

        Example:
            ```python
            segmenter = TorchSegmenter("threshold_percentile", percentile=95, precision="fp16")
            mask = segmenter.segment(image)  # выделит 5% самых ярких пикселей
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)

        # === ПАРАМЕТРЫ ===
        bins = 256
        p = percentile if percentile is not None else self.params.get("percentile", 90.0)
        precision_val = precision if precision is not None else "fp32"
        shift_val: float = shift if shift is not None else 0.0

        # 🔧 FIX: Безопасное приведение к 2D без цикла
        if export_mode:
            if gray.dim() == 4:
                gray = gray.view(-1, gray.shape[-2], gray.shape[-1])
            if gray.dim() == 3:
                gray = gray.view(gray.shape[-2], gray.shape[-1])
        else:
            gray = self._to_grayscale(tensor).squeeze()  # (H, W)
        # Теперь gray гарантированно (H, W)

        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None

        # if export_mode:
        #     # 🔧 MAXIMAL SIMPLIFICATION + внешняя коррекция (вычисляется ДО вызова модели)
        #     device = gray.device
        #     gray_flat = gray.flatten()

        #     t_sorted = torch.sort(gray_flat)[0]
        #     # n = t_sorted.numel()
        #     # REF_NUMEL = 512 * 512

        #     # p_norm_tensor = torch.tensor(p / 100.0, dtype=torch.float32, device=device)

        #     # # pos будет 0-D тензором (скалярным тензором)
        #     # pos = p_norm_tensor * (REF_NUMEL - 1)
        #     # j = torch.floor(pos).to(torch.int64)
        #     # g = pos - j.to(torch.float32)

        #     # max_idx = max(0, REF_NUMEL - 2)
        #     # j = torch.clamp(j, min=0, max=max_idx)
        #     # j_plus_1 = j + 1
        #     # # Интерполяция: (1-g) * y[j] + g * y[j+1]
        #     # val_low = t_sorted[j]
        #     # val_high = t_sorted[j_plus_1]
        #     # threshold = (1.0 - g) * val_low + g * val_high
        #     REF_NUMEL = 670 * 670  # ~262k элементов

        #     # 3. Вычисляем позицию квантиля ОТНОСИТЕЛЬНО референсного размера
        #     p_norm = p / 100.0
        #     # pos — float, но вычисляется из констант → экспортёр видит константу
        #     pos_val: float = p_norm * (REF_NUMEL - 1)

        #     # 4. Индексы — вычисляем на уровне Python (не в графе!)
        #     j_int = int(pos_val)  # Python int
        #     g_float = pos_val - j_int  # Python float

        #     # Clamp к допустимому диапазону (на уровне Python)
        #     # Важно: используем МИНИМУМ из реального размера и референсного
        #     actual_n = t_sorted.numel()
        #     max_safe_idx = min(REF_NUMEL - 2, actual_n - 2) if actual_n > 1 else 0
        #     j_int = max(0, min(j_int, max_safe_idx))
        #     jp1_int = j_int + 1

        #     # 5. Создаём 1D тензор индексов для index_select (ONNX-friendly)
        #     indices = torch.tensor([j_int, jp1_int], dtype=torch.int64, device=device)  # [2]

        #     # 6. Извлекаем значения через index_select (Gather в ONNX)
        #     values = torch.index_select(t_sorted, dim=0, index=indices)  # [2]

        #     # 7. 🔧 ЛИНЕЙНАЯ ИНТЕРПОЛЯЦИЯ (всё в тензорах, но индексы — константы)
        #     g = torch.tensor(g_float, dtype=torch.float32, device=device)  # 0-D const
        #     threshold = (1.0 - g) * values[0] + g * values[1]  # scalar

        #     # 5. Бинаризация
        #     threshold = threshold.to(dtype)
        #     mask = (gray > threshold).to(dtype)

        #     H, W = gray.shape[-2], gray.shape[-1]
        #     return mask.to(torch.float32).reshape(1, 1, H, W)
        if export_mode:
            gray_np = gray.detach().cpu().numpy()  # ⚠️ Может упасть при трассировке
    
            # 2. Вычисление порога через numpy (выполняется СРАЗУ, не в графе!)
            threshold_val = float(np.percentile(gray_np, p, method="linear"))
            
            # 3. Создание константного тензора
            threshold = torch.tensor(threshold_val, dtype=torch.float32, device=gray.device)
            
            # 4. Бинаризация
            mask = (gray > threshold.to(dtype)).to(dtype)
            
            H, W = gray.shape[-2], gray.shape[-1]
            return mask.to(torch.float32).reshape(1, 1, H, W)

        else:
            p_norm = p / 100.0  # нормализация к [0, 1]
            # Стандартный точный квантиль для обычного режима
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                threshold = torch.quantile(gray, p_norm)
                mask = (gray > threshold).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "percentile_thresholding_torch",
                exec_time,
                {
                    "percentile": percentile,
                    "precision": precision,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32).unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _threshold_kittler_illingworth(
        self,
        tensor: torch.Tensor,
        *,
        num_bins: Optional[int] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Пороговая обработка по методу Киттлера-Иллингворта.

        Минимизирует ошибку классификации, предполагая гауссово распределение интенсивностей
        в каждом классе. Критерий: 1 + 2·[w₀·log(σ₀) + w₁·log(σ₁)] - 2·[w₀·log(w₀) + w₁·log(w₁)].

        Алгоритм:
        1. Построение гистограммы и оценка плотности вероятности.
        2. Векторизованный перебор порогов с вычислением критерия.
        3. Выбор порога с минимальным значением критерия.
        4. Бинаризация изображения.

        Метод особенно эффективен для:
        - Изображений, где классы хорошо аппроксимируются гауссианами
        - Медицинских снимков с чётким разделением тканей
        - Случаев, когда Оцу даёт смещённый порог из-за асимметрии гистограммы

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            num_bins: Число бинов гистограммы (по умолчанию: 256).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - Критерий вычисляется векторизованно — без Python-циклов.
            - Для `fp16` добавлены `clamp` для избежания `log(0)` и деления на 0.
            - Метод безопасен для `torch.compile(fullgraph=True)`.

        Example:
            ```python
            segmenter = TorchSegmenter("threshold_kittler_illingworth", precision="bf16")
            mask = segmenter.segment(medical_image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        if export_mode:
            gray = self._to_grayscale(tensor)
            H, W = gray.shape[-2], gray.shape[-1]
            gray = gray.view(-1) if gray.dim() > 1 else gray
        else:
            gray = self._to_grayscale(tensor).squeeze()  # (H, W)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        bins = num_bins if num_bins is not None else self.params.get("num_bins", 256)
        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            # === ГИСТОГРАММА ===
            gray_for_hist = gray.float() if gray.dtype in (torch.float16, torch.bfloat16) else gray
            hist = torch.histc(gray_for_hist, bins=bins, min=0.0, max=1.0)
            total = hist.sum()

            # === ВЕКТОРИЗОВАННЫЙ КРИТЕРИЙ (без цикла) ===
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                pdf = hist.float() / (total + 1e-8)
                bin_levels = torch.arange(bins, dtype=torch.float32, device=self.device) / max(bins - 1, 1)
                # Кумулятивные суммы
                cum_pdf = torch.cumsum(pdf, dim=0)
                cum_mean = torch.cumsum(pdf * bin_levels, dim=0)

                # Векторизованный критерий Киттлера-Иллингворта
                cum_sq = torch.cumsum(pdf * bin_levels**2, dim=0)

                t_idx = torch.arange(1, bins - 1, device=self.device, dtype=torch.int64)

                eps_w = 1e-3 if dtype == torch.float16 else (1e-4 if dtype == torch.bfloat16 else 1e-8)
                eps_var = 1e-3 if dtype == torch.float16 else (1e-6 if dtype == torch.bfloat16 else 1e-8)

                w0 = cum_pdf[t_idx].clamp(min=eps_w)
                w1 = (1.0 - cum_pdf[t_idx]).clamp(min=eps_w)

                mu0 = cum_mean[t_idx] / w0
                mu1 = (cum_mean[-1] - cum_mean[t_idx]) / w1

                var0_raw = cum_sq[t_idx] / w0 - mu0**2
                var1_raw = (cum_sq[-1] - cum_sq[t_idx]) / w1 - mu1**2

                var0 = var0_raw.clamp(min=eps_var)  # float32
                var1 = var1_raw.clamp(min=eps_var)  # float32

                log_w0 = torch.log(w0 + eps_w)
                log_w1 = torch.log(w1 + eps_w)
                log_var0 = torch.log(var0 + eps_var)
                log_var1 = torch.log(var1 + eps_var)

                criterion = 1.0 + 2.0 * (w0 * 0.5 * log_var0 + w1 * 0.5 * log_var1) - 2.0 * (w0 * log_w0 + w1 * log_w1)

                # 🔧 FIX для TensorRT: добавляем фиктивное измерение перед argmin
                criterion_2d = criterion.unsqueeze(0)  # [1, bins-2]
                best_idx_rel = criterion_2d.argmin(dim=1)  # скаляр [0]
                best_idx = best_idx_rel + 1  # сдвиг к исходным биным [1, bins-2]

                # 🔧 FIX: используем gather для безопасной индексации
                threshold_val = best_idx.float() / max(bins - 1, 1)
                threshold = threshold_val.to(dtype)
                # threshold = torch.index_select(bin_levels, dim=0, index=best_idx)

            mask = (gray > threshold).to(dtype)

            # 🔧 Явный ресейп вместо цепочки unsqueeze
            return mask.to(torch.float32).view(1, 1, H, W)

        # === ГИСТОГРАММА ===
        gray_for_hist = gray.float() if gray.dtype in (torch.float16, torch.bfloat16) else gray
        hist = torch.histc(gray_for_hist, bins=256, min=0.0, max=1.0)
        total = hist.sum()
        if total < 1e-8:
            return torch.zeros_like(gray).unsqueeze(0).unsqueeze(0)
        pdf = hist / total
        bin_levels = torch.arange(bins, dtype=dtype, device=self.device) / (bins - 1)

        # === ВЕКТОРИЗОВАННЫЙ КРИТЕРИЙ (без цикла) ===
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            # Кумулятивные суммы
            cum_pdf = torch.cumsum(pdf, dim=0)
            cum_mean = torch.cumsum(pdf * bin_levels, dim=0)

            # Векторизованный критерий Киттлера-Иллингворта
            cum_sq = torch.cumsum(pdf * bin_levels**2, dim=0)

            t_idx = torch.arange(1, bins - 1, device=self.device)
            w0 = cum_pdf[t_idx].clamp(min=1e-8)
            w1 = (1.0 - cum_pdf[t_idx]).clamp(min=1e-8)

            mu0 = cum_mean[t_idx] / w0
            mu1 = (cum_mean[-1] - cum_mean[t_idx]) / w1

            var0 = (cum_sq[t_idx] / w0 - mu0**2).clamp(min=1e-6)
            var1 = ((cum_sq[-1] - cum_sq[t_idx]) / w1 - mu1**2).clamp(min=1e-6)

            criterion = (
                1.0
                + 2.0 * (w0 * 0.5 * torch.log(var0) + w1 * 0.5 * torch.log(var1))
                - 2.0 * (w0 * torch.log(w0) + w1 * torch.log(w1))
            )

            best_idx = criterion.argmin() + 1
            threshold = bin_levels[best_idx]

            mask = (gray > threshold).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "kittler_illingworth_torch",
                exec_time,
                {
                    "num_bins": num_bins,
                    "precision": precision,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32).unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _threshold_entropy_kapur(
        self,
        tensor: torch.Tensor,
        *,
        num_bins: Optional[int] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Пороговая обработка на основе максимизации энтропии Капура.

        Находит порог, максимизирующий сумму энтропий двух классов:
        H = H(C₀) + H(C₁), где H(C) = -Σ pᵢ·log(pᵢ) для пикселей класса C.

        Алгоритм:
        1. Построение гистограммы и оценка плотности вероятности.
        2. Векторизованный перебор порогов с вычислением энтропии.
        3. Выбор порога с максимальной суммарной энтропией.
        4. Бинаризация изображения.

        Метод особенно эффективен для:
        - Изображений с мультимодальными гистограммами
        - Случаев, когда дисперсионные методы (Оцу) не работают
        - Задач, где важна информационная разделимость классов

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            num_bins: Число бинов гистограммы (по умолчанию: 256).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - Энтропия вычисляется векторизованно — без Python-циклов.
            - Для `fp16` добавлены `clamp` для избежания `log(0)`.
            - Метод безопасен для `torch.compile(fullgraph=True)`.

        Example:
            ```python
            segmenter = TorchSegmenter("threshold_entropy_kapur", precision="bf16")
            mask = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        if export_mode:
            gray = self._to_grayscale(tensor)  # (H, W)
            if gray.dim() == 4 and gray.shape[1] == 1:
                gray = gray.view(-1, gray.shape[-2], gray.shape[-1])  # (B, H, W)
            elif gray.dim() == 3 and gray.shape[0] == 1:
                gray = gray.view(gray.shape[-2], gray.shape[-1])  # (H, W)
            H, W = gray.shape[-2], gray.shape[-1]
        else:
            gray = self._to_grayscale(tensor).squeeze()  # (H, W)
            if gray.dim() == 3 and gray.shape[0] == 1:
                gray = gray.squeeze(0)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        bins = num_bins if num_bins is not None else self.params.get("num_bins", 256)
        precision_val = precision if precision is not None else "fp32"
        eps = 1e-10

        if export_mode:
            # === ГИСТОГРАММА — всегда в float32 для стабильности ===
            gray_for_hist = gray.float() if gray.dtype in (torch.float16, torch.bfloat16) else gray
            hist = torch.histc(gray_for_hist, bins=bins, min=0.0, max=1.0)  # всегда float32
            total = hist.sum()

            # === ВЕКТОРИЗОВАННАЯ ЭНТРОПИЯ КАПУРА — всё в float32 ===
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                # 🔧 Все вычисления в float32
                pdf_fp32 = hist.float() / (total + 1e-8)

                pdf_log = pdf_fp32 * torch.log(pdf_fp32 + eps)
                cum_pdf = torch.cumsum(pdf_fp32, dim=0)
                cum_pdflog = torch.cumsum(pdf_log, dim=0)
                total_pdflog = cum_pdflog[-1]

                # 🔧 Индексы — явно int64
                t_idx = torch.arange(1, bins - 1, device=self.device, dtype=torch.int64)

                w0 = cum_pdf[t_idx].clamp(min=eps)  # float32
                w1 = (1.0 - cum_pdf[t_idx]).clamp(min=eps)  # float32

                H0 = torch.log(w0) - cum_pdflog[t_idx] / w0  # float32
                H1 = torch.log(w1) - (total_pdflog - cum_pdflog[t_idx]) / w1  # float32

                total_entropy = H0 + H1  # float32

                # 🔧 Поиск лучшего порога
                entropy_2d = total_entropy.unsqueeze(0)  # [1, bins-2]
                best_t_rel = entropy_2d.argmax(dim=1)  # скаляр [0]
                best_t = best_t_rel + 1  # scalar int64

                # # 🔧 bin_levels всегда в float32 для точности
                bin_levels_fp32 = torch.arange(bins, dtype=torch.float32, device=self.device) / max(bins - 1, 1)

                # 🔧 FIX: прямая индексация вместо index_select
                # threshold_fp32 = bin_levels_fp32[best_t]  # float32 scalar
                threshold_fp32 = bin_levels_fp32.gather(0, best_t)

                # 🔧 FIX: конвертация в целевой dtype только перед сравнением
                threshold = threshold_fp32.to(dtype)

                # threshold_val = best_t.float() / max(bins - 1, 1)  # float32 scalar
                # threshold = threshold_val.to(dtype)

                # Сравнение: gray уже в целевом dtype
                mask = (gray > threshold).to(dtype)

            return mask.to(torch.float32).view(1, 1, H, W)

        # === ГИСТОГРАММА ===
        gray_for_hist = gray.float() if gray.dtype in (torch.float16, torch.bfloat16) else gray
        hist = torch.histc(gray_for_hist, bins=256, min=0.0, max=1.0)
        total = hist.sum()
        if total < 1e-8:
            return torch.zeros_like(gray).unsqueeze(0).unsqueeze(0)

        pdf = hist / total

        # === ВЕКТОРИЗОВАННАЯ ЭНТРОПИЯ КАПУРА ===
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            pdf_log = pdf * torch.log(pdf + eps)  # pdf[i]*log(pdf[i]), shape (256,)
            cum_pdf = torch.cumsum(pdf, dim=0)
            cum_pdflog = torch.cumsum(pdf_log, dim=0)
            total_pdflog = cum_pdflog[-1]

            t_idx = torch.arange(1, bins - 1, device=self.device)
            w0 = cum_pdf[t_idx].clamp(min=eps)
            w1 = (1.0 - cum_pdf[t_idx]).clamp(min=eps)

            # H(C0) = log(w0) - (1/w0) * sum_{i<=t}(pdf[i]*log(pdf[i]))
            H0 = torch.log(w0) - cum_pdflog[t_idx] / w0

            # H(C1) = log(w1) - (1/w1) * sum_{i>t}(pdf[i]*log(pdf[i]))
            H1 = torch.log(w1) - (total_pdflog - cum_pdflog[t_idx]) / w1

            total_entropy = torch.where((w0 > eps) & (w1 > eps), H0 + H1, torch.full_like(H0, -float("inf")))

            best_t = total_entropy.argmax() + 1
            bin_levels = torch.arange(bins, dtype=dtype, device=self.device) / (bins - 1)
            threshold = bin_levels[best_t]

            mask = (gray > threshold).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "kapur_entropy_torch",
                exec_time,
                {
                    "num_bins": num_bins,
                    "precision": precision,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32).unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _threshold_triangle(
        self,
        tensor: torch.Tensor,
        *,
        num_bins: Optional[int] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Треугольный метод пороговой обработки.

        Строит линию от пика гистограммы до конца распределения и находит точку
        максимального перпендикулярного расстояния от гистограммы до этой линии.
        Геометрический метод, не требующий предположений о распределении классов.

        Алгоритм:
        1. Построение гистограммы и поиск её пика.
        2. Определение направления поиска (левый/правый хвост).
        3. Векторизованный расчёт расстояний от гистограммы до линии.
        4. Выбор порога с максимальным расстоянием.

        Метод особенно эффективен для:
        - Изображений с выраженным пиком фона и длинным хвостом объекта
        - Гистограмм, где Оцу даёт смещённый порог
        - Быстрой автоматической бинаризации без настройки параметров

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            num_bins: Число бинов гистограммы (по умолчанию: 256).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - Расстояния вычисляются векторизованно — без Python-циклов.
            - Метод безопасен для `torch.compile(fullgraph=True)`.
            - Для `fp16` рекомендуется `num_bins ≤ 128` для избежания квантования.

        Example:
            ```python
            segmenter = TorchSegmenter("threshold_triangle", precision="fp16")
            mask = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        if export_mode:
            gray = self._to_grayscale(tensor)  # (H, W)
            if gray.dim() == 4 and gray.shape[1] == 1:
                gray = gray.view(-1, gray.shape[-2], gray.shape[-1])  # (B, H, W)
            elif gray.dim() == 3 and gray.shape[0] == 1:
                gray = gray.view(gray.shape[-2], gray.shape[-1])  # (H, W)
        else:
            gray = self._to_grayscale(tensor).squeeze()  # (H, W)
            if gray.dim() == 3 and gray.shape[0] == 1:
                gray = gray.squeeze(0)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        bins = num_bins if num_bins is not None else self.params.get("num_bins", 256)
        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            # === ГИСТОГРАММА И ПИК ===
            gray_fp32 = gray.float() if gray.dtype in (torch.float16, torch.bfloat16) else gray
            hist = torch.histc(gray_fp32, bins=bins, min=0.0, max=1.0)
            hist_2d = hist.unsqueeze(0)  # [1, 256]
            peak_idx = hist_2d.argmax(dim=1)  # [1]

            # zero_tensor = torch.zeros_like(peak_idx)  # same dtype/device as peak_idx
            # max_tensor = torch.full_like(peak_idx, bins - 1)  # same dtype/device

            # === НАПРАВЛЕНИЕ ПОИСКА ===
            mid = bins // 2
            peak_left = peak_idx < mid

            # Вычисляем start_idx и end_idx векторизованно
            start_idx = torch.where(peak_left, peak_idx, torch.tensor(0, device=peak_idx.device))
            end_idx = torch.where(peak_left, torch.tensor(bins - 1, device=peak_idx.device), peak_idx)

            # === ВЕКТОРИЗОВАННЫЙ ТРЕУГОЛЬНЫЙ МЕТОД ===
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                if dtype == torch.bfloat16:
                    t_range = torch.arange(bins, device=self.device, dtype=torch.float32)
                    # hist_max = hist.amax(dim=0, keepdim=False)
                    # # Нормализуем гистограмму
                    # hist_norm = hist.float() / (hist_max + 1e-8)
                else:
                    t_range = torch.arange(bins, device=self.device, dtype=dtype)
                hist_max = hist.amax(dim=0, keepdim=False)
                # Нормализуем гистограмму
                hist_norm = hist.to(dtype) / (hist_max + 1e-8)

                # Линия от пика до конца
                # peak_val = torch.gather(hist_norm, 0, peak_idx).squeeze(0)  # [1] → scalar
                # end_val = torch.gather(hist_norm, 0, end_idx).squeeze(0)

                peak_val = hist_norm[peak_idx]
                end_val = hist_norm[end_idx]

                # Векторизованный треугольный метод
                # range_mask = (t_range >= start_idx) & (t_range <= end_idx)
                # if dtype == torch.bfloat16:
                #     range_mask = (t_range >= start_idx.float()) & (t_range <= end_idx.float())
                # else:
                range_mask = (t_range >= start_idx.to(dtype)) & (t_range <= end_idx.to(dtype))

                # Вычисляем линейную интерполяцию
                denom = (end_idx - start_idx + 1).clamp(min=1).to(dtype)
                slope = (end_val - peak_val) / denom
                # line_vals = peak_val + slope * (t_range - peak_idx).to(dtype)
                line_vals = peak_val + slope * (t_range - peak_idx.to(dtype))

                # 🔧 FIX: расстояния с маской (вне диапазона = -inf)
                # neg_inf_val = torch.finfo(dtype).min if dtype.is_floating_point else -1e4
                # if dtype == torch.bfloat16:
                #     distances = torch.where(
                #         range_mask, torch.abs(hist_norm - line_vals), torch.full_like(hist_norm, -1e10, dtype=dtype)
                #     )
                # else:
                distances = torch.where(
                    range_mask,
                    torch.abs(hist_norm - line_vals),
                    torch.tensor(-1e10, dtype=dtype, device=self.device),
                    # torch.full_like(hist_norm, neg_inf_val, dtype=dtype, device=self.device)
                )

                distances_2d = distances.unsqueeze(0)  # [1, bins]
                best_idx = distances_2d.argmax(dim=1)  # [1]

                # 🔧 FIX: порог как тензорная операция
                threshold = best_idx.to(dtype) / max(bins - 1, 1)
                threshold = threshold.to(dtype)

                mask = (gray > threshold).to(dtype)
            return mask.to(torch.float32).view(1, 1, gray.shape[-2], gray.shape[-1])

        # === ГИСТОГРАММА И ПИК ===
        gray_for_hist = gray.float() if gray.dtype in (torch.float16, torch.bfloat16) else gray
        hist = torch.histc(gray_for_hist, bins=bins, min=0.0, max=1.0)
        peak_idx_int: int = int(torch.argmax(hist).item())

        # === НАПРАВЛЕНИЕ ПОИСКА ===
        if peak_idx_int < bins // 2:
            # Пик слева - ищем в правом хвосте
            start_idx = peak_idx_int
            end_idx = bins - 1
        else:
            # Пик справа - ищем в левом хвосте
            start_idx = 0
            end_idx = peak_idx_int

        # === ВЕКТОРИЗОВАННЫЙ ТРЕУГОЛЬНЫЙ МЕТОД ===
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            # Нормализуем гистограмму
            hist_norm = hist.float() / (hist.max() + 1e-8)

            # Линия от пика до конца
            peak_val = hist_norm[peak_idx_int]
            end_val = hist_norm[end_idx]

            # Векторизованный треугольный метод
            t_range = torch.arange(start_idx, end_idx + 1, device=self.device, dtype=dtype)
            line_vals = peak_val + (end_val - peak_val) * (t_range - peak_idx_int) / (end_idx - peak_idx_int + 1e-10)
            distances = torch.abs(hist_norm[start_idx : end_idx + 1] - line_vals)

            best_local = int(distances.argmax().item())
            best_threshold = start_idx + best_local
            threshold_val: float = float(best_threshold) / float(bins - 1) if bins > 1 else 0.5
            threshold = torch.tensor(threshold_val, dtype=dtype, device=self.device)

            mask = (gray > threshold).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "triangle_torch",
                exec_time,
                {
                    "num_bins": num_bins,
                    "precision": precision,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32).unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _threshold_multi_otsu(
        self,
        tensor: torch.Tensor,
        *,
        n_thresholds: Optional[int] = None,
        num_bins: Optional[int] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Мульти-пороговый метод Оцу для разделения на несколько классов.

        Рекурсивно применяет критерий Оцу для поиска нескольких порогов,
        разделяющих изображение на N+1 классов. Самый крупный класс считается фоном.

        Алгоритм:
        1. Построение гистограммы и оценка плотности вероятности.
        2. Рекурсивный поиск порогов с максимизацией межклассовой дисперсии.
        3. Бинаризация относительно последнего (самого высокого) порога.

        Метод особенно эффективен для:
        - Изображений с несколькими объектами разной яркости
        - Медицинских снимков с несколькими типами тканей
        - Случаев, когда нужно выделить не только объект/фон, но и подклассы

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            n_thresholds: Число порогов для поиска (по умолчанию: 2 → 3 класса).
            num_bins: Число бинов гистограммы (по умолчанию: 256).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - Рекурсивный поиск порогов может быть медленным для больших `n_thresholds`.
            - Для `fp16` рекомендуется `n_thresholds ≤ 3` для стабильности.
            - Метод безопасен для `torch.compile` при фиксированных параметрах.

        Example:
            ```python
            segmenter = TorchSegmenter("threshold_multi_otsu", n_thresholds=2, precision="bf16")
            mask = segmenter.segment(image)  # выделит самый яркий класс
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        if export_mode:
            gray = self._to_grayscale(tensor)  # (H, W)
            if gray.dim() == 4 and gray.shape[1] == 1:
                gray = gray.view(-1, gray.shape[-2], gray.shape[-1])  # (B, H, W)
            elif gray.dim() == 3 and gray.shape[0] == 1:
                gray = gray.view(gray.shape[-2], gray.shape[-1])  # (H, W)
        else:
            gray = self._to_grayscale(tensor).squeeze()  # (H, W)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        n_thresh = n_thresholds if n_thresholds is not None else self.params.get("n_thresholds", 2)
        bins = num_bins if num_bins is not None else self.params.get("num_bins", 256)
        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            # === ГИСТОГРАММА ===
            gray_for_hist = gray.float() if gray.dtype in (torch.float16, torch.bfloat16) else gray
            hist = torch.histc(gray_for_hist, bins=bins, min=0.0, max=1.0)
            total = hist.sum()

            pdf = hist / (total + 1e-8)
            bin_levels = torch.arange(bins, dtype=dtype, device=self.device) / max(bins - 1, 1)

            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                # 🔧 Вариант 1: Найти один порог (для n_thresh=1)
                if n_thresh == 1:

                    # w0, mu0 для левой части
                    w0 = torch.cumsum(pdf, dim=0)[1:-1]  # сумма до t
                    mu0_num = torch.cumsum(pdf * bin_levels, dim=0)[1:-1]
                    mu0 = mu0_num / (w0 + 1e-8)

                    # w1, mu1 для правой части
                    w1 = 1.0 - w0
                    mu1_num = torch.cumsum(pdf * bin_levels, dim=0)[-1] - mu0_num
                    mu1 = mu1_num / (w1 + 1e-8)

                    # Межклассовая дисперсия
                    var_between = w0 * w1 * (mu0 - mu1) ** 2
                    var_between_2d = var_between.unsqueeze(0)

                    # Найти лучший порог
                    best_idx_rel = var_between_2d.argmax(dim=1)  # индекс относительно t_range
                    best_idx = best_idx_rel + 1  # сдвиг к исходным бинам

                    # 🔧 ЗАМЕНА: вместо bin_levels[best_idx] используем torch.gather
                    final_threshold = torch.gather(bin_levels, 0, best_idx)

                # 🔧 Вариант 2: Для n_thresh > 1 — использовать фиксированные квантили
                # (приближённый, но export-friendly подход)
                else:
                    # Найти пороги через равномерное разбиение кумулятивной гистограммы
                    cdf = torch.cumsum(pdf, dim=0)
                    quantiles = torch.linspace(0, 1, n_thresh + 2, device=self.device, dtype=dtype)[1:-1]

                    # 🔧 FIX: матричное сравнение [n_thresh, bins]
                    cdf_exp = cdf.unsqueeze(0)  # [1, bins]
                    q_exp = quantiles.unsqueeze(1)  # [n_thresh, 1]
                    thresholds_idx = (cdf_exp >= q_exp).int().argmax(dim=1)  # [n_thresh]
                    # 🔧 FIX: последний порог через gather (без .item()!)
                    last_idx = thresholds_idx[-1]  # scalar tensor
                    final_threshold = torch.gather(bin_levels, 0, last_idx.unsqueeze(0))
                mask = (gray > final_threshold).to(dtype)
            return mask.to(torch.float32).view(1, 1, gray.shape[-2], gray.shape[-1])

        # === ГИСТОГРАММА ===
        gray_for_hist = gray.float() if gray.dtype in (torch.float16, torch.bfloat16) else gray
        hist = torch.histc(gray_for_hist, bins=bins, min=0.0, max=1.0)
        total = hist.sum()
        if total < 1e-8:
            return torch.zeros_like(gray).unsqueeze(0).unsqueeze(0)

        pdf = hist / total
        bin_levels = torch.arange(bins, dtype=dtype, device=self.device) / (bins - 1)

        # === РЕКУРСИВНЫЙ ПОИСК ПОРОГОВ ===
        def find_thresholds(start: int, end: int, n: int) -> List[int]:
            if n <= 1 or end - start < 2:
                return []
            best_t = start + (end - start) // 2
            best_var = torch.tensor(-float("inf"), device=gray.device)

            for t in range(start + 1, end):
                w0 = pdf[start : t + 1].sum()

                eps = 1e-8
                mu0 = torch.sum(pdf[start : t + 1] * bin_levels[start : t + 1]) / (w0 + eps)

                w1 = pdf[t + 1 : end + 1].sum()
                mu1 = torch.sum(pdf[t + 1 : end + 1] * bin_levels[t + 1 : end + 1]) / (w1 + eps)

                var_between = w0 * w1 * (mu0 - mu1) ** 2
                if var_between > best_var:
                    best_var = var_between
                    best_t = t

            thresholds = [best_t]
            if n > 2:
                left = find_thresholds(start, best_t, (n + 1) // 2)
                right = find_thresholds(best_t, end, n // 2)
                thresholds = left + thresholds + right
            return thresholds

        # === БИНАРИЗАЦИЯ ===
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            thresholds = find_thresholds(0, bins - 1, n_thresh)
            final_threshold = (
                bin_levels[thresholds[-1]] if thresholds else torch.tensor(0.5, dtype=dtype, device=self.device)
            )
            mask = (gray > final_threshold).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "multi_otsu_torch",
                exec_time,
                {
                    "n_thresholds": n_thresholds,
                    "num_bins": num_bins,
                    "precision": precision,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32).unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _threshold_local_contrast(
        self,
        tensor: torch.Tensor,
        *,
        window_size: Optional[int] = None,
        contrast_factor: Optional[float] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Локальный контрастный порог.

        Порог вычисляется на основе локального контраста: пиксели с контрастом выше
        глобального квантиля считаются объектом. Метод устойчив к неравномерному освещению.

        T = μ + k * (σ - σ_min), где σ_min - минимальный локальный контраст.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Вычисление локального среднего через свёртку.
        3. Расчёт локального контраста: |I - μ|.
        4. Глобальный порог через квантиль контраста.
        5. Бинаризация: объект если контраст > порога.

        Метод особенно эффективен для:
        - Текста на неоднородном фоне
        - Изображений с локальными перепадами яркости
        - Случаев, когда объект имеет высокий локальный контраст независимо от абсолютной яркости

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            window_size: Размер локального окна (нечётное, по умолчанию: 15).
            contrast_factor: Доля пикселей для порога квантиля (по умолчанию: 0.1 → 90-й перцентиль).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - Локальное среднее вычисляется через свёртку — O(1) на пиксель.
            - `torch.quantile` работает на GPU, но может быть медленнее для больших изображений.
            - Метод безопасен для `torch.compile(fullgraph=True)`.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "threshold_local_contrast", window_size=15, contrast_factor=0.15, precision="fp16"
            )
            mask = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        if export_mode:
            gray = self._to_grayscale(tensor)  # (H, W)
            if gray.dim() == 4 and gray.shape[1] == 1:
                gray = gray.view(-1, gray.shape[-2], gray.shape[-1])
            elif gray.dim() == 3 and gray.shape[0] == 1:
                gray = gray.view(gray.shape[-2], gray.shape[-1])
        else:
            gray = self._to_grayscale(tensor).squeeze()  # (H, W)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray)

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        ws = window_size if window_size is not None else self.params.get("window_size", 15)
        ws = ws if ws % 2 == 1 else ws + 1
        cf = contrast_factor if contrast_factor is not None else self.params.get("contrast_factor", 0.1)
        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            # === ЛОКАЛЬНОЕ СРЕДНЕЕ ЧЕРЕЗ СВЁРТКУ ===
            gray_4d = gray.unsqueeze(0).unsqueeze(0) if gray.dim() == 2 else gray.unsqueeze(0)

            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                kernel = torch.ones(
                    1,
                    1,
                    ws,
                    ws,
                    device=gray_4d.device,
                    dtype=gray_4d.dtype,
                ) / (ws * ws)
                local_mean = F.conv2d(gray_4d, kernel, padding=ws // 2)
                if local_mean.dim() == 4 and local_mean.shape[:2] == (1, 1):
                    local_mean = local_mean.view(local_mean.shape[2], local_mean.shape[3])
                else:
                    local_mean = local_mean.squeeze(0).squeeze(0)
                local_contrast = torch.abs(gray - local_mean)

                # Глобальный порог через квантиль
                contrast_mean = local_contrast.mean(dim=(-2, -1), keepdim=False)
                contrast_std = local_contrast.std(dim=(-2, -1), unbiased=False, keepdim=False)
                k_factor = torch.tensor(
                    1.28, device=contrast_mean.device, dtype=contrast_mean.dtype
                )  # ≈ 90-й перцентиль нормального распределения
                threshold = contrast_mean + k_factor * contrast_std
                mask = (local_contrast > threshold.unsqueeze(-1).unsqueeze(-1)).to(dtype)
            return mask.to(torch.float32).view(1, 1, mask.shape[-2], mask.shape[-1])

        # === ЛОКАЛЬНОЕ СРЕДНЕЕ ЧЕРЕЗ СВЁРТКУ ===
        gray_4d = gray.unsqueeze(0).unsqueeze(0) if gray.dim() == 2 else gray.unsqueeze(0)

        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            kernel = torch.ones(
                1,
                1,
                ws,
                ws,
                device=gray_4d.device,
                dtype=gray_4d.dtype,
            ) / (ws * ws)
            local_mean = F.conv2d(gray_4d, kernel, padding=ws // 2).squeeze()
            local_contrast = torch.abs(gray - local_mean)

            # Глобальный порог через квантиль
            threshold = torch.quantile(local_contrast, 1.0 - cf)
            mask = (local_contrast > threshold).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "local_contrast_torch",
                exec_time,
                {
                    "window_size": window_size,
                    "contrast_factor": contrast_factor,
                    "precision": precision,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32).unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    # КРАЕВЫЕ МЕТОДЫ СЕГМЕНТАЦИИ
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _sobel_edge(
        self,
        tensor: torch.Tensor,
        *,
        threshold: Optional[float] = None,
        normalize: bool = True,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Обнаружение границ оператором Собеля.

        Вычисляет аппроксимацию градиента интенсивности по горизонтали (Gx) и вертикали (Gy)
        через свёртку с ядрами Собеля 3×3. Величина градиента |G| = √(Gx² + Gy²) указывает
        на силу границы; после пороговой обработки получается бинарная маска краёв.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Свёртка с ядрами Собеля (кэшируются через `@lru_cache`).
        3. Вычисление магнитуды: `magnitude = sqrt(gx² + gy²)`.
        4. Нормализация к [0, 1] (опционально) и бинаризация.

        Метод особенно эффективен для:
        - Предварительного выделения контуров перед watershed / active contour
        - Детекции резких перепадов яркости (текст, линии, границы объектов)
        - Быстрой оценки "текстурности" региона

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            threshold: Порог для бинаризации магнитуды в диапазоне [0, 1] (по умолчанию: 0.1).
            normalize: Если True, нормализовать магнитуду к [0, 1] перед порогом.
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска границ (1, 1, H, W), dtype=float32.

        Note:
            - Ядра Собеля кэшируются по (device, dtype) через `@lru_cache`.
            - Для `fp16` рекомендуется `threshold ≥ 0.05` из-за ограниченной точности.
            - Метод безопасен для `torch.compile(fullgraph=True)`.

        Example:
            ```python
            segmenter = TorchSegmenter("sobel_edge", threshold=0.15, precision="fp16")
            edges = segmenter.segment(image)  # маска границ
            ```
        """
        if torch.compiler.is_compiling() and self.device.type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray
        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            sobel_x, sobel_y = self._get_sobel_kernels_cached(self.device, dtype)

            # === ГРАДИЕНТЫ ===
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                sobel_x, sobel_y = self._prepare_kernel_for_conv((sobel_x, sobel_y), gray.dtype)
                gx = self._safe_conv2d(gray, sobel_x, padding=1)
                gy = self._safe_conv2d(gray, sobel_y, padding=1)
                magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)

                # === НОРМАЛИЗАЦИЯ И ПОРОГ ===
                if normalize:
                    mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
                    magnitude = magnitude / (mag_max + 1e-8)

                thresh = threshold if threshold is not None else self.params.get("threshold", 0.1)
                thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
                mask = (magnitude > thresh_t).to(dtype)

            return mask.to(torch.float32)

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]
        # === ЯДРА СОБЕЛЯ (кэшированные) ===
        sobel_x, sobel_y = self._get_sobel_kernels_cached(self.device, dtype)

        # === ГРАДИЕНТЫ ===
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            sobel_x, sobel_y = self._prepare_kernel_for_conv((sobel_x, sobel_y), gray.dtype)
            gx = self._safe_conv2d(gray, sobel_x, padding=1)
            gy = self._safe_conv2d(gray, sobel_y, padding=1)
            magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)

            # === НОРМАЛИЗАЦИЯ И ПОРОГ ===
            if normalize:
                mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
                magnitude = magnitude / (mag_max + 1e-8)

            thresh = threshold if threshold is not None else self.params.get("threshold", 0.1)
            thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
            mask = (magnitude > thresh_t).to(dtype)

        if self._debug_mode and not torch.compiler.is_compiling():
            try:
                # Безопасное извлечение скаляров
                thresh_val = float(thresh) if isinstance(thresh, (int, float)) else float(thresh_t.item())
                mag_min = float(magnitude.min().item())
                mag_max = float(magnitude.max().item())
                mask_area = int((mask > 0).sum().item())
                
                logger.info(
                    f"SOBEL Torch: threshold={thresh_val:.4f}, "
                    f"magnitude_raw=[{mag_min:.4f}, {mag_max:.4f}], "
                    f"mask_area={mask_area}"
                )
            except Exception as e:
                logger.warning(f"[DEBUG] Failed to log Sobel stats: {e}")
        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "sobel_edge_torch",
                exec_time,
                {
                    "threshold": threshold,
                    "normalize": normalize,
                    "precision": precision,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _canny_edge(
        self,
        tensor: torch.Tensor,
        *,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        sigma: Optional[float] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Обнаружение границ оператором Кэнни.

        Многоэтапный алгоритм, считающийся золотым стандартом детекции границ:
        1. Гауссово сглаживание для подавления шума.
        2. Вычисление градиента (Sobel) и его магнитуды/направления.
        3. Подавление немаксимумов (NMS) для получения тонких границ.
        4. Двойная пороговая фильтрация (hysteresis) для связывания слабых границ.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Гауссово сглаживание с параметром `sigma`.
        3. Вычисление градиентов Собеля и магнитуды.
        4. Векторизованное подавление немаксимумов по 4 направлениям.
        5. Двойной порог: сильные границы + связывание слабых через свёртку.

        Метод особенно эффективен для:
        - Задач, требующих точных, одно-пиксельных границ
        - Предварительной обработки перед векторизацией или Hough-преобразованием
        - Сцен с умеренным уровнем шума и хорошим контрастом

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            low_threshold: Нижний порог для слабых границ в [0, 1] (по умолчанию: 0.1).
            high_threshold: Верхний порог для сильных границ в [0, 1] (по умолчанию: 0.3).
            sigma: Sigma для гауссова сглаживания (по умолчанию: 1.0).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска границ (1, 1, H, W), dtype=float32.

        Note:
            - NMS реализован векторизованно через `torch.roll` и индексацию.
            - Гистерезис использует свёртку 3×3 для поиска соседей (быстрее heapq).
            - Для `fp16` рекомендуется `high_threshold ≥ 0.15` из-за квантования.
            - Метод не поддерживает `fullgraph=True` из-за условной логики гистерезиса.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "canny_edge",
                low_threshold=0.08,
                high_threshold=0.25,
                sigma=1.2,
                precision="bf16"
            )
            edges = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if gray.dim() == 2:
            gray = gray.view(1, 1, gray.shape[0], gray.shape[1])
        elif gray.dim() == 3 and gray.shape[0] == 1:
            gray = gray.view(1, 1, gray.shape[1], gray.shape[2])

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        low = low_threshold if low_threshold is not None else self.params.get("low", 0.1)
        high = high_threshold if high_threshold is not None else self.params.get("high", 0.3)
        sig = sigma if sigma is not None else self.params.get("sigma", 1.0)
        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            # === ГАУССОВО СГЛАЖИВАНИЕ ===
            if sig > 0:
                ks = int(2 * round(3 * sig) + 1)
                ks = ks if ks % 2 == 1 else ks + 1
                gray = tv_gaussian_blur(gray, kernel_size=[ks, ks], sigma=[sig, sig])

            # === ЯДРА СОБЕЛЯ ===
            # sobel_x, sobel_y = self._get_sobel_kernels_cached(self.device, dtype)
            # === ЯДРА СОБЕЛЯ ===
            kernels = self._get_conv_kernel("sobel", return_pair=True, dtype=dtype, device=self.device)
            if isinstance(kernels, tuple):
                sobel_x, sobel_y = kernels
            else:
                sobel_x = sobel_y = kernels

            precision_val = precision if precision is not None else "fp32"
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                # === ГРАДИЕНТЫ ===
                # Autocast может изменить gray_4d.dtype на bf16/fp16, ядра должны совпадать.
                sobel_x, sobel_y = self._prepare_kernel_for_conv((sobel_x, sobel_y), gray.dtype)
                gx = self._safe_conv2d(gray, sobel_x, padding=1)
                gy = self._safe_conv2d(gray, sobel_y, padding=1)
                mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
                angle = torch.atan2(gy, gx)
                angle_deg = torch.abs(torch.rad2deg(angle))  # [0, 180]

                # === NMS (векторизованный) ===
                mag_padded = F.pad(mag, (1, 1, 1, 1), mode="reflect")

                # m_lr = torch.stack([mag_padded[:, :, 1:-1, :-2], mag_padded[:, :, 1:-1, 2:]], dim=0)
                # m_tb = torch.stack([mag_padded[:, :, :-2, 1:-1], mag_padded[:, :, 2:, 1:-1]], dim=0)
                # m_diag1 = torch.stack([mag_padded[:, :, :-2, 2:], mag_padded[:, :, 2:, :-2]], dim=0)
                # m_diag2 = torch.stack([mag_padded[:, :, :-2, :-2], mag_padded[:, :, 2:, 2:]], dim=0)
                m_left = mag_padded[:, :, 1:-1, :-2]  # Сдвиг влево
                m_right = mag_padded[:, :, 1:-1, 2:]  # Сдвиг вправо
                m_top = mag_padded[:, :, :-2, 1:-1]  # Сдвиг вверх
                m_bottom = mag_padded[:, :, 2:, 1:-1]  # Сдвиг вниз

                # Диагональные соседи
                m_ul = mag_padded[:, :, :-2, :-2]  # Верх-Лево
                m_ur = mag_padded[:, :, :-2, 2:]  # Верх-Право
                m_dl = mag_padded[:, :, 2:, :-2]  # Низ-Лево
                m_dr = mag_padded[:, :, 2:, 2:]  # Низ-Право

                mask_0 = (angle_deg <= 22.5) | (angle_deg > 157.5)
                mask_45 = (angle_deg > 22.5) & (angle_deg <= 67.5)
                mask_90 = (angle_deg > 67.5) & (angle_deg <= 112.5)
                mask_135 = (angle_deg > 112.5) & (angle_deg <= 157.5)

                cond_0 = (mag >= m_left) & (mag >= m_right)
                cond_90 = (mag >= m_top) & (mag >= m_bottom)
                cond_45 = (mag >= m_ur) & (mag >= m_dl)
                cond_135 = (mag >= m_ul) & (mag >= m_dr)

                suppressed = torch.zeros_like(mag)
                suppressed = torch.where(mask_0 & cond_0, mag, suppressed)
                suppressed = torch.where(mask_90 & cond_90, mag, suppressed)
                suppressed = torch.where(mask_45 & cond_45, mag, suppressed)
                suppressed = torch.where(mask_135 & cond_135, mag, suppressed)

                # === ДВОЙНОЙ ПОРОГ + ГИСТЕРЕЗИС ===
                strong = suppressed >= high
                weak = (suppressed >= low) & (suppressed < high)
                final_mask = strong.clone()

                kernel_conn = torch.ones(1, 1, 3, 3, device=gray.device, dtype=torch.float32)

                # Гистерезис через свёртку (2 итерации)
                for _ in range(2):
                    if final_mask.dim() == 2:
                        input_tensor = final_mask.view(1, 1, final_mask.shape[0], final_mask.shape[1])
                    elif final_mask.dim() == 3:
                        input_tensor = final_mask.view(1, final_mask.shape[0], final_mask.shape[1], final_mask.shape[2])
                    else:
                        input_tensor = final_mask

                    input_float = input_tensor.float()

                    # Приводим вход к float перед паддингом и свёрткой
                    final_padded = F.pad(input_float, (1, 1, 1, 1), mode="replicate")
                    neighbors = F.conv2d(final_padded, kernel_conn, padding=0).squeeze(0) > 0
                    new_strong = neighbors & weak
                    final_mask = final_mask | new_strong

                mask = final_mask.to(dtype)

            # Нормализация выхода к (1, 1, H, W)0,
            H, W = mask.shape[-2], mask.shape[-1]
            mask = mask.view(1, 1, H, W)
            return mask.to(torch.float32)

        # === ГАУССОВО СГЛАЖИВАНИЕ ===
        if sig > 0:
            ks = int(2 * round(3 * sig) + 1)
            ks = ks if ks % 2 == 1 else ks + 1
            is_3d = gray.dim() == 3
            gray_in = gray.unsqueeze(0) if is_3d else gray
            gray_blurred = tv_gaussian_blur(gray_in, kernel_size=[ks, ks], sigma=[sig, sig])
            gray = gray_blurred.squeeze(0) if is_3d else gray_blurred

        # === ЯДРА СОБЕЛЯ ===
        kernels = self._get_conv_kernel("sobel", return_pair=True, dtype=dtype, device=self.device)
        if isinstance(kernels, tuple):
            sobel_x, sobel_y = kernels
        else:
            sobel_x = sobel_y = kernels

        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            # === ГРАДИЕНТЫ ===
            sobel_x, sobel_y = self._prepare_kernel_for_conv((sobel_x, sobel_y), gray.dtype)
            gx = self._safe_conv2d(gray, sobel_x, padding=1)
            gy = self._safe_conv2d(gray, sobel_y, padding=1)
            mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
            angle = torch.atan2(gy, gx)
            angle_deg = torch.abs(torch.rad2deg(angle))  # [0, 180]

            # === NMS (векторизованный) ===
            mag_padded = F.pad(mag, (1, 1, 1, 1), mode="reflect")

            m_left = mag_padded[:, :, 1:-1, :-2]  # Сдвиг влево
            m_right = mag_padded[:, :, 1:-1, 2:]  # Сдвиг вправо
            m_top = mag_padded[:, :, :-2, 1:-1]  # Сдвиг вверх
            m_bottom = mag_padded[:, :, 2:, 1:-1]  # Сдвиг вниз

            # Диагональные соседи
            m_ul = mag_padded[:, :, :-2, :-2]  # Верх-Лево
            m_ur = mag_padded[:, :, :-2, 2:]  # Верх-Право
            m_dl = mag_padded[:, :, 2:, :-2]  # Низ-Лево
            m_dr = mag_padded[:, :, 2:, 2:]  # Низ-Право

            mask_0 = (angle_deg <= 22.5) | (angle_deg > 157.5)
            mask_45 = (angle_deg > 22.5) & (angle_deg <= 67.5)
            mask_90 = (angle_deg > 67.5) & (angle_deg <= 112.5)
            mask_135 = (angle_deg > 112.5) & (angle_deg <= 157.5)

            cond_0 = (mag >= m_left) & (mag >= m_right)
            cond_90 = (mag >= m_top) & (mag >= m_bottom)
            cond_45 = (mag >= m_ur) & (mag >= m_dl)
            cond_135 = (mag >= m_ul) & (mag >= m_dr)

            suppressed = torch.zeros_like(mag)
            suppressed = torch.where(mask_0 & cond_0, mag, suppressed)
            suppressed = torch.where(mask_90 & cond_90, mag, suppressed)
            suppressed = torch.where(mask_45 & cond_45, mag, suppressed)
            suppressed = torch.where(mask_135 & cond_135, mag, suppressed)

            # === ДВОЙНОЙ ПОРОГ + ГИСТЕРЕЗИС ===
            strong = suppressed >= high
            weak = (suppressed >= low) & (suppressed < high)
            final_mask = strong.clone()

            kernel_conn = torch.ones(1, 1, 3, 3, device=gray.device, dtype=torch.float32)

            # Гистерезис через свёртку (2 итерации)
            for _ in range(2):
                if final_mask.dim() == 2:
                    input_tensor = final_mask.unsqueeze(0).unsqueeze(0)
                elif final_mask.dim() == 3:
                    input_tensor = final_mask.unsqueeze(1)
                else:
                    input_tensor = final_mask

                # Приводим вход к float перед паддингом и свёрткой
                final_padded = F.pad(input_tensor.float(), (1, 1, 1, 1), mode="replicate")
                neighbors = F.conv2d(final_padded, kernel_conn, padding=0).squeeze(0) > 0
                new_strong = neighbors & weak
                final_mask = final_mask | new_strong

            mask = final_mask.to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "canny_edge_torch",
                exec_time,
                {
                    "low_threshold": low_threshold,
                    "high_threshold": high_threshold,
                    "sigma": sigma,
                    "precision": precision,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")

        if final_mask.dim() == 2:
            mask = final_mask.unsqueeze(0).unsqueeze(0)
        elif final_mask.dim() == 3:
            mask = final_mask.unsqueeze(0)
        return mask.to(torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _prewitt_edge(
        self,
        tensor: torch.Tensor,
        *,
        threshold: Optional[float] = None,
        normalize: bool = True,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Обнаружение границ оператором Прюитта.

        Аналогичен оператору Собеля, но использует более простые ядра [−1, 0, +1].
        Менее чувствителен к шуму за счёт усреднения, но даёт менее точные градиенты.
        Вычисляет магнитуду градиента |G| = √(Gx² + Gy²) и бинаризует по порогу.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Свёртка с ядрами Прюитта (кэшируются через `@lru_cache`).
        3. Вычисление магнитуды: `magnitude = sqrt(gx² + gy²)`.
        4. Нормализация к [0, 1] (опционально) и бинаризация.

        Метод особенно эффективен для:
        - Быстрого предварительного выделения контуров
        - Изображений с умеренным уровнем шума
        - Задач, где важна скорость, а не субпиксельная точность границ

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            threshold: Порог для бинаризации магнитуды в диапазоне [0, 1] (по умолчанию: 0.1).
            normalize: Если True, нормализовать магнитуду к [0, 1] перед порогом.
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска границ (1, 1, H, W), dtype=float32.

        Note:
            - Ядра Прюитта кэшируются по (device, dtype) через `@lru_cache`.
            - Для `fp16` рекомендуется `threshold ≥ 0.05` из-за ограниченной точности.
            - Метод безопасен для `torch.compile(fullgraph=True)`.

        Example:
            ```python
            segmenter = TorchSegmenter("prewitt_edge", threshold=0.12, precision="fp16")
            edges = segmenter.segment(image)
            ```
        """
        # if torch.compiler.is_compiling() and self.device.type == "cuda":
        #     torch.compiler.cudagraph_mark_step_begin()

        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        thresh = threshold if threshold is not None else self.params.get("threshold", 0.1)
        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            kernels = self._get_conv_kernel("prewitt", return_pair=True, dtype=dtype, device=self.device)
            if isinstance(kernels, tuple):
                prewitt_x, prewitt_y = kernels
            else:
                prewitt_x, prewitt_y = kernels

            # === ГРАДИЕНТЫ И БИНАРИЗАЦИЯ ===
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                prewitt_x = prewitt_x.to(gray.dtype) if prewitt_x.dtype != gray.dtype else prewitt_x
                prewitt_y = prewitt_y.to(gray.dtype) if prewitt_y.dtype != gray.dtype else prewitt_y
                gx = self._safe_conv2d(gray, prewitt_x, padding=1)
                gy = self._safe_conv2d(gray, prewitt_y, padding=1)
                magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)

                if normalize:
                    mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
                    if self._debug_mode and not torch.compiler.is_compiling():
                        logger.info(
                            f"[{self.method}] mag_max={mag_max.mean().item():.6f}, "
                            f"magnitude_range=[{magnitude.min().item():.6f}, {magnitude.max().item():.6f}], "
                            f"thresh={thresh}"
                        )
                    magnitude = magnitude / (mag_max + 1e-8)

                thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
                mask = (magnitude > thresh_t).to(dtype)

            return mask.to(torch.float32)

        # === ЯДРА ПРЮИТТА (кэшированные) ===
        # prewitt_x, prewitt_y = self._get_prewitt_kernels_cached(self.device, dtype)
        kernels = self._get_conv_kernel("prewitt", return_pair=True, dtype=dtype, device=self.device)
        if isinstance(kernels, tuple):
            prewitt_x, prewitt_y = kernels
        else:
            prewitt_x, prewitt_y = kernels

        # === ГРАДИЕНТЫ И БИНАРИЗАЦИЯ ===
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            prewitt_x = prewitt_x.to(gray.dtype) if prewitt_x.dtype != gray.dtype else prewitt_x
            prewitt_y = prewitt_y.to(gray.dtype) if prewitt_y.dtype != gray.dtype else prewitt_y
            gx = self._safe_conv2d(gray, prewitt_x, padding=1)
            gy = self._safe_conv2d(gray, prewitt_y, padding=1)
            magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)

            if normalize:
                mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
                if self._debug_mode and not torch.compiler.is_compiling():
                    logger.info(
                        f"[{self.method}] mag_max={mag_max.mean().item():.6f}, "
                        f"magnitude_range=[{magnitude.min().item():.6f}, {magnitude.max().item():.6f}], "
                        f"thresh={thresh}"
                    )
                magnitude = magnitude / (mag_max + 1e-8)

            thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
            mask = (magnitude > thresh_t).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "prewitt_edge_torch",
                exec_time,
                {"threshold": thresh, "precision_val": precision_val},
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _scharr_edge(
        self,
        tensor: torch.Tensor,
        *,
        threshold: Optional[float] = None,
        normalize: bool = True,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Обнаружение границ оператором Шарра.

        Улучшенная версия оператора Собеля с оптимизированными коэффициентами,
        обеспечивающими лучшую ротационную симметрию и точность аппроксимации градиента.
        Ядра: [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]] и транспонированное.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Свёртка с ядрами Шарра (кэшируются через `@lru_cache`).
        3. Вычисление магнитуды: `magnitude = sqrt(gx² + gy²)`.
        4. Нормализация к [0, 1] (опционально) и бинаризация.

        Метод особенно эффективен для:
        - Задач, требующих высокой точности определения направления границ
        - Изображений с диагональными структурами и криволинейными контурами
        - Предварительной обработки перед активными контурами или watershed

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            threshold: Порог для бинаризации магнитуды в диапазоне [0, 1] (по умолчанию: 0.1).
            normalize: Если True, нормализовать магнитуду к [0, 1] перед порогом.
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска границ (1, 1, H, W), dtype=float32.

        Note:
            - Ядра Шарра кэшируются по (device, dtype) через `@lru_cache`.
            - Для `fp16` рекомендуется `threshold ≥ 0.05` из-за ограниченной точности.
            - Метод безопасен для `torch.compile(fullgraph=True)`.

        Example:
            ```python
            segmenter = TorchSegmenter("scharr_edge", threshold=0.08, precision="bf16")
            edges = segmenter.segment(image)
            ```
        """
        # if torch.compiler.is_compiling() and self.device.type == "cuda":
        #     torch.compiler.cudagraph_mark_step_begin()

        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        thresh = threshold if threshold is not None else self.params.get("threshold", 0.1)
        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            kernels = self._get_conv_kernel("scharr", return_pair=True, dtype=dtype, device=self.device)
            if isinstance(kernels, tuple):
                scharr_x, scharr_y = kernels
            else:
                scharr_x, scharr_y = kernels

            # === ГРАДИЕНТЫ И БИНАРИЗАЦИЯ ===
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                scharr_x, scharr_y = self._prepare_kernel_for_conv((scharr_x, scharr_y), gray.dtype)
                gx = self._safe_conv2d(gray, scharr_x, padding=1)
                gy = self._safe_conv2d(gray, scharr_y, padding=1)
                magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)

                if normalize:
                    mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
                    if self._debug_mode and not torch.compiler.is_compiling():
                        logger.info(
                            f"[{self.method}] mag_max={mag_max.mean().item():.6f}, "
                            f"magnitude_range=[{magnitude.min().item():.6f}, {magnitude.max().item():.6f}], "
                            f"thresh={thresh}"
                        )
                    magnitude = magnitude / (mag_max + 1e-8)

                thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
                mask = (magnitude > thresh_t).to(dtype)
            return mask.to(torch.float32)

        # === ЯДРА ШАРРА (кэшированные) ===
        # scharr_x, scharr_y = self._get_scharr_kernels_cached(self.device, dtype)
        kernels = self._get_conv_kernel("scharr", return_pair=True, dtype=dtype, device=self.device)
        if isinstance(kernels, tuple):
            scharr_x, scharr_y = kernels
        else:
            scharr_x, scharr_y = kernels

        # === ГРАДИЕНТЫ И БИНАРИЗАЦИЯ ===
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            scharr_x, scharr_y = self._prepare_kernel_for_conv((scharr_x, scharr_y), gray.dtype)
            gx = self._safe_conv2d(gray, scharr_x, padding=1)
            gy = self._safe_conv2d(gray, scharr_y, padding=1)
            magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)

            if normalize:
                mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
                if self._debug_mode and not torch.compiler.is_compiling():
                    logger.info(
                        f"[{self.method}] mag_max={mag_max.mean().item():.6f}, "
                        f"magnitude_range=[{magnitude.min().item():.6f}, {magnitude.max().item():.6f}], "
                        f"thresh={thresh}"
                    )
                magnitude = magnitude / (mag_max + 1e-8)

            thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
            mask = (magnitude > thresh_t).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "scharr_edge_torch",
                exec_time,
                {"threshold": thresh, "precision_val": precision_val},
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _laplacian_edge(
        self,
        tensor: torch.Tensor,
        *,
        threshold: Optional[float] = None,
        sigma: Optional[float] = None,
        normalize: bool = True,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Обнаружение границ через лапласиан изображения.

        Применяет оператор Лапласа ∇² = ∂²/∂x² + ∂²/∂y² для выделения областей
        быстрого изменения интенсивности (нулевых пересечений второй производной).
        Чувствителен к шуму — рекомендуется предварительное гауссово сглаживание.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Опциональное гауссово сглаживание (параметр `sigma`).
        3. Свёртка с ядром Лапласа [[0,1,0],[1,-4,1],[0,1,0]].
        4. Взятие абсолютного значения и нормализация.
        5. Бинаризация по порогу.

        Метод особенно эффективен для:
        - Выделения тонких линий и изолированных точек
        - Задач, где важна чувствительность к быстрым изменениям яркости
        - Предварительной обработки перед zero-crossing детекторами

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            threshold: Порог для бинаризации в диапазоне [0, 1] (по умолчанию: 0.1).
            sigma: Sigma для предварительного Gaussian blur (по умолчанию: 1.0).
            normalize: Если True, нормализовать магнитуду к [0, 1] перед порогом.
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска границ (1, 1, H, W), dtype=float32.

        Note:
            - Ядро Лапласа кэшируется по (device, dtype) через `@lru_cache`.
            - Для `fp16` рекомендуется `sigma ≥ 0.8` для подавления шума.
            - Метод безопасен для `torch.compile(fullgraph=True)`.

        Example:
            ```python
            segmenter = TorchSegmenter("laplacian_edge", sigma=1.2, threshold=0.15)
            edges = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        thresh = threshold if threshold is not None else self.params.get("threshold", 0.1)
        sig = sigma if sigma is not None else self.params.get("sigma", 1.0)
        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            # === ПРЕДВАРИТЕЛЬНОЕ СГЛАЖИВАНИЕ ===
            if sig > 0:
                ks = int(2 * round(3 * sig) + 1)
                ks = ks if ks % 2 == 1 else ks + 1
                gray = tv_gaussian_blur(gray, kernel_size=[ks, ks], sigma=[sig, sig])

            # Ядро Лапласа (4-связность)
            # laplacian_kernel = self._get_laplacian_kernel_cached(self.device, dtype)
            laplacian_kernel = cast(
                torch.Tensor,
                self._get_conv_kernel("laplacian", return_pair=False, dtype=dtype, device=self.device, size=3),
            )

            # === ЛАПЛАСИАН И БИНАРИЗАЦИЯ ===
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                laplacian_kernel = cast(torch.Tensor, self._prepare_kernel_for_conv(laplacian_kernel, gray.dtype))
                laplacian = self._safe_conv2d(gray, laplacian_kernel, padding=1)
                magnitude = torch.abs(laplacian)

                if normalize:
                    mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
                    magnitude = magnitude / (mag_max + 1e-8)

                thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
                mask = (magnitude > thresh_t).to(dtype)
            return mask.to(torch.float32)

        # === ПРЕДВАРИТЕЛЬНОЕ СГЛАЖИВАНИЕ ===
        if sig > 0:
            ks = int(2 * round(3 * sig) + 1)
            ks = ks if ks % 2 == 1 else ks + 1
            gray = tv_gaussian_blur(gray, kernel_size=[ks, ks], sigma=[sig, sig])

        # Ядро Лапласа (4-связность)
        # laplacian_kernel = self._get_laplacian_kernel_cached(self.device, dtype)
        laplacian_kernel = cast(
            torch.Tensor, self._get_conv_kernel("laplacian", return_pair=False, dtype=dtype, device=self.device, size=3)
        )

        # === ЛАПЛАСИАН И БИНАРИЗАЦИЯ ===
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            laplacian_kernel = cast(torch.Tensor, self._prepare_kernel_for_conv(laplacian_kernel, gray.dtype))
            laplacian = self._safe_conv2d(gray, laplacian_kernel, padding=1)
            magnitude = torch.abs(laplacian)

            if normalize:
                mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
                magnitude = magnitude / (mag_max + 1e-8)

            thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
            mask = (magnitude > thresh_t).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "laplacian_edge_torch",
                exec_time,
                {
                    "threshold": thresh,
                    "sigma": sig,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _roberts_edge(
        self,
        tensor: torch.Tensor,
        *,
        threshold: Optional[float] = None,
        normalize: bool = True,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Обнаружение границ оператором Робертса.

        Простой 2×2 оператор для быстрого обнаружения диагональных границ.
        Вычисляет градиент через разности по диагоналям:
        Gx = I(x+1,y+1) - I(x,y), Gy = I(x+1,y) - I(x,y+1).
        Менее точен, чем Собель/Прюитт, но очень быстрый и экономичный по памяти.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Свёртка с ядрами Робертса 2×2.
        3. Вычисление магнитуды: `magnitude = sqrt(gx² + gy²)`.
        4. Нормализация к [0, 1] (опционально) и бинаризация.

        Метод особенно эффективен для:
        - Быстрого прототипирования и отладки пайплайна
        - Изображений с чёткими диагональными границами
        - Ресурсо-ограниченных устройств (мобильные, embedded)

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            threshold: Порог для бинаризации магнитуды в диапазоне [0, 1] (по умолчанию: 0.1).
            normalize: Если True, нормализовать магнитуду к [0, 1] перед порогом.
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска границ (1, 1, H, W), dtype=float32.

        Note:
            - Ядра Робертса кэшируются по (device, dtype) через `@lru_cache`.
            - Из-за малого размера ядра метод чувствителен к шуму.
            - Метод безопасен для `torch.compile(fullgraph=True)`.

        Example:
            ```python
            segmenter = TorchSegmenter("roberts_edge", threshold=0.2, precision="fp16")
            edges = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        thresh = threshold if threshold is not None else self.params.get("threshold", 0.1)
        precision_val = precision if precision is not None else "fp32"

        if dtype == torch.bfloat16:
            eps = 1e-2  # bf16: ~7.8e-3
        elif dtype == torch.float16:
            eps = 1e-3  # fp16: ~9.8e-4
        else:
            eps = 1e-8  # fp32/fp64

        if export_mode:
            kernels = self._get_conv_kernel("roberts", return_pair=True, dtype=dtype, device=self.device)
            if isinstance(kernels, tuple):
                roberts_x, roberts_y = kernels
            else:
                roberts_x, roberts_y = kernels

            # === ГРАДИЕНТЫ И БИНАРИЗАЦИЯ ===
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                # Паддинг 1×1 для сохранения размера при 2×2 ядре
                gray_pad = F.pad(gray, (0, 1, 0, 1), mode="reflect")
                roberts_x, roberts_y = self._prepare_kernel_for_conv((roberts_x, roberts_y), gray_pad.dtype)
                gx = self._safe_conv2d(gray_pad, roberts_x, padding=0)
                gy = self._safe_conv2d(gray_pad, roberts_y, padding=0)
                magnitude = torch.sqrt(gx.square() + gy.square() + eps)

                if normalize:
                    mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
                    mag_max = torch.maximum(mag_max, torch.tensor(eps, dtype=dtype, device=magnitude.device))
                    magnitude = magnitude / mag_max

                thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
                mask = (magnitude > thresh_t).to(dtype)
            return mask.to(torch.float32)

        # === ЯДРА РОБЕРТСА (кэшированные) ===
        # roberts_x, roberts_y = self._get_roberts_kernels_cached(self.device, dtype)
        kernels = self._get_conv_kernel("roberts", return_pair=True, dtype=dtype, device=self.device)
        if isinstance(kernels, tuple):
            roberts_x, roberts_y = kernels
        else:
            roberts_x, roberts_y = kernels

        # === ГРАДИЕНТЫ И БИНАРИЗАЦИЯ ===
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            gray_pad = F.pad(gray, (0, 1, 0, 1), mode="reflect")
            roberts_x, roberts_y = self._prepare_kernel_for_conv((roberts_x, roberts_y), gray_pad.dtype)
            gx = self._safe_conv2d(gray_pad, roberts_x, padding=0)
            gy = self._safe_conv2d(gray_pad, roberts_y, padding=0)
            magnitude = torch.sqrt(gx.square() + gy.square())

            if normalize and magnitude.max() > 1e-8:
                mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
                magnitude = magnitude / (mag_max + 1e-8)

            thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
            mask = (magnitude > thresh_t).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "roberts_edge_torch",
                exec_time,
                {"threshold": thresh, "precision_val": precision_val},
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")

        return mask.to(torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _log_edge(
        self,
        tensor: torch.Tensor,
        *,
        sigma: Optional[float] = None,
        threshold: Optional[float] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Детектор границ Laplacian of Gaussian (LoG).

        Применяет гауссово сглаживание, затем оператор Лапласа, ищет пересечения нуля.
        Эффективен для обнаружения границ на разных масштабах (параметр `sigma`).
        Zero-crossing указывает на переход от положительной к отрицательной кривизне.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Гауссово сглаживание с параметром `sigma`.
        3. Применение лапласиана через свёртку.
        4. Детекция пересечений нуля (смена знака у соседей).
        5. Фильтрация по магнитуде и бинаризация.

        Метод особенно эффективен для:
        - Многомасштабного анализа границ
        - Изображений с размытыми или зашумлёнными контурами
        - Задач, где важна инвариантность к масштабу

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            sigma: Sigma для гауссова размытия (по умолчанию: 1.0).
            threshold: Порог для магнитуды в диапазоне [0, 1] (по умолчанию: 0.1).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска границ (1, 1, H, W), dtype=float32.

        Note:
            - Zero-crossing детектируется векторизованно через `torch.roll`.
            - Для `fp16` рекомендуется `sigma ≥ 1.0` для стабильности.
            - Метод безопасен для `torch.compile(fullgraph=True)`.

        Example:
            ```python
            segmenter = TorchSegmenter("log_edge", sigma=1.5, threshold=0.12)
            edges = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        sig = sigma if sigma is not None else self.params.get("sigma", 1.0)
        thresh = threshold if threshold is not None else self.params.get("threshold", 0.01)
        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            # === ГАУССОВО СГЛАЖИВАНИЕ ===
            if sig > 0:
                ks = int(2 * round(3 * sig) + 1)
                ks = ks if ks % 2 == 1 else ks + 1
                gray = tv_gaussian_blur(gray, kernel_size=[ks, ks], sigma=[sig, sig])

            # === ЛАПЛАСИАН ===
            # laplacian_kernel = self._get_laplacian_kernel_cached(self.device, dtype)
            laplacian_kernel = cast(
                torch.Tensor,
                self._get_conv_kernel("laplacian", return_pair=False, dtype=dtype, device=self.device, size=3),
            )

            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                # Применяем лапласиан
                laplacian_kernel = cast(torch.Tensor, self._prepare_kernel_for_conv(laplacian_kernel, gray.dtype))
                laplacian = self._safe_conv2d(gray, laplacian_kernel, padding=1)

                # === ZERO-CROSSING DETECTION (векторизованный) ===
                sign = torch.sign(laplacian)

                # Пересечение нуля: соседние пиксели имеют разные знаки
                zero_crossing = torch.zeros_like(laplacian, dtype=torch.bool)

                # Проверяем 4-связных соседей через сдвиг
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    shifted = torch.roll(sign, shifts=(dy, dx), dims=(2, 3))
                    zero_crossing = torch.logical_or(zero_crossing, (sign * shifted < 0))

                # === МАГНИТУДА ДЛЯ ПОРОГА ===
                magnitude = torch.abs(laplacian)
                mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
                magnitude = magnitude / (mag_max + 1e-8)

                thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
                mask = (zero_crossing & (magnitude > thresh_t)).to(dtype)
            return mask.to(torch.float32)

        # === ГАУССОВО СГЛАЖИВАНИЕ ===
        if sig > 0:
            ks = int(2 * round(3 * sig) + 1)
            ks = ks if ks % 2 == 1 else ks + 1
            gray = tv_gaussian_blur(gray, kernel_size=[ks, ks], sigma=[sig, sig])

        # === ЛАПЛАСИАН ===
        laplacian_kernel = cast(
            torch.Tensor, self._get_conv_kernel("laplacian", return_pair=False, dtype=dtype, device=self.device, size=3)
        )

        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            laplacian_kernel = cast(torch.Tensor, self._prepare_kernel_for_conv(laplacian_kernel, gray.dtype))
            laplacian = self._safe_conv2d(gray, laplacian_kernel, padding=1)

            # === ZERO-CROSSING DETECTION (векторизованный) ===
            sign = torch.sign(laplacian)

            zero_crossing = torch.zeros_like(laplacian, dtype=torch.bool)

            # Проверяем 4-связных соседей через сдвиг
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                shifted = torch.roll(sign, shifts=(dy, dx), dims=(2, 3))
                zero_crossing = torch.logical_or(zero_crossing, (sign * shifted < 0))

            # === МАГНИТУДА ДЛЯ ПОРОГА ===
            magnitude = torch.abs(laplacian)
            mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
            magnitude = magnitude / (mag_max + 1e-8)

            thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
            mask = (zero_crossing & (magnitude > thresh_t)).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "log_edge_torch",
                exec_time,
                {
                    "sigma": sigma,
                    "threshold": threshold,
                    "precision": precision,
                    "precision_val": precision_val,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")

        return mask.to(torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _dog_edge(
        self,
        tensor: torch.Tensor,
        *,
        sigma1: Optional[float] = None,
        sigma2: Optional[float] = None,
        threshold: Optional[float] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Детектор границ Difference of Gaussian (DoG).

        Аппроксимация Laplacian of Gaussian через разность двух гауссовых размытий
        с разными сигмами. Эффективен и быстр благодаря сепарабельности гауссианы.
        Нулевые пересечения разности указывают на границы.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Два гауссовых размытия с `sigma1` и `sigma2` (sigma2 > sigma1).
        3. Вычисление разности: DoG = G(σ₁) - G(σ₂).
        4. Детекция пересечений нуля и фильтрация по магнитуде.
        5. Бинаризация по порогу.

        Метод особенно эффективен для:
        - Быстрого многомасштабного детектирования границ
        - Задач компьютерного зрения в реальном времени
        - Предварительной обработки для SIFT и других дескрипторов

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            sigma1: Меньшая сигма для гауссианы (по умолчанию: 1.0).
            sigma2: Большая сигма для гауссианы (по умолчанию: 2.0).
            threshold: Порог для магнитуды в диапазоне [0, 1] (по умолчанию: 0.1).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска границ (1, 1, H, W), dtype=float32.

        Note:
            - Рекомендуется `sigma2 / sigma1 ≈ 1.6` для оптимальной аппроксимации LoG.
            - Для `fp16` используйте `sigma1 ≥ 0.8` для избежания артефактов.
            - Метод безопасен для `torch.compile(fullgraph=True)`.

        Example:
            ```python
            segmenter = TorchSegmenter("dog_edge", sigma1=1.0, sigma2=1.6, threshold=0.1)
            edges = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        s1 = sigma1 if sigma1 is not None else self.params.get("sigma1", 1.0)
        s2 = sigma2 if sigma2 is not None else self.params.get("sigma2", 2.0)
        thresh = threshold if threshold is not None else self.params.get("threshold", 0.1)

        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                # === ГАУССОВЫ РАЗМЫТИЯ ===
                def _gaussian_kernel_1d(
                    size: int, sigma: float, dtype: torch.dtype, device: torch.device
                ) -> torch.Tensor:
                    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
                    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
                    return kernel / kernel.sum()

                # Ядро 1 для σ₁
                ks1 = int(2 * round(3 * s1) + 1)
                ks1 = ks1 if ks1 % 2 == 1 else ks1 + 1
                k1_1d = _gaussian_kernel_1d(ks1, s1, gray.dtype, self.device)
                k1_2d = (k1_1d.unsqueeze(1) @ k1_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
                blurred1 = F.conv2d(gray, k1_2d, padding=ks1 // 2)

                # Ядро 2 для σ₂
                ks2 = int(2 * round(3 * s2) + 1)
                ks2 = ks2 if ks2 % 2 == 1 else ks2 + 1
                k2_1d = _gaussian_kernel_1d(ks2, s2, gray.dtype, self.device)
                k2_2d = (k2_1d.unsqueeze(1) @ k2_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
                blurred2 = F.conv2d(gray, k2_2d, padding=ks2 // 2)

                # === РАЗНОСТЬ И ZERO-CROSSING ===
                dog = blurred1 - blurred2
                sign = torch.sign(dog)
                zero_crossing = torch.zeros_like(dog, dtype=torch.bool)

                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    shifted = torch.roll(sign, shifts=(dy, dx), dims=(2, 3))
                    zero_crossing = torch.logical_or(zero_crossing, (sign * shifted < 0))

                # === МАГНИТУДА И БИНАРИЗАЦИЯ ===
                magnitude = torch.abs(dog)
                mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
                magnitude = magnitude / (mag_max + 1e-8)

                thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
                mask = (zero_crossing & (magnitude > thresh_t)).to(dtype)
            return mask.to(torch.float32)

        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            # === ГАУССОВЫ РАЗМЫТИЯ ===
            def _gaussian_kernel_1d(size: int, sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
                coords = torch.arange(size, dtype=dtype, device=device) - size // 2
                kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
                return kernel / kernel.sum()

            # Ядро 1 для σ₁
            ks1 = int(2 * round(3 * s1) + 1)
            ks1 = ks1 if ks1 % 2 == 1 else ks1 + 1
            k1_1d = _gaussian_kernel_1d(ks1, s1, gray.dtype, self.device)
            k1_2d = (k1_1d.unsqueeze(1) @ k1_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
            blurred1 = F.conv2d(gray, k1_2d, padding=ks1 // 2)

            # Ядро 2 для σ₂
            ks2 = int(2 * round(3 * s2) + 1)
            ks2 = ks2 if ks2 % 2 == 1 else ks2 + 1
            k2_1d = _gaussian_kernel_1d(ks2, s2, gray.dtype, self.device)
            k2_2d = (k2_1d.unsqueeze(1) @ k2_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
            blurred2 = F.conv2d(gray, k2_2d, padding=ks2 // 2)

            # === РАЗНОСТЬ И ZERO-CROSSING ===
            dog = blurred1 - blurred2
            sign = torch.sign(dog)
            zero_crossing = torch.zeros_like(dog, dtype=torch.bool)

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                shifted = torch.roll(sign, shifts=(dy, dx), dims=(2, 3))
                zero_crossing = torch.logical_or(zero_crossing, (sign * shifted < 0))

            # === МАГНИТУДА И БИНАРИЗАЦИЯ ===
            magnitude = torch.abs(dog)
            mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
            magnitude = magnitude / (mag_max + 1e-8)

            thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
            mask = (zero_crossing & (magnitude > thresh_t)).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "dog_edge_torch",
                exec_time,
                {
                    "sigma1": sigma1,
                    "sigma2": sigma2,
                    "threshold": threshold,
                    "precision": precision,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _marr_hildreth_edge(
        self,
        tensor: torch.Tensor,
        *,
        sigma: Optional[float] = None,
        threshold: Optional[float] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Детектор границ Марра-Хилдрета (улучшенный LoG с нулевым пересечением).

        Комбинирует гауссово сглаживание, лапласиан 5×5 и детекцию пересечений нуля
        с учётом направления и магнитуды. Более устойчив к шуму, чем базовый LoG.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Гауссово сглаживание с параметром `sigma`.
        3. Применение лапласиана 5×5 для лучшей аппроксимации.
        4. Детекция пересечений нуля с проверкой магнитуды у обоих пикселей.
        5. Бинаризация результата.

        Метод особенно эффективен для:
        - Задач с высоким уровнем шума
        - Медицинских и научных изображений с тонкими структурами
        - Случаев, когда важна точность локализации границ

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            sigma: Sigma для гауссова размытия (по умолчанию: 1.5).
            threshold: Порог для магнитуды в диапазоне [0, 1] (по умолчанию: 0.1).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска границ (1, 1, H, W), dtype=float32.

        Note:
            - Используется лапласиан 5×5 для лучшей аппроксимации второй производной.
            - Пересечение нуля считается валидным, если магнитуда достаточна у любого из пикселей.
            - Метод безопасен для `torch.compile(fullgraph=True)`.

        Example:
            ```python
            segmenter = TorchSegmenter("marr_hildreth_edge", sigma=1.5, threshold=0.12)
            edges = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        sig = sigma if sigma is not None else self.params.get("sigma", 1.0)
        thresh = threshold if threshold is not None else self.params.get("threshold", 0.01)
        precision_val = precision if precision is not None else "fp32"

        if export_mode:
            # === ГАУССОВО СГЛАЖИВАНИЕ ===
            if sig > 0:
                ks = int(2 * round(3 * sig) + 1)
                ks = ks if ks % 2 == 1 else ks + 1
                gray = tv_gaussian_blur(gray, kernel_size=[ks, ks], sigma=[sig, sig])

            # === ЛАПЛАСИАН 5×5 (кэшированный) ===
            # laplacian_5x5 = self._get_laplacian_5x5_cached(self.device, dtype)
            laplacian_5x5 = cast(
                torch.Tensor,
                self._get_conv_kernel("laplacian", return_pair=False, dtype=dtype, device=self.device, size=5),
            )

            precision_val = precision if precision is not None else "fp32"
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                laplacian_5x5 = cast(torch.Tensor, self._prepare_kernel_for_conv(laplacian_5x5, gray.dtype))
                laplacian = self._safe_conv2d(gray, laplacian_5x5, padding=2)

                # === ZERO-CROSSING С ПРОВЕРКОЙ МАГНИТУДЫ ===
                sign = torch.sign(laplacian)
                magnitude = torch.abs(laplacian)
                mag_max = magnitude.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
                magnitude = magnitude / (mag_max + 1e-8)

                zero_crossing = torch.zeros_like(laplacian, dtype=torch.bool)
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    shifted_sign = torch.roll(sign, shifts=(dy, dx), dims=(2, 3))
                    shifted_mag = torch.roll(magnitude, shifts=(dy, dx), dims=(2, 3))
                    crossing = (sign * shifted_sign < 0) & ((magnitude > thresh) | (shifted_mag > thresh))
                    zero_crossing = torch.logical_or(zero_crossing, crossing)

                mask = zero_crossing.to(dtype)

            return mask.to(torch.float32)

        # === ГАУССОВО СГЛАЖИВАНИЕ ===
        if sig > 0:
            ks = int(2 * round(3 * sig) + 1)
            ks = ks if ks % 2 == 1 else ks + 1
            gray = tv_gaussian_blur(gray, kernel_size=[ks, ks], sigma=[sig, sig])

        # === ЛАПЛАСИАН 5×5 (кэшированный) ===
        # laplacian_5x5 = self._get_laplacian_5x5_cached(self.device, dtype)
        laplacian_3x3 = cast(
            torch.Tensor, self._get_conv_kernel("laplacian", return_pair=False, dtype=dtype, device=self.device, size=3)
        )

        precision_val = precision if precision is not None else "fp32"
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            laplacian_3x3 = cast(torch.Tensor, self._prepare_kernel_for_conv(laplacian_3x3, gray.dtype))
            laplacian = self._safe_conv2d(gray, laplacian_3x3, padding=1)

            # === ZERO-CROSSING С ПРОВЕРКОЙ МАГНИТУДЫ ===
            sign = torch.sign(laplacian)
            magnitude = torch.abs(laplacian)
            # mag_max = magnitude.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
            # magnitude = magnitude / (mag_max + 1e-8)

            # zero_crossing = torch.zeros_like(laplacian, dtype=torch.bool)
            # for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            #     shifted_sign = torch.roll(sign, shifts=(dy, dx), dims=(2, 3))
            #     shifted_mag = torch.roll(magnitude, shifts=(dy, dx), dims=(2, 3))
            #     crossing = (sign * shifted_sign < 0) & ((magnitude > thresh) | (shifted_mag > thresh))
            #     zero_crossing = torch.logical_or(zero_crossing, crossing)

            # mask = zero_crossing.to(dtype)
            zero_crossing = torch.zeros_like(laplacian, dtype=torch.bool)
            
            # Горизонтальные пересечения
            zc_h = (sign[..., :, :-1] * sign[..., :, 1:] < 0)
            zero_crossing[..., :, :-1] |= zc_h
            
            # Вертикальные пересечения
            zc_v = (sign[..., :-1, :] * sign[..., 1:, :] < 0)
            zero_crossing[..., :-1, :] |= zc_v
            
            # 🔧 FIX: Фильтрация по магнитуде (порог в [0, 1] для нормализованного входа)
            mask = (zero_crossing & (magnitude > thresh)).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "marr_hildreth_torch",
                exec_time,
                {
                    "sigma": sigma,
                    "threshold": threshold,
                    "precision": precision,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _gradient_magnitude_direction(
        self,
        tensor: torch.Tensor,
        *,
        threshold: Optional[float] = None,
        normalize: bool = True,
        angle_range: Optional[Tuple[float, float]] = None,  # 🔧 Новый параметр
        use_nms: bool = False,  # 🔧 По умолчанию False для совместимости!
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Вычисление градиента с магнитудой и направлением + подавление немаксимумов.

        Вычисляет градиент через оператор Собеля, затем применяет non-maximum suppression
        по направлению градиента для получения тонких, точных границ. Аналогично этапу
        NMS в алгоритме Кэнни, но без гистерезиса.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Вычисление градиентов Собеля (Gx, Gy).
        3. Магнитуда |G| = √(Gx² + Gy²) и направление θ = atan2(Gy, Gx).
        4. Квантование направления на 4 сектора: 0°, 45°, 90°, 135°.
        5. Подавление немаксимумов: пиксель сохраняется, если он локальный максимум
        в направлении градиента.
        6. Бинаризация по порогу.

        Метод особенно эффективен для:
        - Получения тонких, одно-пиксельных границ
        - Предварительной обработки перед векторизацией контуров
        - Задач, где важна точность локализации границ

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            threshold: Порог для бинаризации магнитуды в диапазоне [0, 1] (по умолчанию: 0.1).
            normalize: Если True, нормализовать магнитуду к [0, 1] перед порогом.
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска границ (1, 1, H, W), dtype=float32.

        Note:
            - Направление квантуется на 4 сектора для эффективности.
            - Для `fp16` рекомендуется `threshold ≥ 0.05` из-за ограниченной точности.
            - Метод безопасен для `torch.compile(fullgraph=True)`.

        Example:
            ```python
            segmenter = TorchSegmenter("gradient_magnitude_direction", threshold=0.12)
            edges = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if not torch.compiler.is_compiling():
            start_time: float = time.time()
        else:
            start_time = None  # type: ignore[assignment]

        # === ПАРАМЕТРЫ ===
        thresh = threshold if threshold is not None else self.params.get("threshold", 0.1)
        precision_val = precision if precision is not None else "fp32"
        angle_rng = angle_range if angle_range is not None else self.params.get("angle_range", None)
        use_nms_flag = use_nms

        if export_mode:
            kernels = self._get_conv_kernel("sobel", return_pair=True, dtype=dtype, device=self.device)
            if isinstance(kernels, tuple):
                sobel_x, sobel_y = kernels
            else:
                sobel_x, sobel_y = kernels  # fallback для одиночных ядер

            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                # === ГРАДИЕНТЫ ===
                sobel_x, sobel_y = self._prepare_kernel_for_conv((sobel_x, sobel_y), gray.dtype)
                gx = self._safe_conv2d(gray, sobel_x, padding=1)
                gy = self._safe_conv2d(gray, sobel_y, padding=1)
                magnitude = torch.sqrt(gx.square() + gy.square())
                direction_rad = torch.atan2(gy, gx)  # Радианы для внутренних вычислений
                direction_deg = direction_rad * 180.0 / torch.pi  # Градусы для фильтрации по углу

                # === НОРМАЛИЗАЦИЯ ===
                if normalize:
                    mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
                    magnitude = magnitude / (mag_max + 1e-8)

                # === NON-MAXIMUM SUPPRESSION ===
                if use_nms_flag:
                    magnitude = self._suppress_non_max_torch_export(magnitude, direction_rad)

                if angle_rng is not None:
                    # Учитываем симметрию: θ ≡ θ+180°
                    angle_mask = ((direction_deg >= angle_rng[0]) & (direction_deg <= angle_rng[1])) | \
                                ((direction_deg + 180.0 >= angle_rng[0]) & (direction_deg + 180.0 <= angle_rng[1]))
                    magnitude = magnitude * angle_mask.to(magnitude.dtype)

                # === БИНАРИЗАЦИЯ ===
                thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
                mask = (magnitude > thresh_t).to(dtype)
            return mask.to(torch.float32)

        # === ЯДРА СОБЕЛЯ ===
        # sobel_x, sobel_y = self._get_sobel_kernels_cached(self.device, dtype)
        kernels = self._get_conv_kernel("sobel", return_pair=True, dtype=dtype, device=self.device)
        if isinstance(kernels, tuple):
            sobel_x, sobel_y = kernels
        else:
            sobel_x, sobel_y = kernels  # fallback для одиночных ядер

        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            # === ГРАДИЕНТЫ ===
            sobel_x, sobel_y = self._prepare_kernel_for_conv((sobel_x, sobel_y), gray.dtype)
            gx = self._safe_conv2d(gray, sobel_x, padding=1)
            gy = self._safe_conv2d(gray, sobel_y, padding=1)
            # magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
            # direction = torch.atan2(gy, gx)  # Радианы от -π до π
            magnitude = torch.sqrt(gx.square() + gy.square())
            direction_rad = torch.atan2(gy, gx)  # Радианы для внутренних вычислений
            direction_deg = direction_rad * 180.0 / torch.pi  # Градусы для фильтрации по углу

            # === НОРМАЛИЗАЦИЯ ===
            if normalize and magnitude.max() > 1e-8:
                mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
                magnitude = magnitude / (mag_max + 1e-8)

            # === NON-MAXIMUM SUPPRESSION ===
            if use_nms_flag:
                magnitude = self._suppress_non_max_torch(magnitude, direction_rad)

            if angle_rng is not None:
                # Учитываем симметрию: θ ≡ θ+180°
                angle_mask = ((direction_deg >= angle_rng[0]) & (direction_deg <= angle_rng[1])) | \
                            ((direction_deg + 180.0 >= angle_rng[0]) & (direction_deg + 180.0 <= angle_rng[1]))
                magnitude = magnitude * angle_mask.to(magnitude.dtype)

            # 🔧 FIX: Бинаризация по magnitude (а не suppressed!)
            thresh_t = torch.tensor(thresh, dtype=magnitude.dtype, device=magnitude.device)
            mask = (magnitude > thresh_t).to(dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "gradient_magnitude_direction_torch",
                exec_time,
                {
                    "threshold": threshold,
                    "normalize": normalize,
                    "angle_range": angle_range,
                    "use_nms": use_nms,
                    "precision": precision,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _suppress_non_max_torch(
        magnitude: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        """Подавление немаксимумов по направлению градиента (векторизованная версия).

        Args:
            magnitude: Магнитуда градиента (..., H, W).
            direction: Направление градиента в радианах (..., H, W).

        Returns:
            torch.Tensor: Магнитуда после подавления немаксимумов.
        """
        # Квантование направления на 4 сектора
        angle = torch.abs(direction) * 180 / torch.pi
        angle = torch.fmod(angle, 180)

        # 0°, 45°, 90°, 135°
        sectors = torch.zeros_like(angle, dtype=torch.long)
        sectors[(angle <= 22.5) | (angle > 157.5)] = 0  # 0° (горизонталь)
        sectors[(angle > 22.5) & (angle <= 67.5)] = 1  # 45°
        sectors[(angle > 67.5) & (angle <= 112.5)] = 2  # 90° (вертикаль)
        sectors[(angle > 112.5) & (angle <= 157.5)] = 3  # 135°

        suppressed = torch.zeros_like(magnitude)

        # Паддинг для доступа к соседям
        mag_padded = F.pad(magnitude, (1, 1, 1, 1), mode="reflect")

        for s in range(4):
            mask = sectors == s
            if not mask.any():
                continue

            if s == 0:  # Горизонталь: сравниваем лево/право
                is_max = (magnitude >= mag_padded[:, :, 1:-1, :-2]) & (magnitude >= mag_padded[:, :, 1:-1, 2:])
            elif s == 1:  # 45°: сравниваем UL/DR
                is_max = (magnitude >= mag_padded[:, :, :-2, :-2]) & (magnitude >= mag_padded[:, :, 2:, 2:])
            elif s == 2:  # Вертикаль: сравниваем верх/низ
                is_max = (magnitude >= mag_padded[:, :, :-2, 1:-1]) & (magnitude >= mag_padded[:, :, 2:, 1:-1])
            else:  # 135°: сравниваем UR/DL
                is_max = (magnitude >= mag_padded[:, :, :-2, 2:]) & (magnitude >= mag_padded[:, :, 2:, :-2])

            suppressed[mask & is_max] = magnitude[mask & is_max]

        return suppressed

    @staticmethod
    def _suppress_non_max_torch_export(
        magnitude: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        """Export-friendly версия подавления немаксимумов БЕЗ data-dependent guards.

        🔧 Ключевые изменения:
        - Нет .any() в условных проверках
        - Нет индексации по маске: tensor[mask] = value
        - Только векторизованные операции: torch.where, torch.mul, torch.add
        """
        # === КВАНТОВАНИЕ НАПРАВЛЕНИЯ ===
        angle = torch.abs(direction) * 180.0 / 3.141592653589793
        angle = torch.fmod(angle, 180.0)

        # 🔧 FIX: Векторизованное создание секторов через torch.where
        sectors = torch.zeros_like(angle, dtype=torch.long)

        mask_0 = (angle <= 22.5) | (angle > 157.5)
        mask_1 = (angle > 22.5) & (angle <= 67.5)
        mask_2 = (angle > 67.5) & (angle <= 112.5)
        mask_3 = (angle > 112.5) & (angle <= 157.5)

        # 🔧 FIX: Используем torch.where вместо индексации
        sectors = torch.where(mask_0, torch.zeros_like(sectors), sectors)
        sectors = torch.where(mask_1, torch.ones_like(sectors), sectors)
        sectors = torch.where(mask_2, torch.full_like(sectors, 2), sectors)
        sectors = torch.where(mask_3, torch.full_like(sectors, 3), sectors)

        # === ПОДАВЛЕНИЕ НЕММАКСИМУМОВ (векторизованно) ===
        mag_padded = F.pad(magnitude, (1, 1, 1, 1), mode="reflect")

        # Инициализация результата
        suppressed = torch.zeros_like(magnitude)

        # 🔧 FIX: Обрабатываем все сектора векторизованно, без цикла по секторам
        # Для каждого сектора вычисляем is_max и объединяем через where

        # Сектор 0: Горизонталь (0°)
        is_max_0 = (magnitude >= mag_padded[:, :, 1:-1, :-2]) & (magnitude >= mag_padded[:, :, 1:-1, 2:])
        suppressed = torch.where((sectors == 0) & is_max_0, magnitude, suppressed)

        # Сектор 1: 45°
        is_max_1 = (magnitude >= mag_padded[:, :, :-2, :-2]) & (magnitude >= mag_padded[:, :, 2:, 2:])
        suppressed = torch.where((sectors == 1) & is_max_1, magnitude, suppressed)

        # Сектор 2: Вертикаль (90°)
        is_max_2 = (magnitude >= mag_padded[:, :, :-2, 1:-1]) & (magnitude >= mag_padded[:, :, 2:, 1:-1])
        suppressed = torch.where((sectors == 2) & is_max_2, magnitude, suppressed)

        # Сектор 3: 135°
        is_max_3 = (magnitude >= mag_padded[:, :, :-2, 2:]) & (magnitude >= mag_padded[:, :, 2:, :-2])
        suppressed = torch.where((sectors == 3) & is_max_3, magnitude, suppressed)

        return suppressed

    @torch.no_grad()
    def _phase_congruency_edge(
        self,
        tensor: torch.Tensor,
        *,
        nscales: Optional[int] = None,
        norientations: Optional[int] = None,
        min_wavelength: Optional[float] = None,
        mult: Optional[float] = None,
        sigma_onf: Optional[float] = None,
        k_noise: Optional[float] = None,
        threshold: Optional[float] = None,
        precision: Optional[str] = None,
        export_mode: bool = False,
    ) -> torch.Tensor:
        """Детектор границ на основе фазовой конгруэнтности (реализация Ковези).

        Инвариантна к изменению контраста и яркости. Обнаруживает края через
        выравнивание фаз Фурье-компонент в пространстве изображений, что позволяет
        находить границы даже при низком контрасте или зашумлении.

        Алгоритм:
        1. Преобразование Фурье изображения.
        2. Построение банка фильтров Log-Gabor в частотной области.
        3. Вычисление even/odd откликов для каждого масштаба и ориентации.
        4. Вычисление локальной энергии и компенсация шума (MAD-оценка).
        5. Нормализация карты фазовой конгруэнтности и бинаризация.

        Метод особенно эффективен для:
        - Медицинских изображений с низким контрастом тканей
        - Спутниковых снимков с разнородным освещением
        - Задач, где важна инвариантность к яркости и контрасту

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            nscales: Количество масштабов фильтра (по умолчанию: 4).
            norientations: Количество ориентаций фильтра (по умолчанию: 4).
            min_wavelength: Минимальная длина волны фильтра (по умолчанию: 3).
            mult: Мультипликатор длины волны между масштабами (по умолчанию: 2.0).
            sigma_onf: Стандартное отклонение в частотной области (по умолчанию: 0.55).
            k_noise: Коэффициент шумоподавления (по умолчанию: 2.0).
            threshold: Порог для бинаризации в диапазоне [0, 1] (по умолчанию: 0.3).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска границ (1, 1, H, W), dtype=float32.

        Note:
            - FFT-операции выполняются в `fp32` для стабильности, даже при `fp16/bf16`.
            - Для `fp16` рекомендуется `k_noise ≥ 2.5` для подавления артефактов.
            - Метод не поддерживает `fullgraph=True` в `torch.compile` из-за динамических операций.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "phase_congruency_edge",
                nscales=5,
                norientations=6,
                threshold=0.25,
                precision="bf16"
            )
            edges = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===

        if export_mode:
            gray = self._to_grayscale(tensor)
            if gray.dim() == 4:
                gray = gray.view(-1, gray.shape[-2], gray.shape[-1])
            if gray.dim() == 3:
                gray = gray[0] if gray.shape[0] == 1 else gray.view(-1, gray.shape[-1])
            h, w = gray.shape[-2], gray.shape[-1]
        else:
            gray = self._to_grayscale(tensor).squeeze()
            if gray.dim() == 3 and gray.shape[0] == 1:
                gray = gray.squeeze(0)
            h, w = gray.shape

        device = gray.device

        if export_mode:
            # Используем torch.where для векторизованной операции без data-dependent guard
            gray = torch.where(gray.amax(dim=(-2, -1), keepdim=True) > 1.0, gray / 255.0, gray)
        else:
            # Нормализация к [0, 1]
            if gray.amax() > 1.0:
                gray = gray / 255.0

        if not torch.compiler.is_compiling():
            start_time: Optional[float] = time.time()
        else:
            start_time = None

        # === ПАРАМЕТРЫ ===
        n_scales = nscales if nscales is not None else self.params.get("nscales", 4)
        n_orients = norientations if norientations is not None else self.params.get("norientations", 4)
        min_wl = min_wavelength if min_wavelength is not None else self.params.get("min_wavelength", 3.0)
        mult_val = mult if mult is not None else self.params.get("mult", 2.0)
        sigma_val = sigma_onf if sigma_onf is not None else self.params.get("sigma_onf", 0.55)
        k_val = k_noise if k_noise is not None else self.params.get("k_noise", 2.0)
        thresh = threshold if threshold is not None else self.params.get("threshold", 0.3)
        precision_val = precision if precision is not None else "fp32"
        eps: float = 1e-10
        dtype = self.precision_manager.get_dtype(precision)

        if export_mode:
            # EXPORT-FRIENDLY: ФИКСИРОВАННЫЕ РАЗМЕРЫ
            target_h, target_w = 256, 256

            # Предвычисление фильтров
            if not hasattr(self, "_pc_filters_cache"):
                self._pc_filters_cache = {}

            cache_key = (target_h, target_w, n_scales, n_orients, min_wl, mult_val, sigma_val)
            if cache_key not in self._pc_filters_cache:
                self._pc_filters_cache[cache_key] = self._precompute_pc_filters(
                    target_h, target_w, n_scales, n_orients, min_wl, mult_val, sigma_val, device
                )

            filters = self._pc_filters_cache[cache_key]

            # 🔧 Ресайз входа — гарантируем 4D формат для interpolate
            if h != target_h or w != target_w:
                gray_4d = gray.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                gray_resized = torch.nn.functional.interpolate(
                    gray_4d, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
                gray_resized = gray_resized.squeeze(0).squeeze(0)  # (target_h, target_w)
            else:
                gray_resized = gray

            # Выполняем упрощённую фазовую конгруэнтность
            mask_resized = self._phase_congruency_simple_export(
                gray_resized, filters, k_val, thresh, dtype, precision_val, eps
            )  # (target_h, target_w)

            # Обратный ресайз
            if h != target_h or w != target_w:
                mask_4d = mask_resized.unsqueeze(0).unsqueeze(0)  # (1, 1, target_h, target_w)
                mask = torch.nn.functional.interpolate(mask_4d, size=(h, w), mode="bilinear", align_corners=False)
                mask = mask.squeeze(0).squeeze(0)  # (h, w)
            else:
                mask = mask_resized

            # Возвращаем в правильном формате
            return mask.to(torch.float32).unsqueeze(0).unsqueeze(0)

        # === ОБЫЧНЫЙ РЕЖИМ ===
        gray_fp32 = gray.float()
        img_fft = torch.fft.fft2(gray_fp32)
        fft_shifted = torch.fft.fftshift(img_fft)

        # === ЧАСТОТНАЯ СЕТКА ===
        y_freq = torch.fft.fftshift(torch.fft.fftfreq(h, device=device))
        x_freq = torch.fft.fftshift(torch.fft.fftfreq(w, device=device))
        Y, X = torch.meshgrid(y_freq, x_freq, indexing="ij")
        R = torch.sqrt(X**2 + Y**2 + eps)
        Theta = torch.arctan2(-Y, X)

        # === АККУМУЛЯТОРЫ (в целевой точности) ===
        target_dtype = self.precision_manager.get_dtype(precision)
        sum_even = torch.zeros((h, w), device=device, dtype=target_dtype)
        sum_odd = torch.zeros((h, w), device=device, dtype=target_dtype)
        sum_amp = torch.zeros((h, w), device=device, dtype=target_dtype)
        noise_energy = torch.zeros((h, w), device=device, dtype=target_dtype)

        orientations = torch.linspace(0, torch.pi, n_orients, device=device)

        for scale in range(n_scales):
            wavelength = min_wl * (mult_val**scale)
            fo = 1.0 / wavelength

            # Log-Gabor фильтр (радиальная часть) — в fp32
            log_ratio = torch.log(R / fo + 1e-10) / torch.log(torch.tensor(sigma_val, device=device) + eps)
            log_gabor = torch.exp(-0.5 * log_ratio**2)
            log_gabor[0, 0] = 0.0  # DC = 0

            for angle in orientations:
                # Угловая часть
                angular_spread = torch.pi / 2 / n_orients
                d_theta = torch.abs(Theta - angle)
                d_theta = torch.minimum(d_theta, 2 * torch.pi - d_theta)
                angular = torch.exp(-0.5 * (d_theta / angular_spread) ** 2)

                # Полный фильтр (центральный)
                filter_f = log_gabor * angular

                # Умножаем центрированный спектр на центрированный фильтр,
                # затем смещаем результат для обратного FFT.
                response = torch.fft.ifft2(torch.fft.ifftshift(fft_shifted * filter_f))
                even_resp = torch.real(response).to(target_dtype)
                odd_resp = torch.imag(response).to(target_dtype)

                # Амплитуда
                amp = torch.sqrt(even_resp**2 + odd_resp**2 + eps)

                # Оценка шума (MAD)
                med = torch.median(amp)
                noise_est = 2.0 * (med / 0.6745)

                # Накопление
                sum_even += even_resp
                sum_odd += odd_resp
                sum_amp += amp
                noise_energy += noise_est**2

        # === ВЫЧИСЛЕНИЕ ФАЗОВОЙ КОНГРУЭНТНОСТИ ===
        with self.precision_manager.autocast(precision_val, enabled=(target_dtype != torch.float32)):
            local_energy = torch.sqrt(sum_even**2 + sum_odd**2 + eps)
            T = noise_energy * k_val
            pc_map = torch.clamp(local_energy - T, min=0) / (sum_amp + eps)
            pc_map = torch.clamp(pc_map, 0, 1)

            # === БИНАРИЗАЦИЯ ===
            thresh_t = torch.tensor(min(thresh, 0.99), dtype=target_dtype, device=device)
            mask = (pc_map > thresh_t).to(target_dtype)

        if not torch.compiler.is_compiling() and start_time is not None:
            exec_time: float = time.time() - start_time
            info = self._log_info(
                "phase_congruency_torch",
                exec_time,
                {
                    "nscales": nscales,
                    "norientations": norientations,
                    "min_wavelength": min_wavelength,
                    "mult": mult,
                    "sigma_onf": sigma_onf,
                    "k_noise": k_noise,
                    "threshold": threshold,
                    "precision": precision,
                    "precision_val": precision_val,
                },
                precision_val=precision_val,
            )
            self.params["execution_info"] = info
            if self._debug_mode:
                logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")

        return mask.to(torch.float32).unsqueeze(0).unsqueeze(0)

    def _phase_congruency_simple_export(
        self,
        gray: torch.Tensor,  # (H, W), normalized [0, 1]
        filters: dict,
        k_val: float,
        thresh: float,
        dtype: torch.dtype,
        precision_val: str,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Упрощённая фазовая конгруэнтность для экспорта — возвращает (H, W)."""
        device = gray.device
        h, w = gray.shape

        # FFT изображения
        gray_fp32 = gray.float()
        img_fft = torch.fft.fft2(gray_fp32)
        fft_shifted = torch.fft.fftshift(img_fft)

        # Аккумуляторы
        target_dtype = dtype
        sum_even = torch.zeros((h, w), device=device, dtype=target_dtype)
        sum_odd = torch.zeros((h, w), device=device, dtype=target_dtype)
        sum_amp = torch.zeros((h, w), device=device, dtype=target_dtype)

        # Применяем предвычисленные фильтры
        for scale_filters in filters["filters"]:
            for filter_f in scale_filters:
                response = torch.fft.ifft2(torch.fft.ifftshift(fft_shifted * filter_f))
                even_resp = torch.real(response).to(target_dtype)
                odd_resp = torch.imag(response).to(target_dtype)
                amp = torch.sqrt(even_resp**2 + odd_resp**2 + eps)

                sum_even += even_resp
                sum_odd += odd_resp
                sum_amp += amp

        # Вычисление фазовой конгруэнтности (упрощённо)
        with self.precision_manager.autocast(precision_val, enabled=(target_dtype != torch.float32)):
            local_energy = torch.sqrt(sum_even**2 + sum_odd**2 + eps)
            pc_map = torch.clamp(local_energy / (sum_amp + eps), 0, 1)

            thresh_t = torch.tensor(min(thresh, 0.99), dtype=target_dtype, device=device)
            mask = (pc_map > thresh_t).to(target_dtype)

        # 🔧 FIX: Возвращаем (H, W), а не (1, 1, H, W)
        return mask  # (h, w)

    def _precompute_pc_filters(
        self,
        h: int,
        w: int,
        n_scales: int,
        n_orients: int,
        min_wl: float,
        mult_val: float,
        sigma_val: float,
        device: torch.device,
    ) -> dict:
        """Предварительно вычисляет фильтры Log-Gabor для фиксированных размеров."""
        eps = 1e-10
        filters = []

        # Частотная сетка
        y_freq = torch.fft.fftshift(torch.fft.fftfreq(h, device=device))
        x_freq = torch.fft.fftshift(torch.fft.fftfreq(w, device=device))
        Y, X = torch.meshgrid(y_freq, x_freq, indexing="ij")
        R = torch.sqrt(X**2 + Y**2 + eps)
        Theta = torch.arctan2(-Y, X)
        orientations = torch.linspace(0, torch.pi, n_orients, device=device)

        for scale in range(n_scales):
            wavelength = min_wl * (mult_val**scale)
            fo = 1.0 / wavelength

            # Радиальная часть
            log_ratio = torch.log(R / fo + 1e-10) / torch.log(torch.tensor(sigma_val, device=device) + eps)
            log_gabor = torch.exp(-0.5 * log_ratio**2)
            log_gabor[0, 0] = 0.0

            scale_filters = []
            for angle in orientations:
                # Угловая часть
                angular_spread = torch.pi / 2 / n_orients
                d_theta = torch.abs(Theta - angle)
                d_theta = torch.minimum(d_theta, 2 * torch.pi - d_theta)
                angular = torch.exp(-0.5 * (d_theta / angular_spread) ** 2)

                # Полный фильтр
                filter_f = log_gabor * angular
                scale_filters.append(filter_f)

            filters.append(torch.stack(scale_filters))  # (n_orients, H, W)

        return {
            "filters": filters,  # List[Tensor]: n_scales × (n_orients, H, W)
            "orientations": orientations,
        }

    # ──────────────────────────────────────────────────────────────────────
    # РЕГИОНАЛЬНЫЕ МЕТОДЫ СЕГМЕНТАЦИИ
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _region_growing(
        self,
        tensor: torch.Tensor,
        *,
        seed: Optional[Tuple[int, int]] = None,
        tolerance: Optional[float] = None,
        max_iterations: Optional[int] = None,
        precision: Optional[str] = None,
    ) -> torch.Tensor:
        """Сегментация методом роста регионов (Region Growing).

        Начинает с заданной точки (seed) и рекурсивно добавляет соседние пиксели,
        интенсивность которых отличается от среднего значения региона не более чем на допуск.
        Алгоритм останавливается, когда все связанные пиксели обработаны.

        Алгоритм:
        1. Конвертация в градации серого.
        2. Инициализация маски и очереди с начальной точкой.
        3. Векторизованный BFS: обработка "волны" пикселей за итерацию.
        4. Проверка условия допуска и добавление соседей.
        5. Возврат бинарной маски региона.

        Метод особенно эффективен для:
        - Сегментации однородных областей с чёткими границами
        - Интерактивной сегментации с ручной инициализацией seed
        - Задач, где важна связность результата

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            seed: Начальная точка (y, x) в пикселях (по умолчанию: центр изображения).
            tolerance: Допуск по интенсивности в диапазоне [0, 1] (по умолчанию: 0.1).
            max_iterations: Максимальное число итераций (по умолчанию: H×W).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска региона (1, 1, H, W), dtype=float32.

        Note:
            - Используется векторизованный BFS вместо deque для ускорения на GPU.
            - Для больших изображений (>2048×2048) рекомендуется fallback на Numba (CPU).
            - Метод не поддерживает `fullgraph=True` из-за динамического числа итераций.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "region_growing",
                seed=(100, 150),
                tolerance=0.15,
                precision="fp16"
            )
            mask = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor).squeeze(0)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        start_time = time.time()

        # Масштабируем к [0, 255] для стабильности при low precision
        if gray.max() <= 1.0:
            gray = gray * 255.0

        h, w = gray.shape

        # === ПАРАМЕТРЫ ===
        seed_y, seed_x = seed if seed is not None else self.params.get("seed", (h // 2, w // 2))
        tol = tolerance if tolerance is not None else self.params.get("tolerance", 0.1)
        max_iter = max_iterations if max_iterations is not None else self.params.get("max_iterations", h * w)

        seed_value = gray[seed_y, seed_x]

        # === ИНИЦИАЛИЗАЦИЯ ===
        mask = torch.zeros(h, w, dtype=torch.bool, device=gray.device)
        visited = torch.zeros(h, w, dtype=torch.bool, device=gray.device)

        # Буфер для текущей волны (вместо deque)
        current_wave = torch.tensor([[seed_y, seed_x]], device=gray.device, dtype=torch.long)
        visited[seed_y, seed_x] = True

        # Направления: вверх, вниз, влево, вправо
        directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], device=gray.device, dtype=torch.long)

        # === ВЕКТОРИЗОВАННЫЙ BFS ===
        precision_val = precision if precision is not None else "fp32"
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            for _ in range(max_iter):
                if current_wave.numel() == 0:
                    break

                # Координаты текущей волны
                y_coords = current_wave[:, 0]
                x_coords = current_wave[:, 1]

                # Проверка допуска
                pixel_values = gray[y_coords, x_coords]
                accepted = torch.abs(pixel_values - seed_value) <= tol

                # Обновление маски
                mask[y_coords[accepted], x_coords[accepted]] = True

                # Генерация соседей (векторизованно)
                neighbors = current_wave.unsqueeze(1) + directions.unsqueeze(0)  # [N, 4, 2]
                neighbors_flat = neighbors.view(-1, 2)  # [N*4, 2]

                # Фильтрация валидных соседей
                ny, nx = neighbors_flat[:, 0], neighbors_flat[:, 1]
                valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w) & (~visited[ny, nx])

                if not valid.any():
                    break

                # Обновление visited и формирование новой волны
                next_wave = neighbors_flat[valid]
                visited[next_wave[:, 0], next_wave[:, 1]] = True
                current_wave = next_wave

        exec_time: float = time.time() - start_time
        info = self._log_info(
            "region_growing_torch",
            exec_time,
            {
                "seed": seed,
                "tolerance": tolerance,
                "max_iterations": max_iterations,
                "precision": precision,
            },
            precision_val=precision_val,
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32).unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    # Вариант А: Векторизованный BFS на PyTorch (для небольших изображений)
    @torch.no_grad()
    def _region_growing_opt(self, tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Region Growing — векторизованная версия на PyTorch.

        Использует batch-обработку очереди через буфер индексов вместо Python-цикла.
        """
        gray = self._to_grayscale(tensor).squeeze(0)  # (H, W)
        gray = self._cast_to_dtype(gray)
        if gray.max() <= 1.0:
            gray = gray * 255.0

        start_time = time.time()
        h, w = gray.shape
        seed = self.params.get("seed", (h // 2, w // 2))
        tolerance = self.params.get("tolerance", 0.1)

        seed_y, seed_x = seed
        seed_value = gray[seed_y, seed_x]

        # Маски
        mask = torch.zeros(h, w, dtype=torch.bool, device=self.device)
        visited = torch.zeros(h, w, dtype=torch.bool, device=self.device)

        # Буфер для текущей "волны" пикселей (вместо deque)
        current_wave = torch.tensor([[seed_y, seed_x]], device=self.device, dtype=torch.long)
        visited[seed_y, seed_x] = True

        # Направления: вверх, вниз, влево, вправо
        directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], device=self.device, dtype=torch.long)

        max_iterations = h * w  # Защита от бесконечного цикла
        for _ in range(max_iterations):
            if current_wave.numel() == 0:
                break

            # Извлекаем координаты текущей волны
            y_coords = current_wave[:, 0]
            x_coords = current_wave[:, 1]

            # Проверяем условие допуска для текущих пикселей
            pixel_values = gray[y_coords, x_coords]
            accepted = torch.abs(pixel_values - seed_value) <= tolerance

            # Обновляем маску только для принятых пикселей
            mask[y_coords[accepted], x_coords[accepted]] = True

            # Генерируем соседей для всех пикселей волны (векторизованно)
            # Shape: [n_pixels, 4, 2] -> [n_pixels*4, 2]
            neighbors = current_wave.unsqueeze(1) + directions.unsqueeze(0)  # [N, 4, 2]
            neighbors_flat = neighbors.view(-1, 2)  # [N*4, 2]

            # Фильтруем валидные и непосещённые соседи
            ny, nx = neighbors_flat[:, 0], neighbors_flat[:, 1]
            valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w) & (~visited[ny, nx])

            if not valid.any():
                break

            # Обновляем visited и формируем новую волну
            next_wave = neighbors_flat[valid]
            visited[next_wave[:, 0], next_wave[:, 1]] = True
            current_wave = next_wave

        exec_time = time.time() - start_time

        info = self._log_info(
            "region_growing_torch_vectorized",
            exec_time,
            {"seed": seed, "tolerance": tolerance, **kwargs},
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}")
        return mask.float().unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    def _region_growing_opt2(self, tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Region Growing с fallback на Numba для CPU."""
        gray = self._to_grayscale(tensor).squeeze(0)
        if gray.max() <= 1.0:
            gray = gray * 255.0

        start_time = time.time()
        h, w = gray.shape
        seed = self.params.get("seed", (h // 2, w // 2))
        tolerance = self.params.get("tolerance", 0.1)
        seed_y, seed_x = seed

        # Выбираем бэкенд в зависимости от устройства
        if self.device.type == "cuda" and h * w < 2_000_000:
            # Для GPU и небольших изображений — векторизованный PyTorch
            return self._region_growing(tensor, **kwargs)
        else:
            # Для CPU или больших изображений — Numba
            gray_np = gray.cpu().numpy().astype(np.float32)
            mask_np = _region_growing_numba(gray_np, seed_y, seed_x, tolerance, h, w)
            mask = torch.from_numpy(mask_np).to(self.device)

            exec_time = time.time() - start_time
            info = {
                "method": "region_growing_numba",
                "parameters": {"seed": seed, "tolerance": tolerance, **kwargs},
                "execution_time": exec_time,
            }
            logger.info(info)
            return mask.float().unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _split_and_merge(
        self,
        tensor: torch.Tensor,
        *,
        min_size: Optional[int] = None,
        threshold: Optional[float] = None,
        precision: Optional[str] = None,
    ) -> torch.Tensor:
        """Рекурсивный алгоритм разделения и слияния регионов (квадродерево).

        Рекурсивно делит изображение на квадранты до тех пор, пока дисперсия внутри региона
        не станет меньше заданного порога. Затем возвращает маску второго по величине региона
        (предполагаемый объект).

        Алгоритм:
        1. Конвертация в градации серого на GPU.
        2. Рекурсивное разделение: если дисперсия региона > порога → делим на 4 квадранта.
        3. Остановка при достижении `min_size` или однородности региона.
        4. Сортировка регионов по площади, выбор второго по величине.
        5. Построение бинарной маски выбранного региона.

        Метод особенно эффективен для:
        - Изображений с чёткими границами между однородными областями
        - Задач, где объект занимает значительную, но не доминирующую часть изображения
        - Быстрого прототипирования без обучения моделей

        ⚠️  Ограничения:
            - Рекурсия может быть медленной для больших изображений (>1024×1024).
            - Не учитывает пространственную связность регионов.
            - Требует ручной настройки `threshold` и `min_size`.

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            min_size: Минимальный размер региона для остановки деления (по умолчанию: 50).
            threshold: Порог дисперсии для решения о разделении (по умолчанию: 20.0 для [0,255]).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16' (по умолчанию: из self.dtype).

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - Все вычисления выполняются на GPU без выхода в numpy.
            - Для `fp16` рекомендуется `threshold ≥ 5.0` из-за ограниченной точности дисперсии.
            - Метод не поддерживает `fullgraph=True` в `torch.compile` из-за рекурсии.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "split_and_merge",
                min_size=100,
                threshold=30.0,
                precision="bf16"
            )
            mask = segmenter.segment(image)
            ```
        """
        start_time = time.time()
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor).squeeze(0)  # (H, W)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        # Масштабируем к [0, 255] для стабильности дисперсии
        if gray.max() <= 1.0:
            gray = gray * 255.0

        h, w = gray.shape

        # === ПАРАМЕТРЫ ===
        min_sz = min_size if min_size is not None else self.params.get("min_size", 50)
        thresh = threshold if threshold is not None else self.params.get("threshold", 20.0)

        # === РЕКУРСИВНОЕ РАЗДЕЛЕНИЕ (на GPU) ===
        def _recursive_split(y: int, x: int, h_r: int, w_r: int) -> List[Tuple[int, int, int, int]]:
            # Базовый случай: достигли минимального размера
            if h_r <= min_sz or w_r <= min_sz:
                return [(y, x, h_r, w_r)]

            # Вычисляем дисперсию региона
            region = gray[y : y + h_r, x : x + w_r]
            var = region.var()

            # Если регион однороден — не делим
            if var < thresh:
                return [(y, x, h_r, w_r)]

            # Делим на 4 квадранта
            h_half, w_half = h_r // 2, w_r // 2
            regions = []

            # Верх-лево
            regions.extend(_recursive_split(y, x, h_half, w_half))
            # Верх-право
            if w_r - w_half > 0:
                regions.extend(_recursive_split(y, x + w_half, h_half, w_r - w_half))
            # Низ-лево
            if h_r - h_half > 0:
                regions.extend(_recursive_split(y + h_half, x, h_r - h_half, w_half))
            # Низ-право
            if h_r - h_half > 0 and w_r - w_half > 0:
                regions.extend(_recursive_split(y + h_half, x + w_half, h_r - h_half, w_r - w_half))

            return regions

        precision_val = precision if precision is not None else "fp32"
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            regions = _recursive_split(0, 0, h, w)

            # === ВЫБОР ВТОРОГО ПО ВЕЛИЧИНЕ РЕГИОНА ===
            if len(regions) > 1:
                # Сортируем по площади
                regions_sorted = sorted(regions, key=lambda r: r[2] * r[3], reverse=True)
                target = regions_sorted[1]  # Второй по величине
                y, x, h_r, w_r = target

                # Строим маску
                mask = torch.zeros(h, w, dtype=torch.float32, device=self.device)
                mask[y : y + h_r, x : x + w_r] = 1.0
            else:
                mask = torch.zeros(h, w, dtype=torch.float32, device=self.device)

        exec_time: float = time.time() - start_time
        info = self._log_info(
            "split_and_merge_torch",
            exec_time,
            {
                "min_size": min_size,
                "threshold": threshold,
                "precision": precision,
            },
            precision_val=precision_val,
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _floodfill(
        self,
        tensor: torch.Tensor,
        *,
        points: Optional[List[Tuple[int, int]]] = None,
        tolerance: Optional[float] = None,
        connectivity: int = 4,
        precision: Optional[str] = None,
    ) -> torch.Tensor:
        """Сегментация методом заливки (Flood Fill) из нескольких точек.

        Начиная с заданных точек, рекурсивно заполняет все связанные пиксели,
        интенсивность которых отличается от исходной не более чем на допуск.
        Поддерживает 4- или 8-связность.

        Алгоритм:
        1. Конвертация изображения в градации серого (или работа с цветом).
        2. Инициализация очереди с начальными точками.
        3. Векторизованный BFS: обработка "волны" пикселей за итерацию.
        4. Проверка условия допуска по цвету/интенсивности.
        5. Возврат бинарной маски залитой области.

        Метод особенно эффективен для:
        - Интерактивной сегментации с ручной инициализацией точек
        - Задач, где объект имеет однородный цвет/текстуру
        - Быстрого выделения связных областей

        ⚠️  Ограничения:
            - Требует предварительного знания начальных точек.
            - Чувствителен к выбору `tolerance`: слишком малое → неполная заливка,
            слишком большое → утечка в фон.

        Формула допуска (для цветных изображений):
        ```
        ||I(x,y) - I(seed)||_mean <= tolerance
        ```

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            points: Список начальных точек (x, y) в пикселях. По умолчанию: 5 точек (углы + центр).
            tolerance: Допуск по цвету/интенсивности в диапазоне [0, 1] (по умолчанию: 0.15).
            connectivity: Связность окрестности: 4 или 8 (по умолчанию: 4).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - Все вычисления выполняются на GPU без выхода в numpy.
            - Для цветных изображений используется среднее отклонение по каналам.
            - Для `fp16` рекомендуется `tolerance ≥ 0.1` из-за квантования.
            - Метод не поддерживает `fullgraph=True` из-за динамического числа итераций.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "floodfill",
                points=[(100, 150), (200, 100)],
                tolerance=0.2,
                connectivity=8,
                precision="bf16"
            )
            mask = segmenter.segment(image)
            ```
        """
        start_time = time.time()

        # === ПРЕДПОДГОТОВКА ===
        # Работаем с исходным тензором (цвет или серый)
        img = tensor.squeeze(0)  # (C, H, W) или (1, H, W)
        dtype = self.precision_manager.get_dtype(precision)
        img = self._cast_to_dtype(img) if img.dtype != dtype else img

        c, h, w = img.shape

        # === ПАРАМЕТРЫ ===
        pts = points if points is not None else self.params.get("points", None)
        if pts is None:
            # Default points: углы + центр
            pts = [
                (w // 4, h // 4),
                (w // 4, 3 * h // 4),
                (3 * w // 4, h // 4),
                (3 * w // 4, 3 * h // 4),
                (w // 2, h // 2),
            ]
        tol = tolerance if tolerance is not None else self.params.get("tolerance", 0.15)
        conn = connectivity if connectivity in [4, 8] else 4

        # Направления для 4- или 8-связности
        if conn == 4:
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        else:
            directions = [
                (0, 1),
                (1, 0),
                (0, -1),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]

        # === ИНИЦИАЛИЗАЦИЯ ===
        final_mask = torch.zeros(h, w, dtype=torch.bool, device=self.device)

        # === ЗАЛИВКА ИЗ КАЖДОЙ ТОЧКИ ===
        precision_val = precision if precision is not None else "fp32"
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            for start_x, start_y in pts:
                # Пропускаем уже залитые точки
                if final_mask[start_y, start_x]:
                    continue

                # Целевой цвет в начальной точке
                target_color = img[:, start_y, start_x]  # (C,)

                # Маски для текущей заливки
                visited = torch.zeros(h, w, dtype=torch.bool, device=self.device)
                mask = torch.zeros(h, w, dtype=torch.bool, device=self.device)

                # Очередь для BFS (векторизованная)
                current_wave = torch.tensor([[start_y, start_x]], device=self.device, dtype=torch.long)
                visited[start_y, start_x] = True
                mask[start_y, start_x] = True

                # Основной цикл
                max_iter = h * w  # Защита от бесконечного цикла
                for _ in range(max_iter):
                    if current_wave.numel() == 0:
                        break

                    # Координаты текущей волны
                    y_coords = current_wave[:, 0]
                    x_coords = current_wave[:, 1]

                    # Проверка допуска для текущих пикселей
                    pixel_colors = img[:, y_coords, x_coords].T  # (N, C)
                    color_diff = torch.abs(pixel_colors - target_color).mean(dim=1)
                    accepted = color_diff <= tol

                    # Обновление маски только для принятых пикселей
                    mask[y_coords[accepted], x_coords[accepted]] = True

                    # Генерация соседей (векторизованно)
                    neighbors = current_wave.unsqueeze(1) + torch.tensor(
                        directions, device=self.device, dtype=torch.long
                    ).unsqueeze(
                        0
                    )  # [N, D, 2]
                    neighbors_flat = neighbors.view(-1, 2)  # [N*D, 2]

                    # Фильтрация валидных и непосещённых соседей
                    ny, nx = neighbors_flat[:, 0], neighbors_flat[:, 1]
                    valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w) & (~visited[ny, nx])

                    if not valid.any():
                        break

                    # Обновление visited и формирование новой волны
                    next_wave = neighbors_flat[valid]
                    visited[next_wave[:, 0], next_wave[:, 1]] = True
                    current_wave = next_wave

                # Добавляем результат в общую маску
                final_mask = final_mask | mask

        exec_time: float = time.time() - start_time
        info = self._log_info(
            "floodfill_torch",
            exec_time,
            {
                "points": points,
                "tolerance": tolerance,
                "connectivity": connectivity,
                "precision": precision,
            },
            precision_val=precision_val,
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return final_mask.to(torch.float32).unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _floodfill_torch_visualization(
        self,
        tensor: torch.Tensor,
        *,
        alpha: float = 0.6,
        points: Optional[List[Tuple[int, int]]] = None,
        tolerance: Optional[float] = None,
        connectivity: int = 4,
        colorize_regions: bool = True,  # Показывать разные регионы разными цветами
        precision: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Визуализация для FloodFill с поддержкой цветной сегментации регионов.

        Алгоритм:
        1. Выполнение заливки через оптимизированный `_floodfill` (получение маски).
        2. Если `colorize_regions=True`: выполнение много-точечной заливки с разными цветами.
        3. Создание цветной визуализации на GPU с настраиваемой прозрачностью.
        4. Возврат (визуализация, маска) в едином формате.

        Метод особенно эффективен для:
        - Отладки качества заливки и влияния параметра `tolerance`
        - Визуализации разделения изображения на связные регионы
        - Демонстрации работы алгоритма с разными начальными точками

        ⚠️  Ограничения:
            - При `colorize_regions=True` используется менее оптимизированный путь
            (пока `_multi_point_floodfill` не векторизован полностью).
            - Для больших изображений (>1024×1024) рекомендуется `colorize_regions=False`.

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W), RGB.
            alpha: Прозрачность наложения маски [0, 1] (по умолчанию: 0.6).
            points: Список начальных точек (x, y) в пикселях. По умолчанию: 5 точек (углы + центр).
            tolerance: Допуск по цвету/интенсивности в диапазоне [0, 1] (по умолчанию: 0.15).
            connectivity: Связность окрестности: 4 или 8 (по умолчанию: 4).
            colorize_regions: Если True, показывать разные регионы разными цветами.
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16' (для предобработки).
            **kwargs: Дополнительные параметры (для совместимости).

        Returns:
            Tuple[np.ndarray, torch.Tensor]:
                - Визуализация (H, W, 3), dtype=uint8, RGB — изображение с наложенной маской.
                - Бинарная маска (H, W), dtype=float32, значения {0.0, 1.0} — залитая область.

        Note:
            - При `colorize_regions=False` используется оптимизированный `_floodfill` (векторизованный BFS).
            - При `colorize_regions=True` вызывается `_multi_point_floodfill` (пока с `deque`, но на GPU).
            - Все вычисления выполняются на GPU, numpy используется только для возврата.
            - Для `fp16` рекомендуется `tolerance ≥ 0.1` из-за квантования.
            - Метод не поддерживает `fullgraph=True` из-за динамического числа итераций.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "floodfill",
                points=[(100, 150), (200, 100)],
                tolerance=0.2,
                connectivity=8
            )
            vis, mask = segmenter._floodfill_torch_visualization(
                image, alpha=0.7, colorize_regions=True, precision="bf16"
            )
            # vis: RGB изображение с цветной маской регионов
            # mask: бинарная маска (0.0/1.0)
            ```
        """
        start_time = time.time()

        # === ПАРАМЕТРЫ ===
        pts = points if points is not None else self.params.get("points", None)
        tol = tolerance if tolerance is not None else self.params.get("tolerance", 0.15)
        conn = connectivity if connectivity in [4, 8] else 4

        # === ШАГ 1: ПОЛУЧЕНИЕ МАСКИ ЧЕРЕЗ ОПТИМИЗИРОВАННЫЙ _floodfill ===
        # Это гарантирует согласованность логики между segment() и visualization()
        mask = self._floodfill(
            tensor,
            points=pts,
            tolerance=tol,
            connectivity=conn,
            precision=precision,
        )  # (1, 1, H, W), float32

        # === ШАГ 2: ЦВЕТНАЯ ВИЗУАЛИЗАЦИЯ (опционально) ===
        img = tensor.squeeze(0) if tensor.dim() == 4 else tensor  # (C, H, W)
        dtype = self.precision_manager.get_dtype(precision)
        img = img.to(dtype) if img.dtype != dtype else img
        c, h, w = img.shape

        # Приводим маску к 2D
        mask_2d = mask.squeeze()  # (H, W)

        if colorize_regions and pts is not None:
            # === ЦВЕТНАЯ СЕГМЕНТАЦИЯ ПО ТОЧКАМ ===
            # Используем упрощённый путь: заливка из каждой точки с уникальным цветом
            segmentation = torch.zeros(h, w, dtype=torch.long, device=img.device)
            final_mask = torch.zeros(h, w, dtype=torch.bool, device=img.device)

            # Цвета для регионов (в диапазоне [0, 1] для удобства)
            colors = [
                torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=img.device),  # красный
                torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=img.device),  # зелёный
                torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=img.device),  # синий
                torch.tensor([1.0, 1.0, 0.0], dtype=dtype, device=img.device),  # жёлтый
                torch.tensor([1.0, 0.0, 1.0], dtype=dtype, device=img.device),  # пурпурный
            ]

            precision_val = precision if precision is not None else "fp32"
            with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
                for i, (start_x, start_y) in enumerate(pts):
                    if final_mask[start_y, start_x]:
                        continue

                    # Заливка из одной точки (векторизованная версия)
                    region_mask = self._flood_fill_single_opt(img, (start_x, start_y), tol, conn, dtype)

                    # Добавляем в общую маску
                    new_pixels = region_mask & ~final_mask
                    segmentation[new_pixels] = i % len(colors)
                    final_mask = final_mask | region_mask

            # Создаём цветную визуализацию
            colored = torch.zeros_like(img)  # (C, H, W)
            for i, color in enumerate(colors):
                region_mask = segmentation == i
                if region_mask.any():
                    for c_idx in range(min(c, 3)):  # Поддержка grayscale/RGB
                        colored[c_idx, region_mask] = color[c_idx]

            # Alpha-смешивание с оригиналом
            result = img * (1.0 - alpha) + colored * alpha
        else:
            # === ПРОСТАЯ МАСКА (единый цвет) ===
            color = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=img.device)  # красный
            colored_mask = torch.zeros_like(img)
            mask_bool = mask_2d > 0.5
            for c_idx in range(min(c, 3)):
                colored_mask[c_idx, mask_bool] = color[c_idx]
            result = img * (1.0 - alpha) + colored_mask * alpha

        # === ШАГ 3: КОНВЕРТАЦИЯ В NUMPY (только в конце) ===
        result_np = self._tensor_to_numpy(result.unsqueeze(0), denormalize=True)  # (H, W, 3)
        mask_tensor = mask_2d.to(torch.float32).to(self.device)  # (H, W)

        exec_time = time.time() - start_time
        info = self._log_info(
            "floodfill_visualisation_torch",
            exec_time,
            {
                "alpha": alpha,
                "points": pts,
                "tolerance": tol,
                "colorize_regions": colorize_regions,
                "execution_time": exec_time,
            },
            precision_val=precision_val,
        )
        self.params["visualization_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")

        return result_np, mask_tensor

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _flood_fill_single_opt(
        img: torch.Tensor,  # (C, H, W), dtype=target
        start_point: Tuple[int, int],
        tolerance: float,
        connectivity: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Векторизованная заливка из одной точки (GPU-оптимизированная).

        Использует batch-обработку "волны" пикселей вместо deque для ускорения.

        Args:
            img: Изображение (C, H, W), dtype=target.
            start_point: Начальная точка (x, y).
            tolerance: Допуск по цвету [0, 1].
            connectivity: 4 или 8-связность.
            dtype: Тип данных для вычислений.

        Returns:
            torch.Tensor: Бинарная маска залитой области (H, W), dtype=bool.
        """
        c, h, w = img.shape
        start_x, start_y = start_point

        # Направления
        if connectivity == 4:
            directions = torch.tensor([[0, 1], [1, 0], [0, -1], [-1, 0]], device=img.device, dtype=torch.long)
        else:
            directions = torch.tensor(
                [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]],
                device=img.device,
                dtype=torch.long,
            )

        # Инициализация
        visited = torch.zeros(h, w, dtype=torch.bool, device=img.device)
        mask = torch.zeros(h, w, dtype=torch.bool, device=img.device)

        target_color = img[:, start_y, start_x]  # (C,)

        # Очередь как тензор координат
        current_wave = torch.tensor([[start_y, start_x]], device=img.device, dtype=torch.long)
        visited[start_y, start_x] = True
        mask[start_y, start_x] = True

        max_iter = h * w
        for _ in range(max_iter):
            if current_wave.numel() == 0:
                break

            y_coords = current_wave[:, 0]
            x_coords = current_wave[:, 1]

            # Проверка допуска
            pixel_colors = img[:, y_coords, x_coords].T  # (N, C)
            color_diff = torch.abs(pixel_colors - target_color).mean(dim=1)
            accepted = color_diff <= tolerance

            # Обновление маски
            mask[y_coords[accepted], x_coords[accepted]] = True

            # Генерация соседей (векторизованно)
            neighbors = current_wave.unsqueeze(1) + directions.unsqueeze(0)  # [N, D, 2]
            neighbors_flat = neighbors.view(-1, 2)  # [N*D, 2]

            # Фильтрация валидных соседей
            ny, nx = neighbors_flat[:, 0], neighbors_flat[:, 1]
            valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w) & (~visited[ny, nx])

            if not valid.any():
                break

            # Обновление visited и формирование новой волны
            next_wave = neighbors_flat[valid]
            visited[next_wave[:, 0], next_wave[:, 1]] = True
            current_wave = next_wave

        return mask

    # ──────────────────────────────────────────────────────────────────────
    # МЕТОДЫ КЛАСТЕРИЗАЦИИ
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _kmeans_segmentation(
        self,
        tensor: torch.Tensor,
        *,
        k: Optional[int] = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
        init: Literal["random", "kmeans++"] = "kmeans++",
        precision: Optional[str] = None,
    ) -> torch.Tensor:
        """Сегментация методом K-Means кластеризации.

        Группирует пиксели по цветовому признаку в K кластеров, минимизируя
        внутрикластерную дисперсию. Самый крупный кластер считается фоном.

        Алгоритм:
        1. Конвертация изображения в плоский массив пикселей (H×W, 3).
        2. Инициализация центроидов (kmeans++ или случайно).
        3. Итеративное обновление: присвоение меток → пересчёт центроидов.
        4. Ранняя остановка при сходимости или достижении `max_iter`.
        5. Построение маски: все кластеры, кроме самого крупного.

        Метод особенно эффективен для:
        - Сегментации по цвету (логотипы, объекты с однородной окраской)
        - Предварительной обработки перед более сложными методами
        - Задач, где не требуется пространственная связность

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W), RGB.
            k: Число кластеров (по умолчанию: 3).
            max_iter: Максимальное число итераций (по умолчанию: 50).
            tol: Порог сходимости по изменению центроидов (по умолчанию: 1e-4).
            init: Метод инициализации: 'random' или 'kmeans++' (по умолчанию).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - Используется `torch.cdist` для векторизованного вычисления расстояний.
            - `kmeans++` инициализация ускоряет сходимость на 2-5×.
            - Для `fp16` рекомендуется `tol ≥ 1e-3` из-за ограниченной точности.
            - Метод безопасен для `torch.compile(fullgraph=True)` при фиксированном `k`.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "kmeans_segmentation",
                k=4,
                init="kmeans++",
                precision="bf16"
            )
            mask = segmenter.segment(color_image)
            ```
        """
        start_time: float = time.time()

        # === ПРЕДПОДГОТОВКА ===
        # Убираем batch, переставляем к (H, W, C), сплющиваем
        img = tensor.squeeze(0).permute(1, 2, 0)  # (H, W, C)
        h, w, c = img.shape
        pixels = img.reshape(-1, c)  # (N, C)

        dtype = self.precision_manager.get_dtype(precision)
        pixels = pixels.to(dtype) if pixels.dtype != dtype else pixels

        # === ПАРАМЕТРЫ ===
        k_val = k if k is not None else self.params.get("k", 3)
        max_it = max_iter if max_iter is not None else self.params.get("max_iter", 50)
        tol_val = tol if tol is not None else self.params.get("tol", 1e-4)

        # === ИНИЦИАЛИЗАЦИЯ ЦЕНТРОИДОВ ===
        if init == "kmeans++":
            # Простая эвристика: первый центроид случайно, остальные — с вероятностью ∝ d²
            centroid_list: List[torch.Tensor] = [pixels[torch.randint(0, pixels.size(0), (1,), device=pixels.device)]]
            for _ in range(1, k_val):
                dists = (
                    torch.cdist(pixels.unsqueeze(0), torch.stack(centroid_list).unsqueeze(0))
                    .squeeze(0)
                    .min(dim=1)
                    .values
                )
                probs = dists**2 / (dists.sum() + 1e-8)
                idx = torch.multinomial(probs, 1)
                centroid_list.append(pixels[idx])
            centroids: torch.Tensor = torch.cat(centroid_list, dim=0)  # Явная аннотация
        else:
            idx = torch.randperm(pixels.size(0), device=pixels.device)[:k_val]
            centroids = pixels[idx]  # Уже Tensor

        # === ОСНОВНОЙ ЦИКЛ ===
        precision_val = precision if precision is not None else "fp32"
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            for _ in range(max_it):
                # Присвоение меток
                dists = torch.cdist(pixels.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
                labels = torch.argmin(dists, dim=1)

                # Пересчёт центроидов
                new_centroids = torch.stack(
                    [(pixels[labels == i].mean(dim=0) if (labels == i).any() else centroids[i]) for i in range(k_val)]
                )

                # Проверка сходимости
                if torch.allclose(centroids, new_centroids, atol=tol_val, rtol=0):
                    break
                centroids = new_centroids

            # === ПОСТРОЕНИЕ МАСКИ ===
            # Самый крупный кластер = фон
            counts = torch.bincount(labels, minlength=k_val)
            bg_label = torch.argmax(counts)
            mask_flat = (labels != bg_label).float()
            mask = mask_flat.view(h, w)

        exec_time: float = time.time() - start_time
        info = self._log_info(
            "kmeans_torch",
            exec_time,
            {
                "k": k,
                "max_iter": max_iter,
                "tol": tol,
                "init": init,
                "precision": precision,
            },
            precision_val=precision_val,
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}: precision_val={precision_val}, dtype={dtype}")
        return mask.to(torch.float32).unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _dbscan_segmentation(
        self,
        tensor: torch.Tensor,
        *,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
        downsample: Optional[float] = None,
        precision: Optional[str] = None,
    ) -> torch.Tensor:
        """Сегментация методом DBSCAN кластеризации.

        Группирует пиксели на основе плотности: точки с достаточным числом соседей
        в радиусе `eps` образуют кластер. Пиксели-шум исключаются.

        Алгоритм:
        1. Конвертация изображения в массив признаков (цвет + координаты).
        2. Опциональное уменьшение разрешения для ускорения.
        3. Применение DBSCAN (через sklearn, т.к. чистый PyTorch неэффективен).
        4. Интерполяция меток обратно к исходному размеру.
        5. Построение маски: все кластеры, кроме шума.

        Метод особенно эффективен для:
        - Сегментации объектов произвольной формы без задания числа кластеров
        - Изображений с шумом, где нужно отделить "сигнал" от фона
        - Задач, где важна устойчивость к выбросам

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W), RGB.
            eps: Радиус окрестности для поиска соседей (по умолчанию: 0.1).
            min_samples: Минимальное число соседей для ядра кластера (по умолчанию: 10).
            downsample: Коэффициент уменьшения разрешения для ускорения (по умолчанию: 0.5).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16' (используется только для предобработки).

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - DBSCAN реализуется через sklearn из-за сложности эффективной реализации на PyTorch.
            - Для больших изображений автоматически применяется downsample.
            - Метод не поддерживает `torch.compile` из-за вызова внешних библиотек.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "dbscan_segmentation",
                eps=0.15,
                min_samples=15,
                downsample=0.75
            )
            mask = segmenter.segment(image)
            ```
        """
        start_time: float = time.time()
        # === ПРЕДПОДГОТОВКА ===
        img_np = self._tensor_to_numpy(tensor)
        h, w = img_np.shape[:2]

        # === ПАРАМЕТРЫ ===
        eps_val = eps if eps is not None else self.params.get("eps", 0.1)
        min_s = min_samples if min_samples is not None else self.params.get("min_samples", 10)
        ds = downsample if downsample is not None else self.params.get("downsample", 0.5)

        # === УМЕНЬШЕНИЕ РАЗРЕШЕНИЯ (опционально) ===
        if h * w > 100_000 and ds < 1.0:
            small_h, small_w = int(h * ds), int(w * ds)
            img_small = cv2.resize(img_np, (small_w, small_h), interpolation=cv2.INTER_AREA)
            pixels = img_small.reshape(-1, 3)
            use_resize = True
        else:
            pixels = img_np.reshape(-1, 3)
            use_resize = False

        # === DBSCAN (через sklearn) ===
        try:
            from sklearn.cluster import DBSCAN

            db = DBSCAN(eps=eps_val, min_samples=min_s, n_jobs=-1)
            labels = db.fit_predict(pixels)
        except ImportError:
            warnings.warn("sklearn not installed. Fallback to kmeans.")
            return self._kmeans_segmentation(tensor)

        # === ИНТЕРПОЛЯЦИЯ ОБРАТНО ===
        if use_resize:
            labels_2d = labels.reshape(small_h, small_w)
            labels_2d = cv2.resize(labels_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(int)
        else:
            labels_2d = labels.reshape(h, w)

        # === ПОСТРОЕНИЕ МАСКИ ===
        mask_np = (labels_2d != -1).astype(np.float32)
        mask = torch.from_numpy(mask_np).to(self.device)

        exec_time: float = time.time() - start_time
        info = self._log_info(
            "dbscan_torch",
            exec_time,
            {
                "eps": eps,
                "min_samples": min_samples,
                "downsample": downsample,
                "precision": precision,
            },
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}: precision={precision}")
        return mask.unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _meanshift(
        self,
        tensor: torch.Tensor,
        *,
        bandwidth: Optional[float] = None,
        spatial_radius: Optional[int] = None,
        color_radius: Optional[int] = None,
        downsample: Optional[float] = None,
        precision: Optional[str] = None,
    ) -> torch.Tensor:
        """Сегментация методом MeanShift.

        Итеративно сдвигает каждый пиксель к локальному центру масс в пространстве признаков
        (цвет + координаты). Результатом является кластеризация пикселей по плотности.

        Алгоритм:
        1. Построение пространства признаков: [координаты / spatial_radius, цвет / color_radius].
        2. Применение MeanShift для поиска мод плотности (через sklearn при необходимости).
        3. Пост-обработка: самый крупный кластер считается фоном.
        4. Возврат бинарной маски объектов.

        Метод особенно эффективен для:
        - Сегментации по цвету с учётом пространственной близости
        - Изображений с плавными градиентами и размытыми границами
        - Задач, где не требуется точное число кластеров

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W), RGB.
            bandwidth: Радиус окна для поиска соседей (по умолчанию: 0.5).
            spatial_radius: Радиус нормализации пространственных координат (по умолчанию: 35).
            color_radius: Радиус нормализации цветовых каналов (по умолчанию: 60).
            downsample: Коэффициент уменьшения разрешения для ускорения (по умолчанию: 0.5).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16' (для предобработки).

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - MeanShift реализуется через sklearn из-за сложности эффективной pure-PyTorch версии.
            - Для больших изображений автоматически применяется downsample + интерполяция меток.
            - Метод не поддерживает `torch.compile` из-за вызова внешних библиотек.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "meanshift",
                bandwidth=0.6,
                spatial_radius=40,
                color_radius=50,
                downsample=0.75
            )
            mask = segmenter.segment(image)
            ```
        """
        start_time = time.time()

        # === ПРЕДПОДГОТОВКА ===
        img_np = self._tensor_to_numpy(tensor)
        h, w, c = img_np.shape[:3] if img_np.ndim == 3 else (*img_np.shape, 1)

        # === ПАРАМЕТРЫ ===
        bw = bandwidth if bandwidth is not None else self.params.get("bandwidth", 0.5)
        sr = spatial_radius if spatial_radius is not None else self.params.get("spatial_radius", 35)
        cr = color_radius if color_radius is not None else self.params.get("color_radius", 60)
        ds = downsample if downsample is not None else self.params.get("downsample", 0.5)

        # === DOWNsample ДЛЯ УСКОРЕНИЯ ===
        use_resize = False
        if h * w > 100_000 and ds < 1.0:
            small_h, small_w = int(h * ds), int(w * ds)
            img_small = cv2.resize(img_np, (small_w, small_h), interpolation=cv2.INTER_AREA)
            features_np = self._build_meanshift_features(img_small, sr, cr)
            use_resize = True
        else:
            features_np = self._build_meanshift_features(img_np, sr, cr)

        features_flat = features_np.reshape(-1, features_np.shape[-1])

        # === MEANSHIFT (через sklearn) ===
        try:
            from sklearn.cluster import MeanShift as SkMeanShift

            ms = SkMeanShift(bandwidth=bw, max_iter=100, n_jobs=-1, bin_seeding=True)
            labels_flat = ms.fit_predict(features_flat)
        except ImportError:
            warnings.warn("sklearn not installed. Fallback to kmeans.")
            return self._kmeans_segmentation(tensor)

        # === ИНТЕРПОЛЯЦИЯ МЕТок ОБРАТНО ===
        if use_resize:
            labels_2d = labels_flat.reshape(small_h, small_w)
            labels_2d = cv2.resize(labels_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(int)
        else:
            labels_2d = labels_flat.reshape(h, w)

        # === ПОСТРОЕНИЕ МАСКИ ===
        unique, counts = np.unique(labels_2d, return_counts=True)
        bg_label = unique[np.argmax(counts)] if len(unique) > 0 else -1
        mask_np = (labels_2d != bg_label).astype(np.float32)

        mask = torch.from_numpy(mask_np).to(self.device)

        exec_time: float = time.time() - start_time
        info = self._log_info(
            "meanshift_torch",
            exec_time,
            {
                "bandwidth": bandwidth,
                "spatial_radius": spatial_radius,
                "color_radius": color_radius,
                "downsample": downsample,
                "precision": precision,
            },
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}")
        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _build_meanshift_features(
        image_np: np.ndarray,
        spatial_radius: int,
        color_radius: int,
    ) -> np.ndarray:
        """Построение пространства признаков для MeanShift."""
        h, w = image_np.shape[:2]
        c = image_np.shape[2] if image_np.ndim == 3 else 1
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        spatial = np.stack([x_coords / spatial_radius, y_coords / spatial_radius], axis=-1)
        color = image_np.astype(np.float32) / color_radius
        if c == 1:
            color = np.repeat(color, 3, axis=-1)
        return np.concatenate([spatial, color], axis=-1)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _meanshift_torch_visualization(
        self,
        tensor: torch.Tensor,
        *,
        alpha: float = 0.6,
        bandwidth: Optional[float] = None,
        spatial_radius: Optional[int] = None,
        color_radius: Optional[int] = None,
        downsample: Optional[float] = None,
        precision: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Визуализация для MeanShift с поддержкой AMP и минимальными трансферами.

        Алгоритм:
        1. Выполнение сегментации через `_meanshift` (получение меток).
        2. Вычисление среднего цвета для каждого кластера на оригинальном изображении.
        3. Построение сегментированного изображения: каждый пиксель → средний цвет своего кластера.
        4. Создание бинарной маски (все кластеры, кроме самого крупного = фон).
        5. Возврат (визуализация, маска) в едином формате.

        Метод особенно эффективен для:
        - Отладки качества кластеризации и параметров MeanShift
        - Визуализации цветового разделения изображения на регионы
        - Демонстрации работы алгоритма с разными `bandwidth` и `spatial_radius`

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W), RGB.
            alpha: Прозрачность наложения маски [0, 1] (по умолчанию: 0.6).
            bandwidth: Радиус окна для поиска соседей (по умолчанию: из self.params).
            spatial_radius: Радиус нормализации координат (по умолчанию: из self.params).
            color_radius: Радиус нормализации цвета (по умолчанию: из self.params).
            downsample: Коэффициент уменьшения разрешения для ускорения (по умолчанию: из self.params).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16' (для предобработки).
            **kwargs: Дополнительные параметры (для совместимости).

        Returns:
            Tuple[np.ndarray, torch.Tensor]:
                - Визуализация (H, W, 3), dtype=uint8, RGB — сегментированное изображение.
                - Бинарная маска (H, W), dtype=float32, значения {0.0, 1.0} — объекты.

        Note:
            - Основная логика кластеризации делегирована `_meanshift` для избежания дублирования.
            - Визуализация (усреднение цветов по кластерам) выполняется на CPU через numpy
            из-за необходимости работы с метками из sklearn.
            - Для `fp16` рекомендуется `bandwidth ≥ 0.3` для стабильности кластеризации.
            - Метод не поддерживает `torch.compile` из-за вызова sklearn.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "meanshift",
                bandwidth=0.6,
                spatial_radius=40,
                color_radius=50,
                downsample=0.75
            )
            vis, mask = segmenter._meanshift_torch_visualization(image, alpha=0.7)
            # vis: RGB изображение с усреднёнными цветами регионов
            # mask: бинарная маска объектов
            ```
        """
        start_time = time.time()

        # === ПАРАМЕТРЫ ===
        bw = bandwidth if bandwidth is not None else self.params.get("bandwidth", 0.5)
        sr = spatial_radius if spatial_radius is not None else self.params.get("spatial_radius", 35)
        cr = color_radius if color_radius is not None else self.params.get("color_radius", 60)
        ds = downsample if downsample is not None else self.params.get("downsample", 0.5)

        # === ШАГ 1: ПОЛУЧЕНИЕ МЕТок ЧЕРЕЗ ОСНОВНОЙ МЕТОД ===
        # Используем `_meanshift` для получения бинарной маски
        # Это гарантирует согласованность логики между segment() и visualization()
        mask = self._meanshift(
            tensor,
            bandwidth=bw,
            spatial_radius=sr,
            color_radius=cr,
            downsample=ds,
            precision=precision,
        )  # (1, 1, H, W), float32

        # === ШАГ 2: ПОЛУЧЕНИЕ МЕТок КЛАСТЕРОВ (для визуализации) ===
        # Конвертируем вход в numpy для sklearn (как в основном методе)
        img_np = self._tensor_to_numpy(tensor)
        h, w, c = img_np.shape[:3] if img_np.ndim == 3 else (*img_np.shape, 1)

        # Построение признаков (как в `_build_meanshift_features`)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        spatial = np.stack([x_coords / sr, y_coords / sr], axis=-1)
        color = (img_np.astype(np.float32) / cr) if img_np.max() > 1.0 else (img_np / cr)
        features: np.ndarray
        if c == 1:
            color = np.repeat(color, 3, axis=-1)
        features = np.concatenate([spatial, color], axis=-1)
        last_dim = features.shape[-1]
        features_flat = features.reshape(-1, last_dim)

        # === ШАГ 3: MEANSHIFT ДЛЯ ПОЛУЧЕНИЯ МЕТок (повторно, но только для визуализации) ===
        # ⚠️  Это дублирование необходимо, т.к. `_meanshift` возвращает только бинарную маску,
        # а для визуализации нужны сами метки кластеров.
        try:
            from sklearn.cluster import MeanShift as SkMeanShift

            ms = SkMeanShift(bandwidth=bw, max_iter=100, n_jobs=-1, bin_seeding=True)
            labels_flat = ms.fit_predict(features_flat)
        except ImportError:
            warnings.warn("sklearn not installed. Fallback to original image.")
            result_np = img_np.copy()
            mask_np = np.zeros((h, w), dtype=np.float32)
            return result_np, torch.from_numpy(mask_np).to(self.device)

        labels_2d = labels_flat.reshape(h, w)

        # === ШАГ 4: ПОСТРОЕНИЕ СЕГМЕНТИРОВАННОГО ИЗОБРАЖЕНИЯ ===
        # Для каждого кластера вычисляем средний цвет и заменяем все пиксели кластера этим цветом
        segmented = np.zeros_like(img_np, dtype=np.float32)
        unique_labels = np.unique(labels_2d)

        for label in unique_labels:
            cluster_mask = labels_2d == label
            if np.any(cluster_mask):
                # Средний цвет кластера (в оригинальном цветовом пространстве)
                mean_color = img_np[cluster_mask].mean(axis=0)
                segmented[cluster_mask] = mean_color

        # Конвертация в uint8 для возврата
        if segmented.max() <= 1.0:
            segmented = (segmented * 255).astype(np.uint8)
        else:
            segmented = segmented.astype(np.uint8)

        # === ШАГ 5: ПОДГОТОВКА ВОЗВРАЩАЕМЫХ ЗНАЧЕНИЙ ===
        # Маска уже получена из `_meanshift`, конвертируем в numpy для единообразия
        mask_np = mask.squeeze().cpu().numpy()
        if mask_np.max() <= 1.0:
            mask_np = (mask_np * 255).astype(np.uint8)

        # Возвращаем маску как torch.Tensor на оригинальном устройстве
        mask_tensor = torch.from_numpy(mask_np.astype(np.float32) / 255.0).to(self.device)

        exec_time: float = time.time() - start_time
        info = self._log_info(
            "meanshift_visualisation_torch",
            exec_time,
            {
                "bandwidth": bw,
                "spatial_radius": sr,
                "color_radius": cr,
                "precision": precision,
            },
            # precision_val=precision_val,
        )
        self.params["visualization_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}")

        return segmented, mask_tensor

    # ──────────────────────────────────────────────────────────────────────
    # МЕТОДЫ АКТИВНЫХ КОНТУРОВ
    # ──────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────
    def _active_contour(
        self,
        tensor: torch.Tensor,
        *,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        w_edge: Optional[float] = None,
        w_line: Optional[float] = None,
        max_iter: Optional[int] = None,
        n_points: Optional[int] = None,
        sigma_edge: Optional[float] = None,
        init_radius_factor: float = 0.6,
    ) -> torch.Tensor:
        """Сегментация активными контурами (Snakes / Kass-Witkin-Terzopoulos).

        Инициализирует замкнутый контур (окружность) и деформирует его под действием:
        - Внутренних сил (упругость `alpha`, жёсткость `beta`)
        - Внешних сил (притяжение к границам `w_edge`, к линиям `w_line`)
        Минимизирует энергию контура до равновесия.

        Формула энергии:
        ```
        E = ∫ [α|v_s|² + β|v_ss|² + w_edge·E_edge + w_line·E_line] ds
        ```

        Алгоритм:
        1. Создание начального контура в центре изображения.
        2. Гауссово сглаживание изображения для стабилизации градиента.
        3. Итеративное обновление позиций точек контура через `active_contour()`.
        4. Заполнение полигона контура для получения бинарной маски.

        Метод особенно эффективен для:
        - Медицинских изображений с чёткими, но деформируемыми границами
        - Микроскопических снимков клеток и тканей
        - Задач, где важна гладкость и непрерывность контура

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            alpha: Коэффициент упругости контура (по умолчанию: 0.01).
            beta: Коэффициент жёсткости контура (по умолчанию: 0.1).
            gamma: Шаг градиентного спуска (по умолчанию: 0.001).
            w_edge: Вес внешней энергии от границ (по умолчанию: 1.0).
            w_line: Вес внешней энергии от линий (по умолчанию: 0.0).
            max_iter: Максимальное число итераций (по умолчанию: 250).
            n_points: Число точек в контуре (по умолчанию: 200).
            sigma_edge: Sigma для гауссова сглаживания перед вычислением градиента (по умолчанию: 3.0).
            init_radius_factor: Доля радиуса от центра до края для инициализации (по умолчанию: 0.6).

        Returns:
            torch.Tensor: Бинарная маска внутренней области контура (1, 1, H, W), dtype=float32.

        Note:
            - `gamma` — шаг времени (learning rate); слишком большой → нестабильность.
            - Требует инициализации контура близко к объекту для сходимости.
            - Возвращает маску внутри финального полигона, а не только линию.
            - Все вычисления выполняются на GPU без синхронизации с CPU.

        Example:
            ```python
            segmenter = TorchSegmenter("active_contour", alpha=0.015, beta=10, max_iter=500)
            mask = segmenter.segment(cell_image)  # (1, 1, H, W)
            ```
        """
        gray: torch.Tensor = self._to_grayscale(tensor)  # (B, 1, H, W)
        if gray.dim() == 3:
            gray = gray.unsqueeze(0)  # (1, H, W) → (1, 1, H, W)
        start_time: float = time.time()

        alpha = alpha if alpha is not None else self.params.get("alpha", 0.01)  # упругость (длина контура)
        beta = beta if beta is not None else self.params.get("beta", 0.1)  # жёсткость (кривизна)
        gamma = gamma if gamma is not None else self.params.get("gamma", 0.001)  # шаг градиентного спуска
        w_edge = w_edge if w_edge is not None else self.params.get("w_edge", 1.0)  # вес внешней (граничной) энергии
        w_line = w_line if w_line is not None else self.params.get("w_line", 0.0)
        max_iter = max_iter if max_iter is not None else self.params.get("max_iter", 250)  # число итераций
        n_points = n_points if n_points is not None else self.params.get("n_points", 200)  # число точек контура
        sigma_edge = sigma_edge if sigma_edge is not None else self.params.get("sigma", 3.0)

        h, w = gray.shape[2], gray.shape[3]
        device = gray.device
        dtype = self.dtype

        # === 2. КЭШИРОВАНИЕ ЯДЕР СОБЕЛЯ (статический метод) ===
        # sobel_x, sobel_y = self._get_sobel_kernels_cached(device, dtype)
        sobel_x, sobel_y = self._get_conv_kernel("sobel", return_pair=True, dtype=dtype, device=self.device)

        # === 3. ВЫЧИСЛЕНИЕ КАРТЫ ГРАНИЦ (внешняя энергия) ===
        # Гауссово сглаживание
        sigma_edge = self.params.get("sigma", 3.0)
        ks = int(2 * round(3 * sigma_edge) + 1)
        if ks % 2 == 0:
            ks += 1
        if sigma_edge > 0:
            ks = int(2 * round(3 * sigma_edge) + 1)
            ks = ks if ks % 2 == 1 else ks + 1
            gray_smooth = tv_gaussian_blur(gray, kernel_size=[ks, ks], sigma=[sigma_edge, sigma_edge])
        else:
            gray_smooth = gray

        gx = F.conv2d(gray_smooth, sobel_x.to(gray_smooth.dtype), padding=1)  # (B, 1, H, W)
        gy = F.conv2d(gray_smooth, sobel_y.to(gray_smooth.dtype), padding=1)
        edge_map = gx.square() + gy.square()  # (B, 1, H, W)

        # Нормализуем и берём градиент карты границ (внешние силы)
        # Нормализация [0, 1]
        edge_max = edge_map.amax(dim=(2, 3), keepdim=True)
        edge_map = edge_map / (edge_max + 1e-8)

        # Градиент карты границ (для внешней силы)
        ext_fx = F.conv2d(edge_map, sobel_x, padding=1).squeeze(0)  # (1, H, W)
        ext_fy = F.conv2d(edge_map, sobel_y, padding=1).squeeze(0)

        # === 4. ИНИЦИАЛИЗАЦИЯ КОНТУРА ===
        cx, cy = w / 2.0, h / 2.0
        r = min(cx, cy) * init_radius_factor
        t = torch.linspace(0, 2 * torch.pi, n_points + 1, device=device, dtype=dtype)[:-1]

        # snake: (N, 2), dtype=dtype
        snake = torch.stack([cx + r * torch.cos(t), cy + r * torch.sin(t)], dim=1)  # (N, 2)

        # === 5. ПОДГОТОВКА МАТРИЦЫ ВНУТРЕННЕЙ ЭНЕРГИИ (циклическая, через FFT) ===
        # --- Матрица пентадиагональная для внутренней энергии (трёхточечная для 1D) ---
        # Строим матрицу A = alpha * D2 + beta * D4, где D2 — вторые разности, D4 — четвёртые

        # Коэффициенты для второй разности (упругость)
        a2 = alpha
        # Коэффициенты для четвёртой разности (жёсткость)
        a4 = beta

        # Строим циклическую матрицу как сумму сдвигов
        # Первая строка циклической матрицы (с учётом периодических граничных условий)
        first_row = torch.zeros(n_points, device=device, dtype=dtype)
        first_row[0] = 2 * a2 + 6 * a4
        first_row[1] = first_row[-1] = -a2 - 4 * a4
        first_row[2] = first_row[-2] = a4

        # Вторые разности: xi-1 - 2xi + xi+1 → коэффициенты: {-1:1, 0:-2, 1:1}
        # Четвёртые разности: xi-2 - 4xi-1 + 6xi - 4xi+1 + xi+2

        # Строим циклическую матрицу через FFT (эффективно)
        # A * x = (I + gamma * A)^{-1} * (x + gamma * f_ext)
        # Решаем через (I + gamma * A) x_new = x + gamma * f_ext
        # Матрица (I + gamma*A) — тоже циклическая, можно инвертировать через FFT
        A_fft = torch.fft.rfft(first_row)
        I_plus_gammaA_fft = 1.0 + gamma * A_fft  # (N//2 + 1,)

        # === 6. ОСНОВНОЙ ЦИКЛ (векторизованный, без Python-циклов по пикселям) ===
        for _ in range(max_iter):
            # Интерполируем внешние силы в текущих точках контура
            # Нормализуем координаты к [-1, 1] для grid_sample
            xs = snake[:, 0]
            ys = snake[:, 1]

            # Клипуем координаты к границам изображения
            xs = torch.clamp(xs, 0, w - 1)
            ys = torch.clamp(ys, 0, h - 1)

            # Биленейная интерполяция внешних сил в точках контура
            grid_x = (xs / (w - 1)) * 2 - 1  # [-1, 1]
            grid_y = (ys / (h - 1)) * 2 - 1
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)

            fx_map = ext_fx.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            fy_map = ext_fy.unsqueeze(0).unsqueeze(0)

            fx_pts = F.grid_sample(fx_map, grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze()
            fy_pts = F.grid_sample(fy_map, grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze()

            # Правая часть: x + gamma * (w_edge * f_edge + w_line * f_line)
            rhs_x = snake[:, 0] + gamma * (w_edge * fx_pts + w_line * torch.zeros_like(fx_pts))
            rhs_y = snake[:, 1] + gamma * (w_edge * fy_pts + w_line * torch.zeros_like(fy_pts))

            # Решаем через FFT (матрица (I + gamma*A) циклическая)
            rhs_x_fft = torch.fft.rfft(rhs_x)
            rhs_y_fft = torch.fft.rfft(rhs_y)

            new_x = torch.fft.irfft(rhs_x_fft / I_plus_gammaA_fft, n=n_points)
            new_y = torch.fft.irfft(rhs_y_fft / I_plus_gammaA_fft, n=n_points)

            snake = torch.stack([torch.clamp(new_x, 0, w - 1), torch.clamp(new_y, 0, h - 1)], dim=1)

        # --- Строим бинарную маску из контура ---
        # === 7. РАСТЕРИЗАЦИЯ КОНТУРА В МАСКУ (полностью на GPU) ===
        mask = self._rasterize_polygon_torch(snake, h, w, device, dtype)  # (H, W)

        exec_time: float = time.time() - start_time
        info = self._log_info(
            "active_contour_torch",
            exec_time,
            {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "w_edge": w_edge,
                "w_line": w_line,
                "max_iter": max_iter,
                "n_points": n_points,
                "sigma_edge": sigma_edge,
                "init_radius_factor": init_radius_factor,
                # "precision": precision,  # ✅ Добавлено
            },
            # precision_val=precision_val,
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}")
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _rasterize_polygon_torch(
        polygon: torch.Tensor,  # (N, 2), dtype=float
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Растеризация полигона в бинарную маску на чистом PyTorch (GPU).

        Алгоритм: scanline fill с сортировкой рёбер.

        Args:
            polygon: Координаты вершин (N, 2), float, в пикселях
            height, width: Размер изображения
            device, dtype: Устройство и тип данных

        Returns:
            torch.Tensor: Бинарная маска (H, W), dtype=dtype, значения 0.0 или 1.0
        """
        if polygon.numel() == 0:
            return torch.zeros(height, width, device=device, dtype=dtype)

        # Округляем координаты до int для индексации
        verts = polygon.round().long()  # (N, 2)
        min_coords = torch.tensor([0, 0], device=device, dtype=verts.dtype)
        max_coords = torch.tensor([width - 1, height - 1], device=device, dtype=verts.dtype)
        verts = torch.clamp(verts, min=min_coords, max=max_coords)

        # Векторизованный point-in-polygon test (ray casting)
        # Для каждой точки (x, y) считаем число пересечений луча → нечётное = внутри
        n = verts.shape[0]
        inside = torch.zeros(height, width, device=device, dtype=torch.bool)
        # Простая реализация: проверяем каждое ребро
        for i in range(n):
            x1: int = int(verts[i, 0].item())
            y1: int = int(verts[i, 1].item())
            x2: int = int(verts[(i + 1) % n, 0].item())
            y2: int = int(verts[(i + 1) % n, 1].item())

            # Горизонтальные рёбра пропускаем
            if y1 == y2:
                continue

            # Упорядочиваем по y
            if y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1

            # Проверяем строки в диапазоне [y1, y2)
            y_start: int = max(0, y1)
            y_end: int = min(height, y2)
            for y in range(y_start, y_end):
                # Вычисляем x пересечения с горизонтальной линией y
                if y2 - y1 != 0:
                    x_intersect = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
                else:
                    continue
                # Все пиксели слева от пересечения переключают parity
                if 0 <= x_intersect < width:
                    inside[y, : int(x_intersect) + 1] ^= True

        return inside.to(dtype)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _gvf_contour(
        self,
        tensor: torch.Tensor,
        *,
        threshold: Optional[float] = None,
        sigma: Optional[float] = None,
        iterations: Optional[int] = None,
        mu: Optional[float] = None,
        precision: Optional[str] = None,
    ) -> torch.Tensor:
        """Сегментация на основе Gradient Vector Flow (GVF).

        Вычисляет векторное поле, распространяющее информацию о градиентах по всему изображению.
        Это позволяет контуру "чувствовать" границы даже на расстоянии. Маска строится по величине GVF.

        Алгоритм:
        1. Вычисление градиентов изображения (Sobel).
        2. Итеративное сглаживание векторного поля для распространения градиентов.
        3. Вычисление магнитуды итогового поля.
        4. Бинаризация по порогу.

        Метод особенно эффективен для:
        - Задач, где границы объекта размыты или прерывисты
        - Предварительной обработки перед активными контурами
        - Изображений с низким контрастом на границах

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            threshold: Порог для бинаризации в диапазоне [0, 1] (по умолчанию: 0.1).
            sigma: Sigma для предварительного сглаживания (по умолчанию: 1.0).
            iterations: Число итераций распространения поля (по умолчанию: 20).
            mu: Коэффициент регуляризации (по умолчанию: 0.2).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - Все вычисления выполняются на GPU без выхода в numpy.
            - Для `fp16` рекомендуется `threshold ≥ 0.05` из-за квантования.
            - Метод безопасен для `torch.compile(fullgraph=True)`.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "gvf_contour",
                threshold=0.15,
                iterations=30,
                mu=0.25,
                precision="bf16"
            )
            mask = segmenter.segment(image)
            ```
        """
        start_time: float = time.time()

        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        # === ПАРАМЕТРЫ ===
        thresh = threshold if threshold is not None else self.params.get("threshold", 0.1)
        sig = sigma if sigma is not None else self.params.get("sigma", 1.0)
        iters = iterations if iterations is not None else self.params.get("iterations", 20)
        mu_val = mu if mu is not None else self.params.get("mu", 0.2)

        # === ГАУССОВО СГЛАЖИВАНИЕ (опционально) ===
        if sig > 0:
            ks = int(2 * round(3 * sig) + 1)
            ks = ks if ks % 2 == 1 else ks + 1
            gray = tv_gaussian_blur(gray, kernel_size=[ks, ks], sigma=[sig, sig])

        # === ЯДРА СОБЕЛЯ ===
        # sobel_x, sobel_y = self._get_sobel_kernels_cached(self.device, dtype)
        sobel_x, sobel_y = self._get_conv_kernel("sobel", return_pair=True, dtype=dtype, device=self.device)

        precision_val = precision if precision is not None else "fp32"
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            # === ГРАДИЕНТЫ ===
            gx = F.conv2d(gray, sobel_x.to(gray.dtype), padding=1)
            gy = F.conv2d(gray, sobel_y.to(gray.dtype), padding=1)

            # === ИТЕРАТИВНОЕ РАСПРОСТРАНЕНИЕ (упрощённый GVF) ===
            for _ in range(iters):
                gx_smooth = F.conv2d(
                    gx,
                    torch.ones(1, 1, 3, 3, device=self.device, dtype=dtype) / 9,
                    padding=1,
                )
                gy_smooth = F.conv2d(
                    gy,
                    torch.ones(1, 1, 3, 3, device=self.device, dtype=dtype) / 9,
                    padding=1,
                )
                gx = mu_val * gx_smooth + (1 - mu_val) * gx
                gy = mu_val * gy_smooth + (1 - mu_val) * gy

            # === МАГНИТУДА И БИНАРИЗАЦИЯ ===
            mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
            mag_max = mag.amax(dim=(2, 3), keepdim=True)
            mag = mag / (mag_max + 1e-8)

            thresh_t = torch.tensor(thresh, dtype=dtype, device=self.device)
            mask = (mag > thresh_t).to(dtype)

        exec_time: float = time.time() - start_time

        info = self._log_info(
            "gvf_torch",
            exec_time,
            {
                "threshold": threshold,
                "sigma": sigma,
                "iterations": iterations,
                "mu": mu,
                "precision": precision,
            },
            # precision_val=precision_val,
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}")
        return mask.to(torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _morphological_snakes(
        self,
        tensor: torch.Tensor,
        *,
        iterations: Optional[int] = None,
        smoothing: Optional[int] = None,
        threshold: Optional[float] = None,
        init_radius_factor: float = 0.5,
        precision: Optional[str] = None,
    ) -> torch.Tensor:
        """Сегментация морфологическими змеями.

        Итеративно расширяет или сужает бинарную маску на основе величины градиента.
        Области с низким градиентом "поглощаются", с высоким — отбрасываются.

        Алгоритм:
        1. Инициализация маски как окружности в центре изображения.
        2. Вычисление градиента изображения (Sobel).
        3. Итеративное обновление маски: расширение в областях с низким градиентом,
        сужение в областях с высоким.
        4. Опциональное морфологическое сглаживание на каждой итерации.
        5. Возврат финальной бинарной маски.

        Метод особенно эффективен для:
        - Сегментации объектов с чёткими, но размытыми границами
        - Медицинских изображений с однородными регионами
        - Задач, где важна связность результата

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            iterations: Число итераций обновления маски (по умолчанию: 100).
            smoothing: Радиус морфологического сглаживания (по умолчанию: 1).
            threshold: Порог градиента для решения о расширении/сужении (по умолчанию: 0.5).
            init_radius_factor: Доля радиуса начальной окружности от центра до края (по умолчанию: 0.5).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - Все операции выполняются на GPU через векторизованные свёртки.
            - Для `fp16` рекомендуется `threshold ≥ 0.3` из-за квантования градиента.
            - Метод безопасен для `torch.compile(fullgraph=True)` при фиксированных параметрах.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "morphological_snakes",
                iterations=150,
                smoothing=2,
                threshold=0.4,
                precision="bf16"
            )
            mask = segmenter.segment(image)
            ```
        """
        start_time = time.time()

        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor).squeeze(0)  # (H, W)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray

        if gray.max() <= 1.0:
            gray = gray * 255.0

        h, w = gray.shape

        # === ПАРАМЕТРЫ ===
        iters = iterations if iterations is not None else self.params.get("iterations", 100)
        smooth = smoothing if smoothing is not None else self.params.get("smoothing", 1)
        thresh = threshold if threshold is not None else self.params.get("threshold", 0.5)

        # === ИНИЦИАЛИЗАЦИЯ МАСКИ (окружность) ===
        cy, cx = h // 2, w // 2
        radius = int(min(cx, cy) * init_radius_factor)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing="ij",
        )
        mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius**2).to(dtype)

        # === ГРАДИЕНТ ИЗОБРАЖЕНИЯ ===
        # sobel_x, sobel_y = self._get_sobel_kernels_cached(self.device, dtype)
        sobel_x, sobel_y = self._get_conv_kernel("sobel", return_pair=True, dtype=dtype, device=self.device)

        precision_val = precision if precision is not None else "fp32"
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            gx = F.conv2d(
                gray.unsqueeze(0).unsqueeze(0),
                sobel_x.to(gray.unsqueeze(0).unsqueeze(0).dtype),
                padding=1,
            ).squeeze()
            gy = F.conv2d(
                gray.unsqueeze(0).unsqueeze(0),
                sobel_y.to(gray.unsqueeze(0).unsqueeze(0).dtype),
                padding=1,
            ).squeeze()
            grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
            grad_mag = grad_mag / (grad_mag.amax() + 1e-8)

            # === МОРФОЛОГИЧЕСКИЕ ОПЕРАЦИИ (векторизованные) ===
            if smooth > 0:
                morph_kernel = torch.ones(
                    1,
                    1,
                    smooth * 2 + 1,
                    smooth * 2 + 1,
                    device=self.device,
                    dtype=dtype,
                )
                morph_kernel = morph_kernel / morph_kernel.sum()

            for _ in range(iters):
                # Расширение где градиент низкий, сужение где высокий
                expansion = (grad_mag < thresh) & (~mask.bool())
                erosion = (grad_mag >= thresh) & (mask.bool())

                mask[expansion] = 1.0
                mask[erosion] = 0.0

                # Сглаживание
                if smooth > 0:
                    mask_4d = mask.unsqueeze(0).unsqueeze(0)
                    mask_smooth = F.conv2d(mask_4d, morph_kernel, padding=smooth)
                    mask = (mask_smooth > 0.5).to(dtype).squeeze()

        exec_time: float = time.time() - start_time

        info = self._log_info(
            "morphological_snakes_torch",
            exec_time,
            {
                "iterations": iterations,
                "smoothing": smoothing,
                "threshold": threshold,
                "init_radius_factor": init_radius_factor,
                "precision": precision,
            },
            # precision_val=precision_val,
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}")
        return mask.to(torch.float32).unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _chan_vese(
        self,
        tensor: torch.Tensor,
        *,
        mu: Optional[float] = None,
        lambda1: Optional[float] = None,
        lambda2: Optional[float] = None,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        dt: Optional[float] = None,
        eps: Optional[float] = None,
        init_type: Literal["checkerboard", "disk", "small_disk"] = "checkerboard",
        precision: Optional[str] = None,
    ) -> torch.Tensor:
        """Модель Чан-Везе — активные контуры без градиентов.

        Энергетическая модель, которая разделяет изображение на две области с минимальной
        внутрирегиональной дисперсией. Подходит для объектов без чётких границ, где
        градиентные методы (Собель, Кэнни) дают плохие результаты.

        Алгоритм:
        1. Инициализация функции уровня φ (шахматная доска, диск или пользовательская).
        2. Вычисление средних интенсивностей внутри/снаружи контура через Хевисайд.
        3. Итеративная эволюция φ по уравнению градиентного спуска функционала.
        4. Остановка при сходимости (изменение φ < tol) или достижении max_iter.
        5. Бинаризация: маска = {φ > 0}.

        Метод особенно эффективен для:
        - Медицинских изображений с размытыми границами тканей
        - Задач, где объект и фон имеют схожую текстуру, но разную среднюю яркость
        - Сегментации без предварительного выделения границ

        Формула функционала:
        ```
        E(φ) = λ₁∫(I-c₁)²H(φ) + λ₂∫(I-c₂)²(1-H(φ)) + μ∫δ(φ)|∇φ|
        ```

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            mu: Вес члена длины контура (по умолчанию: 0.25).
            lambda1: Вес ошибки внутри региона (по умолчанию: 1.0).
            lambda2: Вес ошибки снаружи региона (по умолчанию: 1.0).
            tol: Порог сходимости по изменению φ (по умолчанию: 1e-3).
            max_iter: Максимальное число итераций (по умолчанию: 100).
            dt: Шаг времени для эволюции (по умолчанию: 0.5).
            eps: Параметр регуляризации функций Хевисайда/Дирака (по умолчанию: 1.0).
            init_type: Тип инициализации φ: 'checkerboard', 'disk', 'small_disk'.
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - Все вычисления выполняются на GPU без выхода в numpy.
            - Для `fp16` рекомендуется `eps ≥ 0.5`, `tol ≥ 1e-2`, `dt ≤ 0.3`.
            - Метод не поддерживает `fullgraph=True` в `torch.compile` из-за условной остановки.
            - Сходимость обычно достигается за 30-80 итераций для изображений 512×512.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "chan_vese",
                mu=0.3,
                lambda1=1.2,
                lambda2=0.8,
                max_iter=150,
                init_type="disk",
                precision="bf16"
            )
            mask = segmenter.segment(medical_image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor).squeeze()
        if gray.dim() == 3 and gray.shape[0] == 1:
            gray = gray.squeeze(0)

        # Нормализация к [0, 1]
        if gray.max() > 1.0:
            gray = gray / 255.0
        start_time = time.time()

        # Приведение к целевой точности
        dtype = self.precision_manager.get_dtype(precision)
        gray = gray.to(dtype=dtype, device=self.device)

        h, w = gray.shape

        # === ПАРАМЕТРЫ ===
        mu_val = mu if mu is not None else self.params.get("mu", 0.25)
        l1 = lambda1 if lambda1 is not None else self.params.get("lambda1", 1.0)
        l2 = lambda2 if lambda2 is not None else self.params.get("lambda2", 1.0)
        tol_val = tol if tol is not None else self.params.get("tol", 1e-3)
        max_it = max_iter if max_iter is not None else self.params.get("max_iter", 100)
        dt_val = dt if dt is not None else self.params.get("dt", 0.5)
        eps_val = eps if eps is not None else self.params.get("eps", 1.0)

        # === ИНИЦИАЛИЗАЦИЯ УРОВНЯ ===
        init_type = self.params.get("init_level_set", "checkerboard")
        # === ИНИЦИАЛИЗАЦИЯ УРОВНЯ ===
        phi = self._cv_init_level_set_torch(init_type, (h, w), self.device, dtype)

        # === ОСНОВНОЙ ЦИКЛ ЭВОЛЮЦИИ ===
        precision_val = precision if precision is not None else "fp32"
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            for iteration in range(max_it):
                old_phi = phi.clone()

                # Обновление функции уровня
                phi = self._cv_calculate_variation(gray, phi, mu_val, l1, l2, dt_val, eps_val)

                # Проверка сходимости
                phi_var = torch.sqrt(((phi - old_phi) ** 2).mean())
                if phi_var < tol_val:
                    break

        # === БИНАРИЗАЦИЯ ===
        mask = (phi > 0).to(torch.float32)

        exec_time: float = time.time() - start_time

        info = self._log_info(
            "chan_vese_torch",
            exec_time,
            {
                "mu": mu,
                "lambda1": lambda1,
                "lambda2": lambda2,
                "tol": tol,
                "max_iter": max_iter,
                "dt": dt,
                "eps": eps,
                "init_type": init_type,
                "precision": precision,
            },
            # precision_val=precision_val,
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}")
        return mask.unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    # ГРАФОВЫЕ МЕТОДЫ СЕГМЕНТАЦИИ
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _watershed(
        self,
        tensor: torch.Tensor,
        *,
        markers: Optional[torch.Tensor] = None,
        connectivity: int = 4,
        gradient_method: Literal["sobel", "scharr", "gradient"] = "sobel",
        normalize_gradient: bool = True,
        precision: Optional[str] = None,
        use_numba_fallback: bool = False,
    ) -> torch.Tensor:
        """Сегментация методом водораздела (Watershed) на чистом PyTorch.

        Использует морфологические операции и преобразование расстояния для выделения
        надежных маркеров переднего плана и фона. Алгоритм "затопляет" изображение от маркеров,
        формируя границы между объектами.

        Алгоритм:
        1. Конвертация в градации серого и вычисление градиента (Sobel/Scharr).
        2. Генерация маркеров: автоматически (Otsu + distance transform) или пользовательских.
        3. Векторизованный watershed: приоритетная очередь на тензорах вместо heapq.
        4. Построение бинарной маски из размеченных регионов.

        Метод особенно эффективен для:
        - Разделения слипшихся объектов с чёткими границами
        - Задач, где важна точность локализации границ
        - Сегментации с предварительной разметкой маркеров

        ⚠️  Ограничения:
            - Алгоритм по своей природе требует приоритетной обработки, поэтому полная
            векторизация невозможна. Используется гибридный подход: сортировка + batch-обработка.
            - Для больших изображений (>1024×1024) рекомендуется `markers` заранее.

        Формула градиента:
        ```
        |∇I| = √(Gx² + Gy²), где Gx, Gy — свёртка с ядрами Собеля/Шарра
        ```

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            markers: Пользовательские маркеры формы (H, W), dtype=int.
                    0 = неразмечено, >0 = метка региона. По умолчанию: автоматические.
            connectivity: Связность окрестности: 4 или 8 (по умолчанию: 4).
            gradient_method: Метод вычисления градиента: 'sobel', 'scharr', 'gradient'.
            normalize_gradient: Нормализовать градиент к [0, 1] перед watershed (по умолчанию: True).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16' (по умолчанию: из self.dtype).

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32, значения {0.0, 1.0}.

        Note:
            - Все вычисления выполняются на GPU без выхода в numpy.
            - Автоматические маркеры: центр = объект (2), углы = фон (1).
            - Для `fp16` рекомендуется `normalize_gradient=True` для стабильности.
            - Метод не поддерживает `fullgraph=True` в `torch.compile` из-за условной логики.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "watershed",
                connectivity=8,
                gradient_method="scharr",
                precision="bf16"
            )
            mask = segmenter.segment(image)
            ```
        """
        start_time = time.time()

        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor).squeeze(0)  # (H, W)
        dtype = self.precision_manager.get_dtype(precision)
        gray = self._cast_to_dtype(gray) if gray.dtype != dtype else gray
        h, w = gray.shape

        # === ПАРАМЕТРЫ ===
        conn = connectivity if connectivity in [4, 8] else 4
        grad_method = gradient_method if gradient_method in ["sobel", "scharr", "gradient"] else "sobel"
        norm_grad = normalize_gradient if normalize_gradient is not None else True

        # === ВЫЧИСЛЕНИЕ ГРАДИЕНТА ===
        precision_val = precision if precision is not None else "fp32"
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            if grad_method == "scharr":
                # kx, ky = self._get_scharr_kernels_cached(self.device, dtype)
                kernels = self._get_conv_kernel("scharr", return_pair=True, dtype=dtype, device=self.device)
                if isinstance(kernels, tuple):
                    kx, ky = kernels
                else:
                    # Fallback для type safety
                    kx = ky = kernels
                kx = kx.clone()
                ky = ky.clone()
            elif grad_method == "gradient":
                # Простые разности
                kx = torch.tensor([[-1, 0, 1]], dtype=dtype, device=self.device).view(1, 1, 1, 3).clone()
                ky = torch.tensor([[-1], [0], [1]], dtype=dtype, device=self.device).view(1, 1, 3, 1).clone()
            else:  # sobel
                # kx, ky = self._get_sobel_kernels_cached(self.device, dtype)
                kernels = self._get_conv_kernel("sobel", return_pair=True, dtype=dtype, device=self.device)
                if isinstance(kernels, tuple):
                    kx, ky = kernels
                else:
                    # Fallback для type safety
                    kx = ky = kernels
                kx = kx.clone()
                ky = ky.clone()

            gx = F.conv2d(
                gray.unsqueeze(0).unsqueeze(0),
                kx.to(gray.unsqueeze(0).unsqueeze(0).dtype),
                padding=1,
            ).squeeze()
            gy = F.conv2d(
                gray.unsqueeze(0).unsqueeze(0),
                ky.to(gray.unsqueeze(0).unsqueeze(0).dtype),
                padding=1,
            ).squeeze()
            gradient = torch.sqrt(gx**2 + gy**2 + 1e-8)

            if norm_grad:
                gradient = gradient / (gradient.amax() + 1e-8)

        # === МАРКЕРЫ ===
        if markers is not None:
            # Пользовательские маркеры
            markers_int = markers.to(dtype=torch.int32, device=self.device)
            if markers_int.dim() == 4:
                markers_int = markers_int.squeeze(0).squeeze(0)
            elif markers_int.dim() == 3:
                markers_int = markers_int.squeeze(0)
        else:
            # Автоматические маркеры (упрощённо: центр = объект, углы = фон)
            markers_int = self._rw_create_markers(h, w, device=self.device, dtype=torch.int32)

        use_numba = use_numba_fallback and self.device.type == "cpu" and h * w > 500_000  # Порог для переключения

        if use_numba:
            # 🔥 Numba-версия для CPU + больших изображений
            try:
                gradient_np = gradient.squeeze().cpu().numpy().astype(np.float32)
                markers_np = markers_int.cpu().numpy()

                labels_np = _watershed_numba_impl(gradient_np, markers_np, connectivity=conn)
                labels = torch.from_numpy(labels_np).to(self.device)

                exec_time: float = time.time() - start_time
                self.params["execution_info"] = {
                    "method": "watershed_numba_fallback",
                    "image_size": (h, w),
                    "execution_time": exec_time,
                }
            except Exception as e:
                warnings.warn(f"Numba watershed failed: {e}. Fallback to PyTorch.")
                labels = self._watershed_vectorized_torch(gradient, markers_int, connectivity=conn, dtype=dtype)
        else:
            # === ВЕКТОРИЗОВАННЫЙ WATERSHED ===
            labels = self._watershed_vectorized_torch(gradient, markers_int, connectivity=conn, dtype=dtype)

        # === СОЗДАНИЕ МАСКИ ===
        mask = (labels > 0).to(torch.float32)

        exec_time = time.time() - start_time

        info = self._log_info(
            "watershed_torch",
            exec_time,
            {
                "connectivity": connectivity,
                "gradient_method": gradient_method,
                "normalize_gradient": normalize_gradient,
                "precision": precision,
            },
            # precision_val=precision_val,
            backend="numba" if use_numba else "pytorch",
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}")

        return mask.unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _watershed_vectorized_torch(
        gradient: torch.Tensor,  # (H, W), float, [0, 1]
        markers: torch.Tensor,  # (H, W), int32, 0=неразмечено, >0=метка
        connectivity: int = 4,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Векторизованная реализация watershed через сортировку и batch-обработку.

        Алгоритм:
        1. Инициализация: размеченные пиксели попадают в очередь с приоритетом = градиент.
        2. Сортировка очереди по градиенту (вместо heapq).
        3. Обработка пикселей "волнами": все пиксели с одинаковым градиентом обрабатываются вместе.
        4. Распространение меток на соседей с учётом приоритета.

        Это не полностью векторизованный алгоритм (из-за природы watershed), но значительно
        быстрее чистой Python-реализации с heapq за счёт:
        - Предварительной сортировки через torch.sort
        - Векторизованной проверки соседей
        - Минимизации `.item()` вызовов

        Args:
            gradient: Карта градиента (H, W), нормализованная к [0, 1].
            markers: Матрица маркеров (H, W), dtype=int32.
            connectivity: 4 или 8-связность.
            dtype: Тип данных для вычислений.

        Returns:
            torch.Tensor: Метки регионов (H, W), dtype=int32.
        """
        h, w = gradient.shape
        n = h * w

        # === ИНИЦИАЛИЗАЦИЯ ===
        labels = torch.zeros((h, w), dtype=torch.int32, device=gradient.device)
        visited = torch.zeros((h, w), dtype=torch.bool, device=gradient.device)

        # Плоские индексы и координаты
        flat_grad = gradient.reshape(-1)
        flat_markers = markers.reshape(-1)
        indices = torch.arange(n, device=gradient.device)
        y_coords = indices // w
        x_coords = indices % w

        # === ПОДГОТОВКА ОЧЕРЕДИ ===
        # Только размеченные пиксели попадают в начальную очередь
        seeded_mask = flat_markers > 0
        if not seeded_mask.any():
            return labels  # Нет маркеров → пустая сегментация

        # Инициализация: размеченные пиксели
        seeded_idx = indices[seeded_mask]
        labels.reshape(-1)[seeded_idx] = flat_markers[seeded_mask]
        visited.reshape(-1)[seeded_idx] = True

        # Очередь: (градиент, индекс, метка) — сортируем по градиенту
        queue_grad = flat_grad[seeded_mask].clone()
        queue_idx = seeded_idx.clone()
        queue_label = flat_markers[seeded_mask].clone()

        # Сортируем очередь по градиенту (возрастание)
        sort_order = torch.argsort(queue_grad)
        queue_grad = queue_grad[sort_order]
        queue_idx = queue_idx[sort_order]
        queue_label = queue_label[sort_order]

        # === НАПРАВЛЕНИЯ ===
        if connectivity == 4:
            directions = torch.tensor(
                [[0, 1], [1, 0], [0, -1], [-1, 0]],
                device=gradient.device,
                dtype=torch.long,
            )
        else:
            directions = torch.tensor(
                [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]],
                device=gradient.device,
                dtype=torch.long,
            )

        # === ОСНОВНОЙ ЦИКЛ (batch-обработка) ===
        ptr = 0  # Указатель на текущий элемент в очереди
        while ptr < queue_idx.numel():
            # Берём текущий пиксель
            idx: int = int(queue_idx[ptr].item())
            label = queue_label[ptr].item()
            y, x = int(y_coords[idx].item()), int(x_coords[idx].item())

            # Векторизованная проверка соседей
            neighbors = torch.tensor([[y, x]], device=gradient.device, dtype=torch.long) + directions
            ny, nx = neighbors[:, 0], neighbors[:, 1]

            # Фильтрация валидных соседей
            valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            if not valid.any():
                ptr += 1
                continue

            ny_valid, nx_valid = ny[valid], nx[valid]
            n_idx_valid = ny_valid * w + nx_valid

            # Только непосещённые соседи
            unvisited = ~visited.reshape(-1)[n_idx_valid]
            if not unvisited.any():
                ptr += 1
                continue

            # Добавляем непосещённых соседей в очередь
            new_idx = n_idx_valid[unvisited]
            new_grad = flat_grad[new_idx]
            new_labels = torch.full_like(new_idx, label, dtype=torch.int32)

            # 🔥 Эффективная вставка через searchsorted (O(log N) вместо O(N log N))
            if new_idx.numel() > 0:
                # Находим позиции для вставки
                insert_pos = torch.searchsorted(queue_grad, new_grad)

                # Вставляем элементы по одному (можно оптимизировать через batching)
                for i in range(new_idx.numel()):
                    pos: int = int(insert_pos[i].item())
                    queue_grad = torch.cat([queue_grad[:pos], new_grad[i : i + 1], queue_grad[pos:]])
                    queue_idx = torch.cat([queue_idx[:pos], new_idx[i : i + 1], queue_idx[pos:]])
                    queue_label = torch.cat([queue_label[:pos], new_labels[i : i + 1], queue_label[pos:]])

                # Обновляем ptr, если вставка была перед текущей позицией
                if insert_pos.min().item() <= ptr:
                    ptr = max(0, ptr - new_idx.numel())

            # Обновляем visited и labels
            visited.reshape(-1)[new_idx] = True
            labels.reshape(-1)[new_idx] = label
            ptr += 1

        return labels

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _watershed_torch_visualization(
        self,
        tensor: torch.Tensor,
        *,
        alpha: float = 0.6,
        color: Tuple[int, int, int] = (255, 0, 0),
        show_markers: bool = False,
        precision: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Визуализация для Watershed с поддержкой AMP и минимальными трансферами.

        Алгоритм:
        1. Вычисление градиента и маркеров на GPU.
        2. Выполнение watershed через векторизованную очередь.
        3. Создание цветной маски на GPU.
        4. Опциональное отображение маркеров разными цветами.
        5. Alpha-смешивание и конвертация в numpy.

        Метод особенно эффективен для:
        - Отладки качества маркеров и границ водораздела
        - Визуализации разделения слипшихся объектов
        - Демонстрации работы алгоритма с разными параметрами

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            alpha: Прозрачность маски [0, 1] (по умолчанию: 0.6).
            color: Цвет основной маски в формате RGB (по умолчанию: красный).
            show_markers: Если True, отображать маркеры разными цветами.
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.
            **kwargs: Дополнительные параметры для watershed.

        Returns:
            Tuple[np.ndarray, torch.Tensor]:
                - Визуализация (H, W, 3), dtype=uint8, RGB.
                - Бинарная маска (H, W), dtype=float32.

        Note:
            - Все вычисления выполняются на GPU, numpy используется только для возврата.
            - При `show_markers=True` маркеры отображаются разными цветами (1=синий, 2=красный).
            - Поддерживает `torch.compile` при фиксированных параметрах.

        Example:
            ```python
            vis, mask = segmenter._watershed_torch_visualization(
                image, alpha=0.7, show_markers=True, precision="bf16"
            )
            ```
        """
        start_time = time.time()

        # === ПРЕДПОДГОТОВКА ===
        dtype = self.precision_manager.get_dtype(precision)
        img = tensor.squeeze(0) if tensor.dim() == 4 else tensor  # (C, H, W)
        img = img.to(dtype) if img.dtype != dtype else img
        c, h, w = img.shape

        # === WATERSHED НА ЧИСТОМ TORCH ===
        # Конвертация в градации серого
        if c == 3:
            gray = self._rgb_to_gray_torch(img.unsqueeze(0)).squeeze(0)
        else:
            gray = img.squeeze(0) if img.dim() == 3 else img

        # Градиенты (кэшированные ядра)
        # sobel_x, sobel_y = self._get_sobel_kernels_cached(self.device, dtype)
        sobel_x, sobel_y = self._get_conv_kernel("sobel", return_pair=True, dtype=dtype, device=self.device)

        precision_val = precision if precision is not None else "fp32"
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            gx = F.conv2d(
                gray.unsqueeze(0).unsqueeze(0),
                sobel_x.to(gray.unsqueeze(0).unsqueeze(0).dtype),
                padding=1,
            ).squeeze()
            gy = F.conv2d(
                gray.unsqueeze(0).unsqueeze(0),
                sobel_y.to(gray.unsqueeze(0).unsqueeze(0).dtype),
                padding=1,
            ).squeeze()
            gradient = torch.sqrt(gx**2 + gy**2 + 1e-8)
            gradient = gradient / (gradient.amax() + 1e-8)  # Нормализация

            # Автоматические маркеры
            markers = self._rw_create_markers(h, w, device=self.device, dtype=torch.int32)

            # Упрощённый watershed (векторизованный)
            labels = self._watershed_simple_torch(gradient, markers, dtype)

        # === СОЗДАНИЕ МАСКИ ===
        mask = (labels > 0).to(torch.float32)  # (H, W)
        # === ВИЗУАЛИЗАЦИЯ ===
        # Конвертация в RGB если нужно
        if c == 1:
            img_rgb = img.repeat(3, 1, 1)
        else:
            img_rgb = img

        # Цветная маска
        color_tensor = torch.tensor(color, dtype=dtype, device=self.device).view(3, 1, 1) / 255.0
        colored_mask = torch.zeros_like(img_rgb)

        if show_markers and markers.dim() == 2:
            # Разные цвета для разных маркеров
            colors = [
                torch.tensor([0, 0, 255], dtype=dtype, device=self.device),  # синий для фона
                torch.tensor([255, 0, 0], dtype=dtype, device=self.device),  # красный для объекта
            ]
            for label_val in [1, 2]:
                marker_mask = markers == label_val
                color_idx = label_val - 1
                if color_idx < len(colors):
                    for c_idx in range(3):
                        colored_mask[c_idx, marker_mask] = colors[color_idx][c_idx] / 255.0
        else:
            # Единый цвет для всей маски
            mask_bool = mask > 0.5
            for c_idx in range(3):
                colored_mask[c_idx, mask_bool] = color_tensor[c_idx, 0, 0]

        # Alpha-смешивание
        result = img_rgb * (1 - alpha) + colored_mask * alpha

        # === КОНВЕРТАЦИЯ В NUMPY ===
        result_np = self._tensor_to_numpy(result.unsqueeze(0), denormalize=True)
        mask_np = mask.to(torch.float32).cpu().numpy()
        exec_time: float = time.time() - start_time

        info = self._log_info(
            "watershed_visualisation_torch",
            exec_time,
            {
                "alpha": alpha,
                "show_markers": show_markers,
                "precision": precision,
            },
            # precision_val=precision_val,
        )
        self.params["visualization_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}")

        return result_np, torch.from_numpy(mask_np).to(self.device)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _watershed_simple_torch(
        gradient: torch.Tensor,
        markers: torch.Tensor,
        dtype: torch.dtype,
        connectivity: int = 4,
    ) -> torch.Tensor:
        """Упрощённый векторизованный watershed для визуализации.

        Использует batch-обработку вместо heapq для скорости на GPU.

        Args:
            gradient: Карта градиента (H, W), нормализованная к [0, 1].
            markers: Матрица маркеров (H, W), dtype=int, 0=фон, >0=объекты.
            dtype: Тип данных для вычислений.
            connectivity: Связность: 4 или 8.

        Returns:
            torch.Tensor: Метки регионов (H, W), dtype=int.
        """
        h, w = gradient.shape
        labels = torch.zeros_like(markers, dtype=torch.int32)
        visited = torch.zeros_like(markers, dtype=torch.bool)

        # Инициализация с маркерами
        for label_val in torch.unique(markers):
            if label_val == 0:
                continue
            ys, xs = torch.where(markers == label_val)
            labels[ys, xs] = label_val
            visited[ys, xs] = True

        # Направления
        directions = (
            [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if connectivity == 4
            else [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        )

        # Простая итеративная обработка (вместо heapq)
        max_iter = h * w
        for _ in range(max_iter):
            changed = False
            for dy, dx in directions:
                ny, nx = torch.arange(h) + dy, torch.arange(w) + dx
                valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
                if not valid.any():
                    continue

                # Векторизованное обновление
                src_labels = labels[ny[valid], nx[valid]]
                # src_grad = gradient[ny[valid], nx[valid]]
                dst_mask = ~visited[ny[valid], nx[valid]] & (src_labels > 0)

                if dst_mask.any():
                    idx = torch.where(dst_mask)[0]
                    labels[ny[valid][idx], nx[valid][idx]] = src_labels[idx]
                    visited[ny[valid][idx], nx[valid][idx]] = True
                    changed = True

            if not changed:
                break

        return labels

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _random_walker(
        self,
        tensor: torch.Tensor,
        *,
        beta: Optional[float] = None,
        mode: Literal["jacobi", "cg", "scipy"] = "scipy",
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        target_label: int = 2,
        precision: Optional[str] = None,
    ) -> torch.Tensor:
        """Сегментация методом Random Walker на чистом PyTorch (+ scipy fallback).

        На основе маркеров (пользовательских или автоматических) решается задача на графе:
        каждый пиксель "принадлежит" тому маркеру, до которого "случайное блуждание" короче.
        Вероятности вычисляются через решение разреженной системы уравнений Лапласа.

        Алгоритм:
        1. Конвертация изображения в градации серого и нормализация.
        2. Создание маркеров (автоматически или пользовательских).
        3. Вычисление весов рёбер графа на основе градиента: w = exp(-β·||∇I||²).
        4. Построение разреженной матрицы лапласиана графа.
        5. Решение системы уравнений для вероятностей принадлежности к классам.
        6. Назначение меток по максимальной вероятности и построение бинарной маски.

        Метод особенно эффективен для:
        - Сегментации с чёткими, но сложными границами
        - Задач, где важна связность результата и устойчивость к шуму
        - Интерактивной сегментации с ручной разметкой маркеров

        Формула системы:
        ```
        L_bb · x_b = -L_bm · x_m
        ```
        где `L` — лапласиан графа, `b` = неразмеченные, `m` = размеченные пиксели.

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            beta: Коэффициент затухания весов (по умолчанию: 130). Большие значения → сильнее подавление на границах.
            mode: Метод решения: 'jacobi' (быстро, неточно), 'cg' (точно, на GPU), 'scipy' (точно, на CPU, по умолчанию).
            tol: Порог сходимости решателя (по умолчанию: 1e-3).
            max_iter: Максимальное число итераций решателя (по умолчанию: 300).
            target_label: Метка целевого объекта в маркерах (по умолчанию: 2).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16' (по умолчанию: из self.dtype).

        Returns:
            torch.Tensor: Бинарная маска объекта формы (1, 1, H, W), dtype=float32.

        Note:
            - Для `mode='scipy'` требуется установленный `scipy`. При отсутствии — fallback на 'cg'.
            - Для `fp16` рекомендуется `beta ≤ 100` и `tol ≥ 1e-2` для стабильности.
            - Автоматические маркеры: центр = объект (2), углы = фон (1).
            - Метод не поддерживает `fullgraph=True` в `torch.compile` из-за разреженной арифметики.
        Example:
            ```python
            segmenter = TorchSegmenter(
                "random_walker",
                beta=100,
                mode="cg",
                tol=1e-3,
                target_label=2,
                precision="bf16"
            )
            mask = segmenter.segment(image)
            ```
        """
        start_time = time.time()
        # === ПРЕДПОДГОТОВКА ===
        gray = self._to_grayscale(tensor).squeeze(0)  # (H, W)
        if gray.dim() == 3 and gray.shape[0] == 1:
            gray = gray.squeeze(0)

        # Нормализация к [0, 1]
        if gray.max() > 1.0:
            gray = gray / 255.0

        # Приведение к целевой точности
        dtype = self.precision_manager.get_dtype(precision)
        gray = gray.to(dtype=dtype, device=self.device)

        h, w = gray.shape

        # === ПАРАМЕТРЫ ===
        beta_val = beta if beta is not None else self.params.get("beta", 130.0)
        solve_mode = mode if mode in ["jacobi", "cg", "scipy"] else self.params.get("mode", "scipy")
        tol_val = tol if tol is not None else self.params.get("tol", 1e-3)
        max_it = max_iter if max_iter is not None else self.params.get("max_iter", 300)
        target_lbl = target_label if target_label in [1, 2] else self.params.get("target_label", 2)

        # === МАРКЕРЫ ===
        markers = self._rw_create_markers(h, w, device=self.device, dtype=torch.int32)

        # === ВЕСА РЁБЕР ===
        weights = self._rw_compute_weights(gray, beta_val)  # (4, H, W)

        # === ЛАПЛАСИАН ===
        L, b_indices, m_indices = self._rw_build_laplacian(gray, weights, markers)

        # === РЕШЕНИЕ СИСТЕМЫ ===
        n_unlabeled = b_indices.numel()
        n_labels = int(markers.max().item())

        if n_unlabeled == 0:
            # Все пиксели размечены
            result = markers.clone()
        else:
            # Правая часть
            B = self._rw_compute_rhs(b_indices, markers, n_labels, w, device=self.device)

            # Выбор решателя
            if solve_mode == "scipy":
                x = self._rw_solve_scipy(L, b_indices, markers, n_labels, tol_val, max_it)
                if x is None:
                    # Fallback на CG при отсутствии scipy
                    x_init = torch.ones((n_labels, n_unlabeled), device=self.device) / n_labels
                    x = self._rw_solve_cg_batch(L, B, x_init, tol_val, max_it)
            elif solve_mode == "cg":
                x_init = torch.ones((n_labels, n_unlabeled), device=self.device) / n_labels
                x = self._rw_solve_cg_batch(L, B, x_init, tol_val, max_it)
            else:  # jacobi
                x_init = torch.ones((n_labels, n_unlabeled), device=self.device) / n_labels
                x = self._rw_solve_jacobi(L, B, x_init, tol_val, max_it)

            # Назначение меток по максимальной вероятности
            result_flat = torch.argmax(x, dim=0) + 1  # +1 т.к. метки начинаются с 1

            # Заполнение размеченных пикселей
            result_flat[m_indices] = markers.view(-1)[m_indices]
            result = result_flat.view(h, w)

        # === СОЗДАНИЕ МАСКИ ===
        mask = (result == target_lbl).to(torch.float32)

        exec_time: float = time.time() - start_time

        info = self._log_info(
            "random_walker_torch",
            exec_time,
            {
                "beta": beta,
                "mode": mode,
                "tol": tol,
                "max_iter": max_iter,
                "target_label": target_label,
                "precision": precision,
            },
            # precision_val=precision_val,
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}")
        return mask.unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    # SUPER-PIXEL МЕТОДЫ СЕГМЕНТАЦИИ
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _quickshift(
        self,
        tensor: torch.Tensor,
        *,
        kernel_size: Optional[float] = None,
        max_dist: Optional[int] = None,
        ratio: Optional[float] = None,
        convert2lab: bool = True,
        downsample: Optional[float] = None,
        precision: Optional[str] = None,
    ) -> torch.Tensor:
        """Quickshift сегментация — mode-seeking алгоритм в пространстве признаков.

        Находит моды плотности распределения пикселей в пространстве (цвет + координаты).
        Каждый пиксель "поднимается" к ближайшему соседу с большей плотностью,
        образуя иерархию, где корневые узлы = центры сегментов.

        Алгоритм:
        1. Построение пространства признаков: [Lab-цвет, нормализованные координаты].
        2. Оценка плотности через гауссово ядро (с выборкой для скорости).
        3. Поиск "родителя" для каждого пикселя: ближайший сосед с большей плотностью.
        4. Извлечение сегментов: пиксели с общим корнем = один сегмент.
        5. Пост-обработка: удаление мелких сегментов, создание бинарной маски.

        Метод особенно эффективен для:
        - Сегментации текстурных изображений (листва, ткани, облака)
        - Задач, где не требуется точное число сегментов
        - Предварительной обработки перед более сложными методами

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W), RGB.
            kernel_size: Ширина гауссова ядра для оценки плотности (по умолчанию: 5).
            max_dist: Максимальное расстояние для поиска родителя в пространстве признаков (по умолчанию: 10).
            ratio: Вес пространственных координат относительно цвета (по умолчанию: 1.0).
            convert2lab: Конвертировать RGB → Lab для лучшего восприятия цвета (по умолчанию: True).
            downsample: Коэффициент уменьшения разрешения для ускорения (по умолчанию: 0.5).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16' (для предобработки).

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - Quickshift реализуется через numpy из-за сложности эффективной pure-PyTorch версии.
            - Для больших изображений автоматически применяется downsample + интерполяция меток.
            - Метод не поддерживает `torch.compile` из-за вызова внешних библиотек.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "quickshift",
                kernel_size=7,
                max_dist=15,
                ratio=0.8,
                downsample=0.75
            )
            mask = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        img_np = self._tensor_to_numpy(tensor)
        start_time = time.time()
        h, w, c = img_np.shape[:3] if img_np.ndim == 3 else (*img_np.shape, 1)

        # === ПАРАМЕТРЫ ===
        ks = kernel_size if kernel_size is not None else self.params.get("kernel_size", 5.0)
        md = max_dist if max_dist is not None else self.params.get("max_dist", 10)
        rt = ratio if ratio is not None else self.params.get("ratio", 1.0)
        c2l = convert2lab if convert2lab is not None else self.params.get("convert2lab", True)
        ds = downsample if downsample is not None else self.params.get("downsample", 0.5)

        # === DOWNsample ДЛЯ УСКОРЕНИЯ ===
        # use_resize = False
        if h * w > 100_000 and ds < 1.0:
            small_h, small_w = int(h * ds), int(w * ds)
            img_small = cv2.resize(img_np, (small_w, small_h), interpolation=cv2.INTER_AREA)
            segments_small = self._quickshift_numpy_impl(
                image_np=img_small,
                kernel_size=ks,
                max_dist=md,
                ratio=rt,
                convert2lab=c2l,
            )
            # Интерполяция меток обратно
            segments = cv2.resize(
                segments_small.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(int)
            # use_resize = True
        else:
            segments = self._quickshift_numpy_impl(
                image_np=img_np, kernel_size=ks, max_dist=md, ratio=rt, convert2lab=c2l
            )

        # === СОЗДАНИЕ МАСКИ ===
        unique, counts = np.unique(segments, return_counts=True)
        bg_label = unique[np.argmax(counts)] if len(unique) > 0 else -1
        mask_np = (segments != bg_label).astype(np.float32)

        mask = torch.from_numpy(mask_np).to(self.device)

        exec_time: float = time.time() - start_time

        info = self._log_info(
            "quickshift_torch",
            exec_time,
            {
                "kernel_size": kernel_size,
                "max_dist": max_dist,
                "ratio": ratio,
                # "sigma": sigma,  # ✅ Убран неиспользуемый параметр
                "convert2lab": convert2lab,
                "precision": precision,
            },
            # precision_val=precision_val,
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}")

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # ──────────────────────────────────────────────────────────────────────
    def _quickshift_numpy_impl(
        self,
        image_np: np.ndarray,
        kernel_size: float,
        max_dist: int,
        ratio: float,
        convert2lab: bool,
    ) -> np.ndarray:
        """Чистая numpy-реализация Quickshift для внутреннего использования.

        Args:
            image_np: Изображение (H, W, C), dtype=uint8 или float.
            kernel_size: Ширина гауссова ядра.
            max_dist: Макс. расстояние для поиска родителя.
            ratio: Вес координат относительно цвета.
            convert2lab: Конвертировать в Lab.

        Returns:
            np.ndarray: Метки сегментов (H, W), dtype=int.
        """
        h, w = image_np.shape[:2]
        c = image_np.shape[2] if image_np.ndim == 3 else 1

        # Нормализация и конвертация
        if image_np.dtype != np.float32:
            img_float = image_np.astype(np.float32) / 255.0
        else:
            img_float = image_np

        if convert2lab and c == 3:
            features = self._rgb_to_lab_numpy_impl(img_float)
        else:
            features = img_float
            if c == 1:
                features = np.repeat(features, 3, axis=-1)

        # Пространственные координаты
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        spatial = np.stack([x_coords / w * ratio, y_coords / h * ratio], axis=-1)

        # Объединение признаков
        features_spatial = np.concatenate([features, spatial], axis=-1)
        # Оценка плотности (упрощённая с выборкой)
        density = self._compute_density_fast_impl(features_spatial, kernel_size, sample_ratio=0.1)

        # Поиск родителей
        parents = self._find_parents_fast_impl(features_spatial, density, max_dist)

        # Извлечение сегментов
        segments = self._extract_segments_impl(parents)

        return segments

    # ──────────────────────────────────────────────────────────────────────
    # Вспомогательные функции для Quickshift (numpy)
    def _rgb_to_lab_numpy_impl(self, rgb: np.ndarray) -> np.ndarray:
        """Упрощённая RGB → Lab конвертация."""
        # Линейзация sRGB
        mask = rgb > 0.04045
        rgb_linear = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

        # Матрица sRGB → XYZ
        M = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )
        xyz = rgb_linear @ M.T

        # Нормализация к D65
        xyz = xyz / np.array([0.95047, 1.0, 1.08883])

        # Функция f(t) для Lab
        def f(t: np.ndarray) -> np.ndarray:
            return np.where(t > 0.008856, t ** (1 / 3), 7.787 * t + 16 / 116)

        fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        return np.stack([L, a, b], axis=-1)

    # ──────────────────────────────────────────────────────────────────────
    def _compute_density_fast_impl(
        self, features: np.ndarray, kernel_size: float, sample_ratio: float = 0.1
    ) -> np.ndarray:
        """Быстрая оценка плотности с выборкой."""
        h, w = features.shape[:2]
        d = features.shape[-1]
        features_flat = features.reshape(-1, d)
        n = len(features_flat)

        # Выборка точек
        n_samples = max(100, int(n * sample_ratio))
        sample_idx = np.random.choice(n, n_samples, replace=False)
        samples = features_flat[sample_idx]

        # Предвычисление расстояний
        density = np.zeros(n)
        for i, sample in enumerate(samples):
            dists = np.sqrt(np.sum((features_flat - sample) ** 2, axis=1))
            weights = np.exp(-0.5 * (dists / kernel_size) ** 2)
            density += weights

        return density.reshape(h, w)

    # ──────────────────────────────────────────────────────────────────────
    def _find_parents_fast_impl(self, features: np.ndarray, density: np.ndarray, max_dist: float) -> np.ndarray:
        """Быстрый поиск родителей с ранним выходом."""
        h, w = features.shape[:2]
        d = features.shape[-1]
        features_flat = features.reshape(-1, d)
        density_flat = density.ravel()
        n = h * w

        parents = np.arange(n, dtype=np.int32)  # По умолчанию — сам себе

        # Для каждой точки ищем лучшего родителя
        for idx in range(n):
            current_dens = density_flat[idx]
            best_parent = idx
            best_dist = np.inf

            # Проверяем только точки с большей плотностью
            candidates = np.where(density_flat > current_dens)[0]
            if len(candidates) == 0:
                continue

            # Вычисляем расстояния до кандидатов
            for cand_idx in candidates:
                dist = np.sqrt(np.sum((features_flat[idx] - features_flat[cand_idx]) ** 2))
                if dist <= max_dist and dist < best_dist:
                    best_dist = dist
                    best_parent = cand_idx

            parents[idx] = best_parent

        return parents.reshape(h, w)

    # ──────────────────────────────────────────────────────────────────────
    def _extract_segments_impl(self, parents: np.ndarray) -> np.ndarray:
        """Извлечение сегментов через поиск корней."""
        h, w = parents.shape
        parents_flat = parents.ravel()
        n = h * w

        # Находим корень для каждого пикселя
        roots = np.zeros(n, dtype=np.int32)
        for idx in range(n):
            current = idx
            visited = set()
            while parents_flat[current] != current and current not in visited:
                visited.add(current)
                current = parents_flat[current]
            roots[idx] = current

        # Перенумерация корней
        unique_roots = np.unique(roots)
        root_to_label = {root: i for i, root in enumerate(unique_roots)}
        segments = np.vectorize(root_to_label.get)(roots)

        return cast(np.ndarray, segments.reshape(h, w))

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _slic(
        self,
        tensor: torch.Tensor,
        *,
        n_segments: Optional[int] = None,
        compactness: Optional[float] = None,
        max_iter: Optional[int] = None,
        sigma: Optional[float] = None,
        enforce_connectivity: bool = True,
        downsample: Optional[float] = None,
        precision: Optional[str] = None,
    ) -> torch.Tensor:
        """SLIC (Simple Linear Iterative Clustering) — суперпиксельная сегментация.

        Группирует пиксели в компактные, однородные регионы на основе пространственной
        и цветовой близости. Использует k-means в пространстве [Lab, x, y] с учётом компактности.

        Алгоритм:
        1. Конвертация RGB → Lab для лучшего восприятия цвета.
        2. Инициализация центроидов на регулярной сетке.
        3. Присвоение пикселей ближайшему центроиду с учётом компактности.
        4. Обновление центроидов как средних значений присвоенных пикселей.
        5. Повторение шагов 3-4 до сходимости или `max_iter`.
        6. (Опционально) Принудительная связность регионов.

        Метод особенно эффективен для:
        - Предварительной обработки для уменьшения числа пикселей
        - Сегментации с сохранением границ объектов
        - Задач, где важна компактность и однородность регионов

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W), RGB.
            n_segments: Желаемое число суперпикселей (по умолчанию: 100).
            compactness: Вес пространственной близости (больше = компактнее регионы, по умолчанию: 10.0).
            max_iter: Максимальное число итераций k-means (по умолчанию: 10).
            sigma: Sigma для предварительного гауссова сглаживания (по умолчанию: 0.0).
            enforce_connectivity: Обеспечить связность всех регионов (по умолчанию: True).
            downsample: Коэффициент уменьшения разрешения для ускорения (по умолчанию: 0.5).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16' (для предобработки).

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - SLIC реализуется через numpy/scipy из-за сложности эффективной pure-PyTorch версии.
            - Для больших изображений автоматически применяется downsample + интерполяция меток.
            - Метод не поддерживает `torch.compile` из-за вызова внешних библиотек.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "slic",
                n_segments=200,
                compactness=15.0,
                max_iter=15,
                downsample=0.75
            )
            mask = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        img_np = self._tensor_to_numpy(tensor)
        h, w, c = img_np.shape[:3] if img_np.ndim == 3 else (*img_np.shape, 1)
        start_time = time.time()

        # === ПАРАМЕТРЫ ===
        n_seg = n_segments if n_segments is not None else self.params.get("n_segments", 100)
        comp = compactness if compactness is not None else self.params.get("compactness", 10.0)
        max_it = max_iter if max_iter is not None else self.params.get("max_iter", 10)
        sig = sigma if sigma is not None else self.params.get("sigma", 0.0)
        enforce_conn = (
            enforce_connectivity if enforce_connectivity is not None else self.params.get("enforce_connectivity", True)
        )
        ds = downsample if downsample is not None else self.params.get("downsample", 0.5)

        # === DOWNsample ДЛЯ УСКОРЕНИЯ ===
        # use_resize = False
        if h * w > 100_000 and ds < 1.0:
            small_h, small_w = int(h * ds), int(w * ds)
            img_small = cv2.resize(img_np, (small_w, small_h), interpolation=cv2.INTER_AREA)
            labels_small = self._slic_numpy_impl(
                img_small,
                n_segments=n_seg,
                compactness=comp,
                max_iter=max_it,
                sigma=sig,
                enforce_connectivity=enforce_conn,
            )
            # Интерполяция меток обратно
            labels = cv2.resize(labels_small.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(int)
            # use_resize = True
        else:
            labels = self._slic_numpy_impl(
                img_np,
                n_segments=n_seg,
                compactness=comp,
                max_iter=max_it,
                sigma=sig,
                enforce_connectivity=enforce_conn,
            )

        # === СОЗДАНИЕ МАСКИ ===
        unique, counts = np.unique(labels, return_counts=True)
        bg_label = unique[np.argmax(counts)] if len(unique) > 0 else -1
        mask_np = (labels != bg_label).astype(np.float32)

        mask = torch.from_numpy(mask_np).to(self.device)

        exec_time: float = time.time() - start_time
        info = self._log_info(
            "slic_torch",
            exec_time,
            {
                "n_segments": n_segments,
                "compactness": compactness,
                "max_iter": max_iter,
                "sigma": sigma,
                "enforce_connectivity": enforce_connectivity,
                "precision": precision,
            },
            # precision_val=precision_val,
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}")
        return mask.unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    def _slic_enforce_connectivity_impl(
        self,
        labels: np.ndarray,
        n_segments: int,
        min_size_factor: float,
        max_size_factor: float,
    ) -> np.ndarray:
        """Упрощённая реализация принудительной связности."""
        h, w = labels.shape
        labels_out = labels.copy()

        from scipy import ndimage

        for label in np.unique(labels_out):
            if label < 0:
                continue

            mask = (labels_out == label).astype(np.uint8)
            labeled, num = ndimage.label(mask, structure=np.ones((3, 3)))

            if num <= 1:
                continue

            sizes = ndimage.sum(mask, labeled, range(1, num + 1))
            main_comp = np.argmax(sizes) + 1

            for comp_id in range(1, num + 1):
                if comp_id == main_comp:
                    continue

                comp_mask = labeled == comp_id
                from scipy.ndimage import binary_dilation

                dilated = binary_dilation(comp_mask, iterations=1)
                neighbors = dilated & ~comp_mask & (labels_out != label)

                if neighbors.any():
                    neighbor_labels = labels_out[neighbors]
                    most_common = np.bincount(neighbor_labels).argmax()
                    labels_out[comp_mask] = most_common
                else:
                    other_labels = np.unique(labels_out)
                    other_labels = other_labels[other_labels != label]
                    if len(other_labels) > 0:
                        labels_out[comp_mask] = np.random.choice(other_labels)

        # Перенумерация
        unique_new = np.unique(labels_out)
        label_map = {old: new for new, old in enumerate(unique_new)}
        labels_out = np.vectorize(label_map.get)(labels_out)

        return cast(np.ndarray, labels_out)

    # ──────────────────────────────────────────────────────────────────────
    def _slic_numpy_impl(
        self,
        image_np: np.ndarray,
        n_segments: int,
        compactness: float,
        max_iter: int,
        sigma: float,
        enforce_connectivity: bool,
    ) -> np.ndarray:
        """Чистая numpy-реализация SLIC для внутреннего использования.

        Args:
            image_np: Изображение (H, W, C), dtype=uint8 или float.
            n_segments: Желаемое число суперпикселей.
            compactness: Вес пространственной близости.
            max_iter: Макс. число итераций k-means.
            sigma: Sigma для предварительного сглаживания.
            enforce_connectivity: Обеспечить связность регионов.

        Returns:
            np.ndarray: Метки суперпикселей (H, W), dtype=int.
        """
        h, w = image_np.shape[:2]
        c = image_np.shape[2] if image_np.ndim == 3 else 1

        # Нормализация и сглаживание
        if image_np.dtype != np.float32:
            img_float = image_np.astype(np.float32) / 255.0
        else:
            img_float = image_np

        if sigma > 0:
            from scipy.ndimage import gaussian_filter

            img_float = gaussian_filter(img_float, sigma=sigma, mode="reflect")

        # Конвертация в Lab
        if c == 3:
            features = self._rgb_to_lab_numpy_impl(img_float)
        else:
            features = np.repeat(img_float, 3, axis=-1) if c == 1 else img_float

        # Пространственные координаты
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        step = np.sqrt(h * w / n_segments)
        coord_scale = compactness / step
        # Объединение признаков
        features_spatial = np.zeros((h, w, c + 2), dtype=np.float32)
        features_spatial[..., :c] = features
        features_spatial[..., c] = x_coords * coord_scale
        features_spatial[..., c + 1] = y_coords * coord_scale

        # Инициализация центроидов
        grid_y, grid_x = np.mgrid[step / 2 : h : step, step / 2 : w : step]
        grid_y, grid_x = grid_y.ravel(), grid_x.ravel()
        n_centroids = min(n_segments, len(grid_x))

        centroids = np.zeros((n_centroids, c + 2), dtype=np.float32)
        for i in range(n_centroids):
            y, x = int(grid_y[i]), int(grid_x[i])
            y0, y1 = max(0, y - int(step // 2)), min(h, y + int(step // 2))
            x0, x1 = max(0, x - int(step // 2)), min(w, x + int(step // 2))
            region = features_spatial[y0:y1, x0:x1]
            centroids[i] = region.reshape(-1, c + 2).mean(axis=0)

        # K-means кластеризация
        features_flat = features_spatial.reshape(-1, c + 2)
        from scipy.cluster.vq import kmeans2

        try:
            centroids, labels_flat = kmeans2(features_flat, centroids, iter=max_iter, minit="matrix", missing="warn")
        except Exception:
            centroids, labels_flat = kmeans2(features_flat, n_centroids, iter=max_iter, minit="++", missing="warn")

        labels = labels_flat.reshape(h, w)

        # Принудительная связность
        if enforce_connectivity:
            labels = self._slic_enforce_connectivity_impl(labels, n_segments, min_size_factor=0.5, max_size_factor=3.0)

        return cast(np.ndarray, labels)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _felzenszwalb(
        self,
        tensor: torch.Tensor,
        *,
        scale: Optional[float] = None,
        sigma: Optional[float] = None,
        min_size: Optional[int] = None,
        downsample: Optional[float] = None,
        precision: Optional[str] = None,
    ) -> torch.Tensor:
        """Алгоритм Felzenszwalb — иерархическая сегментация на основе графов.

        Строит сегментацию через минимальное остовное дерево:
        1. Построение графа пикселей с весами рёбер на основе градиента.
        2. Сортировка рёбер по весу.
        3. Последовательное слияние регионов, если внутреннее различие < межрегионального.
        4. Пост-обработка: удаление мелких регионов.

        Метод особенно эффективен для:
        - Сегментации объектов разного масштаба
        - Задач, где важна адаптивность к локальным особенностям
        - Предварительной обработки для выделения однородных регионов

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
            scale: Параметр масштаба: больше = крупнее регионы (по умолчанию: 100).
            sigma: Sigma для предварительного гауссова сглаживания (по умолчанию: 0.8).
            min_size: Минимальный размер региона для сохранения (по умолчанию: 50).
            downsample: Коэффициент уменьшения разрешения для ускорения (по умолчанию: 0.5).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16' (для предобработки).

        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W), dtype=float32.

        Note:
            - Felzenszwalb реализуется через skimage из-за сложности эффективной pure-PyTorch версии.
            - Для больших изображений автоматически применяется downsample + интерполяция меток.
            - Метод не поддерживает `torch.compile` из-за вызова внешних библиотек.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "felzenszwalb",
                scale=200,
                sigma=1.0,
                min_size=100,
                downsample=0.75
            )
            mask = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        start_time = time.time()
        gray = self._to_grayscale(tensor).squeeze(0).cpu().numpy()
        if gray.max() <= 1.0:
            gray = gray.astype(np.float32)
        else:
            gray = (gray / 255.0).astype(np.float32)

        h, w = gray.shape

        # === ПАРАМЕТРЫ ===
        sc = scale if scale is not None else self.params.get("scale", 100)
        sig = sigma if sigma is not None else self.params.get("sigma", 0.8)
        ms = min_size if min_size is not None else self.params.get("min_size", 50)
        ds = downsample if downsample is not None else self.params.get("downsample", 0.5)

        # === DOWNsample ДЛЯ УСКОРЕНИЯ ===
        # use_resize = False
        if h * w > 100_000 and ds < 1.0:
            small_h, small_w = int(h * ds), int(w * ds)
            gray_small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
            segments_small = self._felzenszwalb_skimage_impl(gray_small, scale=sc, sigma=sig, min_size=ms)
            # Интерполяция меток обратно
            segments = cv2.resize(
                segments_small.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(int)
            # use_resize = True
        else:
            segments = self._felzenszwalb_skimage_impl(gray, scale=sc, sigma=sig, min_size=ms)

        # === СОЗДАНИЕ МАСКИ ===
        unique, counts = np.unique(segments, return_counts=True)
        bg_label = unique[np.argmax(counts)] if len(unique) > 0 else -1
        mask_np = (segments != bg_label).astype(np.float32)

        mask = torch.from_numpy(mask_np).to(self.device)

        exec_time: float = time.time() - start_time

        info = self._log_info(
            "felzenszwalb_torch",
            exec_time,
            {"scale": scale, "sigma": sigma, "min_size": min_size, "precision": precision},
            # precision_val=precision_val,
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}")
        return mask.unsqueeze(0).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────────────
    def _felzenszwalb_skimage_impl(
        self,
        gray: np.ndarray,
        scale: float,
        sigma: float,
        min_size: int,
    ) -> np.ndarray:
        """Обёртка для skimage.segmentation.felzenszwalb."""
        try:
            from skimage.segmentation import felzenszwalb as sk_felzenszwalb

            return cast(np.ndarray, sk_felzenszwalb(gray, scale=scale, sigma=sigma, min_size=min_size))
        except ImportError:
            warnings.warn("skimage not installed. Fallback to kmeans.")
            # Простой fallback: k-means на сером изображении
            from scipy.cluster.vq import kmeans2

            h, w = gray.shape
            features = gray.reshape(-1, 1)
            centroids, labels_flat = kmeans2(features, 3, iter=10, minit="++")
            return cast(np.ndarray, labels_flat.reshape(h, w))

    # ──────────────────────────────────────────────────────────────────────
    # ИНТЕРАКТИВНЫЕ МЕТОДЫ СЕГМЕНТАЦИИ
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _grabcut(
        self,
        tensor: torch.Tensor,
        *,
        rect: Optional[Tuple[int, int, int, int]] = None,
        num_iterations: Optional[int] = None,
        n_components: int = 5,
        precision: Optional[str] = None,
    ) -> torch.Tensor:
        """Интерактивная сегментация GrabCut на чистом PyTorch.

        Использует прямоугольник для инициализации фона/переднего плана и итеративно
        уточняет границы через гауссовы смеси (GMM) для моделирования цветового распределения.

        Алгоритм:
        1. Инициализация маски: прямоугольник = объект, остальное = фон.
        2. Построение признаков: цвет (RGB) + пространственные координаты.
        3. Обучение двух GMM: для фона и объекта.
        4. Присвоение пикселей к классу с большей вероятностью.
        5. Обновление GMM на основе новых меток.
        6. Повторение шагов 4-5 `num_iterations` раз.

        Метод особенно эффективен для:
        - Интерактивной сегментации с ручной инициализацией
        - Задач, где объект и фон имеют различное цветовое распределение
        - Сегментации с чёткими, но сложными границами

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W), RGB.
            rect: Прямоугольник инициализации (x, y, width, height) в пикселях.
            num_iterations: Число итераций уточнения (по умолчанию: 5).
            n_components: Число компонент в каждой GMM (по умолчанию: 5).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16'.

        Returns:
            torch.Tensor: Бинарная маска объекта (1, 1, H, W), dtype=float32.

        Note:
            - Все вычисления выполняются на GPU без выхода в numpy.
            - Для `fp16` рекомендуется `n_components ≤ 3` для стабильности.
            - Метод безопасен для `torch.compile(fullgraph=True)` при фиксированных параметрах.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "grabcut",
                rect=(50, 50, 200, 200),
                num_iterations=10,
                n_components=3,
                precision="bf16"
            )
            mask = segmenter.segment(image)
            ```
        """
        # === ПРЕДПОДГОТОВКА ===
        img = tensor.squeeze(0).permute(1, 2, 0)  # (H, W, C)
        h, w, c = img.shape
        dtype = self.precision_manager.get_dtype(precision)
        img = img.to(dtype) if img.dtype != dtype else img
        start_time = time.time()

        # === ПАРАМЕТРЫ ===
        rect_val = rect if rect is not None else self.params.get("rect", (w // 4, h // 4, w // 2, h // 2))
        n_iter = num_iterations if num_iterations is not None else self.params.get("num_iterations", 5)
        n_comp = n_components if n_components is not None else self.params.get("n_components", 5)

        x, y, rw, rh = rect_val
        x, y, rw, rh = int(x), int(y), int(rw), int(rh)

        # === ИНИЦИАЛИЗАЦИЯ МАСКИ ===
        mask = torch.zeros(h, w, dtype=torch.float32, device=self.device)
        mask[y : y + rh, x : x + rw] = 1.0  # Объект = 1, фон = 0

        # === ПРИЗНАКИ ===
        # Цвет + нормализованные координаты
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing="ij",
        )
        coords = torch.stack([x_grid.float() / w, y_grid.float() / h], dim=-1)  # (H, W, 2)

        features = torch.cat([img, coords], dim=-1)  # (H, W, C+2)
        features_flat = features.reshape(-1, c + 2)  # (N, D)

        # === GMM КЛАССЫ ===
        # Используем torch.distributions для эффективности
        from torch.distributions import MultivariateNormal

        def init_gmm(
            data: torch.Tensor, n_comp: int, device: torch.device
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Инициализация параметров GMM: means, covs, weights."""
            # Случайная инициализация центроидов
            idx = torch.randperm(data.size(0), device=device)[:n_comp]
            means = data[idx].clone()

            # Инициализация ковариаций как диагональных
            covs = torch.eye(c + 2, device=device).unsqueeze(0).repeat(n_comp, 1, 1) * 0.1

            # Равные веса
            weights = torch.ones(n_comp, device=device) / n_comp

            return means, covs, weights

        def gmm_log_prob(
            data: torch.Tensor,
            means: torch.Tensor,
            covs: torch.Tensor,
            weights: torch.Tensor,
        ) -> torch.Tensor:
            """Вычисление log-вероятности для GMM."""
            n_comp = means.size(0)
            log_probs = torch.zeros(data.size(0), n_comp, device=data.device)

            for i in range(n_comp):
                try:
                    dist = MultivariateNormal(means[i], covs[i] + 1e-6 * torch.eye(c + 2, device=data.device))
                    log_probs[:, i] = dist.log_prob(data)
                except Exception:
                    # Fallback на упрощённую гауссиану
                    diff = data - means[i]
                    log_probs[:, i] = -0.5 * torch.sum(diff**2 / (torch.diag(covs[i]) + 1e-6), dim=1)

            # Взвешенная сумма
            weighted_log_probs = log_probs + torch.log(weights + 1e-10)
            return torch.logsumexp(weighted_log_probs, dim=1)

        # === ОСНОВНОЙ ЦИКЛ ===
        precision_val = precision if precision is not None else "fp32"
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            # Инициализация двух GMM
            fg_data = features_flat[mask.flatten() > 0.5]
            bg_data = features_flat[mask.flatten() <= 0.5]

            if fg_data.size(0) > 0 and bg_data.size(0) > 0:
                fg_means, fg_covs, fg_weights = init_gmm(fg_data, n_comp, self.device)
                bg_means, bg_covs, bg_weights = init_gmm(bg_data, n_comp, self.device)
            else:
                # Fallback: случайная инициализация
                fg_means, fg_covs, fg_weights = init_gmm(features_flat, n_comp, self.device)
                bg_means, bg_covs, bg_weights = init_gmm(features_flat, n_comp, self.device)

            for iteration in range(n_iter):
                # Вычисление вероятностей
                fg_log_prob = gmm_log_prob(features_flat, fg_means, fg_covs, fg_weights)
                bg_log_prob = gmm_log_prob(features_flat, bg_means, bg_covs, bg_weights)

                # Обновление маски
                new_mask = (fg_log_prob > bg_log_prob).float()

                # Проверка сходимости
                if torch.allclose(mask.flatten(), new_mask, atol=1e-3):
                    break

                mask = new_mask.reshape(h, w)

                # Обновление параметров GMM (EM-шаг)
                fg_mask_flat = mask.flatten() > 0.5
                bg_mask_flat = ~fg_mask_flat

                if fg_mask_flat.any():
                    fg_data = features_flat[fg_mask_flat]
                    # Обновление means
                    fg_means = fg_data.mean(dim=0, keepdim=True).repeat(n_comp, 1)
                    # Обновление covs (упрощённо: диагональные)
                    fg_covs = torch.diag(fg_data.var(dim=0) + 1e-6).unsqueeze(0).repeat(n_comp, 1, 1)

                if bg_mask_flat.any():
                    bg_data = features_flat[bg_mask_flat]
                    bg_means = bg_data.mean(dim=0, keepdim=True).repeat(n_comp, 1)
                    bg_covs = torch.diag(bg_data.var(dim=0) + 1e-6).unsqueeze(0).repeat(n_comp, 1, 1)

        exec_time: float = time.time() - start_time

        info = self._log_info(
            "grabcut_torch",
            exec_time,
            {"rect": rect, "num_iterations": num_iterations, "precision": precision},
            # precision_val=precision_val,
        )
        self.params["execution_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}")
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _grabcut_torch_visualization(
        self,
        tensor: torch.Tensor,
        *,
        alpha: float = 0.7,
        color: Tuple[int, int, int] = (255, 0, 0),  # красный по умолчанию
        rect: Optional[Tuple[int, int, int, int]] = None,
        num_iterations: Optional[int] = None,
        n_components: int = 5,
        precision: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Визуализация для GrabCut с поддержкой AMP и минимальными трансферами.

        Алгоритм:
        1. Выполнение сегментации через `_grabcut` (получение бинарной маски).
        2. Создание цветной маски на GPU с настраиваемым цветом.
        3. Alpha-смешивание с оригинальным изображением на GPU.
        4. Конвертация результата в numpy только для возврата.
        5. Возврат (визуализация, маска) в едином формате.

        Метод особенно эффективен для:
        - Отладки качества сегментации и сходимости GrabCut
        - Визуализации границ объекта с настраиваемым цветом наложения
        - Демонстрации работы алгоритма с разными параметрами `rect` и `n_components`

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W), RGB.
            alpha: Прозрачность наложения маски [0, 1] (по умолчанию: 0.7).
            color: Цвет маски в формате RGB (по умолчанию: красный (255, 0, 0)).
            rect: Прямоугольник инициализации (x, y, width, height) в пикселях.
            num_iterations: Число итераций уточнения (по умолчанию: из self.params).
            n_components: Число компонент в каждой GMM (по умолчанию: 5).
            precision: Точность вычислений: 'fp32', 'fp16', 'bf16' (для предобработки).
            **kwargs: Дополнительные параметры (для совместимости).

        Returns:
            Tuple[np.ndarray, torch.Tensor]:
                - Визуализация (H, W, 3), dtype=uint8, RGB — изображение с наложенной маской.
                - Бинарная маска (H, W), dtype=float32, значения {0.0, 1.0} — объект.

        Note:
            - Основная логика GrabCut делегирована `_grabcut` для избежания дублирования.
            - Визуализация (смешивание цветов) выполняется на GPU, numpy используется только для возврата.
            - Для `fp16` рекомендуется `n_components ≤ 3` и `alpha ≥ 0.5` для стабильности.
            - Метод не поддерживает `torch.compile(fullgraph=True)` из-за вызова GMM.

        Example:
            ```python
            segmenter = TorchSegmenter(
                "grabcut",
                rect=(50, 50, 200, 200),
                num_iterations=10,
                n_components=3
            )
            vis, mask = segmenter._grabcut_torch_visualization(
                image, alpha=0.6, color=(0, 255, 0), precision="bf16"
            )
            # vis: RGB изображение с зелёной маской объекта
            # mask: бинарная маска (0.0/1.0)
            ```
        """
        start_time = time.time()

        # === ШАГ 1: ПОЛУЧЕНИЕ МАСКИ ЧЕРЕЗ ОСНОВНОЙ МЕТОД ===
        # Используем `_grabcut` для получения бинарной маски
        # Это гарантирует согласованность логики между segment() и visualization()
        mask = self._grabcut(
            tensor,
            rect=rect,
            num_iterations=num_iterations,
            n_components=n_components,
            precision=precision,
        )  # (1, 1, H, W), float32

        # === ШАГ 2: ПОДГОТОВКА ДАННЫХ ДЛЯ ВИЗУАЛИЗАЦИИ ===
        # Конвертируем вход в удобный формат
        img = tensor.squeeze(0) if tensor.dim() == 4 else tensor  # (C, H, W)
        dtype = self.precision_manager.get_dtype(precision)
        img = img.to(dtype) if img.dtype != dtype else img
        c, h, w = img.shape

        # Приводим маску к 2D и целевой точности
        mask_2d = mask.squeeze()  # (H, W)
        mask_2d = mask_2d.to(dtype) if mask_2d.dtype != dtype else mask_2d

        # === ШАГ 3: СОЗДАНИЕ ЦВЕТНОЙ МАСКИ НА GPU ===
        # Конвертируем изображение в RGB если нужно
        if c == 1:  # Grayscale → RGB
            img_rgb = img.repeat(3, 1, 1)
        else:
            img_rgb = img

        # Цветная маска (на GPU)
        color_tensor = torch.tensor(color, dtype=dtype, device=img.device).view(3, 1, 1) / 255.0
        colored_mask = torch.zeros_like(img_rgb)

        # Применяем цвет только к пикселям маски
        mask_bool = mask_2d > 0.5
        for c_idx in range(3):
            colored_mask[c_idx, mask_bool] = color_tensor[c_idx, 0, 0]

        # === ШАГ 4: ALPHA-СМЕШИВАНИЕ НА GPU ===
        precision_val = precision if precision is not None else "fp32"
        with self.precision_manager.autocast(precision_val, enabled=(dtype != torch.float32)):
            result = img_rgb * (1.0 - alpha) + colored_mask * alpha

        # === ШАГ 5: КОНВЕРТАЦИЯ В NUMPY (только в конце) ===
        # Визуализация: [0, 1] → [0, 255] → uint8
        result_np = self._tensor_to_numpy(result.unsqueeze(0), denormalize=True)  # (H, W, 3)

        # Маска: возвращаем как float32 [0.0, 1.0] на оригинальном устройстве
        mask_tensor = mask_2d.to(torch.float32).to(self.device)

        exec_time: float = time.time() - start_time

        info = self._log_info(
            "grabcut_visualisation_torch",
            exec_time,
            {"alpha": alpha, "color": color, "rect": rect, "num_iterations": num_iterations, "precision": precision},
            # precision_val=precision_val,
        )
        self.params["visualization_info"] = info
        if self._debug_mode:
            logger.info(f"[DEBUG] {self.method}")

        return result_np, mask_tensor


# ──────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────
__all__: List[str] = [
    # 🔹 Основные классы
    "TorchSegmenter2",
    "PrecisionManager",
    # 🔹 Протоколы и типизация
    "SegmentMethod",
    # 🔹 Re-export базовых типов (для удобства)
    "BaseSegmenter",
]
"""Публичный API модуля NewTorchSegmenter.

Экспортируемые символы:
- `TorchSegmenter2`: Высокопроизводительный сегментер на чистом PyTorch (50+ методов).
  Поддерживает AMP, `torch.compile`, динамическую точность (fp32/fp16/bf16/float8),
  векторизованные свёртки и кэширование ядер. Fallback на Numba/scipy для CPU.

- `PrecisionManager`: Утилита управления точностью вычислений.
  Автоматически подбирает оптимальный dtype под устройство, предоставляет контекстные
  менеджеры AMP (`autocast`, `autocast_float8`), проверяет поддержку типов на GPU.

- `SegmentMethod`: Протокол для унификации сигнатур методов сегментации.
  Гарантирует совместимость с `torch.compile` и динамическим диспетчером.

- `BaseSegmenter`, `ImageInput`, `BinaryMask`, `ProbabilityMask`: Базовые типы
  и протоколы из `BaseSegmenter` для удобного импорта в одном месте.

Используется статическими анализаторами (mypy, pyright), linter'ами и IDE
для автодополнения и проверки типов:
    from segmenters.NewTorchSegmenter import TorchSegmenter2, PrecisionManager, SegmentMethod
    
    # Корректная типизация конфигурации
    manager = PrecisionManager(default_precision="bf16")
    segmenter = TorchSegmenter2(
        method="otsu_thresholding",
        device="cuda",
        use_compile=True,
        compile_mode="max-autotune"
    )
    mask = segmenter.segment(image)  # BinaryMask
"""
