# BatchNeuralTester.py

"""Модуль для анализа влияния аугментаций на качество нейросетевых моделей сегментации на датасете ADE20K (или его подмножестве).

Workflow:
1. Загрузка списка изображений из датасета (локально или через HuggingFace).
2. Для каждой комбинации (модель × аугментация):
   - Загрузка чекпоинта через NeuralSegmenter.
   - Пакетное предсказание на подмножестве датасета.
   - Расчёт и агрегация метрик.
   - Очистка памяти.
3. Агрегация результатов в DataFrame.
4. Статистический анализ и визуализация.
5. Экспорт: CSV, JSON, Markdown, PNG-графики.

Example:
    ```bash
    python analyze_batch.py --dataset ./data/ADE20K --subset 50 --output ./results
    ```
"""

"""Модуль для анализа влияния аугментаций на качество нейросетевых моделей сегментации на датасете ADE20K.

Поддерживаемые задачи:
1. **Пакетное тестирование**: инференс множества моделей × уровней аугментаций
2. **Многоклассовые метрики**: mIoU, per-class IoU, Boundary F1 для 150 классов
3. **Статистический анализ**: ANOVA, Tukey HSD, приросты относительно baseline
4. **Экспорт и визуализация**: CSV, JSON, Markdown, PNG-графики, оверлеи
5. **Оптимизация**: кэширование предсказаний, OOM-handling, профилирование

Ключевые особенности:
- ✅ Авто-обнаружение чекпоинтов по шаблону `{model}_{aug}_*.pth`
- ✅ Гибкая загрузка данных: локальный ADE20K или HuggingFace Hub
- ✅ Контроль памяти: автоматическое уменьшение batch_size при OOM
- ✅ LRU-кэш предсказаний на диске с контролем размера
- ✅ Поддержка точностей: fp32/fp16/bf16 с авто-fallback
- ✅ Экспорт в ONNX/TensorRT с fallback-механизмами
- ✅ Интеграция с трекерами: MLflow, Weights & Biases
- ✅ Расширенная визуализация: class-aware оверлеи, сравнительные сетки

Типичный workflow:
```bash
# 1. Подготовка чекпоинтов
ls ./models/unet_smp_{none,basic,medium}_*.pth

# 2. Запуск тестирования
python BatchNeuralTester.py \\
    --dataset ./data/ADE20K \\
    --subset 50 \\
    --output ./results/unet_aug \\
    --cache --resume \\
    --compute-boundary-f1 \\
    --verbose

# 3. Анализ результатов
cat ./results/unet_aug/report.md
open ./results/unet_aug/plots/miou_comparison.png
```

Note:
- Формат чекпоинтов: `{model_type}_{augmentation_level}_*.pth` (например, `unet_smp_basic_20240101.pth`).
- Для воспроизводимости используйте `--seed` и `--resume` при повторных запусках.
- Boundary F1 вычисляется медленно (морфологические операции); используйте `--compute-boundary-f1` только для детального анализа.
- При экспорте в TensorRT требуется установленный `torch-tensorrt` и CUDA-capable GPU.
- Оверлеи сохраняются в `output_dir/overlays/`; используйте `--class-aware-overlays` для легенд классов.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
# from __future__ import annotations  # PEP 563: отложенная оценка аннотаций

import sys
import argparse
import glob
import json
import os
import gc
import time
from hashlib import sha256
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union, Literal, Set, cast, Generator
from dataclasses import dataclass, field, asdict
from collections import OrderedDict

import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from scipy.ndimage import binary_erosion, binary_dilation, zoom
from scipy import stats
import warnings
from contextlib import contextmanager, nullcontext
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from segmenters.NewTorchSegmenter import PrecisionManager
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# HuggingFace для загрузки датасета
from huggingface_hub import hf_hub_download, list_repo_files

import logging

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Настройка путей проекта
project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Локальные импорты
from segmenters.NeuralSegmenter import NeuralSegmenter
from metrics.SegmentationMetrics import SegmentationMetrics


# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & CONSTANTS
# ──────────────────────────────────────────────────────────────────────
MaskArray = np.ndarray
ImageArray = np.ndarray
MetricValue = float
MetricsDict = Dict[str, MetricValue]
PathLike = Union[str, Path]

# Маппинг имён чекпоинтов на ModelType enum (для NeuralSegmenter)
MODEL_TYPE_MAPPING: Dict[str, str] = {
    "unet_smp": "unet_smp",
    "fpn_smp": "fpn_smp",
    "psp_smp": "pspnet_smp",
    "deeplab_tv": "deeplab_tv",
    "fcn_tv": "fcn_tv",
    "segnet": "segnet",
}

# Стандартные метрики для агрегации
DEFAULT_METRICS: List[str] = [
    "iou",
    "dice",
    "f1_score",
    "precision",
    "recall",
    "accuracy",
    "mae",
    "hausdorff_distance",
    "boundary_f1",
]

# ──────────────────────────────────────────────────────────────────────
# CONSTANTS для парсинга
# ──────────────────────────────────────────────────────────────────────
KNOWN_MODEL_PREFIXES: List[str] = [
    "unet_smp",
    "fpn_smp",
    "psp_smp",
    "deeplab_tv",
    "fcn_tv",
    "segnet",
]
KNOWN_AUG_LEVELS: Set[str] = {"none", "basic", "medium"}


# ──────────────────────────────────────────────────────────────────────
class PredictionCache:
    """Кэш предсказаний моделей для ускорения повторных запусков.

    Использует дисковое хранилище с LRU-политикой вытеснения.
    Ключи генерируются на основе mtime чекпоинта, пути к изображению и хэша конфигурации.

    Args:
        cache_dir: Путь к директории для хранения `.npy` файлов.
        max_size_gb: Максимальный размер кэша в гигабайтах.

    Returns:
        PredictionCache: Инициализированный объект кэша.

    Note:
        - При превышении лимита `max_size_gb` удаляются самые старые файлы по `st_mtime`.
        - Повреждённые pickle-файлы автоматически удаляются при чтении.
    """

    def __init__(self, cache_dir: PathLike, max_size_gb: float = 10.0) -> None:
        """Инициализация модуля PredictionCache."""
        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes: int = int(max_size_gb * 1e9)

    # ──────────────────────────────────────────────────────────────────────
    def _get_key(self, model_path: Path, image_path: Path, config_hash: str) -> str:
        """Генерация уникального ключа кэша.

        Args:
            model_path: Путь к файлу весов модели.
            image_path: Путь к входному изображению.
            config_hash: SHA256-хэш конфигурации эксперимента.

        Returns:
            str: 16-символьный hex-ключ.
        """
        content: str = f"{model_path.stat().st_mtime}:{image_path}:{config_hash}"
        return sha256(content.encode()).hexdigest()[:16]

    # ──────────────────────────────────────────────────────────────────────
    def get(self, key: str) -> Optional[np.ndarray]:
        """Извлечение предсказания из кэша по ключу.

        Args:
            key: Уникальный идентификатор кэшированного результата.

        Returns:
            Optional[np.ndarray]: Массив предсказания или None, если кэш отсутствует/повреждён.
        """
        cache_file: Path = self.cache_dir / f"{key}.npy"  # ← меняем расширение
        if cache_file.exists():
            try:
                # np.load безопасен для файлов, созданных np.save
                result = np.load(cache_file, allow_pickle=False)
                if isinstance(result, np.ndarray):
                    return result
                logger.warning(f"⚠️ Неверный тип в кэше {key}: {type(result)}")
                cache_file.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"⚠️ Ошибка чтения кэша {key}: {e}")
                cache_file.unlink(missing_ok=True)
        return None

    # ──────────────────────────────────────────────────────────────────────
    def set(self, key: str, prediction: np.ndarray) -> None:
        """Сохранение предсказания в LRU кэш с контролем лимита размера.

        Args:
            key: Уникальный идентификатор для сохранения.
            prediction: Массив предсказания (маска сегментации).

        Note:
            Перед записью проверяется суммарный размер каталога.
            При превышении лимита удаляются самые старые `.npy` файлы.
        """
        cache_file: Path = self.cache_dir / f"{key}.npy"
        total_size: int = sum(f.stat().st_size for f in self.cache_dir.glob("*.npy"))
        while total_size + prediction.nbytes > self.max_bytes:
            oldest: Optional[Path] = min(
                self.cache_dir.glob("*.npy"),
                key=lambda f: f.stat().st_mtime,
                default=None,
            )
            if oldest is None:
                break
            oldest.unlink()
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.npy"))

        with open(cache_file, "wb") as f:
            pickle.dump(prediction, f)

    # ──────────────────────────────────────────────────────────────────────
    def clear(self) -> int:
        """Полная очистка каталога кэша.

        Returns:
            int: Количество удалённых файлов.
        """
        count: int = 0
        for f in self.cache_dir.glob("*.npy"):
            f.unlink()
            count += 1
        logger.info(f"🗑️  Очищено {count} файлов кэша")
        return count


# ──────────────────────────────────────────────────────────────────────
@dataclass
class TestConfig:
    """Конфигурация тестирования моделей сегментации.

    Содержит все параметры CLI, переданные в скрипт.
    Автоматически преобразует пути и валидирует устройство.

    Attributes:
        dataset_path: Путь к корню датасета или ID HF репозитория.
        output_dir: Директория для артефактов (CSV, PNG, JSON, MD).
        subset_size: Количество изображений для теста. 0 или None = весь датасет.
        random_seed: Seed для воспроизводимости выбора подмножества.
        batch_size: Размер батча (по умолчанию 1 из-за разного размера входов).
        device: "cuda" или "cpu".
        num_classes: Количество классов сегментации (ADE20K = 150).
        ignore_index: Индекс игнорируемого пикселя (обычно 255).
        metrics: Список метрик для расчёта.
        save_overlays: Флаг сохранения визуализаций.
        overlay_cache_max: Лимит оверлеев в оперативной памяти (LRU).
        verbose: Подробный вывод логов и статусов.
        precision: Точность вычислений ("fp32", "fp16", "bf16").
        cache: Включить дисковое кэширование предсказаний.
        resume: Пропускать уже обработанные комбинации (модель+изображение).
        export_onnx: Экспортировать модели в ONNX после теста.
        export_trt: Компилировать в TensorRT.
        profile: Включить torch.profiler для первой модели.
        compute_boundary_f1: Расчёт метрики точности границ (dilation ⊕ erosion).
        per_class_metrics: Включить Per-class Precision/Recall/IoU в отчёт.
        use_mlflow: Логировать метрики в MLflow.
        use_wandb: Логировать в Weights & Biases.
        class_aware_overlays: Рисовать цветные легенды классов на оверлеях.
        overlay_alpha: Прозрачность наложения маски [0.0, 1.0].

    Note:
        При `device='cpu'` точность fp16/bf16 автоматически откатывается до fp32.
    """

    dataset_path: PathLike
    output_dir: PathLike = "./results/augmentation_analysis"
    subset_size: Optional[int] = 50  # None = весь датасет
    random_seed: int = 42
    batch_size: int = 1
    device: str = "cuda"
    num_classes: int = 150
    ignore_index: int = 255
    metrics: List[str] = field(default_factory=lambda: DEFAULT_METRICS.copy())
    save_overlays: bool = True
    overlay_cache_max: int = 3000  # Максимум оверлеев в памяти
    overlay_sample_rate: float = 1.0  # Сохранять 100% визуализаций
    verbose: bool = True
    save_visualizations: bool = True
    viz_alpha: float = 0.4  # Прозрачность для оверлеев
    viz_color: Tuple[int, int, int] = (255, 0, 0)  # Цвет маски
    precision: str = "fp32"
    cache: bool = False
    cache_dir: str = "./cache/predictions"
    cache_max_gb: float = 10.0
    clear_cache: bool = False
    resume: bool = False
    export_onnx: bool = False
    export_trt: bool = False
    trt_precision: str = "fp16"
    opset: int = 17
    dynamic_shapes: bool = False
    models_dir: str = "./models"
    profile: bool = False
    profile_output: str = "./profiling"
    profile_warmup: int = 10
    profile_runs: int = 50
    compute_boundary_f1: bool = False
    per_class_metrics: bool = False
    use_mlflow: bool = False
    use_wandb: bool = False
    class_aware_overlays: bool = False
    overlay_alpha: float = 0.5
    save_viz: bool = False

    # ──────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        """Инициализация модуля TestConfig."""
        self.dataset_path: Path = Path(self.dataset_path)
        self.output_dir: Path = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"Unsupported device: {self.device}")


# ──────────────────────────────────────────────────────────────────────
@dataclass
class ModelCheckpoint:
    """Метаданные найденного чекпоинта модели.

    Attributes:
        key: Уникальный ключ "{model_type}_{aug_level}".
        path: Абсолютный путь к файлу `.pth`.
        model_type: Тип архитектуры (например, "unet_smp").
        augmentation: Уровень аугментации ("none", "basic", "medium").
        original_type: Исходное имя архитектуры из маппинга.

    Methods:
        display_name: Формирует человекочитаемое имя для логов.
    """

    key: str
    path: Path
    model_type: str
    augmentation: str
    original_type: str

    @property
    def display_name(self) -> str:
        """Формирует человекочитаемое имя модели для логов и отчётов.

        Returns:
            str: Строка вида "unet_smp_none".
        """
        return f"{self.original_type}_{self.augmentation}"


# ──────────────────────────────────────────────────────────────────────
@dataclass
class TestResult:
    """Результат тестирования одного изображения.

    Attributes:
        model_key: Ключ модели.
        image_name: Имя файла изображения.
        metrics: Словарь рассчитанных метрик.
        inference_time: Время инференса в секундах.
        precision: Точность, использованная при запуске.
        pred_mask: Предсказанная маска (опционально, для отладки).
        gt_mask: Ground truth маска (опционально).

    Methods:
        to_dict(): Преобразует объект в плоский словарь для `pd.DataFrame`.
    """

    model_key: str
    image_name: str
    metrics: MetricsDict
    inference_time: float
    precision: str = "fp32"
    pred_mask: Optional[MaskArray] = None
    gt_mask: Optional[MaskArray] = None

    # ──────────────────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование результата в плоский словарь для создания DataFrame.

        Returns:
            Dict[str, Any]: Словарь с метаданными, метриками и временем.
        """
        return {
            "model_key": self.model_key,
            "image_name": self.image_name,
            **self.metrics,
            "inference_time": self.inference_time,
            "precision": self.precision,
        }


# ──────────────────────────────────────────────────────────────────────
@contextmanager
def safe_inference_context(model_name: str, image_name: str) -> Generator[None, None, None]:
    """Контекст-менеджер для безопасного выполнения инференса.

    Отлавливает CUDA OOM и другие критические ошибки, очищает память и логирует детали.

    Args:
        model_name: Имя или ключ тестируемой модели.
        image_name: Имя текущего обрабатываемого изображения.

    Note:
        При `OutOfMemoryError` выполняется `torch.cuda.empty_cache()` и `gc.collect()`
        перед повторным выбросом исключения.
    """
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"OOM при {model_name}/{image_name}: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        raise
    except Exception as e:
        logger.error(f"Ошибка {model_name}/{image_name}: {type(e).__name__}: {e}", exc_info=True)
        raise


# ──────────────────────────────────────────────────────────────────────
def _check_precision_support(model: Optional[nn.Module], dtype: torch.dtype, device: str) -> bool:
    """Проверка совместимости точности вычислений с устройством.

    Args:
        model: Экземпляр модели (опционально, для будущих проверок).
        dtype: Требуемый тип данных тензора (torch.float16, bfloat16, float32).
        device: Целевое устройство ("cuda", "cpu").

    Returns:
        bool: True, если точность поддерживается; False, если требуется fallback.

    Note:
        - bf16 требует GPU архитектуры Ampere (Compute Capability >= 8.0).
        - fp16 на CPU официально не поддерживается для инференса.
    """
    if dtype == torch.bfloat16 and device == "cuda" and torch.cuda.is_available():
        cap: Tuple[int, int] = torch.cuda.get_device_capability(0)
        return cap[0] >= 8  # Ampere+
    if dtype == torch.float16 and device == "cpu":
        return False  # fp16 на CPU неэффективен
    return True


# ──────────────────────────────────────────────────────────────────────
def extract_model_aug_from_key(key: str) -> Tuple[str, str]:
    """Корректно извлекает (модель, аугментация) из ключа.

    Формат ключа: "{модель}_{аугментация}_{имя_изображения}"

    Форматы ключей:
      - "fcn_tv_basic_ADE_val_00000001" → ("fcn_tv", "basic").
      - "unet_smp_none_20260413_085847" → ("unet_smp", "none").
      - "segnet_medium_overlay" → ("segnet", "medium").

    Args:
        key: Строка-ключ, например "unet_smp_none_ADE_val_00000001".

    Returns:
        Tuple[str, str]: Кортеж (model_name, augmentation_level).

    Note:
        Алгоритм:
        1. Ищем известный префикс модели в начале ключа.
        2. После префикса ищем известный уровень аугментации.
        3. Возвращаем пару (модель, аугментация).
    """
    key_stripped: str = key.strip()

    # Шаг 1: ищем известный префикс модели
    matched_model: Optional[str] = None
    for model_prefix in sorted(KNOWN_MODEL_PREFIXES, key=len, reverse=True):
        if key_stripped.startswith(model_prefix + "_"):
            matched_model = model_prefix
            break

    if matched_model is None:
        # Fallback: если не нашли префикс, пробуем парсить с конца
        parts: List[str] = key_stripped.split("_")
        for i in range(len(parts) - 1, 0, -1):
            if parts[i].lower() in KNOWN_AUG_LEVELS:
                aug: str = parts[i]
                model: str = "_".join(parts[:i])
                return model, aug
        return key_stripped, "unknown"

    # Шаг 2: после модели ищем аугментацию
    remainder: str = key_stripped[len(matched_model) + 1 :]  # +1 для подчёркивания
    remainder_parts: List[str] = remainder.split("_")

    for part in remainder_parts:
        if part.lower() in KNOWN_AUG_LEVELS:
            return matched_model, part

    # Если аугментация не найдена — возвращаем как есть
    logger.warning(f"⚠️ Не распознан ключ: '{key}' → ('{key}', 'unknown')")
    return matched_model, "unknown"


# ──────────────────────────────────────────────────────────────────────
def save_augmentation_comparison_grid(
    overlay_images: Dict[str, Image.Image],
    output_dir: PathLike = "./test_run1/overlays",
    model_names: Optional[List[str]] = None,
) -> None:
    """Создание и сохранение единой сетки сравнения всех моделей по аугментациям.

    Макет: строки — модели, столбцы — [none, basic, medium].

    Args:
        overlay_images: Словарь `{ключ_оверлея: PIL.Image}`.
        output_dir: Директория для сохранения графика.
        model_names: Опциональный список имён моделей. Если None, извлекается из ключей.

    Note:
        Если для какой-либо комбинации модель/аугментация нет оверлея,
        ячейка помечается как "N/A" с серым фоном.
    """
    output_path: Path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if model_names is None:
        models: List[str] = list(set(extract_model_aug_from_key(k)[0] for k in overlay_images.keys()))
    else:
        models = model_names

    if not models:
        print("⚠️  Нет моделей для визуализации")
        return

    n_models: int = len(models)
    fig, axes = plt.subplots(n_models, 3, figsize=(15, 5 * n_models), squeeze=False)

    for row, model in enumerate(models):
        for col, aug in enumerate(["none", "basic", "medium"]):
            # Ищем ключи, содержащие модель и аугментацию (любое изображение)
            matching_keys: List[str] = [k for k in overlay_images.keys() if k.startswith(f"{model}_{aug}_")]
            ax = axes[row, col]

            if matching_keys and overlay_images[matching_keys[0]] is not None:
                ax.imshow(overlay_images[matching_keys[0]])
                ax.set_title(f"{aug.upper()}", fontsize=10, fontweight="bold", color="darkblue")
                ax.axis("off")
            else:
                ax.set_facecolor("#f5f5f5")
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                    color="gray",
                    fontweight="bold",
                )
                ax.set_title(f"{aug.upper()}", fontsize=10, fontweight="bold", color="gray")
                ax.axis("off")

        axes[row, 0].set_ylabel(
            model.upper(),
            rotation=0,
            labelpad=60,
            fontsize=11,
            fontweight="bold",
            ha="right",
            va="center",
        )

    plt.suptitle(
        "Сравнение влияния аугментаций на визуализацию сегментации",
        fontsize=14,
        y=1.01,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.98))

    grid_path: Path = output_path / Path("full_comparison_grid.png")
    plt.savefig(grid_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"✅ Полная сетка сравнения: {grid_path}")


# ──────────────────────────────────────────────────────────────────────
def save_model_augmentation_comparisons(
    overlay_images: Dict[str, Image.Image],
    output_dir: PathLike = "./test_run1/overlays",
    models: Optional[List[str]] = None,
) -> None:
    """Сохранение отдельных сравнений аугментаций для каждой модели (3 колонки).

    Args:
        overlay_images: Словарь с оверлеями.
        output_dir: Путь для сохранения PNG.
        models: Список моделей для обработки.
    """
    output_path: Path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = list(set(extract_model_aug_from_key(k)[0] for k in overlay_images.keys()))

    print(f"\n🖼️ Сохранение сравнения визуализаций ({len(overlay_images)} оверлеев)...")
    print(f"   Найдено моделей: {models}")

    for model in models:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"{model}: сравнение аугментаций", fontsize=14, fontweight="bold")

        for idx, aug in enumerate(["none", "basic", "medium"]):
            matching_keys: List[str] = [k for k in overlay_images.keys() if k.startswith(f"{model}_{aug}_")]
            ax = axes[idx]

            if matching_keys:
                key: str = matching_keys[0]  # Берём первый найденный
                if overlay_images.get(key) is not None:
                    ax.imshow(overlay_images[key])
                    ax.set_title(f"{aug.upper()}", fontsize=11)
                    ax.axis("off")
                    print(f"   ✅ {model}_{aug}: {key}")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Empty",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=12,
                    )
                    ax.set_title(f"{aug.upper()}", fontsize=11)
                    ax.axis("off")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_title(f"{aug.upper()}", fontsize=11)
                ax.axis("off")
                print(f"   ⚠️  {model}_{aug}: не найдено")

        plt.tight_layout()
        comp_path: Path = output_path / f"comparison_{model}.png"
        plt.savefig(comp_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Сравнение для {model}: {comp_path}")


# ──────────────────────────────────────────────────────────────────────
class BatchNeuralTester:
    """Оркестратор пакетного тестирования нейросетевых моделей сегментации на датасете.

    Управляет загрузкой данных, инференсом, расчётом метрик,
    кэшированием, профилированием, экспортом и визуализацией.

    Аналог BatchClassicTester, но для NeuralSegmenter с поддержкой:
    - Многоклассовой сегментации (ADE20K: 150 классов).
    - Расчёта mIoU и бинарных метрик.
    - Эффективной работы с памятью.
    """

    def __init__(self, config: TestConfig) -> None:
        """Инициализация тестера конфигурацией, кэшем и трекерами.

        Args:
            config: Объект TestConfig с параметрами запуска.
        """
        self.config: TestConfig = config
        self.results: List[TestResult] = []
        # Используем OrderedDict для LRU-кэша оверлеев
        self.overlays: OrderedDict[str, Image.Image] = OrderedDict()
        self.precision_manager: PrecisionManager = PrecisionManager(
            default_precision=getattr(config, "precision", "fp32")
        )

        self.cache: Optional[PredictionCache] = None
        if getattr(config, "cache", False):
            cache_dir: Path = Path(getattr(config, "cache_dir", "./cache/predictions"))
            if getattr(config, "clear_cache", False):
                temp_cache: PredictionCache = PredictionCache(cache_dir)
                temp_cache.clear()
            self.cache = PredictionCache(cache_dir=cache_dir, max_size_gb=getattr(config, "cache_max_gb", 10.0))
            logger.info(f"✅ Кэш инициализирован: {cache_dir}")

        self.tracker: Optional[str] = None
        if getattr(config, "use_mlflow", False) or getattr(config, "use_wandb", False):
            self._init_experiment_tracking()

    # ──────────────────────────────────────────────────────────────────────
    def _find_checkpoints(
        self,
        models_dir: Optional[PathLike] = None,
        model_types: Optional[List[str]] = None,
        augmentation_levels: Optional[List[str]] = None,
    ) -> Dict[str, ModelCheckpoint]:
        """Поиск чекпоинтов по шаблону `{model_type}_{aug_level}_*.pth`.

        Args:
            models_dir: Директория с весами. Если None, берётся из `self.config.models_dir`.
            model_types: Список архитектур для поиска.
            augmentation_levels: Список уровней аугментации.

        Returns:
            Dict[str, ModelCheckpoint]: Маппинг ключей к объектам метаданных чекпоинтов.

        Note:
            Для каждой пары выбирается самый новый файл по времени создания (`os.path.getctime`).
        """
        if models_dir is None:
            models_dir = self.config.models_dir
        if model_types is None:
            model_types = [
                "unet_smp",
                "fpn_smp",
                "psp_smp",
                "deeplab_tv",
                "fcn_tv",
                "segnet",
            ]
        if augmentation_levels is None:
            augmentation_levels = ["none", "basic", "medium"]

        checkpoints: Dict[str, ModelCheckpoint] = {}
        models_path: Path = Path(models_dir)

        for model_type in model_types:
            for aug_level in augmentation_levels:
                # Шаблон: {model_type}_{aug_level}_*.pth
                pattern: str = str(models_path / f"{model_type}_{aug_level}_*.pth")
                files: List[str] = glob.glob(pattern)

                if files:
                    latest: str = max(files, key=os.path.getctime)
                    key: str = f"{model_type}_{aug_level}"  # Ключ для агрегации
                    checkpoints[key] = ModelCheckpoint(
                        key=key,
                        path=Path(latest),
                        model_type=MODEL_TYPE_MAPPING.get(model_type, model_type),
                        augmentation=aug_level,
                        original_type=model_type,
                    )
                    if self.config.verbose:
                        logger.info(f"✓ {key}: {Path(latest).name}")
                else:
                    if self.config.verbose:
                        logger.warning(f"✗ {model_type}_{aug_level}: не найден")

        return checkpoints

    # ──────────────────────────────────────────────────────────────────────
    def _load_ade20k_images(
        self,
        repo_id: str = "hf-internal-testing/fixtures_ade20k",
        local_path: Optional[PathLike] = None,
    ) -> List[Tuple[Path, Path]]:
        """Загрузка пар (изображение, маска) из локальной директории или HuggingFace Hub.

        Args:
            repo_id: ID датасета на HF.
            local_path: Локальный путь к корню датасета.

        Returns:
            List[Tuple[Path, Path]]: Отсортированный список кортежей путей.

        Note:
            Поддерживает несколько вариантов внутренней структуры папок (ADEChallengeData2016, ade20k).
            Если `subset_size` указан, выбирается случайное подмножество с фиксированным seed.
        """
        image_mask_pairs: List[Tuple[Path, Path]] = []

        if local_path and Path(local_path).exists():
            data_path: Path = Path(local_path)
            print(f"🔍 Поиск датасета в: {data_path.resolve()}")
            print(f"   images/validation: {(data_path / 'images' / 'validation').exists()}")
            print(f"   annotations/validation: {(data_path / 'annotations' / 'validation').exists()}")

            images_dir: Path = data_path / "images" / "validation"
            masks_dir: Path = data_path / "annotations" / "validation"

            if not images_dir.exists():
                alt_images: Path = data_path / "ADEChallengeData2016" / "images" / "validation"
                alt_masks: Path = data_path / "ADEChallengeData2016" / "annotations" / "validation"
                if alt_images.exists() and alt_masks.exists():
                    print(f"   [Alt] images/validation: {alt_images.exists()}")
                    print(f"   [Alt] annotations/validation: {alt_masks.exists()}")
                    images_dir, masks_dir = alt_images, alt_masks
                    if self.config.verbose:
                        logger.info("🔍 Используется альтернативная структура: ADEChallengeData2016")

            if not images_dir.exists():
                alt_images = data_path / "ade20k" / "images" / "validation"
                alt_masks = data_path / "ade20k" / "annotations" / "validation"
                if alt_images.exists() and alt_masks.exists():
                    images_dir, masks_dir = alt_images, alt_masks

            if images_dir.exists() and masks_dir.exists():
                for img_path in sorted(images_dir.glob("*.jpg")):
                    mask_path: Path = masks_dir / img_path.with_suffix(".png").name
                    if mask_path.exists():
                        image_mask_pairs.append((img_path, mask_path))
        else:
            try:
                files: List[str] = list_repo_files(repo_id, repo_type="dataset")
                img_files: List[str] = [f for f in files if f.endswith(".jpg")]
                mask_files: List[str] = [f for f in files if f.endswith(".png")]

                for img_file in img_files[: self.config.subset_size or len(img_files)]:
                    img_path = Path(hf_hub_download(repo_id=repo_id, filename=img_file, repo_type="dataset"))
                    mask_file = img_file.replace(".jpg", ".png")
                    if mask_file in mask_files:
                        mask_path = Path(hf_hub_download(repo_id=repo_id, filename=mask_file, repo_type="dataset"))
                        image_mask_pairs.append((Path(img_path), Path(mask_path)))
            except Exception as e:
                logger.error(f"Ошибка загрузки из HF: {e}")
                img_path = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        filename="ADE_val_00000001.jpg",
                        repo_type="dataset",
                    )
                )
                mask_path = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        filename="ADE_val_00000001.png",
                        repo_type="dataset",
                    )
                )
                image_mask_pairs.append((Path(img_path), Path(mask_path)))

        if self.config.subset_size and len(image_mask_pairs) > self.config.subset_size:
            np.random.seed(self.config.random_seed)
            indices: np.ndarray = np.random.choice(len(image_mask_pairs), self.config.subset_size, replace=False)
            image_mask_pairs = [image_mask_pairs[i] for i in sorted(indices)]

        if self.config.verbose:
            logger.info(f"Загружено {len(image_mask_pairs)} пар изображение/маска")

        return image_mask_pairs

    # ──────────────────────────────────────────────────────────────────────
    def _resize_mask(self, mask: MaskArray, target_shape: Tuple[int, int], order: int = 0) -> MaskArray:
        """Ресайз маски предсказания под размер Ground Truth.

        Args:
            mask: Исходная маска (numpy array).
            target_shape: Целевые размеры (H, W).
            order: Порядок интерполяции (0 = nearest для целочисленных меток).

        Returns:
            MaskArray: Изменённая маска с сохранением целочисленных классов.
        """
        if mask.shape == target_shape:
            return mask.copy()
        sh, sw = target_shape[0] / mask.shape[0], target_shape[1] / mask.shape[1]
        resized: np.ndarray = zoom(mask.astype(np.float32), (sh, sw), order=order)
        return np.round(resized).astype(mask.dtype)

    # ──────────────────────────────────────────────────────────────────────
    def _calculate_multiclass_iou(
        self,
        pred: MaskArray,
        gt: MaskArray,
        ignore_index: int = 255,
    ) -> Tuple[float, Dict[int, float]]:
        """Расчёт mIoU для многоклассовой сегментации.

        Args:
            pred: Маска предсказания.
            gt: Ground truth маска.
            ignore_index: Индекс класса, игнорируемый при расчёте (обычно 255).

        Returns:
            Tuple[float, Dict[int, float]]: (средний IoU, словарь {class_id: iou}).

        Note:
            Union=0 обрабатывается как IoU=0.0 для данного класса.
        """
        valid_mask = gt != ignore_index
        if not valid_mask.any():
            return 0.0, {}

        pred_valid: np.ndarray = np.where(valid_mask, pred, ignore_index)
        gt_valid: MaskArray = gt
        classes: np.ndarray = np.unique(np.concatenate([gt_valid[valid_mask], pred_valid[valid_mask]]))
        iou_per_class: Dict[int, float] = {}

        for cls in classes:
            if cls == ignore_index:
                continue
            pred_cls = (pred == cls).astype(np.uint8)
            gt_cls = (gt == cls).astype(np.uint8)
            intersection = np.logical_and(pred_cls, gt_cls).sum()
            union = np.logical_or(pred_cls, gt_cls).sum()
            iou_per_class[int(cls)] = intersection / union if union > 0 else 0.0

        valid_ious: List[float] = [v for v in iou_per_class.values() if v >= 0]
        mean_iou: float = float(np.mean(valid_ious)) if valid_ious else 0.0
        return mean_iou, iou_per_class

    # ──────────────────────────────────────────────────────────────────────
    def _calculate_binary_metrics(
        self,
        pred: MaskArray,
        gt: MaskArray,
        metrics_list: Optional[List[str]] = None,
    ) -> MetricsDict:
        """Расчёт бинарных метрик сегментации (объект vs фон).

        Args:
            pred: Маска предсказания.
            gt: Ground truth маска.
            metrics_list: Список метрик для расчёта.

        Returns:
            MetricsDict: Словарь с IoU, Dice, Precision, Recall, F1, MAE, Hausdorff.
        """
        pred_binary: np.ndarray = (pred > 0).astype(np.uint8)
        gt_binary: np.ndarray = (gt > 0).astype(np.uint8)
        return SegmentationMetrics.calculate_all_metrics(
            pred_mask=pred_binary,
            gt_mask=gt_binary,
            threshold=0.5,
            include_hausdorff=True,
            metrics_list=metrics_list or self.config.metrics,
        )

    # ──────────────────────────────────────────────────────────────────────
    def _calculate_comprehensive_metrics(
        self,
        pred: MaskArray,
        gt: MaskArray,
        num_classes: int,
        ignore_index: int = 255,
        compute_boundary_f1: bool = False,
    ) -> Dict[str, Any]:
        """Расширенный расчёт метрик с per-class статистикой и Boundary F1.

        Args:
            pred: Маска предсказания.
            gt: Ground truth маска.
            num_classes: Общее количество классов в датасете.
            ignore_index: Индекс игнорируемого класса.
            compute_boundary_f1: Флаг расчёта метрики точности границ.

        Returns:
            Dict[str, Any]: Словарь.
            - m_iou, iou_per_class_{cls}.
            - m_dice, dice_per_class_{cls}.
            - boundary_f1 (опционально).
            - per-class precision/recall.
        """
        results: Dict[str, Any] = {}

        # ──────────────────────────────────────────────────────────────
        # 1. mIoU и per-class IoU
        # ──────────────────────────────────────────────────────────────
        m_iou, iou_per_class = self._calculate_multiclass_iou(pred, gt, ignore_index)
        results["m_iou"] = m_iou
        for cls, iou in iou_per_class.items():
            results[f"iou_class_{cls}"] = iou

        # ──────────────────────────────────────────────────────────────
        # 2. mDice и per-class Dice
        # ──────────────────────────────────────────────────────────────
        if getattr(self.config, "per_class_metrics", False):
            dice_per_class: Dict[int, float] = {}
            for cls in range(num_classes):
                if cls == ignore_index:
                    continue
                pred_c: np.ndarray = pred == cls
                gt_c: np.ndarray = gt == cls
                intersection: int = int(np.logical_and(pred_c, gt_c).sum())
                union: int = int(pred_c.sum() + gt_c.sum())
                dice: float = (2.0 * intersection) / (union + 1e-8) if union > 0 else 0.0
                dice_per_class[cls] = dice
                results[f"dice_class_{cls}"] = dice

            results["m_dice"] = float(np.mean(list(dice_per_class.values()))) if dice_per_class else 0.0

            # ──────────────────────────────────────────────────────────────
            # 3. Per-class Precision/Recall (для анализа дисбаланса)
            # ──────────────────────────────────────────────────────────────
            for cls in range(min(20, num_classes)):  # Ограничим первыми 10 классами для экономии
                if cls == ignore_index:
                    continue
                pred_c = (pred == cls).astype(np.uint8)
                gt_c = (gt == cls).astype(np.uint8)
                tp = np.logical_and(pred_c, gt_c).sum()
                fp = np.logical_and(pred_c, ~gt_c).sum()
                fn = np.logical_and(~pred_c, gt_c).sum()

                prec: float = tp / (tp + fp + 1e-8)
                rec: float = tp / (tp + fn + 1e-8)
                results[f"precision_class_{cls}"] = prec
                results[f"recall_class_{cls}"] = rec

        # ──────────────────────────────────────────────────────────────
        # 4. Boundary F1 (точность границ) — опционально, медленно!
        # ──────────────────────────────────────────────────────────────
        if compute_boundary_f1:
            boundary_f1_scores: List[float] = []
            for cls in range(min(5, num_classes)):  # Только первые 5 классов
                if cls == ignore_index:
                    continue
                pred_mask_c: np.ndarray = pred == cls
                gt_mask_c: np.ndarray = gt == cls

                # Границы = dilation XOR erosion
                pred_boundary = binary_dilation(pred_mask_c) ^ binary_erosion(pred_mask_c)
                gt_boundary = binary_dilation(gt_mask_c) ^ binary_erosion(gt_mask_c)

                tp_b = np.logical_and(pred_boundary, gt_boundary).sum()
                fp_b = np.logical_and(pred_boundary, ~gt_boundary).sum()
                fn_b = np.logical_and(~pred_boundary, gt_boundary).sum()

                f1_b: float = (2 * tp_b) / (2 * tp_b + fp_b + fn_b + 1e-8)
                boundary_f1_scores.append(f1_b)

            if boundary_f1_scores:
                results["boundary_f1"] = float(np.mean(boundary_f1_scores))

        return results

    # ──────────────────────────────────────────────────────────────────────
    def _save_overlay_if_needed(
        self,
        checkpoint: ModelCheckpoint,
        img_path: Path,
        overlay: Union[np.ndarray, Image.Image],
    ) -> None:
        """Сохранение оверлея с контролем памяти (LRU-кэш).

        Args:
            checkpoint: Метаданные модели.
            img_path: Путь к исходному изображению.
            overlay: Визуализация (numpy или PIL).

        Note:
            При достижении `overlay_cache_max` самый старый оверлей удаляется из памяти.
        """
        if not self.config.save_overlays:
            return

        # LRU-кэш: удаляем самый старый, если достигнут лимит
        if len(self.overlays) >= self.config.overlay_cache_max:
            oldest_key: str = next(iter(self.overlays))
            self.overlays.pop(oldest_key)
            logger.debug(f"🗑️ Удалён старый оверлей из кэша: {oldest_key}")
            torch.cuda.empty_cache()

        # Ключ в формате "{модель}_{аугментация}_{изображение}"
        overlay_key: str = f"{checkpoint.key}_{img_path.stem}"

        # Конвертация в PIL с обработкой всех форматов
        try:
            overlay_pil: Image.Image = ensure_pil_compatible(overlay)
            self.overlays[overlay_key] = overlay_pil
            logger.debug(f"✅ Оверлей добавлен: {overlay_key}")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось сохранить оверлей {overlay_key}: {e}")

    # ──────────────────────────────────────────────────────────────────────
    def _test_single_model(
        self,
        checkpoint: ModelCheckpoint,
        image_pairs: List[Tuple[Path, Path]],
        precision: str = "fp32",
        cache: Optional[PredictionCache] = None,
        config_hash: str = "",
        profile_dir: Optional[Path] = None,
    ) -> List[TestResult]:
        """Инференс одной модели на наборе изображений с расчётом метрик.

        Args:
            checkpoint: Объект ModelCheckpoint для загрузки.
            image_pairs: Список кортежей (img_path, mask_path).
            precision: Точность вычислений (fp32, fp16, bf16).
            cache: Объект PredictionCache (опционально).
            config_hash: Хэш конфигурации для ключей кэша.
            profile_dir: Директория для профилирования (опционально).

        Returns:
            List[TestResult]: Список результатов по каждому изображению.

        Note:
            Поддерживает resume, кэширование предсказаний, autocast и fallback визуализации.
        """
        results: List[TestResult] = []

        # === 1. Загрузка модели с учётом точности ===
        dtype: torch.dtype = self._resolve_torch_dtype(precision)
        if not _check_precision_support(None, dtype, self.config.device):
            logger.warning(f"⚠️ {precision} не поддерживается на {self.config.device}, fallback на fp32")
            dtype = torch.float32
            precision = "fp32"

        if self.config.verbose:
            actual_precision = "fp16" if dtype == torch.float16 else "bf16" if dtype == torch.bfloat16 else "fp32"
            logger.info(f"🎯 Точность инференса: {actual_precision} (запрошено: {precision})")

        segmenter = NeuralSegmenter(
            model_type=checkpoint.model_type,
            checkpoint_path=str(checkpoint.path),
            device=self.config.device,
            num_classes=self.config.num_classes,
            palette=NeuralSegmenter.ade_palette(),
        )
        segmenter.model.eval()

        if dtype != torch.float32:
            segmenter.model = segmenter.model.to(dtype)

        autocast_enabled: bool = (dtype != torch.float32) and (self.config.device == "cuda")

        if self.config.verbose:
            logger.info(f"Тестирование {checkpoint.display_name} на {len(image_pairs)} изображениях...")

        completed: Set[str] = self._load_completed_runs()

        for idx, (img_path, mask_path) in enumerate(
            tqdm(image_pairs, desc=checkpoint.key, disable=not self.config.verbose)
        ):
            try:
                test_image: Image.Image = Image.open(img_path).convert("RGB")
                gt_mask_pil: Image.Image = Image.open(mask_path)
                gt_mask: np.ndarray = np.array(gt_mask_pil)
                if gt_mask.ndim == 3 and gt_mask.shape[2] == 3:
                    gt_mask = gt_mask[:, :, 0]

                task_key: str = f"{checkpoint.key}:{img_path.name}"
                if task_key in completed:
                    if self.config.verbose:
                        logger.debug(f"⏭  Пропущено: {task_key}")
                    continue

                cached_pred: Optional[np.ndarray] = None
                if cache is not None and config_hash:
                    cache_key: str = cache._get_key(checkpoint.path, img_path, config_hash)
                    cached_pred = cache.get(cache_key)
                    if cached_pred is not None and self.config.verbose:
                        logger.debug(f"♻️  Кэш-хит: {cache_key}")

                # === 2. Инференс с контролем точности ===
                with safe_inference_context(checkpoint.key, img_path.name):
                    amp_ctx = torch.amp.autocast(self.config.device, dtype=dtype) if autocast_enabled else nullcontext()

                    if cached_pred is not None:
                        pred_mask = cached_pred
                        inference_time = 0.0  # Не считаем время для кэша
                    else:
                        with amp_ctx, torch.no_grad():
                            start_time: float = time.perf_counter()
                            input_raw = segmenter.preprocess_image(test_image)

                            # Конвертация в Tensor с явной типизацией
                            if isinstance(input_raw, np.ndarray):
                                input_tensor: torch.Tensor = torch.from_numpy(input_raw).float()
                            elif isinstance(input_raw, torch.Tensor):
                                # Если уже тензор — приводим к нужной точности
                                input_tensor = input_raw.float() if input_raw.is_floating_point() else input_raw
                            else:
                                raise TypeError(f"Неожиданный тип входных данных: {type(input_raw)}")

                            # Перенос на устройство + приведение к точности (одной операцией)
                            input_tensor = input_tensor.to(
                                device=self.config.device,
                                dtype=dtype,
                                non_blocking=True,
                            )
                            if input_tensor.dtype != dtype:
                                input_tensor = input_tensor.to(dtype=dtype, non_blocking=True)

                            if hasattr(segmenter, "_forward"):
                                with (
                                    torch.amp.autocast(self.config.device, dtype=dtype)
                                    if autocast_enabled
                                    else nullcontext()
                                ):
                                    output = segmenter._forward(input_tensor.unsqueeze(0))
                                    pred_mask = output.squeeze(0).argmax(dim=0).cpu().numpy()
                            else:
                                # Fallback: обрабатываем разный return type
                                result: Tuple[np.ndarray, Dict[str, Any]] = segmenter.predict_segmentation_map(
                                    test_image, verbose=False, gt_mask=gt_mask
                                )
                                if isinstance(result, tuple) and len(result) >= 2:
                                    pred_mask, _ = result
                                else:
                                    pred_mask = result

                            if self.config.device == "cuda":
                                torch.cuda.synchronize()
                            inference_time = time.perf_counter() - start_time

                        if cache is not None and config_hash and pred_mask is not None:
                            cache_key = cache._get_key(checkpoint.path, img_path, config_hash)
                            cache.set(cache_key, pred_mask)

                    self._mark_completed(checkpoint.key, img_path.name)

                if gt_mask.shape != pred_mask.shape:
                    pred_resized = self._resize_mask(pred_mask, gt_mask.shape)
                else:
                    pred_resized = pred_mask

                m_iou, _ = self._calculate_multiclass_iou(pred_resized, gt_mask, self.config.ignore_index)
                binary_metrics = self._calculate_binary_metrics(pred_resized, gt_mask, self.config.metrics)
                comprehensive = self._calculate_comprehensive_metrics(
                    pred_resized,
                    gt_mask,
                    num_classes=self.config.num_classes,
                    ignore_index=self.config.ignore_index,
                    compute_boundary_f1=getattr(self.config, "compute_boundary_f1", False),
                )

                # Визуализация: вынесено в отдельный метод + безопасный вызов
                overlay: Image.Image
                if self.config.save_visualizations:
                    try:
                        if dtype != torch.float32 and hasattr(segmenter, "segment_image_unified"):
                            model_for_viz: nn.Module = cast(nn.Module, segmenter.model)
                            original_dtype: torch.dtype = self._get_model_dtype(model_for_viz)

                            model_for_viz = model_for_viz.float()
                            overlay_result = segmenter.segment_image_unified(  # type: ignore[union-attr]
                                test_image,
                                alpha=0.6,
                                class_names=NeuralSegmenter.get_ade_class_names(),
                            )
                            model_for_viz = model_for_viz.to(original_dtype)
                            if overlay_result is None:
                                logger.warning(f"⚠️ segment_image_unified вернул None для {img_path.name}")
                                overlay = self._create_simple_overlay(test_image, pred_resized)
                            else:
                                if isinstance(overlay_result, tuple) and len(overlay_result) > 0:
                                    overlay = overlay_result[0]
                                else:
                                    overlay = overlay_result
                            if overlay is None:
                                if getattr(self.config, "class_aware_overlays", False):
                                    overlay = self._create_class_aware_overlay(
                                        test_image,
                                        pred_resized,
                                        gt_mask,
                                        palette=NeuralSegmenter.ade_palette(),
                                        class_names=NeuralSegmenter.get_ade_class_names(),
                                        alpha=getattr(self.config, "overlay_alpha", 0.5),
                                        show_legend=True,
                                    )
                                    logger.debug(f"🎨 Создан class-aware overlay для {img_path.name}")
                                else:
                                    overlay = self._create_simple_overlay(test_image, pred_resized)
                        elif hasattr(segmenter, "_segment_with_visualization"):
                            tensor_input = segmenter.preprocess_image(test_image)
                            overlay, _ = segmenter._segment_with_visualization(
                                tensor_input,
                                alpha=self.config.viz_alpha,
                                color=self.config.viz_color,
                                precision="fp32",
                            )
                        else:
                            # Fallback: простой оверлей через PIL
                            overlay = self._create_simple_overlay(test_image, pred_resized)

                        # Сохранение через вынесенный метод
                        self._save_overlay_if_needed(checkpoint, img_path, overlay)

                    except Exception as e:
                        logger.warning(f"⏭ Overlay skipped: {e}")
                        # Fallback: сохранить предсказанную маску
                        try:
                            if pred_mask is not None:
                                mask_vis = (pred_mask * 255 / (pred_mask.max() + 1e-8)).astype(np.uint8)
                                if mask_vis.ndim == 2:
                                    mask_vis = np.stack([mask_vis] * 3, axis=-1)
                                viz_dir: Path = Path(self.config.output_dir) / "overlays"
                                viz_dir.mkdir(parents=True, exist_ok=True)
                                Image.fromarray(mask_vis).save(
                                    viz_dir / f"{checkpoint.key}_{img_path.stem}_fallback.png"
                                )
                        except ValueError:
                            pass

                if self.config.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

                # Объединение метрик
                metrics: MetricsDict = {
                    "m_iou": m_iou,
                    "binary_iou": binary_metrics.get("iou", 0),
                    **{k: v for k, v in binary_metrics.items() if k != "iou"},
                }
                if "boundary_f1" in comprehensive:
                    metrics["boundary_f1"] = comprehensive["boundary_f1"]

                test_result: TestResult = TestResult(
                    model_key=checkpoint.key,
                    image_name=img_path.name,
                    metrics=metrics,
                    inference_time=inference_time,
                    precision=precision,
                )
                results.append(test_result)

            except Exception as e:
                logger.error(f"Ошибка при обработке {img_path.name}: {e}")
                continue

        # Очистка памяти
        del segmenter
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        return results

    # ──────────────────────────────────────────────────────────────────────
    def _get_model_dtype(self, model: nn.Module) -> torch.dtype:
        """Получить dtype модели с фоллбэком на первый параметр.

        Args:
            model: PyTorch модуль.

        Returns:
            torch.dtype: Точность модели.
        """
        try:
            # return cast(torch.dtype, model.dtype)  # type: ignore[attr-defined]
            raw_dtype = model.dtype  # type: ignore[attr-defined]
            if isinstance(raw_dtype, torch.dtype):
                return raw_dtype
            else:
                raise TypeError(f"Unexpected dtype type: {type(raw_dtype)}")
        except AttributeError:
            # Фоллбэк: берём dtype первого параметра
            first_param: nn.Parameter = next(iter(model.parameters()))
            return first_param.dtype

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_torch_dtype(precision: str) -> torch.dtype:
        """Преобразование строкового обозначения точности в torch.dtype.

        Args:
            precision: Строка "fp32", "fp16" или "bf16".

        Returns:
            torch.dtype: Соответствующий тип данных. По умолчанию torch.float32.
        """
        mapping: Dict[str, torch.dtype] = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(precision.lower(), torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    def _create_simple_overlay(self, image: Image.Image, pred: MaskArray, alpha: float = 0.5) -> Image.Image:
        """Создание базового оверлея через PIL (fallback метод).

        Args:
            image: Оригинальное RGB изображение.
            pred: Маска предсказания (2D массив).
            alpha: Коэффициент прозрачности наложения.

        Returns:
            Image.Image: Смешанное изображение.
        """
        mask_np: MaskArray = pred.copy()
        if mask_np.max() <= 1.0:
            mask_np = (mask_np * 255).astype(np.uint8)
        elif mask_np.dtype != np.uint8:
            mask_np = mask_np.astype(np.uint8)

        orig: Image.Image = image.convert("RGB")
        mask_pil: Image.Image = Image.fromarray(mask_np, mode="L").convert("RGB")
        # Ресайз маски если размеры не совпадают
        if mask_pil.size != orig.size:
            mask_pil = mask_pil.resize(orig.size, Image.Resampling.NEAREST)

        overlay: Image.Image = Image.blend(orig, mask_pil, alpha=alpha)
        return overlay

    # ──────────────────────────────────────────────────────────────────────
    def _init_experiment_tracking(self) -> None:
        """Инициализация трекеров экспериментов (MLflow или Weights & Biases).

        Note:
            Если оба флага включены, приоритет отдаётся MLflow.
            При отсутствии авторизации W&B автоматически переключается в offline-режим.
        """
        self.tracker = None

        if getattr(self.config, "use_mlflow", False):
            try:
                import mlflow

                mlflow.set_experiment("ADE20K_Augmentation_Analysis")
                mlflow.start_run(run_name=f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}")
                mlflow.log_params(asdict(self.config))
                self.tracker = "mlflow"
                logger.info("✅ MLflow инициализирован")
            except ImportError:
                logger.warning("⚠️  mlflow не установлен, пропуск")
            except Exception as e:
                logger.warning(f"⚠️  Ошибка инициализации MLflow: {e}")

        elif getattr(self.config, "use_wandb", False):
            try:
                import wandb

                # Проверка авторизации
                # if not wandb.api.api_key:
                #     logger.warning("⚠️  WandB не авторизован. Запусти 'wandb login' в терминале")
                #     return
                logger.info("Entered to the function!")
                run = wandb.init(
                    project="segmentation-aug-analysis",
                    config=asdict(self.config),
                    name=f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}",
                    # Авто-режим: онлайн если есть ключ, иначе офлайн
                    mode="online" if wandb.api.api_key else "offline",
                )
                if run:
                    logger.info(f"✅ W&B run URL: {run.url}")
                self.tracker = "wandb"
                if wandb.api.api_key:
                    logger.info("✅ Weights & Biases инициализирован (онлайн)")
                else:
                    logger.info("✅ Weights & Biases инициализирован (офлайн-режим)")
            except ImportError:
                logger.warning("⚠️  wandb не установлен. Установи: pip install wandb")
            except Exception as e:
                logger.warning(f"⚠️  Ошибка инициализации W&B: {type(e).__name__}: {e}")
                # Fallback: продолжаем без wandb
                pass

    # ──────────────────────────────────────────────────────────────────────
    def _log_metrics(
        self,
        model_key: str,
        metrics: MetricsDict,
        step: Optional[int] = None,
        prefix: str = "test",
    ) -> None:
        """Логирование метрик в активный трекер.

        Args:
            model_key: Идентификатор модели.
            metrics: Словарь метрик для логирования.
            step: Номер шага/эпохи (опционально).
            prefix: Префикс для имён метрик в трекере.
        """
        if self.tracker == "mlflow":
            try:
                import mlflow

                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        mlflow.log_metric(f"{prefix}/{model_key}/{k}", v, step=step)
            except ImportError:
                pass
        elif self.tracker == "wandb":
            try:
                import wandb

                log_dict: Dict[str, Any] = {
                    f"{prefix}/{model_key}/{k}": v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and not np.isnan(v)
                }
                if step is not None:
                    log_dict["step"] = step
                wandb.log(log_dict)
            except ImportError:
                pass

    # ──────────────────────────────────────────────────────────────────────
    def _close_experiment_tracking(self) -> None:
        """Корректное завершение сессий трекеров."""
        if self.tracker == "mlflow":
            try:
                import mlflow

                mlflow.end_run()
                logger.info("✅ MLflow закрыт")
            except ImportError:
                pass
        elif self.tracker == "wandb":
            try:
                import wandb

                wandb.finish()
                logger.info("✅ Weights & Biases закрыт")
            except ImportError:
                pass

    # ──────────────────────────────────────────────────────────────────────
    def _load_completed_runs(self) -> Set[str]:
        """Загрузка состояния завершённых задач для resume.

        Returns:
            Set[str]: Множество ключей вида "{model_key}:{image_name}".
        """
        done_file: Path = Path(self.config.output_dir) / ".completed.json"
        if done_file.exists():
            with open(done_file) as f:
                return set(json.load(f))
        return set()

    # ──────────────────────────────────────────────────────────────────────
    def _mark_completed(self, model_key: str, image_name: str) -> None:
        """Сохранение информации о завершённой задаче.

        Args:
            model_key: Ключ модели.
            image_name: Имя изображения.
        """
        done_file: Path = Path(self.config.output_dir) / ".completed.json"
        completed: Set[str] = self._load_completed_runs()
        completed.add(f"{model_key}:{image_name}")
        with open(done_file, "w") as f:
            json.dump(list(completed), f)

    # ──────────────────────────────────────────────────────────────────────
    def _batch_predict_with_memory_control(
        self,
        segmenter: Any,
        images: List[Image.Image],
        batch_size: int = 4,
        target_size: Optional[Tuple[int, int]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> List[np.ndarray]:
        """Пакетное предсказание с автоматическим уменьшением batch_size при OOM.

        Args:
            segmenter: Экземпляр NeuralSegmenter.
            images: Список PIL.Image.
            batch_size: Начальный размер батча.
            target_size: Опциональный ресайз входов.
            dtype: Точность вычислений.

        Returns:
            List[np.ndarray]: Список предсказанных масок.
        """
        predictions: List[np.ndarray] = []
        current_batch_size: int = batch_size

        for i in range(0, len(images), current_batch_size):
            batch: List[Image.Image] = images[i : i + current_batch_size]

            while True:
                try:
                    with torch.no_grad():
                        preds: List[np.ndarray] = []
                        for img in batch:
                            if target_size and img.size != target_size:
                                img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
                                input_tensor = segmenter.preprocess_image(img_resized)
                            else:
                                input_tensor = segmenter.preprocess_image(img)

                            if isinstance(input_tensor, np.ndarray):
                                input_tensor = torch.from_numpy(input_tensor).float()

                            input_tensor = input_tensor.to(self.config.device, dtype=dtype, non_blocking=True)

                            # Инференс
                            if hasattr(segmenter, "_forward"):
                                output = segmenter._forward(input_tensor.unsqueeze(0))
                                pred_mask = output.squeeze(0).argmax(dim=0).cpu().numpy()
                            else:
                                result = segmenter.predict_segmentation_map(img, verbose=False)
                                pred_mask = result[0] if isinstance(result, tuple) else result

                            preds.append(pred_mask)

                    predictions.extend(preds)
                    break

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()

                    if current_batch_size == 1:
                        logger.error("❌ OOM даже при batch_size=1, пропускаем батч")
                        break

                    current_batch_size = max(1, current_batch_size // 2)
                    logger.warning(f"⚠️  OOM, уменьшаем batch_size до {current_batch_size}")
                    batch = images[i : i + current_batch_size]

                except Exception as e:
                    logger.error(f"❌ Ошибка при предсказании: {e}")
                    break

        return predictions

    # ──────────────────────────────────────────────────────────────────────
    def _profile_model_inference(
        self,
        model: nn.Module,
        model_key: str,
        sample_input: torch.Tensor,
        output_dir: Path,
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> Dict[str, Any]:
        """Детальное профилирование инференса через torch.profiler.

        Args:
            model: PyTorch модель.
            model_key: Уникальный ключ модели (например, "unet_smp_none").
            sample_input: Пример входного тензора.
            output_dir: Директория для сохранения trace и стеков.
            num_warmup: Количество итераций прогрева.
            num_runs: Количество измеряемых прогонов.

        Returns:
            Dict[str, Any]: Словарь с метриками времени, памяти, FLOPs и топ-операциями.

        Note:
            Экспортирует Chrome Trace JSON и текстовые стеки вызовов.
            мена файлов включают model_key для уникальности.
        """
        import torch.profiler as profiler

        model.eval()
        sample_input = sample_input.to(self.config.device)

        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(sample_input)
            if self.config.device == "cuda":
                torch.cuda.synchronize()

        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            with profiler.record_function("model_inference"):
                for _ in range(num_runs):
                    with torch.no_grad():
                        _ = model(sample_input)
                    if self.config.device == "cuda":
                        torch.cuda.synchronize()

        results: Dict[str, Any] = {
            "total_time_ms": getattr(prof, "cpu_time_total", getattr(prof, "self_cpu_time_total", 0)) / 1e3 / num_runs,
            "cuda_time_ms": getattr(prof, "cuda_time_total", getattr(prof, "self_cuda_time_total", 0)) / 1e3 / num_runs,
            "cpu_time_ms": getattr(prof, "cpu_time_total", getattr(prof, "self_cpu_time_total", 0)) / 1e3 / num_runs,
            "memory_allocated_mb": (torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0),
            "flops": sum(e.flops for e in prof.key_averages() if e.flops > 0),
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(output_dir / f"trace_{model_key}.json"))
        prof.export_stacks(
            str(output_dir / f"stacks_{model_key}.txt"),
            "self_cuda_time_total",
        )

        try:
            # Новое API + fallback для совместимости
            key_avgs = prof.key_averages()

            if hasattr(key_avgs, "table"):
                top_ops_table = key_avgs.table(sort_by="cuda_time_total", row_limit=10)
                results["top_ops_table"] = str(top_ops_table)

            results["top_ops"] = [
                {
                    "name": getattr(e, "key", str(e)),
                    "cuda_time_ms": getattr(e, "cuda_time_total", getattr(e, "self_cuda_time_total", 0)) / 1e3,
                    "calls": getattr(e, "count", 0),
                }
                for e in list(key_avgs)[:10]
            ]
        except Exception as e_prof:
            logger.warning(f"⚠️  Ошибка форматирования профиля: {e_prof}")
            results["top_ops"] = []

        return results

    # ──────────────────────────────────────────────────────────────────────
    def _create_class_aware_overlay(
        self,
        image: Image.Image,
        pred: MaskArray,
        gt: Optional[MaskArray] = None,
        palette: Optional[List[List[int]]] = None,
        class_names: Optional[List[str]] = None,
        alpha: float = 0.5,
        show_legend: bool = True,
        max_classes_legend: int = 10,
    ) -> Image.Image:
        """Создание overlay с разными цветами для разных классов + опциональная легенда.

        Args:
            image: Оригинальное изображение (PIL).
            pred: Предсказанная маска (H×W, int).
            gt: Ground truth для сравнения (опционально).
            palette: Словарь {class_id: (R,G,B)} или None для дефолтной палитры.
            class_names: Список имён классов для легенды.
            alpha: Прозрачность наложения (0.0–1.0).
            show_legend: Показывать ли легенду.
            max_classes_legend: Максимальное число классов в легенде.

        Returns:
            Image.Image: Итоговое изображение с наложением и текстом.
        """
        # ──────────────────────────────────────────────────────────────
        # 1. Инициализация палитры
        # ──────────────────────────────────────────────────────────────
        if palette is None:
            if hasattr(NeuralSegmenter, "ade_palette"):
                palette = NeuralSegmenter.ade_palette()
            if not palette:
                np.random.seed(42)
                palette = [[int(c) for c in np.random.randint(0, 255, 3)] for _ in range(150)]

        # ──────────────────────────────────────────────────────────────
        # 2. Конвертация предсказания в цветное изображение
        # ──────────────────────────────────────────────────────────────
        h, w = pred.shape
        pred_color: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)

        for cls_id in range(min(len(palette), int(pred.max()) + 1)):
            mask: np.ndarray = pred == cls_id
            if mask.any():
                color: List[int] = palette[cls_id]
                # Конвертируем [R,G,B] в tuple для индексации массива
                pred_color[mask] = tuple(color)  # type: ignore[assignment]

        # ──────────────────────────────────────────────────────────────
        # 3. Подготовка оригинального изображения и ресайз
        # ──────────────────────────────────────────────────────────────
        img_array: np.ndarray = np.array(image.convert("RGB"))

        if img_array.shape[:2] != pred_color.shape[:2]:
            sh: float = img_array.shape[0] / pred_color.shape[0]
            sw: float = img_array.shape[1] / pred_color.shape[1]
            pred_color = zoom(pred_color, (sh, sw, 1), order=1).astype(np.uint8)

        # Наложение с прозрачностью
        overlay_array: np.ndarray = (img_array * (1 - alpha) + pred_color * alpha).astype(np.uint8)

        overlay_pil: Image.Image = Image.fromarray(overlay_array)

        # ──────────────────────────────────────────────────────────────
        # 4. Добавление легенды (если запрошено)
        # ──────────────────────────────────────────────────────────────
        if show_legend and class_names and palette is not None:
            draw: ImageDraw.ImageDraw = ImageDraw.Draw(overlay_pil)

            # Шрифт с безопасным fallback
            font: ImageFont.FreeTypeFont | ImageFont.ImageFont
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=10)
            except (OSError, IOError):
                font = ImageFont.load_default()

            # Позиция легенды
            legend_x: int = overlay_pil.width - 180
            legend_y: int = 10

            # Фон легенды
            draw.rectangle(
                [legend_x - 5, legend_y - 5, legend_x + 175, legend_y + 200],
                fill=(255, 255, 255, 200),
                outline=(0, 0, 0),
            )

            draw.text((legend_x, legend_y), "Classes:", font=font, fill=(0, 0, 0))
            legend_y += 18

            # Фильтрация классов: только те, что есть в предсказании и имеют имя
            available_classes: List[int] = []
            for cls_id in range(len(palette)):
                if cls_id < len(class_names) and np.any(pred == cls_id):
                    available_classes.append(cls_id)

            # Сортировка по частоте встречаемости (по убыванию)
            shown_classes: List[int] = sorted(available_classes, key=lambda c: int(np.sum(pred == c)), reverse=True)[
                :max_classes_legend
            ]

            for cls_id in shown_classes:
                if cls_id >= len(palette) or cls_id >= len(class_names):
                    continue

                # Доступ к цвету: палитра — список, индекс = class_id
                color_list: List[int] = palette[cls_id]
                color_tuple: Tuple[int, int, int] = tuple(color_list)  # type: ignore[assignment]

                name: str = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"

                draw.rectangle(
                    [legend_x, legend_y, legend_x + 12, legend_y + 12],
                    fill=color_tuple,
                    outline=(0, 0, 0),
                )
                draw.text(
                    (legend_x + 18, legend_y + 2),
                    f"{name[:25]}",
                    font=font,
                    fill=(0, 0, 0),
                )
                legend_y += 18

                if legend_y > 190:
                    break

            if len(shown_classes) < len(available_classes):
                draw.text(
                    (legend_x, legend_y + 2),
                    f"... +{len(available_classes) - len(shown_classes)} more",
                    font=font,
                    fill=(100, 100, 100),
                )

        return overlay_pil

    # ──────────────────────────────────────────────────────────────────────
    def _export_model_to_onnx_trt(
        self,
        model: nn.Module,
        model_key: str,
        sample_input: torch.Tensor,
        output_dir: Path,
        opset_version: int = 17,
        trt_precision: Literal["fp32", "fp16"] = "fp16",
    ) -> Dict[str, Optional[Path]]:
        """Экспорт модели в ONNX и компиляция в TensorRT с fallback-механизмами.

        Args:
            model: PyTorch модель.
            model_key: Уникальный ключ модели для именования файлов.
            sample_input: Пример входного тензора.
            output_dir: Директория для сохранения.
            opset_version: Версия ONNX opset.
            trt_precision: Точность TensorRT ("fp32" или "fp16").

        Returns:
            Dict[str, Optional[Path]]: Пути к сгенерированным файлам или None при ошибке.

        Note:
            - Гарантирует нахождение модели и входных данных на CPU перед экспортом.
            - Пробует `export_params=True`, при падении fallback на `False`.
            - TRT использует `ir="ts"` (TorchScript) как workaround для новых версий.
        """
        import torch.onnx

        output_dir.mkdir(parents=True, exist_ok=True)
        results: Dict[str, Optional[Path]] = {"onnx": None, "trt": None}

        onnx_path: Path = output_dir / f"{model_key}.onnx"

        model_cpu = model.to("cpu").eval()
        sample_input_cpu = sample_input.to("cpu", dtype=torch.float32)

        # Очищаем все буферы и параметры от CUDA
        for param in model_cpu.parameters():
            if param.device.type != "cpu":
                param.data = param.data.cpu()
        for buf in model_cpu.buffers():
            if buf.device.type != "cpu":
                buf.data = buf.data.cpu()

        try:
            dynamic_axes: Optional[Dict[str, Dict[int, str]]] = (
                {
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch", 2: "height", 3: "width"},
                }
                if getattr(self.config, "dynamic_shapes", False)
                else None
            )

            torch.onnx.export(
                model_cpu,
                (sample_input_cpu,),
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                verbose=False,
            )

            # Опциональное упрощение
            try:
                import onnx
                from onnxsim import simplify

                model_onnx = onnx.load(str(onnx_path))
                model_simp, check = simplify(model_onnx)
                if check:
                    onnx.save(model_simp, str(onnx_path))
                    logger.info(f"✅ ONNX simplified: {onnx_path}")
            except ImportError:
                pass

            results["onnx"] = onnx_path
            logger.info(f"✅ ONNX exported: {onnx_path}")

        except Exception as e:
            logger.error(f"❌ ONNX export failed (params=True): {type(e).__name__}: {e}")

            # Запасной вариант: экспорт без весов
            try:
                logger.info("🔄 Retrying ONNX export with export_params=False...")
                torch.onnx.export(
                    model_cpu,
                    (sample_input_cpu,),
                    str(onnx_path),
                    export_params=False,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes=(
                        {
                            "input": {0: "batch", 2: "height", 3: "width"},
                            "output": {0: "batch", 2: "height", 3: "width"},
                        }
                        if getattr(self.config, "dynamic_shapes", False)
                        else None
                    ),
                    verbose=False,
                )
                results["onnx"] = onnx_path
                logger.info(f"✅ ONNX exported (no params): {onnx_path}")

            except Exception as e2:
                logger.error(f"❌ ONNX export failed completely: {type(e2).__name__}: {e2}")

                # Последняя попытка: смена opset
                if opset_version != 18:
                    logger.info("🔄 Retrying with opset 18...")
                    return self._export_model_to_onnx_trt(
                        model,
                        model_key,
                        sample_input,
                        output_dir,
                        opset_version=18,
                        trt_precision=trt_precision,
                    )

        # TensorRT экспорт (только если ONNX успешен и на CUDA)
        if results["onnx"] and self.config.device == "cuda" and getattr(self.config, "export_trt", False):
            trt_path: Path = output_dir / f"{model_key}.{trt_precision}.trt"
            try:
                import torch_tensorrt

                trt_version = tuple(map(int, torch_tensorrt.__version__.split(".")[:2]))
                if trt_version < (1, 4):
                    logger.warning(
                        f"⚠️  torch-tensorrt {torch_tensorrt.__version__} может быть несовместим. Пропускаем TRT экспорт."
                    )
                    results["trt"] = None
                else:
                    model_cuda = model.to(self.config.device).eval()
                    input_spec: List[torch_tensorrt.Input] = [
                        torch_tensorrt.Input(
                            sample_input.shape,
                            dtype=(torch.float16 if trt_precision == "fp16" else torch.float32),
                            name="input",
                        )
                    ]

                    trt_model = torch_tensorrt.compile(
                        model_cuda,
                        inputs=input_spec,
                        enabled_precisions={torch.float16 if trt_precision == "fp16" else torch.float32},
                        ir="ts",
                        # min_block_size=1,
                        # fallback_to_torch=True,
                    )

                    torch.jit.save(trt_model, str(trt_path))
                    results["trt"] = trt_path
                    logger.info(f"✅ TensorRT engine saved: {trt_path}")

            except ImportError:
                logger.warning("⚠️  torch-tensorrt not installed. Skip TRT export.")
                results["trt"] = None
            except Exception as e:
                logger.error(f"❌ TRT compile failed: {type(e).__name__}: {e}")
                import traceback

                logger.debug(f"🔍 Full traceback:\n{traceback.format_exc()}")

                # 🔥 Не ретраим с fp32 — ошибка не в точности, а в совместимости
                logger.warning("⚠️  TensorRT экспорт пропущен из-за ошибки компиляции или несовместимости версий")
                results["trt"] = None
                # if trt_precision == "fp16":
                #     logger.info("🔄 Retrying TRT with fp32...")
                #     return self._export_model_to_onnx_trt(
                #         model,
                #         model_key,
                #         sample_input,
                #         output_dir,
                #         opset_version=opset_version,
                #         trt_precision="fp32",
                #     )

        return results

    # ──────────────────────────────────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        """Запуск полного цикла тестирования: поиск чекпоинтов → инференс → агрегация.

        Returns:
            pd.DataFrame: Таблица с метриками, временем и метаданными по каждому изображению.

        Note:
            Включает профилирование первой модели и экспорт ONNX перед основным циклом,
            если соответствующие флаги установлены в config.
        """
        # if self.config.verbose:
        #     logger.info("🔧 Конфигурация запуска:")
        #     for k, v in vars(self.config).items(): logger.info(f"   • {k}: {v}")

        if self.config.verbose:
            logger.info("🔧 Конфигурация запуска:")
            logger.info(f"   • models_dir: {getattr(self.config, 'models_dir', './models')}")
            logger.info(f"   • random_seed: {self.config.random_seed}")
            logger.info(f"   • device: {self.config.device}")
            logger.info(f"   • precision: {self.config.precision}")
            logger.info(f"   • subset_size: {self.config.subset_size}")
            logger.info(f"   • batch_size: {self.config.batch_size}")
            logger.info(f"   • num_classes: {self.config.num_classes}")
            logger.info(f"   • ignore_index: {self.config.ignore_index}")
            logger.info(f"   • save_overlays: {self.config.save_overlays}")
            logger.info(f"   • overlay_cache_max: {self.config.overlay_cache_max}")
            logger.info(f"   • overlay_sample_rate: {self.config.overlay_sample_rate}")
            logger.info(f"   • verbose: {self.config.verbose}")
            logger.info(f"   • save_visualizations: {self.config.save_visualizations}")
            logger.info(f"   • viz_alpha: {self.config.viz_alpha}")
            logger.info(f"   • viz_color: {self.config.viz_color}")
            logger.info(f"   • precision: {self.config.precision}")
            logger.info(f"   • cache: {self.config.cache}")
            logger.info(f"   • cache_dir: {self.config.cache_dir}")
            logger.info(f"   • cache_max_gb: {self.config.cache_max_gb}")
            logger.info(f"   • clear_cache: {self.config.clear_cache}")
            logger.info(f"   • resume: {self.config.resume}")
            logger.info(f"   • export_trt: {self.config.export_trt}")
            logger.info(f"   • trt_precision: {self.config.trt_precision}")
            logger.info(f"   • opset: {self.config.opset}")
            logger.info(f"   • dynamic_shapes: {self.config.dynamic_shapes}")
            logger.info(f"   • profile: {self.config.profile}")
            logger.info(f"   • profile_output: {self.config.profile_output}")
            logger.info(f"   • profile_warmup: {self.config.profile_warmup}")
            logger.info(f"   • profile_runs: {self.config.profile_runs}")
            logger.info(f"   • compute_boundary: {self.config.compute_boundary_f1}")
            logger.info(f"   • per_class_metrics: {self.config.per_class_metrics}")
            logger.info(f"   • use_mlflow: {self.config.use_mlflow}")
            logger.info(f"   • use_wandb: {self.config.use_wandb}")
            logger.info(f"   • class_aware_overlays: {self.config.class_aware_overlays}")
            logger.info(f"   • overlay_alpha: {self.config.overlay_alpha}")
            logger.info(f"   • save vizualization: {self.config.save_viz}")

        checkpoints: Dict[str, ModelCheckpoint] = self._find_checkpoints()
        if not checkpoints:
            logger.error("Не найдено чекпоинтов для тестирования!")
            return pd.DataFrame()

        image_pairs: List[Tuple[Path, Path]] = self._load_ade20k_images(local_path=self.config.dataset_path)
        if not image_pairs:
            logger.error("Не удалось загрузить изображения датасета!")
            return pd.DataFrame()

        profile_dir: Optional[Path] = None
        if getattr(self.config, "profile", False):
            profile_dir = Path(getattr(self.config, "profile_output", "./profiling"))
            profile_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📊 Профилирование включено, вывод: {profile_dir}")

        if profile_dir is not None:
            profile_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📊 Профилирование: результаты будут в {profile_dir}")

        all_results: List[TestResult] = []
        for checkpoint in checkpoints.values():
            if profile_dir is not None and self.config.verbose:
                # Запускаем профилирование для первой модели (если verbose)
                try:
                    segmenter_temp: NeuralSegmenter = NeuralSegmenter(
                        model_type=checkpoint.model_type,
                        checkpoint_path=str(checkpoint.path),
                        device=self.config.device,
                        num_classes=self.config.num_classes,
                    )
                    dummy_input: torch.Tensor = torch.randn(1, 3, 512, 512, device=self.config.device)

                    profile_results: Dict[str, Any] = self._profile_model_inference(
                        model=segmenter_temp.model,
                        model_key=checkpoint.key,
                        sample_input=dummy_input,
                        output_dir=profile_dir,
                        num_warmup=getattr(self.config, "profile_warmup", 10),
                        num_runs=getattr(self.config, "profile_runs", 50),
                    )

                    logger.info(f"✅ Профиль сохранён: {profile_dir / f'trace_{checkpoint.key}.json'}")
                    if self.config.verbose:
                        logger.info(f"   📈 Среднее время: {profile_results['total_time_ms']:.2f} ms")
                        logger.info(f"   🧠 Память: {profile_results['memory_allocated_mb']:.1f} MB")

                except Exception as e:
                    logger.warning(f"⚠️  Ошибка профилирования {checkpoint.key}: {e}")
            config_hash: str = ""
            if self.cache is not None:
                config_dict: Dict[str, Any] = {
                    "precision": getattr(self.config, "precision", "fp32"),
                    "num_classes": self.config.num_classes,
                    "device": self.config.device,
                    "ignore_index": self.config.ignore_index,
                }
                config_hash = sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:16]

            if getattr(self.config, "export_onnx", False):
                export_dir: Path = Path(self.config.output_dir) / "exports"
                logger.info(f"📦 Экспорт моделей в: {export_dir}")
                try:
                    # Загружаем модель для экспорта
                    segmenter_temp = NeuralSegmenter(
                        model_type=checkpoint.model_type,
                        checkpoint_path=str(checkpoint.path),
                        device="cpu",  # Экспорт на CPU надёжнее
                        num_classes=self.config.num_classes,
                    )
                    segmenter_temp.model = segmenter_temp.model.to("cpu").eval()
                    for param in segmenter_temp.model.parameters():
                        param.data = param.data.cpu()
                    if hasattr(segmenter_temp.model, "buffers"):
                        for buf in segmenter_temp.model.buffers():
                            buf.data = buf.data.cpu()

                    # Создаём фиктивный вход для экспорта
                    dummy_input = torch.randn(1, 3, 512, 512, device="cpu", dtype=torch.float32)

                    export_results: Dict[str, Optional[Path]] = self._export_model_to_onnx_trt(
                        model=segmenter_temp.model,
                        model_key=checkpoint.key,
                        sample_input=dummy_input,
                        output_dir=Path(self.config.output_dir) / "exports",
                        opset_version=getattr(self.config, "opset", 17),
                        trt_precision=getattr(self.config, "trt_precision", "fp16"),
                    )

                    if export_results["onnx"]:
                        onnx_size: float = export_results["onnx"].stat().st_size / 1e6  # MB
                        logger.info(f"✅ ONNX: {export_results['onnx'].name} ({onnx_size:.2f} MB)")

                    if export_results["trt"]:
                        trt_size: float = export_results["trt"].stat().st_size / 1e6
                        logger.info(f"✅ TensorRT: {export_results['trt'].name} ({trt_size:.2f} MB)")
                    elif getattr(self.config, "export_trt", False):
                        logger.warning("⚠️  TensorRT экспорт пропущен (возможно, не установлен torch-tensorrt)")

                except Exception as e:
                    logger.error(f"❌ Ошибка экспорта модели {checkpoint.key}: {e}")

            model_results: List[TestResult] = self._test_single_model(
                checkpoint,
                image_pairs,
                precision=getattr(self.config, "precision", "fp32"),
                cache=self.cache,
                config_hash=config_hash,
                profile_dir=profile_dir,
            )
            all_results.extend(model_results)

        self._close_experiment_tracking()

        if not all_results:
            logger.error("Нет результатов для анализа!")
            return pd.DataFrame()

        df: pd.DataFrame = pd.DataFrame([r.to_dict() for r in all_results])
        parsed: pd.DataFrame = df["model_key"].apply(lambda k: pd.Series(extract_model_aug_from_key(k)))
        df["model"] = parsed[0]
        df["augmentation"] = parsed[1]

        self._print_summary_statistics(df)

        if self.config.verbose:
            print("\n🔍 Проверка парсинга в DataFrame:")
            for _, row in df.head(5).iterrows():
                print(f"   {row['model_key']:35} → model='{row['model']}', aug='{row['augmentation']}'")

        if self.config.verbose:
            print("\n🔍 Структура DataFrame:")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Rows: {len(df)}")
            if "precision" in df.columns:
                print(f"   Unique precision values: {df['precision'].unique()}")

            # Проверка размеров групп для агрегации
            if all(col in df.columns for col in ["model", "augmentation", "precision"]):
                group_sizes = df.groupby(["model", "augmentation", "precision"]).size()
                print("\n🔍 Размеры групп для агрегации:")
                print(f"   Всего групп: {len(group_sizes)}")
                print(f"   Распределение размеров: {group_sizes.value_counts().sort_index().to_dict()}")
                if (group_sizes == 1).any():
                    print(f"   ⚠️  {((group_sizes == 1).sum())} групп содержат только 1 запись → std будет 0.0")

        return df

    # ──────────────────────────────────────────────────────────────────────
    def _print_summary_statistics(self, df: pd.DataFrame) -> None:
        """Печать сводной статистики в консоль.

        Args:
            df: DataFrame с результатами тестирования.
        """
        print("\n" + "=" * 80)
        print("СВОДНАЯ СТАТИСТИКА")
        print("=" * 80)

        # Средний mIoU по уровням аугментаций
        if "none" in df["augmentation"].values:
            avg_none: float = df[df["augmentation"] == "none"]["m_iou"].mean()
            print("\n📊 Средний mIoU по уровням аугментаций:")
            print(f"   None:   {avg_none:.4f}")
        else:
            avg_none = 0.0
            print("\n📊 Средний mIoU по уровням аугментаций:")
            print("   None:   N/A")

        if "basic" in df["augmentation"].values:
            avg_basic: float = df[df["augmentation"] == "basic"]["m_iou"].mean()
            gain_basic: float = (avg_basic - avg_none) * 100 if avg_none > 0 else 0
            print(f"   Basic:  {avg_basic:.4f} (прирост: {gain_basic:+.2f}%)")
        else:
            avg_basic = 0.0
            print("   Basic:  N/A")

        if "medium" in df["augmentation"].values:
            avg_medium: float = df[df["augmentation"] == "medium"]["m_iou"].mean()
            gain_medium: float = (avg_medium - avg_none) * 100 if avg_none > 0 else 0
            print(f"   Medium: {avg_medium:.4f} (прирост: {gain_medium:+.2f}%)")
        else:
            print("   Medium: N/A")

        # Лучшая комбинация
        if not df.empty:
            best_idx = df["m_iou"].idxmax()
            best_row: pd.Series = df.loc[best_idx]
            print("\n🏆 Лучшая комбинация:")
            print(f"   Модель: {best_row['model']}")
            print(f"   Аугментации: {best_row['augmentation']}")
            print(f"   mIoU: {best_row['m_iou']:.4f}")

        print("\n" + "=" * 80)

    # ──────────────────────────────────────────────────────────────────────
    def aggregate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Агрегация метрик по группам (model, augmentation, precision).

        Args:
            df: Исходный DataFrame с результатами.

        Returns:
            pd.DataFrame: Агрегированная таблица с mean/std/min/max.

        Note:
            Автоматически фильтрует только числовые колонки.
            Для групп с 1 записью std заполняется 0.0.
        """
        # Перепарсинг если аугментация содержит подозрительные подстроки
        if "augmentation" in df.columns and df["augmentation"].str.contains("smp_|tv_", regex=True, na=False).any():
            logger.warning("⚠️ Подозрительные значения в 'augmentation', перепарсиваем...")
            parsed = df["model_key"].apply(lambda k: pd.Series(extract_model_aug_from_key(k)))
            df["model"] = parsed[0]
            df["augmentation"] = parsed[1]

        # Добавляем precision если нет
        if "precision" not in df.columns:
            logger.warning("⚠️ Колонка 'precision' отсутствует, добавляем 'fp32'")
            df["precision"] = "fp32"

        agg_funcs: List[str] = ["mean", "std", "min", "max"]

        # Fильтруем ТОЛЬКО числовые колонки для агрегации
        all_candidate_cols: List[str] = [
            c for c in df.columns if c in self.config.metrics or c in ["m_iou", "binary_iou", "inference_time"]
        ]

        # Оставляем только числовые (numeric) колонки
        metric_cols: List[str] = [c for c in all_candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

        if getattr(self.config, "compute_boundary_f1", False) and "boundary_f1" in df.columns:
            if "boundary_f1" not in metric_cols and pd.api.types.is_numeric_dtype(df["boundary_f1"]):
                metric_cols.append("boundary_f1")
                logger.info("🔍 Добавлена метрика 'boundary_f1' в агрегацию")

        if getattr(self.config, "per_class_metrics", False):
            # Включаем per-class колонки в агрегацию
            for col in df.columns:
                if col.startswith(("precision_class_", "recall_class_", "iou_class_")):
                    if col not in metric_cols and pd.api.types.is_numeric_dtype(df[col]):
                        metric_cols.append(col)

        if not metric_cols:
            logger.error("❌ Нет числовых колонок для агрегации!")
            logger.info(f"   Доступные колонки: {list(df.columns)}")
            return pd.DataFrame()

        if self.config.verbose:
            logger.info(f"🔍 Агрегация колонок: {metric_cols}")

        grouped = df.groupby(["model", "augmentation", "precision"])[metric_cols]
        aggregated: pd.DataFrame = grouped.agg(agg_funcs)

        # Flatten multi-level columns
        aggregated.columns = [
            "_".join(col).strip() if isinstance(col, tuple) else col for col in aggregated.columns.values
        ]

        # 🔧 Заполняем NaN для std в группах с 1 записью
        std_cols: List[str] = [c for c in aggregated.columns if c.endswith("_std")]
        for col in std_cols:
            counts = df.groupby(["model", "augmentation", "precision"]).size()
            std_fill_map = counts.apply(lambda n: 0.0 if n <= 1 else np.nan).to_dict()
            idx = aggregated.index
            for model, aug, prec in idx:
                if std_fill_map.get((model, aug, prec), np.nan) == 0.0:
                    aggregated.loc[(model, aug, prec), col] = 0.0

        return aggregated.reset_index()

    # ──────────────────────────────────────────────────────────────────────
    def statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Статистический анализ: ANOVA, Tukey HSD, поиск лучших комбинаций.

        Args:
            df: DataFrame с результатами.

        Returns:
            Dict[str, Any]: Словарь со статистикой, результатами тестов и лучшими комбинациями.
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        analysis: Dict[str, Any] = {}

        # ──────────────────────────────────────────────────────────────
        # 1. Сводная статистика по уровням аугментаций
        # ──────────────────────────────────────────────────────────────
        summary: Dict[str, Any] = {}
        for aug in ["none", "basic", "medium"]:
            aug_data: pd.DataFrame = df[df["augmentation"] == aug]
            if not aug_data.empty:
                summary[aug] = {
                    "mean_m_iou": float(aug_data["m_iou"].mean()),
                    "std_m_iou": float(aug_data["m_iou"].std()),
                    "median_m_iou": float(aug_data["m_iou"].median()),
                    "min_m_iou": float(aug_data["m_iou"].min()),
                    "max_m_iou": float(aug_data["m_iou"].max()),
                    "n_samples": int(len(aug_data)),
                }
        analysis["summary_by_augmentation"] = summary

        # ──────────────────────────────────────────────────────────────
        # 2. ANOVA для каждой модели (сравнение аугментаций)
        # ──────────────────────────────────────────────────────────────
        anova_results: Dict[str, Any] = {}
        for model in df["model"].unique():
            model_data: pd.DataFrame = df[df["model"] == model]
            groups = [
                group["m_iou"].dropna().values
                for _, group in model_data.groupby("augmentation")
                if len(group["m_iou"].dropna()) > 1
            ]

            if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    anova_results[model] = {
                        "f_statistic": float(f_stat),
                        "p_value": float(p_value),
                        "significant": bool(p_value < 0.05),
                        "n_groups": len(groups),
                    }
                except Exception:
                    anova_results[model] = {"error": "ANOVA failed"}

        analysis["anova_by_model"] = anova_results

        # ──────────────────────────────────────────────────────────────
        # 3. Пост-хок тесты (Tukey HSD) при значимом ANOVA
        # ──────────────────────────────────────────────────────────────
        posthoc_results: Dict[str, Any] = {}
        try:
            for model, result in anova_results.items():
                if result.get("significant", False):
                    model_data = df[df["model"] == model].copy()
                    if len(model_data) >= 3:
                        tukey = pairwise_tukeyhsd(
                            endog=model_data["m_iou"].values,
                            groups=model_data["augmentation"].values,
                            alpha=0.05,
                        )
                        posthoc_results[model] = {
                            "significant_pairs": [
                                f"{pair[0]} vs {pair[1]}: p={pair[4]:.4f}"
                                for pair in zip(
                                    tukey.groupsunique,
                                    tukey.meandiffs,
                                    tukey.pvalues,
                                    tukey.reject,
                                    range(len(tukey.pvalues)),
                                )
                                if pair[3]  # reject == True
                            ]
                        }
        except ImportError:
            analysis["note"] = "statsmodels not installed — пост-хок тесты пропущены"

        analysis["posthoc_tukey"] = posthoc_results

        # ──────────────────────────────────────────────────────────────
        # 4. Лучшая комбинация (глобально и по моделям)
        # ──────────────────────────────────────────────────────────────
        if not df.empty:
            best_global: pd.Series = df.loc[df["m_iou"].idxmax()]
            analysis["best_global"] = {
                "model": str(best_global["model"]),
                "augmentation": str(best_global["augmentation"]),
                "m_iou": float(best_global["m_iou"]),
                "m_iou_std": float(
                    df[(df["model"] == best_global["model"]) & (df["augmentation"] == best_global["augmentation"])][
                        "m_iou"
                    ].std()
                ),
            }

            best_by_model: Dict[str, Any] = {}
            for model in df["model"].unique():
                model_best = df[df["model"] == model].loc[df[df["model"] == model]["m_iou"].idxmax()]
                best_by_model[model] = {
                    "augmentation": str(model_best["augmentation"]),
                    "m_iou": float(model_best["m_iou"]),
                }
            analysis["best_by_model"] = best_by_model

        return analysis

    # ──────────────────────────────────────────────────────────────────────
    def export_results(
        self,
        df: pd.DataFrame,
        aggregated: pd.DataFrame,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Экспорт результатов в CSV, JSON, Markdown и генерация графиков.

        Args:
            df: Сырой DataFrame.
            aggregated: Агрегированный DataFrame.
            stats: Результаты статистического анализа.

        Returns:
            Dict[str, Path]: Маппинг имён артефактов к путям файлов.
        """
        output: Path = Path(self.config.output_dir)
        exported: Dict[str, Path] = {}

        # ──────────────────────────────────────────────────────────────
        # CSV экспорт
        # ──────────────────────────────────────────────────────────────
        csv_path: Path = output / "detailed_results.csv"
        df.to_csv(csv_path, index=False)
        exported["csv"] = csv_path

        agg_csv: Path = output / "aggregated_metrics.csv"
        aggregated.to_csv(agg_csv, index=False)
        exported["aggregated_csv"] = agg_csv

        # ──────────────────────────────────────────────────────────────
        # JSON со статистикой
        # ──────────────────────────────────────────────────────────────
        if stats:
            json_path: Path = output / "statistical_analysis.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, default=str, ensure_ascii=False)
            exported["stats_json"] = json_path

        # ──────────────────────────────────────────────────────────────
        # Markdown отчёт (расширенный)
        # ──────────────────────────────────────────────────────────────
        md_path: Path = output / "report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Отчёт: Влияние аугментаций на качество сегментации (ADE20K)\n\n")
            f.write(f"**Дата:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Изображений:** {len(df)}\n")
            f.write(f"**Моделей:** {df['model'].nunique()}\n")
            f.write(f"**Аугментаций:** {df['augmentation'].nunique()}\n\n")

            # Сводная таблица
            f.write("## 📊 Сводная таблица метрик (mIoU ± std)\n\n")
            pivot: pd.DataFrame = df.pivot_table(
                values="m_iou",
                index="model",
                columns="augmentation",
                aggfunc=["mean", "std"],
            )
            f.write(pivot.round(4).to_markdown() + "\n\n")

            # Лучшая комбинация
            f.write("## 🏆 Лучшая комбинация по mIoU\n\n")
            if stats and "best_global" in stats:
                best = stats["best_global"]
                f.write(f"- **Модель:** `{best['model']}`\n")
                f.write(f"- **Аугментации:** `{best['augmentation']}`\n")
                f.write(f"- **mIoU:** `{best['m_iou']:.4f} ± {best['m_iou_std']:.4f}`\n\n")

            # Статистика по аугментациям
            if stats and "summary_by_augmentation" in stats:
                f.write("## 📈 Статистика по уровням аугментаций\n\n")
                f.write("| Уровень | Mean mIoU | Std | Median | Min | Max | N |\n")
                f.write("|---------|-----------|-----|--------|-----|-----|---|\n")
                for aug, data in stats["summary_by_augmentation"].items():
                    f.write(
                        f"| {aug} | {data['mean_m_iou']:.4f} | {data['std_m_iou']:.4f} | "
                        f"{data['median_m_iou']:.4f} | {data['min_m_iou']:.4f} | "
                        f"{data['max_m_iou']:.4f} | {data['n_samples']} |\n"
                    )
                f.write("\n")

            # ANOVA результаты
            if stats and "anova_by_model" in stats:
                f.write("## 🔬 Статистическая значимость (ANOVA)\n\n")
                for model, result in stats["anova_by_model"].items():
                    if "error" not in result:
                        sig = "✅" if result.get("significant") else "❌"
                        f.write(f"- `{model}`: p={result['p_value']:.4f} {sig}\n")
                f.write("\n")

            # Прирост относительно baseline
            f.write("## 📈 Прирост относительно baseline (none)\n\n")
            pivot_gain: pd.DataFrame = df.pivot_table(
                values="m_iou", index="model", columns="augmentation", aggfunc="mean"
            )
            if "none" in pivot_gain.columns:
                f.write("| Модель | Basic Δ% | Medium Δ% |\n")
                f.write("|--------|----------|-----------|\n")
                for model in pivot_gain.index:
                    none_val = pivot_gain.loc[model, "none"]
                    basic_gain = (
                        ((pivot_gain.loc[model, "basic"] - none_val) / none_val * 100)
                        if "basic" in pivot_gain.columns
                        else 0
                    )
                    medium_gain = (
                        ((pivot_gain.loc[model, "medium"] - none_val) / none_val * 100)
                        if "medium" in pivot_gain.columns
                        else 0
                    )
                    f.write(f"| {model} | {basic_gain:+.2f}% | {medium_gain:+.2f}% |\n")

            if "boundary_f1" in df.columns:
                f.write("## 🎯 Boundary F1 (точность границ)\n\n")
                f.write(f"- Средний Boundary F1: `{df['boundary_f1'].mean():.4f} ± {df['boundary_f1'].std():.4f}`\n\n")

        exported["report_md"] = md_path

        # ──────────────────────────────────────────────────────────────
        # Визуализации (оверлеи)
        # ──────────────────────────────────────────────────────────────
        if self.overlays and self.config.save_overlays:
            viz_dir: Path = output / "overlays"
            viz_dir.mkdir(exist_ok=True)
            for key, overlay in self.overlays.items():
                try:
                    overlay_pil: Image.Image = ensure_pil_compatible(overlay)
                    overlay_pil.save(viz_dir / f"{key}.png")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось сохранить оверлей {key}: {e}")
            exported["overlays_dir"] = viz_dir

        # ──────────────────────────────────────────────────────────────
        # Графики (через plot_detailed_results)
        # ──────────────────────────────────────────────────────────────
        plots: Dict[str, Path] = self.plot_detailed_results(df)
        exported.update({f"plot_{k}": v for k, v in plots.items()})

        # ──────────────────────────────────────────────────────────────
        # Сетка сравнения визуализаций
        # ──────────────────────────────────────────────────────────────
        if self.overlays:
            unique_models: List[str] = list(set(extract_model_aug_from_key(k)[0] for k in self.overlays.keys()))
            save_augmentation_comparison_grid(
                overlay_images=self.overlays,
                output_dir=output / "overlays",
                model_names=unique_models,
            )
            exported["comparison_grid"] = output / "overlays" / "full_comparison_grid.png"

        return exported

    # ──────────────────────────────────────────────────────────────────────
    def plot_results(self, df: pd.DataFrame, aggregated: pd.DataFrame) -> Dict[str, Path]:
        """Построение базовых графиков: mIoU bar, heatmap прироста, boxplot, время инференса.

        Args:
            df: DataFrame с результатами.
            aggregated: Агрегированный DataFrame (опционально).

        Returns:
            Dict[str, Path]: Пути к сохранённым PNG.
        """
        output = self.config.output_dir
        plots: Dict[str, Path] = {}

        plt.figure(figsize=(14, 6))
        sns.barplot(
            data=df,
            x="model",
            y="m_iou",
            hue="augmentation",
            palette="viridis",
            errorbar="sd",
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("mIoU")
        plt.title("Влияние аугментаций на mIoU (ADE20K)")
        plt.tight_layout()
        plot_path: Path = output / Path("miou_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["miou_bar"] = plot_path

        pivot: pd.DataFrame = df.pivot_table(values="m_iou", index="model", columns="augmentation", aggfunc="mean")
        if "none" in pivot.columns:
            gain: pd.DataFrame = pivot.copy()
            for col in ["basic", "medium"]:
                if col in gain.columns:
                    gain[col] = (gain[col] - gain["none"]) / gain["none"].replace(0, 1e-8) * 100

            plt.figure(figsize=(10, 6))
            sns.heatmap(gain, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
            plt.title("Прирост mIoU относительно baseline (%)")
            plt.ylabel("Модель")
            plt.xlabel("Аугментация")
            plt.tight_layout()
            heatmap_path: Path = output / Path("gain_heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
            plt.close()
            plots["gain_heatmap"] = heatmap_path

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="augmentation", y="m_iou", hue="model", palette="Set2")
        plt.ylabel("mIoU per image")
        plt.title("Распределение mIoU по изображениям")
        plt.tight_layout()
        boxplot_path: Path = output / Path("miou_distribution.png")
        plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["boxplot"] = boxplot_path

        plt.figure(figsize=(12, 5))
        sns.barplot(
            data=df,
            x="model",
            y="inference_time",
            hue="augmentation",
            palette="coolwarm",
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Время (сек)")
        plt.title("Время инференса на изображение")
        plt.tight_layout()
        time_path: Path = output / Path("inference_time.png")
        plt.savefig(time_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["time"] = time_path

        return plots

    # ──────────────────────────────────────────────────────────────────────
    def plot_detailed_results(self, df: pd.DataFrame) -> Dict[str, Path]:
        """Построение детальных графиков: 2x3 сетка метрик, swarmplot, приросты.

        Args:
            df: DataFrame с результатами.

        Returns:
            Dict[str, Path]: Пути к сгенерированным графикам.
        """
        output: Path = Path(self.config.output_dir) / "plots"
        output.mkdir(parents=True, exist_ok=True)
        plots: Dict[str, Path] = {}

        # ──────────────────────────────────────────────────────────────
        # График 1: Все метрики по моделям и аугментациям (2×3 сетка)
        # ──────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        metrics_to_plot: List[str] = [
            "m_iou",
            "binary_iou",
            "dice",
            "f1_score",
            "precision",
            "recall",
        ]
        metric_names: Dict[str, str] = {
            "m_iou": "mIoU",
            "binary_iou": "Binary IoU",
            "dice": "Dice",
            "f1_score": "F1-Score",
            "precision": "Precision",
            "recall": "Recall",
        }

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            if metric not in df.columns:
                ax.text(
                    0.5,
                    0.5,
                    f"'{metric}'\nN/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
                ax.set_title(metric_names.get(metric, metric), fontsize=11)
                ax.axis("off")
                continue

            if not pd.api.types.is_numeric_dtype(df[metric]):
                ax.text(
                    0.5,
                    0.5,
                    f"'{metric}'\n(non-numeric)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
                ax.set_title(metric_names.get(metric, metric), fontsize=11)
                ax.axis("off")
                continue

            # Агрегация: mean по (модель × аугментация)
            plot_data: pd.DataFrame = df.groupby(["model", "augmentation"])[metric].mean().unstack()
            plot_data.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black")
            ax.set_title(f"{metric_names[metric]} по моделям и аугментациям", fontsize=11)
            ax.set_ylabel("Score")
            ax.set_xlabel("Модель")
            ax.legend(title="Аугментации", loc="lower right", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=45, labelsize=8)

        plt.tight_layout()
        plot_path: Path = output / "all_metrics_grid.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["all_metrics_grid"] = plot_path

        # ──────────────────────────────────────────────────────────────
        # График 2: mIoU bar chart (основной)
        # ──────────────────────────────────────────────────────────────
        plt.figure(figsize=(14, 6))
        sns.barplot(
            data=df,
            x="model",
            y="m_iou",
            hue="augmentation",
            palette="viridis",
            errorbar="sd",
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("mIoU")
        plt.title("Влияние аугментаций на качество сегментации (mIoU ± std)")
        plt.tight_layout()
        plot_path = output / "miou_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["miou_bar"] = plot_path

        # ──────────────────────────────────────────────────────────────
        # График 3: Heatmap прироста относительно baseline
        # ──────────────────────────────────────────────────────────────
        pivot: pd.DataFrame = df.pivot_table(values="m_iou", index="model", columns="augmentation", aggfunc="mean")
        if "none" in pivot.columns:
            gain: pd.DataFrame = pivot.copy()
            for col in ["basic", "medium"]:
                if col in gain.columns:
                    gain[col] = (gain[col] - gain["none"]) / gain["none"].replace(0, 1e-8) * 100

            plt.figure(figsize=(10, 8))
            sns.heatmap(gain, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
            plt.title("Прирост mIoU относительно baseline (%)")
            plt.ylabel("Модель")
            plt.xlabel("Аугментация")
            plt.tight_layout()
            plot_path = output / "gain_heatmap.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            plots["gain_heatmap"] = plot_path

        # ──────────────────────────────────────────────────────────────
        # График 4: Распределение mIoU по изображениям (boxplot)
        # ──────────────────────────────────────────────────────────────
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df,
            x="augmentation",
            y="m_iou",
            hue="model",
            palette="Set2",
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": "8",
                "markeredgecolor": "black",
            },
        )
        plt.ylabel("mIoU per image")
        plt.title("Распределение mIoU по изображениям")
        plt.legend(title="Модель", loc="best", fontsize=8)
        plt.tight_layout()
        plot_path = output / "miou_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["boxplot"] = plot_path

        plt.figure(figsize=(12, 6))
        sns.swarmplot(
            data=df,
            x="augmentation",
            y="m_iou",
            hue="model",
            palette="Set2",
            size=4,
            alpha=0.6,
        )
        plt.ylabel("mIoU per image")
        plt.title("Распределение mIoU по изображениям (swarmplot)")
        plt.legend(title="Модель", loc="best", fontsize=8)
        plt.tight_layout()
        plot_path = output / "miou_distribution_swarm.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["swarmplot"] = plot_path

        # ──────────────────────────────────────────────────────────────
        # График 5: Время инференса
        # ──────────────────────────────────────────────────────────────
        plt.figure(figsize=(12, 5))
        sns.barplot(
            data=df,
            x="model",
            y="inference_time",
            hue="augmentation",
            palette="coolwarm",
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Время (сек)")
        plt.title("Время инференса на изображение")
        plt.tight_layout()
        plot_path = output / "inference_time.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["time"] = plot_path

        # ──────────────────────────────────────────────────────────────
        # График 6: Прирост по уровням аугментаций (два графика)
        # ──────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for model in df["model"].unique():
            model_data = df[df["model"] == model]

            none_data = model_data[model_data["augmentation"] == "none"]["m_iou"]
            basic_data = model_data[model_data["augmentation"] == "basic"]["m_iou"]
            medium_data = model_data[model_data["augmentation"] == "medium"]["m_iou"]

            if not none_data.empty and not basic_data.empty:
                none_val = none_data.mean()
                basic_val = basic_data.mean()
                if not np.isnan(none_val) and not np.isnan(basic_val) and none_val != 0:
                    basic_gain = (basic_val - none_val) / none_val * 100
                    axes[0].bar(
                        f"{model}",
                        basic_gain,
                        alpha=0.7,
                        color="steelblue",
                        label=model,
                    )

            if not none_data.empty and not medium_data.empty:
                none_val = none_data.mean()
                medium_val = medium_data.mean()
                if not np.isnan(none_val) and not np.isnan(medium_val) and none_val != 0:
                    medium_gain = (medium_val - none_val) / none_val * 100
                    axes[1].bar(f"{model}", medium_gain, alpha=0.7, color="orange", label=model)

        axes[0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[0].set_title("Прирост mIoU: Basic vs None (%)", fontsize=11)
        axes[0].set_ylabel("Прирост mIoU (%)")
        axes[0].tick_params(axis="x", rotation=45, labelsize=8)
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[1].set_title("Прирост mIoU: Medium vs None (%)", fontsize=11)
        axes[1].set_ylabel("Прирост mIoU (%)")
        axes[1].tick_params(axis="x", rotation=45, labelsize=8)
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plot_path = output / "augmentation_gain.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["gain_comparison"] = plot_path

        return plots


# ──────────────────────────────────────────────────────────────────────
def ensure_pil_compatible(arr: Union[np.ndarray, Image.Image]) -> Image.Image:
    """Преобразование numpy массива или PIL.Image в совместимый RGB PIL.Image.

    Args:
        arr: Входные данные (2D/3D массив или PIL.Image).

    Returns:
        Image.Image: Изображение в формате RGB.

    Note:
        Автоматически нормализует диапазон к [0, 255] и приводит типы.
    """
    if isinstance(arr, Image.Image):
        return arr.convert("RGB") if arr.mode != "RGB" else arr

    arr_np: np.ndarray = np.asarray(arr).copy()

    # Нормализация к [0, 255]
    if arr_np.min() < 0 or arr_np.max() > 255:
        arr_np = (arr_np - arr_np.min()) / (arr_np.max() - arr_np.min() + 1e-8) * 255

    # Конвертация типов
    if arr_np.dtype != np.uint8:
        arr_np = np.clip(arr_np, 0, 255).astype(np.uint8)

    # Обеспечение 3 каналов
    if arr_np.ndim == 2:
        arr_np = np.stack([arr_np] * 3, axis=-1)
    elif arr_np.ndim == 3 and arr_np.shape[2] == 1:
        arr_np = np.repeat(arr_np, 3, axis=2)
    elif arr_np.ndim == 3 and arr_np.shape[2] > 3:
        arr_np = arr_np[:, :, :3]

    return Image.fromarray(arr_np)


# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """Точка входа CLI. Парсит аргументы, запускает тестирование и сохраняет результаты.

    Note:
        Поддерживает все флаги, описанные в epilog argparse.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Анализ влияния аугментаций на сегментацию",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
╔════════════════════════════════════════════════════════════════╗
║                    ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ                       ║
╠════════════════════════════════════════════════════════════════╣
║  🔹 Базовый запуск:                                            ║
║     python BatchNeuralTester.py --dataset ./data/ADE20K \\      ║
║                               --subset 50 \\                     ║
║                               --output ./results               ║
║                                                                ║
║  🔹 С кэшированием и возобновлением:                           ║
║     python BatchNeuralTester.py --cache --resume \\             ║
║                               --output ./results               ║
║                                                                ║
║  🔹 Профилирование инференса:                                  ║
║     python BatchNeuralTester.py --profile \\                    ║
║                               --profile-output ./profiling     ║
║                                                                ║
║  🔹 Экспорт в ONNX:                                            ║
║     python BatchNeuralTester.py --export-onnx --opset 18       ║
║                                                                ║
║  🔹 Экспорт в ONNX + TensorRT:                                 ║
║     python BatchNeuralTester.py --export-onnx --export-trt \\  ║
║                               --trt-precision fp16             ║
║                                                                ║
║  🔹 Многоклассовые метрики + boundary F1:                      ║
║     python BatchNeuralTester.py --compute-boundary-f1 \\       ║
║                               --per-class-metrics              ║
║                                                                ║
║  🔹 Тестирование из кастомной папки моделей:                   ║
║     python BatchNeuralTester.py --models ./my_checkpoints \\   ║
║                               --subset 5                       ║
║                                                                ║
║  🔹 Воспроизводимый эксперимент:                               ║
║     python BatchNeuralTester.py --seed 42 --output ./exp_v1    ║
║     python BatchNeuralTester.py --seed 42 --output ./exp_v1_r  ║
║     # ✅ Одинаковые результаты благодаря фиксированному seed   ║
║                                                                ║
║  🔹 Запуск на CPU (для отладки):                               ║
║     python BatchNeuralTester.py --device cpu \\                ║
║                               --precision fp32 \\              ║
║                               --subset 1                       ║
║                                                                ║
║  🔹 Интеграция с MLflow / Weights & Biases:                    ║
║     python BatchNeuralTester.py --use-mlflow                   ║
║     python BatchNeuralTester.py --use-wandb  # требует wandb login ║
║                                                                ║
║  🔹 Визуализация с легендами классов:                          ║
║     python BatchNeuralTester.py --class-aware-overlays \\      ║
║                               --overlay-alpha 0.6 \\           ║
║                               --save-viz                       ║
╚════════════════════════════════════════════════════════════════╝
""",
        #         epilog="""
        # ### Примеры использования флагов
        # Examples:
        #   # Базовый запуск
        #   python analyze_batch.py --dataset ./data/ADE20K --subset 50
        #   # С кэшированием и возобновлением
        #   python analyze_batch.py --cache --resume --output ./results
        #   # Профилирование инференса
        #   python analyze_batch.py --profile --profile-output ./profiling
        #   # Экспорт в ONNX
        #   python analyze_batch.py --export-onnx --opset 17
        #   # Экспорт в ONNX + TensorRT (fp16)
        #   python analyze_batch.py --export-onnx --export-trt --trt-precision fp16
        #   # Многоклассовые метрики + boundary F1
        #   python analyze_batch.py --compute-boundary-f1 --verbose
        #   # Тестирование конкретной модели из кастомной папки
        #   python BatchNeuralTester.py --models ./my_checkpoints --subset 5
        #   # Воспроизводимый эксперимент
        #   python BatchNeuralTester.py --seed 42 --output ./exp_v1
        #   python BatchNeuralTester.py --seed 42 --output ./exp_v1_retry  # те же данные
        #   # Запуск на CPU (например, для отладки)
        #   python BatchNeuralTester.py --device cpu --precision fp32 --subset 1
        #         """
    )

    # ──────────────────────────────────────────────────────────────
    # Основные параметры
    # ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/ADE20K",
        help="Путь к датасету (локально или через HF)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=50,
        help="Размер подмножества для теста (0 = весь датасет)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/augmentation_analysis",
        help="Директория для результатов",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="./models",
        help="Директория с чекпоинтами моделей (по умолчанию: ./models)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", default=True, help="Подробный вывод")

    # ──────────────────────────────────────────────────────────────
    # Точность и устройство
    # ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Точность вычислений (default: fp32)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Устройство для инференса",
    )

    # ──────────────────────────────────────────────────────────────
    # Кэш, resume, экспорт, профилирование
    # ──────────────────────────────────────────────────────────────
    # Кэширование
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Включить кэширование предсказаний (ускоряет повторные запуски)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache/predictions",
        help="Директория для кэша предсказаний",
    )
    parser.add_argument("--cache-max-gb", type=float, default=10.0, help="Максимальный размер кэша в ГБ")
    parser.add_argument("--clear-cache", action="store_true", help="Очистить кэш перед запуском")

    # Возобновление
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Возобновить прерванный запуск (пропускать выполненные задачи)",
    )

    # Экспорт моделей
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Экспортировать модели в ONNX формат после тестирования",
    )
    parser.add_argument(
        "--export-trt",
        action="store_true",
        help="Экспортировать в TensorRT (требует --export-onnx)",
    )
    parser.add_argument(
        "--trt-precision",
        choices=["fp32", "fp16"],
        default="fp16",
        help="Точность для TensorRT (default: fp16)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument(
        "--dynamic-shapes",
        action="store_true",
        help="Использовать динамические размеры при экспорте в ONNX",
    )

    # Профилирование
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Включить профилирование инференса (torch.profiler)",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default="./profiling",
        help="Директория для результатов профилирования",
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=10,
        help="Количество итераций прогрева перед профилированием",
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=50,
        help="Количество итераций для профилирования",
    )

    # Многоклассовые метрики
    parser.add_argument(
        "--compute-boundary-f1",
        action="store_true",
        help="Вычислять Boundary F1 score (медленно, для детального анализа)",
    )
    parser.add_argument(
        "--per-class-metrics",
        action="store_true",
        help="Сохранять per-class метрики в отчёт",
    )

    # Трекеры экспериментов
    parser.add_argument("--use-mlflow", action="store_true", help="Логировать метрики в MLflow")
    parser.add_argument("--use-wandb", action="store_true", help="Логировать метрики в Weights & Biases")

    # Визуализация
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Сохранять визуализации сегментации (оверлеи)",
    )
    parser.add_argument(
        "--class-aware-overlays",
        action="store_true",
        help="Использовать class-aware оверлеи с легендой",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="Прозрачность оверлеев (0.0–1.0)",
    )

    args = parser.parse_args()

    config: TestConfig = TestConfig(
        dataset_path=args.dataset,
        output_dir=args.output,
        subset_size=args.subset,
        random_seed=args.seed,
        verbose=args.verbose,
        precision=args.precision,
        device=args.device,
        cache=args.cache,
        cache_dir=args.cache_dir,
        cache_max_gb=args.cache_max_gb,
        clear_cache=args.clear_cache,
        resume=args.resume,
        export_onnx=args.export_onnx,
        export_trt=args.export_trt,
        models_dir=args.models,
        trt_precision=args.trt_precision,
        opset=args.opset,
        dynamic_shapes=args.dynamic_shapes,
        profile=args.profile,
        profile_output=args.profile_output,
        profile_warmup=args.profile_warmup,
        profile_runs=args.profile_runs,
        compute_boundary_f1=args.compute_boundary_f1,
        per_class_metrics=args.per_class_metrics,
        use_mlflow=args.use_mlflow,
        use_wandb=args.use_wandb,
        class_aware_overlays=args.class_aware_overlays,
        overlay_alpha=args.overlay_alpha,
        save_viz=args.save_viz,
    )

    tester: BatchNeuralTester = BatchNeuralTester(config)

    if config.verbose:
        print("\n🔍 Проверка парсинга ключей:")
        sample_keys = [
            "fcn_tv_basic_ADE_val_00000001",
            "unet_smp_none_20260413_085847",
            "deeplab_tv_medium_overlay",
            "segnet_none_test",
        ]
        for k in sample_keys:
            model, aug = extract_model_aug_from_key(k)
            print(f"   {k:40} → model='{model}', aug='{aug}'")

    print(f"\n🔍 CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    df: pd.DataFrame = tester.run()
    if df.empty:
        print("❌ Тестирование не дало результатов")
        return

    aggregated: pd.DataFrame = tester.aggregate_metrics(df)
    print(aggregated[["model", "augmentation", "m_iou_mean", "m_iou_std"]].head(10))
    assert not aggregated["m_iou_std"].isna().any(), "Есть NaN в std!"

    stats: Dict[str, Any] = tester.statistical_analysis(df)

    if tester.overlays:
        print(f"\n📊 Генерация сравнительных визуализаций ({len(tester.overlays)} оверлеев)...")

        unique_models: List[str] = list(set(extract_model_aug_from_key(k)[0] for k in tester.overlays.keys()))

        # Сохранение отдельных сравнений по моделям
        save_model_augmentation_comparisons(
            overlay_images=tester.overlays,
            output_dir=Path(args.output) / "overlays",
            models=unique_models,
        )

        # Сохранение общей сетки сравнения
        save_augmentation_comparison_grid(
            overlay_images=tester.overlays,
            output_dir=Path(args.output) / "overlays",
            model_names=unique_models,
        )

    exported: Dict[str, Path] = tester.export_results(df, aggregated, stats)
    print("\n💾 Результаты сохранены:")
    for name, path in exported.items():
        print(f"   • {name}: {path}")

    plots: Dict[str, Path] = tester.plot_results(df, aggregated)
    print("\n📊 Графики сохранены:")
    for name, path in plots.items():
        print(f"   • {name}: {path}")

    print("\n✅ Анализ завершён!")
    print(f"   Всего результатов: {len(df)}")
    print(f"   Моделей: {df['model'].nunique()}")
    print(f"   Аугментаций: {df['augmentation'].nunique()}")

    best: pd.Series = df.loc[df["m_iou"].idxmax()]
    print(f"\n🏆 Лучшая комбинация: {best['model']}_{best['augmentation']}")
    print(f"   mIoU: {best['m_iou']:.4f}")


if __name__ == "__main__":
    main()
