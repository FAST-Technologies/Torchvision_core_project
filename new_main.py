# main.py

"""Основная точка входа фреймворка сегментации.

Выполняет последовательное тестирование классических и нейросетевых методов:
1. Инициализация окружения и автоматический подбор точности/компиляции.
2. Загрузка тестовых изображений (с/без Ground Truth).
3. Регистрация методов сегментации (OpenCV, scikit-learn, PyTorch v1/v2).
4. Опциональный запуск исследовательских блоков:
    - Сравнение бэкендов (PyTorch / ONNX / TensorRT)
    - Профилирование с детекцией CPU↔GPU трансферов
    - Бенчмарк точностей (fp32 vs fp16 vs bf16)
    - Массовое тестирование на ADE20K
    - CPU vs CUDA сравнение
5. Генерация отчётов: таблицы, графики, CSV, HTML.

Args:
    use_optimizations: Если `True`, включает `torch.compile`, AMP и квантование.
                        Если `False`, запускает legacy-пайплайн без оптимизаций.

Returns:
    Tuple[Optional[SegmentationTester], Optional[BenchmarkResult], Optional[SegmentationComparator]]:
    - `tester`: Экземпляр тестера с зарегистрированными методами.
    - `results`: DataFrame с результатами массового тестирования.
    - `comparator`: Экземпляр компаратора для матричных сравнений.

Note:
    - Управляющее поведение задаётся в `configs/main_config.yaml`.
    - При ошибках в отдельных методах выполнение продолжается (graceful degradation).
    - Для прерывания длительных бенчмарков используется `Ctrl+C`.

Example:
    ```python
    if __name__ == "__main__":
        # Запуск с оптимизациями
        tester, results, _ = main(use_optimizations=True)
        if results is not None:
            print(f"Обработано {len(results)} методов")
            print(results.sort_values("IoU", ascending=False).head(10))
    ```
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ И TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций

import os
import sys
import gc
import glob
import json
import warnings
import traceback
import time
import yaml
from io import BytesIO
from pathlib import Path
from typing import (
    Dict,
    Set,
    List,
    Tuple,
    Optional,
    Any,
    Union,
    Literal,
    cast,
)
from matplotlib import colormaps
from tqdm import tqdm

# import re

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import requests
from PIL import Image
from huggingface_hub import hf_hub_download
from tabulate import tabulate

# Локальные импорты
from segmenters.NeuralSegmenter import NeuralSegmenter
from segmenters.OpenCVSegmenter import OpenCVSegmenter
from segmenters.SklearnSegmenter import SklearnSegmenter
from segmenters.TorchSegmenter import TorchSegmenter
from segmenters.NewTorchSegmenter import TorchSegmenter2
from segmenters.ModelTrainer import ModelTrainer, TrainingConfig, TrainingResult
from segmenters.NeuralModelFactory import NeuralModelFactory
from segmenters.BackendSegmenters import ONNXSegmenter, TRTSegmenter
from utils.backend_exporter_new import (load_trt_engine,)
from testing.SegmentationTester import SegmentationTester
from testing.SegmentationComparator import SegmentationComparator
from testing.SegmentationBenchmark import SegmentationBenchmark, export_comparison_table
from testing.TorchImplementationValidator import TorchImplementationValidator
from testing.BatchClassicTester import BatchClassicTester
from testing.CpuCudaBenchmark import CpuCudaBenchmark
from metrics.SegmentationMetrics import SegmentationMetrics, MetricsDict
from utils.warmup import SegmentationWarmUp
from utils.threshold_warmup import ThresholdWarmUp
from utils.strategies import _create_overlay_standalone, segment_image_unified
from utils.batch_exporter import export_all_classical_methods
from utils.backend_exporter_new import TRT_PRESETS, TRT_PRESET_PRODUCTION
from utils.backend_exporter_new import PrecisionType, create_onnx_trt_ep_segmenter

import logging

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler: logging.StreamHandler = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES И КОНСТАНТЫ
# ──────────────────────────────────────────────────────────────────────
# Типы для изображений
ImageArray = npt.NDArray[np.uint8]
"""Тип для входного изображения: (H, W) или (H, W, 3), dtype=uint8."""

MaskArray = npt.NDArray[np.uint8]
"""Тип для бинарной маски: (H, W), dtype=uint8, значения {0, 255}."""

SegmenterDict = Dict[
    str,
    Union[
        OpenCVSegmenter,
        SklearnSegmenter,
        TorchSegmenter,
        TorchSegmenter2,
        NeuralSegmenter,
    ],
]
"""Словарь сегментеров: {имя_метода: экземпляр_сегментера}, dtype=Dict."""

TestImageEntry = Tuple[str, Image.Image, Optional[MaskArray]]
"""Элемент тестового изображения: (путь, PIL.Image, GT-маска или None), dtype=Tuple[str, Image.Image, Optional[MaskArray]]."""

TestImagesDict = Dict[str, TestImageEntry]
"""Словарь тестовых изображений: {имя: (путь, PIL.Image, GT-маска)}, dtype=Dict[str, TestImageEntry]."""

type BenchmarkResult = pd.DataFrame
"""Результат бенчмарка: DataFrame с метриками и временем выполнения, dtype=pd.DataFrame."""

# Константы конфигурации
ADE20K_REPO_ID: str = "hf-internal-testing/fixtures_ade20k"
"""ID репозитория с тестовыми данными ADE20K на HuggingFace, dtype=str."""

NUM_CLASSES_ADE20K: int = 150
"""Количество классов в датасете ADE20K, dtype=int."""

DEFAULT_IMAGE_SIZE: Tuple[int, int] = (512, 512)
"""Размер по умолчанию для ресайза изображений, dtype=Tuple[int, int]."""

TARGET_METHODS_FOR_RESEARCH: List[str] = [
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
    # "sobel_edge",
    # "canny_edge",
    # "prewitt_edge",
    # "scharr_edge",
    # "laplacian_edge",
    # "roberts_cross_edge",
    # "log_edge",
    # "dog_edge",
    # "marr_hildreth_edge",
    # "gradient_magnitude_direction",
    # "phase_congruency_edge",
]
"""Методы для исследования, dtype=List[str]."""

AVAILABLE_PRECISIONS: List[PrecisionType] = ["fp32", "fp16", "bf16"]
"""Возможные точности, dtype=List[str]."""

# Глобальные настройки окружения
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")

num_classes: int = 150

DEFAULT_BENCHMARK_CONFIG: Dict[str, Any] = {
    "enable_trt_ep_benchmark": True,
    "trt_ep_preset": "production",
    "trt_ep_cache_path": "./cache/trt_ep",
}

DEFAULT_TRT_EP_PRECISION: PrecisionType = "fp32"


# ──────────────────────────────────────────────────────────────────────
def _load_config(config_path: str = "configs/main_config.yaml") -> Dict[str, Any]:
    """Загружает конфигурацию фреймворка из YAML-файла.

    Выполняет:
    1. **Чтение файла**: Открытие конфигурационного файла в кодировке UTF-8.
    2. **Парсинг**: Десериализация YAML в словарь через `yaml.safe_load()`.
    3. **Валидация**: Возврат словаря с настройками тестов и окружения.

    Алгоритм:
    ```
    1. Открыть config_path в режиме чтения с encoding='utf-8'.
    2. Вызвать yaml.safe_load(f) → Dict[str, Any].
    3. Вернуть результат.
    ```

    Args:
        config_path: Путь к YAML-файлу конфигурации (по умолчанию `"configs/main_config.yaml"`).

    Returns:
        Dict[str, Any]: Словарь с параметрами:
                        - `test_settings`: флаги `test_classic_logic`, `test_neural_logic`.
                        - `paths`, `devices`, `optimization` и другие секции.

    Note:
        - Используется `safe_load` для защиты от выполнения произвольного кода.
        - Файл должен существовать и быть валидным YAML, иначе возникнет исключение.
        - Все значения сохраняются в исходных типах (bool, int, str, list, dict).

    Example:
        ```python
        config = _load_config("configs/main_config.yaml")
        if config["test_settings"]["test_classic_logic"]:
            print("Классические методы будут протестированы")
        ```
    """
    with open(config_path, "r", encoding="utf-8") as f:
        result: Dict[str, Any] = yaml.safe_load(f)
        return result


# ──────────────────────────────────────────────────────────────────────
# УТИЛИТЫ ДЛЯ ОПРЕДЕЛЕНИЯ ОПТИМАЛЬНЫХ ПАРАМЕТРОВ
# ──────────────────────────────────────────────────────────────────────
def get_optimal_precision(device: torch.device) -> str:
    """Автоматический выбор оптимальной точности вычислений для устройства.

    Выполняет:
    1. **Проверка CUDA**: Если устройство не CUDA → возвращается `fp32`.
    2. **Определение архитектуры**: Чтение `compute capability` GPU.
    3. **Выбор точности**:
       - Ampere+ (≥8.0) → `bf16` (лучший баланс скорости/стабильности).
       - Pascal+ (≥6.0) → `fp16` (хорошая скорость, возможна нестабильность).
       - Старые GPU / CPU → `fp32` (максимальная совместимость).

    Алгоритм:
    ```
    1. Если device.type != "cuda" → вернуть "fp32".
    2. Получить props = torch.cuda.get_device_properties(device.index).
    3. Если props.major >= 8 → вернуть "bf16".
    4. Elif props.major >= 6 → вернуть "fp16".
    5. Else → вернуть "fp32".
    ```

    Args:
        device: Экземпляр `torch.device` для анализа возможностей.

    Returns:
        str: Оптимальная точность: `"bf16"`, `"fp16"` или `"fp32"`.

    Note:
        - `bf16` требует GPU архитектуры Ampere (RTX 30xx, A100) или новее.
        - `fp16` может приводить к числовой нестабильности в некоторых алгоритмах.
        - На CPU всегда используется `fp32` (или `int8` при квантовании).

    Example:
        ```python
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        precision = get_optimal_precision(device)
        print(f"Используемая точность: {precision}")  # bf16 / fp16 / fp32
        ```
    """
    if device.type != "cuda":
        precision: str = "fp32"

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device.index or 0)
        if props.major >= 8:  # Ampere+
            precision = "bf16"  # Лучший баланс скорости и стабильности
        elif props.major >= 6:  # Pascal+
            precision = "fp16"  # Хорошая скорость, но следите за стабильностью
        else:
            precision = "fp32"  # Старые GPU
    else:
        precision = "fp32"  # На CPU только fp32 или int8
    return precision


# ──────────────────────────────────────────────────────────────────────
def get_compile_config(method_name: str, device: torch.device) -> Dict[str, Any]:
    """Возвращает конфигурацию `torch.compile` для указанного метода.

    Выполняет:
    1. **Классификация метода**: Проверка, входит ли метод в список "хорошо компилируемых".
    2. **Подбор режима**: Выбор `compile_mode` в зависимости от устройства.
    3. **Формирование конфига**: Возврат словаря с флагами для `TorchSegmenter2`.

    Алгоритм:
    ```
    1. Определить well_compiled = {глобальные пороги, основные edge-детекторы}.
    2. Если method_name in well_compiled и device.type == "cuda":
       → вернуть {use_compile: True, mode: "reduce-overhead", fullgraph: True, dynamic: False}
    3. Elif method_name in well_compiled (CPU):
       → вернуть {use_compile: True, mode: "default", fullgraph: False, dynamic: True}
    4. Else → вернуть {use_compile: False}.
    ```

    Args:
        method_name: Имя метода сегментации (например, `"otsu_thresholding"`).
        device: Устройство выполнения (`cuda`/`cpu`).

    Returns:
        Dict[str, Any]: Конфигурация компиляции:
                        - `use_compile`: включать ли `torch.compile`.
                        - `compile_mode`: `"reduce-overhead"`, `"max-autotune"` или `"default"`.
                        - `compile_fullgraph`, `compile_dynamic`: флаги графа.

    Note:
        - Методы с динамическим контролем потока плохо компилируются в `fullgraph` режиме.
        - На CPU `dynamic=True` помогает избежать рекомпиляции при разных размерах входа.
        - Для неизвестных методов компиляция отключается во избежание ошибок.

    Example:
        ```python
        cfg = get_compile_config("sobel_edge", torch.device("cuda"))
        print(cfg)
        # {'use_compile': True, 'compile_mode': 'reduce-overhead', ...}
        ```
    """
    # Методы, которые хорошо компилируются
    well_compiled = {
        "global_thresholding",
        "otsu_thresholding",
        "adaptive_thresholding",
        "sobel_edge",
        "prewitt_edge",
        "scharr_edge",
        "laplacian_edge",
    }

    if method_name in well_compiled and device.type == "cuda":
        return {
            "use_compile": True,
            "compile_mode": "reduce-overhead",  # или "max-autotune" для тщательной оптимизации
            "compile_fullgraph": True,
            "compile_dynamic": False,
        }
    elif method_name in well_compiled:
        return {
            "use_compile": True,
            "compile_mode": "default",
            "compile_fullgraph": False,
            "compile_dynamic": True,
        }
    else:
        return {"use_compile": False}


# ──────────────────────────────────────────────────────────────────────
def main(use_optimizations: bool = True) -> Tuple[
    Optional[SegmentationTester],
    Optional[BenchmarkResult],
    Optional[SegmentationComparator],
]:
    """Основная точка входа фреймворка сегментации.

    Выполняет последовательное тестирование классических и нейросетевых методов:
    1. Инициализация окружения и автоматический подбор точности/компиляции.
    2. Загрузка тестовых изображений (с/без Ground Truth).
    3. Регистрация методов сегментации (OpenCV, scikit-learn, PyTorch v1/v2).
    4. Опциональный запуск исследовательских блоков:
       - Мульти-бэкенд бенчмарк с точностями (PyTorch/ONNX/TRT × fp32/fp16/bf16)
       - Экспорт и сравнение бэкендов (классические методы)
       - Профилирование с детекцией CPU↔GPU трансферов
       - Бенчмарк точностей (fp32 vs fp16 vs bf16)
       - Массовое тестирование на ADE20K
       - CPU vs CUDA сравнение
    5. Генерация отчётов: таблицы, графики, CSV, HTML.

    Алгоритм:
    ```
    1. Загрузить конфигурацию из YAML и определить устройство/точность.
    2. Загрузить тестовые изображения и извлечь первое для бенчмарков.
    3. Инициализировать SegmentationTester и зарегистрировать методы (CV2, Sklearn, Torch v1/v2).
    4. Если test_classic_logic:
       - Запустить _run_multi_backend_precision_benchmark()
       - Запустить _run_backend_export_and_comparison()
       - Запустить профилирование и бенчмарк точностей
       - Запустить CPU/CUDA бенчмарк
    5. Если test_neural_logic:
       - Загрузить и протестировать нейросетевые модели
       - Запустить нейросетевой бенчмарк на ADE20K
       - Запустить обучение с аугментациями
    6. Запустить массовое тестирование классических методов (_run_batch_classic_testing_optimized).
    7. Вернуть (tester, results_df, None).
    ```

    Args:
        use_optimizations: Если `True`, включает `torch.compile`, AMP и квантование.
                           Если `False`, запускает legacy-пайплайн без оптимизаций.

    Returns:
        Tuple[Optional[SegmentationTester], Optional[BenchmarkResult], Optional[SegmentationComparator]]:
            - `tester`: Экземпляр тестера с зарегистрированными методами.
            - `results`: DataFrame с результатами массового тестирования.
            - `comparator`: Экземпляр компаратора для матричных сравнений (зарезервировано, возвращается None).

    Note:
        - Управляющее поведение задаётся в `configs/main_config.yaml` (флаги `test_classic_logic`, `test_neural_logic`).
        - При ошибках в отдельных методах выполнение продолжается (graceful degradation).
        - Для прерывания длительных бенчмарков используется `Ctrl+C` (обработка `KeyboardInterrupt`).
        - Все результаты сохраняются в директорию `./data/` с автоматической структурой.
        - Функция поддерживает как оптимизированный (v2, compile, AMP), так и legacy-режимы.

    Example:
        ```python
        if __name__ == "__main__":
            # Запуск с оптимизациями
            tester, results, _ = main(use_optimizations=True)
            if results is not None:
                print(f"Обработано {len(results)} методов")
                print(results.sort_values("IoU", ascending=False).head(10))
        ```
    """
    # ──────────────────────────────────────────────────────────────
    # 1. КОНФИГУРАЦИЯ И ИНИЦИАЛИЗАЦИЯ
    # ──────────────────────────────────────────────────────────────
    config: Dict[str, Any] = _load_config()
    test_neural_logic: bool = False
    # test_classic_logic: bool = config["test_settings"]["test_classic_logic"]
    test_classic_logic: bool = True
    use_torch_v1: bool = True
    use_torch_v2: bool = True
    enable_profiling: bool = True  # Включить профилирование
    enable_benchmark_precision: bool = True  # Бенчмарк точностей

    _log_environment_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimal_precision = get_optimal_precision(device)
    print(f"🚀 Используемое устройство: {device}")
    print(f"⚡ Точность: {optimal_precision}")
    if use_optimizations:
        print(f"⚡ Оптимизации включены: Точность={optimal_precision}, Compile=ON")
    else:
        logger.warning("⚠️  [LEGACY] Оптимизации отключены (fp32, eager mode)")

    print("=" * 60)
    print("ОБЪЕДИНЁННЫЙ ФРЕЙМВОРК СЕГМЕНТАЦИИ")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────────
    # 2. ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ
    # ──────────────────────────────────────────────────────────────
    print("\n1. Загрузка тестовых изображений...")
    test_images: TestImagesDict = load_test_images(use_image_with_mask=False)
    print(f"✅ Загружено изображений: {len(test_images)}")

    first_img_pil: Optional[Image.Image] = None
    first_img_name: Optional[str] = None
    if test_images:
        first_img_name, (_, first_img_pil, _) = next(iter(test_images.items()))
        print(f"📌 Первое изображение для бенчмарков: {first_img_name} ({first_img_pil.size})")

    # ──────────────────────────────────────────────────────────────
    # 3. РЕГИСТРАЦИЯ МЕТОДОВ СЕГМЕНТАЦИИ
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ИНИЦИАЛИЗАЦИЯ МЕТОДОВ СЕГМЕНТАЦИИ")
    print("=" * 60)

    tester: SegmentationTester = SegmentationTester(
        base_output_dir="./data/segmentation_tester_results",
        enable_warmup=True,
        n_warmup_runs=5,
    )

    print("\n1. Загрузка методов OpenCV...")
    cv2_methods: SegmenterDict = _create_cv2_methods()

    print("\n2. Загрузка методов SKlearn...")
    sklearn_methods: SegmenterDict = _create_sklearn_methods()

    print("\n3. Загрузка методов PyTorch...")
    if use_optimizations:
        torch_methods: SegmenterDict = _create_torch_methods_factory(
            use_v1=use_torch_v1,
            use_v2=use_torch_v2,
            device=device,
            precision="fp32" if device.type == "cpu" else optimal_precision,
            enable_compile=True,
        )
    else:
        # 🔥 Fallback на старую реализацию
        torch_methods = _create_torch_methods()
    print(f"   📦 Загружено методов Torch: {len(torch_methods)}")

    if test_classic_logic:
        _register_classic_methods(tester, cv2_methods, sklearn_methods, torch_methods)

    # Очистка памяти перед загрузкой тяжёлых нейросетей
    _clear_gpu_memory()

    # ──────────────────────────────────────────────────────────────
    # 4. ОПЦИОНАЛЬНЫЕ БЛОКИ (вынесены в функции)
    # ──────────────────────────────────────────────────────────────

    if test_classic_logic:
        print("🔬 ИССЛЕДОВАНИЕ: Мульти-бэкенд бенчмарк (PyTorch / ONNX / TensorRT)")
        precision_benchmark_result: Optional[BenchmarkResult] = _run_multi_backend_precision_benchmark(
            tester, first_img_pil
        )
        if precision_benchmark_result is not None:
            print(precision_benchmark_result)

    print("\n⏳ Пауза 15 секунд перед запуском бенчмарка...")
    print("   (нажмите Ctrl+C для отмены, если нужно)")
    try:
        time.sleep(15)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Бенчмарк пропущен по запросу пользователя")
        return tester, None, None

    # 4.1 Профилирование выбранных методов
    if enable_profiling and test_classic_logic and use_torch_v2:
        _run_profiling_demo(tester, test_images, device)

    # 4.2 Бенчмарк точностей (опционально, медленно)
    if enable_benchmark_precision and test_classic_logic and use_torch_v2:
        _run_precision_benchmark_demo(tester, test_images)

        print("\n" + "=" * 60)
        print("⚡ БЕНЧМАРК ТОЧНОСТЕЙ: fp32 / fp16 / bf16")
        print("=" * 60)

        # Берём первое тестовое изображение в формате numpy
        _, (_, first_img_pil, _) = next(iter(test_images.items()))
        test_img_np = np.array(first_img_pil)

        # Выбираем методы для теста (например, только оптимизированные Torch v2)
        target_methods = []
        for name, seg in torch_methods.items():
            if name.endswith("_v2"):
                if hasattr(seg, "method"):
                    target_methods.append(seg.method)
                else:
                    # Fallback на случай, если атрибута method нет
                    target_methods.append(name.replace("_Torch_v2", "").lower())

        print(f"🧪 Выбрано методов для бенчмарка: {len(target_methods)}")
        print(f"📋 Список методов: {target_methods}")

        if target_methods:
            try:
                os.makedirs("./data/reports/precision", exist_ok=True)
                generate_precision_report(
                    methods=target_methods,
                    image=test_img_np,
                    output_path="./data/reports/precision/precision_benchmark.csv",
                    n_warmup=3,
                    n_runs=10,
                    compute_metrics=True,  # Сравнивает IoU относительно fp32
                )
                print("✅ Бенчмарк точностей завершён. CSV-отчёт: ./data/reports/precision/precision_benchmark.csv")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка при запуске бенчмарка точностей: {e}")
                traceback.print_exc()
        else:
            logger.warning("⚠️ Методы Torch v2 не найдены. Пропускаем бенчмарк точностей.")

    # 4.3 Бенчмарк производительности (cold/hot)
    if test_classic_logic:
        perf_results: Optional[pd.DataFrame] = run_performance_benchmark(
            tester=tester,
            test_images=test_images,
            n_runs=10,
            warmup_runs=10,
        )
        print(perf_results)

    # 4.4 Запускает тестирование нейросетевых методов сегментации.
    if test_neural_logic:
        _run_neural_segmentation_tests(tester, device)

    # 4.5 Нейросетевой бенчмарк
    if test_neural_logic:
        neural_results: Optional[Dict[str, Any]] = run_neural_segmentation_benchmark(
            device=device,
            num_classes=NUM_CLASSES_ADE20K,
        )
        print(neural_results)

    #  4.6 Валидация реализаций
    if test_classic_logic:
        print("\n🔬 ИССЛЕДОВАНИЕ: Валидация с поддержкой бэкендов")
        validation_results: Optional[Dict[str, Any]] = run_implementation_validation(
            test_images=test_images,
            output_dir="./data/validation_with_backends",
            image_name="countryside",
            include_backends=True,  # 🔹 Включаем ONNX/TRT
            onnx_dir="./exported_models/onnx",
            trt_dir="./exported_models/tensorrt",
            input_shape=(1, 3, 512, 512),
        )
        if validation_results:
            print(f"✅ Валидация завершена: {len(validation_results['all_results'])} конфигураций")
            print(validation_results)

    # 4.7 Матричное сравнение
    if test_classic_logic:
        matrix_results: Optional[Dict[str, Any]] = run_matrix_comparison(
            test_images=test_images,
            cv2_methods=cv2_methods,
            sklearn_methods=sklearn_methods,
            torch_methods=torch_methods,
            reference_method="global_thresholding_CV2",
            include_backends=True,
            tester=tester
        )
        print(matrix_results)

    # 4.8 Оценка против GT (опционально)
    # if test_classic_logic:
    #     gt_results: Optional[Dict[str, Any]] = run_ground_truth_evaluation(
    #         test_images=test_images,
    #         cv2_methods=cv2_methods,
    #         sklearn_methods=sklearn_methods,
    #         torch_methods=torch_methods,
    #     )
    #     print(gt_results)

    # 4.9 Обучение с аугментациями
    if test_neural_logic:
        aug_results: Optional[Dict[str, Any]] = run_augmentation_training_study(
            root_dir="./data1/ade20k",
            checkpoint_dir="./models",
            device="cuda",
        )
        print(aug_results)

    # 4.10 Тестирование CPU/CUDA бенчмарка
    if test_classic_logic:
        print("\n" + "=" * 80)
        print("🧪 ЗАПУСК MULTI-BACKEND CPU/CUDA БЕНЧМАРКА")
        print("=" * 80)

        # 1. Собираем все методы в один словарь
        all_benchmark_methods: SegmenterDict = {}
        all_benchmark_methods.update(cv2_methods)
        all_benchmark_methods.update(sklearn_methods)
        all_benchmark_methods.update(torch_methods)

        # 2. Добавляем экспортированные ONNX/TRT методы (если они уже зарегистрированы в tester)
        # Можно взять их из tester.methods или создать напрямую
        for name, seg in tester.methods.items():
            if "ONNX" in name or "TRT" in name:
                all_benchmark_methods[name] = seg  # type: ignore[assignment]

        # 3. Выбираем изображение
        test_image = None
        for img_name, (_, img_pil, _) in tqdm(test_images.items(), desc="CUDA/CPU benchmark (Выбор изображения)"):
            test_image = np.array(img_pil)
            print(f"✅ Используем изображение: {img_name} ({test_image.shape})")
            break

        if test_image is not None:
            cpu_cuda_results: BenchmarkResult = run_cpu_cuda_benchmark(
                all_benchmark_methods=all_benchmark_methods,
                test_image=test_image,
                n_runs=10,  # Увеличил прогонов для стабильности
                warmup_runs=5,
            )
            print(cpu_cuda_results.sort_values("mean_time"))

            # Сортировка и вывод топ-10 по времени
            top10: BenchmarkResult = (
                cpu_cuda_results[cpu_cuda_results["error"].isna()].sort_values("mean_time").head(10)
            )
            print(top10[["method", "device", "mean_time", "backend", "precision"]])

    # ──────────────────────────────────────────────────────────────
    # 5. МАССОВОЕ ТЕСТИРОВАНИЕ КЛАССИЧЕСКИХ МЕТОДОВ
    # ──────────────────────────────────────────────────────────────
    print(f"Current precision: {optimal_precision}")
    results_df: Optional[BenchmarkResult] = None
    if test_classic_logic:
        results_df = _run_batch_classic_testing_optimized(
            tester=tester,
            device=device,
            precision=optimal_precision,
        )

    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"✓ Методов протестировано: {len(tester.methods)}")
    print(f"✓ Изображений обработано: {len(test_images)}")
    print("✓ Результаты в: ./data/")

    return tester, results_df, None

def _extract_base_method(parts: List[str], backend: str, 
                        backends: Set[str], composite_backends: Set[str], 
                        precisions: Set[str]) -> str:
    """Извлекает BaseMethod с поддержкой составных бэкендов."""
    
    # 1. Составной бэкенд: ищем последовательность частей
    if backend in composite_backends:
        backend_parts = backend.split("_")
        for start_idx in range(len(parts) - len(backend_parts) + 1):
            if parts[start_idx:start_idx + len(backend_parts)] == backend_parts:
                return "_".join(parts[:start_idx])
    
    # 2. Простой бэкенд
    elif backend in backends and backend in parts:
        backend_idx = parts.index(backend)
        return "_".join(parts[:backend_idx])
    
    # 3. Fallback: удаляем известные суффиксы с конца
    all_suffixes = backends | precisions | composite_backends
    base_parts = parts[:]
    while base_parts and base_parts[-1] in all_suffixes:
        base_parts.pop()
    return "_".join(base_parts) if base_parts else "_".join(parts)


def parse_method_name(method_name: str) -> Dict[str, str]:
    """Надёжно парсит имя метода вида 'base_method_Backend_precision'.

    Выполняет:
    1. **Разбиение**: Сегментация строки по символу подчёркивания.
    2. **Поиск с конца**: Итеративный поиск известных бэкендов и точностей.
    3. **Извлечение базы**: Формирование базового имени до первого найденного бэкенда.
    4. **Валидация**: Фоллбэк на удаление суффиксов, если бэкенд не найден явно.

    Алгоритм:
    ```
    1. Разбить `method_name` на части по "_".
    2. Итерировать с конца списка частей:
       - Если часть в BACKENDS → запомнить Backend, проверить следующую на Precision.
       - Если часть в PRECISIONS и Backend ещё не найден → запомнить Precision.
    3. Если Backend найден → BaseMethod = части до индекса Backend.
    4. Иначе → удалить все известные суффиксы с конца → остаток = BaseMethod.
    5. Вернуть Dict{"BaseMethod": str, "Backend": str, "Precision": str}.
    ```

    Args:
        method_name: Строка с именем метода, например `gradient_magnitude_direction_TRT_fp16`.

    Returns:
        Dict[str, str]: Словарь с полями:
                        - `BaseMethod`: Оригинальное имя алгоритма.
                        - `Backend`: PyTorch/ONNX/TRT/CV2/Sklearn/Unknown.
                        - `Precision`: fp32/fp16/bf16/v1/v2.

    Note:
        - Оптимизирован для имён с множественными подчёркиваниями.
        - Использует `Set` для быстрого поиска (`O(1)`).
        - Автоматически обрабатывает отсутствующие суффиксы, подставляя дефолтные значения.

    Example:
        ```python
        result = parse_method_name("otsu_thresholding_ONNX_fp16")
        print(result)
        # {'BaseMethod': 'otsu_thresholding', 'Backend': 'ONNX', 'Precision': 'fp16'}
        ```
    """
    # 🔥 Известные значения — ключ к надёжному парсингу
    BACKENDS: Set[str] = {"Torch", "ONNX", "TRT", "CV2", "Sklearn"}
    PRECISIONS: Set[str] = {"fp32", "fp16", "bf16", "v1", "v2"}
    COMPOSITE_BACKENDS = {"ONNX_TRT_EP"} 

    parts: List[str] = method_name.split("_")

    # Ищем бэкенд и точность с конца строки
    backend: str = "Unknown"
    precision: str = "fp32"

    # Проверяем последние 1-2 части
    for i in range(len(parts) - 1, max(-1, len(parts) - 3), -1):
        part = parts[i]
        
        if part in PRECISIONS:
            precision = part
            continue
            
        # Проверка составного бэкенда (3 части: X_Y_Z)
        if i >= 2:
            composite = f"{parts[i-2]}_{parts[i-1]}_{part}"
            if composite in COMPOSITE_BACKENDS:
                backend = composite
                break
        
        # Проверка простого бэкенда
        if part in BACKENDS:
            backend = part
            break

    base_method = _extract_base_method(
        parts, backend, BACKENDS, COMPOSITE_BACKENDS, PRECISIONS
    )

    return {"BaseMethod": base_method, "Backend": backend, "Precision": precision}


def _run_multi_backend_precision_benchmark(
    tester: SegmentationTester,
    first_img_pil: Optional[Image.Image],
) -> Optional[BenchmarkResult]:
    """Мульти-бэкенд бенчмарк с поддержкой множественных точностей (fp32/fp16/bf16).

    Выполняет экспорт, регистрацию и сравнение методов сегментации across разных бэкендов и точностей.
    1. **Подготовка**: Определение размера изображения и ожидание подтверждения пользователя.
    2. **Регистрация**: Вызов `_register_backend_methods_with_precision` для создания PyTorch/ONNX/TRT версий с суффиксами точности.
    3. **Бенчмарк**: Запуск `tester.benchmark_methods()` для замера времени выполнения.
    4. **Анализ**: Парсинг имён методов, построение сводной таблицы (pivot), расчёт speedup относительно fp32.

    Алгоритм:
    ```
    1. Проверить наличие `first_img_pil`. Если нет — вернуть None.
    2. Определить real_h, real_w из изображения.
    3. Ожидание 15 сек (с обработкой Ctrl+C).
    4. Вызвать _register_backend_methods_with_precision(...) для генерации моделей.
    5. Ожидание 15 сек (с обработкой Ctrl+C).
    6. Запустить tester.benchmark_methods(test_name="backend_precision_comparison").
    7. Если результаты есть:
       - Применить parse_method_name для разделения BaseMethod, Backend, Precision.
       - Построить pivot_table: index=[BaseMethod, Backend], columns=Precision, values=Mean_Time_s.
       - Вывести сравнение времени.
       - Рассчитать speedup = fp32_time / [fp16|bf16]_time для ONNX и TRT.
    8. Вернуть DataFrame с результатами.
    ```

    Args:
        tester: Экземпляр `SegmentationTester` для регистрации и запуска методов.
        first_img_pil: PIL.Image для определения входного разрешения и проведения бенчмарка.

    Returns:
        Optional[BenchmarkResult]: DataFrame с результатами бенчмарка или `None` при отмене/ошибке.

    Note:
        - Использует жёсткую задержку 15 секунд перед критическими этапами для ручного контроля.
        - Автоматически пропускает блок, если `first_img_pil` отсутствует.
        - Speedup рассчитывается как среднее по всем методам, где присутствуют обе точности.
        - Требует наличия экспортированных моделей в `./exported_models1/`.

    Example:
        ```python
        result = _run_multi_backend_precision_benchmark(tester, first_img_pil)
        if result is not None:
            print(result.pivot_table(index="Backend", columns="Precision", values="Mean_Time_s"))
        ```
    """
    if first_img_pil is None:
        logger.warning("⚠️ Пропуск сравнения бэкендов: нет тестового изображения")
        return None

    real_h, real_w = first_img_pil.size[1], first_img_pil.size[0] # PIL: (W, H)
    print(f"📐 Размер изображения: {real_w}x{real_h}")

    # Пауза перед экспортом/регистрацией
    print("\n⏳ Пауза 15 секунд перед запуском бенчмарка...")
    print("   (нажмите Ctrl+C для отмены, если нужно)")
    try:
        time.sleep(15)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Бенчмарк пропущен по запросу пользователя")
        return None

    exported_methods = export_all_classical_methods(
        output_base_dir="./exported_models1",
        precisions=AVAILABLE_PRECISIONS,
        methods=TARGET_METHODS_FOR_RESEARCH,
        input_shape=(1, 3, real_h, real_w),
        force_reexport=True,
        export_onnx=True,
        export_trt=torch.cuda.is_available(),
    )
    print(exported_methods)

    backend_registration = _register_backend_methods_with_precision(
        tester=tester,
        target_methods=TARGET_METHODS_FOR_RESEARCH,
        output_base_dir="./exported_models1",
        precisions=["fp32", "fp16", "bf16"],
        input_shape=(1, 3, real_h, real_w),
        trt_strategy="auto",
        enable_trt_ep=DEFAULT_BENCHMARK_CONFIG.get("enable_trt_ep_benchmark", True),
        trt_ep_preset=DEFAULT_BENCHMARK_CONFIG.get("trt_ep_preset", "production"),
        trt_ep_cache_path=DEFAULT_BENCHMARK_CONFIG.get("trt_ep_cache_path", "./cache/trt_ep"),
    )
    print(backend_registration)

    print("\n⏳ Пауза 15 секунд перед запуском бенчмарка...")
    print("   (нажмите Ctrl+C для отмены, если нужно)")
    try:
        time.sleep(15)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Бенчмарк пропущен по запросу пользователя")
        return None

    backend_results = tester.benchmark_methods(
        image=np.array(first_img_pil),
        n_runs=10,
        force_warmup=True,
        test_name="backend_precision_comparison",
    )

    if backend_results is not None and not backend_results.empty:
        parsed = backend_results["Method"].apply(parse_method_name)
        backend_results["BaseMethod"] = parsed.apply(lambda x: x["BaseMethod"])
        backend_results["Backend"] = parsed.apply(lambda x: x["Backend"])
        backend_results["Precision"] = parsed.apply(lambda x: x["Precision"])

        summary = backend_results.pivot_table(
            index=["BaseMethod", "Backend"],
            columns="Precision",
            values="Mean_Time_s",
            aggfunc="mean",
        )

        print("\n⚡ Сравнение времени выполнения (мс) по точностям:")
        print(summary.round(4).to_string())

        if "fp32" in summary.columns:
            for backend in ["ONNX", "TRT"]:
                if backend in summary.index.get_level_values("Backend"):
                    print(f"\n🚀 Speedup {backend} относительно PyTorch/fp32:")
                    for precision in ["fp16", "bf16"]:
                        if precision in summary.columns:
                            subset = summary.xs(backend, level="Backend", drop_level=False)
                            if precision in subset.columns and "fp32" in subset.columns:
                                speedup = subset["fp32"] / subset[precision]
                                print(f"   {precision}: {speedup.mean():.2f}x (среднее)")

    print("✅ Сравнение бэкендов завершено. Результаты сохранены.")
    return backend_results

# ──────────────────────────────────────────────────────────────────────
def _register_backend_methods_with_precision(
    tester: SegmentationTester,
    target_methods: List[str],
    output_base_dir: str = "./exported_models",
    precisions: Optional[List[str]] = None,
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    trt_strategy: str = "auto",
    enable_trt_ep: bool = True,
    trt_ep_preset: Optional[str] = "production",
    trt_ep_cache_path: Optional[str] = "./cache/trt_ep",
) -> Dict[str, Any]:
    """Регистрирует методы для разных бэкендов и точностей с суффиксами в названиях.

    Автоматически создаёт и регистрирует в тестере сегментеры для PyTorch, ONNX и TensorRT,
    поддерживая множественные форматы данных (fp32, fp16, bf16).
    1. **PyTorch**: Базовая реализация в eager-режиме (fp32) как референс.
    2. **ONNX**: Загрузка экспортированных моделей для каждой доступной точности.
    3. **TensorRT**: Загрузка оптимизированных движков (только при наличии CUDA).
    4. **Сводка**: Формирование отчёта об успешных, пропущенных и упавших методах.

    Алгоритм:
    ```
    1. Если precisions не указан, автоматически определить доступные на основе torch.cuda.is_available() и device capability.
    2. Для каждого метода из target_methods:
       a. Создать TorchSegmenter2 (fp32, use_compile=False) → добавить как {method}_Torch_fp32.
       b. Для каждой точности:
          - Проверить путь к .onnx файлу.
          - Если существует → создать ONNXSegmenter → добавить как {method}_ONNX_{precision}.
          - Проверить путь к .trt файлу (если CUDA).
          - Если существует → загрузить load_trt_engine → создать TRTSegmenter → добавить как {method}_TRT_{precision}.
       c. Логировать успехи/ошибки/пропуски.
    3. Пауза 15 секунд перед завершением.
    4. Вывести сводку и вернуть словарь с результатами регистрации.
    ```

    Args:
        tester: Экземпляр `SegmentationTester` для добавления методов.
        target_methods: Список имён методов для регистрации (например, `TARGET_METHODS_FOR_RESEARCH`).
        output_base_dir: Базовая директория экспортированных моделей (по умолчанию `"./exported_models"`).
        precisions: Список точностей для регистрации (по умолчанию авто-определение `["fp32", "fp16", "bf16"]`).
        input_shape: Форма входного тензора для ONNX/TRT (по умолчанию `(1, 3, 512, 512)`).
        trt_strategy: Стратегия построения TensorRT (по умолчанию `"auto"`).

    Returns:
        Dict[str, Any]: Словарь с ключами:
                        - `success`: список успешно зарегистрированных имён методов.
                        - `failed`: список методов с ошибками.
                        - `skipped`: список пропущенных методов (файлы не найдены).

    Note:
        - Имена методов формируются по шаблону `{base}_{Backend}_{precision}` для удобного парсинга.
        - ONNX/TensorRT модели должны быть заранее экспортированы в соответствующие подпапки.
        - При отсутствии CUDA TensorRT регистрация полностью пропускается.
        - Функция содержит встроенную 15-секундную задержку перед финальным возвратом.

    Example:
        ```python
        reg_status = _register_backend_methods_with_precision(
            tester=tester,
            target_methods=["otsu_thresholding", "sobel_edge"],
            precisions=["fp32", "fp16"],
            input_shape=(1, 3, 512, 512)
        )
        print(f"Успешно: {len(reg_status['success'])}, Ошибки: {len(reg_status['failed'])}")
        ```
    """
    if precisions is None:
        # Авто-определение доступных точностей
        precisions = ["fp32"]
        if torch.cuda.is_available():
            precisions.append("fp16")
            if torch.cuda.get_device_capability(0)[0] >= 8:  # Ampere+
                precisions.append("bf16")

    registered: Dict[str, Any] = {"success": [], "failed": [], "skipped": []}

    for method_name in target_methods:
        print(f"\n🔹 Регистрация бэкендов: {method_name}")

        # ──────────────────────────────────────────────────
        # 1. PyTorch (оригинал) — только fp32 как референс
        # ──────────────────────────────────────────────────
        try:
            pt_seg = TorchSegmenter2(
                method=method_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                precision="fp32",
                use_compile=False,  # Чистый eager mode для сравнения
            )
            method_key: str = f"{method_name}_Torch_fp32"
            tester.add_method(method_key, pt_seg)
            registered["success"].append(method_key)
            print(f"   ✅ {method_key}")
        except Exception as e:
            registered["failed"].append(f"{method_name}_Torch_fp32: {e}")
            logger.error(f"   ❌ {method_name}_Torch_fp32: {e}")

        # ──────────────────────────────────────────────────
        # 2. ONNX для каждой доступной точности
        # ──────────────────────────────────────────────────
        for precision in precisions:
            onnx_path: str = f"{output_base_dir}/onnx/{precision}/{method_name}.onnx"
            method_key = f"{method_name}_ONNX_{precision}"
            if not os.path.exists(onnx_path):
                registered["skipped"].append(f"{method_name}_ONNX_{precision} (файл не найден)")
                continue

            try:
                onnx_seg = ONNXSegmenter(
                    method_name,
                    onnx_path,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    input_shape=input_shape,
                    precision=precision,  # Передаём точность для корректной инициализации
                    is_neural=False,
                )
                tester.add_method(method_key, onnx_seg)
                registered["success"].append(method_key)
                print(f"✅ Загружен {method_key}")
            except Exception as e:
                registered["failed"].append(f"{method_key}: {e}")
                logger.error(f"   ❌ Не загружен {method_key}: {e}")

        # ───────── 2.1. ONNX с TensorRT EP (опционально) ─────────
        if torch.cuda.is_available() and enable_trt_ep:
            for precision in precisions:
                onnx_path = f"{output_base_dir}/onnx/{precision}/{method_name}.onnx"
                if not os.path.exists(onnx_path):
                    continue
                    
                method_key = f"{method_name}_ONNX_TRT_EP_{precision}"
                
                trt_seg = create_onnx_trt_ep_segmenter(
                    method_name=method_name,
                    onnx_path=onnx_path,
                    precision=precision,
                    input_shape=input_shape,
                    device="cuda",
                    trt_preset=trt_ep_preset,
                    cache_path=trt_ep_cache_path,
                    is_neural=False,
                    num_classes=1,    # Для бинарных классических методов
                    normalization="none",  # Классика не требует ImageNet нормализации
                )
                
                if trt_seg is not None:
                    tester.add_method(method_key, trt_seg)
                    registered["success"].append(method_key)
                    print(f"   ✅ {method_key} (TensorRT EP, preset='{trt_ep_preset}')")
                else:
                    # Fallback: обычный ONNX уже зарегистрирован выше
                    registered["skipped"].append(f"{method_key} (TRT EP unavailable)")
                    logger.warning(f"⚠️ TRT EP не доступен в ONNX Runtime для {method_name}")


        # ──────────────────────────────────────────────────
        # 3. TensorRT для каждой доступной точности (только CUDA)
        # ──────────────────────────────────────────────────
        if not torch.cuda.is_available():
            continue

        for precision in precisions:
            trt_path: str = f"{output_base_dir}/tensorrt/{precision}/{method_name}.trt"
            method_key = f"{method_name}_TRT_{precision}"
            if not os.path.exists(trt_path):
                registered["skipped"].append(f"{method_name}_TRT_{precision} (файл не найден)")
                continue
            try:
                trt_model: Any = load_trt_engine(trt_path, device="cuda")
                if trt_model is not None:
                    trt_seg = TRTSegmenter(method_name, trt_model, device="cuda", precision=precision, is_neural=False)
                    tester.add_method(method_key, trt_seg)
                    registered["success"].append(method_key)
                    print(f"   ✅ Загружен {method_key}")
                else:
                    registered["failed"].append(f"{method_key} (модель=None)")
            except Exception as e:
                registered["failed"].append(f"{method_key}: {e}")
                logger.error(f"   ❌ Не загружен {method_key}: {e}")

        print("\n⏳ Пауза 15 секунд перед запуском бенчмарка...")
        print("   (нажмите Ctrl+C для отмены, если нужно)")
        try:
            time.sleep(15)  # 🔥 Задержка 15 секунд
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Бенчмарк пропущен по запросу пользователя")

    # ──────────────────────────────────────────────────
    # Сводка регистрации
    # ──────────────────────────────────────────────────
    print("\n📊 Сводка регистрации бэкендов:")
    print(f"   ✅ Успешно: {len(registered['success'])}")
    print(f"   ❌ Ошибки: {len(registered['failed'])}")
    print(f"   ⚠️  Пропущено: {len(registered['skipped'])}")

    if registered["failed"]:
        logger.error("\n⚠️  Ошибки:")
        for err in registered["failed"][:20]:
            logger.error(f"   • {err}")

    return registered


# ──────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ main()
# ──────────────────────────────────────────────────────────────────────
def _log_environment_info() -> None:
    """Логирует информацию об окружении: пути, CUDA, память."""
    print(f"📍 CWD: {os.getcwd()}")
    print(f"📍 __file__: {__file__}")
    print(f"📍 sys.path: {sys.path[:3]}...")

    print(f"🚀 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props: Any = torch.cuda.get_device_properties(0)
        print("🔥 CUDA available:")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        vram_gb: float = props.total_memory / 1024**3
        print(f"   VRAM: {vram_gb:.1f} GB")
        print(f"   GPU Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"Full props: {props}")
    else:
        print("💻 CUDA not available, using CPU")


# ──────────────────────────────────────────────────────────────────────
def _create_cv2_methods() -> SegmenterDict:
    """Создаёт словарь методов сегментации на основе OpenCV.

    Returns:
        SegmenterDict: Словарь {имя_метода: OpenCVSegmenter}.
    """
    return {
        # --- Пороговые методы (Threshold) ---
        "global_thresholding_CV2": OpenCVSegmenter("global_thresholding", threshold=0.5),
        "otsu_thresholding_CV2": OpenCVSegmenter("otsu_thresholding"),
        "adaptive_thresholding_CV2": OpenCVSegmenter("adaptive_thresholding", block_size=11, C=2),
        "threshold_niblack_CV2": OpenCVSegmenter("threshold_niblack", window_size=15, k=-0.2),
        "threshold_sauvola_CV2": OpenCVSegmenter("threshold_sauvola", window_size=15, k=0.5, r=128),
        "threshold_bernsen_CV2": OpenCVSegmenter("threshold_bernsen", window_size=15, contrast_threshold=0.15),
        "threshold_phansalkar_CV2": OpenCVSegmenter("threshold_phansalkar", window_size=15, k=0.25, r=128.0, m=0.5),
        "threshold_kittler_illingworth_CV2": OpenCVSegmenter("threshold_kittler_illingworth", num_bins=256),
        "threshold_entropy_kapur_CV2": OpenCVSegmenter("threshold_entropy_kapur", num_bins=256),
        "threshold_triangle_CV2": OpenCVSegmenter("threshold_triangle", num_bins=256),
        "threshold_multi_otsu_CV2": OpenCVSegmenter("threshold_multi_otsu", n_thresholds=2),
        "threshold_percentile_CV2": OpenCVSegmenter("threshold_percentile", percentile=90),
        "threshold_local_contrast_CV2": OpenCVSegmenter(
            "threshold_local_contrast", window_size=15, contrast_factor=0.1
        ),
        # # --- Граничные методы (Edge) ---
        # "sobel_edge_CV2": OpenCVSegmenter("sobel_edge", threshold=0.1),
        # "canny_edge_CV2": OpenCVSegmenter("canny_edge", low=0.1, high=0.3, sigma=1.0),
        # "prewitt_edge_CV2": OpenCVSegmenter("prewitt_edge", threshold=0.1),
        # "scharr_edge_CV2": OpenCVSegmenter("scharr_edge", threshold=0.1),
        # "roberts_cross_edge_CV2": OpenCVSegmenter("roberts_cross_edge", threshold=0.1),
        # "laplacian_edge_CV2": OpenCVSegmenter(
        #     "laplacian_edge", sigma=1.0, ksize=1, threshold=0.1, use_zero_crossing=False
        # ),
        # "log_edge_CV2": OpenCVSegmenter("log_edge", sigma=1.0, threshold=0.01),
        # "dog_edge_CV2": OpenCVSegmenter("dog_edge", sigma1=1.0, sigma2=2.0, threshold=0.01),
        # "marr_hildreth_edge_CV2": OpenCVSegmenter("marr_hildreth_edge", sigma=1.5, threshold=0.01),
        # "gradient_magnitude_direction_CV2": OpenCVSegmenter("gradient_magnitude_direction", threshold=0.1),
        # "phase_congruency_edge_CV2": OpenCVSegmenter(
        #     "phase_congruency_edge",
        #     nscales=4,
        #     norientations=4,
        #     min_wavelength=3,
        #     mult=2.0,
        #     sigma_onf=0.55,
        #     k_noise=2.0,
        #     threshold=0.5,
        # ),
        # "Region_Growing_CV2": OpenCVSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
        # "Split_And_Merge_CV2": OpenCVSegmenter("split_and_merge", min_size=50, threshold=0.1),
        # "Floodfill_CV2": OpenCVSegmenter("floodfill", seed=(100, 100), tolerance=0.15),
        # "KMeans_CV2": OpenCVSegmenter("kmeans_segmentation", k=3),
        # "DBSCAN_CV2": OpenCVSegmenter("dbscan_segmentation", eps=0.1, min_samples=10),
        # "Meanshift_CV2": OpenCVSegmenter("meanshift", spatial_radius=35, color_radius=60),
        # "Active_Contour_CV2": OpenCVSegmenter("active_contour", iterations=10),
        # "GVF_CV2": OpenCVSegmenter("gvf_contour", mu=0.1, iterations=50),
        # "Morphological_Snakes_CV2": OpenCVSegmenter("morphological_snakes", iterations=100),
        # "Chan_Vese_CV2": OpenCVSegmenter("chan_vese", mu=0.25, max_iter=100),
        # "Watershed_CV2": OpenCVSegmenter("watershed"),
        # "Random_Walker_CV2": OpenCVSegmenter("random_walker"),
        # # "Quickshift_CV2": OpenCVSegmenter("quickshift", bandwidth=0.5),
        # "Slic_CV2": OpenCVSegmenter("slic", region_size=20, ruler=10.0),
        # "Felzenszwalb_CV2": OpenCVSegmenter("felzenszwalb"),
        # "GrabCut_CV2": OpenCVSegmenter("grabcut", num_iterations=10),
    }


# ──────────────────────────────────────────────────────────────────────
def _create_sklearn_methods() -> SegmenterDict:
    """Создаёт словарь методов сегментации на основе scikit-learn.

    Returns:
        SegmenterDict: Словарь {имя_метода: SklearnSegmenter}.
    """
    return {
        # --- Пороговые методы (Threshold) ---
        "global_thresholding_Sklearn": SklearnSegmenter("global_thresholding", threshold=0.5, postprocess=False),
        "otsu_thresholding_Sklearn": SklearnSegmenter("otsu_thresholding", postprocess=False),
        "adaptive_thresholding_Sklearn": SklearnSegmenter(
            "adaptive_thresholding", block_size=11, C=2, postprocess=False
        ),
        "threshold_niblack_Sklearn": SklearnSegmenter("threshold_niblack", window_size=15, k=-0.2, postprocess=False),
        "threshold_sauvola_Sklearn": SklearnSegmenter(
            "threshold_sauvola", window_size=15, k=0.5, r=128, postprocess=False
        ),
        "threshold_bernsen_Sklearn": SklearnSegmenter(
            "threshold_bernsen", window_size=15, contrast_threshold=0.15, postprocess=False
        ),
        "threshold_phansalkar_Sklearn": SklearnSegmenter(
            "threshold_phansalkar", window_size=15, k=0.25, r=128.0, m=0.5, postprocess=False
        ),
        "threshold_kittler_illingworth_Sklearn": SklearnSegmenter(
            "threshold_kittler_illingworth", num_bins=256, postprocess=False
        ),
        "threshold_entropy_kapur_Sklearn": SklearnSegmenter("threshold_entropy_kapur", num_bins=256, postprocess=False),
        "threshold_triangle_Sklearn": SklearnSegmenter("threshold_triangle", num_bins=256, postprocess=False),
        "threshold_multi_otsu_Sklearn": SklearnSegmenter("threshold_multi_otsu", n_thresholds=2, postprocess=False),
        "threshold_percentile_Sklearn": SklearnSegmenter("threshold_percentile", percentile=90, postprocess=False),
        "threshold_local_contrast_Sklearn": SklearnSegmenter(
            "threshold_local_contrast", window_size=15, contrast_factor=0.1, postprocess=False
        ),
        # # --- Граничные методы (Edge) ---
        # "sobel_edge_Sklearn": SklearnSegmenter("sobel_edge", threshold=0.1, postprocess=False),
        # "canny_edge_Sklearn": SklearnSegmenter("canny_edge", low=0.1, high=0.3, sigma=1.0, postprocess=False),
        # "prewitt_edge_Sklearn": SklearnSegmenter("prewitt_edge", threshold=0.1, postprocess=False),
        # "scharr_edge_Sklearn": SklearnSegmenter("scharr_edge", threshold=0.1, postprocess=False),
        # "roberts_cross_edge_Sklearn": SklearnSegmenter("roberts_cross_edge", threshold=0.1, postprocess=False),
        # "laplacian_edge_Sklearn": SklearnSegmenter(
        #     "laplacian_edge", sigma=1.0, threshold=0.1, use_zero_crossing=False, postprocess=False
        # ),
        # "log_edge_Sklearn": SklearnSegmenter("log_edge", sigma=1.0, threshold=0.01, postprocess=False),
        # "dog_edge_Sklearn": SklearnSegmenter("dog_edge", sigma1=1.0, sigma2=2.0, threshold=0.01, postprocess=False),
        # "marr_hildreth_edge_Sklearn": SklearnSegmenter(
        #     "marr_hildreth_edge", sigma=1.5, threshold=0.01, postprocess=False
        # ),
        # "gradient_magnitude_direction_Sklearn": SklearnSegmenter(
        #     "gradient_magnitude_direction", threshold=0.1, postprocess=False
        # ),
        # "phase_congruency_edge_Sklearn": SklearnSegmenter(
        #     "phase_congruency_edge",
        #     nscales=4,
        #     norientations=4,
        #     min_wavelength=3,
        #     mult=2.0,
        #     sigma_onf=0.55,
        #     k_noise=2.0,
        #     threshold=0.5,
        #     postprocess=False,
        # ),
        # "Region_Growing_Sklearn": SklearnSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
        # "Split_And_Merge_Sklearn": SklearnSegmenter("split_and_merge", min_size=50, threshold=0.1),
        # "Floodfill_Sklearn": SklearnSegmenter("floodfill", seed=(100, 100), tolerance=0.15),
        # "KMeans_Sklearn": SklearnSegmenter("kmeans_segmentation", k=3),
        # "DBSCAN_Sklearn": SklearnSegmenter("dbscan_segmentation", eps=0.1, min_samples=10),
        # "MeanShift_Sklearn": SklearnSegmenter("meanshift", bandwidth=0.5),
        # "Active_Contour_Sklearn": SklearnSegmenter("active_contour", alpha=0.015, beta=10, gamma=0.001, max_iterations=2000, w_edge=1, w_line=0),
        # "GVF_Sklearn": SklearnSegmenter("gvf_contour", mu=0.1, iterations=50),
        # "Morphological_Snakes_Sklearn": SklearnSegmenter("morphological_snakes", iterations=100, smoothing=1, threshold=0.5),
        # "Chan_Vese_Sklearn": SklearnSegmenter("chan_vese", mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3, max_iter=100),
        # "Watershed_Sklearn": SklearnSegmenter("watershed"),
        # "Random_Walker_Sklearn": SklearnSegmenter("random_walker", beta=10),
        # # "Quickshift_Sklearn": SklearnSegmenter("quickshift", kernel_size=5, max_dist=10, ratio=1.0),
        # "Slic_Sklearn": SklearnSegmenter("slic", n_segments=100, compactness=10.0),
        # "Felzenszwalb_Sklearn": SklearnSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50),
        # "GrabCut_Sklearn": SklearnSegmenter("grabcut"),
    }


# ──────────────────────────────────────────────────────────────────────
def _create_torch_methods_factory(
    use_v1: bool = True,
    use_v2: bool = True,
    device: torch.device = torch.device("cpu"),
    precision: str = "fp32",
    enable_compile: bool = False,
) -> SegmenterDict:
    """🏭 Единая фабрика для создания методов TorchSegmenter v1 и v2.

    Выполняет:
    1. **Определение списков**: Формирование конфигураций для пороговых и граничных методов.
    2. **Создание v1**: Инициализация `TorchSegmenter` (legacy, без оптимизаций).
    3. **Создание v2**: Инициализация `TorchSegmenter2` с поддержкой:
       - Выбора точности (`fp32`/`fp16`/`bf16`)
       - Квантования на CPU
       - `torch.compile` с авто-конфигом
    4. **Формирование имён**: Добавление суффиксов `_v1`/`_v2` для сравнения.

    Алгоритм:
    ```
    1. Определить threshold_methods и edge_methods как список (name, method_name, params).
    2. Объединить в all_methods.
    3. Если use_v1:
       → для каждого: methods[f"{name}_v1"] = TorchSegmenter(method_name, **params)
    4. Если use_v2:
       → получить compile_cfg = get_compile_config(...) если enable_compile.
       → methods[f"{name}_v2"] = TorchSegmenter2(
           method_name, device=device.type, precision=precision,
           use_quantization=(device.type=="cpu"), **params, **compile_cfg)
    5. Вернуть methods.
    ```

    Args:
        use_v1: Создавать ли экземпляры `TorchSegmenter` (по умолчанию `True`).
        use_v2: Создавать ли экземпляры `TorchSegmenter2` (по умолчанию `True`).
        device: Устройство для инициализации сегментеров.
        precision: Точность вычислений для v2 (`"fp32"`, `"fp16"`, `"bf16"`).
        enable_compile: Включать ли авто-конфигурацию `torch.compile`.

    Returns:
        SegmenterDict: Словарь `{имя_метода: сегментер}` с суффиксами `_v1`/`_v2`.

    Note:
        - Суффиксы позволяют одновременно тестировать обе реализации.
        - `use_quantization` автоматически включается на CPU для ускорения.
        - Компиляция применяется только к методам из `well_compiled` списка.

    Example:
        ```python
        methods = _create_torch_methods_factory(
            use_v1=False, use_v2=True,
            device=torch.device("cuda"), precision="bf16", enable_compile=True
        )
        print(list(methods.keys())[:3])
        # ['global_thresholding_Torch_v2', 'otsu_thresholding_Torch_v2', ...]
        ```
    """
    methods: SegmenterDict = {}

    # Базовые списки методов
    threshold_methods: List[Tuple[str, str, Dict[str, Any]]] = [
        ("global_thresholding_Torch", "global_thresholding", {"threshold": 0.5}),
        ("otsu_thresholding_Torch", "otsu_thresholding", {}),
        (
            "adaptive_thresholding_Torch",
            "adaptive_thresholding",
            {"block_size": 11, "C": 2},
        ),
        (
            "threshold_niblack_Torch",
            "threshold_niblack",
            {"window_size": 15, "k": -0.2},
        ),
        (
            "threshold_sauvola_Torch",
            "threshold_sauvola",
            {"window_size": 15, "k": 0.5, "r": 128},
        ),
        (
            "threshold_bernsen_Torch",
            "threshold_bernsen",
            {"window_size": 15, "contrast_threshold": 0.15},
        ),
        (
            "threshold_phansalkar_Torch",
            "threshold_phansalkar",
            {"window_size": 15, "k": 0.25, "r": 128.0, "m": 0.5},
        ),
        (
            "threshold_kittler_illingworth_Torch",
            "threshold_kittler_illingworth",
            {"num_bins": 256},
        ),
        ("threshold_entropy_kapur_Torch", "threshold_entropy_kapur", {"num_bins": 256}),
        ("threshold_triangle_Torch", "threshold_triangle", {"num_bins": 256}),
        ("threshold_multi_otsu_Torch", "threshold_multi_otsu", {"n_thresholds": 2}),
        ("threshold_percentile_Torch", "threshold_percentile", {"percentile": 90}),
        (
            "threshold_local_contrast_Torch",
            "threshold_local_contrast",
            {"window_size": 15, "contrast_factor": 0.1},
        ),
    ]

    edge_methods: List[Tuple[str, str, Dict[str, Any]]] = [
        # ("sobel_edge_Torch", "sobel_edge", {"threshold": 0.1}),
        # ("canny_edge_Torch", "canny_edge", {"low": 0.1, "high": 0.3, "sigma": 1.0}),
        # ("prewitt_edge_Torch", "prewitt_edge", {"threshold": 0.1}),
        # ("scharr_edge_Torch", "scharr_edge", {"threshold": 0.1}),
        # ("roberts_cross_edge_Torch", "roberts_cross_edge", {"threshold": 0.1}),
        # ("log_edge_Torch", "log_edge", {"sigma": 1.0, "threshold": 0.01}),
        # ("laplacian_edge_Torch", "laplacian_edge", {"sigma": 1.0, "threshold": 0.1}),
        # (
        #     "dog_edge_Torch",
        #     "dog_edge",
        #     {"sigma1": 1.0, "sigma2": 2.0, "threshold": 0.01},
        # ),
        # (
        #     "marr_hildreth_edge_Torch",
        #     "marr_hildreth_edge",
        #     {"sigma": 1.5, "threshold": 0.01},
        # ),
        # (
        #     "gradient_magnitude_direction_Torch",
        #     "gradient_magnitude_direction",
        #     {"threshold": 0.1},
        # ),
        # (
        #     "phase_congruency_edge_Torch",
        #     "phase_congruency_edge",
        #     {
        #         "nscales": 4,
        #         "norientations": 4,
        #         "min_wavelength": 3,
        #         "mult": 2.0,
        #         "sigma_onf": 0.55,
        #         "k_noise": 2.0,
        #         "threshold": 0.5,
        #     },
        # ),
    ]

    all_methods = threshold_methods + edge_methods

    # Создание v1
    if use_v1:
        for name, method_name, params in all_methods:
            key = f"{name}_v1"
            methods[key] = TorchSegmenter(method=method_name, **params)

    # Создание v2
    if use_v2:
        for name, method_name, params in all_methods:
            key = f"{name}_v2"
            compile_cfg: Dict[str, Any] = get_compile_config(method_name, device) if enable_compile else {}
            methods[key] = TorchSegmenter2(
                method=method_name,
                device=device.type,
                precision=precision,
                use_quantization=(device.type == "cpu"),
                **params,
                **compile_cfg,
            )

    return methods


# ──────────────────────────────────────────────────────────────────────
def _create_torch_methods() -> SegmenterDict:
    """Создаёт словарь методов сегментации на основе PyTorch.

    Returns:
        SegmenterDict: Словарь {имя_метода: TorchSegmenter}.
    """
    return {
        # --- Пороговые методы (Threshold) ---
        "Global_Threshold_Torch": TorchSegmenter("global_thresholding", threshold=0.5),
        "Otsu_Thresholding_Torch": TorchSegmenter("otsu_thresholding"),
        "Adaptive_Threshold_Torch": TorchSegmenter("adaptive_thresholding", block_size=11, C=2),
        "Niblack_Thresholding_Torch": TorchSegmenter("threshold_niblack", window_size=15, k=-0.2),
        "Sauvola_Thresholding_Torch": TorchSegmenter("threshold_sauvola", window_size=15, k=0.5, r=128),
        "Bernsen_Thresholding_Torch": TorchSegmenter("threshold_bernsen", window_size=15, contrast_threshold=0.15),
        "Phansalkar_Thresholding_Torch": TorchSegmenter("threshold_phansalkar", window_size=15, k=0.25, r=128.0, m=0.5),
        "Kittler_Illingworth_Torch": TorchSegmenter("threshold_kittler_illingworth", num_bins=256),
        "Kapur_Entropy_Torch": TorchSegmenter("threshold_entropy_kapur", num_bins=256),
        "Triangle_Threshold_Torch": TorchSegmenter("threshold_triangle", num_bins=256),
        "Multi_Otsu_Torch": TorchSegmenter("threshold_multi_otsu", n_thresholds=2),
        "Percentile_Threshold_Torch": TorchSegmenter("threshold_percentile", percentile=90),
        "Local_Contrast_Torch": TorchSegmenter("threshold_local_contrast", window_size=15, contrast_factor=0.1),
        # # --- Граничные методы (Edge) ---
        # "Sobel_Torch": TorchSegmenter("sobel_edge", threshold=0.1),
        # "Canny_Torch": TorchSegmenter("canny_edge", low=0.1, high=0.3, sigma=1.0),
        # "Prewitt_Torch": TorchSegmenter("prewitt_edge", threshold=0.1),
        # "Scharr_Torch": TorchSegmenter("scharr_edge", threshold=0.1),
        # "Roberts_Cross_Torch": TorchSegmenter("roberts_cross_edge", threshold=0.1),
        # "LoG_Torch": TorchSegmenter("log_edge", sigma=1.0, threshold=0.01),
        # "DoG_Torch": TorchSegmenter("dog_edge", sigma1=1.0, sigma2=2.0, threshold=0.01),
        # "Laplacian_Torch": TorchSegmenter("laplacian_edge", sigma=1.0, threshold=0.1),
        # "Marr_Hildreth_Torch": TorchSegmenter("marr_hildreth_edge", sigma=1.5, threshold=0.01),
        # "Gradient_Mag_Dir_Torch": TorchSegmenter("gradient_magnitude_direction", threshold=0.1),
        # "Phase_Congruency_Torch": TorchSegmenter(
        #     "phase_congruency_edge",
        #     nscales=4,
        #     norientations=4,
        #     min_wavelength=3,
        #     mult=2.0,
        #     sigma_onf=0.55,
        #     k_noise=2.0,
        #     threshold=0.5,
        # ),
        # "Region_Growing_Torch": TorchSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
        # "Split_And_Merge_Torch": TorchSegmenter("split_and_merge", min_size=50, threshold=20),
        # "Floodfill_Torch": TorchSegmenter("floodfill", seed=(100, 100), tolerance=0.15),
        # "KMeans_Torch": TorchSegmenter("kmeans_segmentation", k=3),
        # "DBSCAN_Torch": TorchSegmenter("dbscan_segmentation", eps=0.5, min_samples=5),
        # "MeanShift_Torch": TorchSegmenter("meanshift", bandwidth=0.5, spatial_radius=35, color_radius=60),
        # "Active_Contour_Torch": TorchSegmenter("active_contour"),
        # "GVF_Torch": TorchSegmenter("gvf_contour"),
        # "Morphological_Snakes_Torch": TorchSegmenter("morphological_snakes", iterations=100, smoothing=1, threshold=0.5),
        # "Chan_Vese_Torch": TorchSegmenter("chan_vese", mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3, max_iter=100, dt=0.5, eps=1.0),
        # "Watershed_Torch": TorchSegmenter("watershed"),
        # "Random_Walker_Torch": TorchSegmenter("random_walker", beta=130, tol=1e-3, max_iter=300, target_label=2),
        # # "Quickshift_Torch": TorchSegmenter("quickshift", kernel_size=5, max_dist=10, ratio=1.0, sigma=0.0, convert2lab=True),
        # "Slic_Torch": TorchSegmenter("slic", n_segments=100, compactness=10.0, max_iter=10, sigma=0.0, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3.0),
        # "Felzenszwalb_Torch": TorchSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50),
        # "GrabCut_Torch": TorchSegmenter("grabcut", rect=(100, 100, 200, 200), num_iterations=5),
    }


# ──────────────────────────────────────────────────────────────────────
# ПРОФИЛИРОВАНИЕ И БЕНЧМАРКИ
# ──────────────────────────────────────────────────────────────────────
def _run_profiling_demo(
    tester: SegmentationTester,
    test_images: TestImagesDict,
    device: torch.device,
) -> None:
    """Демонстрация профилирования с детекцией CPU↔GPU трансферов.

    Выполняет:
    1. **Фильтрация**: Выбор только методов `_v2` (TorchSegmenter2).
    2. **Профилирование**: Запуск `profile_with_transfer_detection` с замером времени и памяти.
    3. **Анализ трансферов**: Вывод предупреждений о неоптимальных перемещениях данных.
    4. **Трассировка**: Генерация детальных трейсов через `profile_with_tracing`.

    Алгоритм:
    ```
    1. Извлечь первое изображение из test_images.
    2. Для каждого метода, оканчивающегося на "_v2":
       a. Проверить тип → только TorchSegmenter2.
       b. Вызвать profile_with_transfer_detection(n_runs=10).
       c. Вывести avg_time_ms, memory_mb, profiler_table.
       d. Если есть transfer_warnings → вывести первые 3.
       e. Вызвать profile_with_tracing(...) → сохранить в ./profiling/tests.
    3. Логировать ошибки без прерывания выполнения.
    ```

    Args:
        tester: Экземпляр `SegmentationTester` с зарегистрированными методами.
        test_images: Словарь тестовых изображений для профилирования.
        device: Устройство выполнения (используется для контекста).

    Returns:
        None. Выводит результаты в stdout и сохраняет трейсы на диск.

    Note:
        - Пропускает методы, не являющиеся `TorchSegmenter2`.
        - Трансфер-детекция помогает найти скрытые `.cpu()`/`.cuda()` вызовы.
        - Трассировка может занимать дополнительную память; результаты сохраняются построчно.

    Example:
        ```python
        _run_profiling_demo(tester, test_images, device)
        # В консоли: 📊 Профилирование: sobel_edge_Torch_v2
        # ⏱️  Среднее время выполнения: 1.24 мс
        # ✅ Трансферов не обнаружено. Память и GPU используются оптимально.
        ```
    """
    print("\n" + "=" * 60)
    print("🔍 ДЕМО ПРОФИЛИРОВАНИЯ")
    print("=" * 60)

    # Берём первое изображение для демо
    _, (_, img_pil, _) = next(iter(test_images.items()))
    img_array = np.array(img_pil)

    # Тестируем несколько методов
    methods_to_profile = [m for m in tester.methods if m.endswith("_v2")]

    for method_name in methods_to_profile:
        if method_name not in tester.methods:
            continue

        segmenter = tester.methods[method_name]
        if not isinstance(segmenter, TorchSegmenter2):
            continue

        print(f"\n📊 Профилирование: {method_name}")
        print("-" * 40)

        try:
            # Профилирование с детекцией трансферов
            profile = segmenter.profile_with_transfer_detection(
                image=img_array,
                n_runs=10,
                detect_transfers=True,
            )

            print(f"   ⏱️  Среднее время выполнения: {profile['avg_time_ms']:.2f} мс")
            print(f"   💾 Память: {profile['memory_mb']:.1f} МБ")

            print(profile["profiler_table"])
            if profile["transfer_warnings"]:
                print("\n⚠️  Предупреждения о трансферах:")
                for w in profile["transfer_warnings"]:
                    print(f"  {w}")
            else:
                print("✅ Трансферов не обнаружено. Память и GPU используются оптимально.")

            # Предупреждения о трансферах
            if profile.get("transfer_warnings"):
                print(f"   ⚠️  Трансферы ({len(profile['transfer_warnings'])}):")
                for w in profile["transfer_warnings"][:3]:  # Показываем первые 3
                    print(f"      • {w}")
            else:
                print("   ✅ Лишних трансферов не обнаружено")

            # Доступ к метаданным выполнения
            if "execution_info" in segmenter.params:
                exec_info = segmenter.params["execution_info"]
                print(f"   📋 Метод: {exec_info.get('method', 'N/A')}")
                print(f"   ⚡ Время: {exec_info.get('execution_time', 0) * 1000:.2f} мс")
                # Доступ к метаданным выполнения
                print(segmenter.params["execution_info"])

            segmenter.profile_with_tracing(segmenter, img_array, output_dir="./profiling/tests")

        except Exception as e:
            logger.error(f"   ❌ Ошибка профилирования: {e}")


def _run_precision_benchmark_demo(
    tester: SegmentationTester,
    test_images: TestImagesDict,
) -> None:
    """Демонстрация бенчмарка точностей (только для TorchSegmenter2).

    Выполняет:
    1. **Проверка окружения**: Выход при отсутствии CUDA.
    2. **Подготовка данных**: Конвертация изображения в градации серого.
    3. **Замер по точностям**: Вызов `benchmark_histc_types` для fp32/fp16/bf16.
    4. **Агрегация**: Формирование сводной таблицы и вывод в консоль.

    Алгоритм:
    ```
    1. Проверить torch.cuda.is_available(). Если False → пропустить.
    2. Взять первое изображение, конвертировать в grayscale.
    3. Для каждого метода "_v2":
       a. Вызвать benchmark_histc_types(gray, device).
       b. Сохранить time_ms для каждой точности.
    4. Собрать результаты в DataFrame.
    5. Вывести pivot_table: index=method, columns=precision, values=time_ms.
    ```

    Args:
        tester: Экземпляр `SegmentationTester` с методами.
        test_images: Словарь изображений (используется первое).

    Returns:
        None. Выводит таблицу сравнения времени выполнения.

    Note:
        - ⚠️ Медленно: последовательный запуск каждой точности.
        - Использует внутренний метод `_to_grayscale` сегментера.
        - Результаты выводятся в stdout, но не сохраняются автоматически.

    Example:
        ```python
        _run_precision_benchmark_demo(tester, test_images)
        # 🔹 otsu_thresholding_Torch_v2:
        #    fp32: 2.15 мс/вызов
        #    fp16: 1.42 мс/вызов
        #    bf16: 1.38 мс/вызов
        ```
    """
    if not torch.cuda.is_available():
        logger.warning("\n⚠️  Бенчмарк точностей требует CUDA. Пропускаем.")
        return

    print("\n" + "=" * 60)
    print("⚡ БЕНЧМАРК ТОЧНОСТЕЙ (fp32/fp16/bf16)")
    print("=" * 60)

    _, (_, img_pil, _) = next(iter(test_images.items()))
    img_array = np.array(img_pil)

    # Тестируем только быстрые методы
    test_methods = [m for m in tester.methods if m.endswith("_v2")]

    results: List[Dict[str, Any]] = []

    for method_name in test_methods:
        if method_name not in tester.methods:
            continue

        segmenter = tester.methods[method_name]
        if not isinstance(segmenter, TorchSegmenter2):
            continue

        print(f"\n🔹 {method_name}:")

        try:
            # Бенчмарк точностей
            precision_results = segmenter.benchmark_histc_types(
                gray=segmenter._to_grayscale(segmenter.preprocess_image(img_array)).squeeze(),
                device=segmenter.device,
            )

            for prec, time_ms in precision_results.items():
                print(f"   {prec}: {time_ms:.2f} мс/вызов")
                results.append(
                    {
                        "method": method_name,
                        "precision": prec,
                        "time_ms": time_ms,
                    }
                )
            # print(segmenter.params["execution_info"])

        except Exception as e:
            logger.error(f"   ❌ Ошибка: {e}")

    # Сводная таблица
    if results:
        df = pd.DataFrame(results)
        print("\n📊 Сводка по точностям:")
        print(
            df.pivot(index="method", columns="precision", values="time_ms").to_string(float_format=lambda x: f"{x:.2f}")
        )


# ──────────────────────────────────────────────────────────────────────
def _register_classic_methods(
    tester: SegmentationTester,
    cv2_methods: SegmenterDict,
    sklearn_methods: SegmenterDict,
    torch_methods: SegmenterDict,
) -> None:
    """Регистрирует классические методы в тестере с обработкой ошибок.

    Выполняет:
    1. **Объединение**: Слияние словарей CV2, Sklearn и Torch.
    2. **Итеративная регистрация**: Попытка добавления каждого метода в `tester`.
    3. **Логирование**: Вывод статуса ✅/⚠️ для каждого сегментера.
    4. **Graceful degradation**: Продолжение работы при падении отдельного метода.

    Алгоритм:
    ```
    1. all_methods = {**cv2_methods, **sklearn_methods, **torch_methods}
    2. Для каждого (name, segmenter) в all_methods:
       a. try: tester.add_method(name, segmenter) → print("✅")
       b. except Exception: logger.warning("⚠️ Не удалось добавить")
    ```

    Args:
        tester: Экземпляр `SegmentationTester` для регистрации.
        cv2_methods: Словарь методов OpenCV.
        sklearn_methods: Словарь методов scikit-learn.
        torch_methods: Словарь методов PyTorch.

    Returns:
        None. Модифицирует `tester` in-place.

    Note:
        - Порядок регистрации: CV2 → Sklearn → Torch.
        - Ошибки не прерывают выполнение, что позволяет тестировать частично рабочие сборки.
        - Дубликаты ключей перезаписываются в пользу последних в цепочке слияния.

    Example:
        ```python
        _register_classic_methods(tester, cv2_dict, sklearn_dict, torch_dict)
        # ✅ global_thresholding_CV2
        # ✅ otsu_thresholding_Sklearn
        # ⚠️ Не удалось добавить custom_method_Torch: Missing dependency
        ```
    """
    all_methods: SegmenterDict = {**cv2_methods, **sklearn_methods, **torch_methods}

    for name, segmenter in all_methods.items():
        try:
            tester.add_method(name, segmenter)
            print(f"   ✅ {name}")
        except Exception as e:
            logger.warning(f"   ⚠️ Не удалось добавить {name}: {e}")


# ──────────────────────────────────────────────────────────────────────
def _clear_gpu_memory() -> None:
    """Очищает память GPU и вызывает сборщик мусора."""
    print("🧹 Очистка памяти CUDA перед загрузкой тяжелой модели...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# ──────────────────────────────────────────────────────────────────────
def _run_neural_segmentation_tests(tester: SegmentationTester, device: torch.device) -> None:
    """Запускает тестирование нейросетевых методов сегментации.

    Args:
        tester: Экземпляр тестера для регистрации методов.
        device: Устройство для выполнения (cuda/cpu).
    """
    neural_models_config: List[Dict[str, Any]] = [
        # Предобученные
        {
            "name": "SegFormer_B5",
            "type": "segformer",
            "model_name": "nvidia/segformer-b5-finetuned-ade-640-640",
            "local_path": "/home/yamshchikov/models/segformer-b5-ready",
        },
        {
            "name": "Mask2Former",
            "type": "mask2former",
            "model_name": "facebook/mask2former-swin-base-ade-semantic",
        },
        # Обученные
        {
            "name": "DeepLabV3+_Trained",
            "type": "deeplab_tv",
            "checkpoint_path": "./models/deeplab_ade20k_best_200_epochs.pth",
        },
        {
            "name": "U-Net_Trained",
            "type": "unet_smp",
            "checkpoint_path": "./models/unet_ade20k_best_200_epochs.pth",
        },
        {
            "name": "FPN_MiT-B5_Trained",
            "type": "fpn_smp",
            "checkpoint_path": "./models/fpn_mit_b5_ade20k_best_200_epochs.pth",
        },
        {
            "name": "PSPNet_MiT-B5_Trained",
            "type": "pspnet_smp",
            "checkpoint_path": "./models/psp_mit_b5_ade20k_best_200_epochs.pth",
        },
        {
            "name": "FCN_ResNet50_Trained",
            "type": "fcn_tv",
            "checkpoint_path": "./models/fcn_resnet50_ade20k_best_200_epochs.pth",
        },
        {
            "name": "SegNet_Trained",
            "type": "segnet",
            "checkpoint_path": "./models/segnet_ade20k_best_200_epochs.pth",
        },
    ]

    print("\n4. Загрузка нейросетевых методов...")
    for config in neural_models_config:
        _load_single_neural_model(tester, config, device)


# ──────────────────────────────────────────────────────────────────────
def _load_single_neural_model(
    tester: SegmentationTester,
    config: Dict[str, Any],
    device: torch.device,
) -> None:
    """Загружает и регистрирует одну нейросетевую модель.

    Args:
        tester: Экземпляр тестера.
        config: Конфигурация модели.
        device: Устройство для выполнения.
    """
    print(f"Current device: {device.type}")
    try:
        if "checkpoint_path" in config and not os.path.exists(config["checkpoint_path"]):
            logger.warning(f"   ⚠️ {config['name']} - чекпоинт не найден: {config['checkpoint_path']}")
            return

        model_name = config.get("model_name")
        if model_name is None:
            model_name = config.get("local_path", "unknown")

        segmenter: NeuralSegmenter = NeuralSegmenter(
            model_type=config["type"],
            checkpoint_path=config.get("checkpoint_path"),
            local_path=config.get("local_path"),
            model_name=model_name,
            num_classes=NUM_CLASSES_ADE20K,
            **{
                k: v
                for k, v in config.items()
                if k not in ["name", "type", "checkpoint_path", "local_path", "model_name"]
            },
        )
        tester.add_method(config["name"], segmenter)
        print(f"   ✅ {config['name']}")
        # print(segmenter.params["execution_info"])
        segmenter.get_class_info()

    except Exception as e:
        logger.error(f"  ⚠️ Нейросетевая сегментация недоступна: {e}")
        logger.error(f"   ❌ {config['name']}: {e}")
        logger.error(traceback.format_exc())
        if os.getenv("DEBUG", "0") == "1":
            traceback.print_exc()

    print(f"\nВсего методов загружено: {len(tester.methods)}")

    # # # ========== ВАРИАНТ 2: Через YAML конфиг ==========
    # print("\n=== ВАРИАНТ 2: Через YAML конфиг ===")
    # _ = NeuralSegmenter(
    #     model_type="segformer",
    #     variant="b5",  # ← Берётся из configs/neural_models.yaml
    #     num_classes=num_classes,
    # )

    # # # ========== ВАРИАНТ 3: Factory + конфиг ==========
    # print("\n=== ВАРИАНТ 3: Factory метод ===")
    # _, _, _ = NeuralModelFactory.create_model_from_config(
    #     model_type="segformer", variant="b2", device="cuda"  # ← Берётся из конфига
    # )

    # # # ========== ВАРИАНТ 4: Обученная модель с чекпоинтом ==========
    # print("\n=== ВАРИАНТ 4: Обученная модель ===")
    # _ = NeuralSegmenter(
    #     model_type="unet_smp",
    #     encoder_name="resnet34",  # ← Можно из конфига
    #     checkpoint_path="./models/unet_ade20k_best.pth",
    #     num_classes=num_classes,
    # )

    # # # ========== ВАРИАНТ 5: Конфиг обучения ==========
    # print("\n=== ВАРИАНТ 5: Конфиг обучения ===")
    # training_config: Dict[str, Any] = NeuralModelFactory.get_training_config("ade20k")
    # print(f"Batch size: {training_config['batch_size']}")
    # print(f"Epochs: {training_config['epochs']}")
    # print(f"LR: {training_config['lr']}")

    # # # ========== ВАРИАНТ 6: Конфиг метрик ==========
    # print("\n=== ВАРИАНТ 6: Конфиг метрик ===")
    # metrics_config: Dict[str, Any] = NeuralModelFactory.get_metrics_config()
    # print(f"Threshold: {metrics_config['threshold']}")
    # print(f"Include Hausdorff: {metrics_config['include_hausdorff']}")

    # # # ========== ВАРИАНТ 7: Массовая загрузка из конфига ==========
    # print("\n=== ВАРИАНТ 7: Массовая загрузка ===")
    # neural_models_config: List[Dict[str, str]] = [
    #     {"name": "SegFormer_B5", "type": "segformer", "variant": "b5"},
    #     {"name": "SegFormer_B2", "type": "segformer", "variant": "b2"},
    #     {"name": "Mask2Former", "type": "mask2former", "variant": "swin_base"},
    #     {"name": "U-Net", "type": "unet_smp", "encoder_name": "resnet34"},
    #     {
    #         "name": "FPN_MiT",
    #         "type": "fpn_smp",
    #         "encoder_name": "mit_b5",
    #         "checkpoint_path": "./models/fpn_mit_b5_ade20k_best_200_epochs.pth",
    #     },
    # ]

    # for config in neural_models_config:
    #     try:
    #         # Проверяем есть ли variant (для HF моделей)
    #         if "variant" in config:
    #             segmenter = NeuralSegmenter(
    #                 model_type=config["type"],
    #                 variant=config["variant"],  # ← Из конфига
    #                 num_classes=num_classes,
    #             )
    #         # Или checkpoint_path (для обученных)
    #         elif "checkpoint_path" in config:
    #             segmenter = NeuralSegmenter(
    #                 model_type=config["type"],
    #                 encoder_name=config.get("encoder_name"),
    #                 checkpoint_path=config["checkpoint_path"],
    #                 num_classes=num_classes,
    #             )
    #         # Или прямое имя модели
    #         else:
    #             segmenter = NeuralSegmenter(
    #                 model_type=config["type"],
    #                 encoder_name=config.get("encoder_name"),
    #                 num_classes=num_classes,
    #             )

    #         print(f"   ✅ {config['name']}")
    #     except Exception as e:
    #         logger.error(f"   ❌ {config['name']} - {e}")


# ──────────────────────────────────────────────────────────────────────
# МАССОВОЕ ТЕСТИРОВАНИЕ С ОПТИМИЗАЦИЯМИ
# ──────────────────────────────────────────────────────────────────────
def _run_batch_classic_testing_optimized(
    tester: SegmentationTester,
    device: torch.device,
    precision: str,
) -> Optional[BenchmarkResult]:
    """Запускает массовое тестирование с учётом оптимизаций."""
    print("\n" + "=" * 80)
    print("🚀 МАССОВОЕ ТЕСТИРОВАНИЕ С ОПТИМИЗАЦИЯМИ")
    print(f"   Устройство: {device}, Точность: {precision}")
    print("=" * 80)

    try:
        # Конфигурация теста
        batch_tester: BatchClassicTester = BatchClassicTester(
            ade20k_root="./data1/ade20k/ADEChallengeData2016",
            output_dir="./data/batch_classic_test_optimized",
            split="validation",
            max_images=3,  # Лимит для быстрого теста
            image_size=DEFAULT_IMAGE_SIZE,
            save_masks=True,
            mask_sample_rate=1.0,  # 20% изображений
            max_mask_samples_per_method=10,
            save_visualizations=True,
            resume=True,
            include_backends=True,
            onnx_dir="./exported_models/onnx",
            trt_dir="./exported_models/tensorrt",
            supported_precisions=["fp32", "fp16", "bf16"] if torch.cuda.is_available() else ["fp32"],
        )

        print("⏳ Запуск тестирования...")
        results_df: BenchmarkResult = batch_tester.run_batch_test()

        # Сохранение и визуализация
        batch_tester.save_results(results_df)
        batch_tester.plot_results(results_df)
        batch_tester.print_summary(results_df)

        print(f"\n💾 Результаты: {batch_tester.output_dir}")
        return results_df

    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        traceback.print_exc()
        return None


# ──────────────────────────────────────────────────────────────────────
def _print_batch_test_summary(results_df: BenchmarkResult) -> None:
    """Выводит сводную статистику по результатам массового тестирования.

    Args:
        results_df: DataFrame с результатами бенчмарка.
    """
    print("\n" + "=" * 80)
    print("📊 СВОДКА ПО МАССОВОМУ ТЕСТИРОВАНИЮ")
    print("=" * 80)

    print("\n🏆 Топ-5 методов по IoU:")
    for i, row in results_df.head(5).iterrows():
        print(f"   {i + 1}. {row['Method']}: IoU={row['iou_mean']:.4f} ± {row['iou_std']:.4f}")

    print("\n⚡ Топ-5 самых быстрых методов:")
    fast_df: BenchmarkResult = results_df.dropna(subset=["time_mean_s"]).sort_values("time_mean_s").head(5)
    for i, row in fast_df.iterrows():
        print(f"   {i + 1}. {row['Method']}: {row['time_mean_s'] * 1000:.1f} мс")

    print("\n❌ Методы с наибольшим числом ошибок:")
    error_df: BenchmarkResult = (
        results_df[results_df["error_count"] > 0].sort_values("error_count", ascending=False).head(5)
    )
    if not error_df.empty:
        for i, row in error_df.iterrows():
            print(f"   {row['Method']}: {row['error_count']} ошибок ({row['error_rate'] * 100:.1f}%)")
    else:
        print("   Нет ошибок!")


# ──────────────────────────────────────────────────────────────────────
def prepare_mask_for_overlay(mask_input: Union[Image.Image, npt.NDArray]) -> MaskArray:
    """Конвертирует входную маску в 2D numpy array для наложения на изображение.

    Поддерживаемые форматы входа:
    - `PIL.Image` (режимы 'L', 'RGB', 'RGBA')
    - `numpy.ndarray` с размерностью 2 или 3
    - Цветные label-изображения (конвертируются в одноканальные)

    Алгоритм обработки:
    1. Конвертация PIL → numpy при необходимости.
    2. Удаление лишних измерений (`squeeze`).
    3. Для RGB: использование первого канала или конвертация через палитру.
    4. Валидация: итоговая маска должна быть 2D.

    Args:
        mask_input: Входная маска в формате PIL.Image или numpy array.

    Returns:
        MaskArray: 2D массив формы (H, W), dtype=uint8, значения {0, 255}.

    Raises:
        ValueError: Если после обработки маска не является 2D.

    Example:
        ```python
        from PIL import Image
        import numpy as np

        # RGB маска → 2D binary
        rgb_mask = Image.open("mask_rgb.png")
        binary = prepare_mask_for_overlay(rgb_mask)
        print(binary.shape)  # (512, 512)
        print(np.unique(binary))  # [0, 255]
        ```
    """
    # Конвертация PIL → numpy
    mask: npt.NDArray = np.array(mask_input) if isinstance(mask_input, Image.Image) else np.asarray(mask_input)

    # Обработка многоканальных масок
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask.squeeze(2)
        elif mask.shape[2] == 3:
            logger.warning("⚠️  Обнаружена RGB маска, используется первый канал")
            mask = mask[:, :, 0]
        else:
            raise ValueError(f"Неподдерживаемая форма маски: {mask.shape}")
    elif mask.ndim > 3:
        mask = np.squeeze(mask)

    # Финальная валидация
    if mask.ndim != 2:
        raise ValueError(f"Маска должна быть 2D после обработки, получено {mask.ndim}D")

    return cast(MaskArray, mask)


def _safe_cuda_synchronize() -> bool:
    """Безопасная синхронизация CUDA с обработкой ошибок.
    
    Returns:
        bool: True если синхронизация успешна, False если была ошибка.
    """
    if not torch.cuda.is_available():
        return True
    try:
        torch.cuda.synchronize()
        return True
    except (torch.AcceleratorError, RuntimeError) as e:
        logger.warning(f"⚠️  CUDA synchronize error (пропускаем): {e}")
        torch.cuda.empty_cache()
        return False


# ──────────────────────────────────────────────────────────────────────
def run_performance_benchmark(
    tester: SegmentationTester,
    test_images: TestImagesDict,
    n_runs: int = 10,
    warmup_runs: int = 10,
    output_dir: str = "./data/performance_benchmark",
) -> Optional[pd.DataFrame]:
    """Запуск бенчмарка производительности с сравнением cold/hot запусков.

    Выполняет двухэтапное тестирование:
    1. **Cold benchmark**: Замер времени без предварительного прогрева.
    2. **Warm-up**: Прогрев сегментеров через `SegmentationWarmUp` и `ThresholdWarmUp`.
    3. **Hot benchmark**: Повторный замер после прогрева.
    4. **Анализ**: Расчёт speedup (cold_time / hot_time) для каждого метода.

    Алгоритм:
    ```
    Для каждого изображения:
        1. Запустить benchmark_methods(cold) → df_cold
        2. Сохранить первое изображение для warmup
        3. Выполнить warmup_all_segmenters() + warmup_threshold/edge_methods()
        4. Запустить benchmark_methods(hot) → df_hot
        5. Рассчитать speedup = cold_mean_ms / hot_mean_ms
        6. Сохранить сравнение в CSV
    ```

    Args:
        tester: Экземпляр `SegmentationTester` с зарегистрированными методами.
        test_images: Словарь тестовых изображений `{имя: (путь, PIL.Image, GT)}`.
        n_runs: Количество прогонов для замера времени (по умолчанию 10).
        warmup_runs: Количество прогонов для прогрева (по умолчанию 10).
        output_dir: Директория для сохранения отчётов и сравнений.

    Returns:
        Optional[pd.DataFrame]: Сводный DataFrame с колонками:
                               [method, image, cold_mean_ms, hot_mean_ms, speedup]
                               или `None` при ошибке.

    Note:
        - Warm-up выполняется только один раз на первом изображении.
        - Для методов с `hot_mean_ms == 0` speedup устанавливается в `float('inf')`.
        - Результаты сохраняются в:
          - `{output_dir}/cold_hot_comparison_{img}.csv` — по изображениям
          - `{output_dir}/cold_hot_comparison_summary.csv` — сводный отчёт

    Example:
        ```python
        df = run_performance_benchmark(
            tester=tester,
            test_images=test_images,
            n_runs=10,
            warmup_runs=10
        )
        if df is not None:
            print(f"Средний speedup: {df['speedup'].mean():.2f}x")
            # Топ-5 ускоренных методов
            print(df.nlargest(5, 'speedup')[['method', 'speedup']])
        ```
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n3. Бенчмарк производительности и оценка качества перед warm up...")

    first_img_array: Optional[ImageArray] = None
    all_comparisons: List[pd.DataFrame] = []
    cold_dfs: Dict[str, pd.DataFrame] = {}

    print("\n" + "=" * 60)
    print("БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ: COLD vs HOT")
    print("=" * 60)

    # ──────────────────────────────────────────────────────
    # ЭТАП 1: Cold benchmark (без прогрева)
    # ──────────────────────────────────────────────────────
    print("\n🔹 Этап 1: Cold benchmark...")
    for img_name, (img_path, img_pil, gt_mask) in tqdm(test_images.items(), desc="Cold benchmark"):
        print(f"\n--- Обработка: {img_name} ---")
        img_array: ImageArray = np.array(img_pil)

        if first_img_array is None:
            first_img_array = img_array.copy()

        df_cold: BenchmarkResult = tester.benchmark_methods(
            image=img_array,
            n_runs=n_runs,
            test_name=f"benchmark_{img_name}_cold",
            save_results=True,
            force_warmup=False,
            ground_truth=gt_mask,
        )
        cold_dfs[img_name] = df_cold.copy()
        print(f"   ✅ Cold: {len(df_cold)} методов")
        print(f"   ✅ Бенчмарк для {img_name} завершён")

    # ──────────────────────────────────────────────────────
    # ЭТАП 2: Warm-up сегментеров
    # ──────────────────────────────────────────────────────
    if first_img_array is not None:
        print("\n🔹 Этап 2: Warm-up сегментеров...")

        warmup_utility: SegmentationWarmUp = SegmentationWarmUp(n_warmup_runs=warmup_runs)

        # Прогрев всех сегментеров
        warmup_results: Dict[str, Any] = warmup_utility.warmup_all_segmenters(
            segmenters_dict=tester.methods,
            image=first_img_array,
            verbose=True,
            # exclude_backends=["_ONNX", "_TRT", "_ONNX_TRT_EP"],
            exclude_backends=["_TRT"],
        )
        print(f"   ✅ Warmup all: {len(warmup_results)} методов")
        print(warmup_results)

        # Прогрев пороговых методов
        threshold_warmup: Dict[str, Any] = ThresholdWarmUp.warmup_threshold_methods(
            segmenters_dict=tester.methods,
            image_sizes=[(128, 128), (256, 256)],
        )
        print(f"   ✅ Threshold warmup: {len(threshold_warmup)} методов")

        # Прогрев граничных методов
        edge_warmup: Dict[str, Any] = ThresholdWarmUp.warmup_edge_methods(
            segmenters_dict=tester.methods,
        )
        print(f"   ✅ Edge warmup: {len(edge_warmup)} методов")

        # Сохранение отчёта о прогреве
        report_path: Path = Path(output_dir) / "warmup_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("WARM-UP ОТЧЁТ\n")
            f.write("=" * 60 + "\n\n")
            f.write(warmup_utility.get_warmup_summary())
            f.write("\n\nПороговые методы:\n")
            f.write(str(threshold_warmup))
            f.write("\n\nГраничные методы:\n")
            f.write(str(edge_warmup))
        print(f"   💾 Отчёт: {report_path}")

        # Синхронизация CUDA
        if torch.cuda.is_available():
            _safe_cuda_synchronize()

    if torch.cuda.is_available():
        print("\n🔄 Сброс состояния CUDA после warmup...")
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _safe_cuda_synchronize()
            print("   ✅ CUDA состояние сброшено")
        except Exception as e:
            logger.warning(f"⚠️  Не удалось полностью сбросить CUDA: {e}")

    # ──────────────────────────────────────────────────────
    # ЭТАП 3: Hot benchmark (после прогрева) + анализ
    # ──────────────────────────────────────────────────────
    print("\n🔹 Этап 3: Hot benchmark + анализ...")
    print("\n3.1. Бенчмарк производительности после warm up...")

    for img_name, (_, img_pil, gt_mask) in tqdm(test_images.items(), desc="Hot benchmark"):
        print(f"\n--- Обработка: {img_name} ---")
        img_array = np.array(img_pil)

        try:
            df_hot: BenchmarkResult = tester.benchmark_methods(
                image=img_array,
                n_runs=n_runs,
                test_name=f"benchmark_{img_name}_hot",
                save_results=True,
                force_warmup=False,
                ground_truth=gt_mask,
            )
        except (torch.AcceleratorError, RuntimeError) as e:
            logger.error(f"❌ Ошибка hot-бенчмарка для {img_name}: {e}")
            # 🔧 Попытка восстановления
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                _safe_cuda_synchronize()
            continue

        df_cold_loaded: Optional[pd.DataFrame] = cold_dfs.get(img_name)

        # Fallback: загрузка из файла, если нет в памяти
        if df_cold_loaded is None:
            cold_dir = tester.get_benchmark_path(f"benchmark_{img_name}_cold", "statistics")
            if cold_dir:
                cold_csv = Path(cold_dir) / "benchmark_results.csv"
                if cold_csv.exists():
                    df_cold_loaded = pd.read_csv(cold_csv)
                    print(f"   ✅ Загружен cold-файл из диска: {cold_csv}")
            else:
                logger.warning(f"   ⚠️  CSV не найден в {cold_csv}")

        if df_cold_loaded is None:
            logger.error(f"   ❌ Пропускаем сравнение для {img_name}: нет cold-данных")
            continue

        # Сравнение cold vs hot
        comparison: pd.DataFrame = _compare_cold_hot(
            df_cold=df_cold_loaded,
            df_hot=df_hot,
            image_name=img_name,
        )

        # Сохранение сравнения
        comp_path: Path = Path(output_dir) / f"cold_hot_comparison_{img_name}.csv"
        comparison.to_csv(comp_path, index=False)
        print("\n" + "=" * 70)
        print(f"🔥 COLD vs HOT BENCHMARK COMPARISON ({img_name})")
        print("=" * 70)
        print(
            comparison.sort_values("speedup", ascending=False).to_string(
                float_format=lambda x: f"{x:.2f}" if not np.isinf(x) else "∞"
            )
        )

        # Вывод топ-5 по speedup
        print(f"\n   🔥 Топ-5 по ускорению ({img_name}):")
        top5: pd.DataFrame = comparison.nlargest(5, "speedup")
        for _, row in top5.iterrows():
            speedup_str: str = f"{row['speedup']:.2f}x" if not np.isinf(row["speedup"]) else "∞"
            print(f"      • {row['method']}: {speedup_str}")

        all_comparisons.append(comparison)

    # ──────────────────────────────────────────────────────
    # СВОДНЫЙ ОТЧЁТ
    # ──────────────────────────────────────────────────────
    if all_comparisons:
        summary: pd.DataFrame = pd.concat(all_comparisons, ignore_index=True)
        summary_path: Path = Path(output_dir) / "cold_hot_comparison_summary.csv"
        summary.to_csv(summary_path, index=False)

        print("\n" + "=" * 70)
        print("📊 СВОДНЫЙ ОТЧЁТ: COLD vs HOT BENCHMARK")
        print("=" * 70)

        # Группировка по методам
        avg_speedup: pd.Series = summary.groupby("method")["speedup"].mean().sort_values(ascending=False)

        print("\n🏆 Топ-10 методов по среднему speedup:")
        for i, (method, speedup) in enumerate(avg_speedup.head(10).items(), 1):
            speedup_str = f"{speedup:.2f}x" if not np.isinf(speedup) else "∞"
            print(f"   {i}. {method}: {speedup_str}")

        print(avg_speedup)

        return summary

    return None


# ──────────────────────────────────────────────────────────────────────
def _compare_cold_hot(
    df_cold: pd.DataFrame,
    df_hot: pd.DataFrame,
    image_name: str,
) -> pd.DataFrame:
    """Сравнивает результаты cold и hot бенчмарков.

    Args:
        df_cold: DataFrame с холодными результатами.
        df_hot: DataFrame с горячими результатами.
        image_name: Имя изображения для логирования.

    Returns:
        pd.DataFrame: Сравнение с колонкой speedup.
    """
    comparison: pd.DataFrame = pd.DataFrame(
        {
            "method": df_cold["Method"],
            "image": image_name,
            "cold_mean_ms": df_cold["Mean_Time_s"] * 1000,
            "hot_mean_ms": df_hot["Mean_Time_s"] * 1000,
        }
    )

    # Расчёт speedup с защитой от деления на ноль
    comparison["speedup"] = comparison.apply(
        lambda row: (row["cold_mean_ms"] / row["hot_mean_ms"] if row["hot_mean_ms"] > 0 else float("inf")),
        axis=1,
    )

    return comparison


# ──────────────────────────────────────────────────────────────────────
def run_neural_segmentation_benchmark(
    device: torch.device,
    num_classes: int = NUM_CLASSES_ADE20K,
    output_dir: str = "./data/neural_benchmark",
    repo_id: str = ADE20K_REPO_ID,
) -> Optional[Dict[str, Any]]:
    """Запуск бенчмарка нейросетевых моделей сегментации на датасете ADE20K.

    Выполняет:
    1. **Загрузка данных**: Скачивание изображения и GT из HuggingFace.
    2. **Подготовка**: Конвертация многоклассовой GT в бинарную маску.
    3. **Инициализация**: Создание `SegmentationBenchmark` с 11+ архитектурами.
    4. **Сравнение**: Последовательный запуск моделей с очисткой VRAM.
    5. **Отчётность**: Генерация визуализаций, метрик и отчётов (CSV, JSON, Markdown, LaTeX).

    Поддерживаемые модели:
    - **Transformers**: SegFormer (B2/B5), Mask2Former, OneFormer, MaskFormer
    - **SMP**: U-Net, FPN, PSPNet, DeepLabV3+ (с энкодерами MiT-B5/ResNet)
    - **TorchVision**: FCN, Mask R-CNN, SegNet
    - **SAM**: MobileSAM, SAM2
    - **Другие**: DPT-Large, UPerNet

    Алгоритм:
    ```
    1. Скачать ADE_val_00000001.jpg/png → конвертировать GT в 2D бинарную маску.
    2. Инициализировать SegmentationBenchmark(device, num_classes, gt_mask).
    3. Загрузить MaskFormer вручную → сохранить результаты → очистить память.
    4. Загрузить остальные модели через _load_benchmark_models().
    5. benchmark.compare(image_input=original_img).
    6. Сохранить оверлеи, метрики, графики, LaTeX-таблицу.
    7. Вернуть Dict{summary, results_map, benchmark, latex_table}.
    ```

    Args:
        device: Устройство для выполнения (`cuda`/`cpu`).
        num_classes: Количество классов (по умолчанию 150 для ADE20K).
        output_dir: Директория для сохранения результатов.
        repo_id: ID репозитория с тестовыми данными (по умолчанию ADE20K fixtures).

    Returns:
        Optional[Dict[str, Any]]: Словарь с результатами:
                                  - `summary`: DataFrame с метриками по моделям
                                  - `results_map`: Словарь `{model_name: overlay_image}`
                                  - `benchmark`: Экземпляр `SegmentationBenchmark`
                                  или `None` при ошибке.

    Note:
        - Требуется ~15–20 ГБ VRAM для одновременной загрузки всех моделей.
        - Автоматически освобождает память через `torch.cuda.empty_cache()` после каждой модели.
        - Генерирует до 18 графиков сравнения и полную таблицу для статей (LaTeX).
        - Ground truth конвертируется в бинарную маску через `_convert_multiclass_to_binary()`.

    Example:
        ```python
        results = run_neural_segmentation_benchmark(
            device=torch.device("cuda"),
            num_classes=150,
            output_dir="./results/neural"
        )
        if results:
            print(f"Лучшая модель по mIoU: {results['summary'].nlargest(1, 'mIoU')['model'].iloc[0]}")
            # Сохранение визуализаций
            for name, overlay in results['results_map'].items():
                if overlay:
                    overlay.save(f"./results/{name}.jpg")
        ```
    """
    from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("НЕЙРОСЕТЕВОЙ БЕНЧМАРК: ADE20K")
    print("=" * 60)

    # ──────────────────────────────────────────────────────
    # 1. ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ
    # ──────────────────────────────────────────────────────
    print("\n🔹 Загрузка данных ADE20K...")

    image_path: str = hf_hub_download(repo_id=repo_id, filename="ADE_val_00000001.jpg", repo_type="dataset")
    original_img: Image.Image = Image.open(image_path)
    original_img.save(Path(output_dir) / "original_image.jpg")

    mask_path: str = hf_hub_download(repo_id=repo_id, filename="ADE_val_00000001.png", repo_type="dataset")
    segmentation_map: Image.Image = Image.open(mask_path)

    # Конвертация GT в бинарную маску
    mask_array: npt.NDArray = np.array(segmentation_map)
    print(f"Mask shape: {mask_array.shape}")
    print(f"Mask dtype: {mask_array.dtype}")
    print(f"Unique values: {np.unique(mask_array)[:20]}")

    gt_mask_2d: MaskArray = prepare_mask_for_overlay(segmentation_map)
    segmentation_map.save(Path(output_dir) / "ground_truth.png")

    print(f"✅ Изображение: {original_img.size}, GT: {gt_mask_2d.shape}")

    # ──────────────────────────────────────────────────────
    # 2. ИНИЦИАЛИЗАЦИЯ БЕНЧМАРКА
    # ──────────────────────────────────────────────────────
    print("\n🔹 Инициализация моделей...")

    infer_res_ade: Image.Image = _create_overlay_standalone(
        original_img,
        gt_mask_2d,
        alpha=0.5,
        palette=NeuralSegmenter.ade_palette(),
    )

    print("🔹 Запуск MaskFormer (Изолированный режим)...")

    model_name_maskformer: str = "facebook/maskformer-resnet50-ade20k-full"
    processor_maskformer = MaskFormerImageProcessor.from_pretrained(model_name_maskformer)
    model_maskformer = MaskFormerForInstanceSegmentation.from_pretrained(model_name_maskformer)
    model_maskformer.to(device)  # type: ignore[arg-type]
    model_maskformer.eval()
    class_names_val: Dict[int, str] = NeuralSegmenter.get_ade_class_names()
    result_mf_ade, result_mf_ade_results = segment_image_unified(
        model_maskformer,
        processor_maskformer,
        original_img,
        "maskformer",
        alpha=0.6,
        palette=NeuralSegmenter.ade_palette,
        num_classes=num_classes,
        class_names=class_names_val,
        gt_mask=gt_mask_2d,
        output_dir=output_dir
    )
    result_mf_ade.save("./data/neural_benchmark/segmented_maskformer_ade.jpg")

    maskformer_manual_ade_result: Dict[str, Any] = {
        "model": "maskformer",
        "overlay": result_mf_ade,
        "mask": result_mf_ade_results.get("mask"),
        "inference_time_ms": result_mf_ade_results.get("inference_time_ms", 0),
        "metrics": result_mf_ade_results.get("metrics", {}),
        "image_size": original_img.size[::-1],
        "output_shape": result_mf_ade_results.get("mask", np.array([])).shape,
        "unique_classes": len(np.unique(result_mf_ade_results.get("mask", np.array([])))),
    }

    del model_maskformer, processor_maskformer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    print(f"✅ MaskFormer готов. VRAM освобождена: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    class_names_dict: Dict[int, str] = NeuralSegmenter.get_ade_class_names()
    class_names: Optional[List[str]] = [class_names_dict.get(i, f"Class {i}") for i in range(num_classes)]
    benchmark: SegmentationBenchmark = SegmentationBenchmark(
        device=device.type,
        num_classes=num_classes,
        class_names=class_names,
        gt_mask=gt_mask_2d,
        palette=NeuralSegmenter.ade_palette,
    )

    # Загрузка моделей (с обработкой ошибок)
    _load_benchmark_models(benchmark, output_dir)

    # ──────────────────────────────────────────────────────
    # 3. ЗАПУСК СРАВНЕНИЯ
    # ──────────────────────────────────────────────────────
    print("\n🔹 Запуск сравнения моделей...")
    print("\n🚀 Running benchmark (this may take 10-15 minutes)...")

    benchmark.compare(image_input=original_img, alpha=0.6)
    benchmark.results["maskformer"] = maskformer_manual_ade_result

    # ──────────────────────────────────────────────────────
    # 4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ──────────────────────────────────────────────────────
    print("\n🔹 Сохранение результатов...")

    # Карта результатов для визуализации
    results_map: Dict[str, Optional[Image.Image]] = {
        model_key: benchmark.results[model_key].get("overlay")
        for model_key in benchmark.results
        if "overlay" in benchmark.results[model_key]
    }

    # Сохранение оверлеев
    for model_key, overlay in results_map.items():
        if overlay is not None:
            overlay_path: Path = Path(output_dir) / f"segmented_{model_key}.jpg"
            overlay.save(overlay_path)
            print(f"✅ Сохранено: {model_key}: {overlay_path}")

    # Метрики и сводка
    print("\n🔍 Checking metrics...")
    summary_ade: Dict[str, Dict[str, Any]] = benchmark.get_summary()
    for metric in ["mIoU", "pixel_acc", "time_ms"]:
        values: List = [summary_ade[m].get(metric, np.nan) for m in summary_ade]
        valid: int = sum(1 for v in values if not np.isnan(v))
        print(f"   {metric}: {valid} models")
    summary: pd.DataFrame = pd.DataFrame(summary_ade).T.sort_values("mIoU", ascending=False)

    for model_name, res in benchmark.results.items():
        print(f"\n{model_name}:")
        print(f"  mIoU: {res['metrics'].get('mIoU', 'N/A'):.4f}")
        print(f"  Time: {res['inference_time_ms']:.1f} ms")
        print(f"  Classes: {res['unique_classes']}")

    # Сохранение отчётов
    print("\n💾 Saving results...")
    benchmark.save_results(str(Path(output_dir) / "ade_benchmark"))

    # Генерация визуализаций
    _generate_neural_benchmark_plots(benchmark, output_dir)

    # LaTeX таблица для публикации
    print("\n" + "=" * 70)
    print("LATEX TABLE FOR PAPER")
    print("=" * 70)
    latex_table: str = benchmark.export_latex_table(caption="Comprehensive Semantic Segmentation Benchmark on ADE20K")
    with open(Path(output_dir) / "benchmark_table.tex", "w") as f:
        f.write(latex_table)
    print(latex_table)

    if "segformer" in summary.index and "segformer_b2" in summary.index:
        print("\n" + "=" * 70)
        print("SEGFORMER SPEED-ACCURACY TRADEOFF")
        print("=" * 70)
        sf: pd.DataFrame = summary.loc[["segformer", "segformer_b2"], ["mIoU", "time_ms"]]
        sf["mIoU"] = sf["mIoU"] * 100
        print(sf.to_string(float_format="%.2f"))
        print(
            f"\n💡 B2 is {summary.loc['segformer', 'time_ms'] / summary.loc['segformer_b2', 'time_ms']:.1f}x faster with {summary.loc['segformer', 'mIoU'] - summary.loc['segformer_b2', 'mIoU']:.1f} pp mIoU drop"
        )

    print("\n" + "=" * 70)
    print("✅ BENCHMARK COMPLETE — Results saved to 'ade20k_11models_comprehensive/'")
    print("=" * 70)

    # Сводная таблица результатов
    table: pd.DataFrame = export_comparison_table(benchmark, str(Path(output_dir) / "model_comparison.md"))
    print(table)

    plt.figure(figsize=(20, 10))
    titles: List[str] = [
        "Original",
        "SegFormer",
        "Mask2Former",
        "OneFormer",
        "U_Net",
        "DeepLabV3+",
        "MobileSAM",
        "SAM2",
        "DPT-Large",
        "UPerNet",
        "SegFormer-B2",
        "FPN + MiT-B5",
        "PSPNet + MiT-B5",
        "MaskFormer",
        "FCN ResNet-50",
        "Mask R-CNN",
        "SegNet",
        "Ground_Truth",
        "Orig_Mask",
    ]

    images: List = [
        original_img,  # Original
        results_map["segformer"],  # SegFormer
        results_map["mask2former"],  # Mask2Former
        results_map["oneformer"],  # OneFormer
        results_map["unet_pretrained"],  # U_Net
        results_map["deeplab_pretrained"],  # DeepLabV3+
        results_map["sam"],  # MobileSAM
        results_map["sam2"],  # SAM2
        results_map["dpt"],  # DPT-Large
        results_map["upernet"],  # UPerNet
        results_map["segformer_b2"],  # SegFormer-B2
        results_map["fpn_mit_b5_pretrained"],  # FPN + MiT-B5
        results_map["psp_mit_b5_pretrained"],  # PSPNet + MiT-B5
        results_map["maskformer"],  # MaskFormer
        results_map["fcn_resnet50_pretrained"],  # FCN ResNet-50
        results_map["maskrcnn_pretrained"],  # Mask R-CNN
        results_map["segnet_resnet34_pretrained"],  # SegNet
        infer_res_ade,  # Ground_Truth
        segmentation_map,  # Orig_Mask
    ]

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(3, 7, i + 1)
        if img is not None:
            plt.imshow(img)
        else:
            plt.text(0.5, 0.5, "N/A", ha="center", va="center")
        plt.title(title, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(
        "./data/neural_benchmark/segmentation_results_ade.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        format="png",
    )
    plt.show()

    print(f"\n✅ Результаты сохранены в: {output_dir}")

    return {
        "summary": summary,
        "results_map": results_map,
        "benchmark": benchmark,
        "latex_table": latex_table,
    }


# ──────────────────────────────────────────────────────────────────────
def _load_benchmark_models(benchmark: SegmentationBenchmark, output_dir: str) -> None:
    """Загружает предобученные модели в бенчмарк с обработкой ошибок и очисткой VRAM.

    Выполняет:
    1. **Диагностика CUDA**: Вывод статистики памяти и настроек cuDNN.
    2. **Очистка памяти**: `empty_cache()`, `synchronize()`, `gc.collect()`.
    3. **Последовательная загрузка**: Итерация по списку моделей с `getattr`-вызовом.
    4. **Обработка ошибок**: Логирование сбоев без прерывания загрузки остальных.
    5. **Пост-очистка**: Освобождение VRAM после каждой модели.

    Алгоритм:
    ```
    1. Вывести VRAM Allocated/Reserved/Max и флаги deterministic/cuDNN.
    2. Очистить память: torch.cuda.empty_cache(), synchronize(), gc.collect().
    3. Для каждой модели в models_to_load:
       a. Получить method = getattr(benchmark, config["method"]).
       b. Вызвать method(*args) или method(**kwargs).
       c. При успехе → print("✅"), при ошибке → print("⚠️") + traceback при DEBUG.
       d. Очистить VRAM после итерации.
    ```

    Args:
        benchmark: Экземпляр `SegmentationBenchmark` для регистрации моделей.
        output_dir: Директория для логирования (используется для отладки).

    Returns:
        None. Модифицирует `benchmark` in-place.

    Note:
        - Модели загружаются последовательно для минимизации пикового потребления VRAM.
        - Поддерживаются как предобученные (HuggingFace), так и локальные чекпоинты.
        - При `DEBUG=1` выводится полный traceback для упавших загрузок.

    Example:
        ```python
        benchmark = SegmentationBenchmark(device="cuda", num_classes=150, gt_mask=gt)
        _load_benchmark_models(benchmark, output_dir="./logs")
        print(f"Загружено моделей: {len(benchmark.models)}")
        ```
    """
    models_to_load: List[Dict[str, Any]] = [
        {
            "method": "load_segformer",
            "args": ["/home/yamshchikov/models/segformer-b5-ready"],
        },
        {
            "method": "load_mask2former",
            "args": ["facebook/mask2former-swin-base-ade-semantic"],
        },
        {"method": "load_oneformer", "args": ["shi-labs/oneformer_ade20k_swin_large"]},
        {
            "method": "load_unet_trained",
            "kwargs": {"checkpoint_path": "models/unet_ade20k_best_200_epochs.pth"},
        },
        {
            "method": "load_deeplab_trained",
            "kwargs": {"checkpoint_path": "models/deeplab_ade20k_best_200_epochs.pth"},
        },
        {"method": "load_sam", "args": ["models/mobile_sam.pt"]},
        {"method": "load_sam", "args": ["models/sam2_t.pt"]},
        {"method": "load_yolov8", "args": ["models/yolov8m-seg.pt"]},
        {"method": "load_yolov8", "args": ["models/yolov8n-seg.pt"]},
        {"method": "load_yolov8", "args": ["models/yolov8s-seg.pt"]},
        {"method": "load_dpt", "args": ["Intel/dpt-large-ade"]},
        {"method": "load_upernet", "args": ["openmmlab/upernet-convnext-small"]},
        {"method": "load_segformer_variant", "args": ["b2"]},
        {
            "method": "load_mask_rcnn_pretrained",
            "kwargs": {"variant": "maskrcnn_resnet50_fpn"},
        },
        {
            "method": "load_fpn_mit_pretrained",
            "kwargs": {
                "variant": "b5",
                "checkpoint_path": "models/fpn_mit_b5_ade20k_best_200_epochs.pth",
            },
        },
        {
            "method": "load_psp_mit_pretrained",
            "kwargs": {
                "variant": "b5",
                "checkpoint_path": "models/psp_mit_b5_ade20k_best_200_epochs.pth",
            },
        },
        {
            "method": "load_fcn_resnet50_pretrained",
            "kwargs": {
                "variant": "fcn_resnet50",
                "checkpoint_path": "models/fcn_resnet50_ade20k_best_200_epochs.pth",
            },
        },
        {
            "method": "load_segnet_pretrained",
            "kwargs": {
                "encoder_name": "resnet34",
                "checkpoint_path": "models/segnet_ade20k_best_200_epochs.pth",
            },
        },
    ]

    print("=" * 50)
    print("CUDA DIAGNOSTICS")
    print("=" * 50)
    print(f"VRAM Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"VRAM Reserved:  {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    print(f"VRAM Max:       {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    print(f"Deterministic:  {torch.are_deterministic_algorithms_enabled()}")
    print(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    print("=" * 50)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    print(f"VRAM After Clean: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    for model_config in models_to_load:
        try:
            method_name: str = model_config["method"]
            method = getattr(benchmark, method_name)

            if "args" in model_config:
                method(*model_config["args"])
            elif "kwargs" in model_config:
                method(**model_config["kwargs"])
            else:
                method()

            print(f"   ✅ {method_name}")

        except Exception as e:
            logger.warning(f"   ⚠️  {model_config.get('method', 'unknown')}: {e}")
            if os.getenv("DEBUG", "0") == "1":
                traceback.print_exc()

        # Очистка памяти после каждой модели
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ──────────────────────────────────────────────────────────────────────
def _generate_neural_benchmark_plots(benchmark: SegmentationBenchmark, output_dir: str) -> None:
    """Генерирует визуализации для нейросетевого бенчмарка.

    Выполняет:
    1. **Подготовка директории**: Создание `plots/` внутри output_dir.
    2. **Генерация графиков**:
       - All metrics (сводный бар-чарт)
       - mIoU comparison (ранжирование по точности)
       - Time comparison (ранжирование по скорости)
       - Per-class IoU (топ-20 классов)
       - Confusion matrix (для лучшей модели)
       - Summary (мульти-метрика)
    3. **Сохранение**: Экспорт в PNG с DPI=150 и `bbox_inches="tight"`.

    Алгоритм:
    ```
    1. Создать plots_dir = Path(output_dir) / "plots".
    2. Для каждого типа графика:
       a. Вызвать соответствующий метод benchmark.plot_*.
       b. Сохранить через plt.savefig(..., dpi=150, bbox_inches="tight").
       c. Закрыть фигуру через plt.close().
    3. Вывести путь к сохранённым графикам.
    ```

    Args:
        benchmark: Экземпляр `SegmentationBenchmark` с результатами.
        output_dir: Базовая директория для сохранения визуализаций.

    Returns:
        None. Графики сохраняются в `{output_dir}/plots/`.

    Note:
        - Все графики закрываются после сохранения для освобождения памяти.
        - `bbox_inches="tight"` обрезает лишние поля для компактности.
        - Конфьюжн-матрица строится для модели с максимальным mIoU.

    Example:
        ```python
        _generate_neural_benchmark_plots(benchmark, output_dir="./results/neural")
        # Сохранит: all_metrics.png, miou_comparison.png, time_comparison.png, ...
        ```
    """
    plots_dir: Path = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 🔧 Проверка: есть ли данные для отрисовки?
    summary = benchmark.get_summary()
    if not summary:
        logger.warning("⚠️ Нет данных для генерации графиков. Пропускаем визуализацию.")
        return

    print("\n🔍 Debug: Available metrics for plotting:")
    for model, metrics in summary.items():
        print(f"   {model}: mIoU={metrics.get('mIoU', 'N/A')}, time={metrics.get('time_ms', 'N/A')}ms")
    
    valid_models = [k for k, v in summary.items() if not np.isnan(v.get("mIoU", np.nan))]
    if not valid_models:
        logger.warning("⚠️ Все метрики mIoU = NaN. Пропускаем визуализацию.")
        return
    
    print("\n📊 Generating visualizations...")
    
    # ──────────────────────────────────────────────────────────────
    # 1. Все метрики (с проверкой)
    # ──────────────────────────────────────────────────────────────
    try:
        fig = plt.figure(figsize=(15, 5))
        benchmark.plot_all_metrics(figsize=(15, 5))
        # 🔧 Явно сохраняем текущую фигуру
        plt.savefig(plots_dir / "all_metrics.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"✅ all_metrics.png")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении all_metrics.png: {e}")
    
    # ──────────────────────────────────────────────────────────────
    # 2. Сравнение по mIoU
    # ──────────────────────────────────────────────────────────────
    try:
        fig = plt.figure(figsize=(12, 6))
        benchmark.plot_comparison_chart("mIoU", title="ADE20K: Mean IoU Comparison", figsize=(12, 6))
        plt.savefig(plots_dir / "miou_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"✅ miou_comparison.png")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении miou_comparison.png: {e}")
    
    # ──────────────────────────────────────────────────────────────
    # 3. Сравнение по времени
    # ──────────────────────────────────────────────────────────────
    try:
        fig = plt.figure(figsize=(12, 6))
        benchmark.plot_comparison_chart("time_ms", title="Inference Time Comparison (ms)", figsize=(12, 6))
        plt.savefig(plots_dir / "time_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"✅ time_comparison.png")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении time_comparison.png: {e}")
    
    # ──────────────────────────────────────────────────────────────
    # 4. Per-class IoU (топ-20 классов)
    # ──────────────────────────────────────────────────────────────
    try:
        fig = plt.figure(figsize=(14, 8))
        benchmark.plot_per_class_iou(top_k=20)
        plt.savefig(plots_dir / "per_class_iou.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"✅ per_class_iou.png")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении per_class_iou.png: {e}")
    
    # ──────────────────────────────────────────────────────────────
    # 5. Confusion matrix для лучшей модели
    # ──────────────────────────────────────────────────────────────
    try:
        summary_df = benchmark.get_summary_dataframe()
        if not summary_df.empty and "mIoU" in summary_df.columns:
            best_model: str = summary_df.sort_values("mIoU", ascending=False).index[0]
            fig = plt.figure(figsize=(10, 8))
            benchmark.plot_confusion_matrix(best_model, normalize="true")
            plt.savefig(plots_dir / f"confusion_matrix_{best_model}.png", dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"✅ confusion_matrix_{best_model}.png")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении confusion matrix: {e}")
    
    # ──────────────────────────────────────────────────────────────
    # 6. Сводный график
    # ──────────────────────────────────────────────────────────────
    try:
        fig = plt.figure(figsize=(15, 5))
        benchmark.plot_summary(metrics=["mIoU", "pixel_acc", "time_ms"])
        plt.savefig(plots_dir / "summary.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"✅ summary.png")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении summary.png: {e}")
    
    print(f"📊 Графики сохранены в: {plots_dir}")


# ──────────────────────────────────────────────────────────────────────
def run_implementation_validation(
    test_images: TestImagesDict,
    output_dir: str = "./data/validation",
    image_name: str = "mountain",
    include_backends: bool = False,
    onnx_dir: Optional[str] = None,
    trt_dir: Optional[str] = None,
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
) -> Optional[Dict[str, Any]]:
    """Валидация согласованности реализаций методов через TorchImplementationValidator (Torch vs OpenCV/Sklearn).

    Выполняет:
    1. **Инициализация**: Создание `TorchImplementationValidator`.
    2. **Валидация**: Запуск `validate_all_methods()` на выбранном изображении.
    3. **Отчётность**: Генерация Markdown-отчёта со статусами PASS/WARNING/FAIL.
    4. **Бенчмарк-сводка**: Опциональное создание отчёта на основе результатов валидации.

    Алгоритм:
    ```
    1. Проверить наличие image_name в test_images.
    2. Инициализировать TorchImplementationValidator(output_dir).
    3. Вызвать validate_all_methods(image_path, use_torch2=True).
    4. Сгенерировать validation_report.md.
    5. Опционально: generate_benchmark_report_from_validation().
    6. Вывести статистику по статусам и топ-5 по IoU.
    7. Вернуть Dict{all_results, benchmark_df, validator}.
    ```

    Аргументы валидации:
    - Сравниваются реализации одного метода в Torch vs OpenCV/Sklearn.
    - Метрики: IoU, Dice, Precision, Recall, F1, MAE, Hausdorff.
    - Статусы: PASS (≥80% метрик в пороге), WARNING (50–80%), FAIL (<50%).

    Args:
        test_images: Словарь тестовых изображений `{имя: (путь, PIL.Image, GT)}`.
        output_dir: Директория для сохранения отчётов валидации.
        image_name: Ключ изображения из `test_images` для валидации.

    Returns:
        Optional[Dict[str, Any]]: Словарь с результатами:
                                  - `all_results`: Результаты валидации по методам
                                  - `benchmark_df`: DataFrame бенчмарк-отчёта (если сгенерирован)
                                  - `validator`: Экземпляр `TorchImplementationValidator`
                                  или `None` при ошибке.

    Note:
        - Сравнивает Torch-реализации с CPU-бэкендами по IoU, Dice, Precision, Recall, F1.
        - Статусы: ✅ PASS (≥80%), ⚠️ WARNING (50–80%), ❌ FAIL (<50%).
        - Ошибки логируются, но не прерывают валидацию остальных методов.
        - Результаты включают:
          - `{output_dir}/validation_report.md` — Markdown-отчёт
          - `{output_dir}/charts/` — Графики сравнения метрик
          - `{output_dir}/masks/` — Примеры масок для визуальной проверки

    Example:
        ```python
        results = run_implementation_validation(
            test_images=test_images,
            output_dir="./results/validation",
            image_name="mountain"
        )
        if results:
            print(f"Валидировано методов: {len(results['all_results'])}")
            # Статистика по статусам
            statuses = [r.get('status') for r in results['all_results'].values()]
            from collections import Counter
            print(f"Статусы: {Counter(statuses)}")
        ```
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("ВАЛИДАЦИЯ РЕАЛИЗАЦИЙ: Torch vs OpenCV/Sklearn")
    print("=" * 60)

    # Проверка наличия изображения
    if image_name not in test_images:
        logger.error(f"❌ Изображение '{image_name}' не найдено в test_images")
        print(f"   Доступные: {list(test_images.keys())}")
        return None

    _, img_pil, _ = test_images[image_name]
    img_array: ImageArray = np.array(img_pil)

    print(f"\n🔹 Валидация на изображении: {image_name} ({img_array.shape})")
    if include_backends:
        print(f"🔹 Включена валидация с бэкендами (ONNX/TRT)")
        print(f"   ONNX dir: {onnx_dir or './exported_models/onnx/fp32'}")
        print(f"   TRT dir:  {trt_dir or './exported_models/tensorrt/fp32'}")

    # ──────────────────────────────────────────────────────
    # 1. ИНИЦИАЛИЗАЦИЯ ВАЛИДАТОРА
    # ──────────────────────────────────────────────────────
    validator: TorchImplementationValidator = TorchImplementationValidator(output_dir=output_dir)

    # ──────────────────────────────────────────────────────
    # 2. ЗАПУСК ВАЛИДАЦИИ
    # ──────────────────────────────────────────────────────
    print("\n🔹 Запуск валидации методов...")

    try:
        # test_images['ade20k_sample'][0]
        # test_images['countryside'][0]
        # all_results = validator.validate_all_methods(test_images["mountain"][0])
        if include_backends:
            # 🔥 Расширенная валидация с поддержкой бэкендов
            all_results: Dict[str, Any] = validator.validate_all_methods_with_backends(
                image_path=img_array,
                use_torch2=True,
                torch2_precision="bf16",
                validate_onnx=True,
                validate_trt=torch.cuda.is_available(),
                onnx_dir=onnx_dir,
                trt_dir=trt_dir,
                input_shape=input_shape,
            )
        else:
            # Базовая валидация (Torch vs CPU-бэкенды)
            all_results = validator.validate_all_methods(image_path=img_array, use_torch2=True, torch2_precision="bf16")
        print(f"   ✅ Валидировано: {len(all_results)} методов")
    except Exception as e:
        logger.error(f"❌ Ошибка валидации: {e}")
        traceback.print_exc()
        return None

    # ──────────────────────────────────────────────────────
    # 3. ГЕНЕРАЦИЯ ОТЧЁТА
    # ──────────────────────────────────────────────────────
    print("\n🔹 Генерация отчёта...")

    try:
        report_path = validator.generate_validation_report(all_results)
        print(f"   ✅ Отчёт сохранён: {report_path}")
        print(f"\n✅ Все результаты сохранены в: {validator.output_dir}")
    except Exception as e:
        logger.error(f"⚠️  Ошибка генерации отчёта: {e}")

    # ──────────────────────────────────────────────────────
    # 4. БЕНЧМАРК-ОТЧЁТ (опционально)
    # ──────────────────────────────────────────────────────
    benchmark_df: Optional[pd.DataFrame] = None
    try:
        benchmark_dir: str = os.path.join(output_dir, "benchmark")
        benchmark_df = validator.generate_benchmark_report_from_validation(all_results, output_dir=benchmark_dir)
        print(f"   ✅ Бенчмарк-отчёт: {benchmark_dir}")
    except Exception as e:
        logger.error(f"⚠️  Ошибка генерации бенчмарка: {e}")

    # ──────────────────────────────────────────────────────
    # 5. СВОДКА
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 СВОДКА ПО ВАЛИДАЦИИ")
    print("=" * 70)

    if all_results:
        # Статистика по статусам (агрегируем по всем конфигурациям)
        statuses: List[str] = []
        for config_results in all_results.values():
            if isinstance(config_results, dict):
                for result in config_results.values():
                    if isinstance(result, dict) and "validation_status" in result:
                        statuses.append(result["validation_status"])

        if statuses:
            from collections import Counter

            status_counts: Counter = Counter(statuses)

            print("\n📈 Распределение статусов:")
            for status, count in status_counts.most_common():
                emoji = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}.get(status, "❓")
                print(f"   {emoji} {status}: {count} методов ({count / len(statuses) * 100:.1f}%)")

            # Топ-5 по согласованности (IoU) — агрегируем по всем конфигурациям
            all_iou_results: List[Tuple[str, float, str]] = []
            for config_name, config_results in all_results.items():
                if isinstance(config_results, dict):
                    for method_name, result in config_results.items():
                        if isinstance(result, dict) and result.get("success") and "metrics" in result:
                            iou = result["metrics"].get("iou", 0)
                            all_iou_results.append((f"{config_name}/{method_name}", iou, result["validation_status"]))

            if all_iou_results:
                sorted_results = sorted(all_iou_results, key=lambda x: x[1], reverse=True)[:5]
                print("\n🏆 Топ-5 по IoU (согласованность):")
                for i, (full_name, iou, status) in enumerate(sorted_results, 1):
                    status_icon = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}.get(status, "❓")
                    print(f"   {i}. {full_name}: IoU = {iou:.4f} {status_icon}")

    print(f"\n💾✅ Все результаты сохранены в: {output_dir}")

    return {
        "all_results": all_results,
        "benchmark_df": benchmark_df,
        "validator": validator,
    }


# ──────────────────────────────────────────────────────────────────────
def run_matrix_comparison(
    test_images: TestImagesDict,
    cv2_methods: SegmenterDict,
    sklearn_methods: SegmenterDict,
    torch_methods: SegmenterDict,
    output_dir: str = "./data/matrix_comparison",
    reference_method: str = "global_thresholding_Sklearn",
    include_backends: bool = False,
    tester: Optional[SegmentationTester] = None
) -> Optional[Dict[str, Any]]:
    """Матричное и пакетное сравнение методов сегментации.

    Выполняет:
    1. **All-vs-All**: Полная матрица попарных сравнений всех методов.
    2. **Batch vs Reference**: Сравнение всех методов относительно одного референса.
    3. **Pairwise**: Детальное сравнение двух конкретных реализаций (CV2 vs Sklearn).
    4. **Агрегация**: Сбор результатов по изображениям и вывод сводки.

    Поддерживаемые типы сравнений:
    - `all_vs_all`: Полная матрица (метод × метод) с метриками согласованности.
    - `batch_vs_reference`: Все методы против одного референсного (по умолчанию Otsu Sklearn).
    - `pairwise`: Сравнение двух конкретных реализаций (например, CV2 vs Sklearn Otsu).

    Алгоритм:
    ```
    1. Объединить методы в all_segmenters.
    2. Инициализировать SegmentationComparator.
    3. Для каждого изображения:
       a. matrix_comparison(type="all_vs_all") → сохранить пары.
       b. batch_comparison(reference=ref_segmenter) → сохранить DF.
       c. compare_methods(segmenter1, segmenter2) → сохранить метрики.
    4. Агрегировать batch_results, посчитать mean similarity.
    5. Вернуть Dict{all_vs_all_results, batch_results, pairwise_results}.
    ```

    Args:
        test_images: Словарь тестовых изображений `{имя: (путь, PIL.Image, GT)}`.
        cv2_methods: Словарь методов OpenCV.
        sklearn_methods: Словарь методов scikit-learn.
        torch_methods: Словарь методов PyTorch.
        output_dir: Базовая директория для сохранения результатов.
        reference_method: Ключ референсного метода для batch-сравнения.
        include_backends: Если True, включает методы с бэкендами из tester.
        tester: Экземпляр SegmentationTester с зарегистрированными бэкенд-методами.

    Returns:
        Optional[Dict[str, Any]]: Словарь с результатами:
                                  - `all_vs_all_results`: Результаты матричного сравнения
                                  - `batch_results`: DataFrame batch-сравнения
                                  - `pairwise_results`: Результаты попарных сравнений
                                  или `None` при ошибке.

    Note:
        - Референсный метод должен присутствовать в объединённом словаре методов.
        - All-vs-All ресурсоёмок для >20 методов (O(n²) сравнений).
        - Результаты сохраняются в:
          - `{output_dir}/matrix_comparison_{img}/` — All-vs-All
          - `{output_dir}/batch_comparison/` — Batch vs Reference
          - `{output_dir}/compare_methods_{img}/` — Pairwise
        - Для больших наборов методов матричное сравнение может быть ресурсоёмким.

    Example:
        ```python
        results = run_matrix_comparison(
            test_images=test_images,
            cv2_methods=cv2_methods,
            sklearn_methods=sklearn_methods,
            torch_methods=torch_methods,
            reference_method="Otsu_Thresholding_Sklearn"
        )
        if results and results['batch_results'] is not None:
            # Топ-3 метода относительно референса
            top3 = results['batch_results'].nlargest(3, 'similarity_score')
            print(top3[['method', 'similarity_score']])
        ```
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("МАТРИЧНОЕ СРАВНЕНИЕ МЕТОДОВ")
    print("=" * 60)

    # Объединение методов
    all_segmenters: SegmenterDict = {**cv2_methods, **sklearn_methods, **torch_methods}

    if include_backends and tester is not None:
        print("\n🔹 Добавление бэкенд-методов из tester...")
        backend_count = 0
        for name, segmenter in tester.methods.items():
            # Фильтруем только бэкенд-методы (содержат ONNX или TRT)
            if "ONNX" in name or "TRT" in name:
                # Парсим имя для извлечения информации
                parsed = parse_method_name(name)
                base_method = parsed["BaseMethod"]
                backend = parsed["Backend"]
                precision = parsed["Precision"]
                
                # Проверяем что базовый метод есть в основных методах
                if any(base_method in m for m in all_segmenters.keys()):
                    all_segmenters[name] = segmenter
                    backend_count += 1
                    print(f"   ✅ {name} ({backend}/{precision})")
        
        print(f"   📦 Всего бэкенд-методов добавлено: {backend_count}")

    # Подготовка конфигурации для сравнения
    methods_config_list: List[Dict[str, Any]] = []
    for name, segmenter in all_segmenters.items():
        # Парсим имя метода для извлечения метаданных
        parsed = parse_method_name(name)
        methods_config_list.append({
            "name": name,
            "segmenter": segmenter,
            "base_method": parsed["BaseMethod"],
            "backend": parsed["Backend"],
            "precision": parsed["Precision"],
        })

    # Референсный сегментер
    if reference_method not in all_segmenters:
        ref_candidates = [k for k in all_segmenters.keys() if k.startswith(reference_method)]
        if ref_candidates:
            reference_method = ref_candidates[0]
            logger.info(f"🔍 Референсный метод найден: {reference_method}")
        else:
            logger.error(f"❌ Референсный метод '{reference_method}' не найден")
            print(f"   Доступные: {list(all_segmenters.keys())[:10]}...")
            return None

    ref_segmenter = all_segmenters[reference_method]

    # Для pairwise: поиск оригинального сегментера
    original_segmenter = None
    for lib_methods in [cv2_methods, sklearn_methods, torch_methods]:
        if reference_method in lib_methods:
            original_segmenter = lib_methods[reference_method]
            break

    if original_segmenter is None:
        # Пробуем найти любую версию референсного метода
        for name, seg in all_segmenters.items():
            parsed = parse_method_name(name)
            if parsed["BaseMethod"] == reference_method or name == reference_method:
                original_segmenter = seg
                break

    if original_segmenter is None:
        logger.warning(f"⚠️  Не найден оригинальный сегментер для {reference_method}")
        original_segmenter = ref_segmenter  # fallback

    # ──────────────────────────────────────────────────────
    # ИНИЦИАЛИЗАЦИЯ КОМПАРАТОРА
    # ──────────────────────────────────────────────────────
    comparator: SegmentationComparator = SegmentationComparator()

    all_vs_all_results: List[Dict[str, Any]] = []
    batch_results_list: List[pd.DataFrame] = []
    pairwise_results_list: List[pd.DataFrame] = []

    # ──────────────────────────────────────────────────────
    # ЦИКЛ ПО ИЗОБРАЖЕНИЯМ
    # ──────────────────────────────────────────────────────
    for img_name, (_, img_pil, _) in tqdm(test_images.items(), desc="Matrix Comparison benchmark"):
        print(f"\n🔹 Обработка: {img_name}")
        img_array: ImageArray = np.array(img_pil)

        img_output_dir: Path = Path(output_dir) / f"matrix_comparison_{img_name}"

        # ──────────────────────────────────────────────────
        # 1. ALL-VS-ALL МАТРИЧНОЕ СРАВНЕНИЕ
        # ──────────────────────────────────────────────────
        try:
            print("   🔸 All-vs-All матрица...")
            matrix_results: Dict[str, Any] = comparator.matrix_comparison(
                image=img_array,
                methods_config=methods_config_list,
                comparison_type="all_vs_all",
                save_results=True,
                output_dir=str(img_output_dir),
            )
            all_vs_all_results.append(
                {
                    "image": img_name,
                    "results": matrix_results,
                    "n_comparisons": len(matrix_results.get("df_comparisons", [])),
                }
            )
            print(f"   ✅ Матрица сравнения для {img_name}")
            print(f"      ✅ Сравнено пар: {len(matrix_results.get('df_comparisons', []))}")
        except Exception as e:
            logger.error(f"      ❌ Ошибка all-vs-all: {e}")
            if os.getenv("DEBUG", "0") == "1":
                traceback.print_exc()

        # ──────────────────────────────────────────────────
        # 2. BATCH COMPARISON VS REFERENCE
        # ──────────────────────────────────────────────────
        try:
            print("   🔸 Batch comparison vs reference...")
            df_batch: pd.DataFrame = comparator.batch_comparison(
                image=img_array,
                methods_config=methods_config_list,
                reference_segmenter=ref_segmenter,
                reference_name=f"Reference_{reference_method}",
                save_results=True,
                output_dir=str(Path(output_dir) / "batch_comparison"),
            )
            batch_results_list.append(df_batch)
            print("   ✅ Пакетное сравнение завершено. Топ-5 метода сохранены.")
            metric_col = "jaccard" if "jaccard" in df_batch.columns else "f1_score"
            if metric_col in df_batch.columns:
                print(f"      ✅ Топ-3: {df_batch.nlargest(3, metric_col)['method'].tolist()}")
        except Exception as e:
            logger.error(f"      ❌ Ошибка batch comparison: {e}")

        # ──────────────────────────────────────────────────
        # 3. PAIRWISE COMPARISON (CV2 vs Sklearn Otsu)
        # ──────────────────────────────────────────────────
        try:
            print("   🔸 Pairwise comparison (CV2 vs Sklearn)...")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            df_pairwise: Dict[str, Any] = comparator.compare_methods(
                image=img_array,
                segmenter1=original_segmenter,
                segmenter2=ref_segmenter,
                name1="Original_CV2_Global",
                name2=f"Reference_{reference_method}",
                save_comparison=True,
                output_path=str(Path(output_dir) / f"compare_methods_{img_name}.jpg"),
            )
            pairwise_results_list.append(df_pairwise)
            print("   ✅ Попарное сравнение сохранено.")
            n_metrics: int = len(df_pairwise.get("metrics", {}))
            print(f"✅ Сохранено: {n_metrics} метрик")
        except TypeError as te:
            logger.error(f"      ⚠️  Pairwise требует старой сигнатуры: {te}")
        except Exception as e:
            logger.error(f"      ❌ Ошибка pairwise: {e}")

    # ──────────────────────────────────────────────────────
    # СВОДНЫЙ ОТЧЁТ
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 СВОДКА ПО МАТРИЧНОМУ СРАВНЕНИЮ")
    print("=" * 70)

    # All-vs-All статистика
    if all_vs_all_results:
        total_comparisons: int = sum(r["n_comparisons"] for r in all_vs_all_results)
        print(f"\n🔗 All-vs-All: {len(all_vs_all_results)} изображений, {total_comparisons} сравнений")

    # Batch comparison: агрегация результатов
    if batch_results_list:
        batch_summary: pd.DataFrame = pd.concat(batch_results_list, ignore_index=True)
        print(f"\n📦 Batch comparison: {len(batch_summary)} записей")

        # 🔹 Группировка по базовому методу и бэкенду для анализа
        if "backend" in batch_summary.columns:
            metric_col = "jaccard" if "jaccard" in batch_summary.columns else "f1_score"
            if metric_col in batch_summary.columns and pd.api.types.is_numeric_dtype(batch_summary[metric_col]):
                print("\n🔹 Статистика по бэкендам:")
                # Удаляем NaN перед агрегацией
                valid_data = batch_summary[[metric_col, "backend"]].dropna(subset=[metric_col])
                if not valid_data.empty:
                    backend_stats = valid_data.groupby("backend")[metric_col].mean()
                    for backend, score in backend_stats.items():
                        print(f"   {backend}: {score:.4f}")
                else:
                    print("   ⚠️  Нет валидных данных для агрегации по бэкендам")
            elif metric_col in batch_summary.columns:
                logger.warning(f"⚠️  Колонка '{metric_col}' не является числовой, пропуск агрегации")

        # Топ-5 методов по средней схожести
        # if "jaccard" in batch_summary.columns:
        #     top_methods: pd.Series = batch_summary.groupby("method")["jaccard"].mean().nlargest(5)
        #     print("\n🏆 Топ-5 по схожести с референсом (IoU):")
        #     for i, (method, score) in enumerate(top_methods.items(), 1):
        #         print(f"   {i}. {method}: {score:.4f}")
        # elif "f1_score" in batch_summary.columns:  # Fallback на F1
        #     top_methods: pd.Series = batch_summary.groupby("method")["f1_score"].mean().nlargest(5)
        #     print("\n🏆 Топ-5 по схожести с референсом (F1):")
        #     for i, (method, score) in enumerate(top_methods.items(), 1):
        #         print(f"   {i}. {method}: {score:.4f}")
        # else:
        #     print("\n⚠️  Нет доступных метрик для ранжирования")
        metric_col = "jaccard" if "jaccard" in batch_summary.columns else "f1_score"
        if metric_col in batch_summary.columns and pd.api.types.is_numeric_dtype(batch_summary[metric_col]):
            # Конвертация к числовому типу с обработкой ошибок
            batch_summary[metric_col] = pd.to_numeric(batch_summary[metric_col], errors='coerce')
            valid_top = batch_summary[[metric_col, "method"]].dropna(subset=[metric_col])
            
            if not valid_top.empty:
                top_methods: pd.Series = valid_top.groupby("method")[metric_col].mean().nlargest(5)
                print(f"\n🏆 Топ-5 по схожести с референсом ({metric_col}):")
                for i, (method, score) in enumerate(top_methods.items(), 1):
                    if pd.notna(score):  # Дополнительная проверка на NaN
                        parsed = parse_method_name(method)
                        backend_info = f" [{parsed['Backend']}/{parsed['Precision']}]" if parsed['Backend'] != "Unknown" else ""
                        print(f"   {i}. {method}{backend_info}: {score:.4f}")
            else:
                print(f"\n⚠️  Нет валидных данных для ранжирования по {metric_col}")
        elif metric_col in batch_summary.columns:
            logger.warning(f"⚠️  Колонка '{metric_col}' не является числовой, пропуск ранжирования")
        else:
            print(f"\n⚠️  Колонка '{metric_col}' не найдена в результатах")

    # Pairwise статистика
    if pairwise_results_list:
        print(f"\n🔍 Pairwise: {len(pairwise_results_list)} сравнений")

    print(f"\n💾 Результаты: {output_dir}")

    return {
        "all_vs_all_results": all_vs_all_results,
        "batch_results": (pd.concat(batch_results_list, ignore_index=True) if batch_results_list else None),
        "pairwise_results": pairwise_results_list,
        "comparator": comparator,
    }


# ──────────────────────────────────────────────────────────────────────
def run_ground_truth_evaluation(
    test_images: TestImagesDict,
    cv2_methods: SegmenterDict,
    sklearn_methods: SegmenterDict,
    torch_methods: SegmenterDict,
    output_dir: str = "./data/gt_evaluation",
    threshold: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Оценка качества сегментации против Ground Truth.

    Выполняет:
    1. **Фильтрация**: Пропуск изображений без GT.
    2. **Запуск методов**: Применение всех классических алгоритмов к каждому изображению.
    3. **Расчёт метрик**: IoU, Dice, Precision, Recall, F1, MAE, Hausdorff.
    4. **Визуализация**: Генерация бар-чартов, scatter-plot и PR-баланса.
    5. **Ранжирование**: Определение топ-5 методов по среднему IoU.

    Алгоритм:
    ```
    1. Объединить методы.
    2. Для каждого изображения с GT:
       a. Бинаризовать GT (threshold=0.5).
       b. Для каждого метода: запустить segment() → вычислить метрики.
       c. Сохранить детальные метрики в JSON.
    3. Агрегировать результаты по изображениям.
    4. Построить графики (metrics_comparison, speed_vs_accuracy, PR).
    5. Сохранить сводный CSV и вернуть Dict{gt_results_summary, summary_df, top_methods}.
    ```

    Args:
        test_images: Словарь `{имя: (путь, PIL.Image, GT-маска)}`.
        cv2_methods: Словарь методов OpenCV.
        sklearn_methods: Словарь методов scikit-learn.
        torch_methods: Словарь методов PyTorch.
        output_dir: Директория для сохранения отчётов.
        threshold: Порог бинаризации для метрик (по умолчанию 0.5).

    Returns:
        Optional[Dict[str, Any]]: Словарь с результатами:
                                  - `gt_results_summary`: {img_name: {method: metrics}}
                                  - `summary_df`: Агрегированный DataFrame
                                  - `top_methods`: Топ-5 методов по среднему IoU
                                  или `None` если нет изображений с GT.

    Note:
        - Изображения без GT пропускаются с предупреждением.
        - Методы с ошибками логируются, но не прерывают выполнение.
        - Визуализации включают:
          - `metrics_comparison_bar.png` — Bar chart метрик
          - `speed_vs_accuracy.png` — Scatter: время vs IoU
          - `precision_recall_balance.png` — PR-баланс

    Example:
        ```python
        results = run_ground_truth_evaluation(
            test_images=test_images,
            cv2_methods=cv2_methods,
            sklearn_methods=sklearn_methods,
            torch_methods=torch_methods
        )
        if results:
            print(f"Обработано изображений с GT: {len(results['gt_results_summary'])}")
            print(f"Топ-3 метода: {[m for m, _ in results['top_methods']]}")
        ```
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("ОЦЕНКА ПРОТИВ GROUND TRUTH")
    print("=" * 60)

    all_segmenters: SegmenterDict = {**cv2_methods, **sklearn_methods, **torch_methods}
    gt_results_summary: Dict[str, Dict[str, MetricsDict]] = {}
    has_gt_images: bool = False

    # ──────────────────────────────────────────────────────
    # ЦИКЛ ПО ИЗОБРАЖЕНИЯМ С GT
    # ──────────────────────────────────────────────────────
    for img_name, (_, img_pil, gt_mask) in tqdm(test_images.items(), desc="Ground Truth benchmark"):
        if gt_mask is None:
            logger.warning(f"⚠️  Пропуск {img_name}: нет Ground Truth")
            continue

        has_gt_images = True
        print(f"\n🎯 Обработка изображения: {img_name} (GT: {gt_mask.shape})")

        img_array: ImageArray = np.array(img_pil)

        # Подготовка GT: бинаризация
        gt_binary: MaskArray
        if gt_mask.max() <= 1.0:
            gt_binary = (gt_mask * 255).astype(np.uint8)
        else:
            gt_binary = gt_mask.astype(np.uint8)

        metrics_all: Dict[str, MetricsDict] = {}

        # Запуск методов
        for name, segmenter in all_segmenters.items():
            try:
                start_time: float = time.time()
                pred_mask: MaskArray = segmenter.segment(img_array)
                exec_time: float = time.time() - start_time

                # Ресайз если нужно
                if pred_mask.shape != gt_binary.shape:
                    from skimage.transform import resize

                    pred_mask = resize(pred_mask, gt_binary.shape, order=0, preserve_range=True).astype(np.uint8)

                # Метрики
                metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(
                    pred_mask=pred_mask,
                    gt_mask=gt_binary,
                    threshold=threshold,
                    include_hausdorff=True,
                )
                metrics["execution_time"] = exec_time
                metrics_all[name] = metrics

                # Статус
                iou: float = metrics.get("iou", 0)
                status: str = "✅" if iou > 0.5 else "⚠️" if iou > 0.2 else "❌"
                print(f"   {status} {name}: IoU={iou:.4f}, Dice={metrics.get('dice', 0):.4f}, t={exec_time:.3f}s")
                print(f"Mask after {name} segment: {pred_mask[:3, :3]}")
                print(segmenter.params["execution_info"])

            except Exception as e:
                logger.error(f"   💥 Критическая ошибка в методе {name}: {e}")
                traceback.print_exc()
                metrics_all[name] = {"error": str(e)}

        gt_results_summary[img_name] = metrics_all

        # Сохранение детальных метрик
        metrics_path: Path = Path(output_dir) / f"gt_metrics_{img_name}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_all, f, indent=2, ensure_ascii=False, default=str)
        print(f"   💾 Детальные метрики сохранены в {metrics_path}")

    # ──────────────────────────────────────────────────────
    # АНАЛИЗ И ВИЗУАЛИЗАЦИИ
    # ──────────────────────────────────────────────────────
    if not has_gt_images:
        logger.warning("⚠️ Ground Truth маски не найдены ни для одного изображения. Пропускаем этап оценки качества.")
        return None

    print("\n📈 Генерация отчётов...")

    # Визуализации
    print("\n📈 Построение сводных графиков по результатам Ground Truth...")
    visualize_gt_results(gt_results_summary, output_dir=output_dir)

    # Топ-5 методов по среднему IoU
    print("\n🏆 ТОП-5 методов по среднему IoU:")
    flat_results: List[Dict[str, Any]] = []
    for img, methods in gt_results_summary.items():
        for method, metrics in methods.items():
            if "iou" in metrics and "error" not in metrics:
                flat_results.append(
                    {
                        "Method": method,
                        "IoU": metrics["iou"],
                        "Image": img,
                    }
                )

    top_methods: List[Tuple[str, float]] = []
    if flat_results:
        df_flat: pd.DataFrame = pd.DataFrame(flat_results)
        top_series: pd.Series = df_flat.groupby("Method")["IoU"].mean().sort_values(ascending=False).head(5)
        top_methods = list(top_series.items())

        print("\n🏆 ТОП-5 методов по среднему IoU:")
        for i, (method, iou) in enumerate(top_methods, 1):
            print(f"   {i}. {method}: IoU = {iou:.4f}")
    else:
        logger.error("   Нет успешных результатов для ранжирования.")

    print("\n" + "=" * 60)
    print("СВОДНЫЙ ОТЧЕТ ПО GROUND TRUTH")
    print("=" * 60)

    # Сводная таблица
    rows: List[Dict[str, Any]] = []
    for img_name, methods_data in tqdm(gt_results_summary.items(), desc="Ground Truth Testing"):
        for method_name, metrics in methods_data.items():
            if "error" not in metrics and "iou" in metrics:
                rows.append(
                    {
                        "Image": img_name,
                        "Method": method_name,
                        "IoU": metrics["iou"],
                        "Dice": metrics.get("dice"),
                        "Precision": metrics.get("precision"),
                        "Recall": metrics.get("recall"),
                        "F1_Score": metrics.get("f1_score"),
                        "Time_s": metrics.get("execution_time"),
                    }
                )

    summary_df: Optional[pd.DataFrame] = None
    if rows:
        summary_df = pd.DataFrame(rows)
        summary_df_sorted = summary_df.sort_values(by=["Image", "IoU"], ascending=[True, False])
        print("\nТоп методов по IoU:")
        print(summary_df_sorted[["Method", "Image", "IoU", "Dice", "Time_s"]].to_string(index=False))
        summary_df_sorted.to_csv("./data/gt_summary_report.csv", index=False)
        summary_path: Path = Path(output_dir) / "gt_summary_report.csv"
        summary_df_sorted.to_csv(summary_path, index=False, float_format="%.4f")
        print(f"\n💾 Сводка: {summary_path}")
        plt.figure(figsize=(12, 6))
        first_img = list(gt_results_summary.keys())[0]
        df_plot = summary_df[summary_df["Image"] == first_img].sort_values("IoU", ascending=False).head(10)
        if not df_plot.empty:
            plt.barh(df_plot["Method"], df_plot["IoU"])
            plt.xlabel("IoU Score")
            plt.title(f"Top 10 Methods by IoU ({first_img})")
            plt.xlim(0, 1)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig("./data/gt_iu_comparison_chart.png")
            print("📊 График сохранен в ./data/gt_iu_comparison_chart.png")
        else:
            logger.warning("⚠️ Не удалось построить график: нет данных для первого изображения.")
        plt.close()
    else:
        logger.error("Нет успешных метрик для отображения.")

    return {
        "gt_results_summary": gt_results_summary,
        "summary_df": summary_df,
        "top_methods": top_methods,
    }


# ──────────────────────────────────────────────────────────────────────
def run_augmentation_training_study(
    root_dir: str = "./data1/ade20k",
    checkpoint_dir: str = "./models",
    device: str = "cuda",
    output_dir: str = "./data/augmentation_study",
) -> Optional[Dict[str, Any]]:
    """Исследование влияния уровней аугментаций на качество обучения моделей.

    Выполняет:
    1. **Конфигурация**: Определение уровней аугментаций и архитектур.
    2. **Обучение**: Запуск экспериментов с разными комбинациями.
    3. **Сравнение**: Анализ метрик (mIoU, val loss) между конфигурациями.
    4. **Визуализация**: Генерация бар-чартов, тепловых карт и топ-комбинаций.
    5. **Оценка**: Валидация сохранённых чекпоинтов на подмножестве данных.

    Алгоритм:
    ```
    1. Инициализировать ModelTrainer(root_dir, checkpoint_dir, device).
    2. Для каждого уровня аугментации (none/basic/medium/aggressive):
       Для каждой модели (unet, fpn, deeplab...):
          → train_experiment(config) → сохранить результат.
    3. Сравнить результаты по уровням.
    4. Сгенерировать графики сравнения и тепловую карту (Model × Augmentation → mIoU).
    5. Оценить чекпоинты на валидации.
    6. Вернуть Dict{results_by_model, summary_df, trainer, checkpoints}.
    ```

    Конфигурации аугментаций:
    - `none`: Без аугментаций (baseline).
    - `basic`: Базовые (flip, rotate, color jitter).
    - `medium`: Расширенные (+ blur, noise, crop).
    - `aggressive`: Агрессивные (+ mixup, cutout, strong color transforms).

    Архитектуры для тестирования:
    - `unet_smp`, `fpn_smp`, `psp_smp` (с энкодерами MiT-B5/ResNet)
    - `deeplab_tv`, `fcn_tv`, `segnet` (TorchVision)

    Args:
        root_dir: Корневая директория датасета (по умолчанию "./data/ade20k").
        checkpoint_dir: Директория для сохранения чекпоинтов.
        device: Устройство для обучения ("cuda" или "cpu").
        output_dir: Директория для сохранения отчётов и графиков.

    Returns:
        Optional[Dict[str, Any]]: Словарь с результатами:
                                  - `results_by_model`: {model_type: {aug_level: metrics}}
                                  - `summary_df`: Сводный DataFrame по всем конфигурациям
                                  - `trainer`: Экземпляр `ModelTrainer`
                                  или `None` при ошибке.

    Note:
        - Обучение выполняется на подмножестве датасета (`subset_fraction=0.05` для скорости).
        - Каждый эксперимент сохраняет чекпоинт с именем `{model}_{aug_level}_*.pth`.
        - Для оценки на валидации используются только чекпоинты с `medium` аугментациями.
        - Тепловая карта строится через `seaborn.heatmap` для наглядности.
        - Топ-3 комбинации сохраняются в отдельный CSV.

    Example:
        ```python
        results = run_augmentation_training_study(
            root_dir="./data/ade20k",
            device="cuda",
            output_dir="./results/aug_study"
        )
        if results:
            # Лучшая комбинация модель+аугментация
            best = results['summary_df'].nlargest(1, 'Best mIoU (%)')
            print(f"🏆 Лучшая: {best['Model'].iloc[0]} + {best['Augmentation Level'].iloc[0]}")
            print(f"   mIoU: {best['Best mIoU (%)'].iloc[0]:.2f}%")
        ```
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("ИССЛЕДОВАНИЕ: ВЛИЯНИЕ АУГМЕНТАЦИЙ НА КАЧЕСТВО ОБУЧЕНИЯ")
    print("=" * 80)

    # ──────────────────────────────────────────────────────
    # 1. ИНИЦИАЛИЗАЦИЯ ТРЕНЕРА
    # ──────────────────────────────────────────────────────
    trainer: ModelTrainer = ModelTrainer(
        checkpoint_dir=checkpoint_dir,
        root_dir=root_dir,
        device=device,
    )

    # ──────────────────────────────────────────────────────
    # 2. КОНФИГУРАЦИИ ЭКСПЕРИМЕНТА
    # ──────────────────────────────────────────────────────
    augmentation_configs: List[Dict[str, Any]] = [
        # {"level": "none", "epochs": 200, "lr": 1e-5, "subset_fraction": 0.05},
        {"level": "basic", "epochs": 200, "lr": 1e-5, "subset_fraction": 0.05},
        # {"level": "medium", "epochs": 200, "lr": 1e-5, "subset_fraction": 0.05},
        # {"level": "aggressive", "epochs": 50, "lr": 5e-5},
    ]

    model_types: List[str] = [
        # "unet_smp",  # U-Net
        # "fpn_smp",  # FPN + MiT-B5
        "psp_smp",  # PSPNet + MiT-B5
        # "deeplab_tv",  # DeepLabV3+
        # "fcn_tv",  # FCN ResNet-50
        # "segnet",  # SegNet
    ]

    results_by_model_and_aug: Dict[str, Dict[str, Any]] = {model_type: {} for model_type in model_types}

    print(f"\n🔧 План: {len(model_types)} моделей × {len(augmentation_configs)} уровней аугментаций")

    # ──────────────────────────────────────────────────────
    # 3. ЦИКЛ ОБУЧЕНИЯ
    # ──────────────────────────────────────────────────────
    for aug_config in augmentation_configs:
        for model_type in model_types:
            print(f"\n{'=' * 60}")
            print(f"🔹 Модель: {model_type} | Аугментации: {aug_config['level']}")
            print(f"{'=' * 60}")

            # Настройка энкодера
            encoder_name: Literal["resnet34", "resnet50", "resnet101", "mit_b5", "efficientnet-b0"]
            if model_type in ["fpn_smp", "psp_smp"]:
                encoder_name = "mit_b5"  # type: ignore[assignment]
            else:
                encoder_name = "resnet34"  # type: ignore[assignment]
            variant: str = "fcn_resnet50" if model_type == "fcn_tv" else "b5" if "mit" in encoder_name else "b5"

            # Конфиг обучения
            training_config: TrainingConfig = TrainingConfig(
                experiment_name=f"{model_type}_aug_{aug_config['level']}",
                model_type=model_type,
                augmentation_level=aug_config["level"],
                epochs=aug_config["epochs"],
                batch_size=4,
                lr=aug_config["lr"],
                encoder_name=encoder_name,
                variant=variant,
                subset_fraction=aug_config.get("subset_fraction", 0.05),
            )

            # Запуск обучения
            try:
                result: TrainingResult = trainer.train_experiment(training_config)
                results_by_model_and_aug[model_type][aug_config["level"]] = result

                miou_pct: float = result["best_miou"] * 100
                print(f"✅ {model_type} ({aug_config['level']}): Best mIoU = {miou_pct:.2f}%")

            except Exception as e:
                logger.error(f"❌ Ошибка обучения {model_type} ({aug_config['level']}): {e}")
                if os.getenv("DEBUG", "0") == "1":
                    traceback.print_exc()

    # ──────────────────────────────────────────────────────
    # 4. СРАВНЕНИЕ РЕЗУЛЬТАТОВ
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 СРАВНЕНИЕ УРОВНЕЙ АУГМЕНТАЦИЙ")
    print("=" * 60)

    for model_type in model_types:
        print(f"\n🔹 {model_type}:")
        print("-" * 60)
        for level, result in results_by_model_and_aug[model_type].items():
            miou: float = result["best_miou"] * 100
            print(f"   {level:10s}: {miou:6.2f}% mIoU")

    # ──────────────────────────────────────────────────────
    # 5. ВИЗУАЛИЗАЦИИ
    # ──────────────────────────────────────────────────────
    trainer.plot_experiment_comparison(output_path="./data/augmentation_comparison.png")

    print("\n" + "=" * 80)
    print("🎨 ГЕНЕРАЦИЯ ВИЗУАЛИЗАЦИЙ")
    print("=" * 80)

    _generate_augmentation_plots(
        results_by_model_and_aug=results_by_model_and_aug,
        model_types=model_types,
        augmentation_configs=augmentation_configs,
        output_dir=output_dir,
    )

    # ──────────────────────────────────────────────────────
    # 6. ОЦЕНКА НА ВАЛИДАЦИИ
    # ──────────────────────────────────────────────────────
    print("\n🔹 Оценка обученных моделей на валидации...")

    comparison_results: Dict[str, float] = trainer.compare_trained_models(augmentation_level="medium")
    print(f"Training results: {comparison_results}")

    # Сбор чекпоинтов с `medium` аугментациями
    checkpoints: Dict[str, str] = _collect_checkpoints(
        model_types=model_types,
        augmentation_levels=["none", "basic", "medium"],
        checkpoint_dir=checkpoint_dir,
    )

    if checkpoints:
        print(f"   🔍 Найдено чекпоинтов: {len(checkpoints)}")

        # Оценка
        try:
            eval_results_old: pd.DataFrame = trainer.evaluate_checkpoints(
                checkpoint_paths=list(checkpoints.values()),
                model_type="unet_smp",
                # val_fraction=0.05,
            )
            print(eval_results_old)

            eval_results: Dict[str, Any] = trainer.evaluate_trained_models_on_val(
                checkpoints=checkpoints,
                val_fraction=0.05,
            )
            print(eval_results)
            print("   ✅ Оценка завершена")
        except Exception as e:
            logger.warning(f"   ⚠️  Ошибка оценки: {e}")

    # ──────────────────────────────────────────────────────
    # 7. СВОДНАЯ ТАБЛИЦА И ТОП-3
    # ──────────────────────────────────────────────────────
    summary_df: pd.DataFrame = _create_augmentation_summary(
        results_by_model_and_aug=results_by_model_and_aug,
        model_types=model_types,
    )
    print(summary_df.to_string(index=False))

    # Сохранение сводки
    summary_path: Path = Path(output_dir) / "augmentation_impact_full_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 Сводка: {summary_path}")

    # Топ-3 комбинации
    top_combinations: pd.DataFrame = summary_df.nlargest(3, "Best mIoU (%)")
    print("\n🏆 ТОП-3 ЛУЧШИХ КОМБИНАЦИЙ:")
    for idx, row in top_combinations.iterrows():
        print(f"\n   {idx + 1}. {row['Model']} + {row['Augmentation Level']}")
        print(f"      mIoU: {row['Best mIoU (%)']:.2f}%")
        print(f"      Epochs: {row['Epochs']}")
        print(f"   Final Val Loss: {row['Final Val Loss']}")

    top_path: Path = Path(output_dir) / "top_3_combinations.csv"
    top_combinations.to_csv(top_path, index=False)
    print(f"\n📊 Топ-3 сохранён: {top_path}")

    # Тепловая карта
    _generate_augmentation_heatmap(summary_df, output_dir)

    return {
        "results_by_model": results_by_model_and_aug,
        "summary_df": summary_df,
        "trainer": trainer,
        "checkpoints": checkpoints,
    }


# ──────────────────────────────────────────────────────────────────────
def _generate_augmentation_plots(
    results_by_model_and_aug: Dict[str, Dict[str, Any]],
    model_types: List[str],
    augmentation_configs: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Генерирует графики сравнения влияния аугментаций на качество обучения.

    Выполняет:
    1. **Подготовка**: Создание директории `plots/`.
    2. **График 1 (модель × аугментации)**:
       - Группированные бар-чарты для каждой архитектуры.
       - Подписи значений поверх столбцов.
    3. **График 2 (аугментация × модели)**:
       - Сравнение всех моделей для каждого уровня аугментаций.
       - Цветовая схема через `viridis` colormap.

    Алгоритм:
    ```
    1. Создать plots_dir.
    2. График 1:
       → Для каждой модели: извлечь mIoU по уровням аугментаций.
       → Построить bar chart с цветами ["#ffcccc", "#ff9999", "#ff6666"].
       → Добавить подписи значений и сетку.
    3. График 2:
       → Для каждого уровня аугментаций: собрать mIoU по моделям.
       → Построить bar chart с цветами из colormaps["viridis"].
       → Повернуть подписи оси X на 45°.
    4. Сохранить оба графика в PNG с DPI=300.
    ```

    Args:
        results_by_model_and_aug: Результаты обучения `{модель: {уровень: метрики}}`.
        model_types: Список типов моделей для визуализации.
        augmentation_configs: Конфигурации уровней аугментаций.
        output_dir: Базовая директория для сохранения.

    Returns:
        None. Графики сохраняются в `{output_dir}/plots/`.

    Note:
        - Цвета для аугментаций подобраны для интуитивного восприятия (светлее → слабее).
        - Подписи значений отображаются с точностью до 2 знаков после запятой.
        - При отсутствии данных для модели/уровня график пропускается.

    Example:
        ```python
        _generate_augmentation_plots(
            results_by_model_and_aug=results,
            model_types=["unet_smp", "fpn_smp"],
            augmentation_configs=[{"level": "basic"}, {"level": "medium"}],
            output_dir="./results/aug_study"
        )
        ```
    """
    plots_dir: Path = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # График 1: Аугментации для каждой модели
    _, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, model_type in enumerate(model_types):
        if idx >= len(axes):
            break
        ax = axes[idx]

        levels = list(results_by_model_and_aug[model_type].keys())
        miou_values: List[float] = [results_by_model_and_aug[model_type][level]["best_miou"] * 100 for level in levels]

        colors = ["#ffcccc", "#ff9999", "#ff6666"][: len(levels)]
        bars = ax.bar(levels, miou_values, color=colors, edgecolor="black", linewidth=1.5)

        ax.set_title(f"{model_type}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Best mIoU (%)")
        ax.set_ylim(0, max(miou_values) * 1.2 if miou_values else 100)
        ax.grid(axis="y", alpha=0.3)

        # Подписи значений
        for bar, val in zip(bars, miou_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.2f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(
        plots_dir / "augmentation_comparison_all_models.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"📊 График 1: {plots_dir / 'augmentation_comparison_all_models.png'}")

    # График 2: Модели для каждого уровня аугментаций
    if augmentation_configs:
        fig, axes = plt.subplots(1, len(augmentation_configs), figsize=(6 * len(augmentation_configs), 5))
        if len(augmentation_configs) == 1:
            axes = [axes]

        for idx, aug_config in enumerate(augmentation_configs):
            ax = axes[idx]
            level = aug_config["level"]

            model_names: List[str] = []
            miou_values = []

            for model_type in model_types:
                if level in results_by_model_and_aug[model_type]:
                    model_names.append(model_type.replace("_smp", "").replace("_tv", ""))
                    miou = results_by_model_and_aug[model_type][level]["best_miou"] * 100
                    miou_values.append(miou)

            if not model_names:
                continue

            colors1: np.ndarray = colormaps["viridis"](np.linspace(0, 1, len(model_names)))
            bars = ax.bar(model_names, miou_values, color=colors1, edgecolor="black", linewidth=1.5)

            ax.set_title(f"Аугментации: {level}", fontsize=12, fontweight="bold")
            ax.set_ylabel("Best mIoU (%)")
            ax.set_ylim(0, max(miou_values) * 1.2)
            ax.grid(axis="y", alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

            # Подписи значений
            for bar, val in zip(bars, miou_values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        plt.tight_layout()
        plt.savefig(
            plots_dir / "model_comparison_by_augmentation.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"📊 График 2: {plots_dir / 'model_comparison_by_augmentation.png'}")


# ──────────────────────────────────────────────────────────────────────
def _collect_checkpoints(
    model_types: List[str],
    augmentation_levels: List[str],
    checkpoint_dir: str,
) -> Dict[str, str]:
    """Собирает актуальные чекпоинты по паттернам имён файлов.

    Выполняет:
    1. **Поиск по шаблону**: Использование `glob` для поиска `*_*.pth`.
    2. **Фильтрация по времени**: Выбор самого свежего файла через `max(..., key=getctime)`.
    3. **Формирование отображения**: Создание `{display_name: checkpoint_path}`.

    Алгоритм:
    ```
    1. Для каждой (model_type, level):
       a. Сформировать pattern = f"{checkpoint_dir}/{model_type}_{level}_*.pth".
       b. Найти files = glob.glob(pattern).
       c. Если files не пуст:
          → latest = max(files, key=os.path.getctime).
          → checkpoints[f"{model_type} ({level})"] = latest.
       d. Иначе → вывести предупреждение.
    2. Вернуть словарь чекпоинтов.
    ```

    Args:
        model_types: Список типов моделей для поиска.
        augmentation_levels: Уровни аугментаций для фильтрации.
        checkpoint_dir: Директория с сохранёнными `.pth` файлами.

    Returns:
        Dict[str, str]: Словарь `{отображаемое_имя: путь_к_файлу}`.

    Note:
        - Используется время создания файла (`getctime`), а не модификации.
        - При отсутствии чекпоинтов выводится предупреждение, но выполнение продолжается.
        - Отображаемое имя форматируется как `"{модель} ({уровень})"`.

    Example:
        ```python
        checkpoints = _collect_checkpoints(
            model_types=["unet_smp", "fpn_smp"],
            augmentation_levels=["basic", "medium"],
            checkpoint_dir="./models"
        )
        print(checkpoints)
        # {'unet_smp (basic)': './models/unet_smp_basic_epoch199.pth', ...}
        ```
    """
    checkpoints: Dict[str, str] = {}

    for model_type in model_types:
        for level in augmentation_levels:
            pattern: str = f"{checkpoint_dir}/{model_type}_{level}_*.pth"
            files: List[str] = glob.glob(pattern)

            if files:
                # Самый свежий файл
                latest: str = max(files, key=os.path.getctime)
                display_name: str = f"{model_type} ({level})"
                checkpoints[display_name] = latest
                print(f"   ✅ {display_name}: {latest}")
            else:
                logger.warning(f"   ⚠️  {model_type} ({level}): не найден")

    return checkpoints


# ──────────────────────────────────────────────────────────────────────
def _create_augmentation_summary(
    results_by_model_and_aug: Dict[str, Dict[str, Any]],
    model_types: List[str],
) -> pd.DataFrame:
    """Создаёт сводный DataFrame по результатам обучения с аугментациями.

    Выполняет:
    1. **Агрегация**: Сбор метрик из вложенной структуры результатов.
    2. **Форматирование**: Конвертация `best_miou` в проценты, обработка `None` для val loss.
    3. **Сортировка**: Упорядочивание по модели и убыванию mIoU.

    Алгоритм:
    ```
    1. Инициализировать пустой список summary_data.
    2. Для каждой (model_type, level, result):
       → Добавить запись с полями:
          - Model, Augmentation Level
          - Best mIoU (%) = result["best_miou"] * 100
          - Epochs, Final Val Loss (с заменой None на "N/A")
    3. Создать DataFrame из списка.
    4. Отсортировать по [Model (asc), Best mIoU (%) (desc)].
    5. Вернуть DataFrame.
    ```

    Args:
        results_by_model_and_aug: Результаты обучения `{модель: {уровень: метрики}}`.
        model_types: Список типов моделей для включения в сводку.

    Returns:
        pd.DataFrame: Сводная таблица с колонками:
                      [Model, Augmentation Level, Best mIoU (%), Epochs, Final Val Loss].

    Note:
        - `Final Val Loss` отображается как `"N/A"`, если значение отсутствует.
        - Сортировка позволяет легко найти лучшую конфигурацию для каждой модели.
        - Все числовые значения сохраняются в исходной точности (кроме mIoU → %).

    Example:
        ```python
        summary = _create_augmentation_summary(results, model_types=["unet_smp", "fpn_smp"])
        print(summary.head())
        #   Model      Augmentation Level  Best mIoU (%)  Epochs  Final Val Loss
        # 0 unet_smp   basic               45.32          200     0.87
        ```
    """
    summary_data: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА: ВЛИЯНИЕ АУГМЕНТАЦИЙ НА КАЧЕСТВО")
    print("=" * 80)

    for model_type in model_types:
        for level, result in results_by_model_and_aug[model_type].items():
            summary_data.append(
                {
                    "Model": model_type,
                    "Augmentation Level": level,
                    "Best mIoU (%)": result["best_miou"] * 100,
                    "Epochs": result["epochs_trained"],
                    "Final Val Loss": (result["final_val_loss"] if result["final_val_loss"] is not None else "N/A"),
                }
            )

    df: pd.DataFrame = pd.DataFrame(summary_data)
    return df.sort_values(["Model", "Best mIoU (%)"], ascending=[True, False])


# ──────────────────────────────────────────────────────────────────────
def _generate_augmentation_heatmap(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Генерирует тепловую карту влияния аугментаций на качество сегментации.

    Выполняет:
    1. **Проверка данных**: Пропуск генерации при пустом DataFrame.
    2. **Pivot-трансформация**: Преобразование в матрицу [Модель × Уровень → mIoU].
    3. **Визуализация**: Построение `seaborn.heatmap` с аннотациями и цветовой схемой.
    4. **Сохранение**: Экспорт в PNG с высоким DPI и подписями.

    Алгоритм:
    ```
    1. Если summary_df.empty → вернуть.
    2. Создать plots_dir = Path(output_dir) / "plots".
    3. pivot_data = summary_df.pivot(index="Model", columns="Augmentation Level", values="Best mIoU (%)").
    4. Построить heatmap:
       - cmap="YlOrRd" (от жёлтого к красному)
       - annot=True с форматом ".2f"
       - linewidths=0.5 для разделения ячеек
    5. Добавить заголовок, подписи осей, сохранить в PNG.
    ```

    Args:
        summary_df: Сводный DataFrame с результатами обучения.
        output_dir: Базовая директория для сохранения визуализаций.

    Returns:
        None. Тепловая карта сохраняется в `{output_dir}/plots/augmentation_heatmap.png`.

    Note:
        - Цветовая схема `YlOrRd` интуитивно отображает рост качества (светлее → хуже).
        - Аннотации отображают точные значения mIoU с двумя знаками после запятой.
        - При отсутствии данных для комбинации ячейка остаётся пустой.

    Example:
        ```python
        _generate_augmentation_heatmap(summary_df, output_dir="./results/aug_study")
        # Сохранит тепловую карту в ./results/aug_study/plots/augmentation_heatmap.png
        ```
    """
    if summary_df.empty:
        return

    print("\n" + "=" * 80)
    print("ТЕПЛОВАЯ КАРТА: МОДЕЛИ × АУГМЕНТАЦИИ")
    print("=" * 80)

    plots_dir: Path = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Pivot таблица: модель × уровень аугментаций → mIoU
    pivot_data: pd.DataFrame = summary_df.pivot(
        index="Model",
        columns="Augmentation Level",
        values="Best mIoU (%)",
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )

    plt.title(
        "Влияние уровня аугментаций на качество сегментации (mIoU %)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Уровень аугментаций", fontsize=12)
    plt.ylabel("Модель", fontsize=12)

    plt.tight_layout()
    heatmap_path: Path = plots_dir / "augmentation_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"🔥 Тепловая карта сохранена: {heatmap_path}")


# ──────────────────────────────────────────────────────────────────────
def visualize_gt_results(
    results_dict: Dict[str, Dict[str, MetricsDict]],
    output_dir: str = "./data/gt_visualization",
) -> None:
    """Построение графиков по результатам тестирования с Ground Truth.

    Генерирует три типа визуализаций:
    1. **Bar Chart**: Сравнение метрик (IoU, Dice, F1) по методам.
    2. **Scatter Plot**: Зависимость точности (IoU) от времени выполнения.
    3. **Precision-Recall**: Баланс между точностью и полнотой.

    Данные агрегируются по изображениям: для каждого метода вычисляется
    среднее значение метрики по всем доступным тестовым изображениям.

    Args:
        results_dict: Словарь результатов вида:
                     `{имя_изображения: {имя_метода: метрики}}`.
        output_dir: Директория для сохранения графиков и CSV-отчёта.

    Returns:
        None. Графики сохраняются на диск, информация выводится в stdout.

    Note:
        - Методы без метрики `iou` или с ключом `error` исключаются из анализа.
        - Для построения графиков требуется минимум 2 метода с валидными данными.
        - Цветовая схема: библиотеки кодируются разными цветами для наглядности.

    Example:
        ```python
        results = {
            "img1": {
                "Otsu_CV2": {"iou": 0.85, "dice": 0.91, "execution_time": 0.12},
                "KMeans_Sklearn": {"iou": 0.72, "dice": 0.83, "execution_time": 1.45},
            }
        }
        visualize_gt_results(results, output_dir="./my_results")
        # Сохранит: metrics_comparison_bar.png, speed_vs_accuracy.png, ...
        ```
    """
    os.makedirs(output_dir, exist_ok=True)

    # Сбор данных в единый DataFrame
    all_rows: List[Dict[str, Any]] = []
    for img_name, methods_data in tqdm(results_dict.items(), desc="Visualisation GT Testing"):
        for method_name, metrics in methods_data.items():
            if "error" in metrics or "iou" not in metrics:
                continue

            row: Dict[str, Any] = {
                "Image": img_name,
                "Method": method_name,
                "Library": _extract_library_from_name(method_name),
                "IoU": metrics["iou"],
                "Dice": metrics["dice"],
                "F1_Score": metrics["f1_score"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "Time_s": metrics.get("execution_time", 0),
                "Area_Diff": metrics.get("area_difference", 0),
            }
            all_rows.append(row)

    if not all_rows:
        logger.warning("⚠️ Нет валидных данных для визуализации.")

        return
    df: pd.DataFrame = pd.DataFrame(all_rows)
    # Группировка по методам для усреднения (если изображений несколько)
    df_avg: pd.DataFrame = _aggregate_metrics_by_method(df)

    # Генерация графиков
    # === ГРАФИК 1: Сравнение метрик (Bar Chart) ===
    _plot_metrics_comparison(df_avg, output_dir)

    # === ГРАФИК 2: Speed vs Accuracy (Scatter Plot) ===
    _plot_speed_vs_accuracy(df_avg, output_dir)

    # === ГРАФИК 3: Precision-Recall Balance ===
    _plot_precision_recall(df_avg, output_dir)

    # Сохранение сводной таблицы
    csv_path: str = os.path.join(output_dir, "gt_summary_table.csv")
    df_avg.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"💾 Сводная таблица: {csv_path}")


# ──────────────────────────────────────────────────────────────────────
def _extract_library_from_name(method_name: str) -> str:
    """Извлекает название библиотеки из имени метода сегментации.

    Выполняет:
    1. **Разбиение**: Сегментация строки по символу подчёркивания.
    2. **Анализ суффикса**: Проверка последнего элемента на известные библиотеки.
    3. **Нормализация**: Приведение к верхнему регистру для сравнения.

    Алгоритм:
    ```
    1. Разбить method_name на части: parts = method_name.split("_").
    2. Если parts пуст → вернуть "Unknown".
    3. Взять suffix = parts[-1].upper().
    4. Если suffix в ["CV2", "SKLEARN", "TORCH", "NEURAL"] → вернуть suffix.
    5. Иначе → вернуть "Unknown".
    ```

    Args:
        method_name: Имя метода, например `"Otsu_Thresholding_CV2"`.

    Returns:
        str: Название библиотеки: `"CV2"`, `"Sklearn"`, `"Torch"`, `"Neural"` или `"Unknown"`.

    Note:
        - Сравнение регистронезависимое благодаря `.upper()`.
        - Поддерживаются только суффиксы в конце имени метода.
        - Для методов без суффикса возвращается `"Unknown"`.

    Example:
        ```python
        lib = _extract_library_from_name("adaptive_thresholding_Sklearn")
        print(lib)  # "SKLEARN"
        ```
    """
    parts: List[str] = method_name.split("_")
    if not parts:
        return "Unknown"

    library_suffix: str = parts[-1].upper()
    if library_suffix in ["CV2", "SKLEARN", "TORCH", "NEURAL"]:
        return library_suffix
    return "Unknown"


# ──────────────────────────────────────────────────────────────────────
def _aggregate_metrics_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует метрики по методам: усреднение по всем изображениям.

    Выполняет:
    1. **Группировка**: Объединение строк по паре [Method, Library].
    2. **Агрегация**: Вычисление среднего для ключевых метрик.
    3. **Сортировка**: Упорядочивание по убыванию IoU для наглядности.

    Алгоритм:
    ```
    1. Группировать df по ["Method", "Library"].
    2. Применить .agg() с функцией "mean" для колонок:
       - IoU, Dice, F1_Score, Time_s, Precision, Recall.
    3. Сбросить индекс через .reset_index().
    4. Отсортировать по "IoU" в убывающем порядке.
    5. Вернуть агрегированный DataFrame.
    ```

    Args:
        df: DataFrame с сырыми результатами тестирования.

    Returns:
        pd.DataFrame: Агрегированные данные с колонками:
                      [Method, Library, IoU, Dice, F1_Score, Time_s, Precision, Recall].

    Note:
        - Агрегация корректно обрабатывает пропущенные значения (NaN).
        - Сортировка по IoU позволяет сразу увидеть лучшие методы.
        - Библиотека сохраняется для сравнения реализаций одного алгоритма.

    Example:
        ```python
        df_avg = _aggregate_metrics_by_method(raw_results_df)
        print(df_avg.head(3)[["Method", "Library", "IoU"]])
        ```
    """
    return (
        df.groupby(["Method", "Library"])
        .agg(
            {
                "IoU": "mean",
                "Dice": "mean",
                "F1_Score": "mean",
                "Time_s": "mean",
                "Precision": "mean",
                "Recall": "mean",
            }
        )
        .reset_index()
        .sort_values("IoU", ascending=False)
    )


# ──────────────────────────────────────────────────────────────────────
def _plot_metrics_comparison(df_avg: pd.DataFrame, output_dir: str) -> None:
    """Построение bar chart сравнения метрик качества сегментации.

    Выполняет:
    1. **Подготовка данных**: Извлечение усреднённых значений метрик.
    2. **Группировка баров**: Размещение IoU/Dice/F1 рядом для каждого метода.
    3. **Оформление**: Подписи осей, легенда, сетка, поворот подписей.
    4. **Сохранение**: Экспорт в PNG с DPI=150.

    Алгоритм:
    ```
    1. Создать фигуру 14×8 дюймов.
    2. Для каждого метода (индекс i):
       → Бар 1 (i - width): IoU, цвет #2ecc71.
       → Бар 2 (i): Dice, цвет #3498db.
       → Бар 3 (i + width): F1-Score, цвет #e74c3c.
    3. Установить подписи: методы на оси X (поворот 45°), "Score" на оси Y.
    4. Добавить легенду, сетку по Y, title.
    5. Сохранить в {output_dir}/metrics_comparison_bar.png.
    ```

    Args:
        df_avg: Агрегированный DataFrame с метриками по методам.
        output_dir: Директория для сохранения графика.

    Returns:
        None. График сохраняется в `{output_dir}/metrics_comparison_bar.png`.

    Note:
        - Ширина баров (0.25) обеспечивает чёткое разделение групп.
        - Цветовая схема подобрана для цветовой доступности (colorblind-friendly).
        - Поворот подписей методов предотвращает наложение при длинных именах.

    Example:
        ```python
        _plot_metrics_comparison(df_avg, output_dir="./results/gt_evaluation")
        # Сохранит график сравнения метрик в metrics_comparison_bar.png
        ```
    """
    plt.figure(figsize=(14, 8))
    x: range = range(len(df_avg))
    width: float = 0.25

    plt.bar([i - width for i in x], df_avg["IoU"], width, label="IoU", color="#2ecc71")
    plt.bar(x, df_avg["Dice"], width, label="Dice", color="#3498db")
    plt.bar(
        [i + width for i in x],
        df_avg["F1_Score"],
        width,
        label="F1-Score",
        color="#e74c3c",
    )

    plt.xticks(x, df_avg["Method"], rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Сравнение метрик качества сегментации", fontsize=14)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path: str = os.path.join(output_dir, "metrics_comparison_bar.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"📊 График метрик: {output_path}")


# ──────────────────────────────────────────────────────────────────────
def _plot_speed_vs_accuracy(df_avg: pd.DataFrame, output_dir: str) -> None:
    """Построение scatter plot: зависимость точности (IoU) от времени выполнения.

    Выполняет:
    1. **Цветовое кодирование**: Разделение точек по библиотекам (CV2/Sklearn/Torch/Neural).
    2. **Размещение точек**: Ось X — время (сек), ось Y — IoU.
    3. **Аннотации**: Подписи методов (обрезанные до 10 символов) рядом с точками.
    4. **Оформление**: Легенда, сетка, заголовок, сохранение в PNG.

    Алгоритм:
    ```
    1. Создать фигуру 10×8 дюймов.
    2. Для каждой библиотеки в df_avg.groupby("Library"):
       → Взять цвет из lib_colors.
       → Построить scatter(Time_s, IoU) с размером точки 100, alpha=0.7.
       → Для каждой точки: добавить аннотацию с именем метода (сдвиг +5,+5).
    3. Установить подписи осей, заголовок, легенду, сетку.
    4. Сохранить в {output_dir}/speed_vs_accuracy.png.
    ```

    Args:
        df_avg: Агрегированный DataFrame с метриками и временем.
        output_dir: Директория для сохранения графика.

    Returns:
        None. График сохраняется в `{output_dir}/speed_vs_accuracy.png`.

    Note:
        - Цвета библиотек фиксированы для консистентности между графиками.
        - Аннотации обрезаны до 10 символов, чтобы избежать наложения.
        - Точки с одинаковыми координатами могут перекрываться (alpha=0.7 для прозрачности).

    Example:
        ```python
        _plot_speed_vs_accuracy(df_avg, output_dir="./results/gt_evaluation")
        # Визуализирует компромисс скорость/точность для всех методов
        ```
    """
    plt.figure(figsize=(10, 8))

    lib_colors: Dict[str, str] = {
        "CV2": "#e67e22",
        "Sklearn": "#9b59b6",
        "Torch": "#34495e",
        "Neural": "#c0392b",
    }

    for lib, group in df_avg.groupby("Library"):
        color: str = lib_colors.get(lib, "#95a5a6")
        plt.scatter(
            group["Time_s"],
            group["IoU"],
            s=100,
            label=lib,
            color=color,
            alpha=0.7,
            edgecolors="black",
        )
        # Подписи методов
        for _, row in group.iterrows():
            plt.annotate(
                row["Method"][-10:],
                (row["Time_s"], row["IoU"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    plt.xlabel("Время выполнения (сек)")
    plt.ylabel("IoU Score")
    plt.title("Зависимость точности от скорости", fontsize=14)
    plt.legend(title="Библиотека")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path: str = os.path.join(output_dir, "speed_vs_accuracy.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"🚀 Speed vs Accuracy: {output_path}")


# ──────────────────────────────────────────────────────────────────────
def _plot_precision_recall(df_avg: pd.DataFrame, output_dir: str) -> None:
    """Построение графика баланса Precision и Recall для методов сегментации.

    Выполняет:
    1. **Построение кривой**: Линия, соединяющая точки (Recall, Precision).
    2. **Аннотации**: Подписи методов (обрезанные до 15 символов) рядом с точками.
    3. **Оформление**: Подписи осей, заголовок, сетка, сохранение в PNG.

    Алгоритм:
    ```
    1. Создать фигуру 10×6 дюймов.
    2. Построить линию: plt.plot(Recall, Precision, "o-", linewidth=2).
    3. Для каждой строки df_avg:
       → Добавить аннотацию с именем метода[:15] (сдвиг +5,+5, шрифт 7pt).
    4. Установить подписи: "Recall (Полнота)", "Precision (Точность)", заголовок.
    5. Добавить сетку, сохранить в {output_dir}/precision_recall_balance.png.
    ```

    Args:
        df_avg: Агрегированный DataFrame с метриками.
        output_dir: Директория для сохранения графика.

    Returns:
        None. График сохраняется в `{output_dir}/precision_recall_balance.png`.

    Note:
        - Идеальный метод находится в правом верхнем углу (Precision=Recall=1).
        - Аннотации обрезаны до 15 символов для читаемости.
        - Линия соединяет точки в порядке следования в DataFrame (не по значению).

    Example:
        ```python
        _plot_precision_recall(df_avg, output_dir="./results/gt_evaluation")
        # Показывает компромисс между точностью и полнотой для каждого метода
        ```
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df_avg["Recall"], df_avg["Precision"], "o-", linewidth=2, markersize=8)

    for _, row in df_avg.iterrows():
        plt.annotate(
            row["Method"][:15],
            (row["Recall"], row["Precision"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )

    plt.xlabel("Recall (Полнота)")
    plt.ylabel("Precision (Точность)")
    plt.title("Баланс Precision и Recall")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path: str = os.path.join(output_dir, "precision_recall_balance.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"⚖️ График Precision-Recall сохранен: {output_path}")


# ──────────────────────────────────────────────────────────────────────
def load_test_images(
    use_image_with_mask: bool = False,
    dataset_repo: str = ADE20K_REPO_ID,
) -> TestImagesDict:
    """Загружает тестовые изображения для бенчмарка.

    Поддерживает два режима:
    1. **С Ground Truth** (`use_image_with_mask=True`):
       - Загружает изображение и маску из репозитория HuggingFace.
       - Конвертирует многоклассовую маску в бинарную (объект/фон).
       - Сохраняет локальные копии для офлайн-работы.
    2. **Без Ground Truth** (`use_image_with_mask=False`):
       - Загружает изображения из предопределённых URL или локальных путей.
       - Генерирует синтетические бинарные маски (половина изображения = объект).
    3. **Валидация**: Проверка успешной загрузки хотя бы одного изображения.

    Алгоритм конвертации маски ADE20K → бинарная:
    1. Находим самый частый класс (предполагаем, что это фон).
    2. Все остальные классы считаем объектом (значение 255).
    3. Если объектов <1% — пробуем второй по величине класс.

    Алгоритм:
    ```
    1. Если use_image_with_mask:
       a. Скачать ADE_val_00000001.jpg/png.
       b. Найти самый частый класс → фон, остальные → объект.
       c. Сохранить локальные копии.
    2. Иначе:
       a. Загрузить изображения по URL или из test_images/.
       b. Сгенерировать синтетическую маску (нижняя половина = 255).
    3. Если test_images пуст → RuntimeError.
    4. Вернуть TestImagesDict.
    ```

    Args:
        use_image_with_mask: Если `True`, загружать с ground truth из ADE20K.
        dataset_repo: ID репозитория на HuggingFace (по умолчанию ADE20K fixtures).

    Returns:
        TestImagesDict: Словарь `{имя: (путь, PIL.Image, GT-маска или None)}`.

    Raises:
        RuntimeError: Если не удалось загрузить ни одного изображения.

    Note:
        - Конвертация ADE20K → бинарная: фон = самый частый класс, объекты = остальные.
        - При <1% пикселей объектов пытается использовать второй по частоте класс.
        - Синтетические маски генерируются как `mask[h//2:, :] = 255`.

    Example:
        ```python
        # Режим с GT
        images = load_test_images(use_image_with_mask=True)
        name, (path, img, gt) = next(iter(images.items()))
        print(f"{name}: {img.size}, GT shape: {gt.shape if gt is not else None}")

        # Режим без GT (синтетические маски)
        images = load_test_images(use_image_with_mask=False)
        ```
    """
    test_images: TestImagesDict = {}

    if use_image_with_mask:
        _load_images_with_ground_truth(test_images, dataset_repo)
    else:
        _load_images_without_ground_truth(test_images)
    if not test_images:
        raise RuntimeError("Не удалось загрузить ни одного тестового изображения")
    return test_images


# ──────────────────────────────────────────────────────────────────────
def _load_images_with_ground_truth(
    test_images: TestImagesDict,
    repo_id: str,
) -> None:
    """Загружает изображение и GT-маску из репозитория HuggingFace.

    Выполняет:
    1. **Скачивание**: Загрузка изображения и маски через `hf_hub_download`.
    2. **Конвертация**: Преобразование многоклассовой маски в бинарную.
    3. **Сохранение локальных копий**: Кэширование для офлайн-работы.
    4. **Добавление в словарь**: Запись в `test_images` с ключом `"ade20k_sample"`.

    Алгоритм:
    ```
    1. Скачать ADE_val_00000001.jpg и .png из repo_id.
    2. Открыть изображения через PIL, конвертировать RGB.
    3. Конвертировать GT: _convert_multiclass_to_binary(gt_np).
    4. Сохранить локальные копии: _save_test_artifacts(...).
    5. Добавить в test_images["ade20k_sample"] = (path, img, binary_gt).
    ```

    Args:
        test_images: Словарь для добавления загруженного образца.
        repo_id: ID репозитория на HuggingFace (например, `ADE20K fixtures`).

    Returns:
        None. Модифицирует `test_images` in-place.

    Note:
        - Конвертация GT: самый частый класс считается фоном, остальные — объектом.
        - Локальные копии сохраняются в `test_images/` для повторного использования.
        - При отсутствии интернета скачивание завершится ошибкой.

    Example:
        ```python
        test_images = {}
        _load_images_with_ground_truth(test_images, "hf-internal-testing/fixtures_ade20k")
        name, (path, img, gt) = test_images["ade20k_sample"]
        print(f"GT shape: {gt.shape}, unique values: {np.unique(gt)}")
        ```
    """
    print(f"📥 Загрузка из репозитория {repo_id}...")

    img_path: str = hf_hub_download(repo_id=repo_id, filename="ADE_val_00000001.jpg", repo_type="dataset")
    mask_path: str = hf_hub_download(repo_id=repo_id, filename="ADE_val_00000001.png", repo_type="dataset")

    img: Image.Image = Image.open(img_path).convert("RGB")
    gt_mask_pil: Image.Image = Image.open(mask_path)

    print(f"✅ Изображение загружено: {os.path.basename(img_path)}")
    print(f"✅ Ground truth загружен: {os.path.basename(mask_path)}")
    print(f"Ground Truth: {mask_path}")
    print(f"Размер изображения: {img.size}")
    print(f"Размер GT: {gt_mask_pil.size}")

    # Конвертация маски в бинарную
    gt_np: npt.NDArray = np.array(gt_mask_pil)
    print(f"\nДиапазон значений Ground Truth: {gt_np.min()} - {gt_np.max()}, min: {gt_np.min()}, max: {gt_np.max()}")
    binary_gt: MaskArray = _convert_multiclass_to_binary(gt_np)

    # Сохранение локальных копий
    local_paths: Dict[str, str] = _save_test_artifacts(img, gt_mask_pil, binary_gt)

    test_images["ade20k_sample"] = (local_paths["img"], img, binary_gt)
    print(f"✅ Загружен образец: {img.size}, GT: {binary_gt.shape}")


# ──────────────────────────────────────────────────────────────────────
def _convert_multiclass_to_binary(gt_np: npt.NDArray) -> MaskArray:
    """Конвертирует многоклассовую маску в бинарную (объект/фон).

    Алгоритм:
    1. Найти самый частый класс → считать фоном.
    2. Все остальные классы → объект (255).
    3. Если объектов <1% → попробовать второй по величине класс.

    Args:
        gt_np: Многоклассовая маска формы (H, W), dtype=uint8.

    Returns:
        MaskArray: Бинарная маска формы (H, W), dtype=uint8, {0, 255}.
    """
    unique: np.ndarray
    counts: np.ndarray
    unique, counts = np.unique(gt_np, return_counts=True)
    bg_class: np.ndarray = unique[np.argmax(counts)]
    print(f"📊 Статистика GT: Всего классов {len(unique)}. Самый частый: {bg_class} ({np.max(counts)} пикселей)")

    binary: MaskArray = (gt_np != bg_class).astype(np.uint8) * 255

    # fallback: если объектов слишком мало
    if np.sum(binary > 0) < (binary.size * 0.01) and len(unique) > 1:
        logger.warning("⚠️ Объектов слишком мало по стратегии 'не фон'. Пробуем взять второй по величине класс.")
        if len(unique) > 1:
            second_common: np.ndarray = unique[np.argsort(counts)[-2]]
            binary = (gt_np == second_common).astype(np.uint8) * 255
        else:
            # Если класс всего один, берем всё изображение как объект
            binary = np.ones_like(gt_np, dtype=np.uint8) * 255

    return binary


# ──────────────────────────────────────────────────────────────────────
def _save_test_artifacts(
    img: Image.Image,
    gt_raw: Image.Image,
    gt_binary: MaskArray,
) -> Dict[str, str]:
    """Сохраняет локальные копии тестовых артефактов для офлайн-работы.

    Выполняет:
    1. **Создание директории**: `test_images/` при отсутствии.
    2. **Сохранение файлов**:
       - Оригинальное изображение → `test_gt_image.jpg`
       - Сырая GT-маска → `test_gt_mask_raw.png`
       - Бинарная GT-маска → `test_gt_mask.png`
    3. **Логирование**: Вывод путей к сохранённым файлам.

    Алгоритм:
    ```
    1. Создать os.makedirs("test_images", exist_ok=True).
    2. Определить paths = {img, mask_raw, mask}.
    3. Сохранить: img.save(paths["img"]), gt_raw.save(paths["mask_raw"]), ...
    4. Вывести подтверждения сохранения для каждого файла.
    5. Вернуть словарь путей.
    ```

    Args:
        img: Оригинальное изображение в формате PIL.
        gt_raw: Сырая многоклассовая маска в формате PIL.
        gt_binary: Бинарная маска в формате numpy array.

    Returns:
        Dict[str, str]: Словарь `{тип: путь_к_файлу}`.

    Note:
        - Бинарная маска конвертируется в PIL перед сохранением.
        - Форматы: JPG для изображения, PNG для масок (без потерь).
        - Пути относительные, относительно текущей рабочей директории.

    Example:
        ```python
        paths = _save_test_artifacts(img, gt_raw, gt_binary)
        print(paths["mask"])  # "test_images/test_gt_mask.png"
        ```
    """
    os.makedirs("test_images", exist_ok=True)

    paths: Dict[str, str] = {
        "img": "test_images/test_gt_image.jpg",
        "mask_raw": "test_images/test_gt_mask_raw.png",
        "mask": "test_images/test_gt_mask.png",
    }

    img.save(paths["img"])
    print(f"✅ Изображение сохранено локально: {paths['img']}")

    gt_raw.save(paths["mask_raw"])
    print(f"✅ Изображение сырой маски сохранено локально: {paths['mask_raw']}")

    Image.fromarray(gt_binary).save(paths["mask"])
    print(f"✅ Изображение маски сохранено локально: {paths['mask']}")

    return paths


# ──────────────────────────────────────────────────────────────────────
def generate_precision_report(
    methods: List[str],
    image: np.ndarray,
    output_path: str = "precision_benchmark.csv",
    n_warmup: int = 3,
    n_runs: int = 10,
    compute_metrics: bool = True,
) -> Optional[pd.DataFrame]:
    """Генерирует CSV-отчёт: метод × точность × метрики производительности.

    Выполняет:
    1. **Подготовка референсов**: Запуск методов в fp32 для сравнения метрик.
    2. **Бенчмарк по точностям**: Замер времени для fp32/fp16/bf16 с прогревом.
    3. **Вычисление метрик**: IoU относительно fp32-референса (опционально).
    4. **Агрегация и сохранение**: Формирование DataFrame и экспорт в CSV.

    Алгоритм:
    ```
    1. Если compute_metrics:
       → Для каждого метода: создать TorchSegmenter2(precision="fp32") → сохранить маску.
    2. Для каждого (метод, точность):
       → Создать сегментер с нужной точностью.
       → Прогреть: n_warmup запусков с torch.cuda.synchronize().
       → Замерить: n_runs запусков, записать времена.
       → Вычислить IoU vs fp32 (если compute_metrics и точность != fp32).
       → Добавить запись в results.
    3. Создать DataFrame, сохранить в CSV.
    4. Вывести pivot-таблицы времени и IoU.
    ```

    Args:
        methods: Список названий методов для тестирования.
        image: Входное изображение (numpy array).
        output_path: Путь для сохранения CSV-отчёта.
        n_warmup: Количество "прогревочных" запусков перед замером.
        n_runs: Количество измерений для статистики.
        compute_metrics: Вычислять ли метрики качества (IoU относительно fp32).

    Returns:
        Optional[pd.DataFrame]: DataFrame с колонками:
                               [method, precision, mean_time_ms, std_time_ms, iou_vs_fp32, memory_mb]
                               или `None` при отсутствии данных.

    Note:
        - `bf16` может работать медленно на GPU старее Ampere (проверяется capability).
        - Замеры времени синхронизируются через `torch.cuda.synchronize()` для точности.
        - При ошибке для одной комбинации метод/точность выполнение продолжается.

    Example:
        ```python
        df = generate_precision_report(
            methods=["otsu_thresholding", "sobel_edge"],
            image=test_img,
            output_path="./reports/precision.csv",
            n_runs=20
        )
        if df is not None:
            print(df.pivot_table(index="method", columns="precision", values="mean_time_ms"))
        ```
    """
    results: List[Dict[str, Any]] = []

    # Предварительная подготовка референсов для fp32
    fp32_refs: Dict[str, Any] = {}
    if compute_metrics:
        print("📦 Подготовка референсных масок (fp32)...")
        for method in methods:
            try:
                ref = TorchSegmenter2(
                    method=method,
                    precision="fp32",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                with torch.no_grad():
                    fp32_refs[method] = ref.segment(image)
            except Exception as e:
                logger.warning(f"⚠️  Не удалось создать референс для {method}: {e}")

    for method in methods:
        for precision in ["fp32", "fp16", "bf16"]:
            # Пропуск неподдерживаемых комбинаций
            if precision != "fp32" and not torch.cuda.is_available():
                continue
            if precision == "bf16" and torch.cuda.is_available():
                if torch.cuda.get_device_capability(0)[0] < 8:
                    logger.warning(f"⚠️  bf16 может работать медленно на {torch.cuda.get_device_name(0)}")

            try:
                seg: TorchSegmenter2 = TorchSegmenter2(
                    method=method,
                    precision=precision,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

                # Warmup
                for _ in range(n_warmup):
                    _ = seg.segment(image)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                # Замер времени
                times: List[float] = []
                for _ in range(n_runs):
                    start: float = time.perf_counter()
                    mask: np.ndarray = seg.segment(image)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)

                # Вычисление метрик
                iou: float = 1.0
                if compute_metrics and precision != "fp32" and method in fp32_refs:
                    ref_mask = fp32_refs[method]
                    iou = SegmentationMetrics.calculate_iou(ref_mask, mask)

                results.append(
                    {
                        "method": method,
                        "precision": precision,
                        "mean_time_ms": np.mean(times) * 1000,
                        "std_time_ms": np.std(times) * 1000,
                        "min_time_ms": np.min(times) * 1000,
                        "max_time_ms": np.max(times) * 1000,
                        "iou_vs_fp32": iou,
                        "memory_mb": (torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0),
                    }
                )

            except Exception as e:
                logger.error(f"❌ Ошибка для {method}/{precision}: {e}")
                continue

    # Сохранение и вывод
    if results:
        df: pd.DataFrame = pd.DataFrame(results)
        df.to_csv(output_path, index=False)

        # Pivot-таблица для наглядности
        print(f"\n📊 Отчёт сохранён: {output_path}")
        print("\n⏱️  Время выполнения (мс):")
        print(df.pivot_table(index="method", columns="precision", values="mean_time_ms").round(2))

        if compute_metrics:
            print("\n🎯 IoU относительно fp32:")
            print(df.pivot_table(index="method", columns="precision", values="iou_vs_fp32").round(4))

        return df
    else:
        logger.warning("⚠️  Нет данных для отчёта")
        return None


# ──────────────────────────────────────────────────────────────────────
def _load_images_without_ground_truth(test_images: TestImagesDict) -> None:
    """Загружает изображения без GT с генерацией синтетических масок.

    Выполняет:
    1. **Загрузка из URL**: Скачивание изображений из предопределённых источников.
    2. **Загрузка локально**: Чтение изображений из локальных путей.
    3. **Генерация масок**: Создание синтетических бинарных масок (нижняя половина = объект).
    4. **Добавление в словарь**: Запись в `test_images` с уникальными ключами.

    Алгоритм:
    ```
    1. Для каждого (name, url) в image_sources:
       → Скачать изображение через _download_image.
       → Сгенерировать маску: _generate_synthetic_mask(img.size).
       → Добавить в test_images[name] = (path, img, mask).
    2. Для каждого (name, path) в image_paths:
       → Открыть изображение локально.
       → Сгенерировать синтетическую маску.
       → Добавить в test_images.
    3. При ошибках: логировать и продолжать загрузку остальных.
    ```

    Args:
        test_images: Словарь для добавления загруженных изображений.

    Returns:
        None. Модифицирует `test_images` in-place.

    Note:
        - Синтетические маски генерируются как `mask[h//2:, :] = 255`.
        - При ошибке загрузки одного изображения остальные продолжают загружаться.
        - Все изображения конвертируются в RGB при загрузке.

    Example:
        ```python
        test_images = {}
        _load_images_without_ground_truth(test_images)
        print(f"Загружено изображений: {len(test_images)}")
        ```
    """
    logger.warning("⚠️ Не удалось загрузить реальные GT. Используем только изображения.")
    image_sources: Dict[str, str] = {
        "countryside": "https://i.pinimg.com/736x/17/e7/fc/17e7fc299466b2afd989e709fe7c9815.jpg",
        # "nature": "https://i.pinimg.com/736x/f7/5a/f2/f75af26820b50c24600f50f3998eb02f.jpg",
        # "architecture": "https://i.pinimg.com/736x/86/f6/07/86f60748d5d9ae4cb9092018d1321648.jpg",
        # "trucks": "https://www.shutterstock.com/shutterstock/videos/1106252821/thumb/1.jpg?ip=x480",
        # "traffic": "https://images.pond5.com/pov-car-and-truck-traffic-footage-190002081_iconl.jpeg",
        # "mountain": "https://i.pinimg.com/736x/17/66/c4/1766c4f667af39f91172ef8eb21ab18a.jpg",
    }

    image_paths: Dict[str, str] = {
        # "war_frame_1": "test_images/2340_frame.jpg",
        # "war_frame_2": "test_images/3330_frame.jpg",
        # "war_frame_3": "test_images/4130_frame.jpg",
        # "war_frame_4": "test_images/4480_frame.jpg",
        # "building": "test_images/test_gt_image.jpg",
        # "animals": "test_images/animals.jpg",
    }

    for name, url in image_sources.items():
        try:
            img: Image.Image = _download_image(url, name)
            gt_synthetic: MaskArray = _generate_synthetic_mask(img.size)
            test_images[name] = (
                f"test_images/test_image_{name}.jpg",
                img,
                gt_synthetic,
            )
            print(f"✅ {name}: {img.size}")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {name}: {e}")

    for name, path in image_paths.items():
        try:
            img = Image.open(path)
            gt_synthetic = _generate_synthetic_mask(img.size)
            test_images[name] = (path, img, gt_synthetic)
            print(f"✅ {name}: {img.size}, ground truth: {gt_synthetic}")

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {name}: {e}")


# ──────────────────────────────────────────────────────────────────────
def _download_image(url: str, name: str) -> Image.Image:
    """Загружает изображение по URL с обработкой ошибок и локальным кэшированием.

    Выполняет:
    1. **Запрос**: `requests.get` с таймаутом 10 секунд.
    2. **Валидация**: `raise_for_status()` для обработки HTTP-ошибок.
    3. **Конвертация**: Открытие через PIL и приведение к режиму RGB.
    4. **Кэширование**: Сохранение локальной копии в `test_images/`.

    Алгоритм:
    ```
    1. Выполнить response = requests.get(url, timeout=10).
    2. Проверить статус: response.raise_for_status().
    3. Открыть изображение: img = Image.open(BytesIO(response.content)).convert("RGB").
    4. Создать директорию test_images/ при необходимости.
    5. Сохранить локально: img.save(f"test_images/test_image_{name}.jpg").
    6. Вернуть PIL.Image.
    ```

    Args:
        url: URL изображения для загрузки.
        name: Уникальное имя для локального файла.

    Returns:
        Image.Image: Загруженное изображение в режиме RGB.

    Note:
        - Таймаут 10 секунд предотвращает зависание при медленном соединении.
        - Локальные копии позволяют работать офлайн после первой загрузки.
        - Конвертация в RGB обеспечивает единообразие входных данных.

    Example:
        ```python
        img = _download_image("https://example.com/image.jpg", "test_sample")
        print(f"Размер: {img.size}, режим: {img.mode}")
        ```
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    img: Image.Image = Image.open(BytesIO(response.content)).convert("RGB")

    os.makedirs("test_images", exist_ok=True)
    local_path: str = f"test_images/test_image_{name}.jpg"
    img.save(local_path)

    return img


# ──────────────────────────────────────────────────────────────────────
def _generate_synthetic_mask(size: Tuple[int, int]) -> MaskArray:
    """Генерирует синтетическую бинарную маску для тестирования без GT.

    Выполняет:
    1. **Инициализация**: Создание нулевого массива формы (H, W).
    2. **Заполнение**: Установка нижней половины изображения в значение 255.
    3. **Возврат**: Массив типа uint8 с значениями {0, 255}.

    Алгоритм:
    ```
    1. Распаковать size → h, w.
    2. Создать mask = np.zeros((h, w), dtype=np.uint8).
    3. Установить mask[h // 2 :, :] = 255.
    4. Вернуть mask.
    ```

    Args:
        size: Размер изображения в формате (width, height).

    Returns:
        MaskArray: Бинарная маска формы (H, W), dtype=uint8, значения {0, 255}.

    Note:
        - Маска делит изображение горизонтально: верх = фон (0), низ = объект (255).
        - Подходит для базового тестирования алгоритмов без реального GT.
        - Для сложных сценариев рекомендуется использовать реальные маски.

    Example:
        ```python
        mask = _generate_synthetic_mask((640, 480))
        print(mask.shape, np.unique(mask))  # (480, 640) [0 255]
        ```
    """
    h, w = size
    mask: MaskArray = np.zeros((h, w), dtype=np.uint8)
    mask[h // 2 :, :] = 255
    return mask


# ──────────────────────────────────────────────────────────────────────
def save_metrics_report(
    metrics_all: Dict[str, MetricsDict],
    path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Сохраняет отчёт с метриками сегментации в JSON-формате.

    Выполняет:
    1. **Подготовка пути**: Создание родительских директорий при необходимости.
    2. **Сериализация**: Запись словаря метрик в файл с форматированием.
    3. **Обработка типов**: Использование `default=str` для numpy-типов.

    Алгоритм:
    ```
    1. Преобразовать path в Path-объект.
    2. Создать path.parent.mkdir(parents=True, exist_ok=True).
    3. Открыть файл в режиме записи с encoding="utf-8".
    4. Записать json.dump(metrics_all, f, indent=indent, ensure_ascii=ensure_ascii, default=str).
    5. Вывести подтверждение сохранения.
    ```

    Args:
        metrics_all: Словарь метрик: `{имя_метода: {метрика: значение}}`.
        path: Путь к выходному файлу (строка или Path).
        indent: Количество пробелов для форматирования (по умолчанию 2).
        ensure_ascii: Экранировать non-ASCII символы (по умолчанию False).

    Returns:
        None. Файл сохраняется на диск.

    Note:
        - `default=str` позволяет сериализовать numpy-скаляры и другие нестандартные типы.
        - `ensure_ascii=False` сохраняет кириллицу и спецсимволы в читаемом виде.
        - Родительские директории создаются автоматически при отсутствии.

    Example:
        ```python
        metrics = {
            "Otsu_CV2": {"iou": 0.85, "dice": 0.91, "execution_time": 0.12},
            "KMeans_Sklearn": {"iou": 0.72, "dice": 0.83},
        }
        save_metrics_report(metrics, "./results/metrics.json")
        ```
    """
    path_obj: Path = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=indent, ensure_ascii=ensure_ascii, default=str)

    print(f"💾 Отчёт сохранён: {path_obj}")


# ──────────────────────────────────────────────────────────────────────
def test_neural_segmentation_variants() -> Tuple[Optional[NeuralSegmenter], Optional[Dict[str, Any]]]:
    """Тестирование различных вариантов нейросетевой сегментации.

    Выполняет:
    1. **Загрузка тестового изображения**: Скачивание по предопределённому URL.
    2. **Инициализация сегментера**: Создание `NeuralSegmenter` с локальной моделью.
    3. **Вариант 1 (alpha)**: Визуализация результатов с разными значениями прозрачности.
    4. **Вариант 2 (анализ)**: Детальный разбор распределения классов по площади.
    5. **Вариант 3 (mask)**: Тестирование `segment_with_mask` с возвратом маски.

    Алгоритм:
    ```
    1. Скачать тестовое изображение через requests.
    2. Создать NeuralSegmenter(local_path="/path/to/model").
    3. Вариант 1:
       → Для alpha in [0.3, 0.5, 0.7, 1.0]: segment_image(alpha=alpha) → показать в subplot.
    4. Вариант 2:
       → detailed_segmentation() → вывести топ-5 классов по пикселям.
    5. Вариант 3:
       → segment_with_mask() → показать оригинал, результат, бинарную маску.
    6. Вернуть (сегментер, детальные результаты).
    ```

    Args:
        Нет аргументов (использует внутренние константы).

    Returns:
        Tuple[Optional[NeuralSegmenter], Optional[Dict[str, Any]]]:
            - Сегментер для дальнейшего использования.
            - Словарь с детальной статистикой сегментации.
            - `(None, None)` при ошибке.

    Note:
        - Требует наличия локальной модели по указанному пути.
        - Визуализации отображаются через `plt.show()` (блокирует выполнение).
        - При ошибке загрузки изображения или модели выводится traceback.

    Example:
        ```python
        segmenter, details = test_neural_segmentation_variants()
        if details:
            print(f"Обнаружено классов: {details['total_classes']}")
        ```
    """
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ВАРИАНТОВ НЕЙРОСЕТЕВОЙ СЕГМЕНТАЦИИ")
    print("=" * 60)

    # Загрузка тестового изображения
    img_url: str = "https://images.pond5.com/pov-car-and-truck-traffic-footage-190002081_iconl.jpeg"

    try:
        response: requests.Response = requests.get(img_url)
        test_image: Image.Image = Image.open(BytesIO(response.content))

        # Создаем нейросетевой сегментатор
        segmenter: NeuralSegmenter = NeuralSegmenter(local_path="/home/yamshchikov/models/segformer-b5-ready")

        # Вариант 1: Различные значения alpha
        print("\n1. Тестирование разных значений alpha:")
        alphas: List[float] = [0.3, 0.5, 0.7, 1.0]

        axes: np.ndarray  # type: ignore[assignment]
        _, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, alpha in enumerate(alphas):
            result: Image.Image = segmenter.segment_image(test_image, alpha=alpha)
            axes[i].imshow(result)
            axes[i].set_title(f"alpha = {alpha}")
            axes[i].axis("off")

            # Сохраняем
            result.save(f"neural_alpha_{alpha}.jpg")

        plt.suptitle("Neural Segmentation with Different Alpha Values", fontsize=14)
        plt.tight_layout()
        plt.show()

        # Вариант 2: Детальный анализ
        print("\n2. Детальный анализ сегментации:")
        detailed_result: Dict[str, Any] = segmenter.detailed_segmentation(test_image)

        # Выводим информацию о классах
        print(f"Обнаружено классов: {detailed_result['total_classes']}")
        print("\nТоп-5 классов по площади:")

        sorted_classes: List[Tuple[str, Dict[str, Any]]] = sorted(
            detailed_result["class_distribution"].items(),
            key=lambda x: x[1]["pixel_count"],
            reverse=True,
        )[:5]

        for class_name, info in sorted_classes:
            print(f"  {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")

        # Вариант 3: segment_with_mask
        print("\n3. Тестирование segment_with_mask:")
        result_np: np.ndarray
        mask: Optional[np.ndarray]
        result_np, mask = segmenter.segment_with_mask(test_image, alpha=0.5)

        axes2: np.ndarray  # type: ignore[assignment]
        _, axes2 = plt.subplots(1, 3, figsize=(15, 5))

        axes2[0].imshow(test_image)  # type: ignore[index]
        axes2[0].set_title("Original")  # type: ignore[index]
        axes2[0].axis("off")  # type: ignore[index]

        axes2[1].imshow(result_np)  # type: ignore[index]
        axes2[1].set_title("Segmentation Result")  # type: ignore[index]
        axes2[1].axis("off")  # type: ignore[index]

        axes2[2].imshow(mask, cmap="gray")  # type: ignore[index]
        axes2[2].set_title("Binary Mask")  # type: ignore[index]
        axes2[2].axis("off")  # type: ignore[index]

        plt.tight_layout()
        plt.show()

        if mask is not None:
            print(f"Размер маски: {mask.shape}")
            print(f"Площадь маски: {np.sum(mask > 0)} пикселей")

        return segmenter, detailed_result

    except Exception as e:
        logger.error(f"❌ Ошибка тестирования нейросетевых вариантов: {e}")
        logger.error(traceback.format_exc())
        return None, None


# ──────────────────────────────────────────────────────────────────────
def run_cpu_cuda_benchmark(
    all_benchmark_methods: Dict,
    test_image: ImageArray,
    n_runs: int = 5,
    warmup_runs: int = 2,
) -> BenchmarkResult:
    """Запуск бенчмарка CPU vs CUDA для классических методов.

    Выполняет:
    1. **Фильтрация**: Исключение нейросетевых методов по ключевым словам.
    2. **Инициализация**: Создание `CpuCudaBenchmark` с параметрами прогонов.
    3. **Бенчмарк**: Замер времени на CPU и CUDA с предварительным прогревом.
    4. **Сводка**: Вывод среднего времени, std и speedup для каждого метода.

    Алгоритм:
    ```
    1. Отфильтровать методы: убрать neural/segformer/unet/sam и т.д.
    2. Инициализировать CpuCudaBenchmark(base_output_dir, n_runs, warmup_runs).
    3. benchmark_all_methods(methods_dict, image, test_name).
    4. Вывести сводку: CPU_time, CUDA_time, Speedup.
    5. Вернуть DataFrame с результатами.
    ```

    Args:
        all_benchmark_methods: Полный словарь зарегистрированных методов.
        test_image: Тестовое изображение формы (H, W) или (H, W, 3).
        n_runs: Количество прогонов для замера времени.
        warmup_runs: Количество "прогревочных" прогонов.

    Returns:
        BenchmarkResult: DataFrame с колонками:
                        [method, device, mean_time, std_time, speedup].

    Note:
        - Использует эвристику `_filter_classical_methods` для отсева нейросетей.
        - Speedup рассчитывается как `CPU_time / CUDA_time`.
        - Если CUDA недоступен, бенчмарк всё равно выполнит CPU-часть.

    Example:
        ```python
        df = run_cpu_cuda_benchmark(cv2_methods, sklearn_methods, torch_methods, test_img)

        # Анализ результатов
        cuda_methods = df[df["device"] == "cuda"]
        print(f"Средний speedup: {cuda_methods['speedup'].mean():.2f}x")
        ```
    """
    print("\n" + "=" * 80)
    print("ЗАПУСК БЕНЧМАРКА: CPU vs CUDA")
    print("=" * 80)

    classical_only: SegmenterDict = _filter_classical_methods(all_benchmark_methods)

    print(f"Количество классических методов: {len(classical_only)}")

    # Бенчмарк CPU vs CUDA для классических методов
    benchmark: CpuCudaBenchmark = CpuCudaBenchmark(
        base_output_dir="./data/cpu_cuda_benchmark",
        n_runs=n_runs,
        warmup_runs=warmup_runs,
    )
    df_results: BenchmarkResult = benchmark.benchmark_all_methods(
        methods_dict=classical_only, image=test_image, test_name="multi_backend_full"
    )

    # Вывод сводки
    _print_cpu_cuda_summary(df_results)

    return df_results


# ──────────────────────────────────────────────────────────────────────
def _print_cpu_cuda_summary(df_results: BenchmarkResult) -> None:
    """Выводит сводку по бенчмарку CPU vs CUDA в табличном формате.

    Выполняет:
    1. **Проверка CUDA**: Пропуск при отсутствии GPU.
    2. **Группировка по методам**: Разделение результатов на CPU и CUDA.
    3. **Расчёт speedup**: Отношение времени CPU к времени CUDA.
    4. **Форматированный вывод**: Таблица с методом, временами и ускорением.

    Алгоритм:
    ```
    1. Если not torch.cuda.is_available() → вывести предупреждение и вернуть.
    2. Для каждого уникального метода в df_results:
       a. Извлечь cpu_data и cuda_data по колонке "device".
       b. Если обе выборки не пустые:
          → Вычислить cpu_time, cuda_time в миллисекундах.
          → Рассчитать speedup = cpu_time / cuda_time.
          → Вывести строку: метод, CPU=XXms, CUDA=YYms, Speedup=ZZx.
    3. Вывести разделитель в конце.
    ```

    Args:
        df_results: DataFrame с результатами бенчмарка (колонки: method, device, mean_time).

    Returns:
        None. Выводит отформатированную сводку в stdout.

    Note:
        - Speedup > 1 означает ускорение на CUDA, < 1 — замедление.
        - Методы без данных для CPU или CUDA пропускаются.
        - Времена конвертируются из секунд в миллисекунды для наглядности.

    Example:
        ```python
        _print_cpu_cuda_summary(benchmark_df)
        # otsu_thresholding_CV2           : CPU= 12.34ms, CUDA=  3.21ms, ⚡ Speedup=3.84x
        ```
    """
    print("\n" + "=" * 80)
    print("СВОДКА ПО БЕНЧМАРКУ CPU vs CUDA")
    print("=" * 80)

    if not torch.cuda.is_available():
        logger.warning("⚠️  CUDA недоступен, пропуск сравнения")
        return

    for method in df_results["method"].unique():
        method_data = df_results[df_results["method"] == method]
        cpu_data = method_data[method_data["device"] == "cpu"]
        cuda_data = method_data[method_data["device"] == "cuda"]

        if cpu_data.empty or cuda_data.empty:
            continue

        cpu_time: float = cpu_data["mean_time"].values[0] * 1000
        cuda_time: float = cuda_data["mean_time"].values[0] * 1000

        speedup: float = cpu_time / cuda_time
        print(f"{method:40s}: CPU={cpu_time:7.2f}ms, CUDA={cuda_time:7.2f}ms, ⚡ Speedup={speedup:.2f}x")

    print("=" * 80)


# ──────────────────────────────────────────────────────────────────────
def _filter_classical_methods(all_methods: SegmenterDict) -> SegmenterDict:
    """Фильтрует методы, исключая нейросетевые, для бенчмарка классических алгоритмов.

    Выполняет:
    1. **Определение ключевых слов**: Список подстрок, характерных для нейросетей.
    2. **Фильтрация по имени**: Исключение методов, содержащих ключевые слова.
    3. **Возврат отфильтрованного словаря**: Только классические методы.

    Алгоритм:
    ```
    1. Определить neural_keywords = ["neural", "segformer", "unet", "fpn", ...].
    2. Вернуть {name: seg for name, seg in all_methods.items()
                if not any(kw in name.lower() for kw in neural_keywords)}.
    ```

    Args:
        all_methods: Полный словарь зарегистрированных методов.

    Returns:
        SegmenterDict: Словарь только с классическими методами (OpenCV, Sklearn, Torch).

    Note:
        - Сравнение регистронезависимое благодаря `.lower()`.
        - Ключевые слова покрывают основные архитектуры нейросетей и библиотеки.
        - Методы с частичным совпадением (например, "neural_network_CV2") также исключаются.

    Example:
        ```python
        classical = _filter_classical_methods(all_methods)
        print(f"Классических методов: {len(classical)} из {len(all_methods)}")
        ```
    """
    neural_keywords: List[str] = [
        "neural",
        "segformer",
        "mask2former",
        "unet",
        "fpn",
        "psp",
        "fcn",
        "deeplab",
        "sam",
        "dpt",
        "upernet",
    ]

    return {name: seg for name, seg in all_methods.items() if not any(kw in name.lower() for kw in neural_keywords)}


# ──────────────────────────────────────────────────────────────────────
def get_device_capabilities() -> Dict[str, Any]:
    """Проверяет возможности CUDA для автоматического подбора настроек оптимизации.

    Выполняет:
    1. **Проверка доступности**: Возврат базового конфига при отсутствии CUDA.
    2. **Чтение свойств GPU**: Получение `compute capability` и имени устройства.
    3. **Определение поддержки**:
       - BF16: Ampere+ (≥8.0)
       - INT8: Volta+ (≥7.0)
    4. **Формирование словаря**: Возврат всех характеристик в едином формате.

    Алгоритм:
    ```
    1. Если not torch.cuda.is_available() → вернуть {"cuda": False, ...}.
    2. Получить props = torch.cuda.get_device_properties(0).
    3. bf16_support = props.major >= 8.
    4. int8_support = props.major >= 7.
    5. Вернуть словарь с флагами и именем устройства.
    ```

    Args:
        Нет аргументов (использует глобальный torch.cuda).

    Returns:
        Dict[str, Any]: Словарь с характеристиками:
                        - `cuda`: доступность CUDA.
                        - `bf16_support`, `int8_support`: поддержка форматов.
                        - `device_name`: имя GPU.

    Note:
        - BF16 требует архитектуры Ampere (RTX 30xx, A100) или новее.
        - INT8 квантование доступно на Volta (T4, V100) и новее.
        - При отсутствии CUDA все флаги устанавливаются в значения для CPU-режима.

    Example:
        ```python
        caps = get_device_capabilities()
        if caps["cuda"] and caps["bf16_support"]:
            print(f"Можно использовать bf16 на {caps['device_name']}")
        ```
    """
    if not torch.cuda.is_available():
        return {"cuda": False, "bf16_support": False, "int8_support": True}

    props = torch.cuda.get_device_properties(0)
    # BF16 поддерживается на Ampere (RTX 3000/A100) и новее (Compute Capability 8.0+)
    bf16_support: bool = props.major >= 8
    # Int8 квантование (INT8 Tensor Cores) поддерживается на Volta (T4/V100) и новее (Compute Capability 7.0+)
    int8_support: bool = props.major >= 7

    return {
        "cuda": True,
        "bf16_support": bf16_support,
        "int8_support": int8_support,
        "device_name": props.name,
    }


# ──────────────────────────────────────────────────────────────────────
def create_segmenter_config(method_name: str, device: Optional[str], **kwargs: Any) -> Any:
    """Фабрика для создания сегментера с нужными флагами оптимизации.

    Выполняет:
    1. **Инициализация**: Попытка создания `TorchSegmenter2` с переданными параметрами.
    2. **Обработка ошибок**: Логирование предупреждения при сбое.
    3. **Возврат результата**: Сегментер или `None` при ошибке.

    Алгоритм:
    ```
    1. try:
       → seg = TorchSegmenter2(method=method_name, device=device, **kwargs)
       → вернуть seg.
    2. except Exception as e:
       → print(f"[WARNING] Не удалось создать конфиг: {e}")
       → вернуть None.
    ```

    Args:
        method_name: Имя метода сегментации.
        device: Устройство для выполнения (`"cuda"`/`"cpu"`/`None`).
        **kwargs: Дополнительные параметры для инициализации сегментера.

    Returns:
        Any: Экземпляр `TorchSegmenter2` или `None` при ошибке создания.

    Note:
        - Ошибки не прерывают выполнение, что позволяет продолжать бенчмарк.
        - Предупреждения выводятся в stdout для отладки.
        - Поддерживает любые параметры, принимаемые конструктором `TorchSegmenter2`.

    Example:
        ```python
        seg = create_segmenter_config(
            "otsu_thresholding", device="cuda", precision="bf16", use_compile=True
        )
        if seg:
            mask = seg.segment(test_image)
        ```
    """
    try:
        seg: TorchSegmenter2 = TorchSegmenter2(method=method_name, device=device, **kwargs)
        return seg
    except Exception as e:
        logger.warning(f"[WARNING] Не удалось создать конфиг для {method_name} ({device}): {e}")
        return None


# ──────────────────────────────────────────────────────────────────────
def _get_device_metrics(device: str) -> Dict[str, Any]:
    """Возвращает словарь с характеристиками устройства и метриками памяти.

    Выполняет:
    1. **Проверка CUDA**: При `device=="cuda"` — сбор статистики через `torch.cuda`.
    2. **Сбор метрик**: Выделение, резервирование, пиковые значения памяти.
    3. **Форматирование**: Округление до 2 знаков для ГБ/МБ.
    4. **CPU-фоллбэк**: Возврат заглушек при отсутствии CUDA.

    Алгоритм:
    ```
    1. Если device == "cuda" и torch.cuda.is_available():
       → Собрать: GPU_Name, Compute_Cap, Total_VRAM_GB, Curr/Peak_Alloc_MB, Curr/Peak_Reserv_MB.
       → Округлить значения через round(..., 2).
    2. Иначе:
       → Вернуть словарь с "CPU", "N/A" и нулевыми метриками.
    3. Вернуть итоговый словарь.
    ```

    Args:
        device: Строка с названием устройства (`"cuda"` или `"cpu"`).

    Returns:
        Dict[str, Any]: Словарь с характеристиками:
                        - `GPU_Name`, `Compute_Cap`, `Total_VRAM_GB`
                        - `Curr_Alloc_MB`, `Peak_Alloc_MB`, `Curr_Reserv_MB`, `Peak_Reserv_MB`

    Note:
        - Метрики памяти в МБ, объём VRAM в ГБ для удобства чтения.
        - При отсутствии CUDA все метрики возвращаются как 0 или "N/A".
        - `Peak_*` метрики сбрасываются через `torch.cuda.reset_peak_memory_stats()`.

    Example:
        ```python
        metrics = _get_device_metrics("cuda")
        print(f"Пиковое выделение: {metrics['Peak_Alloc_MB']} MB")
        ```
    """
    if device == "cuda" and torch.cuda.is_available():
        return {
            "GPU_Name": torch.cuda.get_device_name(0),
            "Compute_Cap": torch.cuda.get_device_capability(0),
            "Total_VRAM_GB": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "Curr_Alloc_MB": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "Peak_Alloc_MB": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
            "Curr_Reserv_MB": round(torch.cuda.memory_reserved() / 1024**2, 2),
            "Peak_Reserv_MB": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
        }
    return {
        "GPU_Name": "CPU",
        "Compute_Cap": "N/A",
        "Total_VRAM_GB": "N/A",
        "Curr_Alloc_MB": 0,
        "Peak_Alloc_MB": 0,
        "Curr_Reserv_MB": 0,
        "Peak_Reserv_MB": 0,
    }


# ──────────────────────────────────────────────────────────────────────
def run_optimization_benchmarks(
    image_path: str,
    methods_list: List[str],
) -> Optional[pd.DataFrame]:
    """Запускает серию тестов производительности для списка методов с разными конфигурациями.

    Выполняет:
    1. **Диагностика системы**: Проверка поддержки BF16/INT8 через `get_device_capabilities`.
    2. **Подготовка конфигураций**: Матрица тестов (Baseline, Compile, BF16, Quantized).
    3. **Цикл по методам**: Для каждого метода — запуск всех конфигураций с замером времени.
    4. **Профилирование**: Вызов `profile_segmentation` с 100 запусками и 20 прогревочными.
    5. **Агрегация**: Формирование сводной таблицы с временем, speedup и памятью.
    6. **Визуализация**: Вывод отформатированной таблицы и сохранение в CSV.

    Алгоритм:
    ```
    1. Получить caps = get_device_capabilities(), вывести системную информацию.
    2. Загрузить изображение, конвертировать в numpy array.
    3. Сформировать configs: список словарей {name, device, kwargs} для тестов.
    4. Для каждого метода в methods_list:
       a. Для каждой конфигурации:
          → Создать сегментер через create_segmenter_config.
          → Запустить profile_segmentation(n_runs=100, warmup=20).
          → Записать время, speedup относительно baseline, метрики памяти.
    5. Создать DataFrame, отсортировать по [Method, Time].
    6. Вывести таблицу через tabulate, сохранить в CSV.
    ```

    Args:
        image_path: Путь к тестовому изображению.
        methods_list: Список имён методов для бенчмарка.

    Returns:
        Optional[pd.DataFrame]: Сводная таблица с колонками:
                               [Method, Config, Device, Time (ms), Speedup, Peak_Alloc_MB, ...]
                               или `None` при ошибке загрузки изображения.

    Note:
        - Профилирование использует 100 запусков для стабильности статистики.
        - Speedup рассчитывается относительно базовой конфигурации (FP32 Eager).
        - При ошибке для одной конфигурации тест продолжается для остальных.

    Example:
        ```python
        df = run_optimization_benchmarks(
            image_path="test.jpg",
            methods_list=["otsu_thresholding", "sobel_edge"]
        )
        if df is not None:
            print(df.groupby("Method")["Speedup"].max())
        ```
    """
    caps: Dict[str, Any] = get_device_capabilities()
    print(f"🖥️  System Info: {caps.get('device_name') if caps.get('cuda') else 'CPU'}")
    print(f" CUDA BF16 Support: {caps.get('bf16_support')} | INT8 Support: {caps.get('int8_support')}")
    print("=" * 80)

    # --- НАСТРОЙКИ ТЕСТА (МАТРИЦА) ---
    try:
        img: Image.Image = Image.open(image_path).convert("RGB")
        img_array: np.ndarray = np.array(img)
        print(f"✅ Изображение загружено: {img_array.shape}")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки изображения {image_path}: {e}")
        return None

    configs: List[Dict[str, Any]] = [
        {
            "name": "Baseline (FP32 Eager)",
            "device": "cuda",
            "kwargs": {"precision": "fp32", "use_compile": False},
        },
        {
            "name": "Optimized (FP32 Compile)",
            "device": "cuda",
            "kwargs": {
                "precision": "fp32",
                "use_compile": True,
                "compile_mode": "reduce-overhead",
            },
        },
        {
            "name": "Max Perf (FP32 Max-Autotune)",
            "device": "cuda",
            "kwargs": {
                "precision": "fp32",
                "use_compile": True,
                "compile_mode": "max-autotune",
            },
        },
    ]

    if caps.get("bf16_support"):
        configs.append(
            {
                "name": "BF16 Native (Ampere+)",
                "device": "cuda",
                "kwargs": {"precision": "bf16", "use_compile": False},
            }
        )
        configs.append(
            {
                "name": "BF16 + Compile",
                "device": "cuda",
                "kwargs": {
                    "precision": "bf16",
                    "use_compile": True,
                    "compile_mode": "reduce-overhead",
                },
            }
        )

    # CPU с квантованием
    configs.append(
        {
            "name": "CPU Int8 Quantized",
            "device": "cpu",
            "kwargs": {
                "precision": "fp32",
                "use_quantization": True,
                "use_compile": False,
            },
        }
    )

    all_results: List[Dict[str, Any]] = []

    for method in methods_list:
        print(f"\n🧪 Тестирование метода: {method.upper()}")
        print("-" * 40)

        baseline_time: Optional[float] = None

        for cfg in configs:
            print(f"   ▶ Запуск: {cfg['name']} ... ", end="", flush=True)

            seg: Optional[TorchSegmenter2] = create_segmenter_config(method, cfg["device"], **cfg["kwargs"])
            if seg is None:
                logger.error("❌ Пропуск")
                continue

            if cfg["device"] == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            try:
                stats: Dict[str, Any] = seg.profile_segmentation(segmenter=seg, image=img_array, n_runs=100, warmup=20)
                mean_time_ms: float = cast(float, stats["mean_time_s"]) * 1000
                print(f"⏱️ {stats['method']} | {mean_time_ms:.2f}ms | Device: {stats['device']}")

                dev_metrics: Dict[str, Any] = _get_device_metrics(cfg["device"])

                speedup: float = 1.0
                if baseline_time is None:
                    baseline_time = mean_time_ms
                elif mean_time_ms > 0:
                    speedup = baseline_time / mean_time_ms

                all_results.append(
                    {
                        "Method": method,
                        "Config": cfg["name"],
                        "Device": dev_metrics["GPU_Name"],
                        "Compute_Cap": dev_metrics["Compute_Cap"],
                        "Total_VRAM_GB": dev_metrics["Total_VRAM_GB"],
                        "Time (ms)": round(mean_time_ms, 2),
                        "Speedup": round(speedup, 2),
                        "Device1": cfg["device"],
                        "Params": cfg["kwargs"],
                        "Peak_Alloc_MB": dev_metrics["Peak_Alloc_MB"],
                        "Peak_Reserv_MB": dev_metrics["Peak_Reserv_MB"],
                    }
                )
                print(f"✅ {mean_time_ms:.2f} ms ({speedup:.2f}x) | Mem: {dev_metrics['Peak_Alloc_MB']:.1f}MB")

            except Exception as e:
                logger.error(f"❌ Ошибка профилирования: {e}")

    print("\n" + "=" * 80)
    print("📊 СВОДНАЯ ТАБЛИЦА ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 80)

    if all_results:
        df_summary: pd.DataFrame = pd.DataFrame(all_results)
        df_summary = df_summary.sort_values(["Method", "Time (ms)"], ascending=[True, True])
        display_cols: List[str] = [
            "Method",
            "Config",
            "Device",
            "Compute_Cap",
            "Total_VRAM_GB",
            "Time (ms)",
            "Speedup",
            "Peak_Alloc_MB",
            "Peak_Reserv_MB",
        ]
        for col in display_cols:
            if col not in df_summary.columns:
                df_summary[col] = "N/A" if "Cap" in col or "VRAM" in col else 0.0

        df_display: pd.DataFrame = df_summary[display_cols].copy()
        df_display["Time (ms)"] = df_summary["Time (ms)"].apply(lambda x: f"{x:.2f}")
        df_display["Speedup"] = df_summary["Speedup"].apply(lambda x: f"{x:.2f}x")
        df_display["Peak_Alloc_MB"] = df_summary["Peak_Alloc_MB"].apply(lambda x: f"{x:.1f}")
        df_display["Peak_Reserv_MB"] = df_summary["Peak_Reserv_MB"].apply(lambda x: f"{x:.1f}")
        df_display["Total_VRAM_GB"] = df_summary["Total_VRAM_GB"].apply(
            lambda x: f"{x:.1f}" if isinstance(x, float) else str(x)
        )
        df_display["Compute_Cap"] = df_summary["Compute_Cap"].apply(
            lambda x: f"{x[0]}.{x[1]}" if isinstance(x, tuple) else str(x)
        )

        print(tabulate(df_display, headers="keys", tablefmt="grid", showindex=False))

        csv_path: str = "optimization_benchmark.csv"
        df_summary.to_csv(csv_path, index=False)
        print(f"\n💾 Таблица сохранена: {csv_path}")
        return df_summary
    else:
        logger.warning("Нет данных для отображения.")
        return None


# ──────────────────────────────────────────────────────────────────────
def benchmark_with_baseline(
    segmenter: TorchSegmenter2,
    image: np.ndarray,
    configurations: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Сравнение различных конфигураций сегментера относительно базовой (baseline).

    Выполняет:
    1. **Запуск baseline**: Создание сегментера с параметрами по умолчанию (fp32, eager).
    2. **Замер базового времени**: 100 запусков через `profile_method`.
    3. **Тестирование конфигураций**: Для каждой — создание сегментера, замер, расчёт speedup.
    4. **Агрегация результатов**: Возврат словаря с временем и ускорением для каждой конфигурации.

    Алгоритм:
    ```
    1. Создать baseline_seg = TorchSegmenter2(method, device="cuda", precision="fp32", use_compile=False).
    2. baseline_time = baseline_seg.profile_method(image, n_runs=100)["mean_time_ms"].
    3. Для каждой (config_name, config_params) в configurations:
       → Создать seg = TorchSegmenter2(method, device="cuda", **config_params).
       → time_ms = seg.profile_method(image, n_runs=100)["mean_time_ms"].
       → speedup = baseline_time / time_ms (или inf при time_ms==0).
       → Добавить в results[config_name] = {time_ms, speedup}.
    4. Вернуть results.
    ```

    Args:
        segmenter: Исходный сегментер (используется только для извлечения `method`).
        image: Тестовое изображение в формате numpy array.
        configurations: Словарь конфигураций: `{имя: {параметры_для_конструктора}}`.

    Returns:
        Dict[str, Any]: Результаты сравнения:
                        - `baseline_fp32_eager`: время базовой конфигурации.
                        - `{config_name}`: {time_ms, speedup} для каждой тестовой конфигурации.

    Note:
        - Все конфигурации тестируются на одном изображении для консистентности.
        - `profile_method` автоматически выполняет прогрев перед замером.
        - При `time_ms == 0` speedup устанавливается в `float('inf')`.

    Example:
        ```python
        configs = {
            "bf16_compile": {"precision": "bf16", "use_compile": True},
            "fp16": {"precision": "fp16", "use_compile": False},
        }
        results = benchmark_with_baseline(seg, test_img, configs)
        print(f"BF16+Compile speedup: {results['bf16_compile']['speedup']:.2f}x")
        ```
    """
    results: Dict[str, Any] = {}

    baseline_seg: TorchSegmenter2 = TorchSegmenter2(
        method=segmenter.method,
        device="cuda",
        precision="fp32",
        use_compile=False,
    )
    # cast нужен, так как profile_method возвращает Dict[str, Any]
    baseline_time: float = cast(float, baseline_seg.profile_method(image, n_runs=100)["mean_time_ms"])
    results["baseline_fp32_eager"] = baseline_time

    for config_name, config_params in configurations.items():
        seg: TorchSegmenter2 = TorchSegmenter2(method=segmenter.method, device="cuda", **config_params)
        time_ms: float = cast(float, seg.profile_method(image, n_runs=100)["mean_time_ms"])
        results[config_name] = {
            "time_ms": time_ms,
            "speedup": baseline_time / time_ms if time_ms > 0 else float("inf"),
        }

    return results


# ──────────────────────────────────────────────────────────────────────
# СОХРАНЕНИЕ СТАРОЙ ВЕРСИИ ДЛЯ СРАВНЕНИЯ
# ──────────────────────────────────────────────────────────────────────
def main_legacy() -> Tuple[
    Optional[SegmentationTester],
    Optional[BenchmarkResult],
    Optional[SegmentationComparator],
]:
    """LEGACY версия main() — для сравнения с оптимизированной. Использует старые параметры TorchSegmenter без оптимизаций."""
    # Копия основной логики, но с созданием методов БЕЗ оптимизаций:
    # - Без precision параметра
    # - Без use_compile
    # - Без квантования
    # - Без профилирования

    # Для экономии места — просто вызов с флагом
    return main(use_optimizations=False)


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Основной тест
    print("ЗАПУСК ОСНОВНОГО ТЕСТА")
    print("=" * 60)
    tester, results, comparator = main()

    print("\n\nЗАПУСК ДОПОЛНИТЕЛЬНОГО ТЕСТА НЕЙРОСЕТЕВЫХ ВАРИАНТОВ")
    print("=" * 60)
    segmenter, detailed_result = test_neural_segmentation_variants()
