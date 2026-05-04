# main.py

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
from io import BytesIO
from pathlib import Path
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Any,
    Union,
    Literal,
    Callable,
    TypeVar,
    cast,
)
from matplotlib import colormaps
from tqdm import tqdm

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import requests
from PIL import Image
from huggingface_hub import hf_hub_download

# Локальные импорты
from segmenters.NeuralSegmenter import NeuralSegmenter
from segmenters.OpenCVSegmenter import OpenCVSegmenter
from segmenters.SklearnSegmenter import SklearnSegmenter
from segmenters.TorchSegmenter import TorchSegmenter
from segmenters.ModelTrainer import ModelTrainer, TrainingConfig, TrainingResult
from segmenters.NeuralModelFactory import NeuralModelFactory
from testing.SegmentationTester import SegmentationTester
from testing.SegmentationComparator import SegmentationComparator
from testing.SegmentationBenchmark import SegmentationBenchmark, export_comparison_table
from testing.TorchImplementationValidator import TorchImplementationValidator
from testing.BatchClassicTester import BatchClassicTester
from metrics.SegmentationMetrics import SegmentationMetrics, MetricsDict
from utils.warmup import SegmentationWarmUp
from utils.threshold_warmup import ThresholdWarmUp
from utils.strategies import _create_overlay_standalone, segment_image_unified

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES И КОНСТАНТЫ
# ──────────────────────────────────────────────────────────────────────
# Типы для изображений
ImageArray = npt.NDArray[np.uint8]
"""Тип для входного изображения: (H, W) или (H, W, 3), dtype=uint8."""

MaskArray = npt.NDArray[np.uint8]
"""Тип для бинарной маски: (H, W), dtype=uint8, значения {0, 255}."""

SegmenterDict = Dict[
    str, Union[OpenCVSegmenter, SklearnSegmenter, TorchSegmenter, NeuralSegmenter]
]
"""Словарь сегментеров: {имя_метода: экземпляр_сегментера}."""

TestImageEntry = Tuple[str, Image.Image, Optional[MaskArray]]
"""Элемент тестового изображения: (путь, PIL.Image, GT-маска или None)."""

TestImagesDict = Dict[str, TestImageEntry]
"""Словарь тестовых изображений: {имя: (путь, PIL.Image, GT-маска)}."""

type BenchmarkResult = pd.DataFrame
"""Результат бенчмарка: DataFrame с метриками и временем выполнения."""

# Константы конфигурации
ADE20K_REPO_ID: str = "hf-internal-testing/fixtures_ade20k"
"""ID репозитория с тестовыми данными ADE20K на HuggingFace."""

NUM_CLASSES_ADE20K: int = 150
"""Количество классов в датасете ADE20K."""

DEFAULT_IMAGE_SIZE: Tuple[int, int] = (512, 512)
"""Размер по умолчанию для ресайза изображений."""

# Глобальные настройки окружения
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")

num_classes: int = 150


# ──────────────────────────────────────────────────────────────────────
def main() -> Tuple[
    Optional[SegmentationTester],
    Optional[BenchmarkResult],
    Optional[SegmentationComparator],
]:
    """
    Основная точка входа фреймворка сегментации.

    Выполняет последовательное тестирование классических и нейросетевых методов:
    1. Инициализация окружения и проверка доступности CUDA.
    2. Загрузка тестовых изображений (с/без ground truth).
    3. Регистрация методов сегментации (OpenCV, Sklearn, Torch, Neural).
    4. Опциональный запуск бенчмарков:
       - Производительность (cold/hot запуски)
       - Валидация согласованности реализаций
       - Сравнение с ground truth
       - Массовое тестирование на датасете
    5. Генерация отчётов: таблицы, графики, HTML.

    Returns:
        Tuple[Optional[SegmentationTester], Optional[BenchmarkResult], Optional[SegmentationComparator]]:
            - `tester`: Экземпляр тестера с зарегистрированными методами (или None).
            - `results`: DataFrame с результатами бенчмарка (или None).
            - `comparator`: Экземпляр компаратора для матричных сравнений (или None).

    Note:
        - Функция управляется флагами `test_neural_logic` и `test_classic_logic`.
        - При ошибках в отдельных методах выполнение продолжается (graceful degradation).
        - Все результаты сохраняются в директорию `./data/` с автоматической структурой.

    Example:
        ```python
        if __name__ == "__main__":
            tester, results, comparator = main()

            if results is not None:
                print(f"Обработано {len(results)} методов")
                print(results.sort_values("IoU", ascending=False).head(10))
        ```
    """
    # ──────────────────────────────────────────────────────────────
    # 1. КОНФИГУРАЦИЯ И ИНИЦИАЛИЗАЦИЯ
    # ──────────────────────────────────────────────────────────────
    test_neural_logic: bool = False
    test_classic_logic: bool = True

    _log_environment_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Используемое устройство: {device}")

    print("=" * 60)
    print("ОБЪЕДИНЁННЫЙ ФРЕЙМВОРК СЕГМЕНТАЦИИ")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────────
    # 2. ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ
    # ──────────────────────────────────────────────────────────────
    print("\n1. Загрузка тестовых изображений...")
    test_images: TestImagesDict = load_test_images(use_image_with_mask=False)
    print(f"✅ Загружено изображений: {len(test_images)}")

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
    torch_methods: SegmenterDict = _create_torch_methods()

    if test_classic_logic:
        _register_classic_methods(tester, cv2_methods, sklearn_methods, torch_methods)

    # Очистка памяти перед загрузкой тяжёлых нейросетей
    _clear_gpu_memory()

    # ──────────────────────────────────────────────────────────────
    # 4. ОПЦИОНАЛЬНЫЕ БЛОКИ (вынесены в функции)
    # ──────────────────────────────────────────────────────────────

    # 4.1 Бенчмарк производительности (cold/hot)
    # if test_classic_logic:
    #     perf_results: Optional[pd.DataFrame] = run_performance_benchmark(
    #         tester=tester,
    #         test_images=test_images,
    #         n_runs=10,
    #         warmup_runs=10,
    #     )
    #     print(perf_results)

    # 4.2 Запускает тестирование нейросетевых методов сегментации.
    if test_neural_logic:
        _run_neural_segmentation_tests(tester, device)

    # 4.3 Нейросетевой бенчмарк
    if test_neural_logic:
        neural_results: Optional[Dict[str, Any]] = run_neural_segmentation_benchmark(
            device=device,
            num_classes=NUM_CLASSES_ADE20K,
        )
        print(neural_results)

    # # 4.4 Валидация реализаций
    # if test_classic_logic:
    #     validation_results: Optional[Dict[str, Any]] = run_implementation_validation(
    #         test_images=test_images,
    #         output_dir="./data/validation",
    #         image_name="mountain",
    #     )
    #     print(validation_results)

    # # 4.5 Матричное сравнение
    # if test_classic_logic:
    #     matrix_results: Optional[Dict[str, Any]] = run_matrix_comparison(
    #         test_images=test_images,
    #         cv2_methods=cv2_methods,
    #         sklearn_methods=sklearn_methods,
    #         torch_methods=torch_methods,
    #         reference_method="Otsu_Thresholding_Sklearn",
    #     )
    #     print(matrix_results)

    # # 4.6 Оценка против GT (опционально)
    # if test_classic_logic:
    #     gt_results: Optional[Dict[str, Any]] = run_ground_truth_evaluation(
    #         test_images=test_images,
    #         cv2_methods=cv2_methods,
    #         sklearn_methods=sklearn_methods,
    #         torch_methods=torch_methods,
    #     )
    #     print(gt_results)

    # 4.7 Обучение с аугментациями
    if test_neural_logic:
        aug_results: Optional[Dict[str, Any]] = run_augmentation_training_study(
            root_dir="./data/ade20k",
            checkpoint_dir="./models",
            device="cuda",
        )
        print(aug_results)

    # 4.8 Тестирование CPU/CUDA бенчмарка
    if test_classic_logic:
        # Выбираем тестовое изображение
        test_image = None
        for img_name, (img_path, img_pil, gt_mask) in tqdm(
            test_images.items(), desc="CUDA/CPU benchmark"
        ):
            test_image = np.array(img_pil)
            print(f"✅ Используем изображение: {img_name} ({test_image.shape})")
            break

        if test_image is not None:
            # Бенчмарк CPU vs CUDA для классических методов
            cpu_cuda_results: BenchmarkResult = run_cpu_cuda_benchmark(
                cv2_methods=cv2_methods,
                sklearn_methods=sklearn_methods,
                torch_methods=torch_methods,
                test_image=test_image,
            )

        print(cpu_cuda_results)

    # ──────────────────────────────────────────────────────────────
    # 5. МАССОВОЕ ТЕСТИРОВАНИЕ КЛАССИЧЕСКИХ МЕТОДОВ
    # ──────────────────────────────────────────────────────────────
    results_df: Optional[BenchmarkResult] = None
    if test_classic_logic:
        results_df = _run_batch_classic_testing()

    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"✓ Методов протестировано: {len(tester.methods)}")
    print(f"✓ Изображений обработано: {len(test_images)}")
    print("✓ Результаты в: ./data/")

    return tester, results_df, None


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
        print("🔥 CUDA available:")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        vram_gb: float = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   VRAM: {vram_gb:.1f} GB")
        print(
            f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("💻 CUDA not available, using CPU")


# ──────────────────────────────────────────────────────────────────────
def _create_cv2_methods() -> SegmenterDict:
    """
    Создаёт словарь методов сегментации на основе OpenCV.

    Returns:
        SegmenterDict: Словарь {имя_метода: OpenCVSegmenter}.
    """
    return {
        # --- Пороговые методы (Threshold) ---
        "Global_Threshold_CV2": OpenCVSegmenter("global_thresholding", threshold=0.5),
        "Otsu_Thresholding_CV2": OpenCVSegmenter("otsu_thresholding"),
        "Adaptive_Threshold_CV2": OpenCVSegmenter(
            "adaptive_thresholding", block_size=11, C=2
        ),
        "Niblack_Thresholding_CV2": OpenCVSegmenter(
            "threshold_niblack", window_size=15, k=-0.2
        ),
        "Sauvola_Thresholding_CV2": OpenCVSegmenter(
            "threshold_sauvola", window_size=15, k=0.5, r=128
        ),
        "Bernsen_Thresholding_CV2": OpenCVSegmenter(
            "threshold_bernsen", window_size=15, contrast_threshold=0.15
        ),
        "Phansalkar_Thresholding_CV2": OpenCVSegmenter(
            "threshold_phansalkar", window_size=15, k=0.25, r=128.0, m=0.5
        ),
        "Kittler_Illingworth_CV2": OpenCVSegmenter(
            "threshold_kittler_illingworth", num_bins=256
        ),
        "Kapur_Entropy_CV2": OpenCVSegmenter("threshold_entropy_kapur", num_bins=256),
        "Triangle_Threshold_CV2": OpenCVSegmenter("threshold_triangle", num_bins=256),
        "Multi_Otsu_CV2": OpenCVSegmenter("threshold_multi_otsu", n_thresholds=2),
        "Percentile_Threshold_CV2": OpenCVSegmenter(
            "threshold_percentile", percentile=90
        ),
        "Local_Contrast_CV2": OpenCVSegmenter(
            "threshold_local_contrast", window_size=15, contrast_factor=0.1
        ),
        # --- Граничные методы (Edge) ---
        "Sobel_CV2": OpenCVSegmenter("sobel_edge", threshold=0.1),
        "Canny_CV2": OpenCVSegmenter("canny_edge", low=0.1, high=0.3, sigma=1.0),
        "Prewitt_CV2": OpenCVSegmenter("prewitt_edge", threshold=0.1),
        "Scharr_CV2": OpenCVSegmenter("scharr_edge", threshold=0.1),
        "Roberts_Cross_CV2": OpenCVSegmenter("roberts_cross_edge", threshold=0.1),
        "LoG_CV2": OpenCVSegmenter("log_edge", sigma=1.0, threshold=0.01),
        "DoG_CV2": OpenCVSegmenter("dog_edge", sigma1=1.0, sigma2=2.0, threshold=0.01),
        "Marr_Hildreth_CV2": OpenCVSegmenter(
            "marr_hildreth_edge", sigma=1.5, threshold=0.01
        ),
        "Gradient_Mag_Dir_CV2": OpenCVSegmenter(
            "gradient_magnitude_direction", threshold=0.1
        ),
        "Phase_Congruency_CV2": OpenCVSegmenter(
            "phase_congruency_edge",
            nscales=4,
            norientations=4,
            min_wavelength=3,
            mult=2.0,
            sigma_onf=0.55,
            k_noise=2.0,
            threshold=0.5,
        ),
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
    """
    Создаёт словарь методов сегментации на основе scikit-learn.

    Returns:
        SegmenterDict: Словарь {имя_метода: SklearnSegmenter}.
    """
    return {
        # --- Пороговые методы (Threshold) ---
        "Global_Threshold_Sklearn": SklearnSegmenter(
            "global_thresholding", threshold=0.5
        ),
        "Otsu_Thresholding_Sklearn": SklearnSegmenter("otsu_thresholding"),
        "Adaptive_Threshold_Sklearn": SklearnSegmenter(
            "adaptive_thresholding", block_size=11, C=2
        ),
        "Niblack_Thresholding_Sklearn": SklearnSegmenter(
            "threshold_niblack", window_size=15, k=-0.2
        ),
        "Sauvola_Thresholding_Sklearn": SklearnSegmenter(
            "threshold_sauvola", window_size=15, k=0.5, r=128
        ),
        "Bernsen_Thresholding_Sklearn": SklearnSegmenter(
            "threshold_bernsen", window_size=15, contrast_threshold=0.15
        ),
        "Phansalkar_Thresholding_Sklearn": SklearnSegmenter(
            "threshold_phansalkar", window_size=15, k=0.25, r=128.0, m=0.5
        ),
        "Kittler_Illingworth_Sklearn": SklearnSegmenter(
            "threshold_kittler_illingworth", num_bins=256
        ),
        "Kapur_Entropy_Sklearn": SklearnSegmenter(
            "threshold_entropy_kapur", num_bins=256
        ),
        "Triangle_Threshold_Sklearn": SklearnSegmenter(
            "threshold_triangle", num_bins=256
        ),
        "Multi_Otsu_Sklearn": SklearnSegmenter("threshold_multi_otsu", n_thresholds=2),
        "Percentile_Threshold_Sklearn": SklearnSegmenter(
            "threshold_percentile", percentile=90
        ),
        "Local_Contrast_Sklearn": SklearnSegmenter(
            "threshold_local_contrast", window_size=15, contrast_factor=0.1
        ),
        # --- Граничные методы (Edge) ---
        "Sobel_Sklearn": SklearnSegmenter("sobel_edge", threshold=0.1),
        "Canny_Sklearn": SklearnSegmenter("canny_edge", low=0.1, high=0.3, sigma=1.0),
        "Prewitt_Sklearn": SklearnSegmenter("prewitt_edge", threshold=0.1),
        "Scharr_Sklearn": SklearnSegmenter("scharr_edge", threshold=0.1),
        "Roberts_Cross_Sklearn": SklearnSegmenter("roberts_cross_edge", threshold=0.1),
        "LoG_Sklearn": SklearnSegmenter("log_edge", sigma=1.0, threshold=0.01),
        "DoG_Sklearn": SklearnSegmenter(
            "dog_edge", sigma1=1.0, sigma2=2.0, threshold=0.01
        ),
        "Marr_Hildreth_Sklearn": SklearnSegmenter(
            "marr_hildreth_edge", sigma=1.5, threshold=0.01
        ),
        "Gradient_Mag_Dir_Sklearn": SklearnSegmenter(
            "gradient_magnitude_direction", threshold=0.1
        ),
        "Phase_Congruency_Sklearn": SklearnSegmenter(
            "phase_congruency_edge",
            nscales=4,
            norientations=4,
            min_wavelength=3,
            mult=2.0,
            sigma_onf=0.55,
            k_noise=2.0,
            threshold=0.5,
        ),
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
def _create_torch_methods() -> SegmenterDict:
    """
    Создаёт словарь методов сегментации на основе PyTorch.

    Returns:
        SegmenterDict: Словарь {имя_метода: TorchSegmenter}.
    """
    return {
        # --- Пороговые методы (Threshold) ---
        "Global_Threshold_Torch": TorchSegmenter("global_thresholding", threshold=0.5),
        "Otsu_Thresholding_Torch": TorchSegmenter("otsu_thresholding"),
        "Adaptive_Threshold_Torch": TorchSegmenter(
            "adaptive_thresholding", block_size=11, C=2
        ),
        "Niblack_Thresholding_Torch": TorchSegmenter(
            "threshold_niblack", window_size=15, k=-0.2
        ),
        "Sauvola_Thresholding_Torch": TorchSegmenter(
            "threshold_sauvola", window_size=15, k=0.5, r=128
        ),
        "Bernsen_Thresholding_Torch": TorchSegmenter(
            "threshold_bernsen", window_size=15, contrast_threshold=0.15
        ),
        "Phansalkar_Thresholding_Torch": TorchSegmenter(
            "threshold_phansalkar", window_size=15, k=0.25, r=128.0, m=0.5
        ),
        "Kittler_Illingworth_Torch": TorchSegmenter(
            "threshold_kittler_illingworth", num_bins=256
        ),
        "Kapur_Entropy_Torch": TorchSegmenter("threshold_entropy_kapur", num_bins=256),
        "Triangle_Threshold_Torch": TorchSegmenter("threshold_triangle", num_bins=256),
        "Multi_Otsu_Torch": TorchSegmenter("threshold_multi_otsu", n_thresholds=2),
        "Percentile_Threshold_Torch": TorchSegmenter(
            "threshold_percentile", percentile=90
        ),
        "Local_Contrast_Torch": TorchSegmenter(
            "threshold_local_contrast", window_size=15, contrast_factor=0.1
        ),
        # --- Граничные методы (Edge) ---
        "Sobel_Torch": TorchSegmenter("sobel_edge", threshold=0.1),
        "Canny_Torch": TorchSegmenter("canny_edge", low=0.1, high=0.3, sigma=1.0),
        "Prewitt_Torch": TorchSegmenter("prewitt_edge", threshold=0.1),
        "Scharr_Torch": TorchSegmenter("scharr_edge", threshold=0.1),
        "Roberts_Cross_Torch": TorchSegmenter("roberts_cross_edge", threshold=0.1),
        "LoG_Torch": TorchSegmenter("log_edge", sigma=1.0, threshold=0.01),
        "DoG_Torch": TorchSegmenter("dog_edge", sigma1=1.0, sigma2=2.0, threshold=0.01),
        "Marr_Hildreth_Torch": TorchSegmenter(
            "marr_hildreth_edge", sigma=1.5, threshold=0.01
        ),
        "Gradient_Mag_Dir_Torch": TorchSegmenter(
            "gradient_magnitude_direction", threshold=0.1
        ),
        "Phase_Congruency_Torch": TorchSegmenter(
            "phase_congruency_edge",
            nscales=4,
            norientations=4,
            min_wavelength=3,
            mult=2.0,
            sigma_onf=0.55,
            k_noise=2.0,
            threshold=0.5,
        ),
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
def _register_classic_methods(
    tester: SegmentationTester,
    cv2_methods: SegmenterDict,
    sklearn_methods: SegmenterDict,
    torch_methods: SegmenterDict,
) -> None:
    """
    Регистрирует классические методы в тестере с обработкой ошибок.

    Args:
        tester: Экземпляр SegmentationTester для регистрации.
        cv2_methods: Словарь методов OpenCV.
        sklearn_methods: Словарь методов scikit-learn.
        torch_methods: Словарь методов PyTorch.
    """
    all_methods: SegmenterDict = {**cv2_methods, **sklearn_methods, **torch_methods}

    for name, segmenter in all_methods.items():
        try:
            tester.add_method(name, segmenter)
            print(f"   ✅ {name}")
        except Exception as e:
            print(f"   ⚠️ Не удалось добавить {name}: {e}")


# ──────────────────────────────────────────────────────────────────────
def _clear_gpu_memory() -> None:
    """Очищает память GPU и вызывает сборщик мусора."""
    print("🧹 Очистка памяти CUDA перед загрузкой тяжелой модели...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# ──────────────────────────────────────────────────────────────────────
def _run_neural_segmentation_tests(
    tester: SegmentationTester, device: torch.device
) -> None:
    """
    Запускает тестирование нейросетевых методов сегментации.

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
    """
    Загружает и регистрирует одну нейросетевую модель.

    Args:
        tester: Экземпляр тестера.
        config: Конфигурация модели.
        device: Устройство для выполнения.
    """
    try:
        if "checkpoint_path" in config and not os.path.exists(
            config["checkpoint_path"]
        ):
            print(
                f"   ⚠️ {config['name']} - чекпоинт не найден: {config['checkpoint_path']}"
            )
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
                if k
                not in ["name", "type", "checkpoint_path", "local_path", "model_name"]
            },
        )
        tester.add_method(config["name"], segmenter)
        print(f"   ✅ {config['name']}")
        segmenter.get_class_info()

    except Exception as e:
        print(f"  ⚠️ Нейросетевая сегментация недоступна: {e}")
        print(f"   ❌ {config['name']}: {e}")
        print(traceback.format_exc())
        if os.getenv("DEBUG", "0") == "1":
            traceback.print_exc()

    print(f"\nВсего методов загружено: {len(tester.methods)}")

    # # ========== ВАРИАНТ 2: Через YAML конфиг ==========
    print("\n=== ВАРИАНТ 2: Через YAML конфиг ===")
    _ = NeuralSegmenter(
        model_type="segformer",
        variant="b5",  # ← Берётся из configs/neural_models.yaml
        num_classes=num_classes,
    )

    # # ========== ВАРИАНТ 3: Factory + конфиг ==========
    print("\n=== ВАРИАНТ 3: Factory метод ===")
    _, _, model_type = NeuralModelFactory.create_model_from_config(
        model_type="segformer", variant="b2", device="cuda"  # ← Берётся из конфига
    )

    # # ========== ВАРИАНТ 4: Обученная модель с чекпоинтом ==========
    print("\n=== ВАРИАНТ 4: Обученная модель ===")
    _ = NeuralSegmenter(
        model_type="unet_smp",
        encoder_name="resnet34",  # ← Можно из конфига
        checkpoint_path="./models/unet_ade20k_best.pth",
        num_classes=num_classes,
    )

    # # ========== ВАРИАНТ 5: Конфиг обучения ==========
    print("\n=== ВАРИАНТ 5: Конфиг обучения ===")
    training_config: Dict[str, Any] = NeuralModelFactory.get_training_config("ade20k")
    print(f"Batch size: {training_config['batch_size']}")
    print(f"Epochs: {training_config['epochs']}")
    print(f"LR: {training_config['lr']}")

    # # ========== ВАРИАНТ 6: Конфиг метрик ==========
    print("\n=== ВАРИАНТ 6: Конфиг метрик ===")
    metrics_config: Dict[str, Any] = NeuralModelFactory.get_metrics_config()
    print(f"Threshold: {metrics_config['threshold']}")
    print(f"Include Hausdorff: {metrics_config['include_hausdorff']}")

    # # ========== ВАРИАНТ 7: Массовая загрузка из конфига ==========
    print("\n=== ВАРИАНТ 7: Массовая загрузка ===")
    neural_models_config: List[Dict[str, str]] = [
        {"name": "SegFormer_B5", "type": "segformer", "variant": "b5"},
        {"name": "SegFormer_B2", "type": "segformer", "variant": "b2"},
        {"name": "Mask2Former", "type": "mask2former", "variant": "swin_base"},
        {"name": "U-Net", "type": "unet_smp", "encoder_name": "resnet34"},
        {
            "name": "FPN_MiT",
            "type": "fpn_smp",
            "encoder_name": "mit_b5",
            "checkpoint_path": "./models/fpn_mit_b5_ade20k_best_200_epochs.pth",
        },
    ]

    for config in neural_models_config:
        try:
            # Проверяем есть ли variant (для HF моделей)
            if "variant" in config:
                segmenter = NeuralSegmenter(
                    model_type=config["type"],
                    variant=config["variant"],  # ← Из конфига
                    num_classes=num_classes,
                )
            # Или checkpoint_path (для обученных)
            elif "checkpoint_path" in config:
                segmenter = NeuralSegmenter(
                    model_type=config["type"],
                    encoder_name=config.get("encoder_name"),
                    checkpoint_path=config["checkpoint_path"],
                    num_classes=num_classes,
                )
            # Или прямое имя модели
            else:
                segmenter = NeuralSegmenter(
                    model_type=config["type"],
                    encoder_name=config.get("encoder_name"),
                    num_classes=num_classes,
                )

            print(f"   ✅ {config['name']}")
        except Exception as e:
            print(f"   ❌ {config['name']} - {e}")


# ──────────────────────────────────────────────────────────────────────
def _run_batch_classic_testing() -> Optional[BenchmarkResult]:
    """
    Запускает массовое тестирование классических методов на датасете.

    Returns:
        Optional[BenchmarkResult]: DataFrame с результатами или None при ошибке.
    """
    print("\n" + "=" * 80)
    print("🚀 ЗАПУСК МАССОВОГО ТЕСТИРОВАНИЯ КЛАССИЧЕСКИХ МЕТОДОВ")
    print("=" * 80)

    try:
        # Конфигурация теста
        batch_tester: BatchClassicTester = BatchClassicTester(
            ade20k_root="./data/ade20k/ADEChallengeData2016",
            output_dir="./data/batch_classic_test",
            split="validation",  # или "training"
            max_images=50,  # Лимит для быстрого теста (None = все)
            image_size=DEFAULT_IMAGE_SIZE,
            save_masks=True,
            mask_sample_rate=1.0,  # 20% изображений
            max_mask_samples_per_method=50,  # максимум 2 образца на метод
            save_visualizations=True,  # генерировать comparison.png
        )

        print("⏳ Запуск массового тестирования...")
        results_df: BenchmarkResult = batch_tester.run_batch_test()

        # Сохранение и визуализация
        # Сохранение результатов
        batch_tester.save_results(results_df)
        # Построение графиков
        batch_tester.plot_results(results_df)

        print(f"\n🎭 Маски сохранены в: {batch_tester.masks_dir}")
        print("📁 Структура: masks/{pair}/{method}/{image}/")

        batch_tester.print_summary(results_df)

        _print_batch_test_summary(results_df)

        print(f"\n💾 Все результаты сохранены в: {batch_tester.output_dir}")

        return results_df

    except Exception as e:
        print(f"❌ Ошибка в массовом тестировании: {e}")
        traceback.print_exc()
        return None


# ──────────────────────────────────────────────────────────────────────
def _print_batch_test_summary(results_df: BenchmarkResult) -> None:
    """
    Выводит сводную статистику по результатам массового тестирования.

    Args:
        results_df: DataFrame с результатами бенчмарка.
    """
    print("\n" + "=" * 80)
    print("📊 СВОДКА ПО МАССОВОМУ ТЕСТИРОВАНИЮ")
    print("=" * 80)

    print("\n🏆 Топ-5 методов по IoU:")
    for i, row in results_df.head(5).iterrows():
        print(
            f"   {i + 1}. {row['Method']}: IoU={row['iou_mean']:.4f} ± {row['iou_std']:.4f}"
        )

    print("\n⚡ Топ-5 самых быстрых методов:")
    fast_df: BenchmarkResult = (
        results_df.dropna(subset=["time_mean_s"]).sort_values("time_mean_s").head(5)
    )
    for i, row in fast_df.iterrows():
        print(f"   {i + 1}. {row['Method']}: {row['time_mean_s'] * 1000:.1f} мс")

    print("\n❌ Методы с наибольшим числом ошибок:")
    error_df: BenchmarkResult = (
        results_df[results_df["error_count"] > 0]
        .sort_values("error_count", ascending=False)
        .head(5)
    )
    if not error_df.empty:
        for i, row in error_df.iterrows():
            print(
                f"   {row['Method']}: {row['error_count']} ошибок ({row['error_rate'] * 100:.1f}%)"
            )
    else:
        print("   Нет ошибок!")


# ──────────────────────────────────────────────────────────────────────
def prepare_mask_for_overlay(mask_input: Union[Image.Image, npt.NDArray]) -> MaskArray:
    """
    Конвертирует входную маску в 2D numpy array для наложения на изображение.

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
    mask: npt.NDArray = (
        np.array(mask_input)
        if isinstance(mask_input, Image.Image)
        else np.asarray(mask_input)
    )

    # Обработка многоканальных масок
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask.squeeze(2)
        elif mask.shape[2] == 3:
            print("⚠️  Обнаружена RGB маска, используется первый канал")
            mask = mask[:, :, 0]
        else:
            raise ValueError(f"Неподдерживаемая форма маски: {mask.shape}")
    elif mask.ndim > 3:
        mask = np.squeeze(mask)

    # Финальная валидация
    if mask.ndim != 2:
        raise ValueError(f"Маска должна быть 2D после обработки, получено {mask.ndim}D")

    return cast(MaskArray, mask)


# ──────────────────────────────────────────────────────────────────────
def run_performance_benchmark(
    tester: SegmentationTester,
    test_images: TestImagesDict,
    n_runs: int = 10,
    warmup_runs: int = 10,
    output_dir: str = "./data/performance_benchmark",
) -> Optional[pd.DataFrame]:
    """
    Запуск бенчмарка производительности с сравнением cold/hot запусков.

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
    for img_name, (img_path, img_pil, gt_mask) in tqdm(
        test_images.items(), desc="Cold benchmark"
    ):
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

        warmup_utility: SegmentationWarmUp = SegmentationWarmUp(
            n_warmup_runs=warmup_runs
        )

        # Прогрев всех сегментеров
        warmup_results: Dict[str, Any] = warmup_utility.warmup_all_segmenters(
            segmenters_dict=tester.methods,
            image=first_img_array,
            verbose=True,
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
            torch.cuda.synchronize()

    # ──────────────────────────────────────────────────────
    # ЭТАП 3: Hot benchmark (после прогрева) + анализ
    # ──────────────────────────────────────────────────────
    print("\n🔹 Этап 3: Hot benchmark + анализ...")
    print("\n3.1. Бенчмарк производительности после warm up...")

    for img_name, (img_path, img_pil, gt_mask) in tqdm(
        test_images.items(), desc="Hot benchmark"
    ):
        print(f"\n--- Обработка: {img_name} ---")
        img_array = np.array(img_pil)

        df_hot: BenchmarkResult = tester.benchmark_methods(
            image=img_array,
            n_runs=n_runs,
            test_name=f"benchmark_{img_name}_hot",
            save_results=True,
            force_warmup=False,
            ground_truth=gt_mask,
        )

        # OLD LOGIC

        # comparison: pd.DataFrame = pd.DataFrame(
        #     {
        #         "method": df_cold["Method"],
        #         "image": img_name,
        #         "cold_mean_ms": df_cold["Mean_Time_s"] * 1000,
        #         "hot_mean_ms": df_hot["Mean_Time_s"] * 1000,
        #     }
        # )
        # comparison["speedup"] = comparison.apply(
        #     lambda row: (
        #         row["cold_mean_ms"] / row["hot_mean_ms"]
        #         if row["hot_mean_ms"] > 0
        #         else float("inf")
        #     ),
        #     axis=1,
        # )

        df_cold_loaded: Optional[pd.DataFrame] = cold_dfs.get(img_name)

        # Fallback: загрузка из файла, если нет в памяти
        if df_cold_loaded is None:
            cold_dir = tester.get_benchmark_path(
                f"benchmark_{img_name}_cold", "statistics"
            )
            if cold_dir:
                cold_csv = Path(cold_dir) / "benchmark_results.csv"
                if cold_csv.exists():
                    df_cold_loaded = pd.read_csv(cold_csv)
                    print(f"   ✅ Загружен cold-файл из диска: {cold_csv}")
            else:
                print(f"   ⚠️  CSV не найден в {cold_csv}")

        if df_cold_loaded is None:
            print(f"   ❌ Пропускаем сравнение для {img_name}: нет cold-данных")
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
            speedup_str: str = (
                f"{row['speedup']:.2f}x" if not np.isinf(row["speedup"]) else "∞"
            )
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
        avg_speedup: pd.Series = (
            summary.groupby("method")["speedup"].mean().sort_values(ascending=False)
        )

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
    """
    Сравнивает результаты cold и hot бенчмарков.

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
        lambda row: (
            row["cold_mean_ms"] / row["hot_mean_ms"]
            if row["hot_mean_ms"] > 0
            else float("inf")
        ),
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
    """
    Запуск бенчмарка нейросетевых моделей сегментации на датасете ADE20K.

    Выполняет:
    1. Загрузку тестового изображения и ground truth из HuggingFace.
    2. Инициализацию `SegmentationBenchmark` с 11+ предобученными моделями.
    3. Запуск сравнения моделей через `benchmark.compare()`.
    4. Генерацию визуализаций, метрик и отчётов (CSV, JSON, Markdown, LaTeX).

    Поддерживаемые модели:
    - **Transformers**: SegFormer (B2/B5), Mask2Former, OneFormer, MaskFormer
    - **SMP**: U-Net, FPN, PSPNet, DeepLabV3+ (с энкодерами MiT-B5/ResNet)
    - **TorchVision**: FCN, Mask R-CNN, SegNet
    - **SAM**: MobileSAM, SAM2
    - **Другие**: DPT-Large, UPerNet

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
        - Для экономии памяти модели загружаются последовательно с `torch.cuda.empty_cache()`.
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

    image_path: str = hf_hub_download(
        repo_id=repo_id, filename="ADE_val_00000001.jpg", repo_type="dataset"
    )
    original_img: Image.Image = Image.open(image_path)
    original_img.save(Path(output_dir) / "original_image.jpg")

    mask_path: str = hf_hub_download(
        repo_id=repo_id, filename="ADE_val_00000001.png", repo_type="dataset"
    )
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
    processor_maskformer = MaskFormerImageProcessor.from_pretrained(
        model_name_maskformer
    )
    model_maskformer = (
        MaskFormerForInstanceSegmentation.from_pretrained(model_name_maskformer)
        .to(device)
        .eval()
    )
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
    )
    result_mf_ade.save("./data/ade20k_test_trained/segmented_maskformer_ade_0.jpg")

    maskformer_manual_ade_result: Dict[str, Any] = {
        "model": "maskformer",
        "overlay": result_mf_ade,
        "mask": result_mf_ade_results.get("mask"),
        "inference_time_ms": result_mf_ade_results.get("inference_time_ms", 0),
        "metrics": result_mf_ade_results.get("metrics", {}),
        "image_size": original_img.size[::-1],
        "output_shape": result_mf_ade_results.get("mask", np.array([])).shape,
        "unique_classes": len(
            np.unique(result_mf_ade_results.get("mask", np.array([])))
        ),
    }

    del model_maskformer, processor_maskformer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    print(
        f"✅ MaskFormer готов. VRAM освобождена: {torch.cuda.memory_allocated() / 1024**2:.1f} MB"
    )
    class_names_dict: Dict[int, str] = NeuralSegmenter.get_ade_class_names()
    class_names: Optional[List[str]] = [
        class_names_dict.get(i, f"Class {i}") for i in range(num_classes)
    ]
    benchmark: SegmentationBenchmark = SegmentationBenchmark(
        device=str(device),
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
    summary: pd.DataFrame = pd.DataFrame(summary_ade).T.sort_values(
        "mIoU", ascending=False
    )

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
    latex_table: str = benchmark.export_latex_table(
        caption="Comprehensive Semantic Segmentation Benchmark on ADE20K"
    )
    with open(Path(output_dir) / "benchmark_table.tex", "w") as f:
        f.write(latex_table)
    print(latex_table)

    if "segformer" in summary.index and "segformer_b2" in summary.index:
        print("\n" + "=" * 70)
        print("SEGFORMER SPEED-ACCURACY TRADEOFF")
        print("=" * 70)
        sf: pd.DataFrame = summary.loc[
            ["segformer", "segformer_b2"], ["mIoU", "time_ms"]
        ]
        sf["mIoU"] = sf["mIoU"] * 100
        print(sf.to_string(float_format="%.2f"))
        print(
            f"\n💡 B2 is {summary.loc['segformer', 'time_ms'] / summary.loc['segformer_b2', 'time_ms']:.1f}x faster with {summary.loc['segformer', 'mIoU'] - summary.loc['segformer_b2', 'mIoU']:.1f} pp mIoU drop"
        )

    print("\n" + "=" * 70)
    print("✅ BENCHMARK COMPLETE — Results saved to 'ade20k_11models_comprehensive/'")
    print("=" * 70)

    # Сводная таблица результатов
    table: pd.DataFrame = export_comparison_table(
        benchmark, str(Path(output_dir) / "model_comparison.md")
    )
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
        results_map["unet_smp"],  # U_Net
        results_map["deeplab_tv"],  # DeepLabV3+
        results_map["sam"],  # MobileSAM
        results_map["sam2"],  # SAM2
        results_map["dpt"],  # DPT-Large
        results_map["upernet"],  # UPerNet
        results_map["segformer_b2"],  # SegFormer-B2
        results_map["fpn_mit"],  # FPN + MiT-B5
        results_map["psp_mit"],  # PSPNet + MiT-B5
        results_map["maskformer"],  # MaskFormer
        results_map["fcn_tv"],  # FCN ResNet-50
        results_map["maskrcnn_tv"],  # Mask R-CNN
        results_map["segnet"],  # SegNet
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
        "./data/ade20k_test_trained/segmentation_results_ade.png",
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
    """
    Загружает модели в бенчмарк с обработкой ошибок.

    Args:
        benchmark: Экземпляр `SegmentationBenchmark`.
        output_dir: Директория для логирования.
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
            print(f"   ⚠️  {model_config.get('method', 'unknown')}: {e}")
            if os.getenv("DEBUG", "0") == "1":
                traceback.print_exc()

        # Очистка памяти после каждой модели
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ──────────────────────────────────────────────────────────────────────
def _generate_neural_benchmark_plots(
    benchmark: SegmentationBenchmark, output_dir: str
) -> None:
    """
    Генерирует визуализации для нейросетевого бенчмарка.

    Args:
        benchmark: Экземпляр `SegmentationBenchmark`.
        output_dir: Директория для сохранения графиков.
    """
    plots_dir: Path = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Все метрики
    print("\n📊 Generating visualizations...")
    benchmark.plot_all_metrics(figsize=(15, 5))
    plt.savefig(plots_dir / "all_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Сравнение по mIoU
    benchmark.plot_comparison_chart(
        "mIoU", title="ADE20K: Mean IoU Comparison", figsize=(12, 6)
    )
    plt.savefig(plots_dir / "miou_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Сравнение по времени
    benchmark.plot_comparison_chart(
        "time_ms", title="Inference Time Comparison (ms)", figsize=(12, 6)
    )
    plt.savefig(plots_dir / "time_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Per-class IoU (топ-20 классов)
    benchmark.plot_per_class_iou(top_k=20)
    plt.savefig(plots_dir / "per_class_iou.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Confusion matrix для лучшей модели
    best_model: str = (
        benchmark.get_summary_dataframe().sort_values("mIoU", ascending=False).index[0]
    )
    benchmark.plot_confusion_matrix(best_model, normalize="true")
    plt.savefig(
        plots_dir / f"confusion_matrix_{best_model}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # 6. Сводный график
    benchmark.plot_summary(metrics=["mIoU", "pixel_acc", "time_ms"])
    plt.savefig(plots_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Графики сохранены в: {plots_dir}")


# ──────────────────────────────────────────────────────────────────────
def run_implementation_validation(
    test_images: TestImagesDict,
    output_dir: str = "./data/validation",
    image_name: str = "mountain",
) -> Optional[Dict[str, Any]]:
    """
    Валидация согласованности реализаций методов через TorchImplementationValidator.

    Выполняет:
    1. Инициализацию валидатора с указанной директорией вывода.
    2. Запуск `validate_all_methods()` на выбранном изображении.
    3. Генерацию отчёта валидации (статусы, метрики, визуализации).
    4. (Опционально) Создание бенчмарк-отчёта на основе результатов валидации.

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
        - Валидация выполняется только для классических методов (не нейросетевых).
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
        print(f"❌ Изображение '{image_name}' не найдено в test_images")
        print(f"   Доступные: {list(test_images.keys())}")
        return None

    img_path, img_pil, gt_mask = test_images[image_name]
    img_array: ImageArray = np.array(img_pil)

    print(f"\n🔹 Валидация на изображении: {image_name} ({img_array.shape})")

    # ──────────────────────────────────────────────────────
    # 1. ИНИЦИАЛИЗАЦИЯ ВАЛИДАТОРА
    # ──────────────────────────────────────────────────────
    validator: TorchImplementationValidator = TorchImplementationValidator(
        output_dir=output_dir
    )

    # ──────────────────────────────────────────────────────
    # 2. ЗАПУСК ВАЛИДАЦИИ
    # ──────────────────────────────────────────────────────
    print("\n🔹 Запуск валидации методов...")

    try:
        # test_images['ade20k_sample'][0]
        # test_images['countryside'][0]
        # all_results = validator.validate_all_methods(test_images["mountain"][0])
        all_results: Dict[str, Any] = validator.validate_all_methods(img_array)
        print(f"   ✅ Валидировано: {len(all_results)} методов")
    except Exception as e:
        print(f"❌ Ошибка валидации: {e}")
        traceback.print_exc()
        return None

    # ──────────────────────────────────────────────────────
    # 3. ГЕНЕРАЦИЯ ОТЧЁТА
    # ──────────────────────────────────────────────────────
    print("\n🔹 Генерация отчёта...")

    try:
        validator.generate_validation_report(all_results)
        print(f"   ✅ Отчёт: {validator.output_dir}/validation_report.md")
        print(f"\n✅ Все результаты сохранены в: {validator.output_dir}")
    except Exception as e:
        print(f"⚠️  Ошибка генерации отчёта: {e}")

    # ──────────────────────────────────────────────────────
    # 4. БЕНЧМАРК-ОТЧЁТ (опционально)
    # ──────────────────────────────────────────────────────
    benchmark_df: Optional[pd.DataFrame] = None
    try:
        benchmark_dir: str = os.path.join(output_dir, "benchmark")
        benchmark_df = validator.generate_benchmark_report_from_validation(
            all_results, output_dir=benchmark_dir
        )
        print(f"   ✅ Бенчмарк-отчёт: {benchmark_dir}")
    except Exception as e:
        print(f"⚠️  Ошибка генерации бенчмарка: {e}")

    # ──────────────────────────────────────────────────────
    # 5. СВОДКА
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 СВОДКА ПО ВАЛИДАЦИИ")
    print("=" * 70)

    if all_results:
        # Статистика по статусам
        statuses: List[str] = [
            result.get("status", "UNKNOWN") for result in all_results.values()
        ]
        from collections import Counter

        status_counts: Counter = Counter(statuses)

        print(f"\n📈 Распределение статусов:")
        for status, count in status_counts.most_common():
            emoji = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}.get(status, "❓")
            print(
                f"   {emoji} {status}: {count} методов ({count/len(statuses)*100:.1f}%)"
            )

        # Топ-5 по согласованности (IoU)
        print(f"\n🏆 Топ-5 по IoU (согласованность):")
        sorted_results: List[Tuple[str, Any]] = sorted(
            all_results.items(),
            key=lambda x: x[1].get("metrics", {}).get("iou", 0),
            reverse=True,
        )
        for i, (method, result) in enumerate(sorted_results[:5], 1):
            iou: float = result.get("metrics", {}).get("iou", 0)
            print(f"   {i}. {method}: IoU = {iou:.4f}")

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
    reference_method: str = "Otsu_Thresholding_Sklearn",
) -> Optional[Dict[str, Any]]:
    """
    Матричное и пакетное сравнение методов сегментации.

    Выполняет три типа сравнений для каждого изображения:
    1. **All-vs-All**: Матрица попарных сравнений всех методов.
    2. **Batch comparison**: Сравнение всех методов относительно референсного.
    3. **Pairwise comparison**: Детальное сравнение двух конкретных реализаций.

    Поддерживаемые типы сравнений:
    - `all_vs_all`: Полная матрица (метод × метод) с метриками согласованности.
    - `batch_vs_reference`: Все методы против одного референсного (по умолчанию Otsu Sklearn).
    - `pairwise`: Сравнение двух конкретных реализаций (например, CV2 vs Sklearn Otsu).

    Args:
        test_images: Словарь тестовых изображений `{имя: (путь, PIL.Image, GT)}`.
        cv2_methods: Словарь методов OpenCV.
        sklearn_methods: Словарь методов scikit-learn.
        torch_methods: Словарь методов PyTorch.
        output_dir: Базовая директория для сохранения результатов.
        reference_method: Ключ референсного метода для batch-сравнения.

    Returns:
        Optional[Dict[str, Any]]: Словарь с результатами:
                                  - `all_vs_all_results`: Результаты матричного сравнения
                                  - `batch_results`: DataFrame batch-сравнения
                                  - `pairwise_results`: Результаты попарных сравнений
                                  или `None` при ошибке.

    Note:
        - Референсный метод должен присутствовать в объединённом словаре методов.
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

    # Подготовка конфигурации для сравнения
    methods_config_list: List[Dict[str, Any]] = [
        {
            "name": name,
            "segmenter": segmenter,
            # "type": "custom"
        }
        for name, segmenter in all_segmenters.items()
    ]

    # Референсный сегментер
    if reference_method not in all_segmenters:
        print(f"❌ Референсный метод '{reference_method}' не найден")
        print(f"   Доступные: {list(all_segmenters.keys())[:10]}...")
        return None

    ref_segmenter = all_segmenters[reference_method]

    # Для pairwise: сравнение двух реализаций Оцу
    original_segmenter = cv2_methods.get("Otsu_Thresholding_CV2")
    if original_segmenter is None:
        print("⚠️  Не найден Otsu_CV2 для pairwise сравнения")
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
    for img_name, (img_path, img_pil, gt_mask) in tqdm(
        test_images.items(), desc="Matrix Comparison benchmark"
    ):
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
            print(
                f"      ✅ Сравнено пар: {len(matrix_results.get('df_comparisons', []))}"
            )
        except Exception as e:
            print(f"      ❌ Ошибка all-vs-all: {e}")
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
            print("   ✅ Пакетное сравнение завершено. Топ-3 метода сохранены.")
            print(
                f"      ✅ Топ-3: {df_batch.nlargest(3, 'similarity_score')['method'].tolist()}"
            )
        except Exception as e:
            print(f"      ❌ Ошибка batch comparison: {e}")

        # ──────────────────────────────────────────────────
        # 3. PAIRWISE COMPARISON (CV2 vs Sklearn Otsu)
        # ──────────────────────────────────────────────────
        try:
            print("   🔸 Pairwise comparison (CV2 vs Sklearn)...")
            df_pairwise: Dict[str, Any] = comparator.compare_methods(
                image=img_array,
                segmenter1=original_segmenter,
                segmenter2=ref_segmenter,
                name1="Original_CV2_Global",
                name2=f"Reference_{reference_method}",
                save_comparison=True,
                output_path=str(Path(output_dir) / f"compare_methods_{img_name}"),
            )
            pairwise_results_list.append(df_pairwise)
            print("   ✅ Попарное сравнение сохранено.")
            n_metrics: int = len(df_pairwise.get("metrics", {}))
            print(f"✅ Сохранено: {n_metrics} метрик")
        except TypeError as te:
            print(f"      ⚠️  Pairwise требует старой сигнатуры: {te}")
        except Exception as e:
            print(f"      ❌ Ошибка pairwise: {e}")

    # ──────────────────────────────────────────────────────
    # СВОДНЫЙ ОТЧЁТ
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 СВОДКА ПО МАТРИЧНОМУ СРАВНЕНИЮ")
    print("=" * 70)

    # All-vs-All статистика
    if all_vs_all_results:
        total_comparisons: int = sum(r["n_comparisons"] for r in all_vs_all_results)
        print(
            f"\n🔗 All-vs-All: {len(all_vs_all_results)} изображений, {total_comparisons} сравнений"
        )

    # Batch comparison: агрегация результатов
    if batch_results_list:
        batch_summary: pd.DataFrame = pd.concat(batch_results_list, ignore_index=True)
        print(f"\n📦 Batch comparison: {len(batch_summary)} записей")

        # Топ-5 методов по средней схожести
        if "similarity_score" in batch_summary.columns:
            top_methods: pd.Series = (
                batch_summary.groupby("method")["similarity_score"].mean().nlargest(5)
            )
            print(f"\n🏆 Топ-5 по схожести с референсом:")
            for i, (method, score) in enumerate(top_methods.items(), 1):
                print(f"   {i}. {method}: {score:.4f}")

    # Pairwise статистика
    if pairwise_results_list:
        print(f"\n🔍 Pairwise: {len(pairwise_results_list)} сравнений")

    print(f"\n💾 Результаты: {output_dir}")

    return {
        "all_vs_all_results": all_vs_all_results,
        "batch_results": (
            pd.concat(batch_results_list, ignore_index=True)
            if batch_results_list
            else None
        ),
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
    """
    Оценка качества сегментации против Ground Truth.

    Для каждого изображения с доступным GT:
    1. Запускает все классические методы.
    2. Вычисляет метрики: IoU, Dice, Precision, Recall, F1, MAE, Hausdorff.
    3. Сохраняет детальные метрики в JSON.
    4. Генерирует сводные графики и топ-методы.

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
    for img_name, (img_path, img_pil, gt_mask) in tqdm(
        test_images.items(), desc="Ground Truth benchmark"
    ):
        if gt_mask is None:
            print(f"⚠️  Пропуск {img_name}: нет Ground Truth")
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

                    pred_mask = resize(
                        pred_mask, gt_binary.shape, order=0, preserve_range=True
                    ).astype(np.uint8)

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
                print(
                    f"   {status} {name}: IoU={iou:.4f}, Dice={metrics.get('dice', 0):.4f}, t={exec_time:.3f}s"
                )
                print(f"Mask after {name} segment: {pred_mask[:3, :3]}")

            except Exception as e:
                print(f"   💥 Критическая ошибка в методе {name}: {e}")
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
        print(
            "⚠️ Ground Truth маски не найдены ни для одного изображения. Пропускаем этап оценки качества."
        )
        return None

    print(f"\n📈 Генерация отчётов...")

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
        top_series: pd.Series = (
            df_flat.groupby("Method")["IoU"].mean().sort_values(ascending=False).head(5)
        )
        top_methods = list(top_series.items())

        print(f"\n🏆 ТОП-5 методов по среднему IoU:")
        for i, (method, iou) in enumerate(top_methods, 1):
            print(f"   {i}. {method}: IoU = {iou:.4f}")
    else:
        print("   Нет успешных результатов для ранжирования.")

    print("\n" + "=" * 60)
    print("СВОДНЫЙ ОТЧЕТ ПО GROUND TRUTH")
    print("=" * 60)

    # Сводная таблица
    rows: List[Dict[str, Any]] = []
    for img_name, methods_data in tqdm(
        gt_results_summary.items(), desc="Ground Truth Testing"
    ):
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
        summary_df_sorted = summary_df.sort_values(
            by=["Image", "IoU"], ascending=[True, False]
        )
        print("\nТоп методов по IoU:")
        print(
            summary_df_sorted[["Method", "Image", "IoU", "Dice", "Time_s"]].to_string(
                index=False
            )
        )
        summary_df_sorted.to_csv("./data/gt_summary_report.csv", index=False)
        summary_path: Path = Path(output_dir) / "gt_summary_report.csv"
        summary_df_sorted.to_csv(summary_path, index=False, float_format="%.4f")
        print(f"\n💾 Сводка: {summary_path}")
        plt.figure(figsize=(12, 6))
        first_img = list(gt_results_summary.keys())[0]
        df_plot = (
            summary_df[summary_df["Image"] == first_img]
            .sort_values("IoU", ascending=False)
            .head(10)
        )
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
            print("⚠️ Не удалось построить график: нет данных для первого изображения.")
        plt.close()
    else:
        print("Нет успешных метрик для отображения.")

    return {
        "gt_results_summary": gt_results_summary,
        "summary_df": summary_df,
        "top_methods": top_methods,
    }


# ──────────────────────────────────────────────────────────────────────
def run_augmentation_training_study(
    root_dir: str = "./data/ade20k",
    checkpoint_dir: str = "./models",
    device: str = "cuda",
    output_dir: str = "./data/augmentation_study",
) -> Optional[Dict[str, Any]]:
    """
    Исследование влияния уровней аугментаций на качество обучения моделей.

    Выполняет:
    1. Обучение нескольких архитектур с разными уровнями аугментаций.
    2. Сравнение метрик (mIoU, val loss) между конфигурациями.
    3. Генерацию визуализаций: бар-чарты, тепловые карты, топ-комбинации.
    4. Оценку обученных моделей на валидации через чекпоинты.

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
        # "psp_smp",  # PSPNet + MiT-B5
        "deeplab_tv",  # DeepLabV3+
        # "fcn_tv",  # FCN ResNet-50
        # "segnet",  # SegNet
    ]

    results_by_model_and_aug: Dict[str, Dict[str, Any]] = {
        model_type: {} for model_type in model_types
    }

    print(
        f"\n🔧 План: {len(model_types)} моделей × {len(augmentation_configs)} уровней аугментаций"
    )

    # ──────────────────────────────────────────────────────
    # 3. ЦИКЛ ОБУЧЕНИЯ
    # ──────────────────────────────────────────────────────
    for aug_config in augmentation_configs:
        for model_type in model_types:
            print(f"\n{'=' * 60}")
            print(f"🔹 Модель: {model_type} | Аугментации: {aug_config['level']}")
            print(f"{'=' * 60}")

            # Настройка энкодера
            encoder_name: Literal[
                "resnet34", "resnet50", "resnet101", "mit_b5", "efficientnet-b0"
            ]
            if model_type in ["fpn_smp", "psp_smp"]:
                encoder_name = "mit_b5"  # type: ignore[assignment]
            else:
                encoder_name = "resnet34"  # type: ignore[assignment]
            variant: str = (
                "fcn_resnet50"
                if model_type == "fcn_tv"
                else "b5" if "mit" in encoder_name else "b5"
            )

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
                print(
                    f"✅ {model_type} ({aug_config['level']}): Best mIoU = {miou_pct:.2f}%"
                )

            except Exception as e:
                print(f"❌ Ошибка обучения {model_type} ({aug_config['level']}): {e}")
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

    comparison_results: Dict[str, float] = trainer.compare_trained_models(
        augmentation_level="medium"
    )
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
                model_type="unet_smp",  # 🔥 Нужно указать тип модели или сделать универсально
                # val_fraction=0.05,
            )
            print(eval_results_old)

            eval_results: Dict[str, Any] = trainer.evaluate_trained_models_on_val(
                checkpoints=checkpoints,
                val_fraction=0.05,
            )
            print(eval_results)
            print(f"   ✅ Оценка завершена")
        except Exception as e:
            print(f"   ⚠️  Ошибка оценки: {e}")

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
    print(f"\n🏆 ТОП-3 ЛУЧШИХ КОМБИНАЦИЙ:")
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
    """
    Генерирует графики сравнения аугментаций.

    1. Бар-чарты: влияние аугментаций для каждой модели.
    2. Групповые бар-чарты: сравнение моделей для каждого уровня аугментаций.
    """
    plots_dir: Path = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # График 1: Аугментации для каждой модели
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, model_type in enumerate(model_types):
        if idx >= len(axes):
            break
        ax = axes[idx]

        levels = list(results_by_model_and_aug[model_type].keys())
        miou_values: List[float] = [
            results_by_model_and_aug[model_type][level]["best_miou"] * 100
            for level in levels
        ]

        colors = ["#ffcccc", "#ff9999", "#ff6666"][: len(levels)]
        bars = ax.bar(
            levels, miou_values, color=colors, edgecolor="black", linewidth=1.5
        )

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
        fig, axes = plt.subplots(
            1, len(augmentation_configs), figsize=(6 * len(augmentation_configs), 5)
        )
        if len(augmentation_configs) == 1:
            axes = [axes]

        for idx, aug_config in enumerate(augmentation_configs):
            ax = axes[idx]
            level = aug_config["level"]

            model_names: List[str] = []
            miou_values = []

            for model_type in model_types:
                if level in results_by_model_and_aug[model_type]:
                    model_names.append(
                        model_type.replace("_smp", "").replace("_tv", "")
                    )
                    miou = (
                        results_by_model_and_aug[model_type][level]["best_miou"] * 100
                    )
                    miou_values.append(miou)

            if not model_names:
                continue

            colors = colormaps["viridis"](np.linspace(0, 1, len(model_names)))
            bars = ax.bar(
                model_names, miou_values, color=colors, edgecolor="black", linewidth=1.5
            )

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
    """
    Собирает актуальные чекпоинты по паттернам.

    Args:
        model_types: Список типов моделей.
        augmentation_levels: Уровни аугментаций для поиска.
        checkpoint_dir: Директория с чекпоинтами.

    Returns:
        Dict[str, str]: {display_name: checkpoint_path}
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
                print(f"   ⚠️  {model_type} ({level}): не найден")

    return checkpoints


# ──────────────────────────────────────────────────────────────────────
def _create_augmentation_summary(
    results_by_model_and_aug: Dict[str, Dict[str, Any]],
    model_types: List[str],
) -> pd.DataFrame:
    """
    Создаёт сводный DataFrame по результатам обучения.

    Args:
        results_by_model_and_aug: Результаты по моделям и аугментациям.
        model_types: Список типов моделей.

    Returns:
        pd.DataFrame: Сводная таблица с колонками:
                     [Model, Augmentation Level, Best mIoU (%), Epochs, Final Val Loss]
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
                    "Final Val Loss": (
                        result["final_val_loss"]
                        if result["final_val_loss"] is not None
                        else "N/A"
                    ),
                }
            )

    df: pd.DataFrame = pd.DataFrame(summary_data)

    # Сортировка: по модели, затем по mIoU (убывание)
    return df.sort_values(["Model", "Best mIoU (%)"], ascending=[True, False])


# ──────────────────────────────────────────────────────────────────────
def _generate_augmentation_heatmap(summary_df: pd.DataFrame, output_dir: str) -> None:
    """
    Генерирует тепловую карту влияния аугментаций.

    Args:
        summary_df: Сводный DataFrame с результатами.
        output_dir: Директория для сохранения.
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
    """
    Построение графиков по результатам тестирования с Ground Truth.

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
    for img_name, methods_data in tqdm(
        results_dict.items(), desc="Visualisation GT Testing"
    ):
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
        print("⚠️ Нет валидных данных для визуализации.")

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
    """
    Извлекает название библиотеки из имени метода.

    Args:
        method_name: Имя метода, например "Otsu_Thresholding_CV2".

    Returns:
        str: Название библиотеки: "CV2", "Sklearn", "Torch" или "Unknown".
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
    """
    Агрегирует метрики по методам: усреднение по всем изображениям.

    Args:
        df: DataFrame с сырыми результатами.

    Returns:
        pd.DataFrame: Агрегированные данные с колонками [Method, Library, IoU, Dice, ...].
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
    """Построение bar chart сравнения метрик."""
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
    """Построение scatter plot: время выполнения vs IoU."""
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
    """Построение графика Precision-Recall."""
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
    """
    Загружает тестовые изображения для бенчмарка.

    Поддерживает два режима:
    1. **С Ground Truth** (`use_image_with_mask=True`):
       - Загружает изображение и маску из репозитория HuggingFace.
       - Конвертирует многоклассовую маску в бинарную (объект/фон).
       - Сохраняет локальные копии для офлайн-работы.
    2. **Без Ground Truth** (`use_image_with_mask=False`):
       - Загружает изображения из предопределённых URL или локальных путей.
       - Генерирует синтетические бинарные маски (половина изображения = объект).

    Алгоритм конвертации маски ADE20K → бинарная:
    1. Находим самый частый класс (предполагаем, что это фон).
    2. Все остальные классы считаем объектом (значение 255).
    3. Если объектов <1% — пробуем второй по величине класс.

    Args:
        use_image_with_mask: Если `True`, загружать с ground truth из ADE20K.
        dataset_repo: ID репозитория на HuggingFace (по умолчанию ADE20K fixtures).

    Returns:
        TestImagesDict: Словарь `{имя: (путь, PIL.Image, GT-маска или None)}`.

    Raises:
        RuntimeError: Если не удалось загрузить ни одного изображения.

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
    """Загружает изображение и GT-маску из репозитория."""
    print(f"📥 Загрузка из репозитория {repo_id}...")

    img_path: str = hf_hub_download(
        repo_id=repo_id, filename="ADE_val_00000001.jpg", repo_type="dataset"
    )
    mask_path: str = hf_hub_download(
        repo_id=repo_id, filename="ADE_val_00000001.png", repo_type="dataset"
    )

    img: Image.Image = Image.open(img_path).convert("RGB")
    gt_mask_pil: Image.Image = Image.open(mask_path)

    print(f"✅ Изображение загружено: {os.path.basename(img_path)}")
    print(f"✅ Ground truth загружен: {os.path.basename(mask_path)}")
    print(f"Ground Truth: {mask_path}")
    print(f"Размер изображения: {img.size}")
    print(f"Размер GT: {gt_mask_pil.size}")

    # Конвертация маски в бинарную
    gt_np: npt.NDArray = np.array(gt_mask_pil)
    print(
        f"\nДиапазон значений Ground Truth: {gt_np.min()} - {gt_np.max()}, min: {gt_np.min()}, max: {gt_np.max()}"
    )
    binary_gt: MaskArray = _convert_multiclass_to_binary(gt_np)

    # Сохранение локальных копий
    local_paths = _save_test_artifacts(img, gt_mask_pil, binary_gt)

    test_images["ade20k_sample"] = (local_paths["img"], img, binary_gt)
    print(f"✅ Загружен образец: {img.size}, GT: {binary_gt.shape}")


# ──────────────────────────────────────────────────────────────────────
def _convert_multiclass_to_binary(gt_np: npt.NDArray) -> MaskArray:
    """
    Конвертирует многоклассовую маску в бинарную (объект/фон).

    Алгоритм:
    1. Найти самый частый класс → считать фоном.
    2. Все остальные классы → объект (255).
    3. Если объектов <1% → попробовать второй по величине класс.

    Args:
        gt_np: Многоклассовая маска формы (H, W), dtype=uint8.

    Returns:
        MaskArray: Бинарная маска формы (H, W), dtype=uint8, {0, 255}.
    """
    unique, counts = np.unique(gt_np, return_counts=True)
    bg_class = unique[np.argmax(counts)]
    print(
        f"📊 Статистика GT: Всего классов {len(unique)}. Самый частый: {bg_class} ({np.max(counts)} пикселей)"
    )

    binary: MaskArray = (gt_np != bg_class).astype(np.uint8) * 255

    # fallback: если объектов слишком мало
    if np.sum(binary > 0) < (binary.size * 0.01) and len(unique) > 1:
        print(
            "⚠️ Объектов слишком мало по стратегии 'не фон'. Пробуем взять второй по величине класс."
        )
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
    """Сохраняет локальные копии тестовых артефактов."""
    os.makedirs("test_images", exist_ok=True)

    paths: Dict[str, str] = {
        "img": "test_images/test_gt_image.jpg",
        "mask_raw": "test_images/test_gt_mask_raw.png",
        "mask": "test_images/test_gt_mask.png",
    }

    img.save(paths["img"])
    print(f"✅ Изображение сохранено локально: {paths["img"]}")

    gt_raw.save(paths["mask_raw"])
    print(f"✅ Изображение сырой маски сохранено локально: {paths["mask_raw"]}")

    Image.fromarray(gt_binary).save(paths["mask"])
    print(f"✅ Изображение маски сохранено локально: {paths["mask"]}")

    return paths


# ──────────────────────────────────────────────────────────────────────
def _load_images_without_ground_truth(test_images: TestImagesDict) -> None:
    """Загружает изображения без GT с генерацией синтетических масок."""
    print("⚠️ Не удалось загрузить реальные GT. Используем только изображения.")
    image_sources: Dict[str, str] = {
        "countryside": "https://i.pinimg.com/736x/17/e7/fc/17e7fc299466b2afd989e709fe7c9815.jpg",
        "nature": "https://i.pinimg.com/736x/f7/5a/f2/f75af26820b50c24600f50f3998eb02f.jpg",
        "architecture": "https://i.pinimg.com/736x/86/f6/07/86f60748d5d9ae4cb9092018d1321648.jpg",
        "trucks": "https://www.shutterstock.com/shutterstock/videos/1106252821/thumb/1.jpg?ip=x480",
        "traffic": "https://images.pond5.com/pov-car-and-truck-traffic-footage-190002081_iconl.jpeg",
        "mountain": "https://i.pinimg.com/736x/17/66/c4/1766c4f667af39f91172ef8eb21ab18a.jpg",
    }

    image_paths: Dict[str, str] = {
        "war_frame_1": "test_images/2340_frame.jpg",
        "war_frame_2": "test_images/3330_frame.jpg",
        "war_frame_3": "test_images/4130_frame.jpg",
        "war_frame_4": "test_images/4480_frame.jpg",
        "building": "test_images/test_gt_image.jpg",
        "animals": "test_images/animals.jpg",
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
            print(f"❌ Ошибка загрузки {name}: {e}")

    for name, path in image_paths.items():
        try:
            img = Image.open(path)
            gt_synthetic = _generate_synthetic_mask(img.size)
            test_images[name] = (path, img, gt_synthetic)
            print(f"✅ {name}: {img.size}, ground truth: {gt_synthetic}")

        except Exception as e:
            print(f"❌ Ошибка загрузки {name}: {e}")


# ──────────────────────────────────────────────────────────────────────
def _download_image(url: str, name: str) -> Image.Image:
    """Загружает изображение по URL с обработкой ошибок."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    img: Image.Image = Image.open(BytesIO(response.content)).convert("RGB")

    os.makedirs("test_images", exist_ok=True)
    local_path: str = f"test_images/test_image_{name}.jpg"
    img.save(local_path)

    return img


# ──────────────────────────────────────────────────────────────────────
def _generate_synthetic_mask(size: Tuple[int, int]) -> MaskArray:
    """Генерирует синтетическую бинарную маску (нижняя половина = объект)."""
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
    """
    Сохраняет отчёт с метриками в JSON-формате.

    Args:
        metrics_all: Словарь метрик: {имя_метода: {метрика: значение}}.
        path: Путь к выходному файлу.
        indent: Количество пробелов для форматирования (по умолчанию 2).
        ensure_ascii: Экранировать non-ASCII символы (по умолчанию False).

    Returns:
        None. Файл сохраняется на диск.

    Note:
        - Используется `default=str` для сериализации numpy-типов.
        - Создаёт родительские директории при необходимости.

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
def test_neural_segmentation_variants() -> (
    Tuple[Optional[NeuralSegmenter], Optional[Dict[str, Any]]]
):
    """Тестирование различных вариантов нейросетевой сегментации"""

    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ВАРИАНТОВ НЕЙРОСЕТЕВОЙ СЕГМЕНТАЦИИ")
    print("=" * 60)

    # Загрузка тестового изображения
    img_url: str = (
        "https://images.pond5.com/pov-car-and-truck-traffic-footage-190002081_iconl.jpeg"
    )

    try:
        response: requests.Response = requests.get(img_url)
        test_image: Image.Image = Image.open(BytesIO(response.content))

        # Создаем нейросетевой сегментатор
        segmenter: NeuralSegmenter = NeuralSegmenter(
            local_path="/home/yamshchikov/models/segformer-b5-ready"
        )

        # Вариант 1: Различные значения alpha
        print("\n1. Тестирование разных значений alpha:")
        alphas: List[float] = [0.3, 0.5, 0.7, 1.0]

        fig: plt.Figure
        axes: np.ndarray  # type: ignore[assignment]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
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
            print(
                f"  {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)"
            )

        # Вариант 3: segment_with_mask
        print("\n3. Тестирование segment_with_mask:")
        result_np: np.ndarray
        mask: Optional[np.ndarray]
        result_np, mask = segmenter.segment_with_mask(test_image, alpha=0.5)

        fig2: plt.Figure
        axes2: np.ndarray  # type: ignore[assignment]
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

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
        print(f"❌ Ошибка тестирования нейросетевых вариантов: {e}")
        print(traceback.format_exc())
        return None, None


# ──────────────────────────────────────────────────────────────────────
def run_cpu_cuda_benchmark(
    cv2_methods: SegmenterDict,
    sklearn_methods: SegmenterDict,
    torch_methods: SegmenterDict,
    test_image: ImageArray,
    n_runs: int = 5,
    warmup_runs: int = 2,
) -> BenchmarkResult:
    """
    Запуск бенчмарка CPU vs CUDA для классических методов.

    Выполняет сравнение времени выполнения методов на CPU и CUDA:
    1. Фильтрует методы: исключает нейросетевые (только классические).
    2. Для каждого метода:
       - Запускает `warmup_runs` раз для "прогрева".
       - Замеряет время выполнения `n_runs` раз на CPU.
       - Замеряет время выполнения `n_runs` раз на CUDA (если доступно).
    3. Агрегирует результаты: среднее, std, speedup.

    Args:
        cv2_methods: Словарь методов OpenCV.
        sklearn_methods: Словарь методов scikit-learn.
        torch_methods: Словарь методов PyTorch.
        test_image: Тестовое изображение формы (H, W) или (H, W, 3).
        n_runs: Количество прогонов для замера времени.
        warmup_runs: Количество "прогревочных" прогонов.

    Returns:
        BenchmarkResult: DataFrame с колонками:
                        [method, device, mean_time, std_time, speedup].

    Example:
        ```python
        df = run_cpu_cuda_benchmark(cv2_methods, sklearn_methods, torch_methods, test_img)

        # Анализ результатов
        cuda_methods = df[df["device"] == "cuda"]
        print(f"Средний speedup: {cuda_methods['speedup'].mean():.2f}x")
        ```
    """
    from testing.CpuCudaBenchmark import CpuCudaBenchmark

    print("\n" + "=" * 80)
    print("ЗАПУСК БЕНЧМАРКА: CPU vs CUDA")
    print("=" * 80)

    # Объединяем все классические методы
    all_classical_methods: SegmenterDict = {
        **cv2_methods,
        **sklearn_methods,
        **torch_methods,
    }
    classical_only: SegmenterDict = _filter_classical_methods(all_classical_methods)

    print(f"Количество классических методов: {len(classical_only)}")

    # Инициализация и запуск бенчмарка
    benchmark: CpuCudaBenchmark = CpuCudaBenchmark(
        base_output_dir="./data/cpu_cuda_benchmark",
        n_runs=n_runs,
        warmup_runs=warmup_runs,
    )

    # Запускаем бенчмарк
    df_results: BenchmarkResult = benchmark.benchmark_all_methods(
        methods_dict=classical_only,
        image=test_image,
        test_name="classical_methods_cpu_cuda",
    )

    # Вывод сводки
    _print_cpu_cuda_summary(df_results)

    return df_results


# ──────────────────────────────────────────────────────────────────────
def _print_cpu_cuda_summary(df_results: BenchmarkResult) -> None:
    """Выводит сводку по бенчмарку CPU vs CUDA."""
    print("\n" + "=" * 80)
    print("СВОДКА ПО БЕНЧМАРКУ CPU vs CUDA")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("⚠️  CUDA недоступен, пропуск сравнения")
        return

    for method in df_results["method"].unique():
        method_data = df_results[df_results["method"] == method]
        cpu_data = method_data[method_data["device"] == "cpu"]
        cuda_data = method_data[method_data["device"] == "cuda"]

        if cpu_data.empty or cuda_data.empty:
            continue

        cpu_time: float = cpu_data["mean_time"].values[0] * 1000
        cuda_time: float = cuda_data["mean_time"].values[0] * 1000

        if cuda_time > 0:
            speedup: float = cpu_time / cuda_time
            print(
                f"{method:40s}: CPU={cpu_time:7.2f}ms, CUDA={cuda_time:7.2f}ms, ⚡ Speedup={speedup:.2f}x"
            )

    print("=" * 80)


# ──────────────────────────────────────────────────────────────────────
def _filter_classical_methods(all_methods: SegmenterDict) -> SegmenterDict:
    """
    Фильтрует методы, исключая нейросетевые.

    Args:
        all_methods: Полный словарь методов.

    Returns:
        SegmenterDict: Только классические методы.
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

    return {
        name: seg
        for name, seg in all_methods.items()
        if not any(kw in name.lower() for kw in neural_keywords)
    }


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Основной тест
    print("ЗАПУСК ОСНОВНОГО ТЕСТА")
    print("=" * 60)
    tester, results, comparator = main()

    print("\n\nЗАПУСК ДОПОЛНИТЕЛЬНОГО ТЕСТА НЕЙРОСЕТЕВЫХ ВАРИАНТОВ")
    print("=" * 60)
    segmenter, detailed_result = test_neural_segmentation_variants()
