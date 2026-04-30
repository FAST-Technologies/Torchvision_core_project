# main.py

# Импорт основных библиотек
from segmenters.NeuralSegmenter import NeuralSegmenter
from segmenters.OpenCVSegmenter import OpenCVSegmenter
from segmenters.SklearnSegmenter import SklearnSegmenter
from segmenters.TorchSegmenter import TorchSegmenter
from segmenters.ModelTrainer import ModelTrainer, TrainingConfig
from segmenters.NeuralModelFactory import NeuralModelFactory
from testing.SegmentationTester import SegmentationTester
from testing.SegmentationComparator import SegmentationComparator
from testing.SegmentationBenchmark import SegmentationBenchmark, export_comparison_table
from testing.TorchImplementationValidator import TorchImplementationValidator
from metrics.SegmentationMetrics import SegmentationMetrics
from utils.warmup import SegmentationWarmUp
from utils.threshold_warmup import ThresholdWarmUp
from testing.BatchClassicTester import BatchClassicTester

from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from utils.strategies import _create_overlay_standalone, segment_image_unified

from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
)

import os
import sys
import traceback
import warnings
import time
import requests
from io import BytesIO
from PIL import Image
import json

from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

warnings.filterwarnings("ignore")

num_classes: int = 150


def main():
    test_neural_logic: bool = False
    test_classic_logic: bool = True
    print(f"📍 CWD: {os.getcwd()}")
    print(f"📍 __file__: {__file__}")
    print(f"📍 sys.path: {sys.path[:3]}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("CUDA")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(
            f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
        print(
            f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("CPU")

    print("=" * 60)
    print("ОБЪЕДИНЁННЫЙ ФРЕЙМВОРК СЕГМЕНТАЦИИ")
    print("=" * 60)

    # ============ 1. ЗАГРУЗКА ДАННЫХ ============
    print("\n1. Загрузка тестовых изображений...")
    test_images = load_test_images(use_image_with_mask=False)
    print(f"⚠️ Количество изображений ({len(test_images)})")

    gt_results_summary = {}

    # ============ 2. ДОБАВЛЕНИЕ МЕТОДОВ ============
    print("=" * 60)
    print("ИНИЦИАЛИЗАЦИЯ МЕТОДОВ СЕГМЕНТАЦИИ")
    print("=" * 60)
    tester = SegmentationTester(
        base_output_dir="./data/segmentation_tester_results",
        enable_warmup=True,
        n_warmup_runs=5,
    )

    print("\n1. Загрузка методов OpenCV...")
    cv2_methods = {
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

    print("\n2. Загрузка методов SKlearn...")
    sklearn_methods = {
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

    print("\n3. Загрузка методов PyTorch...")
    torch_methods = {
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

    if test_classic_logic:
        try:
            for name, segmenter in {
                **cv2_methods,
                **sklearn_methods,
                **torch_methods,
            }.items():
                tester.add_method(name, segmenter)
                print(f"   ✅ {name}")
        except Exception as e:
            print(f"  ⚠️ Загружаемые методы недоступны: {e}")

    print("🧹 Очистка памяти CUDA перед загрузкой тяжелой модели...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    if test_neural_logic:
        neural_models_config = [
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
            try:
                if "checkpoint_path" in config and not os.path.exists(
                    config["checkpoint_path"]
                ):
                    print(
                        f"   ⚠️ {config['name']} - чекпоинт не найден: {config['checkpoint_path']}"
                    )
                    continue

                segmenter = NeuralSegmenter(
                    model_type=config["type"],
                    checkpoint_path=config.get("checkpoint_path"),
                    local_path=config.get("local_path"),
                    model_name=config.get("model_name"),
                    num_classes=num_classes,
                    **{
                        k: v
                        for k, v in config.items()
                        if k
                        not in [
                            "name",
                            "type",
                            "checkpoint_path",
                            "local_path",
                            "model_name",
                        ]
                    },
                )
                tester.add_method(config["name"], segmenter)
                print(f"   ✅ {config['name']}")
                segmenter.get_class_info()
            except Exception as e:
                print(f"  ⚠️ Нейросетевая сегментация недоступна: {e}")
                print(f"   ❌ {config['name']}: {e}")
                print(traceback.format_exc())

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
        training_config = NeuralModelFactory.get_training_config("ade20k")
        print(f"Batch size: {training_config['batch_size']}")
        print(f"Epochs: {training_config['epochs']}")
        print(f"LR: {training_config['lr']}")

        # # ========== ВАРИАНТ 6: Конфиг метрик ==========
        print("\n=== ВАРИАНТ 6: Конфиг метрик ===")
        metrics_config = NeuralModelFactory.get_metrics_config()
        print(f"Threshold: {metrics_config['threshold']}")
        print(f"Include Hausdorff: {metrics_config['include_hausdorff']}")

        # # ========== ВАРИАНТ 7: Массовая загрузка из конфига ==========
        print("\n=== ВАРИАНТ 7: Массовая загрузка ===")
        neural_models_config = [
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

    # ============ 3. БЕНЧМАРК ============
    print("\n3. Бенчмарк производительности и оценка качества перед warm up...")
    first_img_array = None
    all_comparisons = []
    for img_name, (img_path, img_pil, gt_mask) in test_images.items():
        print(f"\n--- Обработка изображения: {img_name} ---")
        img_array = np.array(img_pil)
        if first_img_array is None:
            first_img_array = img_array.copy()
        df_benchmark_before_warm_up = tester.benchmark_methods(
            img_array,
            n_runs=10,
            test_name=f"benchmark_{img_name}_cold_before_warm_up",
            save_results=True,
            force_warmup=False,
            ground_truth=gt_mask,
        )
        print(f"   ✅ Бенчмарк для {img_name} завершён")

    # # # ============ 🔥 ЗАПУСК WARM-UP ============
    print("\n" + "=" * 60)
    print("ЗАПУСК WARM-UP ПЕРЕД БЕНЧМАРКОМ")
    print("=" * 60)

    if first_img_array is not None:
        warmup_utility = SegmentationWarmUp(n_warmup_runs=10)

        warmup_results = warmup_utility.warmup_all_segmenters(
            segmenters_dict=tester.methods,
            image=first_img_array,  # ← Используем сохранённое
            verbose=True,
        )

        threshold_warmup = ThresholdWarmUp.warmup_threshold_methods(
            segmenters_dict=tester.methods, image_sizes=[(128, 128), (256, 256)]
        )

        edge_warmup = ThresholdWarmUp.warmup_edge_methods(
            segmenters_dict=tester.methods
        )

        with open(f"./data/warmup_report.txt", "w") as f:
            f.write("WARM-UP ОТЧЁТ\n")
            f.write("=" * 60 + "\n\n")
            f.write(warmup_utility.get_warmup_summary())
            f.write("\n\nПороговые методы:\n")
            f.write(str(threshold_warmup))
            f.write("\n\nГраничные методы:\n")
            f.write(str(edge_warmup))

        print(f"\n✅ Отчёт о warm-up сохранён в ./data/warmup_report.txt")

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Дождаться завершения всех ядер
            # torch.cuda.empty_cache()  # Очистить кэш (опционально)

    # # # ============ БЕНЧМАРК ПОСЛЕ WARM-UP ============
    print("\n3.1. Бенчмарк производительности после warm up...")
    for img_name, (img_path, img_pil, gt_mask) in test_images.items():
        print(f"\n--- Обработка изображения: {img_name} ---")
        img_array = np.array(img_pil)

        df_benchmark_after_warm_up = tester.benchmark_methods(
            img_array,
            n_runs=10,
            test_name=f"benchmark_{img_name}_hot_after_warm_up",
            save_results=True,
            force_warmup=False,
            ground_truth=gt_mask,
        )

        # 🔥 Сравниваем cold vs hot
        # df_before = pd.read_csv(f"./data/segmentation_tester_results/benchmark_{img_name}_cold_before_warm_up_*/statistics/benchmark_results.csv")
        # df_after = pd.read_csv(f"./data/segmentation_tester_results/benchmark_{img_name}_hot_after_warm_up_*/statistics/benchmark_results.csv")

        comparison = pd.DataFrame(
            {
                "method": df_benchmark_before_warm_up["Method"],
                "image": img_name,
                "cold_mean_ms": df_benchmark_before_warm_up["Mean_Time_s"] * 1000,
                "hot_mean_ms": df_benchmark_after_warm_up["Mean_Time_s"] * 1000,
            }
        )
        comparison["speedup"] = comparison.apply(
            lambda row: (
                row["cold_mean_ms"] / row["hot_mean_ms"]
                if row["hot_mean_ms"] > 0
                else float("inf")
            ),
            axis=1,
        )
        all_comparisons.append(comparison)

        comparison.to_csv(f"./data/cold_hot_comparison_{img_name}.csv", index=False)
        print("\n" + "=" * 70)
        print(f"🔥 COLD vs HOT BENCHMARK COMPARISON ({img_name})")
        print("=" * 70)
        print(
            comparison.sort_values("speedup", ascending=False).to_string(
                float_format=lambda x: f"{x:.2f}" if not np.isinf(x) else "∞"
            )
        )

    if all_comparisons:
        summary = pd.concat(all_comparisons)
        summary.to_csv("./data/cold_hot_comparison_summary.csv")
        print("\n" + "=" * 70)
        print("🔥 СВОДНЫЙ COLD vs HOT BENCHMARK (все изображения)")
        print("=" * 70)
        avg_speedup = (
            summary.groupby("method")["speedup"].mean().sort_values(ascending=False)
        )
        print(avg_speedup)

    if test_neural_logic is True:
        # ============ 4. СЕГМЕНТАЦИОННЫЙ БЕНЧМАРК (опционально) ============
        repo_id = "hf-internal-testing/fixtures_ade20k"
        image_path = hf_hub_download(
            repo_id=repo_id, filename="ADE_val_00000001.jpg", repo_type="dataset"
        )
        original_img_0 = Image.open(image_path)
        original_img_0.save("./data/ade20k_test_trained/original_image_0.jpg")
        segmentation_map_path = hf_hub_download(
            repo_id=repo_id, filename="ADE_val_00000001.png", repo_type="dataset"
        )
        segmentation_map = Image.open(segmentation_map_path)
        mask_array_ade = np.array(segmentation_map)
        print(f"Mask shape: {mask_array_ade.shape}")
        print(f"Mask dtype: {mask_array_ade.dtype}")
        print(f"Unique values: {np.unique(mask_array_ade)[:20]}")
        mask_2d_ade = prepare_mask_for_overlay(segmentation_map)
        segmentation_map.save("./data/ade20k_test_trained/original_image_mask_0.png")

        infer_res_ade = _create_overlay_standalone(
            original_img_0,
            mask_2d_ade,
            alpha=0.5,
            palette=NeuralSegmenter.ade_palette(),
        )

        print("🔹 Запуск MaskFormer (Изолированный режим)...")

        model_name_maskformer = "facebook/maskformer-resnet50-ade20k-full"
        processor_maskformer = MaskFormerImageProcessor.from_pretrained(
            model_name_maskformer
        )
        model_maskformer = (
            MaskFormerForInstanceSegmentation.from_pretrained(model_name_maskformer)
            .to(device)
            .eval()
        )
        result_mf_ade, result_mf_ade_results = segment_image_unified(
            model_maskformer,
            processor_maskformer,
            original_img_0,
            "maskformer",
            alpha=0.6,
            palette=NeuralSegmenter.ade_palette,
            num_classes=num_classes,
            class_names=NeuralSegmenter.get_ade_class_names,
            gt_mask=mask_2d_ade,
        )
        result_mf_ade.save("./data/ade20k_test_trained/segmented_maskformer_ade_0.jpg")

        maskformer_manual_ade_result = {
            "model": "maskformer",
            "overlay": result_mf_ade,
            "mask": result_mf_ade_results.get("mask"),
            "inference_time_ms": result_mf_ade_results.get("inference_time_ms", 0),
            "metrics": result_mf_ade_results.get("metrics", {}),
            "image_size": original_img_0.size[::-1],
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

        # Инициализация бенчмарка (для Cityscapes Classic: 19 классов)
        benchmark_ade = SegmentationBenchmark(
            device="cuda",
            num_classes=num_classes,
            class_names=NeuralSegmenter.get_ade_class_names,
            gt_mask=mask_2d_ade,
            palette=NeuralSegmenter.ade_palette,
        )
        # benchmark_ade.load_all_trained_models(checkpoint_dir="./../models")
        benchmark_ade.load_segformer("/home/yamshchikov/models/segformer-b5-ready")
        benchmark_ade.load_mask2former("facebook/mask2former-swin-base-ade-semantic")
        benchmark_ade.load_oneformer("shi-labs/oneformer_ade20k_swin_large")
        benchmark_ade.load_unet_trained(
            checkpoint_path="models/unet_ade20k_best_200_epochs.pth"
        )
        benchmark_ade.load_deeplab_trained(
            checkpoint_path="models/deeplab_ade20k_best_200_epochs.pth"
        )
        benchmark_ade.load_sam("models/mobile_sam.pt")
        benchmark_ade.load_sam("models/sam2_t.pt")
        benchmark_ade.load_dpt("Intel/dpt-large-ade")
        benchmark_ade.load_upernet("openmmlab/upernet-convnext-small")
        benchmark_ade.load_segformer_variant("b2")
        benchmark_ade.load_mask_rcnn_pretrained(variant="maskrcnn_resnet50_fpn")
        benchmark_ade.load_fpn_mit_pretrained(
            variant="b5", checkpoint_path="models/fpn_mit_b5_ade20k_best_200_epochs.pth"
        )
        benchmark_ade.load_psp_mit_pretrained(
            variant="b5", checkpoint_path="models/psp_mit_b5_ade20k_best_200_epochs.pth"
        )
        benchmark_ade.load_fcn_resnet50_pretrained(
            variant="fcn_resnet50",
            checkpoint_path="models/fcn_resnet50_ade20k_best_200_epochs.pth",
        )
        benchmark_ade.load_segnet_pretrained(
            encoder_name="resnet34",
            checkpoint_path="models/segnet_ade20k_best_200_epochs.pth",
        )

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
        print(f"mask_2d_ade: {mask_2d_ade}")

        print("🚀 Запуск бенчмарка...")
        print("\n🚀 Running benchmark (this may take 10-15 minutes)...")
        benchmark_ade.compare(image_input=original_img_0, alpha=0.6)

        benchmark_ade.results["maskformer"] = maskformer_manual_ade_result
        results_map_ade = {
            "segformer": benchmark_ade.results["segformer"]["overlay"],
            "mask2former": benchmark_ade.results["mask2former"]["overlay"],
            "oneformer": benchmark_ade.results["oneformer"]["overlay"],
            "unet_smp": benchmark_ade.results["unet_pretrained"]["overlay"],
            "deeplab_tv": benchmark_ade.results["deeplab_pretrained"]["overlay"],
            "sam": benchmark_ade.results["sam"]["overlay"],
            "sam2": benchmark_ade.results["sam2"]["overlay"],
            "dpt": benchmark_ade.results["dpt"]["overlay"],
            "upernet": benchmark_ade.results["upernet"]["overlay"],
            "segformer_b2": benchmark_ade.results["segformer_b2"]["overlay"],
            "maskformer": benchmark_ade.results["maskformer"]["overlay"],
            "fpn_mit": benchmark_ade.results["fpn_mit_b5_pretrained"]["overlay"],
            "psp_mit": benchmark_ade.results["psp_mit_b5_pretrained"]["overlay"],
            "fcn_tv": benchmark_ade.results["fcn_resnet50_pretrained"]["overlay"],
            "maskrcnn_tv": benchmark_ade.results["maskrcnn_pretrained"]["overlay"],
            "segnet": benchmark_ade.results["segnet_resnet34_pretrained"]["overlay"],
        }

        for model_key, overlay in results_map_ade.items():
            if overlay is not None:
                overlay.save(
                    f"./data/ade20k_test_trained/segmented_{model_key}_ade.jpg"
                )
                print(f"✅ Сохранено: segmented_{model_key}_ade.jpg")

        print("\n🔍 Checking metrics...")
        summary_ade = benchmark_ade.get_summary()
        for metric in ["mIoU", "pixel_acc", "time_ms"]:
            values = [summary_ade[m].get(metric, np.nan) for m in summary_ade]
            valid = sum(1 for v in values if not np.isnan(v))
            print(f"   {metric}: {valid} models")

        print("\n📊 Generating visualizations...")
        benchmark_ade.plot_all_metrics(figsize=(15, 5))
        benchmark_ade.plot_comparison_chart(
            "mIoU", title="ADE20K: Mean IoU Comparison (11 Models)", figsize=(12, 6)
        )
        benchmark_ade.plot_comparison_chart(
            "time_ms", title="Inference Time Comparison (ms)", figsize=(12, 6)
        )
        benchmark_ade.plot_per_class_iou(top_k=20)
        benchmark_ade.plot_confusion_matrix("segformer", normalize="true")
        benchmark_ade.plot_summary(metrics=["mIoU", "pixel_acc", "time_ms"])

        print("\n💾 Saving results...")
        benchmark_ade.save_results("./data/ade20k_test_trained/ade_benchmark_v1")

        for model_name, res in benchmark_ade.results.items():
            print(f"\n{model_name}:")
            print(f"  mIoU: {res['metrics'].get('mIoU', 'N/A'):.4f}")
            print(f"  Time: {res['inference_time_ms']:.1f} ms")
            print(f"  Classes: {res['unique_classes']}")

        df = pd.DataFrame(benchmark_ade.get_summary()).T.sort_values(
            "mIoU", ascending=False
        )
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY (sorted by mIoU)")
        print("=" * 70)
        print(df.to_string())

        print("\n" + "=" * 70)
        print("LATEX TABLE FOR PAPER")
        print("=" * 70)
        print(
            benchmark_ade.export_latex_table(
                caption="Comprehensive Semantic Segmentation Benchmark on ADE20K"
            )
        )

        if "segformer" in df.index and "segformer_b2" in df.index:
            print("\n" + "=" * 70)
            print("SEGFORMER SPEED-ACCURACY TRADEOFF")
            print("=" * 70)
            sf = df.loc[["segformer", "segformer_b2"], ["mIoU", "time_ms"]]
            sf["mIoU"] = sf["mIoU"] * 100
            print(sf.to_string(float_format="%.2f"))
            print(
                f"\n💡 B2 is {df.loc['segformer', 'time_ms'] / df.loc['segformer_b2', 'time_ms']:.1f}x faster with {df.loc['segformer', 'mIoU'] - df.loc['segformer_b2', 'mIoU']:.1f} pp mIoU drop"
            )

        print("\n" + "=" * 70)
        print(
            "✅ BENCHMARK COMPLETE — Results saved to 'ade20k_11models_comprehensive/'"
        )
        print("=" * 70)
        tabled_7 = export_comparison_table(
            benchmark_ade, "./data/ade20k_test_trained/ade_model_comparison.md"
        )
        print(tabled_7)

        plt.figure(figsize=(20, 10))
        titles = [
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

        images = [
            original_img_0,  # Original
            results_map_ade["segformer"],  # SegFormer
            results_map_ade["mask2former"],  # Mask2Former
            results_map_ade["oneformer"],  # OneFormer
            results_map_ade["unet_smp"],  # U_Net
            results_map_ade["deeplab_tv"],  # DeepLabV3+
            results_map_ade["sam"],  # MobileSAM
            results_map_ade["sam2"],  # SAM2
            results_map_ade["dpt"],  # DPT-Large
            results_map_ade["upernet"],  # UPerNet
            results_map_ade["segformer_b2"],  # SegFormer-B2
            results_map_ade["fpn_mit"],  # FPN + MiT-B5
            results_map_ade["psp_mit"],  # PSPNet + MiT-B5
            results_map_ade["maskformer"],  # MaskFormer
            results_map_ade["fcn_tv"],  # FCN ResNet-50
            results_map_ade["maskrcnn_tv"],  # Mask R-CNN
            results_map_ade["segnet"],  # SegNet
            infer_res_ade,  # Ground_Truth
            segmentation_map,  # Orig_Mask
        ]

        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(3, 7, i + 1)
            plt.imshow(img)
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

        print("\n📊 Сводная таблица результатов:")
        df = pd.DataFrame(benchmark_ade.get_summary()).T
        print(df.sort_values("mIoU", ascending=False).to_string())

    if test_classic_logic is True:
        # ============ 4. ВАЛИДАЦИЯ (Torch vs OpenCV/Sklearn) ============
        print("\n4. Валидация реализаций...")
        validator = TorchImplementationValidator(output_dir="./data/validation")

        all_results = {}
        # test_images['ade20k_sample'][0]
        # test_images['countryside'][0]
        all_results = validator.validate_all_methods(test_images["mountain"][0])
        validator.generate_validation_report(all_results)
        print(f"\n✅ Все результаты сохранены в: {validator.output_dir}")

        # ============ 4.1. БЕНЧМАРК НА ОСНОВЕ ВАЛИДАЦИИ ============
        print("\n4.1. Генерация бенчмарк-отчета...")
        benchmark_df = validator.generate_benchmark_report_from_validation(
            all_results, output_dir="./data/validation_benchmark"
        )
        print(benchmark_df)

        print(f"\n✅ Все результаты сохранены в: {validator.output_dir}")
        print("📊 Бенчмарк-графики в: ./data/validation_benchmark/charts/")

        # ============ 5. МАТРИЧНОЕ СРАВНЕНИЕ ============
        print("\n5. Матричное сравнение методов...")
        comparator = SegmentationComparator()

        all_segmenters = {**cv2_methods, **sklearn_methods, **torch_methods}

        methods_config_list = [
            {
                "name": name,
                "segmenter": segmenter,
                # "type": "custom"
            }
            for name, segmenter in all_segmenters.items()
        ]

        ref_segmenter = sklearn_methods["Otsu_Thresholding_Sklearn"]
        original_segmenter = cv2_methods["Otsu_Thresholding_CV2"]

        for img_name, (img_path, img, gt) in test_images.items():
            img_array = np.array(img)
            print(f"\n--- Обработка изображения: {img_name} ---")
            try:
                results = comparator.matrix_comparison(
                    image=img_array,
                    methods_config=methods_config_list,
                    comparison_type="all_vs_all",
                    save_results=True,
                    output_dir=f"./data/matrix_comparison_{img_name}",
                )
                print(f"   ✅ Матрица сравнения для {img_name}")
                print(f"      - Сравнено пар: {len(results['df_comparisons'])}")
            except Exception as e:
                print(f"   ❌ Ошибка матричного сравнения: {e}")
                traceback.print_exc()

            try:
                df_results = comparator.batch_comparison(
                    image=img_array,
                    methods_config=methods_config_list,
                    reference_segmenter=ref_segmenter,
                    reference_name="Sklearn_Otsu_Ref",
                    save_results=True,
                    output_dir="./data/batch_comparison",
                )
                print("   ✅ Пакетное сравнение завершено. Топ-3 метода сохранены.")
                print(df_results)
            except Exception as e:
                print(f"   ❌ Ошибка пакетного сравнения: {e}")

            try:
                df_compare_methods = comparator.compare_methods(
                    image=img_array,
                    segmenter1=original_segmenter,
                    segmenter2=ref_segmenter,
                    name1="Original_CV2_Global",
                    name2="Reference_Sklearn_Otsu",
                    save_comparison=True,
                    output_path=f"./data/compare_methods_{img_name}",
                )
                print("   ✅ Попарное сравнение сохранено.")
                print(df_compare_methods)
            except TypeError as te:
                print(
                    f"   ⚠️ Метод compare_methods требует старой сигнатуры. Пропускаем или используем альтернативу. {te}"
                )
            except Exception as e:
                print(f"   ❌ Ошибка попарного сравнения: {e}")

    # # ============ 6. СРАВНЕНИЕ С GROUND TRUTH (если есть) ============
    # if test_classic_logic is True:
    #     print("\n6. Сравнение с Ground Truth и оценка качества...")
    #     has_gt_images = False
    #     for img_name, (img_path, img, gt_mask) in test_images.items():
    #         if gt_mask is None:
    #             print(f"⚠️ Пропуск {img_name}: Ground Truth не найден.")
    #             continue

    #         has_gt_images = True
    #         print(f"\n🎯 Обработка изображения: {img_name} (GT available)")
    #         print(f"🎯 Ground Truth найден ({gt_mask.shape}). Запуск оценки метрик...")
    #         metrics_all = {}

    #         if gt_mask.max() <= 1.0:
    #             gt_binary = (gt_mask * 255).astype(np.uint8)
    #         else:
    #             gt_binary = gt_mask.astype(np.uint8)

    #         # Запускаем бенчмарк вручную по каждому методу, чтобы сразу собрать метрики
    #         all_segmenters = {**cv2_methods, **sklearn_methods, **torch_methods}

    #         for name, segmenter in all_segmenters.items():
    #             try:
    #                 start_time = time.time()
    #                 pred_mask = segmenter.segment(img_path)
    #                 exec_time = time.time() - start_time

    #                 if pred_mask.shape != gt_binary.shape:
    #                     from skimage.transform import resize

    #                     # order=0 для бинарных масок (ближайший сосед)
    #                     pred_mask_resized = resize(
    #                         pred_mask, gt_binary.shape, order=0, preserve_range=True
    #                     ).astype(np.uint8)
    #                 else:
    #                     pred_mask_resized = pred_mask

    #                 metrics = SegmentationMetrics.calculate_all_metrics(
    #                     pred_mask, gt_binary, threshold=0.5, include_hausdorff=True
    #                 )
    #                 metrics["execution_time"] = exec_time  # Добавляем время в метрики
    #                 metrics_all[name] = metrics
    #                 status = (
    #                     "✅"
    #                     if metrics["iou"] > 0.5
    #                     else "⚠️" if metrics["iou"] > 0.2 else "❌"
    #                 )
    #                 print(
    #                     f"   {status} {name}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}, Time={exec_time:.3f}s"
    #                 )
    #                 print(f"Mask after {name} segment: {pred_mask_resized[:3, :3]}")

    #             except Exception as e:
    #                 print(f"   💥 Критическая ошибка в методе {name}: {e}")
    #                 traceback.print_exc()
    #                 metrics_all[name] = {"error": str(e)}
    #                 # traceback.print_exc()

    #         gt_results_summary[img_name] = metrics_all
    #         save_metrics_report(metrics_all, f"./data/gt_metrics_{img_name}.json")
    #         print(
    #             f"   💾 Детальные метрики сохранены в ./data/gt_metrics_{img_name}.json"
    #         )

    #     if not has_gt_images:
    #         print(
    #             "⚠️ Ground Truth маски не найдены ни для одного изображения. Пропускаем этап оценки качества."
    #         )
    #     else:
    #         # Запуск визуализации по всем изображениям с GT
    #         print("\n📈 Построение сводных графиков по результатам Ground Truth...")
    #         visualize_gt_results(
    #             gt_results_summary, output_dir="./data/gt_visualization"
    #         )

    #         # Вывод топ-5 методов в консоль
    #         print("\n🏆 ТОП-5 методов по среднему IoU:")
    #         # Плоский список всех результатов
    #         flat_results = []
    #         for img, methods in gt_results_summary.items():
    #             for method, metrics in methods.items():
    #                 if "iou" in metrics and "error" not in metrics:
    #                     flat_results.append(
    #                         {"Method": method, "IoU": metrics["iou"], "Image": img}
    #                     )

    #         if flat_results:
    #             df_flat = pd.DataFrame(flat_results)
    #             top_methods = (
    #                 df_flat.groupby("Method")["IoU"]
    #                 .mean()
    #                 .sort_values(ascending=False)
    #                 .head(5)
    #             )
    #             for i, (method, iou) in enumerate(top_methods.items(), 1):
    #                 print(f"   {i}. {method}: IoU = {iou:.4f}")
    #         else:
    #             print("   Нет успешных результатов для ранжирования.")

    #         print("\n" + "=" * 60)
    #         print("СВОДНЫЙ ОТЧЕТ ПО GROUND TRUTH")
    #         print("=" * 60)

    #         rows = []
    #         for img_name, methods_data in gt_results_summary.items():
    #             for method_name, metrics in methods_data.items():
    #                 if "error" not in metrics and "iou" in metrics:
    #                     rows.append(
    #                         {
    #                             "Image": img_name,
    #                             "Method": method_name,
    #                             "IoU": metrics["iou"],
    #                             "Dice": metrics["dice"],
    #                             "Precision": metrics["precision"],
    #                             "Recall": metrics["recall"],
    #                             "F1_Score": metrics["f1_score"],
    #                             "Time_s": metrics.get("execution_time", 0),
    #                         }
    #                     )

    #         if rows:
    #             df_gt = pd.DataFrame(rows)
    #             df_gt_sorted = df_gt.sort_values(
    #                 by=["Image", "IoU"], ascending=[True, False]
    #             )
    #             print("\nТоп методов по IoU:")
    #             print(
    #                 df_gt_sorted[
    #                     ["Method", "Image", "IoU", "Dice", "Time_s"]
    #                 ].to_string(index=False)
    #             )
    #             df_gt_sorted.to_csv("./data/gt_summary_report.csv", index=False)
    #             print("\n💾 Общая сводка сохранена в ./data/gt_summary_report.csv")
    #             plt.figure(figsize=(12, 6))
    #             first_img = list(gt_results_summary.keys())[0]
    #             df_plot = (
    #                 df_gt[df_gt["Image"] == first_img]
    #                 .sort_values("IoU", ascending=False)
    #                 .head(10)
    #             )
    #             if not df_plot.empty:
    #                 plt.barh(df_plot["Method"], df_plot["IoU"])
    #                 plt.xlabel("IoU Score")
    #                 plt.title(f"Top 10 Methods by IoU ({first_img})")
    #                 plt.xlim(0, 1)
    #                 plt.gca().invert_yaxis()
    #                 plt.tight_layout()
    #                 plt.savefig("./data/gt_iu_comparison_chart.png")
    #                 print("📊 График сохранен в ./data/gt_iu_comparison_chart.png")
    #             else:
    #                 print(
    #                     "⚠️ Не удалось построить график: нет данных для первого изображения."
    #                 )
    #             plt.close()
    #         else:
    #             print("Нет успешных метрик для отображения.")

    # print("\n" + "=" * 60)
    # print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    # print("=" * 60)
    # print(f"✓ Методов протестировано: {len(tester.methods)}")
    # print(f"✓ Изображений обработано: {len(test_images)}")
    # print("✓ Результаты в: ./data/")

    if test_neural_logic is True:
        # ====================================================================
        # 5. ОБУЧЕНИЕ МОДЕЛЕЙ С РАЗНЫМИ УРОВНЯМИ АУГМЕНТАЦИЙ
        # ====================================================================

        trainer = ModelTrainer(
            checkpoint_dir="./models", root_dir="./data/ade20k", device="cuda"
        )

        # Конфигурации для сравнения
        augmentation_configs: List[Dict[str, Any]] = [
            # {"level": "none", "epochs": 200, "lr": 1e-5, "subset_fraction": 0.05},
            {"level": "basic", "epochs": 200, "lr": 1e-5, "subset_fraction": 0.05},
            # {"level": "medium", "epochs": 200, "lr": 1e-5, "subset_fraction": 0.05},
            # {'level': 'aggressive', 'epochs': 50, 'lr': 5e-5},
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

        print(f"\n{'=' * 80}")
        print(
            f"ОБУЧЕНИЕ {len(model_types)} МОДЕЛЕЙ × {len(augmentation_configs)} УРОВНЕЙ АУГМЕНТАЦИЙ"
        )
        print(f"{'=' * 80}")

        for config in augmentation_configs:
            for model_type in model_types:
                print(f"\n{'=' * 60}")
                print(f"🔹 Модель: {model_type} | Аугментации: {config['level']}")
                print(f"{'=' * 60}")

                if model_type in ["fpn_smp", "psp_smp"]:
                    encoder_name = "mit_b5"
                else:
                    encoder_name = "resnet34"

                if model_type == "fcn_tv":
                    variant = "fcn_resnet50"
                elif "mit" in encoder_name:
                    variant = "b5"
                else:
                    variant = "b5"

                training_config = TrainingConfig(
                    experiment_name=f"{model_type}_aug_{config['level']}",
                    model_type=model_type,
                    augmentation_level=config["level"],
                    epochs=config["epochs"],
                    batch_size=4,
                    lr=config["lr"],
                    encoder_name=encoder_name,
                    variant=variant,
                    subset_fraction=0.05,
                )
                result = trainer.train_experiment(training_config)
                results_by_model_and_aug[model_type][config["level"]] = result
                print(
                    f"✅ {model_type} ({config['level']}): Best mIoU = {result['best_miou'] * 100:.2f}%"
                )

        # ====================================================================
        # 5.2. СРАВНЕНИЕ РЕЗУЛЬТАТОВ ПО АУГМЕНТАЦИЯМ
        # ====================================================================
        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ УРОВНЕЙ АУГМЕНТАЦИЙ")
        print("=" * 60)

        for model_type in model_types:
            print(f"\n🔹 {model_type}:")
            print("-" * 60)

            for level, result in results_by_model_and_aug[model_type].items():
                print(f"   {level:10s}: {result['best_miou'] * 100:6.2f}% mIoU")

        # ====================================================================
        # 5.3. ВИЗУАЛИЗАЦИЯ: Влияние аугментаций на каждую модель
        # ====================================================================
        trainer.plot_experiment_comparison(
            output_path="./data/augmentation_comparison.png"
        )

        print("\n" + "=" * 80)
        print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
        print("=" * 80)

        # График 1: Сравнение аугментаций для каждой модели
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, model_type in enumerate(model_types):
            ax = axes[idx]

            levels = list(results_by_model_and_aug[model_type].keys())
            miou_values = [
                results_by_model_and_aug[model_type][level]["best_miou"] * 100
                for level in levels
            ]

            colors = ["#ffcccc", "#ff9999", "#ff6666"][: len(levels)]
            bars = ax.bar(
                levels, miou_values, color=colors, edgecolor="black", linewidth=1.5
            )

            ax.set_title(f"{model_type}", fontsize=12, fontweight="bold")
            ax.set_ylabel("Best mIoU (%)")
            ax.set_ylim(0, max(miou_values) * 1.2)
            ax.grid(axis="y", alpha=0.3)

            # Добавляем значения на столбцы
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
            "./data/augmentation_comparison_all_models.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("📊 График сохранён: ./data/augmentation_comparison_all_models.png")
        plt.show()

        # График 2: Сравнение моделей для каждого уровня аугментаций
        fig, axes = plt.subplots(
            1, len(augmentation_configs), figsize=(6 * len(augmentation_configs), 5)
        )
        if len(augmentation_configs) == 1:
            axes = [axes]

        for idx, config in enumerate(augmentation_configs):
            ax = axes[idx]

            model_names = []
            miou_values = []

            for model_type in model_types:
                if config["level"] in results_by_model_and_aug[model_type]:
                    model_names.append(
                        model_type.replace("_smp", "").replace("_tv", "")
                    )
                    miou_values.append(
                        results_by_model_and_aug[model_type][config["level"]][
                            "best_miou"
                        ]
                        * 100
                    )

            colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
            bars = ax.bar(
                model_names, miou_values, color=colors, edgecolor="black", linewidth=1.5
            )

            ax.set_title(
                f'Аугментации: {config["level"]}', fontsize=12, fontweight="bold"
            )
            ax.set_ylabel("Best mIoU (%)")
            ax.set_ylim(0, max(miou_values) * 1.2)
            ax.grid(axis="y", alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

            # Добавляем значения на столбцы
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
            "./data/model_comparison_by_augmentation.png", dpi=300, bbox_inches="tight"
        )
        print("📊 График сохранён: ./data/model_comparison_by_augmentation.png")
        plt.show()

        # ====================================================================
        # 5.4. ОЦЕНКА ОБУЧЕННЫХ МОДЕЛЕЙ (compare_trained_models)
        # ====================================================================
        print("\n5.4. Оценка обученных моделей на валидации...")

        # Сравниваем модели, обученные с medium аугментациями
        comparison_results = trainer.compare_trained_models(augmentation_level="medium")
        print(f"Training results: {comparison_results}")

        # 🔥 Собираем чекпоинты для всех моделей с medium аугментациями
        checkpoints = {}

        for model_type in model_types:
            # Ищем чекпоинт с medium аугментациями
            pattern = f"./models/{model_type}_medium_*.pth"
            import glob

            files = glob.glob(pattern)

            if files:
                # Берём самый свежий
                checkpoint_path = max(files, key=os.path.getctime)
                checkpoints[f"{model_type} (medium)"] = checkpoint_path
                print(f"   ✅ {model_type} (medium): {checkpoint_path}")
            else:
                print(f"   ⚠️  {model_type} (medium): не найден")

        # Также добавим чекпоинты с none и basic для сравнения
        for model_type in model_types:
            for level in ["none", "basic"]:
                pattern = f"./models/{model_type}_{level}_*.pth"
                files = glob.glob(pattern)

                if files:
                    checkpoint_path = max(files, key=os.path.getctime)
                    checkpoints[f"{model_type} ({level})"] = checkpoint_path
                    print(f"   ✅ {model_type} ({level}): {checkpoint_path}")

        if checkpoints:
            eval_results = trainer.evaluate_checkpoints(
                checkpoint_paths=list(checkpoints.values()),
                model_type="unet_smp",  # 🔥 Нужно указать тип модели или сделать универсально
                # val_fraction=0.05,
            )

        # ====================================================================
        # 5.5. ОЦЕНКА ПО КОНКРЕТНЫМ ЧЕКПОИНТАМ (evaluate_trained_models_on_val)
        # ====================================================================
        print("\n5.5. Оценка по конкретным чекпоинтам...")

        checkpoints = {
            "U-Net (none)": "./models/unet_smp_none_*.pth",
            "U-Net (basic)": "./models/unet_smp_basic_*.pth",
            "U-Net (medium)": "./models/unet_smp_medium_*.pth",
            "DeepLabV3+ (none)": "./models/deeplab_tv_none_*.pth",
            "DeepLabV3+ (basic)": "./models/deeplab_tv_basic_*.pth",
            "DeepLabV3+ (medium)": "./models/deeplab_tv_medium_*.pth",
            "FPN+MiT-B5 (none)": "./models/fpn_smp_none_*.pth",
            "FPN+MiT-B5 (basic)": "./models/fpn_smp_basic_*.pth",
            "FPN+MiT-B5 (medium)": "./models/fpn_smp_medium_*.pth",
            "PSPNet+MiT-B5 (none)": "./models/psp_smp_none_*.pth",
            "PSPNet+MiT-B5 (basic)": "./models/psp_smp_basic_*.pth",
            "PSPNet+MiT-B5 (medium)": "./models/psp_smp_medium_*.pth",
            "FCN ResNet-50 (none)": "./models/fcn_tv_none_*.pth",
            "FCN ResNet-50 (basic)": "./models/fcn_tv_basic_*.pth",
            "FCN ResNet-50 (meduim)": "./models/fcn_tv_medium_*.pth",
            "SegNet (none)": "./models/segnet_none_*.pth",
            "SegNet (basic)": "./models/segnet_basic_*.pth",
            "SegNet (medium)": "./models/segnet_medium_*.pth",
        }

        # Находим актуальные файлы чекпоинтов
        import glob

        checkpoint_paths = {}
        for name, pattern in checkpoints.items():
            files = glob.glob(pattern)
            if files:
                checkpoint_paths[name] = max(files, key=os.path.getctime)
                print(f"   ✅ {name}: {checkpoint_paths[name]}")
            else:
                print(f"   ⚠️  {name}: не найден")

        if checkpoint_paths:
            eval_results = trainer.evaluate_trained_models_on_val(
                checkpoints=checkpoint_paths, val_fraction=0.05
            )
            print(eval_results)

        # ====================================================================
        # 5.6. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ
        # ====================================================================
        print("\n" + "=" * 80)
        print("СВОДНАЯ ТАБЛИЦА: ВЛИЯНИЕ АУГМЕНТАЦИЙ НА КАЧЕСТВО")
        print("=" * 80)

        summary_data = []
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
                            if result["final_val_loss"]
                            else "N/A"
                        ),
                    }
                )

        summary_df = pd.DataFrame(summary_data)
        # Сортировка по модели и mIoU
        summary_df = summary_df.sort_values(
            ["Model", "Best mIoU (%)"], ascending=[True, False]
        )
        print(summary_df.to_string(index=False))

        # Сохранение сводки
        summary_df.to_csv("./data/augmentation_impact_full_summary.csv", index=False)
        print("\n📊 Сводка сохранена: ./data/augmentation_impact_full_summary.csv")

        # ====================================================================
        # 5.7. ГРАФИК: Тепловая карта влияния аугментаций
        # ====================================================================
        print("\n" + "=" * 80)
        print("ТЕПЛОВАЯ КАРТА: МОДЕЛИ × АУГМЕНТАЦИИ")
        print("=" * 80)

        # Создаём pivot таблицу для тепловой карты
        pivot_data = summary_df.pivot(
            index="Model", columns="Augmentation Level", values="Best mIoU (%)"
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
        plt.savefig("./data/augmentation_heatmap.png", dpi=300, bbox_inches="tight")
        print("📊 Тепловая карта сохранена: ./data/augmentation_heatmap.png")
        plt.show()

        # ====================================================================
        # 5.8. ТОП-3 ЛУЧШИХ КОМБИНАЦИЙ
        # ====================================================================
        print("\n" + "=" * 80)
        print("🏆 ТОП-3 ЛУЧШИХ КОМБИНАЦИЙ (МОДЕЛЬ + АУГМЕНТАЦИИ)")
        print("=" * 80)

        top_combinations = summary_df.nlargest(3, "Best mIoU (%)")

        for idx, row in top_combinations.iterrows():
            print(f"\n{idx + 1}. {row['Model']} + {row['Augmentation Level']}")
            print(f"   mIoU: {row['Best mIoU (%)']:.2f}%")
            print(f"   Epochs: {row['Epochs']}")
            print(f"   Final Val Loss: {row['Final Val Loss']}")

        # Сохранение топ-3
        top_combinations.to_csv("./data/top_3_combinations.csv", index=False)
        print("\n📊 Топ-3 сохранён: ./data/top_3_combinations.csv")

    if test_classic_logic is True:
        print("\n" + "=" * 80)
        print("ДОПОЛНИТЕЛЬНЫЕ ИССЛЕДОВАНИЯ")
        print("=" * 80)

        # Выбираем тестовое изображение
        test_image = None
        for img_name, (img_path, img_pil, gt_mask) in test_images.items():
            test_image = np.array(img_pil)
            print(f"✅ Используем изображение: {img_name} ({test_image.shape})")
            break

        if test_image is not None:
            # Бенчмарк CPU vs CUDA для классических методов
            cpu_cuda_results = run_cpu_cuda_benchmark(
                cv2_methods=cv2_methods,
                sklearn_methods=sklearn_methods,
                torch_methods=torch_methods,
                test_image=test_image,
            )

        print(cpu_cuda_results)
    # ====================================================================
    # ДОПОЛНИТЕЛЬНОЕ ИССЛЕДОВАНИЕ: Массовое тестирование классических методов
    # ====================================================================
    if test_classic_logic:
        print("\n" + "=" * 80)
        print("🚀 ЗАПУСК МАССОВОГО ТЕСТИРОВАНИЯ КЛАССИЧЕСКИХ МЕТОДОВ")
        print("=" * 80)

        # Конфигурация теста
        batch_tester = BatchClassicTester(
            ade20k_root="./data/ade20k/ADEChallengeData2016",
            output_dir="./data/batch_classic_test",
            split="validation",  # или "training"
            max_images=50,  # Лимит для быстрого теста (None = все)
            image_size=(512, 512),
        )

        # Запуск тестирования
        print("⏳ Запуск массового тестирования...")
        results_df = batch_tester.run_batch_test()

        # Сохранение результатов
        csv_path, json_path, md_path = batch_tester.save_results(results_df)

        # Построение графиков
        batch_tester.plot_results(results_df)

        # Вывод сводки в консоль
        print("\n" + "=" * 80)
        print("📊 СВОДКА ПО МАССОВОМУ ТЕСТИРОВАНИЮ")
        print("=" * 80)

        print(f"\n🏆 Топ-5 методов по IoU:")
        for i, row in results_df.head(5).iterrows():
            print(
                f"   {i+1}. {row['Method']}: IoU={row['iou_mean']:.4f} ± {row['iou_std']:.4f}"
            )

        print(f"\n⚡ Топ-5 самых быстрых методов:")
        fast_df = (
            results_df.dropna(subset=["time_mean_s"]).sort_values("time_mean_s").head(5)
        )
        for i, row in fast_df.iterrows():
            print(f"   {i+1}. {row['Method']}: {row['time_mean_s']*1000:.1f} мс")

        print(f"\n❌ Методы с наибольшим числом ошибок:")
        error_df = (
            results_df[results_df["error_count"] > 0]
            .sort_values("error_count", ascending=False)
            .head(5)
        )
        if not error_df.empty:
            for i, row in error_df.iterrows():
                print(
                    f"   {row['Method']}: {row['error_count']} ошибок ({row['error_rate']*100:.1f}%)"
                )
        else:
            print("   Нет ошибок!")

        print(f"\n💾 Все результаты сохранены в: {batch_tester.output_dir}")

    return tester, results, comparator


def prepare_mask_for_overlay(mask_input) -> np.ndarray:
    """
    Конвертирует маску в 2D numpy array для create_overlay.

    Handles:
    - PIL Image (RGB or L)
    - numpy array with extra dimensions
    - RGB label images (converts to single channel)
    """

    # Конвертируем PIL → numpy если нужно
    if isinstance(mask_input, Image.Image):
        mask = np.array(mask_input)
    else:
        mask = np.array(mask_input)

    if mask.ndim == 3:
        if mask.shape[2] == 1:
            # (H, W, 1) → (H, W)
            mask = mask.squeeze(2)
        elif mask.shape[2] == 3:
            # RGB изображение → нужно конвертировать в классы
            # Для Cityscapes: используем первый канал или конвертируем через палитру
            print("⚠️  RGB mask detected, using first channel")
            mask = mask[:, :, 0]  # или используйте proper label conversion
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
    elif mask.ndim > 3:
        mask = np.squeeze(mask)

    # Финальная проверка
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D after processing, got {mask.ndim}D")
    return mask


def visualize_gt_results(
    results_dict: Dict[str, Dict], output_dir: str = "./data/gt_visualization"
):
    """
    Построение графиков по результатам тестирования с Ground Truth.

    Args:
        results_dict: Словарь вида {img_name: {method_name: metrics_dict}}
        output_dir: Папка для сохранения графиков
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. Сбор всех данных в единый DataFrame
    all_rows = []
    for img_name, methods_data in results_dict.items():
        for method_name, metrics in methods_data.items():
            if "error" in metrics or "iou" not in metrics:
                continue

            row: Dict[str, Any] = {
                "Image": img_name,
                "Method": method_name,
                "Library": (
                    method_name.split("_")[-1] if "_" in method_name else "Unknown"
                ),  # Извлекаем CV2/Sklearn/Torch
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
        print("⚠️ Нет данных для визуализации.")
        return
    df = pd.DataFrame(all_rows)
    # Группировка по методам для усреднения (если изображений несколько)
    df_avg = (
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
    )

    # Сортировка по IoU
    df_avg = df_avg.sort_values("IoU", ascending=False)

    # === ГРАФИК 1: Сравнение метрик (Bar Chart) ===
    plt.figure(figsize=(14, 8))
    x = range(len(df_avg))
    width = 0.25

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
    plt.title(
        "Сравнение метрик качества сегментации (среднее по изображениям)", fontsize=14
    )
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison_bar.png"), dpi=150)
    plt.close()
    print(f"📊 График метрик сохранен: {output_dir}/metrics_comparison_bar.png")

    # === ГРАФИК 2: Speed vs Accuracy (Scatter Plot) ===
    plt.figure(figsize=(10, 8))

    # Цвета для библиотек
    lib_colors = {
        "CV2": "#e67e22",
        "Sklearn": "#9b59b6",
        "Torch": "#34495e",
        "Neural": "#c0392b",
    }

    for lib, group in df_avg.groupby("Library"):
        color = lib_colors.get(lib, "#95a5a6")
        plt.scatter(
            group["Time_s"],
            group["IoU"],
            s=100,
            label=lib,
            color=color,
            alpha=0.7,
            edgecolors="black",
        )

        # Подписываем точки названиями методов
        for i, row in group.iterrows():
            plt.annotate(
                row["Method"][-10:],
                (row["Time_s"], row["IoU"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    plt.xlabel("Время выполнения (сек)")
    plt.ylabel("IoU Score")
    plt.title("Зависимость точности (IoU) от скорости работы", fontsize=14)
    plt.legend(title="Библиотека")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speed_vs_accuracy.png"), dpi=150)
    plt.close()
    print(f"🚀 График Speed vs Accuracy сохранен: {output_dir}/speed_vs_accuracy.png")

    # === ГРАФИК 3: Precision-Recall Balance ===
    plt.figure(figsize=(10, 6))
    plt.plot(df_avg["Recall"], df_avg["Precision"], "o-", linewidth=2, markersize=8)

    for i, row in df_avg.iterrows():
        plt.annotate(
            row["Method"][:15],
            (row["Recall"], row["Precision"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )

    plt.xlabel("Recall (Полнота)")
    plt.ylabel("Precision (Точность)")
    plt.title("Баланс Precision и Recall для методов сегментации")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_balance.png"), dpi=150)
    plt.close()
    print(
        f"⚖️ График Precision-Recall сохранен: {output_dir}/precision_recall_balance.png"
    )

    # Сохранение сводной таблицы CSV
    csv_path = os.path.join(output_dir, "gt_summary_table.csv")
    df_avg.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"💾 Сводная таблица сохранена: {csv_path}")


def load_test_images(
    use_image_with_mask: bool = False,
) -> Dict[str, Tuple[str, Image.Image, Optional[np.ndarray]]]:
    """
    Загружает тестовые изображения. Возвращает словарь:
    {
        'имя_изображения': (путь, PIL.Image, ground_truth_mask или None)
    }
    """

    test_images = {}

    if use_image_with_mask:
        print("📥 Попытка загрузки тестовых данных с масками (ADE20K)...")
        repo_id = "hf-internal-testing/fixtures_ade20k"
        print(f"Загрузка из репозитория: {repo_id}")

        img_path = hf_hub_download(
            repo_id=repo_id, filename="ADE_val_00000001.jpg", repo_type="dataset"
        )
        img = Image.open(img_path).convert("RGB")

        # Загружаем маску
        mask_path = hf_hub_download(
            repo_id=repo_id, filename="ADE_val_00000001.png", repo_type="dataset"
        )
        gt_mask_pil = Image.open(mask_path)

        print(f"✅ Изображение загружено: {os.path.basename(img_path)}")
        print(f"✅ Ground truth загружен: {os.path.basename(mask_path)}")
        print(f"Ground Truth: {mask_path}")
        print(f"Размер изображения: {img.size}")
        print(f"Размер GT: {gt_mask_pil.size}")

        # Конвертируем маску в бинарную (все, что не фон/черный, считаем объектом для простоты)
        # В ADE20K классы цветные. Для бинарной сегментации просто возьмем все ненулевые пиксели
        # или самый частый класс как фон.
        gt_np = np.array(gt_mask_pil)
        print(
            f"\nДиапазон значений Ground Truth: {gt_np.min()} - {gt_np.max()}, min: {gt_np.min()}, max: {gt_np.max()}"
        )

        # Эвристика: самый частый цвет - фон. Все остальное - объект.
        unique, counts = np.unique(gt_np, return_counts=True)
        bg_class = unique[np.argmax(counts)]
        print(
            f"📊 Статистика GT: Всего классов {len(unique)}. Самый частый: {bg_class} ({np.max(counts)} пикселей)"
        )

        binary_gt = (gt_np != bg_class).astype(np.uint8) * 255

        if np.sum(binary_gt > 0) < (binary_gt.size * 0.01):
            print(
                "⚠️ Объектов слишком мало по стратегии 'не фон'. Пробуем взять второй по величине класс."
            )
            if len(unique) > 1:
                second_common = unique[np.argsort(counts)[-2]]
                binary_gt = (gt_np == second_common).astype(np.uint8) * 255
            else:
                # Если класс всего один, берем всё изображение как объект
                binary_gt = np.ones_like(gt_np, dtype=np.uint8) * 255

        # Сохраняем локально
        local_img_path = "test_images/test_gt_image.jpg"
        local_mask_path_raw = "test_images/test_gt_mask_raw.png"
        local_mask_path = "test_images/test_gt_mask.png"
        img.save(local_img_path)
        print(f"✅ Изображение сохранено локально: {local_img_path}")

        gt_mask_pil.save(local_mask_path_raw)
        print(f"✅ Изображение сырой маски сохранено локально: {local_mask_path_raw}")

        Image.fromarray(binary_gt).save(local_mask_path)
        print(f"✅ Изображение маски сохранено локально: {local_mask_path}")

        test_images["ade20k_sample"] = (local_img_path, img, binary_gt)
        print(f"✅ Загружен образец ADE20K: {img.size}, GT: {binary_gt.shape}")

    else:
        print("⚠️ Не удалось загрузить реальные GT. Используем только изображения.")

        # Примеры изображений с возможными ground truth
        image_urls = {
            "countryside": "https://i.pinimg.com/736x/17/e7/fc/17e7fc299466b2afd989e709fe7c9815.jpg",
            "nature": "https://i.pinimg.com/736x/f7/5a/f2/f75af26820b50c24600f50f3998eb02f.jpg",
            "architecture": "https://i.pinimg.com/736x/86/f6/07/86f60748d5d9ae4cb9092018d1321648.jpg",
            "trucks": "https://www.shutterstock.com/shutterstock/videos/1106252821/thumb/1.jpg?ip=x480",
            "traffic": "https://images.pond5.com/pov-car-and-truck-traffic-footage-190002081_iconl.jpeg",
            "mountain": "https://i.pinimg.com/736x/17/66/c4/1766c4f667af39f91172ef8eb21ab18a.jpg",
        }

        image_paths = {
            "war_frame_1": "test_images/2340_frame.jpg",
            "war_frame_2": "test_images/3330_frame.jpg",
            "war_frame_3": "test_images/4130_frame.jpg",
            "war_frame_4": "test_images/4480_frame.jpg",
            "building": "test_images/test_gt_image.jpg",
            "animals": "test_images/animals.jpg",
        }

        for name, url in image_urls.items():
            try:
                response = requests.get(url, timeout=10)
                img = Image.open(BytesIO(response.content)).convert("RGB")

                local_path = f"test_images/test_image_{name}.jpg"
                img.save(local_path)

                gt_synthetic = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
                gt_synthetic[img.size[1] // 2 :, :] = 255
                test_images[name] = (local_path, img, gt_synthetic)

                print(f"✅ {name}: {img.size}, ground truth: {gt_synthetic}")

            except Exception as e:
                print(f"❌ Ошибка загрузки {name}: {e}")

        for name, path in image_paths.items():
            try:
                img = Image.open(path)
                gt_synthetic = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
                gt_synthetic[img.size[1] // 2 :, :] = 255
                test_images[name] = (path, img, gt_synthetic)
                print(f"✅ {name}: {img.size}, ground truth: {gt_synthetic}")

            except Exception as e:
                print(f"❌ Ошибка загрузки {name}: {e}")
    return test_images


def save_metrics_report(metrics_all: Dict, path: str):
    """Сохранение отчёта с метриками"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False, default=str)


def test_neural_segmentation_variants():
    """Тестирование различных вариантов нейросетевой сегментации"""

    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ВАРИАНТОВ НЕЙРОСЕТЕВОЙ СЕГМЕНТАЦИИ")
    print("=" * 60)

    # Загрузка тестового изображения
    img_url = "https://images.pond5.com/pov-car-and-truck-traffic-footage-190002081_iconl.jpeg"

    try:
        response = requests.get(img_url)
        test_image = Image.open(BytesIO(response.content))

        # Создаем нейросетевой сегментатор
        segmenter = NeuralSegmenter(
            local_path="/home/yamshchikov/models/segformer-b5-ready"
        )

        # Вариант 1: Различные значения alpha
        print("\n1. Тестирование разных значений alpha:")
        alphas = [0.3, 0.5, 0.7, 1.0]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, alpha in enumerate(alphas):
            result = segmenter.segment_image(test_image, alpha=alpha)
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
        detailed_result = segmenter.detailed_segmentation(test_image)

        # Выводим информацию о классах
        print(f"Обнаружено классов: {detailed_result['total_classes']}")
        print("\nТоп-5 классов по площади:")

        sorted_classes = sorted(
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
        result_np, mask = segmenter.segment_with_mask(test_image, alpha=0.5)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(test_image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(result_np)
        axes[1].set_title("Segmentation Result")
        axes[1].axis("off")

        axes[2].imshow(mask, cmap="gray")
        axes[2].set_title("Binary Mask")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

        print(f"Размер маски: {mask.shape}")
        print(f"Площадь маски: {np.sum(mask > 0)} пикселей")

        return segmenter, detailed_result

    except Exception as e:
        print(f"❌ Ошибка тестирования нейросетевых вариантов: {e}")
        print(traceback.format_exc())
        return None, None


# main.py (добавить в конец, перед return)


def run_cpu_cuda_benchmark(
    cv2_methods: Dict,
    sklearn_methods: Dict,
    torch_methods: Dict,
    test_image: np.ndarray,
):
    """
    Запуск бенчмарка CPU vs CUDA для классических методов.
    """
    from testing.CpuCudaBenchmark import CpuCudaBenchmark

    print("\n" + "=" * 80)
    print("ЗАПУСК БЕНЧМАРКА: CPU vs CUDA")
    print("=" * 80)

    # Объединяем все классические методы
    all_classical_methods = {**cv2_methods, **sklearn_methods, **torch_methods}

    # Исключаем нейронные сети из этого бенчмарка
    classical_only = {
        name: seg
        for name, seg in all_classical_methods.items()
        if not any(
            x in name.lower()
            for x in [
                "neural",
                "segformer",
                "mask2former",
                "unet",
                "fpn",
                "psp",
                "fcn",
                "deeplab",
            ]
        )
    }

    print(f"Количество классических методов: {len(classical_only)}")

    # Создаём бенчмарк
    benchmark = CpuCudaBenchmark(
        base_output_dir="./data/cpu_cuda_benchmark",
        n_runs=5,  # Количество прогонов
        warmup_runs=2,  # Warm-up прогонов
    )

    # Запускаем бенчмарк
    df_results = benchmark.benchmark_all_methods(
        methods_dict=classical_only,
        image=test_image,
        test_name="classical_methods_cpu_cuda",
    )

    # Вывод сводки
    print("\n" + "=" * 80)
    print("СВОДКА ПО БЕНЧМАРКУ CPU vs CUDA")
    print("=" * 80)

    if torch.cuda.is_available():
        # Группировка по методам
        for method in df_results["method"].unique():
            method_data = df_results[df_results["method"] == method]
            cpu_data = method_data[method_data["device"] == "cpu"]
            cuda_data = method_data[method_data["device"] == "cuda"]

            if not cpu_data.empty and not cuda_data.empty:
                cpu_time = cpu_data["mean_time"].values[0] * 1000
                cuda_time = cuda_data["mean_time"].values[0] * 1000

                if cuda_time > 0:
                    speedup = cpu_time / cuda_time
                    print(
                        f"{method:40s}: CPU={cpu_time:7.2f}ms, CUDA={cuda_time:7.2f}ms, ⚡ Speedup={speedup:.2f}x"
                    )

    print("=" * 80)

    return df_results


if __name__ == "__main__":
    # Основной тест
    print("ЗАПУСК ОСНОВНОГО ТЕСТА")
    print("=" * 60)
    tester, results, comparator = main()

    print("\n\nЗАПУСК ДОПОЛНИТЕЛЬНОГО ТЕСТА НЕЙРОСЕТЕВЫХ ВАРИАНТОВ")
    print("=" * 60)
    segmenter, detailed_result = test_neural_segmentation_variants()
