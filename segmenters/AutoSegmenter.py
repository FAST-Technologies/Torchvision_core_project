# segmenters/AutoSegmenter.py

import os

if os.getenv("TRACK_FUNCTION_CALLS") == "1":
    from utils.function_tracker import track_calls
else:

    def track_calls(f):
        return f


from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
from dataclasses import dataclass
from enum import Enum
from dataclasses import field


class SegmentationGoal(Enum):
    """Цели сегментации"""

    SPEED = "speed"  # Максимальная скорость
    ACCURACY = "accuracy"  # Максимальная точность
    BALANCED = "balanced"  # Баланс
    LOW_MEMORY = "low_memory"  # Минимальное потребление памяти


class ImageType(Enum):
    """Типы изображений"""

    MEDICAL = "medical"  # МРТ, КТ, рентген
    NATURAL = "natural"  # Фотографии
    DOCUMENT = "document"  # Текст, документы
    SATELLITE = "satellite"  # Спутниковые снимки
    INDUSTRIAL = "industrial"  # Дефекты, контроль качества
    MICROSCOPY = "microscopy"  # Микроскопия
    UNKNOWN = "unknown"  # В случае, если датасет неизвестен


@dataclass
class ImageCharacteristics:
    """Характеристики изображения"""

    width: int
    height: int
    channels: int
    mean_intensity: float
    std_intensity: float
    contrast: float
    noise_level: float
    edge_density: float
    complexity_score: float
    estimated_type: ImageType


@dataclass
class MethodProfile:
    """Профиль метода (из бенчмарков)"""

    name: str
    library: str  # "opencv" | "sklearn" | "torch"
    avg_time_ms: float
    avg_iou: float
    memory_mb: float
    best_for_type: List[ImageType]
    robustness: float  # Устойчивость к шуму
    parameter_sensitivity: float  # Чувствительность к параметрам
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    schema: Dict[str, Any] = field(default_factory=dict)


MethodConfig = Tuple[str, Dict[str, Any]]

METHODS_BY_LIBRARY = {
    "opencv": {
        "global_thresholding": MethodProfile(
            name="global_thresholding",
            library="opencv",
            avg_time_ms=2.0,
            avg_iou=0.62,
            memory_mb=15,
            best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
            robustness=0.4,
            parameter_sensitivity=0.9,
            description="Простое глобальное пороговое значение (OpenCV)",
            params={"threshold": 0.5},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог яркости (0-1)",
                }
            },
        ),
        "otsu_thresholding": MethodProfile(
            name="otsu_thresholding",
            library="opencv",
            avg_time_ms=15.0,
            avg_iou=0.75,
            memory_mb=50,
            best_for_type=[ImageType.DOCUMENT, ImageType.NATURAL],
            robustness=0.8,
            parameter_sensitivity=0.2,
            description="Автоматический порог Оцу (максимизация межклассовой дисперсии) (OpenCV)",
            params={},
            schema={},
        ),
        "adaptive_thresholding": MethodProfile(
            name="adaptive_thresholding",
            library="opencv",
            avg_time_ms=45.0,
            avg_iou=0.82,
            memory_mb=80,
            best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
            robustness=0.9,
            parameter_sensitivity=0.4,
            description="Адаптивный порог с локальным усреднением (OpenCV)",
            params={"block_size": 11, "C": 2},
            schema={
                "block_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер блока (нечетный)",
                },
                "C": {
                    "type": "int",
                    "min": -20,
                    "max": 20,
                    "step": 1,
                    "label": "Константа C (смещение)",
                },
            },
        ),
        "threshold_niblack": MethodProfile(
            name="threshold_niblack",
            library="opencv",
            avg_time_ms=38.0,
            avg_iou=0.71,
            memory_mb=65,
            best_for_type=[ImageType.DOCUMENT, ImageType.MICROSCOPY],
            robustness=0.6,
            parameter_sensitivity=0.7,
            description="Метод Ниблэка для локального порогования (OpenCV)",
            params={"window_size": 15, "k": -0.2},
            schema={
                "window_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер окна",
                },
                "k": {
                    "type": "float",
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Константа k",
                },
            },
        ),
        "threshold_sauvola": MethodProfile(
            name="threshold_sauvola",
            library="opencv",
            avg_time_ms=42.0,
            avg_iou=0.88,
            memory_mb=70,
            best_for_type=[ImageType.DOCUMENT, ImageType.MICROSCOPY],
            robustness=0.92,
            parameter_sensitivity=0.3,
            description="Метод Саволы (улучшенный Ниблэк для текста) (OpenCV)",
            params={"window_size": 15, "k": 0.5, "r": 128},
            schema={
                "window_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер окна",
                },
                "k": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Константа k",
                },
                "r": {
                    "type": "float",
                    "min": 50.0,
                    "max": 255.0,
                    "step": 1.0,
                    "label": "Динамический диапазон R",
                },
            },
        ),
        "threshold_bernsen": MethodProfile(
            name="threshold_bernsen",
            library="opencv",
            avg_time_ms=35.0,
            avg_iou=0.73,
            memory_mb=60,
            best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
            robustness=0.7,
            parameter_sensitivity=0.5,
            description="Метод Бернсена на основе локального контраста (OpenCV)",
            params={"window_size": 15, "contrast_threshold": 0.15},
            schema={
                "window_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер окна",
                },
                "contrast_threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог контраста",
                },
            },
        ),
        "threshold_phansalkar": MethodProfile(
            name="threshold_phansalkar",
            library="opencv",
            avg_time_ms=48.0,
            avg_iou=0.85,
            memory_mb=75,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.88,
            parameter_sensitivity=0.4,
            description="Метод Фансалкара для низкоконтрастных изображений (OpenCV)",
            params={"window_size": 15, "k": 0.25, "r": 128.0, "m": 0.5},
            schema={
                "window_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер окна",
                },
                "k": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Чувствительность k",
                },
                "r": {
                    "type": "float",
                    "min": 50.0,
                    "max": 255.0,
                    "step": 1.0,
                    "label": "Диапазон R",
                },
                "m": {
                    "type": "float",
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "label": "Смещение m",
                },
            },
        ),
        "threshold_kittler_illingworth": MethodProfile(
            name="threshold_kittler_illingworth",
            library="opencv",
            avg_time_ms=25.0,
            avg_iou=0.76,
            memory_mb=55,
            best_for_type=[ImageType.DOCUMENT, ImageType.NATURAL],
            robustness=0.75,
            parameter_sensitivity=0.3,
            description="Минимизация ошибки классификации (Киттлер-Иллингуорт) (OpenCV)",
            params={"num_bins": 256},
            schema={
                "num_bins": {
                    "type": "int",
                    "min": 32,
                    "max": 512,
                    "step": 16,
                    "label": "Кол-во бинов гистограммы",
                }
            },
        ),
        "threshold_entropy_kapur": MethodProfile(
            name="threshold_entropy_kapur",
            library="opencv",
            avg_time_ms=30.0,
            avg_iou=0.74,
            memory_mb=60,
            best_for_type=[ImageType.NATURAL, ImageType.SATELLITE],
            robustness=0.7,
            parameter_sensitivity=0.4,
            description="Максимизация энтропии (Капур) (OpenCV)",
            params={"num_bins": 256},
            schema={
                "num_bins": {
                    "type": "int",
                    "min": 32,
                    "max": 512,
                    "step": 16,
                    "label": "Кол-во бинов гистограммы",
                }
            },
        ),
        "threshold_triangle": MethodProfile(
            name="threshold_triangle",
            library="opencv",
            avg_time_ms=20.0,
            avg_iou=0.69,
            memory_mb=45,
            best_for_type=[ImageType.DOCUMENT, ImageType.MEDICAL],
            robustness=0.65,
            parameter_sensitivity=0.3,
            description="Треугольный метод для унимодальных гистограмм (OpenCV)",
            params={"num_bins": 256},
            schema={
                "num_bins": {
                    "type": "int",
                    "min": 32,
                    "max": 512,
                    "step": 16,
                    "label": "Кол-во бинов гистограммы",
                }
            },
        ),
        "threshold_multi_otsu": MethodProfile(
            name="threshold_multi_otsu",
            library="opencv",
            avg_time_ms=35.0,
            avg_iou=0.78,
            memory_mb=70,
            best_for_type=[ImageType.MEDICAL, ImageType.SATELLITE],
            robustness=0.8,
            parameter_sensitivity=0.5,
            description="Многопороговый Оцу для многоклассовой сегментации (OpenCV)",
            params={"n_thresholds": 2},
            schema={
                "n_thresholds": {
                    "type": "int",
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "label": "Кол-во порогов",
                }
            },
        ),
        "threshold_percentile": MethodProfile(
            name="threshold_percentile",
            library="opencv",
            avg_time_ms=8.0,
            avg_iou=0.65,
            memory_mb=25,
            best_for_type=[ImageType.INDUSTRIAL, ImageType.DOCUMENT],
            robustness=0.5,
            parameter_sensitivity=0.8,
            description="Порог по перцентилю интенсивности (OpenCV)",
            params={"percentile": 90},
            schema={
                "percentile": {
                    "type": "int",
                    "min": 1,
                    "max": 99,
                    "step": 1,
                    "label": "Процентиль (%)",
                }
            },
        ),
        "threshold_local_contrast": MethodProfile(
            name="threshold_local_contrast",
            library="opencv",
            avg_time_ms=40.0,
            avg_iou=0.77,
            memory_mb=68,
            best_for_type=[ImageType.MICROSCOPY, ImageType.MEDICAL],
            robustness=0.82,
            parameter_sensitivity=0.5,
            description="Порог на основе локального контраста (OpenCV)",
            params={"window_size": 15, "contrast_factor": 0.1},
            schema={
                "window_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер окна",
                },
                "contrast_factor": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Фактор контраста",
                },
            },
        ),
        # ===== EDGE DETECTION =====
        "canny_edge": MethodProfile(
            name="canny_edge",
            library="opencv",
            avg_time_ms=25.0,
            avg_iou=0.68,
            memory_mb=60,
            best_for_type=[ImageType.NATURAL, ImageType.INDUSTRIAL],
            robustness=0.7,
            parameter_sensitivity=0.6,
            description="Детектор границ Кэнни (оптимальный по Кэнни) (OpenCV)",
            params={"low": 0.1, "high": 0.3, "sigma": 1.0},
            schema={
                "low": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Нижний порог",
                },
                "high": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Верхний порог",
                },
                "sigma": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Сигма (размытие)",
                },
            },
        ),
        "sobel_edge": MethodProfile(
            name="sobel_edge",
            library="opencv",
            avg_time_ms=12.0,
            avg_iou=0.58,
            memory_mb=35,
            best_for_type=[ImageType.NATURAL, ImageType.DOCUMENT],
            robustness=0.5,
            parameter_sensitivity=0.7,
            description="Градиенты Собеля с порогом (OpenCV)",
            params={"threshold": 0.1},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог градиента",
                }
            },
        ),
        "prewitt_edge": MethodProfile(
            name="prewitt_edge",
            library="opencv",
            avg_time_ms=11.0,
            avg_iou=0.56,
            memory_mb=33,
            best_for_type=[ImageType.NATURAL, ImageType.DOCUMENT],
            robustness=0.48,
            parameter_sensitivity=0.72,
            description="Градиенты Превитта с порогом (OpenCV)",
            params={"threshold": 0.1},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог градиента",
                }
            },
        ),
        "scharr_edge": MethodProfile(
            name="scharr_edge",
            library="opencv",
            avg_time_ms=14.0,
            avg_iou=0.61,
            memory_mb=38,
            best_for_type=[ImageType.NATURAL, ImageType.INDUSTRIAL],
            robustness=0.55,
            parameter_sensitivity=0.65,
            description="Градиенты Шарра (OpenCV)",
            params={"threshold": 0.1},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог градиента",
                }
            },
        ),
        "roberts_cross_edge": MethodProfile(
            name="roberts_cross_edge",
            library="opencv",
            avg_time_ms=8.0,
            avg_iou=0.52,
            memory_mb=28,
            best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
            robustness=0.4,
            parameter_sensitivity=0.8,
            description="Оператор Робертса для диагональных границ (OpenCV)",
            params={"threshold": 0.1},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог градиента",
                }
            },
        ),
        "log_edge": MethodProfile(
            name="log_edge",
            library="opencv",
            avg_time_ms=22.0,
            avg_iou=0.64,
            memory_mb=55,
            best_for_type=[ImageType.NATURAL, ImageType.MEDICAL],
            robustness=0.6,
            parameter_sensitivity=0.5,
            description="Laplacian of Gaussian детектор границ (OpenCV)",
            params={"sigma": 1.0, "threshold": 0.01},
            schema={
                "sigma": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Сигма размытия",
                },
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог детекции",
                },
            },
        ),
        "dog_edge": MethodProfile(
            name="dog_edge",
            library="opencv",
            avg_time_ms=28.0,
            avg_iou=0.66,
            memory_mb=62,
            best_for_type=[ImageType.NATURAL, ImageType.SATELLITE],
            robustness=0.68,
            parameter_sensitivity=0.55,
            description="Difference of Gaussians для мультимасштабных границ (OpenCV)",
            params={"sigma1": 1.0, "sigma2": 2.0, "threshold": 0.01},
            schema={
                "sigma1": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Сигма 1",
                },
                "sigma2": {
                    "type": "float",
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1,
                    "label": "Сигма 2",
                },
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог детекции",
                },
            },
        ),
        "marr_hildreth_edge": MethodProfile(
            name="marr_hildreth_edge",
            library="opencv",
            avg_time_ms=26.0,
            avg_iou=0.63,
            memory_mb=58,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.62,
            parameter_sensitivity=0.58,
            description="Метод Марра-Хилдрета (нулевые пересечения LoG) (OpenCV)",
            params={"sigma": 1.5, "threshold": 0.01},
            schema={
                "sigma": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Сигма размытия",
                },
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог нулевых пересечений",
                },
            },
        ),
        "gradient_magnitude_direction": MethodProfile(
            name="gradient_magnitude_direction",
            library="opencv",
            avg_time_ms=18.0,
            avg_iou=0.59,
            memory_mb=45,
            best_for_type=[ImageType.INDUSTRIAL, ImageType.NATURAL],
            robustness=0.52,
            parameter_sensitivity=0.68,
            description="Сегментация по величине и направлению градиента (OpenCV)",
            params={"threshold": 0.1},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог магнитуды",
                }
            },
        ),
        "phase_congruency_edge": MethodProfile(
            name="phase_congruency_edge",
            library="opencv",
            avg_time_ms=85.0,
            avg_iou=0.79,
            memory_mb=120,
            best_for_type=[
                ImageType.MEDICAL,
                ImageType.SATELLITE,
                ImageType.MICROSCOPY,
            ],
            robustness=0.95,
            parameter_sensitivity=0.3,
            description="Фазовая конгруэнтность (инвариантна к освещению) (OpenCV)",
            params={
                "nscales": 4,
                "norientations": 4,
                "min_wavelength": 3,
                "mult": 2.0,
                "sigma_onf": 0.55,
                "k_noise": 2.0,
                "threshold": 0.5,
            },
            schema={
                "nscales": {
                    "type": "int",
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "label": "Кол-во масштабов",
                },
                "norientations": {
                    "type": "int",
                    "min": 1,
                    "max": 12,
                    "step": 1,
                    "label": "Кол-во ориентаций",
                },
                "min_wavelength": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "label": "Мин. длина волны",
                },
                "mult": {
                    "type": "float",
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Множитель масштаба",
                },
                "sigma_onf": {
                    "type": "float",
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "label": "Сигма частотной области",
                },
                "k_noise": {
                    "type": "float",
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Коэф. шумоподавления",
                },
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог энергии",
                },
            },
        ),
        # ===== REGION-BASED =====
        "region_growing": MethodProfile(
            name="region_growing",
            library="opencv",
            avg_time_ms=55.0,
            avg_iou=0.81,
            memory_mb=90,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.75,
            parameter_sensitivity=0.6,
            description="Рост региона от семян по схожести (OpenCV)",
            params={"tolerance": 0.1},
            schema={
                "tolerance": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Допуск схожести",
                }
            },
        ),
        "split_and_merge": MethodProfile(
            name="split_and_merge",
            library="opencv",
            avg_time_ms=70.0,
            avg_iou=0.76,
            memory_mb=100,
            best_for_type=[ImageType.SATELLITE, ImageType.INDUSTRIAL],
            robustness=0.7,
            parameter_sensitivity=0.5,
            description="Разделение и слияние регионов (OpenCV)",
            params={"min_size": 50, "threshold": 20},
            schema={
                "min_size": {
                    "type": "int",
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "label": "Мин. размер региона",
                },
                "threshold": {
                    "type": "int",
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "label": "Порог слияния",
                },
            },
        ),
        "floodfill": MethodProfile(
            name="floodfill",
            library="opencv",
            avg_time_ms=15.0,
            avg_iou=0.72,
            memory_mb=40,
            best_for_type=[ImageType.DOCUMENT, ImageType.MEDICAL],
            robustness=0.6,
            parameter_sensitivity=0.7,
            description="Заливка области от точки (OpenCV)",
            params={"tolerance": 0.15},
            schema={
                "tolerance": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Допуск заливки",
                }
            },
        ),
        # ===== CLUSTERING =====
        "kmeans_segmentation": MethodProfile(
            name="kmeans_segmentation",
            library="opencv",
            avg_time_ms=120.0,
            avg_iou=0.77,
            memory_mb=150,
            best_for_type=[ImageType.NATURAL, ImageType.SATELLITE],
            robustness=0.65,
            parameter_sensitivity=0.7,
            description="K-means кластеризация в пространстве признаков (OpenCV)",
            params={"k": 3},
            schema={
                "k": {
                    "type": "int",
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "label": "Кол-во кластеров",
                }
            },
        ),
        "dbscan_segmentation": MethodProfile(
            name="dbscan_segmentation",
            library="opencv",
            avg_time_ms=180.0,
            avg_iou=0.74,
            memory_mb=200,
            best_for_type=[ImageType.MICROSCOPY, ImageType.INDUSTRIAL],
            robustness=0.8,
            parameter_sensitivity=0.6,
            description="DBSCAN для сегментации произвольной формы (OpenCV)",
            params={"eps": 0.1, "min_samples": 10},
            schema={
                "eps": {
                    "type": "float",
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Радиус окрестности (eps)",
                },
                "min_samples": {
                    "type": "int",
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "label": "Мин. точек в кластере",
                },
            },
        ),
        "meanshift": MethodProfile(
            name="meanshift",
            library="opencv",
            avg_time_ms=250.0,
            avg_iou=0.83,
            memory_mb=280,
            best_for_type=[ImageType.NATURAL, ImageType.MEDICAL],
            robustness=0.85,
            parameter_sensitivity=0.4,
            description="MeanShift с пространственно-цветовым ядром (OpenCV)",
            params={
                "bandwidth": 0.5,
                "spatial_radius": 35,
                "color_radius": 60,
                "max_level": 1,
            },
            schema={
                "bandwidth": {
                    "type": "float",
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Полоса пропускания",
                },
                "spatial_radius": {
                    "type": "int",
                    "min": 5,
                    "max": 100,
                    "step": 1,
                    "label": "Пространственный радиус",
                },
                "color_radius": {
                    "type": "int",
                    "min": 10,
                    "max": 200,
                    "step": 1,
                    "label": "Цветовой радиус",
                },
                "max_level": {
                    "type": "int",
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "label": "Макс. уровень пирамиды",
                },
            },
        ),
        # ===== ACTIVE CONTOURS =====
        "active_contour": MethodProfile(
            name="active_contour",
            library="opencv",
            avg_time_ms=450.0,
            avg_iou=0.84,
            memory_mb=180,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.78,
            parameter_sensitivity=0.75,
            description="Змеи (snakes) с энергией границ и линий (OpenCV)",
            params={
                "alpha": 0.015,
                "beta": 10,
                "gamma": 0.001,
                "max_iterations": 2000,
                "w_edge": 1,
                "w_line": 0,
            },
            schema={
                "alpha": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "label": "Плавность контура",
                },
                "beta": {
                    "type": "float",
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1,
                    "label": "Жесткость контура",
                },
                "gamma": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "label": "Вязкость (шаг)",
                },
                "max_iterations": {
                    "type": "int",
                    "min": 100,
                    "max": 5000,
                    "step": 100,
                    "label": "Итерации",
                },
                "w_edge": {
                    "type": "float",
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Вес границ",
                },
                "w_line": {
                    "type": "float",
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Вес линий",
                },
            },
        ),
        "gvf_contour": MethodProfile(
            name="gvf_contour",
            library="opencv",
            avg_time_ms=380.0,
            avg_iou=0.86,
            memory_mb=160,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.88,
            parameter_sensitivity=0.5,
            description="Контуры с градиентным векторным потоком (GVF) (OpenCV)",
            params={"mu": 0.1, "iterations": 50},
            schema={
                "mu": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Коэф. диффузии (mu)",
                },
                "iterations": {
                    "type": "int",
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "label": "Итерации GVF",
                },
            },
        ),
        "morphological_snakes": MethodProfile(
            name="morphological_snakes",
            library="opencv",
            avg_time_ms=320.0,
            avg_iou=0.87,
            memory_mb=140,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.92,
            parameter_sensitivity=0.35,
            description="Морфологические змеи (устойчивы к шуму) (OpenCV)",
            params={"iterations": 100, "smoothing": 1, "threshold": 0.5},
            schema={
                "iterations": {
                    "type": "int",
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "label": "Итерации",
                },
                "smoothing": {
                    "type": "int",
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "label": "Степень сглаживания",
                },
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог инициализации",
                },
            },
        ),
        "chan_vese": MethodProfile(
            name="chan_vese",
            library="opencv",
            avg_time_ms=400.0,
            avg_iou=0.89,
            memory_mb=170,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.94,
            parameter_sensitivity=0.3,
            description="Модель Чан-Везе (регион-базированные активные контуры) (OpenCV)",
            params={
                "mu": 0.25,
                "lambda1": 1.0,
                "lambda2": 1.0,
                "tol": 1e-3,
                "max_iter": 100,
                "dt": 0.5,
                "eps": 1.0,
                "init_level_set": "checkerboard",
            },
            schema={
                "mu": {
                    "type": "float",
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "label": "Длина контура (mu)",
                },
                "lambda1": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Внешняя область (lambda1)",
                },
                "lambda2": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Внутренняя область (lambda2)",
                },
                "tol": {
                    "type": "float",
                    "min": 0.0001,
                    "max": 0.01,
                    "step": 0.0001,
                    "label": "Точность сходимости",
                },
                "max_iter": {
                    "type": "int",
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "label": "Макс. итераций",
                },
                "dt": {
                    "type": "float",
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Шаг времени (dt)",
                },
                "eps": {
                    "type": "float",
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Параметр фазового поля",
                },
            },
        ),
        # ===== WATERSHED =====
        "watershed": MethodProfile(
            name="watershed",
            library="opencv",
            avg_time_ms=35.0,
            avg_iou=0.73,
            memory_mb=75,
            best_for_type=[
                ImageType.MEDICAL,
                ImageType.MICROSCOPY,
                ImageType.SATELLITE,
            ],
            robustness=0.65,
            parameter_sensitivity=0.8,
            description="Классический watershed по градиенту (OpenCV)",
            params={},
            schema={},
        ),
        "random_walker": MethodProfile(
            name="random_walker",
            library="opencv",
            avg_time_ms=95.0,
            avg_iou=0.85,
            memory_mb=130,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.9,
            parameter_sensitivity=0.4,
            description="Random walker с вероятностной диффузией (OpenCV)",
            params={
                "beta": 130,
                "tol": 1e-3,
                "max_iter": 300,
                "target_label": 2,
            },
            schema={
                "beta": {
                    "type": "int",
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "label": "Коэф. диффузии (beta)",
                },
                "tol": {
                    "type": "float",
                    "min": 0.0001,
                    "max": 0.01,
                    "step": 0.0001,
                    "label": "Точность",
                },
                "max_iter": {
                    "type": "int",
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "label": "Итерации",
                },
                "target_label": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "label": "Целевая метка объекта",
                },
            },
        ),
        # ===== SUPER-PIXELS =====
        "slic": MethodProfile(
            name="slic",
            library="opencv",
            avg_time_ms=65.0,
            avg_iou=0.79,
            memory_mb=95,
            best_for_type=[
                ImageType.NATURAL,
                ImageType.SATELLITE,
                ImageType.MEDICAL,
            ],
            robustness=0.8,
            parameter_sensitivity=0.4,
            description="SLIC super-pixels в Lab-пространстве (OpenCV)",
            params={
                "n_segments": 100,
                "compactness": 10.0,
                "max_iter": 10,
                "sigma": 0.0,
                "enforce_connectivity": True,
                "min_size_factor": 0.5,
                "max_size_factor": 3.0,
                "ruler": 10.0,
                "region_size": 20,
            },
            schema={
                "n_segments": {
                    "type": "int",
                    "min": 50,
                    "max": 1000,
                    "step": 50,
                    "label": "Кол-во сегментов",
                },
                "compactness": {
                    "type": "float",
                    "min": 0.1,
                    "max": 50.0,
                    "step": 0.1,
                    "label": "Компактность",
                },
                "max_iter": {
                    "type": "int",
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "label": "Итерации",
                },
                "sigma": {
                    "type": "float",
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Сглаживание Гаусса",
                },
                "min_size_factor": {
                    "type": "float",
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "label": "Мин. фактор размера",
                },
                "max_size_factor": {
                    "type": "float",
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Макс. фактор размера",
                },
                "ruler": {
                    "type": "float",
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1,
                    "label": "Масштабная линейка",
                },
                "region_size": {
                    "type": "int",
                    "min": 5,
                    "max": 100,
                    "step": 1,
                    "label": "Базовый размер региона",
                },
            },
        ),
        "felzenszwalb": MethodProfile(
            name="felzenszwalb",
            library="opencv",
            avg_time_ms=85.0,
            avg_iou=0.81,
            memory_mb=110,
            best_for_type=[ImageType.NATURAL, ImageType.INDUSTRIAL],
            robustness=0.78,
            parameter_sensitivity=0.5,
            description="Граф-базированная сегментация Фельценцвальба (OpenCV)",
            params={"scale": 100, "sigma": 0.5, "min_size": 50},
            schema={
                "scale": {
                    "type": "int",
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "label": "Масштаб сегментации",
                },
                "sigma": {
                    "type": "float",
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Сглаживание Гаусса",
                },
                "min_size": {
                    "type": "int",
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "label": "Мин. размер сегмента",
                },
            },
        ),
        # ===== INTERACTIVE =====
        "grabcut": MethodProfile(
            name="grabcut",
            library="opencv",
            avg_time_ms=150.0,
            avg_iou=0.91,
            memory_mb=180,
            best_for_type=[ImageType.NATURAL, ImageType.MEDICAL],
            robustness=0.88,
            parameter_sensitivity=0.5,
            description="GrabCut с итеративной оптимизацией GMM (OpenCV)",
            params={"num_iterations": 5},
            schema={
                "num_iterations": {
                    "type": "int",
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "label": "Итерации оптимизации",
                }
            },
        ),
    },
    "sklearn": {
        "global_thresholding": MethodProfile(
            name="global_thresholding",
            library="sklearn",
            avg_time_ms=2.0,
            avg_iou=0.62,
            memory_mb=15,
            best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
            robustness=0.4,
            parameter_sensitivity=0.9,
            description="Простое глобальное пороговое значение (Sklearn)",
            params={"threshold": 0.5},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог яркости (0-1)",
                }
            },
        ),
        "otsu_thresholding": MethodProfile(
            name="otsu_thresholding",
            library="sklearn",
            avg_time_ms=15.0,
            avg_iou=0.75,
            memory_mb=50,
            best_for_type=[ImageType.DOCUMENT, ImageType.NATURAL],
            robustness=0.8,
            parameter_sensitivity=0.2,
            description="Автоматический порог Оцу (максимизация межклассовой дисперсии) (sklearn)",
            params={},
            schema={},
        ),
        "adaptive_thresholding": MethodProfile(
            name="adaptive_thresholding",
            library="sklearn",
            avg_time_ms=45.0,
            avg_iou=0.82,
            memory_mb=80,
            best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
            robustness=0.9,
            parameter_sensitivity=0.4,
            description="Адаптивный порог с локальным усреднением (sklearn)",
            params={"block_size": 11, "C": 2},
            schema={
                "block_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер блока (нечетный)",
                },
                "C": {
                    "type": "int",
                    "min": -20,
                    "max": 20,
                    "step": 1,
                    "label": "Константа C (смещение)",
                },
            },
        ),
        "threshold_niblack": MethodProfile(
            name="threshold_niblack",
            library="sklearn",
            avg_time_ms=38.0,
            avg_iou=0.71,
            memory_mb=65,
            best_for_type=[ImageType.DOCUMENT, ImageType.MICROSCOPY],
            robustness=0.6,
            parameter_sensitivity=0.7,
            description="Метод Ниблэка для локального порогования (sklearn)",
            params={"window_size": 15, "k": -0.2},
            schema={
                "window_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер окна",
                },
                "k": {
                    "type": "float",
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Константа k",
                },
            },
        ),
        "threshold_sauvola": MethodProfile(
            name="threshold_sauvola",
            library="sklearn",
            avg_time_ms=42.0,
            avg_iou=0.88,
            memory_mb=70,
            best_for_type=[ImageType.DOCUMENT, ImageType.MICROSCOPY],
            robustness=0.92,
            parameter_sensitivity=0.3,
            description="Метод Саволы (улучшенный Ниблэк для текста) (sklearn)",
            params={"window_size": 15, "k": 0.5, "r": 128},
            schema={
                "window_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер окна",
                },
                "k": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Константа k",
                },
                "r": {
                    "type": "float",
                    "min": 50.0,
                    "max": 255.0,
                    "step": 1.0,
                    "label": "Динамический диапазон R",
                },
            },
        ),
        "threshold_bernsen": MethodProfile(
            name="threshold_bernsen",
            library="sklearn",
            avg_time_ms=35.0,
            avg_iou=0.73,
            memory_mb=60,
            best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
            robustness=0.7,
            parameter_sensitivity=0.5,
            description="Метод Бернсена на основе локального контраста (sklearn)",
            params={"window_size": 15, "contrast_threshold": 0.15},
            schema={
                "window_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер окна",
                },
                "contrast_threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог контраста",
                },
            },
        ),
        "threshold_phansalkar": MethodProfile(
            name="threshold_phansalkar",
            library="sklearn",
            avg_time_ms=48.0,
            avg_iou=0.85,
            memory_mb=75,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.88,
            parameter_sensitivity=0.4,
            description="Метод Фансалкара для низкоконтрастных изображений (sklearn)",
            params={"window_size": 15, "k": 0.25, "r": 128.0, "m": 0.5},
            schema={
                "window_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер окна",
                },
                "k": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Чувствительность k",
                },
                "r": {
                    "type": "float",
                    "min": 50.0,
                    "max": 255.0,
                    "step": 1.0,
                    "label": "Диапазон R",
                },
                "m": {
                    "type": "float",
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "label": "Смещение m",
                },
            },
        ),
        "threshold_kittler_illingworth": MethodProfile(
            name="threshold_kittler_illingworth",
            library="sklearn",
            avg_time_ms=25.0,
            avg_iou=0.76,
            memory_mb=55,
            best_for_type=[ImageType.DOCUMENT, ImageType.NATURAL],
            robustness=0.75,
            parameter_sensitivity=0.3,
            description="Минимизация ошибки классификации (Киттлер-Иллингуорт) (sklearn)",
            params={"num_bins": 256},
            schema={
                "num_bins": {
                    "type": "int",
                    "min": 32,
                    "max": 512,
                    "step": 16,
                    "label": "Кол-во бинов гистограммы",
                }
            },
        ),
        "threshold_entropy_kapur": MethodProfile(
            name="threshold_entropy_kapur",
            library="sklearn",
            avg_time_ms=30.0,
            avg_iou=0.74,
            memory_mb=60,
            best_for_type=[ImageType.NATURAL, ImageType.SATELLITE],
            robustness=0.7,
            parameter_sensitivity=0.4,
            description="Максимизация энтропии (Капур) (sklearn)",
            params={"num_bins": 256},
            schema={
                "num_bins": {
                    "type": "int",
                    "min": 32,
                    "max": 512,
                    "step": 16,
                    "label": "Кол-во бинов гистограммы",
                }
            },
        ),
        "threshold_triangle": MethodProfile(
            name="threshold_triangle",
            library="sklearn",
            avg_time_ms=20.0,
            avg_iou=0.69,
            memory_mb=45,
            best_for_type=[ImageType.DOCUMENT, ImageType.MEDICAL],
            robustness=0.65,
            parameter_sensitivity=0.3,
            description="Треугольный метод для унимодальных гистограмм (sklearn)",
            params={"num_bins": 256},
            schema={
                "num_bins": {
                    "type": "int",
                    "min": 32,
                    "max": 512,
                    "step": 16,
                    "label": "Кол-во бинов гистограммы",
                }
            },
        ),
        "threshold_multi_otsu": MethodProfile(
            name="threshold_multi_otsu",
            library="sklearn",
            avg_time_ms=35.0,
            avg_iou=0.78,
            memory_mb=70,
            best_for_type=[ImageType.MEDICAL, ImageType.SATELLITE],
            robustness=0.8,
            parameter_sensitivity=0.5,
            description="Многопороговый Оцу для многоклассовой сегментации (sklearn)",
            params={"n_thresholds": 2},
            schema={
                "n_thresholds": {
                    "type": "int",
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "label": "Кол-во порогов",
                }
            },
        ),
        "threshold_percentile": MethodProfile(
            name="threshold_percentile",
            library="sklearn",
            avg_time_ms=8.0,
            avg_iou=0.65,
            memory_mb=25,
            best_for_type=[ImageType.INDUSTRIAL, ImageType.DOCUMENT],
            robustness=0.5,
            parameter_sensitivity=0.8,
            description="Порог по перцентилю интенсивности (sklearn)",
            params={"percentile": 90},
            schema={
                "percentile": {
                    "type": "int",
                    "min": 1,
                    "max": 99,
                    "step": 1,
                    "label": "Процентиль (%)",
                }
            },
        ),
        "threshold_local_contrast": MethodProfile(
            name="threshold_local_contrast",
            library="sklearn",
            avg_time_ms=40.0,
            avg_iou=0.77,
            memory_mb=68,
            best_for_type=[ImageType.MICROSCOPY, ImageType.MEDICAL],
            robustness=0.82,
            parameter_sensitivity=0.5,
            description="Порог на основе локального контраста (sklearn)",
            params={"window_size": 15, "contrast_factor": 0.1},
            schema={
                "window_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер окна",
                },
                "contrast_factor": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Фактор контраста",
                },
            },
        ),
        # ===== EDGE DETECTION =====
        "canny_edge": MethodProfile(
            name="canny_edge",
            library="sklearn",
            avg_time_ms=25.0,
            avg_iou=0.68,
            memory_mb=60,
            best_for_type=[ImageType.NATURAL, ImageType.INDUSTRIAL],
            robustness=0.7,
            parameter_sensitivity=0.6,
            description="Детектор границ Кэнни (оптимальный по Кэнни) (sklearn)",
            params={"low": 0.1, "high": 0.3, "sigma": 1.0},
            schema={
                "low": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Нижний порог",
                },
                "high": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Верхний порог",
                },
                "sigma": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Сигма (размытие)",
                },
            },
        ),
        "sobel_edge": MethodProfile(
            name="sobel_edge",
            library="sklearn",
            avg_time_ms=12.0,
            avg_iou=0.58,
            memory_mb=35,
            best_for_type=[ImageType.NATURAL, ImageType.DOCUMENT],
            robustness=0.5,
            parameter_sensitivity=0.7,
            description="Градиенты Собеля с порогом (sklearn)",
            params={"threshold": 0.1},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог градиента",
                }
            },
        ),
        "prewitt_edge": MethodProfile(
            name="prewitt_edge",
            library="sklearn",
            avg_time_ms=11.0,
            avg_iou=0.56,
            memory_mb=33,
            best_for_type=[ImageType.NATURAL, ImageType.DOCUMENT],
            robustness=0.48,
            parameter_sensitivity=0.72,
            description="Градиенты Превитта с порогом (sklearn)",
            params={"threshold": 0.1},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог градиента",
                }
            },
        ),
        "scharr_edge": MethodProfile(
            name="scharr_edge",
            library="sklearn",
            avg_time_ms=14.0,
            avg_iou=0.61,
            memory_mb=38,
            best_for_type=[ImageType.NATURAL, ImageType.INDUSTRIAL],
            robustness=0.55,
            parameter_sensitivity=0.65,
            description="Градиенты Шарра (sklearn)",
            params={"threshold": 0.1},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог градиента",
                }
            },
        ),
        "roberts_cross_edge": MethodProfile(
            name="roberts_cross_edge",
            library="sklearn",
            avg_time_ms=8.0,
            avg_iou=0.52,
            memory_mb=28,
            best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
            robustness=0.4,
            parameter_sensitivity=0.8,
            description="Оператор Робертса для диагональных границ (sklearn)",
            params={"threshold": 0.1},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог градиента",
                }
            },
        ),
        "log_edge": MethodProfile(
            name="log_edge",
            library="sklearn",
            avg_time_ms=22.0,
            avg_iou=0.64,
            memory_mb=55,
            best_for_type=[ImageType.NATURAL, ImageType.MEDICAL],
            robustness=0.6,
            parameter_sensitivity=0.5,
            description="Laplacian of Gaussian детектор границ (sklearn)",
            params={"sigma": 1.0, "threshold": 0.01},
            schema={
                "sigma": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Сигма размытия",
                },
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог детекции",
                },
            },
        ),
        "dog_edge": MethodProfile(
            name="dog_edge",
            library="sklearn",
            avg_time_ms=28.0,
            avg_iou=0.66,
            memory_mb=62,
            best_for_type=[ImageType.NATURAL, ImageType.SATELLITE],
            robustness=0.68,
            parameter_sensitivity=0.55,
            description="Difference of Gaussians для мультимасштабных границ (sklearn)",
            params={"sigma1": 1.0, "sigma2": 2.0, "threshold": 0.01},
            schema={
                "sigma1": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Сигма 1",
                },
                "sigma2": {
                    "type": "float",
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1,
                    "label": "Сигма 2",
                },
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог детекции",
                },
            },
        ),
        "marr_hildreth_edge": MethodProfile(
            name="marr_hildreth_edge",
            library="sklearn",
            avg_time_ms=26.0,
            avg_iou=0.63,
            memory_mb=58,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.62,
            parameter_sensitivity=0.58,
            description="Метод Марра-Хилдрета (нулевые пересечения LoG) (sklearn)",
            params={"sigma": 1.5, "threshold": 0.01},
            schema={
                "sigma": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Сигма размытия",
                },
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог нулевых пересечений",
                },
            },
        ),
        "gradient_magnitude_direction": MethodProfile(
            name="gradient_magnitude_direction",
            library="sklearn",
            avg_time_ms=18.0,
            avg_iou=0.59,
            memory_mb=45,
            best_for_type=[ImageType.INDUSTRIAL, ImageType.NATURAL],
            robustness=0.52,
            parameter_sensitivity=0.68,
            description="Сегментация по величине и направлению градиента (sklearn)",
            params={"threshold": 0.1},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог магнитуды",
                }
            },
        ),
        "phase_congruency_edge": MethodProfile(
            name="phase_congruency_edge",
            library="sklearn",
            avg_time_ms=85.0,
            avg_iou=0.79,
            memory_mb=120,
            best_for_type=[
                ImageType.MEDICAL,
                ImageType.SATELLITE,
                ImageType.MICROSCOPY,
            ],
            robustness=0.95,
            parameter_sensitivity=0.3,
            description="Фазовая конгруэнтность (инвариантна к освещению) (sklearn)",
            params={
                "nscales": 4,
                "norientations": 4,
                "min_wavelength": 3,
                "mult": 2.0,
                "sigma_onf": 0.55,
                "k_noise": 2.0,
                "threshold": 0.5,
            },
            schema={
                "nscales": {
                    "type": "int",
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "label": "Кол-во масштабов",
                },
                "norientations": {
                    "type": "int",
                    "min": 1,
                    "max": 12,
                    "step": 1,
                    "label": "Кол-во ориентаций",
                },
                "min_wavelength": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "label": "Мин. длина волны",
                },
                "mult": {
                    "type": "float",
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Множитель масштаба",
                },
                "sigma_onf": {
                    "type": "float",
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "label": "Сигма частотной области",
                },
                "k_noise": {
                    "type": "float",
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Коэф. шумоподавления",
                },
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог энергии",
                },
            },
        ),
        # ===== REGION-BASED =====
        "region_growing": MethodProfile(
            name="region_growing",
            library="sklearn",
            avg_time_ms=55.0,
            avg_iou=0.81,
            memory_mb=90,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.75,
            parameter_sensitivity=0.6,
            description="Рост региона от семян по схожести (sklearn)",
            params={"tolerance": 0.1},
            schema={
                "tolerance": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Допуск схожести",
                }
            },
        ),
        "split_and_merge": MethodProfile(
            name="split_and_merge",
            library="sklearn",
            avg_time_ms=70.0,
            avg_iou=0.76,
            memory_mb=100,
            best_for_type=[ImageType.SATELLITE, ImageType.INDUSTRIAL],
            robustness=0.7,
            parameter_sensitivity=0.5,
            description="Разделение и слияние регионов (sklearn)",
            params={"min_size": 50, "threshold": 20},
            schema={
                "min_size": {
                    "type": "int",
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "label": "Мин. размер региона",
                },
                "threshold": {
                    "type": "int",
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "label": "Порог слияния",
                },
            },
        ),
        "floodfill": MethodProfile(
            name="floodfill",
            library="sklearn",
            avg_time_ms=15.0,
            avg_iou=0.72,
            memory_mb=40,
            best_for_type=[ImageType.DOCUMENT, ImageType.MEDICAL],
            robustness=0.6,
            parameter_sensitivity=0.7,
            description="Заливка области от точки (sklearn)",
            params={"tolerance": 0.15},
            schema={
                "tolerance": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Допуск заливки",
                }
            },
        ),
        # ===== CLUSTERING =====
        "kmeans_segmentation": MethodProfile(
            name="kmeans_segmentation",
            library="sklearn",
            avg_time_ms=120.0,
            avg_iou=0.77,
            memory_mb=150,
            best_for_type=[ImageType.NATURAL, ImageType.SATELLITE],
            robustness=0.65,
            parameter_sensitivity=0.7,
            description="K-means кластеризация в пространстве признаков (sklearn)",
            params={"k": 3},
            schema={
                "k": {
                    "type": "int",
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "label": "Кол-во кластеров",
                }
            },
        ),
        "dbscan_segmentation": MethodProfile(
            name="dbscan_segmentation",
            library="sklearn",
            avg_time_ms=180.0,
            avg_iou=0.74,
            memory_mb=200,
            best_for_type=[ImageType.MICROSCOPY, ImageType.INDUSTRIAL],
            robustness=0.8,
            parameter_sensitivity=0.6,
            description="DBSCAN для сегментации произвольной формы (sklearn)",
            params={"eps": 0.1, "min_samples": 10},
            schema={
                "eps": {
                    "type": "float",
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Радиус окрестности (eps)",
                },
                "min_samples": {
                    "type": "int",
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "label": "Мин. точек в кластере",
                },
            },
        ),
        "meanshift": MethodProfile(
            name="meanshift",
            library="sklearn",
            avg_time_ms=250.0,
            avg_iou=0.83,
            memory_mb=280,
            best_for_type=[ImageType.NATURAL, ImageType.MEDICAL],
            robustness=0.85,
            parameter_sensitivity=0.4,
            description="MeanShift с пространственно-цветовым ядром (sklearn)",
            params={
                "bandwidth": 0.5,
                "spatial_radius": 35,
                "color_radius": 60,
                "max_level": 1,
            },
            schema={
                "bandwidth": {
                    "type": "float",
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Полоса пропускания",
                },
                "spatial_radius": {
                    "type": "int",
                    "min": 5,
                    "max": 100,
                    "step": 1,
                    "label": "Пространственный радиус",
                },
                "color_radius": {
                    "type": "int",
                    "min": 10,
                    "max": 200,
                    "step": 1,
                    "label": "Цветовой радиус",
                },
                "max_level": {
                    "type": "int",
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "label": "Макс. уровень пирамиды",
                },
            },
        ),
        # ===== ACTIVE CONTOURS =====
        "active_contour": MethodProfile(
            name="active_contour",
            library="sklearn",
            avg_time_ms=450.0,
            avg_iou=0.84,
            memory_mb=180,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.78,
            parameter_sensitivity=0.75,
            description="Змеи (snakes) с энергией границ и линий (sklearn)",
            params={
                "alpha": 0.015,
                "beta": 10,
                "gamma": 0.001,
                "max_iterations": 2000,
                "w_edge": 1,
                "w_line": 0,
            },
            schema={
                "alpha": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "label": "Плавность контура",
                },
                "beta": {
                    "type": "float",
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1,
                    "label": "Жесткость контура",
                },
                "gamma": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "label": "Вязкость (шаг)",
                },
                "max_iterations": {
                    "type": "int",
                    "min": 100,
                    "max": 5000,
                    "step": 100,
                    "label": "Итерации",
                },
                "w_edge": {
                    "type": "float",
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Вес границ",
                },
                "w_line": {
                    "type": "float",
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Вес линий",
                },
            },
        ),
        "gvf_contour": MethodProfile(
            name="gvf_contour",
            library="sklearn",
            avg_time_ms=380.0,
            avg_iou=0.86,
            memory_mb=160,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.88,
            parameter_sensitivity=0.5,
            description="Контуры с градиентным векторным потоком (GVF) (sklearn)",
            params={"mu": 0.1, "iterations": 50},
            schema={
                "mu": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Коэф. диффузии (mu)",
                },
                "iterations": {
                    "type": "int",
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "label": "Итерации GVF",
                },
            },
        ),
        "morphological_snakes": MethodProfile(
            name="morphological_snakes",
            library="sklearn",
            avg_time_ms=320.0,
            avg_iou=0.87,
            memory_mb=140,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.92,
            parameter_sensitivity=0.35,
            description="Морфологические змеи (устойчивы к шуму) (sklearn)",
            params={"iterations": 100, "smoothing": 1, "threshold": 0.5},
            schema={
                "iterations": {
                    "type": "int",
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "label": "Итерации",
                },
                "smoothing": {
                    "type": "int",
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "label": "Степень сглаживания",
                },
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог инициализации",
                },
            },
        ),
        "chan_vese": MethodProfile(
            name="chan_vese",
            library="sklearn",
            avg_time_ms=400.0,
            avg_iou=0.89,
            memory_mb=170,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.94,
            parameter_sensitivity=0.3,
            description="Модель Чан-Везе (регион-базированные активные контуры) (sklearn)",
            params={
                "mu": 0.25,
                "lambda1": 1.0,
                "lambda2": 1.0,
                "tol": 1e-3,
                "max_iter": 100,
                "dt": 0.5,
                "eps": 1.0,
                "init_level_set": "checkerboard",
            },
            schema={
                "mu": {
                    "type": "float",
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "label": "Длина контура (mu)",
                },
                "lambda1": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Внешняя область (lambda1)",
                },
                "lambda2": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Внутренняя область (lambda2)",
                },
                "tol": {
                    "type": "float",
                    "min": 0.0001,
                    "max": 0.01,
                    "step": 0.0001,
                    "label": "Точность сходимости",
                },
                "max_iter": {
                    "type": "int",
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "label": "Макс. итераций",
                },
                "dt": {
                    "type": "float",
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Шаг времени (dt)",
                },
                "eps": {
                    "type": "float",
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Параметр фазового поля",
                },
            },
        ),
        # ===== WATERSHED =====
        "watershed": MethodProfile(
            name="watershed",
            library="sklearn",
            avg_time_ms=35.0,
            avg_iou=0.73,
            memory_mb=75,
            best_for_type=[
                ImageType.MEDICAL,
                ImageType.MICROSCOPY,
                ImageType.SATELLITE,
            ],
            robustness=0.65,
            parameter_sensitivity=0.8,
            description="Классический watershed по градиенту (sklearn)",
            params={},
            schema={},
        ),
        "random_walker": MethodProfile(
            name="random_walker",
            library="sklearn",
            avg_time_ms=95.0,
            avg_iou=0.85,
            memory_mb=130,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.9,
            parameter_sensitivity=0.4,
            description="Random walker с вероятностной диффузией (sklearn)",
            params={
                "beta": 130,
                "tol": 1e-3,
                "max_iter": 300,
                "target_label": 2,
            },
            schema={
                "beta": {
                    "type": "int",
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "label": "Коэф. диффузии (beta)",
                },
                "tol": {
                    "type": "float",
                    "min": 0.0001,
                    "max": 0.01,
                    "step": 0.0001,
                    "label": "Точность",
                },
                "max_iter": {
                    "type": "int",
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "label": "Итерации",
                },
                "target_label": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "label": "Целевая метка объекта",
                },
            },
        ),
        # ===== SUPER-PIXELS =====
        "slic": MethodProfile(
            name="slic",
            library="sklearn",
            avg_time_ms=65.0,
            avg_iou=0.79,
            memory_mb=95,
            best_for_type=[
                ImageType.NATURAL,
                ImageType.SATELLITE,
                ImageType.MEDICAL,
            ],
            robustness=0.8,
            parameter_sensitivity=0.4,
            description="SLIC super-pixels в Lab-пространстве (sklearn)",
            params={
                "n_segments": 100,
                "compactness": 10.0,
                "max_iter": 10,
                "sigma": 0.0,
                "enforce_connectivity": True,
                "min_size_factor": 0.5,
                "max_size_factor": 3.0,
                "ruler": 10.0,
                "region_size": 20,
            },
            schema={
                "n_segments": {
                    "type": "int",
                    "min": 50,
                    "max": 1000,
                    "step": 50,
                    "label": "Кол-во сегментов",
                },
                "compactness": {
                    "type": "float",
                    "min": 0.1,
                    "max": 50.0,
                    "step": 0.1,
                    "label": "Компактность",
                },
                "max_iter": {
                    "type": "int",
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "label": "Итерации",
                },
                "sigma": {
                    "type": "float",
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Сглаживание Гаусса",
                },
                "min_size_factor": {
                    "type": "float",
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "label": "Мин. фактор размера",
                },
                "max_size_factor": {
                    "type": "float",
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Макс. фактор размера",
                },
                "ruler": {
                    "type": "float",
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1,
                    "label": "Масштабная линейка",
                },
                "region_size": {
                    "type": "int",
                    "min": 5,
                    "max": 100,
                    "step": 1,
                    "label": "Базовый размер региона",
                },
            },
        ),
        "felzenszwalb": MethodProfile(
            name="felzenszwalb",
            library="sklearn",
            avg_time_ms=85.0,
            avg_iou=0.81,
            memory_mb=110,
            best_for_type=[ImageType.NATURAL, ImageType.INDUSTRIAL],
            robustness=0.78,
            parameter_sensitivity=0.5,
            description="Граф-базированная сегментация Фельценцвальба (sklearn)",
            params={"scale": 100, "sigma": 0.5, "min_size": 50},
            schema={
                "scale": {
                    "type": "int",
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "label": "Масштаб сегментации",
                },
                "sigma": {
                    "type": "float",
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Сглаживание Гаусса",
                },
                "min_size": {
                    "type": "int",
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "label": "Мин. размер сегмента",
                },
            },
        ),
        # ===== INTERACTIVE =====
        "grabcut": MethodProfile(
            name="grabcut",
            library="sklearn",
            avg_time_ms=150.0,
            avg_iou=0.91,
            memory_mb=180,
            best_for_type=[ImageType.NATURAL, ImageType.MEDICAL],
            robustness=0.88,
            parameter_sensitivity=0.5,
            description="GrabCut с итеративной оптимизацией GMM (sklearn)",
            params={"num_iterations": 5},
            schema={
                "num_iterations": {
                    "type": "int",
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "label": "Итерации оптимизации",
                }
            },
        ),
    },
    "torch": {
        "global_thresholding": MethodProfile(
            name="global_thresholding",
            library="torch",
            avg_time_ms=2.0,
            avg_iou=0.62,
            memory_mb=15,
            best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
            robustness=0.4,
            parameter_sensitivity=0.9,
            description="Простое глобальное пороговое значение (torch)",
            params={"threshold": 0.5},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог яркости (0-1)",
                }
            },
        ),
        "otsu_thresholding": MethodProfile(
            name="otsu_thresholding",
            library="torch",
            avg_time_ms=15.0,
            avg_iou=0.75,
            memory_mb=50,
            best_for_type=[ImageType.DOCUMENT, ImageType.NATURAL],
            robustness=0.8,
            parameter_sensitivity=0.2,
            description="Автоматический порог Оцу (максимизация межклассовой дисперсии) (torch)",
            params={},
            schema={},
        ),
        "adaptive_thresholding": MethodProfile(
            name="adaptive_thresholding",
            library="torch",
            avg_time_ms=45.0,
            avg_iou=0.82,
            memory_mb=80,
            best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
            robustness=0.9,
            parameter_sensitivity=0.4,
            description="Адаптивный порог с локальным усреднением (torch)",
            schema={
                "block_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер блока (нечетный)",
                },
                "C": {
                    "type": "int",
                    "min": -20,
                    "max": 20,
                    "step": 1,
                    "label": "Константа C (смещение)",
                },
            },
        ),
        "threshold_niblack": MethodProfile(
            name="threshold_niblack",
            library="torch",
            avg_time_ms=38.0,
            avg_iou=0.71,
            memory_mb=65,
            best_for_type=[ImageType.DOCUMENT, ImageType.MICROSCOPY],
            robustness=0.6,
            parameter_sensitivity=0.7,
            description="Метод Ниблэка для локального порогования (torch)",
            params={"window_size": 15, "k": -0.2},
            schema={
                "window_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер окна",
                },
                "k": {
                    "type": "float",
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Константа k",
                },
            },
        ),
        "threshold_sauvola": MethodProfile(
            name="threshold_sauvola",
            library="torch",
            avg_time_ms=42.0,
            avg_iou=0.88,
            memory_mb=70,
            best_for_type=[ImageType.DOCUMENT, ImageType.MICROSCOPY],
            robustness=0.92,
            parameter_sensitivity=0.3,
            description="Метод Саволы (улучшенный Ниблэк для текста) (torch)",
            params={"window_size": 15, "k": 0.5, "r": 128},
            schema={
                "window_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер окна",
                },
                "k": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Константа k",
                },
                "r": {
                    "type": "float",
                    "min": 50.0,
                    "max": 255.0,
                    "step": 1.0,
                    "label": "Динамический диапазон R",
                },
            },
        ),
        "threshold_bernsen": MethodProfile(
            name="threshold_bernsen",
            library="torch",
            avg_time_ms=35.0,
            avg_iou=0.73,
            memory_mb=60,
            best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
            robustness=0.7,
            parameter_sensitivity=0.5,
            description="Метод Бернсена на основе локального контраста (torch)",
            params={"window_size": 15, "contrast_threshold": 0.15},
            schema={
                "window_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер окна",
                },
                "contrast_threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог контраста",
                },
            },
        ),
        "threshold_phansalkar": MethodProfile(
            name="threshold_phansalkar",
            library="torch",
            avg_time_ms=48.0,
            avg_iou=0.85,
            memory_mb=75,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.88,
            parameter_sensitivity=0.4,
            description="Метод Фансалкара для низкоконтрастных изображений (torch)",
            params={"window_size": 15, "k": 0.25, "r": 128.0, "m": 0.5},
            schema={
                "window_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер окна",
                },
                "k": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Чувствительность k",
                },
                "r": {
                    "type": "float",
                    "min": 50.0,
                    "max": 255.0,
                    "step": 1.0,
                    "label": "Диапазон R",
                },
                "m": {
                    "type": "float",
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "label": "Смещение m",
                },
            },
        ),
        "threshold_kittler_illingworth": MethodProfile(
            name="threshold_kittler_illingworth",
            library="torch",
            avg_time_ms=25.0,
            avg_iou=0.76,
            memory_mb=55,
            best_for_type=[ImageType.DOCUMENT, ImageType.NATURAL],
            robustness=0.75,
            parameter_sensitivity=0.3,
            description="Минимизация ошибки классификации (Киттлер-Иллингуорт) (torch)",
            params={"num_bins": 256},
            schema={
                "num_bins": {
                    "type": "int",
                    "min": 32,
                    "max": 512,
                    "step": 16,
                    "label": "Кол-во бинов гистограммы",
                }
            },
        ),
        "threshold_entropy_kapur": MethodProfile(
            name="threshold_entropy_kapur",
            library="torch",
            avg_time_ms=30.0,
            avg_iou=0.74,
            memory_mb=60,
            best_for_type=[ImageType.NATURAL, ImageType.SATELLITE],
            robustness=0.7,
            parameter_sensitivity=0.4,
            description="Максимизация энтропии (Капур) (torch)",
            params={"num_bins": 256},
            schema={
                "num_bins": {
                    "type": "int",
                    "min": 32,
                    "max": 512,
                    "step": 16,
                    "label": "Кол-во бинов гистограммы",
                }
            },
        ),
        "threshold_triangle": MethodProfile(
            name="threshold_triangle",
            library="torch",
            avg_time_ms=20.0,
            avg_iou=0.69,
            memory_mb=45,
            best_for_type=[ImageType.DOCUMENT, ImageType.MEDICAL],
            robustness=0.65,
            parameter_sensitivity=0.3,
            description="Треугольный метод для унимодальных гистограмм (torch)",
            params={"num_bins": 256},
            schema={
                "num_bins": {
                    "type": "int",
                    "min": 32,
                    "max": 512,
                    "step": 16,
                    "label": "Кол-во бинов гистограммы",
                }
            },
        ),
        "threshold_multi_otsu": MethodProfile(
            name="threshold_multi_otsu",
            library="torch",
            avg_time_ms=35.0,
            avg_iou=0.78,
            memory_mb=70,
            best_for_type=[ImageType.MEDICAL, ImageType.SATELLITE],
            robustness=0.8,
            parameter_sensitivity=0.5,
            description="Многопороговый Оцу для многоклассовой сегментации (torch)",
            params={"n_thresholds": 2},
            schema={
                "n_thresholds": {
                    "type": "int",
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "label": "Кол-во порогов",
                }
            },
        ),
        "threshold_percentile": MethodProfile(
            name="threshold_percentile",
            library="torch",
            avg_time_ms=8.0,
            avg_iou=0.65,
            memory_mb=25,
            best_for_type=[ImageType.INDUSTRIAL, ImageType.DOCUMENT],
            robustness=0.5,
            parameter_sensitivity=0.8,
            description="Порог по перцентилю интенсивности (torch)",
            params={"percentile": 90},
            schema={
                "percentile": {
                    "type": "int",
                    "min": 1,
                    "max": 99,
                    "step": 1,
                    "label": "Процентиль (%)",
                }
            },
        ),
        "threshold_local_contrast": MethodProfile(
            name="threshold_local_contrast",
            library="torch",
            avg_time_ms=40.0,
            avg_iou=0.77,
            memory_mb=68,
            best_for_type=[ImageType.MICROSCOPY, ImageType.MEDICAL],
            robustness=0.82,
            parameter_sensitivity=0.5,
            description="Порог на основе локального контраста (torch)",
            params={"window_size": 15, "contrast_factor": 0.1},
            schema={
                "window_size": {
                    "type": "int",
                    "min": 3,
                    "max": 99,
                    "step": 2,
                    "label": "Размер окна",
                },
                "contrast_factor": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Фактор контраста",
                },
            },
        ),
        # ===== EDGE DETECTION =====
        "canny_edge": MethodProfile(
            name="canny_edge",
            library="torch",
            avg_time_ms=25.0,
            avg_iou=0.68,
            memory_mb=60,
            best_for_type=[ImageType.NATURAL, ImageType.INDUSTRIAL],
            robustness=0.7,
            parameter_sensitivity=0.6,
            description="Детектор границ Кэнни (оптимальный по Кэнни) (torch)",
            params={"low": 0.1, "high": 0.3, "sigma": 1.0},  # ← дефолтные значения
            schema={  # ← ДОБАВЬТЕ СХЕМУ
                "low": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Нижний порог",
                },
                "high": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Верхний порог",
                },
                "sigma": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Сигма (размытие)",
                },
            },
        ),
        "sobel_edge": MethodProfile(
            name="sobel_edge",
            library="torch",
            avg_time_ms=12.0,
            avg_iou=0.58,
            memory_mb=35,
            best_for_type=[ImageType.NATURAL, ImageType.DOCUMENT],
            robustness=0.5,
            parameter_sensitivity=0.7,
            description="Градиенты Собеля с порогом (torch)",
            params={"threshold": 0.1},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог градиента",
                }
            },
        ),
        "prewitt_edge": MethodProfile(
            name="prewitt_edge",
            library="torch",
            avg_time_ms=11.0,
            avg_iou=0.56,
            memory_mb=33,
            best_for_type=[ImageType.NATURAL, ImageType.DOCUMENT],
            robustness=0.48,
            parameter_sensitivity=0.72,
            description="Градиенты Превитта с порогом (torch)",
            params={"threshold": 0.1},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог градиента",
                }
            },
        ),
        "scharr_edge": MethodProfile(
            name="scharr_edge",
            library="torch",
            avg_time_ms=14.0,
            avg_iou=0.61,
            memory_mb=38,
            best_for_type=[ImageType.NATURAL, ImageType.INDUSTRIAL],
            robustness=0.55,
            parameter_sensitivity=0.65,
            description="Градиенты Шарра (torch)",
            params={"threshold": 0.1},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог градиента",
                }
            },
        ),
        "roberts_cross_edge": MethodProfile(
            name="roberts_cross_edge",
            library="torch",
            avg_time_ms=8.0,
            avg_iou=0.52,
            memory_mb=28,
            best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
            robustness=0.4,
            parameter_sensitivity=0.8,
            description="Оператор Робертса для диагональных границ (torch)",
            params={"threshold": 0.1},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог градиента",
                }
            },
        ),
        "log_edge": MethodProfile(
            name="log_edge",
            library="torch",
            avg_time_ms=22.0,
            avg_iou=0.64,
            memory_mb=55,
            best_for_type=[ImageType.NATURAL, ImageType.MEDICAL],
            robustness=0.6,
            parameter_sensitivity=0.5,
            description="Laplacian of Gaussian детектор границ (torch)",
            schema={
                "sigma": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Сигма размытия",
                },
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог детекции",
                },
            },
        ),
        "dog_edge": MethodProfile(
            name="dog_edge",
            library="torch",
            avg_time_ms=28.0,
            avg_iou=0.66,
            memory_mb=62,
            best_for_type=[ImageType.NATURAL, ImageType.SATELLITE],
            robustness=0.68,
            parameter_sensitivity=0.55,
            description="Difference of Gaussians для мультимасштабных границ (torch)",
            params={"sigma1": 1.0, "sigma2": 2.0, "threshold": 0.01},
            schema={
                "sigma1": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Сигма 1",
                },
                "sigma2": {
                    "type": "float",
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1,
                    "label": "Сигма 2",
                },
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог детекции",
                },
            },
        ),
        "marr_hildreth_edge": MethodProfile(
            name="marr_hildreth_edge",
            library="torch",
            avg_time_ms=26.0,
            avg_iou=0.63,
            memory_mb=58,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.62,
            parameter_sensitivity=0.58,
            description="Метод Марра-Хилдрета (нулевые пересечения LoG) (torch)",
            params={"sigma": 1.5, "threshold": 0.01},
            schema={
                "sigma": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Сигма размытия",
                },
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог нулевых пересечений",
                },
            },
        ),
        "gradient_magnitude_direction": MethodProfile(
            name="gradient_magnitude_direction",
            library="torch",
            avg_time_ms=18.0,
            avg_iou=0.59,
            memory_mb=45,
            best_for_type=[ImageType.INDUSTRIAL, ImageType.NATURAL],
            robustness=0.52,
            parameter_sensitivity=0.68,
            description="Сегментация по величине и направлению градиента (torch)",
            params={"threshold": 0.1},
            schema={
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог магнитуды",
                }
            },
        ),
        "phase_congruency_edge": MethodProfile(
            name="phase_congruency_edge",
            library="torch",
            avg_time_ms=85.0,
            avg_iou=0.79,
            memory_mb=120,
            best_for_type=[
                ImageType.MEDICAL,
                ImageType.SATELLITE,
                ImageType.MICROSCOPY,
            ],
            robustness=0.95,
            parameter_sensitivity=0.3,
            description="Фазовая конгруэнтность (инвариантна к освещению) (torch)",
            params={
                "nscales": 4,
                "norientations": 4,
                "min_wavelength": 3,
                "mult": 2.0,
                "sigma_onf": 0.55,
                "k_noise": 2.0,
                "threshold": 0.5,
            },
            schema={
                "nscales": {
                    "type": "int",
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "label": "Кол-во масштабов",
                },
                "norientations": {
                    "type": "int",
                    "min": 1,
                    "max": 12,
                    "step": 1,
                    "label": "Кол-во ориентаций",
                },
                "min_wavelength": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "label": "Мин. длина волны",
                },
                "mult": {
                    "type": "float",
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Множитель масштаба",
                },
                "sigma_onf": {
                    "type": "float",
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "label": "Сигма частотной области",
                },
                "k_noise": {
                    "type": "float",
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Коэф. шумоподавления",
                },
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог энергии",
                },
            },
        ),
        # ===== REGION-BASED =====
        "region_growing": MethodProfile(
            name="region_growing",
            library="torch",
            avg_time_ms=55.0,
            avg_iou=0.81,
            memory_mb=90,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.75,
            parameter_sensitivity=0.6,
            description="Рост региона от семян по схожести (torch)",
            params={"tolerance": 0.1},
            schema={
                "tolerance": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Допуск схожести",
                }
            },
        ),
        "split_and_merge": MethodProfile(
            name="split_and_merge",
            library="torch",
            avg_time_ms=70.0,
            avg_iou=0.76,
            memory_mb=100,
            best_for_type=[ImageType.SATELLITE, ImageType.INDUSTRIAL],
            robustness=0.7,
            parameter_sensitivity=0.5,
            description="Разделение и слияние регионов (torch)",
            params={"min_size": 50, "threshold": 20},
            schema={
                "min_size": {
                    "type": "int",
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "label": "Мин. размер региона",
                },
                "threshold": {
                    "type": "int",
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "label": "Порог слияния",
                },
            },
        ),
        "floodfill": MethodProfile(
            name="floodfill",
            library="torch",
            avg_time_ms=15.0,
            avg_iou=0.72,
            memory_mb=40,
            best_for_type=[ImageType.DOCUMENT, ImageType.MEDICAL],
            robustness=0.6,
            parameter_sensitivity=0.7,
            description="Заливка области от точки (torch)",
            params={"tolerance": 0.15},
            schema={
                "tolerance": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Допуск заливки",
                }
            },
        ),
        # ===== CLUSTERING =====
        "kmeans_segmentation": MethodProfile(
            name="kmeans_segmentation",
            library="torch",
            avg_time_ms=120.0,
            avg_iou=0.77,
            memory_mb=150,
            best_for_type=[ImageType.NATURAL, ImageType.SATELLITE],
            robustness=0.65,
            parameter_sensitivity=0.7,
            description="K-means кластеризация в пространстве признаков (torch)",
            params={"k": 3},
            schema={
                "k": {
                    "type": "int",
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "label": "Кол-во кластеров",
                }
            },
        ),
        "dbscan_segmentation": MethodProfile(
            name="dbscan_segmentation",
            library="torch",
            avg_time_ms=180.0,
            avg_iou=0.74,
            memory_mb=200,
            best_for_type=[ImageType.MICROSCOPY, ImageType.INDUSTRIAL],
            robustness=0.8,
            parameter_sensitivity=0.6,
            description="DBSCAN для сегментации произвольной формы (torch)",
            params={"eps": 0.1, "min_samples": 10},
            schema={
                "eps": {
                    "type": "float",
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Радиус окрестности (eps)",
                },
                "min_samples": {
                    "type": "int",
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "label": "Мин. точек в кластере",
                },
            },
        ),
        "meanshift": MethodProfile(
            name="meanshift",
            library="torch",
            avg_time_ms=250.0,
            avg_iou=0.83,
            memory_mb=280,
            best_for_type=[ImageType.NATURAL, ImageType.MEDICAL],
            robustness=0.85,
            parameter_sensitivity=0.4,
            description="MeanShift с пространственно-цветовым ядром (torch)",
            params={
                "bandwidth": 0.5,
                "spatial_radius": 35,
                "color_radius": 60,
                "max_level": 1,
            },
            schema={
                "bandwidth": {
                    "type": "float",
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Полоса пропускания",
                },
                "spatial_radius": {
                    "type": "int",
                    "min": 5,
                    "max": 100,
                    "step": 1,
                    "label": "Пространственный радиус",
                },
                "color_radius": {
                    "type": "int",
                    "min": 10,
                    "max": 200,
                    "step": 1,
                    "label": "Цветовой радиус",
                },
                "max_level": {
                    "type": "int",
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "label": "Макс. уровень пирамиды",
                },
            },
        ),
        # ===== ACTIVE CONTOURS =====
        "active_contour": MethodProfile(
            name="active_contour",
            library="torch",
            avg_time_ms=450.0,
            avg_iou=0.84,
            memory_mb=180,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.78,
            parameter_sensitivity=0.75,
            description="Змеи (snakes) с энергией границ и линий (torch)",
            params={
                "alpha": 0.015,
                "beta": 10,
                "gamma": 0.001,
                "max_iterations": 2000,
                "w_edge": 1,
                "w_line": 0,
            },
            schema={
                "alpha": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "label": "Плавность контура",
                },
                "beta": {
                    "type": "float",
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1,
                    "label": "Жесткость контура",
                },
                "gamma": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "label": "Вязкость (шаг)",
                },
                "max_iterations": {
                    "type": "int",
                    "min": 100,
                    "max": 5000,
                    "step": 100,
                    "label": "Итерации",
                },
                "w_edge": {
                    "type": "float",
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Вес границ",
                },
                "w_line": {
                    "type": "float",
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Вес линий",
                },
            },
        ),
        "gvf_contour": MethodProfile(
            name="gvf_contour",
            library="torch",
            avg_time_ms=380.0,
            avg_iou=0.86,
            memory_mb=160,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.88,
            parameter_sensitivity=0.5,
            description="Контуры с градиентным векторным потоком (GVF) (torch)",
            params={"mu": 0.1, "iterations": 50},
            schema={
                "mu": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Коэф. диффузии (mu)",
                },
                "iterations": {
                    "type": "int",
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "label": "Итерации GVF",
                },
            },
        ),
        "morphological_snakes": MethodProfile(
            name="morphological_snakes",
            library="torch",
            avg_time_ms=320.0,
            avg_iou=0.87,
            memory_mb=140,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.92,
            parameter_sensitivity=0.35,
            description="Морфологические змеи (устойчивы к шуму) (torch)",
            params={"iterations": 100, "smoothing": 1, "threshold": 0.5},
            schema={
                "iterations": {
                    "type": "int",
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "label": "Итерации",
                },
                "smoothing": {
                    "type": "int",
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "label": "Степень сглаживания",
                },
                "threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Порог инициализации",
                },
            },
        ),
        "chan_vese": MethodProfile(
            name="chan_vese",
            library="torch",
            avg_time_ms=400.0,
            avg_iou=0.89,
            memory_mb=170,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.94,
            parameter_sensitivity=0.3,
            description="Модель Чан-Везе (регион-базированные активные контуры) (torch)",
            params={
                "mu": 0.25,
                "lambda1": 1.0,
                "lambda2": 1.0,
                "tol": 1e-3,
                "max_iter": 100,
                "dt": 0.5,
                "eps": 1.0,
                "init_level_set": "checkerboard",
            },
            schema={
                "mu": {
                    "type": "float",
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "label": "Длина контура (mu)",
                },
                "lambda1": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Внешняя область (lambda1)",
                },
                "lambda2": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Внутренняя область (lambda2)",
                },
                "tol": {
                    "type": "float",
                    "min": 0.0001,
                    "max": 0.01,
                    "step": 0.0001,
                    "label": "Точность сходимости",
                },
                "max_iter": {
                    "type": "int",
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "label": "Макс. итераций",
                },
                "dt": {
                    "type": "float",
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Шаг времени (dt)",
                },
                "eps": {
                    "type": "float",
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Параметр фазового поля",
                },
            },
        ),
        # ===== WATERSHED =====
        "watershed": MethodProfile(
            name="watershed",
            library="torch",
            avg_time_ms=35.0,
            avg_iou=0.73,
            memory_mb=75,
            best_for_type=[
                ImageType.MEDICAL,
                ImageType.MICROSCOPY,
                ImageType.SATELLITE,
            ],
            robustness=0.65,
            parameter_sensitivity=0.8,
            description="Классический watershed по градиенту (torch)",
            params={},
            schema={},
        ),
        "random_walker": MethodProfile(
            name="random_walker",
            library="torch",
            avg_time_ms=95.0,
            avg_iou=0.85,
            memory_mb=130,
            best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
            robustness=0.9,
            parameter_sensitivity=0.4,
            description="Random walker с вероятностной диффузией (torch)",
            params={
                "beta": 130,
                "tol": 1e-3,
                "max_iter": 300,
                "target_label": 2,
            },
            schema={
                "beta": {
                    "type": "int",
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "label": "Коэф. диффузии (beta)",
                },
                "tol": {
                    "type": "float",
                    "min": 0.0001,
                    "max": 0.01,
                    "step": 0.0001,
                    "label": "Точность",
                },
                "max_iter": {
                    "type": "int",
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "label": "Итерации",
                },
                "target_label": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "label": "Целевая метка объекта",
                },
            },
        ),
        # ===== SUPER-PIXELS =====
        "slic": MethodProfile(
            name="slic",
            library="torch",
            avg_time_ms=65.0,
            avg_iou=0.79,
            memory_mb=95,
            best_for_type=[
                ImageType.NATURAL,
                ImageType.SATELLITE,
                ImageType.MEDICAL,
            ],
            robustness=0.8,
            parameter_sensitivity=0.4,
            description="SLIC super-pixels в Lab-пространстве (torch)",
            params={
                "n_segments": 100,
                "compactness": 10.0,
                "max_iter": 10,
                "sigma": 0.0,
                "enforce_connectivity": True,
                "min_size_factor": 0.5,
                "max_size_factor": 3.0,
                "ruler": 10.0,
                "region_size": 20,
            },
            schema={
                "n_segments": {
                    "type": "int",
                    "min": 50,
                    "max": 1000,
                    "step": 50,
                    "label": "Кол-во сегментов",
                },
                "compactness": {
                    "type": "float",
                    "min": 0.1,
                    "max": 50.0,
                    "step": 0.1,
                    "label": "Компактность",
                },
                "max_iter": {
                    "type": "int",
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "label": "Итерации",
                },
                "sigma": {
                    "type": "float",
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Сглаживание Гаусса",
                },
                "min_size_factor": {
                    "type": "float",
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "label": "Мин. фактор размера",
                },
                "max_size_factor": {
                    "type": "float",
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "label": "Макс. фактор размера",
                },
                "ruler": {
                    "type": "float",
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1,
                    "label": "Масштабная линейка",
                },
                "region_size": {
                    "type": "int",
                    "min": 5,
                    "max": 100,
                    "step": 1,
                    "label": "Базовый размер региона",
                },
            },
        ),
        "felzenszwalb": MethodProfile(
            name="felzenszwalb",
            library="torch",
            avg_time_ms=85.0,
            avg_iou=0.81,
            memory_mb=110,
            best_for_type=[ImageType.NATURAL, ImageType.INDUSTRIAL],
            robustness=0.78,
            parameter_sensitivity=0.5,
            description="Граф-базированная сегментация Фельценцвальба (torch)",
            params={"scale": 100, "sigma": 0.5, "min_size": 50},
            schema={
                "scale": {
                    "type": "int",
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "label": "Масштаб сегментации",
                },
                "sigma": {
                    "type": "float",
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "label": "Сглаживание Гаусса",
                },
                "min_size": {
                    "type": "int",
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "label": "Мин. размер сегмента",
                },
            },
        ),
        # ===== INTERACTIVE =====
        "grabcut": MethodProfile(
            name="grabcut",
            library="torch",
            avg_time_ms=150.0,
            avg_iou=0.91,
            memory_mb=180,
            best_for_type=[ImageType.NATURAL, ImageType.MEDICAL],
            robustness=0.88,
            parameter_sensitivity=0.5,
            description="GrabCut с итеративной оптимизацией GMM (torch)",
            params={"num_iterations": 5},
            schema={
                "num_iterations": {
                    "type": "int",
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "label": "Итерации оптимизации",
                }
            },
        ),
    },
}

# Flat dict для быстрого доступа (как было)
ALL_METHODS = {
    name: profile
    for lib_methods in METHODS_BY_LIBRARY.values()
    for name, profile in lib_methods.items()
}


class AutoSegmenter:
    """
    Интеллектуальный селектор методов сегментации.
    Автоматически выбирает оптимальный метод на основе:
    - Характеристик изображения
    - Цели пользователя (скорость/точность)
    - Результатов бенчмарков
    """

    def __init__(
        self,
        goal: SegmentationGoal = SegmentationGoal.BALANCED,
        custom_weights: Optional[Dict[str, float]] = None,
        benchmark_data_path: Optional[str] = None,
    ):
        self.success_thresholds: Dict[str, float] = {
            "iou": 0.80,
            "dice": 0.85,
            "pixel_accuracy": 0.90,
            "precision": 0.80,
            "recall": 0.80,
            "f1_score": 0.82,
            "mae": 0.15,
        }
        self.goal = goal
        self.custom_weights = custom_weights or {}
        self.benchmark_data = self._load_benchmark_data(benchmark_data_path)
        self.available_methods = self._register_methods()

        for name, profile in self.benchmark_data.items():
            if name in self.available_methods and profile.params:
                self.available_methods[name]["params"].update(profile.params)

    def _register_methods(self) -> Dict[str, Any]:
        """Регистрация всех доступных методов сегментации с параметрами по умолчанию и UI-схемами"""
        return {
            # ========== ПОРОГОВЫЕ МЕТОДЫ ==========
            "global_thresholding": {
                "class": "threshold",
                "params": {"threshold": 0.5},
                "schema": {
                    "threshold": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Порог яркости (0-1)",
                    }
                },
                "description": "Простое глобальное пороговое значение",
            },
            "otsu_thresholding": {
                "class": "threshold",
                "params": {},
                "schema": {},
                "description": "Автоматический порог Оцу (максимизация межклассовой дисперсии)",
            },
            "adaptive_thresholding": {
                "class": "threshold",
                "params": {"block_size": 11, "C": 2},
                "schema": {
                    "block_size": {
                        "type": "int",
                        "min": 3,
                        "max": 99,
                        "step": 2,
                        "label": "Размер блока (нечетный)",
                    },
                    "C": {
                        "type": "int",
                        "min": -20,
                        "max": 20,
                        "step": 1,
                        "label": "Константа C (смещение)",
                    },
                },
                "description": "Адаптивный порог с локальным усреднением",
            },
            "threshold_niblack": {
                "class": "threshold",
                "params": {"window_size": 15, "k": -0.2},
                "schema": {
                    "window_size": {
                        "type": "int",
                        "min": 3,
                        "max": 99,
                        "step": 2,
                        "label": "Размер окна",
                    },
                    "k": {
                        "type": "float",
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Константа k",
                    },
                },
                "description": "Метод Ниблэка для локального порогования",
            },
            "threshold_sauvola": {
                "class": "threshold",
                "params": {"window_size": 15, "k": 0.5, "r": 128},
                "schema": {
                    "window_size": {
                        "type": "int",
                        "min": 3,
                        "max": 99,
                        "step": 2,
                        "label": "Размер окна",
                    },
                    "k": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Константа k",
                    },
                    "r": {
                        "type": "float",
                        "min": 50.0,
                        "max": 255.0,
                        "step": 1.0,
                        "label": "Динамический диапазон R",
                    },
                },
                "description": "Метод Саволы (улучшенный Ниблэк для текста)",
            },
            "threshold_bernsen": {
                "class": "threshold",
                "params": {"window_size": 15, "contrast_threshold": 0.15},
                "schema": {
                    "window_size": {
                        "type": "int",
                        "min": 3,
                        "max": 99,
                        "step": 2,
                        "label": "Размер окна",
                    },
                    "contrast_threshold": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Порог контраста",
                    },
                },
                "description": "Метод Бернсена на основе локального контраста",
            },
            "threshold_phansalkar": {
                "class": "threshold",
                "params": {"window_size": 15, "k": 0.25, "r": 128.0, "m": 0.5},
                "schema": {
                    "window_size": {
                        "type": "int",
                        "min": 3,
                        "max": 99,
                        "step": 2,
                        "label": "Размер окна",
                    },
                    "k": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Чувствительность k",
                    },
                    "r": {
                        "type": "float",
                        "min": 50.0,
                        "max": 255.0,
                        "step": 1.0,
                        "label": "Диапазон R",
                    },
                    "m": {
                        "type": "float",
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "label": "Смещение m",
                    },
                },
                "description": "Метод Фансалкара для низкоконтрастных изображений",
            },
            "threshold_kittler_illingworth": {
                "class": "threshold",
                "params": {"num_bins": 256},
                "schema": {
                    "num_bins": {
                        "type": "int",
                        "min": 32,
                        "max": 512,
                        "step": 16,
                        "label": "Кол-во бинов гистограммы",
                    }
                },
                "description": "Минимизация ошибки классификации (Киттлер-Иллингуорт)",
            },
            "threshold_entropy_kapur": {
                "class": "threshold",
                "params": {"num_bins": 256},
                "schema": {
                    "num_bins": {
                        "type": "int",
                        "min": 32,
                        "max": 512,
                        "step": 16,
                        "label": "Кол-во бинов гистограммы",
                    }
                },
                "description": "Максимизация энтропии (Капур)",
            },
            "threshold_triangle": {
                "class": "threshold",
                "params": {"num_bins": 256},
                "schema": {
                    "num_bins": {
                        "type": "int",
                        "min": 32,
                        "max": 512,
                        "step": 16,
                        "label": "Кол-во бинов гистограммы",
                    }
                },
                "description": "Треугольный метод для унимодальных гистограмм",
            },
            "threshold_multi_otsu": {
                "class": "threshold",
                "params": {"n_thresholds": 2},
                "schema": {
                    "n_thresholds": {
                        "type": "int",
                        "min": 1,
                        "max": 5,
                        "step": 1,
                        "label": "Кол-во порогов",
                    }
                },
                "description": "Многопороговый Оцу для многоклассовой сегментации",
            },
            "threshold_percentile": {
                "class": "threshold",
                "params": {"percentile": 90},
                "schema": {
                    "percentile": {
                        "type": "int",
                        "min": 1,
                        "max": 99,
                        "step": 1,
                        "label": "Процентиль (%)",
                    }
                },
                "description": "Порог по перцентилю интенсивности",
            },
            "threshold_local_contrast": {
                "class": "threshold",
                "params": {"window_size": 15, "contrast_factor": 0.1},
                "schema": {
                    "window_size": {
                        "type": "int",
                        "min": 3,
                        "max": 99,
                        "step": 2,
                        "label": "Размер окна",
                    },
                    "contrast_factor": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Фактор контраста",
                    },
                },
                "description": "Порог на основе локального контраста",
            },
            # ========== ГРАНИЧНЫЕ МЕТОДЫ ==========
            "canny_edge": {
                "class": "edge",
                "params": {"low": 0.1, "high": 0.3, "sigma": 1.0},
                "schema": {
                    "low": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Нижний порог",
                    },
                    "high": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Верхний порог",
                    },
                    "sigma": {
                        "type": "float",
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "label": "Сигма (размытие)",
                    },
                },
                "description": "Детектор границ Кэнни (оптимальный по Кэнни)",
            },
            "sobel_edge": {
                "class": "edge",
                "params": {"threshold": 0.1},
                "schema": {
                    "threshold": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Порог градиента",
                    }
                },
                "description": "Градиенты Собеля с порогом",
            },
            "prewitt_edge": {
                "class": "edge",
                "params": {"threshold": 0.1},
                "schema": {
                    "threshold": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Порог градиента",
                    }
                },
                "description": "Градиенты Превитта с порогом",
            },
            "scharr_edge": {
                "class": "edge",
                "params": {"threshold": 0.1},
                "schema": {
                    "threshold": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Порог градиента",
                    }
                },
                "description": "Градиенты Шарра (более точные, чем Собель)",
            },
            "roberts_cross_edge": {
                "class": "edge",
                "params": {"threshold": 0.1},
                "schema": {
                    "threshold": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Порог градиента",
                    }
                },
                "description": "Оператор Робертса для диагональных границ",
            },
            "log_edge": {
                "class": "edge",
                "params": {"sigma": 1.0, "threshold": 0.01},
                "schema": {
                    "sigma": {
                        "type": "float",
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "label": "Сигма размытия",
                    },
                    "threshold": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Порог детекции",
                    },
                },
                "description": "Laplacian of Gaussian детектор границ",
            },
            "dog_edge": {
                "class": "edge",
                "params": {"sigma1": 1.0, "sigma2": 2.0, "threshold": 0.01},
                "schema": {
                    "sigma1": {
                        "type": "float",
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "label": "Сигма 1",
                    },
                    "sigma2": {
                        "type": "float",
                        "min": 0.1,
                        "max": 20.0,
                        "step": 0.1,
                        "label": "Сигма 2",
                    },
                    "threshold": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Порог детекции",
                    },
                },
                "description": "Difference of Gaussians для мультимасштабных границ",
            },
            "marr_hildreth_edge": {
                "class": "edge",
                "params": {"sigma": 1.5, "threshold": 0.01},
                "schema": {
                    "sigma": {
                        "type": "float",
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "label": "Сигма размытия",
                    },
                    "threshold": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Порог нулевых пересечений",
                    },
                },
                "description": "Метод Марра-Хилдрета (нулевые пересечения LoG)",
            },
            "gradient_magnitude_direction": {
                "class": "edge",
                "params": {"threshold": 0.1},
                "schema": {
                    "threshold": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Порог магнитуды",
                    }
                },
                "description": "Сегментация по величине и направлению градиента",
            },
            "phase_congruency_edge": {
                "class": "edge",
                "params": {
                    "nscales": 4,
                    "norientations": 4,
                    "min_wavelength": 3,
                    "mult": 2.0,
                    "sigma_onf": 0.55,
                    "k_noise": 2.0,
                    "threshold": 0.5,
                },
                "schema": {
                    "nscales": {
                        "type": "int",
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "label": "Кол-во масштабов",
                    },
                    "norientations": {
                        "type": "int",
                        "min": 1,
                        "max": 12,
                        "step": 1,
                        "label": "Кол-во ориентаций",
                    },
                    "min_wavelength": {
                        "type": "int",
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "label": "Мин. длина волны",
                    },
                    "mult": {
                        "type": "float",
                        "min": 1.0,
                        "max": 5.0,
                        "step": 0.1,
                        "label": "Множитель масштаба",
                    },
                    "sigma_onf": {
                        "type": "float",
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.05,
                        "label": "Сигма частотной области",
                    },
                    "k_noise": {
                        "type": "float",
                        "min": 0.5,
                        "max": 5.0,
                        "step": 0.1,
                        "label": "Коэф. шумоподавления",
                    },
                    "threshold": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Порог энергии",
                    },
                },
                "description": "Фазовая конгруэнтность (инвариантна к освещению)",
            },
            # ========== РЕГИОНАЛЬНЫЕ МЕТОДЫ ==========
            "region_growing": {
                "class": "region",
                "params": {"tolerance": 0.1},
                "schema": {
                    "tolerance": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Допуск схожести",
                    }
                },
                "description": "Рост региона от семян по схожести",
            },
            "split_and_merge": {
                "class": "region",
                "params": {"min_size": 50, "threshold": 20},
                "schema": {
                    "min_size": {
                        "type": "int",
                        "min": 10,
                        "max": 500,
                        "step": 10,
                        "label": "Мин. размер региона",
                    },
                    "threshold": {
                        "type": "int",
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "label": "Порог слияния",
                    },
                },
                "description": "Разделение и слияние регионов",
            },
            "floodfill": {
                "class": "region",
                "params": {"tolerance": 0.15},
                "schema": {
                    "tolerance": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Допуск заливки",
                    }
                },
                "description": "Заливка области от точки",
            },
            # ========== КЛАСТЕРИЗАЦИЯ ==========
            "kmeans_segmentation": {
                "class": "clustering",
                "params": {"k": 3},
                "schema": {
                    "k": {
                        "type": "int",
                        "min": 2,
                        "max": 20,
                        "step": 1,
                        "label": "Кол-во кластеров",
                    }
                },
                "description": "K-means кластеризация в пространстве признаков",
            },
            "dbscan_segmentation": {
                "class": "clustering",
                "params": {"eps": 0.1, "min_samples": 10},
                "schema": {
                    "eps": {
                        "type": "float",
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Радиус окрестности (eps)",
                    },
                    "min_samples": {
                        "type": "int",
                        "min": 1,
                        "max": 50,
                        "step": 1,
                        "label": "Мин. точек в кластере",
                    },
                },
                "description": "DBSCAN для сегментации произвольной формы",
            },
            "meanshift": {
                "class": "clustering",
                "params": {
                    "bandwidth": 0.5,
                    "spatial_radius": 35,
                    "color_radius": 60,
                    "max_level": 1,
                },
                "schema": {
                    "bandwidth": {
                        "type": "float",
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.1,
                        "label": "Полоса пропускания",
                    },
                    "spatial_radius": {
                        "type": "int",
                        "min": 5,
                        "max": 100,
                        "step": 1,
                        "label": "Пространственный радиус",
                    },
                    "color_radius": {
                        "type": "int",
                        "min": 10,
                        "max": 200,
                        "step": 1,
                        "label": "Цветовой радиус",
                    },
                    "max_level": {
                        "type": "int",
                        "min": 0,
                        "max": 5,
                        "step": 1,
                        "label": "Макс. уровень пирамиды",
                    },
                },
                "description": "MeanShift с пространственно-цветовым ядром",
            },
            # ========== АКТИВНЫЕ КОНТУРЫ ==========
            "active_contour": {
                "class": "active_contour",
                "params": {
                    "alpha": 0.015,
                    "beta": 10,
                    "gamma": 0.001,
                    "max_iterations": 2000,
                    "w_edge": 1,
                    "w_line": 0,
                },
                "schema": {
                    "alpha": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "label": "Плавность контура",
                    },
                    "beta": {
                        "type": "float",
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.1,
                        "label": "Жесткость контура",
                    },
                    "gamma": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "label": "Вязкость (шаг)",
                    },
                    "max_iterations": {
                        "type": "int",
                        "min": 100,
                        "max": 5000,
                        "step": 100,
                        "label": "Итерации",
                    },
                    "w_edge": {
                        "type": "float",
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "label": "Вес границ",
                    },
                    "w_line": {
                        "type": "float",
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "label": "Вес линий",
                    },
                },
                "description": "Змеи (snakes) с энергией границ и линий",
            },
            "gvf_contour": {
                "class": "active_contour",
                "params": {"mu": 0.1, "iterations": 50},
                "schema": {
                    "mu": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Коэф. диффузии (mu)",
                    },
                    "iterations": {
                        "type": "int",
                        "min": 10,
                        "max": 500,
                        "step": 10,
                        "label": "Итерации GVF",
                    },
                },
                "description": "Контуры с градиентным векторным потоком (GVF)",
            },
            "morphological_snakes": {
                "class": "active_contour",
                "params": {"iterations": 100, "smoothing": 1, "threshold": 0.5},
                "schema": {
                    "iterations": {
                        "type": "int",
                        "min": 10,
                        "max": 500,
                        "step": 10,
                        "label": "Итерации",
                    },
                    "smoothing": {
                        "type": "int",
                        "min": 0,
                        "max": 5,
                        "step": 1,
                        "label": "Степень сглаживания",
                    },
                    "threshold": {
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Порог инициализации",
                    },
                },
                "description": "Морфологические змеи (устойчивы к шуму)",
            },
            "chan_vese": {
                "class": "active_contour",
                "params": {
                    "mu": 0.25,
                    "lambda1": 1.0,
                    "lambda2": 1.0,
                    "tol": 1e-3,
                    "max_iter": 100,
                    "dt": 0.5,
                    "eps": 1.0,
                    "init_level_set": "checkerboard",
                },
                "schema": {
                    "mu": {
                        "type": "float",
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "label": "Длина контура (mu)",
                    },
                    "lambda1": {
                        "type": "float",
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "label": "Внешняя область (lambda1)",
                    },
                    "lambda2": {
                        "type": "float",
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "label": "Внутренняя область (lambda2)",
                    },
                    "tol": {
                        "type": "float",
                        "min": 0.0001,
                        "max": 0.01,
                        "step": 0.0001,
                        "label": "Точность сходимости",
                    },
                    "max_iter": {
                        "type": "int",
                        "min": 10,
                        "max": 1000,
                        "step": 10,
                        "label": "Макс. итераций",
                    },
                    "dt": {
                        "type": "float",
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "label": "Шаг времени (dt)",
                    },
                    "eps": {
                        "type": "float",
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.1,
                        "label": "Параметр фазового поля",
                    },
                },
                "description": "Модель Чан-Везе (регион-базированные активные контуры)",
            },
            # ========== WATERSHED ==========
            "watershed": {
                "class": "watershed",
                "params": {},
                "schema": {},
                "description": "Классический watershed по градиенту",
            },
            "random_walker": {
                "class": "watershed",
                "params": {
                    "beta": 130,
                    "tol": 1e-3,
                    "max_iter": 300,
                    "target_label": 2,
                },
                "schema": {
                    "beta": {
                        "type": "int",
                        "min": 10,
                        "max": 500,
                        "step": 10,
                        "label": "Коэф. диффузии (beta)",
                    },
                    "tol": {
                        "type": "float",
                        "min": 0.0001,
                        "max": 0.01,
                        "step": 0.0001,
                        "label": "Точность",
                    },
                    "max_iter": {
                        "type": "int",
                        "min": 10,
                        "max": 1000,
                        "step": 10,
                        "label": "Итерации",
                    },
                    "target_label": {
                        "type": "int",
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "label": "Целевая метка объекта",
                    },
                },
                "description": "Random walker с вероятностной диффузией",
            },
            # ========== SUPER-PIXELS ==========
            "slic": {
                "class": "superpixel",
                "params": {
                    "n_segments": 100,
                    "compactness": 10.0,
                    "max_iter": 10,
                    "sigma": 0.0,
                    "enforce_connectivity": True,
                    "min_size_factor": 0.5,
                    "max_size_factor": 3.0,
                    "ruler": 10.0,
                    "region_size": 20,
                },
                "schema": {
                    "n_segments": {
                        "type": "int",
                        "min": 50,
                        "max": 1000,
                        "step": 50,
                        "label": "Кол-во сегментов",
                    },
                    "compactness": {
                        "type": "float",
                        "min": 0.1,
                        "max": 50.0,
                        "step": 0.1,
                        "label": "Компактность",
                    },
                    "max_iter": {
                        "type": "int",
                        "min": 1,
                        "max": 50,
                        "step": 1,
                        "label": "Итерации",
                    },
                    "sigma": {
                        "type": "float",
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "label": "Сглаживание Гаусса",
                    },
                    "min_size_factor": {
                        "type": "float",
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.1,
                        "label": "Мин. фактор размера",
                    },
                    "max_size_factor": {
                        "type": "float",
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.1,
                        "label": "Макс. фактор размера",
                    },
                    "ruler": {
                        "type": "float",
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.1,
                        "label": "Масштабная линейка",
                    },
                    "region_size": {
                        "type": "int",
                        "min": 5,
                        "max": 100,
                        "step": 1,
                        "label": "Базовый размер региона",
                    },
                },
                "description": "SLIC super-pixels в Lab-пространстве",
            },
            "felzenszwalb": {
                "class": "superpixel",
                "params": {"scale": 100, "sigma": 0.5, "min_size": 50},
                "schema": {
                    "scale": {
                        "type": "int",
                        "min": 10,
                        "max": 1000,
                        "step": 10,
                        "label": "Масштаб сегментации",
                    },
                    "sigma": {
                        "type": "float",
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "label": "Сглаживание Гаусса",
                    },
                    "min_size": {
                        "type": "int",
                        "min": 10,
                        "max": 500,
                        "step": 10,
                        "label": "Мин. размер сегмента",
                    },
                },
                "description": "Граф-базированная сегментация Фельценцвальба",
            },
            # ========== ИНТЕРАКТИВНЫЕ МЕТОДЫ ==========
            "grabcut": {
                "class": "interactive",
                "params": {"num_iterations": 5},
                "schema": {
                    "num_iterations": {
                        "type": "int",
                        "min": 1,
                        "max": 20,
                        "step": 1,
                        "label": "Итерации оптимизации",
                    }
                },
                "description": "GrabCut с итеративной оптимизацией GMM",
            },
        }

    def _load_benchmark_data(self, path: Optional[str]) -> Dict[str, MethodProfile]:
        """
        Профили методов на основе бенчмарков.
        Данные усреднены по тестам на наборах: DIBCO, BSDS500, ISIC, INRIA.
        """
        return {
            # ===== THRESHOLDING =====
            "global_thresholding": MethodProfile(
                name="global_thresholding",
                library="opencv",
                avg_time_ms=2.0,
                avg_iou=0.62,
                memory_mb=15,
                best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
                robustness=0.4,
                parameter_sensitivity=0.9,
            ),
            "otsu_thresholding": MethodProfile(
                name="otsu_thresholding",
                library="opencv",
                avg_time_ms=15.0,
                avg_iou=0.75,
                memory_mb=50,
                best_for_type=[ImageType.DOCUMENT, ImageType.NATURAL],
                robustness=0.8,
                parameter_sensitivity=0.2,
            ),
            "adaptive_thresholding": MethodProfile(
                name="adaptive_thresholding",
                library="opencv",
                avg_time_ms=45.0,
                avg_iou=0.82,
                memory_mb=80,
                best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
                robustness=0.9,
                parameter_sensitivity=0.4,
            ),
            "threshold_niblack": MethodProfile(
                name="threshold_niblack",
                library="opencv",
                avg_time_ms=38.0,
                avg_iou=0.71,
                memory_mb=65,
                best_for_type=[ImageType.DOCUMENT, ImageType.MICROSCOPY],
                robustness=0.6,
                parameter_sensitivity=0.7,
            ),
            "threshold_sauvola": MethodProfile(
                name="threshold_sauvola",
                library="opencv",
                avg_time_ms=42.0,
                avg_iou=0.88,
                memory_mb=70,
                best_for_type=[ImageType.DOCUMENT, ImageType.MICROSCOPY],
                robustness=0.92,
                parameter_sensitivity=0.3,
            ),
            "threshold_bernsen": MethodProfile(
                name="threshold_bernsen",
                library="opencv",
                avg_time_ms=35.0,
                avg_iou=0.73,
                memory_mb=60,
                best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
                robustness=0.7,
                parameter_sensitivity=0.5,
            ),
            "threshold_phansalkar": MethodProfile(
                name="threshold_phansalkar",
                library="opencv",
                avg_time_ms=48.0,
                avg_iou=0.85,
                memory_mb=75,
                best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
                robustness=0.88,
                parameter_sensitivity=0.4,
            ),
            "threshold_kittler_illingworth": MethodProfile(
                name="threshold_kittler_illingworth",
                library="opencv",
                avg_time_ms=25.0,
                avg_iou=0.76,
                memory_mb=55,
                best_for_type=[ImageType.DOCUMENT, ImageType.NATURAL],
                robustness=0.75,
                parameter_sensitivity=0.3,
            ),
            "threshold_entropy_kapur": MethodProfile(
                name="threshold_entropy_kapur",
                library="opencv",
                avg_time_ms=30.0,
                avg_iou=0.74,
                memory_mb=60,
                best_for_type=[ImageType.NATURAL, ImageType.SATELLITE],
                robustness=0.7,
                parameter_sensitivity=0.4,
            ),
            "threshold_triangle": MethodProfile(
                name="threshold_triangle",
                library="opencv",
                avg_time_ms=20.0,
                avg_iou=0.69,
                memory_mb=45,
                best_for_type=[ImageType.DOCUMENT, ImageType.MEDICAL],
                robustness=0.65,
                parameter_sensitivity=0.3,
            ),
            "threshold_multi_otsu": MethodProfile(
                name="threshold_multi_otsu",
                library="opencv",
                avg_time_ms=35.0,
                avg_iou=0.78,
                memory_mb=70,
                best_for_type=[ImageType.MEDICAL, ImageType.SATELLITE],
                robustness=0.8,
                parameter_sensitivity=0.5,
            ),
            "threshold_percentile": MethodProfile(
                name="threshold_percentile",
                library="opencv",
                avg_time_ms=8.0,
                avg_iou=0.65,
                memory_mb=25,
                best_for_type=[ImageType.INDUSTRIAL, ImageType.DOCUMENT],
                robustness=0.5,
                parameter_sensitivity=0.8,
            ),
            "threshold_local_contrast": MethodProfile(
                name="threshold_local_contrast",
                library="opencv",
                avg_time_ms=40.0,
                avg_iou=0.77,
                memory_mb=68,
                best_for_type=[ImageType.MICROSCOPY, ImageType.MEDICAL],
                robustness=0.82,
                parameter_sensitivity=0.5,
            ),
            # ===== EDGE DETECTION =====
            "canny_edge": MethodProfile(
                name="canny_edge",
                library="opencv",
                avg_time_ms=25.0,
                avg_iou=0.68,
                memory_mb=60,
                best_for_type=[ImageType.NATURAL, ImageType.INDUSTRIAL],
                robustness=0.7,
                parameter_sensitivity=0.6,
            ),
            "sobel_edge": MethodProfile(
                name="sobel_edge",
                library="opencv",
                avg_time_ms=12.0,
                avg_iou=0.58,
                memory_mb=35,
                best_for_type=[ImageType.NATURAL, ImageType.DOCUMENT],
                robustness=0.5,
                parameter_sensitivity=0.7,
            ),
            "prewitt_edge": MethodProfile(
                name="prewitt_edge",
                library="opencv",
                avg_time_ms=11.0,
                avg_iou=0.56,
                memory_mb=33,
                best_for_type=[ImageType.NATURAL, ImageType.DOCUMENT],
                robustness=0.48,
                parameter_sensitivity=0.72,
            ),
            "scharr_edge": MethodProfile(
                name="scharr_edge",
                library="opencv",
                avg_time_ms=14.0,
                avg_iou=0.61,
                memory_mb=38,
                best_for_type=[ImageType.NATURAL, ImageType.INDUSTRIAL],
                robustness=0.55,
                parameter_sensitivity=0.65,
            ),
            "roberts_cross_edge": MethodProfile(
                name="roberts_cross_edge",
                library="opencv",
                avg_time_ms=8.0,
                avg_iou=0.52,
                memory_mb=28,
                best_for_type=[ImageType.DOCUMENT, ImageType.INDUSTRIAL],
                robustness=0.4,
                parameter_sensitivity=0.8,
            ),
            "log_edge": MethodProfile(
                name="log_edge",
                library="opencv",
                avg_time_ms=22.0,
                avg_iou=0.64,
                memory_mb=55,
                best_for_type=[ImageType.NATURAL, ImageType.MEDICAL],
                robustness=0.6,
                parameter_sensitivity=0.5,
            ),
            "dog_edge": MethodProfile(
                name="dog_edge",
                library="opencv",
                avg_time_ms=28.0,
                avg_iou=0.66,
                memory_mb=62,
                best_for_type=[ImageType.NATURAL, ImageType.SATELLITE],
                robustness=0.68,
                parameter_sensitivity=0.55,
            ),
            "marr_hildreth_edge": MethodProfile(
                name="marr_hildreth_edge",
                library="opencv",
                avg_time_ms=26.0,
                avg_iou=0.63,
                memory_mb=58,
                best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
                robustness=0.62,
                parameter_sensitivity=0.58,
            ),
            "gradient_magnitude_direction": MethodProfile(
                name="gradient_magnitude_direction",
                library="opencv",
                avg_time_ms=18.0,
                avg_iou=0.59,
                memory_mb=45,
                best_for_type=[ImageType.INDUSTRIAL, ImageType.NATURAL],
                robustness=0.52,
                parameter_sensitivity=0.68,
            ),
            "phase_congruency_edge": MethodProfile(
                name="phase_congruency_edge",
                library="opencv",
                avg_time_ms=85.0,
                avg_iou=0.79,
                memory_mb=120,
                best_for_type=[
                    ImageType.MEDICAL,
                    ImageType.SATELLITE,
                    ImageType.MICROSCOPY,
                ],
                robustness=0.95,
                parameter_sensitivity=0.3,
            ),
            # ===== REGION-BASED =====
            "region_growing": MethodProfile(
                name="region_growing",
                library="opencv",
                avg_time_ms=55.0,
                avg_iou=0.81,
                memory_mb=90,
                best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
                robustness=0.75,
                parameter_sensitivity=0.6,
            ),
            "split_and_merge": MethodProfile(
                name="split_and_merge",
                library="opencv",
                avg_time_ms=70.0,
                avg_iou=0.76,
                memory_mb=100,
                best_for_type=[ImageType.SATELLITE, ImageType.INDUSTRIAL],
                robustness=0.7,
                parameter_sensitivity=0.5,
            ),
            "floodfill": MethodProfile(
                name="floodfill",
                library="opencv",
                avg_time_ms=15.0,
                avg_iou=0.72,
                memory_mb=40,
                best_for_type=[ImageType.DOCUMENT, ImageType.MEDICAL],
                robustness=0.6,
                parameter_sensitivity=0.7,
            ),
            # ===== CLUSTERING =====
            "kmeans_segmentation": MethodProfile(
                name="kmeans_segmentation",
                library="opencv",
                avg_time_ms=120.0,
                avg_iou=0.77,
                memory_mb=150,
                best_for_type=[ImageType.NATURAL, ImageType.SATELLITE],
                robustness=0.65,
                parameter_sensitivity=0.7,
            ),
            "dbscan_segmentation": MethodProfile(
                name="dbscan_segmentation",
                library="opencv",
                avg_time_ms=180.0,
                avg_iou=0.74,
                memory_mb=200,
                best_for_type=[ImageType.MICROSCOPY, ImageType.INDUSTRIAL],
                robustness=0.8,
                parameter_sensitivity=0.6,
            ),
            "meanshift": MethodProfile(
                name="meanshift",
                library="opencv",
                avg_time_ms=250.0,
                avg_iou=0.83,
                memory_mb=280,
                best_for_type=[ImageType.NATURAL, ImageType.MEDICAL],
                robustness=0.85,
                parameter_sensitivity=0.4,
            ),
            # ===== ACTIVE CONTOURS =====
            "active_contour": MethodProfile(
                name="active_contour",
                library="opencv",
                avg_time_ms=450.0,
                avg_iou=0.84,
                memory_mb=180,
                best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
                robustness=0.78,
                parameter_sensitivity=0.75,
            ),
            "gvf_contour": MethodProfile(
                name="gvf_contour",
                library="opencv",
                avg_time_ms=380.0,
                avg_iou=0.86,
                memory_mb=160,
                best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
                robustness=0.88,
                parameter_sensitivity=0.5,
            ),
            "morphological_snakes": MethodProfile(
                name="morphological_snakes",
                library="opencv",
                avg_time_ms=320.0,
                avg_iou=0.87,
                memory_mb=140,
                best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
                robustness=0.92,
                parameter_sensitivity=0.35,
            ),
            "chan_vese": MethodProfile(
                name="chan_vese",
                library="opencv",
                avg_time_ms=400.0,
                avg_iou=0.89,
                memory_mb=170,
                best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
                robustness=0.94,
                parameter_sensitivity=0.3,
            ),
            # ===== WATERSHED =====
            "watershed": MethodProfile(
                name="watershed",
                library="opencv",
                avg_time_ms=35.0,
                avg_iou=0.73,
                memory_mb=75,
                best_for_type=[
                    ImageType.MEDICAL,
                    ImageType.MICROSCOPY,
                    ImageType.SATELLITE,
                ],
                robustness=0.65,
                parameter_sensitivity=0.8,
            ),
            "random_walker": MethodProfile(
                name="random_walker",
                library="opencv",
                avg_time_ms=95.0,
                avg_iou=0.85,
                memory_mb=130,
                best_for_type=[ImageType.MEDICAL, ImageType.MICROSCOPY],
                robustness=0.9,
                parameter_sensitivity=0.4,
            ),
            # ===== SUPER-PIXELS =====
            "slic": MethodProfile(
                name="slic",
                library="opencv",
                avg_time_ms=65.0,
                avg_iou=0.79,
                memory_mb=95,
                best_for_type=[
                    ImageType.NATURAL,
                    ImageType.SATELLITE,
                    ImageType.MEDICAL,
                ],
                robustness=0.8,
                parameter_sensitivity=0.4,
            ),
            "felzenszwalb": MethodProfile(
                name="felzenszwalb",
                library="opencv",
                avg_time_ms=85.0,
                avg_iou=0.81,
                memory_mb=110,
                best_for_type=[ImageType.NATURAL, ImageType.INDUSTRIAL],
                robustness=0.78,
                parameter_sensitivity=0.5,
            ),
            # ===== INTERACTIVE =====
            "grabcut": MethodProfile(
                name="grabcut",
                library="opencv",
                avg_time_ms=150.0,
                avg_iou=0.91,
                memory_mb=180,
                best_for_type=[ImageType.NATURAL, ImageType.MEDICAL],
                robustness=0.88,
                parameter_sensitivity=0.5,
            ),
        }

    def get_available_methods(
        self, library: Optional[str] = None
    ) -> Dict[str, MethodProfile]:
        """Возвращает доступные методы, опционально отфильтрованные по библиотеке"""
        if library:
            return METHODS_BY_LIBRARY.get(library, {})
        return self.benchmark_data

    def analyze_image(self, image: np.ndarray) -> ImageCharacteristics:
        """Анализ характеристик изображения"""
        if len(image.shape) == 3:
            height, width, channels = image.shape
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            height, width = image.shape
            channels = 1
            gray = image

        # Основные статистики
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        contrast = (np.max(gray) - np.min(gray)) / (np.max(gray) + 1e-6)

        # Оценка шума (через локальную дисперсию)
        local_std = (
            cv2.blur(gray.astype(float) ** 2, (3, 3))
            - cv2.blur(gray.astype(float), (3, 3)) ** 2
        )
        noise_level = np.sqrt(np.mean(local_std)) / (std_intensity + 1e-6)

        # Плотность границ
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)

        # Комплексность (энтропия)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / (hist.sum() + 1e-6)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-6))
        complexity_score = entropy / 8.0  # Нормализация

        # Оценка типа изображения
        estimated_type = self._estimate_image_type(
            gray,
            float(mean_intensity),
            float(std_intensity),
            edge_density,
            complexity_score,
        )

        return ImageCharacteristics(
            width=width,
            height=height,
            channels=channels,
            mean_intensity=float(mean_intensity),
            std_intensity=float(std_intensity),
            contrast=float(contrast),
            noise_level=noise_level,
            edge_density=edge_density,
            complexity_score=complexity_score,
            estimated_type=estimated_type,
        )

    def _estimate_image_type(
        self,
        gray: np.ndarray,
        mean_intensity: float,
        std_intensity: float,
        edge_density: float,
        complexity_score: float,
    ) -> ImageType:
        """Эвристическая оценка типа изображения"""
        # Документы: высокий контраст, мало оттенков
        if std_intensity > 80 and complexity_score < 0.6:
            return ImageType.DOCUMENT

        # Медицинские: специфичный диапазон интенсивностей
        if 50 < mean_intensity < 150 and std_intensity < 40:
            return ImageType.MEDICAL

        # Индустриальные: много четких границ
        if edge_density > 0.15:
            return ImageType.INDUSTRIAL

        # Спутниковые: высокая комплексность
        if complexity_score > 0.85:
            return ImageType.SATELLITE

        return ImageType.NATURAL

    def select_best_method(
        self,
        image: np.ndarray,
        characteristics: Optional[ImageCharacteristics] = None,
        library: Optional[str] = None,
    ) -> Tuple[str, str, Dict[str, Any], float]:
        """
        Выбор оптимального метода.

        Returns:
            method_name: Название метода
            library_name: Библиотека метода (opencv/sklearn/torch)
            params: Параметры для метода
            confidence: Уверенность выбора (0-1)
        """
        if characteristics is None:
            characteristics = self.analyze_image(image)

        if library:
            candidates = METHODS_BY_LIBRARY.get(library, {})
        else:
            candidates = self.benchmark_data

        scores = {}

        for method_name, profile in candidates.items():
            if library and profile.library != library:
                continue
            score = self._calculate_method_score(method_name, profile, characteristics)
            scores[method_name] = (score, profile)

        # Выбор лучшего метода
        if not scores:
            raise ValueError(f"No methods found for library='{library}'")

        best_method, (best_score, best_profile) = max(
            scores.items(), key=lambda x: x[1][0]
        )

        # Нормализация уверенности
        all_scores = [s for s, _ in scores.values()]
        confidence = (best_score - np.mean(all_scores)) / (np.std(all_scores) + 1e-6)
        confidence = 1 / (1 + np.exp(-confidence))  # Sigmoid

        params = self.available_methods.get(best_method, {}).get("params", {})

        return best_method, best_profile.library, params, confidence

    @track_calls
    def _calculate_method_score(
        self,
        method_name: str,
        profile: MethodProfile,
        characteristics: ImageCharacteristics,
    ) -> float:
        """Расчет интегральной оценки метода"""
        score = 0.0

        # Весовые коэффициенты в зависимости от цели
        if self.goal == SegmentationGoal.SPEED:
            weights = {"time": 0.7, "accuracy": 0.2, "memory": 0.1}
        elif self.goal == SegmentationGoal.ACCURACY:
            weights = {"time": 0.2, "accuracy": 0.7, "memory": 0.1}
        elif self.goal == SegmentationGoal.LOW_MEMORY:
            weights = {"time": 0.2, "accuracy": 0.3, "memory": 0.5}
        else:  # BALANCED
            weights = {"time": 0.33, "accuracy": 0.34, "memory": 0.33}

        # Нормализация времени (быстрее = лучше)
        max_time = max(p.avg_time_ms for p in self.benchmark_data.values())
        time_score = 1 - (profile.avg_time_ms / max_time)

        # Точность
        accuracy_score = profile.avg_iou

        # Память
        max_memory = max(p.memory_mb for p in self.benchmark_data.values())
        memory_score = 1 - (profile.memory_mb / max_memory)

        # Базовый score
        score = (
            weights["time"] * time_score
            + weights["accuracy"] * accuracy_score
            + weights["memory"] * memory_score
        )

        # Бонус за подходящий тип изображения
        if characteristics.estimated_type in profile.best_for_type:
            score *= 1.3

        # Штраф за чувствительность к шуму (если изображение зашумлено)
        if characteristics.noise_level > 0.3:
            score *= profile.robustness

        # Кастомные веса
        if method_name in self.custom_weights:
            score *= self.custom_weights[method_name]

        return score

    def segment(
        self,
        image: np.ndarray,
        auto_select: bool = True,
        method_name: Optional[str] = None,
        library: Optional[str] = None,
        return_metadata: bool = False,
    ):
        """
        Сегментация изображения.

        Args:
            image: Входное изображение
            auto_select: Автоматически выбрать метод (True) или использовать указанный
            method_name: Название метода (если auto_select=False)
            return_metadata: Вернуть дополнительную информацию

        Returns:
            mask: Маска сегментации
            metadata: Дополнительная информация (если return_metadata=True)
        """

        selected_method = ""
        selected_lib = ""
        params: Dict[str, Any] = {}
        confidence = 0.0

        if auto_select:
            # Автоматический выбор
            selected_method, selected_lib, params, confidence = self.select_best_method(
                image, library=library
            )
            print(
                f"🤖 Auto-selected: {selected_method.upper()} "
                f"(confidence: {confidence:.2f})"
            )
        else:
            if not method_name or not library:
                raise ValueError(
                    "method_name and library required when auto_select=False"
                )
            if method_name not in METHODS_BY_LIBRARY.get(library, {}):
                available = list(METHODS_BY_LIBRARY[library].keys())
                raise ValueError(
                    f"Method '{method_name}' not in library '{library}'. Available: {available}"
                )
            selected_method = method_name
            selected_lib = library
            profile = METHODS_BY_LIBRARY[library][method_name]
            params = profile.params or {}
            confidence = 1.0

        # Определение библиотеки (можно расширить логику)
        segmenter_class = self._get_segmenter_class(selected_method, selected_lib)

        # Создание сегментера
        segmenter = segmenter_class(method=selected_method, **params)

        # Выполнение сегментации
        result, mask = segmenter.segment_with_mask(image)

        if return_metadata:
            characteristics = self.analyze_image(image)
            metadata = {
                "method": selected_method,
                "library": selected_lib,
                "parameters": params,
                "confidence": confidence,
                "image_characteristics": characteristics,
            }
            return mask, metadata

        return mask

    def _get_segmenter_class(self, method_name: str, library: str):
        """Явный выбор сегментера по библиотеке"""
        if library == "sklearn":
            from segmenters.SklearnSegmenter import SklearnSegmenter

            return SklearnSegmenter
        elif library == "torch":
            from segmenters.TorchSegmenter import TorchSegmenter  # если есть

            return TorchSegmenter
        else:  # opencv или по умолчанию
            from segmenters.OpenCVSegmenter import OpenCVSegmenter

            return OpenCVSegmenter

    def get_recommendations(
        self, image: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Получить топ-K рекомендаций методов.

        Returns:
            Список словарей с информацией о методах
        """
        characteristics = self.analyze_image(image)
        scores = {}

        for method_name, profile in self.benchmark_data.items():
            score = self._calculate_method_score(method_name, profile, characteristics)
            scores[method_name] = {
                "score": score,
                "profile": profile,
                "params": self.available_methods[method_name]["params"],
            }

        # Сортировка
        sorted_methods = sorted(
            scores.items(), key=lambda x: x[1]["score"], reverse=True
        )

        recommendations: List[Dict[str, Any]] = []
        for method_name, data in sorted_methods[:top_k]:
            recommendations.append(
                {
                    "rank": len(recommendations) + 1,
                    "method": method_name,
                    "score": data["score"],
                    "estimated_time_ms": data["profile"].avg_time_ms,
                    "estimated_iou": data["profile"].avg_iou,
                    "parameters": data["params"],
                }
            )

        return recommendations


__all__ = ["AutoSegmenter", "SegmentationGoal", "ImageType", "METHODS_BY_LIBRARY"]
