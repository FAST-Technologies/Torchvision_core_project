# testing/TorchImplementationValidator.py

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
import os
import time
import inspect
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union, Type, Set, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch

# Локальные импорты
from segmenters.TorchSegmenter import TorchSegmenter
from segmenters.NewTorchSegmenter import TorchSegmenter2
from segmenters.SklearnSegmenter import SklearnSegmenter
from segmenters.OpenCVSegmenter import OpenCVSegmenter
from metrics.SegmentationMetrics import SegmentationMetrics

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
MethodConfig = Tuple[str, Dict[str, Any]]
MaskArray = np.ndarray  # Binary mask: HxW, dtype uint8/bool
ImageArray = np.ndarray  # RGB/Grayscale: HxW or HxWxC
ImageInput = Union[str, np.ndarray, Image.Image]
MetricDict = Dict[str, float]
PathLike = Union[str, Path]
SegmenterClass = Type[
    Union[TorchSegmenter, TorchSegmenter2, SklearnSegmenter, OpenCVSegmenter]
]
ValidationStatus = Literal["PASS", "WARNING", "FAIL"]


# ──────────────────────────────────────────────────────────────────────
class TorchImplementationValidator:
    """
    Класс для валидации кастомных PyTorch-реализаций методов сегментации
    против эталонных реализаций из библиотек (OpenCV, Scikit-learn).

    Поддерживает как TorchSegmenter, так и TorchSegmenter2 (с параметрами
    точности, компиляции и отладки).

    Основные возможности:
    - Попарное сравнение масок от TorchSegmenter vs SklearnSegmenter/OpenCVSegmenter.
    - Расчёт метрик соответствия: IoU, Dice, F1, Precision, Recall, MAE, Hausdorff.
    - Автоматическая классификация результатов: PASS / WARNING / FAIL на основе порогов.
    - Визуализация различий: оригинал, две маски, heatmap разности.
    - Генерация сводных отчётов: TXT, CSV, JSON, PNG-графики.
    - Поддержка пакетной валидации всех категорий методов (пороговые, граничные, кластеризация и т.д.).

    Workflow:
    1. Создать экземпляр → 2. Вызвать `validate_all_methods(image_path)` → 3. Получить Dict с результатами
       → 4. Сгенерировать отчёт через `generate_validation_report()` или `generate_benchmark_report_from_validation()`.

    Attributes:
        output_dir (Path): Директория для сохранения результатов валидации.
        validation_results (Dict[str, Any]): Кэш последних результатов.
        threshold_methods, edge_methods, ... (List[MethodConfig]): Предустановленные конфигурации методов по категориям.
        success_thresholds (Dict[str, float]): Пороговые значения метрик для классификации статуса валидации.
    """

    def __init__(self, output_dir: str = "./data/validation_results") -> None:
        """
        Инициализация валидатора с настройками путей и порогов успешности.

        Args:
            output_dir: Базовая директория для сохранения артефактов (маски, метрики, графики).
        """
        self.output_dir: str = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.validation_results: Dict[str, Any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ──────────────────────────────────────────────────────────────
        # КОНФИГУРАЦИИ МЕТОДОВ ПО КАТЕГОРИЯМ
        # ──────────────────────────────────────────────────────────────
        self.threshold_methods: List[MethodConfig] = [
            ("global_thresholding", {"threshold": 0.5}),
            ("otsu_thresholding", {}),
            ("adaptive_thresholding", {"block_size": 11, "C": 2}),
            ("threshold_niblack", {"window_size": 15, "k": -0.2}),
            ("threshold_sauvola", {"window_size": 15, "k": 0.5, "r": 128}),
            ("threshold_bernsen", {"window_size": 15, "contrast_threshold": 0.15}),
            (
                "threshold_phansalkar",
                {
                    "window_size": 15,
                    "k": 0.25,  # чувствительность
                    "r": 128.0,  # динамический диапазон [0, 255]
                    "m": 0.5,  # смещение
                },
            ),
            ("threshold_kittler_illingworth", {"num_bins": 256}),
            ("threshold_entropy_kapur", {"num_bins": 256}),
            ("threshold_triangle", {"num_bins": 256}),
            ("threshold_multi_otsu", {"n_thresholds": 2}),
            ("threshold_percentile", {"percentile": 90}),
            (
                "threshold_local_contrast",
                {"window_size": 15, "contrast_factor": 0.1},
            ),
        ]

        self.edge_methods: List[MethodConfig] = [
            ("sobel_edge", {"threshold": 0.1}),
            ("canny_edge", {"low": 0.1, "high": 0.3, "sigma": 1.0}),
            ("prewitt_edge", {"threshold": 0.1}),
            ("scharr_edge", {"threshold": 0.1}),
            ("roberts_cross_edge", {"threshold": 0.1}),
            ("log_edge", {"sigma": 1.0, "threshold": 0.01}),
            ("dog_edge", {"sigma1": 1.0, "sigma2": 2.0, "threshold": 0.01}),
            ("marr_hildreth_edge", {"sigma": 1.5, "threshold": 0.01}),
            (
                "gradient_magnitude_direction",
                {"threshold": 0.1},
            ),
            (
                "phase_congruency_edge",
                {
                    "nscales": 4,
                    "norientations": 4,
                    "min_wavelength": 3,
                    "mult": 2.0,
                    "sigma_onf": 0.55,
                    "k_noise": 2.0,
                    "threshold": 0.5,
                },
            ),
        ]

        self.region_methods: List[MethodConfig] = [
            ("region_growing", {"tolerance": 0.1}),
            ("split_and_merge", {"min_size": 50, "threshold": 20}),
            ("floodfill", {"tolerance": 0.15}),
        ]

        self.clastering_methods: List[MethodConfig] = [
            ("kmeans_segmentation", {"k": 3}),
            ("dbscan_segmentation", {"eps": 0.1, "min_samples": 10}),
            (
                "meanshift",
                {
                    "bandwidth": 0.5,
                    "spatial_radius": 35,
                    "color_radius": 60,
                    "max_level": 1,
                },
            ),
        ]

        self.active_contour_methods: List[MethodConfig] = [
            (
                "active_contour",
                {
                    "alpha": 0.015,
                    "beta": 10,
                    "gamma": 0.001,
                    "max_iterations": 2000,
                    "w_edge": 1,
                    "w_line": 0,
                },
            ),
            ("gvf_contour", {"mu": 0.1, "iterations": 50}),
            (
                "morphological_snakes",
                {"iterations": 100, "smoothing": 1, "threshold": 0.5},
            ),
            (
                "chan_vese",
                {
                    "mu": 0.25,
                    "lambda1": 1.0,
                    "lambda2": 1.0,
                    "tol": 1e-3,
                    "max_iter": 100,
                    "dt": 0.5,
                    "eps": 1.0,
                    "init_level_set": "checkerboard",
                },
            ),
        ]

        self.watershed_methods: List[MethodConfig] = [
            ("watershed", {}),
            (
                "random_walker",
                {"beta": 130, "tol": 1e-3, "max_iter": 300, "target_label": 2},
            ),
        ]

        self.super_pixel_methods: List[MethodConfig] = [
            # ("quickshift", {'kernel_size': 5, 'max_dist': 10, 'ratio': 1.0, 'sigma': 0.0, 'convert2lab': True}),
            (
                "slic",
                {
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
            ),
            ("felzenszwalb", {"scale": 100, "sigma": 0.5, "min_size": 50}),
        ]

        self.interactive_methods: List[MethodConfig] = [
            ("grabcut", {"num_iterations": 5}),
        ]

        # ──────────────────────────────────────────────────────────────
        # ПОРОГИ УСПЕШНОСТИ ВАЛИДАЦИИ
        # ──────────────────────────────────────────────────────────────
        # Разные реализации одного алгоритма дают близкие, но не идентичные результаты
        # из-за различий в padding, численной точности и порядка операций.
        self.success_thresholds: Dict[str, float] = {
            "iou": 0.80,  # IoU > 0.80
            "dice": 0.85,  # Dice > 0.85
            "pixel_accuracy": 0.90,  # Pixel Accuracy > 0.90
            "precision": 0.80,  # Precision > 0.80
            "recall": 0.80,  # Recall > 0.80
            "f1_score": 0.82,  # F1 Score > 0.82
            "mae": 0.15,  # MAE < 0.15
        }

    @staticmethod
    def _filter_params(
        segmenter_class: SegmenterClass, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Фильтрует параметры, оставляя только поддерживаемые сигнатурой __init__ класса.

        Предотвращает ошибки TypeError при передаче параметров, специфичных для
        TorchSegmenter2 (precision, use_compile, etc.) в другие классы.

        Args:
            segmenter_class: Класс сегментера для проверки.
            params: Исходный словарь параметров.

        Returns:
            Dict[str, Any]: Отфильтрованный словарь параметров.
        """
        if not hasattr(segmenter_class, "__init__"):
            return params

        try:
            sig = inspect.signature(segmenter_class.__init__)
            valid_params: Set[str] = set(sig.parameters.keys()) - {"self", "kwargs", "kwds"}
            # Всегда разрешаем 'postprocess' для совместимости
            valid_params.add("postprocess")
            return {k: v for k, v in params.items() if k in valid_params}
        except (ValueError, TypeError):
            # Если не удалось получить сигнатуру, возвращаем как есть
            return params

    @staticmethod
    def _normalize_mask(mask: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Приводит маску к единому формату для сравнения: (H, W), dtype uint8, значения {0, 255}.

        Обрабатывает:
        - torch.Tensor → numpy
        - Разные размерности: (1,1,H,W), (1,H,W), (H,W,1) → (H,W)
        - Нормализованные значения [0,1] → [0,255]
        - Булевы массивы → uint8

        Args:
            mask: Входная маска (Tensor или ndarray).

        Returns:
            np.ndarray: Нормализованная бинарная маска.
        """
        # Конвертация Tensor → numpy
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().detach().cpu().numpy()

        # Удаление лишних измерений
        if mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            elif mask.shape[0] == 1:
                mask = mask.squeeze(0)

        # Конвертация в uint8
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.dtype in [np.float32, np.float64]:
            if mask.max() <= 1.0 + 1e-6:  # Нормализованная [0,1]
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
        elif mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        return mask

    @staticmethod
    def _prepare_torch_params(
        params: Dict[str, Any], use_torch2: bool = False
    ) -> Dict[str, Any]:
        """
        Подготавливает параметры для TorchSegmenter/TorchSegmenter2.

        Для TorchSegmenter2:
        - Добавляет дефолтные значения для параметров точности и компиляции.
        - Отключает torch.compile для стабильности тестов (опционально).

        Args:
            params: Исходные параметры метода.
            use_torch2: Если True, применяет настройки для TorchSegmenter2.

        Returns:
            Dict[str, Any]: Подготовленные параметры.
        """
        if not use_torch2:
            return params

        prepared = params.copy()
        # Дефолтные параметры для TorchSegmenter2
        prepared.setdefault("precision", "fp32")
        prepared.setdefault("use_compile", False)  # Отключаем для стабильности
        prepared.setdefault("debug_mode", False)
        prepared.setdefault("device", None)  # Авто-выбор устройства

        # Удаляем параметры, которые могут конфликтовать
        prepared.pop("postprocess", None)  # postprocess обрабатывается отдельно

        return prepared

    # ──────────────────────────────────────────────────────────────────────
    def _load_image(self, image_path: ImageInput) -> ImageArray:
        """
        Универсальная загрузка изображения для всех сегментаторов.

        Args:
            image_path: Путь к файлу, `np.ndarray` или `PIL.Image`.

        Returns:
            np.ndarray: Изображение в формате RGB (H×W×3), dtype uint8.

        Raises:
            ValueError: Если тип входных данных не поддерживается.
        """
        if isinstance(image_path, str) and Path(image_path).exists():
            return np.array(Image.open(image_path).convert("RGB"))
        elif isinstance(image_path, np.ndarray):
            return image_path
        elif isinstance(image_path, Image.Image):
            return np.array(image_path.convert("RGB"))
        else:
            raise ValueError(f"Неподдерживаемый тип изображения: {type(image_path)}")

    # ──────────────────────────────────────────────────────────────────────
    def validate_segmentation_methods(
        self,
        image_path: ImageInput,
        methods_list: List[MethodConfig],
        first_segmenter_class: SegmenterClass,
        second_segmenter_class: SegmenterClass,
        first_method_name: str = "Torch",
        second_method_name: str = "Reference",
        status_message: str = "ВАЛИДАЦИЯ ПОРОГОВЫХ МЕТОДОВ",
        prefix: str = "threshold_validation",
        validation_type: str = "threshold",
        use_first_method_features: bool = False,
    ) -> Dict[str, Any]:
        """
        Универсальная функция валидации методов сегментации против эталонной реализации.

        Для каждого метода из `methods_list`:
        1. Запускает сегментацию через `torch_segmenter_class` и `reference_segmenter_class`.
        2. Замеряет время выполнения обоих методов.
        3. Рассчитывает метрики соответствия масок через `SegmentationMetrics`.
        4. Определяет статус валидации (PASS / WARNING / FAIL) на основе порогов.
        5. Сохраняет маски, метрики и визуализации.

        Args:
            image_path: Входное изображение (путь, массив или PIL-объект).
            methods_list: Список кортежей `(имя_метода, параметры)` для тестирования.
            torch_segmenter_class: Класс сегментера для тестируемой PyTorch-реализации.
            reference_segmenter_class: Класс сегментера для эталонной реализации.
            reference: Название референсной библиотеки (для отчётов: "sklearn", "opencv").
            status_message: Заголовок для вывода в консоль.
            prefix: Префикс для имён файлов и директорий.
            validation_type: Тип валидации для категоризации в отчётах ("threshold", "edge", ...).
            use_torch2_features: Если True, применяет настройки для TorchSegmenter2.

        Returns:
            Dict[str, Any]: Словарь результатов по методам:
            ```python
            {
                method_name: {
                    "torch_mask": np.ndarray,
                    "reference_mask": np.ndarray,
                    "metrics": Dict[str, float],  # IoU, Dice, F1, ...
                    "parameters": Dict[str, Any],
                    "validation_status": "PASS" | "WARNING" | "FAIL",
                    "success": bool,
                    "reference_library": str,
                    "first_method_time": float,
                    "second_method_time": float,
                    "methods_time_difference": float,
                }
            }
            ```

        Note:
            - Для референсного сегментера автоматически добавляется параметр `postprocess=False`,
              чтобы сравнение было максимально честным (без постобработки).
            - Метрики рассчитываются между двумя предсказаниями, а не против ground truth —
              это проверка консистентности реализаций, а не качества сегментации.
        """
        print(
            f"\n{'=' * 60}\n{status_message}\nСравнение: {first_method_name} vs {second_method_name}\n{'=' * 60}"
        )
        results: Dict[str, Any] = {}
        img_array: ImageArray = self._load_image(image_path)

        is_first_torch2 = (
            first_segmenter_class == TorchSegmenter2
        ) or use_first_method_features

        for method_name, params in methods_list:
            print(f"\n📊 Метод: {method_name}")
            try:
                # ──────────────────────────────────────────────────────
                # Torch / тестируемая реализация
                # ──────────────────────────────────────────────────────
                first_params = self._filter_params(first_segmenter_class, params.copy())
                if is_first_torch2:
                    first_params = self._prepare_torch_params(
                        first_params, use_torch2=True
                    )
                if str(self.device) == "cuda":
                    torch.cuda.synchronize()
                start_method_1_time: float = time.perf_counter()
                segmenter1 = first_segmenter_class(method=method_name, **params)
                mask1_raw = segmenter1.segment(img_array, **params)
                if str(self.device) == "cuda":
                    torch.cuda.synchronize()
                execution_method_1_time: float = (
                    time.perf_counter() - start_method_1_time
                )
                mask1: MaskArray = self._normalize_mask(mask1_raw)

                # ──────────────────────────────────────────────────────
                # Референсная реализация
                # ──────────────────────────────────────────────────────
                ref_params: Dict[str, Any] = self._filter_params(
                    second_segmenter_class, params.copy()
                )
                ref_params["postprocess"] = False
                if str(self.device) == "cuda":
                    torch.cuda.synchronize()
                start_method_2_time: float = time.perf_counter()
                segmenter2 = second_segmenter_class(method=method_name, **ref_params)
                mask2_raw = segmenter2.segment(img_array, **ref_params)
                if str(self.device) == "cuda":
                    torch.cuda.synchronize()
                execution_method_2_time: float = (
                    time.perf_counter() - start_method_2_time
                )

                mask2: MaskArray = self._normalize_mask(mask2_raw)
                difference_methods_time: float = abs(
                    execution_method_2_time - execution_method_1_time
                )

                # ──────────────────────────────────────────────────────
                # Метрики соответствия
                # ──────────────────────────────────────────────────────
                metrics: MetricDict = SegmentationMetrics.calculate_all_metrics(
                    mask1, mask2, threshold=0.5, include_hausdorff=True
                )
                metrics.update(
                    {
                        "first_method_time": execution_method_1_time,
                        "second_method_time": execution_method_2_time,
                        "methods_time_difference": difference_methods_time,
                        "first_method_coverage": float(
                            np.sum(mask1 > 0) / mask1.size * 100
                        ),
                        "second_method_coverage": float(
                            np.sum(mask2 > 0) / mask2.size * 100
                        ),
                    }
                )

                # ──────────────────────────────────────────────────────
                # Статус валидации
                # ──────────────────────────────────────────────────────
                validation_status: ValidationStatus = self._check_validation_status(
                    metrics
                )

                results[method_name] = {
                    "first_method_mask": mask1,
                    "second_method_mask": mask2,
                    "metrics": metrics,
                    "parameters": params,
                    "validation_status": validation_status,
                    "success": True,
                    "first_method_name": first_method_name,
                    "second_method_name": second_method_name,
                    "first_method_time": execution_method_1_time,
                    "second_method_time": execution_method_2_time,
                    "methods_time_difference": difference_methods_time,
                    "is_first_torch2": is_first_torch2,
                }

                # ──────────────────────────────────────────────────────
                # Вывод в консоль
                # ──────────────────────────────────────────────────────
                status_icon = "✅" if validation_status == "PASS" else "⚠️"
                print(f"   {status_icon} IoU: {metrics['iou']:.4f}")
                print(f"   {status_icon} Dice: {metrics['dice']:.4f}")
                print(f"   {status_icon} Precision: {metrics['precision']:.4f}")
                print(f"   {status_icon} Recall: {metrics['recall']:.4f}")
                print(f"   {status_icon} F1-Score: {metrics['f1_score']:.4f}")
                print(f"   {status_icon} MAE: {metrics['mae']:.4f}")
                print(
                    f"   {status_icon} Pixel Accuracy: {metrics['pixel_accuracy']:.4f}"
                )
                print(f"   Hausdorf distance: {metrics['hausdorff_distance']:.4f}")

                print(f"   Predicted Area: {metrics['predicted_area']:.4f}")
                print(f"   Ground Truth Area: {metrics['ground_truth_area']:.4f}")
                print(f"   Area Difference: {metrics['area_difference']:.4f}")
                print(f"   Статус: {validation_status}")

                print(f"    ✅ Время первого метода {execution_method_1_time:.3f}s")
                print(f"    ✅ Время второго метода {execution_method_2_time:.3f}s")
                print(f"    ✅ Разница по времени {difference_methods_time:.3f}s")
                if is_first_torch2:
                    print(f"    🔧 Использован TorchSegmenter2")

            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                traceback.print_exc()
                results[method_name] = {
                    "success": False,
                    "error": str(e),
                }
        # ──────────────────────────────────────────────────────
        # Сохранение и визуализация
        # ──────────────────────────────────────────────────────
        self._save_validation_results(
            results,
            prefix,
            second_method_name,
            first_method_name=first_method_name,
            second_method_name=second_method_name,
        )
        self._visualize_validation(
            results,
            img_array,
            validation_type,
            first_method_name=first_method_name,
            second_method_name=second_method_name,
        )
        return results

    # ──────────────────────────────────────────────────────────────────────
    def _check_validation_status(self, metrics: MetricDict) -> ValidationStatus:
        """
        Определяет статус валидации на основе пороговых значений метрик.

        Логика:
        - Проверяет 7 ключевых метрик: IoU, Dice, Pixel Accuracy, Precision, Recall, F1, MAE.
        - PASS: все 7 метрик проходят пороги.
        - WARNING: ≥ 4 из 7 метрик проходят.
        - FAIL: < 4 метрик проходят.

        Args:
            metrics: Словарь с рассчитанными метриками.

        Returns:
            ValidationStatus: Один из "PASS", "WARNING", "FAIL".
        """
        passed: int = 0
        total: int = 7

        if metrics["iou"] >= self.success_thresholds["iou"]:
            passed += 1
        if metrics["dice"] >= self.success_thresholds["dice"]:
            passed += 1
        if metrics["pixel_accuracy"] >= self.success_thresholds["pixel_accuracy"]:
            passed += 1
        if metrics["precision"] >= self.success_thresholds["precision"]:
            passed += 1
        if metrics["recall"] >= self.success_thresholds["recall"]:
            passed += 1
        if metrics["f1_score"] >= self.success_thresholds["f1_score"]:
            passed += 1
        if metrics["mae"] <= self.success_thresholds["mae"]:
            passed += 1

        if passed == total:
            return "PASS"
        elif passed >= total // 2:
            return "WARNING"
        else:
            return "FAIL"

    # ──────────────────────────────────────────────────────────────────────
    def _save_validation_results(
        self,
        results: Dict[str, Any],
        prefix: str,
        reference: str,
        first_method_name: str = "Torch",
        second_method_name: str = "Reference",
    ) -> None:
        """
        Сохраняет результаты валидации: маски (.npy) и метрики (.txt).

        Структура директории:
        ```
        {output_dir}/{prefix}_{reference}_{timestamp}/
        └── {method_name}/
            ├── torch_mask.npy (или opencv_mask.npy)
            ├── reference_mask.npy
            └── metrics.txt
        ```

        Args:
            results: Результаты `validate_segmentation_methods()`.
            prefix: Префикс имени директории.
            reference: Название референсной библиотеки.
            flag_torch: Если `True`, сохраняет маску как `torch_mask.npy`, иначе `opencv_mask.npy`.
        """
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir: str = os.path.join(
            self.output_dir, f"{prefix}_{reference}_{timestamp}"
        )
        os.makedirs(results_dir, exist_ok=True)

        # Сохраняем маски и метрики
        for method, data in results.items():
            if data.get("success"):
                method_dir: str = os.path.join(results_dir, method)
                os.makedirs(method_dir, exist_ok=True)

                first_mask = data["first_method_mask"]
                second_mask = data["second_method_mask"]
                np.save(
                    os.path.join(method_dir, f"{first_method_name.lower()}_mask.npy"),
                    first_mask,
                )
                np.save(
                    os.path.join(method_dir, f"{second_method_name.lower()}_mask.npy"),
                    second_mask,
                )

                # Метрики
                metrics: Dict[str, Any] = data["metrics"]
                metrics_path: str = os.path.join(method_dir, "metrics.txt")
                with open(metrics_path, "w", encoding="utf-8") as f:
                    f.write(f"Результаты валидации: {method}\n")
                    f.write(f"Референсная библиотека: {reference}\n")
                    f.write(f"Статус: {data['validation_status']}\n")
                    if data.get("is_first_torch2"):
                        f.write("Использован: TorchSegmenter2\n")
                    f.write("=" * 50 + "\n")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.6f}\n")
                        else:
                            f.write(f"{key}: {value}\n")

        print(f"\n💾 Результаты сохранены: {results_dir}")

    # ──────────────────────────────────────────────────────────────────────
    def _visualize_validation(
        self,
        results: Dict[str, Any],
        image_array: ImageArray,
        validation_type: str,
        first_method_name: str = "Method A",
        second_method_name: str = "Method B",
    ) -> None:
        """
        Строит визуализацию сравнения: оригинал, две маски, heatmap разности.

        Макет (на строку метода):
        [Original] | [Tested Mask] | [Reference Mask] | [Difference Heatmap]

        Args:
            results: Результаты валидации.
            image_array: Исходное изображение для отображения.
            validation_type: Категория методов ("threshold", "edge", ...) для заголовка.
            reference: Название референсной библиотеки.
            additional_method: Идентификатор тестируемой реализации.
        """
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        original: ImageArray = image_array

        n_methods: int = len([r for r in results.values() if r.get("success")])
        if n_methods == 0:
            print("⚠️ Нет успешных результатов для визуализации")
            return

        fig, axes = plt.subplots(n_methods, 4, figsize=(20, 5 * n_methods))
        if n_methods == 1:
            axes = axes.reshape(1, -1)

        row: int = 0
        for method, data in results.items():
            if not data.get("success"):
                continue

            mask_a = data["first_method_mask"]
            mask_b = data["second_method_mask"]
            mask_a_np = np.squeeze(mask_a)
            mask_b_np = np.squeeze(mask_b)

            metrics: Dict[str, Any] = data["metrics"]
            status: str = data["validation_status"]

            # Оригинальное изображение
            axes[row, 0].imshow(original)
            axes[row, 0].set_title("Original Image")
            axes[row, 0].axis("off")

            # Torch маска
            axes[row, 1].imshow(mask_a_np, cmap="gray")
            axes[row, 1].set_title(
                f"{method}\n{first_method_name.upper()}\nIoU: {metrics['iou']:.3f}",
                fontsize=8
            )
            axes[row, 1].axis("off")

            # Reference маска
            axes[row, 2].imshow(mask_b_np, cmap="gray")
            axes[row, 2].set_title(
                f"{method}\n{second_method_name.upper()}",
                fontsize=8
            )
            axes[row, 2].axis("off")

            # Heatmap разности
            diff = np.abs(mask_a_np.astype(float) - mask_b_np.astype(float))
            im = axes[row, 3].imshow(diff, cmap="hot")
            status_color: str = {
                "PASS": "green",
                "WARNING": "orange",
                "FAIL": "red",
            }.get(status, "red")
            axes[row, 3].set_title(f"Difference\nStatus: {status}", color=status_color)
            axes[row, 3].axis("off")
            plt.colorbar(im, ax=axes[row, 3], fraction=0.046)
            row += 1

        plt.suptitle(
            f"{validation_type.title()}: {first_method_name} vs {second_method_name}",
            fontsize=14,
        )
        plt.tight_layout()

        viz_path = os.path.join(
            self.output_dir,
            f"{validation_type}_{first_method_name}_vs_{second_method_name}_{timestamp}.jpg",
        )
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"📊 Визуализация: {viz_path}")

    # ──────────────────────────────────────────────────────────────────────
    def generate_validation_report(self, all_results: Dict[str, Any]) -> str:
        """
        Генерирует текстовый сводный отчёт по всем типам валидации.

        Включает:
        - Статистику PASS / WARNING / FAIL по категориям методов.
        - Детальные метрики для каждого метода.
        - Сводные проценты успешности.

        Args:
            all_results: Результаты `validate_all_methods()` — Dict по категориям.

        Returns:
            str: Текст отчёта (также сохраняется в файл).
        """
        report_lines: List[str] = []
        report_lines.append("=" * 60)
        report_lines.append("ОТЧЁТ ПО ВАЛИДАЦИИ TORCH РЕАЛИЗАЦИЙ")
        report_lines.append("=" * 60)
        report_lines.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        total_methods: int = 0
        passed_methods: int = 0
        warning_methods: int = 0
        failed_methods: int = 0
        torch2_count: int = 0

        for validation_type, results in all_results.items():
            report_lines.append(f"\n{validation_type.upper()}")
            report_lines.append("-" * 40)

            for method, data in results.items():
                if not data.get("success"):
                    continue

                total_methods += 1
                if data.get("is_first_torch2"):
                    torch2_count += 1
                status: str = data["validation_status"]

                if status == "PASS":
                    passed_methods += 1
                    icon = "✅"
                elif status == "WARNING":
                    warning_methods += 1
                    icon = "⚠️"
                else:
                    failed_methods += 1
                    icon = "❌"

                metrics: Dict[str, Any] = data["metrics"]
                torch2_tag = " [T2]" if data.get("is_first_torch2") else ""
                report_lines.append(
                    f"{icon} {method}{torch2_tag}: "
                    f"Accuracy={metrics['accuracy']:.3f}, "
                    f"IoU={metrics['iou']:.3f}, "
                    f"Dice={metrics['dice']:.3f}, "
                    f"Precision={metrics['precision']:.3f}, "
                    f"Recall={metrics['recall']:.3f}, "
                    f"F1_Score={metrics['f1_score']:.3f}, "
                    f"Pixel_accuracy={metrics['pixel_accuracy']:.3f} "
                    f"MAE={metrics['mae']:.3f} "
                    f"Hausdorf_distance={metrics.get('hausdorff_distance', float('nan')):.3f} "
                    f"Area_ratio={metrics['area_ratio']:.3f} "
                    f"Area_difference={metrics['area_difference']:.3f} "
                    f"T1={metrics['first_method_time']:.3f}s, T2={metrics['second_method_time']:.3f}s, "
                    f"ΔT={metrics['methods_time_difference']:.3f}s [{status}]"
                )

        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("СВОДНАЯ СТАТИСТИКА")
        report_lines.append("=" * 60)
        report_lines.append(f"Всего методов: {total_methods}")
        if torch2_count > 0:
            report_lines.append(f"Использован TorchSegmenter2: {torch2_count} методов")

        if total_methods > 0:
            report_lines.append(
                f"✅ PASS: {passed_methods} ({passed_methods / total_methods * 100:.2f}%)"
            )
            report_lines.append(
                f"⚠️ WARNING: {warning_methods} ({warning_methods / total_methods * 100:.2f}%)"
            )
            report_lines.append(
                f"❌ FAIL: {failed_methods} ({failed_methods / total_methods * 100:.2f}%)"
            )
        else:
            report_lines.append("⚠️ Нет данных для статистики (все методы не прошли)")
        report_lines.append("=" * 60)
        report: str = "\n".join(report_lines)
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path: str = os.path.join(
            self.output_dir, f"validation_report_{timestamp}.txt"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n📄 Отчёт сохранён: {report_path}")
        print("\n" + report)
        return report

    # ──────────────────────────────────────────────────────────────────────
    def validate_all_methods(
        self,
        image_path: ImageInput,
        use_torch2: bool = False,
        torch2_precision: str = "fp32",
    ) -> Dict[str, Any]:
        """
        Запускает валидацию всех предустановленных категорий методов.

        Выполняет 6 основных конфигураций сравнения:
        1. Threshold: Torch vs Sklearn
        2. Threshold: Torch vs OpenCV
        3. Threshold: OpenCV vs Sklearn
        4. Edge: Torch vs Sklearn
        5. Edge: Torch vs OpenCV
        6. Edge: OpenCV vs Sklearn

        Остальные категории (region, clustering, active_contour, ...) закомментированы
        и могут быть активированы при необходимости.

        Args:
            image_path: Входное изображение для всех тестов.
            use_torch2: Если True, использует TorchSegmenter2 вместо TorchSegmenter.
            torch2_precision: Точность вычислений для TorchSegmenter2 ('fp32', 'fp16', 'bf16').

        Returns:
            Dict[str, Any]: Результаты по ключам:
            `"threshold_sklearn"`, `"threshold_opencv"`, `"edge_sklearn"`, ...
        """
        all_results: Dict[str, Any] = {}
        torch_class = TorchSegmenter2 if use_torch2 else TorchSegmenter

        def _prepare_torch_params_with_precision(
            params: Dict[str, Any], precision: str
        ) -> Dict[str, Any]:
            prepared = params.copy()
            prepared.setdefault("precision", precision)
            prepared.setdefault("use_compile", False)
            prepared.setdefault("debug_mode", False)
            prepared.setdefault("device", None)
            prepared.pop("postprocess", None)
            return prepared

        validation_configs: List[
            Tuple[
                str,
                List[MethodConfig],
                SegmenterClass,
                SegmenterClass,
                str,
                str,
                str,
                str,
            ]
        ] = [
            # Torch vs Sklearn
            (
                "threshold_torch_vs_sklearn",
                self.threshold_methods,
                torch_class,
                SklearnSegmenter,
                "Torch2" if use_torch2 else "Torch",
                "Sklearn",
                "ВАЛИДАЦИЯ ПОРОГОВЫХ МЕТОДОВ (Torch + Sklearn)",
                "threshold",
            ),
            # Torch vs OpenCV
            (
                "threshold_torch_vs_opencv",
                self.threshold_methods,
                torch_class,
                OpenCVSegmenter,
                "Torch2" if use_torch2 else "Torch",
                "OpenCV",
                "ВАЛИДАЦИЯ ПОРОГОВЫХ МЕТОДОВ (Torch + OpenCV)",
                "threshold",
            ),
            # OpenCV vs Sklearn (не-Torch первым!)
            (
                "threshold_opencv_vs_sklearn",
                self.threshold_methods,
                OpenCVSegmenter,
                SklearnSegmenter,
                "OpenCV",
                "Sklearn",
                "ВАЛИДАЦИЯ ПОРОГОВЫХ МЕТОДОВ (Sklearn + OpenCV)",
                "threshold",
            ),
            # Edge methods
            (
                "edge_torch_vs_sklearn",
                self.edge_methods,
                torch_class,
                SklearnSegmenter,
                "Torch2" if use_torch2 else "Torch",
                "Sklearn",
                "ВАЛИДАЦИЯ ОПЕРАТОРОВ ГРАНИЦ (Torch + Sklearn)",
                "edge",
            ),
            (
                "edge_torch_vs_opencv",
                self.edge_methods,
                torch_class,
                OpenCVSegmenter,
                "Torch2" if use_torch2 else "Torch",
                "OpenCV",
                "ВАЛИДАЦИЯ ОПЕРАТОРОВ ГРАНИЦ (Torch + OpenCV)",
                "edge",
            ),
            (
                "edge_opencv_vs_sklearn",
                self.edge_methods,
                OpenCVSegmenter,
                SklearnSegmenter,
                "OpenCV",
                "Sklearn",
                "ВАЛИДАЦИЯ ОПЕРАТОРОВ ГРАНИЦ (OpenCV + Sklearn)",
                "edge",
            ),
            # ('region_sklearn', self.region_methods, torch_class, SklearnSegmenter, "ВАЛИДАЦИЯ РЕГИОНАЛЬНЫХ МЕТОДОВ (Torch + Sklearn)", 'sklearn', 'region', 'Torch'),
            # ('region_opencv', self.region_methods, torch_class, OpenCVSegmenter, "ВАЛИДАЦИЯ РЕГИОНАЛЬНЫХ МЕТОДОВ (Torch + OpenCV)", 'opencv', 'region', 'Torch'),
            # ('region_custom', self.region_methods, OpenCVSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ РЕГИОНАЛЬНЫХ МЕТОДОВ (Sklearn + OpenCV)", 'sklearn', 'region', 'OpenCV'),
            # ('clastering_sklearn', self.clastering_methods, torch_class, SklearnSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ КЛАСТЕРИЗАЦИИ (Torch + Sklearn)", 'sklearn', 'claster', 'Torch'),
            # ('clastering_opencv', self.clastering_methods, torch_class, OpenCVSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ КЛАСТЕРИЗАЦИИ (Torch + OpenCV)", 'opencv', 'claster', 'Torch'),
            # ('clastering_custom', self.clastering_methods, OpenCVSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ КЛАСТЕРИЗАЦИИ (Sklearn + OpenCV)", 'sklearn', 'claster', 'OpenCV'),
            # ('active_contour_sklearn', self.active_contour_methods, torch_class, SklearnSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ АКТИВНЫХ КОНТУРОВ (Torch + Sklearn)", 'sklearn', 'active_contour', 'Torch'),
            # ('active_contour_opencv', self.active_contour_methods, torch_class, OpenCVSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ АКТИВНЫХ КОНТУРОВ (Torch + OpenCV)", 'opencv', 'active_contour', 'Torch'),
            # ('active_contour_custom', self.active_contour_methods, OpenCVSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ АКТИВНЫХ КОНТУРОВ (Sklearn + OpenCV)", 'sklearn', 'active_contour', 'OpenCV'),
            # ('watershed_sklearn', self.watershed_methods, torch_class, SklearnSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ ВОДОРАЗДЕЛА (Torch + Sklearn)", 'sklearn', 'watershed', 'Torch'),
            # ('watershed_opencv', self.watershed_methods, torch_class, OpenCVSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ ВОДОРАЗДЕЛА (Torch + OpenCV)", 'opencv', 'watershed', 'Torch'),
            # ('watershed_custom', self.watershed_methods, OpenCVSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ МЕТОДОВ ВОДОРАЗДЕЛА (Sklearn + OpenCV)", 'sklearn', 'watershed', 'OpenCV'),
            # ('super_pixel_sklearn', self.super_pixel_methods, torch_class, SklearnSegmenter, "ВАЛИДАЦИЯ СУПЕРПИКСЕЛЬНЫХ МЕТОДОВ (Torch + Sklearn)", 'sklearn', 'super_pixel', 'Torch'),
            # ('super_pixel_opencv', self.super_pixel_methods, torch_class, OpenCVSegmenter, "ВАЛИДАЦИЯ СУПЕРПИКСЕЛЬНЫХ МЕТОДОВ (Torch + OpenCV)", 'opencv', 'super_pixel', 'Torch'),
            # ('super_pixel_custom', self.super_pixel_methods, OpenCVSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ СУПЕРПИКСЕЛЬНЫХ МЕТОДОВ (Sklearn + OpenCV)", 'sklearn', 'super_pixel', 'OpenCV'),
            # ('interactive_sklearn', self.interactive_methods, torch_class, SklearnSegmenter, "ВАЛИДАЦИЯ ИНТЕРАКТИВНЫХ МЕТОДОВ (Torch + Sklearn)", 'sklearn', 'interactive', 'Torch'),
            # ('interactive_opencv', self.interactive_methods, torch_class, OpenCVSegmenter, "ВАЛИДАЦИЯ ИНТЕРАКТИВНЫХ МЕТОДОВ (Torch + OpenCV)", 'opencv', 'interactive', 'Torch'),
            # ('interactive_custom', self.interactive_methods, OpenCVSegmenter, SklearnSegmenter, "ВАЛИДАЦИЯ ИНТЕРАКТИВНЫХ МЕТОДОВ (Sklearn + OpenCV)", 'sklearn', 'interactive', 'OpenCV'),
        ]

        # ──────────────────────────────────────────────────────
        # ПРЯМОЕ СРАВНЕНИЕ: TorchSegmenter2 vs TorchSegmenter
        # ──────────────────────────────────────────────────────
        if use_torch2:
            validation_configs.extend([
                # Threshold: Torch2 vs Torch1
                (
                    "threshold_torch2_vs_torch1",
                    self.threshold_methods,
                    TorchSegmenter2,  # first = новая версия
                    TorchSegmenter,   # second = старая версия
                    "Torch2",
                    "Torch1",
                    "ПРЯМОЕ СРАВНЕНИЕ: TorchSegmenter2 vs TorchSegmenter (пороговые)",
                    "threshold",
                ),
                # Edge: Torch2 vs Torch1
                (
                    "edge_torch2_vs_torch1",
                    self.edge_methods,
                    TorchSegmenter2,
                    TorchSegmenter,
                    "Torch2",
                    "Torch1", 
                    "ПРЯМОЕ СРАВНЕНИЕ: TorchSegmenter2 vs TorchSegmenter (граничные)",
                    "edge",
                ),
                # Можно добавить и для других категорий при необходимости
            ])

        for (
            key,
            methods,
            base_class,
            ref_class,
            first_name,
            second_name,
            message,
            v_type,
        ) in validation_configs:
            if use_torch2 and base_class == TorchSegmenter2:
                # Обновляем методы с параметром точности
                methods_with_precision = [
                    (
                        name,
                        _prepare_torch_params_with_precision(params, torch2_precision),
                    )
                    for name, params in methods
                ]
            else:
                methods_with_precision = methods
            all_results[key] = self.validate_segmentation_methods(
                image_path=image_path,
                methods_list=methods_with_precision,
                first_segmenter_class=base_class,
                second_segmenter_class=ref_class,
                first_method_name=first_name,
                second_method_name=second_name,
                status_message=message,
                prefix=f"{v_type}_validation",
                validation_type=v_type,
                use_first_method_features=(
                    use_torch2 and base_class == TorchSegmenter2
                ),
            )
        return all_results

    # ──────────────────────────────────────────────────────────────────────
    def generate_benchmark_report_from_validation(
        self, all_results: Dict[str, Any], output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Генерирует бенчмарк-отчёт на основе результатов валидации.

        Агрегирует данные по всем типам валидации, рассчитывает:
        - Среднее время выполнения по библиотекам.
        - Распределение метрик качества (IoU, Dice, ...).
        - Покрытие масок (% пикселей объекта).
        - Статистику PASS / WARNING / FAIL.

        Args:
            all_results: Результаты `validate_all_methods()`.
            output_dir: Директория для сохранения графиков и CSV. Если `None`, создаётся внутри `self.output_dir`.

        Returns:
            pd.DataFrame: Сводная таблица с колонками:
            `method`, `validation_type`, `torch_time`, `reference_time`, `iou`, `dice`, ..., `validation_status`.
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "benchmark_from_validation")
        os.makedirs(output_dir, exist_ok=True)

        benchmark_data: List[Dict[str, Any]] = []

        # Извлекаем данные из всех типов валидации
        for validation_type, results in all_results.items():
            for method_name, data in results.items():
                if not data.get("success", False):
                    continue

                row: Dict[str, Any] = {
                    "method": method_name,
                    "validation_type": validation_type,
                    "first_method_name": data.get("first_method_name", "Unknown"),
                    "second_method_name": data.get("second_method_name", "Unknown"),
                    "first_method_time": data.get("metrics", {}).get(
                        "first_method_time", 0
                    ),
                    "second_method_time": data.get("metrics", {}).get(
                        "second_method_time", 0
                    ),
                    "time_difference": data.get("methods_time_difference", 0),
                    "first_method_coverage": data.get("metrics", {}).get(
                        "first_method_coverage", 0
                    ),
                    "second_method_coverage": data.get("metrics", {}).get(
                        "second_method_coverage", 0
                    ),
                    "is_first_torch2": data.get("is_first_torch2", False),
                }

                # Добавляем метрики если есть
                if "metrics" in data:
                    metrics: Dict[str, Any] = data["metrics"]
                    row.update(
                        {
                            "iou": metrics.get("iou", 0),
                            "dice": metrics.get("dice", 0),
                            "pixel_accuracy": metrics.get("pixel_accuracy", 0),
                            "precision": metrics.get("precision", 0),
                            "recall": metrics.get("recall", 0),
                            "f1_score": metrics.get("f1_score", 0),
                            "mae": metrics.get("mae", 0),
                            "validation_status": data.get(
                                "validation_status", "UNKNOWN"
                            ),
                        }
                    )

                    # Вычисляем площадь покрытия из масок
                    if (
                        "first_method_mask" in data
                        and data["first_method_mask"] is not None
                    ):
                        torch_mask = data["first_method_mask"]
                        if isinstance(torch_mask, torch.Tensor):
                            torch_mask = torch_mask.cpu().numpy()
                        mask_area: int = np.sum(torch_mask > 0)
                        total_pixels: int = torch_mask.size
                        row["first_method_coverage"] = (mask_area / total_pixels) * 100

                    if (
                        "second_method_mask" in data
                        and data["second_method_mask"] is not None
                    ):
                        ref_mask = data["second_method_mask"]
                        if isinstance(ref_mask, torch.Tensor):
                            ref_mask = ref_mask.cpu().numpy()
                        mask_area = np.sum(ref_mask > 0)
                        total_pixels = ref_mask.size
                        row["second_method_coverage"] = (mask_area / total_pixels) * 100

                benchmark_data.append(row)

        df: pd.DataFrame = pd.DataFrame(benchmark_data)

        if not df.empty:
            # Сохраняем raw данные
            df.to_csv(
                os.path.join(output_dir, "benchmark_validation_data.csv"), index=False
            )

            # Строим графики
            for config in df["validation_type"].unique():
                df_config = df[df["validation_type"] == config]
                first_name = df_config["first_method_name"].iloc[0]
                second_name = df_config["second_method_name"].iloc[0]
                config_dir = os.path.join(output_dir, f"charts_{config}")
                os.makedirs(config_dir, exist_ok=True)
                self._plot_validation_benchmark_charts(
                    df_config,
                    config_dir,
                    first_method_label=first_name,
                    second_method_label=second_name,
                )

            # Генерируем текстовый отчет
            self._generate_validation_benchmark_summary(df, output_dir)

            print(f"📊 Бенчмарк-отчет сохранен в: {output_dir}")

        return df

    # ──────────────────────────────────────────────────────────────────────
    def _plot_validation_benchmark_charts(
        self,
        df: pd.DataFrame,
        output_dir: str,
        first_method_label: str = "First Method",
        second_method_label: str = "Second Method",
    ) -> None:
        """
        Строит 6 типов графиков для бенчмарк-анализа:
        1. Bar-чарт времени выполнения (Torch).
        2. Scatter: Torch time vs Reference time.
        3. Bar-чарт IoU с цветовой индикацией статуса.
        4. Покрытие масок (%).
        5. Heatmap метрик качества.
        6. Trade-off: время vs IoU.

        Args:
            df: DataFrame с агрегированными данными.
            output_dir: Директория для сохранения графиков.
            first_method_label: Подпись для первой библиотеки на графиках.
            second_method_label: Подпись для второй библиотеки.
        """
        charts_dir: str = os.path.join(output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)

        # ──────────────────────────────────────────────────────
        # График 1: Время выполнения (Torch)
        # ──────────────────────────────────────────────────────
        plt.figure(figsize=(14, 8))

        # Группируем по методам и берем среднее время
        if "first_method_time" in df.columns:
            df_sorted = df.sort_values("first_method_time", ascending=True)

            # Динамический размер фигуры
            n_methods: int = len(df_sorted)
            fig_height: float = max(8, n_methods * 0.35)
            plt.figure(figsize=(14, fig_height))

            bars = plt.barh(
                df_sorted["method"],
                df_sorted["first_method_time"],
                color="steelblue",
                edgecolor="black",
                linewidth=0.5,
            )

            plt.xlabel(f"Время выполнения ({first_method_label})", fontsize=11)
            plt.title(f"Производительность: {first_method_label}", fontsize=12)

            # Добавляем подписи значений
            max_time: float = df_sorted["first_method_time"].max()
            for bar, time_val in zip(bars, df_sorted["first_method_time"]):
                plt.text(
                    time_val + max_time * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{time_val:.3f}s",
                    va="center",
                    fontsize=8,
                )

            plt.grid(axis="x", alpha=0.3, linestyle="--")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(
                os.path.join(charts_dir, f"{first_method_label.lower()}_time.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

        # ──────────────────────────────────────────────────────
        # График 2: Scatter Torch vs Reference time
        # ──────────────────────────────────────────────────────
        if "first_method_time" in df.columns and "second_method_time" in df.columns:
            # Берем только методы где есть оба времени
            df_compare: pd.DataFrame = df[
                (df["first_method_time"] > 0) & (df["second_method_time"] > 0)
            ].copy()

            if not df_compare.empty:
                plt.figure(figsize=(12, 8))

                # Scatter plot
                plt.scatter(
                    df_compare["first_method_time"],
                    df_compare["second_method_time"],
                    s=100,
                    alpha=0.7,
                    c="blue",
                    edgecolors="black",
                    linewidth=0.5,
                )

                # Линия y=x для сравнения
                max_t = max(
                    df_compare["first_method_time"].max(),
                    df_compare["second_method_time"].max(),
                )
                plt.plot(
                    [0, max_t],
                    [0, max_t],
                    "r--",
                    linewidth=2,
                    label="y=x (равная скорость)",
                )

                plt.xlabel(f"{first_method_label} время (с)", fontsize=11)
                plt.ylabel(f"{second_method_label} время (с)", fontsize=11)
                plt.title(
                    f"Сравнение скорости: {first_method_label} vs {second_method_label}"
                )
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Подписываем выбросы
                for _, row in df_compare.iterrows():
                    ratio = (
                        row["first_method_time"] / row["second_method_time"]
                        if row["second_method_time"] > 0
                        else 0
                    )
                    if ratio > 2 or ratio < 0.5:  # Только значительные отклонения
                        plt.annotate(
                            row["method"][:20],
                            (row["first_method_time"], row["second_method_time"]),
                            fontsize=7,
                            alpha=0.8,
                        )

                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        charts_dir,
                        f"{first_method_label.lower()}_vs_{second_method_label.lower()}_time.png",
                    ),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()

        # ──────────────────────────────────────────────────────
        # График 3: IoU по методам
        # ──────────────────────────────────────────────────────
        if "iou" in df.columns:
            df_iou: pd.DataFrame = df[df["iou"] > 0].sort_values("iou", ascending=True)

            if not df_iou.empty:
                plt.figure(figsize=(14, 8))
                n_methods = len(df_iou)
                fig_height = max(8, n_methods * 0.35)
                plt.figure(figsize=(14, fig_height))
                colors = [
                    {"PASS": "green", "WARNING": "orange", "FAIL": "red"}.get(
                        row.get("validation_status", "UNKNOWN"), "gray"
                    )
                    for _, row in df_iou.iterrows()
                ]

                bars = plt.barh(
                    df_iou["method"],
                    df_iou["iou"],
                    color=colors,
                    edgecolor="black",
                    linewidth=0.5,
                )

                plt.xlabel("IoU Score", fontsize=11)
                plt.title(
                    "Качество сегментации (IoU) по методам\n"
                    "🟢 PASS | 🟠 WARNING | 🔴 FAIL",
                    fontsize=12,
                    pad=20,
                )

                # Добавляем пороговую линию
                if "iou" in self.success_thresholds:
                    plt.axvline(
                        x=self.success_thresholds["iou"],
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        alpha=0.7,
                        label=f"Threshold ({self.success_thresholds['iou']})",
                    )

                plt.legend(loc="lower right")
                plt.grid(axis="x", alpha=0.3, linestyle="--")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(charts_dir, "iou_comparison.png"),
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close()

        # ──────────────────────────────────────────────────────
        # График 4: Покрытие масок (Coverage)
        # ──────────────────────────────────────────────────────
        if "first_method_coverage" in df.columns:
            df_coverage: pd.DataFrame = df.sort_values(
                "first_method_coverage", ascending=True
            )

            plt.figure(figsize=(14, 8))
            n_methods = len(df_coverage)
            fig_height = max(8, n_methods * 0.35)
            plt.figure(figsize=(14, fig_height))

            bars = plt.barh(
                df_coverage["method"],
                df_coverage["first_method_coverage"],
                color="teal",
                edgecolor="black",
                linewidth=0.5,
            )

            plt.xlabel("Покрытие маски (%)", fontsize=11)
            plt.title(f"Площадь покрытия: {first_method_label}")

            # Добавляем подписи
            max_cov = df_coverage["first_method_coverage"].max()
            for bar, cov in zip(bars, df_coverage["first_method_coverage"]):
                plt.text(
                    cov + max_cov * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{cov:.1f}%",
                    va="center",
                    fontsize=8,
                )

            plt.grid(axis="x", alpha=0.3, linestyle="--")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(
                os.path.join(charts_dir, f"{first_method_label.lower()}_coverage.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

        # ──────────────────────────────────────────────────────
        # График 5: Heatmap метрик
        # ──────────────────────────────────────────────────────
        metric_cols: List[str] = ["iou", "dice", "precision", "recall", "f1_score"]
        available_metrics: List[str] = [m for m in metric_cols if m in df.columns]

        if len(available_metrics) >= 2:
            # Берем top-20 методов по времени
            df_top: pd.DataFrame = (
                df.nlargest(20, "first_method_time") if len(df) > 20 else df
            )

            plt.figure(figsize=(12, 10))

            # Подготавливаем данные для heatmap
            heatmap_data: pd.DataFrame = df_top[["method"] + available_metrics].copy()
            heatmap_data.set_index("method", inplace=True)

            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn",
                center=0.5,
                linewidths=0.5,
                cbar_kws={"label": "Score"},
            )

            plt.title("Матрица метрик качества по методам", fontsize=12, pad=15)
            plt.xticks(rotation=45, ha="right", fontsize=9)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            plt.savefig(
                os.path.join(charts_dir, "metrics_heatmap.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        # ──────────────────────────────────────────────────────
        # График 6: Trade-off время vs IoU
        # ──────────────────────────────────────────────────────
        if "first_method_time" in df.columns and "iou" in df.columns:
            df_tradeoff: pd.Series = df[(df["first_method_time"] > 0) & (df["iou"] > 0)]

            if not df_tradeoff.empty:
                plt.figure(figsize=(12, 8))

                scatter = plt.scatter(
                    df_tradeoff["first_method_time"],
                    df_tradeoff["iou"],
                    s=100,
                    alpha=0.7,
                    c=df_tradeoff["first_method_time"],
                    cmap="RdYlGn_r",
                    edgecolors="black",
                    linewidth=0.5,
                )

                plt.xlabel("Время выполнения (сек)", fontsize=11)
                plt.ylabel("IoU Score", fontsize=11)
                plt.title("Trade-off: Скорость vs Точность", fontsize=12, pad=15)
                plt.colorbar(scatter, label="Время (сек)")
                plt.grid(True, alpha=0.3)

                # Подписываем лучшие компромиссы
                for _, row in df_tradeoff.iterrows():
                    # Подписываем top-5 по IoU и top-5 по скорости
                    if (
                        row["iou"] in df_tradeoff["iou"].nlargest(5).values
                        or row["first_method_time"]
                        in df_tradeoff["first_method_time"].nsmallest(5).values
                    ):
                        plt.annotate(
                            row["method"][:20],
                            (row["first_method_time"], row["iou"]),
                            fontsize=7,
                            bbox=dict(
                                boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5
                            ),
                        )

                plt.tight_layout()
                plt.savefig(
                    os.path.join(charts_dir, "time_vs_iou_tradeoff.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()

        print(f"📈 Графики бенчмарка сохранены в: {charts_dir}")

    # ──────────────────────────────────────────────────────────────────────
    def _generate_validation_benchmark_summary(
        self, df: pd.DataFrame, output_dir: str
    ) -> None:
        """
        Генерирует текстовый сводный отчёт по бенчмарку.

        Включает:
        - Топ-10 самых быстрых и точных методов.
        - Лучшие компромиссы (IoU / время).
        - Статистику по библиотекам и статусам.
        - Анализ покрытия масок.

        Args:
            df: DataFrame с агрегированными данными.
            output_dir: Директория для сохранения `benchmark_summary.txt`.
        """
        summary_path: str = os.path.join(output_dir, "benchmark_summary.txt")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("БЕНЧМАРК МЕТОДОВ СЕГМЕНТАЦИИ (на основе валидации)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Всего методов: {len(df)}\n\n")

            torch2_count = df.get("is_first_torch2", pd.Series([False] * len(df))).sum()
            if torch2_count > 0:
                f.write(f"Использован TorchSegmenter2: {torch2_count} методов\n")
            f.write("\n")

            # ===== Быстрые методы =====
            f.write(
                "=" * 80 + "\nТОП-10 САМЫХ БЫСТРЫХ МЕТОДОВ (Torch)\n" + "=" * 80 + "\n"
            )
            if "first_method_time" in df.columns:
                df_fast: pd.DataFrame = df.nsmallest(10, "first_method_time")
                for i, (_, row) in enumerate(df_fast.iterrows(), 1):
                    t2_tag = " [T2]" if row.get("is_first_torch2") else ""
                    f.write(
                        f"{i:2d}. {row['method']:<35}{t2_tag} {row['first_method_time']:.4f}s\n"
                    )
            f.write("\n")

            # ===== Точные методы =====
            if "iou" in df.columns:
                f.write(
                    "=" * 80
                    + "\nТОП-10 САМЫХ ТОЧНЫХ МЕТОДОВ (по IoU)\n"
                    + "=" * 80
                    + "\n"
                )
                df_accurate: pd.DataFrame = df.nlargest(10, "iou")
                for i, (_, row) in enumerate(df_accurate.iterrows(), 1):
                    t2_tag = " [T2]" if row.get("is_first_torch2") else ""
                    f.write(
                        f"{i:2d}. {row['method']:<40}{t2_tag} IoU: {row['iou']:.4f}\n"
                    )
                f.write("\n")

            # ===== Лучшие компромиссы =====
            if "first_method_time" in df.columns and "iou" in df.columns:
                f.write(
                    "=" * 80
                    + "\nЛУЧШИЕ КОМПРОМИССЫ (скорость × точность)\n"
                    + "=" * 80
                    + "\n"
                )
                # Вычисляем score = IoU / time (чем больше, тем лучше)
                df_compromise: pd.DataFrame = df.copy()
                df_compromise["efficiency_score"] = df_compromise["iou"] / (
                    df_compromise["first_method_time"] + 0.001
                )
                df_best: pd.DataFrame = df_compromise.nlargest(10, "efficiency_score")
                for i, (_, row) in enumerate(df_best.iterrows(), 1):
                    t2_tag = " [T2]" if row.get("is_first_torch2") else ""
                    f.write(
                        f"{i:2d}. {row['method']:<35}{t2_tag} IoU: {row['iou']:.4f}, "
                        f"Time: {row['first_method_time']:.4f}s, "
                        f"Efficiency: {row['efficiency_score']:.2f}\n"
                    )
                f.write("\n")

            # ===== Статистика по библиотекам =====
            if "reference_library" in df.columns:
                f.write("=" * 80 + "\nСРАВНЕНИЕ ПО БИБЛИОТЕКАМ\n" + "=" * 80 + "\n")
                for lib in df["reference_library"].unique():
                    df_lib: pd.DataFrame = df[df["reference_library"] == lib]
                    f.write(f"\n{lib.upper()}:\n")
                    f.write(f"  Количество методов: {len(df_lib)}\n")
                    if "first_method_time" in df_lib.columns:
                        f.write(
                            f"  Среднее время: {df_lib['first_method_time'].mean():.4f}s\n"
                        )
                        f.write(
                            f"  Min время: {df_lib['first_method_time'].min():.4f}s\n"
                        )
                        f.write(
                            f"  Max время: {df_lib['first_method_time'].max():.4f}s\n"
                        )
                    if "iou" in df_lib.columns:
                        f.write(f"  Средний IoU: {df_lib['iou'].mean():.4f}\n")
                f.write("\n")

            # ===== Распределение по статусам =====
            if "validation_status" in df.columns:
                f.write(
                    "=" * 80
                    + "\nРАСПРЕДЕЛЕНИЕ ПО СТАТУСАМ ВАЛИДАЦИИ\n"
                    + "=" * 80
                    + "\n"
                )
                status_counts: pd.Series = df["validation_status"].value_counts()
                for status, count in status_counts.items():
                    percentage: float = (count / len(df)) * 100
                    f.write(f"{status:<10}: {count:3d} методов ({percentage:.1f}%)\n")
                f.write("\n")

            # ===== Покрытие масок =====
            if "first_method_coverage" in df.columns:
                f.write("=" * 80 + "\nСТАТИСТИКА ПОКРЫТИЯ МАСОК\n" + "=" * 80 + "\n")
                f.write(
                    f"Среднее покрытие: {df['first_method_coverage'].mean():.2f}%\n"
                )
                f.write(f"Min покрытие: {df['first_method_coverage'].min():.2f}%\n")
                f.write(f"Max покрытие: {df['first_method_coverage'].max():.2f}%\n")

                # Методы с аномальным покрытием
                df_extreme: pd.Series = df[
                    (df["first_method_coverage"] < 5)
                    | (df["first_method_coverage"] > 95)
                ]
                if not df_extreme.empty:
                    f.write("\nМетоды с экстремальным покрытием (<5% или >95%):\n")
                    for _, row in df_extreme.iterrows():
                        t2_tag = " [T2]" if row.get("is_first_torch2") else ""
                        f.write(
                            f"  - {row['method']}{t2_tag}: {row['first_method_coverage']:.2f}%\n"
                        )
                f.write("\n")

        print(f"📄 Текстовый отчет сохранен: {summary_path}")
