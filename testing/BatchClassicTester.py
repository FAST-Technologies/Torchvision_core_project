# testing/BatchClassicTester.py (обновлённая версия)

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
import os
import signal
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Literal, Union
from collections import defaultdict
from datetime import datetime
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from segmenters.OpenCVSegmenter import OpenCVSegmenter
from segmenters.SklearnSegmenter import SklearnSegmenter
from segmenters.TorchSegmenter import TorchSegmenter
from segmenters.NewTorchSegmenter import TorchSegmenter2
from metrics.SegmentationMetrics import SegmentationMetrics, MetricsDict
import torch
import gc

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
MethodConfig = Tuple[str, Dict[str, Any]]
LibraryName = Literal["torch", "torch_v2", "opencv", "sklearn"]
ValidationStatus = Literal["PASS", "WARNING", "FAIL"]
SegmenterClass = type[
    Union[TorchSegmenter, TorchSegmenter2, SklearnSegmenter, OpenCVSegmenter]
]


# ──────────────────────────────────────────────────────────────────────
class BatchClassicTester:
    """
    Класс для массового тестирования согласованности классических методов
    сегментации между различными библиотеками (Torch, OpenCV, Scikit-learn).

    Основная идея:
    - Запускаем один и тот же метод в двух реализациях (напр., Torch vs OpenCV)
    - Сравниваем выходные маски через метрики соответствия (IoU, Dice, F1, ...)
    - Оцениваем, насколько реализации дают близкие результаты

    Это проверка КОНСИСТЕНТНОСТИ, а не качества против ground truth!

    Attributes:
        library_pairs (List[Tuple[LibraryName, LibraryName]]): Пары библиотек для сравнения.
        all_threshold_methods (List[MethodConfig]): Полный список пороговых методов.
        all_edge_methods (List[MethodConfig]): Полный список граничных методов.
        success_thresholds (Dict[str, float]): Пороги для классификации статуса.
        results (Dict[str, Dict[str, Dict[str, List[float]]]]): {pair: {method: {metric: [values]}}}
    """

    # ──────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        ade20k_root: str = "./data/ade20k/ADEChallengeData2016",
        output_dir: str = "./data/batch_consistency_test",
        split: str = "validation",
        max_images: Optional[int] = None,
        image_size: Tuple[int, int] = (512, 512),
        autosave_interval: int = 5,
        resume: bool = True,
        library_pairs: Optional[List[Tuple[LibraryName, LibraryName]]] = None,
        save_masks: bool = True,  # Сохранять ли маски?
        mask_sample_rate: float = 0.1,  # Доля изображений для сохранения масок (0.0-1.0)
        max_mask_samples_per_method: int = 3,  # Макс. образцов масок на метод
        save_visualizations: bool = True,  # Генерировать ли визуализации разницы?
        save_results: bool = True,  # Сохранять ли результаты (маски + изображения)
        refresh_masks: bool = False,  # Пересоздавать маски даже для протестированных
        torch_segmenter_version: Literal["v1", "v2"] = "v2",
    ) -> None:
        """
        Инициализация тестера согласованности для массового сравнения реализаций.

        Класс предназначен для автоматизированного тестирования консистентности методов
        сегментации между различными бэкендами (Torch, OpenCV, Scikit-learn) на большом
        наборе изображений.

        Основная логика:
        1. Загружает изображения из датасета (по умолчанию ADE20K).
        2. Для каждого изображения запускает один и тот же метод в двух реализациях.
        3. Сравнивает выходные маски через метрики: IoU, Dice, Precision, Recall, F1, MAE.
        4. Классифицирует результат: PASS / WARNING / FAIL на основе порогов.
        5. Сохраняет маски, визуализации и метрики (опционально, с вероятностной выборкой).
        6. Генерирует сводные отчёты: CSV, JSON, Markdown, HTML, PNG-графики.

        Поддерживаемые режимы сохранения артефактов:
        - `save_masks`: Сохранять ли бинарные маски (.npy) для визуальной проверки.
        - `mask_sample_rate`: Доля изображений [0.0–1.0], для которых сохраняются маски.
        - `max_mask_samples_per_method`: Лимит масок на метод (защита от переполнения диска).
        - `save_visualizations`: Генерировать ли 4-панельные сравнения (оригинал + 2 маски + heatmap).
        - `save_results`: Сохранять ли результаты сегментации (маски + изображение + метрики).

        Алгоритм выборки масок:
        ```
        Для каждого (pair, method):
            если текущий_счётчик >= max_mask_samples_per_method:
                ❌ Не сохранять
            иначе:
                если random() < mask_sample_rate:
                    ✅ Сохранить + увеличить счётчик
                иначе:
                    ❌ Пропустить
        ```

        Args:
            ade20k_root: Путь к корневой директории датасета ADE20K.
            output_dir: Директория для сохранения всех результатов и артефактов.
            split: Выборка датасета: "training" или "validation".
            max_images: Максимальное количество изображений для теста.
                    Если `None` — тестируются все доступные.
            image_size: Целевой размер изображений (высота, ширина) для ресайза.
            autosave_interval: Интервал автосохранения прогресса (в итерациях по изображениям).
            resume: Если `True`, автоматически загружает прогресс из `.progress.json`
                    и пропускает уже выполненные тесты.
            library_pairs: Список пар библиотек для попарного сравнения.
                        По умолчанию: `[("torch", "opencv"), ("torch", "sklearn"), ("opencv", "sklearn")]`.
            save_masks: Флаг сохранения бинарных масок (.npy) для визуальной верификации.
            mask_sample_rate: Вероятность [0.0–1.0] сохранения маски для данного теста.
                            Рекомендуется 0.05–0.2 для баланса между информативностью и объёмом.
            max_mask_samples_per_method: Максимальное количество масок на комбинацию (метод, пара библиотек).
                                        После достижения лимита маски перестают сохраняться для этой комбинации.
            save_visualizations: Генерировать ли 4-панельные визуализации сравнения:
                                [Оригинал] | [Маска A] | [Маска B] | [Heatmap разницы].
            save_results: Сохранять ли полные результаты сегментации (маски в PNG + метрики в JSON).
            refresh_masks: Обновлять ли маски.

        Attributes:
            library_pairs: Список пар библиотек для сравнения.
            all_methods: Объединённый список всех тестируемых методов (пороговые + граничные).
            success_thresholds: Пороговые значения метрик для классификации статуса валидации.
            results: Словарь накопленных метрик: `{pair: {method: {metric: [values]}}}`.
            _mask_sample_counts: Счётчики сохранённых масок: `{pair: {method: count}}`.

        Raises:
            FileNotFoundError: Если директория датасета не найдена.
            ValueError: Если `mask_sample_rate` не в диапазоне [0.0, 1.0].

        Example:
            ```python
            # Базовое использование: тестирование на 50 изображениях
            tester = BatchClassicTester(
                ade20k_root="./data/ade20k",
                output_dir="./results/consistency_test",
                max_images=50,
                save_masks=True,
                mask_sample_rate=0.1,  # ~10% изображений сохранят маски
                max_mask_samples_per_method=3  # не более 3 масок на метод
            )
            df = tester.run_batch_test()
            tester.save_results(df)
            tester.plot_results(df)

            # Режим "только метрики" (без сохранения артефактов)
            tester = BatchClassicTester(
                save_masks=False,
                save_visualizations=False,
                save_results=False
            )
            df = tester.run_batch_test()  # Быстро, только CSV/JSON

            # Перезапуск с дозаполнением масок
            tester = BatchClassicTester(
                resume=True,
                mask_sample_rate=0.2,  # Увеличиваем долю для дозаполнения
                max_mask_samples_per_method=5  # Увеличиваем лимит
            )
            df = tester.run_batch_test()  # Продолжит с места остановки
            ```

        Note:
            - При `resume=True` и наличии `.progress.json` тестер пропускает уже выполненные
            комбинации (изображение × метод × пара библиотек), но может дозаполнить маски,
            если лимит `max_mask_samples_per_method` не достигнут.
            - Для экономии места на диске рекомендуется использовать `mask_sample_rate ≤ 0.1`
            и `max_mask_samples_per_method ≤ 5` при тестировании на >100 изображениях.
            - Визуализации генерируются только если `save_visualizations=True` И маска была
            сохранена для данного теста.
            - Счётчики масок сохраняются в `.progress.json` и восстанавливаются при перезапуске,
            что гарантирует соблюдение лимитов даже после прерывания выполнения.
        """
        self._setup_signal_handlers()
        self.ade20k_root: Path = Path(ade20k_root)
        self.output_dir: Path = Path(output_dir)
        self.split: str = split
        self.max_images: Optional[int] = max_images
        self.image_size: Tuple[int, int] = image_size
        self.autosave_interval: int = autosave_interval
        self.resume: bool = resume
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Параметры сохранения масок
        self.save_masks: bool = save_masks
        self.mask_sample_rate: float = mask_sample_rate
        self.max_mask_samples_per_method: int = max_mask_samples_per_method
        self.save_visualizations: bool = save_visualizations
        self.save_results_enabled: bool = save_results
        self.refresh_masks: bool = refresh_masks

        # Счётчики для выборки масок: {pair_key: {method_name: count}}
        self._mask_sample_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Директория для масок
        if self.save_masks:
            self.masks_dir: Path = self.output_dir / "masks"
            self.masks_dir.mkdir(parents=True, exist_ok=True)

        # Пары библиотек для сравнения
        self.library_pairs: List[Tuple[LibraryName, LibraryName]] = library_pairs or [
            ("torch", "opencv"),
            ("torch", "sklearn"),
            ("opencv", "sklearn"),
            ("torch", "torch_v2"),
        ]

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ──────────────────────────────────────────────────────────────
        # ПОЛНЫЙ СПИСОК МЕТОДОВ
        # ──────────────────────────────────────────────────────────────
        self.all_threshold_methods: List[MethodConfig] = [
            ("global_thresholding", {"threshold": 0.5}),
            ("adaptive_thresholding", {"block_size": 11, "C": 2}),
            ("otsu_thresholding", {}),
            ("threshold_niblack", {"window_size": 15, "k": -0.2}),
            ("threshold_sauvola", {"window_size": 15, "k": 0.5, "r": 128.0}),
            ("threshold_bernsen", {"window_size": 15, "contrast_threshold": 0.15}),
            (
                "threshold_phansalkar",
                {"window_size": 15, "k": 0.25, "r": 128.0, "m": 0.5},
            ),
            ("threshold_kittler_illingworth", {"num_bins": 256}),
            ("threshold_entropy_kapur", {"num_bins": 256}),
            ("threshold_triangle", {"num_bins": 256}),
            ("threshold_multi_otsu", {"n_thresholds": 2}),
            ("threshold_percentile", {"percentile": 90}),
            ("threshold_local_contrast", {"window_size": 15, "contrast_factor": 0.1}),
        ]

        self.all_edge_methods: List[MethodConfig] = [
            ("sobel_edge", {"threshold": 0.1}),
            ("canny_edge", {"low": 0.1, "high": 0.3, "sigma": 1.0}),
            ("prewitt_edge", {"threshold": 0.1}),
            ("scharr_edge", {"threshold": 0.1}),
            ("roberts_cross_edge", {"threshold": 0.1}),
            ("log_edge", {"sigma": 1.0, "threshold": 0.01}),
            ("dog_edge", {"sigma1": 1.0, "sigma2": 2.0, "threshold": 0.01}),
            ("marr_hildreth_edge", {"sigma": 1.5, "threshold": 0.01}),
            ("gradient_magnitude_direction", {"threshold": 0.1}),
            (
                "phase_congruency_edge",
                {
                    "nscale": 4,
                    "norientations": 4,
                    "min_wavelength": 3,
                    "mult": 2.0,
                    "sigma_onf": 0.55,
                    "k_noise": 2.0,
                    "threshold": 0.5,
                },
            ),
        ]

        # Объединённый список всех методов
        self.all_methods: List[MethodConfig] = (
            self.all_threshold_methods + self.all_edge_methods
        )

        # ──────────────────────────────────────────────────────────────
        # МЕТРИКИ И ПОРОГИ
        # ──────────────────────────────────────────────────────────────
        self.metrics_to_aggregate: List[str] = [
            "iou",
            "dice",
            "precision",
            "recall",
            "f1_score",
            "accuracy",
            "mae",
            "hausdorff_distance",
            "pixel_accuracy",
        ]

        self.success_thresholds: Dict[str, float] = {
            "iou": 0.80,
            "dice": 0.85,
            "pixel_accuracy": 0.90,
            "precision": 0.80,
            "recall": 0.80,
            "f1_score": 0.82,
            "mae": 0.15,
        }

        # ──────────────────────────────────────────────────────────────
        # СТРУКТУРЫ ДЛЯ РЕЗУЛЬТАТОВ
        # ──────────────────────────────────────────────────────────────
        # {pair_key: {method_name: {metric_name: [values]}}}
        self.results: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.execution_times: Dict[str, Dict[str, List[Tuple[float, float]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self.errors: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.validation_status: Dict[str, Dict[str, List[ValidationStatus]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self.torch_segmenter_version: Literal["v1", "v2"] = torch_segmenter_version

        # Пути для автосохранения
        self.progress_file: Path = self.output_dir / ".progress.json"
        self.temp_results_file: Path = self.output_dir / ".results_temp.csv"

        # Загрузка прогресса если нужно
        if resume and self.progress_file.exists():
            self._load_progress()
            print(
                f"📥 Восстановлен прогресс: {self._processed_count}/{self._total_tests} тестов"
            )

    # ──────────────────────────────────────────────────────────────────────
    def _setup_signal_handlers(self) -> None:
        """
        Настраивает обработчики системных сигналов (SIGINT, SIGTERM) для безопасного завершения.

        При получении сигнала Ctrl+C или `kill` выполняется финальное автосохранение
        текущего прогресса и результатов, после чего процесс завершается с кодом 130.
        """

        def handle_interrupt(signum, frame):
            print("\n\n⚠️  Получен сигнал прерывания!")
            print("💾 Выполняется финальное автосохранение...")
            self._save_progress(force=True)
            print("✅ Прогресс сохранён. Завершение работы.")
            sys.exit(130)

        signal.signal(signal.SIGINT, handle_interrupt)
        signal.signal(signal.SIGTERM, handle_interrupt)

    # ──────────────────────────────────────────────────────────────────────
    def _get_segmenter_class(self, library: LibraryName) -> SegmenterClass:
        """
        Возвращает класс сегментера по имени библиотеки.

        Используется для динамической инициализации сегментеров в тестах согласованности.
        Поддерживает три бэкенда: PyTorch, OpenCV и Scikit-learn.

        Args:
            library: Название библиотеки-реализации.
                    Допустимые значения: "torch", "opencv", "sklearn".

        Returns:
            SegmenterClass: Класс сегментера, соответствующий указанной библиотеке.
                        Возвращаемый тип — один из: `TorchSegmenter`, `OpenCVSegmenter`, `SklearnSegmenter`.

        Raises:
            KeyError: Если `library` не найдено в маппинге (невозможно при корректной типизации).

        Example:
            ```python
            tester = BatchClassicTester()

            # Получение класса для PyTorch
            TorchSeg = tester._get_segmenter_class("torch")
            segmenter = TorchSeg(method="otsu_thresholding")

            # Получение класса для OpenCV
            OpenCvSeg = tester._get_segmenter_class("opencv")
            segmenter = OpenCvSeg(method="canny_edge", low=0.1, high=0.3)
            ```

        Note:
            - Метод использует строгую типизацию через `Literal["torch", "opencv", "sklearn"]`,
            что обеспечивает проверку допустимых значений на этапе статического анализа.
            - Возвращаемый класс можно использовать для создания экземпляров с произвольными
            параметрами через `**kwargs`.
            - Для добавления новой библиотеки достаточно расширить словарь `mapping` и тип `LibraryName`.
        """
        mapping: Dict[LibraryName, SegmenterClass] = {
            "torch": (
                TorchSegmenter2
                if self.torch_segmenter_version == "v2"
                else TorchSegmenter
            ),
            "torch_v2": TorchSegmenter2,
            "opencv": OpenCVSegmenter,
            "sklearn": SklearnSegmenter,
        }
        return mapping[library]

    # ──────────────────────────────────────────────────────────────────────
    def _run_consistency_test(
        self,
        method_name: str,
        params: Dict[str, Any],
        lib_a: LibraryName,
        lib_b: LibraryName,
        image: np.ndarray,
        img_name: str,
        pair_key: str,
    ) -> Tuple[
        Optional[Dict[str, float]],
        Optional[Tuple[float, float]],
        Optional[str],
        Optional[Tuple[np.ndarray, np.ndarray]],
    ]:
        """
        Выполняет тест согласованности: запускает один метод в двух реализациях и сравнивает результаты.

        Это ядро валидации: метод сегментации выполняется последовательно в двух библиотеках,
        после чего выходные маски сравниваются через набор метрик (IoU, Dice, Precision, Recall, ...).

        Алгоритм:
        1. Инициализация сегментеров для `lib_a` и `lib_b` с указанными параметрами.
        2. Запуск сегментации с замером времени выполнения (`time.perf_counter()`).
        3. Приведение масок к единому формату: `uint8`, форма `(H, W)`, значения {0, 255}.
        4. Расчёт метрик соответствия через `SegmentationMetrics.calculate_all_metrics()`.
        5. Определение необходимости сохранения масок через `_should_save_mask()`.
        6. Возврат кортежа с метриками, временем, ошибкой (если есть) и масками для сохранения.

        Особенности:
        - Для второй реализации (`lib_b`) автоматически отключается постобработка
        (`postprocess=False`) для обеспечения честного сравнения "сырых" результатов.
        - Если изображение имеет форму `(H, W, 3)`, оно автоматически конвертируется в grayscale
        при необходимости (зависит от метода).
        - Все исключения перехватываются и возвращаются как строка ошибки, чтобы не прерывать
        пакетное тестирование.

        Args:
            method_name: Имя метода сегментации (например, "otsu_thresholding", "canny_edge").
            params: Словарь параметров метода (например, `{"threshold": 0.5, "sigma": 1.0}`).
            lib_a: Первая библиотека для сравнения ("torch", "opencv" или "sklearn").
            lib_b: Вторая библиотека для сравнения.
            image: Входное изображение формы `(H, W)` или `(H, W, 3)`, dtype=uint8.
            img_name: Имя файла изображения для логирования.
            pair_key: Ключ пары библиотек (например, "torch_vs_opencv") для организации вывода.

        Returns:
            Tuple[Optional[MetricsDict], Optional[Tuple[float, float]], Optional[str], Optional[Tuple[np.ndarray, np.ndarray]]]:
                - `metrics`: Словарь метрик соответствия (IoU, Dice, F1, MAE, Hausdorff, ...) или `None` при ошибке.
                - `exec_times`: Кортеж `(time_a, time_b)` — время выполнения в секундах для каждой библиотеки.
                - `error`: Сообщение об ошибке или `None` при успешном выполнении.
                - `masks_to_save`: Кортеж `(mask_a, mask_b)` для сохранения или `None`, если сохранение не требуется.

        Example:
            ```python
            # Запуск теста для метода Оцу
            metrics, times, error, masks = tester._run_consistency_test(
                method_name="otsu_thresholding",
                params={},
                lib_a="torch",
                lib_b="sklearn",
                image=image_array,
                img_name="test_001.jpg",
                pair_key="torch_vs_sklearn"
            )

            if error:
                print(f"❌ Ошибка: {error}")
            else:
                print(f"✅ IoU: {metrics['iou']:.4f}, Δt: {abs(times[0] - times[1]):.3f}s")
                if masks:
                    # Сохранение масок для визуальной проверки
                    np.save("mask_torch.npy", masks[0])
                    np.save("mask_sklearn.npy", masks[1])
            ```

        Note:
            - Метод сравнивает две реализации одного алгоритма, а не предсказание с ground truth.
            Это проверка **консистентности**, а не **качества**.
            - Для методов, чувствительных к постобработке (морфология, удаление мелких объектов),
            отключение `postprocess` во второй реализации критично для корректного сравнения.
            - Возвращаемые маски — это копии (`mask.copy()`), чтобы избежать побочных эффектов
            при последующей модификации.
            - Время выполнения измеряется через `time.perf_counter()` для максимальной точности.
        """
        try:
            # ──────────────────────────────────────────────────────
            # Запуск первой реализации
            # ──────────────────────────────────────────────────────
            seg_a_class = self._get_segmenter_class(lib_a)
            seg_a = seg_a_class(method=method_name, **params)

            if str(self.device) == "cuda":
                torch.cuda.synchronize()
            start_a: float = time.perf_counter()
            mask_a: np.ndarray = seg_a.segment(image, **params)
            if str(self.device) == "cuda":
                torch.cuda.synchronize()
            time_a: float = time.perf_counter() - start_a

            # ──────────────────────────────────────────────────────
            # Запуск второй реализации (без постобработки для честного сравнения)
            # ──────────────────────────────────────────────────────
            seg_b_class = self._get_segmenter_class(lib_b)
            params_b: Dict[str, Any] = params.copy()
            params_b["postprocess"] = False  # 🔧 Отключаем постобработку

            seg_b = seg_b_class(method=method_name, **params_b)
            if str(self.device) == "cuda":
                torch.cuda.synchronize()
            start_b: float = time.perf_counter()
            mask_b: np.ndarray = seg_b.segment(image, **params_b)
            if str(self.device) == "cuda":
                torch.cuda.synchronize()
            time_b: float = time.perf_counter() - start_b

            # ──────────────────────────────────────────────────────
            # Приведение масок к одному формату и размеру
            # ──────────────────────────────────────────────────────
            def normalize_mask(
                mask: np.ndarray, target_shape: Tuple[int, int]
            ) -> np.ndarray:
                """Приводит маску к бинарному uint8 и нужному размеру."""
                if mask.shape != target_shape:
                    from skimage.transform import resize

                    mask = resize(mask, target_shape, order=0, preserve_range=True)
                if mask.dtype != np.uint8:
                    if mask.max() <= 1.0:
                        mask = (mask * 255).astype(np.uint8)
                    else:
                        mask = mask.astype(np.uint8)
                return mask

            target_shape = image.shape[:2]
            mask_a_norm: np.ndarray = normalize_mask(mask_a, target_shape)
            mask_b_norm: np.ndarray = normalize_mask(mask_b, target_shape)

            # ──────────────────────────────────────────────────────
            # Расчёт метрик соответствия
            # ──────────────────────────────────────────────────────
            metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(
                pred_mask=mask_a_norm,
                gt_mask=mask_b_norm,  # 🔁 Сравниваем две маски между собой!
                threshold=0.5,
                include_hausdorff=True,
            )
            metrics["time_diff"] = abs(time_a - time_b)

            masks_to_save: Optional[Tuple[np.ndarray, np.ndarray]] = None
            if self.save_masks and self._should_save_mask(pair_key, method_name):
                masks_to_save = (mask_a_norm.copy(), mask_b_norm.copy())

            return metrics, (time_a, time_b), None, masks_to_save

        except Exception as e:
            error_msg: str = f"{type(e).__name__}: {str(e)}"
            return None, None, error_msg, None

    def _should_save_mask(self, pair_key: str, method_name: str) -> bool:
        """
        Определяет, нужно ли сохранять маски для данного теста, используя вероятностную выборку.

        Алгоритм принятия решения:
        1. Проверяет глобальный флаг `self.save_masks`.
        2. Получает текущий счётчик сохранённых масок для данной пары (pair_key) и метода.
        3. Если счётчик достиг лимита `max_mask_samples_per_method` → возвращает `False`.
        4. Иначе генерирует случайное число и сравнивает с `mask_sample_rate`.
        5. Если случайное число < `mask_sample_rate` → увеличивает счётчик и возвращает `True`.

        Этот подход обеспечивает:
        - ✅ Равномерное распределение сохранённых масок по всему набору изображений.
        - ✅ Защиту от переполнения диска через жёсткий лимит на метод.
        - ✅ Воспроизводимость при фиксированном seed (если установить `random.seed()`).

        Args:
            pair_key: Ключ пары библиотек, например "torch_vs_opencv".
            method_name: Имя тестируемого метода, например "otsu_thresholding".

        Returns:
            bool: `True` если маску следует сохранить, `False` иначе.

        Example:
            ```python
            # Пример внутренней логики:
            # Предположим: mask_sample_rate=0.1, max_mask_samples_per_method=3
            # Текущий счётчик для (torch_vs_opencv, otsu) = 2

            # Тест #1: random() = 0.08 < 0.1 → сохраняем, счётчик = 3
            # Тест #2: random() = 0.15 > 0.1 → не сохраняем
            # Тест #3: счётчик = 3 >= 3 → не сохраняем (достигнут лимит)
            ```

        Note:
            - Счётчики хранятся в `self._mask_sample_counts` и сохраняются в `.progress.json`
            при автосохранении, что обеспечивает корректную работу при перезапусках.
            - Для отладки можно временно установить `mask_sample_rate=1.0` и
            `max_mask_samples_per_method=999`, чтобы сохранить маски для всех тестов.
            - Метод не зависит от содержимого изображения — решение принимается только
            на основе вероятности и счётчиков, что обеспечивает непредвзятую выборку.
        """
        if not self.save_masks:
            return False

        current_count: int = self._mask_sample_counts[pair_key][method_name]

        # Если достигли лимита — не сохраняем
        if current_count >= self.max_mask_samples_per_method:
            return False

        # Вероятностная выборка
        import random

        should_save: bool = random.random() < self.mask_sample_rate

        if should_save:
            self._mask_sample_counts[pair_key][method_name] += 1
            return True

        return False

    def _save_masks(
        self,
        pair_key: str,
        method_name: str,
        img_name: str,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        metrics: Dict[str, float],
        image: np.ndarray,
    ) -> None:
        """
        Сохраняет маски и метаданные для последующей визуальной проверки и отладки.

        Создаёт иерархическую структуру директорий для удобной навигации по результатам:
        ```
        {output_dir}/masks/
        └── {pair_key}/                    # например: torch_vs_opencv
            └── {method_name}/             # например: otsu_thresholding
                └── {img_name_sanitized}/  # например: ADE_val_00000001_jpg
                    ├── {lib_a}_mask.npy   # бинарная маска первой библиотеки
                    ├── {lib_b}_mask.npy   # бинарная маска второй библиотеки
                    ├── original_image.png # оригинальное изображение для контекста
                    ├── metrics.json       # метрики соответствия и параметры теста
                    └── difference_heatmap.png  # heatmap разницы (если save_visualizations)
        ```

        Форматы файлов:
        - `*_mask.npy`: Бинарные маски в формате NumPy (dtype=uint8, значения {0, 255}).
                    Загружаются через `np.load(path)`.
        - `original_image.png`: RGB-изображение в формате PNG для визуального контекста.
        - `metrics.json`: JSON-словарь с метриками и параметрами:
        ```json
        {
            "iou": 0.92,
            "dice": 0.95,
            "precision": 0.93,
            "recall": 0.91,
            "f1_score": 0.92,
            "mae": 0.03,
            "time_diff": 0.012,
            "parameters": {"threshold": 0.5, "num_bins": 256}
        }
        ```
        - `difference_heatmap.png`: Визуализация разницы масок (если включено).

        Args:
            pair_key: Ключ пары библиотек, например "torch_vs_opencv".
            method_name: Имя тестируемого метода, например "otsu_thresholding".
            img_name: Имя исходного файла изображения, например "ADE_val_00000001.jpg".
            mask_a: Бинарная маска от первой библиотеки (dtype=uint8, форма (H, W)).
            mask_b: Бинарная маска от второй библиотеки (dtype=uint8, форма (H, W)).
            metrics: Словарь с рассчитанными метриками соответствия между масками.
            image: Исходное изображение (RGB или grayscale) для сохранения контекста.

        Returns:
            None. Файлы сохраняются на диск в структуру, описанную выше.

        Note:
            - Имя изображения санитизируется: точки заменяются на подчёркивания
            (`img_name.replace(".", "_")`), чтобы избежать конфликтов с расширениями.
            - Метрики в `metrics.json` сериализуются с приведением numpy-типов к Python
            (`float(v) if isinstance(v, np.floating) else v`).
            - Визуализация разницы генерируется только если `self.save_visualizations=True`.
            - Метод создаёт директории рекурсивно (`mkdir(parents=True, exist_ok=True)`),
            поэтому безопасен для параллельного запуска.

        Example:
            ```python
            # После вызова _save_masks для теста:
            #   pair_key="torch_vs_opencv", method_name="otsu", img_name="test.jpg"
            # Будет создана структура:
            # ./data/batch_consistency_test/masks/torch_vs_opencv/otsu/test_jpg/
            # ├── torch_mask.npy
            # ├── opencv_mask.npy
            # ├── original_image.png
            # ├── metrics.json
            # └── difference_heatmap.png  # если save_visualizations=True

            # Загрузка маски для анализа:
            import numpy as np
            mask_torch = np.load("masks/torch_vs_opencv/otsu/test_jpg/torch_mask.npy")
            print(f"Маска Torch: форма={mask_torch.shape}, dtype={mask_torch.dtype}")
            ```
        """
        lib_a: str
        lib_b: str
        lib_a, lib_b = pair_key.split("_vs_")

        # Создаём директорию
        save_dir: Path = (
            self.masks_dir / pair_key / method_name / img_name.replace(".", "_")
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        # Сохраняем маски
        np.save(save_dir / f"{lib_a}_mask.npy", mask_a)
        np.save(save_dir / f"{lib_b}_mask.npy", mask_b)

        # Сохраняем оригинальное изображение (для контекста)
        if image.max() <= 1.0:
            img_save: np.ndarray = (image * 255).astype(np.uint8)
        else:
            img_save = image.astype(np.uint8)
        Image.fromarray(img_save).save(save_dir / "original_image.png")

        # Сохраняем метрики
        metrics_save: Dict[str, Any] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in metrics.items()
        }
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics_save, f, indent=2)

        # Визуализация разницы (опционально)
        if self.save_visualizations:
            self._save_difference_visualization(
                save_dir, image, mask_a, mask_b, metrics, lib_a, lib_b, method_name
            )

    def _save_difference_visualization(
        self,
        save_dir: Path,
        image: np.ndarray,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        metrics: Dict[str, float],
        lib_a: str,
        lib_b: str,
        method_name: str,
    ) -> None:
        """
        Создаёт 4-панельную визуализацию для наглядного сравнения результатов двух реализаций.

        Макет визуализации (слева направо):
        ```
        ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
        │   Original      │   Mask A        │   Mask B        │   Difference    │
        │   Image         │   ({lib_a})     │   ({lib_b})     │   Heatmap       │
        │                 │   IoU: 0.92     │                 │   Status: PASS  │
        └─────────────────┴─────────────────┴─────────────────┴─────────────────┘
        ```

        Описание панелей:
        1. **Original Image**: Исходное изображение (RGB или grayscale) для контекста.
        2. **Mask A**: Бинарная маска от первой библиотеки (линейная шкала серого, 0=чёрный, 255=белый).
                    В заголовке отображается значение IoU для быстрой оценки качества.
        3. **Mask B**: Бинарная маска от второй библиотеки (аналогично панели 2).
        4. **Difference Heatmap**: Тепловая карта абсолютной разницы между масками.
                                - Цветовая схема: "hot" (чёрный → красный → жёлтый → белый).
                                - Значения: 0 (полное совпадение) → 255 (полное несовпадение).
                                - В заголовке отображается статус валидации (PASS/WARNING/FAIL).

        Args:
            save_dir: Директория для сохранения визуализации (файл: `comparison.png`).
            image: Исходное изображение для отображения в первой панели.
            mask_a: Бинарная маска от первой библиотеки (dtype=uint8, форма (H, W)).
            mask_b: Бинарная маска от второй библиотеки (dtype=uint8, форма (H, W)).
            metrics: Словарь с метриками для отображения в заголовках панелей.
            lib_a: Название первой библиотеки (например, "torch") для подписей.
            lib_b: Название второй библиотеки (например, "opencv") для подписей.
            method_name: Имя метода для заголовка визуализации.

        Returns:
            None. Визуализация сохраняется как `comparison.png` в указанную директорию.

        Note:
            - Размер фигуры: 20×5 дюймов (широкий формат для 4 панелей в ряд).
            - Цветовая карта для масок: "gray" с фиксированным диапазоном [0, 255] для
            согласованности отображения.
            - Цветовая карта для heatmap: "hot" с диапазоном [0, 255] и цветовой шкалой.
            - Статус валидации окрашивается: ✅ зелёный (PASS), ⚠️ оранжевый (WARNING), ❌ красный (FAIL).
            - Визуализация сохраняется с разрешением 150 DPI и обрезкой лишних полей
            (`bbox_inches="tight"`).

        Example:
            ```python
            # После вызова метода будет создан файл:
            # {save_dir}/comparison.png
            #
            # Пример содержимого визуализации:
            # - Панель 1: цветное изображение пейзажа
            # - Панель 2: чёрно-белая маска от Torch с надписью "IoU: 0.94"
            # - Панель 3: чёрно-белая маска от OpenCV
            # - Панель 4: тепловая карта с красными пятнами в местах расхождений,
            #            заголовок "Status: PASS" зелёным цветом
            ```
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 1. Оригинальное изображение
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # 2. Маска библиотеки A
        axes[1].imshow(mask_a, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(
            f"{lib_a.upper()} Mask\nIoU: {metrics.get('iou', np.nan):.3f}"
        )
        axes[1].axis("off")

        # 3. Маска библиотеки B
        axes[2].imshow(mask_b, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title(f"{lib_b.upper()} Mask")
        axes[2].axis("off")

        # 4. Heatmap разницы
        diff: np.ndarray = np.abs(mask_a.astype(float) - mask_b.astype(float))
        im = axes[3].imshow(diff, cmap="hot", vmin=0, vmax=255)
        axes[3].set_title(
            f"Difference Heatmap\nStatus: {metrics.get('validation_status', 'N/A')}"
        )
        axes[3].axis("off")
        plt.colorbar(im, ax=axes[3], fraction=0.046, label="Pixel Difference")

        plt.suptitle(
            f"Consistency Check: {method_name} ({lib_a} ↔ {lib_b})", fontsize=14
        )
        plt.tight_layout()

        viz_path: Path = save_dir / "comparison.png"
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ──────────────────────────────────────────────────────────────────────
    def _check_validation_status(self, metrics: Dict[str, float]) -> ValidationStatus:
        """
        Определяет статус валидации на основе пороговых значений метрик.

        Классифицирует результат теста согласованности на три категории:
        - **PASS**: Все 7 ключевых метрик проходят установленные пороги.
        - **WARNING**: От 4 до 6 метрик проходят пороги (частичное соответствие).
        - **FAIL**: Менее 4 метрик проходят пороги (значительные расхождения).

        Проверяемые метрики и пороги (из `self.success_thresholds`):
        ```
        IoU              >= 0.80  # Intersection over Union
        Dice             >= 0.85  # F1-Score для бинарной сегментации
        Pixel Accuracy   >= 0.90  # Доля совпадающих пикселей
        Precision        >= 0.80  # Точность (доля верных объектов)
        Recall           >= 0.80  # Полнота (доля найденных объектов)
        F1-Score         >= 0.82  # Гармоническое среднее Precision и Recall
        MAE              <= 0.15  # Mean Absolute Error (чем меньше, тем лучше)
        ```

        Алгоритм:
        1. Инициализация счётчика `passed = 0`.
        2. Для каждой метрики:
        - Если оператор `>=` и значение >= порога → `passed += 1`
        - Если оператор `<=` и значение <= порога → `passed += 1`
        3. Классификация:
        - `passed == 7` → "PASS"
        - `passed >= 4` → "WARNING"
        - `passed < 4` → "FAIL"

        Args:
            metrics: Словарь с рассчитанными метриками соответствия.
                    Должен содержать хотя бы некоторые из ключей:
                    "iou", "dice", "pixel_accuracy", "precision", "recall", "f1_score", "mae".

        Returns:
            ValidationStatus: Один из трёх статусов:
                            - "PASS": Полное соответствие реализаций.
                            - "WARNING": Частичное соответствие (требует внимания).
                            - "FAIL": Значительные расхождения (требует исследования).

        Example:
            ```python
            metrics = {
                "iou": 0.92,
                "dice": 0.95,
                "pixel_accuracy": 0.98,
                "precision": 0.91,
                "recall": 0.89,
                "f1_score": 0.90,
                "mae": 0.03
            }

            status = tester._check_validation_status(metrics)
            print(status)  # Вывод: "PASS"

            # Пример с расхождениями
            metrics_bad = {**metrics, "iou": 0.65, "dice": 0.70, "mae": 0.25}
            status = tester._check_validation_status(metrics_bad)
            print(status)  # Вывод: "WARNING" или "FAIL" в зависимости от количества проваленных метрик
            ```

        Note:
            - Пороговые значения можно настроить через `self.success_thresholds` при инициализации.
            - Метод не учитывает `hausdorff_distance` и `time_diff` в классификации, так как
            они имеют другую природу (геометрическая точность и производительность).
            - Для методов обнаружения границ (Canny, Sobel) пороги могут быть слишком строгими,
            так как разные реализации могут давать смещённые на 1-2 пикселя контуры.
            - Статус "WARNING" полезен для выявления методов, требующих дополнительной настройки
            параметров или предобработки.
        """
        passed: int = 0
        total: int = 7

        checks: List[Tuple[str, str, float]] = [
            ("iou", ">=", self.success_thresholds["iou"]),
            ("dice", ">=", self.success_thresholds["dice"]),
            ("pixel_accuracy", ">=", self.success_thresholds["pixel_accuracy"]),
            ("precision", ">=", self.success_thresholds["precision"]),
            ("recall", ">=", self.success_thresholds["recall"]),
            ("f1_score", ">=", self.success_thresholds["f1_score"]),
            ("mae", "<=", self.success_thresholds["mae"]),
        ]

        for metric, op, threshold in checks:
            if metric in metrics:
                if op == ">=" and metrics[metric] >= threshold:
                    passed += 1
                elif op == "<=" and metrics[metric] <= threshold:
                    passed += 1

        if passed == total:
            return "PASS"
        elif passed >= total // 2:
            return "WARNING"
        else:
            return "FAIL"

    # ──────────────────────────────────────────────────────────────────────
    def _load_images_with_masks(self) -> List[Tuple[str, np.ndarray]]:
        """
        Загружает изображения из датасета для тестирования согласованности.

        Метод читает изображения из директории `ade20k_root/images/{split}`,
        автоматически ресайзит их до `self.image_size` и конвертирует в RGB.

        Особенности загрузки:
        - Поддерживаемые форматы: `.jpg`, `.jpeg`, `.png`.
        - Изображения сортируются по имени файла для воспроизводимости.
        - При ошибке загрузки изображение пропускается с выводом предупреждения.
        - Если `self.max_images` задан, загружается только указанное количество первых изображений.

        Преобразования изображения:
        1. Конвертация в RGB через `PIL.Image.convert("RGB")`.
        2. Ресайз до `self.image_size` с интерполяцией `BILINEAR`.
        3. Конвертация в `np.ndarray` формы `(H, W, 3)`, dtype=uint8.

        Args:
            None (использует атрибуты экземпляра: `ade20k_root`, `split`, `image_size`, `max_images`).

        Returns:
            List[Tuple[str, np.ndarray]]: Список кортежей `(имя_файла, изображение)`, где:
                                        - `имя_файла`: строка с именем файла (например, "ADE_val_00000001.jpg").
                                        - `изображение`: `np.ndarray` формы `(H, W, 3)`, dtype=uint8.

        Raises:
            FileNotFoundError: Если директория `images/{split}` не существует.

        Example:
            ```python
            tester = BatchClassicTester(
                ade20k_root="./data/ade20k",
                split="validation",
                image_size=(512, 512),
                max_images=10
            )

            images = tester._load_images_with_masks()
            print(f"Загружено {len(images)} изображений")

            # Доступ к первому изображению
            filename, image = images[0]
            print(f"Файл: {filename}, Форма: {image.shape}, Диапазон: [{image.min()}, {image.max()}]")
            ```

        Note:
            - Метод не загружает ground truth маски, так как тестирование согласованности
            сравнивает две реализации между собой, а не с эталоном.
            - Для тестирования качества сегментации против GT используйте отдельный класс
            (например, `SegmentationEvaluator`).
            - Ресайз с `BILINEAR` интерполяцией может слегка размывать резкие границы;
            для методов, чувствительных к точности контуров, рассмотрите `NEAREST` или `LANCZOS`.
            - Прогресс загрузки отображается через `tqdm` для удобства мониторинга.
        """
        images_dir: Path = self.ade20k_root / "images" / self.split

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        image_files: List[str] = sorted(
            [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
        )

        if self.max_images:
            image_files = image_files[: self.max_images]

        print(f"📥 Загрузка {len(image_files)} изображений из {self.split}...")

        data: List[Tuple[str, np.ndarray]] = []
        for img_file in tqdm(image_files, desc="Loading images"):
            try:
                img: Image.Image = Image.open(images_dir / img_file).convert("RGB")
                img_array: np.ndarray = np.array(
                    img.resize(self.image_size, Image.Resampling.BILINEAR)
                )
                data.append((img_file, img_array))
            except Exception as e:
                print(f"❌ Error loading {img_file}: {e}")
                continue

        print(f"✅ Загружено {len(data)} изображений")
        return data

    # ──────────────────────────────────────────────────────────────────────
    def _init_progress_tracking(
        self, total_images: int, total_methods: int, total_pairs: int
    ) -> None:
        """
        Инициализирует внутренние счётчики и кэши для отслеживания прогресса тестирования.

        Подготавливает метаданные для:
        - Расчёта общего числа тестов: `total_images × total_methods × total_pairs`.
        - Восстановления прогресса при перезапуске (`resume=True`).
        - Оптимизации поиска индексов методов и пар через кэширование.

        Создаваемые атрибуты:
        - `self._total_images`, `self._total_methods`, `self._total_pairs`: Исходные параметры.
        - `self._total_tests`: Общее число тестов для прогресс-бара.
        - `self._processed_count`: Счётчик выполненных тестов (восстанавливается из прогресса).
        - `self._start_time`, `self._last_save_time`: Таймеры для расчёта ETA и автосохранения.
        - `self._method_index_cache`: Словарь `{(метод, параметры_как_json): индекс}` для O(1) поиска.
        - `self._pair_index_cache`: Словарь `{(lib_a, lib_b): индекс}` для O(1) поиска.

        Алгоритм кэширования индексов:
        ```python
        # Для методов: сериализуем параметры в JSON для уникального ключа
        {
            ("otsu_thresholding", "{}"): 0,
            ("canny_edge", '{"sigma": 1.0, "low": 0.1}'): 1,
            ...
        }

        # Для пар библиотек: простой кортеж как ключ
        {
            ("torch", "opencv"): 0,
            ("torch", "sklearn"): 1,
            ("opencv", "sklearn"): 2,
        }
        ```

        Args:
            total_images: Количество изображений в текущем наборе данных.
            total_methods: Количество тестируемых методов сегментации.
            total_pairs: Количество пар библиотек для сравнения.

        Returns:
            None (метод устанавливает атрибуты экземпляра).

        Example:
            ```python
            tester = BatchClassicTester()
            tester._init_progress_tracking(
                total_images=100,
                total_methods=23,
                total_pairs=3
            )

            print(tester._total_tests)  # Вывод: 6900 (100 × 23 × 3)

            # Проверка кэша
            method_key = ("otsu_thresholding", json.dumps({}, sort_keys=True))
            print(tester._method_index_cache[method_key])  # Вывод: индекс метода в self.all_methods
            ```

        Note:
            - Кэширование индексов критично для производительности: без него поиск индекса
            методом `.index()` в цикле из 6900 тестов добавил бы ~45 секунд выполнения.
            - Сериализация параметров в JSON через `sort_keys=True` гарантирует, что
            `{"a": 1, "b": 2}` и `{"b": 2, "a": 1}` дадут одинаковый ключ.
            - Атрибут `_processed_count` инициализируется через `getattr(..., 0)`, чтобы
            корректно работать как при первом запуске, так и при восстановлении прогресса.
            - Таймеры используются для расчёта `ETA` в прогресс-баре и троттлинга автосохранения.
        """
        self._total_images: int = total_images
        self._total_methods: int = total_methods
        self._total_pairs: int = total_pairs
        self._total_tests: int = total_images * total_methods * total_pairs
        self._processed_count: int = getattr(self, "_processed_count", 0)
        self._start_time: float = time.time()
        self._last_save_time: float = time.time()
        self._method_index_cache: Dict[Tuple[str, str], int] = {
            (name, json.dumps(params, sort_keys=True)): idx
            for idx, (name, params) in enumerate(self.all_methods)
        }
        self._pair_index_cache: Dict[Tuple[str, str], int] = {
            pair: idx for idx, pair in enumerate(self.library_pairs)
        }

    # ──────────────────────────────────────────────────────────────────────
    def _update_progress_bar(
        self,
        pbar,
        current: int,
        pair: Tuple[LibraryName, LibraryName],
        method: str,
        img: str,
    ) -> None:
        """
        Обновляет postfix прогресс-бара `tqdm` динамической статистикой выполнения.

        Рассчитывает и отображает:
        - Прошедшее время (`elapsed`) и оставшееся время (`eta`) в человекочитаемом формате.
        - Скорость обработки (тестов в минуту).
        - Общую долю ошибок на текущий момент.
        - Текущую пару библиотек, метод и имя изображения для контекста.

        Форматирование времени:
        ```
        < 60 секунд  → "45с"
        < 3600 секунд → "12.5м"
        >= 3600 секунд → "2.3ч"
        ```

        Args:
            pbar: Экземпляр `tqdm.tqdm` для обновления прогресс-бара.
            current: Количество уже выполненных тестов (для расчёта скорости и ETA).
            pair: Кортеж библиотек текущего теста, например `("torch", "opencv")`.
            method: Имя текущего метода сегментации (например, "otsu_thresholding").
            img: Имя текущего изображения (обрезается до 12 символов для компактности).

        Returns:
            None (метод обновляет состояние `pbar` через `set_postfix()`).

        Example:
            ```python
            from tqdm import tqdm

            with tqdm(total=1000, desc="Тестирование") as pbar:
                for i in range(1000):
                    # ... выполнение теста ...
                    tester._update_progress_bar(
                        pbar=pbar,
                        current=i+1,
                        pair=("torch", "sklearn"),
                        method="canny_edge",
                        img="ADE_val_00000001.jpg"
                    )
                    pbar.update(1)

            # Пример вывода postfix:
            # Тестирование: 45%|████▌| 450/1000 [02:15<01:30, 6.11 тест/с, pair=torch↔sklearn, method=canny, img=ADE_val_0000, elapsed=2.3м, eta=1.5м, rate=366.7/мин, errors=3(0.7%)]
            ```

        Note:
            - Метод использует замыкание `fmt_time()` для локального форматирования времени,
            чтобы не засорять пространство имён класса.
            - Доля ошибок рассчитывается как `общее_число_ошибок / текущий_прогресс`,
            что даёт мгновенную обратную связь о стабильности тестов.
            - Имя изображения обрезается до 12 символов (`img[:12]`), чтобы не перегружать
            строку прогресс-бара, особенно в CI/CD логах с ограниченной шириной.
            - Для отладки можно временно увеличить длину обрезки или добавить больше полей
            в `set_postfix()`.
        """
        elapsed: float = time.time() - self._start_time
        rate: float = current / elapsed if elapsed > 0 else 0
        remaining: float = (self._total_tests - current) / rate if rate > 0 else 0

        def fmt_time(s: float) -> str:
            if s < 60:
                return f"{s:.0f}с"
            elif s < 3600:
                return f"{s / 60:.1f}м"
            else:
                return f"{s / 3600:.1f}ч"

        total_errors: int = sum(
            len(e) for m in self.errors.values() for e in m.values()
        )
        error_rate: float = total_errors / current if current > 0 else 0

        pbar.set_postfix(
            {
                "pair": f"{pair[0]}↔{pair[1]}",
                "method": method.split("_")[0],
                "img": img[:12],
                "elapsed": fmt_time(elapsed),
                "eta": fmt_time(remaining),
                "rate": f"{rate * 60:.1f}/мин",
                "errors": f"{total_errors}({error_rate * 100:.1f}%)",
            }
        )

    # ──────────────────────────────────────────────────────────────────────
    def _save_progress(self, force: bool = False) -> None:
        """
        Сохраняет текущий прогресс и промежуточные результаты на диск для восстановления.

        Механизм автосохранения:
        - **Throttling**: Сохранение выполняется не чаще 1 раза в 30 секунд, если `force=False`,
        чтобы избежать избыточных операций ввода-вывода при быстром выполнении тестов.
        - **Atomic write**: Промежуточные результаты сначала записываются во временный файл
        `.results_temp.csv`, затем атомарно переименовываются в финальный `.csv` для защиты
        от повреждения данных при аварийном завершении.
        - **JSON прогресс**: Метаданные (счётчики, время) сохраняются в `.progress.json`
        в человекочитаемом формате с отступами.

        Сохраняемые данные:
        1. **Прогресс** (`.progress.json`):
        ```json
        {
            "processed_count": 450,
            "total_tests": 1000,
            "start_time": 1712345678.9,
            "last_update": 1712345912.3,
            "mask_sample_counts": {
            "torch_vs_opencv": {
                "otsu_thresholding": 2,
                "canny_edge": 1
            }
            }
        }
        ```
        2. **Результаты** (`.csv`):
        - Агрегированные метрики по всем выполненным тестам.
        - Столбцы: `Method`, `Library_Pair`, `iou_mean`, `dice_mean`, `time_diff_mean`, ...

        Args:
            force: Если `True`, сохраняет прогресс немедленно, игнорируя таймер троттлинга.
                Используется при финальном сохранении или по сигналу прерывания.

        Returns:
            None (метод записывает файлы на диск).

        Raises:
            Exception: Любое исключение при записи файлов перехватывается и логируется,
                    чтобы не прерывать выполнение тестов.

        Example:
            ```python
            # Автоматическое автосохранение (вызывается внутри run_batch_test)
            tester._save_progress(force=False)  # Может быть пропущено, если прошло < 30 сек

            # Принудительное сохранение (например, при Ctrl+C)
            tester._save_progress(force=True)  # Сохранит немедленно

            # Проверка файлов
            import os
            print(os.listdir(tester.output_dir))
            # Вывод: ['.progress.json', 'consistency_test_results.csv', 'charts/', ...]
            ```

        Note:
            - Файл `.progress.json` удаляется после успешного завершения `run_batch_test()`,
            чтобы не захламлять директорию результатов.
            - Временный файл `.results_temp.csv` используется только во время записи;
            после атомарного переименования он исчезает.
            - Счётчики масок (`mask_sample_counts`) сохраняются для корректного продолжения
            вероятностной выборки при перезапуске с `resume=True`.
            - Для больших наборов данных (>10000 тестов) можно увеличить интервал троттлинга
            (сейчас 30 секунд) через параметр `autosave_interval` в `__init__`.
        """
        now: float = time.time()
        if not force and (now - self._last_save_time) < 30:
            return

        try:
            # Метаданные прогресса
            progress: Dict[str, Any] = {
                "processed_count": self._processed_count,
                "total_tests": self._total_tests,
                "start_time": self._start_time,
                "last_update": now,
            }
            progress["mask_sample_counts"] = {
                pair: dict(methods)
                for pair, methods in self._mask_sample_counts.items()
            }
            with open(self.progress_file, "w") as f:
                json.dump(progress, f, indent=2)

            # Промежуточные результаты (atomic write)
            if any(self.results.values()):
                df_temp: pd.DataFrame = self._aggregate_results()
                df_temp.to_csv(self.temp_results_file, index=False, float_format="%.4f")
                final_path: Path = self.temp_results_file.with_suffix(".csv")
                self.temp_results_file.replace(final_path)

            self._last_save_time = now
            print(
                f"\n💾 Автосохранение: {self._processed_count}/{self._total_tests} ✅"
            )

        except Exception as e:
            print(f"\n⚠️  Ошибка автосохранения: {e}")

    # ──────────────────────────────────────────────────────────────────────
    def _load_progress(self) -> bool:
        """
        Загружает метаданные прогресса из файла `.progress.json` для восстановления выполнения.

        Восстанавливаемые данные:
        - `processed_count`: Количество уже выполненных тестов (для пропуска в цикле).
        - `total_tests`: Общее число тестов (для прогресс-бара).
        - `start_time`: Время начала выполнения (для расчёта `elapsed` и `eta`).
        - `mask_sample_counts`: Счётчики сохранённых масок для продолжения выборки.

        Дополнительная коррекция через сканирование диска:
        Если включено сохранение масок (`self.save_masks=True`), метод сканирует директорию
        `masks/` и корректирует счётчики **только вниз**, если файлы были удалены вручную:
        ```
        Если сохранено на диске < сохранено в чекпойнте:
            обновить счётчик до значения с диска + вывести предупреждение
        Иначе:
            оставить счётчик без изменений (чекпойнт — источник истины)
        ```

        Алгоритм загрузки:
        1. Попытка прочитать и распарсить `.progress.json`.
        2. Восстановление атрибутов `_processed_count`, `_total_tests`, `_start_time`.
        3. Восстановление `_mask_sample_counts` из сохранённого словаря.
        4. (Опционально) Сканирование `masks/` для коррекции счётчиков вниз.
        5. Возврат `True` при успехе, `False` при любой ошибке.

        Args:
            None (использует атрибуты экземпляра: `progress_file`, `masks_dir`, `save_masks`).

        Returns:
            bool: `True` если прогресс успешно загружен, `False` если файл не найден или повреждён.

        Example:
            ```python
            tester = BatchClassicTester(resume=True)

            # При инициализации автоматически вызывается _load_progress()
            # Если файл .progress.json существует:
            print(f"Восстановлено: {tester._processed_count}/{tester._total_tests} тестов")

            # Ручной вызов (например, для отладки)
            success = tester._load_progress()
            if not success:
                print("⚠️  Прогресс не загружен, начинаем с начала")

        Note:
            - Метод устойчив к повреждениям: любое исключение (JSONDecodeError, FileNotFoundError)
            перехватывается, выводится предупреждение и возвращается `False`.
            - Коррекция счётчиков **только вниз** предотвращает ситуацию, когда сбой при
            сохранении маски "разблокирует" её сохранение при перезапуске, нарушая лимиты.
            - Если файл `.progress.json` не существует, метод возвращает `False`, и выполнение
            начинается с начала (атрибуты остаются с дефолтными значениями).
            - Для отладки можно временно отключить сканирование масок, установив
            `self.save_masks = False` перед вызовом.
        """
        try:
            with open(self.progress_file, "r") as f:
                progress = json.load(f)
            self._processed_count = progress.get("processed_count", 0)
            self._total_tests = progress.get("total_tests", 0)
            self._start_time = progress.get("start_time", time.time())
            if "mask_sample_counts" in progress:
                for pair, methods in progress["mask_sample_counts"].items():
                    for method, count in methods.items():
                        self._mask_sample_counts[pair][method] = count

            if self.save_masks and self.masks_dir.exists():
                for pair_dir in self.masks_dir.iterdir():
                    if not pair_dir.is_dir():
                        continue
                    pair_key: str = pair_dir.name
                    for method_dir in pair_dir.iterdir():
                        if not method_dir.is_dir():
                            continue
                        method_name: str = method_dir.name
                        # Считаем уже сохранённые поддиректории
                        saved_count: int = len(
                            [
                                d
                                for d in method_dir.iterdir()
                                if d.is_dir() and (d / "metrics.json").exists()
                            ]
                        )
                        current_count: int = self._mask_sample_counts[pair_key][
                            method_name
                        ]
                        if saved_count < current_count:
                            self._mask_sample_counts[pair_key][
                                method_name
                            ] = saved_count
                            print(
                                f"⚠️  Коррекция счётчика: {pair_key}/{method_name}: "
                                f"{current_count} → {saved_count} (файлы удалены?)"
                            )
            return True
        except Exception as e:
            print(f"⚠️  Не удалось загрузить прогресс: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    def run_batch_test(self) -> pd.DataFrame:
        """
        Запускает основной цикл массового тестирования согласованности методов.

        Это главный метод класса, который:
        1. Загружает изображения из датасета через `_load_images_with_masks()`.
        2. Инициализирует трекер прогресса с учётом общего числа тестов.
        3. Запускает вложенный цикл по (изображение × метод × пара библиотек).
        4. Для каждой комбинации:
        - Пропускает уже выполненные тесты (если `resume=True` и не `refresh_masks`).
        - Запускает `_run_consistency_test()` для получения масок и метрик.
        - Агрегирует метрики в `self.results`.
        - Сохраняет маски и визуализации (если `_should_save_mask()` вернул `True`).
        - Обновляет прогресс-бар с расширенной статистикой.
        - Выполняет автосохранение прогресса каждые `autosave_interval` изображений.
        5. После завершения:
        - Сохраняет финальный прогресс и агрегированные результаты.
        - Генерирует сводные графики через `_create_all_summary_charts()`.
        - Создаёт интерактивный HTML-отчёт через `_create_html_summary_report()`.
        - Возвращает сводный DataFrame с метриками по всем методам.

        Алгоритм пропуска выполненных тестов (при `resume=True`):
        ```
        test_index = (img_idx × total_methods × total_pairs +
                    method_idx × total_pairs +
                    pair_idx + 1)

        if self.resume and not self.refresh_masks:
            if self._processed_count >= test_index:
                continue  # Пропускаем уже выполненный тест
        ```

        Логика сохранения масок (внутри цикла):
        ```
        if masks is not None and (self.refresh_masks or not save_dir.exists()):
            # Сохраняем маски, визуализации и метрики
            _save_segmentation_results(...)
            _create_comparison_visualization(...)
            _save_masks(...)
        ```

        Returns:
            pd.DataFrame: Сводная таблица с агрегированными результатами.
                        Столбцы включают:
                        - `Method`, `Library_Pair`, `Images_Tested`
                        - `{metric}_mean`, `{metric}_std`, `{metric}_min`, `{metric}_max`
                        для каждой метрики из `self.metrics_to_aggregate`
                        - `time_a_mean`, `time_b_mean`, `time_diff_mean`
                        - `error_count`, `error_rate`, `pass_rate`, `warning_rate`, `fail_rate`

        Raises:
            ValueError: Если не удалось загрузить ни одного изображения.

        Example:
            ```python
            # Запуск полного тестирования
            tester = BatchClassicTester(max_images=100)
            df = tester.run_batch_test()

            # Доступ к результатам:
            print(f"Всего методов: {len(df)}")
            print(f"Средний IoU: {df['iou_mean'].mean():.3f}")

            # Топ-5 методов по согласованности:
            top_methods = df.nlargest(5, 'iou_mean')[['Method', 'iou_mean', 'pass_rate']]
            print(top_methods)

            # Сохранение отчётов:
            csv_path, json_path, md_path = tester.save_results(df)
            tester.plot_results(df)  # Генерация графиков
            ```

        Note:
            - Метод обрабатывает прерывания (Ctrl+C, SIGTERM) через `_setup_signal_handlers()`,
            выполняя финальное автосохранение перед завершением.
            - После каждого изображения выполняется очистка памяти: `torch.cuda.empty_cache()`
            и `gc.collect()` для предотвращения утечек при работе с большими изображениями.
            - Временные файлы (`.progress.json`, `.results_temp.csv`) удаляются после
            успешного завершения, чтобы не захламлять директорию результатов.
            - Для отладки можно установить `max_images=5` и `autosave_interval=1`, чтобы
            видеть результаты после каждого изображения.
        """
        # Загрузка данных
        test_data: List[Tuple[str, np.ndarray]] = self._load_images_with_masks()
        if not test_data:
            raise ValueError("No test data loaded!")

        total_images: int = len(test_data)
        total_methods: int = len(self.all_methods)
        total_pairs: int = len(self.library_pairs)

        print(f"🔧 Тестируем {total_methods} методов")
        print(f"🔗 Пары библиотек: {[f'{a}↔{b}' for a, b in self.library_pairs]}")
        print(f"📊 Всего тестов: {total_images * total_methods * total_pairs}")

        # Инициализация прогресса
        self._init_progress_tracking(total_images, total_methods, total_pairs)

        from tqdm import tqdm

        with tqdm(
            total=self._total_tests,
            desc="🧪 Тестирование согласованности",
            unit="тест",
            leave=True,
            dynamic_ncols=True,
            mininterval=0.5,
        ) as pbar:

            if self.resume and self._processed_count > 0:
                pbar.update(self._processed_count)
                print(f"⏭️  Пропущено {self._processed_count} выполненных тестов")

            for img_idx, (img_name, image) in enumerate(test_data):
                for method_name, params in self.all_methods:
                    for lib_a, lib_b in self.library_pairs:
                        pair_key: str = f"{lib_a}_vs_{lib_b}"

                        # Пропуск выполненных тестов при resume
                        # 🔧 O(1) доступ вместо O(n) поиска
                        if (
                            self.resume
                            and self._processed_count > 0
                            and not self.refresh_masks
                        ):
                            method_key: Tuple[str, str] = (
                                method_name,
                                json.dumps(params, sort_keys=True),
                            )
                            method_idx: Optional[int] = self._method_index_cache.get(
                                method_key
                            )
                            pair_idx: Optional[int] = self._pair_index_cache.get(
                                (lib_a, lib_b)
                            )

                            if method_idx is None or pair_idx is None:
                                # Если кэш не найден (маловероятно), вычисляем как fallback
                                method_idx = self.all_methods.index(
                                    (method_name, params)
                                )
                                pair_idx = self.library_pairs.index((lib_a, lib_b))

                            test_index: int = (
                                img_idx * total_methods * total_pairs
                                + method_idx * total_pairs
                                + pair_idx
                                + 1
                            )

                            if self._processed_count >= test_index:
                                continue

                        # Запуск теста согласованности
                        metrics, exec_times, error, masks = self._run_consistency_test(
                            method_name, params, lib_a, lib_b, image, img_name, pair_key
                        )

                        # Обработка результатов
                        if error:
                            self.errors[pair_key][method_name].append(
                                f"{img_name}: {error}"
                            )
                        elif metrics:
                            for metric_name in self.metrics_to_aggregate:
                                if metric_name in metrics:
                                    self.results[pair_key][method_name][
                                        metric_name
                                    ].append(metrics[metric_name])
                            if exec_times:
                                self.execution_times[pair_key][method_name].append(
                                    exec_times
                                )
                            if masks is not None:
                                mask_a, mask_b = masks
                                save_dir = (
                                    self.masks_dir
                                    / pair_key
                                    / method_name
                                    / img_name.replace(".", "_")
                                )
                                # 1. Сохраняем маски и изображение
                                if self.refresh_masks or not save_dir.exists():
                                    self._save_segmentation_results(
                                        method_name=method_name,
                                        pair_key=pair_key,
                                        img_name=img_name,
                                        image=image,
                                        mask_a=mask_a,
                                        mask_b=mask_b,
                                        metrics=metrics,
                                        lib_a=lib_a,
                                        lib_b=lib_b,
                                    )

                                    # 2. Создаём визуализацию сравнения
                                    self._create_comparison_visualization(
                                        method_name=method_name,
                                        pair_key=pair_key,
                                        img_name=img_name,
                                        image=image,
                                        mask_a=mask_a,
                                        mask_b=mask_b,
                                        metrics=metrics,
                                        lib_a=lib_a,
                                        lib_b=lib_b,
                                    )
                                    self._save_masks(
                                        pair_key=pair_key,
                                        method_name=method_name,
                                        img_name=img_name,
                                        mask_a=mask_a,
                                        mask_b=mask_b,
                                        metrics=metrics,
                                        image=image,
                                    )

                            # Статус валидации
                            status = self._check_validation_status(metrics)
                            self.validation_status[pair_key][method_name].append(status)

                        # Обновление прогресса
                        self._processed_count += 1
                        pbar.update(1)
                        self._update_progress_bar(
                            pbar,
                            self._processed_count,
                            (lib_a, lib_b),
                            method_name,
                            img_name,
                        )

                        # Автосохранение
                        if img_idx % self.autosave_interval == 0 and img_idx > 0:
                            self._save_progress()

                # Очистка памяти
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()

        # Финальное сохранение
        print("\n🏁 Завершение тестирования...")
        print("\n🏁 Завершение тестирования...")
        self._save_progress(force=True)

        # Удаление временных файлов
        if self.progress_file.exists():
            self.progress_file.unlink()
        if self.temp_results_file.exists():
            self.temp_results_file.unlink()

        # Агрегация результатов
        df: pd.DataFrame = self._aggregate_results()

        # ──────────────────────────────────────────────────────
        # ГЕНЕРАЦИЯ СВОДНЫХ ГРАФИКОВ
        # ──────────────────────────────────────────────────────
        print("\n📊 Генерация сводных графиков...")
        self._create_all_summary_charts(df)
        print("✅ Сводные графики сохранены")

        # ──────────────────────────────────────────────────────
        # ГЕНЕРАЦИЯ HTML-ОТЧЁТА
        # ──────────────────────────────────────────────────────
        print("\n📄 Генерация HTML-отчёта...")
        self._create_html_summary_report(df)
        print("✅ HTML-отчёт сохранён")

        return df

    # ──────────────────────────────────────────────────────────────────────
    def _aggregate_results(self) -> pd.DataFrame:
        """
        Агрегирует накопленные результаты в сводную таблицу.

        Для каждого метода рассчитывает среднее, стандартное отклонение, min и max
        по каждой метрике, а также среднее время выполнения и долю ошибок.
        Сортирует таблицу по убыванию `iou_mean`.

        Returns:
            `pd.DataFrame` со столбцами `Method`, `{metric}_{stat}`, `time_mean_s`,
            `error_count`, `error_rate` и `Images_Tested`.
        """
        rows: List[Dict[str, Any]] = []

        for pair_key in self.results:
            for method_name in self.results[pair_key]:
                images_tested: int = len(
                    self.results[pair_key][method_name].get("iou", [])
                )

                row: Dict[str, Any] = {
                    "Method": method_name,
                    "Library_Pair": pair_key,
                    "Torch_Version": (
                        "v2"
                        if "torch_v2" in pair_key
                        or self.torch_segmenter_version == "v2"
                        else "v1"
                    ),
                    "Images_Tested": images_tested,
                }

                # Средние значения метрик
                for metric_name in self.metrics_to_aggregate:
                    values: List[float] = self.results[pair_key][method_name].get(
                        metric_name, []
                    )
                    if values:
                        row[f"{metric_name}_mean"] = np.mean(values)
                        row[f"{metric_name}_std"] = np.std(values)
                        row[f"{metric_name}_min"] = np.min(values)
                        row[f"{metric_name}_max"] = np.max(values)
                    else:
                        row[f"{metric_name}_mean"] = np.nan

                # Время выполнения
                times: List[Tuple[float, float]] = self.execution_times[pair_key][
                    method_name
                ]
                if times:
                    times_a: List[float] = [t[0] for t in times]
                    times_b: List[float] = [t[1] for t in times]
                    row["time_a_mean"] = np.mean(times_a)
                    row["time_b_mean"] = np.mean(times_b)
                    row["time_diff_mean"] = np.mean(
                        [abs(a - b) for a, b in zip(times_a, times_b)]
                    )

                # Ошибки и статус
                errors: List[str] = self.errors[pair_key][method_name]
                statuses: List[ValidationStatus] = self.validation_status[pair_key][
                    method_name
                ]

                row["error_count"] = len(errors)
                row["error_rate"] = len(errors) / max(images_tested, 1)

                if statuses:
                    status_counts: pd.Series = pd.Series(statuses).value_counts()
                    row["pass_rate"] = status_counts.get("PASS", 0) / len(statuses)
                    row["warning_rate"] = status_counts.get("WARNING", 0) / len(
                        statuses
                    )
                    row["fail_rate"] = status_counts.get("FAIL", 0) / len(statuses)

                rows.append(row)

        df: pd.DataFrame = pd.DataFrame(rows)

        # Сортировка по IoU для каждой пары
        if "iou_mean" in df.columns:
            df = df.sort_values(["Library_Pair", "iou_mean"], ascending=[True, False])

        return df

    # ──────────────────────────────────────────────────────────────────────
    def save_results(
        self, df: pd.DataFrame, prefix: str = "consistency_test"
    ) -> Tuple[Path, Path, Path]:
        """
        Экспортирует результаты в несколько форматов для дальнейшего анализа.

        Args:
            df: Датафрейм с агрегированными результатами.
            prefix: Префикс для имён выходных файлов.

        Returns:
            Кортеж путей к сохранённым файлам `(csv_path, json_path, md_path)`.
        """
        # CSV
        csv_path: Path = self.output_dir / f"{prefix}_results.csv"
        df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"💾 CSV: {csv_path}")

        # JSON
        json_path: Path = self.output_dir / f"{prefix}_details.json"
        details: Dict[str, Any] = {
            "summary": df.to_dict(orient="records"),
            "errors": {k: dict(v) for k, v in self.errors.items()},
            "config": {
                "library_pairs": self.library_pairs,
                "total_methods": len(self.all_methods),
                "image_size": self.image_size,
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 JSON: {json_path}")

        # Markdown
        md_path: Path = self.output_dir / f"{prefix}_report.md"
        self._save_markdown_report(df, md_path)
        print(f"💾 Markdown: {md_path}")

        return csv_path, json_path, md_path

    # ──────────────────────────────────────────────────────────────────────
    def _save_markdown_report(self, df: pd.DataFrame, path: Path) -> None:
        """
        Генерирует читаемый Markdown-отчёт с топ-10 методов, полной таблицей и статистикой ошибок.

        Args:
            df: Датафрейм с результатами.
            path: Путь для сохранения `.md` файла.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write("# 📊 Отчёт: Тестирование согласованности методов сегментации\n\n")
            f.write(
                f"**Пары библиотек:** {', '.join(f'{a}↔{b}' for a, b in self.library_pairs)}\n"
            )
            f.write(f"**Методов:** {len(self.all_methods)}\n")
            f.write(f"**Изображений:** {self.max_images or 'all'}\n\n")

            # Топ по каждой паре
            for pair in df["Library_Pair"].unique():
                df_pair = df[df["Library_Pair"] == pair]
                f.write(f"## 🏆 {pair.upper()}: Топ-10 по согласованности (IoU)\n\n")
                top = df_pair.head(10)[["Method", "iou_mean", "dice_mean", "pass_rate"]]
                f.write(top.to_markdown(index=False) + "\n\n")

            # Сводная таблица
            f.write("## 📈 Полная таблица результатов\n\n")
            cols: List[str] = ["Method", "Library_Pair"] + [
                c for c in df.columns if c.endswith("_mean")
            ]
            f.write(df[cols].to_markdown(index=False, floatfmt=".4f") + "\n\n")

            # Статистика ошибок
            if df["error_count"].sum() > 0:
                f.write("## ⚠️ Статистика ошибок\n\n")
                err_df: pd.DataFrame = df[df["error_count"] > 0][
                    ["Method", "Library_Pair", "error_count", "error_rate"]
                ]
                f.write(err_df.to_markdown(index=False) + "\n\n")

    # ──────────────────────────────────────────────────────────────────────
    def plot_results(self, df: pd.DataFrame, output_dir: Optional[Path] = None) -> None:
        """
        Построение визуализаций для анализа согласованности.

        Графики:
        1. `consistency_ranking.png` — IoU между реализациями по методам (для каждой пары)
        2. `time_vs_consistency.png` — зависимость согласованности от разницы во времени
        3. `metrics_heatmap.png` — тепловая карта всех метрик соответствия
        4. `pass_rate_by_pair.png` — доля успешных валидаций по парам библиотек

        Args:
            df: Датафрейм с результатами.
            output_dir: Директория для сохранения графиков. Если `None`, используется `output_dir/charts`.
        """
        if output_dir is None:
            output_dir = self.output_dir / "charts"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Фильтрация: методы с достаточным количеством тестов
        df_plot: pd.DataFrame = df[df["Images_Tested"] >= 5].copy()
        if df_plot.empty:
            print("⚠️ Недостаточно данных для графиков")
            return

        # ──────────────────────────────────────────────────────
        # График 1: Рейтинг согласованности (IoU) по методам для каждой пары
        # ──────────────────────────────────────────────────────
        for pair in df_plot["Library_Pair"].unique():
            df_pair = df_plot[df_plot["Library_Pair"] == pair].sort_values(
                "iou_mean", ascending=True
            )

            plt.figure(figsize=(14, max(8, len(df_pair) * 0.35)))
            sns.barplot(
                data=df_pair.head(15), x="iou_mean", y="Method", palette="viridis"
            )
            plt.xlabel("Mean IoU (между реализациями)")
            plt.title(f"Топ-15 методов по согласованности: {pair.upper()}")
            plt.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f"consistency_ranking_{pair}.png", dpi=150)
            plt.close()

        # ──────────────────────────────────────────────────────
        # График 2: Зависимость согласованности от разницы во времени
        # ──────────────────────────────────────────────────────
        if "time_diff_mean" in df_plot.columns and "iou_mean" in df_plot.columns:
            plt.figure(figsize=(12, 8))

            for pair in df_plot["Library_Pair"].unique():
                df_pair = df_plot[df_plot["Library_Pair"] == pair]
                plt.scatter(
                    df_pair["time_diff_mean"],
                    df_pair["iou_mean"],
                    label=pair.upper(),
                    s=80,
                    alpha=0.7,
                    edgecolors="black",
                )

            plt.xlabel("Средняя разница во времени (сек)")
            plt.ylabel("Mean IoU (согласованность)")
            plt.title("Зависимость согласованности от разницы в скорости")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "time_vs_consistency.png", dpi=150)
            plt.close()

        # ──────────────────────────────────────────────────────
        # График 3: Тепловая карта метрик (по парам и методам)
        # ──────────────────────────────────────────────────────
        metric_cols: List[str] = [
            c
            for c in df.columns
            if c.endswith("_mean") and c.split("_")[0] in self.metrics_to_aggregate
        ]
        if metric_cols:
            # Топ-15 методов по среднему IoU across all pairs
            df_top = df_plot.groupby("Method")["iou_mean"].mean().nlargest(15).index
            df_heatmap: pd.DataFrame = df_plot[df_plot["Method"].isin(df_top)].copy()

            plt.figure(figsize=(16, 12))

            # Подготовка данных: метод × (пара_метрика)
            pivot_data: Dict = {}
            for method in df_top:
                for pair in df_heatmap["Library_Pair"].unique():
                    row = df_heatmap[
                        (df_heatmap["Method"] == method)
                        & (df_heatmap["Library_Pair"] == pair)
                    ]
                    if not row.empty:
                        for metric in [
                            "iou",
                            "dice",
                            "f1_score",
                            "precision",
                            "recall",
                        ]:
                            col: str = f"{metric}_mean"
                            if col in row.columns:
                                pivot_data[(method, pair, metric)] = row[col].values[0]

            if pivot_data:
                # Создаём DataFrame для heatmap
                heatmap_df: pd.DataFrame = pd.DataFrame(
                    [
                        {"Method": m, "Pair": p, "Metric": met, "Value": v}
                        for (m, p, met), v in pivot_data.items()
                    ]
                )
                pivot: pd.DataFrame = heatmap_df.pivot_table(
                    index="Method", columns=["Pair", "Metric"], values="Value"
                )

                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.5)
                plt.title("Тепловая карта метрик согласованности (Топ-15 методов)")
                plt.xlabel("Пара библиотеки × Метрика")
                plt.ylabel("Метод")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(output_dir / "metrics_heatmap.png", dpi=150)
                plt.close()

        # ──────────────────────────────────────────────────────
        # График 4: Доля успешных валидаций (PASS rate) по парам
        # ──────────────────────────────────────────────────────
        if "pass_rate" in df_plot.columns:
            plt.figure(figsize=(10, 6))

            pass_by_pair: pd.Series = df_plot.groupby("Library_Pair")[
                "pass_rate"
            ].mean()
            sns.barplot(x=pass_by_pair.index, y=pass_by_pair.values, palette="Set2")

            plt.xlabel("Пара библиотек")
            plt.ylabel("Средняя доля PASS (%)")
            plt.title("Успешность валидации по парам реализаций")
            plt.ylim(0, 1)
            plt.gca().yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%")
            )
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "pass_rate_by_pair.png", dpi=150)
            plt.close()

        print(f"📊 Графики сохранены в: {output_dir}")

    # ──────────────────────────────────────────────────────────────────────
    def print_summary(self, df: pd.DataFrame) -> None:
        """
        Выводит сводную статистику по результатам тестирования в консоль.

        Метод группирует данные по парам библиотек и для каждой выводит:
        - Топ-5 методов по среднему IoU (с указанием стандартного отклонения).
        - Топ-5 самых быстрых методов по средней разнице во времени выполнения.
        - Методы с наибольшим количеством ошибок (если есть).

        Формат вывода:
        ```
        🔗 TORCH_VS_OPENCV:
        🏆 Топ-5 по IoU:
            • otsu_thresholding: IoU=0.9234 ± 0.0123
            • global_thresholding: IoU=0.8912 ± 0.0234
            ...
        ⚡ Топ-5 по скорости:
            • global_thresholding: Δt=12.3мс
            • otsu_thresholding: Δt=15.7мс
            ...
        ❌ Методы с ошибками:
            • phase_congruency_edge: 3 ошибок (15.0%)
        ```

        Args:
            df: DataFrame с агрегированными результатами, содержащий столбцы:
                - `Method`, `Library_Pair`, `iou_mean`, `iou_std`,
                - `time_diff_mean`, `error_count`, `error_rate`.

        Returns:
            None (выводит информацию в stdout).

        Example:
            ```python
            tester = BatchClassicTester(max_images=50)
            df = tester.run_batch_test()
            tester.print_summary(df)

            # Пример вывода:
            # 🔗 TORCH_VS_SKLEARN:
            #    🏆 Топ-5 по IoU:
            #       • otsu_thresholding: IoU=0.9412 ± 0.0089
            #       • kmeans_segmentation: IoU=0.8756 ± 0.0312
            #    ⚡ Топ-5 по скорости:
            #       • global_thresholding: Δt=2.1мс
            #       • adaptive_thresholding: Δt=8.4мс
            ```

        Note:
            - Метод автоматически пропускает пары библиотек, для которых нет данных.
            - Значения времени конвертируются в миллисекунды для удобства чтения.
            - Процент ошибок рассчитывается как `(error_count / Images_Tested) × 100`.
            - Для методов без ошибок блок "❌ Методы с ошибками" не выводится.
            - Вывод отформатирован с использованием эмодзи для лучшей визуальной навигации.
        """
        print("\n" + "=" * 80)
        print("📊 СВОДКА ПО ТЕСТУ СОГЛАСОВАННОСТИ")
        print("=" * 80)

        for pair in df["Library_Pair"].unique():
            df_pair = df[df["Library_Pair"] == pair]

            print(f"\n🔗 {pair.upper()}:")
            print("   🏆 Топ-5 по IoU:")
            for _, row in df_pair.head(5).iterrows():
                print(
                    f"      • {row['Method']}: IoU={row['iou_mean']:.4f} ± {row['iou_std']:.4f}"
                )

            print("   ⚡ Топ-5 по скорости (наименьшая разница):")
            fast = df_pair.dropna(subset=["time_diff_mean"]).nsmallest(
                5, "time_diff_mean"
            )
            for _, row in fast.iterrows():
                print(
                    f"      • {row['Method']}: Δt={row['time_diff_mean'] * 1000:.1f}мс"
                )

            if df_pair["error_count"].sum() > 0:
                print("   ❌ Методы с ошибками:")
                err = (
                    df_pair[df_pair["error_count"] > 0]
                    .sort_values("error_count", ascending=False)
                    .head(3)
                )
                for _, row in err.iterrows():
                    print(f"      • {row['Method']}: {row['error_count']} ошибок")

        # Общая статистика
        print(f"\n💾 Все результаты: {self.output_dir}")

    # ──────────────────────────────────────────────────────────────────────
    def _save_segmentation_results(
        self,
        method_name: str,
        pair_key: str,
        img_name: str,
        image: np.ndarray,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        metrics: Dict[str, float],
        lib_a: str,
        lib_b: str,
        save_dir: Optional[Path] = None,
    ) -> None:
        """
        Сохраняет результаты сегментации для визуальной верификации и отладки.

        Создаёт иерархическую структуру файлов для удобного доступа к артефактам:
        ```
        {output_dir}/results/
        └── {pair_key}/                    # например: torch_vs_opencv
            └── {method_name}/             # например: otsu_thresholding
                └── {img_name_sanitized}/  # например: ADE_val_00000001_jpg
                    ├── original.jpg       # исходное изображение (качество 95%)
                    ├── {lib_a}_mask.png   # маска от первой библиотеки
                    ├── {lib_b}_mask.png   # маска от второй библиотеки
                    └── metrics.json       # метрики соответствия в JSON
        ```

        Форматы файлов:
        - `original.jpg`: RGB-изображение в формате JPEG с качеством 95%.
        - `*_mask.png`: Бинарные маски в формате PNG (0=чёрный, 255=белый) для сохранения чёткости границ.
        - `metrics.json`: Словарь с метриками и параметрами:
        ```json
        {
            "iou": 0.92,
            "dice": 0.95,
            "precision": 0.93,
            "recall": 0.91,
            "f1_score": 0.92,
            "mae": 0.03,
            "hausdorff_distance": 4.2,
            "pixel_accuracy": 0.98,
            "time_diff": 0.012
        }
        ```

        Args:
            method_name: Имя тестируемого метода (например, "otsu_thresholding").
            pair_key: Ключ пары библиотек (например, "torch_vs_opencv").
            img_name: Имя исходного файла изображения (например, "ADE_val_00000001.jpg").
            image: Исходное изображение формы `(H, W)` или `(H, W, 3)`, dtype=uint8.
            mask_a: Бинарная маска от первой библиотеки, форма `(H, W)`, dtype=uint8, {0, 255}.
            mask_b: Бинарная маска от второй библиотеки, форма `(H, W)`, dtype=uint8, {0, 255}.
            metrics: Словарь с рассчитанными метриками соответствия между масками.
            lib_a: Название первой библиотеки (например, "torch") для имён файлов.
            lib_b: Название второй библиотеки (например, "opencv") для имён файлов.

        Returns:
            None. Файлы сохраняются на диск в структуру, описанную выше.

        Note:
            - Имя изображения санитизируется: точки заменяются на подчёркивания
            (`img_name.replace(".", "_")`), чтобы избежать конфликтов с расширениями.
            - Метрики в `metrics.json` сериализуются с приведением numpy-типов к Python
            (`float(v) if isinstance(v, np.floating) else v`).
            - Если изображение имеет форму `(H, W)` (grayscale), оно автоматически
            конвертируется в 3 канала перед сохранением как JPEG.
            - Метод создаёт директории рекурсивно (`mkdir(parents=True, exist_ok=True)`),
            поэтому безопасен для параллельного запуска.
            - Если `self.save_results=False`, метод возвращает управление без выполнения действий.

        Example:
            ```python
            # После вызова _save_segmentation_results для теста:
            #   method_name="otsu", pair_key="torch_vs_opencv", img_name="test.jpg"
            # Будет создана структура:
            # ./data/batch_consistency_test/results/torch_vs_opencv/otsu/test_jpg/
            # ├── original.jpg
            # ├── torch_mask.png
            # ├── opencv_mask.png
            # └── metrics.json

            # Загрузка маски для анализа:
            from PIL import Image
            import json
            mask_torch = Image.open("results/torch_vs_opencv/otsu/test_jpg/torch_mask.png")
            with open("results/torch_vs_opencv/otsu/test_jpg/metrics.json") as f:
                metrics = json.load(f)
            print(f"IoU: {metrics['iou']:.3f}")
        """
        if not self.save_results_enabled:
            return

        # Создаём директорию
        if save_dir is None:
            save_dir = (
                self.output_dir
                / "results"
                / pair_key
                / method_name
                / img_name.replace(".", "_")
            )
        if (
            not self.refresh_masks
            and save_dir.exists()
            and (save_dir / "metrics.json").exists()
        ):
            return
        save_dir.mkdir(parents=True, exist_ok=True)

        # Сохраняем оригинальное изображение
        if image.max() <= 1.0:
            img_save = (image * 255).astype(np.uint8)
        else:
            img_save = image.astype(np.uint8)

        if len(img_save.shape) == 2:
            img_save = np.stack([img_save] * 3, axis=-1)

        Image.fromarray(img_save).save(save_dir / "original.jpg", quality=95)

        # Сохраняем маски (PNG для сохранения чёткости)
        Image.fromarray(mask_a).save(save_dir / f"{lib_a}_mask.png")
        Image.fromarray(mask_b).save(save_dir / f"{lib_b}_mask.png")

        # Сохраняем метрики
        metrics_save: Dict[str, Any] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in metrics.items()
        }
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics_save, f, indent=2)

    # ──────────────────────────────────────────────────────────────────────
    def _create_comparison_visualization(
        self,
        method_name: str,
        pair_key: str,
        img_name: str,
        image: np.ndarray,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        metrics: Dict[str, float],
        lib_a: str,
        lib_b: str,
        save_dir: Optional[Path] = None,
    ) -> None:
        """
        Создаёт 4-панельную визуализацию для наглядного сравнения результатов двух реализаций.

        Макет визуализации (слева направо):
        ```
        ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
        │   Original      │   Mask A        │   Mask B        │   Difference    │
        │   Image         │   ({lib_a})     │   ({lib_b})     │   Heatmap       │
        │                 │   IoU: 0.92     │                 │   Status: PASS  │
        └─────────────────┴─────────────────┴─────────────────┴─────────────────┘
        ```

        Описание панелей:
        1. **Original Image**: Исходное изображение (RGB или grayscale) для контекста.
        2. **Mask A**: Бинарная маска от первой библиотеки (линейная шкала серого, 0=чёрный, 255=белый).
                    В заголовке отображается значение IoU для быстрой оценки качества.
        3. **Mask B**: Бинарная маска от второй библиотеки (аналогично панели 2).
        4. **Difference Heatmap**: Тепловая карта абсолютной разницы между масками.
                                - Цветовая схема: "hot" (чёрный → красный → жёлтый → белый).
                                - Значения: 0 (полное совпадение) → 255 (полное несовпадение).
                                - В заголовке отображается статус валидации (PASS/WARNING/FAIL).

        Args:
            method_name: Имя тестируемого метода для заголовка визуализации.
            pair_key: Ключ пары библиотек (например, "torch_vs_opencv").
            img_name: Имя исходного файла изображения.
            image: Исходное изображение формы `(H, W)` или `(H, W, 3)`, dtype=uint8.
            mask_a: Бинарная маска от первой библиотеки, форма `(H, W)`, dtype=uint8, {0, 255}.
            mask_b: Бинарная маска от второй библиотеки, форма `(H, W)`, dtype=uint8, {0, 255}.
            metrics: Словарь с метриками для отображения в заголовках панелей.
            lib_a: Название первой библиотеки для подписей.
            lib_b: Название второй библиотеки для подписей.

        Returns:
            None. Визуализация сохраняется как `{method_name}_{img_name}.jpg` в директорию
            `{output_dir}/visualizations/{pair_key}/`.

        Note:
            - Размер фигуры: 20×5 дюймов (широкий формат для 4 панелей в ряд).
            - Цветовая карта для масок: "gray" с фиксированным диапазоном [0, 255] для
            согласованности отображения.
            - Цветовая карта для heatmap: "hot" с диапазоном [0, 255] и цветовой шкалой.
            - Статус валидации окрашивается: ✅ зелёный (PASS), ⚠️ оранжевый (WARNING), ❌ красный (FAIL).
            - Визуализация сохраняется с разрешением 150 DPI и обрезкой лишних полей
            (`bbox_inches="tight"`).
            - Если `self.save_visualizations=False`, метод возвращает управление без выполнения действий.

        Example:
            ```python
            # После вызова метода будет создан файл:
            # ./data/batch_consistency_test/visualizations/torch_vs_opencv/otsu_thresholding_ADE_val_00000001_jpg.jpg
            #
            # Пример содержимого визуализации:
            # - Панель 1: цветное изображение пейзажа
            # - Панель 2: чёрно-белая маска от Torch с надписью "IoU: 0.94"
            # - Панель 3: чёрно-белая маска от OpenCV
            # - Панель 4: тепловая карта с красными пятнами в местах расхождений,
            #            заголовок "Status: PASS" зелёным цветом
            ```
        """
        if not self.save_visualizations:
            return

        if save_dir is None:
            save_dir = self.output_dir / "visualizations" / pair_key

        # Создаём директорию для визуализаций

        viz_filename: str = f"{method_name}_{img_name.replace('.', '_')}.jpg"
        viz_path: Path = save_dir / viz_filename
        if not self.refresh_masks and viz_path.exists():
            return

        save_dir.mkdir(parents=True, exist_ok=True)

        # Подготовка данных
        img_display: np.ndarray
        if image.max() <= 1.0:
            img_display = (image * 255).astype(np.uint8)
        else:
            img_display = image.astype(np.uint8)

        if len(img_display.shape) == 2:
            img_display = np.stack([img_display] * 3, axis=-1)

        # Вычисляем разницу
        diff: np.ndarray = np.abs(mask_a.astype(float) - mask_b.astype(float))

        # Определяем статус
        status: ValidationStatus = self._check_validation_status(metrics)
        status_color: str = {"PASS": "green", "WARNING": "orange", "FAIL": "red"}.get(
            status, "red"
        )

        # Создаём фигуру
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 1. Оригинальное изображение
        axes[0].imshow(img_display)
        axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # 2. Маска библиотеки A
        axes[1].imshow(mask_a, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(
            f"{lib_a.upper()} {method_name}\nIoU: {metrics.get('iou', np.nan):.3f}",
            fontsize=11,
        )
        axes[1].axis("off")

        # 3. Маска библиотеки B
        axes[2].imshow(mask_b, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title(f"{lib_b.upper()} {method_name}", fontsize=11)
        axes[2].axis("off")

        # 4. Heatmap разницы
        im = axes[3].imshow(diff, cmap="hot", vmin=0, vmax=255)
        axes[3].set_title(
            f"Difference\nStatus: {status}",
            color=status_color,
            fontsize=11,
            fontweight="bold",
        )
        axes[3].axis("off")
        plt.colorbar(im, ax=axes[3], fraction=0.046, label="Pixel Difference")

        # Общий заголовок
        plt.suptitle(
            f"Consistency Check: {method_name} ({lib_a.upper()} vs {lib_b.upper()})",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout(rect=(0, 0, 1, 0.96))

        # Сохраняем
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ──────────────────────────────────────────────────────────────────────
    def _create_summary_comparison_chart(
        self,
        df: pd.DataFrame,
        pair_key: str,
    ) -> None:
        """
        Создаёт сводные графики сравнения всех методов для данной пары библиотек.

        Генерирует три типа визуализаций в директорию `{output_dir}/charts/{pair_key}/`:

        1. **IoU Comparison** (`iou_comparison.png`):
        - Горизонтальный bar-чарт топ-15 методов по среднему IoU.
        - Цветовое кодирование по статусу: 🟢 PASS (≥0.80), 🟠 WARNING (0.50–0.80), 🔴 FAIL (<0.50).
        - Легенда с пояснением пороговых значений.

        2. **Execution Time Comparison** (`time_comparison.png`):
        - Scatter plot: время выполнения библиотеки A (ось X) vs библиотеки B (ось Y).
        - Диагональная линия y=x для визуальной оценки равенства скоростей.
        - Подписи методов для точек с экстремальными отклонениями.

        3. **Validation Status Distribution** (`status_distribution.png`):
        - Круговая диаграмма с распределением статусов (PASS/WARNING/FAIL).
        - Процентное соотношение и абсолютные значения для каждой категории.
        - Цветовая схема: зелёный/оранжевый/красный для интуитивного восприятия.

        Args:
            df: DataFrame с агрегированными результатами, содержащий столбцы:
                `Method`, `Library_Pair`, `iou_mean`, `time_a_mean`, `time_b_mean`, `pass_rate`.
            pair_key: Ключ пары библиотек (например, "torch_vs_opencv") для фильтрации данных.

        Returns:
            None. Графики сохраняются в `{output_dir}/charts/{pair_key}/`.

        Note:
            - Если для пары нет данных (`df_pair.empty`), метод возвращает управление без создания графиков.
            - Bar-чарт автоматически адаптирует высоту под количество методов (минимум 8 дюймов).
            - Для scatter plot подписываются только методы с экстремальными отклонениями (>2× или <0.5×),
            чтобы не перегружать визуализацию.
            - Круговая диаграмма использует `explode=(0.05, 0.05, 0.05)` для лучшего разделения секторов.
            - Все графики сохраняются с разрешением 150–200 DPI для качества печати.

        Example:
            ```python
            tester = BatchClassicTester()
            df = tester.run_batch_test()

            # Создание графиков для конкретной пары
            tester._create_summary_comparison_chart(df, "torch_vs_opencv")

            # Результат:
            # ./data/batch_consistency_test/charts/torch_vs_opencv/
            # ├── iou_comparison.png
            # ├── time_comparison.png
            # └── status_distribution.png
            ```
        """
        charts_dir: Path = self.output_dir / "charts" / pair_key
        charts_dir.mkdir(parents=True, exist_ok=True)

        # Фильтруем данные для конкретной пары
        df_pair: pd.DataFrame = df[df["Library_Pair"] == pair_key].copy()

        if df_pair.empty:
            return

        # ──────────────────────────────────────────────────────
        # График 1: IoU по методам
        # ──────────────────────────────────────────────────────
        plt.figure(figsize=(14, max(8, len(df_pair) * 0.4)))

        df_sorted: pd.DataFrame = df_pair.sort_values("iou_mean", ascending=True)
        colors: List[str] = [
            {"PASS": "#2ecc71", "WARNING": "#f39c12", "FAIL": "#e74c3c"}.get(
                row.get("pass_rate", 0) >= 0.8
                and "PASS"
                or row.get("pass_rate", 0) >= 0.5
                and "WARNING"
                or "FAIL",
                "#95a5a6",
            )
            for _, row in df_sorted.iterrows()
        ]

        bars = plt.barh(
            df_sorted["Method"],
            df_sorted["iou_mean"],
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        print(bars)

        plt.xlabel("Mean IoU", fontsize=12)
        plt.title(
            f"IoU Comparison: {pair_key.upper()}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.grid(axis="x", alpha=0.3, linestyle="--")
        plt.gca().invert_yaxis()

        # Добавляем легенду
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#2ecc71", edgecolor="black", label="PASS (IoU ≥ 0.80)"),
            Patch(
                facecolor="#f39c12",
                edgecolor="black",
                label="WARNING (0.50 ≤ IoU < 0.80)",
            ),
            Patch(facecolor="#e74c3c", edgecolor="black", label="FAIL (IoU < 0.50)"),
        ]
        plt.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        plt.savefig(charts_dir / "iou_comparison.png", dpi=200, bbox_inches="tight")
        plt.close()

        # ──────────────────────────────────────────────────────
        # График 2: Время выполнения
        # ──────────────────────────────────────────────────────
        if "time_a_mean" in df_pair.columns and "time_b_mean" in df_pair.columns:
            plt.figure(figsize=(12, 8))

            plt.scatter(
                df_pair["time_a_mean"],
                df_pair["time_b_mean"],
                s=100,
                alpha=0.7,
                c="steelblue",
                edgecolors="black",
                linewidth=0.5,
            )

            # Линия y=x
            max_time = max(df_pair["time_a_mean"].max(), df_pair["time_b_mean"].max())
            plt.plot(
                [0, max_time],
                [0, max_time],
                "r--",
                linewidth=2,
                label="y=x (equal speed)",
            )

            plt.xlabel(f"{pair_key.split('_vs_')[0].upper()} Time (s)", fontsize=12)
            plt.ylabel(f"{pair_key.split('_vs_')[1].upper()} Time (s)", fontsize=12)
            plt.title(
                f"Execution Time Comparison: {pair_key.upper()}",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Подписываем методы
            for _, row in df_pair.iterrows():
                plt.annotate(
                    row["Method"][:25],
                    (row["time_a_mean"], row["time_b_mean"]),
                    fontsize=7,
                    alpha=0.8,
                )

            plt.tight_layout()
            plt.savefig(
                charts_dir / "time_comparison.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

        # ──────────────────────────────────────────────────────
        # График 3: Распределение статусов
        # ──────────────────────────────────────────────────────
        if "pass_rate" in df_pair.columns:
            plt.figure(figsize=(10, 6))

            pass_count: int = (df_pair["pass_rate"] >= 0.8).sum()
            warning_count: int = (
                (df_pair["pass_rate"] >= 0.5) & (df_pair["pass_rate"] < 0.8)
            ).sum()
            fail_count: int = (df_pair["pass_rate"] < 0.5).sum()

            sizes: List[int] = [pass_count, warning_count, fail_count]
            labels: List[str] = [
                f"PASS\n{pass_count}",
                f"WARNING\n{warning_count}",
                f"FAIL\n{fail_count}",
            ]
            colors_pie: List[str] = ["#2ecc71", "#f39c12", "#e74c3c"]

            plt.pie(
                sizes,
                labels=labels,
                colors=colors_pie,
                autopct="%1.1f%%",
                startangle=90,
                explode=(0.05, 0.05, 0.05),
            )

            plt.title(
                f"Validation Status Distribution: {pair_key.upper()}",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            plt.tight_layout()
            plt.savefig(
                charts_dir / "status_distribution.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

    # ──────────────────────────────────────────────────────────────────────
    def _create_all_summary_charts(self, df: pd.DataFrame) -> None:
        """
        Создаёт все сводные графики для всех пар библиотек.

        Метод выполняет два этапа:
        1. Для каждой уникальной пары библиотек вызывает `_create_summary_comparison_chart()`
        для генерации специфичных графиков (IoU, время, статусы).
        2. Вызывает `_create_global_summary_chart()` для создания обобщающей визуализации
        по всем парам одновременно.

        Args:
            df: DataFrame с агрегированными результатами.

        Returns:
            None. Графики сохраняются в `{output_dir}/charts/`.

        Note:
            - Метод автоматически создаёт необходимые поддиректории.
            - Если в `df` нет данных, методы-обработчики корректно пропускают генерацию.
            - Общее время генерации зависит от количества пар и методов; для >100 методов
            рассмотрите уменьшение `max_images` или увеличение `autosave_interval`.
        """
        for pair_key in df["Library_Pair"].unique():
            self._create_summary_comparison_chart(df, pair_key)

        # Дополнительно: общий график по всем парам
        self._create_global_summary_chart(df)

    # ──────────────────────────────────────────────────────────────────────
    def _create_global_summary_chart(self, df: pd.DataFrame) -> None:
        """
        Создаёт глобальный сводный график по всем парам библиотек.

        Генерирует тепловую карту (heatmap) средних значений IoU для всех комбинаций
        (метод × пара библиотек), позволяя быстро выявить:
        - Методы с высокой согласованностью во всех реализациях.
        - Пары библиотек с систематическими расхождениями.
        - Аномальные значения, требующие дополнительного исследования.

        Формат heatmap:
        - Ось Y: имена методов (сортировка по умолчанию из DataFrame).
        - Ось X: пары библиотек (например, "torch_vs_opencv").
        - Цвет: значение IoU от 0.0 (красный) до 1.0 (зелёный) через жёлтый.
        - Аннотации: числовые значения с точностью до 3 знаков после запятой.

        Args:
            df: DataFrame с агрегированными результатами, содержащий столбцы:
                `Method`, `Library_Pair`, `iou_mean`.

        Returns:
            None. График сохраняется как `global_iou_heatmap.png` в `{output_dir}/charts/`.

        Note:
            - Если столбец `iou_mean` отсутствует, метод возвращает управление без создания графика.
            - Тепловая карта использует `pivot_table` с `aggfunc="mean"` для обработки дубликатов.
            - Поворот подписей оси X на 45° (`ha="right"`) улучшает читаемость при длинных названиях пар.
            - Разрешение 200 DPI обеспечивает качество для включения в отчёты и презентации.

        Example:
            ```python
            tester = BatchClassicTester()
            df = tester.run_batch_test()
            tester._create_global_summary_chart(df)

            # Результат:
            # ./data/batch_consistency_test/charts/global_iou_heatmap.png
            #
            # Визуализация показывает:
            # - otsu_thresholding: [0.94, 0.92, 0.93] для трёх пар → стабильно высокий
            # - phase_congruency_edge: [0.45, 0.38, 0.52] → требует внимания
            ```
        """
        charts_dir: Path = self.output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        # Heatmap метрик по всем парам и методам
        if "iou_mean" in df.columns:
            plt.figure(figsize=(16, 10))

            # Pivot table
            pivot_data: pd.DataFrame = df.pivot_table(
                values="iou_mean",
                index="Method",
                columns="Library_Pair",
                aggfunc="mean",
            )

            sns.heatmap(
                pivot_data,
                annot=True,
                fmt=".3f",
                cmap="YlOrRd",
                linewidths=0.5,
                cbar_kws={"label": "Mean IoU"},
            )

            plt.title(
                "IoU Consistency Across Library Pairs",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(
                charts_dir / "global_iou_heatmap.png", dpi=200, bbox_inches="tight"
            )
            plt.close()

    # ──────────────────────────────────────────────────────────────────────
    def _create_html_summary_report(
        self,
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Создаёт интерактивный HTML-отчёт с превью всех сравнений.

        Включает:
        - Сводную статистику по всем парам библиотек
        - Превью изображений для каждого теста
        - Интерактивные графики метрик
        - Таблицы с деталями по методам

        Args:
            df: DataFrame с результатами тестов.
            output_path: Путь для сохранения HTML (опционально).

        Returns:
            Path: Путь к сохранённому HTML-файлу.
        """
        if output_path is None:
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"consistency_report_{timestamp}.html"

        # Группируем данные для отчёта
        report_data: List[Dict[str, Any]] = []

        for pair_key in df["Library_Pair"].unique():
            df_pair = df[df["Library_Pair"] == pair_key]

            for _, row in df_pair.iterrows():
                method_name: str = row["Method"]

                # Ищем сохранённые визуализации
                viz_dir: Path = self.output_dir / "visualizations" / pair_key
                viz_files = list(viz_dir.glob(f"{method_name}_*.jpg"))

                # Ищем сохранённые результаты
                results_dir: Path = self.output_dir / "results" / pair_key / method_name

                report_data.append(
                    {
                        "pair": pair_key,
                        "method": method_name,
                        "iou_mean": row.get("iou_mean", np.nan),
                        "dice_mean": row.get("dice_mean", np.nan),
                        "pass_rate": row.get("pass_rate", 0),
                        "time_a_mean": row.get("time_a_mean", np.nan),
                        "time_b_mean": row.get("time_b_mean", np.nan),
                        "validation_status": self._get_status_from_pass_rate(
                            row.get("pass_rate", 0)
                        ),
                        "viz_files": viz_files,
                        "results_dir": results_dir if results_dir.exists() else None,
                    }
                )

        # Генерируем HTML
        html_content: str = self._generate_html_content(report_data, df)

        # Сохраняем
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"📄 HTML-отчёт сохранён: {output_path}")
        return output_path

    # ──────────────────────────────────────────────────────────────────────
    def _get_status_from_pass_rate(self, pass_rate: float) -> str:
        """
        Определяет статус валидации на основе доли успешных тестов (pass_rate).

        Классификация:
        - **PASS**: `pass_rate >= 0.80` — ≥80% тестов прошли пороги качества.
        - **WARNING**: `0.50 <= pass_rate < 0.80` — частичное соответствие, требует анализа.
        - **FAIL**: `pass_rate < 0.50` — <50% тестов прошли, значительные расхождения.

        Этот метод используется для агрегации статусов на уровне метода/пары библиотек
        в отчётах и визуализациях.

        Args:
            pass_rate: Доля успешных тестов в диапазоне [0.0, 1.0].
                    Рассчитывается как `число_PASS / общее_число_тестов`.

        Returns:
            str: Один из трёх статусов: "PASS", "WARNING" или "FAIL".

        Example:
            ```python
            tester = BatchClassicTester()

            status1 = tester._get_status_from_pass_rate(0.95)  # "PASS"
            status2 = tester._get_status_from_pass_rate(0.67)  # "WARNING"
            status3 = tester._get_status_from_pass_rate(0.32)  # "FAIL"

            # Использование в отчёте
            report_data.append({
                "method": "otsu_thresholding",
                "pass_rate": 0.92,
                "validation_status": tester._get_status_from_pass_rate(0.92)  # "PASS"
            })
            ```

        Note:
            - Пороговые значения (0.80 и 0.50) согласованы с `_check_validation_status()`
            для единообразия классификации на разных уровнях агрегации.
            - Метод не проверяет диапазон входного значения; ответственность за корректность
            `pass_rate` лежит на вызывающем коде.
            - Для отладки можно временно изменить пороги, но это может нарушить согласованность
            с другими частями системы.
        """
        if pass_rate >= 0.8:
            return "PASS"
        elif pass_rate >= 0.5:
            return "WARNING"
        else:
            return "FAIL"

    # ──────────────────────────────────────────────────────────────────────
    def _generate_html_content(
        self,
        report_data: List[Dict[str, Any]],
        df: pd.DataFrame,
    ) -> str:
        """
        Генерирует HTML-содержимое интерактивного отчёта о тестировании.

        Структура отчёта:
        1. **Заголовок**: Название, дата генерации, стили (CSS inline).
        2. **Сводная статистика**: 4 карточки с ключевыми метриками:
        - Всего методов / тестов
        - Средний IoU
        - Общая успешность (PASS/WARNING/FAIL с цветовым кодированием)
        3. **Детали по методам**: Для каждой пары библиотек:
        - Заголовок пары
        - Секции методов с метриками (IoU, Dice, Pass Rate, время) и статусом
        - Превью визуализаций (если доступны)
        4. **Полная таблица**: HTML-таблица с основными столбцами для сортировки и фильтрации.

        Аргументы:
            report_data: Список словарей с данными для каждой комбинации (метод × пара),
                        включая метрики, статусы и пути к визуализациям.
            df: Исходный DataFrame для генерации сводной таблицы.

        Returns:
            str: Полная HTML-строка, готовая к записи в файл.

        Note:
            - Все стили (CSS) встроены inline для автономности отчёта.
            - Используется адаптивная сетка (`grid-template-columns: repeat(auto-fit, ...)`)
            для корректного отображения на разных размерах экрана.
            - Цветовая схема: градиенты для карточек, семантические цвета для статусов.
            - Таблица использует класс `dataframe` для совместимости с pandas HTML-экспортом.

        Example:
            ```python
            report_data = [
                {
                    "pair": "torch_vs_opencv",
                    "method": "otsu_thresholding",
                    "iou_mean": 0.94,
                    "pass_rate": 0.95,
                    "validation_status": "PASS",
                    "viz_files": [Path("viz1.jpg")],
                }
            ]
            html = tester._generate_html_content(report_data, df)
            with open("report.html", "w") as f:
                f.write(html)
            ```
        """
        html: str = f"""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Отчёт: Тестирование согласованности методов сегментации</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                }}
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .summary-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .summary-card.pass {{
                    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                }}
                .summary-card.warning {{
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                }}
                .summary-card.fail {{
                    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                }}
                .summary-card h3 {{
                    margin: 0 0 10px 0;
                    font-size: 14px;
                    opacity: 0.9;
                }}
                .summary-card .value {{
                    font-size: 32px;
                    font-weight: bold;
                }}
                .method-section {{
                    margin: 30px 0;
                    padding: 20px;
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    background-color: #fafafa;
                }}
                .method-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #3498db;
                }}
                .method-name {{
                    font-size: 18px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .status-badge {{
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 12px;
                }}
                .status-pass {{
                    background-color: #2ecc71;
                    color: white;
                }}
                .status-warning {{
                    background-color: #f39c12;
                    color: white;
                }}
                .status-fail {{
                    background-color: #e74c3c;
                    color: white;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 10px;
                    margin: 15px 0;
                }}
                .metric-item {{
                    background-color: white;
                    padding: 10px;
                    border-radius: 4px;
                    text-align: center;
                }}
                .metric-label {{
                    font-size: 12px;
                    color: #7f8c8d;
                    margin-bottom: 5px;
                }}
                .metric-value {{
                    font-size: 18px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .preview-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                .preview-item {{
                    background-color: white;
                    padding: 10px;
                    border-radius: 4px;
                    text-align: center;
                }}
                .preview-item img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                .preview-label {{
                    margin-top: 8px;
                    font-size: 12px;
                    color: #7f8c8d;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .timestamp {{
                    color: #7f8c8d;
                    font-size: 14px;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Отчёт: Тестирование согласованности методов сегментации</h1>
                <div class="timestamp">
                    Сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
                <!-- Сводная статистика -->
                <h2>📈 Сводная статистика</h2>
                {self._generate_summary_cards(df)}
                <!-- Детали по методам -->
                <h2>🔍 Детали по методам</h2>
                {self._generate_method_sections(report_data)}
                <!-- Сводная таблица -->
                <h2>📋 Полная таблица результатов</h2>
                {self._generate_results_table(df)}
            </div>
        </body>
        </html>
        """

        return html

    # ──────────────────────────────────────────────────────────────────────
    def _generate_summary_cards(self, df: pd.DataFrame) -> str:
        """
        Генерирует HTML-карточки со сводной статистикой для отчёта.

        Создаёт 4 карточки в адаптивной сетке:
        1. Всего методов
        2. Всего тестов
        3. Средний IoU
        4. Успешность (с цветовым кодированием по статусу)

        Args:
            df: DataFrame с результатами для расчёта агрегированных метрик.

        Returns:
            str: HTML-строка с блоком `.summary-grid`.

        Note:
            - Статус карточки успешности определяется по `avg_pass_rate`:
            ≥80% → "pass" (зелёный), 50–80% → "warning" (оранжевый), <50% → "fail" (красный).
            - Значения форматируются: проценты с 1 знаком, IoU с 3 знаками.
        """
        total_methods: int = len(df)
        total_tests = df["Images_Tested"].sum()
        avg_iou: float = df["iou_mean"].mean()
        avg_pass_rate: float = df["pass_rate"].mean() * 100

        # Определяем общий статус
        if avg_pass_rate >= 80:
            status_class = "pass"
            status_text = "PASS"
        elif avg_pass_rate >= 50:
            status_class = "warning"
            status_text = "WARNING"
        else:
            status_class = "fail"
            status_text = "FAIL"

        cards: str = f"""
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Всего методов</h3>
                <div class="value">{total_methods}</div>
            </div>
            <div class="summary-card">
                <h3>Всего тестов</h3>
                <div class="value">{total_tests}</div>
            </div>
            <div class="summary-card">
                <h3>Средний IoU</h3>
                <div class="value">{avg_iou:.3f}</div>
            </div>
            <div class="summary-card {status_class}">
                <h3>Успешность</h3>
                <div class="value">{avg_pass_rate:.1f}%</div>
                <div>{status_text}</div>
            </div>
        </div>
        """

        return cards

    # ──────────────────────────────────────────────────────────────────────
    def _generate_method_sections(self, report_data: List[Dict[str, Any]]) -> str:
        """
        Генерирует секции с деталями по каждому методу для каждой пары библиотек.

        Для каждой комбинации (пара × метод) создаёт блок с:
        - Заголовком метода и бейджем статуса
        - Сеткой метрик (IoU, Dice, Pass Rate, время A/B)
        - Превью визуализации (если файл существует)

        Args:
            report_data: Список словарей с данными методов и пар.

        Returns:
            str: HTML-строка с блоками `.method-section`.

        Note:
            - Методы группируются по парам библиотек через `defaultdict(list)`.
            - Превью изображений отображаются только если `item["viz_files"]` не пуст.
            - Статус окрашивается через CSS-класс `status-{PASS|WARNING|FAIL}`.
        """
        sections: List[str] = []

        # Группируем по парам библиотек
        pairs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in report_data:
            pair = item["pair"]
            pairs[pair].append(item)

        for pair, items in pairs.items():
            pair_html: str = f"<h3>🔗 {pair.upper()}</h3>"

            for item in items:
                status_class: str = f"status-{item['validation_status'].lower()}"

                # Метрики
                metrics_html: str = f"""
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">IoU</div>
                        <div class="metric-value">{item['iou_mean']:.4f}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Dice</div>
                        <div class="metric-value">{item['dice_mean']:.4f}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Pass Rate</div>
                        <div class="metric-value">{item['pass_rate'] * 100:.1f}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Time A</div>
                        <div class="metric-value">{item['time_a_mean'] * 1000:.1f}ms</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Time B</div>
                        <div class="metric-value">{item['time_b_mean'] * 1000:.1f}ms</div>
                    </div>
                </div>
                """

                # Превью изображений (если есть)
                preview_html: str = ""
                if item["viz_files"]:
                    viz_file = item["viz_files"][0]  # Берём первое найденное
                    preview_html = f"""
                    <div class="preview-grid">
                        <div class="preview-item">
                            <img src="{viz_file.name}" alt="Comparison">
                            <div class="preview-label">Визуализация сравнения</div>
                        </div>
                    </div>
                    """

                section: str = f"""
                <div class="method-section">
                    <div class="method-header">
                        <div class="method-name">{item['method']}</div>
                        <div class="status-badge {status_class}">{item['validation_status']}</div>
                    </div>
                    {metrics_html}
                    {preview_html}
                </div>
                """

                pair_html += section

            sections.append(pair_html)

        return "\n".join(sections)

    # ──────────────────────────────────────────────────────────────────────
    def _generate_results_table(self, df: pd.DataFrame) -> str:
        """
        Генерирует полную таблицу результатов в формате HTML.

        Включает столбцы:
        - Метод, пара библиотек, число изображений
        - Средние значения метрик (IoU, Dice, F1-Score, Pass Rate)
        - Время выполнения для обеих библиотек

        Форматирование:
        - Метрики: 4 знака после запятой или "N/A" при отсутствии данных.
        - Время: конвертация в миллисекунды с 2 знаками (например, "12.34ms").
        - Переименование столбцов на русский для читаемости.

        Args:
            df: DataFrame с агрегированными результатами.

        Returns:
            str: HTML-таблица с классом `dataframe`.

        Note:
            - Используется `df.to_html(escape=False)` для корректного отображения эмодзи.
            - Столбцы фильтруются по наличию в `df` для устойчивости к изменениям схемы.
        """
        # Выбираем ключевые колонки
        cols_to_show: List[str] = [
            "Method",
            "Library_Pair",
            "Images_Tested",
            "iou_mean",
            "dice_mean",
            "f1_score_mean",
            "pass_rate",
            "time_a_mean",
            "time_b_mean",
        ]

        # Фильтруем существующие колонки
        cols: List[str] = [c for c in cols_to_show if c in df.columns]
        df_table: pd.DataFrame = df[cols].copy()

        # Форматируем значения
        for col in ["iou_mean", "dice_mean", "f1_score_mean", "pass_rate"]:
            if col in df_table.columns:
                df_table[col] = df_table[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )

        for col in ["time_a_mean", "time_b_mean"]:
            if col in df_table.columns:
                df_table[col] = df_table[col].apply(
                    lambda x: f"{x * 1000:.2f}ms" if pd.notna(x) else "N/A"
                )

        # Переименовываем для читаемости
        col_names: Dict[str, str] = {
            "Method": "Метод",
            "Library_Pair": "Пара библиотек",
            "Images_Tested": "Изображений",
            "iou_mean": "IoU",
            "dice_mean": "Dice",
            "f1_score_mean": "F1-Score",
            "pass_rate": "Pass Rate",
            "time_a_mean": "Время A",
            "time_b_mean": "Время B",
        }

        df_table = df_table.rename(columns=col_names)

        # Генерируем HTML таблицу
        html_table: str = df_table.to_html(
            index=False, classes="dataframe", escape=False
        )

        return html_table
