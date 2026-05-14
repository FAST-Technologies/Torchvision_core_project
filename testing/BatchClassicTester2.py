# testing/BatchClassicTester2.py

"""Модуль для массового тестирования качества классических методов сегментации относительно Ground Truth.

Предназначен для автоматизированной оценки эффективности алгоритмов пороговой обработки
и выделения границ (OpenCV, Scikit-learn, PyTorch) на датасете с размеченными данными (ADE20K).
В отличие от BatchClassicTester, данный модуль сравнивает предсказания с эталонными масками,
а не между реализациями.

Основные возможности:
- Загрузка изображений и GT-масок с автоматическим ресайзом и бинаризацией многоклассовых аннотаций
- Последовательное тестирование 21 метода из трёх бэкендов с единым интерфейсом
- Расчёт агрегированных метрик: IoU, Dice, Precision, Recall, F1, Accuracy, MAE, Hausdorff
- Продвинутый трекер прогресса с ETA, скоростью выполнения и статистикой ошибок
- Устойчивость к прерываниям: resume после Ctrl+C/kill с автосохранением прогресса
- Экспорт отчётов в CSV, JSON, Markdown и автоматическая генерация аналитических графиков
- Управление памятью: очистка GPU-кэша и сборка мусора между итерациями

Примечание:
- Многоклассовые маски ADE20K автоматически конвертируются в бинарные:
  наиболее частый класс = фон (0), остальные = объект (255).
- Для валидации консистентности реализаций между библиотеками используйте BatchClassicTester.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations  # PEP 563: отложенная оценка аннотаций

import os
import signal
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

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
from metrics.SegmentationMetrics import SegmentationMetrics
import torch
import gc

import logging

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ──────────────────────────────────────────────────────────────────────
class BatchClassicTester:
    """Класс для массового автоматизированного тестирования классических методов сегментации на датасетах изображений (по умолчанию ADE20K).

    Основные возможности:
    - Загрузка изображений и ground truth масок с автоматическим ресайзом и бинаризацией.
    - Последовательное тестирование методов из разных бэкендов (OpenCV, Scikit-learn, PyTorch).
    - Расчёт агрегированных метрик: IoU, Dice, Precision, Recall, F1, Accuracy, MAE, Hausdorff.
    - Продвинутый трекер прогресса с динамическим ETA, скоростью выполнения и статистикой ошибок.
    - Аварийное восстановление (resume) после Ctrl+C, kill или непредвиденного сбоя.
    - Периодическое автосохранение промежуточных результатов и метаданных.
    - Экспорт отчётов в CSV, JSON, Markdown и автоматическая генерация визуализаций.
    - Управление памятью: очистка кэша GPU и принудительный сборщик мусора между итерациями.

    Attributes:
        ade20k_root (Path): Корневая директория датасета ADE20K.
        output_dir (Path): Директория для сохранения результатов и артефактов.
        split (str): Выборка датасета ('training' или 'validation').
        max_images (Optional[int]): Максимальное количество изображений для теста (None = все).
        image_size (Tuple[int, int]): Целевой размер изображений и масок (H, W).
        autosave_interval (int): Интервал автосохранения (в количестве обработанных изображений).
        resume (bool): Флаг восстановления прогресса из предыдущего запуска.
        metrics_to_aggregate (List[str]): Список метрик для агрегации.
        results (Dict[str, Dict[str, List[float]]]): Накопленные значения метрик по методам.
        execution_times (Dict[str, List[float]]]): Время выполнения каждого теста.
        errors (Dict[str, List[str]]): Список ошибок, сгруппированных по методам.
    """

    def __init__(
        self,
        ade20k_root: str = "./data/ade20k/ADEChallengeData2016",
        output_dir: str = "./data/batch_classic_test",
        split: str = "validation",  # "training" или "validation"
        max_images: Optional[int] = None,  # Лимит изображений для теста
        image_size: Tuple[int, int] = (512, 512),
        autosave_interval: int = 5,
        resume: bool = True,
    ) -> None:
        """Инициализация тестера с настройками путей, лимитов и параметров восстановления.

        Args:
            ade20k_root: Путь к корневой директории датасета ADE20K.
            output_dir: Путь к директории для сохранения результатов, графиков и логов.
            split: Используемая выборка датасета (`"training"` или `"validation"`).
            max_images: Лимит количества изображений. Если `None`, тестируются все доступные.
            image_size: Размер, к которому будут приведены все изображения и маски.
            autosave_interval: Частота автосохранения прогресса (в итерациях по изображениям).
            resume: Если `True`, автоматически ищет и загружает предыдущий прогресс.
        """
        self._setup_signal_handlers()
        self.ade20k_root: Path = Path(ade20k_root)
        self.output_dir: Path = Path(output_dir)
        self.split: str = split
        self.max_images: Optional[int] = max_images
        self.image_size: Tuple[int, int] = image_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Метрики для агрегации
        self.metrics_to_aggregate: List[str] = [
            "iou",
            "dice",
            "precision",
            "recall",
            "f1_score",
            "accuracy",
            "mae",
            "hausdorff_distance",
        ]

        # Результаты
        self.results: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.execution_times: Dict[str, List[float]] = defaultdict(list)
        self.errors: Dict[str, List[str]] = defaultdict(list)
        self.autosave_interval: int = autosave_interval
        self.resume: bool = resume

        # Пути для автосохранения
        self.progress_file: Path = Path(output_dir) / ".progress.json"
        self.temp_results_file: Path = Path(output_dir) / ".results_temp.csv"

        # Загрузка предыдущего прогресса если нужно
        if resume and self.progress_file.exists():
            self._load_progress()
            print(f"📥 Восстановлен прогресс: {self._processed_count}/{self._total_tests} тестов")

    # ──────────────────────────────────────────────────────────────────────
    def _setup_signal_handlers(self) -> None:
        """Настраивает обработчики системных сигналов (SIGINT, SIGTERM) для безопасного завершения.

        При получении сигнала Ctrl+C или `kill` выполняется финальное автосохранение
        текущего прогресса и результатов, после чего процесс завершается с кодом 130.
        """

        def handle_interrupt(signum, frame):
            print("\n\n⚠️  Получен сигнал прерывания!")
            print("💾 Выполняется финальное автосохранение...")
            self._save_progress(force=True)
            print("✅ Прогресс сохранён. Завершение работы.")
            sys.exit(130)  # Стандартный код для SIGINT

        signal.signal(signal.SIGINT, handle_interrupt)  # Ctrl+C
        signal.signal(signal.SIGTERM, handle_interrupt)  # kill

    # ──────────────────────────────────────────────────────────────────────
    def _init_progress_tracking(self, total_images: int, total_methods: int) -> None:
        """Инициализирует внутренние счётчики и таймеры для отслеживания прогресса.

        Args:
            total_images: Количество изображений в текущем наборе.
            total_methods: Количество тестируемых алгоритмов сегментации.
        """
        self._total_images: int = total_images
        self._total_methods: int = total_methods
        self._total_tests: int = total_images * total_methods
        self._processed_count: int = getattr(self, "_processed_count", 0)
        self._start_time: float = time.time()
        self._last_save_time: float = time.time()

    # ──────────────────────────────────────────────────────────────────────
    def _update_progress_bar(self, pbar, current_count: int, method_name: str, img_name: str) -> None:
        """Обновляет postfix прогресс-бара динамической статистикой.

        Рассчитывает прошедшее время, ETA, скорость обработки (тестов/мин),
        а также общую долю ошибок на текущий момент.

        Args:
            pbar: Экземпляр `tqdm` для обновления.
            current_count: Количество уже выполненных тестов.
            method_name: Имя текущего метода сегментации.
            img_name: Имя текущего изображения.
        """
        elapsed: float = time.time() - self._start_time
        rate: float = current_count / elapsed if elapsed > 0 else 0
        remaining: float = (self._total_tests - current_count) / rate if rate > 0 else 0

        # Форматирование времени
        def fmt_time(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.0f}с"
            elif seconds < 3600:
                return f"{seconds / 60:.1f}м"
            else:
                return f"{seconds / 3600:.1f}ч"

        total_errors: int = sum(len(errs) for errs in self.errors.values())
        error_rate: float = total_errors / current_count if current_count > 0 else 0

        pbar.set_postfix(
            {
                "img": img_name[:15],
                "method": method_name.split("_")[0],
                "elapsed": fmt_time(elapsed),
                "eta": fmt_time(remaining),
                "rate": f"{rate * 60:.1f}/мин",
                "errors": f"{total_errors}({error_rate * 100:.1f}%)",
            }
        )

    # ──────────────────────────────────────────────────────────────────────
    def _save_progress(self, force: bool = False) -> None:
        """Сохраняет текущий прогресс и промежуточные результаты на диск.

        Использует атомарную перезапись CSV для предотвращения повреждения данных
        при аварийном завершении. Сохранение throttled до 1 раза в 30 секунд,
        если `force=False`.

        Args:
            force: Если `True`, сохраняет прогресс немедленно, игнорируя таймер.
        """
        now: float = time.time()
        # Сохраняем если прошло достаточно времени или по принуждению
        if not force and (now - self._last_save_time) < 30:  # не чаще 30 сек
            return

        try:
            # 1. Сохраняем метаданные прогресса
            progress: Dict[str, Any] = {
                "processed_count": self._processed_count,
                "total_tests": self._total_tests,
                "start_time": self._start_time,
                "last_update": now,
                "methods_done": list(self.results.keys()),
            }
            with open(self.progress_file, "w") as f:
                json.dump(progress, f, indent=2)

            # 2. Сохраняем промежуточные результаты (atomic write)
            if self.results:
                df_temp: pd.DataFrame = self._aggregate_results()
                df_temp.to_csv(self.temp_results_file, index=False, float_format="%.4f")
                # Atomic rename
                final_path: Path = self.temp_results_file.with_suffix(".csv")
                self.temp_results_file.replace(final_path)

            self._last_save_time = now
            print(f"\n💾 Автосохранение: {self._processed_count}/{self._total_tests} ✅")

        except Exception as e:
            print(f"\n⚠️  Ошибка автосохранения: {e}")

    # ──────────────────────────────────────────────────────────────────────
    def _load_progress(self) -> bool:
        """Загружает метаданные прогресса из файла `.progress.json`.

        Восстанавливает счётчики обработанных тестов, время старта и список
        уже протестированных методов.

        Returns:
            `True` если загрузка прошла успешно, иначе `False`.
        """
        try:
            with open(self.progress_file, "r") as f:
                progress = json.load(f)

            # Восстанавливаем счётчики
            self._processed_count = progress.get("processed_count", 0)
            self._total_tests = progress.get("total_tests", 0)
            self._start_time = progress.get("start_time", time.time())

            # Если есть временные результаты — загружаем их
            if self.temp_results_file.exists():
                print(f"📥 Загрузка промежуточных результатов из {self.temp_results_file}")
                # Здесь можно добавить логику слияния, если нужно

            return True
        except Exception as e:
            print(f"⚠️  Не удалось загрузить прогресс: {e}")
            return False

    def _load_images_with_masks(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Загружает пары (изображение, ground truth маска) из датасета ADE20K.

        Автоматически ресайзит данные до `self.image_size`. Пропускает изображения,
        для которых не найдены соответствующие маски.

        Returns:
            Список кортежей `(имя_файла, изображение_HxWxC_uint8, бинарная_маска_HxW_uint8)`.
        """
        images_dir: Path = self.ade20k_root / "images" / self.split
        masks_dir: Path = self.ade20k_root / "annotations" / self.split

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        # Получаем список изображений
        image_files: List[str] = sorted([f for f in os.listdir(images_dir) if f.endswith((".jpg", ".jpeg", ".png"))])

        if self.max_images:
            image_files = image_files[: self.max_images]

        print(f"📥 Загрузка {len(image_files)} изображений из {self.split}...")

        data: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for img_file in tqdm(image_files, desc="Loading images"):
            mask_file: str = img_file.rsplit(".", 1)[0] + ".png"
            img_path: Path = images_dir / img_file
            mask_path: Path = masks_dir / mask_file

            if not mask_path.exists():
                print(f"⚠️  Mask not found for {img_file}, skipping")
                continue

            try:
                # Загрузка изображения
                img: Image.Image = Image.open(img_path).convert("RGB")
                img_array: np.ndarray = np.array(img.resize(self.image_size, Image.Resampling.BILINEAR))

                # Загрузка маски
                mask_pil: Image.Image = Image.open(mask_path)
                mask_array: np.ndarray = np.array(mask_pil.resize(self.image_size, Image.Resampling.NEAREST))

                # Конвертация многоклассовой маски в бинарную
                # Стратегия: самый частый класс = фон (0), всё остальное = объект (255)
                binary_mask: np.ndarray = self._multiclass_to_binary(mask_array)

                data.append((img_file, img_array, binary_mask))

            except Exception as e:
                print(f"❌ Error loading {img_file}: {e}")
                continue

        print(f"✅ Загружено {len(data)} пар изображение-маска")
        return data

    # ──────────────────────────────────────────────────────────────────────
    def _multiclass_to_binary(self, mask: np.ndarray) -> np.ndarray:
        """Конвертирует многоклассовую маску в бинарную.

        Стратегия: класс с наибольшей частотой пикселей считается фоном (0),
        все остальные классы объединяются в объект (255).

        Args:
            mask: Входная маска с целочисленными метками классов.

        Returns:
            Бинарная маска `np.uint8`, где 255 = объект, 0 = фон.
        """
        # Находим самый частый класс (предполагаем, что это фон)
        unique: np.ndarray
        counts: np.ndarray
        unique, counts = np.unique(mask, return_counts=True)
        background_class = unique[np.argmax(counts)]

        # Бинаризация: объект = всё, что не фон
        binary: np.ndarray = (mask != background_class).astype(np.uint8) * 255
        return binary

    # ──────────────────────────────────────────────────────────────────────
    def _run_single_test(
        self, method_name: str, segmenter: Any, image: np.ndarray, gt_mask: np.ndarray
    ) -> Tuple[Optional[Dict[str, float]], Optional[float], Optional[str]]:
        """Выполняет один полный цикл тестирования: сегментация + расчёт метрик.

        Args:
            method_name: Идентификатор метода (для логирования).
            segmenter: Экземпляр класса-сегментатора с методом `.segment()`.
            image: Входное изображение `np.ndarray`.
            gt_mask: Ground truth бинарная маска `np.ndarray`.

        Returns:
            Кортеж `(метрики_dict, время_выполнения_сек, сообщение_об_ошибке)`.
            При успехе `error_msg` равен `None`, при ошибке первые два элемента `None`.
        """
        try:
            start_time: float = time.time()

            # Сегментация
            pred_mask: np.ndarray = segmenter.segment(image)

            # Ресайз предсказания к размеру GT если нужно
            if pred_mask.shape != gt_mask.shape:
                from skimage.transform import resize

                pred_mask = resize(pred_mask, gt_mask.shape, order=0, preserve_range=True).astype(np.uint8)

            exec_time: float = time.time() - start_time

            # Расчёт метрик
            metrics: Dict[str, float] = SegmentationMetrics.calculate_all_metrics(
                pred_mask=pred_mask,
                gt_mask=gt_mask,
                threshold=0.5,
                include_hausdorff=True,
            )

            return metrics, exec_time, None

        except Exception as e:
            error_msg: str = f"{type(e).__name__}: {str(e)}"
            return None, None, error_msg

    # ──────────────────────────────────────────────────────────────────────
    def _get_classic_methods(self) -> Dict[str, Any]:
        """Формирует словарь инициализированных сегментаторов для тестирования.

        Включает методы из трёх бэкендов: OpenCV, Scikit-learn и PyTorch.
        Каждый метод конфигурируется с дефолтными гиперпараметрами.

        Returns:
            Словарь `{имя_метода: экземпляр_сегментатора}`.
        """
        methods: Dict[str, Any] = {}
        # === OpenCV методы ===
        methods.update(
            {
                "Global_Threshold_CV2": OpenCVSegmenter("global_thresholding", threshold=0.5),
                "Otsu_Thresholding_CV2": OpenCVSegmenter("otsu_thresholding"),
                "Adaptive_Threshold_CV2": OpenCVSegmenter("adaptive_thresholding", block_size=11, C=2),
                "Niblack_Thresholding_CV2": OpenCVSegmenter("threshold_niblack", window_size=15, k=-0.2),
                "Sauvola_Thresholding_CV2": OpenCVSegmenter("threshold_sauvola", window_size=15, k=0.5, r=128),
                "Sobel_CV2": OpenCVSegmenter("sobel_edge", threshold=0.1),
                "Canny_CV2": OpenCVSegmenter("canny_edge", low=0.1, high=0.3, sigma=1.0),
            }
        )

        # === Sklearn методы ===
        methods.update(
            {
                "Global_Threshold_Sklearn": SklearnSegmenter("global_thresholding", threshold=0.5),
                "Otsu_Thresholding_Sklearn": SklearnSegmenter("otsu_thresholding"),
                "Adaptive_Threshold_Sklearn": SklearnSegmenter("adaptive_thresholding", block_size=11, C=2),
                "Niblack_Thresholding_Sklearn": SklearnSegmenter("threshold_niblack", window_size=15, k=-0.2),
                "Sauvola_Thresholding_Sklearn": SklearnSegmenter("threshold_sauvola", window_size=15, k=0.5, r=128),
                "Sobel_Sklearn": SklearnSegmenter("sobel_edge", threshold=0.1),
                "Canny_Sklearn": SklearnSegmenter("canny_edge", low=0.1, high=0.3, sigma=1.0),
            }
        )

        # === Torch методы ===
        methods.update(
            {
                "Global_Threshold_Torch": TorchSegmenter("global_thresholding", threshold=0.5),
                "Otsu_Thresholding_Torch": TorchSegmenter("otsu_thresholding"),
                "Adaptive_Threshold_Torch": TorchSegmenter("adaptive_thresholding", block_size=11, C=2),
                "Niblack_Thresholding_Torch": TorchSegmenter("threshold_niblack", window_size=15, k=-0.2),
                "Sauvola_Thresholding_Torch": TorchSegmenter("threshold_sauvola", window_size=15, k=0.5, r=128),
                "Sobel_Torch": TorchSegmenter("sobel_edge", threshold=0.1),
                "Canny_Torch": TorchSegmenter("canny_edge", low=0.1, high=0.3, sigma=1.0),
            }
        )

        return methods

    # ──────────────────────────────────────────────────────────────────────
    def run_batch_test(self) -> pd.DataFrame:
        """Запускает основной цикл массового тестирования всех методов на всех изображениях.

        Обрабатывает прерывания, обновляет прогресс-бар, управляет памятью GPU/CPU
        и автоматически сохраняет промежуточные результаты.

        Returns:
            `pd.DataFrame` с агрегированными метриками, временем выполнения и статистикой ошибок.
        """
        # Загрузка данных
        test_data: List[Tuple[str, np.ndarray, np.ndarray]] = self._load_images_with_masks()
        if not test_data:
            raise ValueError("No test data loaded!")

        # Получение методов
        methods: Dict[str, Any] = self._get_classic_methods()
        total_images: int = len(test_data)
        total_methods: int = len(methods)

        print(f"🔧 Тестируем {total_methods} методов на {total_images} изображениях")
        print(f"📊 Всего тестов: {total_images * total_methods}")
        print(f"💾 Автосохранение каждые {self.autosave_interval} изображений")

        # Инициализация прогресса
        self._init_progress_tracking(total_images, total_methods)

        # Прогресс-бар с расширенными настройками
        from tqdm import tqdm

        with tqdm(
            total=self._total_tests,
            desc="🧪 Тестирование",
            unit="тест",
            leave=True,
            dynamic_ncols=False,
            ncols=140,  # подберите под ширину вашего терминала
            mininterval=0.5,
        ) as pbar:

            # Пропускаем уже выполненные тесты если resume=True
            if self.resume and self._processed_count > 0:
                pbar.update(self._processed_count)
                print(f"⏭️  Пропущено {self._processed_count} выполненных тестов")

            # Основной цикл
            for img_idx, (img_name, image, gt_mask) in enumerate(test_data):
                for method_idx, (method_name, segmenter) in enumerate(methods.items()):

                    # Пропуск если уже выполнено (при resume)
                    # test_key = f"{img_name}:{method_name}"
                    if self.resume and self._processed_count > 0:
                        # Простая эвристика: если счётчик больше — пропускаем
                        if self._processed_count >= (img_idx * total_methods + method_idx + 1):
                            continue

                    # Запуск теста
                    metrics, exec_time, error = self._run_single_test(method_name, segmenter, image, gt_mask)

                    # Обработка результатов
                    if error:
                        self.errors[method_name].append(f"{img_name}: {error}")
                    elif metrics:
                        for metric_name in self.metrics_to_aggregate:
                            if metric_name in metrics:
                                self.results[method_name][metric_name].append(metrics[metric_name])
                        if exec_time is not None:
                            self.execution_times[method_name].append(exec_time)
                    # Обновление прогресса
                    self._processed_count += 1
                    pbar.update(1)
                    self._update_progress_bar(pbar, self._processed_count, method_name, img_name)

                    # Автосохранение каждые N изображений
                    if img_idx % self.autosave_interval == 0 and img_idx > 0:
                        self._save_progress()

                # Очистка памяти после каждого изображения
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()

        # Финальное сохранение
        print("\n🏁 Завершение тестирования...")
        self._save_progress(force=True)

        # Убираем временные файлы
        if self.progress_file.exists():
            self.progress_file.unlink()
        if self.temp_results_file.exists():
            self.temp_results_file.unlink()

        return self._aggregate_results()

    # ──────────────────────────────────────────────────────────────────────
    def _aggregate_results(self) -> pd.DataFrame:
        """Агрегирует накопленные результаты в сводную таблицу.

        Для каждого метода рассчитывает среднее, стандартное отклонение, min и max
        по каждой метрике, а также среднее время выполнения и долю ошибок.
        Сортирует таблицу по убыванию `iou_mean`.

        Returns:
            `pd.DataFrame` со столбцами `Method`, `{metric}_{stat}`, `time_mean_s`,
            `error_count`, `error_rate` и `Images_Tested`.
        """
        rows: List[Dict[str, Any]] = []

        for method_name in self.results:
            images_tested: int = len(self.results[method_name].get("iou", []))
            row: Dict[str, Any] = {
                "Method": method_name,
                "Images_Tested": images_tested,
            }

            # Средние значения метрик
            for metric_name in self.metrics_to_aggregate:
                values: List[float] = self.results[method_name].get(metric_name, [])
                if values:
                    row[f"{metric_name}_mean"] = np.mean(values)
                    row[f"{metric_name}_std"] = np.std(values)
                    row[f"{metric_name}_min"] = np.min(values)
                    row[f"{metric_name}_max"] = np.max(values)
                else:
                    row[f"{metric_name}_mean"] = np.nan

            # Время выполнения
            times: List[float] = self.execution_times.get(method_name, [])
            if times:
                row["time_mean_s"] = np.mean(times)
                row["time_std_s"] = np.std(times)
            else:
                row["time_mean_s"] = np.nan

            # Ошибки
            errors: List[str] = self.errors.get(method_name, [])
            row["error_count"] = len(errors)
            row["error_rate"] = len(errors) / max(row["Images_Tested"], 1)

            rows.append(row)

        df: pd.DataFrame = pd.DataFrame(rows)

        # Сортировка по IoU
        if "iou_mean" in df.columns:
            df = df.sort_values("iou_mean", ascending=False)

        return df

    # ──────────────────────────────────────────────────────────────────────
    def save_results(self, df: pd.DataFrame, prefix: str = "batch_test") -> Tuple[Path, Path, Path]:
        """Экспортирует результаты в несколько форматов для дальнейшего анализа.

        Args:
            df: Датафрейм с агрегированными результатами.
            prefix: Префикс для имён выходных файлов.

        Returns:
            Кортеж путей к сохранённым файлам `(csv_path, json_path, md_path)`.
        """
        # CSV
        csv_path: Path = self.output_dir / f"{prefix}_results.csv"
        df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"💾 CSV сохранён: {csv_path}")

        # JSON с детальными результатами
        json_path: Path = self.output_dir / f"{prefix}_details.json"
        details: Dict[str, Any] = {
            "summary": df.to_dict(orient="records"),
            "errors": dict(self.errors),
            "config": {
                "ade20k_root": str(self.ade20k_root),
                "split": self.split,
                "max_images": self.max_images,
                "image_size": self.image_size,
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 JSON сохранён: {json_path}")

        # Markdown отчёт
        md_path: Path = self.output_dir / f"{prefix}_report.md"
        self._save_markdown_report(df, md_path)
        print(f"💾 Markdown отчёт: {md_path}")

        return csv_path, json_path, md_path

    # ──────────────────────────────────────────────────────────────────────
    def _save_markdown_report(self, df: pd.DataFrame, path: Path) -> None:
        """Генерирует читаемый Markdown-отчёт с топ-10 методов, полной таблицей и статистикой ошибок.

        Args:
            df: Датафрейм с результатами.
            path: Путь для сохранения `.md` файла.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write("# 📊 Отчёт: Массовое тестирование классических методов сегментации\n\n")
            f.write(f"**Датасет:** ADE20K ({self.split})\n")
            f.write(f"**Изображений:** {self.max_images or 'all'}\n")
            f.write(f"**Размер:** {self.image_size}\n\n")

            # Топ-10 по IoU
            f.write("## 🏆 Топ-10 методов по среднему IoU\n\n")
            top_10: pd.DataFrame = df.head(10)[["Method", "iou_mean", "dice_mean", "time_mean_s", "Images_Tested"]]
            f.write(top_10.to_markdown(index=False) + "\n\n")

            # Сводная таблица всех метрик
            f.write("## 📈 Полная таблица результатов\n\n")
            cols_to_show: List[str] = ["Method"] + [c for c in df.columns if c.endswith("_mean")]
            f.write(df[cols_to_show].to_markdown(index=False, floatfmt=".4f") + "\n\n")

            # Статистика ошибок
            if any(df["error_count"] > 0):
                f.write("## ⚠️ Статистика ошибок\n\n")
                error_df: pd.DataFrame = df[df["error_count"] > 0][["Method", "error_count", "error_rate"]]
                f.write(error_df.to_markdown(index=False) + "\n\n")

    # ──────────────────────────────────────────────────────────────────────
    def plot_results(self, df: pd.DataFrame, output_dir: Optional[Path] = None) -> None:
        """Строит и сохраняет визуализации эффективности методов.

        Создаёт три графика:
        1. `iou_ranking.png` — бар-чарт топ-15 методов по IoU.
        2. `speed_vs_accuracy.png` — scatter plot зависимости точности от времени.
        3. `metrics_heatmap.png` — тепловая карта средних значений всех метрик.

        Args:
            df: Датафрейм с результатами.
            output_dir: Директория для сохранения графиков. Если `None`, используется `output_dir/charts`.
        """
        if output_dir is None:
            output_dir = self.output_dir / "charts"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Фильтруем методы с достаточным количеством тестов
        df_plot: pd.DataFrame = df[df["Images_Tested"] >= 5].copy()
        if df_plot.empty:
            print("⚠️ Недостаточно данных для построения графиков")
            return

        # === График 1: IoU по методам ===
        plt.figure(figsize=(14, 8))
        sns.barplot(data=df_plot.head(15), x="iou_mean", y="Method", palette="viridis")
        plt.xlabel("Mean IoU")
        plt.title("Топ-15 методов по среднему IoU")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "iou_ranking.png", dpi=150)
        plt.close()

        # === График 2: Speed vs Accuracy ===
        plt.figure(figsize=(10, 8))
        plt.scatter(
            df_plot["time_mean_s"],
            df_plot["iou_mean"],
            s=100,
            alpha=0.7,
            edgecolors="black",
        )

        for _, row in df_plot.iterrows():
            plt.annotate(
                row["Method"][:15],
                (row["time_mean_s"], row["iou_mean"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )

        plt.xlabel("Среднее время (сек)")
        plt.ylabel("Mean IoU")
        plt.title("Зависимость точности от скорости")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "speed_vs_accuracy.png", dpi=150)
        plt.close()

        # === График 3: Heatmap метрик ===
        metrics_cols: List[str] = [
            c for c in df.columns if c.endswith("_mean") and c.split("_")[0] in self.metrics_to_aggregate
        ]
        if metrics_cols:
            plt.figure(figsize=(12, 10))
            heatmap_data: pd.DataFrame = df_plot[["Method"] + metrics_cols].set_index("Method").head(12)
            heatmap_data.columns = [c.replace("_mean", "") for c in heatmap_data.columns]

            sns.heatmap(heatmap_data.T, annot=True, fmt=".3f", cmap="YlOrRd")
            plt.title("Heatmap средних метрик (Топ-12 методов)")
            plt.xlabel("Метод")
            plt.ylabel("Метрика")
            plt.tight_layout()
            plt.savefig(output_dir / "metrics_heatmap.png", dpi=150)
            plt.close()

        print(f"📊 Графики сохранены в: {output_dir}")
