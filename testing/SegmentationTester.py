# testing/SegmentationTester.py

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import (
    List,
    Union,
    Tuple,
    Dict,
    Any,
    Optional,
)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import logging

logger = logging.getLogger(__name__)

# Локальные импорты
from segmenters.BaseSegmenter import BaseSegmenter, BinaryMask, ProbabilityMask
from metrics.SegmentationMetrics import SegmentationMetrics, MetricsDict
from utils.warmup import SegmentationWarmUp

# Настройка путей проекта
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
ArrayLike = Union[np.ndarray, List[Any]]
ImageInput = Union[str, np.ndarray, Image.Image]
MaskArray = np.ndarray  # Binary mask: HxW, dtype uint8/bool
ImageArray = np.ndarray  # RGB/Grayscale: HxW or HxWxC
MetricDict = Dict[str, float]
PathLike = Union[str, Path]
StatsList = List[Dict[str, Any]]


# ──────────────────────────────────────────────────────────────────────
class SegmentationTester:
    """
    Универсальный класс для тестирования, сравнения и бенчмаркинга методов сегментации.

    Основные возможности:
    - Регистрация произвольных сегментеров, наследующих `BaseSegmenter`.
    - Поочерёдное тестирование с замером времени и сохранением артефактов.
    - Автоматический расчёт метрик качества (IoU, Dice, F1, Hausdorff) при наличии GT.
    - Пакетное сравнение методов с визуализацией в grid-формате.
    - Бенчмарк с многократными прогонами, warm-up и статистикой (mean/std/min/max).
    - Экспорт результатов: изображения, маски, overlay, JSON, CSV, TXT, HTML-отчёты.
    - Управление памятью: очистка временных файлов, кэширование результатов.

    Workflow:
    1. Создать экземпляр → 2. Добавить методы через `add_method()` → 3. Загрузить GT (опционально)
       → 4. Вызвать `compare_methods()` / `benchmark_methods()` → 5. Экспортировать отчёт.

    Attributes:
        methods (Dict[str, BaseSegmenter]): Реестр зарегистрированных сегментеров.
        results (Dict[str, Dict]): Кэш результатов по тестам.
        base_output_dir (Path): Базовая директория для сохранения артефактов.
        current_test_id (Optional[str]): ID последнего запущенного теста.
        ground_truth_mask (Optional[MaskArray]): Загруженная маска для расчёта метрик.
        enable_warmup (bool): Флаг включения предварительного "прогрева" моделей.
        n_warmup_runs (int): Количество warm-up прогонов перед основным бенчмарком.
        warmup_utility (SegmentationWarmUp): Утилита для выполнения warm-up.
        warmup_completed (Dict[str, bool]): Трекер выполненных warm-up по методам.
    """

    def __init__(
        self,
        base_output_dir: PathLike = "./../data/segmentation_results",
        ground_truth_path: Optional[PathLike] = None,
        enable_warmup: bool = True,
        n_warmup_runs: int = 3,
    ) -> None:
        """
        Инициализация тестера с настройками путей и параметров тестирования.

        Args:
            base_output_dir: Базовая директория для сохранения результатов.
            ground_truth_path: Путь к ground truth маске (опционально).
            enable_warmup: Если `True`, выполняет warm-up перед первым прогоном каждого метода.
            n_warmup_runs: Количество "разогревочных" итераций для стабилизации производительности.
        """
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.methods: Dict[str, BaseSegmenter] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.base_output_dir: str = str(base_output_dir)
        self.current_test_id: Optional[str] = None
        self.ground_truth_path: Optional[str] = (
            str(ground_truth_path) if ground_truth_path else None
        )
        self.ground_truth_mask: Optional[MaskArray] = None
        self.enable_warmup: bool = enable_warmup
        self.n_warmup_runs: int = n_warmup_runs
        self.warmup_utility = SegmentationWarmUp(n_warmup_runs=n_warmup_runs)
        self.warmup_completed: Dict[str, bool] = {}
        self.last_benchmark_dirs: Dict[str, str] = {}
        self._benchmark_cache: Dict[str, Dict[str, str]] = {}
        if ground_truth_path:
            self.load_ground_truth(ground_truth_path)

    # ──────────────────────────────────────────────────────────────────────
    def load_ground_truth(self, gt_path: PathLike) -> None:
        """
        Загружает ground truth маску из файла.

        Поддерживаемые форматы:
        - Изображения: `.png`, `.jpg`, `.jpeg`, `.bmp` (читаются через OpenCV в grayscale).
        - NumPy-массивы: `.npy` (загружаются через `np.load`).

        Args:
            gt_path: Путь к файлу с маской.

        Note:
            - Если загрузка не удалась, `self.ground_truth_mask` остаётся `None`,
              и метрики качества не будут рассчитываться.
            - Для цветных GT-масок предполагается, что они уже бинаризованы или
              будут обработаны внешним кодом перед передачей в метрики.
        """
        try:
            gt_path_str: str = str(gt_path)
            if gt_path_str.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                self.ground_truth_mask = cv2.imread(gt_path_str, cv2.IMREAD_GRAYSCALE)
            elif gt_path_str.endswith(".npy"):
                self.ground_truth_mask = np.load(gt_path_str)
            else:
                raise ValueError(f"Неизвестный формат ground truth: {gt_path_str}")

            if self.ground_truth_mask is None:
                raise ValueError("Не удалось прочитать файл (возвращён None)")
            print(f"✅ Ground truth загружен: {gt_path_str}")
        except Exception as e:
            print(f"❌ Ошибка загрузки ground truth: {e}")
            self.ground_truth_mask = None

    # ──────────────────────────────────────────────────────────────────────
    def _ensure_warmup(
        self, method_name: str, segmenter: BaseSegmenter, image: ImageArray
    ) -> None:
        """
        Гарантирует выполнение warm-up перед бенчмарком метода.

        Args:
            method_name: Идентификатор метода в реестре.
            segmenter: Экземпляр сегментера для "прогрева".
            image: Реальное изображение для инициализации кэшей/памяти модели.
        """
        if not self.enable_warmup:
            return

        if (
            method_name not in self.warmup_completed
            or not self.warmup_completed[method_name]
        ):
            print(f"\n🔥 Выполняем warm-up для {method_name}...")
            self.warmup_utility.warmup_segmenter(
                segmenter=segmenter,
                method_name=method_name,
                real_image=image,
                verbose=True,
                use_real_image=True,
            )
            self.warmup_completed[method_name] = True

    # ──────────────────────────────────────────────────────────────────────
    def _create_test_directory(self, test_name: Optional[str] = None) -> str:
        """
        Создаёт уникальную иерархию директорий для хранения результатов теста.

        Структура:
        ```
        {base_output_dir}/{test_name}_{timestamp}/
        ├── images/          # Оригиналы, результаты, overlay
        ├── masks/           # Бинарные маски
        ├── comparisons/     # Сводные графики сравнения
        └── statistics/      # JSON/CSV/TXT отчёты
        ```

        Args:
            test_name: Префикс имени теста. Если `None`, используется `"test"`.

        Returns:
            str: Полный путь к созданной директории.
        """
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir: str
        if test_name:
            test_dir = f"{test_name}_{timestamp}"
        else:
            test_dir = f"test_{timestamp}"

        full_path: str = os.path.join(self.base_output_dir, test_dir)
        os.makedirs(full_path, exist_ok=True)
        os.makedirs(os.path.join(full_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "masks"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "comparisons"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "statistics"), exist_ok=True)
        self._benchmark_cache[test_dir] = {
            "root": full_path,
            "statistics": os.path.join(full_path, "statistics"),
            "images": os.path.join(full_path, "images"),
            "masks": os.path.join(full_path, "masks"),
            "comparisons": os.path.join(full_path, "comparisons"),
        }
        self.current_test_id = test_dir
        print(f"📁 Создана директория для теста: {full_path}")
        return str(full_path)

    # ──────────────────────────────────────────────────────────────────────
    def get_benchmark_path(
        self, test_name: str, file_type: str = "root"
    ) -> Optional[str]:
        """Получить путь к файлам бенчмарка по имени теста."""
        return self._benchmark_cache.get(test_name, {}).get(file_type)

    # ──────────────────────────────────────────────────────────────────────
    def add_method(self, name: str, segmenter: BaseSegmenter) -> None:
        """
        Регистрирует новый метод сегментации в тестере.

        Args:
            name: Уникальное имя метода для доступа в отчётах.
            segmenter: Экземпляр класса, наследующего `BaseSegmenter`.
        """
        self.methods[name] = segmenter

    # ──────────────────────────────────────────────────────────────────────
    def test_single_method(
        self,
        image: ImageInput,
        method_name: str,
        save_path: Optional[PathLike] = None,
        output_dir: Optional[PathLike] = None,
    ) -> Dict[str, Any]:
        """
        Выполняет тестирование одного метода с сохранением результатов.

        Логика:
        1. Замер времени выполнения `segment_with_mask()`.
        2. Конвертация входного изображения в `PIL.Image` для сохранения.
        3. Расчёт базовой статистики маски (площадь, процент покрытия).
        4. Сохранение артефактов: оригинал, результат, маска, overlay (если `output_dir` указан).

        Args:
            image: Входное изображение (путь, `np.ndarray` или `PIL.Image`).
            method_name: Ключ метода из реестра `self.methods`.
            save_path: Путь для сохранения только результата (если `output_dir` не указан).
            output_dir: Директория для сохранения полного набора артефактов.

        Returns:
            Dict[str, Any]: Словарь с результатами:
            - `method`, `result`, `mask`, `time`
            - `mask_area`, `mask_percentage`, `image_shape`, `timestamp`
            - Опционально: `result_path`, `mask_path`, `overlay_path`, `original_path`

        Raises:
            ValueError: Если `method_name` отсутствует в реестре.
        """
        if method_name not in self.methods:
            raise ValueError(f"Метод {method_name} не найден")

        segmenter = self.methods[method_name]

        # Замер времени
        if str(self.device) == "cuda":
            torch.cuda.synchronize()
        start_time: float = time.perf_counter()
        result_opt: BinaryMask
        mask_opt: Optional[ProbabilityMask]
        result_opt, mask_opt = segmenter.segment_with_mask(image)
        if result_opt is None or mask_opt is None:
            raise ValueError(f"{method_name}.segment_with_mask() returned None")
        result: np.ndarray = result_opt
        mask: np.ndarray = mask_opt
        if str(self.device) == "cuda":
            torch.cuda.synchronize()
        execution_time: float = time.perf_counter() - start_time

        # Конвертация изображения для сохранения
        original_img: Image.Image
        img_array: np.ndarray
        if isinstance(image, str):
            original_img = Image.open(image).convert("RGB")
            img_array = np.array(original_img)
        elif isinstance(image, Image.Image):
            original_img = image
            img_array = np.array(image)
        else:
            img_array = image
            original_img = Image.fromarray(image.astype(np.uint8))

        # Статистика маски
        mask_area: int = int(np.sum(mask > 0))
        total_pixels: int = int(mask.shape[0] * mask.shape[1])

        result_data: Dict[str, Any] = {
            "method": method_name,
            "result": result,
            "mask": mask,
            "time": execution_time,
            "mask_area": mask_area,
            "mask_percentage": (mask_area / total_pixels) * 100,
            "image_shape": result.shape,
            "timestamp": datetime.now().isoformat(),
        }

        # Сохранение результатов
        if output_dir:
            # Сохраняем оригинальное изображение
            orig_path: str = os.path.join(output_dir, "images", "original.jpg")
            original_img.save(orig_path)

            # Сохраняем результат сегментации
            result_path: str = os.path.join(
                output_dir, "images", f"{method_name}_result.jpg"
            )
            result_pil: Image.Image = Image.fromarray(result.astype(np.uint8))
            result_pil.save(result_path)

            # Сохраняем маску
            mask_path: str = os.path.join(
                output_dir, "masks", f"{method_name}_mask.png"
            )
            mask_pil: Image.Image = Image.fromarray(mask.astype(np.uint8))
            mask_pil.save(mask_path)

            # Сохраняем наложение (overlay)
            try:
                overlay_alpha: float = 0.7  # Яркость наложения
                original_alpha: float = 0.3  # Прозрачность оригинала

                overlay: np.ndarray = (
                    img_array * original_alpha + result * overlay_alpha
                ).astype(np.uint8)
                overlay = overlay.astype(np.uint8)
                overlay_path: str = os.path.join(
                    output_dir, "images", f"{method_name}_overlay.jpg"
                )
                overlay_pil: Image.Image = Image.fromarray(overlay)
                overlay_pil.save(overlay_path)
                result_data["overlay_path"] = overlay_path

                bright_overlay = cv2.addWeighted(img_array, 0.1, result, 0.9, 0)
                bright_overlay_path: str = os.path.join(
                    output_dir, "images", f"{method_name}_bright_overlay.jpg"
                )
                Image.fromarray(bright_overlay.astype(np.uint8)).save(
                    bright_overlay_path
                )
            except Exception as e:
                print(f"⚠️ Ошибка создания overlay для {method_name}: {e}")
                pass

            result_data.update(
                {
                    "result_path": str(result_path),
                    "mask_path": str(mask_path),
                    "original_path": str(orig_path),
                }
            )
            print(f"✅ {method_name}: сохранено в {output_dir}")
        elif save_path:
            result_pil = Image.fromarray(result.astype(np.uint8))
            result_pil.save(save_path)
            print(f"✅ Результат сохранен: {save_path}")
        print(
            f"   ⏱️ Время: {execution_time:.2f}s, 📏 Площадь: {result_data['mask_percentage']:.3f}%"
        )
        return result_data

    # ──────────────────────────────────────────────────────────────────────
    def _save_overlay_image(
        self, result_data: Dict[str, Any], method_dir: PathLike, method_name: str
    ) -> None:
        """
        Сохраняет наложение маски на оригинальное изображение (красный цвет для объекта).

        Args:
            result_data: Словарь с результатами `test_single_method()`.
            method_dir: Директория для сохранения.
            method_name: Имя метода (для именования файлов).
        """
        try:
            mask = result_data.get("mask")
            result_img = result_data.get("result")

            if mask is None or result_img is None:
                return

            if isinstance(result_img, Image.Image):
                result_np: np.ndarray = np.array(result_img)
            else:
                result_np = result_img

            if isinstance(mask, np.ndarray):
                mask_np: np.ndarray = mask.copy()
                if mask_np.dtype != np.uint8:
                    if mask_np.max() <= 1.0:
                        mask_np = (mask_np * 255).astype(np.uint8)
                    else:
                        mask_np = mask_np.astype(np.uint8)
            else:
                return

            # Создаем overlay
            if len(result_np.shape) == 2:
                # Grayscale оригинал
                overlay: np.ndarray = np.stack([result_np] * 3, axis=-1)
            else:
                # RGB оригинал
                overlay = result_np.copy()

            if mask_np.ndim == 2:
                mask_bool: np.ndarray = mask_np > 127
                overlay[mask_bool] = [255, 0, 0]  # Красный

            # Сохраняем overlay
            overlay_path: str = os.path.join(method_dir, "overlay.jpg")
            Image.fromarray(overlay.astype(np.uint8)).save(overlay_path)

            alpha: float = 0.5
            if len(result_np.shape) == 2:
                result_colored: np.ndarray = np.stack([result_np] * 3, axis=-1)
            else:
                result_colored = result_np

            transparent_overlay: np.ndarray = result_colored.copy()
            if mask_np.ndim == 2:
                mask_bool = mask_np > 127
                transparent_overlay[mask_bool] = [255, 0, 0]  # Красный
                blended = cv2.addWeighted(
                    result_colored, 1 - alpha, transparent_overlay, alpha, 0
                )

                blended_path: str = os.path.join(method_dir, "blended_overlay.jpg")
                Image.fromarray(blended.astype(np.uint8)).save(blended_path)

        except Exception as e:
            print(f"    ⚠️ Ошибка создания overlay для {method_name}: {e}")

    # ──────────────────────────────────────────────────────────────────────
    def _save_metrics_file(
        self, result_data: Dict[str, Any], method_dir: PathLike, method_name: str
    ) -> None:
        """
        Сохраняет метрики в JSON и текстовый файл.

        Args:
            result_data: Результаты с ключом `"metrics"`.
            method_dir: Директория для сохранения.
            method_name: Имя метода.
        """
        metrics: Optional[Dict[str, Any]] = result_data.get("metrics", {})
        if not metrics:
            return

        # JSON файл
        json_path: str = os.path.join(method_dir, "metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

        # Текстовый файл
        txt_path: str = os.path.join(method_dir, "metrics.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write(f"МЕТРИКИ СЕГМЕНТАЦИИ: {method_name}\n")
            f.write("=" * 50 + "\n\n")

            f.write(
                f"Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Время выполнения: {result_data.get('time', 0):.3f} секунд\n\n")

            if result_data.get("has_ground_truth"):
                f.write("Метрики качества (с Ground Truth):\n")
                f.write("-" * 50 + "\n")

                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if 0 <= value <= 1:
                            f.write(f"{key:<20}: {value:.4f}\n")
                        else:
                            f.write(f"{key:<20}: {value}\n")
                    else:
                        f.write(f"{key:<20}: {value}\n")
            else:
                f.write("Метрики (без Ground Truth):\n")
                f.write("-" * 50 + "\n")
                f.write(
                    f"Площадь маски: {result_data.get('mask_area', 0):,} пикселей\n"
                )
                f.write(
                    f"Процент покрытия: {result_data.get('mask_percentage', 0):.3f}%\n"
                )
        print(f"    📊 Метрики сохранены для {method_name}")

    # ──────────────────────────────────────────────────────────────────────
    def _save_method_info(
        self, result_data: Dict[str, Any], method_dir: str, method_name: str
    ) -> None:
        """
        Сохраняет информацию о методе и его параметрах.

        Args:
            result_data: Результаты тестирования.
            method_dir: Директория для сохранения.
            method_name: Имя метода.
        """
        try:
            segmenter = self.methods.get(method_name)

            if segmenter is None:
                return

            info_path: str = os.path.join(method_dir, "method_info.txt")

            with open(info_path, "w", encoding="utf-8") as f:
                f.write(
                    f"{'=' * 50}\nИНФОРМАЦИЯ О МЕТОДЕ: {method_name}\n{'=' * 50}\n\n"
                )

                # Основная информация
                f.write(f"Имя метода: {method_name}\n")
                f.write(f"Тип сегментатора: {type(segmenter).__name__}\n")
                f.write(
                    f"Время выполнения: {result_data.get('time', 0):.3f} секунд\n\n"
                )

                # Параметры метода
                if hasattr(segmenter, "method"):
                    f.write(f"Алгоритм: {segmenter.method}\n")

                if hasattr(segmenter, "params") and segmenter.params:
                    f.write("\nПараметры:\n" + "-" * 30 + "\n")
                    for key, value in segmenter.params.items():
                        f.write(f"{key}: {value}\n")

                # Информация о маске
                mask = result_data.get("mask")
                if mask is not None and isinstance(mask, np.ndarray):
                    f.write("\nИнформация о маске:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Размер: {mask.shape}\n")
                    f.write(f"Тип данных: {mask.dtype}\n")
                    f.write(f"Min значение: {mask.min()}\n")
                    f.write(f"Max значение: {mask.max()}\n")
                    mask_binary = mask > (127 if mask.max() > 1 else 0.5)
                    f.write(
                        f"Площадь: {np.sum(mask_binary):,} пикселей, Покрытие: {np.sum(mask_binary) / mask.size * 100:.3f}%\n"
                    )

                # Информация о ground truth
                f.write(
                    f"\nGround Truth: {'Доступен' if result_data.get('has_ground_truth') else 'Отсутствует'}\n"
                )
        except Exception as e:
            print(f"    ⚠️ Ошибка сохранения информации о методе {method_name}: {e}")

    # ──────────────────────────────────────────────────────────────────────
    def _save_method_results(
        self, result_data: Dict[str, Any], output_dir: str, method_name: str
    ) -> None:
        """
        Сохраняет результаты одного метода в указанную директорию.

        Args:
            result_data: Результаты `test_single_method_with_metrics()`.
            output_dir: Базовая директория для сохранения.
            method_name: Имя метода.
        """
        # Создаем поддиректории
        method_dir: str = os.path.join(output_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)

        # Сохраняем изображение результата
        result_img = result_data.get("result")
        if result_img is not None:
            if isinstance(result_img, np.ndarray):
                result_path: str = os.path.join(method_dir, "result.jpg")
                if len(result_img.shape) == 2:
                    # Grayscale
                    Image.fromarray(result_img).save(result_path)
                else:
                    # RGB
                    Image.fromarray(result_img.astype(np.uint8)).save(result_path)
            elif isinstance(result_img, Image.Image):
                result_path = os.path.join(method_dir, "result.jpg")
                result_img.save(result_path)

        # Сохраняем маску
        mask = result_data.get("mask")
        if mask is not None and isinstance(mask, np.ndarray):
            mask_path: str = os.path.join(method_dir, "mask.png")

            if mask.dtype != np.uint8:
                if mask.max() <= 1.0:
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)

            Image.fromarray(mask).save(mask_path)

        # Overlay, метрики, информация
        self._save_overlay_image(result_data, method_dir, method_name)

        # Сохраняем метрики
        if result_data.get("has_ground_truth"):
            self._save_metrics_file(result_data, method_dir, method_name)

        # Сохраняем информацию о методе
        self._save_method_info(result_data, method_dir, method_name)

    # ──────────────────────────────────────────────────────────────────────
    def test_single_method_with_metrics(
        self,
        image: ImageInput,
        method_name: str,
        ground_truth: Optional[MaskArray] = None,
        threshold: float = 0.5,
        output_dir: Optional[PathLike] = None,
    ) -> Dict[str, Any]:
        """
        Тестирование метода с расчётом метрик качества.

        Args:
            image: Входное изображение.
            method_name: Ключ метода в реестре.
            ground_truth: GT-маска. Если `None`, используется `self.ground_truth_mask`.
            threshold: Порог бинаризации для метрик.
            output_dir: Директория для сохранения артефактов.

        Returns:
            Dict[str, Any]: Результаты с ключами:
            - `method`, `result`, `mask`, `time`
            - `metrics` (если есть GT), `has_ground_truth`
            - Базовая статистика маски (если нет GT)
        """
        if method_name not in self.methods:
            raise ValueError(f"Метод {method_name} не найден")

        # Используем ground truth
        gt_mask: Optional[MaskArray] = (
            ground_truth if ground_truth is not None else self.ground_truth_mask
        )

        segmenter = self.methods[method_name]

        # Измеряем время выполнения
        if str(self.device) == "cuda":
            torch.cuda.synchronize()
        start_time: float = time.perf_counter()
        result_img: BinaryMask
        pred_mask: Optional[ProbabilityMask]
        result_img, pred_mask = segmenter.segment_with_mask(image)
        if str(self.device) == "cuda":
            torch.cuda.synchronize()
        execution_time: float = time.perf_counter() - start_time

        if gt_mask is not None:
            if pred_mask is not None and gt_mask is not None:
                metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(
                    pred_mask, gt_mask, threshold=threshold
                )
            else:
                metrics = {}

            result_data: Dict[str, Any] = {
                "method": method_name,
                "result": result_img,
                "mask": pred_mask,
                "time": execution_time,
                "metrics": metrics,
                "has_ground_truth": True,
            }
        else:
            if pred_mask is not None:
                mask_area: int = int(np.sum(pred_mask > 0))
                total_pixels: int = int(pred_mask.shape[0] * pred_mask.shape[1])
            else:
                mask_area = 0
                total_pixels = 1

            result_data = {
                "method": method_name,
                "result": result_img,
                "mask": pred_mask,
                "time": execution_time,
                "mask_area": mask_area,
                "mask_percentage": (mask_area / total_pixels) * 100.0,
                "has_ground_truth": False,
            }
        if output_dir:
            self._save_method_results(result_data, str(output_dir), method_name)
        return result_data

    # ──────────────────────────────────────────────────────────────────────
    def compare_methods(
        self,
        image: ImageInput,
        method_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (20, 15),
        save_comparison: bool = True,
        test_name: Optional[str] = None,
        show_plots: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Визуальное сравнение нескольких методов в одном графике.

        Строит grid-визуализацию: [Оригинал] + [Результаты методов] с подписями
        (время выполнения, процент покрытия). Сохраняет сводный график и статистику.

        Args:
            image: Входное изображение.
            method_names: Список методов для сравнения. Если `None`, все зарегистрированные.
            figsize: Размер фигуры matplotlib.
            save_comparison: Сохранять ли сводный график.
            test_name: Префикс имени теста.
            show_plots: Показывать ли график через `plt.show()`.

        Returns:
            Dict[str, Dict]: Результаты по каждому методу (из `test_single_method()`).
        """
        if method_names is None:
            method_names = list(self.methods.keys())

        test_dir: str = self._create_test_directory(test_name)
        results: Dict[str, Dict[str, Any]] = {}

        original_img: Image.Image

        if isinstance(image, str):
            original_img = Image.open(image).convert("RGB")
            image_path = image
        elif isinstance(image, Image.Image):
            original_img = image
            image_path = None
        else:
            original_img = Image.fromarray(image.astype(np.uint8))
            image_path = None
        print(image_path)
        orig_save_path: str = os.path.join(test_dir, "images", "original.jpg")
        original_img.save(orig_save_path)
        print(f"📸 Оригинальное изображение сохранено: {orig_save_path}")

        # Grid-визуализация
        n_methods: int = len(method_names)
        n_cols: int = min(4, n_methods + 1)
        n_rows: int = (n_methods + n_cols) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        all_stats: StatsList = []
        for i, method_name in enumerate(method_names, 1):
            if i >= len(axes):
                break

            try:
                result_data: Dict[str, Any] = self.test_single_method(
                    image, method_name, output_dir=test_dir
                )
                results[method_name] = result_data

                # Добавляем статистику
                stats: Dict[str, Any] = {
                    "method": method_name,
                    "time_seconds": result_data["time"],
                    "mask_area_pixels": result_data["mask_area"],
                    "mask_percentage": result_data["mask_percentage"],
                    "image_shape": result_data["image_shape"],
                }
                all_stats.append(stats)

                axes[i].imshow(result_data["result"])
                title: str = (
                    f"{method_name}\n{result_data['time']:.2f}s, {result_data['mask_percentage']:.3f}%"
                )
                axes[i].set_title(title, fontsize=9)
                axes[i].axis("off")

                print(
                    f"{method_name}: {result_data['time']:.2f}s, {result_data['mask_percentage']:.3f}% площади"
                )

            except Exception as e:
                error_msg: str = str(e)[:50]
                axes[i].text(
                    0.5,
                    0.5,
                    f"Error:\n{error_msg}",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                    fontsize=8,
                )
                axes[i].set_title(f"{method_name}\n(Error)", fontsize=9)
                axes[i].axis("off")

                print(f"❌ Ошибка в методе {method_name}: {e}")

                stats = {
                    "method": method_name,
                    "error": error_msg,
                    "time_seconds": None,
                    "mask_area_pixels": None,
                    "mask_percentage": None,
                }
                all_stats.append(stats)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle(
            f"Сравнение методов сегментации\n{self.current_test_id}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        # Сохраняем сравнение
        if save_comparison:
            comparison_path: str = os.path.join(
                test_dir, "comparisons", "methods_comparison.jpg"
            )
            plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
            print(f"📊 Сравнительный график сохранен: {comparison_path}")
            comparison_small_path: str = os.path.join(
                test_dir, "comparisons", "methods_comparison_small.jpg"
            )
            plt.savefig(comparison_small_path, dpi=100, bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close()

        # Сохраняем статистику
        self._save_statistics(all_stats, test_dir)

        # Сохраняем результаты
        self._save_results_summary(results, test_dir)

        self.results[test_dir] = results
        print(f"✅ Тестирование завершено. Результаты в: {test_dir}")
        print(f"📋 Протестировано методов: {len(results)}/{len(method_names)}")
        return results

    # ──────────────────────────────────────────────────────────────────────
    def compare_methods_with_metrics(
        self,
        image: ImageInput,
        method_names: Optional[List[str]] = None,
        ground_truth: Optional[MaskArray] = None,
        threshold: float = 0.5,
        figsize: Tuple[int, int] = (20, 15),
        test_name: Optional[str] = None,
        show_plots: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Сравнение методов с отображением метрик качества на графике.

        В отличие от `compare_methods()`, добавляет ground truth и подписи
        с метриками (IoU, Dice, F1, Accuracy) под каждым методом.

        Args:
            image: Входное изображение.
            method_names: Список методов.
            ground_truth: GT-маска. Если `None`, используется загруженная.
            threshold: Порог для метрик.
            figsize: Размер фигуры.
            test_name: Префикс теста.
            show_plots: Показывать ли график.

        Returns:
            Dict[str, Dict]: Результаты с метриками (если есть GT).
        """
        if method_names is None:
            method_names = list(self.methods.keys())

        test_dir: str = self._create_test_directory(test_name)
        results: Dict[str, Dict[str, Any]] = {}

        gt_mask: Optional[MaskArray] = (
            ground_truth if ground_truth is not None else self.ground_truth_mask
        )
        has_gt: bool = gt_mask is not None

        print(f"Сравнение методов {'с' if has_gt else 'без'} ground truth")

        # Grid-визуализация
        n_methods: int = len(method_names)
        n_cols: int = min(4, n_methods + 2 if has_gt else n_methods + 1)
        n_rows: int = (n_methods + n_cols) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        # Оригинальное изображение
        original_img: Union[Image.Image | np.ndarray] = (
            Image.open(image).convert("RGB") if isinstance(image, str) else image
        )
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        start_idx: int = 1
        if has_gt:
            axes[1].imshow(gt_mask, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")
            start_idx = 2

        all_metrics_data: List[MetricDict] = []

        for i, method_name in enumerate(method_names, start_idx):
            if i >= len(axes):
                break

            try:
                result_data: Dict[str, Any] = self.test_single_method_with_metrics(
                    image, method_name, gt_mask, threshold, test_dir
                )
                results[method_name] = result_data
                axes[i].imshow(result_data["result"])

                if has_gt:
                    metrics = result_data["metrics"]
                    title: str = (
                        f"{method_name}\n"
                        f"IoU: {metrics['iou']:.3f}, Dice: {metrics['dice']:.3f}\n"
                        f"F1: {metrics['f1_score']:.3f}, Acc: {metrics['pixel_accuracy']:.3f}"
                    )
                else:
                    title = (
                        f"{method_name}\n"
                        f"Time: {result_data['time']:.2f}s\n"
                        f"Area: {result_data['mask_percentage']:.3f}%"
                    )

                axes[i].set_title(title, fontsize=9)
                axes[i].axis("off")

                if has_gt:
                    metrics = result_data["metrics"].copy()
                    metrics.update({"method": method_name, "time": result_data["time"]})
                    all_metrics_data.append(metrics)

                print(
                    f"{method_name}: {'метрики вычислены' if has_gt else 'без ground truth'}"
                )

            except Exception as e:
                error_msg: str = str(e)[:50]
                axes[i].text(
                    0.5,
                    0.5,
                    f"Error:\n{error_msg}",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                    fontsize=8,
                )
                axes[i].set_title(f"{method_name}\n(Error)", fontsize=9)
                axes[i].axis("off")
                print(f"❌ Ошибка в методе {method_name}: {e}")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle(
            f"Сравнение методов сегментации {'с метриками' if has_gt else ''}\n{self.current_test_id}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        comparison_path: Path = Path(test_dir, "comparisons", "methods_comparison.jpg")
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(comparison_path, dpi=150, bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close()
        if has_gt and all_metrics_data:
            self._save_metrics_comparison(all_metrics_data, test_dir)

        self.results[test_dir] = results
        return results

    # ──────────────────────────────────────────────────────────────────────
    def _save_statistics(self, stats: StatsList, output_dir: PathLike) -> None:
        """
        Сохраняет статистику тестирования в JSON, CSV и TXT форматах.

        Args:
            stats: Список словарей со статистикой по методам.
            output_dir: Директория для сохранения.
        """

        # Функция для конвертации numpy типов в стандартные Python типы
        def convert_numpy_types(obj: Any) -> Any:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # Конвертируем все numpy типы перед сохранением в JSON
        serializable_stats: Any = convert_numpy_types(stats)

        # Сохраняем как JSON
        stats_json_path: str = os.path.join(output_dir, "statistics", "statistics.json")
        try:
            with open(stats_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    serializable_stats, f, indent=2, ensure_ascii=False, default=str
                )
            print(f"📊 Статистика сохранена (JSON): {stats_json_path}")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения JSON статистики: {e}")
            try:

                def default_serializer(o: Any) -> Any:
                    if isinstance(o, np.integer):
                        return int(o)
                    elif isinstance(o, np.floating):
                        return float(o)
                    elif isinstance(o, np.ndarray):
                        return o.tolist()
                    elif hasattr(o, "tolist"):
                        return o.tolist()
                    elif hasattr(o, "__dict__"):
                        return str(o)
                    else:
                        return str(o)

                with open(stats_json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        stats,
                        f,
                        indent=2,
                        ensure_ascii=False,
                        default=default_serializer,
                    )
            except Exception as e2:
                print(f"❌ Критическая ошибка сохранения JSON: {e2}")
                with open(
                    stats_json_path.replace(".json", "_fallback.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(str(stats))

        # Сохраняем как CSV
        try:
            df_stats: List[Dict[str, Any]] = []
            for stat in stats:
                row: Dict[str, Any] = {}
                for key, value in stat.items():
                    if isinstance(value, np.integer):
                        row[key] = int(value)
                    elif isinstance(value, np.floating):
                        row[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        row[key] = str(value.tolist())
                    elif isinstance(value, tuple):
                        row[key] = str(value)
                    else:
                        row[key] = value
                df_stats.append(row)

            df: pd.DataFrame = pd.DataFrame(df_stats)
            stats_csv_path: str = os.path.join(
                output_dir, "statistics", "statistics.csv"
            )
            df.to_csv(stats_csv_path, index=False)
            print(f"📈 Статистика сохранена (CSV): {stats_csv_path}")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения CSV: {e}")

        report_path: str = os.path.join(output_dir, "statistics", "test_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("ОТЧЕТ О ТЕСТИРОВАНИИ МЕТОДОВ СЕГМЕНТАЦИИ\n")
            f.write("=" * 60 + "\n\n")
            f.write(
                f"Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"ID теста: {self.current_test_id}\n")
            f.write(f"Всего методов: {len(stats)}\n\n")

            successful: List[Dict[str, Any]] = [s for s in stats if "error" not in s]
            if successful:
                f.write("УСПЕШНЫЕ МЕТОДЫ:\n")
                f.write("-" * 40 + "\n")
                for stat in successful:
                    f.write(f"{stat['method']}:\n")
                    f.write(f"  Время: {float(stat.get('time_seconds', 0)):.3f} сек\n")
                    f.write(
                        f"  Площадь маски: {int(stat.get('mask_area_pixels', 0)):,} пикселей\n"
                    )
                    f.write(
                        f"  Процент покрытия: {float(stat.get('mask_percentage', 0)):.2f}%\n"
                    )
                    if "image_shape" in stat:
                        shape = stat["image_shape"]
                        if isinstance(shape, tuple):
                            f.write(f"  Размер результата: {shape}\n")
                        elif isinstance(shape, np.ndarray):
                            f.write(f"  Размер результата: {tuple(shape)}\n")
                        else:
                            f.write(f"  Размер результата: {shape}\n")
                    f.write("\n")

            failed: List[Dict[str, Any]] = [s for s in stats if "error" in s]
            if failed:
                f.write("МЕТОДЫ С ОШИБКАМИ:\n")
                f.write("-" * 40 + "\n")
                for stat in failed:
                    f.write(f"{stat['method']}: {stat.get('error', 'Unknown error')}\n")

        print(f"📋 Текстовый отчет сохранен: {report_path}")

    # ──────────────────────────────────────────────────────────────────────
    def _save_metrics_comparison(
        self, metrics_data: List[MetricDict], output_dir: PathLike
    ) -> None:
        """
        Сохраняет сравнение метрик в различных форматах.

        Args:
            metrics_data: Список словарей с метриками по методам.
            output_dir: Базовая директория для сохранения.
        """
        metrics_dir: str = os.path.join(output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # JSON
        json_path: str = os.path.join(metrics_dir, "metrics_comparison.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        print(f"📊 Метрики сохранены (JSON): {json_path}")

        # CSV + таблица-изображение
        try:
            df: pd.DataFrame = pd.DataFrame(metrics_data)
            df = df.sort_values("iou", ascending=False)
            csv_path: str = os.path.join(metrics_dir, "metrics_comparison.csv")
            df.to_csv(csv_path, index=False)
            print(f"📊 Метрики сохранены (CSV): {csv_path}")

            self._create_metrics_table_image(df, metrics_dir)
        except ImportError as e:
            print(f"⚠️ Ошибка сохранения CSV/таблицы: {e}")

    # ──────────────────────────────────────────────────────────────────────
    def _create_metrics_table_image(self, df: pd.DataFrame, metrics_dir: str) -> None:
        """
        Создаёт изображение со сводной таблицей метрик для отчётов.

        Args:
            df: DataFrame с метриками.
            metrics_dir: Директория для сохранения.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, len(df) * 0.4 + 2))
            ax.axis("tight")
            ax.axis("off")

            table_columns: List[str] = [
                "method",
                "iou",
                "dice",
                "f1_score",
                "precision",
                "recall",
                "pixel_accuracy",
                "mae",
                "time",
            ]

            available_columns: List[str] = [
                col for col in table_columns if col in df.columns
            ]
            table_data: pd.DataFrame = df[available_columns].copy()
            for col in table_data.columns:
                if col != "method":
                    table_data[col] = table_data[col].apply(lambda x: f"{x:.4f}")

            table = ax.table(
                cellText=table_data.values,
                colLabels=table_data.columns,
                cellLoc="center",
                loc="center",
                colColours=["#f0f0f0"] * len(table_data.columns),
            )

            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)

            plt.title("Сравнение метрик сегментации", fontsize=14, fontweight="bold")
            plt.tight_layout()

            table_path: str = os.path.join(metrics_dir, "metrics_table.jpg")
            plt.savefig(table_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"📊 Таблица метрик сохранена: {table_path}")

        except Exception as e:
            print(f"⚠️ Ошибка создания таблицы метрик: {e}")

    # ──────────────────────────────────────────────────────────────────────
    def _save_results_summary(
        self, results: Dict[str, Dict], output_dir: PathLike
    ) -> None:
        """
        Сохраняет сводку результатов с конвертацией numpy-типов.

        Args:
            results: Результаты по методам.
            output_dir: Базовая директория.
        """
        summary_path: str = os.path.join(
            output_dir, "statistics", "results_summary.json"
        )

        def convert_for_json(obj: Any) -> Any:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj

        # Подготовка данных для сохранения
        summary_data: Dict[str, Any] = {
            "test_id": self.current_test_id,
            "timestamp": datetime.now().isoformat(),
            "total_methods": len(results),
            "methods": {},
        }

        for method_name, result in results.items():
            method_data: Dict[str, Any] = {}
            for key, value in result.items():
                if key in ("result", "mask"):
                    continue
                elif key == "image_shape":
                    if isinstance(value, np.ndarray):
                        method_data[key] = value.tolist()
                    elif isinstance(value, tuple):
                        method_data[key] = list(value)
                    else:
                        method_data[key] = value
                else:
                    method_data[key] = convert_for_json(value)

            for key in ["result_path", "mask_path", "overlay_path", "original_path"]:
                if key in result:
                    method_data[key] = str(result[key])

            summary_data["methods"][method_name] = method_data

        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"📋 Сводка результатов сохранена: {summary_path}")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения сводки результатов: {e}")
            with open(
                summary_path.replace(".json", "_simple.txt"), "w", encoding="utf-8"
            ) as f:
                for method_name, method_data in summary_data["methods"].items():
                    f.write(f"\n{'=' * 40}\n")
                    f.write(f"Метод: {method_name}\n")
                    f.write(f"{'=' * 40}\n")
                    for key, value in method_data.items():
                        f.write(f"{key}: {value}\n")

    # ──────────────────────────────────────────────────────────────────────
    def benchmark_methods(
        self,
        image: ImageInput,
        n_runs: int = 3,
        save_benchmark: bool = True,
        test_name: Optional[str] = None,
        save_results: bool = True,
        force_warmup: bool = False,
        ground_truth: Optional[MaskArray] = None,
    ) -> pd.DataFrame:
        """
        Бенчмарк методов с многократными прогонами, warm-up и метриками.

        Для каждого метода:
        1. Выполняет warm-up (если включён).
        2. Запускает `n_runs` итераций, замеряя время.
        3. Сохраняет результат первого прогона.
        4. Рассчитывает метрики качества (если есть GT).
        5. Агрегирует статистику: mean, std, min, max времени.

        Args:
            image: Входное изображение.
            n_runs: Количество прогонов для стабильного замера времени.
            save_benchmark: Сохранять ли отчёт бенчмарка.
            test_name: Префикс имени теста.
            save_results: Сохранять ли артефакты (маски, overlay).
            force_warmup: Выполнить warm-up для всех методов принудительно.
            ground_truth: GT-маска для расчёта метрик.

        Returns:
            pd.DataFrame: Сводная таблица с колонками:
            - `Method`, `Mean_Time_s`, `Std_Time_s`, `Time_String`
            - `Mask_Area`, `Mask_Percentage`, `Min_Time_s`, `Max_Time_s`, `Num_Runs`
            - Если есть GT: `IoU`, `Dice`, `F1_Score`, `Precision`, `Recall`, `Accuracy`, `Hausdorff`, `MAE`, `Area_Difference`
        """
        # Конвертация изображения
        if isinstance(image, str):
            original_img: Image.Image = Image.open(image).convert("RGB")
            image_array: np.ndarray = np.array(original_img)
        elif isinstance(image, Image.Image):
            original_img = image.convert("RGB")
            image_array = np.array(original_img)
        elif isinstance(image, np.ndarray):
            original_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
            image_array = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        bench_dir: str = self._create_test_directory(
            f"benchmark_{test_name}" if test_name else "benchmark"
        )
        Path(bench_dir, "images").mkdir(parents=True, exist_ok=True)
        Path(bench_dir, "masks").mkdir(parents=True, exist_ok=True)

        # Сохраняем оригинальное изображение
        orig_path: str = os.path.join(bench_dir, "images", "original.jpg")
        original_img.save(orig_path)
        print(f"📸 Оригинальное изображение сохранено: {orig_path}")

        if force_warmup and self.enable_warmup:
            print("\n" + "=" * 60 + "\n🔥 ЗАПУСК WARM-UP ПЕРЕД БЕНЧМАРКОМ\n" + "=" * 60)

            for method_name, segmenter in self.methods.items():
                self._ensure_warmup(method_name, segmenter, image_array)

        gt_mask_to_use: Optional[MaskArray] = (
            ground_truth if ground_truth is not None else self.ground_truth_mask
        )
        has_gt: bool = gt_mask_to_use is not None
        gt_binary: Optional[MaskArray] = None

        if has_gt and gt_mask_to_use is not None:
            print("🎯 Обнаружен Ground Truth. Будет выполнен расчет метрик качества.")
            gt_binary = (
                (gt_mask_to_use * 255).astype(np.uint8)
                if gt_mask_to_use.max() <= 1.0
                else gt_mask_to_use.astype(np.uint8)
            )
        else:
            print("⚠️ Ground Truth не найден. Метрики качества рассчитываться не будут.")

        print(f"🏃 Запуск бенчмарка ({n_runs} прогонов)...")
        benchmark_results: List[Dict[str, Any]] = []

        for method_name in self.methods.keys():
            print(f"  📊 Тестируем {method_name}...")

            segmenter = self.methods[method_name]

            if not force_warmup:
                self._ensure_warmup(method_name, segmenter, image_array)

            times: List[float] = []
            results_list: List[np.ndarray] = []
            masks_list: List[MaskArray] = []

            is_neural: bool = (
                "neural" in method_name.lower()
                or "segformer" in method_name.lower()
                or "mask2former" in method_name.lower()
                or "deeplab" in method_name.lower()
                or "unet" in method_name.lower()
                or "fpn" in method_name.lower()
                or "psp" in method_name.lower()
                or "fcn" in method_name.lower()
                or "segnet" in method_name.lower()
            )

            input_arg_for_method = image

            if is_neural:
                if isinstance(image, str):
                    input_arg_for_method = image
                else:
                    temp_path: str = os.path.join(bench_dir, "images", "temp_input.jpg")
                    original_img.save(temp_path)
                    input_arg_for_method = temp_path
            else:
                input_arg_for_method = image_array

            for run in range(n_runs):
                if str(self.device) == "cuda":
                    torch.cuda.synchronize()
                start_time: float = time.perf_counter()
                result: np.ndarray
                mask: np.ndarray
                try:
                    result_opt: BinaryMask
                    mask_opt: Optional[ProbabilityMask]
                    result_opt, mask_opt = self.methods[method_name].segment_with_mask(
                        input_arg_for_method
                    )
                    is_backend = method_name.endswith("_ONNX") or method_name.endswith(
                        "_TRT"
                    )
                    if result_opt is None:
                        logger.warning(
                            f"    ⚠️  {method_name} returned None result (run {run + 1})"
                        )
                        if run == 0 and n_runs > 1:
                            continue
                        result = np.zeros(image_array.shape[:2], dtype=np.uint8)
                        mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
                    else:
                        result = result_opt
                        # 🔥 FIX: Для бэкендов используем result как маску, т.к. mask_opt = None
                        if is_backend:
                            mask = result_opt
                        else:
                            mask = mask_opt if mask_opt is not None else result_opt

                    # 🔥 Гарантируем, что mask — numpy array (защита от None)
                    if mask is None:
                        mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
                    if result is None:
                        result = np.zeros(image_array.shape[:2], dtype=np.uint8)
                    if str(self.device) == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - start_time)
                    if run == 0:
                        masks_list.append(mask)
                        results_list.append(result)
                except Exception as e:
                    logger.warning(
                        f"    ⚠️  Ошибка в {method_name} (запуск {run + 1}): {e}"
                    )
                    if run == 0 and n_runs > 1:
                        # 🔥 Пробуем ещё раз при ошибке на первом запуске
                        continue
                    # Если все запуски неудачны — фиксируем нулевые значения
                    times.append(0.0)  # Или np.nan для явного указания ошибки
                    if run == 0:
                        h, w = image_array.shape[:2]
                        masks_list.append(np.zeros((h, w), dtype=np.uint8))
                        results_list.append(np.zeros((h, w), dtype=np.uint8))

            mask_area: int = 0
            total_pixels: int = 1
            metrics_dict: MetricsDict = {}
            if masks_list and results_list:
                mask = masks_list[0]
                result_img: np.ndarray = results_list[0]
                mask_area = int(np.sum(mask > 0))
                total_pixels = int(mask.shape[0] * mask.shape[1])
                if has_gt and gt_binary is not None:
                    try:
                        # Ресайз GT под размер предсказания
                        if gt_binary.shape != mask.shape:
                            from skimage.transform import resize

                            gt_resized: np.ndarray = resize(
                                gt_binary,
                                mask.shape,
                                order=0,
                                preserve_range=True,
                                anti_aliasing=False,
                            ).astype(np.uint8)
                        else:
                            gt_resized = gt_binary

                        metrics_dict = SegmentationMetrics.calculate_all_metrics(
                            pred_mask=mask,
                            gt_mask=gt_resized,
                            threshold=0.5,
                            include_hausdorff=True,
                        )
                        iou: float = metrics_dict.get("iou", 0)
                        dice: float = metrics_dict.get("dice", 0)
                        status = "✅" if iou > 0.5 else "⚠️" if iou > 0.2 else "❌"
                        print(f"    {status} IoU: {iou:.4f}, Dice: {dice:.4f}")

                    except Exception as metric_err:
                        print(f"    ⚠️ Ошибка расчета метрик: {metric_err}")
                        metrics_dict = {"error": str(metric_err)}
            else:
                if not times:
                    print(f"    ❌ Метод {method_name} не вернул результат.")

            mean_time: float = float(np.mean(times)) if times else 0.0
            std_time: float = float(np.std(times)) if times else 0.0

            if save_results and masks_list and results_list:
                try:
                    # Сохраняем результат сегментации
                    result_path: str = os.path.join(
                        bench_dir, "images", f"{method_name}_result.jpg"
                    )
                    result_pil: Image.Image = Image.fromarray(
                        result_img.astype(np.uint8)
                    )
                    result_pil.save(result_path)

                    # Сохраняем маску
                    mask_path: str = os.path.join(
                        bench_dir, "masks", f"{method_name}_mask.png"
                    )
                    mask_pil: Image.Image = Image.fromarray(mask.astype(np.uint8))
                    mask_pil.save(mask_path)

                    # Сохраняем overlay (30% оригинал + 70% результат)
                    if image_array is not None:
                        # 🔥 FIX: Приводим 2D маску к 3 каналам для сложения с RGB
                        if result_img.ndim == 2:  # (H, W)
                            result_3ch = np.stack(
                                [result_img] * 3, axis=-1
                            )  # (H, W, 3)
                        else:
                            result_3ch = result_img

                        overlay: np.ndarray = (
                            image_array * 0.3 + result_3ch * 0.7
                        ).astype(np.uint8)
                        overlay_path: str = os.path.join(
                            bench_dir, "images", f"{method_name}_overlay.jpg"
                        )
                        overlay_pil = Image.fromarray(overlay)
                        overlay_pil.save(overlay_path)

                        print(f"    💾 Overlay сохранён: {overlay_path}")
                except Exception as e:
                    print(f"    ⚠️ Ошибка сохранения результатов: {e}")

            row_data: Dict[str, Any] = {
                "Method": method_name,
                "Mean_Time_s": mean_time,
                "Std_Time_s": std_time,
                "Time_String": f"{mean_time:.3f} ± {std_time:.3f}",
                "Mask_Area": mask_area,
                "Mask_Percentage": (
                    (mask_area / total_pixels * 100) if total_pixels > 0 else 0
                ),
                "Min_Time_s": min(times) if times else 0,
                "Max_Time_s": max(times) if times else 0,
                "Num_Runs": len(times),
                "Has_GT": has_gt,
            }

            if has_gt and metrics_dict and "error" not in metrics_dict:
                row_data.update(
                    {
                        "IoU": metrics_dict.get("iou", 0),
                        "Dice": metrics_dict.get("dice", 0),
                        "F1_Score": metrics_dict.get("f1_score", 0),
                        "Precision": metrics_dict.get("precision", 0),
                        "Recall": metrics_dict.get("recall", 0),
                        "Accuracy": metrics_dict.get("pixel_accuracy", 0),
                        "Hausdorff": metrics_dict.get("hausdorff_distance", 0),
                        "MAE": metrics_dict.get("mae", 0),
                        "Area_Difference": metrics_dict.get("area_difference", 0),
                    }
                )

            benchmark_results.append(row_data)

        df: pd.DataFrame = pd.DataFrame(benchmark_results)
        if "Mean_Time_s" in df.columns:
            df = df.sort_values("Mean_Time_s")
        elif "Mean_Time_s (s)" in df.columns:
            df = df.sort_values("Mean_Time_s (s)")

        print("\n" + "=" * 80 + "\nРЕЗУЛЬТАТЫ БЕНЧМАРКА:\n" + "=" * 80)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(df.to_string(index=False))
        print("=" * 80)
        if save_benchmark:
            self._save_benchmark_results(df, bench_dir)
        self.last_benchmark_dirs[test_name or "default"] = bench_dir
        return df

    # ──────────────────────────────────────────────────────────────────────
    def _save_benchmark_results(self, df: pd.DataFrame, output_dir: PathLike) -> None:
        """
        Сохраняет результаты бенчмарка в CSV, Excel и текстовый отчёт.

        Args:
            df: DataFrame с результатами `benchmark_methods()`.
            output_dir: Базовая директория для сохранения.
        """
        bench_stats_dir: str = os.path.join(output_dir, "statistics")
        os.makedirs(bench_stats_dir, exist_ok=True)

        # Сохраняем как CSV
        csv_path: str = os.path.join(bench_stats_dir, "benchmark_results.csv")
        df.to_csv(csv_path, index=False)

        # Сохраняем как Excel
        try:
            excel_path: str = os.path.join(bench_stats_dir, "benchmark_results.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"📊 Excel отчет сохранен: {excel_path}")
        except Exception:
            pass

        # TXT-отчёт
        report_path: str = os.path.join(bench_stats_dir, "benchmark_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЕТ О БЕНЧМАРКЕ МЕТОДОВ СЕГМЕНТАЦИИ\n")
            f.write("=" * 80 + "\n\n")
            f.write(
                f"Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(
                f"Количество прогонов: {df['Num_Runs'].iloc[0] if not df.empty else 0}\n"
            )
            f.write(f"Всего методов: {len(df)}\n\n")

            f.write("ТАБЛИЦА РЕЗУЛЬТАТОВ:\n")
            f.write("-" * 80 + "\n")

            if "Has_GT" in df.columns and df["Has_GT"].any():
                f.write("Режим: С расчетом метрик качества (Ground Truth available)\n")
                f.write("-" * 80 + "\n")
                f.write(
                    f"{'Метод':<30} {'Время (с)':<20} {'IoU':<10} {'Dice':<10} {'F1':<10}\n"
                )
                f.write("-" * 80 + "\n")
                for _, row in df.iterrows():
                    iou: str = row.get("IoU", "N/A")
                    dice: str = row.get("Dice", "N/A")
                    f1: str = row.get("F1_Score", "N/A")
                    if isinstance(iou, float):
                        f.write(
                            f"{row['Method']:<30} {row['Time_String']:<20} {iou:.4f}     {dice:.4f}     {f1:.4f}\n"
                        )
                    else:
                        f.write(
                            f"{row['Method']:<30} {row['Time_String']:<20} {iou}     {dice}     {f1}\n"
                        )
            else:
                f.write(
                    "Режим: Только производительность (без Ground Truth)\n"
                    + "-" * 80
                    + "\n"
                )
                f.write(
                    f"{'Метод':<30} {'Время (с)':<20} {'Площадь маски':<15} {'Процент':<10}\n"
                    + "-" * 80
                    + "\n"
                )
                for _, row in df.iterrows():
                    f.write(
                        f"{row['Method']:<30} {row['Time_String']:<20} {int(row['Mask_Area']):<15,} {row['Mask_Percentage']:.3f}%\n"
                    )
            f.write("\n" + "=" * 80 + "\nСВОДКА:\n" + "-" * 80 + "\n")
            if not df.empty:
                fastest = df.iloc[0]
                slowest = df.iloc[-1]
                f.write(
                    f"Самый быстрый метод: {fastest['Method']} ({fastest['Mean_Time_s']:.3f} с)\n"
                )
                f.write(
                    f"Самый медленный метод: {slowest['Method']} ({slowest['Mean_Time_s']:.3f} с)\n"
                )
                f.write(f"Среднее время: {df['Mean_Time_s'].mean():.3f} с\n")
                f.write(f"Стандартное отклонение: {df['Mean_Time_s'].std():.3f} с\n")

                if "IoU" in df.columns:
                    best_iou_row = df.loc[df["IoU"].idxmax()]
                    f.write(
                        f"Лучший метод по IoU: {best_iou_row['Method']} (IoU={best_iou_row['IoU']:.4f})\n"
                    )

        print(f"📋 Отчет бенчмарка сохранен: {report_path}")
        print(f"📊 CSV с результатами: {csv_path}")
        self._plot_benchmark_results(df, output_dir)

    # ──────────────────────────────────────────────────────────────────────
    def _plot_benchmark_results(self, df: pd.DataFrame, output_dir: PathLike) -> None:
        """
        Строит графики результатов бенчмарка: время, площадь, IoU vs время.

        Args:
            df: DataFrame с результатами.
            output_dir: Базовая директория для сохранения.
        """

        if df.empty:
            return

        comp_dir: str = os.path.join(output_dir, "comparisons")
        os.makedirs(comp_dir, exist_ok=True)

        # График 1: Время выполнения
        plt.figure(figsize=(12, 6))
        bars: plt.BarContainer = plt.barh(df["Method"], df["Mean_Time_s"])
        plt.xlabel("Время выполнения (секунды)")
        plt.title("Бенчмарк методов сегментации: Время выполнения")

        if "Std_Time_s" in df.columns:
            plt.errorbar(
                df["Mean_Time_s"],
                df["Method"],
                xerr=df["Std_Time_s"],
                fmt="none",
                ecolor="black",
                capsize=5,
            )

        max_time: float = df["Mean_Time_s"].max() if not df.empty else 0
        for bar, time_val in zip(bars, df["Mean_Time_s"]):
            plt.text(
                time_val + max_time * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{time_val:.3f}s",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        bench_plot_path: str = os.path.join(comp_dir, "benchmark_time.png")
        plt.savefig(bench_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        # График 2: Время vs Площадь маски
        plt.figure(figsize=(10, 6))
        if "Mask_Percentage" in df.columns:
            scatter: plt.PathCollection = plt.scatter(
                df["Mean_Time_s"],
                df["Mask_Percentage"],
                s=100,
                alpha=0.7,
                c=range(len(df)),
                cmap="viridis",
            )
            for i, (x, y, method) in enumerate(
                zip(df["Mean_Time_s"], df["Mask_Percentage"], df["Method"])
            ):
                plt.annotate(
                    method,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                )

            plt.xlabel("Время выполнения (секунды)")
            plt.ylabel("Площадь маски (%)")
            plt.title("Бенчмарк: Время vs Площадь покрытия")
            plt.colorbar(scatter, label="Ранг метода")
            plt.grid(True, alpha=0.3)

            bench_scatter_path: str = os.path.join(comp_dir, "benchmark_scatter.png")
            plt.savefig(bench_scatter_path, dpi=150, bbox_inches="tight")
            plt.close()

        # График 3: IoU vs Время (если есть GT)
        if "IoU" in df.columns:
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(
                df["Mean_Time_s"],
                df["IoU"],
                s=100,
                alpha=0.7,
                c=range(len(df)),
                cmap="viridis",
            )

            for i, row in df.iterrows():
                plt.annotate(
                    row["Method"][:15],
                    (row["Mean_Time_s"], row["IoU"]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                )

            plt.xlabel("Время выполнения (секунды)")
            plt.ylabel("IoU Score")
            plt.title("Соотношение точности (IoU) и скорости")
            plt.grid(True, alpha=0.3)
            plt.colorbar(scatter, label="Ранг метода")
            plt.tight_layout()
            plt.savefig(
                os.path.join(comp_dir, "iou_vs_time.png"), dpi=150, bbox_inches="tight"
            )
            plt.close()
            print(f"📈 График IoU vs Time сохранен в {comp_dir}/iou_vs_time.png")

        # График 4: Сравнительная визуализация результатов (маленькие превью)
        self._create_benchmark_preview(df, output_dir, str(comp_dir))
        print(f"📈 Графики бенчмарка сохранены в {comp_dir}/")

    # ──────────────────────────────────────────────────────────────────────
    def _create_metrics_plots(self, df: pd.DataFrame, metrics_dir: PathLike) -> None:
        """
        Создаёт графики сравнения метрик (бар-чарты, scatter IoU vs время).

        Примечание: Метод закомментирован в оригинальном коде — оставлен для совместимости.
        """
        try:
            # График 1: Барчарт основных метрик
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            ax1 = axes[0, 0]
            bars1 = ax1.barh(df["method"], df["iou"])
            ax1.set_xlabel("IoU")
            ax1.set_title("Intersection over Union (IoU) по методам")
            for bar, val in zip(bars1, df["iou"]):
                ax1.text(
                    val + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}",
                    va="center",
                )

            # Dice coefficient сравнение
            ax2 = axes[0, 1]
            bars2 = ax2.barh(df["method"], df["dice"])
            ax2.set_xlabel("Dice Coefficient")
            ax2.set_title("Dice Coefficient по методам")
            for bar, val in zip(bars2, df["dice"]):
                ax2.text(
                    val + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}",
                    va="center",
                )

            # F1 Score сравнение
            ax3 = axes[1, 0]
            bars3 = ax3.barh(df["method"], df["f1_score"])
            ax3.set_xlabel("F1 Score")
            ax3.set_title("F1 Score по методам")
            for bar, val in zip(bars3, df["f1_score"]):
                ax3.text(
                    val + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}",
                    va="center",
                )

            # Время выполнения
            ax4 = axes[1, 1]
            bars4 = ax4.barh(df["method"], df["time"])
            ax4.set_xlabel("Время (секунды)")
            ax4.set_title("Время выполнения по методам")
            for bar, val in zip(bars4, df["time"]):
                ax4.text(
                    val + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}s",
                    va="center",
                )

            plt.tight_layout()
            plots_path = os.path.join(metrics_dir, "metrics_comparison_plots.jpg")
            plt.savefig(plots_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"📊 Графики метрик сохранены: {plots_path}")

            # График 2: Scatter plot IoU vs Время
            fig, ax = plt.subplots(figsize=(10, 6))
            _ = ax.scatter(df["time"], df["iou"], s=100, alpha=0.7)
            for i, row in df.iterrows():
                ax.annotate(
                    row["method"],
                    (row["time"], row["iou"]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                )

            ax.set_xlabel("Время выполнения (секунды)")
            ax.set_ylabel("IoU")
            ax.set_title("Соотношение точности и скорости")
            ax.grid(True, alpha=0.3)
            scatter_path = os.path.join(metrics_dir, "iou_vs_time_scatter.jpg")
            plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        except Exception as e:
            print(f"⚠️ Ошибка создания графиков метрик: {e}")

    # ──────────────────────────────────────────────────────────────────────
    def _create_benchmark_preview(
        self, df: pd.DataFrame, output_dir: PathLike, comp_dir: str
    ) -> None:
        """
        Создаёт превью-галерею результатов всех методов, отсортированных по скорости.

        Args:
            df: DataFrame с результатами.
            output_dir: Базовая директория.
            comp_dir: Директория для сохранения превью.
        """
        images_dir: str = os.path.join(output_dir, "images")
        result_files: List[str] = [
            f for f in os.listdir(images_dir) if f.endswith("_result.jpg")
        ]

        if not result_files:
            return
        sorted_methods: List[Any] = df.sort_values("Mean_Time_s")["Method"].tolist()

        images: List[Image.Image] = []
        titles: List[str] = []
        for method in sorted_methods:
            result_file: str = f"{method}_result.jpg"
            if result_file in result_files:
                img_path: str = os.path.join(images_dir, result_file)
                img: Image.Image = Image.open(img_path)
                images.append(img)
                method_data = df[df["Method"] == method]
                if not method_data.empty:
                    time_val: float = method_data.iloc[0]["Mean_Time_s"]
                    mask_percent: float = (
                        method_data.iloc[0]["Mask_Percentage"]
                        if "Mask_Percentage" in method_data.columns
                        else 0
                    )
                    title: str = f"{method}\n{time_val:.3f}s, {mask_percent:.3f}%"
                else:
                    title = method
                titles.append(title)
        n_images: int = len(images)
        n_cols: int = 4
        n_rows: int = (n_images + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
        axes = axes.flatten()

        for i, (img, title) in enumerate(zip(images, titles)):
            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=8)
            axes[i].axis("off")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle(
            "Превью результатов сегментации (сортировка по скорости)", fontsize=14
        )
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        preview_path: str = os.path.join(comp_dir, "methods_preview.jpg")
        plt.savefig(preview_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ──────────────────────────────────────────────────────────────────────
    def benchmark_all_methods(
        self,
        image: ImageInput,
        method_names: Optional[List[str]] = None,
        n_runs: int = 3,
        test_name: str = "benchmark",
        ground_truth: Optional[MaskArray] = None,
    ) -> pd.DataFrame:
        """
        Упрощённый бенчмарк всех зарегистрированных методов: время + метрики.

        Args:
            image: Входное изображение.
            method_names: Список методов. Если `None`, все зарегистрированные.
            n_runs: Количество прогонов.
            test_name: Префикс имени теста.
            ground_truth: GT-маска для метрик.

        Returns:
            pd.DataFrame: Сводная таблица с метриками и временем.
        """
        if method_names is None:
            method_names = list(self.methods.keys())

        records: List[Dict[str, Any]] = []
        for method_name in method_names:
            if method_name not in self.methods:
                print(f"⚠️ Метод {method_name} не найден, пропускаем")
                continue
            segmenter = self.methods[method_name]
            self._ensure_warmup(
                method_name,
                segmenter,
                np.array(image) if not isinstance(image, np.ndarray) else image,
            )

            times: List[float] = []
            last_result = None
            for _ in range(n_runs):
                if str(self.device) == "cuda":
                    torch.cuda.synchronize()
                t0: float = time.perf_counter()
                try:
                    result_img, mask = segmenter.segment_with_mask(image)
                    if str(self.device) == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                    last_result = (result_img, mask)
                except Exception as e:
                    print(f"⚠️ {method_name}: {e}")
                    break

            if not times or last_result is None:
                records.append({"method": method_name, "error": "failed"})
                continue

            result_img, mask = last_result

            if mask is not None:
                mask_area: int = int(np.sum(mask > 0))
                total_px: int = int(mask.shape[0] * mask.shape[1])
            else:
                mask_area = 0
                total_px = 1

            record: Dict[str, Any] = {
                "method": method_name,
                "mean_time_s": float(np.mean(times)),
                "std_time_s": float(np.std(times)),
                "min_time_s": float(np.min(times)),
                "mask_area_px": int(mask_area),
                "mask_pct": float(100.0 * mask_area / total_px),
            }

            if ground_truth is not None:
                try:
                    if mask is not None and ground_truth is not None:
                        m: MetricsDict = SegmentationMetrics.calculate_all_metrics(
                            mask, ground_truth, threshold=0.5
                        )
                    else:
                        print("⚠️ Cannot compute metrics: mask or ground_truth is None")
                        m = {}
                    record.update(
                        {
                            "iou": m.get("iou", float("nan")),
                            "dice": m.get("dice", float("nan")),
                            "pixel_accuracy": m.get("pixel_accuracy", float("nan")),
                            "precision": m.get("precision", float("nan")),
                            "recall": m.get("recall", float("nan")),
                        }
                    )
                except Exception as e:
                    print(f"⚠️ Метрики для {method_name}: {e}")

            records.append(record)
            print(
                f"  ✅ {method_name}: {record['mean_time_s'] * 1000:.1f}ms "
                f"(±{record['std_time_s'] * 1000:.1f}ms), mask={record['mask_pct']:.1f}%"
            )

        df: pd.DataFrame = pd.DataFrame(records)
        if not df.empty and "mean_time_s" in df.columns:
            df = df.sort_values("mean_time_s")
        return df

    # ──────────────────────────────────────────────────────────────────────
    def visualize_comparison(
        self,
        results: Dict[str, Dict],
        show_masks: bool = True,
        save_visualization: bool = True,
        output_dir: Optional[PathLike] = None,
        show_plots: bool = True,
    ) -> None:
        """
        Визуализация сравнения результатов с сохранением.

        Строит 1 или 2 ряда: [Результаты] + [Маски] (если `show_masks=True`).

        Args:
            results: Результаты `compare_methods()` или `test_single_method()`.
            show_masks: Показывать ли бинарные маски отдельно.
            save_visualization: Сохранять ли итоговую визуализацию.
            output_dir: Директория для сохранения.
            show_plots: Показывать ли через `plt.show()`.
        """
        if output_dir is None and self.current_test_id:
            output_dir = os.path.join(self.base_output_dir, self.current_test_id)
        elif output_dir is None:
            output_dir = self._create_test_directory("./data/visualization")
        n_methods: int = len(results)

        if show_masks:
            fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 10))

            for i, (method_name, result) in enumerate(results.items()):
                # Результат
                axes[0, i].imshow(result["result"])
                title = f"{method_name}\n{result['time']:.2f}s"
                axes[0, i].set_title(title, fontsize=10)
                axes[0, i].axis("off")

                # Маска
                axes[1, i].imshow(result["mask"], cmap="gray")
                mask_title: str = f"Mask\n{result['mask_percentage']:.3f}%"
                axes[1, i].set_title(mask_title, fontsize=10)
                axes[1, i].axis("off")
        else:
            fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))

            for i, (method_name, result) in enumerate(results.items()):
                axes[i].imshow(result["result"])
                title = f"{method_name}\n{result['time']:.2f}s, {result['mask_percentage']:.3f}%"
                axes[i].set_title(title, fontsize=10)
                axes[i].axis("off")

        plt.suptitle(
            "Визуализация результатов сегментации", fontsize=14, fontweight="bold"
        )
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        # Сохраняем визуализацию
        if save_visualization:
            vis_dir: str = os.path.join(output_dir, "comparisons")
            os.makedirs(vis_dir, exist_ok=True)

            vis_path: str = os.path.join(vis_dir, "results_visualization.jpg")
            plt.savefig(vis_path, dpi=150, bbox_inches="tight")
            print(f"🖼️ Визуализация сохранена: {vis_path}")
            for method_name, result in results.items():
                result_fig, result_ax = plt.subplots(1, 1, figsize=(8, 6))
                result_ax.imshow(result["result"])
                result_ax.set_title(f"{method_name} - Result", fontsize=12)
                result_ax.axis("off")

                result_path: str = os.path.join(
                    output_dir, "images", f"{method_name}_large.jpg"
                )
                result_fig.savefig(result_path, dpi=150, bbox_inches="tight")
                plt.close(result_fig)
                mask_fig, mask_ax = plt.subplots(1, 1, figsize=(8, 6))
                mask_ax.imshow(result["mask"], cmap="gray")
                mask_ax.set_title(f"{method_name} - Mask", fontsize=12)
                mask_ax.axis("off")

                mask_path: str = os.path.join(
                    output_dir, "masks", f"{method_name}_large_mask.jpg"
                )
                mask_fig.savefig(mask_path, dpi=150, bbox_inches="tight")
                plt.close(mask_fig)

        if show_plots:
            plt.show()
        else:
            plt.close()

    # ──────────────────────────────────────────────────────────────────────
    def save_results(
        self,
        results: Dict[str, Dict],
        output_dir: Optional[PathLike] = "./../data/segmentation_results",
    ) -> None:
        """
        Сохранение результатов всех методов в указанную директорию.

        Args:
            results: Результаты тестирования.
            output_dir: Директория для сохранения. Если `None`, создаётся новая.
        """

        if output_dir is None:
            output_dir = self._create_test_directory("./../data/results_save")

        print(f"💾 Сохранение результатов в {output_dir}...")

        for method_name, result in results.items():
            # Сохраняем результат
            result_path: str = os.path.join(output_dir, f"{method_name}_result.jpg")
            result_img: Image.Image = Image.fromarray(result["result"].astype(np.uint8))
            result_img.save(result_path)

            # Сохраняем маску
            mask_path: str = os.path.join(output_dir, f"{method_name}_mask.png")
            mask_img: Image.Image = Image.fromarray(result["mask"].astype(np.uint8))
            mask_img.save(mask_path)

            # Сохраняем статистику
            stats_path: str = os.path.join(output_dir, f"{method_name}_stats.txt")
            with open(stats_path, "w") as f:
                f.write(f"Method: {method_name}\n")
                f.write(f"Execution Time: {result['time']:.3f}s\n")
                f.write(f"Mask Area: {result['mask_area']} pixels\n")
                f.write(f"Mask Percentage: {result['mask_percentage']:.2f}%\n")
                f.write(f"Image Shape: {result['image_shape']}\n")
                if "timestamp" in result:
                    f.write(f"Timestamp: {result['timestamp']}\n")
        print(f"✅ Все результаты сохранены в директории: {output_dir}")
