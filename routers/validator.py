# routers/validator.py

"""Модуль API для валидации кросс-библиотечных реализаций методов сегментации.

Предоставляет REST-интерфейс для:
- Асинхронного сравнения реализаций одного метода в разных библиотеках
- Вычисления метрик согласованности (IoU, Dice, F1, Pixel Accuracy, MAE, Hausdorff)
- Генерации визуализаций различий между масками
- Классификации результатов валидации (PASS/WARNING/FAIL по пороговым значениям)

Поддерживаемые библиотеки для сравнения:
- PyTorch (primary): TorchSegmenter, TorchSegmenter2
- OpenCV (reference): OpenCVSegmenter
- scikit-learn (reference): SklearnSegmenter

Поддерживаемые категории методов:
- threshold: global, otsu, adaptive, niblack, sauvola, bernsen, phansalkar, ...
- edge: canny, sobel, prewitt, scharr, roberts, log, dog, marr_hildreth, ...
- region: region_growing, split_and_merge, floodfill
- clustering: kmeans, dbscan, meanshift

Особенности:
- Асинхронное выполнение через asyncio.create_task()
- Прогресс-трекинг с обновлением в реальном времени (пройдено/всего методов)
- Потокобезопасное обновление статуса через asyncio.Lock
- Безопасная сериализация numpy-типов и base64-изображений в JSON
- Поддержка фильтрации методов по категории (threshold/edge/region/clustering/all)
- Детальные метрики времени выполнения для каждой реализации

Пример использования:
```python
# 1. Запуск валидации
POST /api/validate/start
Content-Type: multipart/form-data

file: (изображение для тестирования)
primary_library: "torch"
reference_library: "opencv"
methods_filter: "threshold"  # или "edge", "region", "clustering", "all"

# Response: {"task_id": "uuid...", "status": "running"}

# 2. Проверка статуса
GET /api/validate/status/{task_id}
# Response: {
#     "task_id": "uuid...",
#     "status": "running"|"completed"|"failed",
#     "progress": 45.5,
#     "processed": 3,
#     "total_methods": 10,
#     "elapsed_ms": 1234.5,
#     "message": "Обработка otsu_thresholding (3/10)",
#     "results": [...]  # только при completion
# }

# 3. Результаты валидации включают:
{
    "summary": [
        {
            "method": "otsu_thresholding",
            "success": true,
            "validation_status": "PASS",
            "iou": 0.99,
            "dice": 0.995,
            "f1_score": 0.995,
            "primary_time": 0.042,
            "reference_time": 0.038,
            "time_diff": 0.004,
            "primary_mask_b64": "data:image/png;base64,...",
            "reference_mask_b64": "data:image/png;base64,...",
            "difference_b64": "data:image/png;base64,..."
        },
        ...
    ],
    "benchmark": {
        "methods_count": 10,
        "passed": 8,
        "warning": 1,
        "failed": 1,
        "avg_torch_time": 0.045,
        "avg_iou": 0.97,
        "data": [...]  # детальные метрики по каждому методу
    }
}

Атрибуты модуля:
router (APIRouter): Экземпляр FastAPI router с префиксом "/api/validate".
_validation_tasks (ValidationTaskDict): In-memory хранилище задач валидации.
_validation_lock (asyncio.Lock): Асинхронный лок для потокобезопасного обновления.
Зависимости:
- FastAPI: веб-фреймворк для REST API
- numpy, PIL: обработка изображений и массивов
- testing.TorchImplementationValidator: ядро логики валидации
- segmenters.*: фабрики сегментеров для разных библиотек
- metrics.SegmentationMetrics: вычисление метрик качества
Примечания:
- В продакшене рекомендуется заменить _validation_tasks на Redis/Celery.
- Все изображения конвертируются в RGB перед сегментацией.
- Метрики с NaN/Inf автоматически заменяются на null в JSON-ответах.
- Base64-изображения могут быть большими: рассмотрите отключение в продакшене.
- Пороговые значения для PASS/WARNING/FAIL настраиваются в SegmentationMetrics.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций

import os
import sys
import uuid
import asyncio
import json
import time
import io
import base64
import logging
import traceback
import cv2
from typing import Dict, Any, Optional, List, Tuple, Callable, TypeAlias
import torch
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testing.TorchImplementationValidator import (
    TorchImplementationValidator,
    MethodConfig,
)
from segmenters.TorchSegmenter import TorchSegmenter
from segmenters.NewTorchSegmenter import TorchSegmenter2
from segmenters.OpenCVSegmenter import OpenCVSegmenter
from segmenters.SklearnSegmenter import SklearnSegmenter
from metrics.SegmentationMetrics import SegmentationMetrics, MetricsDict

# logging.basicConfig(level=logging.INFO)
# Настройка логгера
logger: logging.Logger = logging.getLogger("validate")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler: logging.StreamHandler = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    log_dir: Path = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file: Path = log_dir / "validate.log"

    file_handler: logging.FileHandler = logging.FileHandler(
        filename=log_file,
        mode="a",  # 'a' = append, 'w' = overwrite
        encoding="utf-8",  # важно для кириллицы и спецсимволов
        delay=True,  # откладывает создание файла до первой записи
    )

    file_formatter: logging.Formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.info(f"✅ Логгер инициализирован. Логи пишутся в: {log_file.resolve()}")

router: APIRouter = APIRouter(prefix="/api/validate", tags=["validate"])

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & TASK STORAGE
# ──────────────────────────────────────────────────────────────────────
ValidationTaskDict: TypeAlias = Dict[str, Dict[str, Any]]
"""Тип-алиас для хранилища задач валидации, dtype=Dict[str, Dict[str, Any]].

Структура задачи:
{
    "status": str,              # "running" | "completed" | "failed" | "cancelled"
    "progress": float,          # 0.0–100.0
    "message": str,             # Человекочитаемое описание этапа
    "results": Optional[Dict],  # Результаты при completion
    "error": Optional[str],     # Сообщение об ошибке
    "fetched": bool,            # Флаг получения результатов клиентом
    "start_time": float,        # Время начала выполнения (perf_counter)
    "elapsed_ms": float,        # Прошедшее время в миллисекундах
    "processed": int,           # Количество обработанных методов
    "total_methods": int,       # Общее количество методов для обработки
}

Example:
python task: ValidationTaskDict = { "status": "running", "progress": 45.5, "message": "Обработка otsu_thresholding (3/10)", "results": None, "error": None, "fetched": False, "start_time": 123456.789, "elapsed_ms": 1234.5, "processed": 3, "total_methods": 10 }
"""

_validation_tasks: ValidationTaskDict = {}
"""Глобальное хранилище задач валидации в памяти, dtype=ValidationTaskDict.

Ключ: UUID задачи (str).

Значение: Словарь со статусом, прогрессом, результатами и метаданными.

Warning:
В продакшене следует заменить на Redis/Celery для:
- Масштабируемости между воркерами
- Сохранения состояния при перезапуске
- Очистки устаревших задач по TTL
- Распределённой блокировки вместо локального asyncio.Lock
"""

_validation_lock: asyncio.Lock = asyncio.Lock()
"""Асинхронный лок для потокобезопасного обновления _validation_tasks, dtype=asyncio.Lock.

Используется во всех операциях чтения/записи задач для предотвращения состояний гонки при параллельных запросах к одному task_id.

Example:
python async with _validation_lock: _validation_tasks[task_id]["progress"] = 50.0 _validation_tasks[task_id]["processed"] += 1
"""


# 🔹 Энкодер для NaN/Infinity
class NumpyEncoder(json.JSONEncoder):
    """Кастомный JSON-энкодер для безопасной сериализации numpy-типов.

    Обрабатывает специальные случаи:
    - numpy.integer → int
    - numpy.floating → float (с обработкой NaN/Inf → None)
    - numpy.ndarray → list (через tolist())

    Пример использования:
        ```python
        import json
        import numpy as np

        data = {"iou": np.float32(0.85), "mask": np.array([1, 0, 1])}
        json_str = json.dumps(data, cls=NumpyEncoder)
        # Result: {"iou": 0.85, "mask": [1, 0, 1]}
        ```

    Methods:
        default(self, obj: Any) -> Any: Переопределённый метод сериализации.
    """

    def default(self, obj: Any) -> Any:
        """Сериализует объект, не поддерживаемый стандартным JSONEncoder.

        Args:
            obj: Объект для сериализации.

        Returns:
            Any: JSON-совместимое представление объекта.

        Raises:
            TypeError: Если объект не может быть сериализован.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ──────────────────────────────────────────────────────────────────────
def safe_json_response(content: Any, status_code: int = 200) -> JSONResponse:
    """Возвращает FastAPI JSONResponse с безопасной сериализацией контента.

    Использует `NumpyEncoder` для обработки numpy-типов и специальных значений.

    Args:
        content: Данные для ответа (словарь, список, примитивы).
        status_code: HTTP-статус код (по умолчанию 200).

    Returns:
        JSONResponse: Ответ с заголовком application/json.

    Example:
        ```python
        @router.get("/metrics")
        async def get_metrics():
            metrics = {"iou": np.float32(0.78), "methods": np.array(["cv2", "sk"])}
            return safe_json_response(metrics)
        ```
    """
    return JSONResponse(content=content, status_code=status_code, media_type="application/json")


# ──────────────────────────────────────────────────────────────────────
def arr_to_b64(arr: np.ndarray) -> str:
    """Конвертирует numpy-массив в data:image/png;base64,... строку.

    Обрабатывает:
    - Конвертацию float [0,1] → uint8 [0,255] при необходимости.
    - Удаление лишнего измерения для одноканальных масок.
    - Кодирование в PNG через PIL и base64.

    Args:
        arr: Numpy-массив изображения или маски.

    Returns:
        str: Строка формата "data:image/png;base64,{base64_encoded_data}".

    Example:
        ```python
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        b64_str = arr_to_b64(mask)
        # Use in HTML: <img src="{b64_str}">
        ```
    """
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────
def _get_methods_for_filter(
    validator: TorchImplementationValidator, methods_filter: Optional[str]
) -> List[Tuple[str, Dict[str, Any]]]:
    """Возвращает список методов для валидации по заданному фильтру.

    Фильтры:
    - "threshold": пороговые методы (global, otsu, adaptive, niblack, ...)
    - "edge": граничные методы (canny, sobel, prewitt, scharr, ...)
    - "region": региональные методы (region_growing, split_and_merge, floodfill)
    - "clustering": методы кластеризации (kmeans, dbscan, meanshift)
    - None или "all": все доступные методы

    Args:
        validator: Экземпляр TorchImplementationValidator с реестром методов.
        methods_filter: Строка-фильтр категории методов.

    Returns:
        List[Tuple[str, Dict[str, Any]]]: Список кортежей (имя_метода, параметры).

    Example:
        ```python
        methods = _get_methods_for_filter(validator, "threshold")
        # Returns: [("otsu_thresholding", {}), ("adaptive_thresholding", {...}), ...]
        ```
    """
    mapping: Dict[str, List[MethodConfig]] = {
        "threshold": validator.threshold_methods,
        "edge": validator.edge_methods,
        "region": validator.region_methods,
        "clustering": validator.clastering_methods,
    }
    if methods_filter in mapping:
        return mapping[methods_filter]
    return (
        validator.threshold_methods + validator.edge_methods + validator.region_methods + validator.clastering_methods
    )


# ──────────────────────────────────────────────────────────────────────
# BACKGROUND TASK
# ──────────────────────────────────────────────────────────────────────
async def _run_validation_task(
    task_id: str,
    file_content: bytes,
    primary_library: str,
    reference_library: str,
    methods_filter: Optional[str],
) -> None:
    """Асинхронная фоновая задача валидации кросс-библиотечных реализаций.

    Основной рабочий процесс:
    1. Инициализация задачи в _validation_tasks со статусом "running".
    2. Загрузка и предобработка входного изображения.
    3. Инициализация TorchImplementationValidator.
    4. Фильтрация методов по категории (threshold/edge/region/clustering).
    5. Поэтапная валидация каждого метода:
    - Создание сегментеров для primary и reference библиотек.
    - Запуск сегментации с замером времени.
    - Вычисление метрик согласованности (IoU, Dice, F1, MAE, Hausdorff).
    - Классификация результата (PASS/WARNING/FAIL).
    - Генерация base64-визуализаций (оригинал, маски, разница).
    6. Агрегация результатов и подготовка сводной статистики.
    7. Обновление статуса на "completed" с результатами или "failed" с ошибкой.

    Обновление прогресса:
    - 0–10%: Инициализация и загрузка изображения.
    - 10–90%: Пошаговая валидация методов (равномерное распределение).
    - 90–100%: Агрегация, сохранение и финализация.

    Обработка ошибок:
    - Каждый метод обрабатывается в try/except: неудача одного не прерывает остальные.
    - Ошибки логируются с уровнем ERROR и traceback (при DEBUG).
    - При критической ошибке задача помечается "failed" с деталями.

    Args:
        task_id (str): Уникальный идентификатор задачи (UUID).
        file_content (bytes): Содержимое загруженного файла изображения.
        primary_library (str): Библиотека для "первичной" реализации:
            "torch" | "opencv" | "sklearn".
        reference_library (str): Библиотека для "референсной" реализации.
        methods_filter (Optional[str]): Фильтр категории методов:
            "threshold" | "edge" | "region" | "clustering" | None (all).

    Side Effects:
        - Обновляет _validation_tasks[task_id] в реальном времени.
        - Создаёт директорию ./data/validation_web для отчётов.
        - Генерирует base64-изображения для inline-отображения в веб-интерфейсе.
        - Очищает память после обработки каждого метода (gc.collect()).

    Note:
        - Функция не возвращает значение: результаты передаются через _validation_tasks.
        - Для продакшена рекомендуется вынести логику в Celery/RQ с отдельным worker.
        - Base64-изображения могут быть большими: рассмотрите отключение в продакшене.
    """
    t0: float = time.perf_counter()
    async with _validation_lock:
        _validation_tasks[task_id] = {
            "status": "running",
            "progress": 0,
            "message": "Инициализация...",
            "results": None,
            "error": None,
            "fetched": False,
            "start_time": time.time(),
            "elapsed_ms": 0,
            "processed": 0,
            "total_methods": 0,
        }

    try:
        img_array: np.ndarray = np.array(Image.open(io.BytesIO(file_content)).convert("RGB"))
        validator = TorchImplementationValidator(output_dir="./data/validation_web")
        methods_list: List[Tuple[str, Dict[str, Any]]] = _get_methods_for_filter(validator, methods_filter)
        total_methods: int = len(methods_list)

        async with _validation_lock:
            _validation_tasks[task_id]["total_methods"] = total_methods
            _validation_tasks[task_id]["progress"] = 5
            _validation_tasks[task_id]["message"] = f"Запущено {total_methods} методов"

        CLASS_MAP: Dict[str, Any] = {
            "torch": TorchSegmenter,
            "opencv": OpenCVSegmenter,
            "sklearn": SklearnSegmenter,
            "torch_v2": TorchSegmenter2,
        }
        primary_class = CLASS_MAP.get(primary_library, TorchSegmenter)
        reference_class = CLASS_MAP.get(reference_library, OpenCVSegmenter)

        results: Dict[str, Any] = {}

        # 🔹 Пошаговое выполнение с обновлением прогресса
        for idx, (method_name, params) in enumerate(methods_list):
            progress: float = 10 + (idx / total_methods) * 80
            async with _validation_lock:
                _validation_tasks[task_id]["progress"] = round(progress, 2)
                _validation_tasks[task_id]["message"] = f"Обработка {method_name} ({idx + 1}/{total_methods})"
            await asyncio.sleep(0)

            try:
                result: Dict[str, Any] = await asyncio.to_thread(
                    _process_single_method,
                    method_name,
                    params,
                    img_array,
                    primary_class,
                    reference_class,
                    validator,
                )
                results[method_name] = result
            except Exception as e:
                logger.error(f"❌ Error processing {method_name}: {e}")
                results[method_name] = {"success": False, "error": str(e)}

            elapsed_ms = round((time.time() - _validation_tasks[task_id]["start_time"]) * 1000, 1)
            async with _validation_lock:
                _validation_tasks[task_id]["processed"] = idx + 1
                _validation_tasks[task_id]["progress"] = round(10 + (idx + 1) / total_methods * 80, 2)
                _validation_tasks[task_id]["elapsed_ms"] = elapsed_ms
                _validation_tasks[task_id]["message"] = f"Завершён {method_name} ({idx + 1}/{total_methods})"
            await asyncio.sleep(0)

        # 🔹 Финализация
        async with _validation_lock:
            _validation_tasks[task_id]["progress"] = 95
            _validation_tasks[task_id]["message"] = "Сохранение результатов..."

        # 🔹 Подготовка ответа
        summary: List[Dict[str, Any]] = []
        for method, data in results.items():
            if not isinstance(data, dict) or not data.get("success"):
                summary.append({"method": method, "success": False, "error": data.get("error")})
                continue
            metrics: Dict[str, Any] = data.get("metrics", {})
            summary.append(
                {
                    "method": method,
                    "success": True,
                    "validation_status": data.get("validation_status"),
                    "iou": metrics.get("iou"),
                    "dice": metrics.get("dice"),
                    "pixel_accuracy": metrics.get("pixel_accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1_score": metrics.get("f1_score"),
                    "mae": metrics.get("mae"),
                    "hausdorff_distance": metrics.get("hausdorff_distance"),
                    "primary_time": data.get("primary_time"),
                    "reference_time": data.get("reference_time"),
                    "time_diff": data.get("methods_time_difference"),
                    "original_b64": data.get("original_b64"),
                    "primary_mask_b64": data.get("primary_mask_b64"),
                    "reference_mask_b64": data.get("reference_mask_b64"),
                    "difference_b64": data.get("difference_b64"),
                }
            )

        # 🔹 Бенчмарк-данные
        benchmark_data: List[Dict[str, Any]] = []
        for method, data in results.items():
            if isinstance(data, dict) and data.get("success"):
                metrics = data.get("metrics", {})
                pred_area: int = metrics.get("predicted_area", 0) or 0
                gt_area: int = metrics.get("ground_truth_area", 0) or 0
                coverage_pct: float = (pred_area / gt_area * 100) if gt_area > 0 else 0
                benchmark_data.append(
                    {
                        "method": method,
                        "torch_time": data.get("primary_time"),
                        "reference_time": data.get("reference_time"),
                        "time_diff": data.get("time_diff"),
                        "accuracy": metrics.get("accuracy"),
                        "iou": metrics.get("iou"),
                        "dice": metrics.get("dice"),
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1_score": metrics.get("f1_score"),
                        "mae": metrics.get("mae"),
                        "pixel_accuracy": metrics.get("pixel_accuracy"),
                        "hausdorff_distance": metrics.get("hausdorff_distance"),
                        "area_ratio": metrics.get("area_ratio"),
                        "validation_status": data.get("validation_status"),
                        "coverage_pct": round(coverage_pct, 2),
                        "predicted_area": metrics.get("predicted_area"),
                        "ground_truth_area": metrics.get("ground_truth_area"),
                        "area_difference": metrics.get("area_difference"),
                    }
                )

        valid_times: List[Any] = [d["torch_time"] for d in benchmark_data if d.get("torch_time")]
        valid_iou: List[Any] = [d["iou"] for d in benchmark_data if d.get("iou") is not None]
        elapsed_ms = round((time.time() - _validation_tasks[task_id]["start_time"]) * 1000, 1)

        async with _validation_lock:
            _validation_tasks[task_id].update(
                {
                    "status": "completed",
                    "progress": 100,
                    "message": "Готово",
                    "elapsed_ms": elapsed_ms,
                    "fetched": False,
                    "results": {
                        "summary": summary,
                        "passed": sum(1 for s in summary if s.get("validation_status") == "PASS"),
                        "warning": sum(1 for s in summary if s.get("validation_status") == "WARNING"),
                        "failed": sum(1 for s in summary if s.get("validation_status") == "FAIL"),
                        "methods_tested": len(summary),
                        "report_dir": "./data/validation_web",
                        "benchmark": {
                            "methods_count": len(benchmark_data),
                            "passed": sum(1 for s in summary if s.get("validation_status") == "PASS"),
                            "warning": sum(1 for s in summary if s.get("validation_status") == "WARNING"),
                            "failed": sum(1 for s in summary if s.get("validation_status") == "FAIL"),
                            "data": benchmark_data,
                            "avg_torch_time": (sum(valid_times) / len(valid_times) if valid_times else 0),
                            "avg_iou": (sum(valid_iou) / len(valid_iou) if valid_iou else 0),
                        },
                        "benchmark_raw": [
                            {
                                "method": m,
                                "torch_time": d.get("primary_time"),
                                "reference_time": d.get("reference_time"),
                                "iou": d.get("metrics", {}).get("iou"),
                                "status": d.get("validation_status"),
                            }
                            for m, d in results.items()
                            if isinstance(d, dict) and d.get("success")
                        ],
                    },
                }
            )
        elapsed: float = (time.perf_counter() - t0) * 1000
        print(f"Elapsed_time: {round(elapsed, 1)}")

    except Exception as e:
        logger.error(f"Validation task {task_id} failed: {e}", exc_info=True)
        async with _validation_lock:
            start: float = _validation_tasks[task_id].get("start_time", time.perf_counter())
            _validation_tasks[task_id].update(
                {
                    "status": "failed",
                    "message": str(e),
                    "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
                    "error_details": {
                        "error_type": type(e).__name__,
                        "failed_at": _validation_tasks[task_id]["message"],
                        "traceback": (traceback.format_exc() if logger.level == logging.DEBUG else None),
                    },
                }
            )
        logger.error(f"❌ Validation task {task_id} failed: {e}")


# ──────────────────────────────────────────────────────────────────────
def _process_single_method(
    method_name: str,
    params: Dict[str, Any],
    img_array: np.ndarray,
    primary_class: Callable,
    reference_class: Callable,
    validator: Any,
) -> Dict[str, Any]:
    """Синхронная обработка одного метода валидации.

    Выполняет кросс-библиотечное сравнение реализаций одного метода:
    1. Создание сегментеров для primary и reference библиотек.
    2. Запуск сегментации с замером времени выполнения.
    3. Вычисление метрик согласованности через SegmentationMetrics.
    4. Классификация результата через _check_validation_status.
    5. Генерация base64-визуализаций для веб-интерфейса.

    Args:
        method_name (str): Название метода сегментации.
        params (Dict[str, Any]): Параметры инициализации сегментера.
        img_array (np.ndarray): Входное изображение (RGB или grayscale).
        primary_class: Класс сегментера для первичной библиотеки.
        reference_class: Класс сегментера для референсной библиотеки.
        validator: Экземпляр TorchImplementationValidator для проверки статуса.

    Returns:
        Dict[str, Any]: Словарь с результатами валидации:
            - success (bool): Успешность выполнения.
            - validation_status (str): "PASS" | "WARNING" | "FAIL".
            - metrics (MetricsDict): Словарь вычисленных метрик.
            - primary_time, reference_time (float): Время выполнения в секундах.
            - original_b64, primary_mask_b64, reference_mask_b64, difference_b64 (str):
            Base64-кодированные изображения для визуализации.

    Raises:
        Exception: При ошибке в сегментации или вычислении метрик.

    Note:
        - Функция выполняется в отдельном потоке через asyncio.to_thread().
        - Base64-строки могут быть большими: логгируется предупреждение при >500 КБ.
    """
    logger.info(f"🔹 START Processing method: {method_name}")
    try:
        t1: float = time.perf_counter()
        if primary_class == TorchSegmenter2:
            params_primary = {
                "precision": params.get("precision", "fp32"),
                "use_compile": params.get("use_compile", False),
                "device": params.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                **{k: v for k, v in params.items() if k not in ["precision", "use_compile", "device"]},
            }
            seg1 = primary_class(method=method_name, **params_primary)
        else:
            seg1 = primary_class(method=method_name, **params)
        mask1: np.ndarray = seg1.segment(img_array, **params)
        time1: float = time.perf_counter() - t1
        logger.info(f"✅ {method_name} primary done: {time1:.3f}s")

        ref_params: Dict[str, Any] = params.copy()
        if reference_class == TorchSegmenter2:
            ref_params["postprocess"] = False  # TorchSegmenter2 не требует postprocess
            ref_params.setdefault("precision", "fp32")
            ref_params.setdefault("use_compile", False)
        else:
            ref_params["postprocess"] = False  # для старых сегментеров
        t2: float = time.perf_counter()
        seg2 = reference_class(method=method_name, **ref_params)
        mask2: np.ndarray = seg2.segment(img_array, **ref_params)
        time2: float = time.perf_counter() - t2
        logger.info(f"✅ {method_name} primary done: {time2:.3f}s")

        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(mask1, mask2, threshold=0.5)
        metrics.update(
            {
                "first_method_time": time1,
                "second_method_time": time2,
                "methods_time_difference": abs(time1 - time2),
            }
        )
        status = validator._check_validation_status(metrics)

        orig_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if img_array.ndim == 3 else img_array
        mask1_vis: np.ndarray = mask1 if mask1.max() == 255 else mask1 * 255
        mask2_vis: np.ndarray = mask2 if mask2.max() == 255 else mask2 * 255
        diff: np.ndarray = np.abs(mask1_vis.astype(int) - mask2_vis.astype(int))

        logger.info(f"✅ FINISHED {method_name}")
        result: Dict[str, Any] = {
            "success": True,
            "validation_status": status,
            "metrics": metrics,
            "primary_time": time1,
            "reference_time": time2,
            "original_b64": arr_to_b64(orig_gray),
            "primary_mask_b64": arr_to_b64(mask1_vis),
            "reference_mask_b64": arr_to_b64(mask2_vis),
            "difference_b64": arr_to_b64(diff),
        }
        for key in [
            "original_b64",
            "primary_mask_b64",
            "reference_mask_b64",
            "difference_b64",
        ]:
            if key in result:
                size_kb: float = len(result[key]) / 1024
                logger.info(f"📦 {key}: {size_kb:.1f} KB")
                if size_kb > 500:  # >500KB
                    logger.warning(f"⚠️ Large base64 string: {key} ({size_kb:.1f} KB)")
        return result
    except Exception as e:
        logger.error(f"❌ ERROR in {method_name}: {type(e).__name__}: {e}")
        logger.error(f"❌ /api/validate error: {e}\n{traceback.format_exc()}")
        raise


# ──────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────
@router.post("/start")
async def start_validation(
    file: UploadFile = File(...),
    primary_library: str = Form("torch"),  # "torch" | "opencv" | "sklearn" | "torch_v2"
    reference_library: str = Form("opencv"),  # "torch" | "opencv" | "sklearn" | "torch_v2"
    methods_filter: Optional[str] = Form(None),  # "threshold" | "edge" | "region" | "all"
) -> Dict[str, str]:
    """Запускает асинхронную валидацию кросс-библиотечных реализаций.

    Создаёт новую задачу, инициализирует TorchImplementationValidator и планирует
    выполнение через asyncio.create_task().

    Args:
        file (UploadFile): Загружаемое изображение для валидации.
            Конвертируется в RGB numpy array перед обработкой.
        primary_library (str): Библиотека для "первичной" реализации.
            Варианты: "torch" (по умолчанию), "opencv", "sklearn".
        reference_library (str): Библиотека для "референсной" реализации.
            Варианты: "opencv" (по умолчанию), "torch", "sklearn".
        methods_filter (Optional[str]): Фильтр категории методов для валидации.
            Варианты: "threshold" | "edge" | "region" | "clustering" | None (all).

    Returns:
        Dict[str, str]: Словарь с идентификатором задачи и статусом:
            ```json
            {"task_id": "550e8400-e29b-41d4-a716-446655440000", "status": "running"}
            ```

    Example Request (multipart/form-data):
        ```
        POST /api/validate/start
        Content-Type: multipart/form-data

        file: (файл изображения)
        primary_library: "torch"
        reference_library: "opencv"
        methods_filter: "threshold"
        ```

    Note:
        - Задача выполняется асинхронно: ответ возвращается немедленно.
        - Прогресс отслеживается через эндпоинт `/status/{task_id}`.
        - Файл изображения читается в память: для больших файлов может потребоваться
        увеличение лимита `max_file_size` в конфигурации FastAPI.
    """
    file_content = await file.read()
    task_id: str = str(uuid.uuid4())
    temp_validator = TorchImplementationValidator(output_dir="./data/validation_web")
    methods_list: List[Tuple[str, Dict[str, Any]]] = _get_methods_for_filter(temp_validator, methods_filter)
    total_methods: int = len(methods_list)
    logger.info(f"🔧 methods_filter={methods_filter}, total_methods={total_methods}")
    asyncio.create_task(_run_validation_task(task_id, file_content, primary_library, reference_library, methods_filter))
    return {"task_id": task_id, "status": "running"}


# ──────────────────────────────────────────────────────────────────────
@router.get("/status/{task_id}")
async def get_validation_status(task_id: str) -> JSONResponse:
    """Возвращает статус и результаты задачи валидации.

    Args:
        task_id (str): UUID задачи, полученный при запуске через `/start`.

    Returns:
        JSONResponse: Сериализованный объект задачи с полями:
            - task_id (str): Идентификатор задачи.
            - status (str): "running" | "completed" | "failed" | "cancelled".
            - progress (float): Прогресс 0.0–100.0.
            - processed (int): Количество обработанных методов.
            - total_methods (int): Общее количество методов для обработки.
            - elapsed_ms (float): Прошедшее время в миллисекундах.
            - message (str): Описание текущего этапа.
            - results (List[Dict]): Результаты по каждому методу (при completion).
            - benchmark (Dict): Сводная статистика бенчмарка.
            - error_details (Dict): Детали ошибки (при failed).

    Raises:
        HTTPException (404): Если задача с task_id не найдена.

    Example Response (running):
        ```json
        {
            "task_id": "uuid...",
            "status": "running",
            "progress": 45.5,
            "processed": 3,
            "total_methods": 10,
            "elapsed_ms": 1234.5,
            "message": "Обработка otsu_thresholding (3/10)"
        }
        ```

    Example Response (completed):
        ```json
        {
            "task_id": "uuid...",
            "status": "completed",
            "progress": 100.0,
            "processed": 10,
            "total_methods": 10,
            "elapsed_ms": 5678.9,
            "message": "Готово",
            "results": [
                {
                    "method": "otsu_thresholding",
                    "success": true,
                    "validation_status": "PASS",
                    "iou": 0.99,
                    "f1_score": 0.995,
                    "primary_time": 0.042,
                    "reference_time": 0.038
                },
                ...
            ],
            "benchmark": {
                "methods_count": 10,
                "passed": 8,
                "avg_iou": 0.97
            }
        }
        ```

    Note:
        - Метрики с `NaN`/`Inf` автоматически заменяются на `null` перед сериализацией.
        - Base64-изображения включены в каждый элемент results для визуализации.
        - Ответ сериализуется через `safe_json_response()` для обработки numpy-типов.
    """
    async with _validation_lock:
        task: Optional[Dict[str, Any]] = _validation_tasks.get(task_id)

    if not task:
        raise HTTPException(404, detail="Task not found")

    if task["status"] in ("completed", "failed") and task.get("results") is not None:
        task["fetched"] = True

    response: Dict[str, Any] = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", 0),
        "processed": task.get("processed", 0),
        "total_methods": task.get("total_methods", 0),
        "elapsed_ms": task.get("elapsed_ms"),
        "message": task.get("message"),
    }

    if task["status"] == "completed" and task.get("results"):
        results_data = task["results"]
        if isinstance(results_data, dict) and "summary" in results_data:
            for key, value in results_data.items():
                if key != "results":
                    response[key] = value
            response["results"] = results_data.get("summary", [])

        elif isinstance(results_data, dict):
            summary: List[Dict[str, Any]] = []
            for method, data in results_data.items():
                if not isinstance(data, dict) or not data.get("success"):
                    summary.append({"method": method, "success": False, "error": data.get("error")})
                    continue

                metrics: Dict[str, Any] = data.get("metrics", {})
                summary.append(
                    {
                        "method": method,
                        "success": True,
                        "validation_status": data.get("validation_status"),
                        # Метрики
                        "iou": metrics.get("iou"),
                        "dice": metrics.get("dice"),
                        "pixel_accuracy": metrics.get("pixel_accuracy"),
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1_score": metrics.get("f1_score"),
                        "mae": metrics.get("mae"),
                        "hausdorff_distance": metrics.get("hausdorff_distance"),
                        # Временные метрики
                        "primary_time": data.get("primary_time"),
                        "reference_time": data.get("reference_time"),
                        "time_diff": data.get("methods_time_difference"),
                        # Base64 изображения
                        "original_b64": data.get("original_b64"),
                        "primary_mask_b64": data.get("primary_mask_b64"),
                        "reference_mask_b64": data.get("reference_mask_b64"),
                        "difference_b64": data.get("difference_b64"),
                    }
                )

            benchmark_data: List[Dict[str, Any]] = []
            for method, data in results_data.items():
                if not isinstance(data, dict) or not data.get("success"):
                    continue
                metrics = data.get("metrics", {})
                pred_area: int = metrics.get("predicted_area", 0) or 0
                gt_area: int = metrics.get("ground_truth_area", 0) or 0
                coverage_pct: float = (pred_area / gt_area * 100) if gt_area > 0 else 0

                benchmark_data.append(
                    {
                        "method": method,
                        "torch_time": data.get("primary_time"),
                        "reference_time": data.get("reference_time"),
                        "time_diff": data.get("time_diff"),
                        "accuracy": metrics.get("accuracy"),
                        "iou": metrics.get("iou"),
                        "dice": metrics.get("dice"),
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1_score": metrics.get("f1_score"),
                        "mae": metrics.get("mae"),
                        "pixel_accuracy": metrics.get("pixel_accuracy"),
                        "hausdorff_distance": metrics.get("hausdorff_distance"),
                        "area_ratio": metrics.get("area_ratio"),
                        "validation_status": data.get("validation_status"),
                        "coverage_pct": round(coverage_pct, 2),
                        "predicted_area": metrics.get("predicted_area"),
                        "ground_truth_area": metrics.get("ground_truth_area"),
                        "area_difference": metrics.get("area_difference"),
                    }
                )

            valid_times: List[Any] = [d["torch_time"] for d in benchmark_data if d.get("torch_time")]
            valid_iou: List[Any] = [d["iou"] for d in benchmark_data if d.get("iou") is not None]

            response.update(
                {
                    "results": summary,
                    "passed": sum(1 for s in summary if s.get("validation_status") == "PASS"),
                    "warning": sum(1 for s in summary if s.get("validation_status") == "WARNING"),
                    "failed": sum(1 for s in summary if s.get("validation_status") == "FAIL"),
                    "methods_tested": len(summary),
                    "report_dir": "./data/validation_web",
                    "benchmark": {
                        "methods_count": len(benchmark_data),
                        "passed": sum(1 for s in summary if s.get("validation_status") == "PASS"),
                        "warning": sum(1 for s in summary if s.get("validation_status") == "WARNING"),
                        "failed": sum(1 for s in summary if s.get("validation_status") == "FAIL"),
                        "data": benchmark_data,
                        "avg_torch_time": (sum(valid_times) / len(valid_times) if valid_times else 0),
                        "avg_iou": sum(valid_iou) / len(valid_iou) if valid_iou else 0,
                    },
                    "benchmark_raw": [
                        {
                            "method": method,
                            "torch_time": data.get("primary_time"),
                            "reference_time": data.get("reference_time"),
                            "iou": data.get("metrics", {}).get("iou"),
                            "status": data.get("validation_status"),
                        }
                        for method, data in results_data.items()
                        if isinstance(data, dict) and data.get("success")
                    ],
                }
            )

    elif task["status"] == "failed":
        response["error"] = task.get("error")
        if task.get("error_details"):
            response["error_details"] = task["error_details"]

    if response.get("results") and isinstance(response["results"], list):
        for item in response["results"]:
            if item.get("metrics"):
                for k, v in item["metrics"].items():
                    if isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
                        item["metrics"][k] = None

    logger.info(f"🔍 Returning response keys: {list(response.keys())}")
    if "results" in response:
        logger.info(f"🔍 response['results'] type: {type(response['results'])}")
        if isinstance(response["results"], list):
            logger.info(f"🔍 response['results'] length: {len(response['results'])}")
            if response["results"]:
                logger.info(f"🔍 first result keys: {list(response['results'][0].keys())}")
    return safe_json_response(response)


# ──────────────────────────────────────────────────────────────────────
@router.delete("/{task_id}")
async def cancel_validation(task_id: str) -> Dict[str, str]:
    """Отменяет задачу валидации.

    Логика поведения в зависимости от статуса задачи:
    - "running": Устанавливает status="cancelled", message="Отменено пользователем".
    - "completed"|"failed"|"cancelled": Возвращает текущий статус без изменений.
    - Не найдена: Возвращает {"status": "not_found"}.

    Args:
        task_id (str): UUID задачи для отмены.

    Returns:
        Dict[str, str]: Словарь с результатом операции:
            - При отмене: {"status": "cancelled"}
            - При уже завершённой: {"status": "completed", "message": "Already completed"}
            - При не найдено: {"status": "not_found", "message": "Task not found"}

    Example Responses:
        ```json
        // Отмена запущенной задачи
        {"status": "cancelled"}

        // Задача уже завершена
        {
            "status": "completed",
            "message": "Already completed"
        }

        // Задача не найдена
        {
            "status": "not_found",
            "message": "Task not found"
        }
        ```

    Note:
        - Отмена не прерывает выполнение кода немедленно: задача завершит текущий
        шаг и обновит статус при следующей проверке.
        - Для немедленного прерывания требуется интеграция с asyncio cancellation.
    """
    async with _validation_lock:
        task: Optional[Dict[str, Any]] = _validation_tasks.get(task_id)
    if not task:
        return {"status": "not_found", "message": "Task not found"}
    if task["status"] in ("completed", "failed", "cancelled"):
        return {"status": task["status"], "message": f"Already {task['status']}"}
    task["status"] = "cancelled"
    task["message"] = "Отменено пользователем"
    logger.info(f"Task {task_id} cancelled by user")
    return {"status": "cancelled"}
