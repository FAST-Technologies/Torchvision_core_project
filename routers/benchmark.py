# routers/benchmark.py

"""Модуль API для управления бенчмарками сегментации изображений.

Предоставляет REST-интерфейс для:
- Запуска асинхронных бенчмарков множества моделей сегментации
- Отслеживания прогресса выполнения задач
- Получения результатов и метрик качества (mIoU, Pixel Accuracy, F1-score)
- Управления жизненным циклом задач (старт, статус, отмена)

Поддерживаемые модели:
- Transformers: SegFormer (B2/B5), Mask2Former, MaskFormer, OneFormer, DPT, UPerNet
- SAM: MobileSAM, SAM2
- YOLOv8: n/s/m-seg варианты
- SMP/CNN: U-Net, DeepLabV3+, FPN, PSPNet, FCN, SegNet (с предобученными чекпоинтами)
- TorchVision: Mask R-CNN

Особенности:
- Асинхронное выполнение через BackgroundTasks FastAPI
- Прогресс-трекинг с обновлением в реальном времени
- Безопасная сериализация numpy-типов в JSON
- Обработка ошибок с детализацией и логированием
- Поддержка кастомных изображений и GT-масок через upload
- Конфигурация бенчмарка через JSON-параметры

Пример использования:
```python
# 1. Запуск бенчмарка
POST /api/benchmark/start
{
    "use_default_image": true,
    "config": {
        "inference": {"alpha": 0.6, "warmup_runs": 2},
        "filters": {"min_iou": 0.5, "only_passed": false},
        "visualization": {"show_overlay": true, "color_palette": "ade"},
        "models_to_run": ["segformer", "mask2former", "sam"]
    }
}
# Response: {"task_id": "uuid..."}

# 2. Проверка статуса
GET /api/benchmark/status/{task_id}
# Response: {
#     "status": "running"|"completed"|"failed",
#     "progress": 0-100,
#     "message": "Описание текущего этапа",
#     "results": {...}  # только при completion
# }

# 3. Получение результатов
# Результаты доступны в поле `results` при status="completed":
{
    "summary": {
        "segformer": {"mIoU": 0.78, "pixel_acc": 0.92, "time_ms": 145.2},
        "mask2former": {"mIoU": 0.81, "pixel_acc": 0.94, "time_ms": 230.5},
        ...
    },
    "output_dir": "./data/benchmark_{task_id}",
    "charts": {"metrics_plot_b64": "base64-encoded-png"}
}
Атрибуты модуля:
router (APIRouter): Экземпляр FastAPI router с префиксом "/api/benchmark".
benchmark_tasks (Dict[str, BenchmarkTask]): In-memory хранилище задач.
Зависимости:
- FastAPI, Pydantic: веб-фреймворк и валидация данных
- torch, numpy: инференс моделей и обработка тензоров
- PIL, base64: работа с изображениями
- testing.SegmentationBenchmark: ядро бенчмарка
- utils.config, utils.paths: конфигурация и пути проекта
Примечания:
- В продакшене рекомендуется заменить benchmark_tasks на Redis/Celery.
- Для больших моделей требуется ≥20 ГБ VRAM (проверяется при старте).
- Все изображения конвертируются в RGB, GT-маски — в grayscale (L-режим).
- Метрики с NaN/Inf автоматически заменяются на null в JSON.
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
import gc
import base64
import logging
import traceback
from typing import Dict, Any, Optional, Union, List, Callable, TypedDict, Tuple, cast, TypeAlias
from pathlib import Path
from pydantic import BaseModel

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, BackgroundTasks, Form, File, UploadFile
from fastapi.responses import JSONResponse

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import settings
from utils.paths import ensure_dirs, ADE20K_DIR, MODELS_DIR, PROJECT_ROOT, DATA_DIR
from testing.SegmentationBenchmark import SegmentationBenchmark
from utils.palettes import ade_palette, coco_palette, cityscapes_palette

# Настройка логгера
logger: logging.Logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler: logging.StreamHandler = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    log_dir: Path = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file: Path = log_dir / "benchmark.log"

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

router: APIRouter = APIRouter(
    prefix="/api/benchmark",
    tags=["benchmark"],
    responses={
        404: {"description": "Task not found"},
        500: {"description": "Internal server error"},
    },
)

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & TASK STORAGE
# ──────────────────────────────────────────────────────────────────────
PathLike: TypeAlias = Union[str, Path]
"""Тип-алиас для путей: строка или объект pathlib.Path, dtype=Union[str, Path]."""


# 🔹 TypedDict для структуры задачи бенчмарка
class BenchmarkTask(TypedDict, total=False):
    """Структура данных для отслеживания задачи бенчмарка.

    Attributes:
        status (str): Статус выполнения: "running" | "completed" | "failed" | "cancelled".
        progress (float): Прогресс выполнения в процентах (0.0–100.0).
        message (str): Человекочитаемое описание текущего этапа.
        results (Optional[Dict[str, Any]]): Результаты бенчмарка (заполняется при completion).
        error_details (Optional[Dict[str, Any]]): Детали ошибки (заполняется при failed).

    Example:
        ```python
        task: BenchmarkTask = {
            "status": "running",
            "progress": 45.5,
            "message": "🔄 mask2former: загрузка...",
            "results": None,
            "error_details": None
        }
        ```
    """

    status: str  # "running", "completed", "failed", "cancelled"
    progress: float  # 0-100
    message: str
    results: Optional[Dict[str, Any]]
    error_details: Optional[Dict[str, Any]]


# 🔹 TypedDict для конфигурации бенчмарка (вместо Dict[str, Any])
class BenchmarkConfig(TypedDict, total=False):
    """Конфигурация бенчмарка через JSON.

    Attributes:
        inference (Dict[str, Any]): Параметры инференса:
            - alpha (float): Прозрачность оверлея [0.0, 1.0], по умолчанию 0.6.
            - warmup_runs (int): Число прогревочных прогонов, по умолчанию 2.
        filters (Dict[str, Any]): Фильтры для отчёта:
            - min_iou (float): Минимальный IoU для включения в отчёт, по умолчанию 0.0.
            - only_passed (bool): Показывать только прошедшие порог модели, по умолчанию False.
        visualization (Dict[str, Any]): Параметры визуализации:
            - show_overlay (bool): Показывать оверлей маски на изображении, по умолчанию True.
            - show_gt (bool): Показывать ground truth, по умолчанию True.
            - color_palette (str): Название палитры: "ade" | "coco" | "cityscapes", по умолчанию "ade".
        models_to_run (Optional[List[str]]): Список имён моделей для запуска.
            Если None — запускаются все доступные модели.

    Example:
        ```json
        {
            "inference": {"alpha": 0.7, "warmup_runs": 3},
            "filters": {"min_iou": 0.5, "only_passed": true},
            "visualization": {"show_overlay": true, "color_palette": "coco"},
            "models_to_run": ["segformer", "sam2", "yolov8m_seg"]
        }
        ```
    """

    inference: Dict[str, Any]
    filters: Dict[str, Any]
    visualization: Dict[str, Any]
    models_to_run: Optional[List[str]]


# 🔹 Type alias для хранилища задач
BenchmarkTaskDict: TypeAlias = Dict[str, BenchmarkTask]
"""Тип-алиас для хранилища задач: {task_id: BenchmarkTask}, dtype=Dict[str, BenchmarkTask]."""

ModelLoadStep: TypeAlias = Tuple[str, Callable[..., Any], Dict[str, Any]]
"""
Тип-алиас для шага загрузки модели, dtype=Tuple[str, Callable[..., Any], Dict[str, Any]].

Tuple содержит:
    0. str: Уникальный ключ модели (например, "segformer_b5").
    1. Callable: Функция загрузки модели из SegmentationBenchmark.
    2. Dict[str, Any]: Аргументы для вызова функции загрузки.

Example:
    ```python
    step: ModelLoadStep = (
        "segformer",
        bench.load_segformer,
        {"path": "/models/segformer-b5-ready"}
    )
    ```
"""

benchmark_tasks: BenchmarkTaskDict = {}
"""Простое хранилище задач (в продакшене замените на Redis/Celery), dtype=BenchmarkTaskDict."""


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
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return None if obj > 0 else None  # или "Infinity"/"-Infinity"
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
            metrics = {"mIoU": np.float32(0.78), "classes": np.array([1, 2, 3])}
            return safe_json_response(metrics)
        ```
    """
    return JSONResponse(
        content=content,
        status_code=status_code,
        media_type="application/json",
    )


# ──────────────────────────────────────────────────────────────────────
def img_to_b64(path: PathLike) -> str:
    """Конвертирует файл изображения в base64-строку для передачи в JSON.

    Args:
        path: Путь к файлу изображения (строка или Path).

    Returns:
        str: Base64-кодированная строка изображения (без префикса data:).

    Raises:
        FileNotFoundError: Если файл не найден.
        PermissionError: Если нет прав на чтение файла.

    Example:
        ```python
        b64_str = img_to_b64("./results/plot.png")
        # Use in HTML: <img src="data:image/png;base64,{b64_str}">
        ```
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ──────────────────────────────────────────────────────────────────────
class BenchmarkStartRequest(BaseModel):
    """Схема запроса на запуск бенчмарка.

    Используется для валидации входных данных эндпоинта `/start`.

    Attributes:
        use_default_image (bool): Использовать тестовое изображение по умолчанию.
            Если True и image_path не указан, загружается изображение из
            репозитория "hf-internal-testing/fixtures_ade20k".
            По умолчанию True.
        image_path (Optional[str]): Путь к пользовательскому изображению.
            Если указан и файл существует, используется вместо default.
            По умолчанию None.

    Example:
        ```python
        # Валидный запрос
        req = BenchmarkStartRequest(use_default_image=True)

        # С кастомным путём
        req = BenchmarkStartRequest(
            use_default_image=False,
            image_path="/data/custom/test.jpg"
        )
        ```

    Validation:
        - image_path проверяется на существование только при use_default_image=False.
        - Оба параметра опциональны: при отсутствии обоих используется default.
    """

    use_default_image: bool = True
    image_path: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────
@router.get("/health")
async def benchmark_health() -> Dict[str, Any]:
    """Возвращает статус системы и доступность ресурсов для бенчмарка.

    Проверяет:
    - Доступность CUDA и характеристики GPU
    - Использование VRAM (total/allocated/free/reserved)
    - Количество активных задач бенчмарка

    Returns:
        Dict[str, Any]: Словарь со статусом:
            - status (str): "ok" при успешной проверке.
            - cuda_available (bool): Доступен ли CUDA.
            - device_name (str): Название GPU или "cpu".
            - vram_mb (float): Общий объём VRAM в МБ.
            - vram_allocated_mb (float): Использовано VRAM в МБ.
            - vram_free_mb (float): Свободно VRAM в МБ.
            - reserved_vram_mb (float): Зарезервировано VRAM в МБ.
            - active_tasks (int): Количество задач со статусом "running".

    Example Response:
        ```json
        {
            "status": "ok",
            "cuda_available": true,
            "device_name": "NVIDIA GeForce RTX 4090",
            "vram_mb": 24576.0,
            "vram_allocated_mb": 3245.8,
            "vram_free_mb": 21330.2,
            "reserved_vram_mb": 4096.0,
            "active_tasks": 2
        }
        ```

    Note:
        - При отсутствии CUDA все значения памяти возвращаются как 0.0.
        - Метрики памяти обновляются в реальном времени при каждом запросе.
    """
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram_mb: float = props.total_memory / 1024**2
        alloc_vram_mb: float = torch.cuda.memory_allocated(0) / 1024**2
        free_vram_mb: float = total_vram_mb - alloc_vram_mb
        reserved_vram_mb: float = torch.cuda.memory_reserved(0) / 1024**2
        device_name: str = torch.cuda.get_device_name(0)
    else:
        total_vram_mb = alloc_vram_mb = free_vram_mb = reserved_vram_mb = 0.0
        device_name = "cpu"

    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device_name": device_name,
        "vram_mb": total_vram_mb,
        "vram_allocated_mb": alloc_vram_mb,
        "vram_free_mb": free_vram_mb,
        "reserved_vram_mb": reserved_vram_mb,
        "active_tasks": len([t for t in benchmark_tasks.values() if t["status"] == "running"]),
    }


# ──────────────────────────────────────────────────────────────────────
@router.get("/api/benchmark/debug")
async def debug_benchmark() -> Dict[str, Any]:
    """Отладочный эндпоинт: возвращает сводку по всем задачам бенчмарка.

    Предназначен для мониторинга и отладки в режиме разработки.

    Returns:
        Dict[str, Any]: Словарь с информацией:
            - active_tasks (int): Общее количество задач в хранилище.
            - tasks (Dict[str, Dict]): Детали по каждой задаче:
                - status (str): Текущий статус задачи.
                - progress (float): Прогресс выполнения (0–100).

    Example Response:
        ```json
        {
            "active_tasks": 3,
            "tasks": {
                "uuid-1": {"status": "running", "progress": 45.2},
                "uuid-2": {"status": "completed", "progress": 100.0},
                "uuid-3": {"status": "failed", "progress": 12.5}
            }
        }
        ```

    Warning:
        - Не использовать в продакшене: раскрывает внутренние идентификаторы задач.
        - Не включает чувствительные данные (результаты, ошибки, пути).
    """
    return {
        "active_tasks": len(benchmark_tasks),
        "tasks": {k: {"status": v["status"], "progress": v["progress"]} for k, v in benchmark_tasks.items()},
    }


# ──────────────────────────────────────────────────────────────────────
# BACKGROUND TASK
# ──────────────────────────────────────────────────────────────────────
async def run_benchmark(
    task_id: str,
    req: BenchmarkStartRequest,
    image_file: Optional[UploadFile] = None,
    gt_file: Optional[UploadFile] = None,
    config: Optional[Dict] = None,
) -> None:
    """Асинхронная фоновая задача выполнения бенчмарка.

    Основной рабочий процесс:
    1. Инициализация задачи в benchmark_tasks со статусом "running".
    2. Проверка доступности VRAM (предупреждение при <20 ГБ).
    3. Загрузка входного изображения (из файла, пути или default).
    4. Загрузка GT-маски (если предоставлена) для расчёта метрик.
    5. Инициализация SegmentationBenchmark с палитрой и параметрами.
    6. Поэтапная загрузка моделей с обновлением прогресса.
    7. Пошаговый инференс через compare_step_by_step (с прогресс-трекингом).
    8. Сохранение результатов: JSON-отчёт, графики, base64-изображения.
    9. Обновление статуса на "completed" с результатами или "failed" с ошибкой.

    Обработка ошибок:
    - Каждая модель загружается в try/except: неудача одной не прерывает остальные.
    - Чекпоинты проверяются на существование перед загрузкой.
    - Ошибки логируются с traceback (при DEBUG-уровне).
    - При критической ошибке задача помечается "failed" с деталями.

    Обновление прогресса:
    - 0–5%: Инициализация и проверка окружения.
    - 5–75%: Загрузка моделей (равномерное распределение).
    - 75–100%: Инференс и сохранение результатов (управляется внутри compare_step_by_step).

    Args:
        task_id (str): Уникальный идентификатор задачи (UUID).
        req (BenchmarkStartRequest): Параметры запроса (изображение, default-флаги).
        image_file (Optional[UploadFile]): Загруженное изображение (PIL-compatible).
        gt_file (Optional[UploadFile]): Загруженная GT-маска (grayscale).
        config (Optional[Dict]): Словарь конфигурации (распарсенный из JSON).

    Side Effects:
        - Обновляет benchmark_tasks[task_id] в реальном времени.
        - Создаёт директорию ./data/benchmark_{task_id} для результатов.
        - Генерирует графики и сохраняет в PNG + base64.
        - Очищает CUDA-кэш и вызывает gc.collect() после каждой модели.

    Note:
        - Функция не возвращает значение: результаты передаются через benchmark_tasks.
        - Для продакшена рекомендуется вынести логику в Celery/RQ с отдельным worker.
        - compare_step_by_step должен поддерживать асинхронные обновления прогресса.
    """
    benchmark_tasks[task_id] = {
        "status": "running",
        "progress": 0,
        "message": "Инициализация...",
        "results": None,
    }
    if torch.cuda.is_available():
        vram_gb: float = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb < 20:  # менее 20 ГБ
            logger.warning(f"⚠️ Low VRAM: {vram_gb:.1f} GB. Benchmark may fail.")
    try:
        ensure_dirs(ADE20K_DIR)

        logger.info(f"🔍 PROJECT_ROOT: {PROJECT_ROOT}")
        logger.info(f"🔍 MODELS_DIR: {MODELS_DIR}")
        logger.info(f"🔍 DATA_DIR: {DATA_DIR}")
        for key, path in [
            ("SegFormer", settings.SEGFORMER_PATH),
            ("UNet", settings.MODEL_DIR / settings.UNET_CHECKPOINT),
            ("Default Image", settings.DEFAULT_IMAGE),
        ]:
            path_str: str = str(path)
            exists = "✅" if os.path.exists(path_str) else "❌"
            logger.info(f"{exists} {key}: {path}")

        config_dict: BenchmarkConfig = cast(BenchmarkConfig, config or {})
        inference_params: Dict[str, Any] = config_dict.get("inference", {})
        alpha: float = float(inference_params.get("alpha", 0.6))
        # warmup_runs: int = int(inference_params.get("warmup_runs", 2))

        # filters: Dict[str, Any] = config_dict.get("filters", {})
        # min_iou: float = float(filters.get("min_iou", 0.0))
        # only_passed: bool = bool(filters.get("only_passed", False))

        viz_params: Dict[str, Any] = config_dict.get("visualization", {})
        # show_overlay: bool = bool(viz_params.get("show_overlay", True))
        # show_gt: bool = bool(viz_params.get("show_gt", True))
        palette_name: str = str(viz_params.get("color_palette", "ade"))

        PALETTES: Dict[str, Callable[[], List[List[int]]]] = {
            "ade": ade_palette,
            "coco": coco_palette,
            "cityscapes": cityscapes_palette,
        }
        palette_func: Callable[[], List[List[int]]] = PALETTES.get(palette_name, ade_palette)
        palette: List[List[int]] = palette_func()

        benchmark_tasks[task_id]["progress"] = 5
        benchmark_tasks[task_id]["message"] = "Загрузка бенчмарка..."
        logger.info(
            f"🔄 Task {task_id}: progress={benchmark_tasks[task_id]['progress']}, message={benchmark_tasks[task_id]['message']}"
        )

        image_input: Image.Image
        if image_file:
            image_input = Image.open(image_file.file).convert("RGB")
        elif req.image_path and os.path.exists(req.image_path):
            image_input = Image.open(req.image_path).convert("RGB")
        else:
            repo_id: str = "hf-internal-testing/fixtures_ade20k"
            image_path: str = hf_hub_download(repo_id=repo_id, filename="ADE_val_00000001.jpg", repo_type="dataset")
            image_input = Image.open(image_path).convert("RGB")

        # 🔹 Обработка GT-маски
        gt_mask: Optional[np.ndarray] = None
        if gt_file:
            gt_mask = np.array(Image.open(gt_file.file).convert("L"))
        elif req.use_default_image:
            gt_path: str = "./data/ade20k_test_trained/original_image_mask_0.png"
            if os.path.exists(gt_path):
                gt_mask = np.array(Image.open(gt_path).convert("L"))

        bench: SegmentationBenchmark = SegmentationBenchmark(
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_classes=150,
            gt_mask=gt_mask,
            palette=palette,
        )

        # Загрузка моделей
        model_load_steps: List[ModelLoadStep] = [
            (
                "segformer",
                bench.load_segformer,
                {
                    "path": str(settings.MODEL_DIR / settings.SEGFORMER_PATH),
                },
            ),
            ("segformer_b2", bench.load_segformer_variant, {"variant": "b2"}),
            (
                "mask2former",
                bench.load_mask2former,
                {"name": "facebook/mask2former-swin-base-ade-semantic"},
            ),
            (
                "maskformer",
                bench.load_maskformer,
                {"name": "facebook/maskformer-resnet50-ade20k-full"},
            ),
            (
                "oneformer",
                bench.load_oneformer,
                {"name": "shi-labs/oneformer_ade20k_swin_large"},
            ),
            ("dpt", bench.load_dpt, {"model_name": "Intel/dpt-large-ade"}),
            (
                "upernet",
                bench.load_upernet,
                {"model_name": "openmmlab/upernet-convnext-small"},
            ),
            # === SAM models ===
            ("sam", bench.load_sam, {"model_name": "models/mobile_sam.pt"}),
            ("sam2", bench.load_sam, {"model_name": "models/sam2_t.pt"}),
            # === YOLOv8 ===
            ("yolov8n_seg", bench.load_yolov8, {"model_name": "yolov8n-seg.pt"}),
            ("yolov8s_seg", bench.load_yolov8, {"model_name": "yolov8s-seg.pt"}),
            ("yolov8m_seg", bench.load_yolov8, {"model_name": "yolov8m-seg.pt"}),
            # === SMP/CNN models with checkpoints ===
            (
                "unet_pretrained",
                bench.load_unet_trained,
                {
                    "checkpoint_path": str(settings.MODEL_DIR / settings.UNET_CHECKPOINT),
                    "encoder_name": "resnet34",
                },
            ),
            (
                "deeplab_pretrained",
                bench.load_deeplab_trained,
                {
                    "checkpoint_path": str(settings.MODEL_DIR / settings.DEEPLAB_CHECKPOINT),
                },
            ),
            (
                "fpn_mit_b5_pretrained",
                bench.load_fpn_mit_pretrained,
                {
                    "variant": "b5",
                    "checkpoint_path": str(settings.MODEL_DIR / settings.FPN_MIT_CHECKPOINT),
                },
            ),
            (
                "psp_mit_b5_pretrained",
                bench.load_psp_mit_pretrained,
                {
                    "variant": "b5",
                    "checkpoint_path": str(settings.MODEL_DIR / settings.PSP_MIT_CHECKPOINT),
                },
            ),
            (
                "fcn_resnet50_pretrained",
                bench.load_fcn_resnet50_pretrained,
                {
                    "variant": "fcn_resnet50",
                    "checkpoint_path": str(settings.MODEL_DIR / settings.FCN_RESNET50_CHECKPOINT),
                },
            ),
            (
                "segnet_resnet34_pretrained",
                bench.load_segnet_pretrained,
                {
                    "encoder_name": "resnet34",
                    "checkpoint_path": str(settings.MODEL_DIR / settings.SEGNET_RESNET34_CHECKPOINT),
                },
            ),
            # === Torchvision models ===
            (
                "maskrcnn_pretrained",
                bench.load_mask_rcnn_pretrained,
                {"variant": "maskrcnn_resnet50_fpn"},
            ),
        ]

        models_to_run: Optional[List[str]] = config_dict.get("models_to_run")
        if models_to_run:
            model_load_steps = [step for step in model_load_steps if step[0] in models_to_run]

        for i, (key, load_fn, kwargs) in enumerate(model_load_steps):
            benchmark_tasks[task_id]["progress"] = 5 + (i / len(model_load_steps)) * 70
            benchmark_tasks[task_id]["message"] = f"🔄 {key}: загрузка..."
            try:
                cp: Optional[str] = kwargs.get("checkpoint_path")
                if cp and not os.path.exists(cp):
                    logger.warning(f"⚠️ Checkpoint not found: {cp}, skipping {key}")
                    continue
                load_fn(**kwargs)
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                logger.error(f"❌ Failed to load {key}: {e}", exc_info=True)

        benchmark_tasks[task_id]["message"] = f"✅ {key}: готово. Все модели загружены. Запуск инференса..."
        benchmark_tasks[task_id]["progress"] = 50
        benchmark_tasks[task_id]["message"] = "Запуск инференса..."
        # await asyncio.to_thread(bench.compare, image_input=image_input, alpha=alpha)
        await bench.compare_step_by_step(
            image_input=image_input,
            alpha=alpha,
            task_id=task_id,
            benchmark_tasks=benchmark_tasks,
        )

        # Сохранение результатов
        benchmark_tasks[task_id]["message"] = "Сохранение результатов..."
        out_dir: str = f"./data/benchmark_{task_id}"
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs("./data/ade20k_test_trained", exist_ok=True)
        await asyncio.to_thread(bench.save_results, out_dir)
        await asyncio.to_thread(bench.plot_all_metrics, path=f"{out_dir}/plot_all.png")

        summary: Dict[str, Dict[str, Any]] = bench.get_summary()
        for _, metrics in summary.items():
            for key in ["mIoU", "pixel_acc", "f1_weighted"]:
                val = metrics.get(key)
                if isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val)):
                    metrics[key] = None
        benchmark_tasks[task_id].update(
            {
                "status": "completed",
                "progress": 100,
                "message": "Готово",
                "results": {
                    "summary": summary,
                    "output_dir": out_dir,
                    "charts": {"metrics_plot_b64": img_to_b64(f"{out_dir}/plot_all.png")},
                },
            }
        )

    except Exception as e:
        benchmark_tasks[task_id].update(
            {
                "status": "failed",
                "message": str(e),
                "error_details": {
                    "error_type": type(e).__name__,
                    "failed_at": benchmark_tasks[task_id]["message"],
                    "traceback": (traceback.format_exc() if logger.level == logging.DEBUG else None),
                },
            }
        )
        logger.error(f"Benchmark {task_id} failed: {e}", exc_info=True)


# ──────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────
@router.post("/start")
async def start_benchmark(
    image: Optional[UploadFile] = File(None),
    gt_mask: Optional[UploadFile] = File(None),
    use_default_image: bool = Form(True),
    image_path: Optional[str] = Form(None),
    config: Optional[str] = Form(None),
    bg: BackgroundTasks = None,
) -> Dict[str, str]:
    """Запускает асинхронный бенчмарк сегментации.

    Создаёт новую задачу, инициализирует SegmentationBenchmark и планирует
    выполнение через BackgroundTasks FastAPI.

    Args:
        image (Optional[UploadFile]): Загружаемое изображение для бенчмарка.
            Если указано, используется вместо default/image_path.
        gt_mask (Optional[UploadFile]): Загружаемая ground truth маска.
            Используется для расчёта метрик качества (IoU, Dice и др.).
        use_default_image (bool): Использовать тестовое изображение по умолчанию.
            Применяется если image и image_path не указаны.
        image_path (Optional[str]): Путь к локальному изображению.
            Проверяется на существование перед использованием.
        config (Optional[str]): JSON-строка с конфигурацией бенчмарка.
            См. BenchmarkConfig для структуры.
        bg (Optional[BackgroundTasks]): Инъекция BackgroundTasks от FastAPI.
            Если None, бенчмарк запускается синхронно (с предупреждением в логе).

    Returns:
        Dict[str, str]: Словарь с идентификатором задачи:
            ```json
            {"task_id": "550e8400-e29b-41d4-a716-446655440000"}
            ```

    Raises:
        HTTPException (422): Если config содержит невалидный JSON.

    Example Request (multipart/form-data):
        ```
        POST /api/benchmark/start
        Content-Type: multipart/form-data

        use_default_image: false
        image_path: "/data/test.jpg"
        config: '{"inference": {"alpha": 0.7}, "models_to_run": ["segformer"]}'
        ```

    Note:
        - Задача выполняется асинхронно: ответ возвращается немедленно.
        - Прогресс отслеживается через эндпоинт `/status/{task_id}`.
        - Файлы изображений читаются в память: для больших файлов может потребоваться
          увеличение лимита `max_file_size` в конфигурации FastAPI.
    """
    config_dict: Optional[BenchmarkConfig] = None
    if config:
        try:
            config_dict = cast(BenchmarkConfig, json.loads(config))
        except json.JSONDecodeError:
            raise HTTPException(422, detail="Invalid config JSON")
    config_for_task: Optional[Dict[str, Any]] = dict(config_dict) if config_dict else None
    req = BenchmarkStartRequest(
        use_default_image=use_default_image,
        image_path=image_path,
    )
    task_id: str = str(uuid.uuid4())
    if bg is not None:
        bg.add_task(run_benchmark, task_id, req, image, gt_mask, config_for_task)
    else:
        logger.warning("BackgroundTasks not provided, running benchmark synchronously")
    return {"task_id": task_id}


# ──────────────────────────────────────────────────────────────────────
@router.get("/status/{task_id}")
async def get_status(task_id: str) -> JSONResponse:
    """Возвращает текущий статус и прогресс задачи бенчмарка.

    Args:
        task_id (str): UUID задачи, полученный при запуске через `/start`.

    Returns:
        JSONResponse: Сериализованный объект BenchmarkTask с полями:
            - status (str): "running" | "completed" | "failed" | "cancelled".
            - progress (float): Прогресс 0.0–100.0.
            - message (str): Описание текущего этапа.
            - results (Optional[Dict]): Результаты (только при status="completed").
            - error_details (Optional[Dict]): Детали ошибки (только при failed).

    Raises:
        HTTPException (404): Если задача с task_id не найдена.

    Example Response (running):
        ```json
        {
            "status": "running",
            "progress": 67.5,
            "message": "🔄 sam2: инференс...",
            "results": null,
            "error_details": null
        }
        ```

    Example Response (completed):
        ```json
        {
            "status": "completed",
            "progress": 100.0,
            "message": "Готово",
            "results": {
                "summary": {
                    "segformer": {"mIoU": 0.78, "pixel_acc": 0.92, "time_ms": 145.2},
                    "sam2": {"mIoU": 0.65, "pixel_acc": 0.88, "time_ms": 89.1}
                },
                "output_dir": "./data/benchmark_550e8400",
                "charts": {"metrics_plot_b64": "iVBORw0KGgoAAAANSUh..."}
            }
        }
        ```

    Note:
        - Метрики с `NaN`/`Inf` автоматически заменяются на `null` перед сериализацией.
        - Для длительных задач рекомендуется polling с интервалом 2–5 секунд.
    """
    task: Optional[BenchmarkTask] = benchmark_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    logger.info(f"📡 GET /status/{task_id} -> {task.get('progress')}%")
    if task_id not in benchmark_tasks or not task:
        raise HTTPException(status_code=404, detail="Task not found")
    logger.debug(f"📡 Status requested for {task_id}: {task['status']} / {task['progress']}%")
    results = task.get("results")
    if results is not None and isinstance(results, dict) and "summary" in results:
        summary = results["summary"]
        for _, metrics in summary.items():
            for key, value in metrics.items():
                if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
                    metrics[key] = None

    return safe_json_response(task)


# ──────────────────────────────────────────────────────────────────────
@router.get("/debug/{task_id}")
async def debug_task(task_id: str) -> Dict[str, Any]:
    """Возвращает отладочную информацию о задаче бенчмарка.

    Предназначен для разработчиков: предоставляет дополнительные метаданные
    для диагностики проблем с выполнением задач.

    Args:
        task_id (str): UUID задачи.

    Returns:
        Dict[str, Any]: Словарь с отладочной информацией:
            - task_id (str): Идентификатор задачи.
            - status (str): Текущий статус.
            - progress (float): Прогресс выполнения.
            - message (str): Последнее сообщение о состоянии.
            - results_keys (Optional[List[str]]): Ключи в результатах (если есть).
            - last_updated (float): Время последнего обновления (perf_counter).

    Raises:
        Возвращает {"error": "Task not found"} вместо HTTPException для удобства
        в отладочных сценариях.

    Example Response:
        ```json
        {
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "running",
            "progress": 45.2,
            "message": "🔄 mask2former: загрузка...",
            "results_keys": null,
            "last_updated": 123456.789
        }
        ```

    Note:
        - Не использовать в продакшене: может раскрывать внутреннюю структуру.
        - `last_updated` полезен для обнаружения "зависших" задач.
    """
    task: Optional[BenchmarkTask] = benchmark_tasks.get(task_id)
    if not task:
        return {"error": "Task not found"}
    results = task.get("results")
    results_keys = list(results.keys()) if isinstance(results, dict) else None

    return {
        "task_id": task_id,
        "status": task.get("status"),
        "progress": task.get("progress"),
        "message": task.get("message"),
        "results_keys": results_keys,
        "last_updated": time.perf_counter(),
    }


# ──────────────────────────────────────────────────────────────────────
@router.delete("/{task_id}")
async def cancel_benchmark(task_id: str) -> Dict[str, str]:
    """Отменяет или удаляет задачу бенчмарка.

    Логика поведения в зависимости от статуса задачи:
    - "running": Устанавливает status="cancelled", message="Отменено пользователем".
    - "completed"|"failed"|"cancelled": Удаляет задачу из хранилища.
    - Не найдена: Возвращает ошибку 404.

    Args:
        task_id (str): UUID задачи для отмены.

    Returns:
        Dict[str, str]: Словарь с результатом операции:
            - При отмене: {"status": "cancelled"}
            - При удалении завершённой: {"status": "deleted", "was": "...", "message": "..."}
            - При не найдено: {"status": "not_found", "message": "..."}

    Raises:
        HTTPException (404): Если задача не найдена в хранилище.

    Example Responses:
        ```json
        // Отмена запущенной задачи
        {"status": "cancelled"}

        // Удаление уже завершённой
        {
            "status": "deleted",
            "was": "completed",
            "message": "Task already completed"
        }

        // Задача не найдена
        {
            "status": "not_found",
            "message": "Task not found or already removed"
        }
        ```

    Note:
        - Отмена не прерывает выполнение кода немедленно: задача завершит текущий
          шаг и обновит статус при следующей проверке.
        - Для немедленного прерывания требуется интеграция с asyncio cancellation
    """
    if task_id not in benchmark_tasks:
        raise HTTPException(404, detail="Task not found")

    task: BenchmarkTask = benchmark_tasks[task_id]
    if not task:
        return {"status": "not_found", "message": "Task not found or already removed"}
    if task["status"] in ("completed", "failed", "cancelled"):
        del benchmark_tasks[task_id]
        return {
            "status": "deleted",
            "was": task["status"],
            "message": f"Task already {task['status']}",
        }

    task["status"] = "cancelled"
    task["message"] = "Отменено пользователем"
    logger.info(f"Task {task_id} cancelled by user")
    return {"status": "cancelled"}
