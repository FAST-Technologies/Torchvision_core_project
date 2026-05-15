# routers/comparator.py

"""Модуль API для сравнения методов сегментации изображений.

Предоставляет REST-интерфейс для:
- Асинхронного пакетного сравнения методов из разных библиотек (OpenCV, scikit-learn, PyTorch)
- Отслеживания прогресса выполнения задач сравнения
- Получения метрик согласованности (IoU, Dice, F1, Pixel Accuracy) относительно референсного метода
- Визуализации результатов сравнения (оверлеи, матрицы метрик, графики)

Поддерживаемые библиотеки:
- OpenCV: cv2.threshold, cv2.Canny, cv2.Sobel, cv2.adaptiveThreshold и др.
- scikit-learn: filters.threshold_*, feature_detection, morphology и др.
- PyTorch: кастомные реализации через TorchSegmenter/TorchSegmenter2

Особенности:
- Асинхронное выполнение через asyncio.create_task()
- Прогресс-трекинг с обновлением в реальном времени через shared dict
- Потокобезопасное обновление статуса через asyncio.Lock
- Безопасная сериализация numpy-типов в JSON через NumpyEncoder
- Поддержка кастомных параметров для каждого метода
- Генерация визуализаций: сравнение масок, матрицы метрик, сводные графики

Пример использования:
```python
# 1. Запуск сравнения
POST /api/comparator/start
Content-Type: multipart/form-data

image: (файл изображения)
methods: '[
    {"name": "otsu_cv2", "library": "opencv", "method": "otsu_thresholding"},
    {"name": "otsu_sk", "library": "sklearn", "method": "otsu_thresholding"},
    {"name": "otsu_torch", "library": "torch", "method": "otsu_thresholding"}
]'
reference: '{"name": "otsu_cv2", "library": "opencv", "method": "otsu_thresholding"}'
comparison_type: "batch"

# Response: {"task_id": "uuid..."}

# 2. Проверка статуса
GET /api/comparator/status/{task_id}
# Response: {
#     "status": "running"|"completed"|"failed",
#     "progress": 0-100,
#     "message": "Сравнение otsu_sk (2/3)",
#     "results": {...}  # только при completion
# }

# 3. Результаты включают:
{
    "summary": {
        "methods_count": 3,
        "successful": 3,
        "avg_f1": 0.94,
        "top_by_f1": [...]
    },
    "results": [
        {
            "method": "otsu_sk",
            "library": "sklearn",
            "iou": 0.98,
            "dice": 0.99,
            "f1_score": 0.99,
            "test_time": 0.045,
            "ref_time": 0.042
        },
        ...
    ],
    "charts": {
        "comparison_summary.jpg": "base64-encoded-png",
        "f1_score_matrix.png": "base64-encoded-png"
    }
}
Атрибуты модуля:
router (APIRouter): Экземпляр FastAPI router с префиксом "/api/comparator".
_comparator_tasks (ComparatorTaskDict): In-memory хранилище задач сравнения.
_comparator_lock (asyncio.Lock): Асинхронный лок для потокобезопасного обновления задач.
Зависимости:
- FastAPI, Pydantic: веб-фреймворк и валидация данных
- numpy, pandas: обработка массивов и агрегация результатов
- PIL, base64: работа с изображениями и кодирование
- testing.SegmentationComparator: ядро логики сравнения
- segmenters.*: фабрики сегментеров для разных библиотек
Примечания:
- В продакшене рекомендуется заменить _comparator_tasks на Redis/Celery.
- Все изображения конвертируются в RGB перед обработкой.
- Метрики с NaN/Inf автоматически заменяются на null в JSON-ответах.
- Прогресс обновляется каждые ~1 шаг метода для плавного отображения.
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
import base64
import logging
import traceback
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from utils.paths import ensure_dirs, DATA_DIR
from testing.SegmentationComparator import SegmentationComparator
from segmenters.OpenCVSegmenter import OpenCVSegmenter
from segmenters.SklearnSegmenter import SklearnSegmenter
from segmenters.TorchSegmenter import TorchSegmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comparator")

router: APIRouter = APIRouter(
    prefix="/api/comparator",
    tags=["comparator"],
    responses={
        404: {"description": "Task not found"},
        500: {"description": "Internal server error"},
    },
)

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & TASK STORAGE
# ──────────────────────────────────────────────────────────────────────
ComparatorTaskDict = Dict[str, Dict[str, Any]]
"""Тип-алиас для хранилища задач компаратора.
Структура задачи:
{
    "status": str,              # "running" | "completed" | "failed" | "cancelled"
    "progress": float,          # 0.0–100.0
    "message": str,             # Человекочитаемое описание этапа
    "results": Optional[Dict],  # Результаты при completion
    "error": Optional[str],     # Сообщение об ошибке при failed
}
Example:
python task: ComparatorTaskDict = { "status": "running", "progress": 45.5, "message": "Сравнение otsu_sk (2/3)", "results": None, "error": None }
"""

_comparator_tasks: ComparatorTaskDict = {}
"""Глобальное хранилище задач компаратора в памяти.
Ключ: UUID задачи (str).
Значение: Словарь со статусом, прогрессом и результатами.
Warning:
В продакшене следует заменить на Redis/Celery для:
- Масштабируемости между воркерами
- Сохранения состояния при перезапуске
- Очистки устаревших задач по TTL
"""

_comparator_lock = asyncio.Lock()
"""Асинхронный лок для потокобезопасного обновления _comparator_tasks.
Используется во всех операциях чтения/записи задач для предотвращения
состояний гонки при параллельных запросах к одному task_id.
Example:
python async with _comparator_lock: _comparator_tasks[task_id]["progress"] = 50.0
"""


# ──────────────────────────────────────────────────────────────────────
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
def img_to_b64(path: str) -> str:
    """Конвертирует файл изображения в base64-строку для передачи в JSON.

    Args:
        path: Путь к файлу изображения.

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
DEFAULT_COMPARATOR_METHODS: Dict[str, List[str]] = {
    "opencv": [
        "global_thresholding",
        "otsu_thresholding",
        "adaptive_thresholding",
        "canny_edge",
        "sobel_edge",
        "threshold_sauvola",
    ],
    "sklearn": [
        "global_thresholding",
        "otsu_thresholding",
        "adaptive_thresholding",
        "canny_edge",
        "sobel_edge",
        "threshold_sauvola",
    ],
    "torch": [
        "global_thresholding",
        "otsu_thresholding",
        "adaptive_thresholding",
        "canny_edge",
        "sobel_edge",
        "threshold_sauvola",
    ],
}
"""Словарь методов по умолчанию для каждой библиотеки.

Используется как справочник при формировании запросов без явного указания методов.

Ключи: названия библиотек ("opencv", "sklearn", "torch").

Значения: списки имён методов, доступных в данной библиотеке.

Note:
- Список можно расширять по мере добавления новых методов в сегментеры.
- Методы должны иметь совместимые сигнатуры для корректного сравнения.
"""


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────
def _extract_library_from_name(name: str) -> str:
    """Извлекает название библиотеки из имени метода.

    Поддерживает суффиксы: _OpenCV, _Sklearn, _Torch.

    Args:
        name: Имя метода, например "otsu_thresholding_OpenCV".

    Returns:
        str: Название библиотеки в нижнем регистре: "opencv", "sklearn", "torch".

    Example:
        ```python
        _extract_library_from_name("canny_edge_Sklearn")  # returns "sklearn"
        _extract_library_from_name("otsu_OpenCV")  # returns "opencv"
        ```
    """
    if name.endswith("_OpenCV"):
        return "opencv"
    elif name.endswith("_Sklearn"):
        return "sklearn"
    elif name.endswith("_Torch"):
        return "torch"
    return "opencv"


# ──────────────────────────────────────────────────────────────────────
def _create_segmenter(library: str, method: str, params: Dict[str, Any]):
    """Фабричный метод для создания экземпляра сегментера.

    Динамически импортирует и инициализирует класс сегментера на основе названия библиотеки.

    Args:
        library: Название библиотеки: "opencv", "sklearn", "torch".
        method: Название метода сегментации (например, "otsu_thresholding").
        params: Словарь параметров для инициализации сегментера.

    Returns:
        BaseSegmenter: Экземпляр соответствующего класса сегментера.

    Raises:
        ValueError: Если библиотека не распознана.
        ImportError: Если модуль библиотеки не установлен.

    Example:
        ```python
        seg = _create_segmenter("opencv", "canny_edge", {"low": 0.1, "high": 0.3})
        mask = seg.segment(image_array)
        ```
    """
    if library == "opencv":
        return OpenCVSegmenter(method, **params)
    elif library == "sklearn":
        return SklearnSegmenter(method, **params)
    elif library == "torch":
        return TorchSegmenter(method, **params)
    raise ValueError(f"Unknown library: {library}")


# ──────────────────────────────────────────────────────────────────────
# BACKGROUND TASK
# ──────────────────────────────────────────────────────────────────────
async def _run_comparator_task(
    task_id: str,
    image: np.ndarray,
    methods_config: List[Dict[str, Any]],
    reference_config: Dict[str, Any],
    comparison_type: str = "batch",
    output_dir: Optional[str] = None,
) -> None:
    """Асинхронная фоновая задача выполнения сравнения методов.

    Основной рабочий процесс:
    1. Инициализация задачи в _comparator_tasks со статусом "running".
    2. Создание экземпляра SegmentationComparator.
    3. Подготовка сегментеров из конфигурации (с обработкой ошибок).
    4. Поэтапное сравнение каждого метода с референсным:
    - Запуск сегментации тестового и референсного методов.
    - Вычисление метрик согласованности (IoU, Dice, F1, Accuracy).
    - Генерация визуализаций для первых 3 методов.
    5. Агрегация результатов и сохранение в файлы (CSV, PNG, JSON).
    6. Обновление статуса на "completed" с результатами или "failed" с ошибкой.

    Обновление прогресса:
    - 0–20%: Инициализация и подготовка методов.
    - 20–90%: Пошаговое сравнение методов (равномерное распределение).
    - 90–100%: Сохранение результатов и финализация.

    Обработка ошибок:
    - Каждый метод обрабатывается в try/except: неудача одного не прерывает остальные.
    - Ошибки логируются с уровнем WARNING/ERROR.
    - При критической ошибке задача помечается "failed" с деталями.

    Args:
        task_id (str): Уникальный идентификатор задачи (UUID).
        image (np.ndarray): Входное изображение для сегментации (RGB или grayscale).
        methods_config (List[Dict[str, Any]]): Конфигурация сравниваемых методов:
            - name (str): Уникальное имя метода в отчёте.
            - library (str): Библиотека реализации.
            - method (str): Название метода в библиотеке.
            - params (Dict, optional): Параметры инициализации.
        reference_config (Dict[str, Any]): Конфигурация референсного метода:
            - name, library, method, params (аналогично methods_config).
        comparison_type (str): Тип сравнения: "batch" (по умолчанию) или "pairwise".
        output_dir (Optional[str]): Директория для сохранения результатов.
            Если None, создаётся "./data/comparator_{task_id}".

    Side Effects:
        - Обновляет _comparator_tasks[task_id] в реальном времени.
        - Создаёт директорию output_dir с результатами:
            - comparison_summary.jpg: Сводный график метрик.
            - f1_score_matrix.png: Тепловая карта F1-score.
            - accuracy_matrix.png: Тепловая карта accuracy.
            - viz_{method}.png: Визуализации для первых 3 методов.
        - Генерирует base64-кодированные изображения для JSON-ответа.

    Note:
        - Функция не возвращает значение: результаты передаются через _comparator_tasks.
        - Для продакшена рекомендуется вынести логику в Celery/RQ с отдельным worker.
        - compare_step_by_step должен поддерживать асинхронные обновления прогресса.
    """
    async with _comparator_lock:
        _comparator_tasks[task_id] = {
            "status": "running",
            "progress": 0,
            "message": "Инициализация...",
            "results": None,
        }

    try:
        comparator = SegmentationComparator()
        output_dir = output_dir or f"./data/comparator_{task_id}"
        os.makedirs(output_dir, exist_ok=True)

        # Подготовка методов
        segmenters: List[Dict[str, Any]] = []
        for cfg in methods_config:
            try:
                library: str = cfg.get("library") or _extract_library_from_name(cfg["name"])
                method_name: str = cfg["method"]
                params: Dict[str, Any] = cfg.get("params") or {}

                seg = _create_segmenter(library, method_name, params)
                segmenters.append(
                    {
                        "name": cfg["name"],
                        "segmenter": seg,
                        "library": library,
                    }
                )
            except Exception as e:
                logger.warning(f"⚠️ Skip {cfg.get('name', 'unknown')}: {e}")

        ref_seg = _create_segmenter(
            reference_config["library"],
            reference_config["method"],
            reference_config.get("params", {}),
        )
        ref_name: str = reference_config["name"]

        # 🔹 Прогресс: 0-20% подготовка, 20-90% сравнение, 90-100% сохранение
        async with _comparator_lock:
            _comparator_tasks[task_id]["progress"] = 20
            _comparator_tasks[task_id]["message"] = f"Запущено {len(segmenters)} методов"

        # 🔹 Пакетное сравнение с пошаговым обновлением
        results: List[Dict[str, Any]] = []
        for i, cfg in enumerate(segmenters):
            progress: float = 20 + (i / len(segmenters)) * 70
            async with _comparator_lock:
                _comparator_tasks[task_id]["progress"] = progress
                _comparator_tasks[task_id]["message"] = f"Сравнение {cfg['name']} ({i + 1}/{len(segmenters)})"
            await asyncio.sleep(0)

            try:
                t0: float = time.perf_counter()
                test_mask: np.ndarray = cfg["segmenter"].segment(image)
                test_time: float = time.perf_counter() - t0

                ref_mask: np.ndarray = ref_seg.segment(image)
                ref_time: float = time.perf_counter() - t0 - test_time

                metrics: Dict[str, float] = comparator.compute_metrics(ref_mask, test_mask, ref_name, cfg["name"])

                results.append(
                    {
                        "method": cfg["name"],
                        "library": cfg["library"],
                        **metrics,
                        "test_time": test_time,
                        "ref_time": ref_time,
                    }
                )

                if i < 3:
                    out_path: str = os.path.join(output_dir, f"viz_{cfg['name']}.png")
                    comparator.visualize_comparison(
                        image,
                        ref_mask,
                        test_mask,
                        {"method": ref_name, "execution_time": ref_time},
                        {"method": cfg["name"], "execution_time": test_time},
                        metrics,
                        output_path=out_path,
                    )

            except Exception as e:
                logger.error(f"❌ Error comparing {cfg['name']}: {e}")
                results.append({"method": cfg["name"], "error": str(e)})

        # 🔹 Финализация
        async with _comparator_lock:
            _comparator_tasks[task_id]["progress"] = 90
            _comparator_tasks[task_id]["message"] = "Сохранение результатов..."

        # Сохранение
        df: pd.DataFrame = comparator.batch_comparison(
            image=image,
            methods_config=[
                {
                    "name": r["method"],
                    "segmenter": _create_segmenter(
                        next(
                            (c["library"] for c in methods_config if c["name"] == r["method"]),
                            "opencv",
                        ),
                        next(
                            (c["method"] for c in methods_config if c["name"] == r["method"]),
                            "otsu_thresholding",
                        ),
                        {},
                    ),
                }
                for r in results
                if "error" not in r
            ],
            reference_segmenter=ref_seg,
            reference_name=ref_name,
            save_results=True,
            output_dir=output_dir,
        )

        # 🔹 Подготовка ответа
        summary: Dict[str, Any] = {
            "methods_count": len(results),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "top_by_f1": sorted(
                [r for r in results if "f1_score" in r],
                key=lambda x: x["f1_score"],
                reverse=True,
            )[:5],
            "avg_f1": (
                np.mean([r["f1_score"] for r in results if "f1_score" in r])
                if any("f1_score" in r for r in results)
                else None
            ),
            "df": df,
        }

        # 🔹 Сериализация графиков
        charts: Dict = {}
        for fname in [
            "comparison_summary.jpg",
            "f1_score_matrix.png",
            "accuracy_matrix.png",
        ]:
            fpath: str = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                charts[fname] = img_to_b64(fpath)

        async with _comparator_lock:
            _comparator_tasks[task_id]["status"] = "completed"
            _comparator_tasks[task_id]["progress"] = 100
            _comparator_tasks[task_id]["message"] = "Готово"
            _comparator_tasks[task_id]["results"] = {
                "summary": summary,
                "results": results,
                "output_dir": output_dir,
                "charts": charts,
            }

    except Exception as e:
        logger.error(f"Comparator task {task_id} failed: {e}", exc_info=True)
        async with _comparator_lock:
            _comparator_tasks[task_id]["status"] = "failed"
            _comparator_tasks[task_id]["message"] = str(e)
            _comparator_tasks[task_id]["error_details"] = {
                "error_type": type(e).__name__,
                "failed_at": _comparator_tasks[task_id]["message"],
                "traceback": (traceback.format_exc() if logger.level == logging.DEBUG else None),
            }


# ──────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────
@router.post("/start")
async def start_comparator(
    image: UploadFile = File(...),
    methods: str = Form(...),  # JSON string
    reference: str = Form(...),  # JSON string
    comparison_type: str = Form("batch"),
) -> Dict[str, str]:
    """Запускает асинхронное сравнение методов сегментации.

    Создаёт новую задачу, инициализирует SegmentationComparator и планирует выполнение через asyncio.create_task().

    Args:
        image (UploadFile): Загружаемое изображение для сравнения.
            Конвертируется в RGB numpy array перед обработкой.
        methods (str): JSON-строка с конфигурацией сравниваемых методов.
            Формат: список объектов с полями:
            - name (str): Уникальное имя метода в отчёте.
            - library (str, optional): Библиотека ("opencv"|"sklearn"|"torch").
            - method (str): Название метода в библиотеке.
            - params (Dict, optional): Параметры инициализации.
        reference (str): JSON-строка с конфигурацией референсного метода.
            Формат аналогичен methods, но для одного метода.
        comparison_type (str): Тип сравнения: "batch" (по умолчанию) или "pairwise".

    Returns:
        Dict[str, str]: Словарь с идентификатором задачи:
            ```json
            {"task_id": "550e8400-e29b-41d4-a716-446655440000"}
            ```

    Raises:
        HTTPException (422): Если methods или reference содержат невалидный JSON.

    Example Request (multipart/form-data):
        ```
        POST /api/comparator/start
        Content-Type: multipart/form-data

        image: (файл)
        methods: '[
            {"name": "otsu_cv2", "library": "opencv", "method": "otsu_thresholding"},
            {"name": "otsu_sk", "library": "sklearn", "method": "otsu_thresholding"}
        ]'
        reference: '{"name": "otsu_cv2", "library": "opencv", "method": "otsu_thresholding"}'
        comparison_type: "batch"
        ```

    Note:
        - Задача выполняется асинхронно: ответ возвращается немедленно.
        - Прогресс отслеживается через эндпоинт `/status/{task_id}`.
        - Файл изображения читается в память: для больших файлов может потребоваться
        увеличение лимита `max_file_size` в конфигурации FastAPI.
    """
    try:
        methods_config = json.loads(methods)
        reference_config = json.loads(reference)
    except json.JSONDecodeError as e:
        raise HTTPException(422, detail=f"Invalid JSON: {e}")
    from PIL import Image

    img: Image.Image = Image.open(image.file).convert("RGB")
    image_array: np.ndarray = np.array(img)

    task_id: str = str(uuid.uuid4())
    asyncio.create_task(
        _run_comparator_task(
            task_id=task_id,
            image=image_array,
            methods_config=methods_config,
            reference_config=reference_config,
            comparison_type=comparison_type,
        )
    )
    return {"task_id": task_id}


# ──────────────────────────────────────────────────────────────────────
@router.get("/status/{task_id}")
async def get_status(task_id: str) -> JSONResponse:
    """Возвращает текущий статус и прогресс задачи компаратора.

    Args:
        task_id (str): UUID задачи, полученный при запуске через `/start`.

    Returns:
        JSONResponse: Сериализованный объект задачи с полями:
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
            "message": "Сравнение otsu_sk (2/3)",
            "results": null
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
                    "methods_count": 3,
                    "successful": 3,
                    "avg_f1": 0.94
                },
                "results": [
                    {
                        "method": "otsu_sk",
                        "library": "sklearn",
                        "iou": 0.98,
                        "f1_score": 0.99,
                        "test_time": 0.045
                    }
                ],
                "charts": {
                    "comparison_summary.jpg": "iVBORw0KGgoAAAANSUh..."
                }
            }
        }
        ```

    Note:
        - Метрики с `NaN`/`Inf` автоматически заменяются на `null` перед сериализацией.
        - Для длительных задач рекомендуется polling с интервалом 2–5 секунд.
        - Ответ сериализуется через `safe_json_response()` для обработки numpy-типов.
    """
    async with _comparator_lock:
        task: Optional[Dict[str, Any]] = _comparator_tasks.get(task_id)
    if not task:
        raise HTTPException(404, detail="Task not found")

    if task.get("results") and "results" in task["results"]:
        for r in task["results"]["results"]:
            for k, v in r.items():
                if isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
                    r[k] = None

    return safe_json_response(task)


# ──────────────────────────────────────────────────────────────────────
@router.delete("/{task_id}")
async def cancel_comparator(task_id: str) -> Dict[str, str]:
    """Отменяет или удаляет задачу компаратора.

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
    async with _comparator_lock:
        task: Optional[Dict[str, Any]] = _comparator_tasks.get(task_id)
    if not task:
        return {"status": "not_found", "message": "Task not found"}
    if task["status"] in ("completed", "failed", "cancelled"):
        return {"status": task["status"], "message": f"Already {task['status']}"}
    task["status"] = "cancelled"
    task["message"] = "Отменено пользователем"
    return {"status": "cancelled"}
