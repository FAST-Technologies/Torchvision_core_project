# backend/main.py
"""
AutoSegmenter API — улучшенная версия FastAPI бэкенда.

Исправления:
  1. Дублирующийся маршрут /api/methods убран.
  2. Поле best_for добавлено в рекомендации.
  3. Кеш нейронных моделей — модель грузится один раз.
  4. Bare except → except (json.JSONDecodeError, ValueError).
  5. Две функции b64-кодирования объединены в arr_to_b64.
  6. Добавлены /api/health и /api/cache_info.
  7. HTTP 422 вместо 500 при невалидных входных данных.
  8. elapsed_ms и library возвращаются клиенту.
  9. Пользовательские параметры корректно мёржатся с дефолтами.
"""
# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
import os, sys, json, io, math, logging, time, uuid, base64
from typing import (
    Dict,
    Any,
    Optional,
    List,
    Union,
    Callable,
    Literal,
    TypeAlias,
)
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    HTTPException,
    Request,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Локальные импорты
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segmenters.AutoSegmenter import (
    AutoSegmenter,
    SegmentationGoal,
    METHODS_BY_LIBRARY,
    MethodProfile,
)
from metrics.SegmentationMetrics import SegmentationMetrics
from routers import benchmark, comparator, validator
from segmenters.NeuralSegmenter import NeuralSegmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autoseg")

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & CONSTANTS
# ──────────────────────────────────────────────────────────────────────
ImageArray = np.ndarray
MaskArray = np.ndarray
MetricsDict = Dict[str, Any]
PathLike = Union[str, Path]
DeviceStr = Literal["cuda", "cpu"]
ModelConfigDict = Dict[str, Dict[str, Any]]
NeuralConfigDict = Dict[str, ModelConfigDict]
RecommendationDict = Dict[str, Any]
AnalysisDataDict = Dict[str, Any]
ChartDict = Dict[str, str]  # fname -> base64 string
SegmentResponseDict = Dict[str, Any]

# ──────────────────────────────────────────────────────────────────────
# КЕШ МОДЕЛЕЙ
# ──────────────────────────────────────────────────────────────────────
_model_cache: Dict[str, Any] = {}
_CACHE_MAX: int = 3

print(f"🔍 CWD: {os.getcwd()}")
print(f"🔍 __file__: {__file__}")


def _get_or_load_neural(config: Dict[str, Any], task: str) -> Any:
    """
    Загружает нейронный сегментер с LRU-кешем (макс. 3 модели).

    Args:
        config: Конфигурация модели (model_type, model_name, checkpoint_path, ...).
        task: Тип задачи ("semantic", "instance", "panoptic") для формирования ключа кеша.

    Returns:
        NeuralSegmenter: Экземпляр загруженной модели.
    """
    cache_key: str = json.dumps({**config, "_task": task}, sort_keys=True)
    if cache_key not in _model_cache:
        if len(_model_cache) >= _CACHE_MAX:
            oldest: str = next(iter(_model_cache))
            del _model_cache[oldest]
            logger.info("Model cache evicted oldest entry")
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {config.get('model_type')} on {device}")
        _model_cache[cache_key] = NeuralSegmenter(**config, device=device)
    return _model_cache[cache_key]


# ──────────────────────────────────────────────────────────────────────
# УТИЛИТЫ
# ──────────────────────────────────────────────────────────────────────
def arr_to_b64(arr: np.ndarray) -> str:
    """
    Конвертирует numpy-массив в base64 PNG-строку.

    Args:
        arr: Входной массив (H×W или H×W×C). Автоматически приводится к uint8.

    Returns:
        str: Строка формата `data:image/png;base64,...`.
    """
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def analyze_image_data(img_array: ImageArray) -> AnalysisDataDict:
    """
    Извлекает статистические данные изображения для визуализации.

    Args:
        img_array: RGB или grayscale изображение.

    Returns:
        Dict[str, Any]: Гистограмма, плотность границ, Sobel-края в base64.
    """
    from scipy import ndimage

    # Гистограмма интенсивностей
    hist, bins = np.histogram(img_array.flatten(), bins=64, range=(0, 256))
    gray: np.ndarray = (
        np.mean(img_array, axis=2).astype(np.float32)
        if img_array.ndim == 3
        else img_array.astype(np.float32)
    )
    sobel_x: np.ndarray = ndimage.sobel(gray, axis=0)
    sobel_y: np.ndarray = ndimage.sobel(gray, axis=1)
    edges: np.ndarray = np.hypot(sobel_x, sobel_y)
    edges_norm: np.ndarray = (edges / (edges.max() + 1e-8) * 255).astype(np.uint8)

    return {
        "histogram": hist.tolist(),
        "hist_bins": bins.tolist(),
        "edge_density": float(np.mean(edges > edges.max() * 0.3)),
        "edges_b64": arr_to_b64(edges_norm),
    }


def sanitize_metrics(m: MetricsDict) -> MetricsDict:
    """
    Заменяет `inf` и `NaN` на `None` для JSON-совместимости.

    Args:
        m: Словарь с метриками.

    Returns:
        MetricsDict: Очищенный словарь.
    """
    return {
        k: (None if isinstance(v, float) and (math.isinf(v) or math.isnan(v)) else v)
        for k, v in m.items()
    }


def build_overlay(img: ImageArray, mask: MaskArray, alpha: float = 0.4) -> ImageArray:
    """
    Создаёт наложение маски на оригинальное изображение.

    Args:
        img: Оригинальное изображение (H×W или H×W×C).
        mask: Бинарная маска (H×W).
        alpha: Прозрачность маски (0.0–1.0).

    Returns:
        np.ndarray: RGB-изображение с наложенной маской.
    """
    rgb: np.ndarray = np.stack([img] * 3, axis=-1) if img.ndim == 2 else img.copy()
    col: np.ndarray = np.zeros_like(rgb)
    col[mask > 0] = [255, 0, 0]
    return (rgb * (1 - alpha) + col * alpha).astype(np.uint8)


def params_to_schema(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Генерирует JSON-схему UI-конфигуратора из параметров по умолчанию.

    Args:
        params: Словарь параметров метода.

    Returns:
        Dict[str, Any]: Схема с типами, диапазонами и подписями.
    """
    schema: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, bool):
            schema[k] = {"type": "boolean", "default": v}
        elif isinstance(v, int):
            big: bool = any(x in k for x in ("size", "bin", "iter", "scale", "radius"))
            schema[k] = {
                "type": "int",
                "min": 1,
                "max": 500 if big else 100,
                "step": 1,
                "default": v,
            }
        elif isinstance(v, float):
            norm: bool = abs(v) <= 2.0 or any(
                x in k for x in ("threshold", "k", "ratio", "factor")
            )
            schema[k] = {
                "type": "float",
                "min": 0.0,
                "max": 1.0 if norm else 100.0,
                "step": 0.01,
                "default": v,
            }
        else:
            schema[k] = {"type": "string", "default": str(v)}
    return schema


def _best_for(method_name: str) -> List[str]:
    """
    Возвращает список типов изображений, для которых метод оптимален.

    Args:
        method_name: Имя метода.

    Returns:
        List[str]: Список значений `ImageType.value`.
    """
    p = auto_seg.benchmark_data.get(method_name)
    return [t.value for t in p.best_for_type] if p else []


# ──────────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ НЕЙРОСЕТЕЙ
# ──────────────────────────────────────────────────────────────────────
NEURAL_CONFIGS: NeuralConfigDict = {
    "semantic": {
        "segformer_b0": {
            "model_type": "segformer",
            "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
        },
        "segformer_b1": {
            "model_type": "segformer",
            "model_name": "nvidia/segformer-b1-finetuned-ade-512-512",
        },
        "segformer_b2": {
            "model_type": "segformer",
            "model_name": "nvidia/segformer-b2-finetuned-ade-512-512",
        },
        "segformer_b3": {
            "model_type": "segformer",
            "model_name": "nvidia/segformer-b3-finetuned-ade-640-640",
        },
        "segformer_b4": {
            "model_type": "segformer",
            "model_name": "nvidia/segformer-b4-finetuned-ade-640-640",
        },
        "segformer_b5": {
            "model_type": "segformer",
            "model_name": "nvidia/segformer-b5-finetuned-ade-640-640",
        },
        "mask2former_swin_base": {
            "model_type": "mask2former",
            "model_name": "facebook/mask2former-swin-base-ade-semantic",
        },
        "mask2former_swin_large": {
            "model_type": "mask2former",
            "model_name": "facebook/mask2former-swin-large-ade-semantic",
        },
        "oneformer_swin_large": {
            "model_type": "oneformer",
            "model_name": "shi-labs/oneformer_ade20k_swin_large",
        },
        "dpt_large": {"model_type": "dpt", "model_name": "Intel/dpt-large-ade"},
        "upernet_convnext_small": {
            "model_type": "upernet",
            "model_name": "openmmlab/upernet-convnext-small",
        },
        "unet_resnet34": {"model_type": "unet_smp", "encoder_name": "resnet34"},
        "unet_resnet50": {"model_type": "unet_smp", "encoder_name": "resnet50"},
        "unet_efficientnet_b0": {
            "model_type": "unet_smp",
            "encoder_name": "efficientnet-b0",
        },
        "unet_mit_b5": {"model_type": "unet_smp", "encoder_name": "mit_b5"},
        "fpn_mit_b5": {"model_type": "fpn_smp", "encoder_name": "mit_b5"},
        "fpn_efficientnet": {
            "model_type": "fpn_smp",
            "encoder_name": "efficientnet-b5",
        },
        "psp_mit_b5": {"model_type": "pspnet_smp", "encoder_name": "mit_b5"},
        "psp_resnet50": {"model_type": "pspnet_smp", "encoder_name": "resnet50"},
        "deeplab_resnet101": {"model_type": "deeplab_tv"},
        "fcn_resnet50": {"model_type": "fcn_tv", "variant": "fcn_resnet50"},
        "fcn_resnet101": {"model_type": "fcn_tv", "variant": "fcn_resnet101"},
        "segnet_resnet34": {"model_type": "segnet", "encoder_name": "resnet34"},
        "mobile_sam": {"model_type": "sam", "model_name": "mobile_sam.pt"},
        "sam2_tiny": {"model_type": "sam", "model_name": "sam2_t.pt"},
    },
    "instance": {
        "mask2former_coco_instance": {
            "model_type": "mask2former",
            "model_name": "facebook/mask2former-swin-base-coco-instance",
        },
        "maskformer_resnet50": {
            "model_type": "maskformer",
            "model_name": "facebook/maskformer-resnet50-ade20k-full",
        },
        "yolov8n_seg": {"model_type": "yolov8", "model_name": "yolov8n-seg.pt"},
        "yolov8s_seg": {"model_type": "yolov8", "model_name": "yolov8s-seg.pt"},
        "yolov8m_seg": {"model_type": "yolov8", "model_name": "yolov8m-seg.pt"},
        "maskrcnn_resnet50": {
            "model_type": "maskrcnn_tv",
            "variant": "maskrcnn_resnet50_fpn",
        },
        "maskrcnn_resnet50_v2": {
            "model_type": "maskrcnn_tv",
            "variant": "maskrcnn_resnet50_fpn_v2",
        },
        "mobile_sam": {"model_type": "sam", "model_name": "mobile_sam.pt"},
        "sam2_tiny": {"model_type": "sam", "model_name": "sam2_t.pt"},
    },
    "panoptic": {
        "mask2former_ade_panoptic": {
            "model_type": "mask2former",
            "model_name": "facebook/mask2former-swin-base-ade-panoptic",
        },
        "mask2former_coco_panoptic": {
            "model_type": "mask2former",
            "model_name": "facebook/mask2former-swin-base-coco-panoptic",
        },
        "oneformer_coco_panoptic": {
            "model_type": "oneformer",
            "model_name": "shi-labs/oneformer_coco_swin_large",
        },
    },
}


# def build_neural_configs() -> Dict[str, Dict[str, Dict]]:
#     """Авто-генерация NEURAL_CONFIGS из конфига фабрики"""
#     config = NeuralModelFactory.load_config()
#     result = {"semantic": {}, "instance": {}, "panoptic": {}}

#     # SegFormer
#     for variant, name in config["models"]["segformer"]["variants"].items():
#         result["semantic"][f"segformer_{variant}"] = {
#             "model_type": "segformer",
#             "model_name": name,
#         }

#     # Mask2Former
#     for variant, name in config["models"]["mask2former"]["variants"].items():
#         result["semantic"][f"mask2former_{variant}"] = {
#             "model_type": "mask2former",
#             "model_name": name,
#         }
#         result["instance"][f"mask2former_{variant}_instance"] = {
#             "model_type": "mask2former",
#             "model_name": name.replace("-semantic", "-coco-instance"),
#         }
#         result["panoptic"][f"mask2former_{variant}_panoptic"] = {
#             "model_type": "mask2former",
#             "model_name": name.replace("-semantic", "-coco-panoptic"),
#         }

#     # SMP модели
#     for encoder in config["models"]["unet"]["encoders"]:
#         result["semantic"][f"unet_{encoder}"] = {
#             "model_type": "unet_smp",
#             "encoder_name": encoder,
#         }

#     return result


# ──────────────────────────────────────────────────────────────────────
# FASTAPI ПРИЛОЖЕНИЕ
# ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Жизненный цикл приложения: очистка кеша при завершении."""
    logger.info("AutoSegmenter API starting…")
    yield
    _model_cache.clear()
    logger.info("Model cache cleared on shutdown")


app: FastAPI = FastAPI(title="AutoSegmenter API", version="2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(benchmark.router)
app.include_router(comparator.router)
app.include_router(validator.router)
print(f"📋 Registered routes: {[r.path for r in app.routes if hasattr(r, 'path')]}")

auto_seg = AutoSegmenter()


@app.middleware("http")
async def log_benchmark_requests(
    request: Request, call_next: Callable[[Request], Any]
) -> Any:
    """Мидлвэр логирования времени выполнения бенчмарков."""
    if request.url.path.startswith("/api/benchmark"):
        start: float = time.perf_counter()
        response = await call_next(request)
        duration: float = time.perf_counter() - start
        logger.info(f"Benchmark {request.url.path} took {duration:.2f}s")
        return response
    return await call_next(request)


# ── Routes ─────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health() -> Dict[str, Any]:
    """
    Возвращает статус системы: доступность CUDA, использование VRAM, активные задачи.

    Returns:
        Dict[str, Any]: Статус, device_name, vram_mb, active_tasks.
    """
    if torch.cuda.is_available():
        total_vram_mb: float = (
            torch.cuda.get_device_properties(0).total_memory / 1024**2
        )
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
        "active_tasks": len(
            [
                t
                for t in benchmark.benchmark_tasks.values()
                if t.get("status") == "running"
            ]
        ),
        "cached_models": len(_model_cache),
        "cache_max": _CACHE_MAX,
    }


@app.get("/api/cache_info")
async def cache_info() -> Dict[str, Any]:
    """
    Возвращает информацию о кеше загруженных нейронных моделей.

    Returns:
        Dict[str, Any]: Количество закэшированных моделей, список ключей.
    """
    return {"count": len(_model_cache), "models": [k[:80] for k in _model_cache]}


@app.get("/api/methods_library")
async def get_methods_by_library(
    library: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    source_dict: Dict[str, MethodProfile]
    if library and library in METHODS_BY_LIBRARY:
        source_dict = METHODS_BY_LIBRARY.get(library, {})
    else:
        source_dict = {
            name: profile
            for lib_methods in METHODS_BY_LIBRARY.values()
            for name, profile in lib_methods.items()
        }

    result: Dict[str, Any] = {}
    for name, profile in source_dict.items():
        if isinstance(profile, MethodProfile):
            result[name] = {
                "name": profile.name,
                "library": profile.library,
                "avg_iou": profile.avg_iou,
                "avg_time_ms": profile.avg_time_ms,
                "memory_mb": profile.memory_mb,
                "robustness": profile.robustness,
                "description": profile.description,
                "best_for": [t.value for t in profile.best_for_type],
                "defaults": profile.params if profile.params else {},
                "schema": (
                    profile.schema
                    if profile.schema
                    else params_to_schema(profile.params)
                ),
            }
        else:
            # Fallback
            result[name] = {
                "name": profile.get("name", name),
                "avg_iou": profile.get("avg_iou", 0.0),
                "description": profile.get("description", ""),
                "defaults": profile.get("params", {}),
                "schema": profile.get("schema", {}),
            }

    return {"methods": result}


@app.get("/api/methods")
async def get_methods(library: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Возвращает доступные методы для указанной библиотеки (алиас `/api/methods_library`).
    """
    if library and library not in METHODS_BY_LIBRARY:
        raise HTTPException(
            422,
            f"Unknown library: {library!r}. Available: {list(METHODS_BY_LIBRARY.keys())}",
        )
    source: Dict[str, MethodProfile] = (
        METHODS_BY_LIBRARY.get(library, {})
        if library
        else {n: p for lib in METHODS_BY_LIBRARY.values() for n, p in lib.items()}
    )
    result: Dict[str, Any] = {}
    for name, profile in source.items():
        if isinstance(profile, MethodProfile):
            result[name] = {
                "name": profile.name,
                "library": profile.library,
                "avg_iou": profile.avg_iou,
                "avg_time_ms": profile.avg_time_ms,
                "memory_mb": profile.memory_mb,
                "robustness": profile.robustness,
                "description": profile.description,
                "best_for": [t.value for t in profile.best_for_type],
                "defaults": profile.params or {},
                "schema": (
                    profile.schema
                    if profile.schema
                    else params_to_schema(profile.params or {})
                ),
            }
        else:
            result[name] = {
                "name": profile.get("name", name),
                "avg_iou": profile.get("avg_iou", 0.0),
                "description": profile.get("description", ""),
                "defaults": profile.get("params", {}),
                "schema": profile.get("schema", {}),
            }
    return {"methods": result}


@app.post("/api/segment")
async def segment(
    file: UploadFile = File(...),
    mode: str = Form("classical"),  # "classical" | "neural"
    task: str = Form("semantic"),
    model: str = Form("segformer_b2"),
    goal: str = Form("balanced"),
    auto_select: bool = Form(True),
    method: Optional[str] = Form(None),
    library: Optional[str] = Form("opencv"),
    custom_params: str = Form("{}"),
    gt_mask: Optional[UploadFile] = File(default=None),
) -> SegmentResponseDict:
    """
    Основной эндпоинт сегментации изображения.

    Поддерживает:
    - Классические методы (авто-выбор или ручной).
    - Нейронные модели (SegFormer, Mask2Former, SAM, YOLOv8, SMP, Torchvision).
    - Расчёт метрик при наличии GT.
    - Генерацию рекомендаций и анализ изображения.

    Args:
        file: Входное изображение.
        mode: Режим работы ("classical" | "neural").
        task: Тип задачи ("semantic" | "instance" | "panoptic").
        model: Имя нейронной модели (если mode="neural").
        goal: Цель оптимизации ("speed" | "accuracy" | "balanced" | "low_memory").
        auto_select: Автоматически выбрать метод.
        method: Ручной выбор метода.
        library: Библиотека для классических методов.
        custom_params: JSON-строка с пользовательскими параметрами.
        gt_mask: Опциональная GT-маска для расчёта метрик.

    Returns:
        SegmentResponseDict: Словарь с маской, overlay, метриками, рекомендациями и временем.
    """
    t0: float = time.perf_counter()
    try:
        img_pil: Image.Image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_array: np.ndarray = np.array(img_pil)
        mask: np.ndarray
        metadata: Dict[str, Any]
        overlay_np: np.ndarray

        # ─── НЕЙРОННЫЙ РЕЖИМ ───────────────────────────────────────────────
        if mode == "neural":
            from segmenters.NeuralSegmenter import NeuralSegmenter
            from utils.strategies import segment_image_unified
            from utils.palettes import ade_palette, coco_palette, cityscapes_palette

            # 🔹 PALETTES: словарь, где значения — функции без аргументов, возвращающие палитру
            PALETTES: Dict[str, Callable[[], List[List[int]]]] = {
                "semantic": ade_palette,  # ADE20K: 150 классов
                "instance": coco_palette,  # COCO: 80 классов
                "panoptic": cityscapes_palette,  # Cityscapes: 19 классов
            }

            # 🔹 CLASS_FN: словарь, где значения — функции без аргументов, возвращающие имена классов
            CLASS_FN: Dict[str, Callable[[], Dict[int, str]]] = {
                "semantic": NeuralSegmenter.get_ade_class_names,
                "instance": NeuralSegmenter.get_coco_class_names,
                "panoptic": NeuralSegmenter.get_cityscapes_class_names,
            }

            cfg: Optional[Dict[str, Any]] = NEURAL_CONFIGS.get(task, {}).get(model)
            if not cfg:
                raise HTTPException(
                    422, f"Unknown neural config: task={task!r} model={model!r}"
                )
            ns = _get_or_load_neural(cfg, task)
            dev: str = "cuda" if torch.cuda.is_available() else "cpu"

            overlay_pil: Image.Image
            result_info: Dict[str, Any]
            overlay_pil, result_info = segment_image_unified(
                model=ns.model,
                processor=ns.processor,
                image_input=img_pil,
                model_type=cfg["model_type"],
                alpha=0.6,  # Прозрачность наложения
                palette=PALETTES[task],
                device=dev,
                verbose=False,
                num_classes=ns.num_classes,
                class_names=CLASS_FN[task](),
                gt_mask=None,
            )

            raw = result_info.get("mask")
            mask = (
                raw
                if raw is not None
                else (np.array(overlay_pil)[:, :, 0] > 0).astype(np.uint8) * 255
            )

            overlay_np = np.array(overlay_pil)

            metadata = {
                "method": model,
                "library": "neural",
                "task": task,
                "parameters": cfg,
                "confidence": 1.0,
                "image_characteristics": auto_seg.analyze_image(img_array),
                "inference_time_ms": result_info.get("inference_time_ms", 0),
                "unique_classes": result_info.get("unique_classes", 0),
            }
        else:
            auto_seg.goal = (
                SegmentationGoal(goal)
                if goal in ["speed", "accuracy", "balanced", "low_memory"]
                else SegmentationGoal.BALANCED
            )

            try:
                user_params: Dict[str, Any] = json.loads(custom_params)
            except (json.JSONDecodeError, ValueError):
                user_params = {}

            if auto_select:
                mask, metadata = auto_seg.segment(
                    img_array, auto_select=True, library=library, return_metadata=True
                )
            else:
                if not method:
                    raise HTTPException(422, "method required when auto_select=False")

                if library not in METHODS_BY_LIBRARY:
                    raise HTTPException(422, f"Unknown library: {library!r}")
                if method not in METHODS_BY_LIBRARY[library]:
                    raise HTTPException(
                        422,
                        f"Method {method!r} not in {library!r}. "
                        f"Available: {list(METHODS_BY_LIBRARY[library].keys())}",
                    )

                profile: MethodProfile = METHODS_BY_LIBRARY[library][method]
                final_params: Dict[str, Any] = {**(profile.params or {}), **user_params}
                logger.info(
                    f"🛠 Using params for {method}/{library} params={final_params}"
                )

                segmenter = auto_seg._get_segmenter_class(method, library)(
                    **final_params
                )
                result = segmenter.segment_with_mask(img_array)
                if isinstance(result, tuple) and len(result) == 2:
                    _, mask_opt = result
                    mask = (
                        mask_opt
                        if mask_opt is not None
                        else np.zeros_like(img_array[:, :, 0], dtype=np.uint8)
                    )
                else:
                    # Fallback для методов, возвращающих только одну маску
                    mask = result
                metadata = {
                    "method": method,
                    "library": library,
                    "parameters": final_params,
                    "confidence": 1.0,
                    "image_characteristics": auto_seg.analyze_image(img_array),
                }

        # ─── Метрики ───────────────────────────────────────────────────────
        metrics: MetricsDict = {}
        if gt_mask is not None:
            logger.info(f"✅ GT получен: {gt_mask.filename}")
            gt_array: np.ndarray = np.array(
                Image.open(io.BytesIO(await gt_mask.read())).convert("L")
            )
            metrics = sanitize_metrics(
                SegmentationMetrics.calculate_all_metrics(mask, gt_array, threshold=0.5)
            )
        else:
            logger.warning("⚠️ GT не предоставлен, метрики не рассчитываются")

        # ─── Рекомендации & Анализ ──────────────────────────────────────────────────
        recommendations: List[RecommendationDict] = auto_seg.get_recommendations(
            img_array, top_k=5
        )
        analysis_data: AnalysisDataDict = analyze_image_data(img_array)
        chars = metadata["image_characteristics"]

        if mode == "neural":
            pass
        else:
            overlay_np = build_overlay(img_array, mask)

        return {
            "success": True,
            "method": metadata["method"],
            "confidence": float(metadata["confidence"]),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "mask_b64": arr_to_b64(mask),
            "overlay_b64": arr_to_b64(overlay_np),
            "chars": {
                "type": chars.estimated_type.value,
                "size": f"{chars.width}×{chars.height}",
                "contrast": round(float(chars.contrast), 4),
                "noise": round(float(chars.noise_level), 4),
                "channels": chars.channels,
                "mean_intensity": round(float(chars.mean_intensity), 4),
                "edge_density": round(float(chars.edge_density), 4),
                "complexity": round(float(chars.complexity_score), 4),
            },
            "metrics": metrics,
            "recommendations": [
                {
                    "method": r["method"],
                    "score": round(float(r["score"]), 4),
                    "estimated_time_ms": float(r.get("estimated_time_ms", 0)),
                    "estimated_iou": float(r.get("estimated_iou", 0)),
                    "best_for": _best_for(r["method"]),
                }
                for r in recommendations
            ],
            "analysis": analysis_data,
            "examples": {
                "medical": ["otsu", "sauvola", "phansalkar", "adaptive_thresholding"],
                "documents": ["otsu", "adaptive_thresholding", "bernsen", "niblack"],
                "nature": ["canny_edge", "sobel_edge", "watershed", "felzenszwalb"],
                "industrial": [
                    "adaptive_thresholding",
                    "bernsen",
                    "gradient_magnitude_direction",
                    "log_edge",
                ],
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        logger.error(f"❌ /api/segment error: {exc}\n{traceback.format_exc()}")
        raise HTTPException(500, str(exc))


@app.get("/recommendations/")
async def get_recommendations_ep(
    file: UploadFile = File(...),
) -> Dict[str, List[RecommendationDict]]:
    """
    Возвращает топ-5 рекомендаций методов для загруженного изображения.
    """
    img: np.ndarray = np.array(Image.open(io.BytesIO(await file.read())))
    return {"recommendations": auto_seg.get_recommendations(img, top_k=5)}


# ──────────────────────────────────────────────────────────────────────
# STATIC FILES & ENTRY POINT
# ──────────────────────────────────────────────────────────────────────
_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
