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
import asyncio
from typing import Optional, Dict, Any
import json, os, sys, base64, io, math, logging, time
from contextlib import asynccontextmanager
import uuid
from fastapi import BackgroundTasks

import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segmenters.AutoSegmenter import (
    AutoSegmenter,
    SegmentationGoal,
    METHODS_BY_LIBRARY,
    MethodProfile,
)
from metrics.SegmentationMetrics import SegmentationMetrics
from testing.TorchImplementationValidator import TorchImplementationValidator

# from segmenters.NeuralModelFactory import NeuralModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autoseg")

# ── Кеш нейронных моделей ──────────────────────────────────────────────────
_model_cache: Dict[str, Any] = {}
_CACHE_MAX = 3

_validation_tasks: Dict[str, Dict] = {}


def _get_or_load_neural(config: dict, task: str) -> Any:
    from segmenters.NeuralSegmenter import NeuralSegmenter

    cache_key = json.dumps({**config, "_task": task}, sort_keys=True)
    if cache_key not in _model_cache:
        if len(_model_cache) >= _CACHE_MAX:
            oldest = next(iter(_model_cache))
            del _model_cache[oldest]
            logger.info("Model cache evicted oldest entry")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {config.get('model_type')} on {device}")
        _model_cache[cache_key] = NeuralSegmenter(**config, device=device)
    return _model_cache[cache_key]


# ── FastAPI ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AutoSegmenter API starting…")
    yield
    _model_cache.clear()
    logger.info("Model cache cleared on shutdown")


app = FastAPI(title="AutoSegmenter API", version="2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

auto_seg = AutoSegmenter()

# ── Конфиг нейросетей ──────────────────────────────────────────────────────
NEURAL_CONFIGS: Dict[str, Dict[str, dict]] = {
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


def arr_to_b64(arr: np.ndarray) -> str:
    """numpy → data:image/png;base64,…"""
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def analyze_image_data(img_array: np.ndarray) -> dict:
    """Возвращает данные для визуализации анализа"""
    from scipy import ndimage

    # Гистограмма интенсивностей
    hist, bins = np.histogram(img_array.flatten(), bins=64, range=(0, 256))
    gray = (
        np.mean(img_array, axis=2).astype(np.float32)
        if img_array.ndim == 3
        else img_array.astype(np.float32)
    )
    sobel_x = ndimage.sobel(gray, axis=0)
    sobel_y = ndimage.sobel(gray, axis=1)
    edges = np.hypot(sobel_x, sobel_y)
    edges_norm = (edges / edges.max() * 255).astype(np.uint8)

    return {
        "histogram": hist.tolist(),
        "hist_bins": bins.tolist(),
        "edge_density": float(np.mean(edges > edges.max() * 0.3)),
        "edges_b64": arr_to_b64(edges_norm),
    }


def sanitize_metrics(m: dict) -> dict:
    """Заменяет inf/NaN на None для JSON-совместимости"""
    return {
        k: (None if isinstance(v, float) and (math.isinf(v) or math.isnan(v)) else v)
        for k, v in m.items()
    }


def build_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = np.stack([img] * 3, axis=-1) if img.ndim == 2 else img.copy()
    col = np.zeros_like(rgb)
    col[mask > 0] = [255, 0, 0]
    return (rgb * 0.6 + col * 0.4).astype(np.uint8)


def params_to_schema(params: Dict[str, Any]) -> Dict[str, Any]:
    schema: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, bool):
            schema[k] = {"type": "boolean", "default": v}
        elif isinstance(v, int):
            big = any(x in k for x in ("size", "bin", "iter", "scale", "radius"))
            schema[k] = {
                "type": "int",
                "min": 1,
                "max": 500 if big else 100,
                "step": 1,
                "default": v,
            }
        elif isinstance(v, float):
            norm = abs(v) <= 2.0 or any(
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


def _best_for(method_name: str) -> list:
    p = auto_seg.benchmark_data.get(method_name)
    return [t.value for t in p.best_for_type] if p else []


# ── Routes ─────────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "cached_models": len(_model_cache),
        "cache_max": _CACHE_MAX,
    }


@app.get("/api/cache_info")
async def cache_info():
    return {"count": len(_model_cache), "models": [k[:80] for k in _model_cache]}


@app.get("/api/methods_library")
async def get_methods_by_library(library: Optional[str] = None):
    if library and library in METHODS_BY_LIBRARY:
        source_dict = METHODS_BY_LIBRARY.get(library, {})
    else:
        source_dict = {
            name: profile
            for lib_methods in METHODS_BY_LIBRARY.values()
            for name, profile in lib_methods.items()
        }

    result = {}
    for name, profile in source_dict.items():
        # MethodProfile — это dataclass, обращаемся к атрибутам напрямую
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
            # Fallback для словарей (если вдруг source_dict содержит dict)
            result[name] = {
                "name": profile.get("name", name),
                "avg_iou": profile.get("avg_iou", 0.0),
                "description": profile.get("description", ""),
                "defaults": profile.get("params", {}),
                "schema": profile.get("schema", {}),
            }

    return {"methods": result}


@app.get("/api/methods")
async def get_methods(library: Optional[str] = None):
    """Возвращает доступные методы для указанной библиотеки"""
    if library and library not in METHODS_BY_LIBRARY:
        raise HTTPException(
            422,
            f"Unknown library: {library!r}. Available: {list(METHODS_BY_LIBRARY.keys())}",
        )
    source = (
        METHODS_BY_LIBRARY.get(library, {})
        if library
        else {n: p for lib in METHODS_BY_LIBRARY.values() for n, p in lib.items()}
    )
    result = {}
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
):
    t0 = time.perf_counter()
    try:
        img_pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_array = np.array(img_pil)
        mask: np.ndarray
        metadata: Dict[str, Any]
        overlay_np: np.ndarray

        # ─── НЕЙРОННЫЙ РЕЖИМ ───────────────────────────────────────────────
        if mode == "neural":
            from segmenters.NeuralSegmenter import NeuralSegmenter
            from utils.strategies import segment_image_unified
            from utils.palettes import ade_palette, coco_palette, cityscapes_palette

            PALETTES = {
                "semantic": ade_palette,  # ADE20K: 150 классов
                "instance": coco_palette,  # COCO: 80 классов
                "panoptic": cityscapes_palette,  # Cityscapes: 19 классов
            }

            CLASS_FN = {
                "semantic": NeuralSegmenter.get_ade_class_names,
                "instance": NeuralSegmenter.get_coco_class_names,
                "panoptic": NeuralSegmenter.get_cityscapes_class_names,
            }

            cfg = NEURAL_CONFIGS.get(task, {}).get(model)
            if not cfg:
                raise HTTPException(
                    422, f"Unknown neural config: task={task!r} model={model!r}"
                )
            ns = _get_or_load_neural(cfg, task)
            dev = "cuda" if torch.cuda.is_available() else "cpu"

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
                user_params: dict = json.loads(custom_params)
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

                profile = METHODS_BY_LIBRARY[library][method]
                # default_params = auto_seg.available_methods.get(method, {}).get(
                #     "params", {}
                # )
                # final_params = {**default_params, **user_params}
                final_params = {**(profile.params or {}), **user_params}
                logger.info(
                    f"🛠 Using params for {method}/{library} params={final_params}"
                )

                segmenter = auto_seg._get_segmenter_class(method, library)(
                    method=method, **final_params
                )
                _, mask = segmenter.segment_with_mask(img_array)
                metadata = {
                    "method": method,
                    "library": library,
                    "parameters": final_params,
                    "confidence": 1.0,
                    "image_characteristics": auto_seg.analyze_image(img_array),
                }

        # ─── Метрики ───────────────────────────────────────────────────────
        metrics = {}
        if gt_mask is not None:
            logger.info(f"✅ GT получен: {gt_mask.filename}")
            gt_array = np.array(
                Image.open(io.BytesIO(await gt_mask.read())).convert("L")
            )
            metrics = sanitize_metrics(
                SegmentationMetrics.calculate_all_metrics(mask, gt_array, threshold=0.5)
            )
        else:
            logger.warning("⚠️ GT не предоставлен, метрики не рассчитываются")

        # ─── Рекомендации ──────────────────────────────────────────────────
        recommendations = auto_seg.get_recommendations(img_array, top_k=5)
        analysis_data = analyze_image_data(img_array)
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


# ── Валидация методов ──────────────────────────────────────────────────────
# @app.post("/api/validate")
# async def validate_methods(
#     file: UploadFile = File(...),
#     primary_library: str = Form("torch"),      # "torch" | "opencv" | "sklearn"
#     reference_library: str = Form("opencv"),   # "torch" | "opencv" | "sklearn"
#     methods_filter: Optional[str] = Form(None), # "threshold" | "edge" | "region" | "all"
# ):
#     """
#     Запускает кросс-библиотечную валидацию методов сегментации.

#     Returns:
#         Сводные результаты валидации (без тяжёлых изображений)
#     """
#     t0 = time.perf_counter()
#     try:
#         # Загрузка изображения
#         img_array = np.array(Image.open(io.BytesIO(await file.read())).convert("RGB"))

#         # Инициализация валидатора
#         validator = TorchImplementationValidator(output_dir="./data/validation_web")

#         # Выбор методов для валидации
#         if methods_filter == "threshold":
#             methods_list = validator.threshold_methods
#         elif methods_filter == "edge":
#             methods_list = validator.edge_methods
#         elif methods_filter == "region":
#             methods_list = validator.region_methods
#         elif methods_filter == "clustering":
#             methods_list = validator.clastering_methods
#         else:
#             # Все методы (может быть долго!)
#             methods_list = (
#                 validator.threshold_methods +
#                 validator.edge_methods +
#                 validator.region_methods +
#                 validator.clastering_methods
#             )
#         # Маппинг библиотек к классам сегментеров
#         LIB_MAP = {
#             "torch": "TorchSegmenter",
#             "opencv": "OpenCVSegmenter",
#             "sklearn": "SklearnSegmenter",
#         }

# from segmenters.TorchSegmenter import TorchSegmenter
# from segmenters.OpenCVSegmenter import OpenCVSegmenter
# from segmenters.SklearnSegmenter import SklearnSegmenter

#         CLASS_MAP = {
#             "torch": TorchSegmenter,
#             "opencv": OpenCVSegmenter,
#             "sklearn": SklearnSegmenter,
#         }

#         primary_class = CLASS_MAP.get(primary_library, TorchSegmenter)
#         reference_class = CLASS_MAP.get(reference_library, OpenCVSegmenter)

#         # Запуск валидации
#         results = validator.validate_segmentation_methods(
#             image_path=img_array,
#             methods_list=methods_list,
#             torch_segmenter_class=primary_class,
#             reference_segmenter_class=reference_class,
#             reference=reference_library,
#             status_message=f"ВАЛИДАЦИЯ: {LIB_MAP[primary_library]} vs {LIB_MAP[reference_library]}",
#             prefix=f"web_validation_{primary_library}_{reference_library}",
#             validation_type=methods_filter or "all",
#             additional_method=LIB_MAP[primary_library],
#         )
#         # Подготовка лёгкого ответа для фронтенда
#         summary = []
#         for method, data in results.items():
#             if not data.get("success"):
#                 summary.append({
#                     "method": method,
#                     "success": False,
#                     "error": data.get("error", "Unknown error"),
#                 })
#                 continue

#             metrics = data.get("metrics", {})
#             summary.append({
#                 "method": method,
#                 "success": True,
#                 "validation_status": data.get("validation_status"),
#                 "iou": metrics.get("iou"),
#                 "dice": metrics.get("dice"),
#                 "pixel_accuracy": metrics.get("pixel_accuracy"),
#                 "precision": metrics.get("precision"),
#                 "recall": metrics.get("recall"),
#                 "f1_score": metrics.get("f1_score"),
#                 "mae": metrics.get("mae"),
#                 "hausdorff_distance": metrics.get("hausdorff_distance"),
#                 "primary_time": data.get("first_method_time"),
#                 "reference_time": data.get("second_method_time"),
#                 "time_diff": data.get("methods_time_difference"),
#             })

#         elapsed = (time.perf_counter() - t0) * 1000

#         return {
#             "success": True,
#             "elapsed_ms": round(elapsed, 1),
#             "primary_library": primary_library,
#             "reference_library": reference_library,
#             "methods_tested": len(summary),
#             "passed": sum(1 for s in summary if s.get("validation_status") == "PASS"),
#             "warning": sum(1 for s in summary if s.get("validation_status") == "WARNING"),
#             "failed": sum(1 for s in summary if s.get("validation_status") == "FAIL"),
#             "results": summary,
#             "report_dir": validator.output_dir,  # Для отладки
#         }
#     except HTTPException:
#         raise
#     except Exception as exc:
#         import traceback
#         logger.error(f"❌ /api/validate error: {exc}\n{traceback.format_exc()}")
#         raise HTTPException(500, f"Validation failed: {str(exc)}")


def _get_methods_for_filter(
    validator: TorchImplementationValidator, methods_filter: Optional[str]
):
    if methods_filter == "threshold":
        return validator.threshold_methods
    elif methods_filter == "edge":
        return validator.edge_methods
    elif methods_filter == "region":
        return validator.region_methods
    elif methods_filter == "clustering":
        return validator.clastering_methods
    else:
        return (
            validator.threshold_methods
            + validator.edge_methods
            + validator.region_methods
            + validator.clastering_methods
        )


@app.post("/api/validate")
async def start_validation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    primary_library: str = Form("torch"),
    reference_library: str = Form("opencv"),
    methods_filter: Optional[str] = Form(None),
):
    """Запускает валидацию в фоне и возвращает task_id"""
    task_id = str(uuid.uuid4())
    temp_validator = TorchImplementationValidator(output_dir="./data/validation_web")
    methods_list = _get_methods_for_filter(temp_validator, methods_filter)
    total_methods = len(methods_list)
    logger.info(f"🔧 methods_filter={methods_filter}, total_methods={total_methods}")
    _validation_tasks[task_id] = {
        "status": "running",
        "progress": 0,
        "total_methods": total_methods,
        "processed": 0,
        "start_time": time.time(),
        "results": None,
        "error": None,
        "fetched": False,
    }

    background_tasks.add_task(
        _run_validation_task,
        task_id,
        await file.read(),
        primary_library,
        reference_library,
        methods_filter,
    )

    return {"task_id": task_id, "status": "running"}


async def _run_validation_task(
    task_id: str,
    file_content: bytes,
    primary_library: str,
    reference_library: str,
    methods_filter: Optional[str],
):
    """Фоновая задача валидации"""
    try:
        _validation_tasks[task_id]["status"] = "running"
        _validation_tasks[task_id]["progress"] = 0

        # Загрузка изображения
        img_array = np.array(Image.open(io.BytesIO(file_content)).convert("RGB"))

        # Инициализация валидатора
        validator = TorchImplementationValidator(output_dir="./data/validation_web")

        # Выбор методов
        if methods_filter == "threshold":
            methods_list = validator.threshold_methods
        elif methods_filter == "edge":
            methods_list = validator.edge_methods
        elif methods_filter == "region":
            methods_list = validator.region_methods
        elif methods_filter == "clustering":
            methods_list = validator.clastering_methods
        else:
            methods_list = (
                validator.threshold_methods
                + validator.edge_methods
                + validator.region_methods
                + validator.clastering_methods
            )

        _validation_tasks[task_id]["total_methods"] = len(methods_list)
        import cv2
        from segmenters.TorchSegmenter import TorchSegmenter
        from segmenters.OpenCVSegmenter import OpenCVSegmenter
        from segmenters.SklearnSegmenter import SklearnSegmenter

        # Маппинг библиотек
        CLASS_MAP = {
            "torch": TorchSegmenter,
            "opencv": OpenCVSegmenter,
            "sklearn": SklearnSegmenter,
        }
        primary_class = CLASS_MAP.get(primary_library, TorchSegmenter)
        reference_class = CLASS_MAP.get(reference_library, OpenCVSegmenter)

        results = {}
        for idx, (method_name, params) in enumerate(methods_list):
            try:
                result = await asyncio.to_thread(
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
                results[method_name] = {"success": False, "error": str(e)}

            # 🔹 Обновляем прогресс
            _validation_tasks[task_id]["processed"] = idx + 1
            _validation_tasks[task_id]["progress"] = round(
                (idx + 1) / len(methods_list) * 100, 1
            )
            _validation_tasks[task_id]["elapsed_ms"] = round(
                (time.time() - _validation_tasks[task_id]["start_time"]) * 1000, 1
            )
            await asyncio.sleep(0)

        # Завершение
        _validation_tasks[task_id]["status"] = "completed"
        _validation_tasks[task_id]["results"] = results
        _validation_tasks[task_id]["elapsed_ms"] = round(
            (time.time() - _validation_tasks[task_id]["start_time"]) * 1000, 1
        )

    except Exception as e:
        _validation_tasks[task_id]["status"] = "failed"
        _validation_tasks[task_id]["error"] = str(e)
        _validation_tasks[task_id]["elapsed_ms"] = round(
            (time.time() - _validation_tasks[task_id].get("start_time", time.time()))
            * 1000,
            1,
        )
        logger.error(f"❌ Validation task {task_id} failed: {e}")


def _process_single_method(
    method_name: str,
    params: dict,
    img_array: np.ndarray,
    primary_class,
    reference_class,
    validator,
) -> dict:
    """Синхронная обработка одного метода — запускается в отдельном потоке"""
    import cv2
    from metrics.SegmentationMetrics import SegmentationMetrics

    logger.info(f"🔹 START Processing method: {method_name}")
    try:
        # Primary реализация
        start1 = time.time()
        seg1 = primary_class(method=method_name, **params)
        mask1 = seg1.segment(img_array, **params)
        time1 = time.time() - start1
        logger.info(f"✅ {method_name} primary done: {time1:.3f}s")

        # Reference реализация
        start2 = time.time()
        ref_params = params.copy()
        ref_params["postprocess"] = False
        seg2 = reference_class(method=method_name, **ref_params)
        mask2 = seg2.segment(img_array, **ref_params)
        time2 = time.time() - start2
        logger.info(f"✅ {method_name} primary done: {time2:.3f}s")

        # Метрики
        metrics = SegmentationMetrics.calculate_all_metrics(mask1, mask2, threshold=0.5)
        metrics.update(
            {
                "first_method_time": time1,
                "second_method_time": time2,
                "methods_time_difference": abs(time1 - time2),
            }
        )

        status = validator._check_validation_status(metrics)

        # Визуализация
        orig_gray = (
            cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            if img_array.ndim == 3
            else img_array
        )
        mask1_vis = mask1 if mask1.max() == 255 else mask1 * 255
        mask2_vis = mask2 if mask2.max() == 255 else mask2 * 255
        diff = np.abs(mask1_vis.astype(int) - mask2_vis.astype(int))
        logger.info(f"✅ FINISHED {method_name}")
        result = {
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
                size_kb = len(result[key]) / 1024
                logger.info(f"📦 {key}: {size_kb:.1f} KB")
                if size_kb > 500:  # >500KB — предупреждение
                    logger.warning(f"⚠️ Large base64 string: {key} ({size_kb:.1f} KB)")
        return result
    except Exception as e:
        logger.error(f"❌ ERROR in {method_name}: {type(e).__name__}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise  # Перехватится в _run_validation_task


@app.get("/api/validate/status/{task_id}")
async def get_validation_status(task_id: str):
    # 🔹 Сначала проверяем, существует ли задача!
    task = _validation_tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    # 🔹 Только потом работаем с task как с словарём
    if task["status"] in ("completed", "failed") and task.get("results") is not None:
        task["fetched"] = True

    # 🔹 Очистка старых задач (проверить или закомментить)
    now = time.time()
    for tid in list(_validation_tasks.keys()):
        if tid != task_id:
            t = _validation_tasks[tid]
            if (t["status"] in ("completed", "failed") and t.get("fetched", False)) or (
                now - t.get("start_time", 0) > 3600
            ):
                del _validation_tasks[tid]
                logger.info(f"🗑 Cleaned up old task {tid}")

    logger.info(
        f"🔍 Status response for {task_id}: status={task['status']}, total={task.get('total_methods')}, processed={task.get('processed')}"
    )
    logger.info(f"🔍 Task {task_id} found: {task_id in _validation_tasks}")
    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "processed": task["processed"],
        "total_methods": task["total_methods"],
        "elapsed_ms": task.get("elapsed_ms"),
    }

    if task["status"] == "completed":
        summary = []
        for method, data in task["results"].items():
            if not data.get("success"):
                summary.append(
                    {"method": method, "success": False, "error": data.get("error")}
                )
                continue
            metrics = data.get("metrics", {})
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

        response["results"] = summary
        response["passed"] = sum(
            1 for s in summary if s.get("validation_status") == "PASS"
        )
        response["warning"] = sum(
            1 for s in summary if s.get("validation_status") == "WARNING"
        )
        response["failed"] = sum(
            1 for s in summary if s.get("validation_status") == "FAIL"
        )
        response["methods_tested"] = len(summary)
        response["report_dir"] = "./data/validation_web"

    elif task["status"] == "failed":
        response["error"] = task["error"]

    return response


@app.get("/recommendations/")
async def get_recommendations_ep(file: UploadFile = File(...)):
    img = np.array(Image.open(io.BytesIO(await file.read())))
    return {"recommendations": auto_seg.get_recommendations(img, top_k=5)}


_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
