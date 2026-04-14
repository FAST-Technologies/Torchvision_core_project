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
from typing import Optional, Dict, Any, List
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
from routers import benchmark, comparator, validator
from fastapi import Request

# from segmenters.NeuralModelFactory import NeuralModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autoseg")

# ── Кеш нейронных моделей ──────────────────────────────────────────────────
_model_cache: Dict[str, Any] = {}
_CACHE_MAX = 3

print(f"🔍 CWD: {os.getcwd()}")
print(f"🔍 __file__: {__file__}")


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
app.include_router(benchmark.router)
app.include_router(comparator.router)
app.include_router(validator.router)
print(f"📋 Registered routes: {[r.path for r in app.routes]}")

auto_seg = AutoSegmenter()


@app.middleware("http")
async def log_benchmark_requests(request: Request, call_next):
    if request.url.path.startswith("/api/benchmark"):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        logger.info(f"Benchmark {request.url.path} took {duration:.2f}s")
        return response
    return await call_next(request)


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
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "cached_models": len(_model_cache),
        "cache_max": _CACHE_MAX,
    }


@app.get("/api/cache_info")
async def cache_info() -> Dict[str, Any]:
    return {"count": len(_model_cache), "models": [k[:80] for k in _model_cache]}


@app.get("/api/methods_library")
async def get_methods_by_library(library: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
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
) -> Dict[str, Any]:
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


@app.get("/recommendations/")
async def get_recommendations_ep(file: UploadFile = File(...)) -> Dict[str, List[Dict[str, Any]]]:
    img = np.array(Image.open(io.BytesIO(await file.read())))
    return {"recommendations": auto_seg.get_recommendations(img, top_k=5)}


_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
