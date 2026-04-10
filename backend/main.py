# backend/app.py
from typing import Optional, Dict, Any
import json
import os, sys, base64, io
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segmenters.AutoSegmenter import (
    AutoSegmenter,
    SegmentationGoal,
    METHODS_BY_LIBRARY,
    ALL_METHODS,
    ImageType,
    MethodProfile,
)
from metrics.SegmentationMetrics import SegmentationMetrics
from segmenters.NeuralModelFactory import NeuralModelFactory

app = FastAPI(title="AutoSegmenter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

auto_seg = AutoSegmenter()


def build_neural_configs() -> Dict[str, Dict[str, Dict]]:
    """Авто-генерация NEURAL_CONFIGS из конфига фабрики"""
    config = NeuralModelFactory.load_config()
    result = {"semantic": {}, "instance": {}, "panoptic": {}}

    # SegFormer
    for variant, name in config["models"]["segformer"]["variants"].items():
        result["semantic"][f"segformer_{variant}"] = {
            "model_type": "segformer",
            "model_name": name,
        }

    # Mask2Former
    for variant, name in config["models"]["mask2former"]["variants"].items():
        result["semantic"][f"mask2former_{variant}"] = {
            "model_type": "mask2former",
            "model_name": name,
        }
        result["instance"][f"mask2former_{variant}_instance"] = {
            "model_type": "mask2former",
            "model_name": name.replace("-semantic", "-coco-instance"),
        }
        result["panoptic"][f"mask2former_{variant}_panoptic"] = {
            "model_type": "mask2former",
            "model_name": name.replace("-semantic", "-coco-panoptic"),
        }

    # SMP модели
    for encoder in config["models"]["unet"]["encoders"]:
        result["semantic"][f"unet_{encoder}"] = {
            "model_type": "unet_smp",
            "encoder_name": encoder,
        }

    return result


def to_base64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def analyze_image_data(img_array: np.ndarray) -> dict:
    """Возвращает данные для визуализации анализа"""
    # Гистограмма интенсивностей
    hist, bins = np.histogram(img_array.flatten(), bins=64, range=(0, 256))

    # Простая детекция границ (Sobel)
    from scipy import ndimage

    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    sobel_x = ndimage.sobel(gray, axis=0)
    sobel_y = ndimage.sobel(gray, axis=1)
    edges = np.hypot(sobel_x, sobel_y)
    edges_norm = (edges / edges.max() * 255).astype(np.uint8)

    return {
        "histogram": hist.tolist(),
        "hist_bins": bins.tolist(),
        "edge_density": float(np.mean(edges > edges.max() * 0.3)),
        "edges_preview": to_base64(edges_norm),  # reuse to_base64
    }


def sanitize_metrics(metrics: dict) -> dict:
    """Заменяет inf/NaN на None для JSON-совместимости"""
    sanitized = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            if math.isinf(value) or math.isnan(value):
                sanitized[key] = None
            else:
                sanitized[key] = value
        else:
            sanitized[key] = value
    return sanitized


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
    try:
        # Чтение изображения
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)

        if mode == "neural":
            from segmenters.NeuralSegmenter import NeuralSegmenter
            from utils.strategies import segment_image_unified
            from utils.palettes import ade_palette, coco_palette, cityscapes_palette

            PALETTES = {
                "semantic": ade_palette,  # ADE20K: 150 классов
                "instance": coco_palette,  # COCO: 80 классов
                "panoptic": cityscapes_palette,  # Cityscapes: 19 классов
            }

            # Выбор имён классов
            CLASS_NAMES = {
                "semantic": NeuralSegmenter.get_ade_class_names,
                "instance": NeuralSegmenter.get_coco_class_names,
                "panoptic": NeuralSegmenter.get_cityscapes_class_names,
            }

            # NEURAL_CONFIGS = build_neural_configs()
            NEURAL_CONFIGS = {
                "semantic": {
                    # === SegFormer variants ===
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
                    # === Mask2Former semantic ===
                    "mask2former_swin_base": {
                        "model_type": "mask2former",
                        "model_name": "facebook/mask2former-swin-base-ade-semantic",
                    },
                    "mask2former_swin_large": {
                        "model_type": "mask2former",
                        "model_name": "facebook/mask2former-swin-large-ade-semantic",
                    },
                    # === OneFormer ===
                    "oneformer_swin_large": {
                        "model_type": "oneformer",
                        "model_name": "shi-labs/oneformer_ade20k_swin_large",
                    },
                    # === DPT ===
                    "dpt_large": {
                        "model_type": "dpt",
                        "model_name": "Intel/dpt-large-ade",
                    },
                    # === UPerNet ===
                    "upernet_convnext_small": {
                        "model_type": "upernet",
                        "model_name": "openmmlab/upernet-convnext-small",
                    },
                    # === SMP U-Net ===
                    "unet_resnet34": {
                        "model_type": "unet_smp",
                        "encoder_name": "resnet34",
                    },
                    "unet_resnet50": {
                        "model_type": "unet_smp",
                        "encoder_name": "resnet50",
                    },
                    "unet_efficientnet_b0": {
                        "model_type": "unet_smp",
                        "encoder_name": "efficientnet-b0",
                    },
                    "unet_mit_b5": {"model_type": "unet_smp", "encoder_name": "mit_b5"},
                    # === SMP FPN ===
                    "fpn_mit_b5": {"model_type": "fpn_smp", "encoder_name": "mit_b5"},
                    "fpn_efficientnet": {
                        "model_type": "fpn_smp",
                        "encoder_name": "efficientnet-b5",
                    },
                    # === SMP PSPNet ===
                    "psp_mit_b5": {
                        "model_type": "pspnet_smp",
                        "encoder_name": "mit_b5",
                    },
                    "psp_resnet50": {
                        "model_type": "pspnet_smp",
                        "encoder_name": "resnet50",
                    },
                    # === DeepLabV3+ ===
                    "deeplab_resnet101": {
                        "model_type": "deeplab_tv",
                        "encoder_name": "resnet101",
                    },
                    # === Torchvision FCN ===
                    "fcn_resnet50": {"model_type": "fcn_tv", "variant": "fcn_resnet50"},
                    "fcn_resnet101": {
                        "model_type": "fcn_tv",
                        "variant": "fcn_resnet101",
                    },
                    # === SegNet ===
                    "segnet_resnet34": {
                        "model_type": "segnet",
                        "encoder_name": "resnet34",
                        "checkpoint_path": None,  # Можно указать путь к чекпоинту
                    },
                    "segnet_resnet50": {
                        "model_type": "segnet",
                        "encoder_name": "resnet50",
                        "checkpoint_path": None,
                    },
                    # === SAM (семантическая через instance→semantic конверсию) ===
                    "mobile_sam": {
                        "model_type": "sam",
                        "model_name": "mobile_sam.pt",  # Путь к локальному файлу
                    },
                    "sam2_tiny": {"model_type": "sam", "model_name": "sam2_t.pt"},
                },
                "instance": {
                    # === Mask2Former instance ===
                    "mask2former_coco_instance": {
                        "model_type": "mask2former",
                        "model_name": "facebook/mask2former-swin-base-coco-instance",
                    },
                    # === MaskFormer ===
                    "maskformer_resnet50": {
                        "model_type": "maskformer",
                        "model_name": "facebook/maskformer-resnet50-ade20k-full",
                    },
                    # === YOLOv8 segmentation ===
                    "yolov8n_seg": {
                        "model_type": "yolov8",
                        "model_name": "yolov8n-seg.pt",
                    },
                    "yolov8s_seg": {
                        "model_type": "yolov8",
                        "model_name": "yolov8s-seg.pt",
                    },
                    "yolov8m_seg": {
                        "model_type": "yolov8",
                        "model_name": "yolov8m-seg.pt",
                    },
                    # === Mask R-CNN ===
                    "maskrcnn_resnet50": {
                        "model_type": "maskrcnn_tv",
                        "variant": "maskrcnn_resnet50_fpn",
                    },
                    "maskrcnn_resnet50_v2": {
                        "model_type": "maskrcnn_tv",
                        "variant": "maskrcnn_resnet50_fpn_v2",
                    },
                    # === SAM для инстанс-сегментации ===
                    "mobile_sam": {"model_type": "sam", "model_name": "mobile_sam.pt"},
                    "sam2_tiny": {"model_type": "sam", "model_name": "sam2_t.pt"},
                },
                "panoptic": {
                    # === Mask2Former panoptic ===
                    "mask2former_ade_panoptic": {
                        "model_type": "mask2former",
                        "model_name": "facebook/mask2former-swin-base-ade-panoptic",
                    },
                    "mask2former_coco_panoptic": {
                        "model_type": "mask2former",
                        "model_name": "facebook/mask2former-swin-base-coco-panoptic",
                    },
                    # === OneFormer panoptic ===
                    "oneformer_coco_panoptic": {
                        "model_type": "oneformer",
                        "model_name": "shi-labs/oneformer_coco_swin_large",
                    },
                },
            }

            config = NEURAL_CONFIGS.get(task, {}).get(model)
            if not config:
                raise HTTPException(400, f"Unknown neural config: {task}/{model}")

            # Инициализация нейросегментера
            neural_seg = NeuralSegmenter(
                **config, device="cuda" if torch.cuda.is_available() else "cpu"
            )

            # Инференс
            overlay_pil, result_info = segment_image_unified(
                model=neural_seg.model,
                processor=neural_seg.processor,
                image_input=image,  # PIL.Image
                model_type=config["model_type"],
                alpha=0.6,  # Прозрачность наложения
                palette=PALETTES[task],
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=False,
                num_classes=neural_seg.num_classes,
                class_names=CLASS_NAMES[task](),
                gt_mask=None,  # GT передаётся отдельно ниже
            )

            mask = result_info.get("mask")
            if mask is None:
                # Fallback: если маска не вернулась, создаём из overlay
                overlay_np = np.array(overlay_pil)
                mask = (overlay_np[:, :, 0] > 0).astype(np.uint8) * 255

            overlay_np = np.array(overlay_pil)

            metadata = {
                "method": model,
                "library": "neural",
                "task": task,
                "parameters": config,
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
                user_params = json.loads(custom_params)
            except:
                user_params = {}

            # Сегментация
            if auto_select:
                # Автовыбор
                mask, metadata = auto_seg.segment(
                    img_array, auto_select=True, library=library, return_metadata=True
                )
            else:
                # Ручной выбор — валидация
                if not method:
                    raise HTTPException(
                        400, "method_name required when auto_select=False"
                    )

                # Проверка существования метода в выбранной библиотеке
                if library not in METHODS_BY_LIBRARY:
                    raise HTTPException(400, f"Unknown library: {library}")

                if method not in METHODS_BY_LIBRARY[library]:
                    available = list(METHODS_BY_LIBRARY[library].keys())
                    raise HTTPException(
                        400,
                        f"Method '{method}' not found in library '{library}'. Available: {available}",
                    )

                # Получаем параметры из профиля
                profile = METHODS_BY_LIBRARY[library][method]
                default_params = auto_seg.available_methods.get(method, {}).get(
                    "params", {}
                )
                final_params = {**default_params, **user_params}
                print(f"🛠 Using params for {method}: {final_params}")

                segmenter_class = auto_seg._get_segmenter_class(method, library)
                segmenter = segmenter_class(method=method, **final_params)

                # Запускаем сегментацию с указанным методом
                # mask, metadata = auto_seg.segment(
                #     img_array,
                #     auto_select=False,
                #     method_name=method,
                #     library=library,
                #     return_metadata=True,
                # )
                result_img, mask = segmenter.segment_with_mask(img_array)

                metadata = {
                    "method": method,
                    "library": library,
                    "parameters": final_params,  # Возвращаем в UI, чтобы юзер видел, чем считали
                    "confidence": 1.0,
                    "image_characteristics": auto_seg.analyze_image(img_array),
                }
                # Добавляем информацию о библиотеке в метаданные
                metadata["library"] = profile.library

        metrics = {}
        if gt_mask:
            print(f"✅ GT получен: {gt_mask.filename}")
            gt_contents = await gt_mask.read()
            gt_image = Image.open(io.BytesIO(gt_contents)).convert("L")
            gt_array = np.array(gt_image)
            metrics = SegmentationMetrics.calculate_all_metrics(
                mask, gt_array, threshold=0.5
            )
        else:
            print("⚠️ GT не предоставлен, метрики не рассчитываются")

        recommendations = auto_seg.get_recommendations(img_array, top_k=5)

        # Сохранение результата
        analysis_data = analyze_image_data(img_array)

        if mode == "neural":
            # Для нейронных моделей используем цветной оверлей из segment_image_unified
            # overlay_np уже создан выше: overlay_np = np.array(overlay_pil)
            pass
        else:
            if len(img_array.shape) == 2:
                img_rgb = np.stack([img_array] * 3, axis=-1)
            else:
                img_rgb = img_array.copy()

            mask_colored = np.zeros_like(img_rgb)
            mask_colored[mask > 0] = [255, 0, 0]  # Красный для объекта
            overlay_np = (img_rgb * 0.6 + mask_colored * 0.4).astype(np.uint8)

        # 🔹 Конвертация в base64 (универсальная функция)
        def arr_to_b64(arr: np.ndarray) -> str:
            if arr.dtype != np.uint8:
                arr = (
                    (arr * 255).astype(np.uint8)
                    if arr.max() <= 1.0
                    else arr.astype(np.uint8)
                )
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        if metrics:
            metrics = sanitize_metrics(metrics)

        return {
            "success": True,
            "method": metadata["method"],
            "confidence": float(metadata["confidence"]),
            "mask_b64": f"data:image/png;base64,{arr_to_b64(mask)}",
            "overlay_b64": f"data:image/png;base64,{arr_to_b64(overlay_np)}",
            "chars": {
                "type": metadata["image_characteristics"].estimated_type.value,
                "size": f"{metadata['image_characteristics'].width}×{metadata['image_characteristics'].height}",
                "contrast": float(metadata["image_characteristics"].contrast),
                "noise": float(metadata["image_characteristics"].noise_level),
                "channels": metadata["image_characteristics"].channels,
                "mean_intensity": float(
                    metadata["image_characteristics"].mean_intensity
                ),
                "edge_density": float(metadata["image_characteristics"].edge_density),
                "complexity": float(metadata["image_characteristics"].complexity_score),
            },
            "metrics": metrics if metrics else None,  # Только если был GT
            "recommendations": [
                {
                    "method": r["method"],
                    "score": float(r["score"]),
                    "estimated_time_ms": float(r.get("estimated_time_ms", 0)),
                    "estimated_iou": float(r.get("estimated_iou", 0)),
                    "best_for": r.get("best_for", []),
                }
                for r in recommendations
            ],
            "analysis": {
                "histogram": analysis_data["histogram"],
                "edge_density": analysis_data["edge_density"],
                "edges_b64": f"data:image/png;base64,{analysis_data['edges_preview']}",
            },
            "examples": {
                "medical": ["otsu", "sauvola", "adaptive"],
                "documents": ["otsu", "adaptive", "bernson"],
                "nature": ["canny", "sobel", "watershed"],
                "industrial": ["adaptive", "bernson", "nisengard"],
            },
        }
    except Exception as e:
        import traceback

        print(f"❌ Ошибка в /api/segment: {e}")
        traceback.print_exc()
        raise HTTPException(500, str(e))


def params_to_schema(params: Dict[str, Any]) -> Dict[str, Any]:
    """Авто-генерация простой схемы из параметров"""
    schema = {}
    for key, value in params.items():
        if isinstance(value, bool):
            schema[key] = {"type": "boolean", "default": value}
        elif isinstance(value, int):
            # Эвристика: если имя содержит "size"/"bin"/"iter" — большой диапазон
            if any(k in key for k in ["size", "bin", "iter", "scale", "radius"]):
                schema[key] = {
                    "type": "int",
                    "min": 1,
                    "max": 500,
                    "step": 1,
                    "default": value,
                }
            else:
                schema[key] = {
                    "type": "int",
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "default": value,
                }
        elif isinstance(value, float):
            # Эвристика: если значение < 2 — вероятно, нормализованный параметр [0,1]
            if abs(value) <= 1.0 or "threshold" in key or "k" in key:
                schema[key] = {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "default": value,
                }
            else:
                schema[key] = {
                    "type": "float",
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "default": value,
                }
        else:
            schema[key] = {"type": "string", "default": str(value)}
    return schema


@app.get("/api/methods")
async def get_methods_by_library(library: Optional[str] = None):
    if library and library in METHODS_BY_LIBRARY:
        source_dict = METHODS_BY_LIBRARY.get(library, {})
    else:
        # Если библиотека не выбрана, берем дефолтные конфиги (где лежат schema)
        # В вашем коде это auto_seg.available_methods
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
                # params — это Dict[str, Any] в dataclass
                "defaults": profile.params if profile.params else {},
                # schema можно сформировать динамически или задать в profile.params
                # "schema": profile.params.get("schema", {}) if profile.params else {},
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
    if library not in METHODS_BY_LIBRARY:
        raise HTTPException(
            400,
            f"Unknown library: {library}. Available: {list(METHODS_BY_LIBRARY.keys())}",
        )
    methods = auto_seg.get_available_methods(library)
    return {
        "library": library,
        "methods": {
            name: {
                "name": profile.name,
                "library": profile.library,
                "avg_iou": profile.avg_iou,
                "avg_time_ms": profile.avg_time_ms,
                "best_for_type": [t.value for t in profile.best_for_type],
                "robustness": profile.robustness,
                "description": profile.description,
            }
            for name, profile in methods.items()
        },
    }


@app.get("/recommendations/")
async def get_recommendations(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    img_array = np.array(image)

    recs = auto_seg.get_recommendations(img_array, top_k=5)
    return {"recommendations": recs}


if os.path.exists("../frontend/dist"):
    app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
