# backend/app.py
from typing import Optional
import json
import os, sys, base64, io
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
)
from metrics.SegmentationMetrics import SegmentationMetrics

app = FastAPI(title="AutoSegmenter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

auto_seg = AutoSegmenter()


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
    goal: str = Form("balanced"),
    auto_select: bool = Form(True),
    method: Optional[str] = Form(None),
    library: Optional[str] = Form("opencv"),
    gt_mask: Optional[UploadFile] = File(default=None),
):
    try:
        # Чтение изображения
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)

        auto_seg.goal = (
            SegmentationGoal(goal)
            if goal in ["speed", "accuracy", "balanced", "low_memory"]
            else SegmentationGoal.BALANCED
        )

        # Сегментация
        if auto_select:
            # Автовыбор
            mask, metadata = auto_seg.segment(
                img_array, auto_select=True, library=library, return_metadata=True
            )
        else:
            # Ручной выбор — валидация
            if not method:
                raise HTTPException(400, "method_name required when auto_select=False")

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
            params = auto_seg.available_methods.get(method, {}).get("params", {})

            # Запускаем сегментацию с указанным методом
            mask, metadata = auto_seg.segment(
                img_array,
                auto_select=False,
                method_name=method,
                library=library,
                return_metadata=True,
            )
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

        if len(img_array.shape) == 2:
            img_rgb = np.stack([img_array] * 3, axis=-1)
        else:
            img_rgb = img_array.copy()

        mask_colored = np.zeros_like(img_rgb)
        mask_colored[mask > 0] = [255, 0, 0]  # Красный для объекта
        overlay = (img_rgb * 0.6 + mask_colored * 0.4).astype(np.uint8)

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
            "overlay_b64": f"data:image/png;base64,{arr_to_b64(overlay)}",
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
