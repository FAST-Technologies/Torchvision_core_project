# routers/comparator.py

import os
import sys
import uuid
import asyncio
import json
import time
import traceback
import base64
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import torch
import gc
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import settings
from utils.paths import ensure_dirs, DATA_DIR
from testing.SegmentationComparator import SegmentationComparator
from segmenters.OpenCVSegmenter import OpenCVSegmenter
from segmenters.SklearnSegmenter import SklearnSegmenter
from segmenters.TorchSegmenter import TorchSegmenter
from segmenters.NewTorchSegmenter import TorchSegmenter2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comparator")

router = APIRouter(
    prefix="/api/comparator",
    tags=["comparator"],
    responses={
        404: {"description": "Task not found"},
        500: {"description": "Internal server error"},
    },
)

# 🔹 Хранилище задач
_comparator_tasks: Dict[str, Dict[str, Any]] = {}
_comparator_lock = asyncio.Lock()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def safe_json_response(content: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=content, status_code=status_code, media_type="application/json")


def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# 🔹 Конфигурация методов по умолчанию
DEFAULT_COMPARATOR_METHODS = {
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


def _extract_library_from_name(name: str) -> str:
    """Извлекает библиотеку из имени метода: Otsu_OpenCV -> opencv"""
    if name.endswith("_OpenCV"):
        return "opencv"
    elif name.endswith("_Sklearn"):
        return "sklearn"
    elif name.endswith("_Torch_v2"):  # 🔹 Поддержка суффикса v2
        return "torch_v2"
    elif name.endswith("_Torch"):
        return "torch"
    return "opencv"


def _create_segmenter(library: str, method: str, params: Dict[str, Any]):
    """Фабрика сегментеров"""
    if library == "opencv":
        return OpenCVSegmenter(method, **params)
    elif library == "sklearn":
        return SklearnSegmenter(method, **params)
    elif library == "torch":
        return TorchSegmenter(method, **params)
    elif library == "torch_v2":
        return TorchSegmenter2(method, **params)
    raise ValueError(f"Unknown library: {library}")


async def _run_comparator_task(
    task_id: str,
    image: np.ndarray,
    methods_config: List[Dict[str, Any]],
    reference_config: Dict[str, Any],
    comparison_type: str = "batch",
    output_dir: Optional[str] = None,
) -> None:
    """Асинхронная задача компаратора"""
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

        # 🔹 Подготовка методов
        segmenters = []
        for cfg in methods_config:
            try:
                library = cfg.get("library") or _extract_library_from_name(cfg["name"])
                method_name = cfg["method"]
                params = cfg.get("params", {})

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
        ref_name = reference_config["name"]

        # 🔹 Прогресс: 0-20% подготовка, 20-90% сравнение, 90-100% сохранение
        async with _comparator_lock:
            _comparator_tasks[task_id]["progress"] = 20
            _comparator_tasks[task_id]["message"] = f"Запущено {len(segmenters)} методов"

        # 🔹 Пакетное сравнение с пошаговым обновлением
        results = []
        for i, cfg in enumerate(segmenters):
            progress = 20 + (i / len(segmenters)) * 70
            async with _comparator_lock:
                _comparator_tasks[task_id]["progress"] = progress
                _comparator_tasks[task_id]["message"] = f"Сравнение {cfg['name']} ({i+1}/{len(segmenters)})"
            await asyncio.sleep(0)

            try:
                start = time.time()
                test_mask = cfg["segmenter"].segment(image)
                test_time = time.time() - start

                ref_mask = ref_seg.segment(image)
                ref_time = time.time() - start - test_time

                metrics = comparator.compute_metrics(ref_mask, test_mask, ref_name, cfg["name"])

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
                    out_path = os.path.join(output_dir, f"viz_{cfg['name']}.png")
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
        df = comparator.batch_comparison(
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
        summary = {
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
        }

        # 🔹 Сериализация графиков
        charts = {}
        for fname in [
            "comparison_summary.jpg",
            "f1_score_matrix.png",
            "accuracy_matrix.png",
        ]:
            fpath = os.path.join(output_dir, fname)
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


# 🔹 Роуты
@router.post("/start")
async def start_comparator(
    image: UploadFile = File(...),
    methods: str = Form(...),  # JSON string
    reference: str = Form(...),  # JSON string
    comparison_type: str = Form("batch"),
) -> Dict[str, str]:
    try:
        methods_config = json.loads(methods)
        reference_config = json.loads(reference)
    except json.JSONDecodeError as e:
        raise HTTPException(422, detail=f"Invalid JSON: {e}")
    from PIL import Image

    img = Image.open(image.file).convert("RGB")
    image_array = np.array(img)

    task_id = str(uuid.uuid4())
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


@router.get("/status/{task_id}")
async def get_status(task_id: str) -> JSONResponse:
    async with _comparator_lock:
        task = _comparator_tasks.get(task_id)
    if not task:
        raise HTTPException(404, detail="Task not found")

    if task.get("results") and "results" in task["results"]:
        for r in task["results"]["results"]:
            for k, v in r.items():
                if isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
                    r[k] = None

    return safe_json_response(task)


@router.delete("/{task_id}")
async def cancel_comparator(task_id: str) -> Dict[str, str]:
    async with _comparator_lock:
        task = _comparator_tasks.get(task_id)
    if not task:
        return {"status": "not_found", "message": "Task not found"}
    if task["status"] in ("completed", "failed", "cancelled"):
        return {"status": task["status"], "message": f"Already {task['status']}"}
    task["status"] = "cancelled"
    task["message"] = "Отменено пользователем"
    return {"status": "cancelled"}
