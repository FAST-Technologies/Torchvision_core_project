# routers/validate.py

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
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
from typing import Dict, Any, Optional, List, Tuple

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
from segmenters.OpenCVSegmenter import OpenCVSegmenter
from segmenters.SklearnSegmenter import SklearnSegmenter
from metrics.SegmentationMetrics import SegmentationMetrics, MetricsDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("validate")

router = APIRouter(prefix="/api/validate", tags=["validate"])

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & TASK STORAGE
# ──────────────────────────────────────────────────────────────────────
ValidationTaskDict = Dict[str, Dict[str, Any]]
_validation_tasks: ValidationTaskDict = {}
_validation_lock = asyncio.Lock()


# 🔹 Энкодер для NaN/Infinity
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
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
    return JSONResponse(
        content=content, status_code=status_code, media_type="application/json"
    )


# ──────────────────────────────────────────────────────────────────────
def arr_to_b64(arr: np.ndarray) -> str:
    """numpy → data:image/png;base64,..."""
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
    """Возвращает список методов по фильтру: threshold/edge/region/clustering/all."""
    mapping: Dict[str, List[MethodConfig]] = {
        "threshold": validator.threshold_methods,
        "edge": validator.edge_methods,
        "region": validator.region_methods,
        "clustering": validator.clastering_methods,
    }
    if methods_filter in mapping:
        return mapping[methods_filter]
    return (
        validator.threshold_methods
        + validator.edge_methods
        + validator.region_methods
        + validator.clastering_methods
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
    """Асинхронная задача валидации: кросс-библиотечное сравнение методов."""
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
        img_array: np.ndarray = np.array(
            Image.open(io.BytesIO(file_content)).convert("RGB")
        )
        validator = TorchImplementationValidator(output_dir="./data/validation_web")
        methods_list: List[Tuple[str, Dict[str, Any]]] = _get_methods_for_filter(
            validator, methods_filter
        )
        total_methods: int = len(methods_list)

        async with _validation_lock:
            _validation_tasks[task_id]["total_methods"] = total_methods
            _validation_tasks[task_id]["progress"] = 5
            _validation_tasks[task_id]["message"] = f"Запущено {total_methods} методов"

        CLASS_MAP: Dict[str, Any] = {
            "torch": TorchSegmenter,
            "opencv": OpenCVSegmenter,
            "sklearn": SklearnSegmenter,
        }
        primary_class = CLASS_MAP.get(primary_library, TorchSegmenter)
        reference_class = CLASS_MAP.get(reference_library, OpenCVSegmenter)

        results: Dict[str, Any] = {}

        # 🔹 Пошаговое выполнение с обновлением прогресса
        for idx, (method_name, params) in enumerate(methods_list):
            progress: float = 10 + (idx / total_methods) * 80
            async with _validation_lock:
                _validation_tasks[task_id]["progress"] = round(progress, 2)
                _validation_tasks[task_id][
                    "message"
                ] = f"Обработка {method_name} ({idx+1}/{total_methods})"
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

            elapsed_ms = round(
                (time.time() - _validation_tasks[task_id]["start_time"]) * 1000, 1
            )
            async with _validation_lock:
                _validation_tasks[task_id]["processed"] = idx + 1
                _validation_tasks[task_id]["progress"] = round(
                    10 + (idx + 1) / total_methods * 80, 2
                )
                _validation_tasks[task_id]["elapsed_ms"] = elapsed_ms
                _validation_tasks[task_id][
                    "message"
                ] = f"Завершён {method_name} ({idx+1}/{total_methods})"
            await asyncio.sleep(0)

        # 🔹 Финализация
        async with _validation_lock:
            _validation_tasks[task_id]["progress"] = 95
            _validation_tasks[task_id]["message"] = "Сохранение результатов..."

        # 🔹 Подготовка ответа
        summary: List[Dict[str, Any]] = []
        for method, data in results.items():
            if not isinstance(data, dict) or not data.get("success"):
                summary.append(
                    {"method": method, "success": False, "error": data.get("error")}
                )
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

        valid_times: List[Any] = [
            d["torch_time"] for d in benchmark_data if d.get("torch_time")
        ]
        valid_iou: List[Any] = [
            d["iou"] for d in benchmark_data if d.get("iou") is not None
        ]
        elapsed_ms = round(
            (time.time() - _validation_tasks[task_id]["start_time"]) * 1000, 1
        )

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
                        "passed": sum(
                            1 for s in summary if s.get("validation_status") == "PASS"
                        ),
                        "warning": sum(
                            1
                            for s in summary
                            if s.get("validation_status") == "WARNING"
                        ),
                        "failed": sum(
                            1 for s in summary if s.get("validation_status") == "FAIL"
                        ),
                        "methods_tested": len(summary),
                        "report_dir": "./data/validation_web",
                        "benchmark": {
                            "methods_count": len(benchmark_data),
                            "passed": sum(
                                1
                                for s in summary
                                if s.get("validation_status") == "PASS"
                            ),
                            "warning": sum(
                                1
                                for s in summary
                                if s.get("validation_status") == "WARNING"
                            ),
                            "failed": sum(
                                1
                                for s in summary
                                if s.get("validation_status") == "FAIL"
                            ),
                            "data": benchmark_data,
                            "avg_torch_time": (
                                sum(valid_times) / len(valid_times)
                                if valid_times
                                else 0
                            ),
                            "avg_iou": (
                                sum(valid_iou) / len(valid_iou) if valid_iou else 0
                            ),
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
            start: float = _validation_tasks[task_id].get(
                "start_time", time.perf_counter()
            )
            _validation_tasks[task_id].update(
                {
                    "status": "failed",
                    "message": str(e),
                    "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
                    "error_details": {
                        "error_type": type(e).__name__,
                        "failed_at": _validation_tasks[task_id]["message"],
                        "traceback": (
                            traceback.format_exc()
                            if logger.level == logging.DEBUG
                            else None
                        ),
                    },
                }
            )
        logger.error(f"❌ Validation task {task_id} failed: {e}")


# ──────────────────────────────────────────────────────────────────────
def _process_single_method(
    method_name: str,
    params: Dict[str, Any],
    img_array: np.ndarray,
    primary_class,
    reference_class,
    validator,
) -> Dict[str, Any]:
    """Синхронная обработка одного метода валидации."""
    logger.info(f"🔹 START Processing method: {method_name}")
    try:
        import cv2

        t1: float = time.perf_counter()
        seg1 = primary_class(method=method_name, **params)
        mask1: np.ndarray = seg1.segment(img_array, **params)
        time1: float = time.perf_counter() - t1
        logger.info(f"✅ {method_name} primary done: {time1:.3f}s")

        ref_params: Dict[str, Any] = params.copy()
        ref_params["postprocess"] = False
        t2: float = time.perf_counter()
        seg2 = reference_class(method=method_name, **ref_params)
        mask2: np.ndarray = seg2.segment(img_array, **ref_params)
        time2: float = time.perf_counter() - t2
        logger.info(f"✅ {method_name} primary done: {time2:.3f}s")

        metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(
            mask1, mask2, threshold=0.5
        )
        metrics.update(
            {
                "first_method_time": time1,
                "second_method_time": time2,
                "methods_time_difference": abs(time1 - time2),
            }
        )
        status = validator._check_validation_status(metrics)

        orig_gray = (
            cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            if img_array.ndim == 3
            else img_array
        )
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
        import traceback

        logger.error(f"❌ /api/validate error: {e}\n{traceback.format_exc()}")
        raise


# ──────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────
@router.post("/start")
async def start_validation(
    file: UploadFile = File(...),
    primary_library: str = Form("torch"),  # "torch" | "opencv" | "sklearn"
    reference_library: str = Form("opencv"),  # "torch" | "opencv" | "sklearn"
    methods_filter: Optional[str] = Form(
        None
    ),  # "threshold" | "edge" | "region" | "all"
) -> Dict[str, str]:
    """Запускает асинхронную валидацию кросс-библиотечных реализаций."""
    file_content = await file.read()
    task_id: str = str(uuid.uuid4())
    temp_validator = TorchImplementationValidator(output_dir="./data/validation_web")
    methods_list: List[Tuple[str, Dict[str, Any]]] = _get_methods_for_filter(
        temp_validator, methods_filter
    )
    total_methods: int = len(methods_list)
    logger.info(f"🔧 methods_filter={methods_filter}, total_methods={total_methods}")
    asyncio.create_task(
        _run_validation_task(
            task_id, file_content, primary_library, reference_library, methods_filter
        )
    )
    return {"task_id": task_id, "status": "running"}


# ──────────────────────────────────────────────────────────────────────
@router.get("/status/{task_id}")
async def get_validation_status(task_id: str) -> JSONResponse:
    """Возвращает статус и результаты валидации."""
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
                    summary.append(
                        {"method": method, "success": False, "error": data.get("error")}
                    )
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

            valid_times: List[Any] = [
                d["torch_time"] for d in benchmark_data if d.get("torch_time")
            ]
            valid_iou: List[Any] = [
                d["iou"] for d in benchmark_data if d.get("iou") is not None
            ]

            response.update(
                {
                    "results": summary,
                    "passed": sum(
                        1 for s in summary if s.get("validation_status") == "PASS"
                    ),
                    "warning": sum(
                        1 for s in summary if s.get("validation_status") == "WARNING"
                    ),
                    "failed": sum(
                        1 for s in summary if s.get("validation_status") == "FAIL"
                    ),
                    "methods_tested": len(summary),
                    "report_dir": "./data/validation_web",
                    "benchmark": {
                        "methods_count": len(benchmark_data),
                        "passed": sum(
                            1 for s in summary if s.get("validation_status") == "PASS"
                        ),
                        "warning": sum(
                            1
                            for s in summary
                            if s.get("validation_status") == "WARNING"
                        ),
                        "failed": sum(
                            1 for s in summary if s.get("validation_status") == "FAIL"
                        ),
                        "data": benchmark_data,
                        "avg_torch_time": (
                            sum(valid_times) / len(valid_times) if valid_times else 0
                        ),
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
                    if isinstance(v, (float, np.floating)) and (
                        np.isnan(v) or np.isinf(v)
                    ):
                        item["metrics"][k] = None

    logger.info(f"🔍 Returning response keys: {list(response.keys())}")
    if "results" in response:
        logger.info(f"🔍 response['results'] type: {type(response['results'])}")
        if isinstance(response["results"], list):
            logger.info(f"🔍 response['results'] length: {len(response['results'])}")
            if response["results"]:
                logger.info(
                    f"🔍 first result keys: {list(response['results'][0].keys())}"
                )
    return safe_json_response(response)


# ──────────────────────────────────────────────────────────────────────
@router.delete("/{task_id}")
async def cancel_validation(task_id: str) -> Dict[str, str]:
    """Отменяет задачу валидации."""
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
