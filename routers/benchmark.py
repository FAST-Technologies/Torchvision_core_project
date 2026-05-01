# routers/benchmark.py

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
import os
import sys
import uuid
import asyncio
import json
import time
import gc
import base64
import logging
from typing import (
    Dict,
    Any,
    Optional,
    Union,
    List,
    Callable,
    TypedDict,
    Tuple,
    cast,
)
from pathlib import Path
from pydantic import BaseModel

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, BackgroundTasks, Form, File, UploadFile
from fastapi.responses import JSONResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import settings
from utils.paths import ensure_dirs, ADE20K_DIR, MODELS_DIR, PROJECT_ROOT, DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

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
PathLike = Union[str, Path]


# 🔹 TypedDict для структуры задачи бенчмарка
class BenchmarkTask(TypedDict, total=False):
    status: str  # "running", "completed", "failed", "cancelled"
    progress: float  # 0-100
    message: str
    results: Optional[Dict[str, Any]]
    error_details: Optional[Dict[str, Any]]


# 🔹 TypedDict для конфигурации бенчмарка (вместо Dict[str, Any])
class BenchmarkConfig(TypedDict, total=False):
    inference: Dict[str, Any]
    filters: Dict[str, Any]
    visualization: Dict[str, Any]
    models_to_run: Optional[List[str]]


# 🔹 Type alias для хранилища задач
BenchmarkTaskDict = Dict[str, BenchmarkTask]
ModelLoadStep = Tuple[str, Callable[..., Any], Dict[str, Any]]

# Простое хранилище задач (в продакшене замените на Redis/Celery)
benchmark_tasks: BenchmarkTaskDict = {}


class NumpyEncoder(json.JSONEncoder):
    """Кастомный JSON-энкодер для numpy-типов и специальных float-значений"""

    def default(self, obj: Any) -> Any:
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
    """Возвращает JSONResponse с безопасной сериализацией."""
    return JSONResponse(
        content=content,
        status_code=status_code,
        media_type="application/json",
    )


# ──────────────────────────────────────────────────────────────────────
def img_to_b64(path: PathLike) -> str:
    """Конвертирует файл изображения в base64-строку."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ──────────────────────────────────────────────────────────────────────
class BenchmarkStartRequest(BaseModel):
    """Запрос на запуск бенчмарка."""

    use_default_image: bool = True
    image_path: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────
@router.get("/health")
async def benchmark_health() -> Dict[str, Any]:
    """Возвращает статус GPU, VRAM и активных задач бенчмарка."""
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
        "active_tasks": len(
            [t for t in benchmark_tasks.values() if t["status"] == "running"]
        ),
    }


# ──────────────────────────────────────────────────────────────────────
@router.get("/api/benchmark/debug")
async def debug_benchmark() -> Dict[str, Any]:
    """Отладочный эндпоинт: статус всех задач бенчмарка."""
    return {
        "active_tasks": len(benchmark_tasks),
        "tasks": {
            k: {"status": v["status"], "progress": v["progress"]}
            for k, v in benchmark_tasks.items()
        },
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
    """
    Асинхронная задача бенчмарка: загрузка моделей, инференс, визуализация, сохранение.

    Обновляет `benchmark_tasks[task_id]` с прогрессом и результатами.
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
        from testing.SegmentationBenchmark import SegmentationBenchmark
        import numpy as np
        from PIL import Image

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
        warmup_runs: int = int(inference_params.get("warmup_runs", 2))

        filters: Dict[str, Any] = config_dict.get("filters", {})
        min_iou: float = float(filters.get("min_iou", 0.0))
        only_passed: bool = bool(filters.get("only_passed", False))

        viz_params: Dict[str, Any] = config_dict.get("visualization", {})
        show_overlay: bool = bool(viz_params.get("show_overlay", True))
        show_gt: bool = bool(viz_params.get("show_gt", True))
        palette_name: str = str(viz_params.get("color_palette", "ade"))

        from utils.palettes import ade_palette, coco_palette, cityscapes_palette

        PALETTES: Dict[str, Callable[[], List[List[int]]]] = {
            "ade": ade_palette,
            "coco": coco_palette,
            "cityscapes": cityscapes_palette,
        }
        palette_func: Callable[[], List[List[int]]] = PALETTES.get(
            palette_name, ade_palette
        )
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
            from huggingface_hub import hf_hub_download

            repo_id: str = "hf-internal-testing/fixtures_ade20k"
            image_path: str = hf_hub_download(
                repo_id=repo_id, filename="ADE_val_00000001.jpg", repo_type="dataset"
            )
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
                    "checkpoint_path": str(
                        settings.MODEL_DIR / settings.UNET_CHECKPOINT
                    ),
                    "encoder_name": "resnet34",
                },
            ),
            (
                "deeplab_pretrained",
                bench.load_deeplab_trained,
                {
                    "checkpoint_path": str(
                        settings.MODEL_DIR / settings.DEEPLAB_CHECKPOINT
                    ),
                },
            ),
            (
                "fpn_mit_b5_pretrained",
                bench.load_fpn_mit_pretrained,
                {
                    "variant": "b5",
                    "checkpoint_path": str(
                        settings.MODEL_DIR / settings.FPN_MIT_CHECKPOINT
                    ),
                },
            ),
            (
                "psp_mit_b5_pretrained",
                bench.load_psp_mit_pretrained,
                {
                    "variant": "b5",
                    "checkpoint_path": str(
                        settings.MODEL_DIR / settings.PSP_MIT_CHECKPOINT
                    ),
                },
            ),
            (
                "fcn_resnet50_pretrained",
                bench.load_fcn_resnet50_pretrained,
                {
                    "variant": "fcn_resnet50",
                    "checkpoint_path": str(
                        settings.MODEL_DIR / settings.FCN_RESNET50_CHECKPOINT
                    ),
                },
            ),
            (
                "segnet_resnet34_pretrained",
                bench.load_segnet_pretrained,
                {
                    "encoder_name": "resnet34",
                    "checkpoint_path": str(
                        settings.MODEL_DIR / settings.SEGNET_RESNET34_CHECKPOINT
                    ),
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
            model_load_steps = [
                step for step in model_load_steps if step[0] in models_to_run
            ]

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

        benchmark_tasks[task_id][
            "message"
        ] = f"✅ {key}: готово. Все модели загружены. Запуск инференса..."
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
                if isinstance(val, (float, np.floating)) and (
                    np.isnan(val) or np.isinf(val)
                ):
                    metrics[key] = None
        benchmark_tasks[task_id].update(
            {
                "status": "completed",
                "progress": 100,
                "message": "Готово",
                "results": {
                    "summary": summary,
                    "output_dir": out_dir,
                    "charts": {
                        "metrics_plot_b64": img_to_b64(f"{out_dir}/plot_all.png")
                    },
                },
            }
        )

    except Exception as e:
        import traceback

        benchmark_tasks[task_id].update(
            {
                "status": "failed",
                "message": str(e),
                "error_details": {
                    "error_type": type(e).__name__,
                    "failed_at": benchmark_tasks[task_id]["message"],
                    "traceback": (
                        traceback.format_exc()
                        if logger.level == logging.DEBUG
                        else None
                    ),
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
    bg: Optional[BackgroundTasks] = None,
) -> Dict[str, str]:
    """Запускает асинхронный бенчмарк."""
    config_dict: Optional[BenchmarkConfig] = None
    if config:
        try:
            config_dict = cast(BenchmarkConfig, json.loads(config))
        except json.JSONDecodeError:
            raise HTTPException(422, detail="Invalid config JSON")
    config_for_task: Optional[Dict[str, Any]] = (
        dict(config_dict) if config_dict else None
    )
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
    """Возвращает статус и прогресс задачи бенчмарка."""
    task: Optional[BenchmarkTask] = benchmark_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    logger.info(f"📡 GET /status/{task_id} -> {task.get('progress')}%")
    if task_id not in benchmark_tasks or not task:
        raise HTTPException(status_code=404, detail="Task not found")
    logger.debug(
        f"📡 Status requested for {task_id}: {task['status']} / {task['progress']}%"
    )
    results = task.get("results")
    if results is not None and isinstance(results, dict) and "summary" in results:
        summary = results["summary"]
        for _, metrics in summary.items():
            for key, value in metrics.items():
                if isinstance(value, (float, np.floating)) and (
                    np.isnan(value) or np.isinf(value)
                ):
                    metrics[key] = None

    return safe_json_response(task)


# ──────────────────────────────────────────────────────────────────────
@router.get("/debug/{task_id}")
async def debug_task(task_id: str) -> Dict[str, Any]:
    """Отладочная информация о задаче."""
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
    """Отменяет или удаляет задачу бенчмарка."""
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
