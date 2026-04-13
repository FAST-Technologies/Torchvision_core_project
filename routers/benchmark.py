# routers/benchmark.py
import os
import sys
import uuid
import asyncio
import uuid
from typing import Dict, Any, Optional
from fastapi import (
    APIRouter,
    HTTPException,
    BackgroundTasks,
    Request,
    Form,
    File,
    UploadFile,
)
from fastapi.responses import FileResponse
import time
from pydantic import BaseModel
from typing import Dict
import torch
import gc
import base64
import json

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import settings

router = APIRouter(
    prefix="/api/benchmark",
    tags=["benchmark"],
    responses={
        404: {"description": "Task not found"},
        500: {"description": "Internal server error"},
    },
)

# Простое хранилище задач (в продакшене замените на Redis/Celery)
benchmark_tasks: Dict[str, Dict[str, Any]] = {}


def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


class BenchmarkStartRequest(BaseModel):
    use_default_image: bool = True
    image_path: Optional[str] = None


@router.get("/health")
async def benchmark_health():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "vram_mb": (
            torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0
        ),
        "active_tasks": len(
            [t for t in benchmark_tasks.values() if t["status"] == "running"]
        ),
    }


@router.get("/api/benchmark/debug")
async def debug_benchmark():
    return {
        "active_tasks": len(benchmark_tasks),
        "tasks": {
            k: {"status": v["status"], "progress": v["progress"]}
            for k, v in benchmark_tasks.items()
        },
    }


async def run_benchmark(
    task_id: str,
    req: BenchmarkStartRequest,
    image_file: Optional[UploadFile] = None,
    gt_file: Optional[UploadFile] = None,
    config: Optional[Dict] = None,
):
    benchmark_tasks[task_id] = {
        "status": "running",
        "progress": 0,
        "message": "Инициализация...",
        "results": None,
    }
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb < 20:  # менее 20 ГБ
            logger.warning(f"⚠️ Low VRAM: {vram_gb:.1f} GB. Benchmark may fail.")
    try:
        from testing.SegmentationBenchmark import SegmentationBenchmark
        import numpy as np
        from PIL import Image

        inference_params = config.get("inference", {})
        alpha = inference_params.get("alpha", 0.6)
        warmup_runs = inference_params.get("warmup_runs", 2)

        filters = config.get("filters", {})
        min_iou = filters.get("min_iou", 0.0)
        only_passed = filters.get("only_passed", False)

        viz_params = config.get("visualization", {})
        show_overlay = viz_params.get("show_overlay", True)
        show_gt = viz_params.get("show_gt", True)
        palette_name = viz_params.get("color_palette", "ade")

        # Выбор палитры
        from utils.palettes import ade_palette, coco_palette, cityscapes_palette

        PALETTES = {
            "ade": ade_palette,
            "coco": coco_palette,
            "cityscapes": cityscapes_palette,
        }
        palette = PALETTES.get(palette_name, ade_palette)

        benchmark_tasks[task_id]["progress"] = 5
        benchmark_tasks[task_id]["message"] = "Загрузка бенчмарка..."

        if image_file:
            image_input = Image.open(image_file.file).convert("RGB")
        elif req.image_path and os.path.exists(req.image_path):
            image_input = Image.open(req.image_path).convert("RGB")
        else:
            # Дефолтное изображение из ADE20K
            from huggingface_hub import hf_hub_download

            repo_id = "hf-internal-testing/fixtures_ade20k"
            image_path = hf_hub_download(
                repo_id=repo_id, filename="ADE_val_00000001.jpg", repo_type="dataset"
            )
            image_input = Image.open(image_path).convert("RGB")

        # 🔹 Обработка GT-маски
        gt_mask = None
        if gt_file:
            gt_mask = np.array(Image.open(gt_file.file).convert("L"))
        elif req.use_default_image:
            gt_path = "./data/ade20k_test_trained/original_image_mask_0.png"
            if os.path.exists(gt_path):
                gt_mask = np.array(Image.open(gt_path).convert("L"))

        bench = SegmentationBenchmark(
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_classes=150,
            gt_mask=gt_mask,
            palette=palette,
        )

        models_to_run = config.get("models_to_run") if config else None

        # 2. Загрузка моделей (вызывайте только те, что у вас есть в models/)
        model_load_steps = [
            (
                "segformer",
                bench.load_segformer,
                {"path": settings.SEGFORMER_PATH},
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
                    "checkpoint_path": os.path.join(
                        settings.MODEL_DIR, settings.UNET_CHECKPOINT
                    ),
                    "encoder_name": "resnet34",
                },
            ),
            (
                "deeplab_pretrained",
                bench.load_deeplab_trained,
                {
                    "checkpoint_path": os.path.join(
                        settings.MODEL_DIR, settings.DEEPLAB_CHECKPOINT
                    )
                },
            ),
            (
                "fpn_mit_b5_pretrained",
                bench.load_fpn_mit_pretrained,
                {
                    "variant": "b5",
                    "checkpoint_path": os.path.join(
                        settings.MODEL_DIR, settings.FPN_MIT_CHECKPOINT
                    ),
                },
            ),
            (
                "psp_mit_b5_pretrained",
                bench.load_psp_mit_pretrained,
                {
                    "variant": "b5",
                    "checkpoint_path": os.path.join(
                        settings.MODEL_DIR, settings.PSP_MIT_CHECKPOINT
                    ),
                },
            ),
            (
                "fcn_resnet50_pretrained",
                bench.load_fcn_resnet50_pretrained,
                {
                    "variant": "fcn_resnet50",
                    "checkpoint_path": os.path.join(
                        settings.MODEL_DIR, settings.FCN_RESNET50_CHECKPOINT
                    ),
                },
            ),
            (
                "segnet_resnet34_pretrained",
                bench.load_segnet_pretrained,
                {
                    "encoder_name": "resnet34",
                    "checkpoint_path": os.path.join(
                        settings.MODEL_DIR, settings.SEGNET_RESNET34_CHECKPOINT
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

        if models_to_run:
            model_load_steps = [
                step for step in model_load_steps if step[0] in models_to_run
            ]

        for i, (key, load_fn, kwargs) in enumerate(model_load_steps):
            benchmark_tasks[task_id]["progress"] = 5 + (i / len(model_load_steps)) * 70
            benchmark_tasks[task_id]["message"] = f"🔄 {key}: загрузка..."
            try:
                if "checkpoint_path" in kwargs:
                    if not os.path.exists(kwargs["checkpoint_path"]):
                        print(
                            f"⚠️ Checkpoint not found: {kwargs['checkpoint_path']}, skipping {key}"
                        )
                        continue
                load_fn(**kwargs)
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"⚠️ Не удалось загрузить {key}: {e}")

        benchmark_tasks[task_id]["message"] = f"✅ {key}: готово"
        # 3. Запуск сравнения
        benchmark_tasks[task_id]["message"] = "Запуск инференса..."
        await asyncio.to_thread(bench.compare, image_input=image_input, alpha=alpha)

        # 4. Сохранение результатов
        out_dir = f"./data/benchmark_{task_id}"
        await asyncio.to_thread(bench.save_results, out_dir)
        await asyncio.to_thread(bench.plot_all_metrics, path=f"{out_dir}/plot_all.png")

        benchmark_tasks[task_id]["status"] = "completed"
        benchmark_tasks[task_id]["progress"] = 100
        benchmark_tasks[task_id]["message"] = "Готово"
        benchmark_tasks[task_id]["results"] = {
            "summary": bench.get_summary(),
            "output_dir": out_dir,
            "charts": {
                "metrics_plot_b64": img_to_b64(f"{out_dir}/plot_all.png"),
            },
        }

    except Exception as e:
        import traceback

        benchmark_tasks[task_id]["status"] = "failed"
        benchmark_tasks[task_id]["message"] = str(e)
        benchmark_tasks[task_id]["error_details"] = {
            "error_type": type(e).__name__,
            "failed_at": benchmark_tasks[task_id]["message"],  # на каком этапе
            "traceback": (
                traceback.format_exc() if logger.level == logging.DEBUG else None
            ),
        }
        logger.error(f"Benchmark {task_id} failed: {e}", exc_info=True)


@router.post("/start")
async def start_benchmark(
    # req: BenchmarkStartRequest,
    image: Optional[UploadFile] = File(None),
    gt_mask: Optional[UploadFile] = File(None),
    use_default_image: bool = Form(True),
    image_path: Optional[str] = Form(None),
    config: Optional[str] = Form(None),
    bg: BackgroundTasks = None,
):
    config_dict = {}
    if config:
        try:
            config_dict = json.loads(config)
        except json.JSONDecodeError:
            raise HTTPException(422, detail="Invalid config JSON")
    req = BenchmarkStartRequest(
        use_default_image=use_default_image,
        image_path=image_path,
    )
    task_id = str(uuid.uuid4())
    bg.add_task(run_benchmark, task_id, req, image, gt_mask, config_dict)
    return {"task_id": task_id}


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in benchmark_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return benchmark_tasks[task_id]


@router.delete("/{task_id}")
async def cancel_benchmark(task_id: str):
    if task_id not in benchmark_tasks:
        raise HTTPException(404, detail="Task not found")

    task = benchmark_tasks[task_id]
    if task["status"] in ("completed", "failed"):
        raise HTTPException(400, detail="Task already finished")

    task["status"] = "cancelled"
    task["message"] = "Отменено пользователем"

    return {"status": "cancelled"}
