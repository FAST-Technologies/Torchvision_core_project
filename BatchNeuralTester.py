# analyze_batch.py
"""
Модуль для анализа влияния аугментаций на качество нейросетевых моделей сегментации
на датасете ADE20K (или его подмножестве).

Workflow:
1. Загрузка списка изображений из датасета (локально или через HuggingFace).
2. Для каждой комбинации (модель × аугментация):
   - Загрузка чекпоинта через NeuralSegmenter.
   - Пакетное предсказание на подмножестве датасета.
   - Расчёт и агрегация метрик.
   - Очистка памяти.
3. Агрегация результатов в DataFrame.
4. Статистический анализ и визуализация.
5. Экспорт: CSV, JSON, Markdown, PNG-графики.

Example:
    ```bash
    python analyze_batch.py --dataset ./data/ADE20K --subset 50 --output ./results
    ```
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse
import glob
import json
import os
import gc
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, Literal, Set
from dataclasses import dataclass, field, asdict
from collections import OrderedDict, defaultdict

import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import zoom
from scipy import stats
from contextlib import contextmanager, nullcontext
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from segmenters.NewTorchSegmenter import PrecisionManager

# HuggingFace для загрузки датасета
from huggingface_hub import hf_hub_download, list_repo_files

# Локальные импорты
from segmenters.NeuralSegmenter import NeuralSegmenter
from metrics.SegmentationMetrics import SegmentationMetrics

import logging

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & CONSTANTS
# ──────────────────────────────────────────────────────────────────────
MaskArray = np.ndarray
ImageArray = np.ndarray
MetricValue = float
MetricsDict = Dict[str, MetricValue]
PathLike = Union[str, Path]

MODEL_TYPE_MAPPING: Dict[str, str] = {
    "unet_smp": "unet_smp",
    "fpn_smp": "fpn_smp",
    "psp_smp": "pspnet_smp",
    "deeplab_tv": "deeplab_tv",
    "fcn_tv": "fcn_tv",
    "segnet": "segnet",
}

# Стандартные метрики для агрегации
DEFAULT_METRICS: List[str] = [
    "iou",
    "dice",
    "f1_score",
    "precision",
    "recall",
    "accuracy",
    "mae",
    "hausdorff_distance",
]

from hashlib import sha256
import pickle


class PredictionCache:
    """Кэш предсказаний моделей для ускорения повторных запусков"""

    def __init__(self, cache_dir: PathLike, max_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_size_gb * 1e9)

    def _get_key(self, model_path: Path, image_path: Path, config_hash: str) -> str:
        """Генерация уникального ключа кэша"""
        content = f"{model_path.stat().st_mtime}:{image_path}:{config_hash}"
        return sha256(content.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[np.ndarray]:
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except:
                cache_file.unlink(missing_ok=True)
        return None

    def set(self, key: str, prediction: np.ndarray):
        cache_file = self.cache_dir / f"{key}.pkl"
        # Простая политика LRU: удаляем старые если превышен лимит
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        if total_size + prediction.nbytes > self.max_bytes:
            oldest = min(self.cache_dir.glob("*.pkl"), key=lambda f: f.stat().st_mtime)
            oldest.unlink()
        with open(cache_file, "wb") as f:
            pickle.dump(prediction, f)


# ──────────────────────────────────────────────────────────────────────
@dataclass
class TestConfig:
    """Конфигурация тестирования"""

    dataset_path: PathLike
    output_dir: PathLike = "./results/augmentation_analysis"
    subset_size: Optional[int] = 50  # None = весь датасет
    random_seed: int = 42
    batch_size: int = 1  # Для нейросетей обычно 1 из-за разного размера
    device: str = "cuda"
    num_classes: int = 150
    ignore_index: int = 255
    metrics: List[str] = field(default_factory=lambda: DEFAULT_METRICS.copy())
    save_overlays: bool = True
    overlay_cache_max: int = 50  # 🔥 Максимум оверлеев в памяти (исправляет утечку)
    overlay_sample_rate: float = 1.0  # Сохранять 100% визуализаций
    verbose: bool = True
    save_visualizations: bool = True  # Новый флаг
    viz_alpha: float = 0.4  # Прозрачность для оверлеев
    viz_color: Tuple[int, int, int] = (255, 0, 0)  # Цвет маски

    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"Unsupported device: {self.device}")


@dataclass
class ModelCheckpoint:
    """Информация о чекпоинте модели"""

    key: str
    path: Path
    model_type: str
    augmentation: str
    original_type: str

    @property
    def display_name(self) -> str:
        return f"{self.original_type}_{self.augmentation}"


@dataclass
class TestResult:
    """Результат тестирования одного изображения"""

    model_key: str
    image_name: str
    metrics: MetricsDict
    inference_time: float
    precision: str = "fp32"
    pred_mask: Optional[MaskArray] = None
    gt_mask: Optional[MaskArray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "image_name": self.image_name,
            **self.metrics,
            "inference_time": self.inference_time,
        }


@contextmanager
def safe_inference_context(model_name: str, image_name: str):
    """Контекст для отлова ошибок инференса с детальной информацией"""
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"OOM при {model_name}/{image_name}: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        raise
    except Exception as e:
        logger.error(
            f"Ошибка {model_name}/{image_name}: {type(e).__name__}: {e}", exc_info=True
        )
        raise


def _check_precision_support(
    model: Optional[nn.Module], dtype: torch.dtype, device: str
) -> bool:
    """Проверка поддержки точности на текущем устройстве"""
    if dtype == torch.bfloat16 and device == "cuda" and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        return cap[0] >= 8  # Ampere+
    if dtype == torch.float16 and device == "cpu":
        return False  # fp16 на CPU неэффективен
    return True


def extract_model_aug_from_key(key: str) -> Tuple[str, str]:
    """
    Извлекает имя модели и аугментацию из ключа оверлея.
    Формат ключа: "{модель}_{аугментация}_{имя_изображения}"
    Пример: "fpn_smp_none_ADE_val_00000001" → ("fpn_smp", "none")
    """
    parts = key.rsplit("_", 2)  # Разделяем последние 2 подчёркивания
    if len(parts) >= 2:
        return "_".join(parts[:-2]), parts[-2]  # model, aug
    return key, "unknown"


def save_augmentation_comparison_grid(
    overlay_images: Dict[str, Image.Image],
    output_dir: PathLike = "./test_run1/overlays",
    model_names: Optional[List[str]] = None,
) -> None:
    """
    Создаёт единую сетку сравнения всех моделей по уровням аугментаций.

    Макет:
    [Модель 1] [none] [basic] [medium]
    [Модель 2] [none] [basic] [medium]
    ...
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if model_names is None:
        models = list(
            set(extract_model_aug_from_key(k)[0] for k in overlay_images.keys())
        )
    else:
        models = model_names

    if not models:
        print("⚠️  Нет моделей для визуализации")
        return

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 3, figsize=(15, 5 * n_models), squeeze=False)

    for row, model in enumerate(models):
        for col, aug in enumerate(["none", "basic", "medium"]):
            # 🔥 Ищем ключи, содержащие модель и аугментацию (любое изображение)
            matching_keys = [
                k for k in overlay_images.keys() if k.startswith(f"{model}_{aug}_")
            ]
            ax = axes[row, col]

            if matching_keys and overlay_images[matching_keys[0]] is not None:
                ax.imshow(overlay_images[matching_keys[0]])
                ax.set_title(
                    f"{aug.upper()}", fontsize=10, fontweight="bold", color="darkblue"
                )
                ax.axis("off")
            else:
                ax.set_facecolor("#f5f5f5")
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                    color="gray",
                    fontweight="bold",
                )
                ax.set_title(
                    f"{aug.upper()}", fontsize=10, fontweight="bold", color="gray"
                )
                ax.axis("off")

        axes[row, 0].set_ylabel(
            model.upper(),
            rotation=0,
            labelpad=60,
            fontsize=11,
            fontweight="bold",
            ha="right",
            va="center",
        )

    plt.suptitle(
        "Сравнение влияния аугментаций на визуализацию сегментации",
        fontsize=14,
        y=1.01,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.98))

    grid_path = output_path / "full_comparison_grid.png"
    plt.savefig(
        grid_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()
    print(f"✅ Полная сетка сравнения: {grid_path}")


def save_model_augmentation_comparisons(
    overlay_images: Dict[str, Image.Image],
    output_dir: PathLike = "./test_run1/overlays",
    models: Optional[List[str]] = None,
) -> None:
    """
    Сохраняет отдельные сравнения для каждой модели: [none] [basic] [medium]
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = list(
            set(extract_model_aug_from_key(k)[0] for k in overlay_images.keys())
        )

    print(f"\n🖼️ Сохранение сравнения визуализаций ({len(overlay_images)} оверлеев)...")
    print(f"   Найдено моделей: {models}")

    for model in models:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"{model}: сравнение аугментаций", fontsize=14, fontweight="bold")

        for idx, aug in enumerate(["none", "basic", "medium"]):
            # 🔥 Ищем первый подходящий ключ
            matching_keys = [
                k for k in overlay_images.keys() if k.startswith(f"{model}_{aug}_")
            ]
            ax = axes[idx]

            if matching_keys:
                key = matching_keys[0]  # Берём первый найденный
                if overlay_images.get(key) is not None:
                    ax.imshow(overlay_images[key])
                    ax.set_title(f"{aug.upper()}", fontsize=11)
                    ax.axis("off")
                    print(f"   ✅ {model}_{aug}: {key}")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Empty",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=12,
                    )
                    ax.set_title(f"{aug.upper()}", fontsize=11)
                    ax.axis("off")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_title(f"{aug.upper()}", fontsize=11)
                ax.axis("off")
                print(f"   ⚠️  {model}_{aug}: не найдено")

        plt.tight_layout()
        comp_path = output_path / f"comparison_{model}.png"
        plt.savefig(comp_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Сравнение для {model}: {comp_path}")


# ──────────────────────────────────────────────────────────────────────
class BatchNeuralTester:
    """
    Тестирование нейросетевых моделей сегментации на датасете.

    Аналог BatchClassicTester, но для NeuralSegmenter с поддержкой:
    - Многоклассовой сегментации (ADE20K: 150 классов)
    - Расчёта mIoU и бинарных метрик
    - Эффективной работы с памятью
    """

    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[TestResult] = []
        # 🔥 Используем OrderedDict для LRU-кэша оверлеев (исправляет утечку памяти)
        self.overlays: OrderedDict[str, Image.Image] = OrderedDict()
        self.precision_manager = PrecisionManager(
            default_precision=getattr(config, "precision", "fp32")
        )

    def _find_checkpoints(
        self,
        models_dir: PathLike = "./models",
        model_types: Optional[List[str]] = None,
        augmentation_levels: Optional[List[str]] = None,
    ) -> Dict[str, ModelCheckpoint]:
        """Поиск чекпоинтов по шаблону"""
        if model_types is None:
            model_types = [
                "unet_smp",
                "fpn_smp",
                "psp_smp",
                "deeplab_tv",
                "fcn_tv",
                "segnet",
            ]
        if augmentation_levels is None:
            augmentation_levels = ["none", "basic", "medium"]

        checkpoints: Dict[str, ModelCheckpoint] = {}
        models_path = Path(models_dir)

        for model_type in model_types:
            for aug_level in augmentation_levels:
                pattern = str(models_path / f"{model_type}_{aug_level}_*.pth")
                files = glob.glob(pattern)

                if files:
                    latest = max(files, key=os.path.getctime)
                    key = f"{model_type}_{aug_level}"
                    checkpoints[key] = ModelCheckpoint(
                        key=key,
                        path=Path(latest),
                        model_type=MODEL_TYPE_MAPPING.get(model_type, model_type),
                        augmentation=aug_level,
                        original_type=model_type,
                    )
                    if self.config.verbose:
                        logger.info(f"✓ {key}: {Path(latest).name}")
                else:
                    if self.config.verbose:
                        logger.warning(f"✗ {model_type}_{aug_level}: не найден")

        return checkpoints

    def _load_ade20k_images(
        self,
        repo_id: str = "hf-internal-testing/fixtures_ade20k",
        local_path: Optional[PathLike] = None,
    ) -> List[Tuple[Path, Path]]:
        """
        Загрузка списка изображений и масок из датасета.
        """
        image_mask_pairs: List[Tuple[Path, Path]] = []

        if local_path and Path(local_path).exists():
            data_path = Path(local_path)
            images_dir = data_path / "images" / "validation"
            masks_dir = data_path / "annotations" / "validation"

            if images_dir.exists() and masks_dir.exists():
                for img_path in sorted(images_dir.glob("*.jpg")):
                    mask_path = masks_dir / img_path.with_suffix(".png").name
                    if mask_path.exists():
                        image_mask_pairs.append((img_path, mask_path))
        else:
            try:
                files = list_repo_files(repo_id, repo_type="dataset")
                img_files = [f for f in files if f.endswith(".jpg")]
                mask_files = [f for f in files if f.endswith(".png")]

                for img_file in img_files[: self.config.subset_size or len(img_files)]:
                    img_path = hf_hub_download(
                        repo_id=repo_id, filename=img_file, repo_type="dataset"
                    )
                    mask_file = img_file.replace(".jpg", ".png")
                    if mask_file in mask_files:
                        mask_path = hf_hub_download(
                            repo_id=repo_id, filename=mask_file, repo_type="dataset"
                        )
                        image_mask_pairs.append((Path(img_path), Path(mask_path)))
            except Exception as e:
                logger.error(f"Ошибка загрузки из HF: {e}")
                img_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="ADE_val_00000001.jpg",
                    repo_type="dataset",
                )
                mask_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="ADE_val_00000001.png",
                    repo_type="dataset",
                )
                image_mask_pairs.append((Path(img_path), Path(mask_path)))

        if self.config.subset_size and len(image_mask_pairs) > self.config.subset_size:
            np.random.seed(self.config.random_seed)
            indices = np.random.choice(
                len(image_mask_pairs), self.config.subset_size, replace=False
            )
            image_mask_pairs = [image_mask_pairs[i] for i in sorted(indices)]

        if self.config.verbose:
            logger.info(f"Загружено {len(image_mask_pairs)} пар изображение/маска")

        return image_mask_pairs

    def _resize_mask(
        self, mask: MaskArray, target_shape: Tuple[int, int], order: int = 0
    ) -> MaskArray:
        """Ресайз маски с сохранением целочисленных меток классов"""
        if mask.shape == target_shape:
            return mask.copy()
        sh, sw = target_shape[0] / mask.shape[0], target_shape[1] / mask.shape[1]
        resized = zoom(mask.astype(np.float32), (sh, sw), order=order)
        return np.round(resized).astype(mask.dtype)

    def _calculate_multiclass_iou(
        self,
        pred: MaskArray,
        gt: MaskArray,
        ignore_index: int = 255,
    ) -> Tuple[float, Dict[int, float]]:
        """Расчёт mIoU для многоклассовой сегментации"""
        valid_mask = gt != ignore_index
        if not valid_mask.any():
            return 0.0, {}

        pred_valid = np.where(valid_mask, pred, ignore_index)
        gt_valid = gt
        classes = np.unique(
            np.concatenate([gt_valid[valid_mask], pred_valid[valid_mask]])
        )
        iou_per_class: Dict[int, float] = {}

        for cls in classes:
            if cls == ignore_index:
                continue
            pred_cls = (pred == cls).astype(np.uint8)
            gt_cls = (gt == cls).astype(np.uint8)
            intersection = np.logical_and(pred_cls, gt_cls).sum()
            union = np.logical_or(pred_cls, gt_cls).sum()
            iou_per_class[int(cls)] = intersection / union if union > 0 else 0.0

        valid_ious = [v for v in iou_per_class.values() if v >= 0]
        mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
        return mean_iou, iou_per_class

    def _calculate_binary_metrics(
        self,
        pred: MaskArray,
        gt: MaskArray,
        metrics_list: Optional[List[str]] = None,
    ) -> MetricsDict:
        """Расчёт бинарных метрик (объект vs фон)"""
        pred_binary = (pred > 0).astype(np.uint8)
        gt_binary = (gt > 0).astype(np.uint8)
        return SegmentationMetrics.calculate_all_metrics(
            pred_mask=pred_binary,
            gt_mask=gt_binary,
            threshold=0.5,
            include_hausdorff=True,
            metrics_list=metrics_list or self.config.metrics,
        )

    def _save_overlay_if_needed(
        self,
        checkpoint: ModelCheckpoint,
        img_path: Path,
        overlay: Union[np.ndarray, Image.Image],
    ) -> None:
        """
        🔥 Вынесенный метод для сохранения оверлея с контролем памяти.
        Исправляет проблемы с областью видимости и утечкой памяти.
        """
        if not self.config.save_overlays:
            return

        # 🔥 LRU-кэш: удаляем самый старый, если достигнут лимит
        if len(self.overlays) >= self.config.overlay_cache_max:
            oldest_key = next(iter(self.overlays))
            self.overlays.pop(oldest_key)
            logger.debug(f"🗑️ Удалён старый оверлей из кэша: {oldest_key}")

        # 🔥 Ключ в формате "{модель}_{аугментация}_{изображение}"
        overlay_key = f"{checkpoint.key}_{img_path.stem}"

        # 🔥 Конвертация в PIL с обработкой всех форматов
        try:
            overlay_pil = ensure_pil_compatible(overlay)
            self.overlays[overlay_key] = overlay_pil
            logger.debug(f"✅ Оверлей добавлен: {overlay_key}")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось сохранить оверлей {overlay_key}: {e}")

    def _test_single_model(
        self,
        checkpoint: ModelCheckpoint,
        image_pairs: List[Tuple[Path, Path]],
        precision: str = "fp32",
    ) -> List[TestResult]:
        """Тестирование одной модели на наборе изображений"""
        results: List[TestResult] = []

        # === 1. Загрузка модели с учётом точности ===
        dtype = self._resolve_torch_dtype(precision)
        if not _check_precision_support(None, dtype, self.config.device):
            logger.warning(
                f"⚠️ {precision} не поддерживается на {self.config.device}, fallback на fp32"
            )
            dtype = torch.float32
            precision = "fp32"

        segmenter = NeuralSegmenter(
            model_type=checkpoint.model_type,
            checkpoint_path=str(checkpoint.path),
            device=self.config.device,
            num_classes=self.config.num_classes,
            palette=NeuralSegmenter.ade_palette(),
        )
        segmenter.model.eval()

        if dtype != torch.float32:
            segmenter.model = segmenter.model.to(dtype)

        autocast_enabled = (dtype != torch.float32) and (self.config.device == "cuda")

        if self.config.verbose:
            logger.info(
                f"Тестирование {checkpoint.display_name} на {len(image_pairs)} изображениях..."
            )

        completed = self._load_completed_runs()

        for idx, (img_path, mask_path) in enumerate(
            tqdm(image_pairs, desc=checkpoint.key, disable=not self.config.verbose)
        ):
            try:
                test_image = Image.open(img_path).convert("RGB")
                gt_mask_pil = Image.open(mask_path)
                gt_mask = np.array(gt_mask_pil)
                if gt_mask.ndim == 3 and gt_mask.shape[2] == 3:
                    gt_mask = gt_mask[:, :, 0]

                task_key = f"{checkpoint.key}:{img_path.name}"
                if task_key in completed:
                    if self.config.verbose:
                        logger.debug(f"⏭  Пропущено: {task_key}")
                    continue

                # === 2. Инференс с контролем точности ===
                with safe_inference_context(checkpoint.key, img_path.name):
                    amp_ctx = (
                        torch.amp.autocast(self.config.device, dtype=dtype)
                        if autocast_enabled
                        else nullcontext()
                    )

                    with amp_ctx, torch.no_grad():
                        start_time = time.perf_counter()
                        input_tensor = segmenter.preprocess_image(test_image)

                        if isinstance(input_tensor, np.ndarray):
                            input_tensor = torch.from_numpy(input_tensor).float()
                        elif not isinstance(input_tensor, torch.Tensor):
                            raise TypeError(
                                f"Неожиданный тип входных данных: {type(input_tensor)}"
                            )

                        input_tensor = input_tensor.to(
                            self.config.device, non_blocking=True
                        )
                        if input_tensor.dtype != dtype:
                            input_tensor = input_tensor.to(
                                dtype=dtype, non_blocking=True
                            )

                        # 🔥 Безопасный вызов с проверкой метода
                        if hasattr(segmenter, "_forward"):
                            with (
                                torch.amp.autocast(self.config.device, dtype=dtype)
                                if autocast_enabled
                                else nullcontext()
                            ):
                                output = segmenter._forward(input_tensor.unsqueeze(0))
                                pred_mask = (
                                    output.squeeze(0).argmax(dim=0).cpu().numpy()
                                )
                        else:
                            # 🔥 Fallback: обрабатываем разный return type
                            result = segmenter.predict_segmentation_map(
                                test_image, verbose=False, gt_mask=gt_mask
                            )
                            if isinstance(result, tuple) and len(result) >= 2:
                                pred_mask, _ = result
                            else:
                                pred_mask = result

                        if self.config.device == "cuda":
                            torch.cuda.synchronize()
                        inference_time = time.perf_counter() - start_time

                    self._mark_completed(checkpoint.key, img_path.name)

                # Ресайз предсказания под размер GT
                if gt_mask.shape != pred_mask.shape:
                    pred_resized = self._resize_mask(pred_mask, gt_mask.shape)
                else:
                    pred_resized = pred_mask

                # Расчёт метрик
                m_iou, iou_per_class = self._calculate_multiclass_iou(
                    pred_resized, gt_mask, self.config.ignore_index
                )
                binary_metrics = self._calculate_binary_metrics(
                    pred_resized, gt_mask, self.config.metrics
                )

                # 🔥 Визуализация: вынесено в отдельный метод + безопасный вызов
                if self.config.save_visualizations:
                    try:
                        # 🔥 Проверка наличия метода и адаптация под разный API
                        if hasattr(segmenter, "segment_image_unified"):
                            result = segmenter.segment_image_unified(
                                test_image,
                                alpha=0.6,
                                class_names=NeuralSegmenter.get_ade_class_names(),
                            )
                            # 🔥 Адаптация под tuple или одиночный возврат
                            if isinstance(result, tuple) and len(result) >= 2:
                                overlay, _ = result
                            else:
                                overlay = result
                        elif hasattr(segmenter, "_segment_with_visualization"):
                            tensor_input = segmenter.preprocess_image(test_image)
                            overlay, _ = segmenter._segment_with_visualization(
                                tensor_input,
                                alpha=self.config.viz_alpha,
                                color=self.config.viz_color,
                                precision="fp32",
                            )
                        else:
                            # 🔥 Fallback: простой оверлей через PIL
                            overlay = self._create_simple_overlay(
                                test_image, pred_resized
                            )

                        # 🔥 Сохранение через вынесенный метод
                        self._save_overlay_if_needed(checkpoint, img_path, overlay)

                    except Exception as e:
                        logger.warning(f"⏭ Overlay skipped: {e}")
                        # Fallback: сохранить предсказанную маску
                        try:
                            if pred_mask is not None:
                                mask_vis = (
                                    pred_mask * 255 / (pred_mask.max() + 1e-8)
                                ).astype(np.uint8)
                                if mask_vis.ndim == 2:
                                    mask_vis = np.stack([mask_vis] * 3, axis=-1)
                                viz_dir = Path(self.config.output_dir) / "overlays"
                                viz_dir.mkdir(parents=True, exist_ok=True)
                                Image.fromarray(mask_vis).save(
                                    viz_dir
                                    / f"{checkpoint.key}_{img_path.stem}_fallback.png"
                                )
                        except:
                            pass

                if self.config.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

                # Объединение метрик
                metrics: MetricsDict = {
                    "m_iou": m_iou,
                    "binary_iou": binary_metrics.get("iou", 0),
                    **{k: v for k, v in binary_metrics.items() if k != "iou"},
                }

                result = TestResult(
                    model_key=checkpoint.key,
                    image_name=img_path.name,
                    metrics=metrics,
                    inference_time=inference_time,
                    precision=precision,
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Ошибка при обработке {img_path.name}: {e}")
                continue

        # Очистка памяти
        del segmenter
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        return results

    @staticmethod
    def _resolve_torch_dtype(precision: str) -> torch.dtype:
        """Конвертация строки точности в torch.dtype"""
        mapping = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(precision.lower(), torch.float32)

    def _create_simple_overlay(
        self, image: Image.Image, pred: MaskArray, alpha: float = 0.5
    ) -> Image.Image:
        """Создание простого оверлея через PIL (fallback метод)"""
        mask_np = pred.copy()
        if mask_np.max() <= 1.0:
            mask_np = (mask_np * 255).astype(np.uint8)
        elif mask_np.dtype != np.uint8:
            mask_np = mask_np.astype(np.uint8)

        orig = image.convert("RGB")
        mask_pil = Image.fromarray(mask_np, mode="L").convert("RGB")
        # Ресайз маски если размеры не совпадают
        if mask_pil.size != orig.size:
            mask_pil = mask_pil.resize(orig.size, Image.NEAREST)

        overlay = Image.blend(orig, mask_pil, alpha=alpha)
        return overlay

    def _init_experiment_tracking(self):
        """Инициализация трекера экспериментов"""
        if self.config.get("use_mlflow", False):
            import mlflow

            mlflow.set_experiment("ADE20K_Augmentation_Analysis")
            mlflow.start_run(
                run_name=f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
            )
            mlflow.log_params(asdict(self.config))
            self.tracker = "mlflow"
        elif self.config.get("use_wandb", False):
            import wandb

            wandb.init(project="segmentation-aug-analysis", config=asdict(self.config))
            self.tracker = "wandb"
        else:
            self.tracker = None

    def _log_metrics(
        self, model_key: str, metrics: MetricsDict, step: Optional[int] = None
    ):
        """Логирование метрик в трекер"""
        if self.tracker == "mlflow":
            import mlflow

            for k, v in metrics.items():
                mlflow.log_metric(f"{model_key}/{k}", v, step=step)
        elif self.tracker == "wandb":
            import wandb

            wandb.log({f"{model_key}/{k}": v for k, v in metrics.items()}, step=step)

    def _load_completed_runs(self) -> Set[str]:
        """Загрузка уже обработанных комбинаций для resume"""
        done_file = Path(self.config.output_dir) / ".completed.json"
        if done_file.exists():
            with open(done_file) as f:
                return set(json.load(f))
        return set()

    def _mark_completed(self, model_key: str, image_name: str):
        """Пометка завершённой задачи"""
        done_file = Path(self.config.output_dir) / ".completed.json"
        completed = self._load_completed_runs()
        completed.add(f"{model_key}:{image_name}")
        with open(done_file, "w") as f:
            json.dump(list(completed), f)

    def _batch_predict_with_memory_control(
        self,
        segmenter,
        images: List[Image.Image],
        batch_size: int = 4,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> List[np.ndarray]:
        """Пакетное предсказание с контролем памяти"""
        predictions = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            while True:
                try:
                    with torch.no_grad():
                        preds = [
                            segmenter.predict_segmentation_map(img)[0] for img in batch
                        ]
                    predictions.extend(preds)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    logger.warning(f"OOM, уменьшаем batch_size до {batch_size}")
                    batch = images[i : i + batch_size]

        return predictions

    def _profile_model_inference(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_dir: Path,
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> Dict[str, Any]:
        """Детальное профилирование инференса нейросети"""
        import torch.profiler as profiler

        model.eval()
        sample_input = sample_input.to(self.config.device)

        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(sample_input)
            if self.config.device == "cuda":
                torch.cuda.synchronize()

        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            with profiler.record_function("model_inference"):
                for _ in range(num_runs):
                    with torch.no_grad():
                        _ = model(sample_input)
                    if self.config.device == "cuda":
                        torch.cuda.synchronize()

        results = {
            "total_time_ms": prof.self_cpu_time_total / 1e3 / num_runs,
            "cuda_time_ms": prof.self_cuda_time_total / 1e3 / num_runs,
            "cpu_time_ms": prof.self_cpu_time_total / 1e3 / num_runs,
            "memory_allocated_mb": (
                torch.cuda.max_memory_allocated() / 1e6
                if torch.cuda.is_available()
                else 0
            ),
            "flops": sum(e.flops for e in prof.key_averages() if e.flops > 0),
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(
            str(output_dir / f"trace_{model.__class__.__name__}.json")
        )
        prof.export_stacks(
            str(output_dir / f"stacks_{model.__class__.__name__}.txt"),
            "self_cuda_time_total",
        )

        results["top_ops"] = [
            {
                "name": e.key,
                "cuda_time_ms": e.self_cuda_time_total / 1e3,
                "calls": e.count,
            }
            for e in prof.key_averages().sorted_by(
                "self_cuda_time_total", descending=True
            )[:10]
        ]

        return results

    def _export_model_to_onnx_trt(
        self,
        model: nn.Module,
        model_key: str,
        sample_input: torch.Tensor,
        output_dir: Path,
        opset_version: int = 17,
        trt_precision: Literal["fp32", "fp16"] = "fp16",
    ) -> Dict[str, Optional[Path]]:
        """Экспорт модели с обходом известных проблем"""
        from pathlib import Path
        import torch.onnx

        output_dir.mkdir(parents=True, exist_ok=True)
        results = {"onnx": None, "trt": None}

        onnx_path = output_dir / f"{model_key}.onnx"
        try:
            model.eval()
            sample_input = sample_input.to(self.config.device)

            torch.onnx.export(
                model,
                sample_input,
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=(
                    {
                        "input": {0: "batch", 2: "height", 3: "width"},
                        "output": {0: "batch", 2: "height", 3: "width"},
                    }
                    if getattr(self.config, "dynamic_shapes", False)
                    else None
                ),
                verbose=False,
            )

            try:
                import onnx
                from onnxsim import simplify

                model_onnx = onnx.load(str(onnx_path))
                model_simp, check = simplify(model_onnx)
                if check:
                    onnx.save(model_simp, str(onnx_path))
                    logger.info(f"✅ ONNX simplified: {onnx_path}")
            except ImportError:
                pass

            results["onnx"] = onnx_path
            logger.info(f"✅ ONNX exported: {onnx_path}")

        except Exception as e:
            logger.error(f"❌ ONNX export failed: {e}")
            if opset_version != 18:
                logger.info("🔄 Retrying with opset 18...")
                return self._export_model_to_onnx_trt(
                    model,
                    model_key,
                    sample_input,
                    output_dir,
                    opset_version=18,
                    trt_precision=trt_precision,
                )

        if results["onnx"] and self.config.device == "cuda":
            trt_path = output_dir / f"{model_key}.{trt_precision}.trt"
            try:
                import torch_tensorrt

                input_spec = [
                    torch_tensorrt.Input(
                        sample_input.shape,
                        dtype=(
                            torch.float16 if trt_precision == "fp16" else torch.float32
                        ),
                        name="input",
                    )
                ]

                trt_model = torch_tensorrt.compile(
                    model,
                    inputs=input_spec,
                    enabled_precisions={
                        torch.float16 if trt_precision == "fp16" else torch.float32
                    },
                    ir="onnx",
                    min_block_size=1,
                    fallback_to_torch=True,
                )

                torch.jit.save(trt_model, str(trt_path))
                results["trt"] = trt_path
                logger.info(f"✅ TensorRT engine saved: {trt_path}")

            except ImportError:
                logger.warning("⚠️  torch-tensorrt not installed. Skip TRT export.")
            except Exception as e:
                logger.error(f"❌ TRT compile failed: {e}")
                if trt_precision == "fp16":
                    logger.info("🔄 Retrying TRT with fp32...")
                    return self._export_model_to_onnx_trt(
                        model,
                        model_key,
                        sample_input,
                        output_dir,
                        opset_version=opset_version,
                        trt_precision="fp32",
                    )

        return results

    def run(self) -> pd.DataFrame:
        """Запуск полного цикла тестирования"""
        checkpoints = self._find_checkpoints()
        if not checkpoints:
            logger.error("Не найдено чекпоинтов для тестирования!")
            return pd.DataFrame()

        image_pairs = self._load_ade20k_images(local_path=self.config.dataset_path)
        if not image_pairs:
            logger.error("Не удалось загрузить изображения датасета!")
            return pd.DataFrame()

        all_results: List[TestResult] = []
        for checkpoint in checkpoints.values():
            model_results = self._test_single_model(checkpoint, image_pairs)
            all_results.extend(model_results)

        if not all_results:
            logger.error("Нет результатов для анализа!")
            return pd.DataFrame()

        df = pd.DataFrame([r.to_dict() for r in all_results])
        df[["model", "augmentation"]] = df["model_key"].str.split("_", n=1, expand=True)

        self._print_summary_statistics(df)

        return df

    def _print_summary_statistics(self, df: pd.DataFrame) -> None:
        """Печать сводной статистики (как в analyze.py)"""
        print("\n" + "=" * 80)
        print("СВОДНАЯ СТАТИСТИКА")
        print("=" * 80)

        # Средний mIoU по уровням аугментаций
        if "none" in df["augmentation"].values:
            avg_none = df[df["augmentation"] == "none"]["m_iou"].mean()
            print(f"\n📊 Средний mIoU по уровням аугментаций:")
            print(f"   None:   {avg_none:.4f}")
        else:
            avg_none = 0.0
            print(f"\n📊 Средний mIoU по уровням аугментаций:")
            print(f"   None:   N/A")

        if "basic" in df["augmentation"].values:
            avg_basic = df[df["augmentation"] == "basic"]["m_iou"].mean()
            gain_basic = (avg_basic - avg_none) * 100 if avg_none > 0 else 0
            print(f"   Basic:  {avg_basic:.4f} (прирост: {gain_basic:+.2f}%)")
        else:
            avg_basic = 0.0
            print(f"   Basic:  N/A")

        if "medium" in df["augmentation"].values:
            avg_medium = df[df["augmentation"] == "medium"]["m_iou"].mean()
            gain_medium = (avg_medium - avg_none) * 100 if avg_none > 0 else 0
            print(f"   Medium: {avg_medium:.4f} (прирост: {gain_medium:+.2f}%)")
        else:
            print(f"   Medium: N/A")

        # Лучшая комбинация
        if not df.empty:
            best_idx = df["m_iou"].idxmax()
            best_row = df.loc[best_idx]
            print(f"\n🏆 Лучшая комбинация:")
            print(f"   Модель: {best_row['model']}")
            print(f"   Аугментации: {best_row['augmentation']}")
            print(f"   mIoU: {best_row['m_iou']:.4f}")

        print("\n" + "=" * 80)

    def aggregate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Агрегация метрик по моделям и аугментациям"""
        agg_funcs = ["mean", "std", "min", "max"]
        metric_cols = [
            c for c in df.columns if c in self.config.metrics or c == "m_iou"
        ]
        aggregated = df.groupby(["model", "augmentation", "precision"])[
            metric_cols + ["inference_time"]
        ].agg(agg_funcs)
        aggregated.columns = [
            "_".join(col).strip() for col in aggregated.columns.values
        ]
        return aggregated.reset_index()

    def statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Расширенный статистический анализ: ANOVA + пост-хок тесты.
        Работает с 1+ изображениями.
        """
        from scipy import stats
        import warnings

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        analysis: Dict[str, Any] = {}

        # ──────────────────────────────────────────────────────────────
        # 1. Сводная статистика по уровням аугментаций
        # ──────────────────────────────────────────────────────────────
        summary = {}
        for aug in ["none", "basic", "medium"]:
            aug_data = df[df["augmentation"] == aug]
            if not aug_data.empty:
                summary[aug] = {
                    "mean_m_iou": float(aug_data["m_iou"].mean()),
                    "std_m_iou": float(aug_data["m_iou"].std()),
                    "median_m_iou": float(aug_data["m_iou"].median()),
                    "min_m_iou": float(aug_data["m_iou"].min()),
                    "max_m_iou": float(aug_data["m_iou"].max()),
                    "n_samples": int(len(aug_data)),
                }
        analysis["summary_by_augmentation"] = summary

        # ──────────────────────────────────────────────────────────────
        # 2. ANOVA для каждой модели (сравнение аугментаций)
        # ──────────────────────────────────────────────────────────────
        anova_results = {}
        for model in df["model"].unique():
            model_data = df[df["model"] == model]
            groups = [
                group["m_iou"].dropna().values
                for _, group in model_data.groupby("augmentation")
                if len(group["m_iou"].dropna()) > 1
            ]

            if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    anova_results[model] = {
                        "f_statistic": float(f_stat),
                        "p_value": float(p_value),
                        "significant": bool(p_value < 0.05),
                        "n_groups": len(groups),
                    }
                except Exception:
                    anova_results[model] = {"error": "ANOVA failed"}

        analysis["anova_by_model"] = anova_results

        # ──────────────────────────────────────────────────────────────
        # 3. Пост-хок тесты (Tukey HSD) при значимом ANOVA
        # ──────────────────────────────────────────────────────────────
        posthoc_results = {}
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd

            for model, result in anova_results.items():
                if result.get("significant", False):
                    model_data = df[df["model"] == model].copy()
                    if len(model_data) >= 3:
                        tukey = pairwise_tukeyhsd(
                            endog=model_data["m_iou"].values,
                            groups=model_data["augmentation"].values,
                            alpha=0.05,
                        )
                        posthoc_results[model] = {
                            "significant_pairs": [
                                f"{pair[0]} vs {pair[1]}: p={pair[4]:.4f}"
                                for pair in zip(
                                    tukey.groupsunique,
                                    tukey.meandiffs,
                                    tukey.pvalues,
                                    tukey.reject,
                                    range(len(tukey.pvalues)),
                                )
                                if pair[3]  # reject == True
                            ]
                        }
        except ImportError:
            analysis["note"] = "statsmodels not installed — пост-хок тесты пропущены"

        analysis["posthoc_tukey"] = posthoc_results

        # ──────────────────────────────────────────────────────────────
        # 4. Лучшая комбинация (глобально и по моделям)
        # ──────────────────────────────────────────────────────────────
        if not df.empty:
            best_global = df.loc[df["m_iou"].idxmax()]
            analysis["best_global"] = {
                "model": str(best_global["model"]),
                "augmentation": str(best_global["augmentation"]),
                "m_iou": float(best_global["m_iou"]),
                "m_iou_std": float(
                    df[
                        (df["model"] == best_global["model"])
                        & (df["augmentation"] == best_global["augmentation"])
                    ]["m_iou"].std()
                ),
            }

            best_by_model = {}
            for model in df["model"].unique():
                model_best = df[df["model"] == model].loc[
                    df[df["model"] == model]["m_iou"].idxmax()
                ]
                best_by_model[model] = {
                    "augmentation": str(model_best["augmentation"]),
                    "m_iou": float(model_best["m_iou"]),
                }
            analysis["best_by_model"] = best_by_model

        return analysis

    def export_results(
        self,
        df: pd.DataFrame,
        aggregated: pd.DataFrame,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Экспорт результатов с полным набором визуализаций"""
        output = Path(self.config.output_dir)
        exported: Dict[str, Path] = {}

        # ──────────────────────────────────────────────────────────────
        # CSV экспорт
        # ──────────────────────────────────────────────────────────────
        csv_path = output / "detailed_results.csv"
        df.to_csv(csv_path, index=False)
        exported["csv"] = csv_path

        agg_csv = output / "aggregated_metrics.csv"
        aggregated.to_csv(agg_csv, index=False)
        exported["aggregated_csv"] = agg_csv

        # ──────────────────────────────────────────────────────────────
        # JSON со статистикой
        # ──────────────────────────────────────────────────────────────
        if stats:
            json_path = output / "statistical_analysis.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, default=str, ensure_ascii=False)
            exported["stats_json"] = json_path

        # ──────────────────────────────────────────────────────────────
        # Markdown отчёт (расширенный)
        # ──────────────────────────────────────────────────────────────
        md_path = output / "report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Отчёт: Влияние аугментаций на качество сегментации (ADE20K)\n\n")
            f.write(f"**Дата:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Изображений:** {len(df)}\n")
            f.write(f"**Моделей:** {df['model'].nunique()}\n")
            f.write(f"**Аугментаций:** {df['augmentation'].nunique()}\n\n")

            # Сводная таблица
            f.write("## 📊 Сводная таблица метрик (mIoU ± std)\n\n")
            pivot = df.pivot_table(
                values="m_iou",
                index="model",
                columns="augmentation",
                aggfunc=["mean", "std"],
            )
            f.write(pivot.round(4).to_markdown() + "\n\n")

            # Лучшая комбинация
            f.write("## 🏆 Лучшая комбинация по mIoU\n\n")
            if stats and "best_global" in stats:
                best = stats["best_global"]
                f.write(f"- **Модель:** `{best['model']}`\n")
                f.write(f"- **Аугментации:** `{best['augmentation']}`\n")
                f.write(
                    f"- **mIoU:** `{best['m_iou']:.4f} ± {best['m_iou_std']:.4f}`\n\n"
                )

            # Статистика по аугментациям
            if stats and "summary_by_augmentation" in stats:
                f.write("## 📈 Статистика по уровням аугментаций\n\n")
                f.write("| Уровень | Mean mIoU | Std | Median | Min | Max | N |\n")
                f.write("|---------|-----------|-----|--------|-----|-----|---|\n")
                for aug, data in stats["summary_by_augmentation"].items():
                    f.write(
                        f"| {aug} | {data['mean_m_iou']:.4f} | {data['std_m_iou']:.4f} | "
                        f"{data['median_m_iou']:.4f} | {data['min_m_iou']:.4f} | "
                        f"{data['max_m_iou']:.4f} | {data['n_samples']} |\n"
                    )
                f.write("\n")

            # ANOVA результаты
            if stats and "anova_by_model" in stats:
                f.write("## 🔬 Статистическая значимость (ANOVA)\n\n")
                for model, result in stats["anova_by_model"].items():
                    if "error" not in result:
                        sig = "✅" if result.get("significant") else "❌"
                        f.write(f"- `{model}`: p={result['p_value']:.4f} {sig}\n")
                f.write("\n")

            # Прирост относительно baseline
            f.write("## 📈 Прирост относительно baseline (none)\n\n")
            pivot_gain = df.pivot_table(
                values="m_iou", index="model", columns="augmentation", aggfunc="mean"
            )
            if "none" in pivot_gain.columns:
                f.write("| Модель | Basic Δ% | Medium Δ% |\n")
                f.write("|--------|----------|-----------|\n")
                for model in pivot_gain.index:
                    none_val = pivot_gain.loc[model, "none"]
                    basic_gain = (
                        ((pivot_gain.loc[model, "basic"] - none_val) / none_val * 100)
                        if "basic" in pivot_gain.columns
                        else 0
                    )
                    medium_gain = (
                        ((pivot_gain.loc[model, "medium"] - none_val) / none_val * 100)
                        if "medium" in pivot_gain.columns
                        else 0
                    )
                    f.write(f"| {model} | {basic_gain:+.2f}% | {medium_gain:+.2f}% |\n")

        exported["report_md"] = md_path

        # ──────────────────────────────────────────────────────────────
        # Визуализации (оверлеи)
        # ──────────────────────────────────────────────────────────────
        if self.overlays and self.config.save_overlays:
            viz_dir = output / "overlays"
            viz_dir.mkdir(exist_ok=True)
            for key, overlay in self.overlays.items():
                try:
                    overlay_pil = ensure_pil_compatible(overlay)
                    overlay_pil.save(viz_dir / f"{key}.png")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось сохранить оверлей {key}: {e}")
            exported["overlays_dir"] = viz_dir

        # ──────────────────────────────────────────────────────────────
        # Графики (через plot_detailed_results)
        # ──────────────────────────────────────────────────────────────
        plots = self.plot_detailed_results(df)
        exported.update({f"plot_{k}": v for k, v in plots.items()})

        # ──────────────────────────────────────────────────────────────
        # Сетка сравнения визуализаций
        # ──────────────────────────────────────────────────────────────
        if self.overlays:
            unique_models = list(
                set(extract_model_aug_from_key(k)[0] for k in self.overlays.keys())
            )
            save_augmentation_comparison_grid(
                overlay_images=self.overlays,
                output_dir=output / "overlays",
                model_names=unique_models,
            )
            exported["comparison_grid"] = (
                output / "overlays" / "full_comparison_grid.png"
            )

        return exported

    def plot_results(
        self, df: pd.DataFrame, aggregated: pd.DataFrame
    ) -> Dict[str, Path]:
        """Построение и сохранение графиков"""
        output = self.config.output_dir
        plots: Dict[str, Path] = {}

        plt.figure(figsize=(14, 6))
        sns.barplot(
            data=df,
            x="model",
            y="m_iou",
            hue="augmentation",
            palette="viridis",
            errorbar="sd",
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("mIoU")
        plt.title("Влияние аугментаций на mIoU (ADE20K)")
        plt.tight_layout()
        plot_path = output / "miou_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["miou_bar"] = plot_path

        pivot = df.pivot_table(
            values="m_iou", index="model", columns="augmentation", aggfunc="mean"
        )
        if "none" in pivot.columns:
            gain = pivot.copy()
            for col in ["basic", "medium"]:
                if col in gain.columns:
                    gain[col] = (
                        (gain[col] - gain["none"]) / gain["none"].replace(0, 1e-8) * 100
                    )

            plt.figure(figsize=(10, 6))
            sns.heatmap(gain, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
            plt.title("Прирост mIoU относительно baseline (%)")
            plt.ylabel("Модель")
            plt.xlabel("Аугментация")
            plt.tight_layout()
            heatmap_path = output / "gain_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
            plt.close()
            plots["gain_heatmap"] = heatmap_path

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="augmentation", y="m_iou", hue="model", palette="Set2")
        plt.ylabel("mIoU per image")
        plt.title("Распределение mIoU по изображениям")
        plt.tight_layout()
        boxplot_path = output / "miou_distribution.png"
        plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["boxplot"] = boxplot_path

        plt.figure(figsize=(12, 5))
        sns.barplot(
            data=df,
            x="model",
            y="inference_time",
            hue="augmentation",
            palette="coolwarm",
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Время (сек)")
        plt.title("Время инференса на изображение")
        plt.tight_layout()
        time_path = output / "inference_time.png"
        plt.savefig(time_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["time"] = time_path

        return plots

    def plot_detailed_results(self, df: pd.DataFrame) -> Dict[str, Path]:
        """
        Построение детальных графиков как в analyze.py (для 1+ изображений).
        """
        output = Path(self.config.output_dir) / "plots"
        output.mkdir(parents=True, exist_ok=True)
        plots: Dict[str, Path] = {}

        # ──────────────────────────────────────────────────────────────
        # График 1: Все метрики по моделям и аугментациям (2×3 сетка)
        # ──────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        metrics_to_plot: List[str] = [
            "m_iou",
            "binary_iou",
            "dice",
            "f1_score",
            "precision",
            "recall",
        ]
        metric_names: Dict[str, str] = {
            "m_iou": "mIoU",
            "binary_iou": "Binary IoU",
            "dice": "Dice",
            "f1_score": "F1-Score",
            "precision": "Precision",
            "recall": "Recall",
        }

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            if metric not in df.columns:
                ax.text(
                    0.5,
                    0.5,
                    f"'{metric}'\nN/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
                ax.set_title(metric_names.get(metric, metric), fontsize=11)
                ax.axis("off")
                continue

            # Агрегация: mean по (модель × аугментация)
            plot_data = df.groupby(["model", "augmentation"])[metric].mean().unstack()
            plot_data.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black")
            ax.set_title(
                f"{metric_names[metric]} по моделям и аугментациям", fontsize=11
            )
            ax.set_ylabel("Score")
            ax.set_xlabel("Модель")
            ax.legend(title="Аугментации", loc="lower right", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=45, labelsize=8)

        plt.tight_layout()
        plot_path = output / "all_metrics_grid.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["all_metrics_grid"] = plot_path

        # ──────────────────────────────────────────────────────────────
        # График 2: mIoU bar chart (основной)
        # ──────────────────────────────────────────────────────────────
        plt.figure(figsize=(14, 6))
        sns.barplot(
            data=df,
            x="model",
            y="m_iou",
            hue="augmentation",
            palette="viridis",
            errorbar="sd",
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("mIoU")
        plt.title("Влияние аугментаций на качество сегментации (mIoU ± std)")
        plt.tight_layout()
        plot_path = output / "miou_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["miou_bar"] = plot_path

        # ──────────────────────────────────────────────────────────────
        # График 3: Heatmap прироста относительно baseline
        # ──────────────────────────────────────────────────────────────
        pivot = df.pivot_table(
            values="m_iou", index="model", columns="augmentation", aggfunc="mean"
        )
        if "none" in pivot.columns:
            gain = pivot.copy()
            for col in ["basic", "medium"]:
                if col in gain.columns:
                    gain[col] = (
                        (gain[col] - gain["none"]) / gain["none"].replace(0, 1e-8) * 100
                    )

            plt.figure(figsize=(10, 8))
            sns.heatmap(gain, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
            plt.title("Прирост mIoU относительно baseline (%)")
            plt.ylabel("Модель")
            plt.xlabel("Аугментация")
            plt.tight_layout()
            plot_path = output / "gain_heatmap.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            plots["gain_heatmap"] = plot_path

        # ──────────────────────────────────────────────────────────────
        # График 4: Распределение mIoU по изображениям (boxplot)
        # ──────────────────────────────────────────────────────────────
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df,
            x="augmentation",
            y="m_iou",
            hue="model",
            palette="Set2",
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": "8",
            },
        )
        plt.ylabel("mIoU per image")
        plt.title("Распределение mIoU по изображениям")
        plt.legend(title="Модель", loc="best", fontsize=8)
        plt.tight_layout()
        plot_path = output / "miou_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["boxplot"] = plot_path

        plt.figure(figsize=(12, 6))
        sns.swarmplot(
            data=df,
            x="augmentation",
            y="m_iou",
            hue="model",
            palette="Set2",
            size=4,
            alpha=0.6,
        )
        plt.ylabel("mIoU per image")
        plt.title("Распределение mIoU по изображениям (swarmplot)")
        plt.legend(title="Модель", loc="best", fontsize=8)
        plt.tight_layout()
        plot_path = output / "miou_distribution_swarm.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["swarmplot"] = plot_path

        # ──────────────────────────────────────────────────────────────
        # График 5: Время инференса
        # ──────────────────────────────────────────────────────────────
        plt.figure(figsize=(12, 5))
        sns.barplot(
            data=df,
            x="model",
            y="inference_time",
            hue="augmentation",
            palette="coolwarm",
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Время (сек)")
        plt.title("Время инференса на изображение")
        plt.tight_layout()
        plot_path = output / "inference_time.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["time"] = plot_path

        # ──────────────────────────────────────────────────────────────
        # График 6: Прирост по уровням аугментаций (два графика)
        # ──────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for model in df["model"].unique():
            model_data = df[df["model"] == model]

            none_val = model_data[model_data["augmentation"] == "none"]["m_iou"].mean()
            basic_val = model_data[model_data["augmentation"] == "basic"][
                "m_iou"
            ].mean()
            medium_val = model_data[model_data["augmentation"] == "medium"][
                "m_iou"
            ].mean()

            if not np.isnan(none_val) and not np.isnan(basic_val):
                basic_gain = (basic_val - none_val) * 100
                axes[0].bar(
                    f"{model}\n(basic-none)", basic_gain, alpha=0.7, color="steelblue"
                )

            if not np.isnan(none_val) and not np.isnan(medium_val):
                medium_gain = (medium_val - none_val) * 100
                axes[1].bar(
                    f"{model}\n(medium-none)", medium_gain, alpha=0.7, color="orange"
                )

        axes[0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[0].set_title("Прирост mIoU: Basic vs None (%)", fontsize=11)
        axes[0].set_ylabel("Прирост mIoU (%)")
        axes[0].tick_params(axis="x", rotation=45, labelsize=8)
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[1].set_title("Прирост mIoU: Medium vs None (%)", fontsize=11)
        axes[1].set_ylabel("Прирост mIoU (%)")
        axes[1].tick_params(axis="x", rotation=45, labelsize=8)
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plot_path = output / "augmentation_gain.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["gain_comparison"] = plot_path

        return plots


def ensure_pil_compatible(arr: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    🔥 Исправленная функция: гарантирует совместимость с PIL.Image.
    Обрабатывает как numpy-массивы, так и уже готовые PIL.Image объекты.
    """
    # ✅ Если уже PIL.Image — возвращаем как есть
    if isinstance(arr, Image.Image):
        return arr.convert("RGB") if arr.mode != "RGB" else arr

    # Работа с numpy массивом
    arr = np.asarray(arr).copy()

    # 🔥 Нормализация к [0, 255] для любого входного диапазона
    if arr.min() < 0 or arr.max() > 255:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255

    # 🔥 Конвертация типов с защитой от переполнения
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # 🔥 Обеспечение 3 каналов для RGB
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]  # Обрезаем лишние каналы

    return Image.fromarray(arr)


# ──────────────────────────────────────────────────────────────────────
def main():
    """Точка входа для CLI"""
    parser = argparse.ArgumentParser(
        description="Анализ влияния аугментаций на сегментацию"
    )
    parser.add_argument(
        "--dataset", type=str, default="./data/ADE20K", help="Путь к датасету"
    )
    parser.add_argument(
        "--subset", type=int, default=50, help="Размер подмножества для теста"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/augmentation_analysis",
        help="Директория для результатов",
    )
    parser.add_argument(
        "--models", type=str, default="./models", help="Директория с чекпоинтами"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Подробный вывод"
    )
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--profile", action="store_true", help="Enable torch.profiler")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--export-trt", action="store_true")
    parser.add_argument("--trt-precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Сохранять визуализации сегментации (оверлеи)",
    )

    args = parser.parse_args()

    config = TestConfig(
        dataset_path=args.dataset,
        output_dir=args.output,
        subset_size=args.subset,
        random_seed=args.seed,
        verbose=args.verbose,
    )

    tester = BatchNeuralTester(config)

    print(f"\n🔍 CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(
            f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    df = tester.run()
    if df.empty:
        print("❌ Тестирование не дало результатов")
        return

    aggregated = tester.aggregate_metrics(df)
    stats = tester.statistical_analysis(df)

    # 🔥 ИСПРАВЛЕНИЕ #1: Доступ к overlay_images через tester.overlays
    if tester.overlays:
        print(
            f"\n📊 Генерация сравнительных визуализаций ({len(tester.overlays)} оверлеев)..."
        )

        # 🔥 ИСПРАВЛЕНИЕ #2: Используем extract_model_aug_from_key для поиска
        unique_models = list(
            set(extract_model_aug_from_key(k)[0] for k in tester.overlays.keys())
        )

        # Сохранение отдельных сравнений по моделям
        save_model_augmentation_comparisons(
            overlay_images=tester.overlays,  # 🔥 Передаём tester.overlays
            output_dir=Path(args.output) / "overlays",
            models=unique_models,
        )

        # Сохранение общей сетки сравнения
        save_augmentation_comparison_grid(
            overlay_images=tester.overlays,  # 🔥 Передаём tester.overlays
            output_dir=Path(args.output) / "overlays",
            model_names=unique_models,
        )

    exported = tester.export_results(df, aggregated, stats)
    print(f"\n💾 Результаты сохранены:")
    for name, path in exported.items():
        print(f"   • {name}: {path}")

    plots = tester.plot_results(df, aggregated)
    print(f"\n📊 Графики сохранены:")
    for name, path in plots.items():
        print(f"   • {name}: {path}")

    print(f"\n✅ Анализ завершён!")
    print(f"   Всего результатов: {len(df)}")
    print(f"   Моделей: {df['model'].nunique()}")
    print(f"   Аугментаций: {df['augmentation'].nunique()}")

    best = df.loc[df["m_iou"].idxmax()]
    print(f"\n🏆 Лучшая комбинация: {best['model']}_{best['augmentation']}")
    print(f"   mIoU: {best['m_iou']:.4f}")


if __name__ == "__main__":
    main()
