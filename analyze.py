# testing/analyze.py

"""Вспомогательный скрипт для проверки влияния аугментаций на качество обучения сегментационных моделей.

Workflow:
1. Поиск чекпоинтов по шаблону `{model_type}_{augmentation_level}_*.pth`.
2. Загрузка тестового изображения и GT-маски из HuggingFace Hub.
3. Оценка каждой модели: предсказание → расчёт mIoU, Dice, F1, ... → сохранение overlay.
4. Визуализация: бар-чарты, heatmaps, сравнение времени инференса.
5. Экспорт: CSV, Markdown-отчёт, PNG-графики.

Example:
    ```bash
    python analyze.py
    ```
"""

"""Вспомогательный скрипт для проверки влияния аугментаций на качество обучения сегментационных моделей.

Поддерживаемые задачи:
1. **Быстрый анализ**: оценка моделей на одном тестовом изображении из ADE20K
2. **Многоклассовые метрики**: mIoU по 150 классам + бинарные метрики для совместимости
3. **Визуализация**: overlay-визуализации, bar-чарты, heatmaps прироста, сравнение времени
4. **Экспорт**: CSV с метриками, Markdown-отчёт, PNG-графики, оверлеи

Ключевые особенности:
- ✅ Авто-поиск чекпоинтов по шаблону `{model}_{aug}_*.pth`
- ✅ Загрузка данных из HuggingFace Hub (без необходимости локального датасета)
- ✅ Многоклассовый mIoU + бинарные метрики в одном запуске
- ✅ Ресайз предсказаний к размеру GT через nearest-neighbor интерполяцию
- ✅ Очистка CUDA-памяти между моделями для предотвращения OOM
- ✅ Сравнительная сетка оверлеев: модели × уровни аугментаций

Типичный workflow:
```bash
# 1. Подготовка чекпоинтов
ls ./models/unet_smp_{none,basic,medium}_*.pth

# 2. Запуск анализа
python analyze.py

# 3. Просмотр результатов
cat ./data/augmentation_analysis/report.md
open ./data/augmentation_analysis/augmentation_impact_miou.png
```

Note:
- Скрипт использует фиксированное тестовое изображение `ADE_val_00000001.jpg` из репозитория `hf-internal-testing/fixtures_ade20k`.
- Для анализа на других изображениях замените `hf_hub_download()` на локальную загрузку.
- Многоклассовый mIoU рассчитывается только для классов, присутствующих в предсказании или GT.
- Бинарные метрики используют порог 0: всё, кроме класса 0, считается объектом.
- При отсутствии чекпоинта для комбинации модель/аугментация — выводится предупреждение, анализ продолжается.
- Результаты сохраняются в `./data/augmentation_analysis/`; измените `output_dir` при необходимости.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
# from __future__ import annotations  # PEP 563: отложенная оценка аннотаций
import sys
import glob
import os
import gc
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union, TypeAlias

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from scipy.ndimage import zoom

import logging

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler: logging.StreamHandler = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Настройка путей проекта
project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Локальные импорты
from segmenters.NeuralSegmenter import NeuralSegmenter
from metrics.SegmentationMetrics import SegmentationMetrics
from utils.palettes import ade_palette, get_ade_class_names

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & CONSTANTS
# ──────────────────────────────────────────────────────────────────────
MaskArray: TypeAlias = np.ndarray
"""Формат маски сегментации, dtype=np.ndarray."""

ImageArray: TypeAlias = np.ndarray
"""Формат массива пикселей изображения, dtype=np.ndarray."""

MetricValue: TypeAlias = float
"""Поссчитанное значение-результат метрики, dtype=float."""

MetricsDict: TypeAlias = Dict[str, MetricValue]
"""Словарь для хранения результатов метрик, dtype=Dict[str, float]."""

PathLike: TypeAlias = Union[str, Path]
"""Тип пути до файла, исходные форматы: str/Path, dtype=Union[str, Path]."""

MODEL_TYPE_MAPPING: Dict[str, str] = {
    "unet_smp": "unet_smp",
    "fpn_smp": "fpn_smp",
    "psp_smp": "pspnet_smp",
    "deeplab_tv": "deeplab_tv",
    "fcn_tv": "fcn_tv",
    "segnet": "segnet",
}
"""Маппинг имён чекпоинтов на ModelType enum, dtype=Dict[str, str]."""


def get_color_for_class(cls: int, palette: List[List[int]], offset: int = -1) -> List[int]:
    """Получает цвет для класса. Если индекс < 0 (Background), возвращает черный."""
    cls_int = int(cls)
    palette_idx = cls_int + offset

    # 🔧 FIX: Если индекс отрицательный (Background), возвращаем черный [0,0,0]
    if palette_idx < 0:
        return [0, 0, 0]

    if 0 <= palette_idx < len(palette):
        return palette[palette_idx]
    return [0, 0, 0]  # fallback


def get_name_for_class(cls: int, class_names: Dict[int, str], offset: int = -1) -> str:
    """Получает имя класса. Если индекс < 0 (Background), возвращает 'Background'."""
    cls_int = int(cls)
    name_idx = cls_int + offset

    # 🔧 FIX: Если индекс отрицательный, это фон, а не fallback на класс 0 (wall)!
    if name_idx < 0:
        return "Background"

    if name_idx in class_names:
        return class_names[name_idx]
    return f"Class_{name_idx}"


# ──────────────────────────────────────────────────────────────────────
def analyze_augmentation_impact() -> Optional[Tuple[Optional[pd.DataFrame], Optional[Dict[str, Image.Image]]]]:
    """Исследование влияния аугментаций на качество сегментации.

    Исправленная версия с поддержкой всех моделей и корректными метриками.

    Логика:
    1. Поиск чекпоинтов по шаблону в `./models/`.
    2. Загрузка тестового изображения и маски из `hf-internal-testing/fixtures_ade20k`.
    3. Для каждой модели:
       - Загрузка через `NeuralSegmenter`.
       - Предсказание + ресайз к размеру GT.
       - Расчёт mIoU (многоклассовый) и бинарных метрик.
       - Сохранение overlay-визуализации.
       - Очистка CUDA-памяти.
    4. Агрегация результатов в DataFrame.
    5. Визуализация: бар-чарты, heatmaps, сравнение времени.
    6. Экспорт: CSV, Markdown-отчёт.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[Dict[str, PIL.Image]]]:
        - DataFrame с метриками по моделям и аугментациям.
        - Словарь `{key: overlay_image}` для дальнейшего использования.
        Если результаты пусты — возвращает `(None, None)`.
    """
    print("=" * 80)
    print("ИССЛЕДОВАНИЕ: ВЛИЯНИЕ АУГМЕНТАЦИЙ НА КАЧЕСТВО СЕГМЕНТАЦИИ")
    print("=" * 80)

    models_dir: str = "./models"
    model_types: List[str] = [
        "unet_smp",
        "fpn_smp",
        "psp_smp",
        "deeplab_tv",
        "fcn_tv",
        "segnet",
    ]
    augmentation_levels: List[str] = ["none", "basic", "medium"]

    checkpoints: Dict[str, Dict[str, Any]] = {}

    print("\n🔍 Поиск чекпоинтов...")
    for model_type in model_types:
        for aug_level in augmentation_levels:
            pattern: str = f"{models_dir}/{model_type}_{aug_level}_*.pth"
            files: List[str] = glob.glob(pattern)

            if files:
                latest_checkpoint: str = max(files, key=os.path.getctime)
                key: str = f"{model_type}_{aug_level}"
                checkpoints[key] = {
                    "path": latest_checkpoint,
                    "model_type": MODEL_TYPE_MAPPING.get(model_type, model_type),
                    "augmentation": aug_level,
                    "original_type": model_type,
                }
                print(f"   ✅ {key}: {os.path.basename(latest_checkpoint)}")
            else:
                print(f"   ⚠️  {model_type}_{aug_level}: не найден")

    print("\n📥 Загрузка тестовых данных...")
    repo_id: str = "hf-internal-testing/fixtures_ade20k"
    img_path: str = hf_hub_download(repo_id=repo_id, filename="ADE_val_00000001.jpg", repo_type="dataset")
    mask_path: str = hf_hub_download(repo_id=repo_id, filename="ADE_val_00000001.png", repo_type="dataset")

    output_dir: str = "./data/augmentation_analysis2"
    os.makedirs(output_dir, exist_ok=True)

    test_image: Image.Image = Image.open(img_path).convert("RGB")
    test_image.save(f"{output_dir}/test_image.jpg")

    gt_mask_pil: Image.Image = Image.open(mask_path)
    gt_mask_pil.save(f"{output_dir}/test_mask.jpg")

    gt_mask: MaskArray = np.array(gt_mask_pil)
    if gt_mask.ndim == 3 and gt_mask.shape[2] == 3:
        # RGB маска → берём первый канал или конвертируем
        gt_mask = gt_mask[:, :, 0]

    print(f"   ✅ Изображение: {test_image.size}")
    print(f"   ✅ Маска: {gt_mask.shape}, unique values: {len(np.unique(gt_mask))}")

    print("\n🧪 Оценка моделей...")
    results: List[Dict[str, Any]] = []
    overlay_images: Dict[str, Image.Image] = {}

    for key, checkpoint_info in checkpoints.items():
        try:
            model_type = checkpoint_info["model_type"]
            aug_level = checkpoint_info["augmentation"]
            checkpoint_path: str = checkpoint_info["path"]
            display_name: str = checkpoint_info["original_type"]

            print(f"\n   🔹 {key} {display_name}_{aug_level}...")

            segmenter = NeuralSegmenter(
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                device="cuda",
                num_classes=150,
                palette=NeuralSegmenter.ade_palette(),
            )

            # Предсказание
            start_time: float = time.perf_counter()
            pred_mask: MaskArray
            pred_info: Dict[str, Any]
            pred_mask, pred_info = segmenter.predict_segmentation_map(test_image, verbose=False, gt_mask=gt_mask)
            inference_time: float = time.perf_counter() - start_time

            # Бинарная сегментация для совместимости
            pred_mask_2: MaskArray = segmenter.segment(np.array(test_image))
            print(pred_mask_2)

            # Ресайз предсказания под размер GT
            if gt_mask.shape != pred_mask.shape:
                # sh, sw = (
                #     gt_mask.shape[0] / pred_mask.shape[0],
                #     gt_mask.shape[1] / pred_mask.shape[1],
                # )
                # pred_mask_resized: MaskArray = zoom(pred_mask, (sh, sw), order=0)
                pred_pil = Image.fromarray(pred_mask.astype(np.uint16))
                pred_mask_resized = np.array(
                    pred_pil.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.Resampling.NEAREST)
                ).astype(np.uint8)
            else:
                pred_mask_resized = pred_mask

            print("\n🔍 Проверка ignore_index в масках:")
            print(f"   GT mask: min={gt_mask.min()}, max={gt_mask.max()}, unique={np.unique(gt_mask)[:20]}")
            print(f"   Count of 255 in GT: {(gt_mask == 255).sum()}")
            print(f"   Count of 0 in GT: {(gt_mask == 0).sum()}")
            print(f"   Pred mask: min={pred_mask_resized.min()}, max={pred_mask_resized.max()}")
            print(f"   Count of 255 in Pred: {(pred_mask_resized == 255).sum()}")
            print(f"   Count of 0 in Pred: {(pred_mask_resized == 0).sum()}")

            # ──────────────────────────────────────────────────────────────
            # Расчёт mIoU (многоклассовый)
            # ──────────────────────────────────────────────────────────────
            m_iou: MetricValue = SegmentationMetrics.calculate_multiclass_miou(
                pred_mask=pred_mask_resized,
                gt_mask=gt_mask,
                ignore_index=255,
                num_classes=150,  # ADE20K
                return_per_class=False,
            )

            # ──────────────────────────────────────────────────────────────
            # Бинарные метрики (объект vs фон)
            # ──────────────────────────────────────────────────────────────
            # Создаём бинарные маски: 1 = любой семантический класс (1-149), 0 = фон/игнор
            pred_binary = np.where((pred_mask_resized != 255) & (pred_mask_resized != 0), 1, 0).astype(np.uint8)
            gt_binary = np.where((gt_mask != 255) & (gt_mask != 0), 1, 0).astype(np.uint8)

            # Применяем valid_mask для исключения ignore-пикселей из бинарных метрик тоже
            valid_mask = (gt_mask != 255) & (pred_mask_resized != 255)
            pred_binary = pred_binary * valid_mask.astype(np.uint8)
            gt_binary = gt_binary * valid_mask.astype(np.uint8)

            metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(
                pred_mask=pred_binary,
                gt_mask=gt_binary,
                threshold=0.5,
                include_hausdorff=True,
            )

            # ──────────────────────────────────────────────────────────────
            # ДИАГНОСТИКА: СРАВНЕНИЕ РАСПРЕДЕЛЕНИЯ КЛАССОВ
            # ──────────────────────────────────────────────────────────────
            print("\n🔍 ДИАГНОСТИКА МАППИНГА КЛАССОВ")
            print("=" * 60)

            palette = ade_palette()
            class_names = get_ade_class_names()

            # Распределение в GT
            gt_unique, gt_counts = np.unique(gt_mask, return_counts=True)
            print("\n📊 Ground Truth (ADE20K стандарт):")
            for cls, cnt in sorted(zip(gt_unique, gt_counts), key=lambda x: -x[1])[:10]:
                mapped_idx = int(cls) - 1
                name = get_name_for_class(cls, class_names, offset=-1)
                print(f"   {mapped_idx:3d}: {name:20s} {cnt:7,} px ({100*cnt/gt_mask.size:5.2f}%)")

            # Распределение в предсказании
            pred_unique, pred_counts = np.unique(pred_mask_resized, return_counts=True)
            print("\n📊 Prediction (твоя модель):")
            for cls, cnt in sorted(zip(pred_unique, pred_counts), key=lambda x: -x[1])[:10]:
                mapped_idx = int(cls) - 1
                name = get_name_for_class(cls, class_names, offset=-1)
                print(f"   {mapped_idx:3d}: {name:20s} {cnt:7,} px ({100*cnt/pred_mask_resized.size:5.2f}%)")

            # Проверка на частые классы ADE20K
            ade_frequent_classes = {
                0: "wall",
                1: "building",
                2: "sky",
                3: "floor",
                4: "tree",
                5: "ceiling",
                10: "cabinet",
                15: "road",
            }
            print("\n⚠️  Проверка частых классов ADE20K:")
            for cls_id, cls_name in ade_frequent_classes.items():
                gt_pct = 100 * (gt_mask == cls_id).sum() / gt_mask.size
                pred_pct = 100 * (pred_mask_resized == cls_id).sum() / pred_mask_resized.size
                status = "✓" if abs(gt_pct - pred_pct) < 10 else "✗"
                print(f"   {status} {cls_name:12s} (#{cls_id:2d}): GT={gt_pct:5.1f}%, Pred={pred_pct:5.1f}%")

            # ──────────────────────────────────────────────────────────────
            # Сохранение результатов
            # ──────────────────────────────────────────────────────────────
            result: Dict[str, Any] = {
                "model": display_name,
                "augmentation": aug_level,
                "checkpoint": os.path.basename(checkpoint_path),
                "miou": m_iou,  # mIoU для многоклассовой
                "binary_iou": metrics.get("iou", 0),  # Бинарный IoU для совместимости
                "dice": metrics.get("dice", 0),
                "f1_score": metrics.get("f1_score", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "accuracy": metrics.get("accuracy", 0),
                "mae": metrics.get("mae", 0),
                "hausdorff": metrics.get("hausdorff_distance", 0),
                "inference_time": inference_time,
                "pred_mask": pred_mask_resized,
                "gt_mask": gt_mask,
            }
            results.append(result)

            overlay, _ = segmenter.segment_image_unified(
                test_image, alpha=0.6, class_names=NeuralSegmenter.get_ade_class_names()
            )
            overlay_images[key] = overlay

            overlay_path: str = f"{output_dir}/overlay_{display_name}_{aug_level}.jpg"
            overlay.save(overlay_path)

            print(f"      ✅ IoU: {metrics.get("iou", 0):.4f}, mIoU: {m_iou:.4f}, Dice: {metrics.get('dice', 0):.4f}")
            print(f"      ✅ Время: {inference_time:.3f}s")
            print(f"      ✅ Сохранено: {overlay_path}")

            # Создай свою визуализацию с явными подписями
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            axes[0].imshow(test_image)
            axes[0].set_title("Original Image")

            unique_gt: np.ndarray
            counts_gt: np.ndarray
            unique_gt, counts_gt = np.unique(gt_mask, return_counts=True)
            total_pixels_gt: int = gt_mask.size

            unique_pred_mask_resized: np.ndarray
            counts_pred_mask_resized: np.ndarray
            unique_pred_mask_resized, counts_pred_mask_resized = np.unique(pred_mask_resized, return_counts=True)
            total_pixels_pred_mask_resized: int = pred_mask_resized.size

            print("\n📊 Ground Truth Prediction Analysis")
            print(
                f"   Valid pixels: {total_pixels_gt:,} / {gt_mask.size:,} ({100 * total_pixels_gt / gt_mask.size:.3f}%)"
            )
            print(f"   Unique classes: {len(unique_gt)}")

            print("\n📊 Predicted Mask Prediction Analysis")
            print(
                f"   Valid pixels: {total_pixels_pred_mask_resized:,} / {pred_mask_resized.size:,} ({100 * total_pixels_pred_mask_resized / pred_mask_resized.size:.3f}%)"
            )
            print(f"   Unique classes: {len(unique_pred_mask_resized)}")

            # GT с палитрой ADE20K
            gt_color = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
            for cls in np.unique(gt_mask):
                color = get_color_for_class(cls, palette, offset=-1)  # ← offset=-1 для фикса!
                gt_color[gt_mask == cls] = color
                # palette_idx = cls - 1  # ← КЛЮЧЕВОЙ ФИКС!
                # if palette_idx < len(palette):
                #     gt_color[gt_mask == cls] = palette[palette_idx]
            axes[1].imshow(gt_color)
            axes[1].set_title("Ground Truth (ADE20K)")

            # Prediction с палитрой ADE20K
            pred_color = np.zeros((*pred_mask_resized.shape, 3), dtype=np.uint8)
            for cls in np.unique(pred_mask_resized):
                color = get_color_for_class(cls, palette, offset=-1)  # ← Тот же offset!
                pred_color[pred_mask_resized == cls] = color
                # palette_idx = cls - 1  # ← КЛЮЧЕВОЙ ФИКС!
                # if palette_idx < len(palette):
                #     pred_color[pred_mask_resized == cls] = palette[palette_idx]
            axes[2].imshow(pred_color)
            axes[2].set_title("Prediction (твоя модель)")

            plt.tight_layout()
            plt.savefig(f"{output_dir}/palette_check_{display_name}_{aug_level}.png", dpi=300)

            # Выведи топ-20 классов с именами
            print("\n🎨 Визуальная проверка (предсказано):")
            # 🔧 Используем уже вычисленные unique_pred_mask_resized и counts_pred_mask_resized
            sorted_indices_pred = np.argsort(counts_pred_mask_resized)[::-1][:20]  # Топ-20 по количеству
            for idx in sorted_indices_pred:
                cls = unique_pred_mask_resized[idx]
                cnt = counts_pred_mask_resized[idx]
                pct: float = 100 * cnt / total_pixels_pred_mask_resized
                name = get_name_for_class(cls, class_names, offset=-1)
                color = get_color_for_class(cls, palette, offset=-1)
                print(f"   Класс {int(cls)-1:3d}: {name:20s} → RGB {color}, {cnt:7,} px ({pct:5.3f}%)")

            # print("\n🎨 Визуальная проверка (предсказано):")
            # # Сортируем по убыванию количества пикселей
            # sorted_pairs = sorted(
            #     zip(unique_pred_mask_resized, counts_pred_mask_resized),
            #     key=lambda x: x[1],  # Сортировка по counts
            #     reverse=True
            # )[:20]

            # for cls, cnt in sorted_pairs:
            #     pct: float = 100 * cnt / total_pixels_pred_mask_resized
            #     name = class_names.get(cls, f"Class_{cls}")
            #     color = palette[cls] if cls < len(palette) else [0, 0, 0]
            #     print(f"   Класс {cls:3d}: {name:20s} → RGB {color}, {cnt:7,} px ({pct:5.3f}%)")

            print("\n🎨 Визуальная проверка (ground truth):")
            # 🔧 Используем уже вычисленные unique_gt и counts_gt
            sorted_indices_gt = np.argsort(counts_gt)[::-1][:20]  # Топ-20 по количеству
            for idx in sorted_indices_gt:
                cls = unique_gt[idx]
                cnt = counts_gt[idx]
                pct: float = 100 * cnt / total_pixels_gt
                name = get_name_for_class(cls, class_names, offset=-1)
                color = get_color_for_class(cls, palette, offset=-1)
                print(f"   Класс {int(cls)-1:3d}: {name:20s} → RGB {color}, {cnt:7,} px ({pct:5.3f}%)")

            # Топ классы
            del segmenter, pred_mask, pred_info
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

        except Exception as e:
            print(f"   ❌ Ошибка {key}: {e}")
            traceback.print_exc()
            continue

    if not results:
        print("\n❌ Нет результатов для анализа!")
        return None

    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 80)

    df: pd.DataFrame = pd.DataFrame(results)

    # Сводная таблица
    print("\n📊 Сводная таблица метрик (mIoU многоклассовый):")
    pivot_all: pd.DataFrame = df.pivot_table(
        values=["miou", "binary_iou", "dice", "f1_score"],
        index="model",
        columns="augmentation",
        aggfunc="mean",
    )
    print(pivot_all.round(4).to_string())
    pivot_miou: pd.DataFrame = df.pivot_table(values="miou", index="model", columns="augmentation", aggfunc="mean")
    print("\n📊 mIoU (многоклассовый):")
    print(pivot_miou.round(4).to_string())

    # 5. Визуализация
    print("\n📈 Построение графиков...")
    output_dir = "./data/augmentation_analysis2"
    os.makedirs(output_dir, exist_ok=True)

    # График 1: Сравнение IoU по уровням аугментаций
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    metrics_to_plot: List[str] = [
        "miou",
        "binary_iou",
        "dice",
        "f1_score",
        "precision",
        "recall",
        "accuracy",
    ]
    metric_names: Dict[str, str] = {
        "miou": "mIoU (multi-class)",
        "binary_iou": "IoU (binary)",
        "dice": "Dice",
        "f1_score": "F1-Score",
        "precision": "Precision",
        "recall": "Recall",
        "accuracy": "Pixel Accuracy",
    }

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        if metric not in df.columns:
            print(f"   ⚠️  Метрика '{metric}' отсутствует в данных, пропускаем график")
            ax.text(
                0.5,
                0.5,
                f"Metric '{metric}'\nnot available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
            ax.set_title(f"{metric_names.get(metric, metric)}", fontsize=11)
            ax.axis("off")
            continue

        plot_data: pd.DataFrame = df.groupby(["model", "augmentation"])[metric].first().unstack()
        plot_data.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black")
        ax.set_title(f"{metric_names[metric]} по моделям и аугментациям", fontsize=11)
        ax.set_ylabel("Score")
        ax.set_xlabel("Модель")
        ax.legend(title="Аугментации", loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/augmentation_impact_metrics.png", dpi=300, bbox_inches="tight")
    print(f"   ✅ График метрик сохранен: {output_dir}/augmentation_impact_metrics.png")
    plt.close()

    # График 2: Сравнение mIoU по моделям и аугментациям
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x="model", y="miou", hue="augmentation", palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mIoU (multi-class)")
    plt.title("Влияние аугментаций на качество сегментации (многоклассовый mIoU)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/augmentation_impact_miou.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ График mIoU сохранен: {output_dir}/augmentation_impact_miou.png")

    # График 3: Heatmap прироста
    plt.figure(figsize=(10, 8))
    pivot_gain: pd.DataFrame = pivot_miou.copy()
    if "none" in pivot_gain.columns:
        for col in ["basic", "medium"]:
            if col in pivot_gain.columns:
                pivot_gain[col] = (pivot_gain[col] - pivot_gain["none"]) / pivot_gain["none"].replace(0, 1e-8) * 100

    sns.heatmap(pivot_gain, annot=True, fmt=".4f", cmap="RdYlGn", center=0)
    plt.title("Прирост mIoU относительно 'none' (%)")
    plt.ylabel("Модель")
    plt.xlabel("Аугментация")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/augmentation_gain_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Heatmap сохранен: {output_dir}/augmentation_gain_heatmap.png")

    # График 4: Heatmap прироста
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap_data: pd.DataFrame = df.pivot_table(index="model", columns="augmentation", values="miou", aggfunc="first")

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "mIoU Score"},
    )

    ax.set_title("Влияние аугментаций на mIoU", fontsize=14, fontweight="bold")
    ax.set_xlabel("Уровень аугментаций")
    ax.set_ylabel("Модель")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/augmentation_heatmap_miou.png", dpi=300, bbox_inches="tight")
    print(f"   ✅ Heatmap сохранен: {output_dir}/augmentation_heatmap_miou.png")
    plt.close()

    # График 5: Сравнение времени выполнения
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="model", y="inference_time", hue="augmentation", palette="coolwarm")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Время (сек)")
    plt.title("Время инференса по моделям и аугментациям")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/inference_time.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ График времени сохранен: {output_dir}/inference_time.png")

    # График 6: Разница между уровнями аугментаций
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Разница basic - none
    df_analysis: pd.DataFrame = df.copy()

    # Считаем прирост
    for model in df_analysis["model"].unique():
        model_data: pd.DataFrame = df_analysis[df_analysis["model"] == model]

        none_iou = model_data[model_data["augmentation"] == "none"]["miou"].values
        basic_iou = model_data[model_data["augmentation"] == "basic"]["miou"].values
        medium_iou = model_data[model_data["augmentation"] == "medium"]["miou"].values

        if len(none_iou) > 0 and len(basic_iou) > 0:
            basic_gain: float = (basic_iou[0] - none_iou[0]) * 100
            medium_gain: float = (medium_iou[0] - none_iou[0]) * 100 if len(medium_iou) > 0 else 0

            axes[0].bar(f"{model}\n(basic-none)", basic_gain, alpha=0.7)
            axes[1].bar(f"{model}\n(medium-none)", medium_gain, alpha=0.7, color="orange")

    axes[0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[0].set_title("Прирост IoU: Basic vs None (%)", fontsize=11)
    axes[0].set_ylabel("Прирост IoU (%)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[1].set_title("Прирост IoU: Medium vs None (%)", fontsize=11)
    axes[1].set_ylabel("Прирост IoU (%)")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/augmentation_gain.png", dpi=300, bbox_inches="tight")
    print(f"   ✅ График прироста сохранен: {output_dir}/augmentation_gain.png")
    plt.close()

    # ──────────────────────────────────────────────────────────────
    # СОХРАНЕНИЕ СРАВНЕНИЯ ВИЗУАЛИЗАЦИЙ
    # ──────────────────────────────────────────────────────────────
    print("\n🖼️ Сохранение сравнения визуализаций...")

    for model in df["model"].unique():
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"{model}: сравнение аугментаций", fontsize=14, fontweight="bold")

        for idx, aug in enumerate(["none", "basic", "medium"]):
            key = f"{model}_{aug}"
            if key in overlay_images:
                axes[idx].imshow(overlay_images[key])
                axes[idx].set_title(f"{aug.upper()}", fontsize=11)
                axes[idx].axis("off")
            else:
                axes[idx].text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=axes[idx].transAxes,
                )
                axes[idx].set_title(f"{aug.upper()}", fontsize=11)
                axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison_{model}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Сравнение для {model}: {output_dir}/comparison_{model}.png")

    # ──────────────────────────────────────────────────────────────
    # СТАТИСТИЧЕСКИЙ АНАЛИЗ
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("СТАТИСТИЧЕСКИЙ АНАЛИЗ")
    print("=" * 80)

    # Средний mIoU по уровням аугментаций
    avg_none: float = df[df["augmentation"] == "none"]["miou"].mean()
    avg_basic: float = df[df["augmentation"] == "basic"]["miou"].mean()
    avg_medium: float = df[df["augmentation"] == "medium"]["miou"].mean()

    print("\n📊 Средний mIoU по уровням аугментаций:")
    print(f"   None:   {avg_none:.4f}")
    print(f"   Basic:  {avg_basic:.4f} (прирост: {(avg_basic - avg_none) * 100:+.2f}%)")
    print(f"   Medium: {avg_medium:.4f} (прирост: {(avg_medium - avg_none) * 100:+.2f}%)")

    # Лучшая комбинация
    best_idx = df["miou"].idxmax()
    best_row: pd.Series = df.loc[best_idx]

    print("\n🏆 Лучшая комбинация:")
    print(f"   Модель: {best_row['model']}")
    print(f"   Аугментации: {best_row['augmentation']}")
    print(f"   mIoU (multi-class): {best_row['miou']:.4f}")

    # Экспорт результатов
    df.to_csv(f"{output_dir}/augmentation_impact_results.csv", index=False)
    print(f"\n💾 Результаты сохранены: {output_dir}/augmentation_impact_results.csv")

    # Генерация отчёта в Markdown
    report_path: str = f"{output_dir}/report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Отчёт: Влияние аугментаций на качество сегментации\n\n")
        f.write(f"Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Сводная таблица mIoU\n\n")
        f.write(pivot_miou.round(4).to_markdown() + "\n\n")
        f.write("## Статистика\n\n")
        f.write(f"- Средний mIoU (None): {avg_none:.4f}\n")
        f.write(f"- Средний mIoU (Basic): {avg_basic:.4f}\n")
        f.write(f"- Средний mIoU (Medium): {avg_medium:.4f}\n\n")
        f.write(f"### Лучшая комбинация: `{best_row['model']}_{best_row['augmentation']}`\n")
        f.write(f"- mIoU: **{best_row['miou']:.4f}**\n")

    print(f"📄 Отчёт сохранён: {report_path}")

    return df, overlay_images


# ──────────────────────────────────────────────────────────────────────
def save_augmentation_comparison_grid(
    overlay_images: Dict[str, Image.Image],
    output_dir: PathLike = "./data/augmentation_analysis2",
    model_names: Optional[List[str]] = None,
) -> None:
    """Создаёт единую сетку сравнения всех моделей.

    Макет:
    ```
    [Модель 1] [none] [basic] [medium]
    [Модель 2] [none] [basic] [medium]
    ...
    ```

    Args:
        overlay_images: Словарь `{key: PIL.Image}` с визуализациями.
        output_dir: Директория для сохранения.
        model_names: Опциональный список имён моделей (если `None`, извлекается из ключей).
    """
    output_path: Path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if model_names is None:

        def extract_model_name(key: str) -> str:
            """Извлекает имя модели из ключа 'model_aug' (учитывает подчёркивания в имени)."""
            # Разделяем по последнему подчёркиванию: "fpn_smp_none" -> ("fpn_smp", "none")
            parts = key.rsplit("_", 1)
            return parts[0] if len(parts) > 1 else key

        models: List[str] = list(set(extract_model_name(k) for k in overlay_images.keys()))
    else:
        models = model_names

    if not models:
        print("⚠️  Нет моделей для визуализации")
        return

    n_models: int = len(models)
    _, axes = plt.subplots(n_models, 3, figsize=(15, 5 * n_models), squeeze=False)

    for row, model in enumerate(models):
        for col, aug in enumerate(["none", "basic", "medium"]):
            key: str = f"{model}_{aug}"
            ax = axes[row, col]

            if key in overlay_images:
                ax.imshow(overlay_images[key])
                ax.set_title(f"{aug.upper()}", fontsize=10, fontweight="bold", color="darkblue")
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
                ax.set_title(f"{aug.upper()}", fontsize=10, fontweight="bold", color="gray")
                ax.axis("off")

        axes[row, 0].text(
            -0.08,
            0.5,
            model.upper(),
            ha="right",
            va="center",
            transform=axes[row, 0].transAxes,
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.7),
        )

    plt.suptitle(
        "Сравнение влияния аугментаций на визуализацию сегментации",
        fontsize=14,
        y=1.01,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(
        f"{output_dir}/full_comparison_grid.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()
    print(f"✅ Полная сетка сравнения: {output_dir}/full_comparison_grid.png")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔍 CUDA DIAGNOSTICS:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    result_df: Optional[pd.DataFrame]
    overlay_images_result: Optional[Dict[str, Image.Image]]
    analysis_result = analyze_augmentation_impact()

    if analysis_result is None:
        print("\n❌ Анализ не вернул результатов (возможно, не найдены чекпоинты или произошла ошибка).")
        exit(1)

    results_df, overlay_images_result = analysis_result
    if overlay_images_result:
        print("\n🔍 DEBUG: overlay_images keys:")
        for k in sorted(overlay_images_result.keys())[:10]:
            print(f"   {k}")

        print("\n🔍 DEBUG: извлечённые модели:")

        def extract_model_name(key: str) -> str:
            """Вычленяет имя модели из ключа."""
            parts: List[str] = key.rsplit("_", 1)
            return parts[0] if len(parts) > 1 else key

        models: List[str] = list(set(extract_model_name(k) for k in overlay_images_result.keys()))
        print(f"   {models}")

        if results_df is not None:
            model_names: List[str] = results_df["model"].unique().tolist()
            save_augmentation_comparison_grid(overlay_images_result, model_names=model_names)
            print("\n✅ Анализ завершён!")
