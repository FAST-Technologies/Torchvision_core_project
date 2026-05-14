# analyze.py

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
from pathlib import Path
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    Union,
)

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
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Настройка путей проекта
project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Локальные импорты
from segmenters.NeuralSegmenter import NeuralSegmenter
from metrics.SegmentationMetrics import SegmentationMetrics

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & CONSTANTS
# ──────────────────────────────────────────────────────────────────────
MaskArray = np.ndarray
ImageArray = np.ndarray
MetricValue = float
MetricsDict = Dict[str, MetricValue]
PathLike = Union[str, Path]

# Маппинг имён чекпоинтов на ModelType enum (для NeuralSegmenter)
MODEL_TYPE_MAPPING: Dict[str, str] = {
    "unet_smp": "unet_smp",
    "fpn_smp": "fpn_smp",
    "psp_smp": "pspnet_smp",  # 🔧 Исправлено: psp_smp → pspnet_smp
    "deeplab_tv": "deeplab_tv",
    "fcn_tv": "fcn_tv",
    "segnet": "segnet",
}


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

    test_image: Image.Image = Image.open(img_path).convert("RGB")
    gt_mask_pil: Image.Image = Image.open(mask_path)
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
                sh, sw = (
                    gt_mask.shape[0] / pred_mask.shape[0],
                    gt_mask.shape[1] / pred_mask.shape[1],
                )
                pred_mask_resized: MaskArray = zoom(pred_mask, (sh, sw), order=0)
            else:
                pred_mask_resized = pred_mask

            # ──────────────────────────────────────────────────────────────
            # Расчёт mIoU (многоклассовый)
            # ──────────────────────────────────────────────────────────────
            classes: np.ndarray = np.unique(np.concatenate([gt_mask, pred_mask_resized]))
            iou_per_class: List[float] = []

            for cls in classes:
                if cls == 255:  # ignore index
                    continue
                pred_cls: np.ndarray = (pred_mask_resized == cls).astype(np.uint8)
                gt_cls: np.ndarray = (gt_mask == cls).astype(np.uint8)

                intersection: int = int(np.logical_and(pred_cls, gt_cls).sum())
                union: int = int(np.logical_or(pred_cls, gt_cls).sum())

                if union > 0:
                    iou_per_class.append(intersection / union)

            m_iou: MetricValue = float(np.mean(iou_per_class)) if iou_per_class else 0.0

            # ──────────────────────────────────────────────────────────────
            # Бинарные метрики (объект vs фон)
            # ──────────────────────────────────────────────────────────────
            pred_binary: np.ndarray = (pred_mask_resized > 0).astype(np.uint8)
            gt_binary: np.ndarray = (gt_mask > 0).astype(np.uint8)

            metrics: MetricsDict = SegmentationMetrics.calculate_all_metrics(
                pred_mask=pred_binary,
                gt_mask=gt_binary,
                threshold=0.5,
                include_hausdorff=True,
            )

            # ──────────────────────────────────────────────────────────────
            # Сохранение результатов
            # ──────────────────────────────────────────────────────────────
            result: Dict[str, Any] = {
                "model": display_name,
                "augmentation": aug_level,
                "checkpoint": os.path.basename(checkpoint_path),
                "iou": m_iou,  # mIoU для многоклассовой
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

            output_dir: str = "./data/augmentation_analysis"
            os.makedirs(output_dir, exist_ok=True)
            overlay_path: str = f"{output_dir}/overlay_{display_name}_{aug_level}.jpg"
            overlay.save(overlay_path)

            # overlay_path: Path = output_dir / f"overlay_{display_name}_{aug_level}.jpg"
            # overlay.save(overlay_path)

            print(f"      ✅ IoU (mIoU): {m_iou:.4f}, Dice: {metrics.get('dice', 0):.4f}")
            print(f"      ✅ Время: {inference_time:.3f}s")
            print(f"      ✅ Сохранено: {overlay_path}")

            # 🔹 ОЧИСТКА ПАМЯТИ
            del segmenter, pred_mask, pred_info
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

        except Exception as e:
            print(f"   ❌ Ошибка {key}: {e}")
            import traceback

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
    print("\n📊 Сводная таблица метрик (mIoU):")
    pivot_all: pd.DataFrame = df.pivot_table(
        values=["iou", "dice", "f1_score"],
        index="model",
        columns="augmentation",
        aggfunc="mean",
    )
    print(pivot_all.round(4).to_string())
    pivot_iou: pd.DataFrame = df.pivot_table(values="iou", index="model", columns="augmentation", aggfunc="mean")
    print(pivot_iou.round(4).to_string())

    # 5. Визуализация
    print("\n📈 Построение графиков...")
    output_dir = "./data/augmentation_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # График 1: Сравнение IoU по уровням аугментаций
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics_to_plot: List[str] = [
        "iou",
        "dice",
        "f1_score",
        "precision",
        "recall",
        "accuracy",
    ]
    metric_names: Dict[str, str] = {
        "iou": "IoU",
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
    sns.barplot(data=df, x="model", y="iou", hue="augmentation", palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mIoU")
    plt.title("Влияние аугментаций на качество сегментации (mIoU)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/augmentation_impact_miou.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ График mIoU сохранен: {output_dir}/augmentation_impact_miou.png")

    # График 3: Heatmap прироста
    plt.figure(figsize=(10, 8))
    pivot_gain: pd.DataFrame = pivot_iou.copy()
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
    heatmap_data: pd.DataFrame = df.pivot_table(index="model", columns="augmentation", values="iou", aggfunc="first")

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "IoU Score"},
    )

    ax.set_title("Влияние аугментаций на IoU", fontsize=14, fontweight="bold")
    ax.set_xlabel("Уровень аугментаций")
    ax.set_ylabel("Модель")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/augmentation_heatmap_iou.png", dpi=300, bbox_inches="tight")
    print(f"   ✅ Heatmap сохранен: {output_dir}/augmentation_heatmap_iou.png")
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

        none_iou = model_data[model_data["augmentation"] == "none"]["iou"].values
        basic_iou = model_data[model_data["augmentation"] == "basic"]["iou"].values
        medium_iou = model_data[model_data["augmentation"] == "medium"]["iou"].values

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
    avg_none: float = df[df["augmentation"] == "none"]["iou"].mean()
    avg_basic: float = df[df["augmentation"] == "basic"]["iou"].mean()
    avg_medium: float = df[df["augmentation"] == "medium"]["iou"].mean()

    print("\n📊 Средний mIoU по уровням аугментаций:")
    print(f"   None:   {avg_none:.4f}")
    print(f"   Basic:  {avg_basic:.4f} (прирост: {(avg_basic - avg_none) * 100:+.2f}%)")
    print(f"   Medium: {avg_medium:.4f} (прирост: {(avg_medium - avg_none) * 100:+.2f}%)")

    # Лучшая комбинация
    best_idx = df["iou"].idxmax()
    best_row: pd.Series = df.loc[best_idx]

    print("\n🏆 Лучшая комбинация:")
    print(f"   Модель: {best_row['model']}")
    print(f"   Аугментации: {best_row['augmentation']}")
    print(f"   mIoU: {best_row['iou']:.4f}")

    # Экспорт результатов
    df.to_csv(f"{output_dir}/augmentation_impact_results.csv", index=False)
    print(f"\n💾 Результаты сохранены: {output_dir}/augmentation_impact_results.csv")

    # Генерация отчёта в Markdown
    report_path: str = f"{output_dir}/report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Отчёт: Влияние аугментаций на качество сегментации\n\n")
        f.write(f"Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Сводная таблица mIoU\n\n")
        f.write(pivot_iou.round(4).to_markdown() + "\n\n")
        f.write("## Статистика\n\n")
        f.write(f"- Средний mIoU (None): {avg_none:.4f}\n")
        f.write(f"- Средний mIoU (Basic): {avg_basic:.4f}\n")
        f.write(f"- Средний mIoU (Medium): {avg_medium:.4f}\n\n")
        f.write(f"### Лучшая комбинация: `{best_row['model']}_{best_row['augmentation']}`\n")
        f.write(f"- mIoU: **{best_row['iou']:.4f}**\n")

    print(f"📄 Отчёт сохранён: {report_path}")

    return df, overlay_images


# ──────────────────────────────────────────────────────────────────────
def save_augmentation_comparison_grid(
    overlay_images: Dict[str, Image.Image],
    output_dir: PathLike = "./data/augmentation_analysis",
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
    fig, axes = plt.subplots(n_models, 3, figsize=(15, 5 * n_models), squeeze=False)

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
