# analyze.py

"""
Вспомогательный скрипт для проверки влияния аугментаций на качество обучения сегментационных моделей.
"""

# Импорт основных библиотек
import glob
import os
import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from segmenters.NeuralSegmenter import NeuralSegmenter
from metrics.SegmentationMetrics import SegmentationMetrics
import numpy as np
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
import time
from typing import Tuple, Dict, Any


def analyze_augmentation_impact() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Исследование влияния аугментаций на качество сегментации
    Исправленная версия с поддержкой всех моделей и корректными метриками
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

    # Маппинг имён чекпоинтов на ModelType enum
    MODEL_TYPE_MAPPING: Dict[str, str] = {
        "unet_smp": "unet_smp",
        "fpn_smp": "fpn_smp",
        "psp_smp": "pspnet_smp",
        "deeplab_tv": "deeplab_tv",
        "fcn_tv": "fcn_tv",
        "segnet": "segnet",
    }

    checkpoints: dict = {}

    print("\n🔍 Поиск чекпоинтов...")
    for model_type in model_types:
        for aug_level in augmentation_levels:
            pattern: str = f"{models_dir}/{model_type}_{aug_level}_*.pth"
            files: List[str] = glob.glob(pattern)

            if files:
                latest_checkpoint = max(files, key=os.path.getctime)
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
    img_path: str = hf_hub_download(
        repo_id=repo_id, filename="ADE_val_00000001.jpg", repo_type="dataset"
    )
    mask_path: str = hf_hub_download(
        repo_id=repo_id, filename="ADE_val_00000001.png", repo_type="dataset"
    )

    test_image: Image.Image = Image.open(img_path).convert("RGB")
    gt_mask_pil: Image.Image = Image.open(mask_path)
    gt_mask: np.ndarray = np.array(gt_mask_pil)
    if gt_mask.ndim == 3 and gt_mask.shape[2] == 3:
        # RGB маска → берём первый канал или конвертируем
        gt_mask = gt_mask[:, :, 0]

    print(f"   ✅ Изображение: {test_image.size}")
    print(f"   ✅ Маска: {gt_mask.shape}, unique values: {len(np.unique(gt_mask))}")

    print("\n🧪 Оценка моделей...")
    results = []
    overlay_images = {}

    for key, checkpoint_info in checkpoints.items():
        try:
            model_type = checkpoint_info["model_type"]
            aug_level = checkpoint_info["augmentation"]
            checkpoint_path = checkpoint_info["path"]
            display_name = checkpoint_info["original_type"]

            print(f"\n   🔹 {key} {display_name}_{aug_level}...")

            segmenter = NeuralSegmenter(
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                device="cuda",
                num_classes=150,
                palette=NeuralSegmenter.ade_palette(),
            )

            # Предсказание
            start_time = time.time()
            pred_mask, pred_info = segmenter.predict_segmentation_map(
                test_image, verbose=False, gt_mask=gt_mask
            )
            inference_time = time.time() - start_time

            pred_mask_2 = segmenter.segment(np.array(test_image))

            if gt_mask.shape != pred_mask.shape:
                # Ресайз предсказания под размер GT
                from scipy.ndimage import zoom

                sh, sw = (
                    gt_mask.shape[0] / pred_mask.shape[0],
                    gt_mask.shape[1] / pred_mask.shape[1],
                )
                pred_mask_resized = zoom(pred_mask, (sh, sw), order=0)
            else:
                pred_mask_resized = pred_mask

            # Рассчитываем mIoU
            classes = np.unique(np.concatenate([gt_mask, pred_mask_resized]))
            iou_per_class = []

            for cls in classes:
                if cls == 255:  # ignore index
                    continue
                pred_cls = (pred_mask_resized == cls).astype(np.uint8)
                gt_cls = (gt_mask == cls).astype(np.uint8)

                intersection = np.logical_and(pred_cls, gt_cls).sum()
                union = np.logical_or(pred_cls, gt_cls).sum()

                if union > 0:
                    iou_per_class.append(intersection / union)

            m_iou = np.mean(iou_per_class) if iou_per_class else 0.0

            # Бинарные метрики (объект vs фон)
            pred_binary = (pred_mask_resized > 0).astype(np.uint8)
            gt_binary = (gt_mask > 0).astype(np.uint8)

            metrics = SegmentationMetrics.calculate_all_metrics(
                pred_mask=pred_binary,
                gt_mask=gt_binary,
                threshold=0.5,
                include_hausdorff=True,
            )

            # Сохраняем результаты
            result = {
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

            print(
                f"      ✅ IoU (mIoU): {m_iou:.4f}, Dice: {metrics.get('dice', 0):.4f}"
            )
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
    pivot_all = df.pivot_table(
        values=["iou", "dice", "f1_score"],
        index="model",
        columns="augmentation",
        aggfunc="mean",
    )
    print(pivot_all.round(4).to_string())
    pivot_iou = df.pivot_table(
        values="iou", index="model", columns="augmentation", aggfunc="mean"
    )
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

        plot_data = df.groupby(["model", "augmentation"])[metric].first().unstack()
        plot_data.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black")
        ax.set_title(f"{metric_names[metric]} по моделям и аугментациям", fontsize=11)
        ax.set_ylabel("Score")
        ax.set_xlabel("Модель")
        ax.legend(title="Аугментации", loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/augmentation_impact_metrics.png", dpi=300, bbox_inches="tight"
    )
    print(f"   ✅ График метрик сохранен: {output_dir}/augmentation_impact_metrics.png")
    plt.close()

    # График 2: Сравнение mIoU по моделям и аугментациям
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x="model", y="iou", hue="augmentation", palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mIoU")
    plt.title("Влияние аугментаций на качество сегментации (mIoU)")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/augmentation_impact_miou.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"   ✅ График mIoU сохранен: {output_dir}/augmentation_impact_miou.png")

    # График 3: Heatmap прироста
    plt.figure(figsize=(10, 8))
    pivot_gain = pivot_iou.copy()
    if "none" in pivot_gain.columns:
        for col in ["basic", "medium"]:
            if col in pivot_gain.columns:
                pivot_gain[col] = (
                    (pivot_gain[col] - pivot_gain["none"])
                    / pivot_gain["none"].replace(0, 1e-8)
                    * 100
                )

    sns.heatmap(pivot_gain, annot=True, fmt=".4f", cmap="RdYlGn", center=0)
    plt.title("Прирост mIoU относительно 'none' (%)")
    plt.ylabel("Модель")
    plt.xlabel("Аугментация")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/augmentation_gain_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"   ✅ Heatmap сохранен: {output_dir}/augmentation_gain_heatmap.png")

    # График 4: Heatmap прироста
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap_data = df.pivot_table(
        index="model", columns="augmentation", values="iou", aggfunc="first"
    )

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
    plt.savefig(
        f"{output_dir}/augmentation_heatmap_iou.png", dpi=300, bbox_inches="tight"
    )
    print(f"   ✅ Heatmap сохранен: {output_dir}/augmentation_heatmap_iou.png")
    plt.close()

    # График 5: Сравнение времени выполнения
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df, x="model", y="inference_time", hue="augmentation", palette="coolwarm"
    )
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
        model_data = df_analysis[df_analysis["model"] == model]

        none_iou = model_data[model_data["augmentation"] == "none"]["iou"].values
        basic_iou = model_data[model_data["augmentation"] == "basic"]["iou"].values
        medium_iou = model_data[model_data["augmentation"] == "medium"]["iou"].values

        if len(none_iou) > 0 and len(basic_iou) > 0:
            basic_gain = (basic_iou[0] - none_iou[0]) * 100
            medium_gain = (
                (medium_iou[0] - none_iou[0]) * 100 if len(medium_iou) > 0 else 0
            )

            axes[0].bar(f"{model}\n(basic-none)", basic_gain, alpha=0.7)
            axes[1].bar(
                f"{model}\n(medium-none)", medium_gain, alpha=0.7, color="orange"
            )

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
        plt.savefig(
            f"{output_dir}/comparison_{model}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"   ✅ Сравнение для {model}: {output_dir}/comparison_{model}.png")

    print("\n" + "=" * 80)
    print("СТАТИСТИЧЕСКИЙ АНАЛИЗ")
    print("=" * 80)

    # Средний mIoU по уровням аугментаций
    avg_none = df[df["augmentation"] == "none"]["iou"].mean()
    avg_basic = df[df["augmentation"] == "basic"]["iou"].mean()
    avg_medium = df[df["augmentation"] == "medium"]["iou"].mean()

    print(f"\n📊 Средний mIoU по уровням аугментаций:")
    print(f"   None:   {avg_none:.4f}")
    print(f"   Basic:  {avg_basic:.4f} (прирост: {(avg_basic-avg_none)*100:+.2f}%)")
    print(f"   Medium: {avg_medium:.4f} (прирост: {(avg_medium-avg_none)*100:+.2f}%)")

    # Лучшая комбинация
    best_idx = df["iou"].idxmax()
    best_row = df.loc[best_idx]

    print(f"\n🏆 Лучшая комбинация:")
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
        f.write(
            f"### Лучшая комбинация: `{best_row['model']}_{best_row['augmentation']}`\n"
        )
        f.write(f"- mIoU: **{best_row['iou']:.4f}**\n")

    print(f"📄 Отчёт сохранён: {report_path}")

    return df, overlay_images


def save_augmentation_comparison_grid(
    overlay_images, output_dir="./data/augmentation_analysis", model_names=None
) -> None:
    """
    Создаёт единую сетку сравнения всех моделей.

    Args:
        overlay_images: dict {key: PIL.Image}
        output_dir: папка для сохранения
        model_names: опциональный список имён моделей (если None, извлекается из ключей)
    """
    if model_names is None:

        def extract_model_name(key):
            """Извлекает имя модели из ключа 'model_aug' (учитывает подчёркивания в имени)"""
            # Разделяем по последнему подчёркиванию: "fpn_smp_none" -> ("fpn_smp", "none")
            parts = key.rsplit("_", 1)
            return parts[0] if len(parts) > 1 else key

        models = list(set(extract_model_name(k) for k in overlay_images.keys()))
    else:
        models = model_names

    if not models:
        print("⚠️  Нет моделей для визуализации")
        return

    n_models: int = len(models)
    fig, axes = plt.subplots(n_models, 3, figsize=(15, 5 * n_models), squeeze=False)

    for row, model in enumerate(models):
        for col, aug in enumerate(["none", "basic", "medium"]):
            key = f"{model}_{aug}"
            ax = axes[row, col]

            if key in overlay_images:
                ax.imshow(overlay_images[key])
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
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(
        f"{output_dir}/full_comparison_grid.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()
    print(f"✅ Полная сетка сравнения: {output_dir}/full_comparison_grid.png")


if __name__ == "__main__":
    print("\n🔍 CUDA DIAGNOSTICS:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(
            f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    results_df, overlay_images_result = analyze_augmentation_impact()
    print("\n🔍 DEBUG: overlay_images keys:")
    for k in sorted(overlay_images_result.keys())[:10]:
        print(f"   {k}")

    print(f"\n🔍 DEBUG: извлечённые модели:")

    def extract_model_name(key):
        parts = key.rsplit("_", 1)
        return parts[0] if len(parts) > 1 else key

    models = list(set(extract_model_name(k) for k in overlay_images_result.keys()))
    print(f"   {models}")
    if results_df is not None:
        model_names = results_df["model"].unique().tolist()
        save_augmentation_comparison_grid(
            overlay_images_result, model_names=model_names
        )
        print("\n✅ Анализ завершён!")
