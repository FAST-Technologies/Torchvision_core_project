# utils/utils.py

# Импорт основных библиотек
import torch
import numpy as np
import pandas as pd
from typing import (
    Optional,
    Dict,
    Any,
    Union,
    List,
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
MaskArray = np.ndarray  # Semantic mask: H×W, dtype int/uint8
LogitsTensor = Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]
ClassNamesDict = Optional[Dict[Union[int, str], str]]


def compute_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> Dict[str, Any]:
    """
    Вычисляет основные метрики семантической сегментации.

    Поддерживаемые метрики:
    - Pixel Accuracy: Доля правильно классифицированных пикселей.
    - Mean IoU: Среднее пересечение-над-объединением по классам.
    - Weighted F1-Score: F1-мера с учётом дисбаланса классов.
    - Per-class IoU: IoU для каждого класса отдельно.
    - Confusion Matrix: Матрица ошибок размера `num_classes × num_classes`.

    Args:
        pred_mask: Предсказанная маска `[H, W]` с целочисленными метками классов.
        gt_mask: Ground truth маска `[H, W]` с целочисленными метками.
        num_classes: Общее количество классов в задаче.
        ignore_index: Индекс пикселей, которые следует игнорировать при расчёте (обычно 255).

    Returns:
        Dict[str, Any]: Словарь с метриками:
        ```python
        {
            "mIoU": float,                    # Mean IoU (nan если нет валидных пикселей)
            "pixel_acc": float,               # Pixel Accuracy
            "f1_weighted": float,             # Weighted F1-Score
            "per_class_iou": np.ndarray,      # Array[num_classes] с IoU по классам
            "confusion_matrix": np.ndarray,   # Matrix[num_classes, num_classes]
            "unique_pred_classes": int,       # Количество уникальных классов в предсказании
            "valid_pixels": int,              # Количество неигнорируемых пикселей
        }
        ```

    Note:
        - Ground truth значения автоматически обрезаются до `[0, num_classes-1]` с предупреждением при выходе за диапазон.
        - Если все пиксели игнорируются, возвращается `nan` для числовых метрик.
        - `per_class_iou` содержит `nan` для классов, не представленных в предсказании или GT.

    Example:
        ```python
        metrics = compute_metrics(pred, gt, num_classes=150, ignore_index=255)
        print(f"IoU: {metrics['mIoU']:.3f}, Acc: {metrics['pixel_acc']:.3f}")
        ```
    """
    # Маска валидных пикселей (исключаем ignore_index)
    valid = gt_mask != ignore_index
    if not np.any(valid):
        return {
            "mIoU": np.nan,
            "pixel_acc": np.nan,
            "f1_weighted": np.nan,
            "per_class_iou": [np.nan] * num_classes,
            "confusion_matrix": None,
            "unique_pred_classes": len(np.unique(pred_mask)),
            "valid_pixels": 0,
        }

    pred_valid = pred_mask[valid]
    gt_valid = np.clip(gt_mask[valid], 0, num_classes - 1)

    # Значения в ground truth должны быть в диапазоне [0, num_classes-1]
    gt_min, gt_max = gt_valid.min(), gt_valid.max()
    if gt_min < 0 or gt_max >= num_classes:
        print(
            f"⚠️ Warning: gt_mask values out of range [{gt_min}, {gt_max}], expected [0, {num_classes - 1}]"
        )
        gt_valid = np.clip(gt_valid, 0, num_classes - 1)

    # Pixel Accuracy
    pixel_acc: float = accuracy_score(gt_valid, pred_valid)

    # Confusion matrix
    cm: np.ndarray = confusion_matrix(
        gt_valid, pred_valid, labels=list(range(num_classes))
    )

    # Per-class IoU
    iou_per_class: List[float] = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        if tp + fp + fn == 0:
            iou_per_class.append(np.nan)
        else:
            iou_per_class.append(tp / (tp + fp + fn))

    # Mean IoU
    mIoU: float = np.nanmean(iou_per_class)

    # Weighted F1-score
    f1: float = f1_score(
        gt_valid,
        pred_valid,
        average="weighted",
        labels=list(range(num_classes)),
        zero_division=0,
    )

    return {
        "mIoU": mIoU,
        "pixel_acc": pixel_acc,
        "f1_weighted": f1,
        "per_class_iou": np.array(iou_per_class),
        "confusion_matrix": cm,
        "unique_pred_classes": len(np.unique(pred_mask)),
        "valid_pixels": int(np.sum(valid)),
    }


def extract_logits_info(
    outputs: LogitsTensor,
    model_type: str,
) -> Dict[str, Any]:
    """
    Извлекает статистику о логитах из выходных данных модели.

    Поддерживаемые типы моделей:
    - HuggingFace Transformers: `segformer`, `mask2former`, `oneformer`, `dpt`, `upernet`
    - Torchvision: `deeplab_tv`, `fcn_tv`, `maskrcnn_tv`
    - SMP/Custom: `unet_smp`, `fpn_mit`, `psp_mit`, `segnet`, ...
    - Instance segmentation: `sam`, `mobile_sam`, `sam2` (возвращает metadata без статистики)

    Args:
        outputs: Выходные данные модели (Tensor, Dict, Tuple или специфичная структура).
        model_type: Строковый идентификатор типа модели.

    Returns:
        Dict[str, Any]: Словарь с информацией:
        ```python
        {
            "type": str,              # Тип объекта логов
            "shape": Tuple[int, ...], # Форма тензора логов (если применимо)
            "min": float,             # Минимальное значение
            "max": float,             # Максимальное значение
            "mean": float,            # Среднее значение
            "std": float,             # Стандартное отклонение
            "device": str,            # Устройство тензора ("cuda:0", "cpu")
            "num_classes": Optional[int],  # Количество выходных каналов (если определено)
        }
        ```
        Для instance-сегментации (SAM, Mask R-CNN) возвращается metadata без числовой статистики.

    Note:
        - Все вычисления выполняются на CPU после `.cpu().float()`.
        - Для защиты от `nan` используется `np.nanmin`/`nanmax` и fallback для старых версий PyTorch.
    """
    try:
        # ──────────────────────────────────────────────────────────────
        # Извлечение логов в зависимости от типа модели
        # ──────────────────────────────────────────────────────────────
        logits: Optional[torch.Tensor] = None
        # === HuggingFace модели ===
        if model_type in [
            "segformer",
            "segformer_b2",
            "mask2former",
            "oneformer",
            "dpt",
            "upernet",
            "maskformer",
        ]:
            logits = (
                outputs.logits
                if hasattr(outputs, "logits")
                else (
                    outputs[0]
                    if isinstance(outputs, tuple) and len(outputs) > 0
                    else None
                )
            )
        # === Torchvision DeepLab / FCN ===
        elif model_type in ["deeplab_tv", "fcn_tv"]:
            # Torchvision: dict["out"][0] или tensor[0]
            logits = (
                outputs["out"][0]
                if isinstance(outputs, dict) and "out" in outputs
                else (outputs[0] if hasattr(outputs, "__getitem__") else outputs)
            )
        # === Torchvision Mask R-CNN ===
        elif model_type == "maskrcnn_tv":
            return {
                "type": "Mask R-CNN (instance)",
                "note": "No class logits — instance segmentation with masks",
                "output_structure": "list[dict] with 'masks', 'labels', 'scores'",
            }
        elif model_type in ["sam", "mobile_sam", "sam2"]:
            return {
                "type": "SAM (instance masks)",
                "note": "No class logits — promptable instance segmentation",
            }
        elif model_type in [
            "unet_smp",
            "mit_smp",
            "fpn_mit",
            "fpn_effnet",
            "psp_mit",
            "psp_effnet",
            "deeplab_smp",
            "segnet",
            "segnet_custom",
        ]:
            # SMP/Custom: ожидаем Tensor формы [B, C, H, W]
            logits = (
                outputs
                if isinstance(outputs, torch.Tensor) and outputs.dim() == 4
                else None
            )
        else:
            # Fallback: если outputs — Tensor, используем его
            logits = outputs if isinstance(outputs, torch.Tensor) else None

        if logits is None:
            return {"type": "None", "note": "logits not found"}

        # ──────────────────────────────────────────────────────────────
        # Статистика логов
        # ──────────────────────────────────────────────────────────────
        logits_cpu = logits.cpu().float()
        logits_np = logits_cpu.numpy()
        try:
            # PyTorch >= 1.9 с поддержкой nan-статистик
            min_val = float(np.nanmin(logits_np))
            max_val = float(np.nanmax(logits_np))
            mean_val = float(np.nanmean(logits_np))
            std_val = float(np.nanstd(logits_np))
        except AttributeError:
            # Fallback для PyTorch < 1.9
            flat = logits_cpu.flatten()
            flat = flat[~torch.isnan(flat)]
            min_val = float(flat.min()) if len(flat) > 0 else float("nan")
            max_val = float(flat.max()) if len(flat) > 0 else float("nan")
            mean_val = float(flat.mean()) if len(flat) > 0 else float("nan")
            std_val = float(flat.std()) if len(flat) > 0 else float("nan")
        return {
            "type": type(logits).__name__,
            "shape": tuple(logits_cpu.shape),
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
            "device": str(logits.device),
            "num_classes": logits_cpu.shape[1] if logits_cpu.dim() >= 4 else None,
        }
    except Exception as e:
        return {"error": str(e), "type": "extraction_failed"}


def analyze_prediction(
    mask: MaskArray,
    class_names: ClassNamesDict = None,
    ignore_index: int = 255,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Детальный анализ предсказанной маски сегментации.

    Выводит в консоль:
    - Количество валидных пикселей и процент покрытия.
    - Топ-K классов по количеству пикселей с именами (если заданы).
    - Предупреждение о доминирующем классе (>50% пикселей).

    Args:
        mask: Предсказанная маска `[H, W]` с целочисленными метками.
        class_names: Словарь `{класс: имя}` для отображения человекочитаемых названий.
        ignore_index: Индекс пикселей для исключения из анализа.
        top_k: Количество топ-классов для вывода.

    Returns:
        Dict[str, Any]: Словарь с результатами анализа:
        ```python
        {
            "total_pixels": int,           # Количество валидных пикселей
            "unique_classes": int,         # Количество уникальных классов
            "class_counts": Dict[int, int],  # {класс: количество_пикселей}
            "dominant_class": Optional[int],  # Класс с максимальным покрытием (если >50%)
        }
        ```

    Note:
        - Если все пиксели игнорируются, возвращается пустой словарь и выводится предупреждение.
        - Имена классов берутся из `class_names` или генерируются как `"Class_{id}"`.
    """
    # Фильтрация ignore_index
    valid_mask = mask != ignore_index
    mask_valid = mask[valid_mask]

    if len(mask_valid) == 0:
        print("⚠️  No valid pixels (all ignored)")
        return {}

    # Статистика
    unique, counts = np.unique(mask_valid, return_counts=True)
    total: int = len(mask_valid)

    print("\n📊 Prediction Analysis")
    print(
        f"   Valid pixels: {total:,} / {mask.size:,} ({100 * total / mask.size:.3f}%)"
    )
    print(f"   Unique classes: {len(unique)}")

    # Топ классы
    print(f"\n   Top {top_k} classes by pixel count:")
    sorted_idx = np.argsort(counts)[::-1][:top_k]

    for idx in sorted_idx:
        cls = unique[idx]
        cnt = counts[idx]
        pct = 100 * cnt / total
        name = class_names.get(cls, f"Class_{cls}") if class_names else f"Class_{cls}"
        print(f"     {cls:3d}: {name:25s} {cnt:7,} px ({pct:5.3f}%)")

    # Проверка на доминирующий класс
    dominant_class: Optional[int] = None
    if len(counts) > 0 and counts[0] / total > 0.5:
        dominant_cls = int(unique[np.argmax(counts)])
        dominant_class = dominant_cls
        print(
            f"\n   ⚠️  Dominant class: {dominant_cls} ({100 * counts.max() / total:.3f}% of pixels)"
        )
        print("      This may indicate under-segmentation or background bias")

    return {
        "total_pixels": int(total),
        "unique_classes": len(unique),
        "class_counts": {int(c): int(n) for c, n in zip(unique, counts)},
        "dominant_class": dominant_class,
    }


def generate_class_report(
    mask: MaskArray,
    class_names: ClassNamesDict = None,
    ignore_index: int = 255,
    min_pixels: int = 100,
) -> Dict[str, Any]:
    """
    Генерирует детальный отчёт по предсказанным классам для экспорта.

    Фильтрует шумовые классы (< `min_pixels` пикселей) и сортирует по убыванию покрытия.

    Args:
        mask: Предсказанная маска `[H, W]`.
        class_names: Словарь `{класс: имя}` для человекочитаемых названий.
        ignore_index: Индекс игнорируемых пикселей.
        min_pixels: Минимальное количество пикселей для включения класса в отчёт.

    Returns:
        Dict[str, Any]: Словарь с отчётом:
        ```python
        {
            "total_valid_pixels": int,     # Количество валидных пикселей
            "total_image_pixels": int,     # Общее количество пикселей в изображении
            "coverage_pct": float,         # Процент покрытия валидными пикселями
            "unique_classes": int,         # Количество классов в отчёте (после фильтрации)
            "top_class": Optional[str],    # Имя класса с максимальным покрытием
            "top_class_pct": Optional[float],  # Процент покрытия топ-класса
            "dataframe": pd.DataFrame,     # DataFrame с детальной статистикой по классам
        }
        ```

    Note:
        - Если нет валидных пикселей, возвращается `{"error": "⚠️ No valid pixels"}`.
        - DataFrame содержит колонки: `class_id`, `class_name`, `pixel_count`, `percentage`, `rank`.
    """
    # Фильтрация
    valid = mask != ignore_index
    mask_valid = mask[valid]

    if len(mask_valid) == 0:
        return {"error": "⚠️ No valid pixels"}

    # Статистика
    unique, counts = np.unique(mask_valid, return_counts=True)
    total = len(mask_valid)

    # Сбор данных
    rows: List[Dict[str, Any]] = []
    for cls, cnt in zip(unique, counts):
        if cnt >= min_pixels:  # Фильтрация шума
            name = (
                class_names.get(cls, f"Class_{cls}") if class_names else f"Class_{cls}"
            )
            rows.append(
                {
                    "class_id": int(cls),
                    "class_name": name,
                    "pixel_count": int(cnt),
                    "percentage": round(100 * cnt / total, 2),
                    "rank": None,
                }
            )

    # Сортировка и ранжирование
    df: pd.DataFrame = pd.DataFrame(rows).sort_values("pixel_count", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    summary: Dict[str, Any] = {
        "total_valid_pixels": int(total),
        "total_image_pixels": int(mask.size),
        "coverage_pct": round(100 * total / mask.size, 2),
        "unique_classes": len(df),
        "top_class": df.iloc[0]["class_name"] if len(df) > 0 else None,
        "top_class_pct": df.iloc[0]["percentage"] if len(df) > 0 else None,
        "dataframe": df,
    }
    return summary


def export_class_report(
    report: Dict[str, Any],
    output_file: str,
    format: str = "csv",
) -> None:
    """
    Экспортирует отчёт по классам в файл.

    Поддерживаемые форматы:
    - `"csv"`: Таблица в формате CSV (UTF-8).
    - `"markdown"`: Markdown-таблица с заголовком и сводкой.
    - `"json"`: JSON с разделением сводки и детальных данных.

    Args:
        report: Отчёт из `generate_class_report()`.
        output_file: Путь к файлу для сохранения.
        format: Формат экспорта (`"csv"`, `"markdown"` или `"json"`).

    Raises:
        KeyError: Если в `report` отсутствует ключ `"dataframe"`.

    Example:
        ```python
        report = generate_class_report(pred_mask, class_names=ade_classes)
        export_class_report(report, "report.md", format="markdown")
        ```
    """
    if "dataframe" not in report:
        print(f"⚠️ Invalid report structure: missing 'dataframe' key")
        return
    df = report["dataframe"]
    if format == "csv":
        df.to_csv(output_file, index=False, encoding="utf-8")
    elif format == "markdown":
        md = df.to_markdown(index=False)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Class Prediction Report\n\n")
            f.write(
                f"**Valid pixels:** {report['total_valid_pixels']:,} / {report['total_image_pixels']:,} ({report['coverage_pct']}%)\n\n"
            )
            f.write(md)
    elif format == "json":
        import json

        export_data: Dict[str, Any] = {
            "summary": {k: v for k, v in report.items() if k != "dataframe"},
            "classes": report["dataframe"].to_dict(orient="records"),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                export_data,
                f,
                indent=2,
                ensure_ascii=False,
            )
    else:
        print(f"⚠️ Unsupported format: {format}. Use 'csv', 'markdown', or 'json'.")
        return
    print(f"✅ Report exported to {output_file}")
