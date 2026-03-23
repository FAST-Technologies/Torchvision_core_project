# inference/utils.py
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from PIL import Image
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, confusion_matrix

def compute_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int,
    ignore_index: int = 255
) -> Dict[str, Any]:
    """
    Вычисляет основные метрики семантической сегментации.
    
    Args:
        pred_mask: [H, W] предсказанные метки классов
        gt_mask: [H, W] ground truth метки
        num_classes: общее число классов
        ignore_index: индекс для игнорируемых пикселей (фон/неизвестно)
    
    Returns:
        dict с метриками
    """
    # Маска валидных пикселей
    valid = (gt_mask != ignore_index)
    if not np.any(valid):
        return {
            "mIoU": np.nan, 
            "pixel_acc": np.nan, 
            "f1_weighted": np.nan,
            "per_class_iou": [np.nan] * num_classes,
            "confusion_matrix": None,
            "unique_pred_classes": len(np.unique(pred_mask)),
            "valid_pixels": 0
        }
    
    pred_valid = pred_mask[valid]
    gt_valid = np.clip(gt_mask[valid], 0, num_classes - 1)

    # значения в ground truth должны быть в диапазоне [0, num_classes-1]
    gt_min, gt_max = gt_valid.min(), gt_valid.max()
    if gt_min < 0 or gt_max >= num_classes:
        print(f"⚠️ Warning: gt_mask values out of range [{gt_min}, {gt_max}], expected [0, {num_classes-1}]")
        gt_valid = np.clip(gt_valid, 0, num_classes - 1)

    # Pixel Accuracy
    pixel_acc = accuracy_score(gt_valid, pred_valid)
    
    # Confusion matrix
    cm = confusion_matrix(gt_valid, pred_valid, labels=range(num_classes))
    
    # Per-class IoU
    iou_per_class = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        if tp + fp + fn == 0:
            iou_per_class.append(np.nan)
        else:
            iou_per_class.append(tp / (tp + fp + fn))
    
    # Mean IoU
    mIoU = np.nanmean(iou_per_class)

    # Weighted F1-score
    f1 = f1_score(gt_valid, pred_valid, average='weighted', labels=range(num_classes), zero_division=0)
    
    return {
        "mIoU": mIoU,
        "pixel_acc": pixel_acc,
        "f1_weighted": f1,
        "per_class_iou": np.array(iou_per_class),
        "confusion_matrix": cm,
        "unique_pred_classes": len(np.unique(pred_mask)),
        "valid_pixels": int(np.sum(valid))
    }


def extract_logits_info(outputs, model_type: str) -> Dict[str, Any]:
    """
    Извлекает информацию о логитах из outputs модели.
    
    Returns:
        dict с информацией: shape, min, max, mean, std
    """
    try:
        # === HuggingFace модели ===
        if model_type in ["segformer", "segformer_b2", "mask2former", "oneformer", "dpt", "upernet", "maskformer"]:
            logits = outputs.logits if hasattr(outputs, 'logits') else (outputs[0] if isinstance(outputs, tuple) and len(outputs) > 0 else None)
        # === Torchvision DeepLab / FCN ===
        elif model_type in ["deeplab_tv", "fcn_tv"]:
            logits = outputs['out'][0] if isinstance(outputs, dict) and 'out' in outputs else (outputs[0] if hasattr(outputs, '__getitem__') else outputs)
        # === Torchvision Mask R-CNN ===
        elif model_type == "maskrcnn_tv":
            return {
                "type": "Mask R-CNN (instance)", 
                "note": "No class logits — instance segmentation with masks",
                "output_structure": "list[dict] with 'masks', 'labels', 'scores'"
            }
        elif model_type in ["sam", "mobile_sam", "sam2"]:
            return {
                "type": "SAM (instance masks)", 
                "note": "No class logits — promptable instance segmentation"
            }
        elif model_type in ["unet_smp", "mit_smp", "fpn_mit", "fpn_effnet", 
                            "psp_mit", "psp_effnet", "deeplab_smp", 
                            "segnet", "segnet_custom"]:
            logits = outputs if isinstance(outputs, torch.Tensor) and outputs.dim() == 4 else None
        else:
            logits = outputs if isinstance(outputs, torch.Tensor) else None
        
        if logits is None:
            return {"type": "None", "note": "logits not found"}
        
        logits_cpu = logits.cpu().float()
        try:
            min_val = float(torch.nanmin(logits_cpu))
            max_val = float(torch.nanmax(logits_cpu))
            mean_val = float(torch.nanmean(logits_cpu))
            std_val = float(torch.nanstd(logits_cpu))
        except AttributeError:
            # Fallback для PyTorch < 1.9
            flat = logits_cpu.flatten()
            flat = flat[~torch.isnan(flat)]
            min_val = float(flat.min()) if len(flat) > 0 else float('nan')
            max_val = float(flat.max()) if len(flat) > 0 else float('nan')
            mean_val = float(flat.mean()) if len(flat) > 0 else float('nan')
            std_val = float(flat.std()) if len(flat) > 0 else float('nan')
        return {
            "type": type(logits).__name__,
            "shape": tuple(logits_cpu.shape),
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
            "device": str(logits.device),
            "num_classes": logits_cpu.shape[1] if logits_cpu.dim() >= 4 else None
        }
    except Exception as e:
        return {"error": str(e), "type": "extraction_failed"}
    
def analyze_prediction(
    mask: np.ndarray, 
    class_names: dict = None, 
    ignore_index: int = 255, 
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Детальный анализ предсказанной маски.
    
    Args:
        mask: np.ndarray [H, W] с метками классов
        class_names: dict {class_id: name} для отображения имён
        ignore_index: индекс для игнорируемых пикселей
        top_k: показать топ-K классов по количеству пикселей
    """
    # Фильтрация ignore_index
    valid_mask = mask != ignore_index
    mask_valid = mask[valid_mask]
    
    if len(mask_valid) == 0:
        print("⚠️  No valid pixels (all ignored)")
        return
    
    # Статистика
    unique, counts = np.unique(mask_valid, return_counts=True)
    total: int = len(mask_valid)
    
    print(f"\n📊 Prediction Analysis")
    print(f"   Valid pixels: {total:,} / {mask.size:,} ({100*total/mask.size:.1f}%)")
    print(f"   Unique classes: {len(unique)}")
    
    # Топ классы
    print(f"\n   Top {top_k} classes by pixel count:")
    sorted_idx = np.argsort(counts)[::-1][:top_k]
    
    for idx in sorted_idx:
        cls = unique[idx]
        cnt = counts[idx]
        pct = 100 * cnt / total
        name = class_names.get(cls, f"Class_{cls}") if class_names else f"Class_{cls}"
        print(f"     {cls:3d}: {name:25s} {cnt:7,} px ({pct:5.1f}%)")
    
    # Проверка на доминирующий класс
    if len(counts) > 0 and counts[0] / total > 0.5:
        dominant_cls = unique[np.argmax(counts)]
        print(f"\n   ⚠️  Dominant class: {dominant_cls} ({100*counts.max()/total:.1f}% of pixels)")
        print(f"      This may indicate under-segmentation or background bias")
    
    return {
        "total_pixels": int(total),
        "unique_classes": len(unique),
        "class_counts": {int(c): int(n) for c, n in zip(unique, counts)},
        "dominant_class": int(unique[np.argmax(counts)]) if len(counts) > 0 else None
    }

def generate_class_report(
    mask: np.ndarray, 
    class_names: dict = None, 
    ignore_index: int = 255, 
    min_pixels: int = 100
) -> Dict[str, Any]:
    """
    Генерирует детальный отчёт по предсказанным классам.
    
    Returns:
        dict со статистикой для дальнейшего анализа
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
    rows = []
    for cls, cnt in zip(unique, counts):
        if cnt >= min_pixels:  # Фильтрация шума
            name = class_names.get(cls, f"Class_{cls}") if class_names else f"Class_{cls}"
            rows.append({
                "class_id": int(cls),
                "class_name": name,
                "pixel_count": int(cnt),
                "percentage": round(100 * cnt / total, 2),
                "rank": None
            })
    
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
        "dataframe": df
    }
    
    return summary

def export_class_report(
    report: dict, 
    output_file: str, 
    format: str = "csv"
) -> None:
    """Экспорт отчёта по классам в файл"""
    df = report["dataframe"]
    if format == "csv":
        df.to_csv(output_file, index=False, encoding="utf-8")
    elif format == "markdown":
        md = df.to_markdown(index=False)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Class Prediction Report\n\n")
            f.write(f"**Valid pixels:** {report['total_valid_pixels']:,} / {report['total_image_pixels']:,} ({report['coverage_pct']}%)\n\n")
            f.write(md)
    elif format == "json":
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {k: v for k, v in report.items() if k != "dataframe"},
                "classes": report["dataframe"].to_dict(orient="records")
            }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Report exported to {output_file}")
