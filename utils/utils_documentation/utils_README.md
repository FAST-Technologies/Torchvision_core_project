# 🛠️ Utils — Утилиты для анализа и оценки сегментации

## 📖 Описание
Модуль `utils/utils.py` предоставляет **набор вспомогательных функций** для вычисления метрик, анализа предсказаний и экспорта отчётов в задачах семантической сегментации.

> ⚠️ **Важно:** Данный модуль является **инфраструктурным** — он не выполняет сегментацию, а анализирует её результаты. Используйте его в связке с `SegmentationMetrics`, `NeuralSegmenter` или `BatchClassicTester`.

## ✨ Ключевые возможности
### 📊 Метрики качества сегментации

| Метрика | Описание | Диапазон | Когда использовать |
|---------|----------|----------|-------------------|
| **Pixel Accuracy** | Доля верно классифицированных пикселей | [0.0, 1.0] | Быстрая оценка, сбалансированные классы |
| **Mean IoU** | Среднее IoU по всем классам | [0.0, 1.0] | Основной критерий для ADE20K/COCO |
| **Weighted F1** | F1-мера с учётом поддержки классов | [0.0, 1.0] | Дисбаланс классов, медицинская сегментация |
| **Per-class IoU** | IoU для каждого класса отдельно | [0.0, 1.0] | Анализ проблемных классов |
| **Confusion Matrix** | Матрица ошибок `num_classes × num_classes` | — | Детальная диагностика ошибок модели |

### 🔍 Анализ предсказаний
- **Топ-K классов**: Вывод наиболее представленных классов с именами и процентами.
- **Доминирующий класс**: Предупреждение если один класс занимает >50% пикселей.
- **Фильтрация шума**: Исключение классов с `< min_pixels` пикселями из отчётов.
- **Поддержка `ignore_index`**: Автоматическое исключение игнорируемых пикселей (255).

### 📤 Экспорт отчётов
| Формат | Метод | Особенности |
|--------|-------|-------------|
| `csv` | `export_class_report(..., format="csv")` | UTF-8, совместим с Excel/Pandas |
| `markdown` | `export_class_report(..., format="markdown")` | Красивая таблица + сводка для GitHub/документации |
| `json` | `export_class_report(..., format="json")` | Структурированные данные: `summary` + `classes` |

### 🧠 Статистика логов для разных типов моделей
| Тип модели | Поддержка | Что извлекается |
|------------|-----------|-----------------|
| **HF Transformers** | ✅ `segformer`, `mask2former`, `oneformer`, `dpt`, `upernet` | `logits`: shape, min/max/mean/std, device, num_classes |
| **Torchvision** | ✅ `deeplab_tv`, `fcn_tv`, `maskrcnn_tv` | `outputs["out"]` или tensor, metadata для instance-сегментации |
| **SMP/Custom** | ✅ `unet_smp`, `fpn_mit`, `psp_mit`, `segnet` | Tensor `[B, C, H, W]`: полная статистика |
| **Instance Segmentation** | ✅ `sam`, `mobile_sam`, `sam2` | Metadata без числовой статистики (нет class logits) |

## 🚀 Быстрый старт
### Вычисление метрик сегментации
```python
from utils.utils import compute_metrics
import numpy as np

# Предсказание и ground truth
pred_mask = np.random.randint(0, 150, size=(512, 512), dtype=np.uint8)
gt_mask = np.random.randint(0, 150, size=(512, 512), dtype=np.uint8)
gt_mask[gt_mask == 149] = 255  # Добавляем игнорируемые пиксели

# Вычисление метрик
metrics = compute_metrics(
    pred_mask=pred_mask,
    gt_mask=gt_mask,
    num_classes=150,
    ignore_index=255
)

print(f"Mean IoU: {metrics['mIoU']:.3f}")
print(f"Pixel Acc: {metrics['pixel_acc']:.3f}")
print(f"F1 Weighted: {metrics['f1_weighted']:.3f}")
print(f"Valid pixels: {metrics['valid_pixels']:,}")
```

### Анализ предсказанной маски
```python
from utils.utils import analyze_prediction
from segmenters.NeuralSegmenter import NeuralSegmenter

# Загрузка модели и предсказание
segmenter = NeuralSegmenter(model_type="segformer")
mask = segmenter.segment("street_scene.jpg")  # BinaryMask

# Анализ с именами классов
class_names = NeuralSegmenter.get_ade_class_names()
result = analyze_prediction(
    mask=mask,
    class_names=class_names,
    ignore_index=255,
    top_k=10
)
# Вывод в консоль:
# 📊 Prediction Analysis
#    Valid pixels: 245,760 / 262,144 (93.750%)
#    Unique classes: 42
#
#    Top 10 classes by pixel count:
#      0: wall                      89,234 px (36.321%)
#      2: sky                       45,123 px (18.367%)
#      5: tree                      23,456 px ( 9.543%)
#      ...
```

### Генерация и экспорт отчёта
```python
from utils.utils import generate_class_report, export_class_report

# Генерация отчёта
report = generate_class_report(
    mask=mask,
    class_names=class_names,
    ignore_index=255,
    min_pixels=100  # Фильтруем шумовые классы
)

# Экспорт в разные форматы
export_class_report(report, "report.csv", format="csv")
export_class_report(report, "report.md", format="markdown")
export_class_report(report, "report.json", format="json")

# Доступ к DataFrame для дальнейшего анализа
df = report["dataframe"]
print(df.head())
#    class_id  class_name  pixel_count  percentage  rank
# 0         0        wall        89234       36.32     1
# 1         2         sky        45123       18.37     2
# ...
```

### Извлечение статистики логов модели
```python
from utils.utils import extract_logits_info
import torch

# Выход модели (пример для SegFormer)
outputs = model(input_tensor)  # ModelOutput with logits

# Извлечение информации
logits_info = extract_logits_info(
    outputs=outputs,
    model_type="segformer"
)

print(f"Logits shape: {logits_info['shape']}")
print(f"Range: [{logits_info['min']:.3f}, {logits_info['max']:.3f}]")
print(f"Mean ± Std: {logits_info['mean']:.3f} ± {logits_info['std']:.3f}")
print(f"Device: {logits_info['device']}, Classes: {logits_info['num_classes']}")
```

## ⚙️ Конфигурация
### Параметры `compute_metrics()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `pred_mask` | `np.ndarray` | — | Предсказанная маска `[H, W]`, dtype=int/uint8 |
| `gt_mask` | `np.ndarray` | — | Ground truth маска `[H, W]`, dtype=int/uint8 |
| `num_classes` | `int` | — | Общее количество классов в задаче |
| `ignore_index` | `int` | `255` | Индекс пикселей для исключения из расчёта |

### Параметры `extract_logits_info()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `outputs` | `LogitsTensor` | — | Выход модели: `Tensor`, `Dict`, `Tuple` или `ModelOutput` |
| `model_type` | `str` | — | Идентификатор типа модели для корректного парсинга |

### Параметры `analyze_prediction()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `mask` | `MaskArray` | — | Предсказанная маска `[H, W]` |
| `class_names` | `Optional[Dict]` | `None` | Словарь `{класс: имя}` для человекочитаемых названий |
| `ignore_index` | `int` | `255` | Индекс игнорируемых пикселей |
| `top_k` | `int` | `10` | Количество топ-классов для вывода в консоль |

### Параметры `generate_class_report()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `mask` | `MaskArray` | — | Предсказанная маска `[H, W]` |
| `class_names` | `Optional[Dict]` | `None` | Словарь имён классов |
| `ignore_index` | `int` | `255` | Индекс игнорируемых пикселей |
| `min_pixels` | `int` | `100` | Минимальное количество пикселей для включения класса в отчёт |

### Параметры `export_class_report()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `report` | `Dict[str, Any]` | — | Отчёт из `generate_class_report()` |
| `output_file` | `str` | — | Путь к файлу для сохранения |
| `format` | `str` | `"csv"` | Формат экспорта: `"csv"`, `"markdown"`, `"json"` |

### Type Aliases
```python
MaskArray = np.ndarray  # Semantic mask: H×W, dtype int/uint8
LogitsTensor = Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]
ClassNamesDict = Optional[Dict[Union[int, str], str]]
```

## 📚 Справочник функций
### 🔹 Метрики и оценка
| Функция | Параметры | Описание | Возвращает |
|---------|-----------|----------|-----------|
| `compute_metrics()` | `pred_mask`, `gt_mask`, `num_classes`, `ignore_index` | Вычисление основных метрик сегментации | `Dict[str, Any]`: mIoU, accuracy, F1, confusion matrix |

### 🔹 Анализ логов и предсказаний
| Функция | Параметры | Описание | Возвращает |
|---------|-----------|----------|-----------|
| `extract_logits_info()` | `outputs`, `model_type` | Извлечение статистики из выходных данных модели | `Dict[str, Any]`: shape, min/max/mean/std, device |
| `analyze_prediction()` | `mask`, `class_names`, `ignore_index`, `top_k` | Детальный анализ маски с выводом в консоль | `Dict[str, Any]`: total_pixels, unique_classes, class_counts |

### 🔹 Отчёты и экспорт
| Функция | Параметры | Описание | Возвращает |
|---------|-----------|----------|-----------|
| `generate_class_report()` | `mask`, `class_names`, `ignore_index`, `min_pixels` | Генерация структурированного отчёта по классам | `Dict[str, Any]`: summary + `pd.DataFrame` |
| `export_class_report()` | `report`, `output_file`, `format` | Экспорт отчёта в файл (CSV/Markdown/JSON) | `None` (сохраняет файл) |

## 🔄 Конвейер анализа: предсказание → метрики → отчёт → экспорт
### Вычисление метрик с обработкой edge cases
```python
def compute_metrics(pred_mask, gt_mask, num_classes, ignore_index=255):
    # 1. Фильтрация игнорируемых пикселей
    valid = gt_mask != ignore_index
    if not np.any(valid):
        return {"mIoU": np.nan, "pixel_acc": np.nan, ...}
    
    # 2. Валидация диапазона GT
    gt_valid = np.clip(gt_mask[valid], 0, num_classes - 1)
    
    # 3. Confusion matrix и метрики
    cm = confusion_matrix(gt_valid, pred_mask[valid], labels=range(num_classes))
    
    # 4. Per-class IoU с обработкой нулевых знаменателей
    iou_per_class = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
        iou_per_class.append(iou)
    
    return {
        "mIoU": np.nanmean(iou_per_class),
        "pixel_acc": accuracy_score(gt_valid, pred_mask[valid]),
        "f1_weighted": f1_score(..., average="weighted"),
        "per_class_iou": np.array(iou_per_class),
        "confusion_matrix": cm,
        ...
    }
```

### Извлечение логов для разных типов моделей
```python
def extract_logits_info(outputs, model_type):
    # Dispatch по типу модели
    if model_type in ["segformer", "mask2former", ...]:
        logits = outputs.logits if hasattr(outputs, "logits") else None
    elif model_type in ["deeplab_tv", "fcn_tv"]:
        logits = outputs["out"] if isinstance(outputs, dict) else outputs
    elif model_type in ["sam", "sam2"]:
        return {"type": "SAM (instance masks)", "note": "No class logits"}
    
    # Статистика на CPU с защитой от nan
    logits_cpu = logits.cpu().float()
    logits_np = logits_cpu.numpy()
    
    try:
        # PyTorch >= 1.9
        return {
            "shape": tuple(logits_cpu.shape),
            "min": float(np.nanmin(logits_np)),
            "max": float(np.nanmax(logits_np)),
            "mean": float(np.nanmean(logits_np)),
            "std": float(np.nanstd(logits_np)),
            ...
        }
    except AttributeError:
        # Fallback для старых версий
        flat = logits_cpu.flatten()[~torch.isnan(logits_cpu.flatten())]
        return {...}
```

### Генерация отчёта с фильтрацией шума
```python
def generate_class_report(mask, class_names=None, ignore_index=255, min_pixels=100):
    # Фильтрация валидных пикселей
    valid = mask != ignore_index
    mask_valid = mask[valid]
    
    if len(mask_valid) == 0:
        return {"error": "⚠️ No valid pixels"}
    
    # Подсчёт классов
    unique, counts = np.unique(mask_valid, return_counts=True)
    total = len(mask_valid)
    
    # Сбор данных с фильтрацией шума
    rows = []
    for cls, cnt in zip(unique, counts):
        if cnt >= min_pixels:
            name = class_names.get(cls, f"Class_{cls}") if class_names else f"Class_{cls}"
            rows.append({
                "class_id": int(cls),
                "class_name": name,
                "pixel_count": int(cnt),
                "percentage": round(100 * cnt / total, 2),
            })
    
    # Создание DataFrame с ранжированием
    df = pd.DataFrame(rows).sort_values("pixel_count", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    
    return {
        "total_valid_pixels": int(total),
        "coverage_pct": round(100 * total / mask.size, 2),
        "unique_classes": len(df),
        "top_class": df.iloc[0]["class_name"] if len(df) > 0 else None,
        "dataframe": df,
    }
```

## 📊 Интерпретация метрик
### Mean IoU (mIoU)
```python
# Отличная сегментация: >0.70
# Хорошая: 0.50–0.70
# Удовлетворительная: 0.30–0.50
# Плохая: <0.30

if metrics["mIoU"] > 0.7:
    print("✅ Excellent segmentation quality")
elif metrics["mIoU"] > 0.5:
    print("⚠️ Good, but room for improvement")
else:
    print("❌ Poor segmentation, check model/training")
```

### Confusion Matrix анализ
```python
cm = metrics["confusion_matrix"]

# Ошибки между конкретными классами
false_positives = cm[:, class_id].sum() - cm[class_id, class_id]
false_negatives = cm[class_id, :].sum() - cm[class_id, class_id]

print(f"Class {class_id}: FP={false_positives}, FN={false_negatives}")

# Визуализация (опционально)
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
```

### Предупреждение о доминирующем классе
```python
# В analyze_prediction():
if counts[0] / total > 0.5:
    print(f"⚠️ Dominant class: {dominant_cls} ({pct:.3f}% of pixels)")
    print("   This may indicate under-segmentation or background bias")

# Рекомендации:
# - Проверить баланс классов в датасете
# - Увеличить weight для редких классов в лоссе
# - Рассмотреть oversampling миноритарных классов
```

## ⚡ Производительность и оптимизации
### Сложность вычислений
| Функция | Сложность | Примечание |
|---------|-----------|-----------|
| `compute_metrics()` | O(N + C²) | N = пиксели, C = классы; матрица ошибок — самое дорогое |
| `extract_logits_info()` | O(N) | Линейный проход по тензору логов |
| `analyze_prediction()` | O(N log N) | Сортировка уникальных классов |
| `generate_class_report()` | O(N + K log K) | K = количество классов после фильтрации |

### Рекомендации по оптимизации
1. **Сэмплирование для больших изображений**:
   ```python
   # Для изображений >2000×2000 рассмотрите случайное сэмплирование
   if mask.size > 4_000_000:
       indices = np.random.choice(mask.size, size=1_000_000, replace=False)
       mask_sampled = mask.flat[indices]
       metrics = compute_metrics(mask_sampled, gt_mask.flat[indices], ...)
   ```

2. **Кэширование confusion matrix**:
   ```python
   # Если нужно несколько метрик на одной матрице, вычислите её один раз
   cm = confusion_matrix(...)
   iou = cm.diagonal() / (cm.sum(axis=1) + cm.sum(axis=0) - cm.diagonal())
   ```

3. **Использование `ignore_index` для ускорения**:
   ```python
   # Исключение 255-пикселей сокращает объём вычислений
   valid = gt_mask != 255
   # Дальше работаем только с mask[valid], gt_mask[valid]
   ```

4. **Векторизация вместо циклов**:
   ```python
   # ✅ Векторизованно:
   iou = tp / (tp + fp + fn + 1e-10)
   
   # ❌ Избегайте:
   for c in range(num_classes):
       ...
   ```

## 🛠️ Обработка ошибок и устойчивость
### Валидация входных данных
```python
# В compute_metrics():
gt_min, gt_max = gt_valid.min(), gt_valid.max()
if gt_min < 0 or gt_max >= num_classes:
    print(f"⚠️ Warning: gt_mask values out of range [{gt_min}, {gt_max}]")
    gt_valid = np.clip(gt_valid, 0, num_classes - 1)
```

### Обработка пустых/невалидных данных
```python
# Если все пиксели игнорируются:
if not np.any(valid):
    return {
        "mIoU": np.nan,
        "pixel_acc": np.nan,
        "f1_weighted": np.nan,
        "per_class_iou": [np.nan] * num_classes,
        ...
    }

# Если класс не представлен в данных:
if tp + fp + fn == 0:
    iou_per_class.append(np.nan)  # Вместо деления на 0
```

### Fallback для старых версий PyTorch
```python
try:
    # PyTorch >= 1.9
    min_val = float(np.nanmin(logits_np))
except AttributeError:
    # Fallback для < 1.9
    flat = logits_cpu.flatten()
    flat = flat[~torch.isnan(flat)]
    min_val = float(flat.min()) if len(flat) > 0 else float("nan")
```

### Рекомендации по отладке
1. **Проверьте диапазон значений масок**:
   ```python
   print(f"Pred: [{pred_mask.min()}, {pred_mask.max()}]")
   print(f"GT: [{gt_mask.min()}, {gt_mask.max()}]")
   # Должно быть в [0, num_classes-1] или = ignore_index
   ```

2. **Убедитесь в совпадении размеров**:
   ```python
   assert pred_mask.shape == gt_mask.shape, \
       f"Shape mismatch: {pred_mask.shape} vs {gt_mask.shape}"
   ```

3. **Проверьте наличие валидных пикселей**:
   ```python
   valid_ratio = np.sum(gt_mask != ignore_index) / gt_mask.size
   if valid_ratio < 0.01:
       print(f"⚠️ Only {valid_ratio*100:.2f}% valid pixels — metrics may be unreliable")
   ```

4. **Используйте `per_class_iou` для диагностики**:
   ```python
   for c, iou in enumerate(metrics["per_class_iou"]):
       if not np.isnan(iou) and iou < 0.3:
           print(f"⚠️ Class {c} has low IoU: {iou:.3f}")
   ```

## 🤝 Зависимости
```text
numpy>=1.20          # Массивы, статистики, метрики
pandas>=1.3          # DataFrame для отчётов, экспорт
torch>=1.9           # Извлечение логов, nan-статистики
scikit-learn>=1.0    # accuracy_score, f1_score, confusion_matrix
tabulate>=0.8        # Markdown-экспорт (опционально, через pandas)
```

### Опциональные зависимости для визуализации
```bash
# Для визуализации confusion matrix
pip install seaborn matplotlib

# Для расширенного экспорта
pip install tabulate  # Улучшенное форматирование markdown
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование utils.py |
|--------|----------------------|
| `SegmentationMetrics` | Базовые метрики; `utils.compute_metrics()` расширяет функционал |
| `NeuralSegmenter` | Вызов `analyze_prediction()` после `predict_segmentation_map()` |
| `BatchClassicTester2` | Пакетное вычисление метрик через `compute_metrics()` |
| `ModelTrainer` | Валидация mIoU на валидационном наборе |
| `CpuCudaBenchmark` | Сравнение метрик между CPU/GPU инференсом |

## 📄 Лицензия

Проект распространяется под лицензией **MIT**. См. файл [LICENSE](LICENSE) для деталей.

```
MIT License

Copyright (c) 2026 Torchvision_core_project contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---