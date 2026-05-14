# 📊 SegmentationMetrics — Метрики качества сегментации

## 📖 Описание
Модуль `SegmentationMetrics.py` предоставляет **универсальный набор метрик** для оценки качества алгоритмов семантической и бинарной сегментации. Все методы являются статическими — класс используется как пространство имён, не требующее инициализации.

> ⚠️ **Важно:** Модуль работает с *парами масок* (предсказание ↔ Ground Truth). Для массового тестирования на датасетах используйте обёртки: `BatchClassicTester2`, `SegmentationBenchmark`, `SegmentationTester`.

## ✨ Ключевые возможности
- 🎯 **Полный набор метрик:**
  | Категория | Метрики | Диапазон | Интерпретация |
  |-----------|---------|----------|---------------|
  | Пересечение | `IoU`, `Jaccard` | [0.0, 1.0] | Чем выше, тем лучше |
  | Сходство | `Dice` / `F1` | [0.0, 1.0] | Устойчив к дисбалансу классов |
  | Классификация | `Precision`, `Recall`, `Accuracy` | [0.0, 1.0] | Точность / Полнота / Доля верных пикселей |
  | Ошибки | `MAE` | [0.0, 1.0] | Средняя абсолютная ошибка (чем ниже, тем лучше) |
  | Геометрия | `Hausdorff Distance` | [0, ∞) | Макс. расстояние между контурами (чувствительна к выбросам) |
  | Статистика | `Area`, `Ratio`, `Difference` | [0, ∞) | Площади и их соотношения |
  | Конфузия | `TP`, `FP`, `FN`, `TN` | [0, ∞) | Элементы матрицы ошибок |
  | Кластеризация* | `Silhouette`, `CH`, `DB` | [-1,1] / [0,∞) / [0,∞) | Только для >2 классов |

- 🔄 **Автоматическая нормализация:** Поддержка масок в форматах `[0, 1]` (float) и `[0, 255]` (uint8) с адаптивным порогом бинаризации.
- 🛡️ **Устойчивость к крайним случаям:** Защита от деления на ноль, обработка пустых контуров, возврат `np.nan` для неинформативных метрик.
- 🔍 **Верификация реализации:** Опциональное сравнение кастомных формул с эталонными функциями `sklearn.metrics` через параметр `verbose`.
- ⚡ **Оптимизация Hausdorff:** Расстояние вычисляется только по координатам контуров (не по всей маске), что ускоряет расчёт в 10–100×.
- 📦 **Групповой расчёт:** Метод `calculate_all_metrics()` возвращает полный словарь метрик за один проход по данным.
- 📊 **Пакетная оценка:** `evaluate_multiple_masks()` для агрегации средних/стандартных значений по набору изображений.

## 🚀 Быстрый старт
### Базовое использование: расчёт всех метрик
```python
import numpy as np
from metrics.SegmentationMetrics import SegmentationMetrics

# Пример масок (бинарные, форма 256×256)
pred_mask = np.random.randint(0, 2, size=(256, 256), dtype=np.uint8) * 255
gt_mask = np.random.randint(0, 2, size=(256, 256), dtype=np.uint8) * 255

# Расчёт всех метрик
metrics = SegmentationMetrics.calculate_all_metrics(
    pred_mask=pred_mask,
    gt_mask=gt_mask,
    threshold=0.5,              # Порог бинаризации
    include_hausdorff=True,     # Включить медленный Hausdorff
    verbose_comparison=False    # Не выводить сравнение с sklearn
)

# Доступ к результатам
print(f"IoU: {metrics['iou']:.3f}")
print(f"Dice: {metrics['dice']:.3f}")
print(f"F1: {metrics['f1_score']:.3f}")
print(f"Hausdorff: {metrics['hausdorff_distance']:.2f} px")
```

### Расчёт отдельной метрики
```python
# Только IoU (быстрее, чем calculate_all_metrics)
iou = SegmentationMetrics.calculate_iou(pred_mask, gt_mask, threshold=0.5)
print(f"IoU: {iou:.4f}")

# Precision и Recall вместе
precision, recall = SegmentationMetrics.calculate_precision_recall(
    pred_mask, gt_mask, threshold=0.5, verbose=True
)
print(f"P={precision:.3f}, R={recall:.3f}")
```

### Пакетная оценка по набору масок
```python
# Списки масок для нескольких изображений
pred_masks = [pred1, pred2, pred3]  # List[np.ndarray]
gt_masks = [gt1, gt2, gt3]

# Агрегированные результаты
results = SegmentationMetrics.evaluate_multiple_masks(
    pred_masks=pred_masks,
    gt_masks=gt_masks,
    threshold=0.5
)

# Средние значения с отклонениями
avg = results["average_metrics"]
print(f"Средний IoU: {avg['avg_iou']:.3f} ± {avg['std_iou']:.3f}")
print(f"Диапазон F1: [{avg['min_f1_score']:.3f}, {avg['max_f1_score']:.3f}]")
```

## ⚙️ Конфигурация
### Параметры `calculate_all_metrics()`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `pred_mask` | `np.ndarray` | — | Предсказанная маска формы `(H, W)`, dtype `uint8`/`float` |
| `gt_mask` | `np.ndarray` | — | Ground Truth маска той же формы |
| `threshold` | `float` | `0.5` | Порог бинаризации (адаптируется под диапазон [0,1] или [0,255]) |
| `include_hausdorff` | `bool` | `True` | Включать ли расчёт Hausdorff Distance (медленно!) |
| `verbose_comparison` | `bool` | `False` | Выводить ли сравнение custom vs sklearn в лог |
| `metrics_list` | `Optional[List[str]]` | `None` | Список метрик для расчёта; `None` = все доступные |

### Поддерживаемые имена метрик в `metrics_list`
```python
[
    "accuracy", "iou", "jaccard_score", "dice",
    "precision", "recall", "f1_score", "pixel_accuracy", "mae",
    "hausdorff_distance",
    "predicted_area", "ground_truth_area", "area_difference", "area_ratio",
    "true_positive", "false_positive", "false_negative", "true_negative"
]
```

## 📂 Форматы входных данных
### Требования к маскам
| Атрибут | Требование | Пример |
|---------|-----------|--------|
| **Форма** | `(H, W)` — 2D массив | `(512, 512)` |
| **Тип данных** | `uint8`, `int`, `float32/64` | `np.uint8` |
| **Диапазон значений** | `{0, 1}` **или** `{0, 255}` **или** `[0.0, 1.0]` | `0` = фон, `255` = объект |
| **Семантика** | Бинарная: объект/фон; Многоклассовая: целочисленные метки | `0`=фон, `1`=объект |

### Автоматическая нормализация
Метод `_normalize_masks()` приводит маски к единому формату:
```python
# Если макс. значение > 1 → диапазон [0, 255] → порог умножается на 255
if pred_mask.max() > 1:
    pred_binary = (pred_mask > threshold * 255).astype(np.uint8)
else:
    # Диапазон [0, 1] → порог применяется напрямую
    pred_binary = (pred_mask > threshold).astype(np.uint8)
```

> 💡 *Рекомендация: используйте формат `{0, 255}` для визуализации и `{0, 1}` для расчётов — модуль корректно обрабатывает оба.*

## 📊 Описание метрик
### Основные метрики качества
| Метрика | Формула | Интерпретация |
|---------|---------|---------------|
| **IoU** | `|A∩B| / |A∪B|` | Доля пересечения относительно объединения |
| **Dice** | `2\|A∩B\| / (\|A\|+\|B\|)` | Более устойчив к дисбалансу классов |
| **Precision** | `TP / (TP+FP)` | Доля верных объектов среди предсказанных |
| **Recall** | `TP / (TP+FN)` | Доля найденных объектов среди истинных |
| **F1-Score** | `2·P·R / (P+R)` | Гармоническое среднее точности и полноты |
| **Pixel Accuracy** | `верные_пиксели / всего_пикселей` | Может быть завышена при дисбалансе |
| **MAE** | `mean(\|pred - gt\|)` | Средняя абсолютная ошибка (нормализованная) |

### Геометрические и статистические метрики
| Метрика | Описание | Единицы |
|---------|----------|---------|
| **Hausdorff Distance** | Макс. расстояние от точки одного контура до ближайшей точки другого | пиксели |
| **Predicted Area** | Количество пикселей объекта в предсказании | пиксели |
| **Ground Truth Area** | Количество пикселей объекта в эталоне | пиксели |
| **Area Difference** | `\|area_pred - area_gt\|` | пиксели |
| **Area Ratio** | `min(area_pred, area_gt) / max(...)` | безразмерное [0, 1] |

### Элементы матрицы ошибок
| Обозначение | Описание | Пример |
|-------------|----------|--------|
| **TP** (True Positive) | Пиксели, верно предсказанные как объект | Объект → Объект |
| **FP** (False Positive) | Пиксели фона, ошибочно предсказанные как объект | Фон → Объект |
| **FN** (False Negative) | Пиксели объекта, пропущенные как фон | Объект → Фон |
| **TN** (True Negative) | Пиксели фона, верно предсказанные как фон | Фон → Фон |

### Метрики кластеризации* 
> *Применяются только при `len(unique_labels) > 2`*

| Метрика | Диапазон | Интерпретация |
|---------|----------|---------------|
| **Silhouette Score** | [-1, 1] | Чем ближе к 1, тем лучше разделение кластеров |
| **Calinski-Harabasz** | [0, ∞) | Чем выше, тем плотнее и лучше разделены кластеры |
| **Davies-Bouldin** | [0, ∞) | Чем ниже, тем лучше (меньше перекрытия кластеров) |

## ⚡ Производительность и оптимизации
### Скорость расчёта метрик (относительная)
```
✅ Быстро (<1 мс):
   - IoU, Dice, Precision, Recall, F1, Accuracy, MAE
   - Area statistics, Confusion Matrix elements

⚠️ Средне (1–10 мс):
   - Jaccard через sklearn (дополнительные проверки)

❌ Медленно (10–1000 мс, зависит от размера контура):
   - Hausdorff Distance (O(n·m) по точкам контуров)
```

### Рекомендации по использованию
1. **Отключайте Hausdorff** при массовом тестировании: `include_hausdorff=False`.
2. **Используйте `metrics_list`** для расчёта только нужных метрик — это ускоряет выполнение.
3. **Для отладки** включайте `verbose_comparison=True`, чтобы сравнить кастомные формулы с sklearn.
4. **При пустых масках** метрики возвращают `0.0` или `inf` — обрабатывайте эти случаи в вызывающем коде.

### Пример оптимизированного вызова
```python
# Только ключевые метрики для быстрого бенчмарка
metrics = SegmentationMetrics.calculate_all_metrics(
    pred_mask, gt_mask,
    include_hausdorff=False,  # Отключаем медленную метрику
    metrics_list=["iou", "dice", "f1_score", "pixel_accuracy"]  # Только важное
)
```

## 🛠️ Обработка крайних случаев
| Ситуация | Поведение модуля | Рекомендация |
|----------|-----------------|--------------|
| Пустой предсказанный объект | `IoU=0`, `Dice=0`, `Hausdorff=inf` | Проверять `if metrics['iou'] == 0` перед агрегацией |
| Пустой Ground Truth | `Precision=0`, `Recall=1` (по определению) | Использовать `zero_division=0` в sklearn-функциях |
| Маски разного диапазона | Автоматическая нормализация в `_normalize_masks()` | Убедиться, что данные действительно в ожидаемом формате |
| Многоклассовая маска | Метрики кластеризации возвращают значения; бинарные — работают по "объект vs всё остальное" | Для пер-классовых метрик использовать внешнюю обёртку с one-vs-rest |
| Деление на ноль | Защита через `+ 1e-8` в знаменателе | Не требуется дополнительная обработка |

## 🤝 Зависимости
```text
numpy>=1.20
scipy>=1.7  # для directed_hausdorff
scikit-learn>=1.0  # для accuracy_score, jaccard_score, etc.
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование SegmentationMetrics |
|--------|----------------------------------|
| `BatchClassicTester` | Расчёт метрик согласия между реализациями |
| `BatchClassicTester2` | Оценка качества против Ground Truth |
| `SegmentationBenchmark` | Сравнение нейросетей по mIoU, Dice, F1 |
| `SegmentationTester` | Универсальный расчёт метрик + визуализация |
| `SegmentationComparator` | Попарное сравнение методов через IoU/Dice |
| `TorchImplementationValidator` | Валидация PyTorch-реализаций против эталонов |

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