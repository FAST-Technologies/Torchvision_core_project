# 🔍 TorchImplementationValidator — Валидатор PyTorch-реализаций

## 📖 Описание
Модуль `TorchImplementationValidator.py` предназначен для автоматизированной **валидации согласованности** кастомных PyTorch-реализаций алгоритмов сегментации против эталонных библиотек **OpenCV** и **Scikit-learn**.

> ⚠️ **Важно:** Данный модуль проверяет *согласованность между реализациями*, а не качество относительно Ground Truth. Для валидации качества используйте `BatchClassicTester2` или `SegmentationTester`.

## ✨ Ключевые возможности
- 🔁 **Попарное сравнение:** Запуск одного метода в двух бэкендах с замером времени и расчётом метрик соответствия.
- 🎯 **Автоматическая классификация:** Статус `PASS` / `WARNING` / `FAIL` на основе 7 ключевых метрик:
  | Метрика | Порог для PASS | Условие |
  |---------|---------------|---------|
  | IoU | ≥ 0.80 | Intersection over Union |
  | Dice | ≥ 0.85 | F1 для бинарной сегментации |
  | Pixel Accuracy | ≥ 0.90 | Доля совпадающих пикселей |
  | Precision / Recall | ≥ 0.80 | Точность / Полнота |
  | F1-Score | ≥ 0.82 | Гармоническое среднее |
  | MAE | ≤ 0.15 | Mean Absolute Error |
- 🧩 **Поддержка двух версий Torch:**
  - `TorchSegmenter`: Базовая реализация.
  - `TorchSegmenter2`: Расширенная версия с параметрами `precision` (fp32/fp16/bf16), `use_compile`, `debug_mode`.
- 🎨 **Визуализация 4-панельного макета:**
  ```
  ┌─────────────┬─────────────┬─────────────┬─────────────┐
  │  Original   │  Mask A     │  Mask B     │  Difference │
  │  Image      │  (IoU:0.92) │             │  (hot cmap) │
  └─────────────┴─────────────┴─────────────┴─────────────┘
  ```
- 📦 **Пакетная валидация:** 6 предустановленных конфигураций:
  1. Threshold: Torch ↔ Sklearn
  2. Threshold: Torch ↔ OpenCV
  3. Threshold: OpenCV ↔ Sklearn
  4. Edge: Torch ↔ Sklearn
  5. Edge: Torch ↔ OpenCV
  6. Edge: OpenCV ↔ Sklearn
- 📤 **Экспорт в 4 форматах:** `.npy` (маски), `.txt` (метрики), `.png` (визуализации), сводный текстовый отчёт + бенчмарк-графики.

## 🚀 Быстрый старт
### Базовая валидация одного метода
```python
from testing.TorchImplementationValidator import TorchImplementationValidator
from segmenters.TorchSegmenter import TorchSegmenter
from segmenters.OpenCVSegmenter import OpenCVSegmenter

# Инициализация валидатора
validator = TorchImplementationValidator(output_dir="./results/validation")

# Валидация метода Оцу: Torch vs OpenCV
results = validator.validate_segmentation_methods(
    image_path="test.jpg",
    methods_list=[("otsu_thresholding", {})],
    first_segmenter_class=TorchSegmenter,
    second_segmenter_class=OpenCVSegmenter,
    first_method_name="Torch",
    second_method_name="OpenCV",
    status_message="ВАЛИДАЦИЯ ПОРОГОВЫХ МЕТОДОВ",
    prefix="otsu_validation",
    validation_type="threshold"
)

# Доступ к результатам
for method, data in results.items():
    if data.get("success"):
        print(f"{method}: IoU={data['metrics']['iou']:.3f}, статус={data['validation_status']}")
```

### Пакетная валидация всех методов
```python
# Запуск валидации всех пороговых и граничных методов
all_results = validator.validate_all_methods(
    image_path="test.jpg",
    use_torch2=True,          # Использовать TorchSegmenter2
    torch2_precision="fp32"   # Точность вычислений
)

# Генерация сводного отчёта
report = validator.generate_validation_report(all_results)

# Генерация бенчмарк-отчёта с графиками
df_benchmark = validator.generate_benchmark_report_from_validation(
    all_results,
    output_dir="./results/benchmark"
)

# Анализ топ-методов по эффективности
top_efficient = df_benchmark.nlargest(10, "iou") / (df_benchmark["first_method_time"] + 0.001)
print(top_efficient[["method", "iou", "first_method_time", "efficiency_score"]])
```

### Прямое сравнение TorchSegmenter2 vs TorchSegmenter
```python
# Валидация новой версии против старой
results_t2_vs_t1 = validator.validate_segmentation_methods(
    image_path="test.jpg",
    methods_list=[("canny_edge", {"low": 0.1, "high": 0.3})],
    first_segmenter_class=TorchSegmenter2,  # Новая версия
    second_segmenter_class=TorchSegmenter,   # Старая версия
    first_method_name="Torch2",
    second_method_name="Torch1",
    status_message="ПРЯМОЕ СРАВНЕНИЕ: TorchSegmenter2 vs TorchSegmenter",
    prefix="t2_vs_t1",
    validation_type="edge"
)
```

## ⚙️ Конфигурация
### Параметры инициализации
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `output_dir` | `str` | `"./data/validation_results"` | Базовая директория для всех артефактов |

### Параметры `validate_segmentation_methods()`
| Параметр | Тип | Описание |
|---|---|---|
| `image_path` | `ImageInput` | Путь, `np.ndarray` или `PIL.Image` |
| `methods_list` | `List[MethodConfig]` | Список `[(имя_метода, {параметры}), ...]` |
| `first_segmenter_class` | `SegmenterClass` | Класс первого сегментера (например, `TorchSegmenter`) |
| `second_segmenter_class` | `SegmenterClass` | Класс второго сегментера (например, `OpenCVSegmenter`) |
| `first_method_name` / `second_method_name` | `str` | Человекочитаемые имена для отчётов |
| `status_message` | `str` | Заголовок для вывода в консоль |
| `prefix` | `str` | Префикс имён файлов и директорий |
| `validation_type` | `str` | Категория: `"threshold"`, `"edge"`, `"region"`, ... |
| `use_first_method_features` | `bool` | Применять ли специфичные параметры для первого класса |

### Параметры `validate_all_methods()`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `use_torch2` | `bool` | `False` | Использовать `TorchSegmenter2` вместо `TorchSegmenter` |
| `torch2_precision` | `str` | `"fp32"` | Точность вычислений: `"fp32"`, `"fp16"`, `"bf16"` |

## 📂 Структура выходных данных
### Для `validate_segmentation_methods()`
```
{output_dir}/{prefix}_{reference}_{timestamp}/
└── {method_name}/
    ├── {first_method}_mask.npy   # Маска первого сегментера
    ├── {second_method}_mask.npy  # Маска второго сегментера
    └── metrics.txt               # Текстовые метрики + статус
```

### Для `validate_all_methods()` + `generate_benchmark_report_from_validation()`
```
{output_dir}/benchmark_from_validation/
├── benchmark_validation_data.csv  # Сводная таблица всех результатов
├── benchmark_summary.txt          # Текстовый отчёт с топ-методами
└── charts_{validation_type}/      # Графики для каждой конфигурации
    ├── {first_method}_time.png          # Бар-чарт времени выполнения
    ├── {first_method}_vs_{second_method}_time.png  # Scatter: time A vs time B
    ├── iou_comparison.png               # IoU по методам с цветовой индикацией статуса
    ├── {first_method}_coverage.png     # Процент покрытия маски
    ├── metrics_heatmap.png              # Heatmap метрик качества
    └── time_vs_iou_tradeoff.png         # Trade-off: скорость vs точность
```

## 📝 Поддерживаемые методы (по категориям)
### Пороговые (13 алгоритмов)
```python
[
    ("global_thresholding", {"threshold": 0.5}),
    ("otsu_thresholding", {}),
    ("adaptive_thresholding", {"block_size": 11, "C": 2}),
    ("threshold_niblack", {"window_size": 15, "k": -0.2}),
    ("threshold_sauvola", {"window_size": 15, "k": 0.5, "r": 128}),
    ("threshold_bernsen", {"window_size": 15, "contrast_threshold": 0.15}),
    ("threshold_phansalkar", {"window_size": 15, "k": 0.25, "r": 128.0, "m": 0.5}),
    ("threshold_kittler_illingworth", {"num_bins": 256}),
    ("threshold_entropy_kapur", {"num_bins": 256}),
    ("threshold_triangle", {"num_bins": 256}),
    ("threshold_multi_otsu", {"n_thresholds": 2}),
    ("threshold_percentile", {"percentile": 90}),
    ("threshold_local_contrast", {"window_size": 15, "contrast_factor": 0.1}),
]
```

### Граничные (10 алгоритмов)
```python
[
    ("sobel_edge", {"threshold": 0.1}),
    ("canny_edge", {"low": 0.1, "high": 0.3, "sigma": 1.0}),
    ("prewitt_edge", {"threshold": 0.1}),
    ("scharr_edge", {"threshold": 0.1}),
    ("roberts_cross_edge", {"threshold": 0.1}),
    ("log_edge", {"sigma": 1.0, "threshold": 0.01}),
    ("dog_edge", {"sigma1": 1.0, "sigma2": 2.0, "threshold": 0.01}),
    ("marr_hildreth_edge", {"sigma": 1.5, "threshold": 0.01}),
    ("gradient_magnitude_direction", {"threshold": 0.1}),
    ("phase_congruency_edge", {...}),  # Сложные параметры фазовой согласованности
]
```

> 💡 *Остальные категории (региональные, кластеризация, активные контуры, водоразделы, суперпиксели, интерактивные) закомментированы в `validate_all_methods()` и могут быть активированы при необходимости.*

## 📊 Метрики соответствия
Для каждой пары масок рассчитываются:

| Метрика | Описание | Диапазон | Условие для PASS |
|---------|----------|----------|-----------------|
| **IoU** | Intersection over Union | [0.0, 1.0] | ≥ 0.80 |
| **Dice** | Dice / Sørensen coefficient | [0.0, 1.0] | ≥ 0.85 |
| **Pixel Accuracy** | Доля совпадающих пикселей | [0.0, 1.0] | ≥ 0.90 |
| **Precision** | Точность: верные объекты / предсказанные | [0.0, 1.0] | ≥ 0.80 |
| **Recall** | Полнота: найденные объекты / истинные | [0.0, 1.0] | ≥ 0.80 |
| **F1-Score** | Гармоническое среднее Precision и Recall | [0.0, 1.0] | ≥ 0.82 |
| **MAE** | Mean Absolute Error | [0.0, 1.0] | ≤ 0.15 |
| **Hausdorff Distance** | Макс. расстояние между контурами | [0, ∞) | Не используется в классификации |
| **Area Ratio** | Отношение площадей масок | [0, ∞) | Информативная метрика |

## 🎨 Типы визуализаций
### 1. Попарное сравнение (4 панели)
```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   Original      │   Mask A        │   Mask B        │   Difference    │
│   Image         │   (Torch)       │   (OpenCV)      │   Heatmap       │
│                 │   IoU: 0.92     │                 │   Status: PASS  │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```
- **Heatmap разницы**: цветовая схема "hot" (чёрный → красный → жёлтый → белый).
- **Статус**: окрашивается в зелёный (PASS), оранжевый (WARNING), красный (FAIL).

### 2. Бенчмарк-графики (6 типов)
1. **`{method}_time.png`**: Горизонтальный бар-чарт времени выполнения.
2. **`{A}_vs_{B}_time.png`**: Scatter-plot: время A (ось X) vs время B (ось Y) с линией y=x.
3. **`iou_comparison.png`**: Bar-чарт IoU с цветовой индикацией статуса.
4. **`{method}_coverage.png`**: Процент покрытия маски (площадь объекта / общая площадь).
5. **`metrics_heatmap.png`**: Тепловая карта метрик (методы × метрики).
6. **`time_vs_iou_tradeoff.png`**: Trade-off: скорость (ось X) vs точность (ось Y).

## 🔄 Логика фильтрации параметров
Метод `_filter_params()` предотвращает ошибки при передаче параметров, специфичных для одного класса, в другой:

```python
@staticmethod
def _filter_params(segmenter_class: SegmenterClass, params: Dict[str, Any]) -> Dict[str, Any]:
    """Фильтрует параметры, оставляя только поддерживаемые сигнатурой __init__ класса."""
    if not hasattr(segmenter_class, "__init__"):
        return params
    try:
        sig = inspect.signature(segmenter_class.__init__)
        valid_params = set(sig.parameters.keys()) - {"self", "kwargs", "kwds"}
        valid_params.add("postprocess")  # Всегда разрешаем для совместимости
        return {k: v for k, v in params.items() if k in valid_params}
    except (ValueError, TypeError):
        return params  # Fallback: вернуть как есть
```

> 💡 *Это позволяет передавать единый словарь параметров в разные классы сегментеров без риска `TypeError: __init__() got an unexpected keyword argument`.*

## 🛠️ Нормализация масок
Метод `_normalize_mask()` приводит маски к единому формату для сравнения:

```python
@staticmethod
def _normalize_mask(mask: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Приводит маску к формату: (H, W), dtype uint8, значения {0, 255}."""
    # 1. Tensor → numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().detach().cpu().numpy()
    
    # 2. Удаление лишних измерений: (1,1,H,W) → (H,W)
    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        elif mask.shape[0] == 1:
            mask = mask.squeeze(0)
    
    # 3. Конвертация в uint8
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.dtype in [np.float32, np.float64]:
        if mask.max() <= 1.0 + 1e-6:  # Нормализованная [0,1]
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
    
    return mask
```

## ⚡ Рекомендации по использованию
- Для стабильных замеров времени отключайте `torch.compile` через `use_compile=False` (по умолчанию в валидаторе).
- При сравнении с референсными библиотеками всегда устанавливайте `postprocess=False` для второго сегментера — это обеспечивает честное сравнение "сырых" результатов.
- Для отладки конкретного метода установите `methods_list=[(...)]` с одним элементом и `output_dir="./debug"`.
- При использовании `TorchSegmenter2` с `fp16`/`bf16` убедитесь, что ваше оборудование поддерживает соответствующую точность.

## 🤝 Зависимости
```text
torch, numpy, pandas, matplotlib, seaborn, pillow
# Сегментеры: OpenCVSegmenter, SklearnSegmenter, TorchSegmenter, TorchSegmenter2
# Метрики: SegmentationMetrics
```

## 🔗 Сравнение с другими модулями тестирования
| Модуль | Цель | Сравнение | Данные | Вывод |
|--------|------|-----------|--------|-------|
| `BatchClassicTester` | Консистентность реализаций | Библиотека ↔ Библиотека | Только изображения | Метрики согласия между масками |
| `BatchClassicTester2` | Качество классических методов | Предсказание ↔ Ground Truth | Изображения + GT | Метрики качества (IoU, Dice vs GT) |
| `SegmentationBenchmark` | Сравнение нейросетей | Архитектура ↔ Архитектура | Изображения + опционально GT | mIoU, Acc, F1, per-class, time |
| `SegmentationTester` | Универсальное тестирование | Любой сегментер ↔ любой | Изображение + опционально GT | Время, метрики, визуализация |
| **`TorchImplementationValidator`** | **Валидация PyTorch-реализаций** | **Torch ↔ OpenCV/Sklearn** | **Одно изображение** | **Метрики соответствия + статус + визуализация** |

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