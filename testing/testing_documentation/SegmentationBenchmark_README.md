# 🧠 SegmentationBenchmark — Бенчмарк нейросетевых архитектур

## 📖 Описание
Модуль `SegmentationBenchmark.py` предназначен для автоматизированного сравнения **качества** и **производительности** нейросетевых моделей семантической и инстанс-сегментации на едином тестовом изображении или датасете.

> ⚠️ **Важно:** Данный модуль сравнивает *нейросетевые архитектуры*, а не классические алгоритмы. Для тестирования пороговых методов и детекторов границ используйте `BatchClassicTester` или `BatchClassicTester2`.

## ✨ Ключевые возможности
- 🔗 **Fluent Interface:** Цепочка вызовов для загрузки моделей — `bench.load_unet().load_deeplab().load_segformer()...`
- 🧩 **15+ архитектур из 5 категорий:**
  | Категория | Модели |
  |-----------|--------|
  | CNN | UNet, DeepLabV3+, FPN, PSPNet, FCN, SegNet |
  | Transformers | SegFormer (b0–b5), DPT, UPerNet |
  | Universal | Mask2Former, OneFormer, MaskFormer |
  | Promptable | SAM, SAM2 (Segment Anything) |
  | Instance | Mask R-CNN, YOLOv8-seg |
- 💾 **Управление VRAM:** Автоматическая очистка `torch.cuda.empty_cache()` и `gc.collect()` между моделями; поддержка асинхронного режима с обновлением прогресса.
- 📈 **Полный набор метрик:** mIoU, Pixel Accuracy, F1-weighted, Per-Class IoU (массив), Confusion Matrix, количество уникальных классов.
- 🎨 **Визуализация:** 5 типов графиков — бар-чарты метрик, heatmap per-class IoU, матрицы ошибок, наложенные маски, сводный дашборд.
- 📤 **Экспорт в 4 форматах:** CSV (сводка), JSON (детали + сериализация numpy), Markdown (таблица для GitHub), LaTeX (для статей).

## 🚀 Быстрый старт
```python
from testing.SegmentationBenchmark import SegmentationBenchmark
from PIL import Image

# Инициализация бенчмарка
bench = SegmentationBenchmark(
    device="cuda",
    num_classes=150,  # ADE20K
    ignore_index=255,
    gt_mask=Image.open("gt_mask.png")  # Опционально: для расчёта метрик
)

# Загрузка моделей (Fluent Interface)
(
    bench
    .load_unet_trained("checkpoints/unet_ade20k_best.pth")
    .load_deeplab_trained("checkpoints/deeplab_ade20k_best.pth")
    .load_segformer_variant("b2")  # Предобученная из HF
    .load_mask2former()
)

# Запуск сравнения на одном изображении
image = Image.open("test.jpg")
summary = bench.compare(image, alpha=0.6)  # alpha: прозрачность наложения

# Просмотр результатов
df = bench.get_summary_dataframe()
print(df.sort_values("mIoU", ascending=False))

# Визуализация и экспорт
bench.plot_all_metrics(path="results/metrics.png")
bench.save_results(output_dir="results/")
latex_code = bench.export_latex_table(caption="ADE20K Benchmark")
```

## ⚙️ Конфигурация
### Параметры инициализации
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `device` | `str` | `"cuda"` | Устройство для инференса: `"cuda"` или `"cpu"` |
| `num_classes` | `int` | `150` | Количество классов в датасете (например, 150 для ADE20K) |
| `ignore_index` | `int` | `255` | Индекс игнорируемого класса в GT-маске |
| `class_names` | `Optional[List]` | `None` | Список имён классов для визуализации |
| `gt_mask` | `Optional[np.ndarray/Image]` | `None` | Ground Truth маска для расчёта метрик |
| `palette` | `Optional[Union[List, Callable]]` | `ade_palette()` | Цветовая палитра для визуализации масок |

### Методы загрузки моделей (выборка)
| Метод | Описание | Пример вызова |
|---|---|---|
| `load_unet_trained()` | UNet из чекпоинта | `.load_unet_trained("unet.pth", encoder_name="resnet34")` |
| `load_deeplab_trained()` | DeepLabV3+ из чекпоинта | `.load_deeplab_trained("deeplab.pth")` |
| `load_segformer_variant()` | SegFormer из HF | `.load_segformer_variant("b2")` |
| `load_mask2former()` | Mask2Former из HF | `.load_mask2former("facebook/mask2former-swin-base-ade-semantic")` |
| `load_oneformer()` | OneFormer из HF | `.load_oneformer("shi-labs/oneformer_ade20k_swin_large")` |
| `load_sam()` | SAM / SAM2 | `.load_sam("mobile_sam.pt")` |
| `load_yolov8()` | YOLOv8-seg | `.load_yolov8("yolov8n-seg.pt")` |
| `load_all_trained_models()` | Пакетная загрузка всех `.pth` из директории | `.load_all_trained_models("./checkpoints")` |

> 💡 *Все методы загрузки возвращают `self`, что позволяет строить цепочки вызовов.*

## 📂 Структура выходных данных
```
results/
├── summary.csv                 # Сводная таблица: модель × метрики
├── detailed.json              # Детальные метрики + метаданные (с сериализацией numpy)
├── overlay_{model}.jpg        # Изображение с наложенной маской для каждой модели
├── mask_{model}.npy           # Бинарная маска предсказания (NumPy формат)
├── plot_comparison_chart.jpg  # Бар-чарт сравнения по одной метрике
├── plot_per_class_iou.jpg     # Heatmap per-class IoU (топ-20 классов)
├── plot_confusion_matrix.jpg  # Матрица ошибок для указанной модели
├── plot_all_metrics.jpg       # Сводный дашборд (3 графика в ряд)
└── model_comparison.md        # Markdown-таблица для GitHub/отчётов
```

## 📊 Метрики качества
Для каждой модели рассчитываются (при наличии GT-маски):

| Метрика | Описание | Диапазон | Интерпретация |
|---------|----------|----------|---------------|
| **mIoU** | Mean Intersection over Union | [0.0, 1.0] | Чем выше, тем лучше; основной критерий |
| **pixel_acc** | Pixel Accuracy | [0.0, 1.0] | Доля верно классифицированных пикселей |
| **f1_weighted** | Weighted F1-Score | [0.0, 1.0] | Баланс точности и полноты с учётом дисбаланса классов |
| **per_class_iou** | IoU по каждому классу | `np.array` длины `num_classes` | Для heatmap и анализа слабых классов |
| **confusion_matrix** | Матрица ошибок | `np.array` формы `(C, C)` | Для визуализации типичных ошибок |
| **time_ms** | Время инференса | `[0, ∞)` мс | Чем ниже, тем лучше; замеряется с `torch.cuda.synchronize()` |
| **unique_classes** | Количество уникальных классов в предсказании | `[1, num_classes]` | Индикатор "разнообразия" предсказания |

## 🎨 Типы визуализаций
1. **`plot_comparison_chart(metric_name)`** — Бар-чарт сравнения одной метрики по всем моделям с автоформатированием (проценты/десятичные).
2. **`plot_per_class_iou(top_k=20)`** — Heatmap per-class IoU: строки = модели, столбцы = топ-20 присутствующих в GT классов.
3. **`plot_confusion_matrix(model_key)`** — Нормализованная матрица ошибок (recall/precision/counts) с аннотациями.
4. **`plot_all_metrics()`** — Сводный дашборд: 3 графика в ряд (mIoU %, Pixel Acc %, Time ms).
5. **`plot_summary(metrics=[...])`** — Гибкая визуализация произвольного набора метрик.

## 🔄 Управление памятью и асинхронный режим
### Синхронный `compare()`
```python
for i, key in enumerate(model_keys):
    self.run_single(image_input, key)  # Инференс + метрики
    if i < len(model_keys) - 1:
        del self.models[key]["model"]  # Освобождение VRAM
        del self.models[key]["processor"]
        torch.cuda.empty_cache()
        gc.collect()
```

### Асинхронный `compare_step_by_step()`
```python
# Поддержка внешнего трекера прогресса (например, для веб-интерфейса)
summary = await bench.compare_step_by_step(
    image_input=image,
    task_id="benchmark_123",
    benchmark_tasks=tasks_dict  # Мутация in-place: {"benchmark_123": {"progress": 0..100, "message": "..."}}
)
```

> ⚠️ *После `compare()` модели удаляются из памяти, но остаются в `self.results` для доступа к метрикам и визуализациям.*

## 📤 Экспорт результатов
### Форматы вывода
| Формат | Метод | Особенности |
|--------|-------|-------------|
| **CSV** | `save_results()` | Сводная таблица: модель × (mIoU, Acc, F1, time_ms) |
| **JSON** | `save_results()` | Детальные метрики + рекурсивная сериализация numpy-типов |
| **Markdown** | `export_comparison_table()` | Готовая таблица для GitHub/отчётов с категоризацией моделей |
| **LaTeX** | `export_latex_table()` | Код таблицы с `\toprule`/`\bottomrule` для публикаций |

### Пример Markdown-вывода
```markdown
# Segmentation Models Comparison

| Model              | Category    | mIoU (%) | pixel_acc | Time (ms) | unique_classes |
|:-------------------|:------------|---------:|----------:|----------:|---------------:|
| segformer_b2       | Transformer |     45.2 |     0.891 |     124.5 |            142 |
| mask2former        | Universal   |     43.8 |     0.885 |     312.1 |            148 |
| unet_pretrained    | CNN         |     38.1 |     0.852 |      45.3 |            135 |
```

## 🛠️ Обработка ошибок и устойчивость
- Исключения в `run_single()` перехватываются на уровне `segment_image_unified()`; бенчмарк продолжает работу со следующей моделью.
- При отсутствии GT-маски метрики возвращают `np.nan`, но инференс и визуализация выполняются нормально.
- Метод `get_model_num_classes()` использует эвристики (конфигурация HF, последний Conv2d, fallback) для определения выходной размерности модели.
- Все numpy-массивы в JSON сериализуются через `convert_numpy_types()` (рекурсивная конвертация в Python-native типы).

## ⚡ Рекомендации по использованию
- Для экономии памяти при сравнении >5 моделей используйте `compare()` (автоматическая очистка) или асинхронный режим.
- При работе с SAM/SAM2 убедитесь, что изображение предварительно обработано (SAM требует специфичного ресайза).
- Для per-class анализа установите `show_only_present_classes=True` в `plot_per_class_iou()`, чтобы исключить "пустые" классы из визуализации.
- При экспорте в LaTeX используйте `export_latex_table(caption="...")` и вставьте результат в документ с подключённым пакетом `booktabs`.

## 🤝 Зависимости
```text
torch, torchvision, numpy, pandas, matplotlib, seaborn, pillow
transformers, timm, ultralytics  # для отдельных архитектур
segmentation-models-pytorch      # для SMP-моделей (UNet, FPN, PSPNet)
```

## 🔗 Сравнение с другими модулями тестирования
| Модуль | Цель | Типы методов | Данные | Метрики |
|--------|------|--------------|--------|---------|
| `BatchClassicTester` | Консистентность реализаций | Классические (пороги, границы) | Только изображения | IoU/Dice между масками |
| `BatchClassicTester2` | Качество классических методов | Классические | Изображения + GT | IoU, Dice, Precision vs GT |
| **`SegmentationBenchmark`** | **Сравнение нейросетей** | **Нейросетевые (15+ архитектур)** | **Изображения + опционально GT** | **mIoU, Acc, F1, per-class, time** |

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