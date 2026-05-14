# 🧪 BatchClassicTester — Тестирование согласованности классических методов

## 📖 Описание
Модуль `BatchClassicTester.py` предназначен для массового автоматизированного тестирования **консистентности** классических алгоритмов сегментации (пороговые методы и детекторы границ) между различными бэкендами: **PyTorch**, **OpenCV** и **Scikit-learn**.

> ⚠️ **Важно:** Данный тестер проверяет *согласованность реализаций* между библиотеками, а не качество сегментации относительно Ground Truth. Для валидации качества используйте отдельный модуль оценки (например, `SegmentationEvaluator`).

## ✨ Ключевые возможности
- 🔍 **Попарное сравнение:** Настраиваемые пары библиотек (`torch ↔ opencv`, `torch ↔ sklearn`, `opencv ↔ sklearn`, `torch_v1 ↔ torch_v2`).
- 📊 **Метрики согласованности:** IoU, Dice, Precision, Recall, F1-Score, MAE, Hausdorff Distance, Pixel Accuracy.
- 🚦 **Валидация:** Автоматическая классификация результатов на `PASS` / `WARNING` / `FAIL` по порогам.
- 💾 **Экономия ресурсов:** Вероятностная выборка масок (`mask_sample_rate`) и жёсткие лимиты на количество сохранений (`max_mask_samples_per_method`).
- 🔄 **Устойчивость к прерываниям:** Поддержка `resume=True`, автосохранение прогресса, обработка сигналов `SIGINT`/`SIGTERM`.
- 📈 **Аналитика и отчёты:** Генерация CSV/JSON/Markdown/HTML отчётов, тепловых карт, графиков сравнения и 4-панельных визуализаций разницы масок.

## 🚀 Быстрый старт
```python
from testing.BatchClassicTester import BatchClassicTester

# Инициализация тестера
tester = BatchClassicTester(
    ade20k_root="./data/ade20k/ADEChallengeData2016",
    output_dir="./results/consistency_test",
    split="validation",
    max_images=50,
    save_masks=True,
    mask_sample_rate=0.1,
    max_mask_samples_per_method=3,
    resume=True
)

# Запуск тестирования
df_results = tester.run_batch_test()

# Сохранение и визуализация результатов
csv_path, json_path, md_path = tester.save_results(df_results)
tester.plot_results(df_results)
tester.print_summary(df_results)
```

## ⚙️ Конфигурация
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `ade20k_root` | `str` | `./data/ade20k/...` | Путь к корневой директории датасета |
| `output_dir` | `str` | `./data/batch_consistency_test` | Директория для всех артефактов и отчётов |
| `max_images` | `Optional[int]` | `None` | Лимит изображений (`None` = все доступные) |
| `image_size` | `Tuple[int, int]` | `(512, 512)` | Целевой размер для ресайза входных изображений |
| `library_pairs` | `List[Tuple]` | 4 пары | Список пар библиотек для попарного сравнения |
| `save_masks` | `bool` | `True` | Сохранять бинарные маски `.npy` |
| `mask_sample_rate` | `float` | `0.1` | Вероятность сохранения маски `[0.0–1.0]` |
| `max_mask_samples_per_method` | `int` | `3` | Макс. количество масок на комбинацию метод/пара |
| `save_visualizations` | `bool` | `True` | Генерировать 4-панельные сравнения (`comparison.png`) |
| `resume` | `bool` | `True` | Возобновлять тест с места остановки через `.progress.json` |
| `refresh_masks` | `bool` | `False` | Пересоздавать маски и отчёты для уже пройденных тестов |

## 📂 Структура выходных данных
```
{output_dir}/
├── masks/                  # .npy маски + оригиналы + метрики
├── results/                # PNG маски и оригиналы для быстрой навигации
├── visualizations/         # Сводные 4-панельные JPG-визуализации
├── charts/                 # Аналитические графики (IoU, время, статусы, heatmap)
├── consistency_test_results.csv
├── consistency_test_details.json
├── consistency_test_report.md
└── consistency_report_YYYYMMDD_HHMMSS.html
```

## 📝 Тестируемые методы
Всего **23 алгоритма**:
- **Пороговые (13):** `global`, `adaptive`, `otsu`, `niblack`, `sauvola`, `bernsen`, `phansalkar`, `kittler_illingworth`, `entropy_kapur`, `triangle`, `multi_otsu`, `percentile`, `local_contrast`.
- **Граничные (10):** `sobel`, `canny`, `prewitt`, `scharr`, `roberts_cross`, `log`, `dog`, `marr_hildreth`, `gradient_magnitude_direction`, `phase_congruency`.

## 🛡️ Пороги валидации
Статус определяется по 7 ключевым метрикам:
- ✅ `PASS`: Все 7 метрик проходят пороги
- ⚠️ `WARNING`: 4–6 метрик проходят пороги
- ❌ `FAIL`: <4 метрик проходят пороги

| Метрика | Порог (PASS) | Условие |
|---|---|---|
| IoU | `≥ 0.80` | Чем выше, тем лучше |
| Dice | `≥ 0.85` | |
| Pixel Accuracy | `≥ 0.90` | |
| Precision / Recall | `≥ 0.80` | |
| F1-Score | `≥ 0.82` | |
| MAE | `≤ 0.15` | Чем ниже, тем лучше |

> 💡 *Пороги настраиваются через атрибут `success_thresholds` при инициализации или напрямую в коде.*

## ⚡ Рекомендации по производительности
- Для быстрой проверки используйте `max_images=10`, `save_masks=False`, `save_visualizations=False`.
- При `resume=True` тестер пропускает уже выполненные комбинации, но может дозаполнить маски, если лимит `max_mask_samples_per_method` не достигнут.
- Память автоматически очищается после каждого изображения (`torch.cuda.empty_cache()`, `gc.collect()`).
- Для экономии места на диске рекомендуется `mask_sample_rate ≤ 0.1` и `max_mask_samples_per_method ≤ 5` при тестировании на >100 изображениях.

## 🤝 Зависимости
```text
torch, numpy, pandas, matplotlib, seaborn, pillow, tqdm, scikit-image
(opencv-python, scikit-learn, skimage требуются для самих сегментеров)
```

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