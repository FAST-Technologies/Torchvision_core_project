# ⚡ CpuCudaBenchmark — Бенчмарк производительности сегментации

## 📖 Описание
Модуль `CpuCudaBenchmark.py` предназначен для автоматизированного сравнения **производительности** классических и нейросетевых методов сегментации при выполнении на **CPU** и **CUDA (GPU)**.

> ⚠️ **Важно:** Данный модуль измеряет *время выполнения* и *ускорение*, а не качество сегментации. Для валидации качества относительно Ground Truth используйте `BatchClassicTester2`.

## ✨ Ключевые возможности
- 🔁 **Warm-up прогоны:** Стабилизация производительности GPU перед замерами (избегает искажений из-за ленивой инициализации контекста).
- ⏱️ **Точный тайминг:** Использование `time.perf_counter()` + `torch.cuda.synchronize()` для корректных замеров на GPU.
- 🔄 **Автоматическое переключение устройства:** Временная замена `.device` и `.model.to()` у сегментеров без побочных эффектов.
- 🛡️ **Устойчивость к ошибкам:** Исключения в отдельных прогонах логируются, но не прерывают весь бенчмарк.
- 💾 **Сохранение артефактов:** Опциональное сохранение результата, маски и overlay-визуализации для каждой комбинации.
- 📊 **Множество форматов экспорта:** CSV, Excel (с автоподбором ширины колонок), текстовый отчёт со сводкой и топ-ускорениями.
- 📈 **Визуализации:** 5 типов графиков — сравнение времени, speedup, scatter-plot, multi-backend бар-чарт, относительное ускорение.

## 🚀 Быстрый старт
```python
from testing.CpuCudaBenchmark import CpuCudaBenchmark
from segmenters.TorchSegmenter import TorchSegmenter
import numpy as np
from PIL import Image

# Загрузка тестового изображения
image = np.array(Image.open("test.jpg").convert("RGB"))

# Инициализация бенчмарка
benchmark = CpuCudaBenchmark(
    base_output_dir="./data/benchmarks",
    n_runs=10,          # Количество измерительных прогонов
    warmup_runs=3       # Количество разогревочных прогонов
)

# Подготовка методов для тестирования
methods = {
    "otsu_torch_fp32": TorchSegmenter("otsu_thresholding"),
    "canny_opencv_fp32": OpenCVSegmenter("canny_edge", low=0.1, high=0.3),
    # ... другие методы
}

# Запуск бенчмарка
df_results = benchmark.benchmark_all_methods(
    methods_dict=methods,
    image=image,
    test_name="classic_methods_comparison",
    save_artifacts=True  # Сохранять ли результат/маску/overlay
)

# Результаты доступны в df_results и сохранены в ./data/benchmarks/classic_methods_comparison_YYYYMMDD_HHMMSS/
```

## ⚙️ Конфигурация
### Параметры инициализации `CpuCudaBenchmark`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `base_output_dir` | `str` | `"./data/cpu_cuda_benchmark"` | Базовая директория для всех запусков бенчмарка |
| `n_runs` | `int` | `5` | Количество измерительных прогонов на метод/устройство |
| `warmup_runs` | `int` | `2` | Количество "разогревочных" прогонов перед замерами |

### Параметры `benchmark_method()`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `segmenter` | `Any` | — | Экземпляр сегментера с методом `.segment(image)` |
| `image` | `np.ndarray` | — | Тестовое изображение (любой формы/канальности) |
| `method_name` | `str` | — | Уникальное имя в формате `{base}_{backend}_{precision}` |
| `device` | `str` | `"cpu"` | Устройство для запуска: `"cpu"` или `"cuda"` |
| `save_artifacts` | `bool` | `True` | Сохранять ли результат, маску и overlay |
| `output_dir` | `Optional[str]` | `None` | Директория для артефактов (автогенерация если `None`) |

## 📂 Структура выходных данных
```
{base_output_dir}/{test_name}_YYYYMMDD_HHMMSS/
├── artifacts/                    # (опционально) сохранённые артефакты
│   ├── cpu/
│   │   └── {method_name}/
│   │       ├── {method}_cpu_result.jpg
│   │       ├── {method}_cpu_mask.png
│   │       └── {method}_cpu_overlay.jpg
│   └── cuda/
│       └── {method_name}/        # аналогично для GPU
├── charts/                       # (опционально) сгенерированные графики
│   ├── cpu_cuda_comparison.png   # Бар-чарт времени выполнения
│   ├── speedup_comparison.png    # Ускорение (CPU_time / CUDA_time)
│   ├── cpu_cuda_scatter.png      # Scatter-plot зависимости
│   ├── comparison_bar.png        # Multi-backend сравнение на CUDA
│   └── speedup_chart.png         # Относительное ускорение к референсу
├── results.csv                   # Сырые данные бенчмарка
├── results.xlsx                  # Excel-таблица с автоформатированием
└── report.txt                    # Текстовый отчёт со сводкой и топ-ускорениями
```

## 📝 Формат имён методов
Модуль автоматически парсит имена методов вида:
```
{base_method}_{backend}_{precision}
```

| Компонент | Примеры | Описание |
|---|---|---|
| `base_method` | `otsu_thresholding`, `canny_edge`, `unet_inference` | Базовое название алгоритма |
| `backend` | `TORCH`, `CV2`, `SKLEARN`, `ONNX`, `TRT` | Реализация/бэкенд |
| `precision` | `fp32`, `fp16`, `bf16` | Точность вычислений (для нейросетей) |

> 💡 *Если имя не соответствует формату, применяется fallback-парсинг: последний компонент считается точностью (или `fp32` по умолчанию), предпоследний — бэкендом.*

## 📊 Метрики производительности
Для каждого метода/устройства рассчитываются:

| Метрика | Описание | Единицы |
|---------|----------|---------|
| `mean_time` | Среднее время выполнения (после warm-up) | секунды |
| `std_time` | Стандартное отклонение времени | секунды |
| `min_time` / `max_time` | Экстремальные значения времени | секунды |
| `total_time` | Общее время замера (включая накладные расходы) | секунды |
| `n_runs` | Фактическое количество успешных прогонов | целое |
| `speedup` | Отношение `CPU_time / CUDA_time` (рассчитывается постфактум) | безразмерное |

## 📈 Типы визуализаций
1. **`cpu_cuda_comparison.png`** — Группированная гистограмма среднего времени (ms) для CPU и CUDA.
2. **`speedup_comparison.png`** — Бар-чарт ускорения: зелёные столбцы (>1×), красные (<1×).
3. **`cpu_cuda_scatter.png`** — Scatter-plot: CUDA время vs CPU время с линией равенства.
4. **`comparison_bar.png`** — Multi-backend сравнение времени на CUDA (если есть колонка `backend`).
5. **`speedup_chart.png`** — Относительное ускорение к референсу (PyTorch/fp32/CPU).

## 🔄 Логика warm-up и синхронизации
```python
# 1. Warm-up (без замера)
for _ in range(warmup_runs):
    segmenter.segment(image)  # Прогрев кэшей, аллокация памяти

# 2. Замеры с синхронизацией
for run in range(n_runs):
    if device == "cuda":
        torch.cuda.synchronize()  # Ждём завершения всех предыдущих операций
    start = time.perf_counter()
    
    result = segmenter.segment(image)
    
    if device == "cuda":
        torch.cuda.synchronize()  # Ждём завершения текущего ядра
    end = time.perf_counter()
    times.append(end - start)
```

> ⚠️ *Без `torch.cuda.synchronize()` замеры на GPU могут быть некорректными, так как операции выполняются асинхронно.*

## 🛠️ Обработка ошибок и восстановление
- Исключения внутри `segment()` перехватываются, логируются и **не прерывают** цикл прогонов.
- Если все прогоны метода завершились ошибкой, в результатах устанавливается `mean_time = inf`, `error = "Failed/Skipped"`.
- После каждого метода выполняется `torch.cuda.empty_cache()` и `gc.collect()` для предотвращения утечек памяти.
- Исходное устройство сегментера (`.device`, `.model`) восстанавливается после замера.

## ⚡ Рекомендации по использованию
- Для стабильных результатов используйте `n_runs ≥ 5` и `warmup_runs ≥ 2`.
- При тестировании нейросетей с разными точностями (`fp16`, `bf16`) убедитесь, что сегментер поддерживает соответствующий режим.
- Для экономии места на диске установите `save_artifacts=False`, если визуальная верификация не требуется.
- При сравнении бэкендов (ONNX, TRT) предварительно экспортируйте модели и убедитесь в их корректной загрузке.

## 🤝 Зависимости
```text
torch, numpy, pandas, matplotlib, pillow, openpyxl  # для Excel-экспорта
```

## 🔗 Сравнение с другими модулями тестирования
| Модуль | Цель | Данные | Метрики |
|--------|------|--------|---------|
| `BatchClassicTester` | Консистентность реализаций | Только изображения | IoU, Dice между масками |
| `BatchClassicTester2` | Качество относительно GT | Изображения + GT маски | IoU, Dice, Precision vs GT |
| **`CpuCudaBenchmark`** | **Производительность** | Только изображения | Время, speedup, стабильность |

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