# 🚀 main.py — Оркестратор фреймворка сегментации

## 📖 Описание
Файл `main.py` является **центральной точкой входа** и оркестратором всего фреймворка. Он автоматизирует загрузку конфигураций, регистрацию методов из трёх библиотек (OpenCV, scikit-learn, PyTorch), запуск многоэтапных бенчмарков, экспорт моделей в ONNX/TensorRT, сравнение бэкендов, профилирование памяти/трансферов и генерацию итоговых отчётов.

> ⚠️ **Важно:** Данный модуль не реализует алгоритмы сегментации напрямую. Он выступает как **координатор**, объединяющий `OpenCVSegmenter`, `TorchSegmenter`, `NewTorchSegmenter`, `AutoSegmenter`, `SegmentationTester` и нейросетевые бэкенды в единый экспериментальный пайплайн.

## ✨ Ключевые возможности

### 🎛️ Гибкое управление пайплайном
| Параметр | Описание | Значение по умолчанию |
|----------|----------|----------------------|
| `use_optimizations` | Включение `torch.compile`, AMP, квантования | `True` |
| `test_classic_logic` | Запуск классических методов (CV2/Sklearn/Torch) | Из YAML |
| `test_neural_logic` | Запуск нейросетевых моделей (SegFormer, SAM, U-Net...) | `False` |
| `enable_profiling` | Детекция CPU↔GPU трансферов и замеры времени | `True` |
| `enable_benchmark_precision` | Сравнение fp32/fp16/bf16 (требует CUDA) | `True` |

### 🔄 Мульти-бэкенд бенчмаркинг
```
1. PyTorch Eager → Базовая линия
2. PyTorch Compile → reduce-overhead / max-autotune
3. ONNX Runtime → Экспорт + инференс
4. TensorRT → Оптимизированный CUDA-движок
```
Автоматический расчёт **Speedup** относительно baseline, сохранение результатов в CSV и pivot-таблицы.

### 📊 Автоматический подбор окружения
| Устройство | Точность | Компиляция | Квантование |
|------------|----------|------------|-------------|
| CPU (любое) | `fp32` | `False` | `True` (int8) |
| GPU < Pascal (6.x) | `fp32` | `False` | `False` |
| GPU Pascal/Volta (6-7.x) | `fp16` | `True` | `False` |
| GPU Ampere+ (8.x) | `bf16` | `True` | `False` |
| GPU Hopper+ (9.x) | `float8` (эксп.) | `True` | `False` |

## 🚀 Быстрый старт

### Базовый запуск
```bash
# Запуск с полными оптимизациями (рекомендуется)
python main.py

# Запуск legacy-режима (без compile/AMP, для отладки)
python -c "from main import main; main(use_optimizations=False)"
```

### Настройка через конфиг (`configs/main_config.yaml`)
```yaml
test_settings:
  test_classic_logic: true
  test_neural_logic: false
  batch_test_max_images: 50
  batch_test_mask_sample_rate: 0.2
benchmark_settings:
  n_runs: 10
  warmup_runs: 5
  force_warmup: true
export_settings:
  precisions: ["fp32", "fp16", "bf16"]
  input_shape: [1, 3, 512, 512]
  force_reexport: false
```

### Пример программного вызова
```python
from main import main

# 1. Запуск основного пайплайна
tester, results_df, comparator = main(use_optimizations=True)

# 2. Анализ результатов
if results_df is not None:
    top_methods = results_df.sort_values("IoU", ascending=False).head(5)
    print("🏆 Топ-5 методов по IoU:")
    print(top_methods[["Method", "IoU", "time_mean_s"]])
```

## ⚙️ Конфигурация

### Глобальные настройки окружения
```python
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")
```
Автоматически отключают шум логов, включают cuDNN авто-тюнинг и подготавливают среду для стабильных бенчмарков.

### Фабрики сегментеров
| Функция | Назначение | Возвращаемый тип |
|---------|------------|------------------|
| `_create_cv2_methods()` | Инициализация 20+ методов OpenCV | `Dict[str, OpenCVSegmenter]` |
| `_create_sklearn_methods()` | Инициализация 20+ методов scikit-learn | `Dict[str, SklearnSegmenter]` |
| `_create_torch_methods_factory()` | Создание Torch v1/v2 с учётом `use_compile` и `precision` | `Dict[str, TorchSegmenter / TorchSegmenter2]` |
| `_register_backend_methods_with_precision()` | Регистрация PyTorch/ONNX/TRT с суффиксами `{method}_{backend}_{precision}` | `Dict[str, Any]` |

### Управление точностью и компиляцией
```python
def get_optimal_precision(device: torch.device) -> str:
    """Авто-выбор: bf16 (Ampere+), fp16 (Pascal+), fp32 (остальные)."""

def get_compile_config(method_name: str, device: torch.device) -> Dict[str, Any]:
    """Возвращает конфигурацию torch.compile:
    - well_compiled методы: fullgraph=True, reduce-overhead
    - сложные методы: use_compile=False
    """
```

## 📚 Справочник ключевых функций

| Функция | Возвращает | Описание |
|---------|------------|----------|
| `main(use_optimizations)` | `Tuple[Tester, DataFrame, Comparator]` | Главный оркестратор. Загружает конфиг, регистрирует методы, запускает блоки тестов. |
| `load_test_images(use_image_with_mask)` | `TestImagesDict` | Загружает изображения с HuggingFace или локально. Генерирует синтетические маски при отсутствии GT. |
| `run_performance_benchmark(...)` | `Optional[pd.DataFrame]` | Двухэтапный бенчмарк: Cold → Warm-up → Hot. Расчёт speedup, сохранение отчётов. |
| `run_neural_segmentation_benchmark(...)` | `Optional[Dict]` | Запуск сравнения 11+ нейросетей на ADE20K. Генерация LaTeX-таблиц, графиков mIoU, confusion matrix. |
| `run_cpu_cuda_benchmark(...)` | `BenchmarkResult` | Сравнение времени выполнения классических методов на CPU vs GPU. |
| `generate_precision_report(...)` | `Optional[pd.DataFrame]` | Генерация CSV-отчёта: метод × точность × время × IoU_vs_fp32. |
| `parse_method_name(method_name)` | `Dict[str, str]` | Надёжный парсинг имён вида `otsu_thresholding_ONNX_fp16` → `{BaseMethod, Backend, Precision}`. |

## ⚡ Производительность и оптимизации

### Стратегии ускорения
1. **Warm-up:** `SegmentationWarmUp` и `ThresholdWarmUp` прогревают CUDA-ядра перед замерами.
2. **`torch.compile`:** Автоматически применяется к статическим графам (пороговые/граничные методы).
3. **Очистка памяти:** `_clear_gpu_memory()` вызывает `torch.cuda.empty_cache()`, `gc.collect()` перед загрузкой тяжёлых моделей.
4. **Fallback:** При отсутствии CUDA или ошибок экспорта автоматически переключается на CPU/eager mode.

### Рекомендации по запуску
```bash
# Для быстрого прототипирования
python -c "from main import main; main(use_optimizations=True)" 
# Отключите test_neural_logic в конфиге

# Для стресс-теста бэкендов
# В main.py раскомментируйте export_all_classical_methods() 
# и установите force_reexport=True

# Для отладки трансферов
export TRACK_FUNCTION_CALLS=1
python main.py  # Включит логирование вызовов через utils.function_tracker
```

## 🛠️ Обработка ошибок и устойчивость

### Graceful Degradation
```python
try:
    tester.add_method(name, segmenter)
except Exception as e:
    print(f"   ⚠️ Не удалось добавить {name}: {e}")
```
- Ошибки в отдельных методах **не прерывают** пайплайн.
- Ненайденные чекпоинты/экспорты пропускаются с предупреждением.
- `KeyboardInterrupt` корректно обрабатывается в циклах бенчмарков.

### Рекомендации по отладке
1. **Включите подробные логи:**
   ```python
   logging.getLogger("main").setLevel(logging.DEBUG)
   os.environ["DEBUG"] = "1"
   ```
2. **Проверьте конфигурацию экспорта:**
   ```python
   from main import get_compile_config
   print(get_compile_config("otsu_thresholding", torch.device("cuda")))
   ```
3. **Запустите один блок:**
   ```python
   # В main.py закомментируйте блоки 4.1–4.9, оставив только нужный
   ```

## 🤝 Интеграция с другими модулями проекта

| Модуль | Роль в `main.py` |
|--------|------------------|
| `OpenCVSegmenter` / `SklearnSegmenter` | Базовые классические методы для сравнения |
| `TorchSegmenter` / `NewTorchSegmenter` | PyTorch-реализации (v1/v2) с поддержкой AMP/compile |
| `AutoSegmenter` | Интеллектуальный выбор метода (опционально через конфиг) |
| `SegmentationTester` | Регистрация, бенчмаркинг, warm-up, генерация отчётов |
| `BatchClassicTester` | Массовое тестирование на ADE20K с GT |
| `CpuCudaBenchmark` | Сравнение производительности CPU vs GPU |
| `SegmentationBenchmark` | Нейросетевой бенчмарк (11+ моделей) |
| `BackendSegmenters` (ONNX/TRT) | Загрузка экспортированных моделей для мульти-бэкенд тестов |
| `ModelTrainer` / `NeuralModelFactory` | Обучение и загрузка нейросетевых чекпоинтов |

### Пример полного пайплайна данных
```
configs/main_config.yaml
        ↓
load_test_images() → PIL.Image / np.ndarray / GT-маска
        ↓
_create_*_methods() → SegmenterDict
        ↓
register_backend_methods() → PyTorch / ONNX / TensorRT
        ↓
tester.benchmark_methods() → DataFrame (time, IoU, speedup)
        ↓
generate_precision_report() → precision_benchmark.csv
        ↓
visualize_gt_results() → Графики метрик
        ↓
./data/ → Отчёты, маски, графики, CSV, LaTeX
```

## 📦 Зависимости

### Обязательные
```text
torch>=2.0.0          # Ядро фреймворка, compile, AMP
opencv-python>=4.5    # Классические методы, загрузка изображений
scikit-learn>=1.0     # DBSCAN, MeanShift, кластеризация
numpy>=1.20           # Векторизованные операции, маски
pandas>=1.3           # Агрегация бенчмарков, pivot-таблицы
matplotlib>=3.4       # Визуализация графиков, heatmaps
tqdm>=4.60            # Прогресс-бары в циклах
PyYAML>=6.0           # Загрузка конфигурации
huggingface_hub>=0.10 # Загрузка тестовых данных
```

### Опциональные (для расширенных блоков)
```text
transformers>=4.30    # Нейросетевой бенчмарк (SegFormer, Mask2Former)
torchvision>=0.15     # Загрузка предобученных моделей
onnx>=1.12            # Экспорт/загрузка ONNX-моделей
torch-tensorrt>=1.4   # Экспорт в TensorRT Engine
segmentation-models-pytorch  # U-Net, FPN, PSPNet из SMP
```

### Установка
```bash
# Минимальный набор (классические методы + бенчмарки)
pip install torch opencv-python scikit-learn numpy pandas matplotlib tqdm PyYAML huggingface_hub

# Полный набор (нейросети + экспорт + SMP)
pip install transformers torchvision onnx torch-tensorrt segmentation-models-pytorch

# Проверка установки
python -c "from main import main; print('✅ Main orchestrator OK')"
```

## 📄 Лицензия

Проект распространяется под лицензией **MIT**. См. файл [LICENSE](LICENSE) для деталей.

```
MIT License

Copyright (c) 2026 Segmentation Project contributors

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

> 💡 **Совет:** Для воспроизводимости экспериментов зафиксируйте `torch.manual_seed(42)` и используйте `torch.backends.cudnn.deterministic = True` при запуске `main.py` на разных машинах. Отключите `torch.backends.cudnn.benchmark`, если важна стабильность замеров времени.

```python
# В начале main() или в конфиге
import torch, random
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Для стабильных бенчмарков
```