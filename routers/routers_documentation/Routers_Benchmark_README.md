# 🚀 Benchmark API Router — Управление бенчмарками сегментации

## 📖 Описание
Модуль `routers/benchmark.py` предоставляет **REST API интерфейс** для асинхронного запуска, мониторинга и управления бенчмарками нейросетевых моделей сегментации изображений.

> ⚠️ **Важно:** Данный модуль работает исключительно с **нейросетевыми методами** (SegFormer, Mask2Former, SAM, YOLOv8, SMP-модели). Классические алгоритмы (OpenCV, sklearn, PyTorch-эвристики) не поддерживаются в данном бенчмарке.

## ✨ Ключевые возможности

### 🔄 Поддерживаемые модели сегментации

| Категория | Модели | Описание |
|-----------|--------|----------|
| **Transformers** | SegFormer (B2/B5), Mask2Former, MaskFormer, OneFormer, DPT-Large, UPerNet | SOTA-архитектуры на базе Vision Transformers |
| **SAM Family** | MobileSAM, SAM2 | Segment Anything Model с адаптацией для ADE20K |
| **YOLOv8-Seg** | yolov8n-seg, yolov8s-seg, yolov8m-seg | Быстрые детекторы с масками реального времени |
| **SMP/CNN** | U-Net, DeepLabV3+, FPN, PSPNet, FCN, SegNet | Классические архитектуры с предобученными чекпоинтами |
| **TorchVision** | Mask R-CNN (ResNet-50-FPN) | Instance segmentation из torchvision.models |

### 🎛️ Функционал бенчмарка

| Возможность | Описание | Реализация |
|-------------|----------|-----------|
| **Асинхронный запуск** | Фоновое выполнение через `BackgroundTasks` FastAPI | Не блокирует основной поток, поддержка polling |
| **Прогресс-трекинг** | Обновление статуса в реальном времени (0–100%) | `benchmark_tasks` in-memory хранилище |
| **Метрики качества** | mIoU, Pixel Accuracy, F1-score, время инференса | `SegmentationMetrics.calculate_all_metrics()` |
| **Визуализация** | Оверлеи масок, графики сравнения, base64-экспорт | Matplotlib + PIL + base64 encoding |
| **Гибкая конфигурация** | Фильтрация моделей, параметры инференса, палитры | JSON-конфиг через `BenchmarkConfig` TypedDict |
| **Upload изображений** | Поддержка кастомных изображений и GT-масок | `UploadFile` через multipart/form-data |

### 🎚️ Параметры конфигурации бенчмарка

```python
class BenchmarkConfig(TypedDict, total=False):
    inference: Dict[str, Any]      # alpha, warmup_runs
    filters: Dict[str, Any]        # min_iou, only_passed
    visualization: Dict[str, Any]  # show_overlay, color_palette
    models_to_run: Optional[List[str]]  # whitelist моделей
```

## 🚀 Быстрый старт

### 1️⃣ Запуск бенчмарка с параметрами по умолчанию

```bash
curl -X POST http://localhost:8000/api/benchmark/start \
  -F "use_default_image=true"
```

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### 2️⃣ Запуск с кастомным изображением и конфигурацией

```bash
curl -X POST http://localhost:8000/api/benchmark/start \
  -F "image=@test_image.jpg" \
  -F "gt_mask=@ground_truth.png" \
  -F "use_default_image=false" \
  -F 'config={"inference":{"alpha":0.7,"warmup_runs":3},"models_to_run":["segformer","sam2"],"visualization":{"color_palette":"coco"}}'
```

### 3️⃣ Проверка статуса задачи

```bash
curl http://localhost:8000/api/benchmark/status/550e8400-e29b-41d4-a716-446655440000
```

**Response (running):**
```json
{
  "status": "running",
  "progress": 45.5,
  "message": "🔄 sam2: инференс...",
  "results": null,
  "error_details": null
}
```

**Response (completed):**
```json
{
  "status": "completed",
  "progress": 100.0,
  "message": "Готово",
  "results": {
    "summary": {
      "segformer": {"mIoU": 0.78, "pixel_acc": 0.92, "time_ms": 145.2},
      "sam2": {"mIoU": 0.65, "pixel_acc": 0.88, "time_ms": 89.1}
    },
    "output_dir": "./data/benchmark_550e8400",
    "charts": {"metrics_plot_b64": "iVBORw0KGgoAAAANSUh..."}
  }
}
```

### 4️⃣ Отмена задачи

```bash
curl -X DELETE http://localhost:8000/api/benchmark/550e8400-e29b-41d4-a716-446655440000
```

**Response:**
```json
{"status": "cancelled"}
```

### 5️⃣ Python-клиент для взаимодействия

```python
import requests
import time

BASE_URL = "http://localhost:8000/api/benchmark"

# Запуск бенчмарка
response = requests.post(
    f"{BASE_URL}/start",
    data={"use_default_image": True},
    files={"config": ('config.json', '{"models_to_run": ["segformer", "mask2former"]}', 'application/json')}
)
task_id = response.json()["task_id"]

# Polling статуса
while True:
    status = requests.get(f"{BASE_URL}/status/{task_id}").json()
    print(f"Progress: {status['progress']}% - {status['message']}")
    
    if status["status"] == "completed":
        print("✅ Benchmark finished!")
        print(f"Results: {status['results']['summary']}")
        break
    elif status["status"] == "failed":
        print(f"❌ Error: {status['error_details']}")
        break
    
    time.sleep(2)  # Интервал опроса
```

## ⚙️ Конфигурация

### Параметры эндпоинта `/start`

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `image` | `UploadFile` | `None` | Загружаемое изображение (RGB) |
| `gt_mask` | `UploadFile` | `None` | Ground truth маска (grayscale) для расчёта метрик |
| `use_default_image` | `bool` | `true` | Использовать тестовое изображение из ADE20K fixtures |
| `image_path` | `str` | `None` | Путь к локальному изображению (если `use_default_image=false`) |
| `config` | `str` (JSON) | `None` | JSON-строка с конфигурацией бенчмарка |

### Структура `config` (BenchmarkConfig)

```json
{
  "inference": {
    "alpha": 0.6,           // Прозрачность оверлея [0.0, 1.0]
    "warmup_runs": 2        // Число прогревочных прогонов
  },
  "filters": {
    "min_iou": 0.5,         // Минимальный IoU для включения в отчёт
    "only_passed": false    // Показывать только модели с IoU >= min_iou
  },
  "visualization": {
    "show_overlay": true,   // Показывать оверлей маски на изображении
    "show_gt": true,        // Показывать ground truth в визуализации
    "color_palette": "ade"  // Палитра: "ade" | "coco" | "cityscapes"
  },
  "models_to_run": ["segformer", "sam2"]  // Whitelist моделей (опционально)
}
```

### Поддерживаемые палитры для визуализации

| Палитра | Назначение | Количество классов |
|---------|-----------|-------------------|
| `ade` | ADE20K dataset | 150 |
| `coco` | COCO dataset | 91 |
| `cityscapes` | Cityscapes dataset | 19 |

### Type Aliases

```python
PathLike = Union[str, Path]
BenchmarkTaskDict = Dict[str, BenchmarkTask]
ModelLoadStep = Tuple[str, Callable[..., Any], Dict[str, Any]]
```

## 📚 Справочник эндпоинтов

### 🔹 Управление задачами

| Эндпоинт | Метод | Описание | Параметры | Возвращает |
|----------|-------|----------|-----------|-----------|
| `/start` | `POST` | Запуск новой задачи бенчмарка | `image`, `gt_mask`, `use_default_image`, `image_path`, `config` | `{"task_id": "uuid"}` |
| `/status/{task_id}` | `GET` | Получение статуса и прогресса задачи | `task_id` (path) | `BenchmarkTask` JSON |
| `/{task_id}` | `DELETE` | Отмена или удаление задачи | `task_id` (path) | `{"status": "cancelled"\|"deleted"}` |

### 🔹 Диагностика и отладка

| Эндпоинт | Метод | Описание | Параметры | Возвращает |
|----------|-------|----------|-----------|-----------|
| `/health` | `GET` | Проверка доступности системы и ресурсов | — | Статус системы, VRAM, активные задачи |
| `/debug` | `GET` | Сводка по всем задачам (для разработки) | — | Список задач со статусами |
| `/debug/{task_id}` | `GET` | Отладочная информация о конкретной задаче | `task_id` (path) | Детали задачи + метаданные |

### Примеры ответов

#### `/health`
```json
{
  "status": "ok",
  "cuda_available": true,
  "device_name": "NVIDIA GeForce RTX 4090",
  "vram_mb": 24576.0,
  "vram_allocated_mb": 3245.8,
  "vram_free_mb": 21330.2,
  "reserved_vram_mb": 4096.0,
  "active_tasks": 2
}
```

#### `/debug/{task_id}`
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 45.2,
  "message": "🔄 mask2former: загрузка...",
  "results_keys": null,
  "last_updated": 123456.789
}
```

## 🔄 Конвейер выполнения бенчмарка

### 📊 Прогресс-трекинг: этапы выполнения

```
0–5%   : 🔍 Инициализация и проверка окружения
5–75%  : 📦 Загрузка моделей (равномерное распределение)
75–100%: 🚀 Инференс и сохранение результатов
```

### 🔄 Логика `run_benchmark()` (Background Task)

```python
async def run_benchmark(task_id, req, image_file, gt_file, config):
    # 1. Инициализация задачи
    benchmark_tasks[task_id] = {
        "status": "running", "progress": 0, "message": "Инициализация...", ...
    }
    
    # 2. Проверка VRAM
    if torch.cuda.is_available() and vram_gb < 20:
        logger.warning(f"⚠️ Low VRAM: {vram_gb:.1f} GB")
    
    # 3. Загрузка изображения
    if image_file:
        image_input = Image.open(image_file.file).convert("RGB")
    elif req.image_path and os.path.exists(req.image_path):
        image_input = Image.open(req.image_path).convert("RGB")
    else:
        # Fallback на default из HuggingFace
        image_input = hf_hub_download(...)
    
    # 4. Инициализация SegmentationBenchmark
    bench = SegmentationBenchmark(
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_classes=150,
        gt_mask=gt_mask,  # из gt_file или default
        palette=palette_func()
    )
    
    # 5. Поэтапная загрузка моделей с обновлением прогресса
    for i, (key, load_fn, kwargs) in enumerate(model_load_steps):
        benchmark_tasks[task_id]["progress"] = 5 + (i / len(steps)) * 70
        benchmark_tasks[task_id]["message"] = f"🔄 {key}: загрузка..."
        try:
            load_fn(**kwargs)
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.error(f"❌ Failed to load {key}: {e}")
    
    # 6. Инференс с прогресс-трекингом
    await bench.compare_step_by_step(
        image_input=image_input,
        alpha=alpha,
        task_id=task_id,
        benchmark_tasks=benchmark_tasks  # ← callback для обновления
    )
    
    # 7. Сохранение результатов
    bench.save_results(out_dir)
    bench.plot_all_metrics(path=f"{out_dir}/plot_all.png")
    
    # 8. Финализация
    benchmark_tasks[task_id].update({
        "status": "completed",
        "progress": 100,
        "results": {"summary": summary, "output_dir": out_dir, ...}
    })
```

### 🛡️ Обработка ошибок

```python
try:
    # ... основной код бенчмарка ...
    benchmark_tasks[task_id].update({
        "status": "completed",
        "progress": 100,
        "results": {...}
    })
except Exception as e:
    import traceback
    benchmark_tasks[task_id].update({
        "status": "failed",
        "message": str(e),
        "error_details": {
            "error_type": type(e).__name__,
            "failed_at": benchmark_tasks[task_id]["message"],
            "traceback": traceback.format_exc() if DEBUG else None
        }
    })
    logger.error(f"Benchmark {task_id} failed: {e}", exc_info=True)
```

## 📊 Производительность и ресурсы

### Требования к оборудованию

| Компонент | Минимум | Рекомендуется |
|-----------|---------|--------------|
| **GPU VRAM** | 8 ГБ | ≥20 ГБ (RTX 3090/4090, A100) |
| **CUDA Compute Capability** | 6.0+ | 8.0+ (Ampere) для fp16 |
| **RAM** | 16 ГБ | 32+ ГБ |
| **Диск** | 50 ГБ свободного места | 100+ ГБ для кэша моделей |

### Ожидаемое время выполнения (16 моделей, 512×512, RTX 4090)

```
📦 Загрузка моделей: ~2–5 минут (зависит от сети и кэша)
🚀 Инференс: ~3–8 минут (зависит от количества моделей)
📊 Визуализация: ~30 секунд
─────────────────────────────────────
⏱️  Итого: ~6–14 минут
```

### Оптимизации

1. **Кэширование моделей**: Предварительно скачайте чекпоинты в `MODELS_DIR` для ускорения запуска.
2. **Whitelist моделей**: Используйте `models_to_run` для запуска только нужных моделей.
3. **Прогрев инференса**: Параметр `warmup_runs` улучшает стабильность замеров времени.
4. **Очистка памяти**: Автоматический `torch.cuda.empty_cache()` после каждой модели.

## 🛠️ Обработка ошибок и устойчивость

### Валидация входных данных

```python
# Проверка config JSON
if config:
    try:
        config_dict = cast(BenchmarkConfig, json.loads(config))
    except json.JSONDecodeError:
        raise HTTPException(422, detail="Invalid config JSON")

# Проверка существования image_path
if not req.use_default_image and req.image_path:
    if not os.path.exists(req.image_path):
        raise HTTPException(404, detail=f"Image not found: {req.image_path}")

# Проверка чекпоинтов перед загрузкой
cp: Optional[str] = kwargs.get("checkpoint_path")
if cp and not os.path.exists(cp):
    logger.warning(f"⚠️ Checkpoint not found: {cp}, skipping {key}")
    continue
```

### Сериализация numpy-типов в JSON

```python
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None  # JSON не поддерживает NaN/Inf
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def safe_json_response(content: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        content=content,
        status_code=status_code,
        media_type="application/json",
    )
```

### Рекомендации по отладке

1. **Включите DEBUG-логирование**:
   ```python
   import logging
   logging.getLogger("benchmark").setLevel(logging.DEBUG)
   ```

2. **Проверьте пути к моделям**:
   ```bash
   python -c "from utils.config import settings; print(settings.MODEL_DIR)"
   ls -la ./models/
   ```

3. **Тестируйте с одной моделью**:
   ```json
   {"models_to_run": ["segformer"]}
   ```

4. **Мониторьте VRAM во время выполнения**:
   ```bash
   watch -n 1 nvidia-smi
   ```

## 🤝 Зависимости

### Обязательные
```text
fastapi>=0.95.0          # Веб-фреймворк и BackgroundTasks
pydantic>=2.0.0          # Валидация данных и TypedDict
torch>=2.0.0             # Инференс моделей
numpy>=1.24.0            # Обработка тензоров и метрик
Pillow>=9.0.0            # Работа с изображениями
huggingface-hub>=0.16.0  # Загрузка тестовых данных
```

### Опциональные (для расширенного функционала)
```bash
# Для визуализации и графиков
pip install matplotlib seaborn

# Для работы с ADE20K и другими датасетами
pip install datasets

# Для продвинутого логирования
pip install loguru
```

### Установка
```bash
# Базовая установка
pip install -r requirements.txt

# С опциональными зависимостями
pip install -r requirements-optional.txt
```

## 🔗 Интеграция с другими модулями проекта

| Модуль | Использование Benchmark API |
|--------|----------------------------|
| `SegmentationBenchmark` | Ядро бенчмарка: загрузка моделей, инференс, метрики |
| `utils.config` / `utils.paths` | Конфигурация путей и параметров проекта |
| `utils.palettes` | Палитры для визуализации (ade, coco, cityscapes) |
| `testing.SegmentationMetrics` | Расчёт IoU, Dice, F1, Pixel Accuracy |
| `segmenters.NewTorchSegmenter` | Классические методы (не используются в этом бенчмарке) |
| `segmenters.NeuralSegmenter` | Нейросетевые модели (основной источник для бенчмарка) |

### Пример интеграции в основной `main.py`

```python
from routers.benchmark import router as benchmark_router
from fastapi import FastAPI

app = FastAPI()
app.include_router(benchmark_router)

# Теперь доступны эндпоинты:
# - POST /api/benchmark/start
# - GET  /api/benchmark/status/{task_id}
# - DELETE /api/benchmark/{task_id}
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

> 💡 **Совет**: Для продакшен-развёртывания рекомендуется заменить in-memory хранилище `benchmark_tasks` на **Redis** или **Celery** для распределённого выполнения и отказоустойчивости.