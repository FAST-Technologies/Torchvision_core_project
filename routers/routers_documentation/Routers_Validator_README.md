# 📋 routers/validate.py — README

> **Модуль валидации кросс-библиотечных реализаций методов сегментации**

## 🎯 Назначение

Модуль `routers/validate.py` предоставляет REST API для асинхронной валидации согласованности реализаций методов сегментации изображений между различными библиотеками:

- **PyTorch** (`TorchSegmenter`)
- **OpenCV** (`OpenCVSegmenter`)  
- **scikit-learn** (`SklearnSegmenter`)

Основная цель — автоматическое сравнение метрик качества (IoU, Dice, Precision, Recall, F1, MAE, Hausdorff) и времени выполнения для выявления расхождений между реализациями одного и того же алгоритма.

---

## 🚀 Быстрый старт

### Запуск сервера

```bash
# Установка зависимостей
pip install fastapi uvicorn pillow numpy opencv-python scikit-learn torch

# Запуск API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Пример запроса через curl

```bash
# 1. Запуск валидации
curl -X POST "http://localhost:8000/api/validate/start" \
  -F "file=@test_image.jpg" \
  -F "primary_library=torch" \
  -F "reference_library=opencv" \
  -F "methods_filter=threshold"

# Ответ: {"task_id": "abc123...", "status": "running"}

# 2. Проверка статуса
curl "http://localhost:8000/api/validate/status/abc123..."

# 3. Отмена задачи (опционально)
curl -X DELETE "http://localhost:8000/api/validate/abc123..."
```

---

## 📡 API Endpoints

### `POST /api/validate/start` — Запуск валидации

Запускает асинхронную задачу кросс-библиотечного сравнения методов.

#### Параметры формы

| Параметр | Тип | Обязательный | Значения по умолчанию | Описание |
|----------|-----|--------------|----------------------|----------|
| `file` | `UploadFile` | ✅ | — | Изображение для валидации (JPG/PNG) |
| `primary_library` | `str` | ❌ | `"torch"` | Библиотека для первичной реализации: `"torch"`, `"opencv"`, `"sklearn"` |
| `reference_library` | `str` | ❌ | `"opencv"` | Библиотека для эталонной реализации |
| `methods_filter` | `str` | ❌ | `None` | Фильтр методов: `"threshold"`, `"edge"`, `"region"`, `"clustering"`, `None` (все) |

#### Ответ

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running"
}
```

---

### `GET /api/validate/status/{task_id}` — Статус задачи

Возвращает текущий статус и результаты валидации.

#### Параметры пути

| Параметр | Тип | Описание |
|----------|-----|----------|
| `task_id` | `str` | UUID задачи, полученный из `/start` |

#### Ответ (в процессе выполнения)

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 45.5,
  "processed": 12,
  "total_methods": 26,
  "elapsed_ms": 3420.5,
  "message": "Обработка threshold_sauvola (12/26)"
}
```

#### Ответ (по завершении)

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "processed": 26,
  "total_methods": 26,
  "elapsed_ms": 8750.2,
  "message": "Готово",
  "passed": 18,
  "warning": 5,
  "failed": 3,
  "methods_tested": 26,
  "report_dir": "./data/validation_web",
  "benchmark": {
    "methods_count": 26,
    "passed": 18,
    "warning": 5,
    "failed": 3,
    "avg_torch_time": 0.045,
    "avg_iou": 0.923,
    "data": [...]
  },
  "results": [
    {
      "method": "otsu_thresholding",
      "success": true,
      "validation_status": "PASS",
      "iou": 0.998,
      "dice": 0.999,
      "pixel_accuracy": 0.997,
      "precision": 0.996,
      "recall": 0.998,
      "f1_score": 0.997,
      "mae": 0.002,
      "primary_time": 0.015,
      "reference_time": 0.012,
      "time_diff": 0.003,
      "original_b64": "data:image/png;base64,...",
      "primary_mask_b64": "data:image/png;base64,...",
      "reference_mask_b64": "data:image/png;base64,...",
      "difference_b64": "data:image/png;base64,..."
    }
  ]
}
```

---

### `DELETE /api/validate/{task_id}` — Отмена задачи

Отменяет выполнение задачи валидации.

#### Ответ

```json
{
  "status": "cancelled",
  "message": "Отменено пользователем"
}
```

---

## 🧩 Фильтры методов

Параметр `methods_filter` позволяет ограничить набор тестируемых методов:

| Значение | Описание | Примеры методов |
|----------|----------|----------------|
| `threshold` | Пороговые методы | `otsu_thresholding`, `adaptive_thresholding`, `threshold_sauvola` |
| `edge` | Граничные методы | `sobel_edge`, `canny_edge`, `laplacian_edge` |
| `region` | Региональные методы | `region_growing`, `floodfill`, `split_and_merge` |
| `clustering` | Методы кластеризации | `kmeans_segmentation`, `dbscan_segmentation`, `meanshift` |
| `None` | **Все методы** (по умолчанию) | Полный набор (~100 методов) |

---

## 📊 Метрики валидации

Для каждого метода вычисляются следующие метрики:

### Метрики качества сегментации

| Метрика | Диапазон | Описание |
|---------|----------|----------|
| `iou` | [0, 1] | Intersection over Union (Jaccard index) |
| `dice` | [0, 1] | Коэффициент Дайса (F1 для масок) |
| `pixel_accuracy` | [0, 1] | Доля правильно классифицированных пикселей |
| `precision` | [0, 1] | Точность: TP / (TP + FP) |
| `recall` | [0, 1] | Полнота: TP / (TP + FN) |
| `f1_score` | [0, 1] | Гармоническое средное precision и recall |
| `mae` | [0, 1] | Mean Absolute Error между масками |
| `hausdorff_distance` | ≥0 | Максимальное расстояние между границами |

### Метрики производительности

| Метрика | Единицы | Описание |
|---------|---------|----------|
| `primary_time` | секунды | Время выполнения первичной реализации |
| `reference_time` | секунды | Время выполнения эталонной реализации |
| `time_diff` | секунды | Абсолютная разница во времени |

### Статусы валидации

| Статус | Критерий | Интерпретация |
|--------|----------|---------------|
| `PASS` | IoU ≥ 0.80 и остальные метрики в пороге | ✅ Реализации согласованы |
| `WARNING` | 0.50 ≤ IoU < 0.80 | ⚠️ Возможны расхождения, требуется проверка |
| `FAIL` | IoU < 0.50 или ошибка выполнения | ❌ Реализации не согласованы |

---

## 🖼️ Визуализация результатов

Ответ API включает base64-кодированные изображения для визуальной проверки:

| Поле | Описание |
|------|----------|
| `original_b64` | Исходное изображение (grayscale) |
| `primary_mask_b64` | Маска от первичной библиотеки |
| `reference_mask_b64` | Маска от эталонной библиотеки |
| `difference_b64` | Карта различий (белые пиксели = несовпадения) |

**Использование в frontend:**
```html
<img src="{{ result.primary_mask_b64 }}" alt="Primary mask">
<img src="{{ result.reference_mask_b64 }}" alt="Reference mask">
<img src="{{ result.difference_b64 }}" alt="Difference map">
```

---

## ⚙️ Конфигурация

### Глобальные настройки

```python
# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("validate")

# Директория для отчётов
VALIDATION_OUTPUT_DIR = "./data/validation_web"

# Таймауты и лимиты
MAX_IMAGE_SIZE = 4096  # Максимальная сторона изображения в пикселях
MAX_TASK_DURATION = 300  # Максимальное время выполнения задачи (сек)
```

### Параметры сегментеров

Каждый метод сегментации поддерживает собственные параметры, которые передаются через `**params`. Примеры:

```python
# Otsu thresholding
{"method": "otsu_thresholding"}  # Без параметров

# Adaptive thresholding  
{"method": "adaptive_thresholding", "block_size": 11, "C": 2}

# Canny edge
{"method": "canny_edge", "low": 0.1, "high": 0.3, "sigma": 1.0}

# K-means clustering
{"method": "kmeans_segmentation", "k": 3, "max_iter": 100}
```

---

## 🔄 Асинхронная архитектура

### Жизненный цикл задачи

```
1. Клиент → POST /start
   ↓
2. Генерация task_id + создание записи в _validation_tasks
   ↓
3. Запуск _run_validation_task() в фоне (asyncio.create_task)
   ↓
4. Пошаговое выполнение с обновлением прогресса (через _validation_lock)
   ↓
5. Сохранение результатов + установка status="completed"
   ↓
6. Клиент → GET /status/{task_id} для получения результатов
```

### Блокировки и потокобезопасность

```python
# Глобальная асинхронная блокировка для доступа к задачам
_validation_lock = asyncio.Lock()

# Пример безопасного обновления статуса
async with _validation_lock:
    _validation_tasks[task_id]["progress"] = 45.5
    _validation_tasks[task_id]["message"] = "Processing..."
```

### Обработка длительных операций

Тяжёлые вычисления (сегментация) выполняются в отдельном потоке через `asyncio.to_thread()`:

```python
result = await asyncio.to_thread(
    _process_single_method,
    method_name, params, img_array,
    primary_class, reference_class, validator
)
```

---

## 🧪 Тестирование

### Локальное тестирование endpoint'ов

```python
import httpx
import asyncio

async def test_validation():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Запуск
        with open("test.jpg", "rb") as f:
            response = await client.post(
                "/api/validate/start",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={
                    "primary_library": "torch",
                    "reference_library": "opencv",
                    "methods_filter": "threshold"
                }
            )
        task_id = response.json()["task_id"]
        
        # Ожидание завершения
        while True:
            status = await client.get(f"/api/validate/status/{task_id}")
            data = status.json()
            if data["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(1)
        
        print(f"✅ Completed: {data['passed']} passed, {data['failed']} failed")
```

### Pytest fixtures

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def test_image(tmp_path):
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    path = tmp_path / "test.jpg"
    img.save(path)
    return path
```

---

## ⚠️ Обработка ошибок

### Типичные ошибки и решения

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `404 Task not found` | Неверный `task_id` или задача очищена | Проверить UUID, увеличить TTL задач |
| `422 Validation Error` | Неверный тип файла или параметра | Проверить `Content-Type`, допустимые значения `library` |
| `500 Internal Error` | Ошибка в реализации сегментера | Проверить логи, обновить зависимости |
| `MemoryError` | Слишком большое изображение | Добавить ресайз до обработки (`MAX_IMAGE_SIZE`) |

### Логирование

```python
# Уровни логирования
logger.debug("Детальная отладка")      # При --log-level=debug
logger.info("Информационные сообщения") # По умолчанию
logger.warning("Предупреждения")        # Не критичные проблемы
logger.error("Ошибки выполнения")       # Критичные сбои

# Просмотр логов в реальном времени
tail -f logs/validate.log | grep "ERROR\|WARNING"
```

---

## 🚀 Оптимизация производительности

### Рекомендации для продакшена

1. **Кэширование результатов**:
   ```python
   # Добавить хэширование входных данных
   import hashlib
   cache_key = hashlib.md5(image.tobytes() + method_name.encode()).hexdigest()
   ```

2. **Батчинг методов**:
   ```python
   # Группировать методы по библиотеке для минимизации переинициализации
   methods_by_lib = {"torch": [...], "opencv": [...]}
   ```

3. **Лимиты ресурсов**:
   ```bash
   # Запуск с ограничениями через systemd
   MemoryLimit=2G
   CPUQuota=200%
   ```

4. **Асинхронное чтение файлов**:
   ```python
   # Использовать aiofiles для больших изображений
   import aiofiles
   async with aiofiles.open(path, 'rb') as f:
       content = await f.read()
   ```

### Мониторинг

```python
# Добавить метрики Prometheus
from prometheus_client import Counter, Histogram

VALIDATION_REQUESTS = Counter('validate_requests_total', 'Total validation requests', ['status'])
VALIDATION_DURATION = Histogram('validate_duration_seconds', 'Validation duration')

@VALIDATION_DURATION.time()
async def start_validation(...):
    try:
        # ... логика ...
        VALIDATION_REQUESTS.labels(status='success').inc()
    except Exception:
        VALIDATION_REQUESTS.labels(status='error').inc()
        raise
```

---

## 📁 Структура выходных данных

```
./data/validation_web/
├── validation_report.md          # Markdown-отчёт
├── benchmark_results.csv         # Сводная таблица метрик
├── masks/                        # Примеры масок для визуальной проверки
│   ├── otsu_thresholding_primary.png
│   ├── otsu_thresholding_reference.png
│   └── otsu_thresholding_diff.png
└── charts/                       # Графики сравнения
    ├── iou_comparison.png
    └── time_comparison.png
```

---

## 🔐 Безопасность

### Ограничения загрузки

```python
# Максимальный размер файла (10 MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Разрешённые расширения
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def validate_file(file: UploadFile) -> bool:
    ext = os.path.splitext(file.filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS
```

### Санитизация входных данных

```python
# Проверка библиотек
VALID_LIBRARIES = {"torch", "opencv", "sklearn"}
if primary_library not in VALID_LIBRARIES:
    raise HTTPException(400, detail=f"Invalid library: {primary_library}")

# Очистка параметров от опасных значений
def sanitize_params(params: Dict) -> Dict:
    return {k: v for k, v in params.items() 
            if isinstance(v, (str, int, float, bool))}
```

---

## 📈 Масштабирование

### Горизонтальное масштабирование

```yaml
# docker-compose.yml для реплик API
version: '3.8'
services:
  validate-api:
    image: segmentation-validator:latest
    replicas: 3
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
```

### Распределённая очередь задач

Для высокой нагрузки рекомендуется заменить `asyncio.create_task()` на очередь задач:

```python
# С использованием Celery + Redis
from celery import Celery

app = Celery('validate', broker='redis://redis:6379/0')

@app.task(bind=True, max_retries=3)
def run_validation_task(self, task_id, file_content, ...):
    # Логика валидации
    ...
```

---

## 🤝 Вклад в разработку

### Добавление нового метода

1. Реализовать метод в соответствующем сегментере (`TorchSegmenter`, `OpenCVSegmenter`, `SklearnSegmenter`)
2. Добавить профиль в `METHODS_BY_LIBRARY` в `AutoSegmenter`
3. Обновить `MethodConfig` в `TorchImplementationValidator`
4. Протестировать кросс-библиотечную согласованность

### Чеклист перед мерджем

- [ ] Все методы имеют согласованные сигнатуры `segment(image, **kwargs)`
- [ ] Возвращаемые маски имеют одинаковый формат (uint8, 0/255)
- [ ] Добавлены юнит-тесты для новых методов
- [ ] Обновлена документация API (OpenAPI/Swagger)

---

## 📄 Лицензия

MIT License — см. файл `LICENSE` в корне репозитория.

---

> 💡 **Совет**: Для интерактивного тестирования используйте встроенную документацию Swagger UI:  
> 🌐 `http://localhost:8000/docs` или Redoc: `http://localhost:8000/redoc`