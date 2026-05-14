# AutoSegmenter API — Бэкенд на FastAPI

> **Версия:** 2.0  
> **Фреймворк:** FastAPI + PyTorch + OpenCV + scikit-learn  
> **Язык:** Python 3.9+

## 📋 Обзор

`backend/main.py` — это улучшенная версия REST API бэкенда для системы интеллектуальной сегментации изображений **AutoSegmenter**. API предоставляет единый интерфейс для:

- 🎯 **Автоматического выбора** оптимального метода сегментации на основе характеристик изображения
- ⚡ **Классических методов** (пороговые, градиентные, кластеризация, активные контуры, watershed)
- 🧠 **Нейросетевых моделей** (SegFormer, Mask2Former, SAM, YOLOv8, SMP, TorchVision)
- 📊 **Расчёта метрик качества** при наличии Ground Truth масок
- 🔍 **Анализа изображений** и генерации рекомендаций

---

## ✨ Улучшения версии 2.0

| № | Исправление | Описание |
|---|-------------|----------|
| 1 | 🔄 Убран дублирующий маршрут `/api/methods` | Единая точка входа для получения методов |
| 2 | 🎯 Поле `best_for` в рекомендациях | Клиент видит, для каких типов изображений метод оптимален |
| 3 | 🧠 LRU-кеш нейронных моделей | Модель загружается один раз, макс. 3 модели в кеше |
| 4 | 🛡️ Безопасная обработка исключений | `except (json.JSONDecodeError, ValueError)` вместо `bare except` |
| 5 | 🔗 Объединение функций base64-кодирования | Единая функция `arr_to_b64()` для всех типов массивов |
| 6 | 🏥 Добавлены `/api/health` и `/api/cache_info` | Мониторинг состояния системы и кеша |
| 7 | ⚠️ Корректные коды ошибок | `HTTP 422` вместо `500` при невалидных входных данных |
| 8 | ⏱️ Возврат `elapsed_ms` и `library` | Клиент получает метаданные выполнения |
| 9 | ⚙️ Корректный мёрдж параметров | Пользовательские параметры объединяются с дефолтными |

---

## 🚀 Быстрый старт

### Требования

```bash
Python >= 3.9
CUDA >= 11.7 (опционально, для GPU-ускорения)
```

### Установка зависимостей

```bash
cd backend
pip install -r requirements.txt
```

### Запуск сервера

```bash
# Режим разработки с авто-релоадом
python main.py

# Или через uvicorn напрямую
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Продакшн-режим
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Проверка работы

```bash
# Health check
curl http://localhost:8000/api/health

# Swagger UI
open http://localhost:8000/docs

# ReDoc
open http://localhost:8000/redoc
```

---

## 📡 API Endpoints

### 🔍 Мониторинг

#### `GET /api/health`
Возвращает статус системы, информацию о GPU и активных задачах.

```json
{
  "status": "ok",
  "cuda_available": true,
  "device_name": "NVIDIA GeForce RTX 4090",
  "vram_mb": 24576.0,
  "vram_allocated_mb": 1024.5,
  "vram_free_mb": 23551.5,
  "reserved_vram_mb": 2048.0,
  "active_tasks": 0,
  "cached_models": 2,
  "cache_max": 3
}
```

#### `GET /api/cache_info`
Информация о закэшированных нейронных моделях.

```json
{
  "count": 2,
  "models": [
    "{\"model_type\": \"segformer\", \"model_name\": \"nvidia/segformer-b2...}",
    "{\"model_type\": \"mask2former\", \"model_name\": \"facebook/mask2former...}"
  ]
}
```

### 📚 Библиотеки методов

#### `GET /api/methods_library`
Получение всех доступных методов сегментации.

**Параметры:**
| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `library` | `str` | ❌ | Фильтр по библиотеке: `opencv`, `sklearn`, `torch` |

**Пример запроса:**
```bash
curl "http://localhost:8000/api/methods_library?library=opencv"
```

**Пример ответа:**
```json
{
  "methods": {
    "otsu_thresholding": {
      "name": "otsu_thresholding",
      "library": "opencv",
      "avg_iou": 0.75,
      "avg_time_ms": 15.0,
      "memory_mb": 50,
      "robustness": 0.8,
      "description": "Автоматический порог Оцу...",
      "best_for": ["document", "natural"],
      "defaults": {},
      "schema": {}
    }
  }
}
```

#### `GET /api/methods`
Алиас `/api/methods_library` для обратной совместимости.

### 🎨 Сегментация

#### `POST /api/segment`
**Основной эндпоинт** для выполнения сегментации изображения.

**Параметры формы:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `file` | `UploadFile` | ✅ | Входное изображение (JPG/PNG) |
| `mode` | `str` | `"classical"` | Режим: `classical` \| `neural` |
| `task` | `str` | `"semantic"` | Задача: `semantic` \| `instance` \| `panoptic` |
| `model` | `str` | `"segformer_b2"` | Имя нейронной модели (если `mode=neural`) |
| `goal` | `str` | `"balanced"` | Цель: `speed` \| `accuracy` \| `balanced` \| `low_memory` |
| `auto_select` | `bool` | `true` | Автоматический выбор метода |
| `method` | `str` | `null` | Ручной выбор метода (если `auto_select=false`) |
| `library` | `str` | `"opencv"` | Библиотека: `opencv` \| `sklearn` \| `torch` |
| `custom_params` | `str` | `"{}"` | JSON с пользовательскими параметрами |
| `gt_mask` | `UploadFile` | `null` | Ground Truth маска для расчёта метрик |

**Пример запроса (curl):**
```bash
curl -X POST http://localhost:8000/api/segment \
  -F "file=@image.jpg" \
  -F "mode=classical" \
  -F "auto_select=true" \
  -F "library=opencv" \
  -F "goal=accuracy"
```

**Пример запроса (Python requests):**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/segment",
    files={
        "file": open("image.jpg", "rb"),
        "gt_mask": open("mask.png", "rb"),  # опционально
    },
    data={
        "mode": "neural",
        "task": "semantic",
        "model": "segformer_b5",
        "goal": "accuracy",
    }
)
result = response.json()
```

**Пример ответа:**
```json
{
  "success": true,
  "method": "otsu_thresholding",
  "confidence": 0.92,
  "elapsed_ms": 45.3,
  "mask_b64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "overlay_b64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "chars": {
    "type": "document",
    "size": "1920×1080",
    "contrast": 0.85,
    "noise": 0.12,
    "channels": 3,
    "mean_intensity": 128.5,
    "edge_density": 0.08,
    "complexity": 0.65
  },
  "metrics": {
    "iou": 0.89,
    "dice": 0.94,
    "precision": 0.91,
    "recall": 0.88,
    "f1_score": 0.89
  },
  "recommendations": [
    {
      "method": "adaptive_thresholding",
      "score": 0.95,
      "estimated_time_ms": 45.0,
      "estimated_iou": 0.82,
      "best_for": ["document", "industrial"]
    }
  ],
  "analysis": {
    "histogram": [...],
    "hist_bins": [...],
    "edge_density": 0.08,
    "edges_b64": "data:image/png;base64,..."
  },
  "examples": {
    "medical": ["otsu", "sauvola", "phansalkar", "adaptive_thresholding"],
    "documents": ["otsu", "adaptive_thresholding", "bernsen", "niblack"],
    "nature": ["canny_edge", "sobel_edge", "watershed", "felzenszwalb"],
    "industrial": ["adaptive_thresholding", "bernsen", "gradient_magnitude_direction", "log_edge"]
  }
}
```

### 🎯 Рекомендации

#### `GET /recommendations/`
Получение топ-5 рекомендаций методов для изображения.

```bash
curl -X GET http://localhost:8000/recommendations/ \
  -F "file=@image.jpg"
```

---

## 🧠 Поддерживаемые нейронные модели

### Семантическая сегментация (`task=semantic`)

| Модель | Ключ | Описание |
|--------|------|----------|
| SegFormer-B0 | `segformer_b0` | Легковесная, быстрая |
| SegFormer-B2 | `segformer_b2` | **Баланс скорости/точности** |
| SegFormer-B5 | `segformer_b5` | Максимальная точность |
| Mask2Former (Swin-B) | `mask2former_swin_base` | State-of-the-art |
| Mask2Former (Swin-L) | `mask2former_swin_large` | Высокая точность |
| OneFormer | `oneformer_swin_large` | Универсальная архитектура |
| DPT-Large | `dpt_large` | Детализированные границы |
| U-Net (ResNet-34) | `unet_resnet34` | Классика для медицинских изображений |
| FPN (MiT-B5) | `fpn_mit_b5` | Многоуровневые признаки |
| DeepLabV3+ | `deeplab_resnet101` | Расширенное контекстное моделирование |
| MobileSAM | `mobile_sam` | Быстрая адаптивная сегментация |
| SAM2-Tiny | `sam2_tiny` | Segment Anything v2 |

### Instance & Panoptic сегментация

| Задача | Модели |
|--------|--------|
| `instance` | `mask2former_*_instance`, `yolov8*_seg`, `maskrcnn_*`, `mobile_sam`, `sam2_tiny` |
| `panoptic` | `mask2former_*_panoptic`, `oneformer_*_panoptic` |

---

## ⚙️ Конфигурация

### Переменные окружения

```bash
# Порт сервера
export PORT=8000

# Логирование
export LOG_LEVEL=INFO

# Ограничения
export MAX_UPLOAD_SIZE=50MB
export CACHE_MAX_MODELS=3

# Пути к моделям (опционально)
export MODEL_CACHE_DIR=./models_cache
```

### Параметры классических методов

Передавайте через `custom_params` в JSON-формате:

```json
{
  "threshold": 0.5,
  "block_size": 11,
  "C": 2,
  "window_size": 15,
  "k": 0.5
}
```

**Пример для adaptive_thresholding:**
```bash
curl -X POST http://localhost:8000/api/segment \
  -F "file=@doc.jpg" \
  -F "auto_select=false" \
  -F "method=adaptive_thresholding" \
  -F "library=opencv" \
  -F 'custom_params={"block_size":25,"C":5}'
```

---

## 📊 Цели оптимизации (`goal` параметр)

| Значение | Приоритет | Когда использовать |
|----------|-----------|-------------------|
| `speed` | ⚡ Скорость | Реальное время, потоковая обработка |
| `accuracy` | 🎯 Точность | Научные исследования, медицина |
| `balanced` | ⚖️ Баланс | **По умолчанию**, универсальный случай |
| `low_memory` | 💾 Память | Ограниченные устройства, мобильные |

---

## 🔄 Кэширование моделей

- **Алгоритм:** LRU (Least Recently Used)
- **Макс. размер:** 3 модели
- **Ключ кеша:** `json.dumps({**config, "_task": task}, sort_keys=True)`

```python
# При повторном запросе той же модели:
# 1. Проверяется наличие в _model_cache
# 2. Если есть — возвращается готовый экземпляр
# 3. Если нет — загружается и добавляется в кеш
# 4. При переполнении удаляется самая старая запись
```

---

## 🛠️ Разработка и отладка

### Логирование

```python
import logging
logger = logging.getLogger("autoseg")
logger.setLevel(logging.DEBUG)  # Для детальной отладки
```

### Тестирование эндпоинтов

```bash
# Тест health
http GET http://localhost:8000/api/health

# Тест методов
http GET "http://localhost:8000/api/methods?library=torch"

# Тест сегментации с выводом в файл
http --form POST http://localhost:8000/api/segment \
  file@image.jpg mode=classical auto_select=true > response.json
```

### Профилирование

```python
# Включить профилирование бенчмарков
export BENCHMARK_PROFILING=1

# Логирование времени выполнения
# Все запросы к /api/benchmark/* логируются автоматически
```

---

## 🚨 Обработка ошибок

| Код | Ситуация | Решение |
|-----|----------|---------|
| `422` | Невалидный `library` или `method` | Проверьте доступные значения через `/api/methods` |
| `422` | Невалидный JSON в `custom_params` | Убедитесь в корректности JSON-синтаксиса |
| `422` | Неизвестная нейронная конфигурация | Проверьте `task` + `model` в `NEURAL_CONFIGS` |
| `500` | Внутренняя ошибка | Проверьте логи сервера, наличие CUDA, память |

---

## 📦 Структура проекта

```
backend/
├── main.py                 # Этот файл — точка входа API
├── requirements.txt        # Зависимости Python
├── routers/
│   ├── benchmark.py        # Эндпоинты бенчмарков
│   ├── comparator.py       # Сравнение методов
│   └── validator.py        # Валидация реализаций
├── ../segmenters/
│   ├── AutoSegmenter.py    # Ядро интеллектуального выбора
│   ├── TorchSegmenter.py   # PyTorch реализации
│   ├── OpenCVSegmenter.py  # OpenCV реализации
│   ├── SklearnSegmenter.py # scikit-learn реализации
│   └── NeuralSegmenter.py  # Нейросетевые модели
└── ../metrics/
    └── SegmentationMetrics.py  # Расчёт метрик качества
```

---

## 🔗 Полезные ссылки

- 📖 [Swagger UI](http://localhost:8000/docs) — интерактивная документация
- 📚 [ReDoc](http://localhost:8000/redoc) — альтернативная документация
- 🧪 [Health Check](http://localhost:8000/api/health) — статус системы
- 🗄️ [Cache Info](http://localhost:8000/api/cache_info) — информация о кеше

---

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте ветку (`git checkout -b feature/amazing-feature`)
3. Закоммитьте изменения (`git commit -m 'Add amazing feature'`)
4. Запушьте ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

---

> ⚠️ **Важно:** Для работы с нейронными моделями требуется ~15–20 ГБ VRAM при одновременной загрузке нескольких моделей. Используйте параметр `CACHE_MAX_MODELS` для ограничения потребления памяти.