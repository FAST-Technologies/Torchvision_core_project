# 📦 AutoSegmenter — Документация по сборке и запуску

> **AutoSegmenter** — фреймворк для автоматизированной сегментации изображений с поддержкой классических методов (OpenCV, scikit-learn, PyTorch) и нейросетевых моделей (Transformers, SAM, SMP).

---

## 📋 Оглавление

1. [Требования](#-требования)
2. [Структура проекта](#-структура-проекта)
3. [Настройка окружения](#-настройка-окружения)
4. [Сценарии использования](#-сценарии-использования)
5. [Тестирование](#-тестирование)
6. [Диагностика и отладка](#-диагностика-и-отладка)
7. [API Reference](#-api-reference)
8. [Устранение неполадок](#-устранение-неполадок)

---

## 🔧 Требования

### Аппаратные

| Компонент | Минимум | Рекомендуемые |
|-----------|---------|--------------|
| **GPU** | NVIDIA с 8 ГБ VRAM | NVIDIA RTX 4000+ с 20+ ГБ |
| **CPU** | 4 ядра | 8+ ядер |
| **RAM** | 16 ГБ | 32+ ГБ |
| **Диск** | 50 ГБ свободного места | 100+ ГБ SSD |

### Программные

```bash
# Проверка версий
docker --version          # ≥ 24.0
docker compose version    # ≥ 2.20
nvidia-smi                # Driver ≥ 535, CUDA ≥ 12.1
python --version          # ≥ 3.10 (для локальной разработки)
```

### Зависимости проекта

Все зависимости указаны в `requirements.txt`. Основные:

```txt
torch>=2.6.0
torchvision>=0.21.0
opencv-python>=4.12.0
fastapi>=0.135.1
uvicorn>=0.46.0
segmentation_models_pytorch>=0.5.0
```

---

## 📁 Структура проекта

```
Torchvision_core_project/
├── 📄 docker-compose.yml          # Оркестрация сервисов
├── 📄 Dockerfile.backend          # Сборка бэкенда (Python/FastAPI)
├── 📄 Dockerfile.frontend         # Сборка фронтенда (Node/Vite)
├── 📄 requirements.txt            # Python-зависимости
├── 📄 .env.dev                    # Переменные для разработки
├── 📄 .env.prod                   # Переменные для продакшена
│
├── 📂 backend/                    # FastAPI приложение
│   ├── main.py                    # Точка входа, роуты
│   ├── services/                  # Бизнес-логика
│   └── models/                    # Pydantic-схемы
│
├── 📂 frontend/                   # React + TypeScript UI
│   ├── src/
│   ├── package.json
│   └── nginx.conf                 # Конфигурация прокси
│
├── 📂 segmenters/                 # Реализации методов сегментации
│   ├── OpenCVSegmenter.py
│   ├── SklearnSegmenter.py
│   ├── TorchSegmenter.py
│   ├── NewTorchSegmenter.py       # Оптимизированная версия (AMP, compile)
│   └── NeuralSegmenter.py         # Transformers, SAM, SMP
│
├── 📂 testing/                    # Тестирование и бенчмарки
│   ├── SegmentationTester.py
│   ├── SegmentationBenchmark.py
│   └── TorchImplementationValidator.py
│
├── 📂 metrics/                    # Метрики качества (IoU, Dice, F1)
├── 📂 utils/                      # Утилиты: warmup, экспорт, стратегии
├── 📂 configs/                    # YAML-конфигурации
└── 📂 data/                       # Данные (загружается при первом запуске)
```

---

## ⚙️ Настройка окружения

### Файлы переменных

#### 🔹 `.env.dev` — Разработка

```bash
# Hot-reload для FastAPI
API_RELOAD=--reload

# GPU конфигурация
GPU_COUNT=1
CUDA_VISIBLE_DEVICES=0

# API настройки
API_HOST=0.0.0.0
API_PORT=8000

# Пути (опционально)
DATA_DIR=./data
MODELS_DIR=./models
```

#### 🔹 `.env.prod` — Продакшен

```bash
# Отключить hot-reload
API_RELOAD=

# GPU конфигурация
GPU_COUNT=1
CUDA_VISIBLE_DEVICES=0

# API настройки
API_HOST=0.0.0.0
API_PORT=8000

# Оптимизации
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

> 💡 **Важно**: Файлы `.env.*` не отслеживаются в Git. Создайте их вручную на основе шаблонов.

---

## 🚀 Сценарии использования

### 🔧 Разработка (с hot-reload)

```bash
# 1. Сборка и запуск
docker compose --env-file .env.dev up --build

# 2. Бэкенд доступен на: http://localhost:8000
#    - API Docs: http://localhost:8000/docs
#    - Health:   http://localhost:8000/api/health

# 3. Фронтенд доступен на: http://localhost:3000

# 4. Логи в реальном времени
docker compose logs -f backend
docker compose logs -f frontend

# 5. Остановка
docker compose down
```

> ✅ При изменении файлов в `backend/` приложение автоматически перезагружается.

---

### 📦 Продакшен

```bash
# 1. Сборка и запуск в фоне
docker compose --env-file .env.prod up -d --build

# 2. Проверка статуса
docker compose ps
# ✅ autoseg-backend   Up (healthy)
# ✅ autoseg-frontend  Up

# 3. Проверка здоровья
curl http://localhost:8000/api/health | jq
# {
#   "status": "ok",
#   "cuda_available": true,
#   "device_name": "NVIDIA RTX 4000 SFF Ada Generation",
#   "vram_mb": 20475.0,
#   ...
# }

# 4. Обновление без простоя (zero-downtime)
docker compose up -d --no-deps --build backend

# 5. Просмотр логов
docker compose logs -f backend
docker compose logs -f frontend

# 6. Остановка
docker compose down
```

---

### 🧪 Тестирование

```bash
# 🔹 Запуск тестов внутри контейнера
docker compose run --rm backend pytest tests/ -v -m "not slow"

# 🔹 Smoke-тест импортов
docker compose run --rm backend python -c "from backend.main import app; print('✅ OK')"

# 🔹 Проверка типов во фронтенде
cd frontend && npx tsc --noEmit
# Ожидаемый вывод: ✓ 0 errors

# 🔹 Сборка фронтенда
cd frontend && npm run build

# 🔹 Тестовый запрос к API сегментации
curl -X POST http://localhost:8000/api/segment \
  -F "file=@test_images/test_gt_image.jpg" \
  -F "mode=classical" \
  -F "auto_select=true" \
  -F "library=opencv" \
  -o response.json -w "HTTP %{http_code}\n"
# Ожидаемый вывод: HTTP 200
```

---

### 🐳 Прямая сборка образов (без compose)

```bash
# 🔹 Сборка бэкенда
docker build -f Dockerfile.backend -t autoseg-backend:latest .

# 🔹 Проверка прав на uvicorn
docker run --rm autoseg-backend:latest ls -la /usr/local/bin/uvicorn
# ✅ -rwxr-xr-x 1 appuser appuser ... uvicorn

# 🔹 Проверка наличия curl для healthcheck
docker run --rm autoseg-backend:latest which curl
# ✅ /usr/bin/curl

# 🔹 Тестовый запуск бэкенда
docker run --rm -p 8000:8000 \
  -e API_HOST=0.0.0.0 \
  -e API_PORT=8000 \
  autoseg-backend:latest \
  uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 🔹 Сборка фронтенда
docker build -f Dockerfile.frontend -t autoseg-frontend:latest .
```

---

### 🔄 Пересборка с очисткой кэша

```bash
# 🔹 Через compose
docker compose --env-file .env.prod build --no-cache backend

# 🔹 Прямая сборка
docker build --no-cache -f Dockerfile.backend -t autoseg-backend:latest .

# 🔹 Проверка после сборки
docker run --rm autoseg-backend:latest which uvicorn
# ✅ /usr/local/bin/uvicorn

docker run --rm autoseg-backend:latest ls -la /usr/local/bin/uvicorn
# ✅ -rwxr-xr-x 1 appuser appuser ... uvicorn
```

---

### 🛠 Отладка внутри контейнера

```bash
# 🔹 Запуск shell как root для диагностики
docker run --rm -it --user root autoseg-backend:latest sh

# Внутри контейнера:
ls -la /usr/local/bin/uvicorn    # Проверка прав
which uvicorn                     # Проверка PATH
id                                # Текущий пользователь
python -c "import torch; print(torch.cuda.is_available())"  # Проверка CUDA

# Выход
exit
```

---

## 📊 Мониторинг и диагностика

### Проверка здоровья

```bash
# 🔹 Бэкенд
curl http://localhost:8000/api/health | jq

# 🔹 Фронтенд
curl -I http://localhost:3000
# HTTP/1.1 200 OK

# 🔹 Docker healthcheck
docker compose ps
# ✅ autoseg-backend  Up (healthy)
```

### Просмотр ресурсов

```bash
# 🔹 Использование GPU
docker compose exec backend nvidia-smi

# 🔹 Использование памяти
docker compose exec backend python -c "import torch; print(f'Allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB')"

# 🔹 Логи с фильтрацией
docker compose logs backend | grep -i error
docker compose logs backend | grep -i "loaded model"
```

### Проверка API

```bash
# 🔹 Список доступных методов
curl http://localhost:8000/api/methods?library=opencv | jq

# 🔹 Пример запроса сегментации
curl -X POST http://localhost:8000/api/segment \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_images/test_gt_image.jpg" \
  -F "mode=classical" \
  -F "library=opencv" \
  -F "method=otsu_thresholding" \
  -F "auto_select=true" \
  -o result.json

# 🔹 Проверка ответа
cat result.json | jq '.success, .method, .elapsed_ms'
```

---

## 📚 API Reference

### 🔹 Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "ok",
  "cuda_available": true,
  "device_name": "NVIDIA RTX 4000 SFF Ada Generation",
  "vram_mb": 20475.0,
  "vram_allocated_mb": 128.5,
  "vram_free_mb": 20346.5,
  "reserved_vram_mb": 256.0,
  "active_tasks": 0,
  "cached_models": 0,
  "cache_max": 3
}
```

### 🔹 Сегментация

```http
POST /api/segment
Content-Type: multipart/form-data
```

**Параметры:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|-------------|----------|
| `file` | File | ✅ | Изображение для сегментации |
| `mode` | String | ✅ | `classical` или `neural` |
| `library` | String | ⚠️ | `opencv`, `sklearn`, `torch`, `torch_v2` (для classical) |
| `method` | String | ⚠️ | Имя метода (если `auto_select=false`) |
| `auto_select` | Boolean | ❌ | Автоматический выбор метода (по умолчанию: `true`) |
| `gt_mask` | File | ❌ | Ground truth для вычисления метрик |
| `custom_params` | JSON | ❌ | Дополнительные параметры метода |

**Пример запроса:**
```bash
curl -X POST http://localhost:8000/api/segment \
  -F "file=@image.jpg" \
  -F "mode=classical" \
  -F "library=opencv" \
  -F "method=otsu_thresholding" \
  -F "auto_select=false" \
  -F "custom_params={\"threshold\": 0.5}"
```

**Response:**
```json
{
  "success": true,
  "method": "otsu_thresholding",
  "library": "opencv",
  "confidence": 0.92,
  "elapsed_ms": 45.3,
  "mask_b64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "overlay_b64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "chars": {
    "type": "document",
    "size": "1024x768",
    "contrast": 0.85,
    "noise": 0.12
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
      "estimated_time_ms": 52,
      "estimated_iou": 0.91,
      "best_for": ["documents", "low_contrast"]
    }
  ]
}
```

### 🔹 Доступные методы

```http
GET /api/methods?library=opencv
```

**Response:**
```json
{
  "methods": {
    "otsu_thresholding": {
      "name": "Otsu Thresholding",
      "avg_iou": 0.85,
      "avg_time_ms": 42,
      "memory_mb": 128,
      "robustness": 0.92,
      "description": "Автоматический порог по методу Оцу",
      "best_for": ["documents", "high_contrast"],
      "defaults": {},
      "schema": {}
    }
  }
}
```

---

## 🚨 Устранение неполадок

### ❌ Контейнер не запускается / Permission denied

```bash
# 🔹 Проверьте права на uvicorn
docker run --rm autoseg-backend:latest ls -la /usr/local/bin/uvicorn
# Ожидаемо: -rwxr-xr-x 1 appuser appuser

# 🔹 Если права неверные — пересоберите с очисткой кэша
docker build --no-cache -f Dockerfile.backend -t autoseg-backend:latest .

# 🔹 Проверьте, что curl установлен (для healthcheck)
docker run --rm autoseg-backend:latest which curl
# Ожидаемо: /usr/bin/curl
```

### ❌ CUDA не определяется

```bash
# 🔹 Проверьте, что nvidia-container-toolkit установлен
nvidia-smi  # Должен работать на хосте

# 🔹 Проверьте запуск с GPU
docker run --rm --gpus all autoseg-backend:latest python -c "import torch; print(torch.cuda.is_available())"
# Ожидаемо: True

# 🔹 В docker-compose.yml убедитесь, что есть:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: ${GPU_COUNT:-1}
#           capabilities: [gpu]
```

### ❌ Healthcheck показывает (unhealthy)

```bash
# 🔹 Проверьте логи
docker compose logs backend | tail -50

# 🔹 Проверьте, что curl доступен внутри контейнера
docker compose exec backend which curl

# 🔹 Проверьте, что приложение слушает правильный порт
docker compose exec backend netstat -tlnp | grep 8000

# 🔹 Если healthcheck использует localhost — замените на 127.0.0.1 в Dockerfile:
# CMD wget --quiet --tries=1 --spider http://127.0.0.1:80/index.html || exit 1
```

### ❌ Медленная работа / мало VRAM

```bash
# 🔹 Проверьте использование памяти
docker compose exec backend python -c "import torch; print(f'Used: {torch.cuda.memory_allocated()/1e6:.1f} MB')"

# 🔹 Очистите кэш CUDA
docker compose exec backend python -c "import torch; torch.cuda.empty_cache()"

# 🔹 Уменьшите batch_size в configs/main_config.yaml
# training:
#   batch_size: 2  # вместо 4 или 8

# 🔹 Используйте bf16 вместо fp32 (если поддерживается)
# segmenter = TorchSegmenter2(..., precision="bf16")
```

### ❌ Ошибки импорта / модулей

```bash
# 🔹 Проверьте, что все зависимости установлены
docker compose run --rm backend pip list | grep torch

# 🔹 Если не хватает — обновите requirements.txt и пересоберите
docker compose build --no-cache backend

# 🔹 Проверьте PYTHONPATH внутри контейнера
docker compose exec backend python -c "import sys; print('\n'.join(sys.path))"
# Должен содержать: /app
```

---

## 📈 Оптимизация производительности

### 🔹 Использование AMP (Automatic Mixed Precision)

```python
# В коде: указание точности при инициализации
segmenter = TorchSegmenter2(
    method="otsu_thresholding",
    device="cuda",
    precision="bf16",  # или "fp16" для старых GPU
    use_compile=True
)
```

### 🔹 torch.compile для ускорения

```yaml
# configs/main_config.yaml
torch_compile:
  enabled: true
  mode: "reduce-overhead"  # или "max-autotune"
  fullgraph: false  # true может не скомпилироваться для сложных методов
```

### 🔹 Кэширование результатов

```python
# Использование кэша при повторных вызовах
mask = segmenter.segment_with_cache(image, use_cache=True)
```

### 🔹 Warm-up перед бенчмарком

```python
# Прогрев модели для стабильных замеров
from utils.warmup import SegmentationWarmUp
warmup = SegmentationWarmUp(n_warmup_runs=5)
warmup.warmup_all_segmenters(segmenters_dict, image)
```

---

## 🔄 Обновление проекта

```bash
# 🔹 Обновление зависимостей
pip install --upgrade -r requirements.txt
# Затем пересоберите образ:
docker compose build --no-cache backend

# 🔹 Обновление моделей
# Модели загружаются при первом запросе из HuggingFace.
# Для кэширования локально:
# 1. Скачайте модель вручную
# 2. Укажите local_path в configs/neural_models.yaml

# 🔹 Обновление фронтенда
cd frontend
npm install
npm run build
cd ..
docker compose build frontend
```

---

## 📄 Лицензия

MIT License — см. файл `LICENSE` в корне проекта.

---

> 💡 **Совет**: Для быстрой проверки работоспособности после любой правки используйте:
> ```bash
> docker compose --env-file .env.dev up --build -d
> curl http://localhost:8000/api/health | jq .status
> # ✅ "ok"
> ```

*Документация актуальна на момент: Май 2026* 🚀