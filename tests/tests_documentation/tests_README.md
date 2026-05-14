# 🧪 Документация тестовых модулей

Ниже представлены модульные докстринги и структурированные README для каждого тестового файла в едином стиле проекта.

---

## 1. `tests/test_new_torch_segmenter.py`

## 🧪 Тесты: `TorchSegmenter2` (NewTorchSegmenter)

## 🎯 Назначение
Комплексная проверка оптимизированной PyTorch-реализации сегментаторов с поддержкой смешанной точности, JIT-компиляции и аппаратных ускорений.

## 🔑 Ключевые тестовые сценарии
| Категория | Что проверяется | Ожидаемый результат |
|-----------|----------------|---------------------|
| **Точность** | `fp32` vs `fp16` vs `bf16` | Корректный fallback на CPU, IoU отклонение < 0.01 |
| **Компиляция** | `torch.compile` (fullgraph/dynamic) | Ускорение ≥ 1.2×, отсутствие runtime-ошибок |
| **Кэширование** | Повторные вызовы с теми же параметрами | Hit rate > 90%, время 2-го вызова < 1 мс |
| **Профилирование** | `profile_with_transfer_detection` | Корректный подсчёт времени, отсутствие лишних трансферов |
| **Fallback** | Большие изображения (>2K×2K) на CPU | Автоматический переход на Numba/векторизованный путь |
| **Методы** | 50+ методов (пороговые, граничные, контуры) | Возврат `np.ndarray` формы `(H, W)`, `dtype=uint8` |

## 🛠️ Фикстуры и зависимости
```bash
# Обязательные
pytest>=7.0
torch>=2.0.0
numpy>=1.20

# Опционально (для CPU fallback)
numba>=0.56.0
```

## ▶️ Запуск
```bash
# Базовые тесты
pytest tests/test_new_torch_segmenter.py -v

# С профилированием
pytest tests/test_new_torch_segmenter.py -v --durations=5

# Только тесты точности
pytest tests/test_new_torch_segmenter.py -k "precision" -v
```

## 💡 Примечания
- На CPU тесты `bf16`/`fp16` должны автоматически fallback к `fp32`
- `torch.compile` требует тёплый запуск (первый прогон медленнее)
- Для стабильных бенчмарков используйте `torch.backends.cudnn.deterministic = True`

---

## 2. `tests/test_segmentation_metrics.py`

## 📊 Тесты: `SegmentationMetrics`

## 🎯 Назначение
Валидация математической корректности и устойчивости функций расчёта метрик качества сегментации.

## 📐 Тестируемые метрики
| Метрика | Диапазон | Проверяемые сценарии |
|---------|----------|----------------------|
| `iou` | `[0.0, 1.0]` | Идеальное совпадение, пустой предикт, частичное пересечение |
| `dice` | `[0.0, 1.0]` | Эквивалентность F1, обработка дисбаланса классов |
| `precision` / `recall` | `[0.0, 1.0]` | Ложные срабатывания, пропущенные объекты |
| `f1_score` | `[0.0, 1.0]` | Гармоническое среднее, границы значений |
| `mae` | `[0.0, 1.0]` | Абсолютная ошибка, нормализация входов |
| `hausdorff_distance` | `[0, ∞)` | Граничные расхождения, устойчивость к выбросам |

## ⚠️ Крайние случаи (Edge Cases)
```python
# 1. Полное совпадение
gt = np.ones((100, 100), dtype=np.uint8) * 255
assert metrics["iou"] == pytest.approx(1.0)

# 2. Пустой предикт
pred = np.zeros((100, 100), dtype=np.uint8)
assert metrics["iou"] == pytest.approx(0.0)

# 3. Вероятностная маска + threshold
prob = np.random.rand(100, 100).astype(np.float32)
metrics = SegmentationMetrics.calculate_all_metrics(prob, gt, threshold=0.5)
assert "iou" in metrics
```

## ▶️ Запуск
```bash
pytest tests/test_segmentation_metrics.py -v
pytest tests/test_segmentation_metrics.py -k "hausdorff" -v  # Только Hausdorff
pytest tests/test_segmentation_metrics.py --tb=short
```

## 📝 Примечания
- Все метрики возвращают `float`, совместимые с JSON-сериализацией
- `NaN`/`Inf` заменяются на `None` в API-ответах
- Многоклассовые метрики вычисляются как macro-average по умолчанию

---

## 3. `tests/test_sklearn_segmenter.py`

## 🐍 Тесты: `SklearnSegmenter`

## 🎯 Назначение
Проверка совместимости и корректности реализации классических методов сегментации через `scikit-image` и `scikit-learn`.

## 🔑 Ключевые сценарии
| Тест | Вход | Ожидание |
|------|------|----------|
| `test_initialization` | `SklearnSegmenter("adaptive", block_size=11)` | `seg.params["block_size"] == 11` |
| `test_segment_rgb` | `(256, 256, 3) uint8` | Маска `(256, 256) uint8` |
| `test_segment_grayscale_normalized` | `(128, 128) float32 [0,1]` | Корректная бинаризация, без переполнения |
| `test_sauvola_with_r_parameter` | `r=128` | Учёт динамического диапазона в формуле |
| `test_canny_with_quantiles` | `use_quantiles=False` | Абсолютные пороги вместо квантилей |
| `test_invalid_method_raises` | `"unknown_method"` | `ValueError` при создании или вызове |

## 📦 Фикстуры
- `rgb_image`: `(256, 256, 3)` случайные пиксели `[0, 255]`
- `gray_image`: `(256, 256)` случайные пиксели `[0, 255]`
- `segmenter`: Базовый `SklearnSegmenter("global_thresholding", threshold=0.5)`

## ▶️ Запуск
```bash
pytest tests/test_sklearn_segmenter.py -v
pytest tests/test_sklearn_segmenter.py -k "sauvola or canny" -v
```

## ⚙️ Зависимости
```text
scikit-image>=0.19.0
scikit-learn>=1.0.0
opencv-python>=4.5.0  # для конвертации RGB→Gray
```

---

## 4. `tests/test_torch_segmenter.py`

## 🔥 Тесты: `TorchSegmenter` (Базовый)

## 🎯 Назначение
Валидация базовой PyTorch-реализации без оптимизаций (eager mode, fp32, без compile).

## 🔑 Ключевые сценарии
| Тест | Описание | Статус |
|------|----------|--------|
| `test_import` | Импорт класса | ✅ |
| `test_initialization` | Проверка `method` и `params` | ✅ |
| `test_segment_rgb` | RGB → `(H, W) uint8` | ✅ |
| `test_segment_grayscale` | Grayscale → `(H, W) uint8` | ✅ |
| `test_unknown_method` | `ValueError` при неверном имени | ✅ |
| `test_methods` | Параметризация 5 методов | ✅ |
| `test_cuda_availability` | Проверка `device="cuda"` | ⚠️ (требует GPU) |
| `test_segment_returns_numpy` | Тип возврата из файла | ✅ |

## 🖥️ Маркеры pytest
- `@pytest.mark.gpu`: Запускается только при `torch.cuda.is_available()`
- Без маркеров: CPU-only тесты

## ▶️ Запуск
```bash
# Все тесты (CPU + GPU если доступен)
pytest tests/test_torch_segmenter.py -v

# Только CPU тесты
pytest tests/test_torch_segmenter.py -v -m "not gpu"

# Только GPU тесты
pytest tests/test_torch_segmenter.py -v -m "gpu"
```

## 💡 Примечания
- На CPU `device="cuda"` должен автоматически fallback к `"cpu"`
- Все методы возвращают `np.ndarray`, а не `torch.Tensor`
- Для стабильности тестов используется фиксированный `seed` в `conftest.py`

---

## 5. `tests/integration/test_validation_pipeline.py`

## 🔗 Интеграционные тесты: Validation & Benchmark Pipeline

## 🎯 Назначение
Сквозная проверка взаимодействия модулей валидации, бенчмаркинга и расчёта метрик.

## 🔑 Тестируемые пайплайны
| Тест | Что проверяется | Допуск |
|------|----------------|--------|
| `test_torch_vs_opencv_validation` | Кросс-библиотечная валидация | ≥0 результатов, без исключений |
| `test_metrics_calculation_consistency` | Детерминизм метрик | `metrics1 == metrics2` |
| `test_benchmark_reproducibility` | Стабильность замеров времени | `|t1 - t2| / max(t1, t2) < 0.5` |

## 📁 Структура тестов
```python
@pytest.mark.integration
class TestValidationPipeline:
    def test_torch_vs_opencv_validation(...)
    def test_metrics_calculation_consistency(...)
    def test_benchmark_reproducibility(...)
```

## ⚙️ Конфигурация
- Использует `tempfile.TemporaryDirectory()` для изоляции
- Требует `pytest>=7.0`, `pandas>=1.3`, `numpy>=1.20`
- Запускается отдельно от unit-тестов

## ▶️ Запуск
```bash
# Только интеграционные тесты
pytest tests/integration/test_validation_pipeline.py -v -m integration

# С подробным выводом
pytest tests/integration/test_validation_pipeline.py -v -s

# Без кэша pytest
pytest tests/integration/test_validation_pipeline.py -v --cache-clear
```

---

## 6. `conftest.py` — Глобальные фикстуры и среда тестирования

## 🎯 Назначение
Централизованное управление общими данными, маркерами pytest и логикой пропуска тестов в зависимости от доступности GPU.

### 🔑 Ключевые возможности
| Фикстура | Область | Описание |
|----------|---------|----------|
| `test_data_dir` | `session` | Путь к директории `tests/test_data` |
| `rgb_image` | `function` | Синтетическое RGB-изображение `256×256×3` |
| `gray_image` | `function` | Синтетическое grayscale-изображение `256×256` |
| `textured_gray_image` | `function` | Изображение с двумя текстурными областями + градиент |
| `binary_mask` | `function` | Бинарная маска с квадратом в центре `(256, 256)` |
| `temp_image_file` | `function` | Временный JPEG-файл в `tmp_path` |
| `temp_mask_file` | `function` | Временный PNG-файл маски в `tmp_path` |
| `small_image` | `function` | `64×64×3` для быстрых юнит-тестов |
| `large_image` | `function` | `512×512×3` для стресс-тестов производительности |

### ⚙️ Автоматический пропуск GPU-тестов
```python
@pytest.fixture(autouse=True)
def skip_if_no_gpu(request) -> None:
    if request.node.get_closest_marker("gpu"):
        has_hardware = torch.cuda.is_available()
        force_run = os.getenv("FORCE_GPU_TEST", "false").lower() == "true"
        if not has_hardware and not force_run:
            pytest.skip("CUDA hardware not available")
```
**Запуск:** `pytest tests/ -m gpu` (пропустит тесты, если нет GPU)  
**Принудительный запуск:** `FORCE_GPU_TEST=1 pytest tests/ -m gpu`

---

## 7. `test_base_segmenter.py` — Валидация контракта базового класса

## 🎯 Назначение
Проверка корректности реализации абстрактного интерфейса `BaseSegmenter` и методов предобработки/визуализации.

### 🔍 Что тестируется
| Тест | Проверяемое поведение |
|------|----------------------|
| `test_import` | Корректный импорт `BaseSegmenter` из модуля |
| `test_preprocess_image_from_path` | Конвертация пути `str` → `torch.Tensor` на устройстве |
| `test_preprocess_image_from_pil` | Конвертация `PIL.Image` → тензор с нормализацией |
| `test_preprocess_image_from_numpy` | Обработка `np.ndarray` (включая 2D→3D конвертацию) |
| `test_segment_with_mask_base` | Возврат кортежа `(overlay, mask)` с проверкой форм и типов |

### 🧩 Паттерн мокирования
В тестах используется `DummySegmenter`, наследующий `BaseSegmenter`:
```python
class DummySegmenter(BaseSegmenter):
    def segment(self, image, **kwargs) -> np.ndarray:
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)
```
Это позволяет тестировать **инфраструктуру** (препроцессинг, возврат типов) без привязки к конкретному алгоритму.

---

## 8. `test_datasets.py` — Пайплайн загрузки ADE20K

## 🎯 Назначение
Проверка корректности инициализации, индексации, аугментаций и метаданных датасета `ADE20KDataset`.

### 📦 Структура тестов
| Тест | Ключевые проверки |
|------|------------------|
| `test_import` | Доступность класса `ADE20KDataset` |
| `test_dataset_initialization` | `len(dataset) == 3`, `image_size == (128, 128)` |
| `test_dataset_getitem` | Ключи `{"image", "mask", "image_id"}`, формы `(3,128,128)`/`(128,128)`, типы `float32`/`int64` |
| `test_dataset_with_augmentation` | Согласованность форм при `augment=True` |
| `test_subset_fraction` | Сокращение выборки до `≤ 50%` при `subset_fraction=0.5` |
| `test_ignore_index_in_mask` | Отсутствуют значения за пределами `[0, 149]` при `ignore_index=255` |
| `test_validation_split` | Корректное чтение из `validation/` директории |

### 🛠️ Динамическая генерация данных
Фикстура `temp_dataset_dir` создаёт изолированную структуру ADE20K в `tmp_path`:
```
tmp_path/
└── ADEChallengeData2016/
    ├── images/training/  (3 случайных JPEG)
    └── annotations/training/ (3 случайных PNG, значения 0–149)
```
Это гарантирует **детерминизм** и отсутствие зависимости от внешних файлов.

---

## 9. `test_opencv_segmenter.py` — Алгоритмическая валидация OpenCV

## 🎯 Назначение
Проверка корректности работы классических методов, обработки граничных случаев и сравнения алгоритмов.

### 🧪 Матрица тестов
| Тест | Стратегия валидации |
|------|---------------------|
| `test_import` / `test_initialization` | Доступность класса, сохранение `method` и `params` |
| `test_segment_rgb` / `test_segment_grayscale` | Форма выхода `(H, W)`, тип `uint8`, значения `{0, 255}` |
| `@pytest.mark.parametrize` | 7 методов: `global`, `otsu`, `adaptive`, `niblack`, `sauvola`, `sobel`, `canny` |
| `test_canny_edge_output` | Плотность границ `< 50%` (`np.mean(mask > 0) < 0.5`) |
| `test_sauvola_vs_niblack` | Сравнение на синтетической текстуре: оба метода возвращают `uint8`, но дают разные результаты на низкоконтрастных участках |
| `test_invalid_method_raises` | Выброс `ValueError("Неизвестный метод")` |
| `test_segment_with_mask` | Возврат `(overlay, mask)` с корректными размерами и типами |

### 💡 Пример параметризации
```python
@pytest.mark.parametrize(
    "method,params",
    [
        ("global_thresholding", {"threshold": 0.5}),
        ("canny_edge", {"low": 0.1, "high": 0.3}),
        # ... ещё 5 методов
    ],
)
def test_methods_basic(self, rgb_image, method, params):
    seg = OpenCVSegmenter(method, **params)
    mask = seg.segment(rgb_image)
    assert mask.shape == rgb_image.shape[:2]
    assert mask.dtype == np.uint8
```

---

### 🚀 Как запускать тесты

| Команда | Что запускает |
|---------|---------------|
| `pytest tests/test_opencv_segmenter.py -v` | Тесты OpenCV-сегментера |
| `pytest tests/test_datasets.py -v` | Тесты пайплайна данных |
| `pytest tests/test_base_segmenter.py -v` | Тесты базового контракта |
| `pytest tests/ -v --tb=short` | Все модульные тесты проекта |
| `pytest tests/ -m "not slow"` | Исключение долгих интеграционных тестов |
| `pytest tests/ --cov=segmenters --cov-report=html` | Генерация отчёта покрытия кода |

> 💡 **Совет:** Для CI/CD добавьте шаг `pytest tests/ -v -x --tb=short -m "not gpu"` чтобы гарантировать быстрый проход пайплайна без зависимости от CUDA-окружения.

## ⚠️ Примечания
- Тесты могут быть медленными (загрузка моделей, бенчмарки)
- Допуск 50% на время обусловлен фоновыми процессами ОС
- Для CI/CD рекомендуется запускать в изолированном контейнере
- Результаты записываются во временные папки и автоматически удаляются

---

## 📋 Сводная таблица запуска всех тестов

| Команда | Что запускает | Время |
|---------|---------------|-------|
| `pytest tests/ -v --tb=short` | Все модули | ~2-5 мин |
| `pytest tests/ -v -m "not integration and not gpu"` | CPU unit-тесты | ~1-2 мин |
| `pytest tests/ -v -k "metric"` | Только метрики | ~10-20 сек |
| `pytest tests/ -v -k "torch"` | PyTorch сегментеры | ~30-60 сек |
| `pytest tests/integration/ -v -m integration` | Интеграция | ~3-10 мин |

> 💡 **Совет**: Для локальной разработки используйте `pytest --maxfail=3` для быстрого обнаружения первых ошибок. Для CI/CD добавьте `--junitxml=report.xml` и `--cov=segmenters --cov=metrics --cov-report=xml`.