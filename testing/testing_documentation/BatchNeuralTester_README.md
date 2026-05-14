# 🧪 BatchNeuralTester — Пакетное тестирование нейросетевых моделей сегментации

## 📖 Описание
Модуль `BatchNeuralTester.py` предоставляет **оркестратор для масштабного анализа** влияния аугментаций данных на качество нейросетевых моделей семантической сегментации на датасете **ADE20K** (150 классов).

> ⚠️ **Важно:** Данный модуль работает с **нейросетевыми моделями** (через `NeuralSegmenter`). Для классических методов используйте `BatchClassicTester` или `BatchClassicTester2`.

## ✨ Ключевые возможности
### 🔄 Полный цикл тестирования и анализа

| Этап | Функционал | Инструменты |
|------|-----------|------------|
| **Поиск моделей** | Авто-обнаружение чекпоинтов по шаблону `{model}_{aug}_*.pth` | `glob`, `os.path.getctime` |
| **Загрузка данных** | Локальный ADE20K или загрузка через HuggingFace Hub | `hf_hub_download`, `list_repo_files` |
| **Инференс** | Пакетное предсказание с контролем памяти (OOM-handling) | `torch.no_grad()`, `torch.cuda.empty_cache()` |
| **Метрики** | mIoU, Binary IoU, Dice, F1, Precision, Recall, Boundary F1 | `SegmentationMetrics`, `sklearn.metrics` |
| **Статистика** | ANOVA, Tukey HSD, приросты относительно baseline | `scipy.stats`, `statsmodels` |
| **Экспорт** | CSV, JSON, Markdown-отчёт, PNG-графики, оверлеи | `pandas`, `matplotlib`, `seaborn` |

### 🎚️ Гибкая конфигурация через CLI
```bash
# Базовый запуск на 50 изображениях
python BatchNeuralTester.py --dataset ./data/ADE20K --subset 50

# С кэшированием и возобновлением прерванного запуска
python BatchNeuralTester.py --cache --resume --output ./results

# Профилирование инференса (torch.profiler)
python BatchNeuralTester.py --profile --profile-output ./profiling

# Экспорт моделей в ONNX + TensorRT
python BatchNeuralTester.py --export-onnx --export-trt --trt-precision fp16

# Многоклассовые метрики + Boundary F1
python BatchNeuralTester.py --compute-boundary-f1 --per-class-metrics

# Интеграция с трекерами экспериментов
python BatchNeuralTester.py --use-mlflow
python BatchNeuralTester.py --use-wandb  # требует 'wandb login'
```

### 🧠 Умное кэширование предсказаний
- **LRU-политика**: автоматическое вытеснение старых файлов при превышении лимита.
- **Ключи на основе mtime**: повторное использование кэша только если чекпоинт не изменился.
- **Дисковое хранилище**: `.pkl` файлы в `cache_dir` с контролем размера (`max_size_gb`).

```python
# Пример использования кэша
cache = PredictionCache(cache_dir="./cache", max_size_gb=10.0)
key = cache._get_key(model_path, image_path, config_hash)
cached_pred = cache.get(key)  # None если кэш отсутствует
if cached_pred is None:
    pred = model.inference(image)
    cache.set(key, pred)  # Сохранение с контролем лимита
```

### 📊 Расширенные метрики для многоклассовой сегментации
| Метрика | Описание | Формула |
|---------|----------|---------|
| **mIoU** | Mean Intersection over Union (150 классов) | `mean(IoU_c for c in classes)` |
| **Binary IoU** | IoU для бинарной задачи (объект vs фон) | `TP / (TP + FP + FN)` |
| **Dice** | Dice / Sørensen coefficient | `2·TP / (2·TP + FP + FN)` |
| **Boundary F1** | Точность границ через dilation ⊕ erosion | `2·TP_b / (2·TP_b + FP_b + FN_b)` |
| **Per-class Precision/Recall** | Детальная диагностика по классам | `TP/(TP+FP)`, `TP/(TP+FN)` |

### 🎨 Визуализация и отчётность
- **Оверлеи с классами**: цветные маски с легендой (опция `--class-aware-overlays`).
- **Сравнительные сетки**: 3 колонки (none/basic/medium) × N строк (модели).
- **Графики**: bar-чарты mIoU, heatmap приростов, boxplot распределения, swarmplot.
- **Markdown-отчёт**: автоматически генерируемый `report.md` со сводными таблицами и статистикой.

## 🚀 Быстрый старт
### Базовое тестирование: влияние аугментаций на U-Net
```bash
# Подготовка: чекпоинты должны лежать в ./models с именами вида:
# unet_smp_none_*.pth, unet_smp_basic_*.pth, unet_smp_medium_*.pth

python BatchNeuralTester.py \
    --dataset ./data/ADE20K \
    --subset 50 \
    --output ./results/unet_aug_test \
    --verbose
```

**Ожидаемый вывод:**
```
🔍 Поиск датасета в: /path/to/ADE20K
✓ unet_smp_none: unet_smp_none_20240101_120000.pth
✓ unet_smp_basic: unet_smp_basic_20240101_120000.pth
✓ unet_smp_medium: unet_smp_medium_20240101_120000.pth
Загружено 50 пар изображение/маска

🎯 Точность инференса: fp32 (запрошено: fp32)
Тестирование unet_smp_none на 50 изображениях...
100%|██████████| 50/50 [02:15<00:00,  2.70s/it]

================================================================================
СВОДНАЯ СТАТИСТИКА
================================================================================

📊 Средний mIoU по уровням аугментаций:
   None:   0.4523
   Basic:  0.4687 (прирост: +3.62%)
   Medium: 0.4812 (прирост: +6.39%)

🏆 Лучшая комбинация:
   Модель: unet_smp
   Аугментации: medium
   mIoU: 0.4812
================================================================================

💾 Результаты сохранены:
   • csv: ./results/unet_aug_test/detailed_results.csv
   • aggregated_csv: ./results/unet_aug_test/aggregated_metrics.csv
   • stats_json: ./results/unet_aug_test/statistical_analysis.json
   • report_md: ./results/unet_aug_test/report.md
   • overlays_dir: ./results/unet_aug_test/overlays
   • plot_miou_bar: ./results/unet_aug_test/plots/miou_comparison.png
   • ...
```

### Сравнение нескольких архитектур
```bash
# Чекпоинты для разных моделей:
# unet_smp_*.pth, fpn_smp_*.pth, deeplab_tv_*.pth, segnet_*.pth

python BatchNeuralTester.py \
    --dataset ./data/ADE20K \
    --subset 100 \
    --output ./results/arch_comparison \
    --models ./checkpoints \
    --compute-boundary-f1 \
    --per-class-metrics
```

### Экспорт моделей для продакшена
```bash
# Экспорт в ONNX + TensorRT после тестирования
python BatchNeuralTester.py \
    --dataset ./data/ADE20K \
    --subset 10 \  # Быстрый тест перед экспортом
    --export-onnx \
    --export-trt \
    --trt-precision fp16 \
    --opset 18 \
    --dynamic-shapes
```

**Результат:**
```
📦 Экспорт моделей в: ./results/exports
✅ ONNX: unet_smp_none.onnx (45.2 MB)
✅ TensorRT: unet_smp_none.fp16.trt (22.1 MB)
```

### Профилирование производительности
```bash
# Детальный анализ времени и памяти инференса
python BatchNeuralTester.py \
    --dataset ./data/ADE20K \
    --subset 1 \  # Достаточно одного изображения для профилирования
    --profile \
    --profile-output ./profiling \
    --profile-warmup 10 \
    --profile-runs 50
```

**Файлы в `./profiling/`:**
- `trace_unet_smp_none.json` — Chrome Trace для анализа в `chrome://tracing`
- `stacks_unet_smp_none.txt` — стек вызовов по времени CUDA
- Консольный вывод: среднее время, память, FLOPs, топ-10 операций

## ⚙️ Конфигурация
### Основные параметры CLI
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--dataset` | `str` | `"./data/ADE20K"` | Путь к датасету (локально) или ID HF-репозитория |
| `--subset` | `int` | `50` | Количество изображений для теста (0 = весь датасет) |
| `--output` | `str` | `"./results/augmentation_analysis"` | Директория для артефактов |
| `--models` | `str` | `"./models"` | Директория с чекпоинтами моделей |
| `--seed` | `int` | `42` | Random seed для воспроизводимости |
| `--verbose` | `bool` | `True` | Подробный вывод логов и статусов |

### Точность и устройство
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--precision` | `str` | `"fp32"` | Точность вычислений: `"fp32"`, `"fp16"`, `"bf16"` |
| `--device` | `str` | `"cuda"` | Устройство: `"cuda"` или `"cpu"` |

> ⚠️ **Важно**: При `device="cpu"` точности `fp16`/`bf16` автоматически откатываются к `fp32`.

### Кэширование и возобновление
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--cache` | `bool` | `False` | Включить дисковое кэширование предсказаний |
| `--cache-dir` | `str` | `"./cache/predictions"` | Директория для `.pkl` файлов кэша |
| `--cache-max-gb` | `float` | `10.0` | Максимальный размер кэша в гигабайтах |
| `--clear-cache` | `bool` | `False` | Очистить кэш перед запуском |
| `--resume` | `bool` | `False` | Пропускать уже обработанные комбинации (модель+изображение) |

### Экспорт и профилирование
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--export-onnx` | `bool` | `False` | Экспортировать модели в ONNX после теста |
| `--export-trt` | `bool` | `False` | Компилировать в TensorRT (требует `--export-onnx`) |
| `--trt-precision` | `str` | `"fp16"` | Точность для TensorRT: `"fp32"` или `"fp16"` |
| `--opset` | `int` | `17` | Версия ONNX opset |
| `--dynamic-shapes` | `bool` | `False` | Использовать динамические размеры при экспорте |
| `--profile` | `bool` | `False` | Включить `torch.profiler` для первой модели |
| `--profile-output` | `str` | `"./profiling"` | Директория для результатов профилирования |
| `--profile-warmup` | `int` | `10` | Итераций прогрева перед профилированием |
| `--profile-runs` | `int` | `50` | Измеряемых прогонов для профилирования |

### Расширенные метрики и визуализация
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--compute-boundary-f1` | `bool` | `False` | Вычислять Boundary F1 score (медленно, для детального анализа) |
| `--per-class-metrics` | `bool` | `False` | Сохранять per-class Precision/Recall/IoU в отчёт |
| `--class-aware-overlays` | `bool` | `False` | Рисовать цветные легенды классов на оверлеях |
| `--overlay-alpha` | `float` | `0.5` | Прозрачность наложения маски [0.0, 1.0] |
| `--save-viz` | `bool` | `False` | Сохранять визуализации сегментации (оверлеи) |

### Трекеры экспериментов
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--use-mlflow` | `bool` | `False` | Логировать метрики в MLflow |
| `--use-wandb` | `bool` | `False` | Логировать в Weights & Biases (требует `wandb login`) |

### Dataclass `TestConfig` (внутреннее представление)
```python
@dataclass
class TestConfig:
    dataset_path: PathLike
    output_dir: PathLike = "./results/augmentation_analysis"
    subset_size: Optional[int] = 50
    random_seed: int = 42
    batch_size: int = 1  # Из-за разного размера входов
    device: str = "cuda"
    num_classes: int = 150  # ADE20K
    ignore_index: int = 255
    metrics: List[str] = field(default_factory=lambda: DEFAULT_METRICS)
    # ... остальные поля (см. таблицу выше)
```

## 📚 Справочник классов и функций
### 🔹 Основные классы
| Класс | Описание | Ключевые методы |
|-------|----------|----------------|
| `PredictionCache` | LRU-кэш предсказаний на диске | `_get_key()`, `get()`, `set()`, `clear()` |
| `TestConfig` | Конфигурация тестирования (dataclass) | `__post_init__()`, валидация путей |
| `ModelCheckpoint` | Метаданные найденного чекпоинта | `display_name` (property) |
| `TestResult` | Результат тестирования одного изображения | `to_dict()` для создания DataFrame |
| `BatchNeuralTester` | Оркестратор полного цикла | `run()`, `aggregate_metrics()`, `export_results()` |

### 🔹 Ключевые функции
| Функция | Описание | Возвращает |
|---------|----------|-----------|
| `extract_model_aug_from_key()` | Парсинг ключа `"{model}_{aug}_{image}"` → `(model, aug)` | `Tuple[str, str]` |
| `save_augmentation_comparison_grid()` | Создание сетки сравнения оверлеев (модели × аугментации) | `None` (сохраняет PNG) |
| `_check_precision_support()` | Проверка совместимости точности с устройством | `bool` |
| `_calculate_multiclass_iou()` | Расчёт mIoU для 150 классов | `Tuple[float, Dict[int, float]]` |
| `_calculate_boundary_f1()` | Точность границ через морфологические операции | `float` |

### 🔹 Вспомогательные утилиты
| Функция | Описание |
|---------|----------|
| `ensure_pil_compatible()` | Конвертация `np.ndarray`/`Image` → RGB `PIL.Image` |
| `_resize_mask()` | Ресайз маски предсказания под размер GT (nearest-neighbor) |
| `_create_simple_overlay()` | Базовый оверлей через `Image.blend()` |
| `_create_class_aware_overlay()` | Оверлей с цветными классами и легендой |

## 🔄 Конвейер тестирования: поиск → загрузка → инференс → анализ → экспорт
### Логика `BatchNeuralTester.run()`
```python
def run(self) -> pd.DataFrame:
    # 1. Поиск чекпоинтов по шаблону
    checkpoints = self._find_checkpoints()  # Dict[key, ModelCheckpoint]
    
    # 2. Загрузка пар (изображение, маска) из ADE20K
    image_pairs = self._load_ade20k_images()  # List[Tuple[Path, Path]]
    
    # 3. Для каждой модели: инференс + метрики
    for checkpoint in checkpoints.values():
        results = self._test_single_model(
            checkpoint, image_pairs,
            precision=self.config.precision,
            cache=self.cache
        )
        all_results.extend(results)
    
    # 4. Агрегация в DataFrame
    df = pd.DataFrame([r.to_dict() for r in all_results])
    df["model"] = df["model_key"].apply(lambda k: extract_model_aug_from_key(k)[0])
    df["augmentation"] = df["model_key"].apply(lambda k: extract_model_aug_from_key(k)[1])
    
    # 5. Статистический анализ и экспорт
    aggregated = self.aggregate_metrics(df)
    stats = self.statistical_analysis(df)
    exported = self.export_results(df, aggregated, stats)
    
    return df
```

### Инференс с контролем памяти и точности
```python
def _test_single_model(self, checkpoint, image_pairs, precision, cache, config_hash):
    # Проверка и приведение точности
    dtype = self._resolve_torch_dtype(precision)
    if not _check_precision_support(None, dtype, self.config.device):
        dtype = torch.float32  # fallback
    
    # Загрузка модели
    segmenter = NeuralSegmenter(
        model_type=checkpoint.model_type,
        checkpoint_path=str(checkpoint.path),
        device=self.config.device,
        num_classes=self.config.num_classes
    )
    segmenter.model.eval()
    if dtype != torch.float32:
        segmenter.model = segmenter.model.to(dtype)
    
    # Инференс с autocast (если нужно)
    for img_path, mask_path in image_pairs:
        with safe_inference_context(checkpoint.key, img_path.name):
            amp_ctx = torch.amp.autocast(self.config.device, dtype=dtype) if autocast_enabled else nullcontext()
            with amp_ctx, torch.no_grad():
                pred_mask = segmenter.predict_segmentation_map(test_image, verbose=False)
            
            # Расчёт метрик
            m_iou, _ = self._calculate_multiclass_iou(pred_mask, gt_mask, ignore_index)
            binary_metrics = self._calculate_binary_metrics(pred_mask, gt_mask)
            
            # Сохранение результата
            results.append(TestResult(...))
    
    return results
```

### Агрегация метрик с обработкой групп из 1 элемента
```python
def aggregate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
    # Фильтрация только числовых колонок
    metric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    # Группировка и агрегация
    grouped = df.groupby(["model", "augmentation", "precision"])[metric_cols]
    aggregated = grouped.agg(["mean", "std", "min", "max"])
    
    # 🔧 Заполнение NaN для std в группах с 1 записью
    std_cols = [c for c in aggregated.columns if c.endswith("_std")]
    for col in std_cols:
        counts = df.groupby([...]).size()
        for (model, aug, prec), count in counts.items():
            if count <= 1:
                aggregated.loc[(model, aug, prec), col] = 0.0
    
    return aggregated.reset_index()
```

## 📊 Статистический анализ
### ANOVA и Tukey HSD для сравнения аугментаций
```python
def statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
    analysis = {}
    
    # 1. Сводная статистика по аугментациям
    for aug in ["none", "basic", "medium"]:
        aug_data = df[df["augmentation"] == aug]
        analysis["summary_by_augmentation"][aug] = {
            "mean_m_iou": float(aug_data["m_iou"].mean()),
            "std_m_iou": float(aug_data["m_iou"].std()),
            # ... остальные статистики
        }
    
    # 2. ANOVA для каждой модели
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        groups = [group["m_iou"].dropna().values for _, group in model_data.groupby("augmentation")]
        if len(groups) >= 2:
            f_stat, p_value = stats.f_oneway(*groups)
            analysis["anova_by_model"][model] = {
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05)
            }
    
    # 3. Пост-хок тесты (Tukey HSD) при значимом ANOVA
    for model, result in analysis["anova_by_model"].items():
        if result.get("significant"):
            tukey = pairwise_tukeyhsd(...)
            analysis["posthoc_tukey"][model] = {
                "significant_pairs": [...]
            }
    
    return analysis
```

### Интерпретация результатов статистики
| Метрика | Значение | Интерпретация |
|---------|----------|---------------|
| **p-value < 0.05** (ANOVA) | ✅ Значимо | Аугментации статистически влияют на качество |
| **p-value ≥ 0.05** (ANOVA) | ❌ Не значимо | Различия могут быть случайными |
| **Tukey HSD: "basic vs none: p=0.012"** | ✅ | Basic значимо лучше none |
| **Прирост +6.39%** | 📈 | Умеренное улучшение, стоит использовать |
| **Boundary F1 < 0.3** | ⚠️ | Модель плохо детектирует границы |

## ⚡ Производительность и оптимизации
### Управление памятью при пакетном инференсе
```python
# Автоматическое уменьшение batch_size при OOM
def _batch_predict_with_memory_control(self, segmenter, images, batch_size=4):
    current_batch_size = batch_size
    for i in range(0, len(images), current_batch_size):
        while True:
            try:
                # Инференс батча
                predictions.extend(batch_preds)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                if current_batch_size == 1:
                    break  # Даже при batch=1 OOM
                current_batch_size = max(1, current_batch_size // 2)
                logger.warning(f"⚠️ OOM, уменьшаем batch_size до {current_batch_size}")
```

### Кэширование для ускорения повторных запусков
```bash
# Первый запуск: предсказания вычисляются и сохраняются в кэш
$ python BatchNeuralTester.py --cache --subset 50
# Время: ~15 минут

# Второй запуск с --resume: кэшированные предсказания загружаются
$ python BatchNeuralTester.py --cache --resume --subset 50
# Время: ~2 минуты (только метрики и отчёты)
```

### Профилирование: что смотреть в результатах
```json
// trace_unet_smp_none.json (Chrome Trace)
{
  "traceEvents": [
    {"name": "model_inference", "dur": 45230, "args": {"device": "cuda"}},
    {"name": "conv2d", "dur": 12340, "args": {"flops": 1.2e9}},
    ...
  ]
}

// Консольный вывод профилирования
📈 Среднее время: 45.23 ms
🧠 Память: 1245.6 MB
🔢 FLOPs: 12.4G
📋 Топ-операции:
  1. conv2d: 12.3 ms (27.2%)
  2. batch_norm: 8.1 ms (17.9%)
  3. relu: 4.5 ms (10.0%)
  ...
```

## 🛠️ Обработка ошибок и устойчивость
### Контекст-менеджер для безопасного инференса
```python
@contextmanager
def safe_inference_context(model_name: str, image_name: str):
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"OOM при {model_name}/{image_name}: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        raise
    except Exception as e:
        logger.error(f"Ошибка {model_name}/{image_name}: {type(e).__name__}: {e}", exc_info=True)
        raise
```

### Fallback-механизмы при экспорте
```python
# Экспорт ONNX: пробует export_params=True → False → смена opset
try:
    torch.onnx.export(model, ..., export_params=True)
except Exception:
    try:
        torch.onnx.export(model, ..., export_params=False)
    except Exception:
        if opset_version != 18:
            return self._export_model_to_onnx_trt(..., opset_version=18)

# TensorRT: fallback на fp32 если fp16 не поддерживается
if trt_precision == "fp16" and not _check_precision_support(...):
    logger.warning("⚠️ fp16 не поддерживается, fallback на fp32")
    return self._export_model_to_onnx_trt(..., trt_precision="fp32")
```

### Валидация входных данных
```python
# Проверка существования датасета
if not images_dir.exists() or not masks_dir.exists():
    logger.error("Не удалось найти директорию с изображениями/масками!")
    return []

# Проверка совместимости точности и устройства
if dtype == torch.bfloat16 and device == "cuda":
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 8:  # Ampere+
        logger.warning("⚠️ bf16 требует GPU Ampere+, fallback на fp32")
        dtype = torch.float32
```

### Рекомендации по отладке
1. **Запустите с `--verbose --subset 1`** для быстрой проверки конвейера:
   ```bash
   python BatchNeuralTester.py --verbose --subset 1 --device cpu
   ```

2. **Проверьте структуру датасета**:
   ```bash
   tree ./data/ADE20K -L 3
   # Ожидаемо:
   # ├── images/
   # │   └── validation/
   # │       ├── ADE_val_00000001.jpg
   # │       └── ...
   # └── annotations/
   #     └── validation/
   #         ├── ADE_val_00000001.png
   #         └── ...
   ```

3. **Убедитесь в наличии чекпоинтов**:
   ```bash
   ls ./models/*_none_*.pth ./models/*_basic_*.pth ./models/*_medium_*.pth
   ```

4. **Проверьте логи при падении**:
   ```bash
   # Включите debug-логирование
   export LOG_LEVEL=DEBUG
   python BatchNeuralTester.py --verbose 2>&1 | tee debug.log
   ```

## 🤝 Зависимости
```text
torch>=2.0                    # Инференс, autocast, profiler
torchvision>=0.15             # Предобученные модели (опционально)
pandas>=1.5                   # Агрегация результатов, DataFrame
numpy>=1.24                   # Массивы, метрики
scipy>=1.10                   # Статистика, морфологические операции
scikit-learn>=1.2             # Метрики сегментации
matplotlib>=3.7               # Визуализация
seaborn>=0.12                 # Статистические графики
tqdm>=4.65                    # Прогресс-бары
pillow>=9.5                   # Работа с изображениями
huggingface_hub>=0.16         # Загрузка датасета (опционально)
statsmodels>=0.14             # ANOVA, Tukey HSD (опционально)
mlflow>=2.5                   # Трекинг экспериментов (опционально)
wandb>=0.15                   # Weights & Biases (опционально)
onnx>=1.14                    # Экспорт в ONNX (опционально)
onnxsim>=0.4.0                # Упрощение ONNX (опционально)
torch-tensorrt>=2.0           # Компиляция в TensorRT (опционально)
```

### Установка зависимостей
```bash
# Базовый набор (без экспорта и трекеров)
pip install torch torchvision pandas numpy scipy scikit-learn matplotlib seaborn tqdm pillow huggingface_hub

# + статистический анализ
pip install statsmodels

# + трекеры экспериментов
pip install mlflow wandb

# + экспорт моделей
pip install onnx onnxsim torch-tensorrt

# Или всё сразу:
pip install -r requirements_batch_neural.txt
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование BatchNeuralTester |
|--------|--------------------------------|
| `NeuralSegmenter` | Загрузка моделей и инференс через единый интерфейс |
| `SegmentationMetrics` | Расчёт IoU, Dice, Boundary F1 и других метрик |
| `NeuralModelFactory` | Создание моделей для тестирования (опционально) |
| `utils.strategies` | Единая стратегия инференса для всех архитектур |
| `utils.palettes` | Цветовые палитры для визуализации оверлеев |
| `utils.backend_exporter` | Альтернативный экспорт в ONNX/TRT (если нужен) |

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