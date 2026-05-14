# 🔍 Analyze — Скрипт анализа влияния аугментаций на качество сегментации

## 📖 Описание
Скрипт `analyze.py` предоставляет **быстрый способ оценки** влияния уровней аугментации данных на качество предобученных моделей семантической сегментации на датасете **ADE20K** (150 классов).

> ⚠️ **Важно:** Данный скрипт предназначен для **пост-тренировочного анализа** уже обученных моделей. Для обучения используйте `ModelTrainer`, для пакетного тестирования — `BatchNeuralTester`.

## ✨ Ключевые возможности
### 🔄 Полный цикл анализа в одном скрипте

| Этап | Функционал | Инструменты |
|------|-----------|------------|
| **Поиск моделей** | Авто-обнаружение чекпоинтов по шаблону `{model}_{aug}_*.pth` | `glob`, `os.path.getctime` |
| **Загрузка данных** | Тестовое изображение и маска из HuggingFace Hub | `hf_hub_download` |
| **Инференс** | Предсказание с ресайзом к размеру GT, расчёт метрик | `NeuralSegmenter`, `SegmentationMetrics` |
| **Метрики** | Многоклассовый mIoU + бинарные метрики (Dice, F1, Precision, Recall) | `sklearn`, `scipy` |
| **Визуализация** | 2×3 сетка метрик, bar-чарты, heatmaps, сравнение времени | `matplotlib`, `seaborn` |
| **Экспорт** | CSV, Markdown-отчёт, PNG-графики, overlay-визуализации | `pandas`, `PIL` |

### 🎚️ Поддерживаемые модели и аугментации
| Модель | Ключ в чекпоинте | `ModelType` enum |
|--------|-----------------|-----------------|
| U-Net (SMP) | `unet_smp_*_*.pth` | `"unet_smp"` |
| FPN (SMP) | `fpn_smp_*_*.pth` | `"fpn_smp"` |
| PSPNet (SMP) | `psp_smp_*_*.pth` | `"pspnet_smp"` |
| DeepLabV3+ (TV) | `deeplab_tv_*_*.pth` | `"deeplab_tv"` |
| FCN (TV) | `fcn_tv_*_*.pth` | `"fcn_tv"` |
| SegNet | `segnet_*_*.pth` | `"segnet"` |

| Уровень аугментации | Описание |
|---------------------|----------|
| `none` | Только ресайз и нормализация |
| `basic` | + горизонтальный флип |
| `medium` | + ротация, color jitter, масштабирование |

### 📊 Расчёт метрик: многоклассовый + бинарный режим
```python
# Многоклассовый mIoU (150 классов)
classes = np.unique(np.concatenate([gt_mask, pred_mask]))
for cls in classes:
    if cls == 255: continue  # ignore index
    pred_cls = (pred_mask == cls).astype(np.uint8)
    gt_cls = (gt_mask == cls).astype(np.uint8)
    iou = intersection / union if union > 0 else 0.0
m_iou = np.mean(iou_per_class)

# Бинарные метрики (объект vs фон)
pred_binary = (pred_mask > 0).astype(np.uint8)
gt_binary = (gt_mask > 0).astype(np.uint8)
metrics = SegmentationMetrics.calculate_all_metrics(
    pred_mask=pred_binary, gt_mask=gt_binary, include_hausdorff=True
)
```

## 🚀 Быстрый старт
### Базовый запуск анализа
```bash
# Убедитесь, что чекпоинты лежат в ./models с именами вида:
# unet_smp_none_*.pth, unet_smp_basic_*.pth, unet_smp_medium_*.pth

python analyze.py
```

**Ожидаемый вывод:**
```
================================================================================
ИССЛЕДОВАНИЕ: ВЛИЯНИЕ АУГМЕНТАЦИЙ НА КАЧЕСТВО СЕГМЕНТАЦИИ
================================================================================

🔍 Поиск чекпоинтов...
   ✅ unet_smp_none: unet_smp_none_20240101_120000.pth
   ✅ unet_smp_basic: unet_smp_basic_20240101_120000.pth
   ✅ unet_smp_medium: unet_smp_medium_20240101_120000.pth
   ⚠️  fcn_tv_none: не найден

📥 Загрузка тестовых данных...
   ✅ Изображение: (640, 480)
   ✅ Маска: (480, 640), unique values: 87

🧪 Оценка моделей...
   🔹 unet_smp_none unet_smp_none...
      ✅ IoU (mIoU): 0.4523, Dice: 0.6234
      ✅ Время: 0.234s
      ✅ Сохранено: ./data/augmentation_analysis/overlay_unet_smp_none.jpg

================================================================================
РЕЗУЛЬТАТЫ ОЦЕНКИ
================================================================================

📊 Сводная таблица метрик (mIoU):
augmentation     basic   medium     none
model                              
fcn_tv          0.4123   0.4312   0.3987
unet_smp        0.4687   0.4812   0.4523

📈 Построение графиков...
   ✅ График метрик сохранен: ./data/augmentation_analysis/augmentation_impact_metrics.png
   ✅ График mIoU сохранен: ./data/augmentation_analysis/augmentation_impact_miou.png
   ✅ Heatmap сохранен: ./data/augmentation_analysis/augmentation_gain_heatmap.png

================================================================================
СТАТИСТИЧЕСКИЙ АНАЛИЗ
================================================================================

📊 Средний mIoU по уровням аугментаций:
   None:   0.4255
   Basic:  0.4405 (прирост: +3.53%)
   Medium: 0.4562 (прирост: +7.21%)

🏆 Лучшая комбинация:
   Модель: unet_smp
   Аугментации: medium
   mIoU: 0.4812

💾 Результаты сохранены: ./data/augmentation_analysis/augmentation_impact_results.csv
📄 Отчёт сохранён: ./data/augmentation_analysis/report.md
```

### Анализ с кастомной директорией моделей
```bash
# Если чекпоинты лежат в другом месте
python analyze.py
# (отредактируйте models_dir в скрипте или создайте symlink)
```

### Интерпретация результатов
```python
# После запуска анализ доступен в переменных
result_df, overlay_images = analyze_augmentation_impact()

# Доступ к метрикам
print(result_df[["model", "augmentation", "iou", "dice", "inference_time"]])

# Визуализация оверлеев
for key, overlay in overlay_images.items():
    overlay.show()  # или overlay.save(f"{key}.png")
```

## ⚙️ Конфигурация
### Параметры внутри `analyze_augmentation_impact()`
| Переменная | Тип | По умолчанию | Описание |
|-----------|-----|--------------|----------|
| `models_dir` | `str` | `"./models"` | Директория с чекпоинтами моделей |
| `model_types` | `List[str]` | `["unet_smp", "fpn_smp", ...]` | Список архитектур для поиска |
| `augmentation_levels` | `List[str]` | `["none", "basic", "medium"]` | Уровни аугментации для анализа |
| `repo_id` | `str` | `"hf-internal-testing/fixtures_ade20k"` | ID датасета на HuggingFace Hub |
| `output_dir` | `str` | `"./data/augmentation_analysis"` | Директория для результатов |

### Возвращаемое значение
```python
Tuple[Optional[pd.DataFrame], Optional[Dict[str, PIL.Image]]]:
- DataFrame с колонками:
  • model, augmentation, checkpoint
  • iou (mIoU), binary_iou, dice, f1_score, precision, recall, accuracy
  • mae, hausdorff, inference_time
  • pred_mask, gt_mask (numpy arrays)
- Dict с оверлеями: {ключ_модели: PIL.Image в режиме "RGB"}
```

## 📚 Справочник функций
### 🔹 Основная функция
| Функция | Параметры | Описание | Возвращает |
|---------|-----------|----------|-----------|
| `analyze_augmentation_impact()` | — | Полный цикл: поиск → загрузка → инференс → метрики → визуализация → экспорт | `Tuple[DataFrame, Dict[overlay]]` или `(None, None)` при ошибке |

### 🔹 Вспомогательная функция
| Функция | Параметры | Описание | Возвращает |
|---------|-----------|----------|-----------|
| `save_augmentation_comparison_grid()` | `overlay_images`, `output_dir`, `model_names` | Создание единой сетки сравнения всех моделей × аугментаций | `None` (сохраняет PNG) |

### 🔹 Type Aliases
```python
MaskArray = np.ndarray          # Маска сегментации (H, W), dtype=uint8/int
ImageArray = np.ndarray         # Изображение (H, W, 3), dtype=uint8
MetricValue = float             # Числовое значение метрики
MetricsDict = Dict[str, float]  # Словарь метрик {name: value}
PathLike = Union[str, Path]     # Путь к файлу или директории
```

## 🔄 Конвейер анализа: поиск → загрузка → инференс → метрики → визуализация
### Логика `analyze_augmentation_impact()`
```python
def analyze_augmentation_impact():
    # 1. Поиск чекпоинтов по шаблону
    for model_type in model_types:
        for aug in augmentation_levels:
            pattern = f"{models_dir}/{model_type}_{aug}_*.pth"
            files = glob.glob(pattern)
            if files:
                latest = max(files, key=os.path.getctime)
                checkpoints[f"{model_type}_{aug}"] = {...}

    # 2. Загрузка тестовых данных из HF Hub
    img_path = hf_hub_download(repo_id, "ADE_val_00000001.jpg")
    mask_path = hf_hub_download(repo_id, "ADE_val_00000001.png")
    test_image = Image.open(img_path).convert("RGB")
    gt_mask = np.array(Image.open(mask_path))

    # 3. Инференс и метрики для каждой модели
    for key, info in checkpoints.items():
        segmenter = NeuralSegmenter(model_type=info["model_type"], ...)
        
        # Предсказание
        pred_mask, _ = segmenter.predict_segmentation_map(test_image, verbose=False)
        
        # Ресайз к размеру GT
        if gt_mask.shape != pred_mask.shape:
            sh, sw = gt_mask.shape[0]/pred_mask.shape[0], ...
            pred_mask = zoom(pred_mask, (sh, sw), order=0)
        
        # Многоклассовый mIoU
        for cls in np.unique(np.concatenate([gt_mask, pred_mask])):
            if cls == 255: continue
            iou = intersection / union
        m_iou = np.mean(iou_per_class)
        
        # Бинарные метрики
        metrics = SegmentationMetrics.calculate_all_metrics(
            pred_mask=(pred_mask>0).astype(uint8),
            gt_mask=(gt_mask>0).astype(uint8),
            ...
        )
        
        # Сохранение результатов и оверлея
        results.append({...})
        overlay, _ = segmenter.segment_image_unified(test_image, alpha=0.6)
        overlay_images[key] = overlay
        
        # Очистка памяти
        del segmenter; torch.cuda.empty_cache(); gc.collect()

    # 4. Агрегация и визуализация
    df = pd.DataFrame(results)
    # ... построение графиков через matplotlib/seaborn ...
    
    # 5. Экспорт
    df.to_csv(f"{output_dir}/results.csv")
    # ... генерация Markdown-отчёта ...
    
    return df, overlay_images
```

### Создание сравнительной сетки оверлеев
```python
def save_augmentation_comparison_grid(overlay_images, output_dir, model_names=None):
    # Извлечение имён моделей из ключей "model_aug"
    if model_names is None:
        models = list(set(key.rsplit("_", 1)[0] for key in overlay_images))
    
    # Создание сетки: строки = модели, столбцы = [none, basic, medium]
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 5*len(models)))
    
    for row, model in enumerate(models):
        for col, aug in enumerate(["none", "basic", "medium"]):
            key = f"{model}_{aug}"
            ax = axes[row, col]
            if key in overlay_images:
                ax.imshow(overlay_images[key])
                ax.set_title(f"{aug.upper()}")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
    
    plt.savefig(f"{output_dir}/full_comparison_grid.png", dpi=300)
```

## 📊 Интерпретация метрик и визуализаций
### Многоклассовый mIoU vs Бинарный IoU
| Метрика | Описание | Когда использовать |
|---------|----------|-------------------|
| **mIoU** | Среднее IoU по 150 классам | Основная метрика для ADE20K, учитывает все классы |
| **Binary IoU** | IoU для задачи "объект vs фон" | Быстрая оценка, если важен только факт детекции |
| **Dice** | 2·TP / (2·TP + FP + FN) | Более устойчив к дисбалансу классов |
| **Precision/Recall** | Точность и полнота детекции | Диагностика ошибок: ложные срабатывания/пропуски |

### Интерпретация heatmaps прироста
```python
# Heatmap показывает прирост относительно baseline (none) в процентах:
#   +10% (зелёный) = значительное улучшение
#    0% (жёлтый) = без изменений
#   -10% (красный) = ухудшение качества

# Пример вывода:
#           none    basic   medium
# unet_smp   0.45   +3.5%   +7.2%   ✅ Аугментации помогают
# fcn_tv     0.40   -1.2%   +0.8%   ⚠️  Basic ухудшил, medium — нейтрально
```

### Анализ времени инференса
```python
# График inference_time показывает:
# - Абсолютное время на изображение (сек)
# - Относительную сложность моделей
# - Влияние аугментаций на время (обычно незначительное)

# Рекомендация: если разница во времени >2×, рассмотрите более лёгкую архитектуру
```

## ⚡ Производительность и оптимизации
### Ожидаемое время выполнения (на одном изображении 640×480, RTX 3090)
```
✅ Быстро (<0.5 сек/модель):
   - unet_smp + resnet34
   - fcn_tv + resnet50

⚠️ Средне (0.5–2 сек/модель):
   - fpn_smp + mit_b5
   - psp_smp + mit_b5
   - deeplab_tv + resnet101

❌ Медленно (2–5+ сек/модель):
   - Любая модель с загрузкой тяжёлых весов впервые
   - Модели с post-processing (CRF, морфология)
```

### Рекомендации по ускорению
1. **Кэшируйте загрузку данных**: тестовое изображение скачивается один раз из HF Hub.
2. **Используйте `torch.cuda.empty_cache()`** между моделями для предотвращения OOM.
3. **Отключайте `verbose=False`** в `predict_segmentation_map()` для ускорения.
4. **Запускайте на одном изображении** для быстрой проверки конвейера.

## 🛠️ Обработка ошибок и устойчивость
### Поиск чекпоинтов с логированием
```python
for model_type in model_types:
    for aug in augmentation_levels:
        files = glob.glob(f"{models_dir}/{model_type}_{aug}_*.pth")
        if files:
            latest = max(files, key=os.path.getctime)
            # ... сохранение ...
        else:
            print(f"   ⚠️  {model_type}_{aug}: не найден")  # Не прерываем цикл
```

### Безопасный инференс с очисткой памяти
```python
try:
    segmenter = NeuralSegmenter(...)
    pred_mask, _ = segmenter.predict_segmentation_map(test_image, verbose=False)
    # ... расчёт метрик ...
    overlay, _ = segmenter.segment_image_unified(...)
    
    # 🔹 ОЧИСТКА ПАМЯТИ
    del segmenter, pred_mask
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    
except Exception as e:
    print(f"   ❌ Ошибка {key}: {e}")
    traceback.print_exc()
    continue  # Переходим к следующей модели
```

### Ресайз маски с nearest-neighbor интерполяцией
```python
# Сохранение целочисленных меток классов при ресайзе
if gt_mask.shape != pred_mask.shape:
    sh, sw = gt_mask.shape[0]/pred_mask.shape[0], gt_mask.shape[1]/pred_mask.shape[1]
    pred_mask_resized = zoom(pred_mask, (sh, sw), order=0)  # order=0 = nearest
```

### Рекомендации по отладке
1. **Проверьте наличие CUDA**:
   ```python
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"Device: {torch.cuda.get_device_name(0)}")
   ```

2. **Убедитесь в наличии чекпоинтов**:
   ```bash
   ls ./models/*_none_*.pth ./models/*_basic_*.pth ./models/*_medium_*.pth
   ```

3. **Запустите с одним изображением для теста**:
   ```python
   # Временно замените загрузку из HF на локальный файл
   test_image = Image.open("local_test.jpg").convert("RGB")
   ```

4. **Проверьте размеры масок**:
   ```python
   print(f"GT shape: {gt_mask.shape}, Pred shape: {pred_mask.shape}")
   # Должны совпадать после ресайза
   ```

## 🤝 Зависимости
```text
torch>=1.9                    # Инференс моделей, CUDA-поддержка
pandas>=1.3                   # Агрегация результатов, DataFrame
numpy>=1.20                   # Массивы, метрики, ресайз
matplotlib>=3.4               # Визуализация графиков
seaborn>=0.12                 # Heatmaps, статистические графики
Pillow>=8.0                   # Загрузка и обработка изображений
huggingface_hub>=0.10         # Загрузка тестовых данных из HF Hub
scipy>=1.7                    # zoom для ресайза масок
scikit-learn>=1.0             # Метрики сегментации (опционально)
```

### Установка зависимостей
```bash
# Базовый набор
pip install torch pandas numpy matplotlib seaborn Pillow huggingface_hub scipy scikit-learn

# Или из requirements-файла проекта
pip install -r requirements_analyze.txt
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование analyze.py |
|--------|-------------------------|
| `NeuralSegmenter` | Загрузка моделей и инференс через единый интерфейс |
| `SegmentationMetrics` | Расчёт бинарных метрик (IoU, Dice, Hausdorff) |
| `ModelTrainer` | Чекпоинты, обученные через `train_experiment()`, совместимы с анализом |
| `BatchNeuralTester` | `analyze.py` — упрощённая версия для быстрого анализа одной картинки |
| `utils.palettes` | Цветовая палитра ADE20K для визуализации оверлеев |

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