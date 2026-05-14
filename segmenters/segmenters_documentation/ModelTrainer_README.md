# 🧠⚙️ ModelTrainer — Универсальный трейнер для обучения моделей семантической сегментации

## 📖 Описание
Модуль `ModelTrainer.py` предоставляет **единую точку входа** для обучения, валидации и сравнения современных архитектур семантической сегментации на датасете **ADE20K** (150 классов).

> ⚠️ **Важно:** Данный модуль работает с **нейросетевыми методами** (SMP, Torchvision). Для классических алгоритмов используйте `OpenCVSegmenter` или `SklearnSegmenter`.

## ✨ Ключевые возможности
### 🏗️ Поддерживаемые архитектуры (6 семейств)

| Семейство | Модели | Encoder'ы | Особенности |
|-----------|--------|-----------|-------------|
| **SMP** (3) | `unet_smp`, `fpn_smp`, `psp_smp` | `resnet34/50/101`, `mit_b0-b5`, `efficientnet-b0` | Предобученные веса ImageNet, гибкая настройка декодера |
| **Torchvision** (2) | `deeplab_tv`, `fcn_tv` | `resnet50`, `resnet101` | Официальные реализации, предобучение на COCO+VOC |
| **Custom** (1) | `segnet` | proxy через `unet_smp` | Эмуляция SegNet через U-Net или кастомная реализация |

### 🎚️ 4 уровня аугментации данных

| Уровень | Трансформации | Сценарии использования |
|---------|--------------|----------------------|
| `none` | Ресайз, нормализация | Отладка, baseline, воспроизводимость |
| `basic` | + горизонтальный флип | Быстрое улучшение обобщения |
| `medium` | + ротация (30%), color jitter, масштабирование (0.9–1.1×) | Стандарт для большинства задач |
| `aggressive` | + вертикальный флип, усиленный jitter, большее масштабирование | Маленькие датасеты, сложная инвариантность |

### 🔧 Автоматическая настройка под тип модели
- **Правильный `ignore_index`**: таблица `IGNORE_INDEX_BY_MODEL` гарантирует корректную обработку игнорируемых пикселей для каждого типа модели.
- **Разморозка бэкбона**: для Torchvision-моделей бэкбон автоматически размораживается на эпохе 5 с уменьшением LR в 10×.
- **Взвешенный лосс**: опциональный `compute_class_weights()` для борьбы с дисбалансом классов (median frequency balancing).
- **Early stopping**: остановка по валидационному mIoU с настраиваемым `patience`.

### 🔄 Единый интерфейс для всех экспериментов
```python
from segmenters.ModelTrainer import ModelTrainer, TrainingConfig

# Инициализация трейнера
trainer = ModelTrainer(
    checkpoint_dir="./models",
    root_dir="./data/ade20k",
    device="cuda"
)

# Конфигурация эксперимента
config = TrainingConfig(
    experiment_name="unet_baseline",
    model_type="unet_smp",
    augmentation_level="medium",
    epochs=100,
    batch_size=4,
    lr=1e-4,
    encoder_name="resnet34",
    use_class_weights=True
)

# Обучение
result = trainer.train_experiment(config)
print(f"Best mIoU: {result['best_miou'] * 100:.2f}%")
print(f"Checkpoint: {result['checkpoint_path']}")

# Сравнение моделей
results = trainer.compare_trained_models(
    augmentation_level="medium",
    model_types=["unet_smp", "deeplab_tv", "fpn_smp"]
)
```

### 📊 Визуализация и анализ результатов
```python
# Сравнение аугментаций
df = trainer.compare_augmentations(
    model_type="unet_smp",
    augmentation_levels=["none", "basic", "medium", "aggressive"]
)

# Визуализация всех экспериментов
trainer.plot_experiment_comparison(output_path="./plots/comparison.png")

# Оценка конкретных чекпоинтов
eval_df = trainer.evaluate_checkpoints(
    checkpoint_paths=["./models/unet_epoch_50.pth", "./models/unet_epoch_100.pth"],
    model_type="unet_smp"
)
```

## 🚀 Быстрый старт
### Базовое обучение: U-Net на ADE20K
```python
from segmenters.ModelTrainer import ModelTrainer, TrainingConfig

trainer = ModelTrainer()

config = TrainingConfig(
    experiment_name="unet_quickstart",
    model_type="unet_smp",
    augmentation_level="basic",
    epochs=20,          # Для теста; в продакшене 100–200
    batch_size=4,
    lr=1e-4,
    subset_fraction=0.05  # 5% данных для быстрой проверки
)

result = trainer.train_experiment(config)
```

### Сравнение архитектур (бенчмарк)
```python
# Обучение всех поддерживаемых моделей с одинаковыми гиперпараметрами
all_results = trainer.train_all_models(
    augmentation_level="medium",
    epochs=20,
    batch_size=4,
    lr=1e-4,
    subset_fraction=0.05
)

# Сводная таблица результатов
for name, res in all_results.items():
    print(f"{name:15s} → mIoU: {res['best_miou']*100:5.2f}%")
```

### Загрузка и инференс обученной модели
```python
import torch
from segmenters.ModelTrainer import ModelTrainer

trainer = ModelTrainer()

# Создание модели
model = trainer.create_model("unet_smp", encoder_name="resnet34")

# Загрузка весов
checkpoint = torch.load("./models/unet_smp_medium_20240101_120000.pth", map_location="cuda")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval().to("cuda")

# Инференс
with torch.no_grad():
    output = model(image_tensor.to("cuda"))  # [B, 3, H, W]
    pred_mask = output.argmax(1)             # [B, H, W], значения 0–149
```

## ⚙️ Конфигурация
### Параметры `TrainingConfig.__init__()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `experiment_name` | `str` | — | Человекочитаемое имя эксперимента для логирования |
| `model_type` | `ModelType` | — | Тип архитектуры: `"unet_smp"`, `"deeplab_tv"`, и т.д. |
| `augmentation_level` | `AugmentationLevel` | `"none"` | Уровень аугментаций: `"none"`, `"basic"`, `"medium"`, `"aggressive"` |
| `epochs` | `int` | `20` | Максимальное количество эпох обучения |
| `batch_size` | `int` | `4` | Размер батча для `DataLoader` |
| `lr` | `float` | `1e-4` | Начальный learning rate |
| `encoder_name` | `EncoderName` | `"resnet34"` | Название encoder'а (для SMP-моделей) |
| `variant` | `str` | `"b5"` | Вариант модели (например, `"b5"` для MiT, `"fcn_resnet50"` для FCN) |
| `subset_fraction` | `float` | `0.05` | Доля данных для использования (1.0 = весь датасет) |
| `early_stop_patience` | `int` | `200` | Эпох без улучшения mIoU для early stopping |
| `use_class_weights` | `bool` | `False` | Использовать ли взвешенный CrossEntropyLoss |
| `checkpoint_name` | `Optional[str]` | `None` | Имя файла чекпоинта (генерируется автоматически, если `None`) |

### Возвращаемое значение `train_experiment()` — `TrainingResult`
| Ключ | Тип | Описание |
|------|-----|----------|
| `experiment_name` | `str` | Имя эксперимента |
| `augmentation_level` | `str` | Использованный уровень аугментаций |
| `model_type` | `str` | Тип обученной модели |
| `ignore_index` | `int` | Индекс игнорируемых пикселей в лоссе |
| `epochs_trained` | `int` | Фактическое количество проведённых эпох |
| `best_miou` | `float` | Лучший достигнутый mIoU на валидации |
| `final_train_loss` | `Optional[float]` | Значение тренировочного лосса на последней эпохе |
| `final_val_loss` | `Optional[float]` | Значение валидационного лосса на последней эпохе |
| `checkpoint_path` | `str` | Путь к сохранённому чекпоинту |
| `history` | `Dict[str, List[float]]` | Словарь с историей метрик: `["train_loss", "val_loss", "val_miou"]` |
| `config` | `TrainingConfig` | Исходная конфигурация эксперимента |

## 📚 Справочник методов
### 🔹 Создание модели и данных
| Метод | Параметры | Описание | Возвращает |
|-------|-----------|----------|-----------|
| `create_model()` | `model_type`, `encoder_name`, `variant`, `for_training` | Создаёт экземпляр модели сегментации по указанному типу | `nn.Module` |
| `create_dataloaders()` | `augmentation_level`, `batch_size`, `subset_fraction`, `ignore_index` | Создаёт `DataLoader` для тренировочного и валидационного наборов | `Tuple[DataLoader, DataLoader]` |
| `compute_class_weights()` | `train_loader`, `num_classes`, `ignore_index`, `max_batches` | Считает инвертированные частоты классов для взвешенного лосса | `torch.Tensor` |

### 🔹 Обучение и оценка
| Метод | Параметры | Описание | Возвращает |
|-------|-----------|----------|-----------|
| `train_experiment()` | `config: TrainingConfig` | Обучает один эксперимент с заданной конфигурацией | `TrainingResult` |
| `compare_trained_models()` | `augmentation_level`, `checkpoint_paths`, `model_types` | Сравнивает обученные модели на валидационном наборе | `Dict[str, float]` |
| `evaluate_trained_models_on_val()` | `checkpoints`, `val_fraction`, `model_types` | Оценивает модели по явным путям к чекпоинтам | `Dict[str, float]` |
| `evaluate_checkpoints()` | `checkpoint_paths`, `model_type`, `encoder_name`, `variant` | Оценивает список чекпоинтов одной архитектуры | `pd.DataFrame` |

### 🔹 Сравнение и визуализация
| Метод | Параметры | Описание | Возвращает |
|-------|-----------|----------|-----------|
| `compare_augmentations()` | `model_type`, `augmentation_levels`, `base_config` | Сравнивает обучение с разными уровнями аугментаций | `pd.DataFrame` |
| `plot_experiment_comparison()` | `output_path` | Визуализирует сравнение проведённых экспериментов (2×2 grid) | `None` (сохраняет PNG) |
| `train_all_models()` | `augmentation_level`, `epochs`, `batch_size`, `lr`, `subset_fraction` | Обучает все поддерживаемые модели с одинаковыми гиперпараметрами | `Dict[str, TrainingResult]` |

## 🔄 Конвейер обучения: подготовка → цикл → сохранение
### Автоматическая подготовка (`train_experiment`)
```python
# 1. Выбор ignore_index по таблице IGNORE_INDEX_BY_MODEL
ignore_index = IGNORE_INDEX_BY_MODEL.get(config.model_type, 255)

# 2. Создание модели с правильной инициализацией
model = self.create_model(model_type, encoder_name, variant, for_training=True)

# 3. Создание DataLoader с нужными аугментациями и игнорированием
train_loader, val_loader = self.create_dataloaders(
    config.augmentation_level, config.batch_size, config.subset_fraction, ignore_index
)

# 4. Опциональные веса классов
if config.use_class_weights:
    class_weights = self.compute_class_weights(train_loader, NUM_CLASSES, ignore_index)

# 5. Критерий с правильным ignore_index
criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)

# 6. Optimizer только на trainable параметры
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=1e-4)

# 7. Scheduler на всё обучение (CosineAnnealingLR)
total_steps = len(train_loader) * config.epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=config.lr * 0.01)
```

### Цикл обучения с разморозкой бэкбона
```python
for epoch in range(config.epochs):
    # Разморозка бэкбона для Torchvision-моделей на эпохе 5
    if config.model_type in ["deeplab_tv", "fcn_tv"] and epoch == 5:
        for param in model.backbone.parameters():
            param.requires_grad = True
        # Пересоздание optimizer/scheduler с уменьшенным LR
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr / 10)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_steps)
    
    # Один шаг обучения и валидации
    train_loss = trainer.train_epoch()
    val_loss, val_miou = trainer.validate()
    
    # Early stopping по mIoU
    if epoch >= config.early_stop_patience:
        recent = history["val_miou"][-config.early_stop_patience:]
        if max(recent) <= best_miou * 0.999:
            break
    
    # Сохранение лучшего чекпоинта
    if val_miou > best_miou:
        torch.save({...}, checkpoint_path)
```

## 📊 Метрики качества
При валидации используется **weighted mIoU** через `sklearn.metrics.jaccard_score`:

```python
from sklearn.metrics import jaccard_score

miou = jaccard_score(
    all_targets,
    all_preds,
    average="weighted",      # Взвешенное среднее по классам
    labels=range(150),       # ADE20K: 150 классов
    zero_division=0          # Защита от деления на 0
)
```

| Метрика | Описание | Диапазон | Интерпретация |
|---------|----------|----------|---------------|
| **mIoU (weighted)** | Среднее IoU, взвешенное по поддержке классов | [0.0, 1.0] | Основной критерий для ADE20K; устойчив к дисбалансу |
| **Train/Val Loss** | CrossEntropyLoss с учётом `ignore_index` | [0, ∞) | Мониторинг сходимости; важно смотреть на разрыв между train/val |

## ⚡ Производительность и оптимизации
### Относительная скорость обучения (на подмножестве 5% ADE20K, 512×512, batch=4)
```
✅ Быстро (<30 сек/эпоха):
   - unet_smp + resnet34
   - fcn_tv + resnet50

⚠️ Средне (30–90 сек/эпоха):
   - fpn_smp + mit_b5
   - psp_smp + mit_b5
   - deeplab_tv + resnet101

❌ Медленно (90–180+ сек/эпоха):
   - segnet (custom)
   - Любая модель с use_class_weights=True (доп. проход по данным)
```

### Рекомендации по оптимизации
1. **Используйте `subset_fraction`** для отладки: `0.01`–`0.05` ускоряет итерации в 20–100×.
2. **Заморозка бэкбона**: первые 5 эпох с `for_training=True` экономят память и ускоряют сходимость.
3. **`num_workers=0`** в `DataLoader`: для ADE20K с малым подмножеством это часто быстрее из-за накладных расходов на multiprocessing.
4. **Очистка памяти**: `torch.cuda.empty_cache()` и `gc.collect()` после каждой эпохи предотвращают утечки.
5. **Пакетная оценка**: `compare_trained_models()` загружает модели по очереди, минимизируя пиковое потребление VRAM.

## 🛠️ Обработка ошибок и устойчивость
### Стратегия при сбое загрузки чекпоинта
```python
try:
    checkpoint = torch.load(path, map_location=self.device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)  # Прямая загрузка state_dict
except Exception as e:
    logger.error(f"Failed to load checkpoint {path}: {e}")
    continue  # Пропуск модели в сравнении
```

### Валидация данных перед обучением
```python
# Проверка диапазона значений масок (исключая ignore_index)
valid_masks = masks[masks != ignore_index]
assert valid_masks.min() >= 0 and valid_masks.max() <= NUM_CLASSES - 1, \
    f"Mask values out of range! [{valid_masks.min()}, {valid_masks.max()}]"

# Предупреждение при ignore_index=0 (класс "wall" доминирует в ADE20K)
if ignore_index == 0:
    logger.warning("ignore_index=0, но класс 0='wall' доминирует в ADE20K!")
```

### Рекомендации по отладке
1. **Включите детальное логирование**:
   ```python
   import logging
   logging.getLogger("segmenters.ModelTrainer").setLevel(logging.DEBUG)
   ```

2. **Проверьте конфигурацию перед запуском**:
   ```python
   print(f"Model: {config.model_type}, ignore_index: {IGNORE_INDEX_BY_MODEL[config.model_type]}")
   print(f"Augmentations: {config.augmentation_level}, subset: {config.subset_fraction*100:.1f}%")
   ```

3. **Тестируйте на 1 эпохе и 1% данных**:
   ```python
   config = TrainingConfig(..., epochs=1, subset_fraction=0.01)
   result = trainer.train_experiment(config)  # Быстрая проверка конвейера
   ```

## 🤝 Зависимости
```text
torch>=1.9                    # Основные тензорные операции, autograd
torchvision>=0.10             # Предобученные модели сегментации (DeepLab, FCN)
segmentation-models-pytorch>=0.2  # SMP: U-Net, FPN, PSPNet с предобученными encoder'ами
scikit-learn>=1.0             # jaccard_score для оценки mIoU
pandas>=1.3                   # Таблицы результатов, сравнение экспериментов
matplotlib>=3.4               # Визуализация learning curves
numpy>=1.20                   # Массивы, статистики
Pillow>=8.0                   # Загрузка изображений
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование ModelTrainer |
|--------|---------------------------|
| `ADE20KDataset` | Основной датасет для обучения и валидации |
| `NeuralTrainer` | Базовый класс для цикла обучения/валидации (наследуется внутри) |
| `SegmentationTester` | Универсальное тестирование: можно добавить обученную модель через `add_method()` |
| `CpuCudaBenchmark` | Бенчмарк производительности: сравнение времени инференса обученных моделей |
| `TorchImplementationValidator` | Валидация: сравнение предсказаний обученной модели с эталонной реализацией |

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