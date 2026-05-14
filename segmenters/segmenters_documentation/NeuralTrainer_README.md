# 🧠⚡ NeuralTrainer — Трейнер для fine-tuning моделей семантической сегментации

## 📖 Описание
Модуль `NeuralTrainer.py` предоставляет **универсальный цикл обучения** для дообучения нейронных моделей семантической сегментации на датасетах типа ADE20K (150 классов).

> ⚠️ **Важно:** Данный модуль отвечает за **обучение и валидацию** моделей. Для создания моделей используйте `NeuralModelFactory`, для инференса — `NeuralSegmenter`.

## ✨ Ключевые возможности
### 🔧 Поддержка современных практик обучения

| Фича | Описание | Применение |
|------|----------|-----------|
| **`ignore_index`** | Игнорирование пикселей с заданным индексом в `CrossEntropyLoss` | ADE20K (255), Cityscapes (255), пользовательские маски |
| **Auxiliary loss** | Взвешенный вспомогательный лосс для моделей с `aux` выходом | DeepLabV3+, FCN (улучшает сходимость на ранних эпохах) |
| **Gradient clipping** | Ограничение нормы градиента (`max_norm=1.0`) | Стабилизация обучения, предотвращение взрывов градиента |
| **CosineAnnealingLR** | Плавное уменьшение LR по косинусоиде | Более стабильная сходимость, избегание локальных минимумов |
| **Early stopping** | Остановка по валидационному mIoU с `patience` | Экономия времени, предотвращение переобучения |
| **Macro mIoU** | Расчёт метрики через `sklearn.metrics.jaccard_score` | Учёт только присутствующих классов, `zero_division=0` |

### 🔄 Единый интерфейс обучения
```python
from segmenters.NeuralTrainer import NeuralTrainer

# Инициализация трейнера
trainer = NeuralTrainer(
    model=unet_model,              # nn.Module с выходом [B, C, H, W]
    train_loader=train_dl,         # DataLoader с батчами {"image": ..., "mask": ...}
    val_loader=val_dl,             # Валидационный DataLoader
    num_classes=150,               # Количество выходных классов
    ignore_index=255,              # Индекс игнорируемых пикселей
    aux_loss_weight=0.4,           # Вес aux-лосса (для DeepLab/FCN)
    device="cuda",
    lr=1e-4,
    weight_decay=1e-4
)

# Обучение с early stopping
history = trainer.fit(
    epochs=100,
    checkpoint_path="./models/unet_best.pth",
    early_stop_patience=200
)

# Доступ к результатам
print(f"Best mIoU: {trainer.best_miou:.4f}")
print(f"Train losses: {history['train_loss'][-10:]}")
print(f"Val mIoU curve: {history['val_miou']}")
```

### 🎚️ Автоматическая обработка данных
- **Валидация масок**: невалидные значения (`<0` или `>=num_classes`) автоматически заменяются на `ignore_index`.
- **Поддержка dict-выхода**: модели с `{"out": ..., "aux": ...}` (DeepLab, FCN) обрабатываются корректно.
- **Отладка первого батча**: опциональный `verbose_first_batch` выводит статистику по маскам для быстрой диагностики.

## 🚀 Быстрый старт
### Базовое обучение: U-Net на подмножестве ADE20K
```python
import torch
from torch.utils.data import DataLoader
from segmenters.NeuralTrainer import NeuralTrainer
from dataseters.ADE20KDataset import ADE20KDataset

# Подготовка данных
train_dataset = ADE20KDataset(root_dir="./data/ade20k", split="training", augment=True)
val_dataset = ADE20KDataset(root_dir="./data/ade20k", split="validation", augment=False)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# Создание модели (через NeuralModelFactory или напрямую)
from segmenters.NeuralModelFactory import NeuralModelFactory, ModelType
model, _, _ = NeuralModelFactory.create_model(
    ModelType.UNET_SMP,
    encoder_name="resnet34",
    num_classes=150,
    device="cuda"
)

# Инициализация и запуск трейнера
trainer = NeuralTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_classes=150,
    ignore_index=255,  # ADE20K стандарт
    aux_loss_weight=0.0,  # U-Net не имеет aux выхода
    device="cuda",
    lr=1e-4
)

history = trainer.fit(
    epochs=50,
    checkpoint_path="./checkpoints/unet_resnet34.pth",
    early_stop_patience=100
)
```

### Обучение DeepLabV3+ с auxiliary loss
```python
# Для моделей с aux выходом (DeepLab, FCN) включите aux_loss_weight
trainer = NeuralTrainer(
    model=deeplab_model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_classes=150,
    ignore_index=255,
    aux_loss_weight=0.4,  # Рекомендуемое значение для DeepLab
    device="cuda",
    lr=1e-4
)

# В тренировочном цикле лосс автоматически считается как:
# loss = CE(main_out, mask) + 0.4 * CE(aux_out, mask)
history = trainer.fit(epochs=100)
```

### Мониторинг обучения в реальном времени
```python
# История доступна в реальном времени через trainer.history
for epoch in range(len(trainer.history["val_miou"])):
    print(
        f"Epoch {epoch+1}: "
        f"train_loss={trainer.history['train_loss'][epoch]:.4f}, "
        f"val_miou={trainer.history['val_miou'][epoch]:.4f}"
    )

# Построение learning curves
import matplotlib.pyplot as plt
epochs = range(1, len(history["val_miou"]) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, history["train_loss"], label="Train")
plt.plot(epochs, history["val_loss"], label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss curves")

plt.subplot(1, 2, 2)
plt.plot(epochs, [m*100 for m in history["val_miou"]])
plt.xlabel("Epoch")
plt.ylabel("Val mIoU (%)")
plt.title("Validation mIoU")
plt.tight_layout()
plt.savefig("training_curves.png")
```

## ⚙️ Конфигурация
### Параметры `__init__()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `model` | `nn.Module` | — | Обучаемая модель с выходом `[B, C, H, W]` или `dict` |
| `train_loader` | `DataLoader` | — | DataLoader для тренировочного набора |
| `val_loader` | `DataLoader` | — | DataLoader для валидационного набора |
| `num_classes` | `int` | `150` | Количество выходных классов (ADE20K) |
| `device` | `str` | `"cuda"` | Устройство: `"cuda"` или `"cpu"` |
| `lr` | `float` | `1e-4` | Начальный learning rate |
| `weight_decay` | `float` | `1e-4` | Коэффициент L2-регуляризации (AdamW) |
| `ignore_index` | `int` | `255` | Индекс пикселей для игнорирования в лоссе |
| `aux_loss_weight` | `float` | `0.4` | Вес вспомогательного лосса (0.0 = отключён) |
| `verbose_first_batch` | `bool` | `False` | Вывод отладочной информации по первому батчу |

### Параметры `fit()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `epochs` | `int` | `20` | Максимальное количество эпох обучения |
| `checkpoint_path` | `str` | `"./models/best_model.pth"` | Путь для сохранения лучшего чекпоинта |
| `early_stop_patience` | `int` | `200` | Эпох без улучшения `val_miou` для остановки |

### Возвращаемое значение `fit()` — `HistoryDict`
```python
{
    "train_loss": List[float],  # Средний лосс за эпоху (train)
    "val_loss": List[float],    # Средний лосс за эпоху (val)
    "val_miou": List[float],    # Macro mIoU на валидации
}
```

## 📚 Справочник методов
### 🔹 Основные методы обучения
| Метод | Параметры | Описание | Возвращает |
|-------|-----------|----------|-----------|
| `train_epoch()` | — | Одна эпоха обучения: forward, backward, optimizer step | `float`: средний train loss |
| `validate()` | — | Валидация + расчёт macro mIoU | `Tuple[float, float]`: (val_loss, val_miou) |
| `fit()` | `epochs`, `checkpoint_path`, `early_stop_patience` | Полный цикл обучения с логированием и сохранением | `HistoryDict`: история метрик |

### 🔹 Внутренняя логика (для расширения)
| Метод | Описание |
|-------|----------|
| `_compute_loss(outputs, masks)` | Расчёт основного + aux лосса (если применимо) |
| `_update_scheduler()` | Шаг scheduler'а после каждого батча |
| `_save_checkpoint(path, epoch, miou)` | Сохранение state_dict модели и оптимизатора |

## 🔄 Конвейер обучения: forward → loss → backward → update
### Обработка выхода модели
```python
# Поддержка двух форматов выхода
outputs = model(images)  # [B, C, H, W] или {"out": ..., "aux": ...}

if isinstance(outputs, dict):
    main_out = outputs["out"]      # Основной выход для лосса
    aux_out = outputs.get("aux")   # Вспомогательный выход (опционально)
else:
    main_out = outputs
    aux_out = None
```

### Расчёт лосса с auxiliary компонентой
```python
# Основной лосс
loss = criterion(main_out, masks)

# Вспомогательный лосс (если есть и вес > 0)
if aux_out is not None and aux_loss_weight > 0:
    loss = loss + aux_loss_weight * criterion(aux_out, masks)
```

### Валидация масок перед лоссом
```python
# Замена невалидных значений на ignore_index
invalid = ((masks < 0) | (masks >= num_classes)) & (masks != ignore_index)
if torch.any(invalid):
    masks = masks.clone()
    masks[invalid] = ignore_index
```

### Градиентный клиппинг и шаг оптимизатора
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Стабильность
optimizer.step()
scheduler.step()  # CosineAnnealingLR после каждого батча
```

## 📊 Метрики качества
### Macro mIoU через `sklearn.metrics.jaccard_score`
```python
from sklearn.metrics import jaccard_score

# Фильтрация игнорируемых пикселей
filtered = [(p, t) for p, t in zip(preds, targets) if t != ignore_index]
if filtered:
    f_preds, f_targets = zip(*filtered)
    present_labels = list(set(f_targets))  # Только присутствующие классы
    
    miou = jaccard_score(
        f_targets,
        f_preds,
        average="macro",           # Среднее по классам, не по пикселям
        labels=present_labels,     # Игнорировать отсутствующие классы
        zero_division=0            # Защита от деления на 0
    )
```

| Метрика | Описание | Диапазон | Интерпретация |
|---------|----------|----------|---------------|
| **Train/Val Loss** | CrossEntropyLoss с `ignore_index` | [0, ∞) | Мониторинг сходимости; разрыв train/val → переобучение |
| **Val mIoU (macro)** | Среднее IoU по присутствующим классам | [0.0, 1.0] | Основной критерий для ADE20K; устойчив к дисбалансу |

## ⚡ Производительность и оптимизации
### Рекомендации по ускорению обучения
1. **Mixed Precision Training** (PyTorch 1.6+):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   with autocast():
       outputs = model(images)
       loss = criterion(outputs, masks)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

2. **Градиентный аккумулирование** для больших batch_size:
   ```python
   # Эмуляция batch_size=16 при GPU-памяти на 4
   accumulation_steps = 4
   for batch_idx, batch in enumerate(train_loader):
       outputs = model(batch["image"])
       loss = criterion(outputs, batch["mask"]) / accumulation_steps
       loss.backward()
       
       if (batch_idx + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **Предварительная загрузка данных**:
   ```python
   # В DataLoader: num_workers > 0, pin_memory=True для CUDA
   train_loader = DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=True)
   ```

4. **Кэширование аугментаций**: для стабильных экспериментов используйте `worker_init_fn` с фиксированным seed.

### Относительная скорость эпохи (на подмножестве 5% ADE20K, 512×512, batch=4, RTX 3090)
```
✅ Быстро (<30 сек/эпоха):
   - U-Net + ResNet34 (без aux)
   - FCN + ResNet50 (с aux, но frozen backbone)

⚠️ Средне (30–90 сек/эпоха):
   - DeepLabV3+ + ResNet101 (с aux, unfrozen)
   - PSPNet + MiT-B5

❌ Медленно (90–180+ сек/эпоха):
   - Любая модель с full fine-tuning больших encoder'ов
   - Обучение с `verbose_first_batch=True` (доп. логирование)
```

## 🛠️ Обработка ошибок и устойчивость
### Валидация масок перед лоссом
```python
# Автоматическая замена невалидных значений
invalid = ((masks < 0) | (masks >= num_classes)) & (masks != ignore_index)
if torch.any(invalid):
    if batch_idx == 0:  # Только первый батч для избежания спама
        print(f"⚠️ Found {invalid.sum().item()} invalid mask values")
    masks = masks.clone()
    masks[invalid] = ignore_index
```

### Безопасное сохранение чекпоинта
```python
try:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "miou": val_miou,
        "ignore_index": ignore_index,
    }, checkpoint_path)
except Exception as e:
    logger.error(f"Failed to save checkpoint: {e}")
    # Не прерываем обучение, пробуем сохранить в backup
    backup_path = checkpoint_path + ".backup"
    torch.save({...}, backup_path)
```

### Early stopping с защитой от флуктуаций
```python
# Остановка только если val_miou не превышает best * 0.999
# (защита от микро-улучшений из-за шума)
if patience_counter >= early_stop_patience:
    recent_miou = history["val_miou"][-early_stop_patience:]
    if max(recent_miou) <= best_miou * 0.999:
        print("⏹️ Early stopping triggered")
        break
```

### Рекомендации по отладке
1. **Включите `verbose_first_batch=True`** для проверки данных:
   ```python
   trainer = NeuralTrainer(..., verbose_first_batch=True)
   # Вывод: диапазон масок, уникальные классы, счётчики ключевых классов
   ```

2. **Проверьте `ignore_index`**:
   ```python
   print(f"Criterion ignore_index: {trainer.criterion.ignore_index}")
   print(f"Trainer ignore_index: {trainer.ignore_index}")
   # Должны совпадать!
   ```

3. **Мониторьте градиенты** при нестабильном обучении:
   ```python
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
   ```

4. **Тестируйте на 1 эпохе и 1% данных**:
   ```python
   history = trainer.fit(epochs=1, early_stop_patience=1)
   # Быстрая проверка конвейера без долгого ожидания
   ```

## 🤝 Зависимости
```text
torch>=1.9                    # Основные тензорные операции, autograd
torchvision>=0.10             # Предобученные модели (опционально)
scikit-learn>=1.0             # jaccard_score для расчёта mIoU
numpy>=1.20                   # Массивы, метрики
```

### Опциональные зависимости для расширенного мониторинга
```bash
# TensorBoard для визуализации обучения
pip install tensorboard

# Weights & Biases для экспериментов
pip install wandb
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование NeuralTrainer |
|--------|----------------------------|
| `ModelTrainer` | Высокоуровневый обёрточный класс, использующий `NeuralTrainer` для обучения |
| `NeuralModelFactory` | Создание моделей, которые передаются в `NeuralTrainer` |
| `ADE20KDataset` | Основной датасет для `train_loader`/`val_loader` |
| `SegmentationTester` | Тестирование обученных моделей через `add_method()` |
| `CpuCudaBenchmark` | Бенчмарк времени обучения/инференса после `fit()` |

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
