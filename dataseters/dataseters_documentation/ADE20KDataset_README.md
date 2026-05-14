# 🗂️ ADE20KDataset — Загрузчик датасета ADE20K с расширенными аугментациями

## 📖 Описание
Модуль `datasets/ADE20KDataset.py` предоставляет **специализированный PyTorch Dataset** для работы с датасетом ADE20K, включающий синхронизированные аугментации для изображений и масок семантической сегментации.

> ⚠️ **Важно:** Данный модуль предназначен *специфически для ADE20K* со структурой `ADEChallengeData2016/`. Для других датасетов используйте универсальный `DatasetManager` из `load_datasets.py` или создайте аналогичный класс.

## ✨ Ключевые возможности

### 🔄 Синхронизированные аугментации
| Тип | Применяется к | Особенности |
|-----|---------------|-------------|
| **Геометрические** | Изображение + Маска | Одинаковые трансформации, `fill=ignore_index` для маски |
| **Фотометрические** | Только изображение | Цветовые искажения не влияют на семантику маски |
| **Ресайз/Кроп** | Оба | Билинейная интерполяция для фото, nearest для масок |

### 🎚️ 4 уровня аугментаций

| Уровень | Аугментации | Рекомендации |
|---------|-------------|--------------|
| `"none"` | Только ресайз + нормализация | Валидация, тестирование, инференс |
| `"basic"` | + Horizontal Flip (p=0.5) | Базовое обучение, стабильная сходимость |
| `"medium"` | + Vertical Flip, Rotation (±30°), Color Jitter, Scale (0.9–1.1) | Стандартное обучение, хороший баланс |
| `"aggressive"` | + Affine (±15°, shear, translate), Gamma, Grayscale | Большие датасеты, борьба с переобучением |

### 🎯 Поддержка форматов и валидация
```python
# Автоматическая проверка соответствия изображение-маска
- Фильтрация битых пар при инициализации
- Клиппинг значений маски: [0, 149] (кроме ignore_index=255)
- Валидация размеров после аугментаций (unit-тест test_augmentation_sync)
```

## 🚀 Быстрый старт

### Базовое использование
```python
from datasets.ADE20KDataset import ADE20KDataset
from torch.utils.data import DataLoader

# Инициализация датасета
dataset = ADE20KDataset(
    root_dir="./data/ade20k",
    split="training",           # "training" | "validation" | "testing"
    image_size=(512, 512),      # Целевой размер (ширина, высота)
    augment=True,               # Применять аугментации
    augmentation_level="medium",# Уровень: none/basic/medium/aggressive
    ignore_index=255,           # Индекс для игнорирования в лоссе
)

# Создание DataLoader
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Training loop
for batch in loader:
    images = batch["image"]   # [B, 3, H, W], нормализованные
    masks = batch["mask"]     # [B, H, W], dtype long, значения 0..149 или 255
    image_ids = batch["image_id"]  # Список имён файлов
    
    # ... forward pass, loss, backward ...
```

### Настройка аугментаций
```python
# Тонкая настройка вероятностей аугментаций
dataset = ADE20KDataset(
    root_dir="./data/ade20k",
    augment=True,
    augmentation_level="medium",
    
    # Переопределение вероятностей
    hflip_prob=0.7,           # Горизонтальный флип (по умолчанию 0.5)
    vflip_prob=0.1,           # Вертикальный флип (по умолчанию 0.0)
    rotation_prob=0.4,        # Ротация ±30° (по умолчанию 0.0 для basic)
    color_jitter_prob=0.3,    # Цветовые искажения (по умолчанию 0.0)
    
    # Диапазон масштабирования
    scale_range=(0.85, 1.15), # Random scale перед crop/pad
)
```

### Быстрое тестирование на подвыборке
```python
# Использование subset_fraction для отладки (1% данных)
dataset = ADE20KDataset(
    root_dir="./data/ade20k",
    split="training",
    subset_fraction=0.01,  # Использовать только 1% данных
    augment=False,          # Отключить аугментации для стабильности
)

print(f"Используем {len(dataset)} из {len(dataset._valid_indices_full)} примеров")
```

## ⚙️ Конфигурация

### Параметры инициализации `ADE20KDataset.__init__()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `root_dir` | `PathLike` | `"./data/ade20k"` | Корневая директория датасета |
| `split` | `str` | `"training"` | Сплит: `"training"`, `"validation"`, `"testing"` |
| `image_size` | `Tuple[int,int]` | `(512, 512)` | Целевой размер (ширина, высота) |
| `augment` | `bool` | `False` | Применять ли аугментации |
| `subset_fraction` | `Optional[float]` | `None` | Доля данных для использования (0.0–1.0) |
| `augmentation_level` | `str` | `"basic"` | Уровень: `"none"`, `"basic"`, `"medium"`, `"aggressive"` |
| `hflip_prob` | `float` | `0.5` | Вероятность горизонтального флипа |
| `vflip_prob` | `float` | `0.0` | Вероятность вертикального флипа |
| `rotation_prob` | `float` | `0.0` | Вероятность ротации ±30° |
| `color_jitter_prob` | `float` | `0.0` | Вероятность цветовых искажений |
| `scale_range` | `Tuple[float,float]` | `(0.8, 1.2)` | Диапазон случайного масштабирования |
| `ignore_index` | `int` | `255` | Индекс пикселей для игнорирования в лоссе |

### Структура возвращаемого батча
```python
batch = dataset[0]  # или next(iter(loader))
{
    "image": torch.Tensor,  # [3, H, W], float32, нормализовано ImageNet статистиками
    "mask": torch.Tensor,   # [H, W], int64, значения 0..149 или ignore_index
    "image_id": str         # Имя файла: "ADE_train_00000001.jpg"
}
```

### Нормализация изображений
Изображения нормализуются статистиками ImageNet:
```python
mean = [0.485, 0.456, 0.406]  # RGB
std = [0.229, 0.224, 0.225]
# Формула: normalized = (pixel / 255 - mean) / std
```

## 📚 Справочник методов

### 🔹 Основные методы `ADE20KDataset`
| Метод | Возвращает | Описание |
|-------|------------|----------|
| `__len__()` | `int` | Количество валидных примеров в датасете |
| `__getitem__(idx)` | `Dict[str, Any]` | Возвращает пример: `{"image", "mask", "image_id"}` |
| `_configure_augmentations(...)` | `None` | Настраивает вероятности аугментаций по уровню |
| `_apply_geometric_augmentations(img, mask)` | `Tuple[Image, Image]` | Синхронные геометрические трансформации |
| `_apply_photometric_augmentations(img)` | `Image` | Цветовые аугментации только для изображения |
| `test_augmentation_sync()` | `bool` | Unit-тест: проверка согласованности размеров и значений |

### 🔹 Внутренняя логика аугментаций
```python
# Геометрические (применяются к обоим):
1. RandomHorizontalFlip(p=hflip_prob)
2. RandomVerticalFlip(p=vflip_prob)
3. RandomRotation(±30°, p=rotation_prob)
   - img: fill=(0,0,0)  # чёрный фон
   - mask: fill=ignore_index  # 255 = игнорировать
4. RandomScale + Crop/Pad (если scale_range != (1.0, 1.0))
5. RandomAffine (только aggressive, p=0.3)

# Фотометрические (только изображение):
1. ColorJitter: brightness/contrast/saturation ±20%
2. RandomGrayscale (aggressive, p=0.1) → конвертация обратно в RGB
3. Gamma correction (aggressive, p=0.2, γ∈[0.7, 1.5])
```

### 🔹 Альтернативная реализация: `ADE20KDatasetWithTransforms`
Более чистая архитектура с `transforms.Compose`, но менее гибкая:
```python
dataset = ADE20KDatasetWithTransforms(
    root_dir="./data/ade20k",
    augment=True,
    image_size=(512, 512)
)
# Использует предопределённые цепочки трансформаций
# Меньше контроля над вероятностями, но проще для стандартных сценариев
```

## 🧪 Тестирование и отладка

### Встроенный тест загрузчика
```python
from datasets.ADE20KDataset import test_dataloader

# Запуск комплексного теста: загрузка, валидация, визуализация
success = test_dataloader()
if success:
    print("✅ Все тесты пройдены!")
    # Результат: ./data/ade20k_augmentations_preview.png
```

### Unit-тест синхронизации аугментаций
```python
dataset = ADE20KDataset(augment=True, augmentation_level="aggressive")
assert dataset.test_augmentation_sync(), "Аугментации рассинхронизированы!"
```

### Отладка аугментаций
```python
# 1. Включите логирование аугментаций
import logging
logging.getLogger("datasets.ADE20KDataset").setLevel(logging.WARNING)

# 2. Проверьте применяемые трансформации в логах:
# [14:32:15] ⚠️ 🔧 Applied: hflip, rotate_-12.3°, scale_1.05

# 3. Визуализируйте батч вручную
batch = dataset[0]
img = batch["image"].permute(1,2,0).numpy()
mask = batch["mask"].numpy()

# Денормализация для визуализации
img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
img = np.clip(img, 0, 1)

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img); ax1.set_title("Image")
ax2.imshow(mask, cmap="tab20"); ax2.set_title("Mask")
plt.show()
```

## ⚡ Производительность и оптимизации

### Рекомендации по настройке DataLoader
```python
# Для CPU-тренировки:
DataLoader(..., num_workers=4, pin_memory=False)

# Для GPU-тренировки:
DataLoader(..., num_workers=8, pin_memory=True, persistent_workers=True)

# Для отладки:
DataLoader(..., num_workers=0, pin_memory=False)  # Упрощает трассировку ошибок
```

### Влияние уровня аугментаций на скорость
```
Уровень      | Относительная скорость | Рекомендация
-------------|------------------------|-------------
"none"       | ████████████ 100%      | Валидация, инференс
"basic"      | ██████████  85%        | Базовое обучение
"medium"     | ████████    70%        | Стандартное обучение
"aggressive" | ██████      50%        | Только при избытке данных
```

### Оптимизация памяти
```python
# 1. Используйте subset_fraction для прототипирования
dataset = ADE20KDataset(subset_fraction=0.01)  # 1% данных

# 2. Отключите аугментации при отладке архитектуры модели
dataset = ADE20KDataset(augment=False)

# 3. Уменьшите image_size для быстрых экспериментов
dataset = ADE20KDataset(image_size=(256, 256))  # Вместо (512, 512)
```

## 🛠️ Обработка ошибок и устойчивость

### Валидация при инициализации
```python
# Автоматические проверки:
- ✅ Существование директорий images/ и annotations/
- ✅ Соответствие имён файлов (.jpg ↔ .png)
- ✅ Фильтрация битых пар (изображение без маски)

# При ошибке:
FileNotFoundError: Images dir not found: ./data/ade20k/ADEChallengeData2016/images/training
```

### Защита масок при аугментациях
```python
# Клиппинг значений маски после трансформаций:
mask_np = np.array(mask_pil, dtype=np.int64)
valid_mask = mask_np != self.ignore_index  # 255
mask_np[valid_mask] = np.clip(mask_np[valid_mask], 0, 149)  # ADE20K: 150 классов

# Это предотвращает:
- Выход значений за диапазон [0, 149] из-за интерполяции
- Коррупцию ignore_index при аугментациях
```

### Логирование и отладка
```python
# Формат логов аугментаций:
[14:32:15] ⚠️ 🔧 Applied: hflip, rotate_-12.3°, scale_1.05

# Ошибки в __getitem__ логируются с трейсом:
❌ Ошибка: ...
Traceback (most recent call last):
  ...
```

## 🤝 Интеграция с другими модулями проекта

| Модуль | Использование ADE20KDataset |
|--------|----------------------------|
| `TorchSegmenter` | Обучение моделей через `create_pytorch_dataset()` |
| `BatchClassicTester` | Тестирование классических методов на реальных данных |
| `SegmentationMetrics` | Оценка качества предсказаний против масок ADE20K |
| `VisualizationTool` | Визуализация предсказаний с наложением на оригиналы |
| `BenchmarkRunner` | Стандартизированный бенчмарк на подмножестве ADE20K |

### Пример полного пайплайна обучения
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.ADE20KDataset import ADE20KDataset
from segmenters.TorchSegmenter import DeepLabV3Plus

# 1. Подготовка данных
train_dataset = ADE20KDataset(
    root_dir="./data/ade20k",
    split="training",
    image_size=(512, 512),
    augment=True,
    augmentation_level="medium",
    ignore_index=255
)

val_dataset = ADE20KDataset(
    root_dir="./data/ade20k",
    split="validation",
    image_size=(512, 512),
    augment=False,  # Нет аугментаций на валидации
    ignore_index=255
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# 2. Модель и лосс
model = DeepLabV3Plus(num_classes=150).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 3. Training loop
for epoch in range(50):
    model.train()
    for batch in train_loader:
        images, masks = batch["image"].cuda(), batch["mask"].cuda()
        
        logits = model(images)  # [B, 150, H, W]
        loss = criterion(logits, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 4. Валидация
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images, masks = batch["image"].cuda(), batch["mask"].cuda()
            preds = model(images).argmax(dim=1)
            # ... расчёт метрик (IoU, Dice) ...
```

## 📦 Зависимости

### Обязательные
```text
torch>=1.9.0           # PyTorch Dataset, DataLoader, тензоры
torchvision>=0.10.0    # transforms, functional API
numpy>=1.20.0          # Массивы, клиппинг, валидация
Pillow>=9.0.0          # Загрузка изображений и масок
matplotlib>=3.4.0      # Визуализация в test_dataloader()
```

### Опциональные (для расширенного тестирования)
```text
tqdm>=4.60.0           # Прогресс-бары в бенчмарках
```

### Установка
```bash
# Базовые зависимости (обычно уже установлены в проекте)
pip install torch torchvision numpy Pillow matplotlib

# Проверка установки
python -c "from datasets.ADE20KDataset import ADE20KDataset; print('✅ OK')"
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

> 💡 **Совет:** Для воспроизводимости результатов зафиксируйте `random.seed()` и `torch.manual_seed()` перед созданием DataLoader. Аугментации используют `random.random()`, поэтому детерминизм требует явной настройки.

```python
import random
import torch

random.seed(42)
torch.manual_seed(42)
# Опционально для CUDA:
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```