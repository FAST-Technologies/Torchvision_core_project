# Torchvision_core_project

> 🎯 **Универсальный фреймворк для сравнительного тестирования и бенчмаркинга методов сегментации изображений**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Оглавление

- [О проекте](#-о-проекте)
- [Возможности](#-возможности)
- [Установка](#-установка)
- [Структура проекта](#-структура-проекта)
- [Быстрый старт](#-быстрый-старт)
- [Использование](#-использование)
- [Поддерживаемые модели](#-поддерживаемые-модели)
- [Метрики качества](#-метрики-качества)
- [Конфигурация](#-конфигурация)
- [Примеры](#-примеры)
- [Вклад в проект](#-вклад-в-проект)
- [Лицензия](#-лицензия)

---

## 🔍 О проекте

**Torchvision_core_project** — это масштабируемый фреймворк для исследования, сравнения и бенчмаркинга алгоритмов семантической сегментации изображений. Проект объединяет классические методы компьютерного зрения и современные нейросетевые архитектуры в едином интерфейсе.

### Ключевые особенности:

✅ **Единый интерфейс** для 50+ методов сегментации;  
✅ **Поддержка библиотек**: OpenCV, Scikit-Learn, Scikit-Image, PyTorch, Transformers, SMP;  
✅ **Бенчмаркинг**: замеры времени, памяти, метрик качества;  
✅ **Визуализация**: автоматическое построение графиков и отчётов;  
✅ **Валидация**: сравнение реализаций между библиотеками;  
✅ **Обучение**: fine-tuning моделей на ADE20K и других датасетах;  
✅ **Warm-up утилиты**: точные замеры производительности (cold/hot run).  

---

## ✨ Возможности

### 🧩 Методы сегментации

| Категория | Методы |
|-----------|--------|
| **Пороговые** | Global, Adaptive, Otsu, Niblack, Sauvola |
| **Градиентные** | Sobel, Canny |
| **Региональные** | Region Growing, Split-and-Merge, Flood Fill, Watershed, Random Walker |
| **Кластеризация** | K-Means, DBSCAN, MeanShift |
| **Активные контуры** | Active Contour, GVF, Morphological Snakes, Chan-Vese |
| **Суперпиксели** | SLIC, Felzenszwalb, QuickShift |
| **Интерактивные** | GrabCut |
| **Нейросетевые** | SegFormer, Mask2Former, OneFormer, DeepLabV3+, U-Net, FPN, PSPNet, FCN, SegNet, SAM, DPT, UPerNet, Mask R-CNN |

### 📊 Метрики оценки

```python
# Доступные метрики для бинарной и многоклассовой сегментации
- IoU (Intersection over Union) / Jaccard Index
- Dice Coefficient / F1-Score
- Precision, Recall, Accuracy
- Pixel Accuracy, MAE
- Hausdorff Distance
- Confusion Matrix, Per-class IoU
- Area metrics (difference, ratio)
```

### 🚀 Производительность

- 🔥 **Cold/Hot benchmarking** с warm-up фазами
- ⚡ **CPU vs CUDA** сравнение скорости
- 💾 **Автоматическое освобождение** VRAM между моделями
- 📈 **Детальные отчёты**: CSV, JSON, HTML, LaTeX

---

## 📦 Установка

### Системные требования

- Python 3.12+
- CUDA 121.0+ (опционально, для GPU-ускорения)
- ~20-22 ГБ свободного места для моделей

### Параметры CUDA платформы

```
yamshchikov@rcws-gpu02:~/ML_practice/Torchvision_core_project$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0
```

### Установка зависимостей

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/torchvision_core_project.git
cd torchvision_core_project

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установка базовых зависимостей
pip install -r requirements.txt

# Установка опциональных зависимостей для нейросетей
pip install -r requirements_neural.txt  # transformers, ultralytics, segmentation-models-pytorch
```

### requirements.txt (базовый)
```txt
datasets>=4.4.1
huggingface-hub>=0.36.0
matplotlib>=3.10.7
numpy>=2.2.0
opencv-python>=4.12.0
pandas>=2.3.3
pillow>=8.3.0
pyyaml>=6.0
requests>=2.32.0
scipy>=1.16.0
scikit-learn>=1.8.0
scikit-image>=0.25.0
seaborn>=0.13.0
tabulate>=0.10.0
torch>=2.6.0
torchmetrics>=1.8.2
torchvision>=0.21.0
transformers>=4.57.3
tqdm>=4.67.1
```

---

## 🗂️ Структура проекта

```
torchvision_core_project/
├── main.py                          # Точка входа и демонстрация всех возможностей
├── README.md                        # Документация
├── requirements.txt                 # Зависимости
├── LICENSE                          # Лицензия проекта
├── .gitignore                       # Игнор модулей
├── .gitattributes                   # Гит-атрибуты
├── __init.py__                      # Инициализация
│
├── segmenters/                      # Реализации сегментаторов
│   ├── __init.py__                 # Инициализация
│   ├── BaseSegmenter.py            # Абстрактный базовый класс
│   ├── ModelTrainer.py             # Тренировка изначальных моделей
│   ├── OpenCVSegmenter.py          # Методы на OpenCV
│   ├── SklearnSegmenter.py         # Методы на Scikit-learn
│   ├── TorchSegmenter.py           # Методы на чистом PyTorch
│   ├── NeuralSegmenter.py          # Универсальный нейросетевой сегментатор
│   ├── NeuralModelFactory.py       # Фабрика моделей + YAML-конфиги
│   └── NeuralTrainer.py            # Трейнер для fine-tuning
│
├── testing/                         # Инструменты тестирования
│   ├── SegmentationTester.py       # Тестирование отдельных методов
│   ├── SegmentationComparator.py   # Попарное и матричное сравнение
│   ├── SegmentationBenchmark.py    # Бенчмарк нейросетевых моделей
│   ├── TorchImplementationValidator.py  # Валидация PyTorch-реализаций
│   └── CpuCudaBenchmark.py         # Сравнение CPU vs CUDA
│
├── metrics/                         # Метрики качества
│   └── SegmentationMetrics.py      # Расчёт всех метрик
│
├── inference/                       # Стратегии инференса
│   ├── strategies.py               # Dispatch-функции для разных архитектур
│   ├── utils.py                    # Утилиты анализа логов и предсказаний
│   └── palettes.py                 # Цветовые палитры (ADE20K, COCO, Cityscapes)
│
├── datasets/                        # Загрузчики датасетов
│   ├── LoadDatasets.py             # Загрузка датасетов с HF
│   └── ADE20KDataset.py            # Датасет ADE20K с аугментациями
│
├── utils/                           # Вспомогательные утилиты
│   ├── warmup.py                   # Warm-up для бенчмарков
│   ├── threshold_warmup.py         # Специализированный warm-up
│   └── config.py                   # Управление конфигурацией
│
├── configs/                         # YAML-конфигурации
│   └── neural_models.yaml          # Параметры нейросетевых моделей
├── reports/                         # YAML-конфигурации
│   └── *_report.md                  # Предикты нейросетевых моделей
│
├── data/                            # Выходные данные (генерируется)
│   ├── segmentation_tester_results/
│   ├── validation/
│   ├── ade20k_test_trained/
│   └── ...
│
└── models/                          # Сохранённые чекпоинты (опционально)
    ├── *.pt
    └── *.pth
```

---

## 🚀 Быстрый старт

### 1. Запуск основного теста

```bash
python main.py
```

Проект автоматически:
- Загрузит тестовые изображения
- Инициализирует методы сегментации
- Выполнит бенчмарк производительности
- Проведёт валидацию реализаций
- Сохранит результаты в `./data/`

### 2. Минимальный пример использования

```python
from segmenters.OpenCVSegmenter import OpenCVSegmenter
from segmenters.SklearnSegmenter import SklearnSegmenter
from testing.SegmentationTester import SegmentationTester
from PIL import Image
import numpy as np

# Загрузка изображения
image = Image.open("test.jpg").convert("RGB")
img_array = np.array(image)

# Инициализация тестера
tester = SegmentationTester(base_output_dir="./results")

# Добавление методов
tester.add_method("Otsu_CV2", OpenCVSegmenter("otsu_thresholding"))
tester.add_method("Otsu_Sklearn", SklearnSegmenter("otsu_thresholding"))

# Запуск сравнения
results = tester.compare_methods(
    image=img_array,
    method_names=["Otsu_CV2", "Otsu_Sklearn"],
    save_comparison=True
)

# Вывод результатов
for name, data in results.items():
    print(f"{name}: {data['time']:.3f}s, coverage: {data['mask_percentage']:.1f}%")
```

### 3. Бенчмарк нейросетевой модели

```python
from segmenters.NeuralSegmenter import NeuralSegmenter
from testing.SegmentationBenchmark import SegmentationBenchmark

# Загрузка предобученной модели
segmenter = NeuralSegmenter(
    model_type="segformer",
    model_name="nvidia/segformer-b5-finetuned-ade-640-640",
    num_classes=150
)

# Бенчмарк
benchmark = SegmentationBenchmark(device="cuda", num_classes=150)
benchmark.load_model("segformer_b5", segmenter.model, segmenter.processor, "segformer")

# Запуск инференса
result = benchmark.run_single("test.jpg", "segformer_b5", alpha=0.6)
print(f"IoU: {result['metrics'].get('iou', 'N/A'):.4f}")
```

---

## 🧠 Поддерживаемые модели

### 🔹 Transformer-based (HuggingFace)

| Модель | Ключ | Описание |
|--------|------|----------|
| SegFormer | `segformer` | Efficient Transformer для семантической сегментации |
| Mask2Former | `mask2former` | Универсальная сегментация (semantic/instance/panoptic) |
| OneFormer | `oneformer` | Multi-task универсальный сегментатор |
| DPT | `dpt` | Dense Prediction Transformer |
| UPerNet | `upernet` | CNN + FPN для семантической сегментации |

### 🔹 Torchvision

| Модель | Ключ | Описание |
|--------|------|----------|
| DeepLabV3+ | `deeplab_tv` | Atrous Spatial Pyramid Pooling |
| FCN | `fcn_tv` | Fully Convolutional Networks |
| Mask R-CNN | `maskrcnn_tv` | Instance segmentation (конвертируется в semantic) |

### 🔹 Segmentation Models Pytorch (SMP)

| Архитектура | Encoder | Ключ |
|-------------|---------|------|
| U-Net | ResNet, EfficientNet, MiT | `unet_smp` |
| FPN | ResNet, EfficientNet, MiT | `fpn_smp` |
| PSPNet | ResNet, EfficientNet, MiT | `psp_smp` |
| DeepLabV3+ | ResNet, EfficientNet | `deeplab_smp` |

### 🔹 Промпт-сегментация

| Модель | Ключ | Описание |
|--------|------|----------|
| MobileSAM | `sam` | Легковесная версия Segment Anything |
| SAM 2 | `sam2` | Обновлённая версия SAM с видео-поддержкой |

---

## 📐 Метрики качества

Все метрики рассчитываются через модуль `metrics.SegmentationMetrics`:

```python
from metrics.SegmentationMetrics import SegmentationMetrics

metrics = SegmentationMetrics.calculate_all_metrics(
    pred_mask=prediction,
    gt_mask=ground_truth,
    threshold=0.5,
    include_hausdorff=True
)

print(f"IoU: {metrics['iou']:.4f}")
print(f"Dice: {metrics['dice']:.4f}")
print(f"Hausdorff: {metrics['hausdorff_distance']:.2f}")
```

### Поддерживаемые метрики:

| Метрика | Диапазон | Описание |
|---------|----------|----------|
| `iou` / `jaccard` | [0, 1] | Пересечение над объединением |
| `dice` / `f1_score` | [0, 1] | Гармоническое среднее precision/recall |
| `precision` | [0, 1] | Доля верных положительных предсказаний |
| `recall` | [0, 1] | Полнота обнаружения объектов |
| `pixel_accuracy` | [0, 1] | Доля правильно классифицированных пикселей |
| `mae` | [0, 1] | Средняя абсолютная ошибка |
| `hausdorff_distance` | [0, ∞) | Максимальное расстояние между границами |
| `per_class_iou` | List[float] | IoU для каждого класса отдельно |

---

## ⚙️ Конфигурация

### 📄 Конфигурация нейросетевых моделей (`configs/neural_models.yaml`)

```yaml
models:
  segformer:
    default: b5
    variants:
      b0: "nvidia/segformer-b0-finetuned-ade-512-512"
      b1: "nvidia/segformer-b1-finetuned-ade-512-512"
      b2: "nvidia/segformer-b2-finetuned-ade-512-512"
      b3: "nvidia/segformer-b3-finetuned-ade-640-640"
      b4: "nvidia/segformer-b4-finetuned-ade-640-640"
      b5: "nvidia/segformer-b5-finetuned-ade-640-640"
  
  mask2former:
    default: "facebook/mask2former-swin-base-ade-semantic"
  
  unet:
    encoders: ["resnet34", "resnet50", "efficientnet-b0", "mit_b5"]
    default_encoder: "resnet34"

training:
  ade20k:
    batch_size: 4
    epochs: 20
    lr: 1e-4
    image_size: [512, 512]
    augmentations:
      basic: ["flip", "rotate"]
      medium: ["flip", "rotate", "color_jitter"]
      aggressive: ["flip", "rotate", "color_jitter", "random_crop", "blur"]

metrics:
  threshold: 0.5
  include_hausdorff: true
  ignore_index: 255
```

### 🔧 Использование конфига

```python
from segmenters.NeuralModelFactory import NeuralModelFactory

# Загрузка модели из конфига
model, processor, model_type = NeuralModelFactory.create_model_from_config(
    model_type="segformer",
    variant="b2",  # Берётся из YAML
    device="cuda"
)

# Получение параметров обучения
train_config = NeuralModelFactory.get_training_config("ade20k")
print(f"Batch size: {train_config['batch_size']}")
```

---

## 💡 Примеры

### 📊 Сравнение методов с Ground Truth

```python
from testing.SegmentationTester import SegmentationTester
from metrics.SegmentationMetrics import SegmentationMetrics
import numpy as np

# Загрузка GT
gt_mask = np.load("ground_truth.npy")

# Тестирование с метриками
result = tester.test_single_method_with_metrics(
    image="test.jpg",
    method_name="Otsu_CV2",
    ground_truth=gt_mask,
    output_dir="./results"
)

print(f"IoU: {result['metrics']['iou']:.4f}")
print(f"Time: {result['time']:.3f}s")
```

### 🔄 Валидация PyTorch-реализации против OpenCV

```python
from testing.TorchImplementationValidator import TorchImplementationValidator

validator = TorchImplementationValidator(output_dir="./validation")

# Валидация пороговых методов
results = validator.validate_segmentation_methods(
    image_path="test.jpg",
    methods_list=validator.threshold_methods,
    torch_segmenter_class=TorchSegmenter,
    reference_segmenter_class=OpenCVSegmenter,
    reference="opencv",
    validation_type="threshold"
)

# Генерация отчёта
validator.generate_validation_report({"threshold": results})
```

### 📈 Визуализация результатов бенчмарка

```python
from testing.SegmentationBenchmark import SegmentationBenchmark

benchmark = SegmentationBenchmark(device="cuda")
benchmark.load_segformer("/path/to/model")
benchmark.load_mask2former()

# Запуск сравнения
benchmark.compare(image_input="test.jpg")

# Построение графиков
benchmark.plot_comparison_chart("mIoU", title="IoU Comparison")
benchmark.plot_per_class_iou(top_k=20)
benchmark.plot_confusion_matrix("segformer", normalize='true')

# Экспорт в LaTeX для публикации
latex_table = benchmark.export_latex_table("Results on ADE20K")
print(latex_table)
```

### 🔥 Cold vs Hot бенчмарк

```python
from utils.warmup import SegmentationWarmUp
from testing.SegmentationTester import SegmentationTester

# Инициализация
tester = SegmentationTester(enable_warmup=True)
warmup = SegmentationWarmUp(n_warmup_runs=5)

# Cold run (без warm-up)
cold_results = tester.benchmark_methods(img, n_runs=10, force_warmup=False)

# Warm-up фаза
warmup.warmup_all_segmenters(tester.methods, image=img)

# Hot run (после warm-up)
hot_results = tester.benchmark_methods(img, n_runs=10, force_warmup=False)

# Сравнение
speedup = cold_results['Mean_Time_s'] / hot_results['Mean_Time_s']
print(f"Speedup после warm-up: {speedup.mean():.2f}x")
```

---

## 🤝 Вклад в проект

Мы приветствуем вклад в развитие проекта! 

### Как внести изменения:

1. **Fork** репозиторий
2. Создайте ветку для фичи: `git checkout -b feature/amazing-feature`
3. Внесите изменения и добавьте тесты
4. Закоммитьте: `git commit -m 'Add: amazing feature'`
5. Запушьте: `git push origin feature/amazing-feature`
6. Откройте **Pull Request**

### Стандарты кода:

- Используйте **type hints** для всех функций
- Документируйте публичные методы в **Google-style docstrings**
- Следуйте **PEP 8** для форматирования
- Добавляйте юнит-тесты для новых функций

### Запрос новых функций:

Откройте **Issue** с меткой `enhancement`, описав:
- Какую проблему решает фича
- Предлагаемый API/интерфейс
- Примеры использования

---

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

## 🙏 Благодарности

- [HuggingFace Transformers](https://huggingface.co/transformers/) — предобученные модели
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) — архитектуры сегментации
- [Ultralytics](https://github.com/ultralytics) — реализации SAM
- [ADE20K Dataset](http://sceneparsing.csail.mit.edu/) — датасет для обучения
- [OpenCV](https://opencv.org/), [Scikit-learn](https://scikit-learn.org/) — классические алгоритмы

---

> 💡 **Совет**: Для воспроизводимости результатов фиксируйте версии зависимостей и используйте `torch.use_deterministic_algorithms(True)` при необходимости.

*Последнее обновление: Апрель 2026*