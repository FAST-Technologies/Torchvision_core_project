# 🌐 Torchvision_core_project

> 🇬🇧 English version: [README.en.md](README.en.md)  
> 🇷🇺 Russian version: [README.md](README.md)

> 🎯 **Universal Framework for Comparative Testing and Benchmarking of Image Segmentation Methods**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Links](https://github.com/FAST-Technologies/Torchvision_core_project/actions/workflows/check-links.yml/badge.svg)](https://github.com/FAST-Technologies/Torchvision_core_project/actions/workflows/check-links.yml)

---

## 📋 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Supported Models](#-supported-models)
- [Quality Metrics](#-quality-metrics)
- [Configuration](#-configuration)
- [Performance Optimizations](#-performance-optimizations)
- [Examples](#-examples)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 About

**Torchvision_core_project** is a scalable framework for researching, comparing, and benchmarking image semantic segmentation algorithms. The project unifies classical computer vision methods and modern neural network architectures under a single interface.

### Key Features:

✅ **Unified interface** for 50+ segmentation methods;  
✅ **Library support**: OpenCV, Scikit-Learn, Scikit-Image, PyTorch, Transformers, SMP;  
✅ **Two PyTorch implementations**: `TorchSegmenter` (baseline) and `TorchSegmenter2` (optimized);  
✅ **Precision management**: fp32/fp16/bf16/int8 with automatic device-aware selection;  
✅ **torch.compile support**: graph optimization with configurable modes;  
✅ **Benchmarking**: cold/hot runs, timing, memory, and quality metrics;  
✅ **Model export**: TorchScript, ONNX, TensorRT (JIT/Dynamo);  
✅ **Visualization**: automatic chart generation and reporting;  
✅ **Validation**: cross-library implementation comparison;  
✅ **Training**: fine-tuning models on ADE20K and other datasets;  
✅ **Warm-up utilities**: precise performance measurements.  

---

## ✨ Features

### 🧩 Segmentation Methods

| Category | Methods | TorchSegmenter2 Status |
|----------|---------|----------------------|
| **Thresholding** | Global, Adaptive, Otsu, Niblack, Sauvola, Bernsen, Phansalkar, Kittler-Illingworth, Kapur, Triangle, Multi-Otsu, Percentile, Local Contrast | ✅ Fully optimized |
| **Edge Detection** | Sobel, Canny, Prewitt, Scharr, Laplacian, Roberts, LoG, DoG, Marr-Hildreth, Gradient Magnitude/Direction, Phase Congruency | ✅ Vectorized + NMS |
| **Region-Based** | Region Growing, Split-and-Merge, Flood Fill, Watershed, Random Walker | ✅ BFS + Numba fallback |
| **Clustering** | K-Means, DBSCAN, MeanShift | ⚠️ K-Means optimized, others via sklearn |
| **Active Contours** | Active Contour, GVF, Morphological Snakes, Chan-Vese | ✅ FFT solution + vectorization |
| **Superpixels** | SLIC, Felzenszwalb, QuickShift | ⚠️ Via numpy/scipy (limited optimization) |
| **Interactive** | GrabCut | ✅ GMM on PyTorch |
| **Neural Networks** | SegFormer, Mask2Former, OneFormer, DeepLabV3+, U-Net, FPN, PSPNet, FCN, SegNet, SAM, DPT, UPerNet, Mask R-CNN |

### 📊 Evaluation Metrics

```python
# Available metrics for binary and multi-class segmentation
- IoU (Intersection over Union) / Jaccard Index
- Dice Coefficient / F1-Score
- Precision, Recall, Accuracy, F1-Score
- Pixel Accuracy, MAE (Mean Absolute Error)
- Hausdorff Distance (95th percentile)
- Confusion Matrix, Per-class IoU
- Area metrics (difference, ratio, overlap)
```

### 🚀 Performance

| Feature | Description |
|---------|-------------|
| 🔥 **Cold/Hot benchmarking** | Measurements before/after warm-up with CPU↔GPU transfer detection |
| ⚡ **Precision management** | Automatic fp32/fp16/bf16 selection per device (Ampere+ → bf16) |
| 🔄 **torch.compile** | Graph optimization with `reduce-overhead` / `max-autotune` modes |
| 💾 **VRAM optimization** | Automatic memory cleanup between models |
| 📈 **Detailed reports** | CSV, JSON, HTML, LaTeX + matplotlib/seaborn visualizations |
| 🧪 **Cross-backend export** | ONNX/TensorRT export with result consistency validation |
| ⚡ **CPU vs CUDA** | Speed comparison |

---

## 🌐 Web Interface

The project includes a modern **React + TypeScript** web interface with intuitive controls and real-time result visualization.

### 🔹 Starting the Interface

```bash
# 1. Start the backend (if not already running)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 2. In a separate terminal, start the frontend
cd frontend
npm install
npm run dev
```
Open in browser: `http://localhost:5173`

### 🔹 Main Tabs

| Tab | Description |
|-----|-------------|
| 🖼 **Result** | Preview original, mask, overlay, and boundaries |
| 📊 **Metrics** | IoU, Dice, F1, Precision, Recall, MAE, Hausdorff + confusion matrix |
| 💡 **Recommendations** | Optimal method selection based on image type and goal |
| 🔍 **Analysis** | Intensity histogram, image characteristics, scene examples |
| 📈 **Benchmark** | Cross-model neural network comparison with progress bars and charts |
| 🔬 **Validation** | Compare implementations of the same method in OpenCV / Sklearn / Torch |
| ⚖️ **Comparator** | Detailed pairwise and matrix comparison of classical methods |

### 🔹 Key UI Features

✅ **Interactive progress**: animated progress bars for benchmarks and validation  
✅ **Real-time polling**: automatic status updates without page reload  
✅ **Mask visualization**: overlay, difference, boundaries — all in one interface  
✅ **Export results**: download reports in CSV, JSON, PNG, HTML  
✅ **Responsive design**: proper display on desktop and tablet  
✅ **Preset saving**: configure benchmarks and comparators with save capability  

### 🔹 Benchmark Workflow Example

1. Load an image (or use default from ADE20K)
2. Select models for comparison (or keep defaults)
3. Click **"▶ Run Benchmark"**
4. Monitor progress: model loading → inference → saving
5. Review results: metrics table, IoU/time charts, summary visualization

### 🔹 Cross-Library Validation

```typescript
// Example validation configuration
{
  "primary_library": "torch",
  "reference_library": "opencv", 
  "methods_filter": "threshold",  // threshold | edge | region | clustering
  "image": "test.jpg"
}
```

The interface automatically:
- Runs selected methods in both libraries
- Compares masks by metrics (IoU, Dice, F1, area_ratio)
- Generates coverage charts and difference matrices
- Produces HTML report with visualization

### 🔹 Browser Requirements

- Chrome 120+ / Firefox 115+ / Safari 17+
- WebGL support (for Recharts graphs)
- Screen resolution: ≥ 1280×720 (recommended 1920×1080)

> 💡 **Tip**: For heavy benchmarks, use "Results Only" mode — it disables mask previews and speeds up loading.

### 📸 Interface Gallery

![Main Dashboard](docs/screenshots/main_dashboard.png)
*Intuitive controls: mode, goal, method selection*

![Method Recommendations](docs/screenshots/recomendations.png)
*Method recommendations for specific tasks*

![Analysis](docs/screenshots/analysis.png)
*Image analysis with characteristics and intensity histogram*

![Benchmark in Progress](docs/screenshots/benchmark_progress.png)
*Animated progress with step-by-step details*

![Benchmark Results](docs/screenshots/benchmark_progress_results.png)
*Comparison table and metric charts*

![Validation Process](docs/screenshots/validation_process.png)
![Validation Results](docs/screenshots/validation_results.png)
![Validation Results](docs/screenshots/validation_results_2.png)
![Validation Results](docs/screenshots/validation_results_3.png)
![Validation Results](docs/screenshots/validation_results_4.png)
![Validation Results](docs/screenshots/validation_results_5.png)
*OpenCV vs Torch comparison with metrics and visualization*

![Comparator Results](docs/screenshots/comparator_results.png)
*Summary visualization based on reference method*

---

## 📦 Installation

### System Requirements

- Python 3.12+
- CUDA 12.0+ (optional, for GPU acceleration)
- ~20-22 GB free space for models

### CUDA Platform Parameters

```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
```

### Installing Dependencies

```bash
# Clone repository
git clone https://github.com/yourusername/torchvision_core_project.git
cd torchvision_core_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install base dependencies
pip install -r requirements.txt

# Install optional neural network dependencies
pip install -r requirements_neural.txt  # transformers, ultralytics, segmentation-models-pytorch

# For TensorRT export (optional)
pip install torch-tensorrt  # or use torch2trt
```

### requirements.txt (base)
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
numba>=0.60.0  # For CPU optimizations
```

---

## 🗂️ Project Structure

```
torchvision_core_project/
├── main.py                          # Entry point and feature demonstration
├── README.md                        # Documentation (Russian)
├── README.en.md                     # Documentation (English)
├── requirements.txt                 # Dependencies
├── LICENSE                          # Project license
├── .gitignore                       # Git ignore rules
├── .gitattributes                   # Git attributes
├── __init__.py                      # Package initialization
│
├── segmenters/                      # Segmenter implementations
│   ├── __init__.py                 # Initialization
│   ├── BaseSegmenter.py            # Abstract base class
│   ├── ModelTrainer.py             # Model training
│   ├── OpenCVSegmenter.py          # OpenCV-based methods
│   ├── SklearnSegmenter.py         # Scikit-learn-based methods
│   ├── TorchSegmenter.py           # Pure PyTorch methods (v1)
│   ├── NewTorchSegmenter.py        # Optimized methods (TorchSegmenter2, v2)
│   ├── NeuralSegmenter.py          # Universal neural segmenter
│   ├── NeuralModelFactory.py       # Model factory + YAML configs
│   ├── NeuralTrainer.py            # Fine-tuning trainer
│   └── BackendSegmenters.py        # ONNX/TensorRT wrappers
│
├── testing/                         # Testing tools
│   ├── SegmentationTester.py       # Single method testing
│   ├── SegmentationComparator.py   # Pairwise and matrix comparison
│   ├── SegmentationBenchmark.py    # Neural model benchmarking
│   ├── TorchImplementationValidator.py  # PyTorch implementation validation
│   ├── BatchClassicTester.py       # Batch classical method testing
│   └── CpuCudaBenchmark.py         # CPU vs CUDA comparison
│
├── metrics/                         # Quality metrics
│   └── SegmentationMetrics.py      # All metrics calculation
│
├── inference/                       # Inference strategies
│   ├── strategies.py               # Dispatch functions for different architectures
│   ├── utils.py                    # Log analysis and prediction utilities
│   └── palettes.py                 # Color palettes (ADE20K, COCO, Cityscapes)
│
├── datasets/                        # Dataset loaders
│   ├── LoadDatasets.py             # HF dataset loading
│   └── ADE20KDataset.py            # ADE20K dataset with augmentations
├── reports/                         # YAML configurations
│   └── *_report.md                  # Neural model predictions
│
├── utils/                           # Helper utilities
│   ├── warmup.py                   # Benchmark warm-up
│   ├── threshold_warmup.py         # Specialized warm-up
│   ├── backend_exporter.py         # ONNX/TensorRT export
│   └── config.py                   # Configuration management
│
├── configs/                         # YAML configurations
│   └── neural_models.yaml          # Neural model parameters
│
├── data/                            # Output data (generated)
│   ├── segmentation_tester_results/
│   ├── validation/
│   ├── ade20k_test_trained/
│   ├── backend_comparison/
│   └── ...
│
└── models/                          # Saved checkpoints (optional)
    ├── *.pt
    └── *.pth
```

---

## 🚀 Quick Start

### 1. Run Main Test

#### Basic run
```bash
python main.py
```

#### Debug mode with verbose errors
```bash
DEBUG=1 python main.py
```

#### Type checking
```bash
mypy main.py --ignore-missing-imports
```

#### Documentation check
```bash
pydocstyle main.py --convention=google
```

The project automatically:
- Loads test images
- Initializes segmentation methods (including optimized TorchSegmenter2)
- Runs performance benchmarks (cold/hot)
- Performs implementation validation
- Saves results to `./data/`

### 2. Minimal Usage Example

```python
from segmenters.OpenCVSegmenter import OpenCVSegmenter
from segmenters.SklearnSegmenter import SklearnSegmenter
from segmenters.NewTorchSegmenter import TorchSegmenter2
from testing.SegmentationTester import SegmentationTester
from PIL import Image
import numpy as np
import torch

# Load image
image = Image.open("test.jpg").convert("RGB")
img_array = np.array(image)

# Initialize tester
tester = SegmentationTester(base_output_dir="./results")

# Add methods (including optimized version)
tester.add_method("Otsu_CV2", OpenCVSegmenter("otsu_thresholding"))
tester.add_method("Otsu_Sklearn", SklearnSegmenter("otsu_thresholding"))
tester.add_method("Otsu_Torch_v2", TorchSegmenter2(
    method="otsu_thresholding",
    device="cuda" if torch.cuda.is_available() else "cpu",
    precision="bf16",  # Automatic precision selection
    use_compile=True   # Enable torch.compile
))

# Run comparison
results = tester.compare_methods(
    image=img_array,
    method_names=["Otsu_CV2", "Otsu_Torch_v2", "Otsu_Sklearn"],
    save_comparison=True
)

# Output results
for name, data in results.items():
    print(f"{name}: {data['time']:.3f}s, IoU: {data.get('iou', 'N/A')}")
```

### 3. Benchmark with Precision Management

```python
from segmenters.NewTorchSegmenter import TorchSegmenter2
import torch
import numpy as np

# Automatic precision selection per device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    precision = "bf16" if props.major >= 8 else "fp16" if props.major >= 6 else "fp32"
else:
    precision = "fp32"

# Create optimized segmenter
segmenter = TorchSegmenter2(
    method="sobel_edge",
    device=str(device),
    precision=precision,
    use_compile=True,
    compile_mode="reduce-overhead"
)

# Run segmentation
mask = segmenter.segment(np.array(Image.open("test.jpg")))
print(f"Mask: {mask.shape}, dtype: {mask.dtype}")
```

### 4. Neural Model Benchmark

```python
from segmenters.NeuralSegmenter import NeuralSegmenter
from testing.SegmentationBenchmark import SegmentationBenchmark

# Load pretrained model
segmenter = NeuralSegmenter(
    model_type="segformer",
    model_name="nvidia/segformer-b5-finetuned-ade-640-640",
    num_classes=150
)

# Benchmark
benchmark = SegmentationBenchmark(device="cuda", num_classes=150)
benchmark.load_model("segformer_b5", segmenter.model, segmenter.processor, "segformer")

# Run inference
result = benchmark.run_single("test.jpg", "segformer_b5", alpha=0.6)
print(f"IoU: {result['metrics'].get('iou', 'N/A'):.4f}")
```

---

## 🧠 Supported Models

### 🔹 Transformer-based (HuggingFace)

| Model | Key | Description |
|-------|-----|-------------|
| SegFormer | `segformer` | Efficient Transformer for semantic segmentation |
| Mask2Former | `mask2former` | Universal segmentation (semantic/instance/panoptic) |
| OneFormer | `oneformer` | Multi-task universal segmenter |
| DPT | `dpt` | Dense Prediction Transformer |
| UPerNet | `upernet` | CNN + FPN for semantic segmentation |

### 🔹 Torchvision

| Model | Key | Description |
|-------|-----|-------------|
| DeepLabV3+ | `deeplab_tv` | Atrous Spatial Pyramid Pooling |
| FCN | `fcn_tv` | Fully Convolutional Networks |
| Mask R-CNN | `maskrcnn_tv` | Instance segmentation (converted to semantic) |

### 🔹 Segmentation Models Pytorch (SMP)

| Architecture | Encoder | Key |
|-------------|---------|-----|
| U-Net | ResNet, EfficientNet, MiT | `unet_smp` |
| FPN | ResNet, EfficientNet, MiT | `fpn_smp` |
| PSPNet | ResNet, EfficientNet, MiT | `psp_smp` |
| DeepLabV3+ | ResNet, EfficientNet | `deeplab_smp` |

### 🔹 Prompt Segmentation

| Model | Key | Description |
|-------|-----|-------------|
| MobileSAM | `sam` | Lightweight Segment Anything version |
| SAM 2 | `sam2` | Updated SAM with video support |

---

## 📐 Quality Metrics

All metrics are calculated via `metrics.SegmentationMetrics`:

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

### Supported Metrics:

| Metric | Range | Description |
|--------|-------|-------------|
| `iou` / `jaccard` | [0, 1] | Intersection over Union |
| `dice` / `f1_score` | [0, 1] | Harmonic mean of precision/recall |
| `precision` | [0, 1] | Fraction of correct positive predictions |
| `recall` | [0, 1] | Completeness of object detection |
| `pixel_accuracy` | [0, 1] | Fraction of correctly classified pixels |
| `mae` | [0, 1] | Mean Absolute Error |
| `hausdorff_distance` | [0, ∞) | Maximum distance between boundaries |
| `per_class_iou` | List[float] | IoU for each class separately |

---

## ⚙️ Configuration

### 📄 Neural Model Configuration (`configs/neural_models.yaml`)

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

### 🔧 Using the Config

```python
from segmenters.NeuralModelFactory import NeuralModelFactory

# Load model from config
model, processor, model_type = NeuralModelFactory.create_model_from_config(
    model_type="segformer",
    variant="b2",  # Loaded from YAML
    device="cuda"
)

# Get training parameters
train_config = NeuralModelFactory.get_training_config("ade20k")
print(f"Batch size: {train_config['batch_size']}")
```

---

## ⚡ Performance Optimizations

### 🔹 Precision Management

```python
from segmenters.NewTorchSegmenter import TorchSegmenter2, PrecisionManager
import torch

device = torch.device("cuda")

# Automatic precision selection
precision = PrecisionManager.get_optimal_precision(device)  # 'bf16' for Ampere+

segmenter = TorchSegmenter2(
    method="otsu_thresholding",
    device=str(device),
    precision=precision,  # fp32/fp16/bf16
    use_compile=True
)
```

| Precision | Support | When to Use |
|-----------|---------|-------------|
| `fp32` | All devices | Default, maximum accuracy |
| `fp16` | CUDA ≥ 6.0 (Pascal+) | 1.5-2× speedup, moderate accuracy loss |
| `bf16` | CUDA ≥ 8.0 (Ampere+) | 2-3× speedup, minimal accuracy loss |
| `int8` | CPU (dynamic quantization) | Maximum CPU speedup |

### 🔹 torch.compile Configuration

```python
# Optimal settings for different methods
compile_configs = {
    "global_thresholding": {"fullgraph": True, "dynamic": True, "mode": "reduce-overhead"},
    "canny_edge": {"fullgraph": False, "dynamic": True, "mode": "reduce-overhead"},  # Conditional logic
    "watershed": {"fullgraph": False, "dynamic": True, "mode": "reduce-overhead"},   # heapq
    "quickshift": {"use_compile": False},  # numpy-heavy, compilation won't help
}

segmenter = TorchSegmenter2(
    method="sobel_edge",
    use_compile=True,
    compile_mode="reduce-overhead",  # or "max-autotune" for thorough optimization
    compile_fullgraph=True,  # False if method has conditional logic
    compile_dynamic=True  # Support different image sizes
)
```

### 🔹 Export to ONNX / TensorRT

```python
from utils.backend_exporter import export_method_to_onnx_safe, export_method_to_trt_jit

# Export to ONNX
export_method_to_onnx_safe(
    segmenter, 
    method_name="otsu_thresholding",
    output_path="./exports/otsu.onnx",
    opset_version=17,
    precision="fp16"
)

# Export to TensorRT (via TorchScript)
export_method_to_trt_jit(
    segmenter,
    method_name="otsu_thresholding", 
    output_path="./exports/otsu.trt",
    precision="fp16",
    input_shape=(1, 3, 512, 512),
    min_shape=(1, 3, 256, 256),
    max_shape=(1, 3, 1024, 1024)
)
```

### 🔹 Precision Benchmark

```python
from main import generate_precision_report

# Compare time and quality across precisions
df = generate_precision_report(
    methods=["otsu_thresholding", "sobel_edge"],
    image=np.array(Image.open("test.jpg")),
    output_path="./reports/precision_benchmark.csv",
    n_warmup=3,
    n_runs=10,
    compute_metrics=True  # Compares IoU relative to fp32
)

print(df.pivot_table(index="method", columns="precision", values="mean_time_ms"))
```

---

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# Only fast tests (exclude slow and integration)
pytest tests/ -v -m "not slow and not integration"

# Only TorchSegmenter tests
pytest tests/test_torch_segmenter.py -v

# With code coverage
pytest tests/ --cov=segmenters --cov=metrics --cov-report=html

# Show slowest tests
pytest tests/ --durations=10

# Only GPU tests (requires CUDA)
pytest tests/ -v -m gpu

# Parallel execution (requires pytest-xdist)
pytest tests/ -n auto
```

---

## 💡 Examples

### 📊 Compare Methods with Ground Truth

```python
from testing.SegmentationTester import SegmentationTester
from metrics.SegmentationMetrics import SegmentationMetrics
import numpy as np

# Load GT
gt_mask = np.load("ground_truth.npy")

# Test with metrics
result = tester.test_single_method_with_metrics(
    image="test.jpg",
    method_name="Otsu_CV2",
    ground_truth=gt_mask,
    output_dir="./results"
)

print(f"IoU: {result['metrics']['iou']:.4f}")
print(f"Time: {result['time']:.3f}s")
```

### 🔄 Validate PyTorch Implementation vs OpenCV

```python
from testing.TorchImplementationValidator import TorchImplementationValidator

validator = TorchImplementationValidator(output_dir="./validation")

# Validate thresholding methods
results = validator.validate_segmentation_methods(
    image_path="test.jpg",
    methods_list=validator.threshold_methods,
    torch_segmenter_class=TorchSegmenter2,  # Use optimized version
    reference_segmenter_class=OpenCVSegmenter,
    reference="opencv",
    validation_type="threshold"
)

# Generate report
validator.generate_validation_report({"threshold": results})
```

### 📈 Visualize Benchmark Results

```python
from testing.SegmentationBenchmark import SegmentationBenchmark

benchmark = SegmentationBenchmark(device="cuda")
benchmark.load_segformer("/path/to/model")
benchmark.load_mask2former()

# Run comparison
benchmark.compare(image_input="test.jpg")

# Generate charts
benchmark.plot_comparison_chart("mIoU", title="IoU Comparison")
benchmark.plot_per_class_iou(top_k=20)
benchmark.plot_confusion_matrix("segformer", normalize='true')

# Export to LaTeX for publication
latex_table = benchmark.export_latex_table("Results on ADE20K")
print(latex_table)
```

### 🔥 Cold vs Hot Benchmark

```python
from utils.warmup import SegmentationWarmUp
from testing.SegmentationTester import SegmentationTester

# Initialize
tester = SegmentationTester(enable_warmup=True)
warmup = SegmentationWarmUp(n_warmup_runs=5)

# Cold run (no warm-up)
cold_results = tester.benchmark_methods(img, n_runs=10, force_warmup=False)

# Warm-up phase
warmup.warmup_all_segmenters(tester.methods, image=img)

# Hot run (after warm-up)
hot_results = tester.benchmark_methods(img, n_runs=10, force_warmup=False)

# Compare
speedup = cold_results['Mean_Time_s'] / hot_results['Mean_Time_s']
print(f"Speedup after warm-up: {speedup.mean():.2f}x")
```

### 🚀 Profiling with Transfer Detection

```python
from segmenters.NewTorchSegmenter import TorchSegmenter2

segmenter = TorchSegmenter2(method="canny_edge", device="cuda", precision="bf16")

# Profile with CPU↔GPU transfer detection
profile = segmenter.profile_with_transfer_detection(
    image=np.array(Image.open("test.jpg")),
    n_runs=10,
    detect_transfers=True
)

print(f"Average time: {profile['avg_time_ms']:.2f} ms")
print(f"Memory: {profile['memory_mb']:.1f} MB")

# Warnings about unwanted transfers
if profile.get("transfer_warnings"):
    print("⚠️  Problematic transfers found:")
    for w in profile["transfer_warnings"]:
        print(f"   • {w}")
```

# 📘 Documentation: `BatchNeuralTester.py`

## 📖 Overview
Module for batch testing, profiling, and analysis of neural semantic segmentation models (SMP, TorchVision) trained with different augmentation strategies on the ADE20K dataset. Automates metric calculation, prediction caching, model export, and report generation.

## 🏗️ Architecture and Workflow
1. **CLI parsing** → create `TestConfig`
2. **Checkpoint discovery** in `--models` via pattern `{model}_{aug}_*.pth`
3. **Dataset loading** (local or via HuggingFace Hub)
4. **Testing loop** per model:
   - Load via `NeuralSegmenter`
   - Inference with `--cache` and `--resume` support
   - Calculate metrics (mIoU, Binary IoU, Dice, Boundary F1, per-class stats)
   - Generate overlays (`--class-aware-overlays`)
5. **Aggregation** → `pd.DataFrame` grouped by `(model, augmentation, precision)`
6. **Statistics** → ANOVA, Tukey HSD, summary tables
7. **Export** → CSV, JSON, Markdown, PNG charts, ONNX/TensorRT
8. **Logging** → MLflow (optional)

## 📦 Key Classes and Data Structures

| Class / Object | Purpose |
|----------------|---------|
| `TestConfig` (dataclass) | Centralized experiment configuration. All CLI flags as fields. |
| `ModelCheckpoint` (dataclass) | Checkpoint metadata: path, model type, augmentation level, aggregation key. |
| `TestResult` (dataclass) | Single-image inference result: metrics, time, precision, masks. |
| `PredictionCache` | Disk-based LRU prediction cache. Keys generated via SHA256 from `(mtime_ckpt, img_path, config_hash)`. |
| `BatchNeuralTester` | Main orchestrator. Manages loading, inference, profiling, export, and visualization. |

## 🛠️ Key `BatchNeuralTester` Methods

| Method | Description |
|--------|-------------|
| `__init__(config)` | Initialize cache, experiment trackers, precision manager. |
| `_find_checkpoints()` | Find `.pth` files in `config.models_dir`. Group by `(model_type, aug_level)`. |
| `_load_ade20k_images()` | Load `(image, mask)` pairs. Support local path and HF Hub. |
| `_calculate_multiclass_iou()` | Calculate mIoU and per-class IoU with `ignore_index=255`. |
| `_calculate_binary_metrics()` | Binary metrics (IoU, Dice, Precision, Recall, F1, MAE, Hausdorff). |
| `_calculate_comprehensive_metrics()` | Extended metrics: per-class Dice/Precision/Recall + Boundary F1 (dilation⊕erosion). |
| `_test_single_model()` | Main inference loop for single checkpoint. Supports cache, resume, fp16/autocast. |
| `_profile_model_inference()` | Profiling via `torch.profiler`. Export Chrome Trace and stacks. |
| `_export_model_to_onnx_trt()` | Export to ONNX (with `export_params=False` fallback) and TensorRT (`ir="ts"`). |
| `run()` | Run full pipeline. Returns `pd.DataFrame` with results. |
| `aggregate_metrics()` | Group by `(model, aug, precision)`. Calculate `mean/std/min/max`. |
| `statistical_analysis()` | ANOVA on augmentations, Tukey HSD post-hoc tests, find best combination. |
| `export_results()` | Save CSV, JSON, Markdown report, overlays, charts. |
| `plot_results()` / `plot_detailed_results()` | Visualization: bar charts, gain heatmaps, box/swarm plots. |

## 💻 CLI Flags (Current State)

### 🔹 Basic
| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `./data/ADE20K` | Dataset path or HF Hub ID |
| `--models` | `./models` | Directory with `.pth` checkpoints |
| `--subset` | `50` | Number of images (`0` = full dataset) |
| `--output` | `./results/augmentation_analysis` | Output folder |
| `--seed` | `42` | Random seed for reproducibility |
| `--verbose` | `True` | Verbose logging |

### ⚡ Performance
| Flag | Default | Description |
|------|---------|-------------|
| `--precision` | `fp32` | Inference precision: `fp32`, `fp16`, `bf16` |
| `--device` | `cuda` | Device: `cuda` or `cpu` |
| `--cache` | `False` | Enable prediction caching |
| `--cache-dir` | `./cache/predictions` | Cache path |
| `--cache-max-gb` | `10.0` | Cache size limit in GB |
| `--clear-cache` | `False` | Clear cache before run |
| `--resume` | `False` | Skip processed `(model, image)` pairs |
| `--batch-size` | `1` | Batch size (for batch inference) |

### 📦 Export and Profiling
| Flag | Default | Description |
|------|---------|-------------|
| `--export-onnx` | `False` | Export to ONNX |
| `--export-trt` | `False` | Compile to TensorRT |
| `--trt-precision` | `fp16` | TRT precision: `fp32`, `fp16` |
| `--opset` | `17` | ONNX opset version |
| `--dynamic-shapes` | `False` | Dynamic shapes in ONNX |
| `--profile` | `False` | Enable `torch.profiler` |
| `--profile-output` | `./profiling` | Folder for trace files |
| `--profile-warmup` | `10` | Warm-up iterations |
| `--profile-runs` | `50` | Profiling iterations |

### 📊 Metrics and Tracking
| Flag | Default | Description |
|------|---------|-------------|
| `--compute-boundary-f1` | `False` | Calculate Boundary F1 (slow) |
| `--per-class-metrics` | `False` | Save per-class Precision/Recall/IoU |
| `--use-mlflow` | `False` | Log metrics to MLflow |
| `--use-wandb` | `False` | Log to Weights & Biases *(deferred)* |

### 🎨 Visualization
| Flag | Default | Description |
|------|---------|-------------|
| `--save-viz` | `False` | Save overlays |
| `--class-aware-overlays` | `False` | Color overlays with class legend |
| `--overlay-alpha` | `0.5` | Overlay transparency |
| `--border-thickness` | `2` | Border thickness in overlays |

## 📁 Output File Structure
```text
{output_dir}/
├── detailed_results.csv          # All per-image results
├── aggregated_metrics.csv        # Aggregated metrics (mean/std/min/max)
├── statistical_analysis.json     # ANOVA, Tukey HSD, best combination
├── report.md                     # Markdown report with tables and gains
├── plots/                        # Charts
│   ├── miou_comparison.png
│   ├── gain_heatmap.png
│   ├── miou_distribution.png
│   ├── inference_time.png
│   └── augmentation_gain.png
├── overlays/                     # Visualizations
│   ├── comparison_{model}.png
│   └── full_comparison_grid.png
├── exports/                      # Exported models
│   ├── {model_key}.onnx
│   └── {model_key}.{trt_precision}.trt
└── .completed.json               # Task status (for --resume)
```

## 🔧 Ready-to-Use `docstring` Templates (for code embedding)

```python
class BatchNeuralTester:
    """
    Orchestrator for batch testing of segmentation models.
    
    Supports:
    - Multi-class and binary metrics (mIoU, Dice, Boundary F1)
    - Prediction caching and resume of interrupted runs
    - Inference profiling (CPU/CUDA time, memory, FLOPs)
    - ONNX and TensorRT export with fallback mechanisms
    - MLflow integration
    - Report generation (CSV, JSON, Markdown, PNG)
    """

    def _find_checkpoints(self, models_dir: Optional[PathLike] = None, ...) -> Dict[str, ModelCheckpoint]:
        """
        Find checkpoints by pattern {model_type}_{aug_level}_*.pth.
        
        Args:
            models_dir: Path to weights directory. If None, taken from config.
            model_types: List of architectures to search.
            augmentation_levels: List of augmentation levels.
            
        Returns:
            Dict[str, ModelCheckpoint]: Mapping "{model}_{aug}" → checkpoint data.
        """

    def _test_single_model(self, checkpoint: ModelCheckpoint, image_pairs: List[Tuple[Path, Path]], ...) -> List[TestResult]:
        """
        Run inference for one model on a set of images.
        
        Supports:
        - Automatic fallback to fp32 if fp16/bf16 unsupported
        - Load predictions from cache
        - Skip processed pairs with --resume
        - Calculate comprehensive metrics and generate overlays
        
        Returns:
            List[TestResult]: List of results per image.
        """

    def run(self) -> pd.DataFrame:
        """
        Run full testing pipeline: checkpoint discovery → data loading → inference → aggregation.
        
        Returns:
            pd.DataFrame: Table with metrics, inference time, and metadata.
        """

    def export_results(self, df: pd.DataFrame, aggregated: pd.DataFrame, stats: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
        """
        Export results to various formats.
        
        Returns:
            Dict[str, Path]: Mapping of artifact name → file path.
        """
```

## ⚠️ Known Limitations and Notes
| Feature | Status | Note |
|---------|--------|--------|
| `--use-wandb` | ⏸️ Deferred | Requires `api_key` check refinement and offline fallback. |
| `--export-trt` | ⚠️ Version-dependent | `ValueError: Unknown ir was requested` in newer `torch-tensorrt`. Using `ir="ts"` as temporary workaround. |
| `--precision fp16` on CPU | 🔄 Auto-fallback | PyTorch doesn't support fp16 inference on CPU. Script auto-fallbacks to `fp32` with warning. |
| `--resume` | ✅ Works | Status stored in `output_dir/.completed.json`. Don't change `--dataset`, `--subset`, or `--models` between `--resume` runs. |

## 🧩 Main Classes and Functions

### `PredictionCache`
```python
"""
Model prediction cache for accelerating repeated runs.

Uses disk storage with LRU eviction policy.
Keys generated based on checkpoint mtime, image path, and config hash.

Args:
    cache_dir: Path to directory for storing `.pkl` files.
    max_size_gb: Maximum cache size in gigabytes.

Returns:
    PredictionCache: Initialized cache object.

Note:
    - When exceeding `max_size_gb`, oldest files by `st_mtime` are deleted.
    - Corrupted pickle files are auto-deleted on read.
"""
```

| Method | Description |
|--------|-------------|
| `_get_key()` | Generate 16-char hex key from `(mtime, image_path, config_hash)`. |
| `get()` | Load prediction from `.pkl`. Returns `None` if missing/error. |
| `set()` | Save array. Auto-cleans old files when limit exceeded. |
| `clear()` | Full cache directory cleanup. Returns deleted file count. |

---

### `BatchNeuralTester`
```python
"""
Orchestrator for batch testing of segmentation models.

Manages data loading, inference, metric calculation,
caching, profiling, export, and visualization.

Args:
    config: TestConfig object with run parameters.
"""
```

| Method | Purpose | Key Features |
|--------|---------|-------------|
| `__init__()` | Initialization | Creates `PredictionCache`, initializes `mlflow`/`wandb`. |
| `_find_checkpoints()` | Find `.pth` | Groups by `(model, aug)`, takes newest by `ctime`. |
| `_load_ade20k_images()` | Load data | Support local paths and HF Hub. Random subset. |
| `_resize_mask()` | Resize GT | `scipy.ndimage.zoom` with `order=0` to preserve integer labels. |
| `_calculate_multiclass_iou()` | mIoU | Calculate on valid pixels (`ignore_index=255`). |
| `_calculate_binary_metrics()` | Binary metrics | IoU, Dice, F1, Precision, Recall, MAE, Hausdorff. |
| `_calculate_comprehensive_metrics()` | Extended metrics | Per-class stats + Boundary F1 (dilation⊕erosion). |
| `_test_single_model()` | Inference loop | `autocast`, `torch.no_grad()`, cache, resume, visualization fallback. |
| `_profile_model_inference()` | Profiling | `torch.profiler` → Chrome Trace, FLOPs, CPU/CUDA time. |
| `_export_model_to_onnx_trt()` | Export | Fallback `export_params=False`, `ir="ts"` for TRT, cleanup CUDA→CPU. |
| `run()` | Main pipeline | Coordinates all steps. Returns `pd.DataFrame`. |
| `aggregate_metrics()` | Aggregation | Group by `(model, aug, precision)`. `mean/std/min/max`. |
| `statistical_analysis()` | Statistics | ANOVA, Tukey HSD, find best combinations. |
| `export_results()` | Save artifacts | CSV, JSON, Markdown report, PNG, overlays. |
| `plot_results()` / `plot_detailed_results()` | Visualization | Bar charts, gain heatmaps, box/swarm plots, baseline-relative gains. |

---

## 🛠️ Helper Functions

| Function | Description |
|----------|-------------|
| `extract_model_aug_from_key()` | Parse `"{model}_{aug}_{img}"` → `(model, aug)`. Searches known prefixes, fallback by suffix. |
| `safe_inference_context()` | Context manager. Catches OOM/errors, cleans memory, logs details. |
| `_check_precision_support()` | Check `dtype/device` compatibility. bf16 → Ampere+, fp16 on CPU → `False`. |
| `_resolve_torch_dtype()` | Map `"fp16"/"bf16"/"fp32"` → `torch.dtype`. |
| `ensure_pil_compatible()` | Normalize `[0,255]`, type conversion, ensure 3 RGB channels. |
| `save_augmentation_comparison_grid()` | Unified grid: rows=models, columns=`[none, basic, medium]`. |
| `save_model_augmentation_comparisons()` | Separate PNGs with 3 columns per model. |

---

## 📁 Output File Structure

```text
{output_dir}/
├── detailed_results.csv          # All per-image results
├── aggregated_metrics.csv        # Aggregation (mean/std/min/max)
├── statistical_analysis.json     # ANOVA, Tukey HSD, best combinations
├── report.md                     # Markdown report with tables
├── plots/                        # Charts
│   ├── miou_comparison.png
│   ├── gain_heatmap.png
│   ├── miou_distribution.png
│   ├── miou_distribution_swarm.png
│   ├── inference_time.png
│   └── augmentation_gain.png
├── overlays/                     # Visualizations
│   ├── comparison_{model}.png
│   └── full_comparison_grid.png
├── exports/                      # Exported models
│   ├── {model_key}.onnx
│   └── {model_key}.{fp16|fp32}.trt
├── profiling/                    # Profiling results
│   ├── trace_{model}.json        # Chrome Trace
│   └── stacks_{model}.txt
└── .completed.json               # Status for --resume
```

---

## 🚀 CLI Usage Examples

```bash
# 🔹 Basic run
python BatchNeuralTester.py --dataset ./data/ADE20K --subset 50 --output ./results

# 🔹 With caching and resume
python BatchNeuralTester.py --cache --resume --output ./results

# 🔹 Inference profiling
python BatchNeuralTester.py --profile --profile-output ./profiling

# 🔹 Export to ONNX
python BatchNeuralTester.py --export-onnx --opset 18

# 🔹 Export to ONNX + TensorRT
python BatchNeuralTester.py --export-onnx --export-trt --trt-precision fp16

# 🔹 Multi-class metrics + boundary F1
python BatchNeuralTester.py --compute-boundary-f1 --per-class-metrics

# 🔹 Test from custom model folder
python BatchNeuralTester.py --models ./my_checkpoints --subset 5

# 🔹 Reproducible experiment
python BatchNeuralTester.py --seed 42 --output ./exp_v1
python BatchNeuralTester.py --seed 42 --output ./exp_v1_retry  # same data

# 🔹 Run on CPU (for debugging)
python BatchNeuralTester.py --device cpu --precision fp32 --subset 1

# 🔹 MLflow / Weights & Biases integration
python BatchNeuralTester.py --use-mlflow
python BatchNeuralTester.py --use-wandb  # requires wandb login

# 🔹 Visualization with class legends
python BatchNeuralTester.py --class-aware-overlays --overlay-alpha 0.6 --save-viz
```

---

## ⚠️ Known Limitations and Notes

| Feature | Status | Note |
|---------|--------|--------|
| `--use-wandb` | ⏸️ Deferred | Auto-fallback to offline mode. Requires `wandb login` for online. |
| `--export-trt` | ⚠️ Workaround | Uses `ir="ts"` (TorchScript) due to `ValueError: Unknown ir` bug in newer `torch-tensorrt`. |
| `--precision fp16` on CPU | 🔄 Auto-fallback | Script auto-fallbacks to `fp32` with warning. |
| `--resume` | ✅ Works | Status stored in `output_dir/.completed.json`. Don't change `--dataset`/`--models` between runs. |
| `--per-class-metrics` | ✅ Works | Limited to first 20 classes to save RAM/CSV size. |

---

## 🛠️ How to Generate Auto-Documentation

To create HTML/PDF documentation from code:

```bash
pip install pdoc  # or mkdocs, sphinx
pdoc BatchNeuralTester.py --output-dir ./docs
# Open ./docs/index.html in browser
```
---

## 🐳 Running in Docker

```bash
# Build image
docker build -t torchvision-core .

# Run with GPU
docker run --gpus all -v ./data:/app/data torchvision-core python main.py

# Or via docker-compose
docker-compose up
```

### 📄 Dockerfile (example):

```dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.0-cudnn8-runtime
# ... dependencies, code copy, entrypoint
```

## ❓ Common Issues

### ❌ "CUDA out of memory"
```python
# Solution: reduce batch_size or use gradient accumulation
# Or switch precision: precision="bf16"
```

### ❌ "ModuleNotFoundError: torch_tensorrt"
```bash
# TensorRT is optional dependency
pip install torch-tensorrt  # or skip TRT export
```

### ❌ "ONNX export failed for method X"
```python
# Some methods contain dynamic control flow
# Use export_method_to_onnx_safe() with fallback
```

## ⏱️ Expected Execution Times

| Method | Image Size | Time (CPU) | Time (CUDA, bf16) |
|--------|-----------|------------|-------------------|
| otsu_thresholding | 512×512 | ~15 ms | ~2 ms |
| canny_edge | 512×512 | ~25 ms | ~4 ms |
| chan_vese | 512×512 | ~400 ms | ~45 ms |
| segformer-b5 | 512×512 | N/A | ~120 ms |

> ⚠️ Numbers are approximate and depend on hardware and system load.


## 🔐 Privacy and Security

- All computations run locally; no data is sent to the cloud
- Models are loaded from trusted sources (HuggingFace, PyTorch Hub)
- When using `--use-mlflow`, metrics are logged locally by default

## 🧪 Testing and Validation

The project includes a comprehensive testing system to ensure quality, correctness, and performance of all components.

### 🔹 Test Types

| Type | Description | Run Command |
|------|-------------|-------------|
| **Unit tests** | Test individual functions and classes | `pytest tests/unit/` |
| **Integration** | Test full segmentation pipeline | `pytest tests/integration/ -m "not slow"` |
| **Benchmarks** | Measure time, memory, accuracy (CPU/CUDA) | `pytest tests/benchmarks/ -m benchmark` |
| **Implementation validation** | Compare Torch/OpenCV/Sklearn method versions | `python -m testing.TorchImplementationValidator` |
| **Ground Truth evaluation** | Metrics (IoU, Dice, F1) against labeled data | `pytest tests/gt_evaluation/` |
| **Export tests** | Validate ONNX/TensorRT export | `pytest tests/export/ -m export` |

### 🔹 Quick Start

```bash
# Run all tests (excluding slow)
pytest -m "not slow"

# Only unit tests with coverage
pytest tests/unit/ --cov=segmenters --cov-report=html

# Performance benchmark on specific image
python -m testing.SegmentationBenchmark --image path/to/image.jpg

# Validate Torch vs OpenCV consistency
python -m testing.TorchImplementationValidator --image path/to/image.jpg
```

### 🔹 pytest Markers

```bash
# Skip slow tests
pytest -m "not slow"

# Only GPU tests (requires CUDA)
pytest -m gpu

# Only integration tests
pytest -m integration

# Combine: fast + non-GPU
pytest -m "not slow and not gpu"
```

### 🔹 Quality Metrics

When running tests with Ground Truth, the following are automatically calculated:

- **IoU (Intersection over Union)** — primary segmentation quality metric
- **Dice Coefficient** — overlap measure between prediction and GT
- **Precision / Recall / F1-Score** — accuracy/completeness balance
- **Pixel Accuracy** — fraction of correctly classified pixels
- **Hausdorff Distance** — boundary distance (for medical tasks)

Results are saved to `./data/reports/` in CSV, JSON, and Markdown formats.

### 🔹 Profiling and Debugging

```bash
# Profile method execution time
python -c "from segmenters.TorchSegmenter2 import TorchSegmenter2; \
           s = TorchSegmenter2('otsu_thresholding'); \
           s.profile_method('image.jpg', n_runs=100)"

# Detailed trace for Chrome DevTools
python -m testing.TorchImplementationValidator --profile --output ./profiling/

# Compare precisions (fp32/fp16/bf16)
python -m testing.CpuCudaBenchmark --precisions fp32 fp16 bf16
```

### 🔹 CI/CD Integration

GitHub Actions configuration (`./.github/workflows/test.yml`) includes:

- ✅ Run tests on Python 3.13
- ✅ Type checking via mypy
- ✅ Linting via ruff/black
- ✅ Coverage collection (requires ≥80%)
- ✅ Optional GPU tests (if runner has CUDA)

```yaml
# Example workflow step
- name: Run tests
  run: |
    pytest -m "not slow and not gpu" --cov=segmenters --cov-fail-under=80
```

> 💡 **Tip**: For local debugging, use `--pdb` flag to enter interactive debugger on test failure:
> ```bash
> pytest tests/unit/test_thresholding.py::test_otsu --pdb
> ```

---


## 🤝 Contributing

We welcome contributions to the project!

### How to contribute:

1. **Fork** the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Commit: `git commit -m 'Add: amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open a **Pull Request**

### Code Standards:

- Use **type hints** for all functions
- Document public methods with **Google-style docstrings**
- Follow **PEP 8** for formatting
- Add unit tests for new features
- For optimizations: use `@torch.no_grad()` and `autocast` where applicable

### Requesting New Features:

Open an **Issue** with the `enhancement` label, describing:
- What problem the feature solves
- Proposed API/interface
- Usage examples
- Expected performance impact

---

## 📄 License

The project is distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

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

## 🙏 Acknowledgments

- [HuggingFace Transformers](https://huggingface.co/transformers/) — pretrained models
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) — segmentation architectures
- [Ultralytics](https://github.com/ultralytics) — SAM implementations
- [OpenCV](https://opencv.org/), [Scikit-learn](https://scikit-learn.org/) — classical algorithms
- [Numba](https://numba.pydata.org/) — JIT compilation for CPU optimizations

---

> 💡 **Tip**: For reproducible results, pin dependency versions and use `torch.use_deterministic_algorithms(True)` when needed. For maximum GPU performance, use `bf16` on Ampere+ and `fp16` on older architectures.

*Last updated: May 2026*
```

---