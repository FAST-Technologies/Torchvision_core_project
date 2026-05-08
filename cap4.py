# cap4_test_benchmark.py
import torch
import numpy as np
from segmenters.BackendSegmenters import ONNXSegmenter, TRTSegmenter
from testing.SegmentationTester import SegmentationTester
from utils.backend_exporter import load_trt_model

print("🔧 Тест segment_with_mask для ONNX...")
onnx_seg = ONNXSegmenter(
    "otsu_thresholding",
    "./data/backend_comparison/otsu_thresholding.onnx",
    device="cuda",
)
img = np.random.randint(0, 255, (610, 735, 3), dtype=np.uint8)

# Прямой вызов segment_with_mask
result, mask = onnx_seg.segment_with_mask(img)
print(
    f"✅ ONNX segment_with_mask: result shape={result.shape}, mask shape={mask.shape if mask is not None else None}"
)

print("\n🔧 Тест segment_with_mask для TRT...")
trt_model = load_trt_model("./data/backend_comparison/otsu_thresholding_fp32.trt")
trt_seg = TRTSegmenter("otsu_thresholding", trt_model, device="cuda")
result, mask = trt_seg.segment_with_mask(img)
print(
    f"✅ TRT segment_with_mask: result shape={result.shape}, mask shape={mask.shape if mask is not None else None}"
)

print("\n🔧 Тест в SegmentationTester...")
tester = SegmentationTester(enable_warmup=False)
tester.add_method("test_onnx", onnx_seg)
tester.add_method("test_trt", trt_seg)

# Мини-бенчмарк на 1 запуск
df = tester.benchmark_methods(
    image=img, n_runs=1, test_name="quick_test", save_benchmark=False
)
print("\n📊 Результаты бенчмарка:")
print(df[["Method", "Mean_Time_s", "Mask_Area", "Mask_Percentage"]])
