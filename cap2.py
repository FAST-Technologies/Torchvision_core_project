# test_trt_jit.py
from segmenters.NewTorchSegmenter import TorchSegmenter2
from utils.backend_exporter import export_method_to_trt_jit

methods = ["otsu_thresholding", "sobel_edge", "canny_edge"]
seg_base = TorchSegmenter2(method="otsu_thresholding", device="cuda", precision="fp32")

print("🔹 TRT JIT export для всех методов...")
for method in methods:
    print(f"\n--- {method} ---")
    seg = TorchSegmenter2(method=method, device="cuda", precision="fp32")
    ok = export_method_to_trt_jit(
        seg, 
        method, 
        f"./test_{method}.trt",
        input_shape=(1, 3, 512, 512)
    )
    print(f"{method}: {'✅' if ok else '❌'}")

print("\n🎉 Готово!")