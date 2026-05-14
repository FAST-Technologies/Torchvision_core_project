#!/usr/bin/env python3
import numpy as np, cv2, torch
from segmenters.NewTorchSegmenter import TorchSegmenter2
from segmenters.BackendSegmenters import TRTSegmenter
from utils.backend_exporter import export_method_to_trt_jit, load_trt_model

# 1. Загружаем изображение
img = cv2.imread("test_images/test_image_countryside.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
orig_h, orig_w = img_rgb.shape[:2]
print(f"📸 Original image: {orig_w}x{orig_h}")

# 2. Создаём и экспортируем модель под ФИКСИРОВАННЫЙ размер
seg = TorchSegmenter2(method="otsu_thresholding", device="cuda")
trt_path = "./test_otsu.trt"

# 🔥 Экспортируем под размер входа (512×512)
export_method_to_trt_jit(
    seg,
    "otsu_thresholding",
    trt_path,
    precision="fp32",
    input_shape=(1, 3, 512, 512),  # Фиксировано!
)

# 3. Загружаем и тестируем
trt_model = load_trt_model(trt_path)
trt_seg = TRTSegmenter(
    "otsu_thresholding",
    trt_model,
    device="cuda",
    input_shape=(1, 3, 512, 512),  # Должно совпадать с экспортом!
)

# 4. Запускаем инференс (изображение будет ресайзено к 512×512 внутри)
mask = trt_seg.segment(img_rgb)

# 5. Валидация
print(f"✅ Mask shape: {mask.shape}, dtype: {mask.dtype}")
print(f"✅ Mask min/max: {mask.min()}/{mask.max()}, unique values: {np.unique(mask)}")
print(f"✅ Mask area: {np.sum(mask > 0)} pixels ({np.sum(mask > 0) / mask.size * 100:.2f}%)")

if mask.shape == (orig_h, orig_w) and mask.max() == 255 and np.sum(mask > 0) > 0:
    print("🎉 TRT сегментация работает корректно!")
    cv2.imwrite("test_mask_trt.png", mask)
else:
    print("❌ Что-то пошло не так — проверь логи выше")
