from segmenters.NewTorchSegmenter import TorchSegmenter2
from segmenters.BackendSegmenters import ONNXSegmenter, TRTSegmenter
from utils.backend_exporter import load_trt_model
import numpy as np
import cv2

# Загружаем модель
trt_model = load_trt_model("./test_canny_edge.trt")
seg_trt = TRTSegmenter("canny_edge", trt_model, device="cuda")

# Тестовое изображение 610×735
img = np.random.randint(0, 255, (610, 735, 3), dtype=np.uint8)

# 🔥 Вариант А: Ресайз вручную перед вызовом
img_resized = cv2.resize(img, (512, 512))
mask = seg_trt.segment(img_resized)
print(f"✅ TRT с ресайзом: {mask.shape}")