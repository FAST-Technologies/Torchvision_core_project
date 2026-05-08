import onnxruntime as ort
import numpy as np
import torch
from utils.backend_exporter import SegmenterMethodWrapper

sess = ort.InferenceSession("data/backend_comparison/otsu_thresholding.onnx")
inp = np.random.randn(1, 3, 512, 512).astype(np.float32)
out = sess.run(None, {"input": inp})[0]
print(
    f"ONNX output: shape={out.shape}, dtype={out.dtype}, min={out.min()}, max={out.max()}"
)
