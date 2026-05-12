import onnx

model = onnx.load("exported_models/onnx/fp16/adaptive_thresholding.onnx")
opset_version = model.opset_import[0].version if len(model.opset_import) > 0 else None

print(opset_version)
