def check_export_compatibility():
    import torch
    import torch_tensorrt

    print(f"PyTorch version: {torch.__version__}")
    print(f"torch-tensorrt version: {torch_tensorrt.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    # Проверка поддержки Dynamo IR
    try:
        from torch_tensorrt.dynamo import compile as dynamo_compile

        print("✅ torch_tensorrt.dynamo available")
    except ImportError:
        print("❌ torch_tensorrt.dynamo NOT available - update torch-tensorrt")


# Запустить перед экспортом
check_export_compatibility()
