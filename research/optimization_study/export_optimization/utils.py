"""
Вспомогательные утилиты для модуля экспорта.
"""

import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch


def get_available_providers() -> List[str]:
    """
    Возвращает доступные ONNX Runtime провайдеры.

    Returns:
        Список доступных провайдеров: ['CUDAExecutionProvider', ...]
    """
    try:
        import onnxruntime as ort

        return ort.get_available_providers()
    except ImportError:
        return ["CPUExecutionProvider"]


def get_available_tensorrt_backends() -> Dict[str, bool]:
    """
    Проверяет доступность TensorRT-бэкендов.

    Returns:
        Dict с флагами доступности:
        {
            "tensorrt": bool,      # tensorrt package
            "torch_tensorrt": bool, # torch-tensorrt package
            "torch2trt": bool,      # torch2trt package
        }
    """
    backends = {}

    # Проверка tensorrt
    try:
        import tensorrt as trt

        backends["tensorrt"] = True
        backends["tensorrt_version"] = trt.__version__
    except ImportError:
        backends["tensorrt"] = False
        backends["tensorrt_version"] = None

    # Проверка torch-tensorrt
    try:
        import torch_tensorrt

        backends["torch_tensorrt"] = True
        backends["torch_tensorrt_version"] = torch_tensorrt.__version__
    except ImportError:
        backends["torch_tensorrt"] = False
        backends["torch_tensorrt_version"] = None

    # Проверка torch2trt
    try:
        from torch2trt import torch2trt

        backends["torch2trt"] = True
    except ImportError:
        backends["torch2trt"] = False

    return backends


def format_time(seconds: float) -> str:
    """Форматирует время в человекочитаемый вид."""
    if seconds < 0.001:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.3f} ms"
    else:
        return f"{seconds:.3f} s"


def format_speedup(ratio: float) -> str:
    """Форматирует коэффициент ускорения."""
    if ratio >= 10:
        return f"{ratio:.1f}×"
    elif ratio >= 2:
        return f"{ratio:.2f}×"
    else:
        return f"{ratio:.3f}×"


def save_benchmark_results(results: Dict[str, Dict], output_path: Union[str, Path], format: str = "json") -> Path:
    """
    Сохраняет результаты бенчмарка в файл.

    Args:
        results: Словарь с результатами
        output_path: Путь к файлу
        format: Формат ('json' или 'csv')

    Returns:
        Path к сохранённому файлу
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        if output_path.suffix != ".json":
            output_path = output_path.with_suffix(".json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    elif format == "csv":
        if output_path.suffix != ".csv":
            output_path = output_path.with_suffix(".csv")

        if not results:
            output_path.write_text("")
            return output_path

        # Получаем все возможные ключи из всех результатов
        all_keys = set()
        for method_data in results.values():
            all_keys.update(method_data.keys())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["method"] + sorted(all_keys))
            writer.writeheader()
            for method, data in results.items():
                row = {"method": method, **data}
                writer.writerow(row)

    print(f"💾 Результаты сохранены: {output_path}")
    return output_path


def warmup_inference(func, input_tensor, n_warmup: int = 10):
    """Прогрев инференса для стабильных замеров."""
    for _ in range(n_warmup):
        _ = func(input_tensor)
        if input_tensor.device.type == "cuda":
            torch.cuda.synchronize()


def measure_inference(func, input_tensor, n_runs: int = 100, sync_cuda: bool = True) -> Dict[str, float]:
    """
    Замер времени инференса.

    Returns:
        Dict со статистикой: mean, std, min, max, p95
    """
    times = []

    with torch.inference_mode():
        for _ in range(n_runs):
            if sync_cuda and input_tensor.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = func(input_tensor)

            if sync_cuda and input_tensor.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    times_np = np.array(times)
    return {
        "mean_ms": float(np.mean(times_np)),
        "std_ms": float(np.std(times_np)),
        "min_ms": float(np.min(times_np)),
        "max_ms": float(np.max(times_np)),
        "p50_ms": float(np.percentile(times_np, 50)),
        "p95_ms": float(np.percentile(times_np, 95)),
        "p99_ms": float(np.percentile(times_np, 99)),
    }
