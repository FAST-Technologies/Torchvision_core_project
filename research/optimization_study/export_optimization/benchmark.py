#!/usr/bin/env python3
"""
Единый скрипт бенчмаркинга экспортированных методов.

Примеры:
    # Базовый запуск с ONNX
    python benchmark.py --methods sobel_edge,global_thresholding --image test.jpg

    # Сравнение всех доступных бэкендов
    python benchmark.py --methods all --backend auto --image test.jpg --output results/

    # Только torch-tensorrt с FP16
    python benchmark.py --methods sobel_edge --backend torch_tensorrt --precision fp16
"""
import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np

# Добавляем корень проекта в path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from segmenters.NewTorchSegmenter import TorchSegmenter
from .backend_registry import get_registry, BackendRegistry
from .utils import save_benchmark_results, format_time, format_speedup


def parse_args():
    parser = argparse.ArgumentParser(
        description="Бенчмарк экспортированных методов сегментации"
    )

    # Входные данные
    parser.add_argument(
        "--image", type=str, required=True, help="Путь к тестовому изображению"
    )
    parser.add_argument(
        "--image-size",
        type=str,
        default="512x512",
        help="Размер для ресайза: 'WxH' (по умолчанию: 512x512)",
    )

    # Методы
    parser.add_argument(
        "--methods",
        type=str,
        default="sobel_edge,global_thresholding",
        help="Список методов через запятую или 'all' для всех",
    )

    # Бэкенд
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "onnx", "torch_tensorrt", "torch2trt"],
        help="Бэкенд экспорта (auto = лучший доступный)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16"],
        help="Точность для TensorRT-бэкендов",
    )

    # Параметры бенчмарка
    parser.add_argument(
        "--n-runs", type=int, default=100, help="Количество запусков для замера"
    )
    parser.add_argument("--n-warmup", type=int, default=10, help="Прогревочные запуски")
    parser.add_argument(
        "--compare-all-providers",
        action="store_true",
        help="Сравнить все доступные ONNX Runtime провайдеры",
    )

    # Вывод
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Папка для сохранения результатов (по умолчанию: не сохранять)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")

    return parser.parse_args()


def main():
    args = parse_args()

    # === Инициализация ===
    print("🔧 Export Optimization Benchmark")
    print("=" * 60)

    # Парсинг размера изображения
    try:
        w, h = map(int, args.image_size.split("x"))
        image_shape = (3, h, w)  # (C, H, W)
    except ValueError:
        print(f"❌ Invalid image size: {args.image_size}. Use 'WxH' format.")
        sys.exit(1)

    # Загрузка сегментера
    print(f"📦 Loading TorchSegmenter...")
    segmenter = TorchSegmenter(device="cuda" if torch.cuda.is_available() else "cpu")

    # Выбор методов
    if args.methods == "all":
        methods = list(segmenter.method_map.keys())
    else:
        methods = [m.strip() for m in args.methods.split(",")]
        # Проверка существования методов
        for m in methods:
            if m not in segmenter.method_map:
                print(f"⚠️ Method '{m}' not found, skipping")
                methods.remove(m)

    if not methods:
        print("❌ No valid methods to benchmark")
        sys.exit(1)

    print(f"🎯 Methods: {', '.join(methods)}")

    # === Выбор бэкенда ===
    registry = get_registry()

    if args.verbose:
        registry.print_status()

    if args.backend == "auto":
        backend_name = registry.get_best_backend()
        print(f"🎯 Auto-selected backend: {backend_name}")
    else:
        backend_name = args.backend
        backend_info = registry.get_backend_info(backend_name)
        if backend_info and not backend_info.available:
            print(f"❌ Backend '{backend_name}' not available:")
            print(f"   {backend_info.notes}")
            sys.exit(1)

    # Получение конвертера
    try:
        converter = registry.get_converter(
            backend_name, segmenter, image_shape=image_shape
        )
    except ImportError as e:
        print(f"❌ Failed to initialize converter: {e}")
        sys.exit(1)

    # === Бенчмарк ===
    results = {}

    for method_name in methods:
        print(f"\n🔬 Benchmarking: {method_name}")
        print("-" * 40)

        try:
            if backend_name == "onnx":
                # Экспорт в ONNX
                onnx_path = f"/tmp/{method_name}.onnx"
                converter.export_method_to_onnx(
                    method_name, onnx_path, verbose=args.verbose
                )

                if args.compare_all_providers:
                    # Бенчмарк на всех провайдерах
                    method_results = converter.benchmark_all_providers(
                        method_name, onnx_path, n_runs=args.n_runs
                    )
                else:
                    # Бенчмарк на лучшем провайдере
                    method_results = converter.benchmark_onnx_vs_torch(
                        method_name,
                        onnx_path,
                        n_runs=args.n_runs,
                        n_warmup=args.n_warmup,
                    )

            elif backend_name == "torch_tensorrt":
                # Компиляция через torch-tensorrt
                compiled = converter.convert_method(
                    method_name, precision=args.precision
                )
                method_results = converter.benchmark(
                    method_name,
                    compiled,
                    n_runs=args.n_runs,
                    n_warmup=args.n_warmup,
                    precision=args.precision,
                )

            elif backend_name == "torch2trt":
                # Конвертация через torch2trt
                method_results = converter.benchmark_trt_vs_torch(
                    method_name,
                    n_runs=args.n_runs,
                    fp16_mode=(args.precision == "fp16"),
                )

            # Сохранение результатов
            results[method_name] = method_results

            # Вывод
            if "speedup" in method_results:
                print(f"   ⚡ Speedup: {method_results['speedup_formatted']}")
            if "torch_mean_ms" in method_results:
                print(
                    f"   🐌 Torch: {format_time(method_results['torch_mean_ms']/1000)}"
                )
            if "onnx_mean_ms" in method_results:
                print(
                    f"   🚀 ONNX:  {format_time(method_results['onnx_mean_ms']/1000)}"
                )
            if "trt_mean_ms" in method_results:
                print(f"   🚀 TRT:   {format_time(method_results['trt_mean_ms']/1000)}")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[method_name] = {"error": str(e)}
            if args.verbose:
                import traceback

                traceback.print_exc()

    # === Сохранение результатов ===
    if args.output:
        output_path = (
            Path(args.output)
            / f"benchmark_{backend_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        save_benchmark_results(results, output_path)
        print(f"\n💾 Results saved: {output_path}")

    # === Сводка ===
    print(f"\n{'='*60}")
    print("📊 Summary:")
    successful = sum(1 for r in results.values() if "error" not in r)
    print(f"   Completed: {successful}/{len(methods)} methods")

    if successful > 0:
        speedups = [
            r["speedup"]
            for r in results.values()
            if "speedup" in r and isinstance(r["speedup"], (int, float))
        ]
        if speedups:
            avg_speedup = np.mean(speedups)
            print(f"   Avg speedup: {format_speedup(avg_speedup)}")
            print(f"   Best: {format_speedup(max(speedups))}")

    print("=" * 60)


if __name__ == "__main__":
    main()
