#!/usr/bin/env python3
"""
CLI-скрипт для бенчмаркинга kernel fusion.
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import torch

from .config import FusionConfig, FusionStrategy
from .fusion_optimizer import FusionOptimizer
from .graph_profiler import GraphProfiler
from .report_generator import FusionReportGenerator
from .visualization import FusionVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="⚡ Kernel Fusion Benchmark — исследование оптимизации через fusion"
    )

    # Выбор методов
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Методы для тестирования (через запятую или 'all')",
    )

    # Стратегия fusion
    parser.add_argument(
        "--strategy",
        type=str,
        default="graph",
        choices=["none", "graph", "manual", "custom", "vectorized"],
        help="Стратегия fusion",
    )

    # Параметры компиляции
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Режим torch.compile",
    )

    # Устройство
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Устройство для бенчмарка",
    )

    # Параметры бенчмарка
    parser.add_argument(
        "--n-runs", type=int, default=50, help="Количество запусков для замера"
    )
    parser.add_argument(
        "--input-size",
        type=str,
        default="512x512",
        help="Размер тестового изображения: 'WxH'",
    )

    # Вывод
    parser.add_argument(
        "--output", type=str, default=None, help="Папка для сохранения результатов"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")
    parser.add_argument(
        "--profile-graph", action="store_true", help="Профилировать граф вычислений"
    )

    # Визуализация
    parser.add_argument("--plot", action="store_true", help="Сгенерировать графики")

    return parser.parse_args()


def main():
    args = parse_args()

    # === Инициализация ===
    print("⚡ Kernel Fusion Benchmark")
    print("=" * 60)

    # Парсинг размера изображения
    try:
        w, h = map(int, args.input_size.split("x"))
        example_input = torch.randn(1, 1, h, w, device=args.device)
    except ValueError:
        print(f"❌ Invalid input size: {args.input_size}. Use 'WxH' format.")
        sys.exit(1)

    # Инициализация сегментера
    print(f"📦 Loading TorchSegmenter...")
    from segmenters.TorchSegmenter import TorchSegmenter

    segmenter = TorchSegmenter(device=args.device)

    # Выбор методов
    if args.methods == "all":
        methods = list(segmenter.method_map.keys())
    else:
        methods = [m.strip() for m in args.methods.split(",")]

    # Конфигурация
    strategy_map = {
        "none": FusionStrategy.NONE,
        "graph": FusionStrategy.GRAPH_FUSION,
        "manual": FusionStrategy.MANUAL_FUSION,
        "custom": FusionStrategy.CUSTOM_KERNEL,
        "vectorized": FusionStrategy.VECTORIZED,
    }

    config = FusionConfig(
        strategy=strategy_map[args.strategy],
        compile_mode=args.compile_mode,
        profile_graph=args.profile_graph,
        verbose=args.verbose,
        output_dir=args.output or "./results/kernel_fusion",
    )

    # === Оптимизация ===
    optimizer = FusionOptimizer(segmenter, config, device=args.device)

    if args.profile_graph:
        profiler = GraphProfiler(verbose=args.verbose)
        for method in methods[:5]:  # Ограничим для скорости
            if method in segmenter.method_map:
                profiler.profile_function(
                    segmenter.method_map[method],
                    example_input,
                    method_name=method,
                )

    # Fusion методов
    fused_ops = optimizer.fuse_all_methods(methods, example_input)

    # === Бенчмарк ===
    print(f"\n🔬 Running benchmarks...")
    results = []

    for method_name, fused_op in fused_ops.items():
        try:
            result = optimizer.benchmark_fusion(
                method_name,
                fused_op,
                example_input,
                n_runs=args.n_runs,
            )
            results.append(result)

            if args.verbose:
                print(
                    f"   {method_name:30s}: "
                    f"{result['speedup_formatted']} speedup "
                    f"({result['original_mean_ms']:.3f} → {result['fused_mean_ms']:.3f} ms)"
                )

        except Exception as e:
            print(f"   ❌ {method_name}: {e}")

    # === Отчёт ===
    if args.output:
        reporter = FusionReportGenerator(output_dir=args.output)
        report_path = reporter.generate_markdown(results, config)
        print(f"📄 Report saved: {report_path}")

    # === Визуализация ===
    if args.plot:
        viz = FusionVisualizer(output_dir=args.output or "./plots")
        plots = viz.plot_all(results)
        print(f"📊 Plots saved: {list(plots.values())}")

    # === Сводка ===
    print(f"\n{'='*60}")
    print("📊 Summary:")

    if results:
        speedups = [r["speedup"] for r in results if "speedup" in r]
        if speedups:
            avg_speedup = np.mean(speedups)
            max_speedup = np.max(speedups)
            print(f"   Avg speedup: {format_speedup(avg_speedup)}")
            print(f"   Max speedup: {format_speedup(max_speedup)}")
            print(f"   Methods fused: {len(results)}")

    print("=" * 60)


if __name__ == "__main__":
    main()
