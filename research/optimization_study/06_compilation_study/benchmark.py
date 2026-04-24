#!/usr/bin/env python3
"""
CLI-скрипт для бенчмаркинга компиляционной оптимизации.
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import torch

from .config import CompilationConfig, CompilationStrategy
from .graph_optimizer import GraphOptimizer
from .report_generator import CompilationReportGenerator
from .visualization import CompilationVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="🔧 Compilation Study Benchmark — исследование компиляционной оптимизации"
    )

    # Выбор методов
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Методы для тестирования (через запятую или 'all')",
    )

    # Стратегия компиляции
    parser.add_argument(
        "--strategy",
        type=str,
        default="torch_compile",
        choices=["none", "jit_script", "jit_trace", "torch_compile", "graph_freeze"],
        help="Стратегия компиляции",
    )

    # Параметры torch.compile
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Режим torch.compile",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="inductor",
        choices=["inductor", "cudagraphs", "aot_eager"],
        help="Бэкенд компиляции",
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
        "--analyze-graph", action="store_true", help="Анализировать структуру графа"
    )

    # Визуализация
    parser.add_argument("--plot", action="store_true", help="Сгенерировать графики")

    # Сравнение стратегий
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Сравнить разные стратегии компиляции",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # === Инициализация ===
    print("🔧 Compilation Study Benchmark")
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
        "none": CompilationStrategy.NONE,
        "jit_script": CompilationStrategy.JIT_SCRIPT,
        "jit_trace": CompilationStrategy.JIT_TRACE,
        "torch_compile": CompilationStrategy.TORCH_COMPILE,
        "graph_freeze": CompilationStrategy.GRAPH_FREEZE,
    }

    config = CompilationConfig(
        strategy=strategy_map[args.strategy],
        compile_mode=args.compile_mode,
        backend=args.backend,
        device=args.device,
        n_runs=args.n_runs,
        verbose=args.verbose,
        output_dir=args.output or "./results/compilation_study",
    )

    # === Оптимизация ===
    optimizer = GraphOptimizer(segmenter, config, device=args.device)

    if args.analyze_graph:
        print(f"\n🔍 Analyzing graph structure...")
        for method in methods[:5]:  # Ограничим для скорости
            if method in segmenter.method_map:
                from .utils import analyze_graph_structure

                info = analyze_graph_structure(
                    segmenter.method_map[method], example_input
                )
                print(
                    f"   {method}: {info['num_operations']} ops, "
                    f"potential: {info['optimization_potential']:.2f}"
                )

    # Запуск оптимизации
    if args.compare_strategies:
        print(f"\n🔬 Comparing compilation strategies...")
        # Для каждого метода тестируем несколько стратегий
        reports = []
        for method_name in methods:
            if method_name not in segmenter.method_map:
                continue
            for strat in ["jit_script", "jit_trace", "torch_compile"]:
                config.strategy = strategy_map[strat]
                optimizer = GraphOptimizer(segmenter, config, device=args.device)
                report = optimizer.optimize_method(method_name, example_input)
                if report:
                    reports.append(report)
    else:
        print(f"\n🔬 Running compilation benchmark...")
        reports = optimizer.optimize_all_methods(methods, example_input)

    # === Отчёт ===
    if args.output:
        reporter = CompilationReportGenerator(output_dir=args.output)
        report_path = reporter.generate_markdown(reports, config)
        print(f"📄 Report saved: {report_path}")

    # === Визуализация ===
    if args.plot:
        viz = CompilationVisualizer(output_dir=args.output or "./plots")
        plots = viz.plot_all(reports)
        print(f"📊 Plots saved: {list(plots.values())}")

    # === Сводка ===
    print(f"\n{'='*60}")
    print("📊 Summary:")

    if reports:
        valid = [r for r in reports if r.is_worthwhile]
        print(f"   Methods optimized: {len(reports)}")
        print(f"   Methods worthwhile: {len(valid)}")

        speedups = [r.speedup for r in reports]
        print(f"   Avg speedup: {np.mean(speedups):.2f}×")
        print(f"   Max speedup: {max(speedups):.2f}×")

        # Топ-3 по ускорению
        top = sorted(reports, key=lambda r: r.speedup, reverse=True)[:3]
        print(f"\n   🏆 Top 3 by speedup:")
        for i, r in enumerate(top, 1):
            print(
                f"      {i}. {r.method_name}: {r.speedup_formatted} " f"({r.strategy})"
            )

    print("=" * 60)


if __name__ == "__main__":
    main()
