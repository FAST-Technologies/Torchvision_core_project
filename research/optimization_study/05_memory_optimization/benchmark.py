#!/usr/bin/env python3
"""
CLI-скрипт для бенчмаркинга оптимизации памяти.
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import torch

from .config import MemoryConfig, MemoryPolicy
from .memory_optimizer import MemoryOptimizer
from .report_generator import MemoryReportGenerator
from .visualization import MemoryVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="🧠 Memory Optimization Benchmark — исследование оптимизации памяти")

    # Выбор методов
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Методы для тестирования (через запятую или 'all')",
    )

    # Политика памяти
    parser.add_argument(
        "--policy",
        type=str,
        default="pooled",
        choices=["lazy", "pooled", "reuse", "aggressive_gc", "pinned"],
        help="Политика управления памятью",
    )

    # Параметры кэширования
    parser.add_argument("--cache-size", type=int, default=100, help="Макс. размер кэша ядер")
    parser.add_argument("--cache-ttl", type=float, default=300.0, help="Время жизни кэша (сек)")

    # Устройство
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Устройство для бенчмарка",
    )

    # Параметры бенчмарка
    parser.add_argument("--n-runs", type=int, default=20, help="Количество запусков для замера")
    parser.add_argument(
        "--input-size",
        type=str,
        default="512x512",
        help="Размер тестового изображения: 'WxH'",
    )

    # Вывод
    parser.add_argument("--output", type=str, default=None, help="Папка для сохранения результатов")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")
    parser.add_argument("--detect-leaks", action="store_true", help="Активировать детектор утечек")

    # Визуализация
    parser.add_argument("--plot", action="store_true", help="Сгенерировать графики")

    # Сравнение
    parser.add_argument("--compare", action="store_true", help="Сравнить до/после оптимизации")

    return parser.parse_args()


def main():
    args = parse_args()

    # === Инициализация ===
    print("🧠 Memory Optimization Benchmark")
    print("=" * 60)

    # Парсинг размера изображения
    try:
        w, h = map(int, args.input_size.split("x"))
        input_tensor = torch.randn(1, 1, h, w, device=args.device)
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
    policy_map = {
        "lazy": MemoryPolicy.LAZY,
        "pooled": MemoryPolicy.POOLED,
        "reuse": MemoryPolicy.REUSE,
        "aggressive_gc": MemoryPolicy.AGGRESSIVE_GC,
        "pinned": MemoryPolicy.PINNED,
    }

    config = MemoryConfig(
        policy=policy_map[args.policy],
        cache_max_size=args.cache_size,
        cache_ttl_seconds=args.cache_ttl,
        detect_leaks=args.detect_leaks,
        n_runs=args.n_runs,
        device=args.device,
        verbose=args.verbose,
        output_dir=args.output or "./results/memory_optimization",
    )

    # === Оптимизация ===
    optimizer = MemoryOptimizer(segmenter, config, device=args.device)

    if args.compare:
        print(f"\n🔬 Running comparative benchmark...")
        reports = optimizer.optimize_all_methods(methods, input_tensor)
    else:
        print(f"\n🔬 Profiling methods...")
        reports = []
        for method_name in methods:
            try:
                report = optimizer.optimize_method(method_name, input_tensor)
                reports.append(report)
            except Exception as e:
                print(f"   ❌ {method_name}: {e}")

    # === Отчёт ===
    if args.output:
        reporter = MemoryReportGenerator(output_dir=args.output)
        report_path = reporter.generate_markdown(reports, config)
        print(f"📄 Report saved: {report_path}")

    # === Визуализация ===
    if args.plot:
        viz = MemoryVisualizer(output_dir=args.output or "./plots")
        plots = viz.plot_all(reports)
        print(f"📊 Plots saved: {list(plots.values())}")

    # === Сводка ===
    print(f"\n{'='*60}")
    print("📊 Summary:")

    if reports:
        total_saved = sum(r.memory_saved_mb for r in reports)
        avg_reduction = np.mean([r.reduction_pct for r in reports])
        avg_speedup = np.mean([r.speedup for r in reports])

        print(f"   Methods optimized: {len(reports)}")
        print(f"   Total memory saved: {total_saved:.2f} MB")
        print(f"   Avg reduction: {avg_reduction:.1f}%")
        print(f"   Avg speedup: {avg_speedup:.2f}×")

        # Топ-3 по экономии
        top_saved = sorted(reports, key=lambda r: r.memory_saved_mb, reverse=True)[:3]
        print(f"\n   🏆 Top 3 by memory saved:")
        for i, r in enumerate(top_saved, 1):
            print(f"      {i}. {r.method_name}: {r.memory_saved_mb:.2f} MB ({r.reduction_pct:.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    main()
