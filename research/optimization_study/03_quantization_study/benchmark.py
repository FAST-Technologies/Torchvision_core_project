"""
CLI-скрипт для запуска бенчмарка квантования.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch

from .config import QuantizationConfig, DEFAULT_CONFIG
from .quantizer import QuantizedSegmenter
from .report_generator import QuantizationReportGenerator
from .visualization import QuantizationVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="🔢 Quantization Benchmark — исследование квантования методов сегментации"
    )

    # Входные данные
    parser.add_argument(
        "--image", type=str, required=True, help="Путь к тестовому изображению"
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=None,
        help="Папка с изображениями для калибровки (для static INT8)",
    )

    # Выбор методов
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Методы для тестирования (через запятую или 'all')",
    )

    # Схемы квантования
    parser.add_argument(
        "--schemes",
        type=str,
        default="fp32,fp16,int8_dynamic",
        help="Схемы квантования: fp32,fp16,int8_dynamic,int8_static",
    )

    # Параметры бенчмарка
    parser.add_argument(
        "--n-runs", type=int, default=50, help="Количество запусков для замера"
    )
    parser.add_argument(
        "--calibration-steps",
        type=int,
        default=100,
        help="Шагов калибровки для static INT8",
    )

    # Устройство
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Устройство для бенчмарка (квантование лучше на CPU)",
    )

    # Вывод
    parser.add_argument(
        "--output", type=str, default=None, help="Папка для сохранения результатов"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")

    # Визуализация
    parser.add_argument("--plot", action="store_true", help="Сгенерировать графики")

    return parser.parse_args()


def load_calibration_images(calibration_dir: str, max_images: int = 100) -> list:
    """Загружает изображения для калибровки из папки."""
    from PIL import Image

    images = []
    calibration_path = Path(calibration_dir)

    if not calibration_path.exists():
        print(f"⚠️ Calibration directory not found: {calibration_dir}")
        return []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        for img_path in calibration_path.glob(ext):
            if len(images) >= max_images:
                break
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                images.append(img)
            except Exception as e:
                print(f"⚠️ Failed to load {img_path}: {e}")

    if not images:
        print(f"⚠️ No valid images found in {calibration_dir}")

    return images


def main():
    args = parse_args()

    # === Инициализация ===
    print("🔢 Quantization Benchmark")
    print("=" * 60)

    # Загрузка изображения
    try:
        from PIL import Image

        test_image = np.array(Image.open(args.image).convert("RGB"))
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        sys.exit(1)

    # Загрузка калибровочных данных
    calibration_data = None
    if args.calibration_dir and "int8_static" in args.schemes:
        calibration_data = load_calibration_images(
            args.calibration_dir, max_images=args.calibration_steps
        )
        print(f"📦 Loaded {len(calibration_data)} calibration images")

    # Инициализация сегментера
    print(f"📦 Loading TorchSegmenter...")
    from segmenters.TorchSegmenter import TorchSegmenter

    segmenter = TorchSegmenter(device=args.device)

    # Выбор методов
    if args.methods == "all":
        methods = list(segmenter.method_map.keys())
    else:
        methods = [m.strip() for m in args.methods.split(",")]

    # Выбор схем
    schemes = [s.strip() for s in args.schemes.split(",")]

    # Конфигурация
    config = QuantizationConfig(
        schemes=schemes,
        n_runs=args.n_runs,
        calibration_steps=args.calibration_steps,
        target_device=args.device,
        verbose=args.verbose,
        output_dir=args.output or "./results/quantization_study",
    )

    # === Бенчмарк ===
    quantizer = QuantizedSegmenter(segmenter, config)
    results = quantizer.run_full_benchmark(
        methods=methods,
        schemes=schemes,
        calibration_data=calibration_data,
        test_image=test_image,
    )

    # === Отчёт ===
    if args.output:
        reporter = QuantizationReportGenerator(output_dir=args.output)
        report_path = reporter.generate_markdown(results, config)
        print(f"📄 Report saved: {report_path}")

    # === Визуализация ===
    if args.plot:
        viz = QuantizationVisualizer(output_dir=args.output or "./plots")
        plots = viz.plot_all(results)
        print(f"📊 Plots saved: {list(plots.values())}")

    # === Сводка ===
    print(f"\n{'='*60}")
    print("📊 Summary:")

    if results:
        # Группировка по схемам
        from collections import defaultdict

        by_scheme = defaultdict(list)
        for r in results:
            by_scheme[r["scheme"]].append(r)

        for scheme, scheme_results in by_scheme.items():
            if not scheme_results:
                continue
            avg_speedup = np.mean([r["speedup"] for r in scheme_results])
            avg_agree = np.mean([r["pixel_agreement"] for r in scheme_results])
            print(
                f"   {scheme:15s}: {format_speedup(avg_speedup)} speedup, "
                f"{avg_agree*100:.2f}% agreement"
            )

    print("=" * 60)


if __name__ == "__main__":
    main()
