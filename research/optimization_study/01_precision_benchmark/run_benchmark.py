"""
Главный скрипт запуска исследования точности вычислений.
Интегрирует все модули бенчмарка, учитывает поддержку типов данных на текущем устройстве
и генерирует полный отчёт с графиками.
"""

import os
import sys
import torch
import warnings
from pathlib import Path

# Добавляем корень проекта в sys.path (если запускается из подпапки)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from segmenters.TorchSegmenter import TorchSegmenter
from research.optimization_study.config.precision import PrecisionConfig
from research.optimization_study.precision_benchmark import PrecisionBenchmark
from research.optimization_study.report_generator import ReportGenerator
from research.optimization_study.visualizer import PrecisionVisualizer
from research.optimization_study.config.precision import (
    is_dtype_supported,
    PRECISION_TO_DTYPE,
)


def main():
    print("🔬 Запуск исследования точности вычислений (Precision Benchmark)")
    print("=" * 60)

    # 1. Настройка устройства и проверка поддержки типов
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Устройство: {device}")

    # Динамическая фильтрация доступных прецизионных режимов
    available_precisions = [
        p for p in ["fp32", "fp16", "bf16", "int8"] if is_dtype_supported(PRECISION_TO_DTYPE[p], str(device.type))
    ]
    if not available_precisions:
        raise RuntimeError("Нет поддерживаемых типов данных для текущего устройства!")
    print(f"✅ Доступные режимы: {', '.join(available_precisions)}")
    if "bf16" in available_precisions and device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"   ℹ️  BF16 требует Compute Capability ≥ 8.0 (у вас: {cap[0]}.{cap[1]})")

    # 2. Инициализация сегментера и загрузка тестового изображения
    print("\n📦 Инициализация TorchSegmenter...")
    segmenter = TorchSegmenter(method="sobel_edge", device=str(device))

    # Загрузка тестового изображения (замените путь на свой)
    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        print(f"⚠️  {image_path} не найден. Создаю синтетическое изображение...")
        import numpy as np
        from PIL import Image

        img_np = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(img_np).save(image_path)

    print(f"🖼️  Загрузка {image_path}...")
    # TorchSegmenter.preprocess_image ожидает np.ndarray или PIL.Image
    from PIL import Image

    test_image = Image.open(image_path)

    # 3. Настройка конфигурации бенчмарка
    config = PrecisionConfig(
        precisions=available_precisions,
        n_runs=30,  # Количество измерений
        warmup_runs=5,  # Прогрев
        sync_cuda=True,  # Синхронизация для точных замеров на GPU
        reference_precision="fp32",
        tolerance=1e-4,
        output_dir="./results/precision_benchmark",
        verbose=True,
    )

    # 4. Запуск бенчмарка
    print("\n⏱️  Запуск бенчмарка...")
    benchmark = PrecisionBenchmark(segmenter, test_image, config, device=str(device))

    # Тестируемые методы (можно расширить)
    methods_to_test = [
        "sobel_edge",
        "otsu_thresholding",
        "adaptive_thresholding",
        "canny_edge",
    ]

    df = benchmark.run_full_benchmark(methods=methods_to_test, include_accuracy=True)

    if df.empty:
        print("❌ Бенчмарк не вернул данных. Проверьте логи.")
        return

    # 5. Генерация отчётов и графиков
    print("\n📊 Генерация отчётов...")
    summary = benchmark.get_summary(df)
    benchmark.print_summary(df)

    output_dir = Path(config.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Отчёты
    reporter = ReportGenerator(output_dir=str(output_dir))
    report_paths = reporter.generate_all(df, summary, base_name=f"precision_report_{timestamp}")
    print(f"📄 Отчёты сохранены в: {output_dir}")

    # Графики
    if MATPLOTLIB_AVAILABLE:
        print("\n📈 Генерация графиков...")
        visualizer = PrecisionVisualizer(output_dir=str(output_dir / "plots"))
        plot_paths = visualizer.plot_all(df, prefix=f"{timestamp}_")
        print(f"📊 Графики сохранены в: {output_dir / 'plots'}")
    else:
        print("⚠️  matplotlib/seaborn не установлены. Графики не будут сгенерированы.")

    print("\n✅ Исследование завершено успешно!")


if __name__ == "__main__":
    from datetime import datetime

    try:
        import matplotlib

        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False

    main()
