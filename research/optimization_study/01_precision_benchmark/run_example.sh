#!/bin/bash
# Пример запуска бенчмарка точности

set -e

echo "🔬 Precision Benchmark — Example Run"
echo "======================================"

# Параметры
IMAGE="${1:-./data/test.jpg}"
METHODS="${2:-sobel_edge,otsu_thresholding,canny_edge}"
PRECISIONS="${3:-fp32,fp16,bf16}"
OUTPUT_DIR="./results/precision_benchmark"

# Проверка изображения
if [ ! -f "$IMAGE" ]; then
    echo "❌ Image not found: $IMAGE"
    exit 1
fi

# Запуск через Python
python -c "
import sys
sys.path.insert(0, '.')

from segmenters.TorchSegmenter import TorchSegmenter
from optimization_study.01_precision_benchmark import (
    PrecisionBenchmark, 
    PrecisionVisualizer,
    ReportGenerator
)

# Инициализация
segmenter = TorchSegmenter(method='sobel_edge')
benchmark = PrecisionBenchmark(segmenter, '$IMAGE')

# Запуск бенчмарка
methods = '$METHODS'.split(',')
precisions = '$PRECISIONS'.split(',')

print(f'🎯 Methods: {methods}')
print(f'🎯 Precisions: {precisions}')
print()

df = benchmark.run_full_benchmark(methods=methods, precisions=precisions)

# Сводка
benchmark.print_summary(df)

# Визуализация
viz = PrecisionVisualizer(output_dir='$OUTPUT_DIR/plots')
plots = viz.plot_all(df, prefix='example_')
print(f'📊 Plots saved: {list(plots.values())}')

# Отчёт
reporter = ReportGenerator(output_dir='$OUTPUT_DIR/reports')
reports = reporter.generate_all(df, benchmark.get_summary(df))
print(f'📄 Reports saved: {list(reports.values())}')
"

echo ""
echo "✅ Done! Check $OUTPUT_DIR for results"