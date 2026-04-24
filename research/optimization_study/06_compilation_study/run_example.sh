#!/bin/bash
# Пример запуска бенчмарка компиляционной оптимизации

set -e

echo "🔧 Compilation Study Benchmark — Example Run"
echo "============================================"

# Параметры
METHODS="${1:-sobel_edge,otsu_thresholding,adaptive_thresholding}"
STRATEGY="${2:-torch_compile}"
DEVICE="${3:-cuda}"
OUTPUT_DIR="./results/compilation_study"

# Запуск
echo "🎯 Methods: $METHODS"
echo "🎯 Strategy: $STRATEGY"
echo "🎯 Device: $DEVICE"
echo ""

python -m optimization_study.06_compilation_study.benchmark \
    --methods "$METHODS" \
    --strategy "$STRATEGY" \
    --device "$DEVICE" \
    --compile-mode "reduce-overhead" \
    --backend "inductor" \
    --n-runs 50 \
    --input-size "512x512" \
    --output "$OUTPUT_DIR" \
    --analyze-graph \
    --plot \
    --verbose

echo ""
echo "✅ Done! Results saved to $OUTPUT_DIR"