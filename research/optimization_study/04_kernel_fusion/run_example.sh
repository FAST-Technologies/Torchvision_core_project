#!/bin/bash
# Пример запуска бенчмарка kernel fusion

set -e

echo "⚡ Kernel Fusion Benchmark — Example Run"
echo "========================================"

# Параметры
METHODS="${1:-sauvola_thresholding,niblack_thresholding,sobel_edge}"
STRATEGY="${2:-graph}"
DEVICE="${3:-cuda}"
OUTPUT_DIR="./results/kernel_fusion"

# Запуск
echo "🎯 Methods: $METHODS"
echo "🎯 Strategy: $STRATEGY"
echo "🎯 Device: $DEVICE"
echo ""

python -m optimization_study.04_kernel_fusion.benchmark \
    --methods "$METHODS" \
    --strategy "$STRATEGY" \
    --device "$DEVICE" \
    --n-runs 50 \
    --input-size "512x512" \
    --output "$OUTPUT_DIR" \
    --profile-graph \
    --plot \
    --verbose

echo ""
echo "✅ Done! Results saved to $OUTPUT_DIR"