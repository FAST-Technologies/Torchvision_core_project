#!/bin/bash
# Пример запуска бенчмарка оптимизации памяти

set -e

echo "🧠 Memory Optimization Benchmark — Example Run"
echo "=============================================="

# Параметры
METHODS="${1:-sauvola_thresholding,niblack_thresholding,adaptive_thresholding}"
POLICY="${2:-pooled}"
DEVICE="${3:-cuda}"
OUTPUT_DIR="./results/memory_optimization"

# Запуск
echo "🎯 Methods: $METHODS"
echo "🎯 Policy: $POLICY"
echo "🎯 Device: $DEVICE"
echo ""

python -m optimization_study.05_memory_optimization.benchmark \
    --methods "$METHODS" \
    --policy "$POLICY" \
    --device "$DEVICE" \
    --n-runs 20 \
    --input-size "512x512" \
    --output "$OUTPUT_DIR" \
    --detect-leaks \
    --plot \
    --verbose

echo ""
echo "✅ Done! Results saved to $OUTPUT_DIR"