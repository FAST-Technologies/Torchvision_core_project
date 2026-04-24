#!/bin/bash
# Пример запуска бенчмарка

set -e

echo "🔧 Export Optimization Benchmark — Example Run"
echo "=============================================="

# Параметры
IMAGE="${1:-./data/test.jpg}"
METHODS="${2:-sobel_edge,global_thresholding,adaptive_thresholding}"
BACKEND="${3:-auto}"
OUTPUT_DIR="./results/export_benchmark"

# Проверка изображения
if [ ! -f "$IMAGE" ]; then
    echo "❌ Image not found: $IMAGE"
    exit 1
fi

# Запуск
echo "🎯 Image: $IMAGE"
echo "🎯 Methods: $METHODS"
echo "🎯 Backend: $BACKEND"
echo ""

python -m optimization_study.02_export_optimization.benchmark \
    --image "$IMAGE" \
    --methods "$METHODS" \
    --backend "$BACKEND" \
    --n-runs 50 \
    --output "$OUTPUT_DIR" \
    --verbose

echo ""
echo "✅ Done! Results saved to $OUTPUT_DIR"