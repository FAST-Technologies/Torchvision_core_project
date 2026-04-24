#!/bin/bash
# Пример запуска бенчмарка квантования

set -e

echo "🔢 Quantization Benchmark — Example Run"
echo "========================================"

# Параметры
IMAGE="${1:-./data/test.jpg}"
METHODS="${2:-sobel_edge,global_thresholding,otsu_thresholding}"
SCHEMES="${3:-fp32,fp16,int8_dynamic}"
DEVICE="${4:-cpu}"
OUTPUT_DIR="./results/quantization_study"
CALIB_DIR="${5:-./data/calibration/}"

# Проверка изображения
if [ ! -f "$IMAGE" ]; then
    echo "❌ Image not found: $IMAGE"
    exit 1
fi

# Запуск
echo "🎯 Image: $IMAGE"
echo "🎯 Methods: $METHODS"
echo "🎯 Schemes: $SCHEMES"
echo "🎯 Device: $DEVICE"
echo ""

python -m optimization_study.03_quantization_study.benchmark \
    --image "$IMAGE" \
    --methods "$METHODS" \
    --schemes "$SCHEMES" \
    --device "$DEVICE" \
    --calibration-dir "$CALIB_DIR" \
    --n-runs 50 \
    --output "$OUTPUT_DIR" \
    --plot \
    --verbose

echo ""
echo "✅ Done! Results saved to $OUTPUT_DIR"