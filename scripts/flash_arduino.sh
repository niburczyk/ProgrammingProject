#!/bin/bash
set -e

PORT=/dev/ttyACM0
BOARD=arduino:avr:uno
SKETCH=src/svm_predict_poly

echo "🔌 Arduino flashen..."
arduino-cli compile --fqbn $BOARD $SKETCH
arduino-cli upload -p $PORT --fqbn $BOARD $SKETCH
echo "✅ Arduino geflasht!"
