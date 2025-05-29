#!/bin/bash
set -e

PORT=/dev/ttyACM0
BOARD=arduino:avr:uno
SKETCH=src/svm_predict_poly/emg_transmitter.ino

echo "🔌 Arduino flashen..."
arduino-cli compile --fqbn $BOARD $SKETCH
arduino-cli upload -p $PORT --fqbn $BOARD $SKETCH
echo "✅ Arduino geflasht!"
