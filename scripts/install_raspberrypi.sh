#!/bin/bash
# ==============================================================================
# M.A.R. — Script di Installazione Completo per Raspberry Pi 5 / 4
# ==============================================================================
# Configura l'ambiente Debian / Raspberry Pi OS a 64-bit per l'esecuzione
# del sistema di stereovisione, YOLOv8/v11 e OpenCV.

set -e

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "========================================================================"
echo "  🐾 M.A.R. — Setup Dipendenze Raspberry Pi (64-bit)"
echo "========================================================================"
echo ""

# 1. Pacchetti di sistema
echo "[1/4] Installazione librerie di sistema via APT..."
sudo apt update
sudo apt install -y \
    python3-opencv \
    python3-numpy \
    python3-pip \
    python3-serial \
    python3-dev \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libatlas-base-dev \
    v4l-utils

echo ""
echo "[2/4] Installazione pacchetti Python via pip..."
pip3 install --break-system-packages --default-timeout=300 \
    ultralytics \
    pyserial \
    psutil \
    pyyaml \
    pillow \
    requests \
    scipy

echo ""
echo "[3/4] Configurazione permessi porte seriali e video..."
sudo usermod -a -G dialout,video $USER || true

echo ""
echo "[4/4] Verifica finale ambiente..."
python3 -c "
import cv2, numpy, ultralytics
print('✓ OpenCV:', cv2.__version__)
print('✓ NumPy:', numpy.__version__)
print('✓ Ultralytics YOLO:', ultralytics.__version__)
"

echo ""
echo "========================================================================"
echo "  🎉 Installazione completata con successo!"
echo "  Per avviare il sistema: cd .. && ./scripts/start_stream.sh"
echo "========================================================================"
