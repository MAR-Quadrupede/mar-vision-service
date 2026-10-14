#!/usr/bin/env python3
"""
M.A.R. — Stereovisione Dependency Verification Script
Checks and verifies all required and optional runtime dependencies.
"""

import sys
import platform

print("=" * 64)
print("     🐾 M.A.R. — VERIFICA AMBIENTE E DIPENDENZE STEREOVISIONE")
print("=" * 64)
print(f"Sistema Operativo: {platform.system()} ({platform.release()})")
print(f"Architettura:      {platform.machine()}")
print(f"Versione Python:   {sys.version.split()[0]}")
print("-" * 64)

all_core_ok = True

# 1. NumPy
try:
    import numpy as np
    print(f"  [✓ OK] NumPy: {np.__version__}")
except Exception as e:
    print(f"  [✗ ERRORE] NumPy: {e}")
    all_core_ok = False

# 2. OpenCV
try:
    import cv2
    print(f"  [✓ OK] OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"  [✗ ERRORE] OpenCV: {e}")
    all_core_ok = False

# 3. Ultralytics YOLO
try:
    import ultralytics
    from ultralytics import YOLO
    print(f"  [✓ OK] Ultralytics YOLO: {ultralytics.__version__}")
except Exception as e:
    print(f"  [✗ ERRORE] Ultralytics: {e}")
    all_core_ok = False

# 4. PyTorch
try:
    import torch
    print(f"  [✓ OK] PyTorch: {torch.__version__} (MPS/CUDA: {torch.cuda.is_available() or torch.backends.mps.is_available()})")
except Exception as e:
    print(f"  [⚠ INFO] PyTorch standalone non importato: {e}")

# 5. PySerial
try:
    import serial
    import serial.tools.list_ports
    ports = [p.device for p in serial.tools.list_ports.comports()]
    print(f"  [✓ OK] PySerial: {serial.__version__} (Porte rilevate: {ports or 'Nessuna'})")
except Exception as e:
    print(f"  [⚠ INFO] PySerial non presente: {e} (Disabiliterà la connessione seriale ad Arduino)")

print("-" * 64)
if all_core_ok:
    print("  🎉 Tutte le dipendenze essenziali sono installate e funzionanti!")
    print("  Puoi avviare il sistema con: python3 main.py")
else:
    print("  ✗ Alcune dipendenze essenziali mancano. Esegui:")
    print("     pip install -r requirements.txt")
print("=" * 64)

sys.exit(0 if all_core_ok else 1)
