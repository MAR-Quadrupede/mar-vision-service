#!/usr/bin/env python3
"""
M.A.R. — YOLO Models Benchmark & Loading Verification
Tests loading, warm-up, and latency benchmarking of YOLO models on edge hardware.
"""

import sys
import time
from pathlib import Path
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERRORE: Ultralytics non è installato. Esegui: pip install ultralytics")
    sys.exit(1)


MODELS_TO_BENCHMARK = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolo11n.pt",
    "yolo11s.pt"
]


def benchmark_model(model_name: str):
    print("\n" + "=" * 60)
    print(f"  Test Modello: {model_name}")
    print("=" * 60)

    # Check local models dir
    local_path = Path(__file__).parent.parent / "models" / model_name
    target = str(local_path) if local_path.exists() else model_name

    try:
        t0 = time.time()
        print(f"Caricamento {target}...")
        model = YOLO(target)
        load_time = (time.time() - t0) * 1000
        print(f"✓ Caricato in {load_time:.1f} ms")

        # Create dummy stereo frame (1280x800)
        dummy_frame = np.zeros((800, 1280, 3), dtype=np.uint8)

        # Warm-up run
        _ = model(dummy_frame, verbose=False)

        # Benchmark 5 iterations
        latencies = []
        for _ in range(5):
            t_start = time.time()
            _ = model(dummy_frame, verbose=False)
            latencies.append((time.time() - t_start) * 1000)

        avg_latency = float(np.mean(latencies))
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0
        print(f"✓ Latenza media di inferenza: {avg_latency:.1f} ms (~{fps:.1f} FPS)")
        return True, avg_latency, fps

    except Exception as e:
        print(f"✗ Errore test modello {model_name}: {e}")
        return False, 0.0, 0.0


def main():
    print("=" * 64)
    print("     🐾 M.A.R. — BENCHMARK MODELLI YOLO EDGE AI")
    print("=" * 64)

    results = {}
    for m in MODELS_TO_BENCHMARK:
        success, latency, fps = benchmark_model(m)
        results[m] = (success, latency, fps)

    print("\n" + "=" * 64)
    print("                     RIEPILOGO BENCHMARK")
    print("=" * 64)
    print(f"{'MODELLO':<16} | {'STATO':<8} | {'LATENZA':<12} | {'FPS STIMATI'}")
    print("-" * 64)
    for m, (ok, lat, fps) in results.items():
        status = "✓ OK" if ok else "✗ FAIL"
        lat_str = f"{lat:.1f} ms" if ok else "N/A"
        fps_str = f"{fps:.1f} FPS" if ok else "N/A"
        print(f"{m:<16} | {status:<8} | {lat_str:<12} | {fps_str}")
    print("=" * 64)


if __name__ == "__main__":
    main()
