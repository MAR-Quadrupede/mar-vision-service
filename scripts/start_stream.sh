#!/bin/bash
# ==============================================================================
# M.A.R. — Script di Avvio Rapido Streaming Stereocamera 3D & Edge AI
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "========================================================================"
echo "  🐾 M.A.R. — Avvio Sistema Stereovisione 3D Global Shutter"
echo "========================================================================"
echo ""

# Verifica Python
if ! command -v python3 &> /dev/null; then
    echo "✗ Errore: Python 3 non trovato nel sistema."
    exit 1
fi

# Controllo dipendenze essenziali
echo "Controllo dipendenze..."
python3 scripts/test_dependencies.py || {
    echo ""
    echo "⚠️ Alcune dipendenze non sono presenti. Vuoi installarle? (s/n)"
    read -r resp
    if [[ "$resp" =~ ^[sSyY]$ ]]; then
        pip3 install -r requirements.txt
    fi
}

echo ""
echo "🚀 Avvio applicazione con server di streaming sulla porta 8443..."
echo "  • Stream SX + YOLO: http://localhost:8443/stream_left.mjpg"
echo "  • Stream DX + YOLO: http://localhost:8443/stream_right.mjpg"
echo "  • Dashboard UI:     http://localhost:8443/"
echo ""
echo "Per fermare il server: Premi Ctrl+C"
echo "========================================================================"
echo ""

# Esecuzione principale
python3 main.py "$@"
