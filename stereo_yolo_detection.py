#!/usr/bin/env python3
"""
M.A.R. — Interactive Stereo YOLO Detection (Backwards Compatibility Wrapper)
Launches the interactive model selector and stereo detector.
"""

import sys
from config import parse_cli_arguments
from main import run_system

if __name__ == "__main__":
    if "--interactive" not in sys.argv and "--model" not in sys.argv:
        sys.argv.append("--interactive")
    config = parse_cli_arguments()
    run_system(config)
