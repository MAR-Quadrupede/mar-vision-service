#!/usr/bin/env python3
"""
M.A.R. — Standalone Global Shutter Stereo Navigator (Backwards Compatibility Wrapper)
Launches the unified M.A.R. Stereovisione Navigator.
"""

from config import parse_cli_arguments
from main import run_system

if __name__ == "__main__":
    config = parse_cli_arguments()
    run_system(config)
