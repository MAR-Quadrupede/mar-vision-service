"""
M.A.R. Stereovisione Server Package.
High-performance MJPEG multi-stream and REST API server for the web dashboard.
"""

from .mjpeg_server import MultiStreamServer

__all__ = ["MultiStreamServer"]
