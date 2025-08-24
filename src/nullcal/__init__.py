"""
A package to constrain calibration errors of a closed-geometry network
of gravitational-wave detectors.
"""

from __future__ import annotations

from .utils.log import setup_logger
from .version import __version__

setup_logger()


__all__ = [
    "__version__",
]
