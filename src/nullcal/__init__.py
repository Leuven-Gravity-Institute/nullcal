from __future__ import annotations

from .utils import logger

__version__ = '1.0.0'


def log_version_information():
    logger.info(f"Running nullcal: {__version__}")
