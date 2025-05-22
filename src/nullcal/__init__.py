from __future__ import annotations

from .utils import logger

__version__ = '0.5.0'


def log_version_information():
    logger.info(f"Running nullcal: {__version__}")
