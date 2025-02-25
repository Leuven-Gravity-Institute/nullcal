from . import detector
from . import likelihood
from . import utils
from .utils import logger
from ._version import __version__


def get_version_information():
    return __version__


def log_version_information():
    logger.info(f"Running nullcal: {__version__}")
