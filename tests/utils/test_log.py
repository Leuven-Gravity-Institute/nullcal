"""Tests for logger setup.

``setup_logger`` is called once at the start of a run and configures the package's only logger. The
things that can go wrong are all idempotence and level-propagation faults: handlers added twice on a
second call, a level set on the logger but not on its handlers, a file handler created without its
directory. Each has its own test.
"""

from __future__ import annotations

import logging

import pytest

from nullcal.utils.log import get_version_information, setup_logger
from nullcal.version import __version__


@pytest.fixture(autouse=True)
def restore_logger():
    """Give every test a clean ``nullcal`` logger and put the caller's back afterwards.

    Without this the tests would configure each other, which is precisely the failure mode being
    tested for.
    """
    logger = logging.getLogger("nullcal")
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    saved_propagate = logger.propagate
    logger.handlers = []

    yield logger

    for handler in logger.handlers:
        handler.close()
    logger.handlers = saved_handlers
    logger.setLevel(saved_level)
    logger.propagate = saved_propagate


@pytest.mark.unit
def test_get_version_information_returns_the_package_version():
    """The value logged as provenance must be the version the package actually reports."""
    assert get_version_information() == __version__


@pytest.mark.unit
def test_adds_a_stream_handler_and_stops_propagating(restore_logger):
    """Propagation is disabled so the package's records do not appear twice under a root handler."""
    setup_logger()

    assert restore_logger.propagate is False
    assert any(isinstance(handler, logging.StreamHandler) for handler in restore_logger.handlers)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("log_level", "expected"),
    [("DEBUG", logging.DEBUG), ("debug", logging.DEBUG), ("warning", logging.WARNING), ("INFO", logging.INFO)],
)
def test_accepts_level_names_in_any_case(restore_logger, log_level, expected):
    setup_logger(log_level=log_level)

    assert restore_logger.level == expected


@pytest.mark.unit
@pytest.mark.parametrize("log_level", [10, logging.WARNING, logging.ERROR])
def test_accepts_integer_levels(restore_logger, log_level):
    """Integer levels are a documented input, as an alternative to the level names."""
    setup_logger(log_level=log_level)

    assert restore_logger.level == log_level


@pytest.mark.unit
def test_rejects_a_level_given_as_a_numeric_string(restore_logger):
    """``"10"`` is not accepted, only the integer ``10``.

    A string is routed to ``getattr(logging, ...)``, which has no attribute named ``"10"``. The
    docstring offers "a string from the list above, or an integer", so refusing a numeric string is
    consistent with the documented contract -- but the boundary is easy to trip over when levels
    arrive from a config file, where everything is a string, so it is pinned rather than left to be
    discovered.
    """
    with pytest.raises(ValueError, match="log_level 10 not understood"):
        setup_logger(log_level="10")


@pytest.mark.unit
def test_rejects_an_unknown_level_name(restore_logger):
    """A typo in a config file must stop the run, not silently select a default verbosity."""
    with pytest.raises(ValueError, match="log_level nonsense not understood"):
        setup_logger(log_level="nonsense")


@pytest.mark.unit
def test_handlers_are_set_to_the_requested_level(restore_logger):
    """A handler left at its own level filters records the logger already accepted.

    Setting the level on the logger alone is the classic half-fix: it looks right in the logger's
    attributes and drops records at the handler.
    """
    setup_logger(log_level="DEBUG")

    assert restore_logger.handlers
    assert all(handler.level == logging.DEBUG for handler in restore_logger.handlers)


@pytest.mark.unit
def test_calling_twice_does_not_duplicate_handlers(restore_logger):
    """Every record would otherwise be emitted once per call to ``setup_logger``."""
    setup_logger()
    handler_count = len(restore_logger.handlers)

    setup_logger()

    assert len(restore_logger.handlers) == handler_count


@pytest.mark.unit
def test_second_call_updates_the_level_of_existing_handlers(restore_logger):
    """Re-running with a new level must take effect even though no handler is added."""
    setup_logger(log_level="WARNING")

    setup_logger(log_level="DEBUG")

    assert restore_logger.level == logging.DEBUG
    assert all(handler.level == logging.DEBUG for handler in restore_logger.handlers)


@pytest.mark.unit
def test_writes_a_log_file_when_a_label_is_given(tmp_path, restore_logger):
    """The output directory is created if missing -- a run should not fail on a fresh outdir."""
    outdir = tmp_path / "fresh" / "nested"

    setup_logger(outdir=str(outdir), label="analysis")
    logging.getLogger("nullcal").info("a recorded message")

    for handler in restore_logger.handlers:
        handler.flush()
    log_file = outdir / "analysis.log"
    assert log_file.exists()
    assert "a recorded message" in log_file.read_text(encoding="utf-8")


@pytest.mark.unit
def test_no_file_handler_without_a_label(restore_logger):
    """Without a label there is no file name to use, so logging stays on the stream only."""
    setup_logger(outdir=".")

    assert not any(isinstance(handler, logging.FileHandler) for handler in restore_logger.handlers)


@pytest.mark.unit
def test_version_is_logged_on_request(tmp_path, restore_logger):
    """``print_version`` is how a run records which build produced it."""
    setup_logger(outdir=str(tmp_path), label="versioned", log_level="INFO", print_version=True)

    for handler in restore_logger.handlers:
        handler.flush()
    contents = (tmp_path / "versioned.log").read_text(encoding="utf-8")
    assert f"Running nullcal version: {__version__}" in contents


@pytest.mark.unit
def test_version_is_not_logged_by_default(tmp_path, restore_logger):
    setup_logger(outdir=str(tmp_path), label="quiet")

    for handler in restore_logger.handlers:
        handler.flush()
    assert "Running nullcal version" not in (tmp_path / "quiet.log").read_text(encoding="utf-8")
