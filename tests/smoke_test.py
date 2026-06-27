"""Smoke tests before publishing to verify the wheel and source distribution."""

from __future__ import annotations

import sys

import nullcal

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - used in isolated wheel smoke runs.
    pytestmark = ()
else:
    pytestmark = pytest.mark.integration


def test_basic_import() -> None:
    """Test basic import."""
    print(f"Python version: {sys.version}")
    print(f"Package version: {nullcal.__version__}")

    assert "site-packages" in nullcal.__file__ or "dist" in nullcal.__file__, (
        f"Package imported from unexpected location: {nullcal.__file__}"
    )


if __name__ == "__main__":
    test_basic_import()
    print("Smoke test passed: Package is importable.")
