"""Unit tests for the JAX WDM verification report."""

from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/jax_wdm_verification.py"
SPEC = importlib.util.spec_from_file_location("jax_wdm_verification", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
jax_wdm_verification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(jax_wdm_verification)


class FakeArchive(dict):
    """Minimal npz-like mapping for the report test."""

    @property
    def files(self):
        return list(self)


def test_reference_table_excludes_fed_back_inputs(monkeypatch):
    """A stored output reused as input must not be reported as reproduced."""
    reference = FakeArchive(
        {
            "independent_output": np.array([1.0]),
            "time_frequency_filter": np.array([True]),
        }
    )
    actual = {name: value.copy() for name, value in reference.items()}

    monkeypatch.setattr(jax_wdm_verification.np, "load", lambda _path: nullcontext(reference))
    monkeypatch.setattr(jax_wdm_verification.pipeline, "build_likelihood_from_reference_inputs", object)
    monkeypatch.setattr(jax_wdm_verification.pipeline, "compute_artifacts", lambda _likelihood: actual)

    names = [row[0] for row in jax_wdm_verification.reference_table()]

    assert names == ["independent_output"]


def test_report_identity_includes_the_platform(monkeypatch):
    """Last-bit verification values identify the platform that produced them."""
    monkeypatch.setattr(jax_wdm_verification.platform, "platform", lambda: "example-platform")

    assert jax_wdm_verification.report_identity("abc123") == ("Revision: `abc123`\nPlatform: `example-platform`")
