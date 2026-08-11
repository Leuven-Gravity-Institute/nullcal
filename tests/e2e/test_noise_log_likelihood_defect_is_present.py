"""Pins the known defect precisely, at the call site, with no ``xfail`` involved.

This module exists because ``xfail`` is a blunt instrument. It was measured, not assumed: with
``pytest.mark.xfail(strict=True, raises=IndexError)`` applied at module level, an ``IndexError``
raised *from fixture setup* is reported as ``XFAIL``, and so is an unrelated ``IndexError`` from
anywhere in the test body. So the companion module's three tests would report the defect as
"expected" without having exercised the filter-domain behaviour at all — if, say,
``pipeline.build_likelihood`` started raising ``IndexError`` for an unrelated reason, the suite
would stay green and say the defect was as expected.

The tests here close that gap two ways:

* they assert the exception at the exact call that is broken, with a message match, so nothing
  else can satisfy them;
* they carry no ``xfail``, so they are ordinary passing tests today. That makes them the canary
  for the whole module: they build the same ``likelihood`` fixture, so a fixture that starts
  raising shows up here as an error rather than being absorbed into an expected failure.

When R17 fixes the defect these tests **fail**, which is the intended signal: the fixing commit
deletes this module and removes the ``xfail`` markers from the companion one, in the same commit.
"""

from __future__ import annotations

import pytest

from . import pipeline

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


@pytest.fixture(scope="module")
def likelihood(tmp_path_factory):
    return pipeline.build_likelihood(tmp_path_factory.mktemp("defect_present"))


def test_uncalibrated_time_frequency_null_stream_raises_index_error(likelihood):
    """The precise, currently-broken call raises ``IndexError`` — nothing weaker."""
    null_stream = likelihood.null_stream_calculator
    with pytest.raises(IndexError, match="too many indices for array"):
        null_stream.compute_uncalibrated_time_frequency_domain_null_stream()


def test_noise_log_likelihood_raises_index_error(likelihood):
    """The public bilby entry point inherits the failure, which is why it matters."""
    with pytest.raises(IndexError, match="too many indices for array"):
        likelihood.noise_log_likelihood()


def test_the_calibrated_path_is_unaffected(likelihood):
    """The other branch works, which is why published results are not implicated.

    Kept here rather than in the companion module because it must run unconditionally: it is the
    evidence for the scope claim, and an ``xfail`` marker would let it stop being checked.
    """
    from . import config

    likelihood.parameters = dict(config.calibration_parameters())
    value = float(likelihood.log_likelihood())
    assert value < 0.0, f"log_likelihood() = {value!r}; it is -0.5 * residual energy"
