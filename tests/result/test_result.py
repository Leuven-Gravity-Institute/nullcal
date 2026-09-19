"""Tests for the package-owned sampler result container."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from nullcal.result import Result


def test_result_is_built_from_blackjax_style_states():
    states = SimpleNamespace(
        position={"amplitude": np.arange(6.0).reshape(3, 2), "phase": np.ones((3, 2))},
        logdensity=np.array([-3.0, -2.0, -1.0]),
    )

    result = Result.from_states(states, metadata={"warmup_steps": 100})

    assert result.samples is states.position
    assert result.logdensity is states.logdensity
    assert result.sample_count == 3
    assert result.metadata == {"warmup_steps": 100}


def test_result_rejects_parameter_arrays_with_different_sample_counts():
    with pytest.raises(ValueError, match="sample dimension"):
        Result(
            samples={"amplitude": np.ones((3, 2)), "phase": np.ones((4, 2))},
            logdensity=np.ones(3),
        )


def test_result_rejects_logdensity_with_a_different_sample_count():
    with pytest.raises(ValueError, match="logdensity"):
        Result(samples={"amplitude": np.ones((3, 2))}, logdensity=np.ones(2))


def test_result_requires_at_least_one_sample_array():
    with pytest.raises(ValueError, match="at least one"):
        Result(samples={}, logdensity=np.ones(2))


def test_result_is_frozen():
    result = Result(samples={"amplitude": np.ones((3, 2))}, logdensity=np.ones(3))

    with pytest.raises(FrozenInstanceError):
        result.logdensity = np.zeros(3)
