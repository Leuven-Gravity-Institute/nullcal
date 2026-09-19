"""Sampler result types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Result:
    """Posterior samples extracted from a sequence of sampler states."""

    samples: Mapping[str, Any]
    logdensity: Any
    info: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("samples must contain at least one parameter array")
        sample_counts = {value.shape[0] for value in self.samples.values() if value.ndim > 0}
        if len(sample_counts) != 1 or any(value.ndim == 0 for value in self.samples.values()):
            raise ValueError("all parameter arrays must have the same leading sample dimension")
        if self.logdensity.ndim != 1 or self.logdensity.shape[0] != self.sample_count:
            raise ValueError("logdensity must contain one value per posterior sample")

    @property
    def sample_count(self) -> int:
        """Number of retained posterior samples."""
        return next(iter(self.samples.values())).shape[0]

    @classmethod
    def from_states(
        cls,
        states: Any,
        *,
        info: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Result:
        """Extract positions and log densities from stacked sampler states."""
        return cls(
            samples=states.position,
            logdensity=states.logdensity,
            info=info,
            metadata={} if metadata is None else metadata,
        )


__all__ = ["Result"]
