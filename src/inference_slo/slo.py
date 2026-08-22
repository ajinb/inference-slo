"""Service level objectives, per criticality class.

A single SLO across all traffic is the mistake that makes shedding impossible: if batch and
interactive traffic are held to the same target, there is no principled basis for preferring
one over the other under pressure. Different classes get different targets, and that is what
gives an admission policy something to optimise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Criticality


@dataclass(frozen=True)
class Target:
    """The latency contract for one criticality class."""

    ttft_s: float
    tpot_s: float

    def __post_init__(self) -> None:
        if self.ttft_s <= 0 or self.tpot_s <= 0:
            raise ValueError("SLO targets must be positive")

    def met_by(self, ttft_s: float, tpot_s: float) -> bool:
        return ttft_s <= self.ttft_s and tpot_s <= self.tpot_s


def _default_targets() -> dict[Criticality, Target]:
    return {
        # A human is watching the cursor blink.
        Criticality.CRITICAL: Target(ttft_s=1.0, tpot_s=0.05),
        Criticality.STANDARD: Target(ttft_s=3.0, tpot_s=0.10),
        # Nobody is watching. Minutes are fine.
        Criticality.SHEDDABLE: Target(ttft_s=30.0, tpot_s=0.50),
    }


@dataclass(frozen=True)
class SLO:
    """The full set of targets, plus the headroom rule that protects the top class."""

    targets: dict[Criticality, Target] = field(default_factory=_default_targets)
    #: Fraction of capacity kept clear of non-critical traffic. The reservation is what stops
    #: a burst of batch work from queueing ahead of the interactive traffic that pays for it.
    critical_headroom: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.critical_headroom < 1.0:
            raise ValueError("critical_headroom must be in [0, 1)")
        missing = [c for c in Criticality if c not in self.targets]
        if missing:
            raise ValueError(f"no target for {[c.value for c in missing]}")

    def target_for(self, criticality: Criticality) -> Target:
        return self.targets[criticality]

    def met(self, criticality: Criticality, ttft_s: float, tpot_s: float) -> bool:
        return self.targets[criticality].met_by(ttft_s, tpot_s)
