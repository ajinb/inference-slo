"""Deterministic workload generation.

Seeded and reproducible: the same seed produces the same arrivals on every machine, which is
what makes a claim about one policy beating another checkable rather than anecdotal.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .types import Criticality, Request


@dataclass(frozen=True)
class WorkloadSpec:
    """Shape of the offered load."""

    duration_s: float = 60.0
    #: Requests per second, Poisson arrivals.
    rate_rps: float = 12.0
    mix: tuple[float, float, float] = (0.25, 0.45, 0.30)
    prompt_tokens_mean: int = 1_800
    output_tokens_mean: int = 320
    seed: int = 42

    def __post_init__(self) -> None:
        if self.duration_s <= 0 or self.rate_rps <= 0:
            raise ValueError("duration and rate must be positive")
        if abs(sum(self.mix) - 1.0) > 1e-9:
            raise ValueError(f"mix must sum to 1.0, got {sum(self.mix)}")
        if any(p < 0 for p in self.mix):
            raise ValueError("mix fractions cannot be negative")


def generate(spec: WorkloadSpec) -> list[Request]:
    """Poisson arrivals with log-normally distributed token counts.

    Log-normal because real prompt length distributions are heavy-tailed — a mean with no tail
    makes every admission policy look better than it is, since the tail is the whole problem.
    """
    rng = random.Random(spec.seed)
    classes = (Criticality.CRITICAL, Criticality.STANDARD, Criticality.SHEDDABLE)

    requests: list[Request] = []
    now = 0.0
    request_id = 0

    while True:
        now += rng.expovariate(spec.rate_rps)
        if now >= spec.duration_s:
            break
        requests.append(Request(
            id=request_id,
            arrival_s=now,
            criticality=rng.choices(classes, weights=spec.mix, k=1)[0],
            prompt_tokens=_lognormal(rng, spec.prompt_tokens_mean),
            output_tokens=_lognormal(rng, spec.output_tokens_mean),
        ))
        request_id += 1

    return requests


def _lognormal(rng: random.Random, mean: int, sigma: float = 0.6) -> int:
    """A log-normal draw with the given arithmetic mean, floored at 1."""
    mu = math.log(mean) - (sigma ** 2) / 2
    return max(1, int(rng.lognormvariate(mu, sigma)))
