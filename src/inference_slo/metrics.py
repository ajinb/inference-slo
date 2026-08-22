"""SLIs.

**Goodput, not throughput.** A server that completes a thousand requests and misses the
target on all of them has done no useful work. Counting completions rather than completions
*within SLO* is how an overloaded system reports itself as busy and healthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .slo import SLO
from .types import Criticality, Outcome


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. No interpolation anywhere in this package."""
    if not values:
        return 0.0
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    ordered = sorted(values)
    index = max(0, math.ceil(p * len(ordered)) - 1)
    return ordered[index]


@dataclass(frozen=True)
class ClassMetrics:
    criticality: Criticality
    offered: int
    admitted: int
    shed: int
    completed: int
    within_slo: int
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    tpot_p95: float

    @property
    def shed_rate(self) -> float:
        return self.shed / self.offered if self.offered else 0.0

    @property
    def attainment(self) -> float:
        """Share of *offered* requests that completed inside their SLO.

        Deliberately measured against offered rather than admitted load: measuring against
        admitted lets a policy score perfectly by shedding almost everything.
        """
        return self.within_slo / self.offered if self.offered else 0.0


@dataclass(frozen=True)
class RunMetrics:
    policy: str
    offered: int
    admitted: int
    shed: int
    completed: int
    goodput: int
    by_class: dict[Criticality, ClassMetrics] = field(default_factory=dict)

    @property
    def goodput_rate(self) -> float:
        return self.goodput / self.offered if self.offered else 0.0

    @property
    def shed_rate(self) -> float:
        return self.shed / self.offered if self.offered else 0.0

    def attainment(self, criticality: Criticality) -> float:
        metrics = self.by_class.get(criticality)
        return metrics.attainment if metrics else 0.0


def summarise(policy: str, outcomes: list[Outcome], slo: SLO) -> RunMetrics:
    by_class: dict[Criticality, ClassMetrics] = {}
    total_goodput = 0

    for criticality in Criticality:
        group = [o for o in outcomes if o.request.criticality is criticality]
        completed = [o for o in group if o.completed]
        within = [o for o in completed if slo.met(criticality, o.ttft_s, o.tpot_s)]
        total_goodput += len(within)

        ttfts = [o.ttft_s for o in completed]
        tpots = [o.tpot_s for o in completed]
        by_class[criticality] = ClassMetrics(
            criticality=criticality,
            offered=len(group),
            admitted=sum(1 for o in group if o.admitted),
            shed=sum(1 for o in group if not o.admitted),
            completed=len(completed),
            within_slo=len(within),
            ttft_p50=percentile(ttfts, 0.50),
            ttft_p95=percentile(ttfts, 0.95),
            ttft_p99=percentile(ttfts, 0.99),
            tpot_p95=percentile(tpots, 0.95),
        )

    return RunMetrics(
        policy=policy,
        offered=len(outcomes),
        admitted=sum(1 for o in outcomes if o.admitted),
        shed=sum(1 for o in outcomes if not o.admitted),
        completed=sum(1 for o in outcomes if o.completed),
        goodput=total_goodput,
        by_class=by_class,
    )
