"""The load sweep.

One question, asked across a range of offered load: as the accelerator saturates, what does
each admission policy do to goodput, and to the class you actually care about?
"""

from __future__ import annotations

from dataclasses import dataclass

from .backend import ServerSpec
from .metrics import RunMetrics
from .policies import default_policies
from .replay import ReplayConfig, replay
from .slo import SLO
from .types import Criticality
from .workload import WorkloadSpec, generate


@dataclass(frozen=True)
class SweepPoint:
    rate_rps: float
    policy: str
    goodput_rate: float
    shed_rate: float
    critical_attainment: float
    standard_attainment: float
    sheddable_attainment: float
    critical_ttft_p95: float


def sustainable_rps(spec: ServerSpec, workload: WorkloadSpec) -> float:
    """Roughly where this server tips over, from first principles.

    Decode is the binding constraint in almost every realistic mix: aggregate decode
    throughput divided by mean output length is the request rate the server can actually
    retire. Useful as an anchor so the sweep is read against capacity, not against a
    number someone picked.
    """
    return spec.decode_tokens_per_s / workload.output_tokens_mean


def sweep(
    rates: tuple[float, ...] = (4.0, 6.0, 8.0, 12.0, 16.0, 20.0),
    *,
    spec: ServerSpec | None = None,
    slo: SLO | None = None,
    seed: int = 42,
    duration_s: float = 60.0,
) -> list[SweepPoint]:
    spec = spec or ServerSpec()
    slo = slo or SLO()
    points: list[SweepPoint] = []

    for rate in rates:
        requests = generate(WorkloadSpec(duration_s=duration_s, rate_rps=rate, seed=seed))
        for policy in default_policies(slo):
            metrics = replay(requests, policy, spec=spec, slo=slo, config=ReplayConfig())
            points.append(_point(rate, metrics))

    return points


def headline(
    rate_rps: float = 12.0, *, seed: int = 42, duration_s: float = 60.0,
    spec: ServerSpec | None = None, slo: SLO | None = None,
) -> list[RunMetrics]:
    """One overloaded operating point, all three policies."""
    spec = spec or ServerSpec()
    slo = slo or SLO()
    requests = generate(WorkloadSpec(duration_s=duration_s, rate_rps=rate_rps, seed=seed))
    return [
        replay(requests, policy, spec=spec, slo=slo) for policy in default_policies(slo)
    ]


def _point(rate: float, metrics: RunMetrics) -> SweepPoint:
    critical = metrics.by_class[Criticality.CRITICAL]
    return SweepPoint(
        rate_rps=rate,
        policy=metrics.policy,
        goodput_rate=round(metrics.goodput_rate, 4),
        shed_rate=round(metrics.shed_rate, 4),
        critical_attainment=round(critical.attainment, 4),
        standard_attainment=round(metrics.attainment(Criticality.STANDARD), 4),
        sheddable_attainment=round(metrics.attainment(Criticality.SHEDDABLE), 4),
        critical_ttft_p95=round(critical.ttft_p95, 4),
    )
