"""The replay harness.

Fixed-slice simulation over a deterministic arrival trace. Every policy sees the *same*
requests in the same order, so a difference in the results is a difference in the policy and
nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

from .backend import Server, ServerSpec
from .metrics import RunMetrics, summarise
from .policies import AdmissionPolicy
from .slo import SLO
from .types import Outcome, Request


@dataclass(frozen=True)
class ReplayConfig:
    #: Simulation slice. Small enough that prefill and decode interleave believably.
    dt: float = 0.02
    #: Time allowed after the last arrival for in-flight work to drain.
    drain_s: float = 120.0

    def __post_init__(self) -> None:
        if self.dt <= 0 or self.drain_s < 0:
            raise ValueError("dt must be positive and drain_s non-negative")


def replay(
    requests: list[Request],
    policy: AdmissionPolicy,
    *,
    spec: ServerSpec | None = None,
    slo: SLO | None = None,
    config: ReplayConfig | None = None,
) -> RunMetrics:
    """Run one workload against one policy."""
    spec = spec or ServerSpec()
    slo = slo or SLO()
    config = config or ReplayConfig()

    server = Server(spec=spec)
    arrivals = sorted(requests, key=lambda r: (r.arrival_s, r.id))
    outcomes: dict[int, Outcome] = {}

    now = 0.0
    index = 0
    horizon = (arrivals[-1].arrival_s if arrivals else 0.0) + config.drain_s

    while now < horizon:
        while index < len(arrivals) and arrivals[index].arrival_s <= now:
            request = arrivals[index]
            index += 1
            decision = policy.decide(request, server.state(now))
            if decision.admit:
                server.enqueue(request, now)
            else:
                outcomes[request.id] = Outcome(
                    request=request, admitted=False, shed_reason=decision.reason
                )

        for outcome in server.step(now, config.dt):
            outcomes[outcome.request.id] = outcome

        now += config.dt
        if index >= len(arrivals) and not server.queue and not server.running:
            break

    # Anything still in flight when the clock runs out never completed. It counts as admitted
    # and missed — which is the honest accounting, and the reason admitting everything scores
    # badly rather than merely looking slow.
    for item in list(server.queue) + list(server.running):
        outcomes.setdefault(item.request.id, Outcome(request=item.request, admitted=True))

    return summarise(policy.name, [outcomes[r.id] for r in arrivals], slo)
