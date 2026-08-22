"""Admission policies.

The Gateway API Inference Extension routes on model awareness and exposes per-request
criticality. What it leaves to you is the decision this module is about: under saturation,
*which* request do you refuse, and on what evidence?

Every policy here is a pure function of the request and the gateway state — no lookahead, no
backend internals a real endpoint picker could not scrape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .backend import GatewayState
from .slo import SLO
from .types import Criticality, Request


@dataclass(frozen=True)
class Admission:
    admit: bool
    reason: str = ""

    @staticmethod
    def yes() -> Admission:
        return Admission(True)

    @staticmethod
    def no(reason: str) -> Admission:
        return Admission(False, reason)


class AdmissionPolicy(Protocol):
    name: str

    def decide(self, request: Request, state: GatewayState) -> Admission: ...


@dataclass
class AdmitAll:
    """Take everything. What plain load balancing gives you, and the baseline to beat.

    Nothing is refused, so nothing is protected: under overload every class degrades together
    and the queue absorbs work the server will never clear in time.
    """

    name: str = "admit-all"

    def decide(self, request: Request, state: GatewayState) -> Admission:
        return Admission.yes()


@dataclass
class QueueDepthShedding:
    """Shed by class once the queue passes a fixed depth. The obvious implementation.

    It is a real improvement on admitting everything, and it has a structural flaw: queue
    *depth* is a poor proxy for queue *time*. Twenty short prompts and twenty long ones are
    the same depth and wildly different waits, so the threshold is either too tight under one
    workload or too loose under the other — and the workload changes during the day.
    """

    max_queue: int = 24
    #: Classes at or below this rank are protected from depth-based shedding.
    protect_rank: int = 0
    name: str = "queue-depth"

    def decide(self, request: Request, state: GatewayState) -> Admission:
        if request.criticality.rank <= self.protect_rank:
            return Admission.yes()
        if state.queue_depth >= self.max_queue:
            return Admission.no(f"queue depth {state.queue_depth} >= {self.max_queue}")
        return Admission.yes()


@dataclass
class SLOAwareAdmission:
    """Admit only if this request's own SLO can still be met, and reserve headroom above it.

    Two rules, in order:

    1. **Headroom.** Non-critical traffic is refused once KV utilisation passes
       `1 - critical_headroom`. Capacity kept clear of batch work is capacity available to the
       interactive traffic that pays for it — and the reservation has to be enforced *before*
       saturation, because after it there is nothing left to reserve.

    2. **Predicted attainment.** Estimate TTFT from the prefill work already queued and TPOT
       from what one more stream does to the batch. If the request cannot meet its own target,
       refuse it now rather than admitting it to miss later.

    Rule 2 is the one that matters. Admitting a request you can already predict will breach
    is strictly worse than refusing it: the client waits *and* the work displaces something
    that would have made its target. A fast refusal is a better answer than a slow failure.
    """

    slo: SLO = field(default_factory=SLO)
    name: str = "slo-aware"

    def decide(self, request: Request, state: GatewayState) -> Admission:
        if request.criticality is not Criticality.CRITICAL:
            ceiling = 1.0 - self.slo.critical_headroom
            if state.kv_utilization >= ceiling:
                return Admission.no(
                    f"KV {state.kv_utilization:.0%} >= {ceiling:.0%} reserved for critical"
                )

        target = self.slo.target_for(request.criticality)
        ttft = state.projected_ttft_s(request.prompt_tokens)
        if ttft > target.ttft_s:
            return Admission.no(f"predicted TTFT {ttft:.2f}s > {target.ttft_s:.2f}s")

        tpot = state.projected_tpot_s()
        if tpot > target.tpot_s:
            return Admission.no(f"predicted TPOT {tpot * 1000:.0f}ms > {target.tpot_s * 1000:.0f}ms")

        return Admission.yes()


def default_policies(slo: SLO | None = None) -> list[AdmissionPolicy]:
    """The three arms of the study, worst to best."""
    return [AdmitAll(), QueueDepthShedding(), SLOAwareAdmission(slo=slo or SLO())]
