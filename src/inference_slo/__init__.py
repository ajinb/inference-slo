"""inference-slo — SLO-aware admission control for LLM inference serving."""

from .backend import GatewayState, Phase, Server, ServerSpec
from .metrics import ClassMetrics, RunMetrics, percentile, summarise
from .policies import (
    Admission,
    AdmissionPolicy,
    AdmitAll,
    QueueDepthShedding,
    SLOAwareAdmission,
    default_policies,
)
from .replay import ReplayConfig, replay
from .slo import SLO, Target
from .study import SweepPoint, headline, sustainable_rps, sweep
from .types import SHED_ORDER, Criticality, Outcome, Request
from .workload import WorkloadSpec, generate

__version__ = "0.1.0"

__all__ = [
    "SHED_ORDER", "SLO", "Admission", "AdmissionPolicy", "AdmitAll", "ClassMetrics",
    "Criticality", "GatewayState", "Outcome", "Phase", "QueueDepthShedding", "ReplayConfig",
    "Request", "RunMetrics", "SLOAwareAdmission", "Server", "ServerSpec", "SweepPoint",
    "Target", "WorkloadSpec", "default_policies", "generate", "headline", "percentile",
    "replay", "summarise", "sustainable_rps", "sweep",
]
