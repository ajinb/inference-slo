"""Core types for the inference workload."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Criticality(str, Enum):
    """Per-request criticality, as the Gateway API Inference Extension exposes it.

    The extension gives you the *primitive*. What it does not give you is the policy: which
    of these to shed, at what point, against which target. That gap is what this repo is about.
    """

    #: Interactive, user-facing. Shedding this is a visible outage.
    CRITICAL = "critical"
    #: Normal serving traffic.
    STANDARD = "standard"
    #: Batch, evaluation, background enrichment. Latency-insensitive by construction.
    SHEDDABLE = "sheddable"

    @property
    def rank(self) -> int:
        return _RANK[self]


_RANK = {Criticality.CRITICAL: 0, Criticality.STANDARD: 1, Criticality.SHEDDABLE: 2}

#: Least important first — the order a shedding policy walks.
SHED_ORDER: tuple[Criticality, ...] = (
    Criticality.SHEDDABLE, Criticality.STANDARD, Criticality.CRITICAL,
)


@dataclass(frozen=True)
class Request:
    """One inference request."""

    id: int
    arrival_s: float
    criticality: Criticality
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if self.prompt_tokens < 1 or self.output_tokens < 1:
            raise ValueError("token counts must be positive")


@dataclass
class Outcome:
    """What happened to one request.

    `ttft_s` is time to first token — queue wait plus prefill. `tpot_s` is time per output
    token during decode. Those two are the SLIs that matter for inference serving; a single
    end-to-end latency number hides which half is broken.
    """

    request: Request
    admitted: bool
    shed_reason: str | None = None
    queued_s: float = 0.0
    ttft_s: float = 0.0
    tpot_s: float = 0.0
    completed_s: float | None = None

    @property
    def completed(self) -> bool:
        return self.completed_s is not None
