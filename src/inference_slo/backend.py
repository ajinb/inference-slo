"""A model server, modelled at the level a gateway can actually see.

Two phases, because they saturate on different resources and break in different ways:

**Prefill** is compute-bound and proportional to prompt length. It is what stands between a
request and its first token, so it owns TTFT.

**Decode** is memory-bound. Continuous batching means aggregate throughput rises with batch
size but saturates, so per-stream throughput *falls* as the batch grows — which is why TPOT
degrades for everybody when one more request is admitted.

The **KV cache** is the hard constraint. It grows with every generated token, so a request's
footprint is not known at admission time, and a server that admits on prompt length alone
discovers the problem several hundred tokens later.

See the README for what this deliberately does not model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .types import Criticality, Outcome, Request


class Phase(str, Enum):
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    DONE = "done"


@dataclass(frozen=True)
class ServerSpec:
    """Capacity of one accelerator-backed model server."""

    #: Total KV cache, in tokens.
    kv_tokens: int = 90_000
    #: Aggregate prefill throughput.
    prefill_tokens_per_s: float = 30_000.0
    #: Aggregate decode throughput across the whole batch.
    decode_tokens_per_s: float = 2_400.0
    #: Ceiling for any single stream, however empty the batch.
    per_stream_decode_tokens_per_s: float = 55.0
    max_batch: int = 48

    def __post_init__(self) -> None:
        for name in ("kv_tokens", "prefill_tokens_per_s", "decode_tokens_per_s",
                     "per_stream_decode_tokens_per_s", "max_batch"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass
class InFlight:
    """A request the server has accepted, and where it has got to."""

    request: Request
    admitted_s: float
    phase: Phase = Phase.QUEUED
    prefill_done_tokens: float = 0.0
    generated_tokens: float = 0.0
    first_token_s: float | None = None

    @property
    def kv_tokens(self) -> int:
        return self.request.prompt_tokens + int(self.generated_tokens)


@dataclass(frozen=True)
class GatewayState:
    """What an admission policy is allowed to look at.

    Every field here is something a real endpoint picker can scrape from a model server's
    metrics endpoint. A policy that needs anything else is not implementable.
    """

    now_s: float
    queue_depth: int
    queued_prompt_tokens: int
    queued_by_class: dict[Criticality, int]
    batch_size: int
    kv_used_tokens: int
    spec: ServerSpec

    @property
    def kv_utilization(self) -> float:
        return self.kv_used_tokens / self.spec.kv_tokens

    def projected_tpot_s(self, extra: int = 1) -> float:
        """Time per output token if `extra` more streams joined the batch."""
        batch = max(1, self.batch_size + extra)
        aggregate = min(
            self.spec.decode_tokens_per_s, batch * self.spec.per_stream_decode_tokens_per_s
        )
        return batch / aggregate

    def projected_ttft_s(self, prompt_tokens: int) -> float:
        """Time to first token for a request joining the back of the queue.

        Prefill is FIFO and shares one throughput budget, so everything already queued has to
        clear first. This is the estimate an endpoint picker can actually compute.
        """
        ahead = self.queued_prompt_tokens
        return (ahead + prompt_tokens) / self.spec.prefill_tokens_per_s


@dataclass
class Server:
    """One model server, stepped forward in fixed time slices."""

    spec: ServerSpec = field(default_factory=ServerSpec)
    queue: list[InFlight] = field(default_factory=list)
    running: list[InFlight] = field(default_factory=list)
    completed: list[InFlight] = field(default_factory=list)

    # ---- observable state ------------------------------------------------------------

    @property
    def kv_used_tokens(self) -> int:
        return sum(f.kv_tokens for f in self.running)

    @property
    def queued_prompt_tokens(self) -> int:
        return sum(f.request.prompt_tokens - int(f.prefill_done_tokens) for f in self.queue)

    @property
    def batch_size(self) -> int:
        return sum(1 for f in self.running if f.phase is Phase.DECODE)

    def state(self, now_s: float) -> GatewayState:
        by_class: dict[Criticality, int] = {c: 0 for c in Criticality}
        for item in self.queue:
            by_class[item.request.criticality] += 1
        return GatewayState(
            now_s=now_s,
            queue_depth=len(self.queue),
            queued_prompt_tokens=self.queued_prompt_tokens,
            queued_by_class=by_class,
            batch_size=self.batch_size,
            kv_used_tokens=self.kv_used_tokens,
            spec=self.spec,
        )

    # ---- mutation --------------------------------------------------------------------

    def enqueue(self, request: Request, now_s: float) -> InFlight:
        item = InFlight(request=request, admitted_s=now_s)
        self.queue.append(item)
        return item

    def step(self, now_s: float, dt: float) -> list[Outcome]:
        """Advance one time slice. Returns outcomes for anything that finished."""
        self._start_queued(now_s)
        self._advance_prefill(now_s, dt)
        return self._advance_decode(now_s, dt)

    def _start_queued(self, now_s: float) -> None:
        """Promote queued work into the batch while KV and batch slots allow.

        Admission reserves the *prompt* only; the KV footprint then grows with every token
        generated. That gap is the reason a server can look healthy at admission time and be
        saturated four hundred tokens later.
        """
        while self.queue:
            if len(self.running) >= self.spec.max_batch:
                return
            candidate = self.queue[0]
            if self.kv_used_tokens + candidate.request.prompt_tokens > self.spec.kv_tokens:
                return
            self.queue.pop(0)
            candidate.phase = Phase.PREFILL
            self.running.append(candidate)

    def _advance_prefill(self, now_s: float, dt: float) -> None:
        prefilling = [f for f in self.running if f.phase is Phase.PREFILL]
        if not prefilling:
            return

        share = (self.spec.prefill_tokens_per_s * dt) / len(prefilling)
        for item in prefilling:
            item.prefill_done_tokens += share
            if item.prefill_done_tokens >= item.request.prompt_tokens:
                item.phase = Phase.DECODE
                item.first_token_s = now_s + dt

    def _advance_decode(self, now_s: float, dt: float) -> list[Outcome]:
        decoding = [f for f in self.running if f.phase is Phase.DECODE]
        if not decoding:
            return []

        batch = len(decoding)
        aggregate = min(
            self.spec.decode_tokens_per_s, batch * self.spec.per_stream_decode_tokens_per_s
        )
        per_stream = (aggregate / batch) * dt

        finished: list[Outcome] = []
        for item in decoding:
            item.generated_tokens += per_stream
            if item.generated_tokens >= item.request.output_tokens:
                item.phase = Phase.DONE
                finished.append(self._complete(item, now_s + dt))

        if finished:
            done = {id(o.request) for o in finished}
            self.running = [f for f in self.running if id(f.request) not in done]
            self.completed.extend(f for f in decoding if f.phase is Phase.DONE)

        return finished

    def _complete(self, item: InFlight, now_s: float) -> Outcome:
        first_token_s = item.first_token_s if item.first_token_s is not None else now_s
        decode_s = max(now_s - first_token_s, 1e-9)
        return Outcome(
            request=item.request,
            admitted=True,
            queued_s=max(0.0, first_token_s - item.admitted_s),
            ttft_s=max(0.0, first_token_s - item.request.arrival_s),
            tpot_s=decode_s / item.request.output_tokens,
            completed_s=now_s,
        )
