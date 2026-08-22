# inference-slo

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

> **SLO-aware admission control for LLM inference serving.** The Gateway API Inference
> Extension gives you per-request criticality. It does not tell you what to shed, when, or
> against which target when the accelerator saturates. This is a deterministic replay harness
> for answering that.

Classic web load-shedding assumes short, stateless, uniform requests. Inference is none of
those: a request is long-running, partially stateful, and its KV-cache footprint **grows with
every token it generates** — so its cost is not known at admission time. Admitting on queue
depth, or on nothing at all, breaks in ways the web playbook does not predict.

## The 30-second demo (offline, no cluster, no GPU)

```bash
inference-slo demo
```

```
inference-slo — 12 rps offered against a server that retires ~7.5 rps

policy          goodput   shed  crit SLO  std SLO  shed SLO  crit TTFT p95
admit-all           32%     0%       11%      12%       83%         35.48s
queue-depth         30%    35%       11%      25%       56%          5.53s
slo-aware           52%    48%      100%      36%       36%          0.25s
```

Same requests, same server, same seed. Only the admission policy changes.

**Two results worth arguing with:**

1. **Admitting everything protects nothing.** Zero shed, and every class misses together —
   critical TTFT p95 lands at 35 seconds against a 1-second target. The queue fills with work
   the server will never retire in time, and the class that pays for the cluster suffers
   exactly as much as the batch job.

2. **Naive queue-depth shedding is not an improvement.** It drops 35% of the traffic and gets
   *lower* goodput than admitting everything, with critical attainment unchanged at 11%. It
   sheds by depth, and depth is a poor proxy for time:

   ```bash
   python examples/why_depth_fails.py
   ```
   ```
     depth   queued prompt tokens   implied prefill wait
         8                 16,442                  0.55s
        40                 66,070                  2.20s

   prompt tokens: min 233  median 1,468  max 11,036
   ```

   A threshold on the left-hand column cannot see the right-hand column. Prompt lengths are
   heavy-tailed, so any fixed depth is simultaneously too tight for one workload and too loose
   for another — and the workload changes during the day.

## What SLO-aware admission does

Two rules, in order:

**Headroom.** Non-critical traffic is refused once KV utilisation passes `1 - headroom`
(default 25%). Capacity kept clear of batch work is capacity available to the interactive
traffic that pays for it, and the reservation has to be enforced *before* saturation — after
it, there is nothing left to reserve.

**Predicted attainment.** Estimate TTFT from the prefill work already queued, and TPOT from
what one more stream does to the batch. If the request cannot meet *its own class's* target,
refuse it now.

The second rule is the one that matters. Admitting a request you can already predict will
breach is strictly worse than refusing it: the client waits *and* the work displaces something
that would have made its target. **A fast refusal is a better answer than a slow failure.**

Every input is something a real endpoint picker can scrape from a model server's metrics
endpoint. There is a test that pins this — a policy needing backend internals or lookahead is
not implementable, and this repo will not pretend otherwise.

## The honest version: the load sweep

```bash
inference-slo sweep
```

```
  rps  policy          goodput   shed  crit SLO  std SLO
    4  admit-all         100%     0%      100%     100%
    4  slo-aware          99%     1%      100%      99%

    6  admit-all         100%     0%       99%     100%
    6  slo-aware          86%    14%      100%      83%      <- SLO-aware costs 14% here

    8  admit-all          47%     0%       18%      29%      <- capacity, ~7.5 rps
    8  slo-aware          70%    30%      100%      60%

   12  admit-all          32%     0%       11%      12%
   12  slo-aware          52%    48%      100%      36%

   20  admit-all          13%     0%        6%       6%
   20  slo-aware          34%    66%      100%      12%
```

**Below capacity, SLO-aware admission loses.** At 6 rps it gives up 14% of goodput refusing
work the server could actually have retired, because the estimate is conservative. That cost
is real and it is only worth paying if you are ever overloaded. Above capacity, the same
conservatism is the only thing holding critical attainment at 100% while everything else
degrades.

Stating that trade-off is the point of the sweep. A harness that only showed the operating
point where the new policy wins would not be worth running.

## What this model does and does not capture

It is a **queueing model of the bottleneck structure**, not a simulator of vLLM. It exists to
reason about admission policy, and it earns that by getting the shape right:

**Modelled** — two-phase serving (prefill is compute-bound and owns TTFT; decode is
memory-bound and owns TPOT) · continuous batching, where aggregate throughput saturates so
per-stream throughput *falls* as the batch grows · KV cache as the hard constraint, growing
with every generated token · heavy-tailed prompt and output lengths · Poisson arrivals ·
per-class SLOs on TTFT and TPOT.

**Not modelled** — chunked prefill (prefill here shares one throughput budget FIFO) ·
preemption and recompute of running sequences under KV pressure · prefix/KV cache reuse across
requests · prefill/decode disaggregation and tiered KV offload, as in llm-d · multi-replica
routing and autoscaling (one server, so this is about admission and not placement) ·
speculative decoding · quantisation effects.

Several of those would move the numbers. **None of them changes the direction of the result**,
which is that shedding on a latency prediction beats shedding on queue depth, which is roughly
tied with not shedding at all. Treat the percentages as a demonstration of mechanism, not as a
capacity plan for your cluster.

## Using it

```python
from inference_slo import (
    SLO, ServerSpec, SLOAwareAdmission, WorkloadSpec, generate, replay,
)

requests = generate(WorkloadSpec(duration_s=60, rate_rps=12.0, seed=42))
metrics = replay(
    requests,
    SLOAwareAdmission(slo=SLO(critical_headroom=0.25)),
    spec=ServerSpec(kv_tokens=90_000, decode_tokens_per_s=2_400),
)

print(metrics.goodput_rate, metrics.attainment(Criticality.CRITICAL))
```

Write your own policy by implementing `decide(request, state) -> Admission`:

```python
class MyPolicy:
    name = "mine"

    def decide(self, request, state):
        if state.projected_ttft_s(request.prompt_tokens) > 2.0:
            return Admission.no("too slow")
        return Admission.yes()
```

## Roadmap

**v0.1 (this)** — the replay harness and three policies. Shipped first because it is the
citable half: a deterministic study anyone can rerun and disagree with.

**v0.2** — the same `AdmissionPolicy` interface as an endpoint-picker plugin for the
[Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension),
so a policy validated here can run in a cluster unchanged.

## Design notes

- **Deterministic end to end.** Every source of randomness takes an explicit seed and the same
  seed gives identical results everywhere. There is a test that pins this.
- **Goodput, not throughput.** A server that completes a thousand requests and misses the
  target on all of them has done no useful work.
- **Attainment is measured against offered load, not admitted load** — otherwise a policy
  scores perfectly by shedding almost everything.
- **Zero runtime dependencies.** Percentiles are nearest-rank throughout.

```bash
pytest -q                       # 88 tests
ruff check src tests examples
inference-slo demo
python examples/load_sweep.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Fidelity claims must be honest: if you add a serving
behaviour, say plainly what it approximates and where the approximation breaks.

## License

Apache 2.0. More at [cloudandsre.com](https://cloudandsre.com).
