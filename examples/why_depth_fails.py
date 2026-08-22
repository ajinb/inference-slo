"""Why queue depth is the wrong signal.

Depth counts requests. What a request actually waits for is the *prefill work* queued ahead
of it, and prompt lengths are heavy-tailed. Two queues of identical depth can differ by an
order of magnitude in wait — so any fixed depth threshold is simultaneously too tight for one
workload and too loose for another, and the workload changes during the day.
"""

from inference_slo import ServerSpec, WorkloadSpec, generate
from inference_slo.backend import Server

spec = ServerSpec()
requests = generate(WorkloadSpec(rate_rps=12.0, seed=42))

server = Server(spec=spec)
print(f"{'depth':>7}{'queued prompt tokens':>23}{'implied prefill wait':>23}")

for n, request in enumerate(requests[:40], start=1):
    server.enqueue(request, now_s=0.0)
    if n % 8 == 0:
        state = server.state(0.0)
        wait = state.queued_prompt_tokens / spec.prefill_tokens_per_s
        print(f"{state.queue_depth:>7}{state.queued_prompt_tokens:>23,}{wait:>22.2f}s")

lengths = [r.prompt_tokens for r in requests]
print(f"\nprompt tokens: min {min(lengths):,}  median {sorted(lengths)[len(lengths) // 2]:,}"
      f"  max {max(lengths):,}")
print("A threshold on the left-hand column cannot see the right-hand column.")
