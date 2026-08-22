"""Goodput and critical-class SLO attainment across a range of offered load.

The sweep is the honest version of the story. Below capacity, SLO-aware admission *costs*
you goodput: it refuses work the server could actually have retired, because its estimate is
conservative. Above capacity, that same conservatism is the only thing keeping the class you
care about inside its target.
"""

from inference_slo import ServerSpec, WorkloadSpec, sustainable_rps, sweep

capacity = sustainable_rps(ServerSpec(), WorkloadSpec())
print(f"server retires roughly {capacity:.1f} rps (decode-bound)\n")
print(f"{'rps':>5}  {'policy':<14}{'goodput':>9}{'shed':>7}{'crit SLO':>10}"
      f"{'crit TTFT p95':>15}")

last = None
for point in sweep():
    if last is not None and point.rate_rps != last:
        print()
    print(f"{point.rate_rps:>5.0f}  {point.policy:<14}{point.goodput_rate:>8.0%}"
          f"{point.shed_rate:>7.0%}{point.critical_attainment:>10.0%}"
          f"{point.critical_ttft_p95:>14.2f}s")
    last = point.rate_rps

print(
    "\nTwo things worth arguing with. Naive queue-depth shedding drops a third of the "
    "traffic and ends up with *lower* goodput than admitting everything — it sheds by depth, "
    "which is a poor proxy for time, so it sheds the wrong requests. And SLO-aware admission "
    "gives up real goodput below capacity; that is the price of the estimate being "
    "conservative, and it is only worth paying if you are ever overloaded."
)
