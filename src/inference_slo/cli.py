"""inference-slo command-line interface."""

from __future__ import annotations

import argparse
import json
import sys

from .backend import ServerSpec
from .study import headline, sustainable_rps, sweep
from .types import Criticality
from .workload import WorkloadSpec


def cmd_demo(args) -> int:
    runs = headline(rate_rps=args.rate)
    if args.json:
        print(json.dumps([{
            "policy": r.policy, "offered": r.offered, "goodput": r.goodput,
            "goodput_rate": r.goodput_rate, "shed_rate": r.shed_rate,
            "by_class": {c.value: vars(m) | {"criticality": c.value}
                         for c, m in r.by_class.items()},
        } for r in runs], indent=2, default=str))
        return 0

    capacity = sustainable_rps(ServerSpec(), WorkloadSpec())
    print(f"inference-slo — {args.rate:g} rps offered against a server that retires "
          f"~{capacity:.1f} rps\n")
    print(f"{'policy':<14}{'goodput':>9}{'shed':>7}"
          f"{'crit SLO':>10}{'std SLO':>9}{'shed SLO':>10}{'crit TTFT p95':>15}")
    for run in runs:
        print(f"{run.policy:<14}{run.goodput_rate:>8.0%}{run.shed_rate:>7.0%}"
              f"{run.attainment(Criticality.CRITICAL):>10.0%}"
              f"{run.attainment(Criticality.STANDARD):>9.0%}"
              f"{run.attainment(Criticality.SHEDDABLE):>10.0%}"
              f"{run.by_class[Criticality.CRITICAL].ttft_p95:>14.2f}s")
    print("\nAdmitting everything protects nothing: the queue fills with work the server "
          "will never retire in time, and every class misses together.")
    return 0


def cmd_sweep(args) -> int:
    points = sweep()
    if args.json:
        print(json.dumps([vars(p) for p in points], indent=2))
        return 0

    capacity = sustainable_rps(ServerSpec(), WorkloadSpec())
    print(f"load sweep — server retires ~{capacity:.1f} rps\n")
    print(f"{'rps':>5}  {'policy':<14}{'goodput':>9}{'shed':>7}{'crit SLO':>10}{'std SLO':>9}")
    last = None
    for point in points:
        if last is not None and point.rate_rps != last:
            print()
        marker = "  <- capacity" if abs(point.rate_rps - capacity) < 1.0 else ""
        print(f"{point.rate_rps:>5.0f}  {point.policy:<14}{point.goodput_rate:>8.0%}"
              f"{point.shed_rate:>7.0%}{point.critical_attainment:>10.0%}"
              f"{point.standard_attainment:>9.0%}{marker}")
        last = point.rate_rps
    return 0


def cmd_classes(args) -> int:
    runs = headline(rate_rps=args.rate)
    if args.json:
        print(json.dumps([{
            "policy": r.policy,
            "classes": {c.value: vars(m) | {"criticality": c.value}
                        for c, m in r.by_class.items()},
        } for r in runs], indent=2, default=str))
        return 0

    for run in runs:
        print(f"\n{run.policy}")
        print(f"  {'class':<12}{'offered':>8}{'shed':>7}{'in SLO':>8}"
              f"{'attain':>8}{'ttft p95':>10}{'tpot p95':>10}")
        for criticality in Criticality:
            m = run.by_class[criticality]
            print(f"  {criticality.value:<12}{m.offered:>8}{m.shed:>7}{m.within_slo:>8}"
                  f"{m.attainment:>8.0%}{m.ttft_p95:>9.2f}s{m.tpot_p95 * 1000:>9.0f}ms")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="machine-readable output")
    common.add_argument("--rate", type=float, default=argparse.SUPPRESS,
                        help="offered load in rps")

    parser = argparse.ArgumentParser(
        prog="inference-slo",
        description="SLO-aware admission control for LLM inference. All runs are offline.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo", parents=[common], help="three policies at one overloaded point")
    sub.add_parser("sweep", parents=[common], help="goodput across a range of offered load")
    sub.add_parser("classes", parents=[common], help="per-class breakdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.json = getattr(args, "json", False)
    args.rate = getattr(args, "rate", 12.0)
    handlers = {"demo": cmd_demo, "sweep": cmd_sweep, "classes": cmd_classes, None: cmd_demo}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
