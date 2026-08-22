"""The claims the README makes. If these break, the README is wrong."""

import pytest

from inference_slo import (
    Criticality,
    ServerSpec,
    WorkloadSpec,
    headline,
    sustainable_rps,
    sweep,
)


@pytest.fixture(scope="module")
def runs():
    return headline(rate_rps=12.0)


@pytest.fixture(scope="module")
def by_policy(runs):
    return {r.policy: r for r in runs}


def test_capacity_is_decode_bound():
    capacity = sustainable_rps(ServerSpec(), WorkloadSpec())
    assert 6.0 < capacity < 9.0


def test_the_headline_point_is_genuinely_overloaded(by_policy):
    assert by_policy["admit-all"].goodput_rate < 0.5


def test_slo_aware_beats_both_baselines_on_goodput(by_policy):
    slo_aware = by_policy["slo-aware"].goodput_rate
    assert slo_aware > by_policy["admit-all"].goodput_rate
    assert slo_aware > by_policy["queue-depth"].goodput_rate


def test_slo_aware_holds_critical_attainment(by_policy):
    assert by_policy["slo-aware"].attainment(Criticality.CRITICAL) == 1.0


def test_admitting_everything_protects_nothing(by_policy):
    """Every class degrades together — that is the failure mode."""
    run = by_policy["admit-all"]
    assert run.shed_rate == 0.0
    assert run.attainment(Criticality.CRITICAL) < 0.3
    assert run.attainment(Criticality.STANDARD) < 0.3


def test_naive_depth_shedding_sheds_a_lot_and_buys_nothing(by_policy):
    """The counterintuitive result: shedding by depth is not better than not shedding."""
    depth = by_policy["queue-depth"]
    admit_all = by_policy["admit-all"]
    assert depth.shed_rate > 0.25
    assert depth.goodput_rate <= admit_all.goodput_rate
    assert depth.attainment(Criticality.CRITICAL) == pytest.approx(
        admit_all.attainment(Criticality.CRITICAL), abs=0.05
    )


def test_slo_aware_keeps_critical_ttft_inside_its_target(by_policy):
    from inference_slo import SLO
    p95 = by_policy["slo-aware"].by_class[Criticality.CRITICAL].ttft_p95
    assert p95 <= SLO().target_for(Criticality.CRITICAL).ttft_s


def test_results_are_deterministic():
    first = [(r.policy, r.goodput) for r in headline(rate_rps=12.0)]
    second = [(r.policy, r.goodput) for r in headline(rate_rps=12.0)]
    assert first == second


@pytest.fixture(scope="module")
def points():
    return sweep()


class TestSweep:
    def test_covers_every_rate_and_policy(self, points):
        assert len({p.rate_rps for p in points}) == 6
        assert len({p.policy for p in points}) == 3

    def test_below_capacity_everything_works(self, points):
        for point in [p for p in points if p.rate_rps <= 4.0]:
            assert point.goodput_rate > 0.95

    def test_slo_aware_costs_goodput_below_capacity(self, points):
        """The honest cost of a conservative estimate. Do not hide this."""
        under = {p.policy: p for p in points if p.rate_rps == 6.0}
        assert under["slo-aware"].goodput_rate < under["admit-all"].goodput_rate

    def test_slo_aware_wins_at_and_above_capacity(self, points):
        for rate in (8.0, 12.0, 16.0, 20.0):
            arms = {p.policy: p for p in points if p.rate_rps == rate}
            assert arms["slo-aware"].goodput_rate > arms["admit-all"].goodput_rate, rate

    def test_critical_attainment_holds_all_the_way_up(self, points):
        for point in [p for p in points if p.policy == "slo-aware"]:
            assert point.critical_attainment >= 0.99, point.rate_rps

    def test_baseline_critical_attainment_collapses_under_load(self, points):
        heavy = [p for p in points if p.policy == "admit-all" and p.rate_rps >= 12.0]
        assert all(p.critical_attainment < 0.2 for p in heavy)

    def test_shedding_rises_monotonically_with_load(self, points):
        rates = [p.shed_rate for p in sorted(
            (p for p in points if p.policy == "slo-aware"), key=lambda p: p.rate_rps
        )]
        assert rates == sorted(rates)
