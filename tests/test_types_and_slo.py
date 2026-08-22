import pytest

from inference_slo import SHED_ORDER, SLO, Criticality, Request, Target


def test_shed_order_runs_least_important_first():
    assert SHED_ORDER == (Criticality.SHEDDABLE, Criticality.STANDARD, Criticality.CRITICAL)


def test_criticality_ranks_ascend_with_expendability():
    assert Criticality.CRITICAL.rank < Criticality.STANDARD.rank < Criticality.SHEDDABLE.rank


@pytest.mark.parametrize("kwargs", [{"prompt_tokens": 0}, {"output_tokens": 0},
                                    {"prompt_tokens": -1}])
def test_requests_need_positive_token_counts(kwargs):
    base = {"id": 1, "arrival_s": 0.0, "criticality": Criticality.STANDARD,
            "prompt_tokens": 10, "output_tokens": 10}
    with pytest.raises(ValueError):
        Request(**(base | kwargs))


class TestTarget:
    def test_both_dimensions_must_be_met(self):
        target = Target(ttft_s=1.0, tpot_s=0.05)
        assert target.met_by(0.9, 0.04)
        assert not target.met_by(1.1, 0.04)
        assert not target.met_by(0.9, 0.06)

    def test_boundary_counts_as_met(self):
        assert Target(ttft_s=1.0, tpot_s=0.05).met_by(1.0, 0.05)

    @pytest.mark.parametrize("kwargs", [{"ttft_s": 0}, {"tpot_s": -1}])
    def test_targets_must_be_positive(self, kwargs):
        with pytest.raises(ValueError):
            Target(**({"ttft_s": 1.0, "tpot_s": 0.05} | kwargs))


class TestSLO:
    def test_defaults_get_stricter_as_criticality_rises(self):
        slo = SLO()
        ttfts = [slo.target_for(c).ttft_s for c in
                 (Criticality.CRITICAL, Criticality.STANDARD, Criticality.SHEDDABLE)]
        assert ttfts == sorted(ttfts), "a more critical class must have a tighter target"

    def test_every_class_needs_a_target(self):
        with pytest.raises(ValueError, match="no target"):
            SLO(targets={Criticality.CRITICAL: Target(1.0, 0.05)})

    @pytest.mark.parametrize("headroom", [-0.1, 1.0, 1.5])
    def test_headroom_bounds(self, headroom):
        with pytest.raises(ValueError, match="critical_headroom"):
            SLO(critical_headroom=headroom)

    def test_met_delegates_to_the_right_class(self):
        slo = SLO()
        # 2s TTFT breaches critical but not standard.
        assert not slo.met(Criticality.CRITICAL, 2.0, 0.01)
        assert slo.met(Criticality.STANDARD, 2.0, 0.01)
