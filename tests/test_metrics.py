import pytest

from inference_slo import SLO, Criticality, Outcome, Request, percentile, summarise


def test_percentile_uses_nearest_rank():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 1.0) == 5.0
    assert percentile(values, 0.2) == 1.0


def test_percentile_of_nothing_is_zero():
    assert percentile([], 0.95) == 0.0


@pytest.mark.parametrize("p", [0.0, -0.1, 1.1])
def test_percentile_rejects_invalid_p(p):
    with pytest.raises(ValueError):
        percentile([1.0], p)


def _outcome(rid, criticality, *, admitted=True, ttft=0.1, tpot=0.01, completed=True):
    request = Request(id=rid, arrival_s=0.0, criticality=criticality,
                      prompt_tokens=100, output_tokens=100)
    return Outcome(request=request, admitted=admitted, ttft_s=ttft, tpot_s=tpot,
                   completed_s=1.0 if completed else None)


def test_goodput_counts_only_completions_inside_slo():
    outcomes = [
        _outcome(0, Criticality.CRITICAL, ttft=0.1),    # inside
        _outcome(1, Criticality.CRITICAL, ttft=99.0),   # completed, missed
    ]
    metrics = summarise("p", outcomes, SLO())
    assert metrics.completed == 2
    assert metrics.goodput == 1


def test_attainment_is_measured_against_offered_not_admitted():
    """Otherwise a policy scores perfectly by shedding almost everything."""
    outcomes = [_outcome(0, Criticality.CRITICAL, ttft=0.1)]
    outcomes += [
        _outcome(i, Criticality.CRITICAL, admitted=False, completed=False)
        for i in range(1, 10)
    ]
    metrics = summarise("p", outcomes, SLO())
    assert metrics.by_class[Criticality.CRITICAL].attainment == pytest.approx(0.1)


def test_shed_requests_never_count_as_goodput():
    outcomes = [_outcome(i, Criticality.SHEDDABLE, admitted=False, completed=False)
                for i in range(5)]
    metrics = summarise("p", outcomes, SLO())
    assert metrics.goodput == 0
    assert metrics.shed_rate == 1.0


def test_admitted_but_never_completed_counts_as_a_miss():
    outcomes = [_outcome(0, Criticality.CRITICAL, completed=False)]
    metrics = summarise("p", outcomes, SLO())
    assert metrics.admitted == 1
    assert metrics.completed == 0
    assert metrics.goodput == 0


def test_every_class_appears_even_when_empty():
    metrics = summarise("p", [_outcome(0, Criticality.CRITICAL)], SLO())
    assert set(metrics.by_class) == set(Criticality)
    assert metrics.by_class[Criticality.SHEDDABLE].offered == 0
    assert metrics.by_class[Criticality.SHEDDABLE].attainment == 0.0
