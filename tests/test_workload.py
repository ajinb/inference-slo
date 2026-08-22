import pytest

from inference_slo import Criticality, WorkloadSpec, generate


def test_generation_is_deterministic():
    assert generate(WorkloadSpec(seed=7)) == generate(WorkloadSpec(seed=7))


def test_different_seeds_give_different_traces():
    assert generate(WorkloadSpec(seed=1)) != generate(WorkloadSpec(seed=2))


def test_arrivals_are_ordered_and_inside_the_window():
    spec = WorkloadSpec(duration_s=30.0, rate_rps=10.0)
    requests = generate(spec)
    arrivals = [r.arrival_s for r in requests]
    assert arrivals == sorted(arrivals)
    assert all(0 <= a < spec.duration_s for a in arrivals)


def test_ids_are_dense_and_unique():
    requests = generate(WorkloadSpec())
    assert [r.id for r in requests] == list(range(len(requests)))


def test_arrival_count_tracks_the_requested_rate():
    spec = WorkloadSpec(duration_s=200.0, rate_rps=10.0, seed=3)
    expected = spec.duration_s * spec.rate_rps
    assert 0.85 * expected <= len(generate(spec)) <= 1.15 * expected


def test_class_mix_is_roughly_honoured():
    spec = WorkloadSpec(duration_s=600.0, rate_rps=10.0, mix=(0.25, 0.45, 0.30), seed=5)
    requests = generate(spec)
    share = {
        c: sum(1 for r in requests if r.criticality is c) / len(requests) for c in Criticality
    }
    assert share[Criticality.CRITICAL] == pytest.approx(0.25, abs=0.04)
    assert share[Criticality.STANDARD] == pytest.approx(0.45, abs=0.04)
    assert share[Criticality.SHEDDABLE] == pytest.approx(0.30, abs=0.04)


def test_token_counts_are_heavy_tailed():
    """A mean with no tail makes every admission policy look better than it is."""
    requests = generate(WorkloadSpec(duration_s=600.0, rate_rps=10.0, seed=11))
    lengths = sorted(r.prompt_tokens for r in requests)
    median = lengths[len(lengths) // 2]
    p99 = lengths[int(0.99 * len(lengths))]
    assert p99 > 3 * median


def test_token_counts_are_never_zero():
    requests = generate(WorkloadSpec(duration_s=300.0, rate_rps=20.0, prompt_tokens_mean=2))
    assert all(r.prompt_tokens >= 1 and r.output_tokens >= 1 for r in requests)


@pytest.mark.parametrize("kwargs", [
    {"duration_s": 0}, {"rate_rps": 0}, {"mix": (0.5, 0.4, 0.4)}, {"mix": (1.5, -0.3, -0.2)},
])
def test_invalid_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        WorkloadSpec(**kwargs)
