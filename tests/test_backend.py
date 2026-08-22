import pytest

from inference_slo import Criticality, Request, Server, ServerSpec
from inference_slo.backend import Phase


def _request(rid=0, prompt=1000, output=100, criticality=Criticality.STANDARD):
    return Request(id=rid, arrival_s=0.0, criticality=criticality,
                   prompt_tokens=prompt, output_tokens=output)


@pytest.fixture
def server():
    return Server(spec=ServerSpec())


@pytest.mark.parametrize("field", ["kv_tokens", "prefill_tokens_per_s",
                                   "decode_tokens_per_s", "max_batch"])
def test_capacities_must_be_positive(field):
    with pytest.raises(ValueError, match=field):
        ServerSpec(**{field: 0})


def test_a_request_walks_queued_to_prefill_to_decode_to_done(server):
    server.enqueue(_request(), now_s=0.0)
    assert server.queue[0].phase is Phase.QUEUED

    server.step(0.0, 0.02)
    assert server.running[0].phase in (Phase.PREFILL, Phase.DECODE)

    outcomes = []
    now = 0.02
    for _ in range(5_000):
        outcomes.extend(server.step(now, 0.02))
        now += 0.02
        if outcomes:
            break
    assert len(outcomes) == 1
    assert outcomes[0].completed


def test_ttft_covers_queue_wait_plus_prefill(server):
    server.enqueue(_request(prompt=30_000), now_s=0.0)
    now = 0.0
    outcome = None
    for _ in range(20_000):
        found = server.step(now, 0.02)
        now += 0.02
        if found:
            outcome = found[0]
            break
    assert outcome is not None
    # 30k prompt tokens at 30k tokens/s is about a second of prefill alone.
    assert outcome.ttft_s >= 0.9


def test_kv_pressure_keeps_work_queued(server):
    """The KV cache is the hard constraint; a full cache means nothing else starts."""
    for i in range(200):
        server.enqueue(_request(rid=i, prompt=5_000), now_s=0.0)
    server.step(0.0, 0.02)
    assert server.queue, "the whole backlog should not fit in KV at once"
    assert server.kv_used_tokens <= server.spec.kv_tokens


def test_batch_size_is_capped(server):
    for i in range(500):
        server.enqueue(_request(rid=i, prompt=10), now_s=0.0)
    for _ in range(50):
        server.step(0.0, 0.02)
    assert len(server.running) <= server.spec.max_batch


def test_per_stream_decode_rate_falls_as_the_batch_grows():
    """Continuous batching: aggregate throughput saturates, so each stream gets less."""
    state_small = Server(spec=ServerSpec()).state(0.0)
    tpot_small = state_small.projected_tpot_s(extra=1)
    tpot_large = state_small.projected_tpot_s(extra=200)
    assert tpot_large > tpot_small


def test_projected_ttft_grows_with_queued_work(server):
    empty = server.state(0.0).projected_ttft_s(1_000)
    for i in range(20):
        server.enqueue(_request(rid=i, prompt=4_000), now_s=0.0)
    loaded = server.state(0.0).projected_ttft_s(1_000)
    assert loaded > empty


def test_gateway_state_counts_the_queue_by_class(server):
    server.enqueue(_request(rid=0, criticality=Criticality.CRITICAL), 0.0)
    server.enqueue(_request(rid=1, criticality=Criticality.SHEDDABLE), 0.0)
    server.enqueue(_request(rid=2, criticality=Criticality.SHEDDABLE), 0.0)
    by_class = server.state(0.0).queued_by_class
    assert by_class[Criticality.CRITICAL] == 1
    assert by_class[Criticality.SHEDDABLE] == 2
    assert by_class[Criticality.STANDARD] == 0


def test_kv_utilisation_is_a_fraction(server):
    assert server.state(0.0).kv_utilization == 0.0
    server.enqueue(_request(prompt=1_000), 0.0)
    server.step(0.0, 0.02)
    assert 0.0 < server.state(0.0).kv_utilization < 1.0
