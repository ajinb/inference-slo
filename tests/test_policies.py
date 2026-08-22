import pytest

from inference_slo import (
    SLO,
    AdmitAll,
    Criticality,
    QueueDepthShedding,
    Request,
    Server,
    ServerSpec,
    SLOAwareAdmission,
)


def _request(criticality=Criticality.STANDARD, prompt=1_000):
    return Request(id=0, arrival_s=0.0, criticality=criticality,
                   prompt_tokens=prompt, output_tokens=100)


def _server_with(queued=0, prompt=2_000):
    server = Server(spec=ServerSpec())
    for i in range(queued):
        server.enqueue(
            Request(id=i, arrival_s=0.0, criticality=Criticality.STANDARD,
                    prompt_tokens=prompt, output_tokens=100),
            now_s=0.0,
        )
    return server


class TestAdmitAll:
    @pytest.mark.parametrize("criticality", list(Criticality))
    def test_admits_regardless_of_pressure(self, criticality):
        state = _server_with(queued=500).state(0.0)
        assert AdmitAll().decide(_request(criticality), state).admit


class TestQueueDepthShedding:
    def test_admits_below_the_threshold(self):
        policy = QueueDepthShedding(max_queue=10)
        assert policy.decide(_request(), _server_with(queued=5).state(0.0)).admit

    def test_sheds_at_the_threshold(self):
        policy = QueueDepthShedding(max_queue=10)
        decision = policy.decide(_request(), _server_with(queued=10).state(0.0))
        assert not decision.admit
        assert "queue depth" in decision.reason

    def test_critical_is_protected_from_depth_shedding(self):
        policy = QueueDepthShedding(max_queue=10)
        state = _server_with(queued=500).state(0.0)
        assert policy.decide(_request(Criticality.CRITICAL), state).admit

    def test_depth_is_blind_to_the_work_behind_it(self):
        """The structural flaw: identical depth, wildly different wait."""
        policy = QueueDepthShedding(max_queue=10)
        short = _server_with(queued=9, prompt=200).state(0.0)
        long = _server_with(queued=9, prompt=20_000).state(0.0)
        assert policy.decide(_request(), short).admit
        assert policy.decide(_request(), long).admit
        assert long.projected_ttft_s(1_000) > 10 * short.projected_ttft_s(1_000)


class TestSLOAwareAdmission:
    def test_admits_on_an_idle_server(self):
        policy = SLOAwareAdmission()
        for criticality in Criticality:
            assert policy.decide(_request(criticality), _server_with().state(0.0)).admit

    def test_sheds_when_predicted_ttft_breaches_the_target(self):
        policy = SLOAwareAdmission()
        state = _server_with(queued=40, prompt=4_000).state(0.0)
        decision = policy.decide(_request(Criticality.CRITICAL), state)
        assert not decision.admit
        assert "predicted TTFT" in decision.reason

    def test_uses_each_class_own_target(self):
        """The same queue is a breach for critical and fine for sheddable."""
        policy = SLOAwareAdmission()
        state = _server_with(queued=20, prompt=3_000).state(0.0)
        assert not policy.decide(_request(Criticality.CRITICAL), state).admit
        assert policy.decide(_request(Criticality.SHEDDABLE), state).admit

    def test_reserves_kv_headroom_for_critical(self):
        policy = SLOAwareAdmission(slo=SLO(critical_headroom=0.25))
        server = Server(spec=ServerSpec(kv_tokens=10_000))
        server.enqueue(
            Request(id=0, arrival_s=0.0, criticality=Criticality.CRITICAL,
                    prompt_tokens=8_000, output_tokens=100),
            now_s=0.0,
        )
        server.step(0.0, 0.02)
        state = server.state(0.0)
        assert state.kv_utilization >= 0.75

        decision = policy.decide(_request(Criticality.SHEDDABLE), state)
        assert not decision.admit
        assert "reserved for critical" in decision.reason
        # Critical itself is never refused on headroom grounds.
        assert "reserved for critical" not in (
            policy.decide(_request(Criticality.CRITICAL, prompt=10), state).reason
        )

    def test_a_long_prompt_can_be_refused_where_a_short_one_is_admitted(self):
        policy = SLOAwareAdmission()
        state = _server_with(queued=8, prompt=2_000).state(0.0)
        assert policy.decide(_request(Criticality.CRITICAL, prompt=100), state).admit
        assert not policy.decide(_request(Criticality.CRITICAL, prompt=60_000), state).admit


def test_policies_only_read_scrapable_state():
    """A policy that needs backend internals or lookahead is not implementable."""
    from inference_slo.backend import GatewayState
    allowed = set(GatewayState.__dataclass_fields__)
    assert allowed == {
        "now_s", "queue_depth", "queued_prompt_tokens", "queued_by_class",
        "batch_size", "kv_used_tokens", "spec",
    }
