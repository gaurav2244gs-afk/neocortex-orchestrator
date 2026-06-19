from neocortex.agents.supervisor import SupervisorAgent, route_after_supervisor
from neocortex.graph.state import initial_state


def make_state(confidence, retry_count=0, max_retries=2, used_corrective=False):
    state = initial_state(session_id="s1", query="q", max_retries=max_retries)
    state["confidence_score"] = confidence
    state["retry_count"] = retry_count
    state["used_corrective_subagent"] = used_corrective
    return state


def test_high_confidence_is_accepted():
    supervisor = SupervisorAgent(confidence_threshold=0.75)
    state = make_state(confidence=0.9)
    result = supervisor.run(state)
    assert route_after_supervisor(result) == "accept"
    assert result["status"] == "accepted"


def test_low_confidence_with_retries_left_routes_to_retry():
    supervisor = SupervisorAgent(confidence_threshold=0.75)
    state = make_state(confidence=0.3, retry_count=0, max_retries=2)
    result = supervisor.run(state)
    assert route_after_supervisor(result) == "retry"
    assert result["retry_count"] == 1
    assert result["corrective_context"] is not None
    assert result["status"] == "correcting"


def test_retries_exhausted_spawns_corrective_subagent():
    supervisor = SupervisorAgent(confidence_threshold=0.75)
    state = make_state(confidence=0.3, retry_count=2, max_retries=2, used_corrective=False)
    result = supervisor.run(state)
    assert route_after_supervisor(result) == "spawn_corrective"
    assert result["used_corrective_subagent"] is True


def test_corrective_subagent_also_failing_routes_to_fail():
    supervisor = SupervisorAgent(confidence_threshold=0.75)
    state = make_state(confidence=0.3, retry_count=2, max_retries=2, used_corrective=True)
    result = supervisor.run(state)
    assert route_after_supervisor(result) == "fail"
    assert result["status"] == "failed"


def test_confidence_exactly_at_threshold_is_accepted():
    supervisor = SupervisorAgent(confidence_threshold=0.75)
    state = make_state(confidence=0.75)
    result = supervisor.run(state)
    assert route_after_supervisor(result) == "accept"
