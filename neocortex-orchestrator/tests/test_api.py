from fastapi.testclient import TestClient

from neocortex.api.main import app


def test_health_check():
    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_create_session_returns_expected_shape():
    # Defaults to USE_MOCK_LLM=true and falls back to offline embeddings/NLI/Chroma,
    # so this exercises the full graph (executor -> auditor -> supervisor loop)
    # without any external services or API keys.
    with TestClient(app) as client:
        resp = client.post("/api/sessions", json={"query": "When was the Eiffel Tower completed?"})
        assert resp.status_code == 200
        body = resp.json()
        for key in ["session_id", "query", "response", "confidence_score", "drift_score", "status"]:
            assert key in body
        assert body["status"] in {"accepted", "failed"}


def test_get_unknown_session_returns_404():
    with TestClient(app) as client:
        resp = client.get("/api/sessions/does-not-exist")
        assert resp.status_code == 404
