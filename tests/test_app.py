"""Tests for the Flask API, driven with a stub engine (no model, no network)."""

from __future__ import annotations

import pytest

from src.app import create_app


@pytest.fixture
def client(engine):
    app = create_app(engine=engine)
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture
def unloaded_client(monkeypatch):
    """A client whose engine fails to load, to exercise the 503 paths."""
    from src import app as app_module

    def explode():
        raise FileNotFoundError("metadata not found at models/metadata.pkl")

    monkeypatch.setattr(app_module, "build_default_engine", explode)
    application = create_app(eager=False)
    application.config.update(TESTING=True)
    return application.test_client()


class TestHealth:
    def test_healthy_when_engine_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.get_json()["status"] == "healthy"
        assert response.get_json()["num_articles"] == 5

    def test_unhealthy_when_engine_missing(self, unloaded_client):
        response = unloaded_client.get("/health")
        assert response.status_code == 503
        assert response.get_json()["status"] == "unhealthy"
        assert "metadata not found" in response.get_json()["details"]


class TestInfo:
    def test_reports_service_details(self, client):
        payload = client.get("/info").get_json()
        assert payload["service"] == "Search Relevancy API"
        assert payload["num_articles"] == 5
        assert payload["index_backend"] == "exact"

    def test_unavailable_without_engine(self, unloaded_client):
        assert unloaded_client.get("/info").status_code == 503


class TestSearch:
    def test_returns_ranked_results(self, client):
        response = client.post("/search", json={"query": "climate", "num_results": 2})
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["query"] == "climate"
        assert payload["num_results"] == 2
        assert {"article_id", "title", "relevance_score"} <= set(payload["results"][0])

    def test_missing_query_field(self, client):
        response = client.post("/search", json={})
        assert response.status_code == 400
        assert "query" in response.get_json()["error"]

    def test_missing_body(self, client):
        assert client.post("/search").status_code == 400

    def test_short_query_rejected(self, client):
        response = client.post("/search", json={"query": "a"})
        assert response.status_code == 400

    def test_overlong_query_rejected(self, client):
        response = client.post("/search", json={"query": "x" * 5000})
        assert response.status_code == 400

    def test_non_string_query_rejected(self, client):
        assert client.post("/search", json={"query": 7}).status_code == 400

    def test_bad_num_results_rejected(self, client):
        response = client.post("/search", json={"query": "climate", "num_results": "x"})
        assert response.status_code == 400

    def test_num_results_clamped(self, client):
        payload = client.post(
            "/search", json={"query": "climate", "num_results": 9999}
        ).get_json()
        assert payload["num_results"] == 5

    def test_unavailable_without_engine(self, unloaded_client):
        response = unloaded_client.post("/search", json={"query": "climate"})
        assert response.status_code == 503


class TestBatchSearch:
    def test_returns_one_block_per_query(self, client):
        response = client.post(
            "/search/batch", json={"queries": ["climate", "markets"], "num_results": 2}
        )
        assert response.status_code == 200
        assert len(response.get_json()["batch_results"]) == 2

    def test_empty_list_rejected(self, client):
        assert client.post("/search/batch", json={"queries": []}).status_code == 400

    def test_non_list_rejected(self, client):
        assert client.post("/search/batch", json={"queries": "climate"}).status_code == 400

    def test_missing_field_rejected(self, client):
        assert client.post("/search/batch", json={}).status_code == 400

    def test_oversized_batch_rejected(self, client):
        response = client.post("/search/batch", json={"queries": ["climate"] * 50})
        assert response.status_code == 400

    def test_invalid_member_rejected(self, client):
        response = client.post("/search/batch", json={"queries": ["climate", "a"]})
        assert response.status_code == 400


class TestErrorHandlers:
    def test_unknown_endpoint_returns_json_404(self, client):
        response = client.get("/nope")
        assert response.status_code == 404
        assert response.get_json()["error"] == "Endpoint not found"

    def test_wrong_method_returns_json_405(self, client):
        response = client.get("/search")
        assert response.status_code == 405
        assert response.get_json()["error"] == "Method not allowed"
