import json

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from webapp.app import app  # noqa: E402


def test_metrics_is_valid_json_after_route_requests():
    with TestClient(app) as client:
        for algorithm in ("pointwise", "dp"):
            response = client.get("/api/route", params={"algorithm": algorithm})
            assert response.status_code in (200, 422)
        metrics = client.get("/api/metrics")
        assert metrics.status_code == 200
        json.loads(metrics.text)


def test_unknown_algorithm_is_rejected():
    with TestClient(app) as client:
        assert client.get("/api/route", params={"algorithm": "mpc"}).status_code == 422
