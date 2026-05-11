import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app


def _mock_discount_model(pred=42.5):
    m = MagicMock()
    m.predict.return_value = np.array([pred])
    return m


@pytest.fixture
def client(tmp_path):
    stats = {
        "actual_price": {"mean": 1000.0, "std": 200.0},
        "discounted_price": {"mean": 600.0, "std": 150.0},
        "rating": {"mean": 4.0, "std": 0.3},
        "rating_count": {"mean": 10000.0, "std": 5000.0},
    }
    stats_path = tmp_path / "training_stats.json"
    stats_path.write_text(json.dumps(stats))

    with TestClient(app) as c:
        app.state.discount_model = _mock_discount_model()
        app.state.rag_index = MagicMock()
        app.state.rag_embed = MagicMock()
        app.state.rag_docs = ["doc1", "doc2", "doc3"]
        app.state.stats_path = str(stats_path)
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_discount_returns_prediction(client):
    payload = {"actual_price": 999.0, "discounted_price": 599.0, "rating": 4.2, "rating_count": 1000.0}
    resp = client.post("/predict_discount", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_discount_percentage"] == 42.5
    assert "drift_warning" in data
    assert "inputs" in data


def test_predict_discount_503_when_no_model(client):
    app.state.discount_model = None
    resp = client.post(
        "/predict_discount",
        json={"actual_price": 999.0, "discounted_price": 599.0, "rating": 4.2, "rating_count": 1000.0},
    )
    assert resp.status_code == 503
    app.state.discount_model = _mock_discount_model()


def test_answer_question(client):
    with patch("src.api.retrieve_relevant_docs", return_value=["relevant doc"]) as mock_r:
        with patch("src.api.generate_answer", return_value="A great cable.") as mock_g:
            resp = client.post("/answer_question", json={"query": "What cable should I buy?"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "A great cable."
    assert data["question"] == "What cable should I buy?"
    assert "sources" in data


def test_answer_question_503_when_no_rag(client):
    app.state.rag_index = None
    resp = client.post("/answer_question", json={"query": "Test?"})
    assert resp.status_code == 503
    app.state.rag_index = MagicMock()
