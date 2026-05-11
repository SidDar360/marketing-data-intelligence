import json
import os

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.models import (
    check_drift,
    evaluate_model,
    load_model,
    save_model,
    train_linear_regression,
    train_random_forest,
)


@pytest.fixture
def xy():
    rng = np.random.default_rng(42)
    X = rng.random((120, 4))
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + rng.normal(0, 0.05, 120)
    return X, y


def test_train_random_forest_returns_model_and_metrics(xy):
    X, y = xy
    model, metrics = train_random_forest(X, y)
    assert isinstance(model, RandomForestRegressor)
    assert {"rmse", "mae", "r2"} == set(metrics)
    assert metrics["r2"] > 0.5


def test_train_linear_regression_high_r2(xy):
    X, y = xy
    model, metrics = train_linear_regression(X, y)
    assert isinstance(model, LinearRegression)
    assert metrics["r2"] > 0.9


def test_evaluate_model(xy):
    X, y = xy
    model, _ = train_random_forest(X, y)
    metrics = evaluate_model(model, X, y)
    assert metrics["rmse"] >= 0
    assert metrics["r2"] <= 1.0


def test_save_and_load_model(xy, tmp_path, monkeypatch):
    import src.models as m

    monkeypatch.setattr(m, "ARTIFACTS_DIR", str(tmp_path))

    X, y = xy
    model, _ = train_random_forest(X, y)
    save_model(model, "test_rf")
    loaded = load_model("test_rf")

    np.testing.assert_array_almost_equal(model.predict(X[:5]), loaded.predict(X[:5]))


def test_check_drift_no_drift(tmp_path):
    stats = {
        "actual_price": {"mean": 1000.0, "std": 200.0},
        "discounted_price": {"mean": 600.0, "std": 150.0},
        "rating": {"mean": 4.0, "std": 0.3},
        "rating_count": {"mean": 10000.0, "std": 5000.0},
    }
    stats_path = str(tmp_path / "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f)

    result = check_drift({"actual_price": 950.0, "discounted_price": 580.0}, stats_path)
    assert result["drift_detected"] is False


def test_check_drift_detected(tmp_path):
    stats = {
        "actual_price": {"mean": 1000.0, "std": 10.0},
    }
    stats_path = str(tmp_path / "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f)

    result = check_drift({"actual_price": 99999.0}, stats_path)
    assert result["drift_detected"] is True
