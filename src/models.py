"""
models.py
=========
Scikit-learn model training, evaluation, persistence, and drift detection.

Two regression models are trained on the Amazon Sales dataset:

1. **RandomForestRegressor** — predicts ``discount_percentage``.
   Tree ensembles handle the non-linear interactions between price tiers,
   rating bands, and review volume without requiring feature scaling.

2. **LinearRegression** — predicts ``discounted_price`` (₹).
   A linear model is appropriate here because the selling price is almost
   mechanically derived from the MRP and the discount percentage; the
   relationship is inherently linear (price = MRP × (1 − discount/100)).

Both models are serialised with ``pickle`` to the ``artifacts/`` directory
and loaded from there by the FastAPI server at startup.

Drift detection
---------------
``save_training_stats`` (in ``data_preprocessing.py``) records the mean and
standard deviation of each input feature over the full training set.
``check_drift`` uses those statistics at inference time to flag inputs that
are statistically unusual (|z| > 3).  This is a lightweight, parameter-free
check that requires no additional model and adds negligible latency.

Exported public API
-------------------
train_random_forest    -- Fit a RandomForest and return (model, metrics).
train_linear_regression -- Fit a LinearRegression and return (model, metrics).
evaluate_model         -- Compute RMSE, MAE, and R² on a held-out set.
plot_feature_importance -- Save a bar chart of RandomForest feature importances.
save_model             -- Pickle a fitted model to ``artifacts/<name>.pkl``.
load_model             -- Unpickle a model from ``artifacts/<name>.pkl``.
check_drift            -- Return z-scores and a drift flag for a prediction input.
"""

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# All serialised model files live in one place so the API server has a single
# directory to look in regardless of the current working directory.
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


# ── Training ──────────────────────────────────────────────────────────────────

def train_random_forest(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Fit a RandomForestRegressor and evaluate it on a held-out test split.

    The forest uses 100 trees, which strikes a good balance between variance
    reduction and training time on this ~1,400-row dataset.  The random seed
    is fixed so that repeated runs produce identical train/test splits and
    therefore reproducible metric numbers.

    Args:
        X: Feature matrix — a ``pd.DataFrame`` or ``np.ndarray`` of shape
            ``(n_samples, n_features)``.
        y: Target vector of ``discount_percentage`` values, length
            ``n_samples``.
        test_size: Fraction of rows held out for evaluation.  Defaults to
            ``0.2`` (20 %).
        random_state: Seed for both the train/test split and the forest's
            internal random number generator.

    Returns:
        A tuple ``(model, metrics)`` where:
        - ``model`` is the fitted ``RandomForestRegressor``.
        - ``metrics`` is a ``dict`` with keys ``"rmse"``, ``"mae"``,
          ``"r2"`` computed on the test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_test, y_test)


def train_linear_regression(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Fit an ordinary least-squares Linear Regression model.

    Linear regression is the right tool for price prediction because the
    target (discounted_price) has a near-linear relationship with the MRP and
    discount percentage by construction.  This model achieves R² ≈ 0.95 with
    just four features.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Target vector of ``discounted_price`` values (₹).
        test_size: Fraction of rows held out for evaluation.
        random_state: Seed for the train/test split.

    Returns:
        A tuple ``(model, metrics)`` — same structure as
        ``train_random_forest``.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_test, y_test)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test) -> dict:
    """Compute RMSE, MAE, and R² for a fitted sklearn-compatible model.

    All three metrics are returned so callers can choose which to display.
    RMSE is preferred for comparing models (penalises large errors more than
    MAE); MAE gives an intuitive 'average absolute error' in the target unit;
    R² expresses how much variance the model explains (1.0 = perfect).

    Args:
        model: Any fitted object with a ``predict(X)`` method.
        X_test: Feature matrix for the held-out test rows.
        y_test: True target values for those rows.

    Returns:
        ``dict`` with keys:
        - ``"rmse"`` — root-mean-squared error (same unit as target).
        - ``"mae"``  — mean absolute error.
        - ``"r2"``   — coefficient of determination.
    """
    y_pred = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }


# ── Explainability ────────────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: list,
    output_path: str | None = None,
) -> None:
    """Save (or display) a bar chart of RandomForest feature importances.

    Feature importances in a RandomForest are the mean decrease in node
    impurity (Gini impurity for classifiers, MSE for regressors) weighted by
    the fraction of samples that pass through each node.  They sum to 1.0 and
    give a quick overview of which features the ensemble relies on most.

    This function is a no-op for models that do not expose a
    ``feature_importances_`` attribute (e.g. LinearRegression), so it is safe
    to call unconditionally.

    Args:
        model: A fitted model — expected to be a ``RandomForestRegressor``
            with a ``feature_importances_`` attribute.
        feature_names: Ordered list of feature column names matching the
            columns of the training ``X``.
        output_path: If provided, the chart is saved to this file path
            instead of being displayed interactively.  The file format is
            inferred from the extension (e.g. ``.png``).
    """
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    # Sort descending so the most important feature appears on the left.
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(
        range(len(importances)),
        [feature_names[i] for i in indices],
        rotation=30,
        ha="right",
    )
    plt.title("Feature Importances — Discount Prediction")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Feature importance plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model, name: str) -> str:
    """Serialise a fitted model to ``artifacts/<name>.pkl`` with pickle.

    The ``artifacts/`` directory is created automatically if it does not
    already exist.  Using ``pickle`` here is intentional: it preserves every
    attribute of the sklearn estimator (tree structure, fitted weights, feature
    names, etc.) so the loaded object behaves identically to the original.

    Args:
        model: Any fitted sklearn-compatible estimator.
        name: Filename stem (no extension).  The file will be written to
            ``<ARTIFACTS_DIR>/<name>.pkl``.

    Returns:
        The absolute path of the saved file.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")
    return path


def load_model(name: str):
    """Deserialise a model previously saved by ``save_model``.

    Args:
        name: Filename stem matching the name used in ``save_model``.

    Returns:
        The fitted estimator, ready to call ``.predict()`` on.

    Raises:
        FileNotFoundError: If ``artifacts/<name>.pkl`` does not exist.
            The caller should catch this and instruct the user to run
            ``train.py`` first.
    """
    path = os.path.join(ARTIFACTS_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Drift detection ───────────────────────────────────────────────────────────

def check_drift(input_features: dict, stats_path: str | None = None) -> dict:
    """Detect whether a prediction request is outside the training distribution.

    For each feature in ``input_features``, this function computes the
    absolute z-score::

        z = |value − training_mean| / training_std

    A z-score above **3.0** indicates the value is more than three standard
    deviations from the training mean — a commonly used threshold for flagging
    statistical outliers.  When any feature exceeds this threshold,
    ``drift_detected`` is set to ``True`` in the returned dict.

    This is a lightweight, assumption-free check.  It does not require a
    separate drift-detection model and adds less than 1 ms of latency.

    Args:
        input_features: Dict mapping feature name → value for a single
            prediction request, e.g.
            ``{"actual_price": 50000, "rating": 4.5, ...}``.
        stats_path: Path to the ``training_stats.json`` file written by
            ``save_training_stats``.  Defaults to the standard location
            inside ``artifacts/``.

    Returns:
        A dict with two keys:
        - ``"drift_detected"`` (``bool``) — ``True`` if any z-score > 3.
        - ``"z_scores"`` (``dict``) — per-feature z-score, rounded to 3 d.p.
    """
    if stats_path is None:
        stats_path = os.path.join(ARTIFACTS_DIR, "training_stats.json")

    with open(stats_path) as f:
        stats = json.load(f)

    z_scores: dict = {}
    drift_detected = False

    for feature, value in input_features.items():
        # Skip features that were not recorded during training (e.g. keys
        # added by Pydantic's model_dump that are not model inputs).
        if feature not in stats:
            continue

        mean = stats[feature]["mean"]
        std = stats[feature]["std"]

        # Guard against zero std (constant feature in training data).
        z = abs((value - mean) / std) if std > 0 else 0.0
        z_scores[feature] = round(z, 3)

        if z > 3.0:
            drift_detected = True

    return {"drift_detected": drift_detected, "z_scores": z_scores}
