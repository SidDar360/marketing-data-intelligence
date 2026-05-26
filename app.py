"""
Marketing Data Intelligence — Streamlit Demo App
Run: streamlit run app.py

How this file is structured
---------------------------
1. Imports & constants      -- Third-party libraries, colour palette, file paths.
2. Model registry           -- _MODELS dict: one entry per algorithm with its
                               sklearn class, default hyperparams, and UI copy.
3. Tunable-parameter spec   -- _TUNABLE dict: slider definitions for each model.
4. Helper functions         -- dark_fig() for themed matplotlib figures,
                               _model_param_ui() for the hyperparameter sliders.
5. Cached loaders           -- @st.cache_resource / @st.cache_data functions that
                               load data and train models once per session.
6. Bootstrap                -- Auto-trains models on first run if artifacts are
                               missing (needed for Streamlit Community Cloud).
7. Page routing             -- if/elif blocks, one per sidebar page.

Pages
-----
Overview        -- Dataset KPIs and exploratory charts.
Feature Explorer-- Correlation heatmap, plain-English insights, feature selection.
Predict Discount-- Live discount prediction with model selector + param tuning.
Predict Price   -- Live price prediction with model selector + param tuning.
Model Insights  -- Evaluation metrics, feature importance, residual plots.
"""
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

# "Agg" is a non-interactive backend — required when matplotlib is used inside
# a web server (Streamlit) where no display is available.
matplotlib.use("Agg")
# Allow "from src.xxx import ..." regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import (
    encode_category,
    get_feature_target_for_discount,
    get_feature_target_for_price,
    get_top_level_category,
    load_and_clean_data,
)
from src.models import check_drift, evaluate_model, load_model

# ── Model registry ────────────────────────────────────────────────────────────
# Each key is the display name shown in the selectbox.
# Each value is a dict with:
#   cls    -- the sklearn class to instantiate
#   params -- default constructor kwargs (merged with user overrides at train time)
#   speed  -- rough training time shown in the UI
#   desc   -- plain-English explanation shown in the "What is X?" expander
#   pros   -- strengths shown as a green info box
#   cons   -- limitations shown as a yellow warning box
_MODELS = {
    "Linear Regression": {
        "cls": LinearRegression, "params": {},
        "speed": "⚡ Instant",
        "desc": (
            "Finds the best straight line (or flat plane) through the data. "
            "Each feature gets a weight — the prediction is just a weighted sum of your inputs. "
            "You can inspect exactly which factors matter and by how much."
        ),
        "pros": "Extremely fast, fully interpretable, great baseline.",
        "cons": "Assumes a perfectly linear relationship — misses curves, thresholds, and feature interactions.",
    },
    "Ridge Regression": {
        "cls": Ridge, "params": {"alpha": 1.0},
        "speed": "⚡ Instant",
        "desc": (
            "Linear Regression with a built-in 'penalty' that shrinks large coefficients. "
            "When two input features are correlated (e.g. actual price and discounted price move together), "
            "Ridge handles that more gracefully than plain Linear Regression."
        ),
        "pros": "More stable than Linear Regression when features are correlated. Still fully interpretable.",
        "cons": "Still linear — same ceiling on complexity as Linear Regression.",
    },
    "Random Forest": {
        "cls": RandomForestRegressor, "params": {"n_estimators": 200, "random_state": 42},
        "speed": "🕐 ~2 s",
        "desc": (
            "Trains hundreds of independent decision trees on random slices of the data, "
            "then averages all their predictions. Each tree learns slightly different patterns, "
            "so the group is much more robust and accurate than any single tree."
        ),
        "pros": "Handles non-linear patterns and feature interactions. Robust to outliers. Usually very accurate.",
        "cons": "Slower to train. Hard to interpret — you can see feature importance but not exact decision rules.",
    },
    "Gradient Boosting": {
        "cls": GradientBoostingRegressor,
        "params": {"n_estimators": 200, "learning_rate": 0.1, "random_state": 42},
        "speed": "🕐 ~3 s",
        "desc": (
            "Builds trees one at a time, where each new tree focuses on correcting the mistakes "
            "of all the previous ones. This 'boosting' process squeezes out maximum accuracy "
            "and is one of the top-performing algorithms for structured data."
        ),
        "pros": "Often the most accurate on tabular data. Systematically learns from its own errors.",
        "cons": "Slowest to train. More sensitive to settings. Can overfit if not tuned carefully.",
    },
}

# Features used to train each model.  Must stay in sync with the feature
# extraction functions in src/data_preprocessing.py.
_DISCOUNT_FEATURES = ["actual_price", "discounted_price", "rating", "rating_count"]
_PRICE_FEATURES    = ["actual_price", "rating", "rating_count", "discount_percentage"]

# ── Hyperparameter tuning spec ────────────────────────────────────────────────
# Each entry is a list of tuples, one per tunable parameter.
# Tuple format: (param_name, type, min, max, default, step, short_help, explanation)
#   param_name  -- kwarg name passed to the sklearn constructor
#   type        -- "float", "int", or "int_none" (0 maps to Python None)
#   min/max     -- slider range
#   default     -- initial slider value (matches the _MODELS params default)
#   step        -- slider increment
#   short_help  -- tooltip shown on hover
#   explanation -- always-visible caption rendered below the slider
_TUNABLE = {
    "Ridge Regression": [
        ("alpha", "float", 0.01, 100.0, 1.0, 0.01,
         "Regularization strength.",
         "Controls how aggressively the model shrinks its coefficients. "
         "**Low (0.01):** barely any shrinkage — behaves like plain Linear Regression and may overfit noisy data. "
         "**High (100):** heavily shrinks all weights toward zero, which can underfit. "
         "Start at 1 and increase if training score is much higher than test score."),
    ],
    "Random Forest": [
        ("n_estimators", "int", 50, 500, 200, 10,
         "Number of trees.",
         "Each tree is trained on a random subset of the data. More trees reduce random variance — predictions become more stable and consistent. "
         "Returns diminish past ~200; raise it if metric scores change noticeably between runs, lower it if training feels slow."),
        ("max_depth", "int_none", 0, 30, 0, 1,
         "Max depth of each tree (0 = unlimited).",
         "**0 (unlimited):** trees grow until every leaf contains a single training sample — maximum fit but high memory use. "
         "**Low depth (5–10):** shallower, faster trees that generalise better when the model is overfitting (test R² much lower than training R²). "
         "Try reducing this first if you suspect overfitting."),
        ("min_samples_split", "int", 2, 20, 2, 1,
         "Min samples required to split a node.",
         "**Low (2):** a node splits even with just two samples — very detailed trees, high overfitting risk. "
         "**High (10–20):** forces each split to represent more data, creating simpler trees that generalise better. "
         "Increase this alongside max_depth to control model complexity."),
    ],
    "Gradient Boosting": [
        ("n_estimators", "int", 50, 500, 200, 10,
         "Number of boosting stages.",
         "Each stage adds one shallow tree that corrects the previous errors. More stages improve training fit but slow down training and increase overfitting risk. "
         "Best used together with a lower learning_rate: halve the rate and double the estimators for similar accuracy with better generalisation."),
        ("learning_rate", "float", 0.01, 0.5, 0.1, 0.01,
         "Shrinks each tree's contribution.",
         "**High (0.3–0.5):** learns quickly but can overshoot the optimum and overfit. "
         "**Low (0.01–0.05):** each tree makes a smaller correction, so the model is more careful — but needs many more n_estimators to converge. "
         "This is the most impactful knob: lower rate + more estimators almost always improves generalisation."),
        ("max_depth", "int", 2, 10, 3, 1,
         "Max depth of each tree.",
         "Unlike Random Forest, Gradient Boosting works best with **shallow trees (depth 3–5)**. "
         "Each tree only needs to capture one layer of error, not the full pattern. "
         "Deeper trees capture more complex corrections but overfit quickly. "
         "Increase only if R² is low and you've already tried more estimators with a lower learning rate."),
    ],
}


def _model_param_ui(model_name: str, key_prefix: str) -> str:
    """Render an '⚙️ Fine-tune parameters' expander and return user choices as JSON.

    The returned JSON string is passed to train_dynamic_model() as `params_json`.
    It is serialised rather than returned as a dict because @st.cache_resource
    requires all arguments to be hashable — dicts are not, strings are.

    Widget keys use `key_prefix` ("disc" or "price") so the same slider values
    persist in session state and sync between the Predict page and Model Insights.

    Args:
        model_name: Key into _MODELS — determines which sliders to show.
        key_prefix: "disc" for the discount model, "price" for the price model.

    Returns:
        JSON string of param overrides, e.g. '{"n_estimators": 100}'.
        Returns '{}' for Linear Regression (no tunable params).
    """
    specs = _TUNABLE.get(model_name)
    if not specs:
        with st.expander("⚙️ Fine-tune parameters"):
            st.info(
                "Linear Regression has no tunable parameters — it solves for the optimal weights analytically. "
                "To control complexity, switch to Ridge Regression and adjust its alpha."
            )
        return "{}"

    params: dict = {}
    with st.expander("⚙️ Fine-tune parameters"):
        for param, ptype, lo, hi, default, step, short_help, explanation in specs:
            # Widget key is scoped to the model task so discount and price sliders
            # don't collide even when both pages are open in the same session.
            wkey = f"params_{key_prefix}_{param}"
            if ptype == "float":
                val = st.slider(param, float(lo), float(hi), float(default), float(step),
                                key=wkey, help=short_help)
                params[param] = val
            elif ptype == "int":
                val = st.slider(param, int(lo), int(hi), int(default), int(step),
                                key=wkey, help=short_help)
                params[param] = val
            elif ptype == "int_none":
                # Slider value 0 maps to Python None (sklearn's "unlimited" sentinel).
                val = st.slider(param, int(lo), int(hi), int(default), int(step),
                                key=wkey, help=short_help)
                params[param] = None if val == 0 else val
            st.caption(explanation)
    return json.dumps(params, sort_keys=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "amazon.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
STATS_PATH = os.path.join(ARTIFACTS_DIR, "training_stats.json")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marketing Data Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Card containers */
    .kpi-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #181825 100%);
        border: 1px solid #313244;
        border-radius: 14px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .kpi-value { font-size: 2rem; font-weight: 800; color: #cba6f7; }
    .kpi-label { font-size: 0.82rem; color: #a6adc8; margin-top: 0.2rem; letter-spacing: 0.04em; text-transform: uppercase; }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #1e2d4a 0%, #182040 100%);
        border-left: 4px solid #89b4fa;
        border-radius: 10px;
        padding: 1.1rem 1.5rem;
        margin: 0.8rem 0;
        color: #cdd6f4;
        font-size: 1.05rem;
        line-height: 1.65;
    }

    /* Drift banners */
    .drift-warn {
        background: #2d1800;
        border-left: 4px solid #fab387;
        border-radius: 8px;
        padding: 0.75rem 1.1rem;
        color: #fab387;
        margin-top: 0.8rem;
    }
    .drift-ok {
        background: #0d2b0d;
        border-left: 4px solid #a6e3a1;
        border-radius: 8px;
        padding: 0.75rem 1.1rem;
        color: #a6e3a1;
        margin-top: 0.8rem;
    }

    /* Source docs */
    .source-doc {
        background: #181825;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: #cdd6f4;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }

    /* Chat bubble */
    .chat-q {
        background: #2a2a3e;
        border-radius: 12px 12px 4px 12px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.4rem;
        display: inline-block;
        color: #cdd6f4;
        font-weight: 600;
    }

    /* Section divider */
    hr { border-color: #313244 !important; }

    /* Sidebar tweaks */
    [data-testid="stSidebar"] { background: #11111b; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
# Catppuccin Mocha palette — matches the CSS dark theme defined above so that
# matplotlib charts blend seamlessly with the rest of the Streamlit UI.
DARK   = "#1e1e2e"   # figure background
SURFACE= "#181825"   # axes background (slightly darker than DARK)
BORDER = "#313244"   # axis spines and grid lines
TEXT   = "#cdd6f4"   # axis labels, tick marks, legend text
PURPLE = "#cba6f7"   # primary highlight colour (used for key data series)
BLUE   = "#89b4fa"   # secondary colour
GREEN  = "#a6e3a1"   # positive values / success indicators
RED    = "#f38ba8"   # negative values / error indicators
PEACH  = "#fab387"   # warnings


def dark_fig(w=7, h=4):
    """Create a matplotlib Figure and Axes pre-styled to match the dark UI theme.

    Every chart in the app calls this helper instead of plt.subplots() directly
    so that background colours, axis colours, and spine colours are consistent
    without repeating the same six lines everywhere.

    Args:
        w: Figure width in inches.
        h: Figure height in inches.

    Returns:
        (fig, ax) tuple ready for plotting.
    """
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    return fig, ax


# ── Cached loaders and trainers ───────────────────────────────────────────────
# Streamlit re-runs the entire script on every user interaction.  Caching
# prevents expensive work (disk I/O, model training) from repeating on each run.
#
# @st.cache_data   -- for functions that return plain data (DataFrames, arrays,
#                     dicts).  Results are serialised and stored per argument
#                     combination.  The leading underscore on a DataFrame arg
#                     (e.g. _df) tells Streamlit to skip hashing that arg and
#                     use object identity instead — DataFrames are unhashable.
#
# @st.cache_resource -- for heavyweight objects that should not be serialised
#                       (sklearn models, database connections).  One instance is
#                       shared across all user sessions for the lifetime of the
#                       server process.

@st.cache_data
def get_data():
    """Load and clean the Amazon CSV once per session."""
    return load_and_clean_data(CSV_PATH)


@st.cache_resource
def get_rf_model():
    """Load the pre-trained RandomForest from artifacts/ (used by legacy paths)."""
    return load_model("random_forest_discount")


@st.cache_resource
def get_lr_model():
    """Load the pre-trained LinearRegression from artifacts/ (used by legacy paths)."""
    return load_model("linear_regression_price")


@st.cache_resource
def train_dynamic_model(model_name: str, task: str, params_json: str = "{}"):
    """Train a model from scratch and cache it by (model_name, task, params_json).

    Called whenever the user picks a model or changes a hyperparameter slider.
    Because the result is cached, switching back to a previously used combination
    is instant — no retraining.

    params_json is a JSON string (not a dict) because @st.cache_resource requires
    all arguments to be hashable; plain dicts are not.

    Args:
        model_name: Key into _MODELS, e.g. "Random Forest".
        task:       "discount" or "price" — selects the feature/target pair.
        params_json: JSON string of hyperparameter overrides, e.g. '{"n_estimators": 100}'.
                     Merged on top of the model's defaults from _MODELS.

    Returns:
        A fitted sklearn estimator ready to call .predict() on.
    """
    df_tr = load_and_clean_data(CSV_PATH)
    if task == "discount":
        X, y = get_feature_target_for_discount(df_tr)
    else:
        X, y = get_feature_target_for_price(df_tr)
    cfg = _MODELS[model_name]
    # Merge registry defaults with user overrides — overrides take precedence.
    merged = {**cfg["params"], **json.loads(params_json)}
    m = cfg["cls"](**merged)
    m.fit(X, y)
    return m


@st.cache_data
def dynamic_test_predictions(_df, model_name: str, task: str, params_json: str = "{}"):
    """Return (y_test, y_pred, metrics) for the chosen model on the held-out test split.

    Uses the same 80/20 random_state=42 split as train_dynamic_model so that
    the test rows are never seen during training.

    Returns:
        (y_test, y_pred, metrics) where metrics is a dict with "r2", "rmse", "mae".
    """
    model = train_dynamic_model(model_name, task, params_json)
    if task == "discount":
        X, y = get_feature_target_for_discount(_df)
    else:
        X, y = get_feature_target_for_price(_df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return y_test.values, model.predict(X_test), evaluate_model(model, X_test, y_test)


@st.cache_data
def rf_test_predictions(_df):
    """Test predictions for the artifact-trained RandomForest (legacy, used by older code paths)."""
    model = get_rf_model()
    X, y = get_feature_target_for_discount(_df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return y_test.values, model.predict(X_test), evaluate_model(model, X_test, y_test)


@st.cache_data
def lr_test_predictions(_df):
    """Test predictions for the artifact-trained LinearRegression (legacy)."""
    model = get_lr_model()
    X, y = get_feature_target_for_price(_df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return y_test.values, model.predict(X_test), evaluate_model(model, X_test, y_test)


@st.cache_data
def correlation_matrix(_df):
    """Compute and cache the Pearson correlation matrix for all five numeric features."""
    cols = ["actual_price", "discounted_price", "discount_percentage", "rating", "rating_count"]
    return _df[cols].corr()


@st.cache_data
def encode_category_cached(_df, method: str, target_col: str = "discount_percentage"):
    """Encode the top-level product category and cache the result by (method, target_col).

    Wraps ``src.data_preprocessing.encode_category`` so Streamlit only
    re-encodes when the user changes the encoding method or target column —
    not on every UI interaction.

    The leading underscore on ``_df`` tells Streamlit's cache to use object
    identity rather than hashing the DataFrame, which avoids the
    ``UnhashableTypeError`` that plain ``df`` would raise.

    Args:
        _df: Cleaned dataset as returned by ``load_and_clean_data``.
        method: Encoding method — ``"label"``, ``"frequency"``, ``"target"``,
            or ``"onehot"``.
        target_col: Column used as the target when ``method="target"``.

    Returns:
        ``(encoded_df, col_names)`` as returned by ``encode_category``.
    """
    return encode_category(_df, method, target_col)


@st.cache_data
def feature_selection_path(_df):
    """Run greedy forward selection and return the R² gain at each step.

    Algorithm:
    1. Start with an empty feature set.
    2. Try adding each remaining feature using LinearRegression on the train split.
    3. Permanently add whichever feature gave the highest test R².
    4. Repeat until all _DISCOUNT_FEATURES have been added.

    LinearRegression is used here (not the user-selected model) because it is
    fast and deterministic — the goal is to measure raw feature signal, not to
    match the prediction model.

    Returns:
        List of dicts: [{"step": 1, "feature_added": "...", "r2": 0.xx}, ...]
    """
    X_all, y = get_feature_target_for_discount(_df)
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.2, random_state=42)
    features = list(_DISCOUNT_FEATURES)
    selected, remaining, results = [], features[:], []
    while remaining:
        best, best_r2 = None, -1.0
        for f in remaining:
            cols = selected + [f]
            m = LinearRegression().fit(X_tr[cols], y_tr)
            r2 = float(m.score(X_te[cols], y_te))
            if r2 > best_r2:
                best, best_r2 = f, r2
        selected.append(best)
        remaining.remove(best)
        results.append({"step": len(selected), "feature_added": best, "r2": best_r2})
    return results


def artifacts_ok() -> bool:
    """Return True if all required model artifact files exist on disk.

    The three files are produced by train.py (or by _run_training on first boot).
    If any are missing the app triggers _run_training before rendering any page.
    """
    needed = ["random_forest_discount.pkl", "linear_regression_price.pkl", "training_stats.json"]
    return all(os.path.exists(os.path.join(ARTIFACTS_DIR, f)) for f in needed)


def _run_training():
    """Train the baseline RF and LR models and save them to artifacts/.

    Called automatically on the very first run (e.g. on Streamlit Community Cloud
    where there is no local artifacts/ directory).  Imports are deferred inside
    the function so that the top-level import block stays fast for normal runs
    where artifacts already exist.
    """
    from src.data_preprocessing import (
        get_feature_target_for_discount,
        get_feature_target_for_price,
        load_and_clean_data,
        save_training_stats,
    )
    from src.models import save_model, train_linear_regression, train_random_forest
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df_train = load_and_clean_data(CSV_PATH)
    X_disc, y_disc = get_feature_target_for_discount(df_train)
    rf, _ = train_random_forest(X_disc, y_disc)
    save_model(rf, "random_forest_discount")
    # Training stats are used by check_drift() at inference time.
    save_training_stats(X_disc)
    X_price, y_price = get_feature_target_for_price(df_train)
    lr, _ = train_linear_regression(X_price, y_price)
    save_model(lr, "linear_regression_price")


# ── Sidebar navigation ───────────────────────────────────────────────────────
# `page` holds the selected radio label.  The page blocks below use `in` checks
# (e.g. "Discount" in page) rather than exact equality so emoji prefixes don't
# matter and the routing is resilient to label wording changes.
with st.sidebar:
    st.markdown("## 📊 Marketing Data\nIntelligence")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠  Overview", "🔍  Feature Explorer", "🏷️  Predict Discount", "💰  Predict Price", "📈  Model Insights"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if artifacts_ok():
        st.success("Models loaded ✅")

# ── First-run bootstrap ───────────────────────────────────────────────────────
# On Streamlit Community Cloud the artifacts/ directory doesn't exist until the
# app runs for the first time.  This block trains the baseline models in the
# background and then calls st.rerun() so the page renders with fresh data.
if not artifacts_ok():
    with st.spinner("⚙️ First run — training models on the dataset (takes ~30 s)…"):
        _run_training()
    st.rerun()

# Load the cleaned dataset once; all page blocks share this same DataFrame.
df = get_data()

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("Marketing Data Intelligence")
    st.markdown(
        "An end-to-end ML system for e-commerce analytics: "
        "**discount prediction** and **price forecasting** on the Amazon Sales dataset."
    )
    st.markdown("---")

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df):,}</div><div class="kpi-label">Products</div></div>', unsafe_allow_html=True)
        with st.popover("ℹ️", use_container_width=True):
            st.markdown("**Total Products**")
            st.markdown("The number of unique Amazon products in this dataset after cleaning. A larger dataset gives the models more examples to learn from, making predictions more reliable.")
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{df["category"].nunique()}</div><div class="kpi-label">Categories</div></div>', unsafe_allow_html=True)
        with st.popover("ℹ️", use_container_width=True):
            st.markdown("**Product Categories**")
            st.markdown("The number of distinct product types covered — from electronics to kitchen items. Wider variety means the models have learned patterns across many different kinds of products.")
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{df["discount_percentage"].mean():.1f}%</div><div class="kpi-label">Avg Discount</div></div>', unsafe_allow_html=True)
        with st.popover("ℹ️", use_container_width=True):
            st.markdown("**Average Discount**")
            st.markdown("The typical discount across all products in the dataset. This is the baseline the discount model was trained on — predictions near this value are most reliable.")
    with k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{df["rating"].mean():.2f}</div><div class="kpi-label">Avg Rating</div></div>', unsafe_allow_html=True)
        with st.popover("ℹ️", use_container_width=True):
            st.markdown("**Average Rating**")
            st.markdown("The mean customer satisfaction score (out of 5) across all products. Higher ratings tend to correlate with higher prices — the models use this as a feature.")
    with k5:
        med_price = df["discounted_price"].median()
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">₹{med_price:,.0f}</div><div class="kpi-label">Median Price</div></div>', unsafe_allow_html=True)
        with st.popover("ℹ️", use_container_width=True):
            st.markdown("**Median Selling Price**")
            st.markdown("Half of all products sell for less than this, half for more. The median is more reliable than the average here because a few very expensive products would skew the average upward.")

    st.markdown("---")

    # Charts row
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Discount Distribution")
        fig, ax = dark_fig(5, 3.5)
        ax.hist(df["discount_percentage"], bins=40, color=PURPLE, edgecolor=SURFACE, linewidth=0.4)
        ax.set_xlabel("Discount (%)")
        ax.set_ylabel("Count")
        ax.axvline(df["discount_percentage"].mean(), color=PEACH, linewidth=1.5, linestyle="--", label=f"Mean {df['discount_percentage'].mean():.1f}%")
        ax.legend(labelcolor=TEXT, facecolor=DARK, edgecolor=BORDER, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Discount Distribution**")
            st.markdown("How discounts are spread across all products. The dashed line is the average. Most products cluster between 20–70% off — this is the range the model learned the most from and where it's most accurate.")

    with c2:
        st.subheader("Rating Distribution")
        fig, ax = dark_fig(5, 3.5)
        ax.hist(df["rating"], bins=25, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Rating Distribution**")
            st.markdown("How customer ratings are distributed across products. The strong skew toward 4–5 stars is typical for Amazon listings — few products survive long with poor reviews.")

    with c3:
        st.subheader("Price Distribution")
        fig, ax = dark_fig(5, 3.5)
        clipped = df["discounted_price"].clip(upper=df["discounted_price"].quantile(0.97))
        ax.hist(clipped, bins=40, color=GREEN, edgecolor=SURFACE, linewidth=0.4)
        ax.set_xlabel("Discounted Price (₹)")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Price Distribution**")
            st.markdown("How selling prices are spread, clipped at the 97th percentile to remove extreme outliers. The long tail to the right means a few high-end products are priced far above the majority — the model is less confident for those.")

    st.markdown("---")

    # Price vs Discount scatter
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Actual Price vs Discount %")
        fig, ax = dark_fig(8, 4)
        sc = ax.scatter(
            df["actual_price"].clip(upper=df["actual_price"].quantile(0.95)),
            df["discount_percentage"],
            c=df["rating"],
            cmap="cool",
            alpha=0.4,
            s=15,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Rating", color=TEXT)
        cbar.ax.yaxis.set_tick_params(color=TEXT)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)
        ax.set_xlabel("Actual Price (₹)")
        ax.set_ylabel("Discount (%)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Actual Price vs Discount %**")
            st.markdown("Each dot is one product. Position shows its original price (x-axis) and discount (y-axis). Dot colour indicates the rating. No strong pattern means price alone doesn't determine the discount — the model uses all four features together.")

    with c2:
        st.subheader("System Architecture")
        _disc_sel = st.session_state.get("disc_model", "Random Forest")
        _price_sel = st.session_state.get("price_model", "Linear Regression")
        st.markdown(
            f"""
| Component | Technology |
|-----------|-----------|
| Data | Amazon Sales CSV |
| Discount model | {_disc_sel} |
| Price model | {_price_sel} |
| UI | Streamlit |
"""
        )

    st.markdown("---")
    st.subheader("Sample Products")
    cols = ["product_name", "discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]
    st.dataframe(
        df[cols].rename(columns={"discounted_price": "price (₹)", "actual_price": "MRP (₹)", "discount_percentage": "discount %"}).head(30),
        use_container_width=True,
        height=320,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif "Explorer" in page:
    st.title("🔍 Feature Explorer")
    st.markdown("Understand how the dataset's features relate to each other and to the discount — before any model is involved.")
    st.markdown("---")

    # ── Category encoding selector ──────────────────────────────────────────
    # The raw ``category`` column is a pipe-separated string hierarchy.  It
    # must be converted to numbers before it can enter a correlation matrix or
    # a model.  The selector below controls which conversion strategy is used;
    # the choice propagates through every chart on this page.
    st.subheader("Category Encoding")
    st.markdown(
        "**category** stores a pipe-separated hierarchy such as  \n"
        "`Computers&Accessories | Accessories&Peripherals | Cables&Accessories | …`  \n"
        "Only the **top-level segment** is used here, giving **9 distinct groups**.  \n"
        "Select an encoding method to include it in the heatmap and correlation analysis."
    )

    _ENC_LABELS = [
        "None — numeric only",
        "Label Encoding",
        "Frequency Encoding",
        "Target Encoding — Avg Discount %",
        "Target Encoding — Avg Price ₹",
        "One-Hot Encoding",
    ]
    enc_choice = st.radio("Encoding method", _ENC_LABELS, horizontal=True, key="cat_enc")

    # Plain-English explanation for each encoding method.
    _ENC_DESC = {
        "None — numeric only": (
            "The heatmap and correlation analysis use only the five numeric columns. "
            "Use this as the baseline to compare against category-encoded variants."
        ),
        "Label Encoding": (
            "**How it works:** Each of the 9 top-level categories is assigned a unique integer in "
            "alphabetical order — *Car&Motorbike* = 0, *Computers&Accessories* = 1, *Electronics* = 2, "
            "*Health&PersonalCare* = 3, and so on up to 8.  \n\n"
            "**Strength:** One extra column, no additional memory, works out-of-the-box with any "
            "sklearn estimator.  \n"
            "**Weakness:** The integers imply a magnitude ordering that doesn't exist — a linear model "
            "incorrectly treats *Electronics* (2) as 'twice as much' as *Car&Motorbike* (0). "
            "Tree-based models (Random Forest, Gradient Boosting) are immune to this because they "
            "split on arbitrary thresholds, not on the numeric magnitude."
        ),
        "Frequency Encoding": (
            "**How it works:** Each category is replaced with its *proportion of all products*. "
            "*Electronics* (526 products, ≈36 %) becomes 0.36; *Toys&Games* (1 product) becomes 0.0007.  \n\n"
            "**Strength:** Captures 'how mainstream is this category?' without imposing an ordering. "
            "Rare categories that behave unusually are distinguishable from dominant ones.  \n"
            "**Weakness:** Two categories with identical product counts would collapse to the same "
            "encoded value, making them indistinguishable to the model."
        ),
        "Target Encoding — Avg Discount %": (
            "**How it works:** Each category is replaced with the *mean discount percentage* for all "
            "products in that category — computed here on the full dataset for exploration.  \n\n"
            "**Strength:** Highest single-feature correlation with the discount target. It directly "
            "captures 'how aggressively does this category typically discount?'  \n"
            "**Weakness:** Uses the target variable in the encoding — **data leakage** in production. "
            "Real pipelines compute this mean only on the training fold (k-fold target encoding) to "
            "prevent the model from seeing information about the test labels during training."
        ),
        "Target Encoding — Avg Price ₹": (
            "**How it works:** Same as above but uses *mean discounted price* per category. "
            "*Electronics* may average ₹1 500; USB cables may average ₹250.  \n\n"
            "**Strength:** Captures category-level price positioning — naturally useful for the "
            "price prediction model since it summarises typical price tiers.  \n"
            "**Weakness:** Same data leakage caveat as discount target encoding."
        ),
        "One-Hot Encoding": (
            "**How it works:** Creates **9 binary columns** — one per top-level category. "
            "A product in *Electronics* has `cat_Electronics = 1` and all other category dummies = 0.  \n\n"
            "**Strength:** No false ordering; each category's signal is captured independently. "
            "Ideal for tree-based models that can freely split on any subset of dummies.  \n"
            "**Weakness:** Adds 9 features instead of 1. For linear models the 9 dummies always "
            "sum to 1 — perfect multicollinearity — so one dummy should be dropped. "
            "Because OHE doesn't fit in a single heatmap cell, the main heatmap stays unchanged "
            "and a dedicated **cross-correlation heatmap** (9 categories × 5 numeric features) "
            "is shown below it."
        ),
    }

    with st.expander(f"📖 About **{enc_choice}**", expanded=True):
        st.markdown(_ENC_DESC[enc_choice])

    st.markdown("---")

    # ── Build augmented correlation matrix ─────────────────────────────────
    # Maps each radio label to (method_str, target_col) or None for "no category".
    _ENC_PARAMS = {
        "None — numeric only":              None,
        "Label Encoding":                   ("label",     "discount_percentage"),
        "Frequency Encoding":               ("frequency", "discount_percentage"),
        "Target Encoding — Avg Discount %": ("target",    "discount_percentage"),
        "Target Encoding — Avg Price ₹":    ("target",    "discounted_price"),
        "One-Hot Encoding":                 ("onehot",    "discount_percentage"),
    }
    enc_params = _ENC_PARAMS[enc_choice]

    _NUMERIC_COLS = ["actual_price", "discounted_price", "discount_percentage", "rating", "rating_count"]

    # Human-readable labels for every possible column that may appear in the heatmap.
    _FEAT_LABELS = {
        "actual_price":        "Actual Price (MRP)",
        "discounted_price":    "Discounted Price",
        "discount_percentage": "Discount %",
        "rating":              "Rating",
        "rating_count":        "Rating Count",
        "category_label":      "Category (Label)",
        "category_freq":       "Category (Frequency)",
        "category_target":     "Category (Target)",
    }

    # Plain-English context used in the "What the Data Tells Us" section.
    _FEAT_CONTEXT = {
        "actual_price":      "Premium products (higher MRP) are often discounted more aggressively to attract buyers.",
        "discounted_price":  "The selling price naturally moves with the discount — a lower selling price relative to MRP means a bigger discount.",
        "rating":            "Customer satisfaction and discount levels are linked — either high-rated products attract larger discounts, or discounts drive more purchases and reviews.",
        "rating_count":      "More popular products (more reviews) may face different pricing dynamics than niche ones.",
        "category_label":    "Integer codes are arbitrary alphabetical assignments — tree models discover meaningful splits regardless of ordering, but linear models may not extract full signal.",
        "category_freq":     "Higher values mean a more-populated category. Captures how 'mainstream' a product type is within this dataset.",
        "category_target":   "Directly encodes average pricing behaviour per category — the most predictive single-column representation, but uses the target variable in its construction.",
    }

    # Build the correlation matrix and track which columns appear in the heatmap.
    # OHE is handled separately below (9 dummy columns don't fit in a single cell).
    is_ohe = enc_params is not None and enc_params[0] == "onehot"

    if enc_params is None or is_ohe:
        corr = correlation_matrix(df)
        heatmap_cols = list(_NUMERIC_COLS)
    else:
        method, tgt = enc_params
        enc_df, enc_col_names = encode_category_cached(df, method, tgt)
        df_aug = df[_NUMERIC_COLS].copy()
        for col in enc_col_names:
            df_aug[col] = enc_df[col].values
        corr = df_aug.corr()
        heatmap_cols = list(_NUMERIC_COLS) + enc_col_names

    # ── Section 1: Correlation Heatmap ─────────────────────────────────────
    st.subheader("Correlation Heatmap")
    readable_labels = [_FEAT_LABELS.get(c, c) for c in heatmap_cols]
    corr_display = corr.copy()
    corr_display.index = readable_labels
    corr_display.columns = readable_labels

    n_cols = len(heatmap_cols)
    fig, ax = dark_fig(10 if n_cols > 5 else 9, 6)
    sns.heatmap(
        corr_display,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor=SURFACE,
        ax=ax,
        annot_kws={"size": 9 if n_cols > 5 else 10},
        cbar_kws={"shrink": 0.8},
    )
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", color=TEXT)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color=TEXT)
    ax.collections[0].colorbar.ax.tick_params(colors=TEXT)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Inline interpretation guide ─────────────────────────────────────────
    # Always visible so users don't need to click anything to understand the chart.
    st.markdown("**What the numbers mean**")
    gi1, gi2, gi3 = st.columns(3)
    with gi1:
        st.success(
            "**Positive — red cells (0 to +1)**  \n"
            "Both features rise and fall together.  \n"
            "**+1.00** = perfect positive link  \n"
            "**+0.70+** = strong  \n"
            "**+0.30–0.69** = moderate  \n"
            "**< +0.30** = weak  \n\n"
            "Higher values of one → higher values of the other."
        )
    with gi2:
        st.info(
            "**Near zero — white cells (−0.3 to +0.3)**  \n"
            "No meaningful linear relationship.  \n"
            "Knowing one feature's value tells you almost nothing about the other.  \n\n"
            "The features may still interact non-linearly — tree models can detect this even when r ≈ 0."
        )
    with gi3:
        st.error(
            "**Negative — blue cells (−1 to 0)**  \n"
            "As one feature rises, the other tends to fall.  \n"
            "**−1.00** = perfect inverse link  \n"
            "**−0.70+** = strong  \n"
            "**−0.30–0.69** = moderate  \n"
            "**> −0.30** = weak  \n\n"
            "Higher values of one → lower values of the other."
        )

    # Dynamic sentence highlighting the strongest positive and negative
    # relationships with discount % in whichever encoding is currently active.
    _disc_corr_vals = corr["discount_percentage"].drop("discount_percentage")
    _pos_feat = _disc_corr_vals.idxmax()
    _neg_feat = _disc_corr_vals.idxmin()
    _pos_r    = _disc_corr_vals[_pos_feat]
    _neg_r    = _disc_corr_vals[_neg_feat]
    _pos_lbl  = _FEAT_LABELS.get(_pos_feat, _pos_feat)
    _neg_lbl  = _FEAT_LABELS.get(_neg_feat, _neg_feat)

    _neg_sentence = (
        f"The most negative is **{_neg_lbl}** (r = {_neg_r:+.2f}) — higher {_neg_lbl.lower()} → smaller discount."
        if _neg_r < -0.1
        else "No strong negative relationships exist in this view — all features correlate positively or near-zero with discount %."
    )
    st.caption(
        f"In this heatmap: the strongest positive link with Discount % is **{_pos_lbl}** "
        f"(r = {_pos_r:+.2f}) — higher {_pos_lbl.lower()} → larger discount. {_neg_sentence}"
    )

    with st.popover("ℹ️ How to read this", use_container_width=True):
        st.markdown("**Correlation Matrix**")
        st.markdown(
            "Each cell shows the Pearson correlation (r) between two features. "
            "**r = 1** — they move perfectly together. **r = −1** — when one goes up the other goes down. "
            "**r ≈ 0** — no linear relationship. "
            "Red = strong positive, blue = strong negative, white = no correlation. "
            "The diagonal is always 1 (every feature is perfectly correlated with itself)."
        )

    with st.expander("🔢 How each number is calculated"):
        st.markdown("#### The Pearson Correlation Formula")
        st.markdown(
            "Every cell value is computed using this formula:\n\n"
            "$$r = \\frac{\\sum_{i=1}^{n}(x_i - \\bar{x})(y_i - \\bar{y})}"
            "{\\sqrt{\\sum_{i=1}^{n}(x_i - \\bar{x})^2 \\;\\cdot\\; \\sum_{i=1}^{n}(y_i - \\bar{y})^2}}$$\n\n"
            "Where **x** and **y** are the two features being compared, and the bar (x̄, ȳ) means the average."
        )
        st.markdown("#### What each part means")
        st.markdown(
            "| Part | What it does |\n"
            "|------|--------------|\n"
            "| $x_i - \\bar{x}$ | How far product *i*'s value is from the average (its deviation) |\n"
            "| $(x_i - \\bar{x})(y_i - \\bar{y})$ | Multiply the two deviations together. Positive when both are above/below average simultaneously. |\n"
            "| $\\sum$ (numerator) | Sum those products across all 1,465 products. Large positive sum → features tend to rise and fall together. |\n"
            "| $\\sqrt{\\cdots}$ (denominator) | Scales the result to always land between −1 and +1, regardless of the units. |"
        )
        st.markdown("#### Worked example — 3 products")
        st.markdown(
            "Suppose we want the correlation between **Actual Price** and **Discount %** for 3 products:\n\n"
            "| Product | Actual Price (₹) | Discount % |\n"
            "|---------|-----------------|------------|\n"
            "| A | 500 | 20 |\n"
            "| B | 1 000 | 50 |\n"
            "| C | 2 000 | 70 |\n\n"
            "**Step 1 — Compute averages:**  \n"
            "Mean price = (500 + 1000 + 2000) ÷ 3 = **1 167**  \n"
            "Mean discount = (20 + 50 + 70) ÷ 3 = **46.7%**\n\n"
            "**Step 2 — Compute deviations** (value − mean):\n\n"
            "| Product | Price dev | Discount dev | Product of devs |\n"
            "|---------|-----------|--------------|----------------|\n"
            "| A | 500 − 1167 = **−667** | 20 − 46.7 = **−26.7** | (−667)(−26.7) = **+17 809** |\n"
            "| B | 1000 − 1167 = **−167** | 50 − 46.7 = **+3.3** | (−167)(+3.3) = **−551** |\n"
            "| C | 2000 − 1167 = **+833** | 70 − 46.7 = **+23.3** | (+833)(+23.3) = **+19 409** |\n\n"
            "**Step 3 — Sum the products of deviations (numerator):**  \n"
            "17 809 − 551 + 19 409 = **+36 667**  \n"
            "Positive sum → the two features tend to move in the same direction.\n\n"
            "**Step 4 — Compute the denominator (the scaling factor):**  \n"
            "Price variance: (−667)² + (−167)² + (+833)² = 444 889 + 27 889 + 693 889 = 1 166 667  \n"
            "Discount variance: (−26.7)² + (+3.3)² + (+23.3)² = 712.9 + 10.9 + 542.9 = 1 266.7  \n"
            "Denominator = √(1 166 667 × 1 266.7) = **√1 477 477 889 ≈ 38 438**\n\n"
            "**Step 5 — Divide:**  \n"
            "r = 36 667 ÷ 38 438 ≈ **+0.95**  \n\n"
            "This strong positive value confirms: in these 3 products, higher MRP → bigger discount. "
            "The real dataset runs the same calculation across all **1,465 products** to produce each cell in the heatmap."
        )
        st.markdown("#### Why the diagonal is always 1.00")
        st.markdown(
            "When both features are the same (e.g. actual_price vs actual_price), "
            "every deviation product $(x_i - \\bar{x})^2$ is always positive, and the numerator equals the denominator exactly — so r = 1."
        )

    # ── OHE cross-correlation heatmap ──────────────────────────────────────
    # One-hot encoding produces 9 binary columns that don't fit as a single
    # heatmap cell.  Instead we compute a cross-correlation matrix:
    #   rows  = 9 category dummies
    #   cols  = 5 numeric features
    # This shows how each category's presence/absence relates to prices,
    # ratings, and discount levels.
    if is_ohe:
        st.markdown("#### One-Hot Encoding — Category × Numeric Cross-Correlation")
        st.markdown(
            "Since one-hot encoding generates 9 binary columns the main heatmap above is "
            "unchanged (5 numeric features only). The chart below shows a **cross-correlation** "
            "heatmap: each **row** is a category dummy (1 = product belongs to that category, 0 = it doesn't), "
            "each **column** is one of the 5 numeric features."
        )
        ohe_df, ohe_col_names = encode_category_cached(df, "onehot", "discount_percentage")
        # Build a single DataFrame with both OHE dummies and numeric features, then
        # slice the cross-correlation block we care about from the full corr matrix.
        combined = pd.concat(
            [ohe_df.reset_index(drop=True), df[_NUMERIC_COLS].reset_index(drop=True)],
            axis=1,
        )
        full_ohe_corr = combined.corr()
        cross_corr = full_ohe_corr.loc[ohe_col_names, _NUMERIC_COLS]

        # Strip the "cat_" prefix for cleaner row labels.
        row_labels = [c.replace("cat_", "") for c in ohe_col_names]
        col_labels  = [_FEAT_LABELS.get(c, c) for c in _NUMERIC_COLS]

        fig, ax = dark_fig(10, 5)
        sns.heatmap(
            cross_corr.values,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=col_labels,
            yticklabels=row_labels,
            linewidths=0.5,
            linecolor=SURFACE,
            ax=ax,
            annot_kws={"size": 9},
            cbar_kws={"shrink": 0.8},
        )
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", color=TEXT)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color=TEXT)
        ax.collections[0].colorbar.ax.tick_params(colors=TEXT)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ How to read this", use_container_width=True):
            st.markdown("**OHE Cross-Correlation Heatmap**")
            st.markdown(
                "Each row is a binary 0/1 indicator for one top-level category. "
                "Each column is one of the five numeric features. "
                "**Positive (red)** — products in that category tend to have *above-average* values for that feature. "
                "**Negative (blue)** — products tend to fall *below average*. "
                "For example, a red cell at *Electronics × Actual Price* means "
                "Electronics products cost more than the overall dataset average."
            )

    st.markdown("---")

    # ── Section 2: Plain-English Insights ─────────────────────────────────
    st.subheader("What the Data Tells Us")
    # target_corr includes the encoded category column when one is present,
    # giving an automatic plain-English summary for whichever encoding is active.
    target_corr = corr["discount_percentage"].drop("discount_percentage").sort_values(key=abs, ascending=False)

    for feat, r in target_corr.items():
        strength = "strongly" if abs(r) > 0.6 else "moderately" if abs(r) > 0.3 else "weakly"
        direction = "higher" if r > 0 else "lower"
        polarity = "positively" if r > 0 else "negatively"
        label = _FEAT_LABELS.get(feat, feat)
        context = _FEAT_CONTEXT.get(feat, "")
        msg = (
            f"**{label}** is {strength} {polarity} correlated with discount (r = {r:+.2f}) — "
            f"higher {label} tends to mean a **{direction} discount**. {context}"
        )
        if abs(r) > 0.6:
            st.success(msg)
        elif abs(r) > 0.3:
            st.info(msg)
        else:
            st.warning(msg)

    st.markdown("##### Feature-to-Feature Relationships")
    pairs = []
    for i, c1 in enumerate(heatmap_cols):
        for j, c2 in enumerate(heatmap_cols):
            if j <= i or c1 == "discount_percentage" or c2 == "discount_percentage":
                continue
            pairs.append((abs(corr.loc[c1, c2]), c1, c2, corr.loc[c1, c2]))
    pairs.sort(reverse=True)
    for _, c1, c2, r in pairs[:3]:
        l1, l2 = _FEAT_LABELS.get(c1, c1), _FEAT_LABELS.get(c2, c2)
        if abs(r) > 0.7:
            st.warning(
                f"⚠️ **{l1}** and **{l2}** are highly correlated (r = {r:+.2f}). "
                "They carry overlapping information — but both contribute to computing the price ratio, so both are kept."
            )
        else:
            st.info(f"**{l1}** and **{l2}**: r = {r:+.2f}.")

    st.markdown("---")

    # ── Section 3: Feature vs Discount bar chart ───────────────────────────
    st.subheader("Each Feature's Correlation with Discount %")
    bar_colors = [GREEN if v > 0 else RED for v in target_corr.values]
    bar_labels = [_FEAT_LABELS.get(f, f) for f in target_corr.index]

    # Grow chart height proportionally when extra features are present.
    fig, ax = dark_fig(8, max(3.0, len(target_corr) * 0.6))
    bars = ax.barh(bar_labels, target_corr.values, color=bar_colors, edgecolor=SURFACE)
    ax.axvline(0, color=TEXT, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Pearson r with Discount %")
    for bar, val in zip(bars, target_corr.values):
        xpos = val + 0.015 if val >= 0 else val - 0.015
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{val:+.2f}", va="center", ha=ha, color=TEXT, fontsize=9)
    spread = max(abs(target_corr.values)) * 1.3
    ax.set_xlim(-spread, spread)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    with st.popover("ℹ️ What does this show?", use_container_width=True):
        st.markdown("**Feature–Target Correlation**")
        st.markdown(
            "Each bar is one input feature. The length shows how strongly that feature alone is linearly correlated with discount. "
            "**Green** = higher value → higher discount. **Red** = higher value → lower discount. "
            "A longer bar = stronger individual predictor. Note: features can still be useful in combination even if weak individually."
        )

    st.markdown("---")

    # ── Section 4: Minimum Features Needed ────────────────────────────────
    st.subheader("Minimum Features Needed to Predict Discount")
    st.markdown(
        "This adds one feature at a time — always choosing the feature that improves test R² the most — "
        "to show how quickly predictive power is captured."
    )

    with st.spinner("Running feature selection…"):
        path = feature_selection_path(df)

    steps = [p["step"] for p in path]
    r2s = [p["r2"] for p in path]
    feats_added = [_FEAT_LABELS.get(p["feature_added"], p["feature_added"]) for p in path]
    gains = [r2s[0]] + [r2s[i] - r2s[i - 1] for i in range(1, len(r2s))]

    fig, ax = dark_fig(7, 3.5)
    ax.plot(steps, r2s, marker="o", color=PURPLE, linewidth=2.5, markersize=8)
    for x, y_val, feat in zip(steps, r2s, feats_added):
        ax.annotate(
            f"+ {feat}\nR²={y_val:.3f}",
            (x, y_val),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            color=TEXT,
            fontsize=8,
        )
    ax.set_xlabel("Number of features")
    ax.set_ylabel("R² on test set")
    ax.set_xticks(steps)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.9, color=GREEN, linewidth=1, linestyle=":", alpha=0.7, label="R² = 0.90 threshold")
    ax.legend(labelcolor=TEXT, facecolor=DARK, edgecolor=BORDER, fontsize=8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    tbl_df = pd.DataFrame({
        "Step": steps,
        "Feature added": feats_added,
        "Cumulative R²": [f"{r:.4f}" for r in r2s],
        "R² gain": [f"+{g:.4f}" for g in gains],
    })
    st.dataframe(tbl_df, use_container_width=True, hide_index=True)

    threshold_step = next((p for p in path if p["r2"] >= 0.90), path[-1])
    n_needed = threshold_step["step"]
    r2_achieved = threshold_step["r2"]
    top_feats = ", ".join(_FEAT_LABELS.get(path[i]["feature_added"], path[i]["feature_added"]) for i in range(n_needed))
    st.success(
        f"**{n_needed} feature{'s' if n_needed > 1 else ''} ({top_feats}) achieve R² = {r2_achieved:.3f}**, "
        f"explaining {r2_achieved * 100:.1f}% of discount variation. "
        f"Additional features provide diminishing returns."
    )
    with st.popover("ℹ️ What does this show?", use_container_width=True):
        st.markdown("**Minimum Feature Analysis (Greedy Forward Selection)**")
        st.markdown(
            "Starts with no features. Each round adds whichever remaining feature improves test R² the most, using a fast Linear Regression. "
            "The line shows how much predictive power you gain with each addition. "
            "A flat tail means the last features overlap with existing ones or have low signal on their own."
        )

    # ── Section 5: Category Deep-Dive ─────────────────────────────────────
    # Raw per-category statistics — before encoding — so users can see the
    # underlying distributions that each encoding method tries to capture.
    st.markdown("---")
    st.subheader("Category Analysis")
    st.markdown(
        "Raw category-level statistics *before* any encoding. These are the "
        "ground-truth distributions that encoding methods attempt to summarise numerically. "
        "Target encoding will approximate the 'Avg Discount %' or 'Avg Price ₹' columns; "
        "frequency encoding will reflect the 'Products' column."
    )

    top_cat = get_top_level_category(df)
    cat_stats = (
        df.assign(top_category=top_cat)
        .groupby("top_category")
        .agg(
            count=("discount_percentage", "count"),
            avg_discount=("discount_percentage", "mean"),
            avg_actual_price=("actual_price", "mean"),
            avg_disc_price=("discounted_price", "mean"),
            avg_rating=("rating", "mean"),
        )
        .sort_values("count", ascending=False)
        .reset_index()
    )

    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**Avg Discount % by Category**")
        fig, ax = dark_fig(6, 4)
        sorted_d = cat_stats.sort_values("avg_discount", ascending=True)
        bars = ax.barh(sorted_d["top_category"], sorted_d["avg_discount"], color=PURPLE, edgecolor=SURFACE)
        for bar, val in zip(bars, sorted_d["avg_discount"]):
            ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", color=TEXT, fontsize=8)
        ax.set_xlabel("Avg Discount %")
        ax.set_xlim(0, sorted_d["avg_discount"].max() * 1.22)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Average Discount % by Category**")
            st.markdown(
                "The typical discount level for products in each top-level category. "
                "This is what *Target Encoding (Avg Discount %)* uses as the encoded value — "
                "categories with longer bars get higher numeric representations."
            )

    with ch2:
        st.markdown("**Avg Discounted Price ₹ by Category**")
        fig, ax = dark_fig(6, 4)
        sorted_p = cat_stats.sort_values("avg_disc_price", ascending=True)
        bars = ax.barh(sorted_p["top_category"], sorted_p["avg_disc_price"], color=BLUE, edgecolor=SURFACE)
        for bar, val in zip(bars, sorted_p["avg_disc_price"]):
            ax.text(val + 15, bar.get_y() + bar.get_height() / 2, f"₹{val:,.0f}", va="center", color=TEXT, fontsize=8)
        ax.set_xlabel("Avg Discounted Price (₹)")
        ax.set_xlim(0, sorted_p["avg_disc_price"].max() * 1.22)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Average Discounted Price by Category**")
            st.markdown(
                "The typical selling price per category. This is what *Target Encoding (Avg Price ₹)* uses — "
                "higher-priced categories like Musical Instruments or Electronics get larger encoded values."
            )

    st.dataframe(
        cat_stats.rename(columns={
            "top_category": "Category",
            "count":         "Products",
            "avg_discount":  "Avg Discount %",
            "avg_actual_price": "Avg MRP (₹)",
            "avg_disc_price":   "Avg Price (₹)",
            "avg_rating":       "Avg Rating",
        }).style.format({
            "Avg Discount %": "{:.1f}%",
            "Avg MRP (₹)":    "₹{:,.0f}",
            "Avg Price (₹)":  "₹{:,.0f}",
            "Avg Rating":     "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT DISCOUNT
# ══════════════════════════════════════════════════════════════════════════════
elif "Discount" in page:
    st.title("🏷️ Predict Discount Percentage")
    st.markdown("Choose a model, adjust the product attributes, and see the predicted discount in real-time.")
    st.markdown("---")

    # ── Model selector ─────────────────────────────────────────────────────
    disc_model_name = st.selectbox(
        "Model",
        list(_MODELS.keys()),
        index=2,   # Random Forest default
        key="disc_model",
    )
    _dcfg = _MODELS[disc_model_name]
    with st.expander(f"📖 What is {disc_model_name}?  {_dcfg['speed']}"):
        st.markdown(_dcfg["desc"])
        pc1, pc2 = st.columns(2)
        pc1.success(f"✅ **Strengths:** {_dcfg['pros']}")
        pc2.warning(f"⚠️ **Limitations:** {_dcfg['cons']}")
    disc_params_json = _model_param_ui(disc_model_name, "disc")
    st.markdown("---")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Product Attributes")
        actual_price = st.number_input("Actual / MRP Price (₹)", min_value=10.0, max_value=200000.0, value=999.0, step=50.0)
        discounted_price = st.number_input("Selling / Discounted Price (₹)", min_value=1.0, max_value=200000.0, value=599.0, step=50.0)

        if discounted_price >= actual_price:
            st.warning("Selling price should be less than MRP for a discount to exist.")

        rating = st.slider("Product Rating", 1.0, 5.0, 4.2, 0.1)
        rating_count = st.number_input("Number of Ratings", min_value=1, max_value=500000, value=1200, step=100)

        # Quick presets
        st.markdown("**Quick Presets**")
        p1, p2, p3 = st.columns(3)
        if p1.button("Budget item"):
            actual_price, discounted_price, rating, rating_count = 299.0, 149.0, 3.8, 500
        if p2.button("Mid-range"):
            actual_price, discounted_price, rating, rating_count = 1999.0, 1199.0, 4.2, 5000
        if p3.button("Premium"):
            actual_price, discounted_price, rating, rating_count = 15000.0, 9999.0, 4.5, 20000

        with st.expander("📖 Why these features?"):
            _corr_disc = correlation_matrix(df)["discount_percentage"]
            st.dataframe(
                pd.DataFrame([
                    {"Feature": "actual_price", "Role": "Original list price (MRP). Higher-priced products are often discounted more aggressively to attract buyers.", "r with Discount": f"{_corr_disc['actual_price']:+.2f}"},
                    {"Feature": "discounted_price", "Role": "Selling price. Together with MRP it defines the price ratio — the most direct signal of discount depth.", "r with Discount": f"{_corr_disc['discounted_price']:+.2f}"},
                    {"Feature": "rating", "Role": "Customer satisfaction (1–5). Links consumer perception to pricing strategy.", "r with Discount": f"{_corr_disc['rating']:+.2f}"},
                    {"Feature": "rating_count", "Role": "Number of reviews — proxy for sales volume and product maturity.", "r with Discount": f"{_corr_disc['rating_count']:+.2f}"},
                ]),
                use_container_width=True,
                hide_index=True,
            )

    with col_out:
        st.subheader("Prediction")
        with st.spinner(f"Running {disc_model_name}…"):
            disc_model = train_dynamic_model(disc_model_name, "discount", disc_params_json)
        # sklearn expects a 2-D array even for a single prediction row.
        feats = np.array([[actual_price, discounted_price, rating, float(rating_count)]])
        pred = float(disc_model.predict(feats)[0])
        # Tree models can extrapolate slightly outside [0, 100] on unusual inputs.
        pred = np.clip(pred, 0.0, 100.0)

        # Big result
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{pred:.1f}%</div>'
            f'<div class="kpi-label">Predicted Discount</div></div>',
            unsafe_allow_html=True,
        )
        with st.popover("ℹ️ What is this?", use_container_width=True):
            st.markdown("**Predicted Discount**")
            st.markdown("The RandomForest model's estimate of what discount percentage this product would typically carry. It learned this from 1,465 Amazon products — similar price-to-rating combinations in the training data shape this number.")
        st.progress(int(pred))

        # Implied discount is pure arithmetic: (MRP - Selling Price) / MRP × 100.
        # The model prediction often differs because it captures market patterns beyond the raw price ratio.
        implied = (1 - discounted_price / actual_price) * 100 if actual_price > 0 else 0.0
        colA, colB = st.columns(2)
        with colA:
            st.metric("Model Prediction", f"{pred:.1f}%")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**Model Prediction**")
                st.markdown("The RandomForest's output based on the pattern it learned from the training data. It captures non-obvious relationships between price, ratings, and typical discount levels.")
        with colB:
            st.metric("Implied by Prices", f"{implied:.1f}%", delta=f"{pred - implied:+.1f}pp vs implied")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**Implied by Prices**")
                st.markdown("Simple maths: `(MRP − Selling Price) ÷ MRP × 100`. This is the literal discount if you were to list at exactly these two prices. The model prediction often differs because real-world discounts reflect market positioning, not just the price gap.")

        # Drift detection: check_drift() compares each input value against the training
        # distribution (mean and std dev saved in training_stats.json).  It returns a dict
        # with "drift_detected" (bool) and "z_scores" (per-feature z-score).  The threshold
        # is 3.0 standard deviations — beyond that the input is in a region the model saw
        # very rarely, so predictions carry higher uncertainty.
        drift = check_drift(
            {"actual_price": actual_price, "discounted_price": discounted_price,
             "rating": rating, "rating_count": float(rating_count)},
            STATS_PATH,
        )
        if drift["drift_detected"]:
            st.markdown(
                f'<div class="drift-warn">⚠️ <strong>Input drift detected</strong> — '
                f'one or more values are >3σ from training distribution.<br>'
                f'<small>Z-scores: {drift["z_scores"]}</small></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="drift-ok">✅ Inputs within training distribution.</div>', unsafe_allow_html=True)
        with st.popover("ℹ️ What is drift?", use_container_width=True):
            st.markdown("**Input Drift**")
            st.markdown("Drift means your inputs look very different from what the model was trained on. Specifically, one or more values is more than 3 standard deviations from the training average. When drift is detected, the prediction is technically valid but comes from a region of the data the model has seen fewer examples of — treat it with extra caution.")

        # Context histogram
        st.markdown("**Where does this prediction fall in the dataset?**")
        pct = (df["discount_percentage"] <= pred).mean() * 100
        st.caption(f"Higher than {pct:.0f}% of products in the dataset.")
        fig, ax = dark_fig(6, 2.8)
        ax.hist(df["discount_percentage"], bins=40, color=BORDER, edgecolor=SURFACE, linewidth=0.3)
        ax.axvline(pred, color=PURPLE, linewidth=2.5, label=f"Predicted {pred:.1f}%")
        ax.legend(labelcolor=TEXT, facecolor=DARK, edgecolor=BORDER, fontsize=9)
        ax.set_xlabel("Discount (%)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Context Histogram**")
            st.markdown("The grey bars show the full spread of discount percentages in the dataset. The purple line marks your prediction. If the line is in the tail (far left or right), the model has seen fewer similar examples and may be less accurate.")


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT PRICE
# ══════════════════════════════════════════════════════════════════════════════
elif "Price" in page:
    st.title("💰 Predict Discounted Price")
    st.markdown("Choose a model, enter product details, and see the predicted selling price.")
    st.markdown("---")

    # ── Model selector ─────────────────────────────────────────────────────
    price_model_name = st.selectbox(
        "Model",
        list(_MODELS.keys()),
        index=0,   # Linear Regression default
        key="price_model",
    )
    _pcfg = _MODELS[price_model_name]
    with st.expander(f"📖 What is {price_model_name}?  {_pcfg['speed']}"):
        st.markdown(_pcfg["desc"])
        pc1, pc2 = st.columns(2)
        pc1.success(f"✅ **Strengths:** {_pcfg['pros']}")
        pc2.warning(f"⚠️ **Limitations:** {_pcfg['cons']}")
    price_params_json = _model_param_ui(price_model_name, "price")
    st.markdown("---")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Product Attributes")
        ap = st.number_input("Actual / MRP Price (₹)", min_value=10.0, max_value=200000.0, value=999.0, step=50.0)
        disc_pct = st.slider("Discount Percentage (%)", 0.0, 95.0, 40.0, 1.0)
        rat = st.slider("Rating", 1.0, 5.0, 4.0, 0.1)
        rat_cnt = st.number_input("Number of Ratings", min_value=1, max_value=500000, value=5000, step=100)

        with st.expander("📖 Why these features?"):
            _corr_price = correlation_matrix(df)["discounted_price"]
            st.dataframe(
                pd.DataFrame([
                    {"Feature": "actual_price", "Role": "Original list price (MRP). The strongest predictor of selling price — higher MRP almost always means higher selling price.", "r with Price": f"{_corr_price['actual_price']:+.2f}"},
                    {"Feature": "discount_percentage", "Role": "Discount applied. Directly determines how far below MRP the product sells.", "r with Price": f"{_corr_price['discount_percentage']:+.2f}"},
                    {"Feature": "rating", "Role": "Customer satisfaction (1–5). Better-rated products can sustain higher prices.", "r with Price": f"{_corr_price['rating']:+.2f}"},
                    {"Feature": "rating_count", "Role": "Number of reviews — proxy for sales volume. High-volume products face different price pressures.", "r with Price": f"{_corr_price['rating_count']:+.2f}"},
                ]),
                use_container_width=True,
                hide_index=True,
            )

    with col_out:
        st.subheader("Prediction")
        with st.spinner(f"Running {price_model_name}…"):
            price_model = train_dynamic_model(price_model_name, "price", price_params_json)
        feats = np.array([[ap, rat, float(rat_cnt), disc_pct]])
        pred_price = float(price_model.predict(feats)[0])
        # Formula price is the arithmetic baseline: MRP × (1 − discount/100).
        # Comparing it to the model prediction highlights the non-linear market signals
        # (brand, review volume, category) that the formula cannot capture.
        formula_price = ap * (1 - disc_pct / 100)

        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value">₹{pred_price:,.0f}</div>'
            f'<div class="kpi-label">Predicted Selling Price</div></div>',
            unsafe_allow_html=True,
        )
        with st.popover("ℹ️ What is this?", use_container_width=True):
            st.markdown("**Predicted Selling Price**")
            st.markdown("The LinearRegression model's estimate of what this product would actually sell for given its MRP, rating, review count, and discount level. It learned from 1,465 real Amazon listings.")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Model Prediction", f"₹{pred_price:,.0f}")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**Model Prediction**")
                st.markdown("The LinearRegression's output. It weights each input feature (MRP, rating, etc.) differently based on what it learned from the data, giving a more nuanced estimate than a simple formula.")
        with c2:
            st.metric("Formula (MRP × discount)", f"₹{formula_price:,.0f}", delta=f"₹{pred_price - formula_price:+,.0f}")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**Formula Price**")
                st.markdown("Simple maths: `MRP × (1 − discount %)`. The model differs from this because real product prices are influenced by brand positioning, review volume, and category norms — not just the raw discount applied to MRP.")

        st.markdown("---")
        # coef_ exists only on linear models (LinearRegression, Ridge).
        # feature_importances_ exists only on tree-based models (RandomForest, GradientBoosting).
        # Checking hasattr avoids a hard-coded model-type check and keeps the UI adaptive.
        if hasattr(price_model, "coef_"):
            st.subheader("Model Coefficients")
            coef_df = pd.DataFrame(
                {"Feature": _PRICE_FEATURES, "Coefficient": price_model.coef_}
            )
            fig, ax = dark_fig(6, 3)
            colors = [GREEN if c > 0 else RED for c in coef_df["Coefficient"]]
            ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
            ax.axvline(0, color=TEXT, linewidth=0.8)
            ax.set_xlabel("Coefficient value")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            with st.popover("ℹ️ What does this show?", use_container_width=True):
                st.markdown("**Model Coefficients**")
                st.markdown("Each bar shows how much one feature shifts the predicted price. **Green (positive)** = higher value raises the predicted price. **Red (negative)** = higher value lowers it. The longer the bar, the stronger the influence. For example, a high `discount_percentage` is negative — more discount means lower selling price.")
        elif hasattr(price_model, "feature_importances_"):
            st.subheader("Feature Importance")
            imp_df = pd.DataFrame({"Feature": _PRICE_FEATURES, "Importance": price_model.feature_importances_}).sort_values("Importance")
            fig, ax = dark_fig(6, 3)
            ax.barh(imp_df["Feature"], imp_df["Importance"], color=PURPLE, edgecolor=SURFACE)
            ax.set_xlabel("Importance")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            with st.popover("ℹ️ What does this show?", use_container_width=True):
                st.markdown("**Feature Importance**")
                st.markdown("Which inputs this model relied on most. A longer bar means that feature had more influence on the prediction.")

        st.markdown("---")
        st.subheader("Price Range in Dataset")
        fig, ax = dark_fig(6, 2.5)
        clipped = df["discounted_price"].clip(upper=df["discounted_price"].quantile(0.97))
        ax.hist(clipped, bins=40, color=BORDER, edgecolor=SURFACE, linewidth=0.3)
        ax.axvline(pred_price, color=GREEN, linewidth=2.5, label=f"Predicted ₹{pred_price:,.0f}")
        ax.legend(labelcolor=TEXT, facecolor=DARK, edgecolor=BORDER, fontsize=9)
        ax.set_xlabel("Discounted Price (₹)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Price Range Context**")
            st.markdown("The grey bars show the full spread of selling prices in the dataset (top 3% clipped). The green line marks your prediction. If the line is far from the main cluster, the model has seen fewer similar products and the prediction may be less accurate.")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif "Insights" in page:
    st.title("📈 Model Insights")
    st.markdown("---")

    _model_names = list(_MODELS.keys())
    ic1, ic2 = st.columns(2)
    with ic1:
        # key="disc_model" is shared with the Predict Discount page selectbox.
        # Streamlit session state means whichever page the user visited last sets the
        # initial value here — the two pages stay in sync without any extra wiring.
        ins_disc_model = st.selectbox(
            "Discount model",
            _model_names,
            index=_model_names.index(st.session_state.get("disc_model", "Random Forest")),
            key="disc_model",
        )
        ins_disc_params_json = _model_param_ui(ins_disc_model, "disc")
    with ic2:
        # Same session state sharing for the price model.
        ins_price_model = st.selectbox(
            "Price model",
            _model_names,
            index=_model_names.index(st.session_state.get("price_model", "Linear Regression")),
            key="price_model",
        )
        ins_price_params_json = _model_param_ui(ins_price_model, "price")

    # params_json must be passed to both dynamic_test_predictions and train_dynamic_model
    # so the cache key matches.  If they differed, the metrics would be computed with
    # different hyperparameters than the model shown in the importance chart below.
    with st.spinner("Computing evaluation metrics…"):
        y_test_disc, y_pred_disc, disc_metrics = dynamic_test_predictions(df, ins_disc_model, "discount", ins_disc_params_json)
        y_test_price, y_pred_price, price_metrics = dynamic_test_predictions(df, ins_price_model, "price", ins_price_params_json)

    # ── Metrics cards ──────────────────────────────────────────────────────
    st.subheader("Evaluation Metrics")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"#### Discount — {ins_disc_model}")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("R²", f"{disc_metrics['r2']:.4f}")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**R² (R-squared)**")
                st.markdown("How much of the variation in discount percentages the model explains. 0 = no better than guessing the average; 1 = perfect. Closer to 1 is better.")
        with m2:
            st.metric("RMSE", f"{disc_metrics['rmse']:.2f} pp")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**RMSE (Root Mean Squared Error)**")
                st.markdown("Typical prediction error in percentage points. Larger errors are penalised more than smaller ones.")
        with m3:
            st.metric("MAE", f"{disc_metrics['mae']:.2f} pp")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**MAE (Mean Absolute Error)**")
                st.markdown("Average absolute difference between predicted and actual discounts, in percentage points. Unlike RMSE, large errors aren't penalised extra.")

    with c2:
        st.markdown(f"#### Price — {ins_price_model}")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("R²", f"{price_metrics['r2']:.4f}")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**R² (R-squared)**")
                st.markdown("How much of the variation in selling prices the model explains. Closer to 1 is better.")
        with m2:
            st.metric("RMSE", f"₹{price_metrics['rmse']:.0f}")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**RMSE (Root Mean Squared Error)**")
                st.markdown("Typical prediction error in rupees. High-value products naturally have larger absolute errors.")
        with m3:
            st.metric("MAE", f"₹{price_metrics['mae']:.0f}")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**MAE (Mean Absolute Error)**")
                st.markdown("Average absolute price prediction error. A few large misses can inflate RMSE more than MAE.")

    st.markdown("---")

    # ── Feature importance / coefficients ──────────────────────────────────
    disc_model_obj = train_dynamic_model(ins_disc_model, "discount", ins_disc_params_json)
    feature_names_disc = ["actual_price", "discounted_price", "rating", "rating_count"]

    # Same adaptive hasattr check as on the Predict Price page — shows importances for tree
    # models and coefficients for linear models without branching on model name.
    if hasattr(disc_model_obj, "feature_importances_"):
        st.subheader(f"Feature Importance — {ins_disc_model}")
        imp_df = pd.DataFrame({"Feature": feature_names_disc, "Importance": disc_model_obj.feature_importances_}).sort_values("Importance")
        fig, ax = dark_fig(8, 3)
        bars = ax.barh(imp_df["Feature"], imp_df["Importance"], color=PURPLE, edgecolor=SURFACE)
        for bar, val in zip(bars, imp_df["Importance"]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", color=TEXT, fontsize=9)
        ax.set_xlabel("Importance")
        ax.set_xlim(0, imp_df["Importance"].max() * 1.18)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Feature Importance**")
            st.markdown("Which inputs the model relied on most to predict discounts. A higher bar = that feature drove more of the decision. Price features dominate because the price ratio directly determines the discount — rating and review count add nuance.")
    elif hasattr(disc_model_obj, "coef_"):
        st.subheader(f"Feature Coefficients — {ins_disc_model}")
        coef_df = pd.DataFrame({"Feature": feature_names_disc, "Coefficient": disc_model_obj.coef_}).sort_values("Coefficient")
        fig, ax = dark_fig(8, 3)
        colors = [GREEN if v >= 0 else RED for v in coef_df["Coefficient"]]
        bars = ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor=SURFACE)
        ax.axvline(0, color=TEXT, linewidth=0.8, linestyle="--")
        ax.set_xlabel("Coefficient")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Feature Coefficients**")
            st.markdown("Each bar shows how much one unit increase in that feature changes the predicted discount. Green = raises the predicted discount; red = lowers it. Longer bar = stronger effect.")

    st.markdown("---")

    # ── Actual vs Predicted ────────────────────────────────────────────────
    st.subheader("Actual vs Predicted")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"**Discount % — {ins_disc_model}**")
        fig, ax = dark_fig(6, 4.5)
        ax.scatter(y_test_disc, y_pred_disc, alpha=0.35, color=BLUE, s=18, label="Predictions")
        mn, mx = min(y_test_disc.min(), y_pred_disc.min()), max(y_test_disc.max(), y_pred_disc.max())
        ax.plot([mn, mx], [mn, mx], "--", color=RED, linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual Discount (%)")
        ax.set_ylabel("Predicted Discount (%)")
        ax.legend(labelcolor=TEXT, facecolor=DARK, edgecolor=BORDER, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Actual vs Predicted — Discount**")
            st.markdown("Each dot is one product from the held-out test set. The x-axis is the real discount; the y-axis is what the model predicted. Dots along the red diagonal = perfect predictions. Scatter away from it = error. A tight cluster along the line means the model generalises well.")

    with c2:
        st.markdown(f"**Discounted Price ₹ — {ins_price_model}**")
        fig, ax = dark_fig(6, 4.5)
        cap = np.percentile(y_test_price, 97)
        mask = (y_test_price <= cap) & (y_pred_price <= cap)
        ax.scatter(y_test_price[mask], y_pred_price[mask], alpha=0.35, color=GREEN, s=18, label="Predictions")
        mn, mx = y_test_price[mask].min(), y_test_price[mask].max()
        ax.plot([mn, mx], [mn, mx], "--", color=RED, linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual Price (₹)")
        ax.set_ylabel("Predicted Price (₹)")
        ax.legend(labelcolor=TEXT, facecolor=DARK, edgecolor=BORDER, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Actual vs Predicted — Price**")
            st.markdown("Same chart for the price model (top 3% of prices clipped so the main cluster is visible). Dots on the diagonal = accurate. Spread at higher prices reflects that premium products are harder to price precisely from just four features.")

    st.markdown("---")

    # ── Residuals ──────────────────────────────────────────────────────────
    st.subheader("Residuals Distribution")
    c1, c2 = st.columns(2)

    with c1:
        residuals_disc = y_test_disc - y_pred_disc
        fig, ax = dark_fig(6, 3.5)
        ax.hist(residuals_disc, bins=40, color=PURPLE, edgecolor=SURFACE, linewidth=0.3)
        ax.axvline(0, color=RED, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Residual (Actual − Predicted) %")
        ax.set_ylabel("Count")
        ax.set_title(f"{ins_disc_model} Residuals")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Discount Model Residuals**")
            st.markdown("A residual is `actual − predicted` for each test product. A bell-shaped histogram centred on 0 (the red dashed line) means errors are random and symmetric — the model isn't systematically over- or under-predicting. That's a sign of a healthy, unbiased model.")

    with c2:
        residuals_price = y_test_price - y_pred_price
        # Clip to the 2nd–98th percentile so a handful of very expensive outliers don't
        # collapse the histogram into a spike, hiding the shape of the main distribution.
        clipped_res = np.clip(residuals_price, np.percentile(residuals_price, 2), np.percentile(residuals_price, 98))
        fig, ax = dark_fig(6, 3.5)
        ax.hist(clipped_res, bins=40, color=GREEN, edgecolor=SURFACE, linewidth=0.3)
        ax.axvline(0, color=RED, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Residual (Actual − Predicted) ₹")
        ax.set_ylabel("Count")
        ax.set_title(f"{ins_price_model} Residuals (2–98th pct)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Price Model Residuals**")
            st.markdown("Same idea for the price model, clipped at the 2nd–98th percentile to remove extreme outliers. Centred near 0 = good. A slight right skew would mean the model occasionally under-predicts high-end prices.")

    st.markdown("---")

    # ── Training stats table ───────────────────────────────────────────────
    st.subheader("Training Data Statistics")
    with open(STATS_PATH) as f:
        stats = json.load(f)
    stats_df = pd.DataFrame(stats).T
    stats_df.columns = ["Mean", "Std Dev"]
    st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
