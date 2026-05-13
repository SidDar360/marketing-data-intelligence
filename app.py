"""
Marketing Data Intelligence — Streamlit Demo App
Run: streamlit run app.py
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

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import (
    get_feature_target_for_discount,
    get_feature_target_for_price,
    load_and_clean_data,
)
from src.models import check_drift, evaluate_model, load_model

# ── Model registry ────────────────────────────────────────────────────────────
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

_DISCOUNT_FEATURES = ["actual_price", "discounted_price", "rating", "rating_count"]
_PRICE_FEATURES    = ["actual_price", "rating", "rating_count", "discount_percentage"]

# Per-model tunable parameters: (param_name, type, min, max, default, step, short_help, explanation)
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
    """Render parameter sliders for `model_name` and return a JSON string of overrides."""
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

# ── Matplotlib dark theme helper ──────────────────────────────────────────────
DARK = "#1e1e2e"
SURFACE = "#181825"
BORDER = "#313244"
TEXT = "#cdd6f4"
PURPLE = "#cba6f7"
BLUE = "#89b4fa"
GREEN = "#a6e3a1"
RED = "#f38ba8"
PEACH = "#fab387"


def dark_fig(w=7, h=4):
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


# ── Cached resource loaders ───────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_and_clean_data(CSV_PATH)


@st.cache_resource
def get_rf_model():
    return load_model("random_forest_discount")


@st.cache_resource
def get_lr_model():
    return load_model("linear_regression_price")


@st.cache_resource
def train_dynamic_model(model_name: str, task: str, params_json: str = "{}"):
    df_tr = load_and_clean_data(CSV_PATH)
    if task == "discount":
        X, y = get_feature_target_for_discount(df_tr)
    else:
        X, y = get_feature_target_for_price(df_tr)
    cfg = _MODELS[model_name]
    merged = {**cfg["params"], **json.loads(params_json)}
    m = cfg["cls"](**merged)
    m.fit(X, y)
    return m


@st.cache_data
def dynamic_test_predictions(_df, model_name: str, task: str, params_json: str = "{}"):
    model = train_dynamic_model(model_name, task, params_json)
    if task == "discount":
        X, y = get_feature_target_for_discount(_df)
    else:
        X, y = get_feature_target_for_price(_df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return y_test.values, model.predict(X_test), evaluate_model(model, X_test, y_test)


@st.cache_data
def rf_test_predictions(_df):
    model = get_rf_model()
    X, y = get_feature_target_for_discount(_df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return y_test.values, model.predict(X_test), evaluate_model(model, X_test, y_test)


@st.cache_data
def lr_test_predictions(_df):
    model = get_lr_model()
    X, y = get_feature_target_for_price(_df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return y_test.values, model.predict(X_test), evaluate_model(model, X_test, y_test)


@st.cache_data
def correlation_matrix(_df):
    cols = ["actual_price", "discounted_price", "discount_percentage", "rating", "rating_count"]
    return _df[cols].corr()


@st.cache_data
def feature_selection_path(_df):
    """Greedy forward selection on discount features using LinearRegression."""
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


def artifacts_ok():
    needed = ["random_forest_discount.pkl", "linear_regression_price.pkl", "training_stats.json"]
    return all(os.path.exists(os.path.join(ARTIFACTS_DIR, f)) for f in needed)


def _run_training():
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
    save_training_stats(X_disc)
    X_price, y_price = get_feature_target_for_price(df_train)
    lr, _ = train_linear_regression(X_price, y_price)
    save_model(lr, "linear_regression_price")


# ── Sidebar ───────────────────────────────────────────────────────────────────
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

if not artifacts_ok():
    with st.spinner("⚙️ First run — training models on the dataset (takes ~30 s)…"):
        _run_training()
    st.rerun()

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

    corr = correlation_matrix(df)
    _NUMERIC_COLS = ["actual_price", "discounted_price", "discount_percentage", "rating", "rating_count"]
    _FEAT_LABELS = {
        "actual_price": "Actual Price (MRP)",
        "discounted_price": "Discounted Price",
        "discount_percentage": "Discount %",
        "rating": "Rating",
        "rating_count": "Rating Count",
    }
    _FEAT_CONTEXT = {
        "actual_price": "Premium products (higher MRP) are often discounted more aggressively to attract buyers.",
        "discounted_price": "The selling price naturally moves with the discount — a lower selling price relative to MRP means a bigger discount.",
        "rating": "Customer satisfaction and discount levels are linked — either high-rated products attract larger discounts, or discounts drive more purchases and reviews.",
        "rating_count": "More popular products (more reviews) may face different pricing dynamics than niche ones.",
    }

    # ── Section 1: Correlation Heatmap ─────────────────────────────────────
    st.subheader("Correlation Heatmap")
    readable_labels = [_FEAT_LABELS.get(c, c) for c in _NUMERIC_COLS]
    corr_display = corr.copy()
    corr_display.index = readable_labels
    corr_display.columns = readable_labels

    fig, ax = dark_fig(9, 6)
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
        annot_kws={"size": 10},
        cbar_kws={"shrink": 0.8},
    )
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", color=TEXT)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color=TEXT)
    ax.collections[0].colorbar.ax.tick_params(colors=TEXT)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
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

    st.markdown("---")

    # ── Section 2: Plain-English Insights ─────────────────────────────────
    st.subheader("What the Data Tells Us")
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
    for i, c1 in enumerate(_NUMERIC_COLS):
        for j, c2 in enumerate(_NUMERIC_COLS):
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

    fig, ax = dark_fig(8, 3)
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

    # ── Section 4: Minimum Feature Analysis ───────────────────────────────
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
        feats = np.array([[actual_price, discounted_price, rating, float(rating_count)]])
        pred = float(disc_model.predict(feats)[0])
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

        # Implied vs actual discount
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

        # Drift
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
        ins_disc_model = st.selectbox(
            "Discount model",
            _model_names,
            index=_model_names.index(st.session_state.get("disc_model", "Random Forest")),
            key="disc_model",
        )
        ins_disc_params_json = _model_param_ui(ins_disc_model, "disc")
    with ic2:
        ins_price_model = st.selectbox(
            "Price model",
            _model_names,
            index=_model_names.index(st.session_state.get("price_model", "Linear Regression")),
            key="price_model",
        )
        ins_price_params_json = _model_param_ui(ins_price_model, "price")

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
