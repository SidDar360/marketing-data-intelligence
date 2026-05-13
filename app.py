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
import streamlit as st
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import (
    get_feature_target_for_discount,
    get_feature_target_for_price,
    load_and_clean_data,
)
from src.models import check_drift, evaluate_model, load_model

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
        ["🏠  Overview", "🏷️  Predict Discount", "💰  Predict Price", "📈  Model Insights"],
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
        st.markdown(
            """
| Component | Technology |
|-----------|-----------|
| Data | Amazon Sales CSV |
| Discount model | Random Forest |
| Price model | Linear Regression |
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
# PREDICT DISCOUNT
# ══════════════════════════════════════════════════════════════════════════════
elif "Discount" in page:
    st.title("🏷️ Predict Discount Percentage")
    st.markdown("Adjust the product attributes — the RandomForest model predicts the discount in real-time.")
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

    with col_out:
        st.subheader("Prediction")
        rf = get_rf_model()
        feats = np.array([[actual_price, discounted_price, rating, float(rating_count)]])
        pred = float(rf.predict(feats)[0])
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
    st.markdown("Enter product details to forecast the selling price using the Linear Regression model.")
    st.markdown("---")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Product Attributes")
        ap = st.number_input("Actual / MRP Price (₹)", min_value=10.0, max_value=200000.0, value=999.0, step=50.0)
        disc_pct = st.slider("Discount Percentage (%)", 0.0, 95.0, 40.0, 1.0)
        rat = st.slider("Rating", 1.0, 5.0, 4.0, 0.1)
        rat_cnt = st.number_input("Number of Ratings", min_value=1, max_value=500000, value=5000, step=100)

    with col_out:
        st.subheader("Prediction")
        lr = get_lr_model()
        feats = np.array([[ap, rat, float(rat_cnt), disc_pct]])
        pred_price = float(lr.predict(feats)[0])
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
        st.subheader("Model Coefficients")
        coef_df = pd.DataFrame(
            {"Feature": ["actual_price", "rating", "rating_count", "discount_percentage"], "Coefficient": lr.coef_}
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

    with st.spinner("Computing evaluation metrics…"):
        y_test_rf, y_pred_rf, rf_metrics = rf_test_predictions(df)
        y_test_lr, y_pred_lr, lr_metrics = lr_test_predictions(df)

    # ── Metrics cards ──────────────────────────────────────────────────────
    st.subheader("Evaluation Metrics")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 🌲 RandomForest — Discount Prediction")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("R²", f"{rf_metrics['r2']:.4f}")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**R² (R-squared)**")
                st.markdown("How much of the variation in discount percentages the model explains. 0 = no better than guessing the average; 1 = perfect. **0.967** means the model explains 96.7% of the variation — excellent.")
        with m2:
            st.metric("RMSE", f"{rf_metrics['rmse']:.2f} pp")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**RMSE (Root Mean Squared Error)**")
                st.markdown("The typical size of prediction errors, in percentage points. **3.78 pp** means the model is off by about 3.78 percentage points on average. Larger errors are penalised more than smaller ones.")
        with m3:
            st.metric("MAE", f"{rf_metrics['mae']:.2f} pp")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**MAE (Mean Absolute Error)**")
                st.markdown("The average absolute difference between predicted and actual discounts, in percentage points. **2.13 pp** means predictions are typically within ~2 points of the real value. Unlike RMSE, large errors aren't penalised extra.")

    with c2:
        st.markdown("#### 📉 LinearRegression — Price Prediction")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("R²", f"{lr_metrics['r2']:.4f}")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**R² (R-squared)**")
                st.markdown("How much of the variation in selling prices the model explains. **0.951** means 95.1% explained — very strong. The remaining 4.9% reflects factors not captured in the four input features.")
        with m2:
            st.metric("RMSE", f"₹{lr_metrics['rmse']:.0f}")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**RMSE (Root Mean Squared Error)**")
                st.markdown("Typical prediction error in rupees. **₹1,200** means the model's price estimates are off by about ₹1,200 on average. High-value products naturally have larger absolute errors.")
        with m3:
            st.metric("MAE", f"₹{lr_metrics['mae']:.0f}")
            with st.popover("ℹ️", use_container_width=True):
                st.markdown("**MAE (Mean Absolute Error)**")
                st.markdown("Average absolute price prediction error. **₹733** means typical predictions are within ₹733 of the actual selling price — better than RMSE suggests because a few large misses inflate RMSE.")

    st.markdown("---")

    # ── Feature importance ─────────────────────────────────────────────────
    st.subheader("Feature Importance — RandomForest")
    rf = get_rf_model()
    feature_names = ["actual_price", "discounted_price", "rating", "rating_count"]
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": rf.feature_importances_}).sort_values("Importance")

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
        st.markdown("Which inputs the RandomForest relied on most to predict discounts. A higher bar = that feature was used in more decision splits. **actual_price** and **discounted_price** dominate because the price ratio directly determines the discount — rating and review count add nuance.")

    st.markdown("---")

    # ── Actual vs Predicted ────────────────────────────────────────────────
    st.subheader("Actual vs Predicted")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Discount % (RandomForest)**")
        fig, ax = dark_fig(6, 4.5)
        ax.scatter(y_test_rf, y_pred_rf, alpha=0.35, color=BLUE, s=18, label="Predictions")
        mn, mx = min(y_test_rf.min(), y_pred_rf.min()), max(y_test_rf.max(), y_pred_rf.max())
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
        st.markdown("**Discounted Price ₹ (LinearRegression)**")
        fig, ax = dark_fig(6, 4.5)
        cap = np.percentile(y_test_lr, 97)
        mask = (y_test_lr <= cap) & (y_pred_lr <= cap)
        ax.scatter(y_test_lr[mask], y_pred_lr[mask], alpha=0.35, color=GREEN, s=18, label="Predictions")
        mn, mx = y_test_lr[mask].min(), y_test_lr[mask].max()
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
        residuals_rf = y_test_rf - y_pred_rf
        fig, ax = dark_fig(6, 3.5)
        ax.hist(residuals_rf, bins=40, color=PURPLE, edgecolor=SURFACE, linewidth=0.3)
        ax.axvline(0, color=RED, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Residual (Actual − Predicted) %")
        ax.set_ylabel("Count")
        ax.set_title("RF Residuals")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Discount Model Residuals**")
            st.markdown("A residual is `actual − predicted` for each test product. A bell-shaped histogram centred on 0 (the red dashed line) means errors are random and symmetric — the model isn't systematically over- or under-predicting. That's a sign of a healthy, unbiased model.")

    with c2:
        residuals_lr = y_test_lr - y_pred_lr
        clipped_res = np.clip(residuals_lr, np.percentile(residuals_lr, 2), np.percentile(residuals_lr, 98))
        fig, ax = dark_fig(6, 3.5)
        ax.hist(clipped_res, bins=40, color=GREEN, edgecolor=SURFACE, linewidth=0.3)
        ax.axvline(0, color=RED, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Residual (Actual − Predicted) ₹")
        ax.set_ylabel("Count")
        ax.set_title("LR Residuals (2–98th pct)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.popover("ℹ️ What does this show?", use_container_width=True):
            st.markdown("**Price Model Residuals**")
            st.markdown("Same idea for the price model, clipped at the 2nd–98th percentile to remove extreme outliers. Centred near 0 = good. A slight right skew would mean the model occasionally under-predicts high-end prices — worth watching if you're using this for premium product pricing.")

    st.markdown("---")

    # ── Training stats table ───────────────────────────────────────────────
    st.subheader("Training Data Statistics")
    with open(STATS_PATH) as f:
        stats = json.load(f)
    stats_df = pd.DataFrame(stats).T
    stats_df.columns = ["Mean", "Std Dev"]
    st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
