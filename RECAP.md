# Marketing Data Intelligence — Project Recap

A full record of what was built, why decisions were made, and how the app evolved.

---

## Where It Started

The project began as a Jupyter notebook doing exploratory data analysis on the Amazon Sales Dataset (1,465 products). A basic linear regression model was added to predict selling price. From there, the system was rebuilt from scratch into a production-grade ML application with:

- A proper data preprocessing pipeline
- Two trained models saved as `.pkl` artifacts
- A RAG (Retrieval-Augmented Generation) assistant using FAISS + flan-t5
- A FastAPI REST API with Docker support
- A Streamlit UI

That full rebuild was the starting point for all the work described below.

---

## Phase 1 — Streamlit Community Cloud Deployment

**Goal:** Host the app for free on Streamlit Community Cloud.

**Problem:** The original app depended on `torch`, `transformers`, `faiss-cpu`, and `sentence-transformers` to run the local flan-t5 AI assistant. Streamlit's free tier has ~1 GB RAM — these packages alone exceed that.

**What changed:**

- Removed all heavy ML packages (`torch`, `transformers`, `faiss-cpu`, `sentence-transformers`, `fastapi`, `uvicorn`, `httpx`, `pydantic`) from `requirements.txt`
- The app no longer loads pre-trained model artifacts from disk. Instead, it trains all models fresh on first boot using `@st.cache_resource` so training only happens once per session
- Added `_run_training()` — auto-detects missing artifacts and trains on startup
- Added `.streamlit/config.toml` with `headless = true` and dark theme settings
- Added `.streamlit/secrets.toml` to `.gitignore` to keep API keys out of the repo

**Result:** `requirements.txt` reduced to 6 packages (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`).

---

## Phase 2 — Metric Explanation Popovers

**Goal:** Make every number in the app self-explanatory — users should be able to click any KPI or chart and understand what it means without any ML background.

**What changed:**

Added `st.popover("ℹ️")` buttons throughout the entire app:

| Page | Elements with popovers |
|------|----------------------|
| Overview | 5 KPI cards, 4 charts |
| Predict Discount | Predicted discount card, implied discount metric, drift warning, context histogram |
| Predict Price | Predicted price card, formula price metric, coefficients/importance chart, price range histogram |
| Model Insights | R², RMSE, MAE for both models, feature importance chart, actual vs predicted charts, residuals charts |

Each popover contains a plain-English explanation of what the number means, how to interpret it, and what it would mean if it were higher or lower.

---

## Phase 3 — AI Assistant (Added then Removed)

**Added:** A chat page powered by the Claude API (`claude-haiku-4-5-20251001`). Used `st.chat_input` and `st.chat_message` for native Streamlit chat UI with streaming responses. The system prompt injected dataset context (model metrics, avg discount, category count, etc.) so Claude could answer questions about the specific data.

**Removed:** At user request, the feature was pulled out entirely and merged via PR #2. The `anthropic` package was removed from `requirements.txt`. The rationale: keeping the app dependency-free and self-contained.

---

## Phase 4 — Model Selector

**Goal:** Let users choose which ML algorithm to use instead of being locked into Random Forest for discount and Linear Regression for price.

**Four models added:**

| Model | Task default | Tunable |
|-------|-------------|---------|
| Linear Regression | Price (default) | None — closed-form solution |
| Ridge Regression | — | `alpha` |
| Random Forest | Discount (default) | `n_estimators`, `max_depth`, `min_samples_split` |
| Gradient Boosting | — | `n_estimators`, `learning_rate`, `max_depth` |

**How it works technically:**

- `_MODELS` registry stores each model's class, default params, speed label, description, pros, and cons
- `train_dynamic_model(model_name, task, params_json)` — `@st.cache_resource`, keyed on (model, task, params). Each unique combination trains once and is reused.
- `dynamic_test_predictions(_df, model_name, task, params_json)` — `@st.cache_data`, splits the dataset 80/20, returns `(y_test, y_pred, metrics)`
- Model selectboxes use `key="disc_model"` and `key="price_model"` across all pages — Streamlit session state means the selection is shared automatically between Predict, Insights, and Overview pages

**UI on each page:**

- A `st.selectbox` to pick the model
- A `📖 What is X?` expander explaining the algorithm, its strengths, and its limitations
- A `⚙️ Fine-tune parameters` expander with sliders for tunable params (see Phase 5)

**Pages updated to be model-aware:**

- **Predict Discount** — selectbox, trains chosen model, shows prediction
- **Predict Price** — selectbox, trains chosen model, adaptive chart (coefficients for linear models, feature importance for tree models)
- **Model Insights** — two selectboxes at top; all metrics, charts, and residuals driven by the selected models; feature section adapts (importance bars for trees, coefficient bars for linear models)
- **Overview** — architecture table now reads from session state and shows the currently selected models

---

## Phase 5 — Parameter Tuning UI

**Goal:** Let users fine-tune hyperparameters directly in the app and immediately see the effect on model performance.

**How it works:**

- `_TUNABLE` dict defines each tunable parameter: name, type, range, default, step, tooltip, and a detailed explanation
- `_model_param_ui(model_name, key_prefix)` renders sliders inside a `⚙️ Fine-tune parameters` expander and returns a JSON string of the user's chosen values
- That JSON string is passed to `train_dynamic_model` and merged with the model's base params — e.g. changing `n_estimators` from 200 to 50 retriggers training and caches the new model under a different key
- Widget keys use a consistent prefix (`params_disc_*` / `params_price_*`) so values persist across pages

**Parameters exposed:**

| Model | Parameter | Range | What it does |
|-------|-----------|-------|-------------|
| Ridge | `alpha` | 0.01 – 100 | Regularization strength. Low = like plain LR. High = aggressive shrinkage, may underfit. |
| Random Forest | `n_estimators` | 50 – 500 | Number of trees. More = stable but slower. Diminishing returns past ~200. |
| Random Forest | `max_depth` | 0 – 30 | Max tree depth. 0 = unlimited. Lower reduces overfitting. |
| Random Forest | `min_samples_split` | 2 – 20 | Min samples to split a node. Higher = simpler trees. |
| Gradient Boosting | `n_estimators` | 50 – 500 | Boosting stages. Pair with lower learning rate for best results. |
| Gradient Boosting | `learning_rate` | 0.01 – 0.5 | Most impactful knob. Lower rate + more estimators = better generalisation. |
| Gradient Boosting | `max_depth` | 2 – 10 | Keep at 3–5. Deeper trees overfit quickly with boosting. |

Linear Regression shows an info message explaining it has no tunable parameters.

---

## Current State of the App

### Pages

**Overview**
- 5 KPI cards: total products, categories, avg discount, avg rating, median price
- 4 charts: discount distribution, rating distribution, category breakdown, price vs discount scatter
- Architecture table showing currently selected models (reads from session state)
- Sample products table (top 30 rows)

**Predict Discount**
- Model selector (default: Random Forest)
- Model description expander
- Parameter tuning expander
- Inputs: MRP, selling price, rating, rating count + quick presets (Budget / Mid-range / Premium)
- Outputs: predicted discount %, implied discount from formula, drift warning if inputs are out of distribution, context histogram showing where prediction falls in dataset

**Predict Price**
- Model selector (default: Linear Regression)
- Model description expander
- Parameter tuning expander
- Inputs: MRP, discount %, rating, rating count
- Outputs: predicted price, formula price, adaptive chart (coefficients or feature importance depending on model), price range histogram

**Model Insights**
- Two model selectors (one for discount, one for price) with parameter tuners — synced with predict pages via session state
- Evaluation metrics (R², RMSE, MAE) for both models
- Feature importance or coefficient chart (adapts to model type)
- Actual vs predicted scatter plots for both models
- Residuals distribution histograms for both models

### File Structure

```
app.py                  Entire Streamlit app (~900 lines)
amazon.csv              Raw dataset — 1,465 Amazon products
train.py                Standalone training script (for local/Docker use)
requirements.txt        6 packages: pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit
src/
  data_preprocessing.py  load_and_clean_data(), feature extraction functions
  models.py              evaluate_model(), check_drift(), save/load_model()
  rag.py                 Original RAG pipeline (not used by Streamlit app)
  api.py                 FastAPI app (not used by Streamlit app)
tests/                  Unit tests for src/ modules
artifacts/              Gitignored — .pkl models, training_stats.json (generated at runtime)
.streamlit/
  config.toml           headless = true, dark theme
  secrets.toml          ANTHROPIC_API_KEY placeholder (gitignored)
Dockerfile              For running the FastAPI service
docker-compose.yml      Mounts artifacts/, sets LLM_MODEL env var
```

### Key Technical Decisions

**Auto-training on boot** — `@st.cache_resource` on `train_dynamic_model` means every unique (model, task, params) combination trains once per session. No pre-built artifacts required. This is what makes Streamlit Community Cloud deployment work.

**Session state key sharing** — Using the same `key="disc_model"` on selectboxes across the Predict and Insights pages means the user's model choice is automatically shared between pages without any manual sync code.

**`params_json` as cache key** — Hyperparameter changes are captured as a JSON string which becomes part of the `@st.cache_resource` key. Different param combinations each get their own cached model, so switching between configurations is instant after the first train.

**Adaptive charts** — `hasattr(model, "coef_")` for linear models vs `hasattr(model, "feature_importances_")` for tree models lets the same chart section work correctly for all four model types.

---

## Git History Summary

| PR / Commit | What it did |
|-------------|------------|
| Initial commits | EDA notebook, basic linear regression, production rebuild (FastAPI + RAG + Streamlit) |
| `23829d1` | Prepared for Streamlit Community Cloud — stripped heavy deps, added auto-training |
| `c8eacc6` | Added AI assistant (Claude API) + all metric explanation popovers |
| PR #2 | Removed AI assistant |
| PR #3 | Model selector + parameter tuning + insights/overview updates |
| PR #4 | Removed CLAUDE.md |
