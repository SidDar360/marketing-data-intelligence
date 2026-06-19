# Marketing Data Intelligence

An end-to-end machine learning system for e-commerce analytics built on the [Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset). The system provides discount and price prediction models with multiple selectable algorithms, an interactive Streamlit demo with Feature Explorer and PCA Explorer pages, and a REST API.

---

## Features

| Capability | Details |
|---|---|
| **Discount Prediction** | Choose between Linear Regression, Ridge, Random Forest, or Gradient Boosting — predict `discount_percentage` |
| **Price Prediction** | Same four-model selector — predict `discounted_price` |
| **Per-model Parameter Tuning** | Each non-linear model exposes its key hyperparameters as sliders with plain-English explanations |
| **Feature Explorer** | Pearson correlation heatmap with worked example, feature selection path, and category target-encoding |
| **PCA Explorer** | Step-by-step principal component analysis ending with per-feature and per-category impact on discount % |
| **Drift Detection** | Per-feature z-score check flags out-of-distribution prediction inputs |
| **REST API** | FastAPI with `/predict_discount` and `/answer_question` (RAG) endpoints |
| **Docker** | Single `docker compose up --build` starts the API server |

---

## Architecture

```
amazon.csv
    │
    ▼
data_preprocessing.py        Clean raw strings (₹, %, commas) → numeric DataFrame
    │                         + normalize_features() for PCA
    │
    ├──▶ models.py            Train selected algorithm for discount + price tasks
    │         │                Save baseline models to artifacts/
    │         └──▶ check_drift()   Z-score check at inference time
    │
    └──▶ rag.py               Build text documents per product
              │                Encode with all-MiniLM-L6-v2 → FAISS IndexFlatL2
              │                Save index to artifacts/
              └──▶ generate_answer()   Retrieve top-k docs → flan-t5-base

artifacts/                   Pickled baseline models, FAISS index, training stats
    │
    ├──▶ api.py               FastAPI: /predict_discount, /answer_question
    └──▶ app.py               Streamlit:
                                · Overview
                                · Feature Explorer
                                · PCA Explorer
                                · Predict Discount
                                · Predict Price
                                · Model Insights
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- ~2 GB disk space (for HuggingFace model cache, only required if you use the RAG endpoint)

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Train baseline models and build the RAG index

```bash
python train.py
```

This writes six files to `artifacts/`:
- `random_forest_discount.pkl`
- `linear_regression_price.pkl`
- `training_stats.json`
- `feature_importance.png`
- `faiss_index.pkl`
- `documents.pkl`

Expected runtime: ~30 s on a modern laptop.

> The Streamlit app trains additional models on-the-fly when you switch algorithms or move the parameter sliders — results are cached so each combination only trains once per session.

### 3 — Launch the Streamlit demo

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser. On Streamlit Community Cloud the app bootstraps its own artifacts on first run.

### 4 — (Optional) Start the REST API

```bash
uvicorn src.api:app --reload
```

Interactive docs at http://localhost:8000/docs.

---

## Streamlit Pages

| Page | What it shows |
|---|---|
| **Overview** | Dataset summary, distributions, headline KPIs |
| **Feature Explorer** | Pearson correlation heatmap with an inline worked example; feature selection path showing how R² changes as features are added; category target-encoding panel |
| **PCA Explorer** | Standardised features → PCA components, variance explained, biplots, then Step 7 (definitive per-feature impact on discount %) and Step 8 (per-category breakdown) |
| **Predict Discount** | Pick a model, tune its parameters, enter a product's features, get a discount prediction with drift warning |
| **Predict Price** | Same selector for predicting discounted price |
| **Model Insights** | Metrics, coefficients/feature-importances, and a residual plot for the currently selected model |

---

## Project Structure

```
marketing-data-intelligence/
├── amazon.csv                 Raw dataset (1,465 products, 16 columns)
├── train.py                   Training pipeline entry point
├── app.py                     Streamlit demo app (6 pages)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── data_preprocessing.py  Loading, cleaning, feature extraction, normalize_features
│   ├── models.py              Model training, evaluation, persistence, drift detection
│   ├── rag.py                 RAG pipeline: corpus, FAISS index, retrieval, generation
│   └── api.py                 FastAPI application
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_api.py
└── artifacts/                 Generated by train.py (gitignored)
```

---

## API Reference

### `GET /health`

Liveness probe.

**Response**
```json
{ "status": "ok" }
```

---

### `POST /predict_discount`

Predict the discount percentage for a product. Backed by the baseline RandomForest saved by `train.py`.

**Request body**
```json
{
  "actual_price": 999.0,
  "discounted_price": 599.0,
  "rating": 4.2,
  "rating_count": 1200.0
}
```

**Response**
```json
{
  "predicted_discount_percentage": 41.83,
  "drift_warning": false,
  "drift_details": {
    "actual_price": 0.021,
    "discounted_price": 0.003,
    "rating": 0.667,
    "rating_count": 0.040
  },
  "inputs": { ... }
}
```

`drift_warning` is `true` when any input feature is more than 3 standard deviations from its training-set mean.

**Example with curl**
```bash
curl -X POST http://localhost:8000/predict_discount \
  -H "Content-Type: application/json" \
  -d '{"actual_price": 999, "discounted_price": 599, "rating": 4.2, "rating_count": 1200}'
```

---

### `POST /answer_question`

Answer a product-related question via RAG + LLM. The AI assistant lives at the API layer only — there is no chat page in the Streamlit UI.

**Request body**
```json
{
  "query": "What charging cables are compatible with iPhone?",
  "top_k": 3
}
```

**Response**
```json
{
  "question": "What charging cables are compatible with iPhone?",
  "answer": "The Wayona Nylon Braided USB to Lightning cable is compatible with iPhone 12 and other models.",
  "sources": [
    "Product: Wayona Nylon Braided USB to Lightning ...",
    "..."
  ]
}
```

`sources` contains the raw product documents that were passed to the LLM as context — useful for verifying grounding.

**Example with curl**
```bash
curl -X POST http://localhost:8000/answer_question \
  -H "Content-Type: application/json" \
  -d '{"query": "What earbuds have the best ratings?"}'
```

---

## Running with Docker

```bash
docker compose up --build
```

The `artifacts/` directory is volume-mounted, so you can run `python train.py` locally first and the container will use the pre-built models without downloading anything.

```yaml
# docker-compose.yml (excerpt)
volumes:
  - ./artifacts:/app/artifacts
environment:
  - LLM_MODEL=google/flan-t5-base
```

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `google/flan-t5-base` | Any HuggingFace `text2text-generation` model. Use `google/flan-t5-large` for better answer quality. |

---

## Running Tests

```bash
pytest tests/
```

Run a single file:
```bash
pytest tests/test_api.py -v
```

The test suite contains 18 tests across `test_preprocessing.py` (7), `test_models.py` (6), and `test_api.py` (5). Mocks are used for the sklearn models and the RAG pipeline, so no artifacts are required to run tests.

---

## Dataset

The [Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset) contains 1,465 product listings across 16 columns:

| Column | Type | Description |
|---|---|---|
| `product_id` | str | Amazon ASIN |
| `product_name` | str | Full product title |
| `category` | str | Pipe-separated category hierarchy |
| `discounted_price` | str → float | Selling price (e.g. "₹399") |
| `actual_price` | str → float | Original MRP (e.g. "₹1,099") |
| `discount_percentage` | str → float | Discount (e.g. "64%") |
| `rating` | str → float | Average star rating (1–5) |
| `rating_count` | str → float | Number of ratings |
| `about_product` | str | Bullet-point product description |
| `review_content` | str | Concatenated customer reviews |

---

## Model Details

### Available Algorithms

Both the discount and price tasks let you swap between four algorithms from the Streamlit UI:

| Model | Tunable Parameters | Notes |
|---|---|---|
| Linear Regression | — | Closed-form baseline, fully interpretable |
| Ridge Regression | `alpha` | Linear with L2 regularisation; more stable when features are correlated |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` | Handles non-linear patterns and interactions |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth` | Often the most accurate on tabular data |

Each slider has a tooltip plus an always-visible plain-English explanation describing what low vs. high values do.

### Baseline Metrics

These are the metrics for the models saved by `train.py` (RandomForest for discount, LinearRegression for price). Other algorithm/parameter combinations are trained on-the-fly in the UI and their metrics are shown live on the Model Insights page.

**RandomForest — Discount Prediction**

| Metric | Value |
|---|---|
| R² | 0.967 |
| RMSE | 3.78 percentage points |
| MAE | 2.13 percentage points |

Features used: `actual_price`, `discounted_price`, `rating`, `rating_count`.

**LinearRegression — Price Prediction**

| Metric | Value |
|---|---|
| R² | 0.951 |
| RMSE | ₹1,200 |
| MAE | ₹733 |

Features used: `actual_price`, `rating`, `rating_count`, `discount_percentage`.

### RAG Assistant (API only)

| Component | Choice | Reason |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | 22 M params, 384-dim, strong semantic similarity, CPU-friendly |
| Vector store | FAISS `IndexFlatL2` | Exact search, no tuning, fast for <10 k documents |
| LLM | `google/flan-t5-base` | Open-source, ~250 MB, instruction-following, no GPU needed |

---

## Tech Stack

- **ML**: scikit-learn, NumPy, pandas
- **NLP / RAG**: sentence-transformers, FAISS, Hugging Face Transformers
- **API**: FastAPI, Uvicorn, Pydantic
- **UI**: Streamlit, Matplotlib, Seaborn
- **Container**: Docker, Docker Compose
- **Testing**: pytest, httpx
