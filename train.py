#!/usr/bin/env python3
"""
train.py
========
Entry point for the offline training pipeline.

Run this script once before starting the API server or the Streamlit app.
It performs four sequential steps and writes all output to the ``artifacts/``
directory:

    Step 1  Load & clean data
            Reads ``amazon.csv``, strips currency symbols and noise
            characters, drops rows with missing numeric values.

    Step 2  Train RandomForestRegressor
            Target: ``discount_percentage``.
            Features: actual_price, discounted_price, rating, rating_count.
            Saves model to ``artifacts/random_forest_discount.pkl``.
            Saves feature importance chart to ``artifacts/feature_importance.png``.
            Saves per-feature mean/std to ``artifacts/training_stats.json``
            for use by the drift detection logic at inference time.

    Step 3  Train LinearRegression
            Target: ``discounted_price`` (₹).
            Features: actual_price, rating, rating_count, discount_percentage.
            Saves model to ``artifacts/linear_regression_price.pkl``.

    Step 4  Build RAG index
            Converts each product row into a rich text document, encodes the
            documents with ``all-MiniLM-L6-v2``, and stores the FAISS index
            to ``artifacts/faiss_index.pkl`` alongside the raw document
            strings in ``artifacts/documents.pkl``.

Usage
-----
From the repository root::

    python train.py

Expected runtime: ~30 s on a modern MacBook (mostly the FAISS encoding step).
The LLM (flan-t5-base) is NOT downloaded here; it is fetched on the first
call to ``/answer_question`` and cached by HuggingFace locally.

Dependencies
------------
All required packages are listed in ``requirements.txt``.  Install with::

    pip install -r requirements.txt
"""

import os
import sys

# Make the ``src`` package importable when the script is run from the repo
# root as ``python train.py`` (as opposed to ``python -m train``).
sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import (
    get_feature_target_for_discount,
    get_feature_target_for_price,
    load_and_clean_data,
    save_training_stats,
)
from src.models import (
    plot_feature_importance,
    save_model,
    train_linear_regression,
    train_random_forest,
)
from src.rag import build_document_corpus, build_faiss_index

CSV_PATH = os.path.join(os.path.dirname(__file__), "amazon.csv")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def main() -> None:
    """Execute the full training pipeline end-to-end."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── Step 1: Data loading & cleaning ──────────────────────────────────────
    print("=" * 50)
    print("Step 1: Loading and cleaning data")
    print("=" * 50)
    df = load_and_clean_data(CSV_PATH)
    print(f"  {len(df)} rows after cleaning.\n")

    # ── Step 2: RandomForest — discount prediction ────────────────────────────
    print("=" * 50)
    print("Step 2: Train RandomForest — discount prediction")
    print("=" * 50)
    X_disc, y_disc = get_feature_target_for_discount(df)
    rf_model, rf_metrics = train_random_forest(X_disc, y_disc)
    print(f"  RMSE : {rf_metrics['rmse']:.4f}")
    print(f"  MAE  : {rf_metrics['mae']:.4f}")
    print(f"  R²   : {rf_metrics['r2']:.4f}")

    save_model(rf_model, "random_forest_discount")
    plot_feature_importance(
        rf_model,
        list(X_disc.columns),
        output_path=os.path.join(ARTIFACTS_DIR, "feature_importance.png"),
    )
    # Save training distribution statistics for drift detection at inference time.
    save_training_stats(X_disc)
    print()

    # ── Step 3: LinearRegression — price prediction ───────────────────────────
    print("=" * 50)
    print("Step 3: Train LinearRegression — price prediction")
    print("=" * 50)
    X_price, y_price = get_feature_target_for_price(df)
    lr_model, lr_metrics = train_linear_regression(X_price, y_price)
    print(f"  RMSE : {lr_metrics['rmse']:.4f}")
    print(f"  MAE  : {lr_metrics['mae']:.4f}")
    print(f"  R²   : {lr_metrics['r2']:.4f}")
    save_model(lr_model, "linear_regression_price")
    print()

    # ── Step 4: RAG index ─────────────────────────────────────────────────────
    print("=" * 50)
    print("Step 4: Building RAG index")
    print("=" * 50)
    documents = build_document_corpus(df)
    build_faiss_index(documents)
    print()

    # ── Done ──────────────────────────────────────────────────────────────────
    print("=" * 50)
    print("All done. Artifacts saved to ./artifacts/")
    print("Start API:  uvicorn src.api:app --reload")
    print("Start app:  streamlit run app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
