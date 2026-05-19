"""
data_preprocessing.py
=====================
Responsible for loading the raw Amazon Sales CSV and transforming it into
clean, numeric DataFrames that are ready for model training.

The raw dataset stores prices as Indian-Rupee strings (e.g. "₹1,299"),
ratings and discounts as strings with extra characters ("4.2", "64%"),
and some rows have missing values.  All of that is resolved here before
any modelling code ever touches the data.

Exported public API
-------------------
load_and_clean_data      -- Load CSV, clean all numeric columns, drop bad rows.
get_feature_target_for_discount -- Return (X, y) for the discount model.
get_feature_target_for_price    -- Return (X, y) for the price model.
save_training_stats      -- Persist per-feature mean/std for drift detection.
"""

import json
import os
import re

import numpy as np
import pandas as pd

# Resolved at import time so every function in this module can write to the
# same artifacts directory without needing it passed in explicitly.
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean_price(x) -> float:
    """Convert an Indian-Rupee price string to a plain float.

    Handles values like "₹1,299", "₹ 999", or already-clean floats.

    Args:
        x: Raw cell value from the CSV (str or numeric).

    Returns:
        Price as a Python float.
    """
    return float(str(x).replace("₹", "").replace(",", "").strip())


def _clean_numeric(x):
    """Strip every character that is not a digit or a decimal point.

    Used for columns such as ``rating`` ("4.2 out of 5"), ``rating_count``
    ("24,269"), and ``discount_percentage`` ("64%").

    Args:
        x: Raw cell value from the CSV.

    Returns:
        Float value, or ``None`` if no digits are found (so the row can be
        dropped later by ``dropna``).
    """
    s = re.sub(r"[^\d.]", "", str(x))
    return float(s) if s else None


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_clean_data(csv_path: str = "amazon.csv") -> pd.DataFrame:
    """Load the Amazon Sales CSV and return a fully numeric, clean DataFrame.

    Cleaning steps applied in order:
    1. Read the CSV (all columns land as ``str`` or ``object`` dtype).
    2. Convert price columns (``actual_price``, ``discounted_price``) by
       stripping the ₹ symbol and comma thousands-separators.
    3. Strip non-numeric characters from ``rating``, ``rating_count``, and
       ``discount_percentage``.
    4. Drop any row that is still missing a value in the five numeric columns
       we rely on for modelling.
    5. Reset the index so downstream code can rely on a clean 0-based range
       index.

    Args:
        csv_path: Path to the ``amazon.csv`` file.  Defaults to a file in
            the current working directory.

    Returns:
        A ``pd.DataFrame`` where the five numeric columns have ``float64``
        dtype and no ``NaN`` values.  All other columns (product_name,
        category, review text, etc.) are kept as-is for use by the RAG
        pipeline.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    df = pd.read_csv(csv_path)

    # Price columns carry a currency symbol and thousands commas.
    df["actual_price"] = df["actual_price"].apply(_clean_price)
    df["discounted_price"] = df["discounted_price"].apply(_clean_price)

    # These columns contain extra text or punctuation that must be stripped
    # before they can be cast to float.
    df["rating"] = df["rating"].apply(_clean_numeric)
    df["rating_count"] = df["rating_count"].apply(_clean_numeric)
    df["discount_percentage"] = df["discount_percentage"].apply(_clean_numeric)

    # Drop rows where any of the five critical numeric columns is missing.
    # The original dataset has 2 rows with a null rating_count; those are
    # removed here.
    df = df.dropna(
        subset=[
            "actual_price",
            "discounted_price",
            "rating",
            "rating_count",
            "discount_percentage",
        ]
    )

    return df.reset_index(drop=True)


def get_feature_target_for_discount(df: pd.DataFrame):
    """Extract the feature matrix and target vector for discount prediction.

    The RandomForest model learns to predict ``discount_percentage`` from the
    four numeric product attributes that are observable at listing time.

    Feature choices:
    - ``actual_price``      -- The original MRP; higher-priced items tend to
                               carry larger headline discounts.
    - ``discounted_price``  -- The selling price; together with actual_price
                               this encodes the absolute size of the discount.
    - ``rating``            -- Average customer rating (1–5 stars).
    - ``rating_count``      -- Number of ratings; a proxy for product maturity
                               and sales volume.

    Args:
        df: A cleaned DataFrame as returned by ``load_and_clean_data``.

    Returns:
        A tuple ``(X, y)`` where:
        - ``X`` is a ``pd.DataFrame`` of shape ``(n, 4)`` with the four
          feature columns.
        - ``y`` is a ``pd.Series`` of ``discount_percentage`` values.
    """
    features = ["actual_price", "discounted_price", "rating", "rating_count"]
    return df[features].copy(), df["discount_percentage"]


def get_feature_target_for_price(df: pd.DataFrame):
    """Extract the feature matrix and target vector for price prediction.

    The Linear Regression model predicts ``discounted_price`` from three
    product attributes plus the discount percentage.

    Feature choices:
    - ``actual_price``          -- Strong linear predictor; the discounted
                                   price scales closely with MRP.
    - ``rating``                -- Captures some quality premium or penalty.
    - ``rating_count``          -- Volume proxy; popular products may be
                                   priced more aggressively.
    - ``discount_percentage``   -- Direct driver of the selling price; included
                                   because the model is asked to reason about
                                   price *given* an intended discount level.

    Args:
        df: A cleaned DataFrame as returned by ``load_and_clean_data``.

    Returns:
        A tuple ``(X, y)`` where:
        - ``X`` is a ``pd.DataFrame`` of shape ``(n, 4)``.
        - ``y`` is a ``pd.Series`` of ``discounted_price`` values (₹).
    """
    features = ["actual_price", "rating", "rating_count", "discount_percentage"]
    return df[features].copy(), df["discounted_price"]


def save_training_stats(X: pd.DataFrame, path: str | None = None) -> None:
    """Persist the per-feature mean and standard deviation of the training set.

    These statistics are loaded at inference time by ``src.models.check_drift``
    to compute z-scores for incoming prediction requests.  If any feature
    value sits more than 3 standard deviations from its training mean, the API
    returns a ``drift_warning`` flag so callers know the model is operating
    outside its training distribution.

    The saved JSON has this shape::

        {
          "actual_price": {"mean": 1234.5, "std": 678.9},
          "discounted_price": {"mean": ...},
          ...
        }

    Args:
        X: The feature DataFrame used for training (before the train/test
            split, so statistics reflect the full distribution).
        path: Optional explicit file path.  Defaults to
            ``<repo_root>/artifacts/training_stats.json``.
    """
    if path is None:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        path = os.path.join(ARTIFACTS_DIR, "training_stats.json")

    stats = {
        col: {"mean": float(X[col].mean()), "std": float(X[col].std())}
        for col in X.columns
    }

    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Training stats saved to {path}")


# ── Category encoding helpers ─────────────────────────────────────────────────

def get_top_level_category(df: pd.DataFrame) -> pd.Series:
    """Extract the first segment of the pipe-separated category hierarchy.

    The raw ``category`` column stores a full path such as
    ``"Computers&Accessories|Accessories&Peripherals|Cables&Accessories|…"``.
    Only the first pipe-delimited segment is used here because it collapses
    the full hierarchy into 9 mutually exclusive top-level groups that are
    large enough to carry statistically meaningful signal.

    Args:
        df: Any DataFrame that contains a ``category`` column, as returned by
            ``load_and_clean_data``.

    Returns:
        A ``pd.Series`` of top-level category strings with the same index as
        *df*.  Rows with a missing ``category`` value are filled with
        ``"Unknown"``.
    """
    return df["category"].fillna("Unknown").str.split("|").str[0]


def encode_category(
    df: pd.DataFrame,
    method: str,
    target_col: str = "discount_percentage",
) -> tuple[pd.DataFrame, list[str]]:
    """Encode the top-level product category as one or more numeric columns.

    Four encoding strategies are supported.  Each converts the 9 categorical
    labels (Electronics, Computers&Accessories, Home&Kitchen, …) into numbers
    using a different philosophy, which produces different correlation signals
    and model behaviours.

    ``"label"``
        Assigns each unique category an integer in alphabetical order
        (Car&Motorbike = 0, Computers&Accessories = 1, Electronics = 2, …).
        Fast and memory-efficient, but imposes an arbitrary magnitude ordering.
        Tree models are immune to this artefact; linear models are not.

    ``"frequency"``
        Replaces each category with its proportion of all rows.  Electronics
        (526 products, ≈36 %) becomes 0.36; Toys&Games (1 product) becomes
        0.0007.  Captures 'market-size' signal without inventing an ordering.

    ``"target"``
        Replaces each category with the mean of *target_col* for all products
        in that category.  Produces the highest individual correlation with the
        target, but uses the target variable in the encoding — a data-leakage
        risk in production pipelines where the mapping should be fitted only on
        the training fold.

    ``"onehot"``
        Creates one binary dummy column per top-level category (9 columns
        total).  No false ordering, each category's signal learned
        independently.  Suitable for tree models; linear models should drop one
        dummy to avoid perfect multicollinearity.

    Args:
        df: Cleaned DataFrame from ``load_and_clean_data``.
        method: One of ``"label"``, ``"frequency"``, ``"target"``, ``"onehot"``.
        target_col: Column whose per-category mean is used when
            ``method="target"``.  Defaults to ``"discount_percentage"``; set to
            ``"discounted_price"`` for price-oriented analysis.

    Returns:
        A tuple ``(encoded_df, col_names)`` where:

        - ``encoded_df`` is a ``pd.DataFrame`` aligned to *df*'s index with one
          column for label / frequency / target methods and nine columns for
          one-hot encoding.
        - ``col_names`` is a list of the column names added.

    Raises:
        ValueError: If *method* is not one of the four supported strings.
    """
    cat = get_top_level_category(df)

    if method == "label":
        # Sort alphabetically so codes are deterministic across Python sessions.
        codes = pd.Categorical(cat, categories=sorted(cat.unique())).codes
        return (
            pd.DataFrame({"category_label": codes.astype(float)}, index=df.index),
            ["category_label"],
        )

    if method == "frequency":
        # normalize=True → proportions rather than raw counts, range 0–1.
        freq_map = cat.value_counts(normalize=True)
        return (
            pd.DataFrame({"category_freq": cat.map(freq_map)}, index=df.index),
            ["category_freq"],
        )

    if method == "target":
        mean_map = df.groupby(cat)[target_col].mean()
        return (
            pd.DataFrame({"category_target": cat.map(mean_map)}, index=df.index),
            ["category_target"],
        )

    if method == "onehot":
        # astype(float) keeps values consistent with other numeric features.
        dummies = pd.get_dummies(cat, prefix="cat").astype(float)
        return dummies, list(dummies.columns)

    raise ValueError(
        f"Unknown category encoding method: {method!r}. "
        "Choose one of 'label', 'frequency', 'target', 'onehot'."
    )
