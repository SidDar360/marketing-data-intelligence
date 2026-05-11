import os
import json
import pytest
import pandas as pd

from src.data_preprocessing import (
    load_and_clean_data,
    get_feature_target_for_discount,
    get_feature_target_for_price,
    save_training_stats,
)


@pytest.fixture
def sample_csv(tmp_path):
    rows = {
        "product_id": ["A1", "A2", "A3"],
        "product_name": ["Prod 1", "Prod 2", "Prod 3"],
        "category": ["Electronics", "Computers", "Electronics"],
        "discounted_price": ["₹399", "₹199", "₹599"],
        "actual_price": ["₹1,099", "₹349", "₹1,299"],
        "discount_percentage": ["64%", "43%", "54%"],
        "rating": ["4.2", "4.0", "3.9"],
        "rating_count": ["24,269", "43,994", "7,928"],
        "about_product": ["About 1", "About 2", "About 3"],
        "user_id": ["U1", "U2", "U3"],
        "user_name": ["User1", "User2", "User3"],
        "review_id": ["R1", "R2", "R3"],
        "review_title": ["Good", "Great", "OK"],
        "review_content": ["Content 1", "Content 2", "Content 3"],
        "img_link": ["http://img1", "http://img2", "http://img3"],
        "product_link": ["http://p1", "http://p2", "http://p3"],
    }
    path = tmp_path / "test.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_load_converts_prices(sample_csv):
    df = load_and_clean_data(sample_csv)
    assert df["discounted_price"].dtype == float
    assert df["actual_price"].dtype == float
    assert df["discounted_price"].iloc[0] == 399.0
    assert df["actual_price"].iloc[0] == 1099.0


def test_load_converts_percentage(sample_csv):
    df = load_and_clean_data(sample_csv)
    assert df["discount_percentage"].dtype == float
    assert df["discount_percentage"].iloc[0] == 64.0


def test_load_converts_rating_count(sample_csv):
    df = load_and_clean_data(sample_csv)
    assert df["rating_count"].iloc[0] == 24269.0


def test_load_drops_nulls(tmp_path):
    rows = {
        "product_id": ["A1", "A2"],
        "product_name": ["P1", "P2"],
        "category": ["C", "C"],
        "discounted_price": ["₹100", "₹200"],
        "actual_price": ["₹200", "₹400"],
        "discount_percentage": ["50%", None],
        "rating": ["4.0", "3.5"],
        "rating_count": ["100", "200"],
        "about_product": ["x", "y"],
        "user_id": ["u", "v"],
        "user_name": ["n", "m"],
        "review_id": ["r", "s"],
        "review_title": ["t", "t"],
        "review_content": ["c", "c"],
        "img_link": ["i", "i"],
        "product_link": ["l", "l"],
    }
    path = tmp_path / "nulls.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    df = load_and_clean_data(str(path))
    assert len(df) == 1


def test_feature_target_discount(sample_csv):
    df = load_and_clean_data(sample_csv)
    X, y = get_feature_target_for_discount(df)
    assert list(X.columns) == ["actual_price", "discounted_price", "rating", "rating_count"]
    assert len(X) == len(y) == 3


def test_feature_target_price(sample_csv):
    df = load_and_clean_data(sample_csv)
    X, y = get_feature_target_for_price(df)
    assert list(X.columns) == ["actual_price", "rating", "rating_count", "discount_percentage"]
    assert len(X) == len(y) == 3


def test_save_training_stats(sample_csv, tmp_path):
    df = load_and_clean_data(sample_csv)
    X, _ = get_feature_target_for_discount(df)
    stats_path = str(tmp_path / "stats.json")
    save_training_stats(X, path=stats_path)
    with open(stats_path) as f:
        stats = json.load(f)
    assert "actual_price" in stats
    assert "mean" in stats["actual_price"]
    assert "std" in stats["actual_price"]
