import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.portfolio import load_portfolio, portfolio_summary


def make_csv(tmp_path, content):
    f = tmp_path / "portfolio.csv"
    f.write_text(content)
    return f


VALID_CSV = """isin,issuer_name,coupon,maturity_date,face_value,rating
INE202B07IK1,DHFL,9.10,2022-03-15,5000,AA
INE134E08KL2,HDFC,8.45,2025-06-15,10000,AAA
INE115A07FH6,LIC HFL,8.30,2024-01-20,8000,AAA
"""


def test_basic_load(tmp_path):
    f = make_csv(tmp_path, VALID_CSV)
    df = load_portfolio(f)
    assert len(df) == 3
    assert "weight" in df.columns
    assert "years_to_maturity" in df.columns


def test_weights_sum_to_100(tmp_path):
    f = make_csv(tmp_path, VALID_CSV)
    df = load_portfolio(f)
    assert abs(df["weight"].sum() - 100.0) < 0.01


def test_stress_tag_detected(tmp_path):
    f = make_csv(tmp_path, VALID_CSV)
    df = load_portfolio(f)
    dhfl_rows = df[df["isin"] == "INE202B07IK1"]
    assert dhfl_rows.iloc[0]["stress_tag"] == "DHFL"
    assert dhfl_rows.iloc[0]["is_stress_issuer"] == True


def test_missing_required_column_raises(tmp_path):
    csv = """isin,coupon,face_value
INE202B07IK1,9.10,5000
"""
    f = make_csv(tmp_path, csv)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_portfolio(f)


def test_invalid_isin_raises(tmp_path):
    csv = """isin,coupon,maturity_date,face_value
INVALID123,9.10,2025-01-01,5000
"""
    f = make_csv(tmp_path, csv)
    with pytest.raises(ValueError, match="Invalid ISINs"):
        load_portfolio(f)


def test_portfolio_summary_structure(tmp_path):
    f = make_csv(tmp_path, VALID_CSV)
    df = load_portfolio(f)
    summary = portfolio_summary(df)
    assert "total_holdings" in summary
    assert "total_aum_lacs" in summary
    assert "stress_holdings" in summary
    assert "holdings" in summary
    assert len(summary["holdings"]) == 3


def test_sorted_by_face_value_descending(tmp_path):
    f = make_csv(tmp_path, VALID_CSV)
    df = load_portfolio(f)
    assert df.iloc[0]["face_value"] >= df.iloc[1]["face_value"]