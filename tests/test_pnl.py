import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.pnl import (
    compute_bond_price, compute_modified_duration,
    compute_dv01, enrich_portfolio_with_duration
)


def test_par_bond_price():
    """A bond trading at par (coupon == ytm) should price near 100."""
    price = compute_bond_price(
        face_value=100, coupon_rate=8.0,
        ytm=8.0, years_to_maturity=5.0
    )
    assert abs(price - 100.0) < 0.5


def test_premium_bond():
    """Coupon > YTM → price above par."""
    price = compute_bond_price(
        face_value=100, coupon_rate=10.0,
        ytm=8.0, years_to_maturity=5.0
    )
    assert price > 100.0


def test_discount_bond():
    """Coupon < YTM → price below par."""
    price = compute_bond_price(
        face_value=100, coupon_rate=7.0,
        ytm=9.0, years_to_maturity=5.0
    )
    assert price < 100.0


def test_modified_duration_positive():
    """Duration should always be positive for standard bonds."""
    dur = compute_modified_duration(
        face_value=100, coupon_rate=8.0,
        ytm=8.5, years_to_maturity=5.0
    )
    assert dur > 0


def test_longer_maturity_higher_duration():
    """Longer maturity → higher duration."""
    dur_5y = compute_modified_duration(100, 8.0, 8.5, 5.0)
    dur_10y = compute_modified_duration(100, 8.0, 8.5, 10.0)
    assert dur_10y > dur_5y


def test_dv01_scales_with_face_value():
    """Doubling face value should double DV01."""
    dv01_1 = compute_dv01(1000, 8.0, 8.5, 5.0)
    dv01_2 = compute_dv01(2000, 8.0, 8.5, 5.0)
    assert abs(dv01_2 - 2 * dv01_1) < 0.001


def test_matured_bond_returns_par():
    """A matured bond should return face value."""
    price = compute_bond_price(100, 8.0, 9.0, 0.0)
    assert price == 100.0


def test_matured_bond_zero_duration():
    """A matured bond has zero duration."""
    dur = compute_modified_duration(100, 8.0, 9.0, 0.0)
    assert dur == 0.0


def test_enrich_portfolio():
    """enrich_portfolio_with_duration adds dv01 and modified_duration."""
    import pandas as pd
    portfolio = pd.DataFrame([{
        "isin": "INE134E08KL2",
        "issuer_name": "HDFC",
        "coupon": 8.45,
        "maturity_date": pd.Timestamp("2031-06-15"),
        "face_value": 10000,
        "weight": 100.0,
        "rating": "AAA",
        "years_to_maturity": 5.5,
        "stress_tag": None,
        "is_stress_issuer": False,
    }])
    metrics = pd.DataFrame([{
        "isin": "INE134E08KL2",
        "date": pd.Timestamp("2019-01-15"),
        "avg_ytm": 8.50,
    }])
    enriched = enrich_portfolio_with_duration(portfolio, metrics)
    assert "dv01" in enriched.columns
    assert "modified_duration" in enriched.columns
    assert enriched.iloc[0]["dv01"] > 0