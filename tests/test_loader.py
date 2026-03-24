import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.loader import load_trades, validate_dataframe


def make_csv(tmp_path, content):
    f = tmp_path / "trades.csv"
    f.write_text(content)
    return f


def test_basic_load(tmp_path):
    csv = make_csv(tmp_path, """date,isin,ytm,volume
2019-01-01,INE202E01016,9.5,10000000
2019-01-02,INE202E01016,9.8,5000000
""")
    df = load_trades(csv)
    assert len(df) == 2
    assert list(df.columns) == ["date", "isin", "ytm", "volume"]


def test_missing_volume_fills_nan(tmp_path):
    csv = make_csv(tmp_path, """date,isin,ytm
2019-01-01,INE202E01016,9.5
""")
    df = load_trades(csv)
    assert "volume" in df.columns
    assert pd.isna(df["volume"].iloc[0])


def test_invalid_ytm_dropped(tmp_path):
    csv = make_csv(tmp_path, """date,isin,ytm,volume
2019-01-01,INE202E01016,9.5,1000
2019-01-02,INE202E01016,-1.0,1000
2019-01-03,INE202E01016,150.0,1000
""")
    df = load_trades(csv)
    assert len(df) == 1


def test_missing_required_column_raises(tmp_path):
    csv = make_csv(tmp_path, """date,volume
2019-01-01,10000
""")
    with pytest.raises(ValueError, match="Missing required columns"):
        load_trades(csv)


def test_column_aliases(tmp_path):
    csv = make_csv(tmp_path, """Date,ISIN,YTM,Volume
2019-01-01,INE202E01016,9.5,10000
""")
    df = load_trades(csv)
    assert "ytm" in df.columns


def test_validate_passes(tmp_path):
    csv = make_csv(tmp_path, """date,isin,ytm,volume
2019-01-01,INE202E01016,9.5,10000
""")
    df = load_trades(csv)
    assert validate_dataframe(df)


from engine.features import compute_features, get_feature_matrix
from engine.aggregator import compute_daily_ytm


def make_multi_day_df():
    """10 days of trades for one ISIN."""
    dates = pd.date_range("2018-09-01", periods=10, freq="B")
    rows = []
    for i, d in enumerate(dates):
        for _ in range(3):
            rows.append({
                "date": d,
                "isin": "INE202E01016",
                "ytm": 9.0 + i * 0.1,
                "volume": 1_000_000.0,
            })
    return pd.DataFrame(rows)


def test_features_computed():
    raw = make_multi_day_df()
    daily = compute_daily_ytm(raw)
    feat = compute_features(daily)
    assert "d1" in feat.columns
    assert "d5" in feat.columns
    assert "spread_bps" in feat.columns
    assert "z_score_21d" in feat.columns
    assert "vol_log" in feat.columns


def test_d1_is_correct():
    raw = make_multi_day_df()
    daily = compute_daily_ytm(raw)
    feat = compute_features(daily)
    # d1 for row 1 should be ~10 bps (0.1% * 100)
    d1_vals = feat["d1"].dropna()
    assert abs(d1_vals.iloc[0] - 10.0) < 0.5


def test_spread_bps_reasonable():
    raw = make_multi_day_df()
    daily = compute_daily_ytm(raw)
    feat = compute_features(daily)
    # 2018 benchmark is 7.76, YTM ~9%, spread should be ~125-200bps
    spread = feat["spread_bps"].dropna()
    assert (spread > 50).all()
    assert (spread < 500).all()


def test_feature_matrix_no_nans():
    raw = make_multi_day_df()
    daily = compute_daily_ytm(raw)
    feat = compute_features(daily)
    X, _ = get_feature_matrix(feat)
    assert X.isna().sum().sum() == 0

from engine.detector import train_model, predict_anomalies, run_detection_pipeline


def make_feature_df(n_days=60):
    """Generate a feature DataFrame with one obvious anomaly injected."""
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    rows = []
    for i, d in enumerate(dates):
        for _ in range(3):
            rows.append({
                "date": d, "isin": "INE202E01016",
                "ytm": 9.0 + (50.0 if i == 45 else 0),  # spike on day 45
                "volume": 1_000_000.0,
            })
    raw = pd.DataFrame(rows)
    from engine.aggregator import compute_daily_ytm
    from engine.features import compute_features
    daily = compute_daily_ytm(raw)
    return compute_features(daily)


def test_model_trains_without_error():
    feat = make_feature_df()
    result = run_detection_pipeline(feat, contamination=0.05, save=False)
    assert "is_anomaly" in result.columns
    assert "anomaly_score_norm" in result.columns


def test_injected_spike_is_flagged():
    feat = make_feature_df(n_days=60)
    result = run_detection_pipeline(feat, contamination=0.05, save=False)
    # The day with YTM spike (day 45) should be flagged
    spike_rows = result[result["avg_ytm"] > 50]
    assert len(spike_rows) > 0
    assert spike_rows["is_anomaly"].iloc[0] == 1


def test_anomaly_score_norm_bounded():
    feat = make_feature_df()
    result = run_detection_pipeline(feat, contamination=0.05, save=False)
    assert result["anomaly_score_norm"].between(0, 1).all()