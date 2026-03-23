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