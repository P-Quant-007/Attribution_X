import pandas as pd
import numpy as np
from pathlib import Path


REQUIRED_COLUMNS = {"date", "isin", "ytm"}
OPTIONAL_COLUMNS = {"volume"}

COLUMN_ALIASES = {
    "Date": "date",
    "DATE": "date",
    "Trade Date": "date",
    "ISIN": "isin",
    "Isin": "isin",
    "Bond": "isin",
    "YTM": "ytm",
    "Yield": "ytm",
    "yield": "ytm",
    "Volume": "volume",
    "Vol": "volume",
    "vol": "volume",
    "Notional": "volume",
}


def load_trades(filepath: str | Path) -> pd.DataFrame:
    """
    Load bond trade data from CSV or XLSX.
    Returns a clean DataFrame with columns: date, isin, ytm, volume (optional).
    Raises ValueError if required columns are missing.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(filepath)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use CSV or XLSX.")

    # Normalise column names using aliases
    df = df.rename(columns=COLUMN_ALIASES)
    df.columns = [c.strip().lower() for c in df.columns]

    # Validate required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"File has: {list(df.columns)}"
        )

    # Add volume column with NaN if not present
    if "volume" not in df.columns:
        df["volume"] = np.nan

    # Keep only relevant columns
    df = df[["date", "isin", "ytm", "volume"]].copy()

    # Parse and clean date
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    dropped_dates = df["date"].isna().sum()
    if dropped_dates > 0:
        print(f"[loader] Warning: dropped {dropped_dates} rows with unparseable dates.")
    df = df.dropna(subset=["date"])

    # Clean ISIN
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    df = df[df["isin"].str.len() == 12]  # valid ISINs are 12 chars

    # Clean YTM — must be numeric, positive, reasonable (0 < ytm < 100)
    df["ytm"] = pd.to_numeric(df["ytm"], errors="coerce")
    df = df.dropna(subset=["ytm"])
    df = df[(df["ytm"] > 0) & (df["ytm"] < 100)]

    # Clean volume — numeric, positive if present
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df.loc[df["volume"] <= 0, "volume"] = np.nan

    # Sort and reset index
    df = df.sort_values(["date", "isin"]).reset_index(drop=True)

    print(f"[loader] Loaded {len(df)} trades | "
          f"{df['isin'].nunique()} ISINs | "
          f"{df['date'].min().date()} to {df['date'].max().date()}")

    return df


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Quick sanity check on a loaded DataFrame."""
    assert "date" in df.columns
    assert "isin" in df.columns
    assert "ytm" in df.columns
    assert "volume" in df.columns
    assert df["ytm"].between(0, 100).all()
    assert df["date"].notna().all()
    return True

def load_processed_trades(filepath: str | Path) -> pd.DataFrame:
    """
    Load the pre-processed CBRICS dataset (output of data/load_cbrics.py).
    This is the fast path — data is already clean, just parse dates and types.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Processed file not found: {filepath}\n"
                                f"Run: python data/load_cbrics.py first.")

    df = pd.read_csv(filepath, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df["ytm"] = pd.to_numeric(df["ytm"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.sort_values(["date", "isin"]).reset_index(drop=True)

    print(f"[loader] Loaded processed dataset: {len(df):,} trades | "
          f"{df['isin'].nunique():,} ISINs | "
          f"{df['date'].min().date()} to {df['date'].max().date()}")
    return df