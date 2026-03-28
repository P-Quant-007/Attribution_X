import pandas as pd
import numpy as np
from pathlib import Path


REQUIRED_COLUMNS = {"date", "isin", "ytm"}
OPTIONAL_COLUMNS = {"volume"}

# Applied AFTER stripping and lowercasing — all keys must be lowercase+stripped
COLUMN_ALIASES = {
    # Date variants
    "date": "date",
    "trade date": "date",
    "trade date & time": "date",
    "trade date&time": "date",
    "tradedatetime": "date",
    # ISIN variants
    "isin": "isin",
    "bond": "isin",
    # YTM / Yield variants
    "ytm": "ytm",
    "yield": "ytm",
    "yld": "ytm",
    # Volume variants
    "volume": "volume",
    "vol": "volume",
    "notional": "volume",
    "trade value in rs. lacs": "volume",
    "trade value in rs lacs": "volume",
    "trade value (rs. lacs)": "volume",
    # Yield type (kept for filtering)
    "yield type": "yield_type",
    # Settlement status (kept for filtering)
    "settlement status": "settlement_status",
    # Issuer name
    "issuer name": "issuer_name",
    "issuername": "issuer_name",
}


def load_trades(filepath: str | Path) -> pd.DataFrame:
    """
    Load bond trade data from CSV or XLSX.
    Handles both old CBRICS format and new CBRICS format with leading spaces.
    Returns a clean DataFrame with columns: date, isin, ytm, volume.
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

    # Step 1: Strip leading/trailing whitespace from column names first
    df.columns = [c.strip() for c in df.columns]

    # Step 2: Lowercase all column names
    df.columns = [c.lower() for c in df.columns]

    # Step 3: Apply aliases (now safely matching on clean lowercase names)
    df = df.rename(columns=COLUMN_ALIASES)

    # Validate required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"File has: {list(df.columns)}"
        )

    # Filter by yield type if column present — keep YTM only
    if "yield_type" in df.columns:
        df = df[df["yield_type"].astype(str).str.upper().str.contains("YTM", na=False)]

    # Filter by settlement status if column present — keep Settled only
    if "settlement_status" in df.columns:
        df = df[df["settlement_status"].astype(str).str.upper().str.contains("SETTLED", na=False)]

    # Add volume column with NaN if not present
    if "volume" not in df.columns:
        df["volume"] = np.nan

    # Keep only relevant columns
    keep = ["date", "isin", "ytm", "volume"]
    if "issuer_name" in df.columns:
        keep.append("issuer_name")
    df = df[[c for c in keep if c in df.columns]].copy()

    # Parse and clean date
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    dropped_dates = df["date"].isna().sum()
    if dropped_dates > 0:
        print(f"[loader] Warning: dropped {dropped_dates} rows with unparseable dates.")
    df = df.dropna(subset=["date"])

    # Clean ISIN
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    df = df[df["isin"].str.match(r"^[A-Z]{2}[A-Z0-9]{10}$")]

    # Clean YTM — must be numeric, in range 0.5–30%
    df["ytm"] = pd.to_numeric(df["ytm"], errors="coerce")
    df = df.dropna(subset=["ytm"])
    df = df[(df["ytm"] > 0.5) & (df["ytm"] < 30.0)]

    # Clean volume — numeric, positive
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df.loc[df["volume"] <= 0, "volume"] = np.nan

    # Sort and reset index
    df = df.sort_values(["date", "isin"]).reset_index(drop=True)

    print(f"[loader] Loaded {len(df):,} trades | "
          f"{df['isin'].nunique():,} ISINs | "
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
    Fast path — data is already clean.
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
