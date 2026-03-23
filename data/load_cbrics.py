"""
CBRICS Multi-file Loader for Attribution X
Loads all weekly CSV files from data/cbrics/ and produces a single
clean trades DataFrame ready for the analytics engine.

Usage:
    python data/load_cbrics.py                    # loads all files
    python data/load_cbrics.py --year 2018        # single year
    python data/load_cbrics.py --from 2017-01-01 --to 2019-12-31
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

# ── Column mapping from CBRICS format → internal format ───────────────────
CBRICS_COLUMN_MAP = {
    "ISIN":                      "isin",
    "Issuer Name":               "issuer_name",
    "Issue Description":         "issue_description",
    "Coupon":                    "coupon",
    "Yield":                     "ytm",
    "Yield Type":                "yield_type",
    "Price":                     "price",
    "Trade Value in Rs. Lacs":   "volume",
    "Trade Date & Time":         "date",
    "Settlement Date":           "settlement_date",
    "Settlement Status":         "settlement_status",
    "Seller Deal Type":          "deal_type",
    "Remarks":                   "remarks",
    "Listed / Unlisted Security": "listed_status",
}

# ── Known stress issuers (for flagging) ───────────────────────────────────
STRESS_ISSUERS = {
    "DEWAN HOUSING FINANCE":     "DHFL",
    "IL&FS":                     "ILFS",
    "INFRASTRUCTURE LEASING":    "ILFS",
    "YES BANK":                  "YES_BANK",
    "RELIANCE CAPITAL":          "RELIANCE_CAP",
    "RELIANCE HOME FINANCE":     "RELIANCE_CAP",
    "VODAFONE":                  "VODAFONE",
    "IDEA CELLULAR":             "VODAFONE",
    "FUTURE RETAIL":             "FUTURE",
    "FUTURE ENTERPRISES":        "FUTURE",
    "SREI INFRASTRUCTURE":       "SREI",
}


def load_single_file(filepath: Path) -> pd.DataFrame | None:
    """Load and minimally clean one CBRICS weekly CSV.
    Strategy: try fast C engine first, fall back to python engine for bad-line files.
    """
    # Attempt 1: fast C engine
    try:
        df = pd.read_csv(filepath, low_memory=False)
        if df.empty:
            return None
        df.columns = [c.strip() for c in df.columns]
        required = {"ISIN", "Yield", "Trade Date & Time", "Trade Value in Rs. Lacs"}
        if not required.issubset(set(df.columns)):
            return None
        return df.rename(columns=CBRICS_COLUMN_MAP)[
            [v for v in CBRICS_COLUMN_MAP.values() if v in df.rename(columns=CBRICS_COLUMN_MAP).columns]
        ].copy()
    except Exception:
        pass

    # Attempt 2: python engine with bad-line skipping (for malformed files)
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
        if df.empty:
            return None
        df.columns = [c.strip() for c in df.columns]
        required = {"ISIN", "Yield", "Trade Date & Time", "Trade Value in Rs. Lacs"}
        if not required.issubset(set(df.columns)):
            print(f"  [skip] {filepath.name} — missing columns")
            return None
        renamed = df.rename(columns=CBRICS_COLUMN_MAP)
        keep = [v for v in CBRICS_COLUMN_MAP.values() if v in renamed.columns]
        return renamed[keep].copy()
    except Exception as e:
        print(f"  [skip] {filepath.name} — failed both engines: {e}")
        return None


def clean_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning and filtering rules to combined DataFrame."""

    original_len = len(df)

    # ── 1. Parse dates ─────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(
        df["date"], dayfirst=True, errors="coerce"
    ).dt.normalize()  # strip time, keep date only
    df = df.dropna(subset=["date"])

    # ── 2. Clean ISIN ──────────────────────────────────────────────────────
    df = df.copy()
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    df = df[df["isin"].str.match(r"^[A-Z]{2}[A-Z0-9]{10}$")]

    # ── 3. YTM only — drop YTC and YTP ────────────────────────────────────
    if "yield_type" in df.columns:
        df = df[df["yield_type"] == "YTM"]

    # ── 4. Settled trades only ─────────────────────────────────────────────
    if "settlement_status" in df.columns:
        df = df[df["settlement_status"] == "Settled"]

    # ── 5. Clean YTM values ────────────────────────────────────────────────
    df["ytm"] = pd.to_numeric(df["ytm"], errors="coerce")
    df = df.dropna(subset=["ytm"])
    df = df[(df["ytm"] > 0.5) & (df["ytm"] < 30.0)]  # realistic bond yield range

    # ── 6. Clean volume ────────────────────────────────────────────────────
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df.loc[df["volume"] <= 0, "volume"] = np.nan

    # ── 7. Clean issuer name ───────────────────────────────────────────────
    if "issuer_name" in df.columns:
        df["issuer_name"] = df["issuer_name"].astype(str).str.strip().str.upper()
    else:
        df["issuer_name"] = "UNKNOWN"

    # ── 8. Tag stress issuers ──────────────────────────────────────────────
    def tag_issuer(name: str) -> str | None:
        for keyword, tag in STRESS_ISSUERS.items():
            if keyword in name:
                return tag
        return None

    df["stress_tag"] = df["issuer_name"].apply(tag_issuer)

    # ── 9. Flag defaulted ISINs from Remarks ──────────────────────────────
    if "remarks" in df.columns:
        df["is_defaulted_isin"] = df["remarks"].astype(str).str.contains(
            "Defaulted", case=False, na=False
        )
    else:
        df["is_defaulted_isin"] = False

    # ── 10. Final sort and dedup ───────────────────────────────────────────
    df = df.sort_values(["date", "isin"]).reset_index(drop=True)

    removed = original_len - len(df)
    print(f"  [clean] {original_len:>7,} → {len(df):>7,} rows "
          f"({removed:,} removed by filters)")

    return df


def load_all_cbrics(
    data_dir: Path,
    date_from: str | None = None,
    date_to:   str | None = None,
    year:      int | None = None,
) -> pd.DataFrame:
    """
    Load all weekly CBRICS CSV files from data_dir.
    Returns a single clean DataFrame.
    """
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            f"Please copy your weekly CBRICS files there."
        )

    print(f"\n{'='*60}")
    print(f"Attribution X — CBRICS Data Loader")
    print(f"{'='*60}")
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    print(f"Loading all files...\n")

    frames = []
    for i, f in enumerate(csv_files, 1):
        if i % 50 == 0 or i == len(csv_files):
            print(f"  Progress: {i}/{len(csv_files)} files")
        df = load_single_file(f)
        if df is not None:
            frames.append(df)

    if not frames:
        raise ValueError("No valid data loaded from any file.")

    print(f"\nCombining {len(frames)} files...")
    combined = pd.concat(frames, ignore_index=True)

    print("Cleaning combined dataset...")
    combined = clean_trades(combined)

    # ── Date filtering ─────────────────────────────────────────────────────
    if year:
        combined = combined[combined["date"].dt.year == year]
        print(f"Filtered to year {year}: {len(combined):,} rows")

    if date_from:
        combined = combined[combined["date"] >= pd.Timestamp(date_from)]
    if date_to:
        combined = combined[combined["date"] <= pd.Timestamp(date_to)]
    if date_from or date_to:
        print(f"Filtered to {date_from} – {date_to}: {len(combined):,} rows")

    return combined


def print_summary(df: pd.DataFrame):
    """Print a rich summary of the loaded dataset."""
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total trades:     {len(df):>10,}")
    print(f"Unique ISINs:     {df['isin'].nunique():>10,}")
    print(f"Unique issuers:   {df['issuer_name'].nunique():>10,}")
    print(f"Date range:       {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"YTM range:        {df['ytm'].min():.2f}% – {df['ytm'].max():.2f}%")
    print(f"Avg daily volume: ₹{df.groupby('date')['volume'].sum().mean():>10,.0f} Lacs")

    print(f"\nSTRESS ISSUER TRADES:")
    stress = df[df["stress_tag"].notna()]
    if len(stress):
        summary = stress.groupby("stress_tag").agg(
            trades=("ytm", "count"),
            isins=("isin", "nunique"),
            ytm_max=("ytm", "max"),
            ytm_mean=("ytm", "mean"),
            date_first=("date", "min"),
            date_last=("date", "max"),
        ).round(2)
        print(summary.to_string())
    else:
        print("  No stress issuers found in this date range.")

    print(f"\nDefaulted ISIN trades: {df['is_defaulted_isin'].sum():,}")
    print(f"{'='*60}\n")


def save_processed(df: pd.DataFrame, out_path: Path):
    """Save the processed dataset."""
    df.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}  ({len(df):,} rows, "
          f"{out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load CBRICS weekly CSV files")
    parser.add_argument("--year",  type=int,   help="Filter to a single year")
    parser.add_argument("--from",  dest="date_from", help="Start date YYYY-MM-DD")
    parser.add_argument("--to",    dest="date_to",   help="End date YYYY-MM-DD")
    parser.add_argument("--out",   default="data/processed_trades.csv",
                        help="Output file path")
    args = parser.parse_args()

    data_dir = Path(__file__).parent / "cbrics"

    df = load_all_cbrics(
        data_dir=data_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        year=args.year,
    )

    print_summary(df)

    out_path = Path(args.out)
    save_processed(df, out_path)