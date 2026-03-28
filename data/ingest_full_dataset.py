"""
One-time ingestion of combined_all_records.csv into Neon.
Runs locally — no file size limit, no Render disk needed.
"""
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'backend')

import pandas as pd
from engine.aggregator import compute_daily_ytm
from engine.features   import compute_features
from engine.detector   import run_detection_pipeline
from backend.database  import get_engine, upsert_daily_metrics, upsert_anomalies

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE    = "data/combined_all_records.csv"
CONTAMINATION = 0.03
CHUNK_SIZE    = 100_000   # process in chunks to avoid memory issues

# ── CBRICS column map (same as load_cbrics.py) ────────────────────────────────
COL_MAP = {
    "ISIN":                    "isin",
    "Issuer Name":             "issuer_name",
    "Yield":                   "ytm",
    "Trade Value in Rs. Lacs": "volume",
    "Trade Date & Time":       "date",
    "Yield Type":              "yield_type",
    "Settlement Status":       "settlement_status",
    "Remarks":                 "remarks",
    "Coupon":                  "coupon",
    "Price":                   "price",
}

STRESS_ISSUERS = {
    "DEWAN HOUSING": "DHFL", "DHFL": "DHFL",
    "IL&FS": "ILFS", "INFRASTRUCTURE LEASING": "ILFS",
    "YES BANK": "YES_BANK",
    "RELIANCE CAPITAL": "RELIANCE_CAP", "RELIANCE HOME": "RELIANCE_CAP",
    "VODAFONE": "VODAFONE", "IDEA CELLULAR": "VODAFONE",
    "FUTURE RETAIL": "FUTURE", "FUTURE ENTERPRISES": "FUTURE",
    "SREI": "SREI",
}

def get_stress_tag(issuer: str) -> str | None:
    if not isinstance(issuer, str):
        return None
    issuer_up = issuer.upper()
    for kw, tag in STRESS_ISSUERS.items():
        if kw in issuer_up:
            return tag
    return None

def load_and_clean(filepath: str) -> pd.DataFrame:
    print(f"[ingest] Loading {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)

    # Drop the malformed duplicate-header column (last column)
    bad_cols = [c for c in df.columns if c.count(',') > 3]
    if bad_cols:
        df = df.drop(columns=bad_cols)
        print(f"[ingest] Dropped {len(bad_cols)} malformed column(s)")

    # Rename to internal names
    df = df.rename(columns=COL_MAP)
    df.columns = [c.strip() for c in df.columns]

    # Keep only needed columns
    keep = ["isin", "issuer_name", "ytm", "volume", "date",
            "yield_type", "settlement_status", "remarks"]
    df = df[[c for c in keep if c in df.columns]].copy()

    print(f"[ingest] Raw rows: {len(df):,}")

    # Parse date
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.normalize()

    # Validate ISIN
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    df = df[df["isin"].str.match(r"^[A-Z]{2}[A-Z0-9]{10}$")]

    # YTM numeric + range filter
    df["ytm"] = pd.to_numeric(df["ytm"], errors="coerce")
    df = df[df["ytm"].between(0.5, 30.0)]

    # Yield type filter — YTM only
    if "yield_type" in df.columns:
        df = df[df["yield_type"].astype(str).str.upper().str.contains("YTM", na=False)]

    # Settlement filter — Settled only
    if "settlement_status" in df.columns:
        df = df[df["settlement_status"].astype(str).str.upper().str.contains("SETTLED", na=False)]

    # Volume
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    # Stress tags
    df["stress_tag"] = df["issuer_name"].apply(get_stress_tag)

    df = df.reset_index(drop=True)
    print(f"[ingest] Clean rows: {len(df):,} | "
          f"ISINs: {df['isin'].nunique():,} | "
          f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def main():
    # 1. Load and clean
    trades = load_and_clean(INPUT_FILE)

    # 2. Aggregate to daily YTM
    print("\n[ingest] Computing daily YTM (VWAP)...")
    daily = compute_daily_ytm(trades)

    # 3. Compute features
    print("[ingest] Computing features...")
    features = compute_features(daily)

    # 4. Run detection pipeline
    print(f"[ingest] Running Isolation Forest (contamination={CONTAMINATION})...")
    result = run_detection_pipeline(
        features, contamination=CONTAMINATION, save=True
    )

    confirmed = int(result["confirmed_anomaly"].sum())
    print(f"[ingest] Anomalies: {int(result['is_anomaly'].sum())} raw, "
          f"{confirmed} confirmed")

    # 5. Push to Neon
    print("\n[ingest] Writing to Neon PostgreSQL...")
    engine = get_engine()
    upsert_daily_metrics(features, engine)
    upsert_anomalies(result, engine)

    print(f"\n[ingest] ✅ Complete.")
    print(f"  Daily metric rows: {len(features):,}")
    print(f"  Anomaly rows:      {len(result):,}")
    print(f"  Confirmed anomalies: {confirmed}")
    print(f"\nNow call POST /retrain on Render to update the live model.")


if __name__ == "__main__":
    main()