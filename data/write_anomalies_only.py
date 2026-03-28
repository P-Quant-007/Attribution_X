import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'backend')
import pandas as pd
from sqlalchemy import text
from engine.aggregator import compute_daily_ytm
from engine.features   import compute_features
from engine.detector   import run_detection_pipeline
from backend.database  import get_engine

INPUT_FILE    = "data/combined_all_records.csv"
CONTAMINATION = 0.03

COL_MAP = {
    "ISIN": "isin", "Issuer Name": "issuer_name",
    "Yield": "ytm", "Trade Value in Rs. Lacs": "volume",
    "Trade Date & Time": "date", "Yield Type": "yield_type",
    "Settlement Status": "settlement_status", "Remarks": "remarks",
}

print("Loading data...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
bad_cols = [c for c in df.columns if c.count(',') > 3]
df = df.drop(columns=bad_cols)
df = df.rename(columns=COL_MAP)

df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"])
df["date"] = df["date"].dt.normalize()
df["isin"] = df["isin"].astype(str).str.strip().str.upper()
df = df[df["isin"].str.match(r"^[A-Z]{2}[A-Z0-9]{10}$")]
df["ytm"] = pd.to_numeric(df["ytm"], errors="coerce")
df = df[df["ytm"].between(0.5, 30.0)]
if "yield_type" in df.columns:
    df = df[df["yield_type"].astype(str).str.upper().str.contains("YTM", na=False)]
if "settlement_status" in df.columns:
    df = df[df["settlement_status"].astype(str).str.upper().str.contains("SETTLED", na=False)]
df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
print(f"Clean rows: {len(df):,}")

daily    = compute_daily_ytm(df)
features = compute_features(daily)
result   = run_detection_pipeline(features, contamination=CONTAMINATION, save=True)

# Confidence score inline
result["prints_norm"] = (result["prints"] / 20).clip(upper=1.0)
result["vol_norm"]    = (result["vol_log"] / 20).clip(upper=1.0)
result["confidence_score"] = (
    0.4 * result["prints_norm"] +
    0.3 * result["vol_norm"] +
    0.3 * result["confirmed_anomaly"].astype(float)
).round(4)

keep = ["date","isin","avg_ytm","spread_bps","d1","z_score_21d",
        "vol_log","prints","anomaly_score","anomaly_score_norm",
        "is_anomaly","confirmed_anomaly","cusum_signal","confidence_score"]
out = result[[c for c in keep if c in result.columns]].copy()
out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
out = out.fillna(0).replace([float("inf"), float("-inf")], 0)

engine = get_engine()
with engine.connect() as conn:
    conn.execute(text("TRUNCATE TABLE anomalies"))
    conn.commit()

out.to_sql("anomalies", engine, if_exists="replace",
           index=False, method="multi", chunksize=500)

confirmed = int(result["confirmed_anomaly"].sum())
print(f"\n✅ Done. {len(out):,} anomaly rows written. {confirmed} confirmed.")
print("Now run: curl -X POST https://attribution-x.onrender.com/retrain")