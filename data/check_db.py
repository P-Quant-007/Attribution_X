import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'backend')

from engine.loader import load_processed_trades
from engine.aggregator import compute_daily_ytm
from engine.features import compute_features
from engine.detector import run_detection_pipeline
from backend.database import get_engine, upsert_daily_metrics, upsert_anomalies, fetch_anomalies

# Use DHFL only for this test
df = load_processed_trades('data/processed_trades.csv')
dhfl = df[df['stress_tag'] == 'DHFL'].copy()

daily  = compute_daily_ytm(dhfl)
feat   = compute_features(daily)
result = run_detection_pipeline(feat, contamination=0.03, save=False)

engine = get_engine()
upsert_daily_metrics(feat, engine)
upsert_anomalies(result, engine)

# Read back and verify
anomalies = fetch_anomalies(engine)
print(f"\n[db] Verified read-back: {len(anomalies)} anomaly rows in DB")
print(anomalies[["date","isin","avg_ytm","anomaly_score_norm","confirmed_anomaly"]].head(10).to_string(index=False))