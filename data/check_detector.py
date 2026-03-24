import pandas as pd
import sys
sys.path.insert(0, '.')
from engine.loader import load_processed_trades
from engine.aggregator import compute_daily_ytm
from engine.features import compute_features
from engine.detector import run_detection_pipeline

df = load_processed_trades('data/processed_trades.csv')
dhfl = df[df['stress_tag'] == 'DHFL'].copy()

daily  = compute_daily_ytm(dhfl)
feat   = compute_features(daily)
result = run_detection_pipeline(feat, contamination=0.03, save=False)

# Show top anomalies
top = (result[result["is_anomaly"] == 1]
       .sort_values("anomaly_score_norm", ascending=False)
       [["date", "isin", "avg_ytm", "spread_bps",
         "d1", "z_score_21d", "anomaly_score_norm"]]
       .head(15))

print("=== TOP 15 ANOMALIES (DHFL) ===")
print(top.to_string(index=False))

print(f"\nTotal anomalies: {result['is_anomaly'].sum()}")
print(f"Date range of anomalies: "
      f"{result[result['is_anomaly']==1]['date'].min().date()} to "
      f"{result[result['is_anomaly']==1]['date'].max().date()}")