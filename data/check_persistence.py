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

print("=== FILTER SUMMARY ===")
print(f"Raw anomalies:       {result['is_anomaly'].sum()}")
print(f"Confirmed anomalies: {result['confirmed_anomaly'].sum()}")
print(f"CUSUM signals:       {result['cusum_signal'].sum()}")
print(f"Either signal:       {((result['confirmed_anomaly']==1)|(result['cusum_signal']==1)).sum()}")

print("\n=== CONFIRMED ANOMALY DATES (DHFL) ===")
confirmed = result[result['confirmed_anomaly'] == 1]
monthly = confirmed.groupby(confirmed['date'].dt.to_period('M')).size()
print(monthly.to_string())