import pandas as pd
import sys
sys.path.insert(0, '.')
from engine.loader import load_processed_trades
from engine.aggregator import compute_daily_ytm
from engine.features import compute_features

df = load_processed_trades('data/processed_trades.csv')
dhfl = df[df['stress_tag'] == 'DHFL'].copy()

daily = compute_daily_ytm(dhfl)
feat  = compute_features(daily)

stress = feat[feat['date'] >= '2018-09-01'].groupby(
    feat['date'].dt.to_period('M')
)[['avg_ytm', 'spread_bps', 'd1', 'z_score_21d']].mean().round(2)

print("=== DHFL Feature Evolution (stress period) ===")
print(stress[stress.index <= '2019-08'].to_string())