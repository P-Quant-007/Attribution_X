"""
Write daily metrics to Neon DB from combined_all_records.csv.
Run this once to populate the daily_metrics table for charting.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from engine.loader import load_trades
from engine.aggregator import compute_daily_ytm
from engine.features import compute_features
from backend.database import get_engine, upsert_daily_metrics

DATA_FILE = os.path.join(os.path.dirname(__file__), "combined_all_records.csv")

print("Loading data...")
df = load_trades(DATA_FILE)
print(f"Clean rows: {len(df):,}")

print("Aggregating daily YTM...")
daily = compute_daily_ytm(df)
print(f"Daily rows: {len(daily):,}")

print("Computing features...")
feat = compute_features(daily)
print(f"Feature rows: {len(feat):,}")

print("Writing to Neon daily_metrics table...")
engine = get_engine()
upsert_daily_metrics(feat, engine)
print(f"Done. {len(feat):,} rows written to daily_metrics.")