import pandas as pd
import numpy as np


# GOI benchmark curve — approximate 3yr ZCYC by year
# Source: RBI FBIL historical data (hardcoded for deterministic computation)
GOI_BENCHMARK = {
    2015: 7.72, 2016: 7.11, 2017: 6.79, 2018: 7.76,
    2019: 6.84, 2020: 5.77, 2021: 5.98, 2022: 7.27,
    2023: 7.18, 2024: 7.02,
}


def get_benchmark_ytm(date: pd.Timestamp) -> float:
    """Return approximate GOI 3yr benchmark YTM for a given date."""
    return GOI_BENCHMARK.get(date.year, 7.50)


def compute_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute anomaly detection features from daily aggregated YTM data.

    Input:  DataFrame with columns (date, isin, avg_ytm, prints, volume_sum)
            — output of aggregator.compute_daily_ytm()

    Output: DataFrame with additional columns:
            d1          — 1-day YTM change (bps)
            d5          — 5-day YTM change (bps)
            vol_log     — log(volume_sum + 1)
            spread_bps  — YTM minus GOI benchmark (bps)
            spread_d1   — 1-day change in spread (bps)
            z_score_21d — rolling 21-day z-score of avg_ytm
    """
    if daily_df.empty:
        return daily_df

    df = daily_df.copy().sort_values(["isin", "date"])

    # ── Per-ISIN rolling features ─────────────────────────────────────────
    df["d1"] = (
        df.groupby("isin")["avg_ytm"]
        .diff(1)
        .mul(100)   # convert % to bps
        .round(4)
    )

    df["d5"] = (
        df.groupby("isin")["avg_ytm"]
        .diff(5)
        .mul(100)
        .round(4)
    )

    # Rolling 21-day z-score (measures how far today's YTM is from recent norm)
    def rolling_zscore(s: pd.Series, window: int = 21) -> pd.Series:
        mean = s.rolling(window, min_periods=5).mean()
        std  = s.rolling(window, min_periods=5).std()
        return ((s - mean) / std.replace(0, np.nan)).round(4)

    df["z_score_21d"] = (
        df.groupby("isin")["avg_ytm"]
        .transform(rolling_zscore)
    )

    # ── Volume feature ─────────────────────────────────────────────────────
    df["vol_log"] = np.log1p(df["volume_sum"].fillna(0)).round(4)

    # ── Benchmark spread ──────────────────────────────────────────────────
    df["benchmark_ytm"] = df["date"].apply(get_benchmark_ytm)
    df["spread_bps"] = ((df["avg_ytm"] - df["benchmark_ytm"]) * 100).round(2)

    df["spread_d1"] = (
        df.groupby("isin")["spread_bps"]
        .diff(1)
        .round(4)
    )

    # ── Prints (already present, ensure it's numeric) ─────────────────────
    df["prints"] = pd.to_numeric(df["prints"], errors="coerce").fillna(0)

    # ── Drop rows with insufficient history (first 5 days per ISIN) ───────
    # Keep NaN rows — Isolation Forest will handle via imputation in detector
    df = df.sort_values(["isin", "date"]).reset_index(drop=True)

    print(f"[features] {len(df)} rows | "
          f"Features: d1, d5, z_score_21d, vol_log, spread_bps, spread_d1, prints")

    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return clean numeric feature matrix ready for model input.
    Fills NaN with column medians (safe for Isolation Forest).
    Only returns rows where we have at least d1 computed (min 2 days history).
    """
    feature_cols = ["d1", "d5", "z_score_21d", "vol_log",
                    "spread_bps", "spread_d1", "prints"]

    # Need at least d1
    df_clean = df.dropna(subset=["d1"]).copy()

    X = df_clean[feature_cols].copy()

    # Fill remaining NaN with column medians
    for col in feature_cols:
        median = X[col].median()
        X[col] = X[col].fillna(median)

    return X, df_clean