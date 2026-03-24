import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import joblib
from pathlib import Path


FEATURE_COLS = [
    "d1", "d5", "z_score_21d",
    "vol_log", "spread_bps", "spread_d1", "prints"
]

MODEL_PATH = Path("engine/isolation_forest.pkl")
SCALER_PATH = Path("engine/scaler.pkl")


def train_model(
    X: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42,
) -> tuple:
    """
    Train Isolation Forest on feature matrix X.

    contamination: expected fraction of anomalies (3% is conservative for
                   bond markets — real stress events are rare)
    Returns: (model, scaler)
    """
    scaler = RobustScaler()  # robust to outliers — better than StandardScaler here
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    print(f"[detector] Model trained on {len(X):,} samples | "
          f"contamination={contamination} | "
          f"estimators=200")

    return model, scaler


def predict_anomalies(
    df: pd.DataFrame,
    X: pd.DataFrame,
    model: IsolationForest,
    scaler: RobustScaler,
) -> pd.DataFrame:
    """
    Apply trained model to feature matrix and attach scores to df.

    anomaly_score: raw isolation forest score (lower = more anomalous)
    anomaly_score_norm: normalised to [0,1] where 1 = most anomalous
    is_anomaly: binary flag (1 = anomaly)
    """
    X_scaled = scaler.transform(X)

    # Raw scores: negative values = more anomalous in sklearn convention
    raw_scores = model.score_samples(X_scaled)

    # Normalise: flip and scale to [0,1]
    score_min, score_max = raw_scores.min(), raw_scores.max()
    norm_scores = 1 - (raw_scores - score_min) / (score_max - score_min + 1e-10)

    # Binary flag: sklearn predict returns -1 for anomaly, 1 for normal
    predictions = model.predict(X_scaled)
    is_anomaly = (predictions == -1).astype(int)

    result = df.copy()
    result["anomaly_score"]      = raw_scores.round(6)
    result["anomaly_score_norm"] = norm_scores.round(4)
    result["is_anomaly"]         = is_anomaly

    n_anomalies = is_anomaly.sum()
    pct = 100 * n_anomalies / len(result)
    print(f"[detector] {n_anomalies:,} anomalies flagged "
          f"({pct:.1f}% of {len(result):,} rows)")

    return result


def save_model(model, scaler):
    """Persist model and scaler to disk."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[detector] Model saved to {MODEL_PATH}")


def load_model():
    """Load persisted model and scaler."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run the pipeline first."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def run_detection_pipeline(
    daily_features_df: pd.DataFrame,
    contamination: float = 0.03,
    save: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: feature matrix → train → predict → return annotated df.
    This is the single entry point called by FastAPI.
    """
    from engine.features import get_feature_matrix

    X, df_clean = get_feature_matrix(daily_features_df)

    if len(X) < 50:
        raise ValueError(
            f"Too few samples ({len(X)}) to train model. "
            f"Need at least 50 rows with computed features."
        )

    model, scaler = train_model(X, contamination=contamination)

    if save:
        save_model(model, scaler)

    result = predict_anomalies(df_clean, X, model, scaler)
    result = apply_persistence_filter(result)
    result = apply_cusum(result)
    return result

def apply_persistence_filter(
    result_df: pd.DataFrame,
    window: int = 3,
    min_hits: int = 2,
) -> pd.DataFrame:
    """
    Apply 2-out-of-3 rolling persistence filter per ISIN.

    For each ISIN, a confirmed_anomaly is flagged only when
    at least min_hits anomalies appear within a rolling window of days.

    This eliminates single-day data glitches while preserving
    genuine sustained stress periods.

    Adds column: confirmed_anomaly (0 or 1)
    """
    df = result_df.copy().sort_values(["isin", "date"])

    df["confirmed_anomaly"] = (
        df.groupby("isin")["is_anomaly"]
        .transform(
            lambda s: s.rolling(window, min_periods=1)
                       .sum()
                       .ge(min_hits)
                       .astype(int)
        )
    )

    n_confirmed = df["confirmed_anomaly"].sum()
    n_raw       = df["is_anomaly"].sum()
    filtered    = n_raw - n_confirmed

    print(f"[persistence] Raw anomalies: {n_raw} | "
          f"Confirmed (≥{min_hits}/{window}): {n_confirmed} | "
          f"Filtered out: {filtered}")

    return df

def apply_cusum(
    result_df: pd.DataFrame,
    threshold: float = 5.0,
    drift: float = 0.5,
) -> pd.DataFrame:
    """
    CUSUM (Cumulative Sum) regime shift detector on avg_ytm per ISIN.

    Detects sustained upward shifts in YTM that persist beyond
    day-to-day noise — ideal for catching slow-building credit stress.

    threshold: sensitivity (lower = more sensitive). 5.0 is standard.
    drift:     allowance for natural drift (0.5 = 50bps tolerance).

    Adds column: cusum_signal (0 or 1)
    """
    df = result_df.copy().sort_values(["isin", "date"])

    def cusum_per_isin(series: pd.Series) -> pd.Series:
        s = series.values
        n = len(s)
        cusum_pos = np.zeros(n)   # upward CUSUM
        signal    = np.zeros(n, dtype=int)

        for i in range(1, n):
            # Standardise by rolling std (min 5 periods)
            window_vals = s[max(0, i-20):i]
            mu  = window_vals.mean() if len(window_vals) > 1 else s[i]
            std = window_vals.std()  if len(window_vals) > 1 else 1.0
            std = max(std, 0.01)    # avoid division by zero

            z = (s[i] - mu) / std
            cusum_pos[i] = max(0, cusum_pos[i-1] + z - drift)

            if cusum_pos[i] > threshold:
                signal[i] = 1

        return pd.Series(signal, index=series.index)

    df["cusum_signal"] = (
        df.groupby("isin")["avg_ytm"]
        .transform(cusum_per_isin)
    )

    n_cusum = df["cusum_signal"].sum()
    print(f"[cusum] Regime shifts detected: {n_cusum} rows flagged")

    return df