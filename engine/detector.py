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
    return result