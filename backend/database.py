import os
from sqlalchemy import (
    create_engine, text,
    Column, String, Float, Integer, Boolean, Date, DateTime, Text
)
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

load_dotenv()

Base = declarative_base()


def get_engine():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL not set in .env")
    # NullPool is important for serverless (Neon/Render) — avoids stale connections
    return create_engine(url, poolclass=NullPool)


# ── ORM Models ────────────────────────────────────────────────────────────

class Trade(Base):
    __tablename__ = "trades"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    date          = Column(Date, nullable=False, index=True)
    isin          = Column(String(12), nullable=False, index=True)
    issuer_name   = Column(String(255))
    ytm           = Column(Float, nullable=False)
    volume        = Column(Float)
    stress_tag    = Column(String(50))
    is_defaulted  = Column(Boolean, default=False)
    loaded_at     = Column(DateTime, default=datetime.utcnow)


class DailyMetric(Base):
    __tablename__ = "daily_metrics"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    date           = Column(Date, nullable=False, index=True)
    isin           = Column(String(12), nullable=False, index=True)
    avg_ytm        = Column(Float)
    prints         = Column(Integer)
    volume_sum     = Column(Float)
    method         = Column(String(10))
    d1             = Column(Float)
    d5             = Column(Float)
    z_score_21d    = Column(Float)
    vol_log        = Column(Float)
    spread_bps     = Column(Float)
    spread_d1      = Column(Float)
    benchmark_ytm  = Column(Float)
    computed_at    = Column(DateTime, default=datetime.utcnow)


class Anomaly(Base):
    __tablename__ = "anomalies"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    date                = Column(Date, nullable=False, index=True)
    isin                = Column(String(12), nullable=False, index=True)
    issuer_name         = Column(String(255))
    avg_ytm             = Column(Float)
    spread_bps          = Column(Float)
    d1                  = Column(Float)
    z_score_21d         = Column(Float)
    anomaly_score       = Column(Float)
    anomaly_score_norm  = Column(Float)
    is_anomaly          = Column(Integer)
    confirmed_anomaly   = Column(Integer)
    cusum_signal        = Column(Integer)
    confidence_score    = Column(Float)
    detected_at         = Column(DateTime, default=datetime.utcnow)

class Portfolio(Base):
    __tablename__ = "portfolios"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id        = Column(String(50), nullable=False, index=True)
    isin                = Column(String(12), nullable=False, index=True)
    issuer_name         = Column(String(255))
    coupon              = Column(Float)
    maturity_date       = Column(Date)
    face_value          = Column(Float)
    weight              = Column(Float)
    rating              = Column(String(20))
    years_to_maturity   = Column(Float)
    stress_tag          = Column(String(50))
    is_stress_issuer    = Column(Boolean, default=False)
    call_date           = Column(Date, nullable=True)
    put_date            = Column(Date, nullable=True)
    uploaded_at         = Column(DateTime, default=datetime.utcnow)


class PortfolioPnL(Base):
    __tablename__ = "portfolio_pnl"

    id                       = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id             = Column(String(50), nullable=False, index=True)
    date                     = Column(Date, nullable=False, index=True)
    isin                     = Column(String(12), nullable=False, index=True)
    issuer_name              = Column(String(255))
    face_value               = Column(Float)
    weight                   = Column(Float)
    coupon                   = Column(Float)
    avg_ytm                  = Column(Float)
    d1_bps                   = Column(Float)
    modified_duration        = Column(Float)
    dv01                     = Column(Float)
    daily_pnl                = Column(Float)
    benchmark_pnl            = Column(Float)
    spread_pnl               = Column(Float)
    cumulative_pnl           = Column(Float)
    portfolio_daily_pnl      = Column(Float)
    portfolio_benchmark_pnl  = Column(Float)
    portfolio_spread_pnl     = Column(Float)
    portfolio_cumulative_pnl = Column(Float)
    stress_tag               = Column(String(50))
    is_anomaly               = Column(Integer, default=0)
    computed_at              = Column(DateTime, default=datetime.utcnow)

# ── Table management ──────────────────────────────────────────────────────

def create_tables():
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("[db] Tables created: trades, daily_metrics, anomalies")


def drop_tables():
    """Use with caution — for dev resets only."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    print("[db] All tables dropped.")


# ── Write helpers ─────────────────────────────────────────────────────────

def upsert_daily_metrics(df, engine=None):
    """Write daily metrics to DB. Replaces all existing data."""
    if engine is None:
        engine = get_engine()

    cols = [
        "date", "isin", "avg_ytm", "prints", "volume_sum", "method",
        "d1", "d5", "z_score_21d", "vol_log",
        "spread_bps", "spread_d1", "benchmark_ytm"
    ]
    available = [c for c in cols if c in df.columns]
    records = df[available].copy()
    records["date"] = pd.to_datetime(records["date"]).dt.date
    records["computed_at"] = datetime.utcnow()

    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE daily_metrics"))
        conn.commit()

    records.to_sql("daily_metrics", engine, if_exists="append",
                   index=False, method="multi", chunksize=500)
    print(f"[db] Wrote {len(records):,} rows to daily_metrics")


def upsert_anomalies(df, engine=None):
    """Write anomaly results to DB."""
    if engine is None:
        engine = get_engine()

    cols = [
        "date", "isin", "avg_ytm", "spread_bps", "d1", "z_score_21d",
        "anomaly_score", "anomaly_score_norm",
        "is_anomaly", "confirmed_anomaly", "cusum_signal"
    ]
    available = [c for c in cols if c in df.columns]
    records = df[available].copy()
    records["detected_at"] = datetime.utcnow()

    # Compute confidence scores before saving
    from engine.evidence import compute_confidence_score
    records["confidence_score"] = records.to_dict(orient="records")
    records["confidence_score"] = [
        compute_confidence_score(r) for r in records.to_dict(orient="records")
    ]
    anomalies_only = records[records["is_anomaly"] == 1]

    with engine.connect() as conn:
        conn.execute(text("DELETE FROM anomalies"))
        conn.commit()

    anomalies_only.to_sql("anomalies", engine, if_exists="append",
                          index=False, method="multi", chunksize=500)
    print(f"[db] Wrote {len(anomalies_only):,} anomaly rows to anomalies")


# ── Read helpers ──────────────────────────────────────────────────────────

def fetch_anomalies(engine=None) -> "pd.DataFrame":
    import pandas as pd
    if engine is None:
        engine = get_engine()
    return pd.read_sql(
        "SELECT * FROM anomalies ORDER BY anomaly_score_norm DESC",
        engine
    )


def fetch_daily_metrics(isin: str = None, engine=None) -> "pd.DataFrame":
    import pandas as pd
    if engine is None:
        engine = get_engine()
    query = "SELECT * FROM daily_metrics ORDER BY isin, date"
    if isin:
        query = f"SELECT * FROM daily_metrics WHERE isin = '{isin}' ORDER BY date"
    return pd.read_sql(query, engine)

def save_portfolio(df: pd.DataFrame, portfolio_id: str, engine=None):
    """Save portfolio holdings to DB."""
    if engine is None:
        engine = get_engine()

    records = df.copy()
    records["portfolio_id"] = portfolio_id
    records["uploaded_at"]  = datetime.utcnow()

    # Convert dates to strings for DB
    for col in ["maturity_date", "call_date", "put_date"]:
        if col in records.columns:
            records[col] = pd.to_datetime(
                records[col], errors="coerce"
            ).dt.date

    # Drop columns not in schema
    keep = [
        "portfolio_id", "isin", "issuer_name", "coupon",
        "maturity_date", "face_value", "weight", "rating",
        "years_to_maturity", "stress_tag", "is_stress_issuer",
        "call_date", "put_date", "uploaded_at"
    ]
    records = records[[c for c in keep if c in records.columns]]

    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM portfolios WHERE portfolio_id = :pid"),
            {"pid": portfolio_id}
        )
        conn.commit()

    records.to_sql("portfolios", engine, if_exists="append",
                   index=False, method="multi", chunksize=500)
    print(f"[db] Saved {len(records)} holdings for portfolio '{portfolio_id}'")


def fetch_portfolio(portfolio_id: str, engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine()
    return pd.read_sql(
        f"SELECT * FROM portfolios WHERE portfolio_id = '{portfolio_id}' ORDER BY face_value DESC",
        engine
    )


def save_portfolio_pnl(df: pd.DataFrame, portfolio_id: str, engine=None):
    """Save PnL attribution results to DB."""
    if engine is None:
        engine = get_engine()

    records = df.copy()
    records["portfolio_id"] = portfolio_id
    records["computed_at"]  = datetime.utcnow()
    records["date"] = pd.to_datetime(records["date"], errors="coerce").dt.date

    # Clean NaN/Inf
    records = records.fillna(0).replace(
        [float("inf"), float("-inf")], 0
    )

    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM portfolio_pnl WHERE portfolio_id = :pid"),
            {"pid": portfolio_id}
        )
        conn.commit()

    records.to_sql("portfolio_pnl", engine, if_exists="append",
                   index=False, method="multi", chunksize=500)
    print(f"[db] Saved {len(records)} PnL rows for portfolio '{portfolio_id}'")


def fetch_portfolio_pnl(portfolio_id: str, engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine()
    return pd.read_sql(
        f"SELECT * FROM portfolio_pnl WHERE portfolio_id = '{portfolio_id}' ORDER BY date, isin",
        engine
    )