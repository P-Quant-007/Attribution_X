import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import tempfile
import traceback
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from engine.evidence import (
    get_qdrant_client, get_embedding_model,
    retrieve_evidence, explain_anomaly
)
import anthropic

from engine.loader import load_trades, load_processed_trades
from engine.aggregator import compute_daily_ytm
from engine.features import compute_features
from engine.detector import run_detection_pipeline, load_model, predict_anomalies
from engine.features import get_feature_matrix
from backend.database import (
    get_engine, upsert_daily_metrics,
    upsert_anomalies, fetch_anomalies, fetch_daily_metrics
)

app = FastAPI(
    title="Attribution X",
    description="AI Agent for Credit Market Stress Detection",
    version="1.0.0",
)
@app.on_event("startup")
async def startup_event():
    """Startup event — keep lightweight, no model loading here."""
    import logging
    logging.info("Attribution X API starting up...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "attribution-x"}


# ── Main analysis endpoint ────────────────────────────────────────────────

@app.post("/run-analysis")
async def run_analysis(
    file: UploadFile = File(...),
    contamination: float = Query(default=0.03, ge=0.01, le=0.10),
    stress_tag: str = Query(default=None),
):
    """
    Upload a CSV/XLSX of bond trades → run full pipeline → return anomalies.

    Parameters:
    - file:          CSV or XLSX with columns: date, isin, ytm, volume
    - contamination: expected anomaly fraction (default 3%)
    - stress_tag:    optional filter e.g. 'DHFL' to focus on one issuer
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    try:
        # Save upload to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load — try processed format first, fall back to raw
        try:
            df = load_processed_trades(tmp_path)
        except Exception:
            df = load_trades(tmp_path)

        # Optional issuer filter
        if stress_tag and "stress_tag" in df.columns:
            df = df[df["stress_tag"] == stress_tag.upper()]
            if df.empty:
                raise HTTPException(404, f"No trades found for stress_tag={stress_tag}")

        # Run pipeline — use pre-trained model for inference, never retrain here
        daily   = compute_daily_ytm(df)
        feat    = compute_features(daily)

        try:
            # Use the pre-trained model from disk (trained on full 13L dataset)
            model, scaler = load_model()
            X, feat_clean = get_feature_matrix(feat)
            from engine.detector import apply_persistence_filter, apply_cusum
            result = predict_anomalies(feat_clean, X, model, scaler)
            result = apply_persistence_filter(result)
            result = apply_cusum(result)
        except FileNotFoundError:
            # Fallback: no pre-trained model found — train fresh (first-time setup only)
            result = run_detection_pipeline(feat, contamination=contamination, save=False)

        # Persist to DB
        engine = get_engine()
        upsert_daily_metrics(feat, engine)
        upsert_anomalies(result, engine)

        # Build response — clean NaN/Inf before JSON serialisation
        anomalies = result[result["is_anomaly"] == 1].copy()
        anomalies["date"] = anomalies["date"].astype(str)
        anomalies = anomalies.fillna(0).replace(
            [float("inf"), float("-inf")], 0
        )

        return {
            "status": "success",
            "summary": {
                "total_trades":       len(df),
                "total_daily_rows":   len(daily),
                "total_anomalies":    int(result["is_anomaly"].sum()),
                "confirmed_anomalies": int(result["confirmed_anomaly"].sum()),
                "cusum_signals":       int(result["cusum_signal"].sum()),
                "date_range": {
                    "from": str(df["date"].min().date()),
                    "to":   str(df["date"].max().date()),
                },
            },
            "anomalies": anomalies[[
                "date", "isin", "avg_ytm", "spread_bps",
                "d1", "z_score_21d", "anomaly_score_norm",
                "is_anomaly", "confirmed_anomaly", "cusum_signal"
            ]].to_dict(orient="records"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {str(e)}\n{traceback.format_exc()}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Results endpoint ──────────────────────────────────────────────────────

@app.get("/get-results")
def get_results(
    confirmed_only: bool = Query(default=False),
    limit: int = Query(default=100, le=1000),
):
    """
    Fetch stored anomaly results from DB.

    Parameters:
    - confirmed_only: if True, return only confirmed anomalies (2/3 rule)
    - limit:          max rows to return
    """
    try:
        engine = get_engine()
        df = fetch_anomalies(engine)

        if confirmed_only:
            df = df[df["confirmed_anomaly"] == 1]

        df = df.head(limit)
        df["date"] = df["date"].astype(str)
        df = df.fillna(0).replace([float("inf"), float("-inf")], 0)

        return {
            "status": "success",
            "count": len(df),
            "anomalies": df.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(500, f"DB error: {str(e)}")


# ── Metrics endpoint ──────────────────────────────────────────────────────

@app.get("/get-metrics")
def get_metrics(isin: str = Query(default=None)):
    """
    Fetch daily metrics for charting.
    Optionally filter by ISIN.
    """
    try:
        engine = get_engine()
        df = fetch_daily_metrics(isin=isin, engine=engine)
        df["date"] = df["date"].astype(str)
        df = df.fillna(0).replace([float("inf"), float("-inf")], 0)

        return {
            "status": "success",
            "count":  len(df),
            "metrics": df[[
                "date", "isin", "avg_ytm", "spread_bps",
                "d1", "z_score_21d", "prints", "vol_log"
            ]].to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(500, f"DB error: {str(e)}")

# ── Portfolio endpoints ───────────────────────────────────────────────────

@app.post("/upload-portfolio")
async def upload_portfolio(
    file: UploadFile = File(...),
    portfolio_id: str = Query(default="default_portfolio"),
):
    """
    Upload a portfolio CSV/XLSX and persist to DB.
    Returns portfolio summary with DV01 profile.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        from engine.portfolio import load_portfolio, portfolio_summary
        from engine.loader import load_processed_trades
        from engine.aggregator import compute_daily_ytm
        from engine.features import compute_features
        from engine.pnl import enrich_portfolio_with_duration
        from backend.database import save_portfolio

        portfolio_df = load_portfolio(tmp_path)

        # Enrich with duration using stored metrics
        engine_db = get_engine()
        metrics_df = fetch_daily_metrics(engine=engine_db)
        if not metrics_df.empty:
            metrics_df["date"] = pd.to_datetime(metrics_df["date"])
            enriched = enrich_portfolio_with_duration(portfolio_df, metrics_df)
        else:
            enriched = portfolio_df.copy()
            enriched["current_ytm"] = enriched["coupon"]
            enriched["modified_duration"] = 0.0
            enriched["dv01"] = 0.0
            enriched["dv01_contribution_pct"] = 0.0
            enriched["price_per_lac"] = 100.0

        save_portfolio(enriched, portfolio_id=portfolio_id, engine=engine_db)

        summary = portfolio_summary(enriched)
        summary["portfolio_id"] = portfolio_id
        summary["portfolio_dv01"] = round(float(enriched.get("dv01", pd.Series([0])).sum()), 4) if "dv01" in enriched.columns else 0.0

        # Clean NaN
        import math
        def clean(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return 0.0
            return v

        summary["holdings"] = [
            {k: clean(val) if isinstance(val, float) else val
             for k, val in h.items()}
            for h in summary["holdings"]
        ]

        return {"status": "success", "summary": summary}

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Portfolio error: {str(e)}\n{traceback.format_exc()}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.get("/get-portfolio")
def get_portfolio(portfolio_id: str = Query(default="demo_portfolio")):
    """Fetch stored portfolio holdings from DB with computed DV01 and stress metrics."""
    try:
        from backend.database import fetch_portfolio
        from engine.portfolio import portfolio_summary
        from engine.pnl import enrich_portfolio_with_duration

        engine_db    = get_engine()
        df           = fetch_portfolio(portfolio_id, engine=engine_db)
        if df.empty:
            raise HTTPException(404, f"Portfolio '{portfolio_id}' not found.")

        df["maturity_date"] = pd.to_datetime(df["maturity_date"])
        metrics_df = fetch_daily_metrics(engine=engine_db)

        if not metrics_df.empty:
            metrics_df["date"] = pd.to_datetime(metrics_df["date"])
            enriched = enrich_portfolio_with_duration(df, metrics_df)
        else:
            enriched = df.copy()
            enriched["modified_duration"]      = 0.0
            enriched["dv01"]                   = 0.0
            enriched["dv01_contribution_pct"]  = 0.0

        summary = portfolio_summary(enriched)
        summary["portfolio_id"]   = portfolio_id
        summary["portfolio_dv01"] = round(float(enriched["dv01"].sum()), 4) if "dv01" in enriched.columns else 0.0

        import math
        def clean(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return 0.0
            return v

        summary["holdings"] = [
            {k: clean(val) if isinstance(val, float) else val for k, val in h.items()}
            for h in summary["holdings"]
        ]

        return {"status": "success", **summary}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Portfolio fetch error: {str(e)}\n{traceback.format_exc()}")


@app.get("/get-pnl-attribution")
def get_pnl_attribution(
    portfolio_id: str = Query(default="default_portfolio"),
    start_date:   str = Query(default="2018-01-01"),
    end_date:     str = Query(default="2019-12-31"),
):
    """
    Run PnL attribution for a stored portfolio.
    Returns per-bond and portfolio-level attribution report.
    """
    try:
        from engine.pnl import (
            enrich_portfolio_with_duration,
            compute_daily_pnl,
            generate_attribution_report,
        )
        from backend.database import fetch_portfolio

        engine_db = get_engine()
        portfolio_df = fetch_portfolio(portfolio_id, engine=engine_db)

        if portfolio_df.empty:
            raise HTTPException(404, f"Portfolio '{portfolio_id}' not found. Upload it first.")

        portfolio_df["maturity_date"] = pd.to_datetime(portfolio_df["maturity_date"])

        metrics_df = fetch_daily_metrics(engine=engine_db)
        metrics_df["date"] = pd.to_datetime(metrics_df["date"])

        enriched = enrich_portfolio_with_duration(portfolio_df, metrics_df)
        all_metrics = pd.read_sql(
            "SELECT * FROM daily_metrics ORDER BY isin, date",
            engine_db
        )
        all_metrics["date"] = pd.to_datetime(all_metrics["date"])
        all_metrics = all_metrics.fillna(0).replace([float("inf"), float("-inf")], 0)

        pnl = compute_daily_pnl(enriched, all_metrics, start_date, end_date)

        if pnl.empty:
            return {
                "status": "success",
                "message": "No PnL data found for portfolio ISINs in date range.",
                "report": {}
            }

        from backend.database import save_portfolio_pnl
        save_portfolio_pnl(pnl, portfolio_id=portfolio_id, engine=engine_db)

        report = generate_attribution_report(pnl)

        # Clean NaN/Inf
        import json
        report_clean = json.loads(
            json.dumps(report, default=lambda x: 0 if isinstance(x, float) and (x != x or abs(x) == float("inf")) else x)
        )

        return {"status": "success", "portfolio_id": portfolio_id, "report": report_clean}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Attribution error: {str(e)}\n{traceback.format_exc()}")


@app.get("/get-suggestions")
def get_suggestions(
    portfolio_id: str = Query(default="default_portfolio"),
    start_date:   str = Query(default="2018-01-01"),
    end_date:     str = Query(default="2019-12-31"),
):
    """
    Generate 3 AI reallocation suggestions for a portfolio.
    Combines PnL attribution + anomaly signals.
    """
    try:
        from engine.pnl import (
            enrich_portfolio_with_duration,
            compute_daily_pnl,
            generate_attribution_report,
        )
        from engine.suggestions import generate_suggestions
        from backend.database import fetch_portfolio

        engine_db   = get_engine()
        portfolio_df = fetch_portfolio(portfolio_id, engine=engine_db)

        if portfolio_df.empty:
            raise HTTPException(404, f"Portfolio '{portfolio_id}' not found.")

        portfolio_df["maturity_date"] = pd.to_datetime(portfolio_df["maturity_date"])

        metrics_df = fetch_daily_metrics(engine=engine_db)
        metrics_df["date"] = pd.to_datetime(metrics_df["date"])

        enriched = enrich_portfolio_with_duration(portfolio_df, metrics_df)

        all_metrics = pd.read_sql(
            "SELECT * FROM daily_metrics ORDER BY isin, date",
            engine_db
        )
        all_metrics["date"] = pd.to_datetime(all_metrics["date"])
        all_metrics = all_metrics.fillna(0).replace([float("inf"), float("-inf")], 0)

        pnl    = compute_daily_pnl(enriched, all_metrics, start_date, end_date)
        report = generate_attribution_report(pnl)

        anomalies = fetch_anomalies(engine_db)
        anomalies  = anomalies.fillna(0).replace([float("inf"), float("-inf")], 0)

        ant_key    = os.getenv("ANTHROPIC_API_KEY", "")
        suggestions = generate_suggestions(report, anomalies, portfolio_df, ant_key)

        return {
            "status":      "success",
            "portfolio_id": portfolio_id,
            "suggestions": suggestions,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Suggestion error: {str(e)}\n{traceback.format_exc()}")

@app.post("/retrain")
def retrain_model(contamination: float = Query(default=0.03, ge=0.01, le=0.10)):
    """
    Retrain Isolation Forest on all features currently stored in Neon.
    Call this after ingesting new data locally via ingest_full_dataset.py.
    """
    try:
        import pickle
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import RobustScaler
        from sqlalchemy import text

        engine_db = get_engine()
        with engine_db.connect() as conn:
            result = conn.execute(text(
                "SELECT d1, z_score_21d, vol_log, spread_bps, spread_d1, prints FROM anomalies"
            ))
            rows = result.fetchall()

        if not rows:
            raise HTTPException(404, "No data in anomalies table.")

        X = np.array([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows], dtype=float)
        scaler   = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
        clf.fit(X_scaled)

        model_path = os.path.join(os.path.dirname(__file__), "..", "engine", "isolation_forest.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"model": clf, "scaler": scaler}, f)

        return {
            "status":        "retrained",
            "rows_used":     len(rows),
            "contamination": contamination,
            "model_path":    "engine/isolation_forest.pkl",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Retrain error: {str(e)}\n{traceback.format_exc()}")

@app.get("/explain-anomaly")
def explain_anomaly_endpoint(
    isin: str = Query(...),
    date: str = Query(...),
):
    """
    Get AI explanation for a specific anomaly.
    Retrieves evidence from Qdrant and generates LLM narrative.
    """
    try:
        engine = get_engine()
        df = fetch_anomalies(engine)
        df["date"] = df["date"].astype(str)

        row = df[(df["isin"] == isin) & (df["date"].str[:10] == date[:10])]
        if row.empty:
            raise HTTPException(404, f"Anomaly not found for {isin} on {date}")

        anomaly = row.iloc[0].to_dict()

        # Get Qdrant client and embedding model
        qdrant  = get_qdrant_client()
        model   = get_embedding_model()

        # Get Anthropic client if key available
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        ant_client = None
        if anthropic_key and anthropic_key != "your_anthropic_api_key":
            ant_client = anthropic.Anthropic(api_key=anthropic_key)

        result = explain_anomaly(anomaly, qdrant, model, ant_client)
        result = {k: (0 if isinstance(v, float) and
                      (v != v or v == float("inf") or v == float("-inf"))
                      else v)
                  for k, v in result.items()}

        return {"status": "success", "explanation": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Explanation error: {str(e)}")
    

def write_anomalies_direct(result_df, engine):
    """Write anomalies to DB without importing engine.evidence (avoids onnxruntime on Windows)."""
    from sqlalchemy import text
    import math

    df = result_df.copy()

    # Compute confidence score inline (same formula as compute_confidence_score)
    df["prints_norm"] = (df["prints"] / 20).clip(upper=1.0)
    df["vol_norm"]    = (df["vol_log"] / 20).clip(upper=1.0)
    df["confidence_score"] = (
        0.4 * df["prints_norm"] +
        0.3 * df["vol_norm"] +
        0.3 * df["confirmed_anomaly"].astype(float)
    ).round(4)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.fillna(0).replace([float("inf"), float("-inf")], 0)

    keep = [
        "date", "isin", "avg_ytm", "spread_bps", "d1", "d5",
        "z_score_21d", "vol_log", "prints",
        "anomaly_score", "anomaly_score_norm",
        "is_anomaly", "confirmed_anomaly", "cusum_signal",
        "confidence_score",
    ]
    df = df[[c for c in keep if c in df.columns]]

    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE anomalies"))
        conn.commit()

    df.to_sql("anomalies", engine, if_exists="append",
              index=False, method="multi", chunksize=500)
    print(f"[db] Wrote {len(df)} rows to anomalies")

