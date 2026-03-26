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
from engine.detector import run_detection_pipeline
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

        # Run pipeline
        daily   = compute_daily_ytm(df)
        feat    = compute_features(daily)
        result  = run_detection_pipeline(feat, contamination=contamination, save=True)

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