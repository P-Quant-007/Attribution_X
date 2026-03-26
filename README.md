# Attribution X
**AI Agent for Credit Market Stress Detection and NAV Attribution**

## Live Demo
- **Frontend:** https://attributionx-is7crrba7zg98kanjxgbgu.streamlit.app
- **Backend API:** https://attribution-x.onrender.com
- **API Docs:** https://attribution-x.onrender.com/docs

## Overview
An AI-powered system for the Indian fixed-income market that detects anomalies in bond yield behaviour and identifies early signs of credit stress using real CBRICS/FIMMDA trade data.

Built for the ET Hackathon — PS-5 (Domain-Specific AI Agent).

## What it does
1. Ingests real CBRICS bond trade data (643,067 trades, 2015–2024)
2. Computes daily VWAP-weighted YTM and benchmark spread per ISIN
3. Engineers features: d1, d5, spread_bps, z_score_21d, vol_log
4. Detects anomalies using Isolation Forest + 2/3 persistence filter + CUSUM
5. Retrieves relevant historical credit events from Qdrant vector DB
6. Generates AI explanations for each anomaly with confidence scores

## The DHFL Story
DHFL bonds traded at ~9% YTM in early 2018. By April 2019, yields had spiked to 29.84% — a 2,000+ bps spread above the GOI benchmark. The system:
- Flags Apr 18, 2019 with anomaly score **1.0** (maximum)
- Confirms via persistence filter and CUSUM regime detection
- Retrieves evidence: Cobrapost expose (Jan 2019), halted repayments (Apr 2019), IBC proceedings (Nov 2019)

## Architecture
```
CBRICS CSVs → Data Loader → Daily VWAP Aggregator → Feature Engineering
→ Isolation Forest → Persistence Filter (2/3 rule) → CUSUM
→ Neon PostgreSQL → FastAPI (Render) → Streamlit (Streamlit Cloud)
                                     ↘ Qdrant Cloud (evidence retrieval)
```

## Tech Stack
| Layer | Technology |
|---|---|
| Engine | scikit-learn Isolation Forest, CUSUM |
| Backend | FastAPI, Uvicorn, Render |
| Frontend | Streamlit, Plotly |
| Database | Neon PostgreSQL (SQLAlchemy) |
| Vector DB | Qdrant Cloud (fastembed ONNX) |
| AI | Anthropic Claude (explanations) |
| Data | CBRICS/FIMMDA real trade data |

## Project Structure
```
attribution-x/
├── backend/           # FastAPI service (main.py, database.py)
├── engine/            # Analytics pipeline
│   ├── loader.py      # Data ingestion
│   ├── aggregator.py  # Daily VWAP
│   ├── features.py    # Feature engineering
│   ├── detector.py    # Isolation Forest + CUSUM
│   └── evidence.py    # Qdrant retrieval + AI explanation
├── frontend/          # Streamlit dashboard
│   └── app.py
├── data/              # CBRICS loader scripts
├── tests/             # 16 unit tests
├── requirements.txt
└── .env.example
```

## Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/attribution-x.git
cd attribution-x
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env         # fill in DATABASE_URL, QDRANT_URL, QDRANT_API_KEY
```

Start backend:
```bash
cd backend
python run.py
```

Start frontend (new terminal):
```bash
cd frontend
streamlit run app.py
```

## Environment Variables
```
DATABASE_URL        # Neon PostgreSQL connection string
QDRANT_URL          # Qdrant Cloud cluster URL
QDRANT_API_KEY      # Qdrant API key
ANTHROPIC_API_KEY   # Anthropic API key (optional, fallback explanation used if absent)
```

## Key Results
- **643,067** real bond trades ingested (2015–2024)
- **17,561** unique ISINs across 1,774 issuers
- **71** anomalies detected on DHFL, **28** confirmed by persistence filter
- **Apr 18, 2019** scores 1.0 — peak DHFL stress correctly identified
- **5** major stress issuers tracked: DHFL, IL&FS, Yes Bank, Reliance Capital, Vodafone

## Demo Flow
1. Open the Streamlit app
2. Click **"Load last results from DB"**
3. Go to **Results** tab — YTM chart loads with anomaly markers
4. Select `INE202B07HQ0` (DHFL bond) → click **"Explain this anomaly"**
5. See AI explanation with DHFL-specific evidence from Qdrant
```

