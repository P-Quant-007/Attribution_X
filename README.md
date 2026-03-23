# Attribution X
**AI Agent for Credit Market Stress Detection and NAV Attribution**

## Overview
An AI-powered system for the Indian fixed-income market that explains daily NAV movements, detects anomalies in bond yield behaviour, and identifies early signs of credit stress (e.g. DHFL default).

## Architecture
- **Engine**: Isolation Forest + CUSUM anomaly detection on bond YTMs
- **Backend**: FastAPI on Render
- **Frontend**: Streamlit on Streamlit Cloud
- **Database**: Neon PostgreSQL
- **Vector DB**: Qdrant Cloud (evidence retrieval)

## Project Structure
```
attribution-x/
├── backend/        # FastAPI service
├── engine/         # Analytics pipeline
├── frontend/       # Streamlit dashboard
├── data/           # Synthetic demo dataset
├── tests/          # Unit tests
├── requirements.txt
└── .env.example
```

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your credentials
```

## Demo
Upload the synthetic dataset → anomalies detected → DHFL stress period highlighted → AI explanation retrieved.