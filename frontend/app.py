import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ── Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Attribution X",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"  # replaced with Render URL on deployment

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Attribution X")
    st.caption("AI Agent for Credit Market Stress Detection")
    st.divider()

    api_url = st.text_input("API URL", value=API_URL)

    st.subheader("Analysis Settings")
    contamination = st.slider(
        "Anomaly sensitivity",
        min_value=0.01, max_value=0.10,
        value=0.03, step=0.01,
        help="Expected fraction of anomalies. Higher = more flags."
    )
    stress_tag = st.selectbox(
        "Focus on issuer",
        options=["ALL", "DHFL", "ILFS", "YES_BANK",
                 "RELIANCE_CAP", "VODAFONE", "FUTURE"],
        index=1,
    )
    confirmed_only = st.checkbox("Show confirmed anomalies only", value=False)
    st.divider()

    # Health check
    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        if r.status_code == 200:
            st.success("API connected")
        else:
            st.error("API error")
    except Exception:
        st.warning("API unreachable — start backend first")


# ── Main ──────────────────────────────────────────────────────────────────
st.title("📊 Attribution X")
st.markdown("**Credit Market Stress Detection & NAV Attribution** — Indian Fixed Income")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Run Analysis", "📈 Results", "ℹ️ About"])

# ── TAB 1: Upload & Run ───────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Upload Trade Data")
        uploaded = st.file_uploader(
            "Upload CSV or XLSX",
            type=["csv", "xlsx"],
            help="CBRICS format or processed_trades.csv"
        )

    with col2:
        st.subheader("Quick Stats")
        if "results" in st.session_state:
            r = st.session_state["results"]
            s = r["summary"]
            total = s['total_trades']
            st.metric("Total Trades",    f"{total:,}" if isinstance(total, int) else str(total))
            st.metric("Anomalies Found", s["total_anomalies"])
            st.metric("Confirmed",       s["confirmed_anomalies"])

    if uploaded:
        if st.button("▶ Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Running pipeline... (this takes ~30 seconds)"):
                try:
                    params = {"contamination": contamination}
                    if stress_tag != "ALL":
                        params["stress_tag"] = stress_tag

                    resp = requests.post(
                        f"{api_url}/run-analysis",
                        files={"file": (uploaded.name,
                                        uploaded.getvalue(),
                                        "text/csv")},
                        params=params,
                        timeout=120,
                    )

                    if resp.status_code == 200:
                        st.session_state["results"] = resp.json()
                        st.success("Analysis complete!")
                        st.rerun()
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text[:300]}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Is the backend running?")
                except Exception as e:
                    st.error(f"Error: {e}")

    else:
        st.info("Upload your CBRICS trade file to begin analysis.")

        # Demo mode — load from DB
        st.divider()
        st.subheader("Or load stored results")
        if st.button("📂 Load last results from DB"):
            with st.spinner("Fetching from database..."):
                try:
                    params = {"confirmed_only": confirmed_only, "limit": 500}
                    resp = requests.get(f"{api_url}/get-results", params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state["results"] = {
                            "summary": {
                                "total_trades": "—",
                                "total_daily_rows": "—",
                                "total_anomalies": data["count"],
                                "confirmed_anomalies": sum(
                                    1 for a in data["anomalies"]
                                    if a.get("confirmed_anomaly") == 1
                                ),
                                "cusum_signals": "—",
                                "date_range": {"from": "—", "to": "—"},
                            },
                            "anomalies": data["anomalies"],
                        }
                        st.success(f"Loaded {data['count']} anomalies from DB")
                        st.rerun()
                    else:
                        st.error("Failed to fetch results")
                except Exception as e:
                    st.error(f"Error: {e}")


# ── TAB 2: Results & Charts ───────────────────────────────────────────────
with tab2:
    if "results" not in st.session_state:
        st.info("Run an analysis first or load stored results from the Run Analysis tab.")
    else:
        results  = st.session_state["results"]
        summary  = results["summary"]
        anomalies_list = results["anomalies"]

        if not anomalies_list:
            st.warning("No anomalies found.")
        else:
            df_anom = pd.DataFrame(anomalies_list)
            df_anom["date"] = pd.to_datetime(df_anom["date"])

            if confirmed_only and "confirmed_anomaly" in df_anom.columns:
                df_anom = df_anom[df_anom["confirmed_anomaly"] == 1]

            # ── Summary metrics ───────────────────────────────────────
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Trades",     f"{summary['total_trades']:,}" if isinstance(summary['total_trades'], int) else summary['total_trades'])
            c2.metric("Daily Rows",       f"{summary['total_daily_rows']:,}" if isinstance(summary['total_daily_rows'], int) else summary['total_daily_rows'])
            c3.metric("Raw Anomalies",    summary["total_anomalies"])
            c4.metric("Confirmed",        summary["confirmed_anomalies"])
            c5.metric("Date Range",       f"{summary['date_range']['from']} → {summary['date_range']['to']}")

            st.divider()

            # ── Fetch metrics for charting ─────────────────────────────
            st.subheader("YTM Time Series with Anomaly Flags")

            try:
                isin_filter = None
                if stress_tag != "ALL":
                    # Get most anomalous ISIN for this tag
                    top_isin = (df_anom.sort_values("anomaly_score_norm", ascending=False)
                                       .iloc[0]["isin"])
                    isin_filter = top_isin

                metrics_resp = requests.get(
                    f"{api_url}/get-metrics",
                    params={"isin": isin_filter} if isin_filter else {},
                    timeout=15,
                )

                if metrics_resp.status_code == 200:
                    df_metrics = pd.DataFrame(metrics_resp.json()["metrics"])
                    df_metrics["date"] = pd.to_datetime(df_metrics["date"])

                    if isin_filter:
                        df_plot = df_metrics[df_metrics["isin"] == isin_filter]
                        anom_plot = df_anom[df_anom["isin"] == isin_filter]
                        chart_title = f"YTM — {isin_filter}"
                    else:
                        # Aggregate across all ISINs
                        df_plot = (df_metrics.groupby("date")["avg_ytm"]
                                             .mean()
                                             .reset_index())
                        anom_plot = df_anom
                        chart_title = "Average YTM — All ISINs"

                    # Build chart
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.05,
                    )

                    # YTM line
                    fig.add_trace(go.Scatter(
                        x=df_plot["date"],
                        y=df_plot["avg_ytm"],
                        mode="lines",
                        name="Avg YTM (%)",
                        line=dict(color="#4C9BE8", width=1.5),
                    ), row=1, col=1)

                    # Anomaly markers
                    if len(anom_plot):
                        confirmed = anom_plot[anom_plot.get("confirmed_anomaly", pd.Series([0]*len(anom_plot))) == 1]
                        unconfirmed = anom_plot[anom_plot.get("confirmed_anomaly", pd.Series([0]*len(anom_plot))) != 1]

                        if len(confirmed):
                            fig.add_trace(go.Scatter(
                                x=confirmed["date"],
                                y=confirmed["avg_ytm"],
                                mode="markers",
                                name="Confirmed anomaly",
                                marker=dict(color="red", size=10,
                                            symbol="circle",
                                            line=dict(color="darkred", width=1)),
                                hovertemplate=(
                                    "<b>%{x}</b><br>"
                                    "YTM: %{y:.2f}%<br>"
                                    "Score: %{customdata:.3f}<extra></extra>"
                                ),
                                customdata=confirmed["anomaly_score_norm"],
                            ), row=1, col=1)

                        if len(unconfirmed):
                            fig.add_trace(go.Scatter(
                                x=unconfirmed["date"],
                                y=unconfirmed["avg_ytm"],
                                mode="markers",
                                name="Raw anomaly",
                                marker=dict(color="orange", size=7,
                                            symbol="circle-open",
                                            line=dict(color="orange", width=1.5)),
                            ), row=1, col=1)

                    # Spread chart
                    if "spread_bps" in df_plot.columns:
                        fig.add_trace(go.Bar(
                            x=df_plot["date"],
                            y=df_plot["spread_bps"],
                            name="Spread (bps)",
                            marker_color="#7FC97F",
                            opacity=0.6,
                        ), row=2, col=1)

                    fig.update_layout(
                        title=chart_title,
                        height=550,
                        hovermode="x unified",
                        legend=dict(orientation="h", y=1.05),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    fig.update_yaxes(title_text="YTM (%)", row=1, col=1)
                    fig.update_yaxes(title_text="Spread (bps)", row=2, col=1)
                    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
                    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Chart unavailable: {e}")

            # ── Anomaly table ──────────────────────────────────────────
            st.subheader("Anomaly Details")

            display_cols = [c for c in [
                "date", "isin", "avg_ytm", "spread_bps",
                "d1", "z_score_21d", "anomaly_score_norm",
                "confirmed_anomaly", "cusum_signal"
            ] if c in df_anom.columns]

            st.dataframe(
                df_anom[display_cols]
                    .sort_values("anomaly_score_norm", ascending=False)
                    .style.format({
                        "avg_ytm": "{:.2f}",
                        "spread_bps": "{:.0f}",
                        "d1": "{:.1f}",
                        "z_score_21d": "{:.2f}",
                        "anomaly_score_norm": "{:.4f}",
                    }),
                use_container_width=True,
                height=400,
            )

            # ── Download ───────────────────────────────────────────────
            csv = df_anom[display_cols].to_csv(index=False)
            st.download_button(
                "⬇ Download anomalies CSV",
                data=csv,
                file_name="attribution_x_anomalies.csv",
                mime="text/csv",
            )

            # ── AI Explanation Panel ───────────────────────────────────
            st.divider()
            st.subheader("🤖 AI Anomaly Explanation")

            col_a, col_b = st.columns([1, 2])
            with col_a:
                selected_isin = st.selectbox(
                    "Select anomaly to explain",
                    options=df_anom.sort_values(
                        "anomaly_score_norm", ascending=False
                    )["isin"].tolist(),
                    key="explain_isin"
                )
                selected_date = df_anom[
                    df_anom["isin"] == selected_isin
                ]["date"].iloc[0]
                st.caption(f"Date: {str(selected_date)[:10]}")

                explain_btn = st.button(
                    "🔍 Explain this anomaly",
                    type="primary",
                    use_container_width=True
                )

            with col_b:
                if explain_btn:
                    with st.spinner("Retrieving evidence and generating explanation..."):
                        try:
                            resp = requests.get(
                                f"{api_url}/explain-anomaly",
                                params={
                                    "isin": selected_isin,
                                    "date": str(selected_date)[:10]
                                },
                                timeout=30,
                            )
                            if resp.status_code == 200:
                                expl = resp.json()["explanation"]
                                st.session_state["last_explanation"] = expl
                            else:
                                st.error(f"API error: {resp.text[:200]}")
                        except Exception as e:
                            st.error(f"Error: {e}")

                if "last_explanation" in st.session_state:
                    expl = st.session_state["last_explanation"]

                    # Confidence score
                    conf = expl.get("confidence_score", 0)
                    conf_color = "🟢" if conf > 0.6 else "🟡" if conf > 0.3 else "🔴"
                    st.metric("Confidence Score", f"{conf_color} {conf:.2%}")

                    # Explanation text
                    st.markdown("**Explanation**")
                    st.info(expl.get("explanation", "No explanation available."))

                    # Evidence
                    st.markdown("**Supporting Evidence**")
                    for ev in expl.get("evidence", []):
                        with st.expander(
                            f"📄 {ev['title']} ({ev['date']}) — similarity: {ev['score']:.3f}"
                        ):
                            st.write(ev["text"])


# ── TAB 3: About ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("About Attribution X")
    st.markdown("""
    **Attribution X** is an AI-powered system for the Indian fixed-income market that:

    - Detects anomalies in bond yield behaviour using **Isolation Forest + CUSUM**
    - Identifies early signs of credit stress (DHFL, IL&FS, Yes Bank)
    - Provides explainable, audit-ready outputs with confidence scores
    - Runs on real CBRICS/FIMMDA trade data (2015–2024)

    ### Pipeline
```
    CBRICS trades → Daily VWAP aggregation → Feature engineering
    → Isolation Forest → Persistence filter → CUSUM → Anomaly results
```

    ### Key Features
    | Feature | Description |
    |---|---|
    | d1 | 1-day YTM change (bps) |
    | d5 | 5-day YTM change (bps) |
    | spread_bps | YTM minus GOI benchmark |
    | z_score_21d | 21-day rolling z-score |
    | vol_log | Log trade volume |

    ### Tech Stack
    - **Backend**: FastAPI on Render
    - **Frontend**: Streamlit Cloud
    - **Database**: Neon PostgreSQL
    - **Vector DB**: Qdrant Cloud (evidence retrieval)
    - **Model**: scikit-learn Isolation Forest
    """)