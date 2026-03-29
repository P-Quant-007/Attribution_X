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

# Render URL on deployment
API_URL = "https://attribution-x.onrender.com"

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

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Run Analysis", "📈 Results", "💼 Portfolio", "ℹ️ About"])

# ── TAB 1: Upload & Run ───────────────────────────────────────────────────
with tab1:
    st.subheader("Credit Market Stress Detection")
    st.caption(
        "Pre-trained on the full CBRICS dataset. "
        "Click below to load results instantly."
    )

    col_btn, col_stats = st.columns([1, 2])

    with col_btn:
        if st.button("📂 Load Results from DB", type="primary", use_container_width=True):
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
                        st.success(f"Loaded {data['count']} anomalies")
                        st.rerun()
                    else:
                        st.error("Failed to fetch results")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col_stats:
        if "results" in st.session_state:
            r = st.session_state["results"]
            s = r["summary"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Anomalies", s["total_anomalies"])
            m2.metric("Confirmed", s["confirmed_anomalies"])
            m3.metric("Model", "Pre-trained")

    st.divider()
    with st.expander("Advanced — score a new CBRICS file using the pre-trained model"):
        st.caption(
            "Upload any CBRICS CSV/XLSX. The pre-trained model will score it. "
            "This does NOT retrain the model."
        )
        uploaded = st.file_uploader(
            "Upload CSV or XLSX",
            type=["csv", "xlsx"],
            help="CBRICS format or processed_trades.csv"
        )
        if uploaded:
            if st.button("▶ Score with Pre-trained Model", type="primary", use_container_width=True):
                with st.spinner("Scoring trades using pre-trained model..."):
                    try:
                        params = {}
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
                            st.success("Scoring complete!")
                            st.rerun()
                        else:
                            st.error(f"API error {resp.status_code}: {resp.text[:300]}")

                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API. Is the backend running?")
                    except Exception as e:
                        st.error(f"Error: {e}")


# ── TAB 2: Results & Charts ───────────────────────────────────────────────
with tab2:
    if "results" not in st.session_state:
        st.info("Run an analysis first or load stored results from the Run Analysis tab.")
    else:
        results        = st.session_state["results"]
        summary        = results["summary"]
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
            c1.metric("Total Trades",  f"{summary['total_trades']:,}" if isinstance(summary['total_trades'], int) else summary['total_trades'])
            c2.metric("Daily Rows",    f"{summary['total_daily_rows']:,}" if isinstance(summary['total_daily_rows'], int) else summary['total_daily_rows'])
            c3.metric("Raw Anomalies", summary["total_anomalies"])
            c4.metric("Confirmed",     summary["confirmed_anomalies"])
            c5.metric("Date Range",    f"{summary['date_range']['from']} → {summary['date_range']['to']}")

            st.divider()

            # ── YTM Chart ─────────────────────────────────────────────
            st.subheader("YTM Time Series with Anomaly Flags")

            try:
                isin_filter = None
                if stress_tag != "ALL":
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
                    if df_metrics.empty:
                        st.info("No metric history available for charting.")
                    else:
                        df_metrics["date"] = pd.to_datetime(df_metrics["date"])

                        if isin_filter:
                            df_plot    = df_metrics[df_metrics["isin"] == isin_filter]
                            anom_plot  = df_anom[df_anom["isin"] == isin_filter]
                            chart_title = f"YTM — {isin_filter}"
                        else:
                            df_plot = (df_metrics.groupby("date")["avg_ytm"]
                                                 .mean()
                                                 .reset_index())
                            anom_plot  = df_anom
                            chart_title = "Average YTM — All ISINs"

                        # Build chart — always runs when df_metrics is not empty
                        fig = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            vertical_spacing=0.05,
                        )

                        fig.add_trace(go.Scatter(
                            x=df_plot["date"],
                            y=df_plot["avg_ytm"],
                            mode="lines",
                            name="Avg YTM (%)",
                            line=dict(color="#4C9BE8", width=1.5),
                        ), row=1, col=1)

                        if len(anom_plot):
                            confirmed_anom = anom_plot[
                                anom_plot.get("confirmed_anomaly",
                                             pd.Series([0] * len(anom_plot))) == 1
                            ]
                            unconfirmed_anom = anom_plot[
                                anom_plot.get("confirmed_anomaly",
                                             pd.Series([0] * len(anom_plot))) != 1
                            ]

                            if len(confirmed_anom):
                                fig.add_trace(go.Scatter(
                                    x=confirmed_anom["date"],
                                    y=confirmed_anom["avg_ytm"],
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
                                    customdata=confirmed_anom["anomaly_score_norm"],
                                ), row=1, col=1)

                            if len(unconfirmed_anom):
                                fig.add_trace(go.Scatter(
                                    x=unconfirmed_anom["date"],
                                    y=unconfirmed_anom["avg_ytm"],
                                    mode="markers",
                                    name="Raw anomaly",
                                    marker=dict(color="orange", size=7,
                                                symbol="circle-open",
                                                line=dict(color="orange", width=1.5)),
                                ), row=1, col=1)

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

                    conf = expl.get("confidence_score", 0)
                    conf_color = "🟢" if conf > 0.6 else "🟡" if conf > 0.3 else "🔴"
                    st.metric("Confidence Score", f"{conf_color} {conf:.2%}")

                    st.markdown("**Explanation**")
                    st.info(expl.get("explanation", "No explanation available."))

                    st.markdown("**Supporting Evidence**")
                    for ev in expl.get("evidence", []):
                        with st.expander(
                            f"📄 {ev['title']} ({ev['date']}) — similarity: {ev['score']:.3f}"
                        ):
                            st.write(ev["text"])


# ── TAB 3: Portfolio ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Portfolio PnL Attribution")

    portfolio_id = "demo_portfolio"
    start_date   = "2018-01-01"
    end_date     = "2019-12-31"

    if "portfolio_summary" not in st.session_state:
        st.caption(
            "Demo portfolio: 8 bonds including DHFL, IL&FS, Yes Bank, NABARD, HDFC. "
            "Period: 2018–2019 DHFL stress cycle."
        )
        if st.button("📂 Load Demo Portfolio", type="primary", use_container_width=True):
            with st.spinner("Loading demo portfolio..."):
                try:
                    # Fetch holdings
                    port_resp = requests.get(
                        f"{api_url}/get-portfolio",
                        params={"portfolio_id": portfolio_id},
                        timeout=30,
                    )
                    # Fetch PnL
                    pnl_resp = requests.get(
                        f"{api_url}/get-pnl-attribution",
                        params={"portfolio_id": portfolio_id,
                                "start_date": start_date,
                                "end_date": end_date},
                        timeout=120,
                    )
                    if port_resp.status_code == 200 and pnl_resp.status_code == 200:
                        port_data = port_resp.json()
                        st.session_state["portfolio_summary"] = {
                            "portfolio_id": portfolio_id,
                            "holdings": port_data["holdings"],
                            "total_holdings": port_data["total_holdings"],
                            "total_aum_lacs": port_data["total_aum_lacs"],
                            "stress_aum_pct": 0,
                            "portfolio_dv01": 0,
                        }
                        st.session_state["pnl_report"] = pnl_resp.json()["report"]
                        st.success("Demo portfolio loaded!")
                        st.rerun()
                    else:
                        st.error(f"Failed to load portfolio: {port_resp.text[:200]}")
                except Exception as e:
                    st.error(f"Error: {e}")

    with st.expander("Advanced — upload your own portfolio"):
        col_up, col_set = st.columns([2, 1])
        with col_up:
            portfolio_file = st.file_uploader(
                "Upload portfolio CSV",
                type=["csv", "xlsx"],
                help="Columns: isin, issuer_name, coupon, maturity_date, face_value, rating",
                key="portfolio_uploader"
            )
        with col_set:
            custom_portfolio_id = st.text_input("Portfolio ID", value="custom_portfolio")
            custom_start = st.text_input("Analysis start", value="2018-01-01")
            custom_end   = st.text_input("Analysis end",   value="2019-12-31")

        if portfolio_file:
            if st.button("📤 Upload Portfolio", type="primary"):
                with st.spinner("Uploading and computing DV01 profile..."):
                    try:
                        resp = requests.post(
                            f"{api_url}/upload-portfolio",
                            files={"file": (portfolio_file.name,
                                            portfolio_file.getvalue(),
                                            "text/csv")},
                            params={"portfolio_id": custom_portfolio_id},
                            timeout=60,
                        )
                        if resp.status_code == 200:
                            st.session_state["portfolio_summary"] = resp.json()["summary"]
                            portfolio_id = custom_portfolio_id
                            start_date   = custom_start
                            end_date     = custom_end
                            st.success("Portfolio uploaded successfully!")
                            st.rerun()
                        else:
                            st.error(f"Upload failed: {resp.text[:200]}")
                    except Exception as e:
                        st.error(f"Error: {e}")

    if "portfolio_summary" in st.session_state:
        psummary = st.session_state["portfolio_summary"]

        if "holdings" in psummary:
            st.divider()
            st.subheader("Holdings")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total AUM",       f"₹{psummary.get('total_aum_lacs', 0):,.0f}L")
            c2.metric("Holdings",         psummary.get("total_holdings", 0))
            c3.metric("Stress exposure", f"{psummary.get('stress_aum_pct', 0):.1f}%")
            c4.metric("Portfolio DV01",  f"₹{psummary.get('portfolio_dv01', 0):.2f}L/bp")

            holdings_df = pd.DataFrame(psummary["holdings"])
            holdings_df["stress"] = holdings_df["is_stress_issuer"].apply(
                lambda x: "⚠️" if x else ""
            )
            display_cols = ["stress", "isin", "issuer_name", "coupon",
                            "face_value", "weight", "rating", "stress_tag"]
            available = [c for c in display_cols if c in holdings_df.columns]
            st.dataframe(
                holdings_df[available].style.apply(
                    lambda row: ["background-color: rgba(255,100,100,0.15)"] * len(row)
                    if row.get("is_stress_issuer", False) else [""] * len(row),
                    axis=1
                ),
                use_container_width=True,
                height=300,
            )

        st.divider()
        st.subheader("PnL Attribution")

        if st.button("📊 Run PnL Attribution", type="primary"):
            with st.spinner("Running attribution engine..."):
                try:
                    resp = requests.get(
                        f"{api_url}/get-pnl-attribution",
                        params={
                            "portfolio_id": portfolio_id,
                            "start_date":   start_date,
                            "end_date":     end_date,
                        },
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        st.session_state["pnl_report"] = resp.json()["report"]
                        st.success("Attribution complete!")
                        st.rerun()
                    else:
                        st.error(f"Attribution failed: {resp.text[:200]}")
                except Exception as e:
                    st.error(f"Error: {e}")

        if "pnl_report" in st.session_state:
            report  = st.session_state["pnl_report"]
            summary = report.get("summary", {})

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total PnL",      f"₹{summary.get('total_pnl_lacs', 0):,.2f}L")
            m2.metric("Benchmark PnL",  f"₹{summary.get('benchmark_pnl_lacs', 0):,.2f}L")
            m3.metric("Spread PnL",     f"₹{summary.get('spread_pnl_lacs', 0):,.2f}L")
            m4.metric("Spread % total", f"{summary.get('spread_pct_of_total', 0):.1f}%")

            daily = pd.DataFrame(report.get("daily_portfolio", []))
            if not daily.empty:
                daily["date"] = pd.to_datetime(daily["date"])
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=daily["date"],
                    y=daily["portfolio_cumulative_pnl"],
                    mode="lines",
                    fill="tozeroy",
                    name="Cumulative PnL",
                    line=dict(color="#E24B4A", width=2),
                    fillcolor="rgba(226,75,74,0.15)",
                ))
                fig_cum.update_layout(
                    title="Cumulative Portfolio PnL (₹ Lacs)",
                    height=350,
                    hovermode="x unified",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                fig_cum.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
                fig_cum.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
                st.plotly_chart(fig_cum, use_container_width=True)

            by_bond = pd.DataFrame(report.get("by_bond", []))
            if not by_bond.empty:
                fig_wf = go.Figure(go.Bar(
                    x=by_bond["issuer_name"].str[:25],
                    y=by_bond["total_pnl"],
                    marker_color=[
                        "#E24B4A" if v < 0 else "#639922"
                        for v in by_bond["total_pnl"]
                    ],
                    text=by_bond["total_pnl"].apply(lambda x: f"₹{x:,.0f}L"),
                    textposition="outside",
                ))
                fig_wf.update_layout(
                    title="PnL by Bond (₹ Lacs)",
                    height=350,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_wf, use_container_width=True)

        st.divider()
        st.subheader("🤖 AI Reallocation Suggestions")

        if st.button("✨ Generate AI Suggestions", type="primary"):
            with st.spinner("Analysing portfolio and generating suggestions..."):
                try:
                    resp = requests.get(
                        f"{api_url}/get-suggestions",
                        params={"portfolio_id": portfolio_id,
                                "start_date": start_date,
                                "end_date": end_date},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        st.session_state["suggestions"] = resp.json()["suggestions"]
                        st.success("Suggestions generated!")
                        st.rerun()
                    else:
                        st.error(f"Failed: {resp.text[:200]}")
                except Exception as e:
                    st.error(f"Error: {e}")

        if "suggestions" in st.session_state:
            suggestions = st.session_state["suggestions"]
            action_colors  = {"REDUCE": "#E24B4A", "ADD": "#639922", "SWITCH": "#BA7517"}
            priority_icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}

            st.caption("⚠️ AI-generated advisory. Not financial advice. Verify independently.")
            st.divider()

            for i, s in enumerate(suggestions, 1):
                action   = s.get("action", "")
                color    = action_colors.get(action, "#888")
                priority = s.get("priority", "MEDIUM")
                icon     = priority_icons.get(priority, "🟡")

                st.markdown(
                    f"""<div style="border-left: 4px solid {color}; padding: 12px 16px;
                    border-radius: 0 8px 8px 0; margin-bottom: 12px;
                    background: rgba(128,128,128,0.05);">
                    <span style="font-weight:500; color:{color};">{action}</span>
                    &nbsp;&nbsp;{icon} <b>{s.get('target', '')}</b>
                    </div>""",
                    unsafe_allow_html=True
                )
                st.write(f"**Rationale:** {s.get('rationale', '')}")
                st.caption(f"⚡ Risk: {s.get('risk_note', '')}")
                if i < len(suggestions):
                    st.divider()


# ── TAB 4: About ──────────────────────────────────────────────────────────
with tab4:
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