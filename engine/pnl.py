import pandas as pd
import numpy as np
from datetime import date


# ── Bond math ─────────────────────────────────────────────────────────────

def compute_bond_price(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    years_to_maturity: float,
    freq: int = 2,
) -> float:
    """
    Compute dirty price of a fixed-rate bond.

    face_value:        par value (any unit — ₹ Lacs)
    coupon_rate:       annual coupon in % (e.g. 8.5 for 8.5%)
    ytm:               yield to maturity in % (e.g. 9.0 for 9%)
    years_to_maturity: years remaining to final maturity
    freq:              coupon frequency (2 = semi-annual, default for India)
    """
    if years_to_maturity <= 0:
        return face_value  # matured bond — return par

    y = ytm / 100 / freq
    c = (coupon_rate / 100 / freq) * face_value
    n = max(1, round(years_to_maturity * freq))

    periods = np.arange(1, n + 1)
    pv_coupons = np.sum(c / (1 + y) ** periods)
    pv_principal = face_value / (1 + y) ** n

    return pv_coupons + pv_principal


def compute_modified_duration(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    years_to_maturity: float,
    freq: int = 2,
) -> float:
    """
    Compute modified duration of a fixed-rate bond.
    Returns duration in years.
    """
    if years_to_maturity <= 0:
        return 0.0

    y = ytm / 100 / freq
    c = (coupon_rate / 100 / freq) * face_value
    n = max(1, round(years_to_maturity * freq))

    periods = np.arange(1, n + 1)
    pv_coupons   = c / (1 + y) ** periods
    pv_principal = np.zeros(n)
    pv_principal[-1] = face_value / (1 + y) ** n

    total_pv   = pv_coupons + pv_principal
    price      = total_pv.sum()

    if price <= 0:
        return 0.0

    # Macaulay duration — weight each period by PV fraction
    macaulay = np.sum((periods / freq) * total_pv) / price

    # Modified duration
    modified = macaulay / (1 + y)

    return round(float(modified), 6)


def compute_dv01(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    years_to_maturity: float,
    freq: int = 2,
) -> float:
    """
    Compute DV01 (dollar value of 1 basis point) for a bond.

    DV01 = face_value × modified_duration / 10,000
    Units: same as face_value per basis point (e.g. ₹ Lacs per bp)

    Alternatively computed as price difference for +1bp shock
    (we use both and return the analytical version).
    """
    mod_dur = compute_modified_duration(
        face_value, coupon_rate, ytm, years_to_maturity, freq
    )
    dv01 = face_value * mod_dur / 10_000
    return round(float(dv01), 6)


def enrich_portfolio_with_duration(
    portfolio_df: pd.DataFrame,
    daily_metrics_df: pd.DataFrame,
    as_of_date: str | None = None,
) -> pd.DataFrame:
    """
    Enrich portfolio holdings with duration and DV01 using
    the most recent available YTM from daily_metrics.

    portfolio_df:      output of engine.portfolio.load_portfolio()
    daily_metrics_df:  output of engine.features.compute_features()
    as_of_date:        use YTM as of this date (default: most recent)

    Returns portfolio_df with added columns:
    - current_ytm, modified_duration, dv01, price_per_lac
    """
    df = portfolio_df.copy()

    # Get most recent YTM per ISIN from daily_metrics
    dm = daily_metrics_df.copy()
    dm["date"] = pd.to_datetime(dm["date"])

    if as_of_date:
        dm = dm[dm["date"] <= pd.Timestamp(as_of_date)]

    # Latest YTM per ISIN
    latest_ytm = (
        dm.sort_values("date")
          .groupby("isin")["avg_ytm"]
          .last()
          .reset_index()
          .rename(columns={"avg_ytm": "current_ytm"})
    )

    df = df.merge(latest_ytm, on="isin", how="left")

    # Fallback to coupon rate if no market YTM available
    df["current_ytm"] = df["current_ytm"].fillna(df["coupon"])

    # Compute duration and DV01 per bond
    results = []
    for _, row in df.iterrows():
        ytm = float(row["current_ytm"])
        coupon = float(row["coupon"])
        fv = float(row["face_value"])
        # Always recompute from maturity_date to avoid stale values
        today = pd.Timestamp.today().normalize()
        mat = pd.Timestamp(row["maturity_date"])
        ytm_years = max(0.0, (mat - today).days / 365.25)

        # Use absolute years for matured bonds — set to 0
        ytm_years = max(0.0, ytm_years)

        mod_dur = compute_modified_duration(fv, coupon, ytm, ytm_years)
        dv01    = compute_dv01(fv, coupon, ytm, ytm_years)
        price   = compute_bond_price(fv, coupon, ytm, ytm_years)

        results.append({
            "modified_duration": mod_dur,
            "dv01":              dv01,
            "price_per_lac":     round(price / fv * 100, 4) if fv > 0 else 100.0,
        })

    enriched = pd.DataFrame(results)
    df["modified_duration"] = enriched["modified_duration"].values
    df["dv01"]              = enriched["dv01"].values
    df["price_per_lac"]     = enriched["price_per_lac"].values

    # Portfolio-level DV01
    total_dv01 = df["dv01"].sum()
    df["dv01_contribution_pct"] = (df["dv01"] / total_dv01 * 100).round(4)

    print(f"[pnl] Portfolio DV01: ₹{total_dv01:,.2f} Lacs/bp | "
          f"Avg modified duration: {df['modified_duration'].mean():.2f}y")

    return df

def compute_daily_pnl(
    enriched_portfolio: pd.DataFrame,
    daily_features: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    port = enriched_portfolio.copy()
    feat = daily_features.copy()
    feat["date"] = pd.to_datetime(feat["date"])

    if start_date:
        feat = feat[feat["date"] >= pd.Timestamp(start_date)]
    if end_date:
        feat = feat[feat["date"] <= pd.Timestamp(end_date)]

    portfolio_isins = set(port["isin"].tolist())
    feat = feat[feat["isin"].isin(portfolio_isins)]

    if feat.empty:
        print("[pnl] Warning: no daily features found for portfolio ISINs.")
        return pd.DataFrame()

    dv01_map   = port.set_index("isin")["dv01"].to_dict()
    fv_map     = port.set_index("isin")["face_value"].to_dict()
    weight_map = port.set_index("isin")["weight"].to_dict()
    coupon_map = port.set_index("isin")["coupon"].to_dict()
    issuer_map = port.set_index("isin")["issuer_name"].to_dict()
    stress_map = port.set_index("isin")["stress_tag"].to_dict()
    dur_map    = port.set_index("isin")["modified_duration"].to_dict()

    rows = []
    for _, r in feat.iterrows():
        isin = r["isin"]
        dv01 = dv01_map.get(isin, 0)

        d1_bps       = float(r.get("d1", 0) or 0)
        spread_d1    = float(r.get("spread_d1", 0) or 0)
        benchmark_d1 = d1_bps - spread_d1

        daily_pnl     = -dv01 * d1_bps
        spread_pnl    = -dv01 * spread_d1
        benchmark_pnl = -dv01 * benchmark_d1

        rows.append({
            "date":               r["date"],
            "isin":               isin,
            "issuer_name":        issuer_map.get(isin, ""),
            "face_value":         fv_map.get(isin, 0),
            "weight":             weight_map.get(isin, 0),
            "coupon":             coupon_map.get(isin, 0),
            "avg_ytm":            float(r.get("avg_ytm", 0) or 0),
            "d1_bps":             round(d1_bps, 4),
            "modified_duration":  dur_map.get(isin, 0),
            "dv01":               dv01,
            "daily_pnl":          round(daily_pnl, 4),
            "benchmark_pnl":      round(benchmark_pnl, 4),
            "spread_pnl":         round(spread_pnl, 4),
            "stress_tag":         stress_map.get(isin),
            "is_anomaly":         int(r.get("is_anomaly", 0) or 0),
        })

    result = pd.DataFrame(rows).sort_values(["isin", "date"])
    result["cumulative_pnl"] = (
        result.groupby("isin")["daily_pnl"].cumsum().round(4)
    )

    port_daily = (
        result.groupby("date")[["daily_pnl", "benchmark_pnl", "spread_pnl"]]
        .sum().reset_index()
        .rename(columns={
            "daily_pnl":     "portfolio_daily_pnl",
            "benchmark_pnl": "portfolio_benchmark_pnl",
            "spread_pnl":    "portfolio_spread_pnl",
        })
    )
    port_daily["portfolio_cumulative_pnl"] = (
        port_daily["portfolio_daily_pnl"].cumsum().round(4)
    )

    result = result.merge(port_daily, on="date", how="left")

    n_days    = result["date"].nunique()
    total_pnl = port_daily["portfolio_daily_pnl"].sum()
    print(f"[pnl] Attribution complete: {n_days} days | "
          f"{len(result):,} bond-day rows | "
          f"Total PnL: ₹{total_pnl:,.2f} Lacs")

    return result


def generate_attribution_report(pnl_df: pd.DataFrame) -> dict:
    if pnl_df.empty:
        return {"error": "No PnL data available"}

    total_pnl     = round(float(pnl_df.groupby("date")["portfolio_daily_pnl"].first().sum()), 2)
    benchmark_pnl = round(float(pnl_df.groupby("date")["portfolio_benchmark_pnl"].first().sum()), 2)
    spread_pnl    = round(float(pnl_df.groupby("date")["portfolio_spread_pnl"].first().sum()), 2)

    by_bond = (
        pnl_df.groupby(["isin", "issuer_name", "stress_tag"])
        .agg(
            total_pnl     = ("daily_pnl",     "sum"),
            benchmark_pnl = ("benchmark_pnl", "sum"),
            spread_pnl    = ("spread_pnl",    "sum"),
            avg_ytm       = ("avg_ytm",        "mean"),
            dv01          = ("dv01",           "first"),
            weight        = ("weight",         "first"),
            n_days        = ("daily_pnl",      "count"),
        )
        .reset_index().round(4)
        .sort_values("total_pnl", ascending=False)
    )

    daily = (
        pnl_df.groupby("date")
        .agg(
            portfolio_daily_pnl      = ("portfolio_daily_pnl",      "first"),
            portfolio_benchmark_pnl  = ("portfolio_benchmark_pnl",  "first"),
            portfolio_spread_pnl     = ("portfolio_spread_pnl",     "first"),
            portfolio_cumulative_pnl = ("portfolio_cumulative_pnl", "first"),
        )
        .reset_index()
    )
    daily["date"] = daily["date"].astype(str)

    return {
        "summary": {
            "total_pnl_lacs":      total_pnl,
            "benchmark_pnl_lacs":  benchmark_pnl,
            "spread_pnl_lacs":     spread_pnl,
            "spread_pct_of_total": round(
                spread_pnl / total_pnl * 100 if total_pnl != 0 else 0, 2
            ),
            "n_days":  int(pnl_df["date"].nunique()),
            "n_bonds": int(pnl_df["isin"].nunique()),
        },
        "by_bond": by_bond.assign(
            stress_tag=by_bond["stress_tag"].fillna("")
        ).to_dict(orient="records"),
        "top_contributors": by_bond.nlargest(3, "total_pnl")[
            ["isin", "issuer_name", "total_pnl", "spread_pnl", "weight"]
        ].to_dict(orient="records"),
        "top_detractors": by_bond.nsmallest(3, "total_pnl")[
            ["isin", "issuer_name", "total_pnl", "spread_pnl", "weight"]
        ].to_dict(orient="records"),
        "daily_portfolio": daily.to_dict(orient="records"),
    }