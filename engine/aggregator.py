import pandas as pd
import numpy as np


def compute_daily_ytm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily average YTM per ISIN.

    Logic:
    - If volume is present for ALL trades on a given day → use VWAP
    - If volume is missing for ANY trade on that day → simple average fallback
    - Also computes: prints (trade count), volume_sum

    Input:  clean DataFrame from loader (date, isin, ytm, volume)
    Output: DataFrame with (date, isin, avg_ytm, prints, volume_sum, method)
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "isin", "avg_ytm",
                                     "prints", "volume_sum", "method"])

    results = []

    for (date, isin), group in df.groupby(["date", "isin"]):
        prints = len(group)
        has_volume = group["volume"].notna().all() and (group["volume"] > 0).all()

        if has_volume:
            total_vol = group["volume"].sum()
            avg_ytm = (group["ytm"] * group["volume"]).sum() / total_vol
            volume_sum = total_vol
            method = "vwap"
        else:
            avg_ytm = group["ytm"].mean()
            volume_sum = group["volume"].sum() if group["volume"].notna().any() else np.nan
            method = "simple"

        results.append({
            "date": date,
            "isin": isin,
            "avg_ytm": round(float(avg_ytm), 6),
            "prints": prints,
            "volume_sum": float(volume_sum) if not np.isnan(volume_sum) else np.nan,
            "method": method,
        })

    out = pd.DataFrame(results)
    out = out.sort_values(["isin", "date"]).reset_index(drop=True)

    print(f"[aggregator] {len(out)} daily rows | "
          f"VWAP: {(out['method']=='vwap').sum()} | "
          f"Simple avg: {(out['method']=='simple').sum()}")

    return out