"""
Generate synthetic bond trade data for the DHFL stress period demo.
Run once: python data/generate_synthetic.py
Output:   data/synthetic_trades.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

# ── ISINs ──────────────────────────────────────────────────────────────────
ISINS = {
    "INE202E01016": "DHFL 9.10% 2021",        # primary stress bond
    "INE202E01024": "DHFL 9.20% 2022",        # secondary DHFL
    "INE134E08KL2": "HDFC Ltd 8.45% 2021",    # investment grade control
    "INE115A07FH6": "LIC Housing 8.30% 2021", # investment grade control
    "INE020B08BY1": "IL&FS 9.50% 2020",       # contagion bond
}

# ── Date range ─────────────────────────────────────────────────────────────
dates = pd.bdate_range("2018-01-01", "2019-03-31")  # business days only

# ── Baseline YTMs (pre-stress, annualised %) ──────────────────────────────
BASE_YTM = {
    "INE202E01016": 9.10,
    "INE202E01024": 9.20,
    "INE134E08KL2": 8.45,
    "INE115A07FH6": 8.30,
    "INE020B08BY1": 9.50,
}

# ── Stress events (date → isin → ytm_shock in bps, liquidity_drop 0-1) ───
# DHFL stress builds from Aug 2018, spikes Sep-Nov 2018
# IL&FS crisis: Sep-Oct 2018
STRESS_EVENTS = [
    # (start_date, end_date, isin, ytm_ramp_bps, liquidity_factor)
    ("2018-09-01", "2018-09-30", "INE020B08BY1", 180, 0.5),   # IL&FS shock
    ("2018-09-15", "2018-10-31", "INE202E01016", 120, 0.6),   # DHFL early stress
    ("2018-09-15", "2018-10-31", "INE202E01024", 110, 0.6),
    ("2018-11-01", "2018-12-31", "INE202E01016", 280, 0.3),   # DHFL acute stress
    ("2018-11-01", "2018-12-31", "INE202E01024", 260, 0.3),
    ("2019-01-01", "2019-03-31", "INE202E01016", 420, 0.15),  # DHFL near-default
    ("2019-01-01", "2019-03-31", "INE202E01024", 400, 0.15),
    ("2018-10-01", "2018-11-30", "INE020B08BY1", 350, 0.2),   # IL&FS contagion
]

# ── GOI benchmark curve (approximate ZCYC for 3yr tenor, %) ───────────────
# Rises ~50bps during 2018 tightening cycle
def goi_yield(date):
    base = 7.20
    days_from_start = (date - pd.Timestamp("2018-01-01")).days
    trend = min(days_from_start / 365 * 0.50, 0.50)
    noise = np.random.normal(0, 0.02)
    return round(base + trend + noise, 4)


def build_stress_map():
    """Build (date, isin) → (ytm_shock_bps, liquidity_factor) map."""
    stress = {}
    for start, end, isin, bps, liq in STRESS_EVENTS:
        period = pd.bdate_range(start, end)
        n = len(period)
        for i, d in enumerate(period):
            # Ramp up linearly over first 1/3, hold, slight release
            ramp = min((i / (n / 3)), 1.0)
            shock = bps * ramp
            key = (d, isin)
            if key not in stress or stress[key][0] < shock:
                stress[key] = (shock, liq)
    return stress


def generate_trades():
    stress_map = build_stress_map()
    rows = []

    for date in dates:
        bench = goi_yield(date)

        for isin, name in ISINS.items():
            base = BASE_YTM[isin]
            shock_bps, liq_factor = stress_map.get((date, isin), (0, 1.0))

            # Daily noise (stressed bonds are more volatile)
            vol_scale = 1 + (shock_bps / 100)
            daily_noise = np.random.normal(0, 0.04 * vol_scale)

            ytm_mid = base + (shock_bps / 100) + daily_noise

            # Number of trades (sparse during stress)
            base_prints = 4 if "DHFL" in name else 6
            n_trades = max(1, int(np.random.poisson(base_prints * liq_factor)))

            # Skip some days entirely during deep stress (illiquidity)
            if liq_factor < 0.2 and np.random.rand() > liq_factor * 3:
                continue

            for _ in range(n_trades):
                trade_noise = np.random.normal(0, 0.03 * vol_scale)
                ytm = round(max(0.5, ytm_mid + trade_noise), 4)

                # Volume (face value in INR, sparse during stress)
                base_vol = np.random.lognormal(mean=16.5, sigma=0.8)  # ~₹15cr avg
                volume = round(base_vol * liq_factor, 0)

                # Randomly drop volume for ~15% of trades (simulates missing data)
                if np.random.rand() < 0.15:
                    volume = np.nan

                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "isin": isin,
                    "ytm": ytm,
                    "volume": volume,
                    "benchmark_ytm": bench,
                    "bond_name": name,
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating synthetic DHFL stress period dataset...")
    df = generate_trades()

    out_path = Path(__file__).parent / "synthetic_trades.csv"
    df.to_csv(out_path, index=False)

    print(f"\nDataset saved to: {out_path}")
    print(f"Total trades:     {len(df):,}")
    print(f"Date range:       {df['date'].min()} to {df['date'].max()}")
    print(f"ISINs:            {df['isin'].nunique()}")
    print(f"\nTrades per ISIN:")
    print(df.groupby("bond_name")["ytm"].count().to_string())
    print(f"\nYTM range by ISIN (showing stress):")
    print(df.groupby("bond_name")["ytm"].agg(["min","max","mean"]).round(2).to_string())