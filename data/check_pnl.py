import sys, pandas as pd
sys.path.insert(0, '.')
from engine.portfolio import load_portfolio
from engine.loader import load_processed_trades
from engine.aggregator import compute_daily_ytm
from engine.features import compute_features
from engine.pnl import (
    enrich_portfolio_with_duration,
    compute_daily_pnl,
    generate_attribution_report
)

portfolio = load_portfolio('data/sample_portfolio.csv')
trades    = load_processed_trades('data/processed_trades.csv')
daily     = compute_daily_ytm(trades)
features  = compute_features(daily)
enriched  = enrich_portfolio_with_duration(portfolio, features)

pnl = compute_daily_pnl(
    enriched, features,
    start_date="2018-01-01",
    end_date="2019-12-31"
)

if pnl.empty:
    print("No PnL data — ISINs in portfolio not found in features")
else:
    report = generate_attribution_report(pnl)
    s = report["summary"]
    print(f"\n=== PnL ATTRIBUTION SUMMARY (2018-2019) ===")
    print(f"Total PnL:          ₹{s['total_pnl_lacs']:>10,.2f} Lacs")
    print(f"Benchmark PnL:      ₹{s['benchmark_pnl_lacs']:>10,.2f} Lacs")
    print(f"Spread PnL:         ₹{s['spread_pnl_lacs']:>10,.2f} Lacs")
    print(f"Spread % of total:  {s['spread_pct_of_total']:>10.1f}%")
    print(f"Days covered:       {s['n_days']:>10}")
    print(f"Bonds covered:      {s['n_bonds']:>10}")

    print(f"\n=== BY BOND ===")
    for b in report["by_bond"]:
        tag = f" [{b['stress_tag']}]" if b['stress_tag'] else ""
        print(f"  {b['isin']} | {b['issuer_name'][:28]:<28} | "
              f"PnL: ₹{b['total_pnl']:>9,.2f}L | "
              f"Spread: ₹{b['spread_pnl']:>9,.2f}L{tag}")

    print(f"\n=== TOP CONTRIBUTORS ===")
    for b in report["top_contributors"]:
        print(f"  {b['isin']} | {b['issuer_name'][:30]} | ₹{b['total_pnl']:,.2f}L")

    print(f"\n=== TOP DETRACTORS ===")
    for b in report["top_detractors"]:
        print(f"  {b['isin']} | {b['issuer_name'][:30]} | ₹{b['total_pnl']:,.2f}L")