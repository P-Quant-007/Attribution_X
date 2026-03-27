import sys, pandas as pd
sys.path.insert(0, '.')
from engine.portfolio import load_portfolio
from engine.loader import load_processed_trades
from engine.aggregator import compute_daily_ytm
from engine.features import compute_features
from engine.pnl import enrich_portfolio_with_duration

portfolio = load_portfolio('data/sample_portfolio.csv')
trades    = load_processed_trades('data/processed_trades.csv')
daily     = compute_daily_ytm(trades)
features  = compute_features(daily)
enriched  = enrich_portfolio_with_duration(portfolio, features)

print("\n=== PORTFOLIO DURATION PROFILE ===")
print(f"{'ISIN':<15} {'Issuer':<30} {'Coupon':>6} {'YTM':>6} "
      f"{'Dur':>6} {'DV01 (₹L/bp)':>12} {'Wt%':>6}")
print("-" * 85)
for _, r in enriched.iterrows():
    stress = " ⚠" if r['is_stress_issuer'] else ""
    print(f"{r['isin']:<15} {r['issuer_name'][:30]:<30} "
          f"{r['coupon']:>6.2f} {r['current_ytm']:>6.2f} "
          f"{r['modified_duration']:>6.2f} {r['dv01']:>12.4f} "
          f"{r['weight']:>6.1f}{stress}")

total_dv01 = enriched['dv01'].sum()
print("-" * 85)
print(f"{'PORTFOLIO TOTAL':<52} {'':>6} {total_dv01:>12.4f}")
print(f"\nFor a 10bp parallel shift, portfolio PnL = ₹{total_dv01 * 10:,.2f} Lacs")