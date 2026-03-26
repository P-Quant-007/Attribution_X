import sys, pandas as pd
sys.path.insert(0, '.')
from engine.portfolio import load_portfolio
from engine.loader import load_processed_trades
from engine.aggregator import compute_daily_ytm
from engine.features import compute_features
from engine.pnl import enrich_portfolio_with_duration, compute_daily_pnl, generate_attribution_report
from engine.suggestions import generate_suggestions

portfolio = load_portfolio('data/sample_portfolio.csv')
trades    = load_processed_trades('data/processed_trades.csv')
daily     = compute_daily_ytm(trades)
features  = compute_features(daily)
enriched  = enrich_portfolio_with_duration(portfolio, features)
pnl       = compute_daily_pnl(enriched, features, "2018-01-01", "2019-12-31")
report    = generate_attribution_report(pnl)

# Pass empty anomalies — suggestions engine handles it gracefully
anomalies = pd.DataFrame()

suggestions = generate_suggestions(report, anomalies, portfolio)

print("\n=== AI REALLOCATION SUGGESTIONS ===\n")
for i, s in enumerate(suggestions, 1):
    priority_icon = "🔴" if s['priority'] == "HIGH" else "🟡"
    print(f"{i}. [{s['action']}] {priority_icon} {s['target']}")
    print(f"   Rationale: {s['rationale']}")
    print(f"   Risk note: {s['risk_note']}")
    print()