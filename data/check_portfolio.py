import sys
sys.path.insert(0, '.')
from engine.portfolio import load_portfolio, portfolio_summary
import json

df = load_portfolio('data/sample_portfolio.csv')
summary = portfolio_summary(df)

print(f"\nTotal AUM:        ₹{summary['total_aum_lacs']:,.0f} Lacs")
print(f"Total holdings:   {summary['total_holdings']}")
print(f"Stress holdings:  {summary['stress_holdings']} "
      f"({summary['stress_aum_pct']}% of AUM)")
print(f"Avg coupon:       {summary['avg_coupon']}%")
print(f"Avg maturity:     {summary['avg_maturity_years']} years")
print(f"\nRating breakdown: {summary['rating_breakdown']}")
print(f"\nHoldings (sorted by size):")
for h in summary['holdings']:
    stress = f" ⚠ {h['stress_tag']}" if h['is_stress_issuer'] else ""
    print(f"  {h['isin']} | {h['issuer_name'][:30]:<30} | "
          f"₹{h['face_value']:>6,.0f}L | {h['weight']:>5.1f}% | "
          f"{h['rating']}{stress}")