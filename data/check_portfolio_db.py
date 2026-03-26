import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'backend')

from engine.portfolio import load_portfolio, portfolio_summary
from backend.database import get_engine, save_portfolio, fetch_portfolio

df = load_portfolio('data/sample_portfolio.csv')
engine = get_engine()

save_portfolio(df, portfolio_id="demo_portfolio", engine=engine)

fetched = fetch_portfolio("demo_portfolio", engine=engine)
print(f"\n[db] Read back {len(fetched)} holdings from DB")
print(fetched[["isin", "issuer_name", "face_value",
               "weight", "rating", "stress_tag"]].to_string(index=False))
