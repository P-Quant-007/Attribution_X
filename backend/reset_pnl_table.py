import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from database import get_engine, Base
from sqlalchemy import text

engine = get_engine()
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS portfolio_pnl"))
    conn.commit()
    print("Dropped portfolio_pnl")

Base.metadata.create_all(engine)
print("Recreated all tables")