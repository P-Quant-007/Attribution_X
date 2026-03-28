import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from database import get_engine, Base
from sqlalchemy import text

engine = get_engine()
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS anomalies"))
    conn.commit()
    print("Dropped anomalies table")

Base.metadata.create_all(engine)
print("Recreated all tables with correct schema")