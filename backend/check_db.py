import sys 
sys.path.insert(0, 'backend') 
from database import get_engine 
from sqlalchemy import text 
engine = get_engine() 
conn = engine.connect() 
r = conn.execute(text('SELECT COUNT(*), SUM(confirmed_anomaly), SUM(is_anomaly) FROM anomalies')) 
print(r.fetchone()) 
conn.close() 
