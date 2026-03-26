"""
Run once to create all tables in Neon PostgreSQL.
Usage: python backend/setup_db.py
"""
from database import create_tables, get_engine
from sqlalchemy import text
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    engine = get_engine()

    # Test connection first
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"[db] Connected to: {version[:50]}")
    except Exception as e:
        print(f"[db] Connection failed: {e}")
        print("Check your DATABASE_URL in .env")
        sys.exit(1)

    create_tables()
    print("[db] Setup complete. Ready for data.")