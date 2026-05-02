"""
Emergency migration: Add currency column to portfolios table if it doesn't exist.
Run this directly on the production database when alembic migrations can't be run.
"""
import os
import sys
from sqlalchemy import create_engine, inspect, text

# Get DATABASE_URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "")
DATABASE_PATH = os.getenv("DATABASE_PATH", "../dev_portfolio.db")

if DATABASE_URL:
    # PostgreSQL
    db_url = DATABASE_URL
    print(f"Using PostgreSQL: {DATABASE_URL[:30]}...")
else:
    # SQLite
    db_url = f"sqlite:///{DATABASE_PATH}"
    print(f"Using SQLite: {DATABASE_PATH}")

try:
    engine = create_engine(db_url)
    inspector = inspect(engine)
    
    # Check if portfolios table exists
    if "portfolios" not in inspector.get_table_names():
        print("ERROR: portfolios table does not exist")
        print("Run 'alembic upgrade head' first to create all tables")
        sys.exit(1)
    
    # Check if currency column exists
    columns = {col["name"]: col for col in inspector.get_columns("portfolios")}
    
    if "currency" in columns:
        print("✓ currency column already exists")
    else:
        print("Adding currency column to portfolios table...")
        with engine.connect() as conn:
            # PostgreSQL and SQLite both support ALTER TABLE ADD COLUMN
            conn.execute(text("ALTER TABLE portfolios ADD COLUMN currency VARCHAR(10) NOT NULL DEFAULT 'KWD'"))
            conn.commit()
        print("✓ currency column added successfully")
    
    engine.dispose()
    print("\nDone!")
    
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
